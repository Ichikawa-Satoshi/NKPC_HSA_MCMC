"""HSA dynamic-theta NKPC with an MA(3) inflation disturbance.

The additive counterpart of
``nkpc_hsa.gibbs.hsa_dynamic.model.func_nkpc_hsa_decomp_joint_fullSigma``, which
is left untouched. The model is identical except that the inflation shock is an
MA(3)-filtered innovation rather than the innovation itself:

    pi_t    = alpha*pi_{t-1} + (1-alpha)*E_t pi_{t+1} + kappa*x_t - theta*Nhat_t + e_t
    e_t     = psi(L) v_t                                       <-- the only change
    x_t     = phi_1*x_{t-1} + zeta_t
    N_obs_t = Nhat_t + Nbar_t + nu_t
    Nhat_t  = rho_1*Nhat_{t-1} + rho_2*Nhat_{t-2} + u_t
    Nbar_t  = n + Nbar_{t-1} + epsilon_t

    [v_t, zeta_t, u_t, epsilon_t]' ~ N(0, Sigma)

Reading Sigma's first coordinate as the *innovation* keeps
``covariance_structure`` meaning what it meant before: ``e_zeta_only`` is still
one free correlation, now between ``v_t`` and ``zeta_t``. With ``psi = 0`` the
model collapses to production exactly.

Conditioning on the innovation, not the disturbance
---------------------------------------------------
Production's coefficient, ``phi``, ``rho`` and ``n`` blocks condition on ``e_t``
through Sigma. Here they condition on ``v_t``, which is what Sigma is now about.
``v`` is recovered from the disturbance by ``inverse_ma_filter`` after every
coefficient move, seeded with the pre-sample ``v_{-1..-q}`` the augmented FFBS
state carries at ``t = 0`` -- so the recovery is exact, not an approximation
that conditions the first ``q`` periods away.

The coefficient block needs more than a reweighting. Conditional on
``(zeta, u, eps)`` the innovation is ``v_t ~ N(mean_v_t, var_v_t)``, independent
across ``t`` but *heteroskedastic*: ``var_v`` at ``t = 0`` differs from
``t >= 1`` because only ``zeta_0`` is available to condition on -- ``u_0`` and
``eps_0`` do not exist. The disturbance therefore has mean ``psi(L) mean_v`` and
the banded, non-Toeplitz covariance ``L_psi diag(var_v) L_psi'`` that
``MAWeighting(..., innovation_var=...)`` builds, with the pre-sample rows given
the unconditional variance ``Sigma[0, 0]``.

Block order
-----------
``beta`` -> ``phi_1`` -> ``rho`` -> ``n`` -> ``Sigma`` -> ``sigma_N2`` ->
``states`` -> ``psi``. ``psi`` is last for the Chib reason documented in
``ma_error``.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
from numpy.linalg import inv

from nkpc_hsa.error_robustness.joint_ffbs_ma3 import sample_joint_states_ffbs_dynamic_ma
from nkpc_hsa.error_robustness.ma_error import (
    MA_ORDER,
    AdaptiveRandomWalk,
    MAWeighting,
    PsiPrior,
    autocovariance,
    inverse_ma_filter,
    is_invertible,
    ma_filter,
)
from nkpc_hsa.gibbs.common.competition import finite_N_residuals
from nkpc_hsa.gibbs.common.constraints import constraint_stats_summary, draw_with_constraints
from nkpc_hsa.gibbs.hsa_dynamic.model import (
    KAPPA_SCALE,
    _ar2_stats_summary,
    _as_1d,
    _assert_all_pos,
    _common_priors,
    _compute_state_residuals,
    _conditional_e_all,
    _getd,
    _init_states,
    _mvnrnd,
    _restrict_sigma_structure,
    _sample_ar2_coeffs_full,
    _sample_invgamma,
    _sample_n_full,
    _sample_phi_full,
    _sample_Sigma,
    _summary,
)

__all__ = ["func_nkpc_hsa_dynamic_ma3"]


def _sample_beta_gls(
    *,
    y: np.ndarray,
    X: np.ndarray,
    weighting: MAWeighting,
    prior_mean: np.ndarray,
    prior_var: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Production ``_sample_beta_gaussian_weighted`` with a banded covariance.

    Production weights row by row because its disturbance is serially
    independent; here the covariance has bandwidth ``q``, so the diagonal weight
    ``1/var_t`` becomes ``Omega^{-1}``.
    """
    y = _as_1d(y)
    X = np.asarray(X, dtype=float)
    prior_mean = _as_1d(prior_mean)
    prior_var = _as_1d(prior_var)

    if X.ndim != 2:
        raise ValueError("X must be 2D.")
    if X.shape[0] != y.size:
        raise ValueError("y and X have incompatible lengths.")
    if X.shape[1] != prior_mean.size or prior_mean.size != prior_var.size:
        raise ValueError("Prior dimensions do not match X.")
    _assert_all_pos(prior_var, "Prior variances must be positive.")

    XtWX, XtWy = weighting.gls_moments(y, X)
    V0_inv = np.diag(1.0 / prior_var)
    Vn = inv(XtWX + V0_inv)
    mn = Vn @ (XtWy + V0_inv @ prior_mean)
    return _mvnrnd(mn, Vn, rng)


def func_nkpc_hsa_dynamic_ma3(
    pi_data,
    pi_prev_data,
    Epi_data,
    x_data,
    x_prev_data,
    N_data,
    n_burn: int,
    n_keep: int,
    priors: Optional[dict[str, Any]] = None,
    opts: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Gibbs/Metropolis sampler for hsa_dynamic with an MA(q) inflation disturbance.

    ``opts`` additions over the production sampler: ``ma_order`` (default 3, 0
    gives the i.i.d. specification), ``psi0``, ``n_psi_steps``,
    ``psi_init_scale``.
    """
    pi_t = _as_1d(pi_data)
    pi_tm1 = _as_1d(pi_prev_data)
    pi_expect = _as_1d(Epi_data)
    x_t = _as_1d(x_data)
    x_tm1 = _as_1d(x_prev_data)
    N_obs = _as_1d(N_data)

    T = pi_t.size
    if not (pi_tm1.size == pi_expect.size == x_t.size == x_tm1.size == N_obs.size == T):
        raise ValueError("All input series must have the same length.")
    if T < 3:
        raise ValueError("Need T >= 3.")
    if n_burn < 0:
        raise ValueError("n_burn must be nonnegative.")
    if n_keep <= 0:
        raise ValueError("n_keep must be positive.")

    pri = _common_priors(priors or {})
    opts = opts or {}

    _assert_all_pos(
        [
            pri["sigma_alpha"], pri["sigma_kappa"], pri["sigma_theta"], pri["sigma_phi"],
            pri["sigma_rho1"], pri["sigma_rho2"], pri["sigma_n"], pri["a_N"], pri["b_N"],
            pri["P0_Nhat"], pri["P0_Nhat_lag"], pri["P0_Nbar"],
        ],
        "Prior scales and inverse-gamma hyperparameters must be positive.",
    )
    if pri["nu_Sigma"] <= 3:
        raise ValueError("nu_Sigma must be greater than 3 for a 4x4 inverse-Wishart prior.")

    covariance_structure = str(_getd(opts, "covariance_structure", "e_zeta_only"))
    S_Sigma = _restrict_sigma_structure(
        np.asarray(pri["S_Sigma"], dtype=float), covariance_structure
    )
    if S_Sigma.shape != (4, 4):
        raise ValueError("S_Sigma must be a 4x4 matrix.")

    ma_order = int(_getd(opts, "ma_order", MA_ORDER))
    if ma_order < 0:
        raise ValueError("ma_order must be nonnegative.")
    psi_prior = PsiPrior.from_config(priors, order=ma_order) if ma_order else None
    psi = np.asarray(_getd(opts, "psi0", np.zeros(ma_order)), dtype=float).reshape(-1)
    if psi.size != ma_order:
        raise ValueError(f"psi0 must have length ma_order={ma_order}.")
    if not is_invertible(psi):
        raise ValueError(f"psi0={psi} is not invertible.")
    n_psi_steps = int(_getd(opts, "n_psi_steps", 2))

    rng = np.random.default_rng(_getd(opts, "seed", None))
    alpha = float(_getd(opts, "alpha0", pri["mu_alpha"]))
    kappa = float(_getd(opts, "kappa0", pri["mu_kappa"]))
    theta = float(_getd(opts, "theta0", pri["mu_theta"]))
    phi_1 = float(_getd(opts, "phi10", pri["mu_phi"]))
    rho1 = float(_getd(opts, "rho10", pri["mu_rho1"]))
    rho2 = float(_getd(opts, "rho20", pri["mu_rho2"]))
    n_drift = float(_getd(opts, "n0", pri["mu_n"]))
    sigma_N2 = float(_getd(opts, "sigma_N20", 1.0))

    # These defaults must match production exactly. rho_1/rho_2 have an
    # integrated autocorrelation time around 90, so a 2000-sweep burn-in is only
    # ~20 effective draws for that block -- far too few to forget a different
    # starting sigma_u2/sigma_eps2. Getting them wrong shifts the posterior
    # means by half a standard deviation and looks exactly like a sampler bug.
    Sigma0 = _getd(opts, "Sigma0", None)
    if Sigma0 is None:
        Sigma = np.diag(
            [
                float(_getd(opts, "sigma_e20", 1.0)),
                float(_getd(opts, "sigma_zeta20", 1.0)),
                float(_getd(opts, "sigma_u20", 0.5)),
                float(_getd(opts, "sigma_eps20", 0.1)),
            ]
        )
    else:
        Sigma = np.asarray(Sigma0, dtype=float)
    Sigma = _restrict_sigma_structure(Sigma, covariance_structure)
    if Sigma.shape != (4, 4):
        raise ValueError("Sigma0 must be 4x4.")

    enforce_stationary = bool(_getd(opts, "enforce_stationary", True))
    ar2_max_tries = int(max(1, _getd(opts, "ar2_max_tries", 2000)))
    store_every = int(max(1, _getd(opts, "store_every", 1)))
    verbose = bool(_getd(opts, "verbose", False))
    # Display-only hook installed by the run driver; it never touches the draws.
    progress_callback = _getd(opts, "progress_callback", None)
    coefficient_constraints = _getd(opts, "coefficient_constraints", {})
    constraint_stats: dict[str, int] = {}
    ar2_stats: dict[str, int] = {}

    n_store = int(n_keep // store_every)
    if n_store <= 0:
        raise ValueError("No draws would be stored. Use n_keep >= store_every.")

    Nbar, Nhat, states = _init_states(N_obs)
    a_t = pi_tm1 - pi_expect
    y = pi_t - pi_expect

    m0 = np.array(
        [
            float(_getd(opts, "m0_Nhat", pri["m0_Nhat"])),
            float(_getd(opts, "m0_Nhat_lag", pri["m0_Nhat_lag"])),
            float(_getd(opts, "m0_Nbar", pri["m0_Nbar"])),
        ],
        dtype=float,
    )
    P0 = np.diag(
        [
            float(_getd(opts, "P0_Nhat", pri["P0_Nhat"])),
            float(_getd(opts, "P0_Nhat_lag", pri["P0_Nhat_lag"])),
            float(_getd(opts, "P0_Nbar", pri["P0_Nbar"])),
        ]
    )

    proposal = (
        AdaptiveRandomWalk(ma_order, init_scale=float(_getd(opts, "psi_init_scale", 0.08)))
        if ma_order
        else None
    )
    # Pre-sample innovations, updated from the augmented state each sweep.
    psi_presample = np.zeros(ma_order, dtype=float)

    alpha_draws = np.zeros(n_store)
    kappa_draws = np.zeros(n_store)
    theta_draws = np.zeros(n_store)
    phi_draws = np.zeros(n_store)
    rho1_draws = np.zeros(n_store)
    rho2_draws = np.zeros(n_store)
    n_draws = np.zeros(n_store)
    sigma_N_draws = np.zeros(n_store)
    sigma_e_draws = np.zeros(n_store)
    sigma_v_draws = np.zeros(n_store)
    sigma_zeta_draws = np.zeros(n_store)
    sigma_u_draws = np.zeros(n_store)
    sigma_eps_draws = np.zeros(n_store)
    corr_e_zeta_draws = np.zeros(n_store)
    corr_e_u_draws = np.zeros(n_store)
    corr_e_eps_draws = np.zeros(n_store)
    corr_u_eps_draws = np.zeros(n_store)
    Sigma_draws = np.zeros((n_store, 4, 4))
    Nbar_draws = np.zeros((n_store, T))
    Nhat_draws = np.zeros((n_store, T))
    psi_draws = np.zeros((n_store, ma_order))
    e_acf_draws = np.zeros((n_store, ma_order))

    total_iter = n_burn + n_keep
    store_idx = 0

    def innovation(disturbance: np.ndarray) -> np.ndarray:
        """psi(L)^{-1} applied to the disturbance, seeded with the sampled pre-sample."""
        return inverse_ma_filter(disturbance, psi, psi_presample) if ma_order else disturbance

    for it in range(1, total_iter + 1):
        zeta = x_t - phi_1 * x_tm1
        u, eps = _compute_state_residuals(states, rho1, rho2, n_drift)

        # ---- 1. beta = (alpha, kappa, theta), GLS under the banded covariance ----
        mean_v, var_v = _conditional_e_all(Sigma, zeta, u, eps)
        if ma_order:
            weighting = MAWeighting(
                psi, T, innovation_var=var_v, presample_var=float(Sigma[0, 0])
            )
            # Pre-sample innovations have unconditional mean zero.
            mean_xi = ma_filter(mean_v, psi, presample=0.0)
        else:
            weighting = MAWeighting(psi, T, innovation_var=var_v)
            mean_xi = mean_v

        X = np.column_stack([a_t, x_t / KAPPA_SCALE, -Nhat])
        beta = draw_with_constraints(
            lambda: _sample_beta_gls(
                y=y - mean_xi,
                X=X,
                weighting=weighting,
                prior_mean=np.array(
                    [pri["mu_alpha"], pri["mu_kappa"], pri["mu_theta"]], dtype=float
                ),
                prior_var=np.array(
                    [pri["sigma_alpha"] ** 2, pri["sigma_kappa"] ** 2, pri["sigma_theta"] ** 2],
                    dtype=float,
                ),
                rng=rng,
            ),
            ("alpha", "kappa", "theta"),
            coefficient_constraints,
            stats=constraint_stats,
        )
        alpha, kappa, theta = (float(beta[0]), float(beta[1]), float(beta[2]))

        kappa_eff = kappa / KAPPA_SCALE
        e = y - alpha * a_t - kappa_eff * x_t + theta * Nhat
        v = innovation(e)

        # ---- 2. phi_1 ----
        phi_1 = _sample_phi_full(
            x_t=x_t, x_tm1=x_tm1, e=v, u=u, eps=eps, Sigma=Sigma,
            mu_phi=pri["mu_phi"], sigma_phi=pri["sigma_phi"], rng=rng,
        )
        zeta = x_t - phi_1 * x_tm1

        # ---- 3. rho1, rho2 ----
        rho1, rho2 = _sample_ar2_coeffs_full(
            states=states, e=v, zeta=zeta, eps=eps, Sigma=Sigma,
            mu_rho1=pri["mu_rho1"], sigma_rho1=pri["sigma_rho1"],
            mu_rho2=pri["mu_rho2"], sigma_rho2=pri["sigma_rho2"],
            enforce_stationary=enforce_stationary, rng=rng,
            max_tries=ar2_max_tries, current=(rho1, rho2), stats=ar2_stats,
        )
        u, eps = _compute_state_residuals(states, rho1, rho2, n_drift)

        # ---- 4. n ----
        n_drift = _sample_n_full(
            states=states, e=v, zeta=zeta, u=u, Sigma=Sigma,
            mu_n=pri["mu_n"], sigma_n=pri["sigma_n"], rng=rng,
        )
        u, eps = _compute_state_residuals(states, rho1, rho2, n_drift)

        # ---- 5. Sigma, over [v, zeta, u, epsilon] ----
        Sigma = _sample_Sigma(
            e=v, zeta=zeta, u=u, eps=eps, nu0=pri["nu_Sigma"], S0=S_Sigma,
            structure=covariance_structure, rng=rng, current_Sigma=Sigma,
        )

        # ---- 6. sigma_N2 ----
        resid_N = finite_N_residuals(N_obs, Nhat, Nbar)
        sigma_N2 = _sample_invgamma(
            pri["a_N"] + 0.5 * resid_N.size,
            pri["b_N"] + 0.5 * float(np.sum(resid_N**2)),
            rng,
        )

        # ---- 7. joint state draw, MA innovation carried in the state ----
        Nbar, Nhat, states_aug, v_path = sample_joint_states_ffbs_dynamic_ma(
            N_obs=N_obs, pi_t=pi_t, pi_tm1=pi_tm1, pi_expect=pi_expect, x_t=x_t,
            zeta=zeta, alpha=alpha, kappa=kappa, theta=theta,
            rho1=rho1, rho2=rho2, n_drift=n_drift, Sigma=Sigma, psi=psi,
            sigma_N2=sigma_N2, kappa_scale=KAPPA_SCALE, m0=m0, P0=P0, rng=rng,
        )
        states = states_aug[:, :3]
        if ma_order:
            # states_aug[0, 4:] is [v_{-1}, ..., v_{-q}] as sampled.
            psi_presample = states_aug[0, 4:].copy()

        # ---- 8. psi, last ----
        if ma_order:
            kappa_eff = kappa / KAPPA_SCALE
            xi_post = y - alpha * a_t - kappa_eff * x_t + theta * Nhat
            mean_v_post, var_v_post = _conditional_e_all(Sigma, zeta, *_compute_state_residuals(
                states, rho1, rho2, n_drift
            ))
            centred = xi_post - ma_filter(mean_v_post, psi, presample=0.0)
            current_post = None
            for _ in range(max(1, n_psi_steps)):
                if current_post is None:
                    w_cur = MAWeighting(
                        psi, T, innovation_var=var_v_post, presample_var=float(Sigma[0, 0])
                    )
                    current_post = w_cur.log_likelihood(centred, 1.0) + psi_prior.log_pdf(psi)
                candidate = proposal.propose(psi, rng)
                accepted = False
                if is_invertible(candidate):
                    try:
                        w_can = MAWeighting(
                            candidate, T, innovation_var=var_v_post,
                            presample_var=float(Sigma[0, 0]),
                        )
                        cand_centred = xi_post - ma_filter(mean_v_post, candidate, presample=0.0)
                        cand_post = (
                            w_can.log_likelihood(cand_centred, 1.0)
                            + psi_prior.log_pdf(candidate)
                        )
                    except (ValueError, np.linalg.LinAlgError):
                        cand_post = -np.inf
                    if np.isfinite(cand_post) and np.log(rng.random()) < cand_post - current_post:
                        psi, current_post, centred = candidate, cand_post, cand_centred
                        accepted = True
                proposal.register(psi, accepted)
            if it == n_burn:
                proposal.freeze()

        # ---- moments of the reduced-form disturbance ----
        gamma = autocovariance(psi, float(Sigma[0, 0]))
        sigma_e2 = gamma[0]

        if it > n_burn and (it - n_burn) % store_every == 0:
            alpha_draws[store_idx] = alpha
            kappa_draws[store_idx] = kappa / KAPPA_SCALE
            theta_draws[store_idx] = theta
            phi_draws[store_idx] = phi_1
            rho1_draws[store_idx] = rho1
            rho2_draws[store_idx] = rho2
            n_draws[store_idx] = n_drift
            sigma_N_draws[store_idx] = np.sqrt(sigma_N2)
            sigma_e_draws[store_idx] = np.sqrt(sigma_e2)
            sigma_v_draws[store_idx] = np.sqrt(Sigma[0, 0])
            sigma_zeta_draws[store_idx] = np.sqrt(Sigma[1, 1])
            sigma_u_draws[store_idx] = np.sqrt(Sigma[2, 2])
            sigma_eps_draws[store_idx] = np.sqrt(Sigma[3, 3])
            corr_e_zeta_draws[store_idx] = Sigma[0, 1] / np.sqrt(Sigma[0, 0] * Sigma[1, 1])
            corr_e_u_draws[store_idx] = Sigma[0, 2] / np.sqrt(Sigma[0, 0] * Sigma[2, 2])
            corr_e_eps_draws[store_idx] = Sigma[0, 3] / np.sqrt(Sigma[0, 0] * Sigma[3, 3])
            corr_u_eps_draws[store_idx] = Sigma[2, 3] / np.sqrt(Sigma[2, 2] * Sigma[3, 3])
            Sigma_draws[store_idx] = Sigma
            Nbar_draws[store_idx] = Nbar
            Nhat_draws[store_idx] = Nhat
            if ma_order:
                psi_draws[store_idx] = psi
                e_acf_draws[store_idx] = gamma[1:] / sigma_e2
            store_idx += 1

        if progress_callback is not None:
            progress_callback(it, total_iter)

        if verbose and it % 2000 == 0:
            acc = proposal.acceptance_rate if proposal is not None else float("nan")
            print(
                f"Iter {it}/{total_iter}: alpha={alpha:.3f}, kappa={kappa:.3f}, "
                f"theta={theta:.3f}, psi={np.round(psi, 3)}, accept={acc:.2f}"
            )

    error_structure: dict[str, Any] = {
        "family": "ma" if ma_order else "iid",
        "order": ma_order,
        "state_dim": 3 + ma_order + 1,
        "sigma_order_note": "Sigma's first coordinate is the MA innovation v_t, not e_t",
    }
    if ma_order and proposal is not None:
        error_structure["psi_acceptance_rate"] = proposal.acceptance_rate
        error_structure["psi_proposal_scale"] = float(np.exp(proposal.log_scale))
        error_structure["psi_metropolis_steps_per_sweep"] = n_psi_steps

    out: dict[str, Any] = {
        "alpha": _summary(alpha_draws),
        "kappa": _summary(kappa_draws),
        "theta": _summary(theta_draws),
        "phi_1": _summary(phi_draws),
        "rho1": _summary(rho1_draws),
        "rho2": _summary(rho2_draws),
        "n": _summary(n_draws),
        "sigma_N": _summary(sigma_N_draws),
        "sigma_e": _summary(sigma_e_draws),
        "sigma_v": _summary(sigma_v_draws),
        "sigma_zeta": _summary(sigma_zeta_draws),
        "sigma_u": _summary(sigma_u_draws),
        "sigma_eps": _summary(sigma_eps_draws),
        "corr_e_zeta": _summary(corr_e_zeta_draws),
        "corr_e_u": _summary(corr_e_u_draws),
        "corr_e_eps": _summary(corr_e_eps_draws),
        "corr_u_eps": _summary(corr_u_eps_draws),
        "Sigma": _summary(Sigma_draws),
        "state_draws": {"Nbar": Nbar_draws, "Nhat": Nhat_draws},
        "priors": priors or {},
        "opts": opts,
        "model": {
            "inflation": (
                "pi_t = alpha*pi_{t-1} + (1-alpha)*E_t*pi_{t+1} + kappa*x_t "
                "- theta*Nhat_t + psi(L)v_t"
            ),
            "x": "x_t = phi_1*x_{t-1} + zeta_t",
            "measurement": "N_obs_t = Nhat_t + Nbar_t + nu_t",
            "gap": "Nhat_t = rho1*Nhat_{t-1} + rho2*Nhat_{t-2} + u_t",
            "trend": "Nbar_t = n + Nbar_{t-1} + epsilon_t",
            "Sigma_order": "[v_t, zeta_t, u_t, epsilon_t]",
            "covariance_structure": covariance_structure,
            "covariance_update": (
                "restricted_conjugate"
                if covariance_structure != "full"
                else "iw_missing_shock_augmentation"
            ),
            "nu_independent": True,
            "state_sampler": "joint_ffbs_ma",
            "state_vector": (
                "[Nhat_t, Nhat_{t-1}, Nbar_t, v_t, ..., v_{t-%d}]'" % ma_order
                if ma_order
                else "[Nhat_t, Nhat_{t-1}, Nbar_t, v_t]'"
            ),
            "kappa_scale": KAPPA_SCALE,
            "kappa_internal": "stored kappa * KAPPA_SCALE",
            "stored_units": "physical",
            "coefficient_constraints": coefficient_constraints,
            "coefficient_constraint_stats": constraint_stats_summary(constraint_stats),
            "ar2_stationarity": {
                "enforce_stationary": enforce_stationary,
                "max_tries": ar2_max_tries,
                **_ar2_stats_summary(ar2_stats),
            },
            "error_structure": error_structure,
        },
        "error_structure": error_structure,
    }
    if ma_order:
        out["psi"] = _summary(psi_draws)
        out["e_acf"] = _summary(e_acf_draws)
    return out
