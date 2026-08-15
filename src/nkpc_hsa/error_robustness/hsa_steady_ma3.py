"""HSA steady-state NKPC with an MA(3) inflation disturbance.

The additive counterpart of
``nkpc_hsa.gibbs.hsa_steady.model.func_nkpc_hsa_decomp_tv_kappa_kalman``, which
is left untouched. The model is identical except for the disturbance:

    pi_t     = alpha*pi_{t-1} + (1-alpha)*E_t pi_{t+1} + kappa_t*x_t + e_t
    kappa_t  = kappa_0 + delta*Nbar_t
    x_t      = phi_1*x_{t-1} + zeta_t
    N_obs_t  = Nhat_t + Nbar_t + nu_t
    Nhat_t   = rho_1*Nhat_{t-1} + rho_2*Nhat_{t-2} + u_t
    Nbar_t   = n + Nbar_{t-1} + epsilon_t

    e_t      = lambda_ez*zeta_t + xi_t
    xi_t     = psi(L) v_t,   v_t ~ iid N(0, sigma_v^2)      <-- the only change

Every coefficient block is the production block with ``sum`` replaced by the
GLS form under ``Omega_0(psi)``; with ``psi = 0`` the algebra collapses back
exactly. The state draw needs more than a reweighting, because a serially
dependent disturbance cannot live in the measurement equation -- see
``joint_ffbs_ma3``.

Relation to the ``no_inertia`` restriction
------------------------------------------
The production sampler already flags four-quarter overlap as a reason a large
``alpha`` may be an artefact, and offers ``no_inertia`` (alpha = 0) as the
response. That restriction removes the contaminated regressor. Modelling the
disturbance instead removes the contamination, which is the treatment the
overlap actually calls for: on a recursive NKPC design with ``psi =
(0.45, 0.30, 0.55)`` the i.i.d. sampler overstates ``alpha`` by +0.22 and
attenuates ``kappa`` by 71%, while the MA(3) sampler is close to unbiased in
both. ``production/main_scripts/error_robustness`` reproduces that experiment.

Block order
-----------
``beta`` -> ``lambda_ez`` -> ``phi_1`` -> ``sigma_zeta2`` -> ``sigma_v2`` ->
``rho`` -> ``sigma_u2`` -> ``n`` -> ``sigma_eps2`` -> ``sigma_N2`` ->
``states`` -> ``psi``.

``psi`` is last on purpose: it is the only Metropolis block, and Chib's final
ordinate factor is the one evaluated with every other block at its starred
value, so a single numerical normalisation of ``p(psi* | .)`` replaces a
Chib-Jeliazkov correction inside every reduced run.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
from numpy.linalg import inv

from nkpc_hsa.error_robustness.joint_ffbs_ma3 import sample_joint_states_ffbs_ma
from nkpc_hsa.error_robustness.ma_error import (
    MA_ORDER,
    AdaptiveRandomWalk,
    MAWeighting,
    PsiPrior,
    autocovariance,
    sample_psi,
)
from nkpc_hsa.gibbs.common.competition import finite_N_residuals
from nkpc_hsa.gibbs.common.constraints import draw_with_constraints
from nkpc_hsa.gibbs.hsa_steady.model import (
    KAPPA_SCALE,
    _ar2_stats_summary,
    _as_1d,
    _assert_all_pos,
    _common_priors,
    _getd,
    _init_states,
    _kappa_t_constraint_validators,
    _mvnrnd,
    _sample_ar2_coeffs,
    _sample_invgamma,
    _summary,
)

__all__ = ["func_nkpc_hsa_steady_ma3"]

_FIXED_BLOCKS = {
    "beta", "lambda_ez", "phi_1", "sigma_zeta2", "sigma_v2", "sigma_eta2",
    "rho", "sigma_u2", "n", "sigma_eps2", "sigma_N2", "psi",
}


def _sample_beta_gls(
    y: np.ndarray,
    X: np.ndarray,
    sigma2: float,
    prior_mean: np.ndarray,
    prior_var: np.ndarray,
    weighting: MAWeighting,
    rng: np.random.Generator,
) -> np.ndarray:
    """Production ``_sample_beta_gaussian`` with ``X'X`` replaced by ``X' Omega_0^{-1} X``."""
    y = _as_1d(y)
    X = np.asarray(X, dtype=float)
    prior_mean = _as_1d(prior_mean)
    prior_var = _as_1d(prior_var)

    if X.ndim != 2:
        raise ValueError("X must be 2D.")
    if X.shape[0] != y.size:
        raise ValueError("X and y lengths do not match.")
    if X.shape[1] != prior_mean.size or prior_mean.size != prior_var.size:
        raise ValueError("Prior dimensions do not match X.")

    _assert_all_pos(prior_var, "Prior variances must be positive.")
    _assert_all_pos([sigma2], "sigma2 must be positive.")

    XtWX, XtWy = weighting.gls_moments(y, X)
    V0_inv = np.diag(1.0 / prior_var)
    Vn = inv(XtWX / sigma2 + V0_inv)
    mn = Vn @ (XtWy / sigma2 + V0_inv @ prior_mean)
    return _mvnrnd(mn, Vn, rng)


def func_nkpc_hsa_steady_ma3(
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
    *,
    orth: bool = False,
) -> dict[str, Any]:
    """Gibbs/Metropolis sampler for hsa_steady with an MA(q) inflation disturbance.

    ``opts`` additions over the production sampler
    ---------------------------------------------
    ``ma_order``        MA order, default 3 (0 gives the i.i.d. specification)
    ``psi0``            starting value, default zeros
    ``n_psi_steps``     Metropolis steps per sweep, default 2
    ``psi_init_scale``  initial random-walk scale, default 0.08
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
        raise ValueError("Need T >= 3 for the AR(2) gap equation.")
    if n_burn < 0:
        raise ValueError("n_burn must be nonnegative.")
    if n_keep <= 0:
        raise ValueError("n_keep must be positive.")

    pri = _common_priors(priors or {})
    _assert_all_pos(
        [
            pri["sigma_alpha"], pri["sigma_kappa0"], pri["sigma_delta"], pri["sigma_phi"],
            pri["sigma_lambda"], pri["sigma_rho1"], pri["sigma_rho2"], pri["sigma_n"],
            pri["a_e"], pri["b_e"], pri["a_u"], pri["b_u"], pri["a_eps"], pri["b_eps"],
            pri["a_z"], pri["b_z"], pri["a_N"], pri["b_N"],
            pri["P0_Nhat"], pri["P0_Nhat_lag"], pri["P0_Nbar"],
        ],
        "Prior scales and inverse-gamma hyperparameters must be positive.",
    )

    opts = opts or {}
    rng = np.random.default_rng(_getd(opts, "seed", None))

    ma_order = int(_getd(opts, "ma_order", MA_ORDER))
    if ma_order < 0:
        raise ValueError("ma_order must be nonnegative.")
    psi_prior = PsiPrior.from_config(priors, order=ma_order) if ma_order else None

    alpha = float(_getd(opts, "alpha0", pri["mu_alpha"]))
    kappa0 = float(_getd(opts, "kappa0", pri["mu_kappa0"]))
    delta = float(_getd(opts, "delta0", pri["mu_delta"]))
    phi_1 = float(_getd(opts, "phi10", pri["mu_phi"]))
    lambda_ez = 0.0 if orth else float(_getd(opts, "lambda0", pri["mu_lambda"]))
    rho1 = float(_getd(opts, "rho10", pri["mu_rho1"]))
    rho2 = float(_getd(opts, "rho20", pri["mu_rho2"]))
    n_drift = float(_getd(opts, "n0", pri["mu_n"]))

    sigma_v2 = float(_getd(opts, "sigma_v20", _getd(opts, "sigma_e20", 1.0)))
    sigma_zeta2 = float(_getd(opts, "sigma_zeta20", 1.0))
    sigma_u2 = float(_getd(opts, "sigma_u20", 1.0))
    sigma_eps2 = float(_getd(opts, "sigma_eps20", 1.0))
    sigma_N2 = float(_getd(opts, "sigma_N20", _getd(opts, "sigma_m20", 1.0)))
    _assert_all_pos(
        [sigma_v2, sigma_zeta2, sigma_u2, sigma_eps2, sigma_N2],
        "Initial variances must be positive.",
    )

    psi = np.asarray(_getd(opts, "psi0", np.zeros(ma_order)), dtype=float).reshape(-1)
    if psi.size != ma_order:
        raise ValueError(f"psi0 must have length ma_order={ma_order}.")
    n_psi_steps = int(_getd(opts, "n_psi_steps", 2))

    no_inertia = bool(_getd(opts, "no_inertia", False))
    enforce_stationary = bool(_getd(opts, "enforce_stationary", True))
    ar2_max_tries = int(max(1, _getd(opts, "ar2_max_tries", 2000)))
    store_every = int(max(1, _getd(opts, "store_every", 1)))
    verbose = bool(_getd(opts, "verbose", False))
    # Display-only hook installed by the run driver; it never touches the draws.
    progress_callback = _getd(opts, "progress_callback", None)
    coefficient_constraints = _getd(opts, "coefficient_constraints", {})
    constraint_stats: dict[str, int] = {}
    ar2_stats: dict[str, int] = {}

    fixed = dict(_getd(opts, "fixed", {}) or {})
    unknown_fixed = set(fixed) - _FIXED_BLOCKS
    if unknown_fixed:
        raise ValueError(f"Unknown fixed block(s): {sorted(unknown_fixed)}")
    if "beta" in fixed:
        alpha, kappa0, delta = (float(v) for v in fixed["beta"])
    if "lambda_ez" in fixed:
        lambda_ez = float(fixed["lambda_ez"])
    if "phi_1" in fixed:
        phi_1 = float(fixed["phi_1"])
    if "rho" in fixed:
        rho1, rho2 = (float(v) for v in fixed["rho"])
    if "sigma_zeta2" in fixed:
        sigma_zeta2 = float(fixed["sigma_zeta2"])
    # "sigma_eta2" is accepted as an alias so Chib reduced runs written against
    # the production block names keep working.
    if "sigma_v2" in fixed or "sigma_eta2" in fixed:
        sigma_v2 = float(fixed.get("sigma_v2", fixed.get("sigma_eta2")))
    if "sigma_u2" in fixed:
        sigma_u2 = float(fixed["sigma_u2"])
    if "sigma_eps2" in fixed:
        sigma_eps2 = float(fixed["sigma_eps2"])
    if "sigma_N2" in fixed:
        sigma_N2 = float(fixed["sigma_N2"])
    if "n" in fixed:
        n_drift = float(fixed["n"])
    if "psi" in fixed:
        psi = np.asarray(fixed["psi"], dtype=float).reshape(-1)

    n_store = int(n_keep // store_every)
    if n_store <= 0:
        raise ValueError("No draws would be stored. Use n_keep >= store_every.")

    Nbar, Nhat = _init_states(N_obs)
    a_t = pi_tm1 - pi_expect
    y = pi_t - pi_expect
    lambda_prec0 = 0.0 if orth else 1.0 / pri["sigma_lambda"] ** 2

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

    states = np.zeros((T, 3), dtype=float)
    states[:, 0] = Nhat
    states[:, 2] = Nbar
    states[0, 1] = m0[1]
    if T > 1:
        states[1:, 1] = Nhat[:-1]

    proposal = (
        AdaptiveRandomWalk(ma_order, init_scale=float(_getd(opts, "psi_init_scale", 0.08)))
        if ma_order
        else None
    )
    weighting = MAWeighting(psi, T)

    alpha_draws = np.zeros(n_store)
    kappa0_draws = np.zeros(n_store)
    delta_draws = np.zeros(n_store)
    phi_draws = np.zeros(n_store)
    lambda_draws = np.zeros(n_store)
    rho1_draws = np.zeros(n_store)
    rho2_draws = np.zeros(n_store)
    n_draws = np.zeros(n_store)
    sigma_e_draws = np.zeros(n_store)
    sigma_v_draws = np.zeros(n_store)
    sigma_zeta_draws = np.zeros(n_store)
    sigma_u_draws = np.zeros(n_store)
    sigma_eps_draws = np.zeros(n_store)
    sigma_N_draws = np.zeros(n_store)
    rho_ez_draws = np.zeros(n_store)
    Nbar_draws = np.zeros((n_store, T))
    Nhat_draws = np.zeros((n_store, T))
    kappa_t_draws = np.zeros((n_store, T))
    nhat_lag_draws = np.zeros(n_store)
    psi_draws = np.zeros((n_store, ma_order))
    e_acf_draws = np.zeros((n_store, ma_order))

    total_iter = n_burn + n_keep
    store_idx = 0

    for it in range(1, total_iter + 1):
        # ---- 1. beta = (alpha, kappa_0, delta), GLS under Omega_0(psi) ----
        zeta = x_t - phi_1 * x_tm1
        y_adj = y - lambda_ez * zeta

        columns = [] if no_inertia else [a_t]
        columns += [x_t / KAPPA_SCALE, (x_t * Nbar) / KAPPA_SCALE]
        X = np.column_stack(columns)

        prior_means = [] if no_inertia else [pri["mu_alpha"]]
        prior_means += [pri["mu_kappa0"], pri["mu_delta"]]
        prior_vars = [] if no_inertia else [pri["sigma_alpha"] ** 2]
        prior_vars += [pri["sigma_kappa0"] ** 2, pri["sigma_delta"] ** 2]
        beta_names = ("kappa_0", "delta") if no_inertia else ("alpha", "kappa_0", "delta")

        if "beta" not in fixed:
            beta = draw_with_constraints(
                lambda: _sample_beta_gls(
                    y=y_adj,
                    X=X,
                    sigma2=sigma_v2,
                    prior_mean=np.array(prior_means, dtype=float),
                    prior_var=np.array(prior_vars, dtype=float),
                    weighting=weighting,
                    rng=rng,
                ),
                beta_names,
                coefficient_constraints,
                validators=_kappa_t_constraint_validators(
                    Nbar, coefficient_constraints, offset=0 if no_inertia else 1
                ),
                stats=constraint_stats,
            )
            if no_inertia:
                alpha = 0.0
                kappa0 = float(beta[0])
                delta = float(beta[1])
            else:
                alpha = float(beta[0])
                kappa0 = float(beta[1])
                delta = float(beta[2])

        kappa_t = kappa0 + delta * Nbar
        kappa_t_eff = kappa_t / KAPPA_SCALE

        # ---- 2. cross-equation loading, GLS ----
        if orth:
            lambda_ez = 0.0
        elif "lambda_ez" not in fixed:
            e_base = y - alpha * a_t - kappa_t_eff * x_t
            Wzeta = weighting.solve(zeta)
            post_var_lambda = 1.0 / (lambda_prec0 + float(zeta @ Wzeta) / sigma_v2)
            post_mean_lambda = post_var_lambda * (
                pri["mu_lambda"] * lambda_prec0 + float(e_base @ Wzeta) / sigma_v2
            )
            lambda_ez = float(
                post_mean_lambda + np.sqrt(post_var_lambda) * rng.standard_normal()
            )

        # ---- 3. phi_1, GLS on the inflation-equation contribution ----
        y_tilde_phi = y - alpha * a_t - kappa_t_eff * x_t
        if "phi_1" not in fixed:
            Wx_prev = weighting.solve(x_tm1)
            prec = (
                1.0 / pri["sigma_phi"] ** 2
                + float(np.sum(x_tm1**2)) / sigma_zeta2
                + (lambda_ez**2) * float(x_tm1 @ Wx_prev) / sigma_v2
            )
            mean_num = (
                pri["mu_phi"] / pri["sigma_phi"] ** 2
                + float(np.dot(x_tm1, x_t)) / sigma_zeta2
                - lambda_ez * float((y_tilde_phi - lambda_ez * x_t) @ Wx_prev) / sigma_v2
            )
            phi_1 = float(mean_num / prec + rng.standard_normal() / np.sqrt(prec))

        # ---- 4. sigma_zeta2 and sigma_v2 ----
        zeta = x_t - phi_1 * x_tm1
        xi = y - alpha * a_t - kappa_t_eff * x_t - lambda_ez * zeta

        if "sigma_zeta2" not in fixed:
            sigma_zeta2 = _sample_invgamma(
                pri["a_z"] + 0.5 * T,
                pri["b_z"] + 0.5 * float(np.sum(zeta**2)),
                rng,
            )
        if not ("sigma_v2" in fixed or "sigma_eta2" in fixed):
            # Rao-Blackwellised: the banded quadratic form integrates the
            # innovation path out instead of reusing the FFBS draw of it.
            sigma_v2 = _sample_invgamma(
                pri["a_e"] + 0.5 * T,
                pri["b_e"] + 0.5 * weighting.quadratic_form(xi),
                rng,
            )

        # ---- 5. rho1, rho2, sigma_u2 ----
        if "rho" not in fixed:
            rho1, rho2 = _sample_ar2_coeffs(
                Nhat=Nhat,
                sigma_state2=sigma_u2,
                mu_rho1=pri["mu_rho1"],
                sigma_rho1=pri["sigma_rho1"],
                mu_rho2=pri["mu_rho2"],
                sigma_rho2=pri["sigma_rho2"],
                enforce_stationary=enforce_stationary,
                rng=rng,
                max_tries=ar2_max_tries,
                current=(rho1, rho2),
                stats=ar2_stats,
                initial_lag=float(states[0, 1]),
            )
        resid_u = states[1:, 0] - rho1 * states[:-1, 0] - rho2 * states[:-1, 1]
        if "sigma_u2" not in fixed:
            sigma_u2 = _sample_invgamma(
                pri["a_u"] + 0.5 * resid_u.size,
                pri["b_u"] + 0.5 * float(np.sum(resid_u**2)),
                rng,
            )

        # ---- 6. n and sigma_eps2 ----
        dNbar = Nbar[1:] - Nbar[:-1]
        post_var_n = 1.0 / (1.0 / pri["sigma_n"] ** 2 + dNbar.size / sigma_eps2)
        post_mean_n = post_var_n * (
            pri["mu_n"] / pri["sigma_n"] ** 2 + float(np.sum(dNbar)) / sigma_eps2
        )
        if "n" not in fixed:
            n_drift = float(post_mean_n + np.sqrt(post_var_n) * rng.standard_normal())
        resid_eps = Nbar[1:] - n_drift - Nbar[:-1]
        if "sigma_eps2" not in fixed:
            sigma_eps2 = _sample_invgamma(
                pri["a_eps"] + 0.5 * resid_eps.size,
                pri["b_eps"] + 0.5 * float(np.sum(resid_eps**2)),
                rng,
            )

        # ---- 7. sigma_N2 ----
        resid_N = finite_N_residuals(N_obs, Nhat, Nbar)
        if "sigma_N2" not in fixed:
            sigma_N2 = _sample_invgamma(
                pri["a_N"] + 0.5 * resid_N.size,
                pri["b_N"] + 0.5 * float(np.sum(resid_N**2)),
                rng,
            )

        # ---- 8. joint state draw, MA block carried inside the state ----
        y_tilde = (
            pi_t - pi_expect
            - alpha * (pi_tm1 - pi_expect)
            - (kappa0 / KAPPA_SCALE) * x_t
            - lambda_ez * zeta
        )
        Nbar, Nhat, states_aug, _v_path = sample_joint_states_ffbs_ma(
            N_obs=N_obs,
            y_tilde=y_tilde,
            h_nhat=np.zeros(T, dtype=float),
            h_nbar=(delta / KAPPA_SCALE) * x_t,
            n_drift=n_drift,
            rho1=rho1,
            rho2=rho2,
            psi=psi,
            sigma_v2=sigma_v2,
            sigma_u2=sigma_u2,
            sigma_eps2=sigma_eps2,
            sigma_N2=sigma_N2,
            m0=m0,
            P0=P0,
            rng=rng,
        )
        states = states_aug[:, :3]
        kappa_t = kappa0 + delta * Nbar

        # ---- 9. psi, last: Metropolis on the banded likelihood ----
        if ma_order and "psi" not in fixed:
            kappa_t_eff = kappa_t / KAPPA_SCALE
            xi_post = y - alpha * a_t - kappa_t_eff * x_t - lambda_ez * zeta
            psi, weighting = sample_psi(
                psi,
                xi_post,
                sigma_v2,
                prior=psi_prior,
                proposal=proposal,
                rng=rng,
                n_steps=n_psi_steps,
                weighting=weighting,
            )
            if it == n_burn:
                proposal.freeze()

        gamma = autocovariance(psi, sigma_v2)
        sigma_e2 = lambda_ez**2 * sigma_zeta2 + gamma[0]
        sigma_e = float(np.sqrt(sigma_e2))
        rho_ez = 0.0 if orth else float(
            (lambda_ez * np.sqrt(sigma_zeta2)) / max(sigma_e, 1e-12)
        )

        if it > n_burn and (it - n_burn) % store_every == 0:
            # Stored in physical units, matching the production schema: the
            # sampler works on the KAPPA_SCALE-multiplied regression columns.
            alpha_draws[store_idx] = alpha
            kappa0_draws[store_idx] = kappa0 / KAPPA_SCALE
            delta_draws[store_idx] = delta / KAPPA_SCALE
            phi_draws[store_idx] = phi_1
            lambda_draws[store_idx] = lambda_ez
            rho1_draws[store_idx] = rho1
            rho2_draws[store_idx] = rho2
            n_draws[store_idx] = n_drift
            sigma_e_draws[store_idx] = sigma_e
            sigma_v_draws[store_idx] = np.sqrt(sigma_v2)
            sigma_zeta_draws[store_idx] = np.sqrt(sigma_zeta2)
            sigma_u_draws[store_idx] = np.sqrt(sigma_u2)
            sigma_eps_draws[store_idx] = np.sqrt(sigma_eps2)
            sigma_N_draws[store_idx] = np.sqrt(sigma_N2)
            rho_ez_draws[store_idx] = rho_ez
            Nbar_draws[store_idx] = Nbar
            Nhat_draws[store_idx] = Nhat
            kappa_t_draws[store_idx] = kappa_t / KAPPA_SCALE
            nhat_lag_draws[store_idx] = float(states[0, 1])
            if ma_order:
                psi_draws[store_idx] = psi
                e_acf_draws[store_idx] = gamma[1:] / sigma_e2
            store_idx += 1

        if progress_callback is not None:
            progress_callback(it, total_iter)

        if verbose and it % 2000 == 0:
            acc = proposal.acceptance_rate if proposal is not None else float("nan")
            print(
                f"Iter {it}/{total_iter}: alpha={alpha:.3f}, kappa0={kappa0:.3f}, "
                f"delta={delta:.4f}, psi={np.round(psi, 3)}, accept={acc:.2f}"
            )

    # The return schema mirrors the production sampler exactly -- same keys,
    # same units, same nesting -- so anything downstream that reads a run
    # directory works unchanged against an error_robustness run. ``sigma_eta``
    # keeps its slot but now means the MA *innovation* sd; ``sigma_e`` is still
    # the sd of the reduced-form disturbance and now includes the MA
    # amplification. New keys (``psi``, ``e_acf``) are additions, not renames.
    error_structure: dict[str, Any] = {
        "family": "ma" if ma_order else "iid",
        "order": ma_order,
        "state_dim": 3 + ma_order + 1,
        "sigma_eta_meaning": (
            "sd of the MA innovation v_t" if ma_order else "sd of the iid disturbance"
        ),
    }
    if ma_order and proposal is not None:
        error_structure["psi_acceptance_rate"] = proposal.acceptance_rate
        error_structure["psi_proposal_scale"] = float(np.exp(proposal.log_scale))
        error_structure["psi_metropolis_steps_per_sweep"] = n_psi_steps

    out: dict[str, Any] = {
        "alpha": _summary(alpha_draws),
        "kappa_0": _summary(kappa0_draws),
        "delta": _summary(delta_draws),
        "phi_1": _summary(phi_draws),
        "lambda_ez": _summary(lambda_draws),
        "rho": _summary(rho_ez_draws),
        "rho1": _summary(rho1_draws),
        "rho2": _summary(rho2_draws),
        "n": _summary(n_draws),
        "sigma_e": _summary(sigma_e_draws),
        "sigma_eta": _summary(sigma_v_draws),
        "sigma_zeta": _summary(sigma_zeta_draws),
        "sigma_u": _summary(sigma_u_draws),
        "sigma_eps": _summary(sigma_eps_draws),
        "sigma_N": _summary(sigma_N_draws),
        "state_draws": {
            "Nbar": Nbar_draws,
            "Nhat": Nhat_draws,
            "kappa_t": kappa_t_draws,
            **(
                {"Nhat_lag": nhat_lag_draws}
                if bool(_getd(opts, "return_state_lag", False))
                else {}
            ),
        },
        "priors": priors or {},
        "opts": opts,
        "model": {
            "N_measurement_error": True,
            "N_measurement_equation": "N_obs_t = Nhat_t + Nbar_t + measurement_error_t",
            "state_sampler": "joint_ffbs_ma",
            "theta_sampled": False,
            "inflation_equation": (
                "y_t = kappa_t*x_t + lambda_ez*zeta_t + psi(L)v_t  (alpha restricted to 0)"
                if no_inertia
                else "y_t = alpha*a_t + kappa_t*x_t + lambda_ez*zeta_t + psi(L)v_t"
            ),
            "no_inertia": no_inertia,
            "state_vector": (
                "[Nhat_t, Nhat_{t-1}, Nbar_t, v_t, ..., v_{t-%d}]'" % ma_order
                if ma_order
                else "[Nhat_t, Nhat_{t-1}, Nbar_t, v_t]'"
            ),
            "kappa_scale": KAPPA_SCALE,
            "kappa_internal": "stored kappa_0, delta, and kappa_t multiplied by KAPPA_SCALE",
            "stored_units": "physical",
            "coefficient_constraints": coefficient_constraints,
            "coefficient_constraint_stats": constraint_stats,
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
