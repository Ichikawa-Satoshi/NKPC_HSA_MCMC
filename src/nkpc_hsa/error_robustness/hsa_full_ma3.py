"""``hsa_full`` (Particle Gibbs) with an MA(3) inflation disturbance.

The additive counterpart of
``nkpc_hsa.gibbs.hsa_full_pg.model.func_nkpc_hsa_full_pg``, which is left
untouched. The model is identical except for the disturbance:

    y_t = alpha*a_t + kappa0*x_t + delta*x_t*Nbar_t
          - theta0*Nhat_t - gamma*Nbar_t*Nhat_t + lambda_ez*zeta_t + xi_t
    xi_t = psi(L) v_t,   v_t ~ iid N(0, sigma_v^2)          <-- the only change

Every coefficient block is the production block with the row-by-row weight
``1/sigma_eta2`` replaced by ``Omega_0(psi)^{-1}/sigma_v2``. The state block is
the conditional-SMC sweep in ``particle_gibbs_ma3``, which solves for ``v_t``
rather than proposing it -- see that module for why a naive augmentation would
give every particle zero weight.

``psi`` is last in the block order, for the Chib reason documented in
``ma_error``. With ``psi = 0`` and ``ma_order = 0`` the sampler reduces to
production, and the state block reduces to it bit for bit.

Scope note: ``static_theta`` is supported exactly as in production. The
identification problems ``gamma`` has in this model are orthogonal to the error
structure and are not addressed here.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
from numpy.linalg import inv

from nkpc_hsa.error_robustness.ma_error import (
    MA_ORDER,
    AdaptiveRandomWalk,
    MAWeighting,
    PsiPrior,
    autocovariance,
    is_invertible,
    sample_psi,
)
from nkpc_hsa.error_robustness.particle_gibbs_ma3 import sample_states_particle_gibbs_ma
from nkpc_hsa.gibbs.common.competition import finite_N_residuals
from nkpc_hsa.gibbs.common.constraints import constraint_stats_summary, draw_with_constraints
from nkpc_hsa.gibbs.hsa_full.model import (
    KAPPA_SCALE,
    _ar2_stats_summary,
    _common_priors,
    _getd,
    _init_states,
    _kappa_t_constraint_validators,
    _mvnrnd,
    _sample_ar2_coeffs,
    _sample_invgamma,
    _summary,
)
from nkpc_hsa.gibbs.hsa_full_pg.model import DEFAULT_N_PARTICLES

__all__ = ["func_nkpc_hsa_full_ma3"]


def _sample_beta_gls(
    y: np.ndarray,
    X: np.ndarray,
    *,
    sigma2: float,
    weighting: MAWeighting,
    prior_mean: np.ndarray,
    prior_var: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Production ``_sample_beta_gaussian`` with ``X'X`` replaced by ``X' Omega_0^{-1} X``."""
    XtWX, XtWy = weighting.gls_moments(np.asarray(y, dtype=float).reshape(-1), X)
    V0_inv = np.diag(1.0 / np.asarray(prior_var, dtype=float).reshape(-1))
    Vn = inv(XtWX / sigma2 + V0_inv)
    mn = Vn @ (XtWy / sigma2 + V0_inv @ np.asarray(prior_mean, dtype=float).reshape(-1))
    return _mvnrnd(mn, Vn, rng)


def func_nkpc_hsa_full_ma3(
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
    """Gibbs/Particle-Gibbs/Metropolis sampler for hsa_full with an MA(q) disturbance.

    ``opts`` additions over production: ``ma_order`` (default 3, 0 gives the
    i.i.d. specification), ``psi0``, ``n_psi_steps``, ``psi_init_scale``.
    """
    pi_t = np.asarray(pi_data, dtype=float).reshape(-1)
    pi_tm1 = np.asarray(pi_prev_data, dtype=float).reshape(-1)
    pi_expect = np.asarray(Epi_data, dtype=float).reshape(-1)
    x_t = np.asarray(x_data, dtype=float).reshape(-1)
    x_tm1 = np.asarray(x_prev_data, dtype=float).reshape(-1)
    N_obs = np.asarray(N_data, dtype=float).reshape(-1)
    T = pi_t.size
    if not (pi_tm1.size == pi_expect.size == x_t.size == x_tm1.size == N_obs.size == T):
        raise ValueError("All input series must have the same length.")

    pri = _common_priors(priors or {})
    opts = opts or {}

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

    alpha = float(_getd(opts, "alpha0", pri["mu_alpha"]))
    kappa0 = float(_getd(opts, "kappa00", pri["mu_kappa0"]))
    delta = float(_getd(opts, "delta0", pri["mu_delta"]))
    theta0 = float(_getd(opts, "theta00", pri["mu_theta"]))
    gamma = float(_getd(opts, "gamma0", pri["mu_gamma"]))
    phi_1 = float(_getd(opts, "phi10", pri["mu_phi"]))
    lambda_ez = 0.0 if orth else float(_getd(opts, "lambda0", 0.0))
    rho1 = float(_getd(opts, "rho10", 0.5))
    rho2 = float(_getd(opts, "rho20", -0.5))
    n_drift = float(_getd(opts, "n0", 0.01))
    sigma_v2 = float(_getd(opts, "sigma_v20", _getd(opts, "sigma_e20", 1.0)))
    sigma_zeta2 = float(_getd(opts, "sigma_zeta20", 1.0))
    sigma_u2 = float(_getd(opts, "sigma_u20", _getd(opts, "sigma_eps20", 0.5)))
    sigma_eps2 = float(_getd(opts, "sigma_eps20", _getd(opts, "sigma_eta20", 0.1)))
    sigma_N2 = float(_getd(opts, "sigma_N20", _getd(opts, "sigma_m20", 1.0)))
    enforce_stationary = bool(_getd(opts, "enforce_stationary", True))
    ar2_max_tries = int(max(1, _getd(opts, "ar2_max_tries", 2000)))
    static_theta = bool(_getd(opts, "static_theta", False))
    if static_theta:
        gamma = 0.0
    n_particles = int(_getd(opts, "n_particles", DEFAULT_N_PARTICLES))
    store_every = int(max(1, _getd(opts, "store_every", 1)))
    verbose = bool(_getd(opts, "verbose", False))
    coefficient_constraints = _getd(opts, "coefficient_constraints", {})
    constraint_stats: dict[str, int] = {}
    ar2_stats: dict[str, int] = {}
    rng = np.random.default_rng(_getd(opts, "seed", None))

    Nbar, Nhat = _init_states(N_obs)
    Nhat_initial_lag = float(_getd(opts, "m0_Nhat_lag", pri["m0_Nhat_lag"]))
    a_t = pi_tm1 - pi_expect
    lambda_prec0 = 0.0 if orth else 1.0 / pri["sigma_lambda"] ** 2

    m0_Nhat = float(_getd(opts, "m0_Nhat", pri["m0_Nhat"]))
    P0_Nhat = float(_getd(opts, "P0_Nhat", pri["P0_Nhat"]))
    m0_Nhat_lag = float(_getd(opts, "m0_Nhat_lag", pri["m0_Nhat_lag"]))
    P0_Nhat_lag = float(_getd(opts, "P0_Nhat_lag", pri["P0_Nhat_lag"]))
    m0_Nbar = float(_getd(opts, "m0_Nbar", pri["m0_Nbar"]))
    P0_Nbar = float(_getd(opts, "P0_Nbar", pri["P0_Nbar"]))

    proposal = (
        AdaptiveRandomWalk(ma_order, init_scale=float(_getd(opts, "psi_init_scale", 0.08)))
        if ma_order
        else None
    )
    weighting = MAWeighting(psi, T)
    v_presample = np.zeros(ma_order, dtype=float)

    n_store = int(n_keep // store_every)
    alpha_draws = np.zeros(n_store)
    kappa0_draws = np.zeros(n_store)
    delta_draws = np.zeros(n_store)
    theta0_draws = np.zeros(n_store)
    gamma_draws = np.zeros(n_store)
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
    theta_t_draws = np.zeros((n_store, T))
    pg_ess_mean_draws = np.zeros(n_store)
    pg_ess_min_draws = np.zeros(n_store)
    pg_moved_draws = np.zeros(n_store)
    psi_draws = np.zeros((n_store, ma_order))
    e_acf_draws = np.zeros((n_store, ma_order))

    total_iter = n_burn + n_keep
    store_idx = 0

    for it in range(1, total_iter + 1):
        zeta = x_t - phi_1 * x_tm1
        y = pi_t - pi_expect
        y_adj = y - lambda_ez * zeta

        # ---- 1. coefficients, GLS under Omega_0(psi) ----
        columns = [a_t, x_t / KAPPA_SCALE, (x_t * Nbar) / KAPPA_SCALE, -Nhat]
        prior_means = [pri["mu_alpha"], pri["mu_kappa0"], pri["mu_delta"], pri["mu_theta"]]
        prior_vars = [
            pri["sigma_alpha"] ** 2, pri["sigma_kappa0"] ** 2,
            pri["sigma_delta"] ** 2, pri["sigma_theta"] ** 2,
        ]
        beta_names = ["alpha", "kappa_0", "delta", "theta" if static_theta else "theta_0"]
        if not static_theta:
            columns.append(-(Nhat * Nbar))
            prior_means.append(pri["mu_gamma"])
            prior_vars.append(pri["sigma_gamma"] ** 2)
            beta_names.append("gamma")
        X = np.column_stack(columns)

        beta = draw_with_constraints(
            lambda: _sample_beta_gls(
                y_adj, X, sigma2=sigma_v2, weighting=weighting,
                prior_mean=np.array(prior_means, dtype=float),
                prior_var=np.array(prior_vars, dtype=float),
                rng=rng,
            ),
            tuple(beta_names),
            coefficient_constraints,
            validators=_kappa_t_constraint_validators(Nbar, coefficient_constraints),
            stats=constraint_stats,
        )
        alpha = float(beta[0])
        kappa0 = float(beta[1])
        delta = float(beta[2])
        theta0 = float(beta[3])
        if not static_theta:
            gamma = float(beta[4])
        kappa_t_eff = (kappa0 + delta * Nbar) / KAPPA_SCALE
        theta_t = theta0 + gamma * Nbar

        # ---- 2. cross-equation loading, GLS ----
        if not orth:
            e_base = y - alpha * a_t - kappa_t_eff * x_t + theta_t * Nhat
            Wzeta = weighting.solve(zeta)
            post_var_lambda = 1.0 / (lambda_prec0 + float(zeta @ Wzeta) / sigma_v2)
            post_mean_lambda = post_var_lambda * (
                pri["mu_lambda"] * lambda_prec0 + float(e_base @ Wzeta) / sigma_v2
            )
            lambda_ez = float(
                post_mean_lambda + np.sqrt(post_var_lambda) * rng.standard_normal()
            )
        else:
            lambda_ez = 0.0

        # ---- 3. phi_1, GLS on the inflation-equation contribution ----
        y_tilde_phi = y - alpha * a_t - kappa_t_eff * x_t + theta_t * Nhat
        Wx_prev = weighting.solve(x_tm1)
        prec_phi = (
            1.0 / pri["sigma_phi"] ** 2
            + float(np.sum(x_tm1**2)) / sigma_zeta2
            + (lambda_ez**2) * float(x_tm1 @ Wx_prev) / sigma_v2
        )
        mean_num_phi = (
            pri["mu_phi"] / pri["sigma_phi"] ** 2
            + float(np.dot(x_tm1, x_t)) / sigma_zeta2
            - lambda_ez * float((y_tilde_phi - lambda_ez * x_t) @ Wx_prev) / sigma_v2
        )
        phi_1 = float(mean_num_phi / prec_phi + rng.standard_normal() / np.sqrt(prec_phi))

        # ---- 4. variances ----
        zeta = x_t - phi_1 * x_tm1
        xi = y - alpha * a_t - kappa_t_eff * x_t + theta_t * Nhat - lambda_ez * zeta
        sigma_zeta2 = _sample_invgamma(
            pri["a_z"] + 0.5 * T, pri["b_z"] + 0.5 * float(np.sum(zeta**2)), rng
        )
        sigma_v2 = _sample_invgamma(
            pri["a_e"] + 0.5 * T, pri["b_e"] + 0.5 * weighting.quadratic_form(xi), rng
        )

        # ---- 5-6. firm-count blocks, unchanged from production ----
        if T >= 2:
            rho1, rho2 = _sample_ar2_coeffs(
                Nhat, sigma_u2, pri["mu_rho1"], pri["sigma_rho1"],
                pri["mu_rho2"], pri["sigma_rho2"], enforce_stationary, rng,
                max_tries=ar2_max_tries, current=(rho1, rho2), stats=ar2_stats,
                initial_lag=Nhat_initial_lag,
            )
            second_lag = np.concatenate([[Nhat_initial_lag], Nhat[:-2]])
            resid_u = Nhat[1:] - rho1 * Nhat[:-1] - rho2 * second_lag
            sigma_u2 = _sample_invgamma(
                pri["a_u"] + 0.5 * resid_u.size,
                pri["b_u"] + 0.5 * float(np.sum(resid_u**2)), rng,
            )
            dNbar = Nbar[1:] - Nbar[:-1]
            post_var_n = 1.0 / (1.0 / pri["sigma_n"] ** 2 + dNbar.size / sigma_eps2)
            post_mean_n = post_var_n * (
                pri["mu_n"] / pri["sigma_n"] ** 2 + float(np.sum(dNbar)) / sigma_eps2
            )
            n_drift = float(post_mean_n + np.sqrt(post_var_n) * rng.standard_normal())
            resid_eps = Nbar[1:] - n_drift - Nbar[:-1]
            sigma_eps2 = _sample_invgamma(
                pri["a_eps"] + 0.5 * resid_eps.size,
                pri["b_eps"] + 0.5 * float(np.sum(resid_eps**2)), rng,
            )

        resid_N = finite_N_residuals(N_obs, Nhat, Nbar)
        sigma_N2 = _sample_invgamma(
            pri["a_N"] + 0.5 * resid_N.size,
            pri["b_N"] + 0.5 * float(np.sum(resid_N**2)), rng,
        )

        # ---- 7. joint Particle Gibbs state update, v solved rather than proposed ----
        pg = sample_states_particle_gibbs_ma(
            y=y, a_t=a_t, x_t=x_t, zeta=zeta, N_obs=N_obs,
            alpha=alpha, kappa0_eff=kappa0 / KAPPA_SCALE, delta_eff=delta / KAPPA_SCALE,
            theta0=theta0, gamma=gamma, lambda_ez=lambda_ez,
            rho1=rho1, rho2=rho2, n_drift=n_drift,
            psi=psi, sigma_v2=sigma_v2, sigma_u2=sigma_u2,
            sigma_eps2=sigma_eps2, sigma_N2=sigma_N2,
            Nbar_ref=Nbar, Nhat_ref=Nhat, Nhat_ref_lag=Nhat_initial_lag,
            v_presample_ref=v_presample,
            m0_Nhat=m0_Nhat, P0_Nhat=P0_Nhat, m0_Nhat_lag=m0_Nhat_lag,
            P0_Nhat_lag=P0_Nhat_lag, m0_Nbar=m0_Nbar, P0_Nbar=P0_Nbar,
            n_particles=n_particles, rng=rng,
        )
        Nhat = pg["Nhat"]
        Nbar = pg["Nbar"]
        Nhat_initial_lag = pg["Nhat_lag"]
        v_presample = pg["v_presample"]

        kappa_t_eff = (kappa0 + delta * Nbar) / KAPPA_SCALE
        theta_t = theta0 + gamma * Nbar

        # ---- 8. psi, last ----
        if ma_order:
            xi_post = y - alpha * a_t - kappa_t_eff * x_t + theta_t * Nhat - lambda_ez * zeta
            psi, weighting = sample_psi(
                psi, xi_post, sigma_v2, prior=psi_prior, proposal=proposal,
                rng=rng, n_steps=n_psi_steps, weighting=weighting,
            )
            if it == n_burn:
                proposal.freeze()
        else:
            weighting = MAWeighting(psi, T)

        gamma_acov = autocovariance(psi, sigma_v2)
        sigma_e2 = lambda_ez**2 * sigma_zeta2 + gamma_acov[0]
        sigma_e = float(np.sqrt(sigma_e2))
        rho_ez = 0.0 if orth else float(
            (lambda_ez * np.sqrt(sigma_zeta2)) / max(sigma_e, 1e-12)
        )

        if it > n_burn and (it - n_burn) % store_every == 0:
            alpha_draws[store_idx] = alpha
            kappa0_draws[store_idx] = kappa0 / KAPPA_SCALE
            delta_draws[store_idx] = delta / KAPPA_SCALE
            theta0_draws[store_idx] = theta0
            gamma_draws[store_idx] = gamma
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
            kappa_t_draws[store_idx] = kappa_t_eff
            theta_t_draws[store_idx] = theta_t
            pg_ess_mean_draws[store_idx] = pg["ess_mean"]
            pg_ess_min_draws[store_idx] = pg["ess_min"]
            pg_moved_draws[store_idx] = pg["moved_frac"]
            if ma_order:
                psi_draws[store_idx] = psi
                e_acf_draws[store_idx] = gamma_acov[1:] / sigma_e2
            store_idx += 1

        if verbose and it % 2000 == 0:
            acc = proposal.acceptance_rate if proposal is not None else float("nan")
            print(
                f"Iter {it}/{total_iter}: alpha={alpha:.3f}, delta={delta:.3f}, "
                f"gamma={gamma:.3f}, psi={np.round(psi, 3)}, accept={acc:.2f}, "
                f"pg_ess={pg['ess_mean']:.0f}"
            )

    theta_keys = (
        {"theta": _summary(theta0_draws)}
        if static_theta
        else {"theta_0": _summary(theta0_draws), "gamma": _summary(gamma_draws)}
    )
    error_structure: dict[str, Any] = {
        "family": "ma" if ma_order else "iid",
        "order": ma_order,
        "innovation_solved_in_particle_filter": bool(ma_order),
    }
    if ma_order and proposal is not None:
        error_structure["psi_acceptance_rate"] = proposal.acceptance_rate
        error_structure["psi_proposal_scale"] = float(np.exp(proposal.log_scale))
        error_structure["psi_metropolis_steps_per_sweep"] = n_psi_steps

    out: dict[str, Any] = {
        "alpha": _summary(alpha_draws),
        "kappa_0": _summary(kappa0_draws),
        "delta": _summary(delta_draws),
        **theta_keys,
        "phi_1": _summary(phi_draws),
        "lambda_ez": _summary(lambda_draws),
        "rho": _summary(rho_ez_draws),
        "rho1": _summary(rho1_draws),
        "rho2": _summary(rho2_draws),
        "n": _summary(n_draws),
        "sigma_e": _summary(sigma_e_draws),
        "sigma_v": _summary(sigma_v_draws),
        "sigma_zeta": _summary(sigma_zeta_draws),
        "sigma_u": _summary(sigma_u_draws),
        "sigma_eps": _summary(sigma_eps_draws),
        "sigma_N": _summary(sigma_N_draws),
        "state_draws": {
            "Nbar": Nbar_draws,
            "Nhat": Nhat_draws,
            "kappa_t": kappa_t_draws,
            "theta_t": theta_t_draws,
        },
        "pg_diagnostics": {
            "n_particles": n_particles,
            "ess_mean": pg_ess_mean_draws,
            "ess_min": pg_ess_min_draws,
            "moved_frac": pg_moved_draws,
        },
        "priors": priors or {},
        "opts": opts,
        "model": {
            "N_measurement_error": True,
            "N_measurement_equation": "N_obs_t = Nhat_t + Nbar_t + nu_t, nu_t ~ N(0, sigma_N^2)",
            "state_blocks": (
                f"JOINT Particle Gibbs (conditional SMC, {n_particles} particles) with the "
                "MA innovation solved from the inflation equation rather than proposed."
            ),
            "state_sampler": "particle_gibbs_ma",
            "n_particles": n_particles,
            "kappa_scale": KAPPA_SCALE,
            "kappa_internal": "stored kappa_0, delta, and kappa_t multiplied by KAPPA_SCALE",
            "stored_units": "physical",
            "theta_specification": (
                "static (theta_t = theta, gamma fixed at 0)"
                if static_theta
                else "time-varying (theta_t = theta_0 + gamma * Nbar_t)"
            ),
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
