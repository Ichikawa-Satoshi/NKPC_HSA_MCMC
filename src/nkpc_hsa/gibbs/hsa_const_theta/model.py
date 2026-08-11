"""HSA const-theta: time-varying slope, constant entry coefficient, joint FFBS.

Model (identical to ``hsa_full`` with ``gamma`` restricted to zero)::

    y_t      = alpha*a_t + kappa_t*x_t - theta*Nhat_t + lambda_ez*zeta_t + eta_t
    kappa_t  = kappa_0 + delta*Nbar_t
    x_t      = phi_1*x_{t-1} + zeta_t
    N_obs_t  = Nbar_t + Nhat_t + nu_t
    Nhat_t   = rho_1*Nhat_{t-1} + rho_2*Nhat_{t-2} + u_t
    Nbar_t   = n + Nbar_{t-1} + epsilon_t

with ``y_t = pi_t - E_t pi_{t+1}`` and ``a_t = pi_{t-1} - E_t pi_{t+1}``.

Why this module exists
----------------------
This specification was previously produced by ``func_nkpc_hsa_full(static_theta=True)``,
which draws the firm-count state in two alternating conditional blocks,
``Nhat | Nbar`` then ``Nbar | Nhat``. Each block is an exact conditional, so that
kernel is *valid* -- but ``N_obs_t = Nbar_t + Nhat_t + nu_t`` with a small sigma_N
pins the sum of the states while leaving the split nearly free, and the posterior
correlation between the two blocks is about -0.999. A two-block Gibbs moves the
shared level with autocorrelation rho^2 per sweep, i.e. an integrated
autocorrelation time of order 10^3, which is exactly what the old runs showed
(Nbar path Rhat ~ 2.7, bulk ESS ~ 2, and the trend drift ``n`` at Rhat ~ 1.84).

With ``gamma = 0`` the inflation observation is *linear* in the joint state
``s_t = (Nhat_t, Nhat_{t-1}, Nbar_t)'``, so the whole model is linear-Gaussian and
the entire path can be drawn in one exact FFBS sweep -- the same routine that
makes ``hsa_steady`` mix. ``hsa_const_theta`` is the ``h_nhat = -theta_0`` case of
``hsa_steady``'s ``h_nhat = 0``; both go through
``nkpc_hsa.gibbs.common.joint_ffbs.sample_joint_competition_states_ffbs``.

This is a *computational* fix. It removes a sampler pathology; it does not create
identification that the data do not contain. The latent level ridge is a property
of the model and the data and survives the change.

``hsa_full`` (gamma free) genuinely is non-linear-Gaussian in the joint state and
keeps its alternating blocks / Particle-Gibbs sweep.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from nkpc_hsa.gibbs.common.competition import finite_N_residuals
from nkpc_hsa.gibbs.common.constraints import constraint_stats_summary, draw_with_constraints
from nkpc_hsa.gibbs.common.joint_ffbs import sample_joint_competition_states_ffbs
from nkpc_hsa.gibbs.hsa_full.model import (
    KAPPA_SCALE,
    _ar2_stats_summary,
    _common_priors,
    _getd,
    _init_states,
    _kappa_t_constraint_validators,
    _sample_ar2_coeffs,
    _sample_beta_gaussian,
    _sample_invgamma,
    _sample_phi_joint,
    _summary,
)

__all__ = ["func_nkpc_hsa_const_theta"]


def func_nkpc_hsa_const_theta(
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
    Ehat_data=None,
    E_data=None,
) -> dict[str, Any]:
    """Gibbs sampler for HSA const-theta with an exact joint FFBS state block.

    Signature, priors, scaling conventions, missing-N handling and output keys
    match the previous ``func_nkpc_hsa_full_static_theta`` exactly, so this can be
    dispatched through ``run_model`` and consumed by the existing table builders
    without changes. The only difference is that the latent firm-count path is
    drawn jointly instead of in two alternating blocks.
    """
    if E_data is not None:
        if Ehat_data is not None:
            raise ValueError("Supply E_data levels or legacy Ehat_data, not both.")
        from nkpc_hsa.gibbs.joint_ne import func_nkpc_hsa_joint_ne

        return func_nkpc_hsa_joint_ne(
            pi_data=pi_data,
            pi_prev_data=pi_prev_data,
            Epi_data=Epi_data,
            x_data=x_data,
            x_prev_data=x_prev_data,
            N_data=N_data,
            E_data=E_data,
            n_burn=n_burn,
            n_keep=n_keep,
            priors=priors,
            opts=opts,
            orth=orth,
            const_theta=True,
        )

    pi_t = np.asarray(pi_data, dtype=float).reshape(-1)
    pi_tm1 = np.asarray(pi_prev_data, dtype=float).reshape(-1)
    pi_expect = np.asarray(Epi_data, dtype=float).reshape(-1)
    x_t = np.asarray(x_data, dtype=float).reshape(-1)
    x_tm1 = np.asarray(x_prev_data, dtype=float).reshape(-1)
    N_obs = np.asarray(N_data, dtype=float).reshape(-1)
    Ehat_obs = None if Ehat_data is None else np.asarray(Ehat_data, dtype=float).reshape(-1)
    T = pi_t.size
    if not (pi_tm1.size == pi_expect.size == x_t.size == x_tm1.size == N_obs.size == T):
        raise ValueError("All input series must have the same length.")
    if Ehat_obs is not None and Ehat_obs.size != T:
        raise ValueError("Ehat_data must have the same length as the other model series.")
    if Ehat_obs is not None and np.isfinite(Ehat_obs).sum() < 3:
        raise ValueError("Ehat_data must contain at least three finite quarterly observations.")
    if T < 3:
        raise ValueError("Need T >= 3 for the AR(2) gap equation.")

    pri = _common_priors(priors or {})
    opts = opts or {}
    rng = np.random.default_rng(_getd(opts, "seed", None))

    alpha = float(_getd(opts, "alpha0", pri["mu_alpha"]))
    kappa0 = float(_getd(opts, "kappa00", pri["mu_kappa0"]))
    delta = float(_getd(opts, "delta0", pri["mu_delta"]))
    theta = float(_getd(opts, "theta00", pri["mu_theta"]))
    phi_1 = float(_getd(opts, "phi10", pri["mu_phi"]))
    lambda_ez = 0.0 if orth else float(_getd(opts, "lambda0", pri["mu_lambda"]))
    lambda_E = float(_getd(opts, "lambda_E0", pri["mu_lambda_E"]))
    rho1 = float(_getd(opts, "rho10", pri["mu_rho1"]))
    rho2 = float(_getd(opts, "rho20", pri["mu_rho2"]))
    n_drift = float(_getd(opts, "n0", pri["mu_n"]))

    sigma_eta2 = float(_getd(opts, "sigma_e20", _getd(opts, "sigma_eta20", 1.0)))
    sigma_zeta2 = float(_getd(opts, "sigma_zeta20", 1.0))
    sigma_u2 = float(_getd(opts, "sigma_u20", 1.0))
    sigma_eps2 = float(_getd(opts, "sigma_eps20", 1.0))
    sigma_N2 = float(_getd(opts, "sigma_N20", _getd(opts, "sigma_m20", 1.0)))
    sigma_E2 = float(_getd(opts, "sigma_E20", 0.01))

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

    Nbar, Nhat = _init_states(N_obs)
    initialize_from_Ehat = bool(_getd(opts, "initialize_from_Ehat", Ehat_obs is not None))
    if Ehat_obs is not None and initialize_from_Ehat:
        finite_E = np.isfinite(Ehat_obs)
        loading = lambda_E if abs(lambda_E) > 1e-8 else 1.0
        target = np.empty(T, dtype=float)
        target[finite_E] = Ehat_obs[finite_E] / loading
        if not np.all(finite_E):
            positions = np.arange(T)
            target[~finite_E] = np.interp(positions[~finite_E], positions[finite_E], target[finite_E])
        total = Nbar + Nhat
        Nhat = target
        Nbar = total - Nhat
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

    # states[:, 1] carries Nhat_{t-1}; states[0, 1] is the sampled Nhat_{-1}.
    states = np.zeros((T, 3), dtype=float)
    states[:, 0] = Nhat
    states[:, 2] = Nbar
    states[0, 1] = m0[1]
    if T > 1:
        states[1:, 1] = Nhat[:-1]

    alpha_draws = np.zeros(n_store)
    kappa0_draws = np.zeros(n_store)
    delta_draws = np.zeros(n_store)
    theta_draws = np.zeros(n_store)
    phi_draws = np.zeros(n_store)
    lambda_draws = np.zeros(n_store)
    lambda_E_draws = np.zeros(n_store) if Ehat_obs is not None else None
    rho1_draws = np.zeros(n_store)
    rho2_draws = np.zeros(n_store)
    n_draws = np.zeros(n_store)
    sigma_e_draws = np.zeros(n_store)
    sigma_eta_draws = np.zeros(n_store)
    sigma_zeta_draws = np.zeros(n_store)
    sigma_u_draws = np.zeros(n_store)
    sigma_eps_draws = np.zeros(n_store)
    sigma_N_draws = np.zeros(n_store)
    sigma_E_draws = np.zeros(n_store) if Ehat_obs is not None else None
    rho_ez_draws = np.zeros(n_store)
    Nbar_draws = np.zeros((n_store, T))
    Nhat_draws = np.zeros((n_store, T))
    kappa_t_draws = np.zeros((n_store, T))
    theta_t_draws = np.zeros((n_store, T))

    total_iter = n_burn + n_keep
    store_idx = 0

    for it in range(1, total_iter + 1):
        # --- 1. (alpha, kappa_0, delta, theta) from the inflation regression ---
        zeta = x_t - phi_1 * x_tm1
        y_adj = y - lambda_ez * zeta
        X = np.column_stack(
            [
                a_t,
                x_t / KAPPA_SCALE,
                (x_t * Nbar) / KAPPA_SCALE,
                -Nhat,
            ]
        )
        beta_prior_mean = np.array(
            [pri["mu_alpha"], pri["mu_kappa0"], pri["mu_delta"], pri["mu_theta"]], dtype=float
        )
        beta_prior_var = np.array(
            [
                pri["sigma_alpha"] ** 2,
                pri["sigma_kappa0"] ** 2,
                pri["sigma_delta"] ** 2,
                pri["sigma_theta"] ** 2,
            ],
            dtype=float,
        )
        beta = draw_with_constraints(
            lambda: _sample_beta_gaussian(
                y_adj,
                X,
                sigma2=sigma_eta2,
                prior_mean=beta_prior_mean,
                prior_var=beta_prior_var,
                rng=rng,
            ),
            ("alpha", "kappa_0", "delta", "theta"),
            coefficient_constraints,
            validators=_kappa_t_constraint_validators(Nbar, coefficient_constraints),
            stats=constraint_stats,
        )
        alpha = float(beta[0])
        kappa0 = float(beta[1])
        delta = float(beta[2])
        theta = float(beta[3])

        kappa_t = kappa0 + delta * Nbar
        kappa_t_eff = kappa_t / KAPPA_SCALE

        # --- 2. lambda_ez ---
        if not orth:
            e_base = y - alpha * a_t - kappa_t_eff * x_t + theta * Nhat
            post_var_lambda = 1.0 / (lambda_prec0 + float(np.sum(zeta**2)) / sigma_eta2)
            post_mean_lambda = post_var_lambda * (
                pri["mu_lambda"] * lambda_prec0 + float(np.dot(zeta, e_base)) / sigma_eta2
            )
            lambda_ez = float(post_mean_lambda + np.sqrt(post_var_lambda) * rng.standard_normal())
        else:
            lambda_ez = 0.0

        # --- 3. phi_1 ---
        y_tilde_phi = y - alpha * a_t - kappa_t_eff * x_t + theta * Nhat
        phi_1 = _sample_phi_joint(
            x_t=x_t,
            x_tm1=x_tm1,
            y_tilde=y_tilde_phi,
            lambda_ez=lambda_ez,
            sigma_zeta2=sigma_zeta2,
            sigma_eta2=sigma_eta2,
            mu_phi=pri["mu_phi"],
            sigma_phi=pri["sigma_phi"],
            rng=rng,
        )

        # --- 4. sigma_zeta2, sigma_eta2 ---
        zeta = x_t - phi_1 * x_tm1
        eta = y - alpha * a_t - kappa_t_eff * x_t + theta * Nhat - lambda_ez * zeta
        sigma_zeta2 = _sample_invgamma(
            pri["a_z"] + 0.5 * T, pri["b_z"] + 0.5 * float(np.sum(zeta**2)), rng
        )
        sigma_eta2 = _sample_invgamma(
            pri["a_e"] + 0.5 * T, pri["b_e"] + 0.5 * float(np.sum(eta**2)), rng
        )

        # --- 5. (rho_1, rho_2) and sigma_u2, using the sampled Nhat_{-1} lag ---
        rho1, rho2 = _sample_ar2_coeffs(
            Nhat,
            sigma_u2,
            pri["mu_rho1"],
            pri["sigma_rho1"],
            pri["mu_rho2"],
            pri["sigma_rho2"],
            enforce_stationary,
            rng,
            max_tries=ar2_max_tries,
            current=(rho1, rho2),
            stats=ar2_stats,
            initial_lag=float(states[0, 1]),
        )
        resid_u = states[1:, 0] - rho1 * states[:-1, 0] - rho2 * states[:-1, 1]
        sigma_u2 = _sample_invgamma(
            pri["a_u"] + 0.5 * resid_u.size, pri["b_u"] + 0.5 * float(np.sum(resid_u**2)), rng
        )

        # --- 6. n and sigma_eps2 ---
        dNbar = Nbar[1:] - Nbar[:-1]
        post_var_n = 1.0 / (1.0 / pri["sigma_n"] ** 2 + dNbar.size / sigma_eps2)
        post_mean_n = post_var_n * (
            pri["mu_n"] / pri["sigma_n"] ** 2 + float(np.sum(dNbar)) / sigma_eps2
        )
        n_drift = float(post_mean_n + np.sqrt(post_var_n) * rng.standard_normal())
        resid_eps = Nbar[1:] - n_drift - Nbar[:-1]
        sigma_eps2 = _sample_invgamma(
            pri["a_eps"] + 0.5 * resid_eps.size,
            pri["b_eps"] + 0.5 * float(np.sum(resid_eps**2)),
            rng,
        )

        # --- 7. sigma_N2, from observed quarters only ---
        resid_N = finite_N_residuals(N_obs, Nhat, Nbar)
        sigma_N2 = _sample_invgamma(
            pri["a_N"] + 0.5 * resid_N.size, pri["b_N"] + 0.5 * float(np.sum(resid_N**2)), rng
        )

        # --- 8. lambda_E and sigma_E2 for the quarterly Ehat observation ---
        if Ehat_obs is not None:
            finite_E = np.isfinite(Ehat_obs)
            e_obs = Ehat_obs[finite_E]
            e_state = Nhat[finite_E]
            prior_prec_E = 1.0 / pri["sigma_lambda_E"]**2
            post_var_lambda_E = 1.0 / (
                prior_prec_E + float(np.sum(e_state**2)) / sigma_E2
            )
            post_mean_lambda_E = post_var_lambda_E * (
                pri["mu_lambda_E"] * prior_prec_E
                + float(np.dot(e_state, e_obs)) / sigma_E2
            )
            lambda_E = float(
                post_mean_lambda_E + np.sqrt(post_var_lambda_E) * rng.standard_normal()
            )
            resid_E = e_obs - lambda_E * e_state
            sigma_E2 = _sample_invgamma(
                pri["a_E"] + 0.5 * resid_E.size,
                pri["b_E"] + 0.5 * float(np.sum(resid_E**2)),
                rng,
            )

        # --- 9. JOINT exact FFBS for the whole state path ---
        # With gamma = 0 the inflation row is linear in s_t, loading
        # -theta on Nhat_t and delta*x_t/KAPPA_SCALE on Nbar_t.
        y_tilde_state = y - alpha * a_t - (kappa0 / KAPPA_SCALE) * x_t - lambda_ez * zeta
        Nbar, Nhat, states = sample_joint_competition_states_ffbs(
            N_obs=N_obs,
            y_tilde=y_tilde_state,
            h_nhat=np.full(T, -theta, dtype=float),
            h_nbar=(delta / KAPPA_SCALE) * x_t,
            n_drift=n_drift,
            rho1=rho1,
            rho2=rho2,
            sigma_eta2=sigma_eta2,
            sigma_u2=sigma_u2,
            sigma_eps2=sigma_eps2,
            sigma_N2=sigma_N2,
            m0=m0,
            P0=P0,
            rng=rng,
            Ehat_obs=Ehat_obs,
            lambda_E=lambda_E,
            sigma_E2=sigma_E2,
        )

        kappa_t = kappa0 + delta * Nbar

        # --- 10. store ---
        sigma_e = float(np.sqrt(lambda_ez**2 * sigma_zeta2 + sigma_eta2))
        rho_ez = (
            0.0
            if orth
            else float((lambda_ez * np.sqrt(sigma_zeta2)) / max(sigma_e, 1e-12))
        )

        if it > n_burn and (it - n_burn) % store_every == 0:
            alpha_draws[store_idx] = alpha
            kappa0_draws[store_idx] = kappa0 / KAPPA_SCALE
            delta_draws[store_idx] = delta / KAPPA_SCALE
            theta_draws[store_idx] = theta
            phi_draws[store_idx] = phi_1
            lambda_draws[store_idx] = lambda_ez
            if lambda_E_draws is not None:
                lambda_E_draws[store_idx] = lambda_E
            rho1_draws[store_idx] = rho1
            rho2_draws[store_idx] = rho2
            n_draws[store_idx] = n_drift
            sigma_e_draws[store_idx] = sigma_e
            sigma_eta_draws[store_idx] = np.sqrt(sigma_eta2)
            sigma_zeta_draws[store_idx] = np.sqrt(sigma_zeta2)
            sigma_u_draws[store_idx] = np.sqrt(sigma_u2)
            sigma_eps_draws[store_idx] = np.sqrt(sigma_eps2)
            sigma_N_draws[store_idx] = np.sqrt(sigma_N2)
            if sigma_E_draws is not None:
                sigma_E_draws[store_idx] = np.sqrt(sigma_E2)
            rho_ez_draws[store_idx] = rho_ez
            Nbar_draws[store_idx] = Nbar
            Nhat_draws[store_idx] = Nhat
            kappa_t_draws[store_idx] = kappa_t / KAPPA_SCALE
            theta_t_draws[store_idx] = np.full(T, theta, dtype=float)
            store_idx += 1

        if progress_callback is not None:
            progress_callback(it, total_iter)

        if verbose and it % 5000 == 0:
            print(
                f"Iter {it}/{total_iter}: alpha={alpha:.3f}, kappa0={kappa0:.3f}, "
                f"delta={delta:.3f}, theta={theta:.3f}, n={n_drift:.4f}"
            )

    return {
        "alpha": _summary(alpha_draws),
        "kappa_0": _summary(kappa0_draws),
        "delta": _summary(delta_draws),
        "theta": _summary(theta_draws),
        "phi_1": _summary(phi_draws),
        "lambda_ez": _summary(lambda_draws),
        **({"lambda_E": _summary(lambda_E_draws)} if lambda_E_draws is not None else {}),
        "rho": _summary(rho_ez_draws),
        "rho1": _summary(rho1_draws),
        "rho2": _summary(rho2_draws),
        "n": _summary(n_draws),
        "sigma_e": _summary(sigma_e_draws),
        "sigma_eta": _summary(sigma_eta_draws),
        "sigma_zeta": _summary(sigma_zeta_draws),
        "sigma_u": _summary(sigma_u_draws),
        "sigma_eps": _summary(sigma_eps_draws),
        "sigma_N": _summary(sigma_N_draws),
        **({"sigma_E": _summary(sigma_E_draws)} if sigma_E_draws is not None else {}),
        "state_draws": {
            "Nbar": Nbar_draws,
            "Nhat": Nhat_draws,
            "kappa_t": kappa_t_draws,
            "theta_t": theta_t_draws,
        },
        "priors": priors or {},
        "opts": opts,
        "model": {
            "N_measurement_error": True,
            "N_measurement_equation": "N_obs_t = Nhat_t + Nbar_t + nu_t, nu_t ~ N(0, sigma_N^2)",
            "establishment_measurement": Ehat_obs is not None,
            "establishment_measurement_equation": (
                "Ehat_obs_t = lambda_E * Nhat_t + omega_t"
                if Ehat_obs is not None
                else None
            ),
            "state_sampler": "joint_ffbs",
            "state_blocks": (
                "Exact joint Kalman/FFBS draw of s_t = [Nhat_t, Nhat_{t-1}, Nbar_t]'. "
                "Valid because gamma = 0 makes the inflation observation linear in the "
                "joint state; replaces the alternating Nhat|Nbar and Nbar|Nhat blocks."
            ),
            "state_vector": "[Nhat_t, Nhat_{t-1}, Nbar_t]'",
            "kappa_scale": KAPPA_SCALE,
            "kappa_internal": "stored kappa_0, delta, and kappa_t multiplied by KAPPA_SCALE",
            "stored_units": "physical",
            "theta_specification": "static (theta_t = theta, gamma fixed at 0)",
            "initialize_from_Ehat": initialize_from_Ehat,
            "coefficient_constraints": coefficient_constraints,
            "coefficient_constraint_stats": constraint_stats_summary(constraint_stats),
            "ar2_stationarity": {
                "enforce_stationary": enforce_stationary,
                "max_tries": ar2_max_tries,
                **_ar2_stats_summary(ar2_stats),
            },
        },
    }
