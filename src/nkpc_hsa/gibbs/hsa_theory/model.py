"""Samplers for the F0/U/R1/R2/R3 theory-model namespace.

The historical sampler functions remain untouched at their public slugs.  U is
the same unrestricted moving-reference likelihood as historical ``hsa_full``
but its output schema uses the new coefficient names.  R1/R2/R3 activate the
cross-restricted coefficient block in the shared Particle-Gibbs implementation.
F0 has a separate fixed-reference state system and is therefore not obtained by
constraining a moving-reference model.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional

import numpy as np

from nkpc_hsa.gibbs.common.competition import finite_N_residuals
from nkpc_hsa.gibbs.common.constraints import constraint_stats_summary, draw_with_constraints
from nkpc_hsa.gibbs.hsa_full.model import (
    KAPPA_SCALE,
    _ar2_stats_summary,
    _common_priors,
    _getd,
    _sample_ar2_coeffs,
    _sample_beta_gaussian,
    _sample_invgamma,
    _sample_phi_joint,
    _summary,
)
from nkpc_hsa.gibbs.hsa_full_pg.model import func_nkpc_hsa_full_pg


def _rename_unrestricted_result(result: Mapping[str, Any], *, zeta0: float) -> dict[str, Any]:
    out = dict(result)
    delta = out.pop("delta")
    delta_draws = np.asarray(delta["draws"], dtype=float)
    out["kappa_N_empirical"] = delta
    out["d_kappa_d_logN"] = _summary(10.0 * delta_draws)
    out["zeta0"] = _summary(np.full_like(delta_draws, float(zeta0)))
    out["mu0"] = _summary(np.full_like(delta_draws, float(zeta0) / (float(zeta0) - 1.0)))
    model_meta = dict(out.get("model", {}) or {})
    model_meta.update(
        {
            "model_hierarchy": "U",
            "cross_equation_restriction": False,
            "kappa_N_sampling": "independent",
        }
    )
    out["model"] = model_meta
    return out


def func_hsa_u(*args, **kwargs) -> dict[str, Any]:
    opts = dict(kwargs.get("opts") or {})
    opts.setdefault("restriction_level", "U")
    kwargs["opts"] = opts
    result = func_nkpc_hsa_full_pg(*args, **kwargs)
    return _rename_unrestricted_result(result, zeta0=float(opts.get("zeta0", 6.0)))


def func_hsa_restricted(*args, **kwargs) -> dict[str, Any]:
    opts = dict(kwargs.get("opts") or {})
    level = str(opts.get("restriction_level", "R1"))
    if level not in {"R1", "R2", "R3"}:
        raise ValueError("Restricted theory sampler level must be R1, R2, or R3.")
    opts["cross_restriction"] = True
    opts["static_theta"] = level == "R3"
    kwargs["opts"] = opts
    return func_nkpc_hsa_full_pg(*args, **kwargs)


def _fixed_reference_ffbs(
    *,
    y: np.ndarray,
    a_t: np.ndarray,
    x_t: np.ndarray,
    zeta: np.ndarray,
    N_obs: np.ndarray,
    alpha: float,
    kappa0_eff: float,
    theta0: float,
    lambda_ez: float,
    rho1: float,
    rho2: float,
    sigma_eta2: float,
    sigma_u2: float,
    sigma_N2: float,
    m0: np.ndarray,
    P0: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, float]:
    """Exact FFBS for q_t = log-N deviation around one fixed N0."""
    T = y.size
    F = np.array([[rho1, rho2], [1.0, 0.0]])
    Q = np.diag([sigma_u2, 0.0])
    ytilde = y - alpha * a_t - kappa0_eff * x_t - lambda_ez * zeta
    m_pred = np.empty((T, 2))
    P_pred = np.empty((T, 2, 2))
    m_filt = np.empty((T, 2))
    P_filt = np.empty((T, 2, 2))
    eye = np.eye(2)
    for t in range(T):
        if t == 0:
            m_pred[t], P_pred[t] = m0, P0
        else:
            m_pred[t] = F @ m_filt[t - 1]
            P_pred[t] = F @ P_filt[t - 1] @ F.T + Q
        rows, obs, variances = [], [], []
        if np.isfinite(N_obs[t]):
            rows.append([1.0, 0.0])
            obs.append(N_obs[t])
            variances.append(sigma_N2)
        rows.append([-theta0, 0.0])
        obs.append(ytilde[t])
        variances.append(sigma_eta2)
        H = np.asarray(rows, dtype=float)
        R = np.diag(variances)
        S = H @ P_pred[t] @ H.T + R
        K = np.linalg.solve(S, H @ P_pred[t]).T
        innovation = np.asarray(obs) - H @ m_pred[t]
        m_filt[t] = m_pred[t] + K @ innovation
        P_filt[t] = (eye - K @ H) @ P_pred[t] @ (eye - K @ H).T + K @ R @ K.T
        P_filt[t] = 0.5 * (P_filt[t] + P_filt[t].T) + 1e-12 * eye
    states = np.empty((T, 2))
    states[-1] = rng.multivariate_normal(m_filt[-1], P_filt[-1])
    for t in range(T - 2, -1, -1):
        A = np.linalg.solve(P_pred[t + 1], F @ P_filt[t]).T
        mean = m_filt[t] + A @ (states[t + 1] - F @ m_filt[t])
        cov = P_filt[t] - A @ P_pred[t + 1] @ A.T
        cov = 0.5 * (cov + cov.T) + 1e-12 * eye
        states[t] = rng.multivariate_normal(mean, cov)
    return states[:, 0], float(states[0, 1])


def func_hsa_f0(
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
    """Fixed-N0 local benchmark with no Nbar state or moving-reference law."""
    pi_t = np.asarray(pi_data, dtype=float).reshape(-1)
    pi_tm1 = np.asarray(pi_prev_data, dtype=float).reshape(-1)
    pi_expect = np.asarray(Epi_data, dtype=float).reshape(-1)
    x_t = np.asarray(x_data, dtype=float).reshape(-1)
    x_tm1 = np.asarray(x_prev_data, dtype=float).reshape(-1)
    N_obs = np.asarray(N_data, dtype=float).reshape(-1)
    T = pi_t.size
    if not (pi_tm1.size == pi_expect.size == x_t.size == x_tm1.size == N_obs.size == T):
        raise ValueError("All F0 input series must have the same length.")
    pri = _common_priors(priors or {})
    opts = dict(opts or {})
    rng = np.random.default_rng(_getd(opts, "seed", None))
    store_every = int(max(1, _getd(opts, "store_every", 1)))
    coefficient_constraints = dict(_getd(opts, "coefficient_constraints", {}) or {})
    coefficient_constraints["enabled"] = True
    coefficient_constraints.setdefault("max_tries", 2000)
    enforce_stationary = bool(_getd(opts, "enforce_stationary", True))
    ar2_max_tries = int(_getd(opts, "ar2_max_tries", 2000))
    # Display-only hook installed by the run driver; it never touches the draws.
    progress_callback = _getd(opts, "progress_callback", None)

    alpha = float(pri["mu_alpha"])
    kappa0 = float(pri["mu_kappa0"])
    theta0 = max(float(pri["mu_theta"]), 0.01)
    phi_1 = float(pri["mu_phi"])
    lambda_ez = 0.0 if orth else float(pri["mu_lambda"])
    rho1, rho2 = 0.5, -0.2
    sigma_eta2 = sigma_zeta2 = 1.0
    sigma_u2, sigma_N2 = 0.05, 0.05
    finite = np.isfinite(N_obs)
    if not finite.any():
        raise ValueError("F0 requires at least one finite competition observation.")
    q = np.interp(np.arange(T), np.flatnonzero(finite), N_obs[finite])
    q_lag0 = float(q[0])
    a_t = pi_tm1 - pi_expect
    total_iter = n_burn + n_keep
    n_store = int(n_keep // store_every)
    saved = {name: np.zeros(n_store) for name in (
        "alpha", "kappa_0", "theta0", "phi_1", "lambda_ez", "rho1", "rho2",
        "sigma_e", "sigma_zeta", "sigma_u", "sigma_N", "rho_ez",
    )}
    q_draws = np.zeros((n_store, T))
    kappa_t_draws = np.zeros((n_store, T))
    theta_t_draws = np.zeros((n_store, T))
    constraint_stats: dict[str, int] = {}
    ar2_stats: dict[str, int] = {}
    idx = 0

    for it in range(1, total_iter + 1):
        y = pi_t - pi_expect
        zeta = x_t - phi_1 * x_tm1
        y_adj = y - lambda_ez * zeta
        X = np.column_stack([a_t, x_t / KAPPA_SCALE, -q])

        def _valid(beta: np.ndarray) -> bool:
            return bool(beta[1] > 0.0 and beta[2] > 0.0)

        beta = draw_with_constraints(
            lambda: _sample_beta_gaussian(
                y_adj, X, sigma_eta2,
                np.array([pri["mu_alpha"], pri["mu_kappa0"], pri["mu_theta"]]),
                np.array([pri["sigma_alpha"] ** 2, pri["sigma_kappa0"] ** 2, pri["sigma_theta"] ** 2]),
                rng,
            ),
            ("alpha", "kappa_0", "theta_0"),
            coefficient_constraints,
            validators=[_valid],
            stats=constraint_stats,
        )
        alpha, kappa0, theta0 = map(float, beta)
        if not orth:
            e_base = y - alpha * a_t - (kappa0 / KAPPA_SCALE) * x_t + theta0 * q
            precision = 1.0 / pri["sigma_lambda"] ** 2 + float(zeta @ zeta) / sigma_eta2
            variance = 1.0 / precision
            mean = variance * (pri["mu_lambda"] / pri["sigma_lambda"] ** 2 + float(zeta @ e_base) / sigma_eta2)
            lambda_ez = float(mean + np.sqrt(variance) * rng.standard_normal())
        ytilde_phi = y - alpha * a_t - (kappa0 / KAPPA_SCALE) * x_t + theta0 * q
        phi_1 = _sample_phi_joint(
            x_t=x_t, x_tm1=x_tm1, y_tilde=ytilde_phi,
            lambda_ez=lambda_ez, sigma_zeta2=sigma_zeta2,
            sigma_eta2=sigma_eta2, mu_phi=pri["mu_phi"],
            sigma_phi=pri["sigma_phi"], rng=rng,
        )
        zeta = x_t - phi_1 * x_tm1
        eta = y - alpha * a_t - (kappa0 / KAPPA_SCALE) * x_t + theta0 * q - lambda_ez * zeta
        sigma_zeta2 = _sample_invgamma(pri["a_z"] + 0.5 * T, pri["b_z"] + 0.5 * float(zeta @ zeta), rng)
        sigma_eta2 = _sample_invgamma(pri["a_e"] + 0.5 * T, pri["b_e"] + 0.5 * float(eta @ eta), rng)
        if T >= 2:
            rho1, rho2 = _sample_ar2_coeffs(
                q, sigma_u2, pri["mu_rho1"], pri["sigma_rho1"],
                pri["mu_rho2"], pri["sigma_rho2"], enforce_stationary, rng,
                max_tries=ar2_max_tries, current=(rho1, rho2), stats=ar2_stats,
                initial_lag=q_lag0,
            )
            second_lag = np.concatenate([[q_lag0], q[:-2]])
            u = q[1:] - rho1 * q[:-1] - rho2 * second_lag
            sigma_u2 = _sample_invgamma(pri["a_u"] + 0.5 * u.size, pri["b_u"] + 0.5 * float(u @ u), rng)
        residual_N = N_obs[finite] - q[finite]
        sigma_N2 = _sample_invgamma(pri["a_N"] + 0.5 * residual_N.size, pri["b_N"] + 0.5 * float(residual_N @ residual_N), rng)
        q, q_lag0 = _fixed_reference_ffbs(
            y=y, a_t=a_t, x_t=x_t, zeta=zeta, N_obs=N_obs,
            alpha=alpha, kappa0_eff=kappa0 / KAPPA_SCALE, theta0=theta0,
            lambda_ez=lambda_ez, rho1=rho1, rho2=rho2,
            sigma_eta2=sigma_eta2, sigma_u2=sigma_u2, sigma_N2=sigma_N2,
            m0=np.zeros(2), P0=10.0 * np.eye(2), rng=rng,
        )
        if it > n_burn and (it - n_burn) % store_every == 0:
            sigma_e = float(np.sqrt(lambda_ez**2 * sigma_zeta2 + sigma_eta2))
            saved["alpha"][idx] = alpha
            saved["kappa_0"][idx] = kappa0 / KAPPA_SCALE
            saved["theta0"][idx] = theta0
            saved["phi_1"][idx] = phi_1
            saved["lambda_ez"][idx] = lambda_ez
            saved["rho1"][idx], saved["rho2"][idx] = rho1, rho2
            saved["sigma_e"][idx] = sigma_e
            saved["sigma_zeta"][idx] = np.sqrt(sigma_zeta2)
            saved["sigma_u"][idx] = np.sqrt(sigma_u2)
            saved["sigma_N"][idx] = np.sqrt(sigma_N2)
            saved["rho_ez"][idx] = 0.0 if orth else lambda_ez * np.sqrt(sigma_zeta2) / max(sigma_e, 1e-12)
            q_draws[idx] = q
            kappa_t_draws[idx] = kappa0 / KAPPA_SCALE
            theta_t_draws[idx] = theta0
            idx += 1

        if progress_callback is not None:
            progress_callback(it, total_iter)

    return {
        "alpha": _summary(saved["alpha"]),
        "kappa_0": _summary(saved["kappa_0"]),
        "theta_0": _summary(saved["theta0"]),
        "phi_1": _summary(saved["phi_1"]),
        "lambda_ez": _summary(saved["lambda_ez"]),
        "rho": _summary(saved["rho_ez"]),
        "rho1": _summary(saved["rho1"]),
        "rho2": _summary(saved["rho2"]),
        "sigma_e": _summary(saved["sigma_e"]),
        "sigma_zeta": _summary(saved["sigma_zeta"]),
        "sigma_u": _summary(saved["sigma_u"]),
        "sigma_N": _summary(saved["sigma_N"]),
        "state_draws": {
            "N_deviation": q_draws,
            "kappa_t": kappa_t_draws,
            "theta_t": theta_t_draws,
        },
        "priors": priors or {},
        "opts": opts,
        "model": {
            "model_hierarchy": "F0",
            "reference_design": "fixed_reference",
            "moving_reference": False,
            "state_sampler": "fixed_reference_ffbs",
            "state_vector": "(N_deviation_t, N_deviation_t-1); no Nbar state",
            "kappa_scale": KAPPA_SCALE,
            "stored_units": "physical",
            "coefficient_constraint_stats": constraint_stats_summary(constraint_stats),
            "ar2_stationarity": _ar2_stats_summary(ar2_stats),
        },
    }


__all__ = ["func_hsa_f0", "func_hsa_u", "func_hsa_restricted"]
