"""Gibbs sampler for the six-state firm/establishment measurement system.

This module is deliberately separate from the historical three-state sampler.
It is selected only when quarterly establishment *levels* are supplied, and it
implements the model in which N and E have distinct trends and distinct AR(2)
cycles whose innovations may be correlated.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from nkpc_hsa.gibbs.common.competition import finite_N_residuals
from nkpc_hsa.gibbs.common.constraints import constraint_stats_summary, draw_with_constraints
from nkpc_hsa.gibbs.common.joint_ffbs import sample_joint_ne_states_ffbs
from nkpc_hsa.gibbs.hsa_dynamic.model import _sample_invwishart
from nkpc_hsa.gibbs.hsa_full.model import (
    KAPPA_SCALE,
    _common_priors,
    _getd,
    _init_states,
    _kappa_t_constraint_validators,
    _sample_beta_gaussian,
    _sample_invgamma,
    _sample_phi_joint,
    _summary,
)

__all__ = ["func_nkpc_hsa_joint_ne"]


def _is_stationary_ar2(r1: float, r2: float) -> bool:
    return abs(r2) < 1.0 and r1 + r2 < 1.0 and r2 - r1 < 1.0


def _ar_design(states: np.ndarray, current_index: int, lag_index: int) -> tuple[np.ndarray, np.ndarray]:
    y = states[1:, current_index]
    X = np.column_stack([states[:-1, current_index], states[:-1, lag_index]])
    return y, X


def _sample_correlated_ar2(
    *,
    y: np.ndarray,
    X: np.ndarray,
    other_resid: np.ndarray,
    covariance: np.ndarray,
    own_index: int,
    prior_mean: np.ndarray,
    prior_sd: np.ndarray,
    current: tuple[float, float],
    enforce_stationary: bool,
    max_tries: int,
    rng: np.random.Generator,
    stats: dict[str, int],
) -> tuple[float, float]:
    other_index = 1 - own_index
    slope = covariance[own_index, other_index] / covariance[other_index, other_index]
    conditional_var = (
        covariance[own_index, own_index]
        - covariance[own_index, other_index] ** 2 / covariance[other_index, other_index]
    )
    conditional_var = max(float(conditional_var), 1e-12)
    adjusted_y = y - slope * other_resid
    prior_prec = np.diag(1.0 / np.square(prior_sd))
    post_cov = np.linalg.inv(X.T @ X / conditional_var + prior_prec)
    post_mean = post_cov @ (X.T @ adjusted_y / conditional_var + prior_prec @ prior_mean)
    stats["draw_calls"] = stats.get("draw_calls", 0) + 1
    tries = 1 if not enforce_stationary else max_tries
    for attempt in range(1, tries + 1):
        draw = rng.multivariate_normal(post_mean, (post_cov + post_cov.T) / 2.0)
        if not enforce_stationary or _is_stationary_ar2(float(draw[0]), float(draw[1])):
            stats["proposals"] = stats.get("proposals", 0) + attempt
            stats["rejections"] = stats.get("rejections", 0) + attempt - 1
            return float(draw[0]), float(draw[1])
    stats["proposals"] = stats.get("proposals", 0) + tries
    stats["rejections"] = stats.get("rejections", 0) + tries
    stats["fallbacks"] = stats.get("fallbacks", 0) + 1
    if _is_stationary_ar2(*current):
        return current
    return 0.0, 0.0


def _sample_drift_and_variance(
    path: np.ndarray,
    drift: float,
    variance: float,
    *,
    mu: float,
    sigma: float,
    a: float,
    b: float,
    rng: np.random.Generator,
) -> tuple[float, float]:
    differences = np.diff(path)
    post_var = 1.0 / (1.0 / sigma**2 + differences.size / variance)
    post_mean = post_var * (mu / sigma**2 + float(np.sum(differences)) / variance)
    drift = float(post_mean + np.sqrt(post_var) * rng.standard_normal())
    residual = differences - drift
    variance = _sample_invgamma(a + 0.5 * residual.size, b + 0.5 * float(residual @ residual), rng)
    return drift, variance


def _finite_level_residuals(obs: np.ndarray, trend: np.ndarray, cycle: np.ndarray) -> np.ndarray:
    mask = np.isfinite(obs)
    return obs[mask] - trend[mask] - cycle[mask]


def func_nkpc_hsa_joint_ne(
    pi_data,
    pi_prev_data,
    Epi_data,
    x_data,
    x_prev_data,
    N_data,
    E_data,
    n_burn: int,
    n_keep: int,
    priors: dict[str, Any] | None = None,
    opts: dict[str, Any] | None = None,
    *,
    orth: bool = False,
    const_theta: bool = False,
) -> dict[str, Any]:
    """Estimate HSA steady or const-theta with one six-state FFBS draw."""
    pi_t = np.asarray(pi_data, dtype=float).reshape(-1)
    pi_tm1 = np.asarray(pi_prev_data, dtype=float).reshape(-1)
    pi_expect = np.asarray(Epi_data, dtype=float).reshape(-1)
    x_t = np.asarray(x_data, dtype=float).reshape(-1)
    x_tm1 = np.asarray(x_prev_data, dtype=float).reshape(-1)
    N_obs = np.asarray(N_data, dtype=float).reshape(-1)
    E_obs = np.asarray(E_data, dtype=float).reshape(-1)
    T = pi_t.size
    if any(v.size != T for v in (pi_tm1, pi_expect, x_t, x_tm1, N_obs, E_obs)):
        raise ValueError("All joint N--E model inputs must have the same length.")
    if T < 3 or np.isfinite(N_obs).sum() < 3 or np.isfinite(E_obs).sum() < 3:
        raise ValueError("The joint N--E model needs T >= 3 and at least three observations of each level.")

    raw_priors = dict(priors or {})
    pri = _common_priors(raw_priors)
    opts = dict(opts or {})
    rng = np.random.default_rng(_getd(opts, "seed", None))
    store_every = int(max(1, _getd(opts, "store_every", 1)))
    n_store = int(n_keep // store_every)
    if n_burn < 0 or n_store <= 0:
        raise ValueError("n_burn must be nonnegative and n_keep >= store_every.")
    enforce_stationary = bool(_getd(opts, "enforce_stationary", True))
    max_tries = int(max(1, _getd(opts, "ar2_max_tries", 2000)))
    coefficient_constraints = _getd(opts, "coefficient_constraints", {})
    constraint_stats: dict[str, int] = {}
    ar_N_stats: dict[str, int] = {}
    ar_E_stats: dict[str, int] = {}

    def p(name: str, default: float) -> float:
        return float(_getd(raw_priors, name, default))

    mu_Erho = np.array([p("mu_rho_E1", pri["mu_rho1"]), p("mu_rho_E2", pri["mu_rho2"])])
    sd_Erho = np.array([p("sigma_rho_E1", pri["sigma_rho1"]), p("sigma_rho_E2", pri["sigma_rho2"])])
    mu_nE, sd_nE = p("mu_n_E", pri["mu_n"]), p("sigma_n_E", pri["sigma_n"])
    a_epsE, b_epsE = p("a_epsE", pri["a_eps"]), p("b_epsE", pri["b_eps"])
    nu_NE = p("nu_NE", 5.0)
    S_NE = np.asarray(raw_priors.get("S_NE", [[2.0 * pri["b_u"], 0.0], [0.0, 2.0 * p("b_uE", pri["b_u"])]]), dtype=float)
    if S_NE.shape != (2, 2) or nu_NE <= 1.0 or np.linalg.eigvalsh(S_NE).min() <= 0.0:
        raise ValueError("S_NE must be positive definite 2x2 and nu_NE must exceed 1.")

    alpha = float(_getd(opts, "alpha0", pri["mu_alpha"]))
    kappa0 = float(_getd(opts, "kappa00", pri["mu_kappa0"]))
    delta = float(_getd(opts, "delta0", pri["mu_delta"]))
    theta = float(_getd(opts, "theta00", pri["mu_theta"])) if const_theta else 0.0
    phi = float(_getd(opts, "phi10", pri["mu_phi"]))
    lambda_ez = 0.0 if orth else float(_getd(opts, "lambda0", pri["mu_lambda"]))
    rho_N1, rho_N2 = float(_getd(opts, "rho_N10", pri["mu_rho1"])), float(_getd(opts, "rho_N20", pri["mu_rho2"]))
    rho_E1, rho_E2 = float(_getd(opts, "rho_E10", mu_Erho[0])), float(_getd(opts, "rho_E20", mu_Erho[1]))
    n_N, n_E = float(_getd(opts, "n_N0", pri["mu_n"])), float(_getd(opts, "n_E0", mu_nE))
    sigma_eta2 = float(_getd(opts, "sigma_eta20", 1.0))
    sigma_zeta2 = float(_getd(opts, "sigma_zeta20", 1.0))
    Sigma_NE = np.asarray(_getd(opts, "Sigma_NE0", S_NE / max(nu_NE + 3.0, 1.0)), dtype=float)
    sigma_epsN2 = float(_getd(opts, "sigma_epsN20", 0.01))
    sigma_epsE2 = float(_getd(opts, "sigma_epsE20", 0.01))
    sigma_N2 = float(_getd(opts, "sigma_N20", 0.01))
    sigma_E2 = float(_getd(opts, "sigma_E20", 0.01))

    Nbar, Nhat = _init_states(N_obs)
    Ebar, Ehat = _init_states(E_obs)
    states = np.zeros((T, 6))
    states[:, [0, 2, 3, 5]] = np.column_stack([Nhat, Nbar, Ehat, Ebar])
    states[1:, 1], states[1:, 4] = Nhat[:-1], Ehat[:-1]

    m0 = np.array([
        p("m0_Nhat", pri["m0_Nhat"]), p("m0_Nhat_lag", pri["m0_Nhat_lag"]), p("m0_Nbar", pri["m0_Nbar"]),
        p("m0_Ehat", 0.0), p("m0_Ehat_lag", 0.0), p("m0_Ebar", 0.0),
    ])
    states[0, 1], states[0, 4] = m0[1], m0[4]
    P0 = np.diag([
        p("P0_Nhat", pri["P0_Nhat"]), p("P0_Nhat_lag", pri["P0_Nhat_lag"]), p("P0_Nbar", pri["P0_Nbar"]),
        p("P0_Ehat", 10.0), p("P0_Ehat_lag", 10.0), p("P0_Ebar", 10.0),
    ])
    y = pi_t - pi_expect
    a_t = pi_tm1 - pi_expect
    lambda_prec = 0.0 if orth else 1.0 / pri["sigma_lambda"] ** 2

    names = ["alpha", "kappa_0", "delta", "phi_1", "lambda_ez", "rho", "rho_N1", "rho_N2", "rho_E1", "rho_E2", "n_N", "n_E", "sigma_e", "sigma_eta", "sigma_zeta", "sigma_uN", "sigma_uE", "sigma_epsN", "sigma_epsE", "sigma_N", "sigma_E", "rho_NE"]
    if const_theta:
        names.append("theta")
    scalar = {name: np.zeros(n_store) for name in names}
    paths = {name: np.zeros((n_store, T)) for name in ("Nbar", "Nhat", "Ebar", "Ehat", "kappa_t")}
    if const_theta:
        paths["theta_t"] = np.zeros((n_store, T))

    for it in range(1, n_burn + n_keep + 1):
        zeta = x_t - phi * x_tm1
        columns = [a_t, x_t / KAPPA_SCALE, x_t * Nbar / KAPPA_SCALE]
        means = [pri["mu_alpha"], pri["mu_kappa0"], pri["mu_delta"]]
        variances = [pri["sigma_alpha"] ** 2, pri["sigma_kappa0"] ** 2, pri["sigma_delta"] ** 2]
        beta_names = ["alpha", "kappa_0", "delta"]
        if const_theta:
            columns.append(-Nhat)
            means.append(pri["mu_theta"])
            variances.append(pri["sigma_theta"] ** 2)
            beta_names.append("theta")
        beta = draw_with_constraints(
            lambda: _sample_beta_gaussian(y - lambda_ez * zeta, np.column_stack(columns), sigma_eta2, np.asarray(means), np.asarray(variances), rng),
            tuple(beta_names), coefficient_constraints,
            validators=_kappa_t_constraint_validators(Nbar, coefficient_constraints), stats=constraint_stats,
        )
        alpha, kappa0, delta = map(float, beta[:3])
        theta = float(beta[3]) if const_theta else 0.0
        kappa_t = kappa0 + delta * Nbar
        e_base = y - alpha * a_t - kappa_t * x_t / KAPPA_SCALE + theta * Nhat
        if orth:
            lambda_ez = 0.0
        else:
            post_var = 1.0 / (lambda_prec + float(zeta @ zeta) / sigma_eta2)
            post_mean = post_var * (pri["mu_lambda"] * lambda_prec + float(zeta @ e_base) / sigma_eta2)
            lambda_ez = float(post_mean + np.sqrt(post_var) * rng.standard_normal())
        phi = _sample_phi_joint(
            x_t=x_t, x_tm1=x_tm1, y_tilde=e_base, lambda_ez=lambda_ez,
            sigma_zeta2=sigma_zeta2, sigma_eta2=sigma_eta2,
            mu_phi=pri["mu_phi"], sigma_phi=pri["sigma_phi"], rng=rng,
        )
        zeta = x_t - phi * x_tm1
        eta = y - alpha * a_t - kappa_t * x_t / KAPPA_SCALE + theta * Nhat - lambda_ez * zeta
        sigma_zeta2 = _sample_invgamma(pri["a_z"] + T / 2, pri["b_z"] + float(zeta @ zeta) / 2, rng)
        sigma_eta2 = _sample_invgamma(pri["a_e"] + T / 2, pri["b_e"] + float(eta @ eta) / 2, rng)

        y_N, X_N = _ar_design(states, 0, 1)
        y_E, X_E = _ar_design(states, 3, 4)
        resid_E = y_E - X_E @ np.array([rho_E1, rho_E2])
        rho_N1, rho_N2 = _sample_correlated_ar2(
            y=y_N, X=X_N, other_resid=resid_E, covariance=Sigma_NE, own_index=0,
            prior_mean=np.array([pri["mu_rho1"], pri["mu_rho2"]]),
            prior_sd=np.array([pri["sigma_rho1"], pri["sigma_rho2"]]), current=(rho_N1, rho_N2),
            enforce_stationary=enforce_stationary, max_tries=max_tries, rng=rng, stats=ar_N_stats,
        )
        resid_N = y_N - X_N @ np.array([rho_N1, rho_N2])
        rho_E1, rho_E2 = _sample_correlated_ar2(
            y=y_E, X=X_E, other_resid=resid_N, covariance=Sigma_NE, own_index=1,
            prior_mean=mu_Erho, prior_sd=sd_Erho, current=(rho_E1, rho_E2),
            enforce_stationary=enforce_stationary, max_tries=max_tries, rng=rng, stats=ar_E_stats,
        )
        innovations = np.column_stack([y_N - X_N @ [rho_N1, rho_N2], y_E - X_E @ [rho_E1, rho_E2]])
        Sigma_NE = _sample_invwishart(nu_NE + innovations.shape[0], S_NE + innovations.T @ innovations, rng)
        n_N, sigma_epsN2 = _sample_drift_and_variance(Nbar, n_N, sigma_epsN2, mu=pri["mu_n"], sigma=pri["sigma_n"], a=pri["a_eps"], b=pri["b_eps"], rng=rng)
        n_E, sigma_epsE2 = _sample_drift_and_variance(Ebar, n_E, sigma_epsE2, mu=mu_nE, sigma=sd_nE, a=a_epsE, b=b_epsE, rng=rng)
        residual_Nobs = finite_N_residuals(N_obs, Nhat, Nbar)
        residual_Eobs = _finite_level_residuals(E_obs, Ebar, Ehat)
        sigma_N2 = _sample_invgamma(pri["a_N"] + residual_Nobs.size / 2, pri["b_N"] + float(residual_Nobs @ residual_Nobs) / 2, rng)
        sigma_E2 = _sample_invgamma(pri["a_E"] + residual_Eobs.size / 2, pri["b_E"] + float(residual_Eobs @ residual_Eobs) / 2, rng)

        y_state = y - alpha * a_t - kappa0 * x_t / KAPPA_SCALE - lambda_ez * zeta
        sd_uN, sd_uE = np.sqrt(Sigma_NE[0, 0]), np.sqrt(Sigma_NE[1, 1])
        rho_NE = float(Sigma_NE[0, 1] / (sd_uN * sd_uE))
        Nbar, Nhat, Ebar, Ehat, states = sample_joint_ne_states_ffbs(
            N_obs=N_obs, E_obs=E_obs, y_tilde=y_state,
            h_nhat=np.full(T, -theta), h_nbar=delta * x_t / KAPPA_SCALE,
            rho_N1=rho_N1, rho_N2=rho_N2, rho_E1=rho_E1, rho_E2=rho_E2,
            n_N=n_N, n_E=n_E, sigma_eta2=sigma_eta2,
            sigma_uN=sd_uN, sigma_uE=sd_uE, rho_NE=rho_NE,
            sigma_epsN=np.sqrt(sigma_epsN2), sigma_epsE=np.sqrt(sigma_epsE2),
            sigma_N2=sigma_N2, sigma_E2=sigma_E2, m0=m0, P0=P0, rng=rng,
        )
        kappa_t = kappa0 + delta * Nbar

        if it > n_burn and (it - n_burn) % store_every == 0:
            j = (it - n_burn) // store_every - 1
            sigma_e = float(np.sqrt(lambda_ez**2 * sigma_zeta2 + sigma_eta2))
            values = {
                "alpha": alpha, "kappa_0": kappa0 / KAPPA_SCALE, "delta": delta / KAPPA_SCALE,
                "phi_1": phi, "lambda_ez": lambda_ez,
                "rho": 0.0 if orth else lambda_ez * np.sqrt(sigma_zeta2) / max(sigma_e, 1e-12),
                "rho_N1": rho_N1, "rho_N2": rho_N2, "rho_E1": rho_E1, "rho_E2": rho_E2,
                "n_N": n_N, "n_E": n_E, "sigma_e": sigma_e, "sigma_eta": np.sqrt(sigma_eta2),
                "sigma_zeta": np.sqrt(sigma_zeta2), "sigma_uN": sd_uN, "sigma_uE": sd_uE,
                "sigma_epsN": np.sqrt(sigma_epsN2), "sigma_epsE": np.sqrt(sigma_epsE2),
                "sigma_N": np.sqrt(sigma_N2), "sigma_E": np.sqrt(sigma_E2), "rho_NE": rho_NE,
            }
            if const_theta:
                values["theta"] = theta
            for key, value in values.items():
                scalar[key][j] = value
            for key, value in (("Nbar", Nbar), ("Nhat", Nhat), ("Ebar", Ebar), ("Ehat", Ehat)):
                paths[key][j] = value
            paths["kappa_t"][j] = kappa_t / KAPPA_SCALE
            if const_theta:
                paths["theta_t"][j] = theta

    def stat_summary(stats: dict[str, int]) -> dict[str, float | int]:
        proposals = int(stats.get("proposals", 0))
        rejections = int(stats.get("rejections", 0))
        return {**stats, "proposal_rejection_rate": rejections / proposals if proposals else 0.0}

    return {
        **{key: _summary(value) for key, value in scalar.items()},
        "state_draws": paths,
        "priors": raw_priors,
        "opts": opts,
        "model": {
            "joint_establishment_state": True,
            "state_sampler": "joint_ffbs",
            "state_vector": "[Nhat_t, Nhat_{t-1}, Nbar_t, Ehat_t, Ehat_{t-1}, Ebar_t]'",
            "N_measurement_equation": "N_obs_t = Nhat_t + Nbar_t + nu_N,t",
            "E_measurement_equation": "E_obs_t = Ehat_t + Ebar_t + nu_E,t",
            "cycle_covariance": "Cov(u_N,t,u_E,t) = rho_NE*sigma_uN*sigma_uE",
            "inflation_establishment_loading": 0.0,
            "theta_specification": "constant" if const_theta else "zero (steady)",
            "stored_units": "physical",
            "kappa_scale": KAPPA_SCALE,
            "coefficient_constraint_stats": constraint_stats_summary(constraint_stats),
            "N_ar2_stationarity": stat_summary(ar_N_stats),
            "E_ar2_stationarity": stat_summary(ar_E_stats),
        },
    }
