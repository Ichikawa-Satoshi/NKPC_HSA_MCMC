from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np


Family = Literal["ces", "dynamic", "steady", "full"]

# Default prior hyperparameters in the physical units used by stored posterior
# draws. These match the sampler defaults in func_gibbs; run-specific priors
# from priors.json should be passed to the chib_* entry points so the prior
# and posterior-ordinate terms match the priors that actually generated the
# posterior draws.
_DEFAULT_PRIORS: dict[str, tuple[float, float]] = {
    "alpha": (0.5, 0.2),
    "kappa": (0.1, 0.2),
    "kappa_0": (0.1, 0.2),
    "delta": (0.1, 0.2),
    "theta": (0.1, 0.2),
    "theta_0": (0.1, 0.2),
    "gamma": (0.1, 0.2),
    "phi_1": (0.7, 0.2),
    "lambda_ez": (0.0, 0.5),
    "rho_1": (0.5, 0.2),
    "rho_2": (-0.5, 0.2),
    "n": (0.0, 0.1),
}

_DEFAULT_IG: dict[str, float] = {
    "a_e": 2.0,
    "b_e": 2.0,
    "a_z": 0.001,
    "b_z": 0.001,
    "a_u": 2.0,
    "b_u": 2.0,
    "a_eps": 2.0,
    "b_eps": 2.0,
    "a_N": 2.0,
    "b_N": 2.0,
}


def _resolve_priors(priors: dict | None) -> dict[str, float | tuple[float, float]]:
    """Merge user/run priors (priors_*.yaml shape, physical units) with defaults."""
    out: dict[str, float | tuple[float, float]] = {}
    out.update(_DEFAULT_PRIORS)
    out.update(_DEFAULT_IG)
    for key, value in (priors or {}).items():
        if key in _DEFAULT_IG:
            out[key] = float(value)
        elif key in _DEFAULT_PRIORS:
            if isinstance(value, dict):
                out[key] = (float(value["mean"]), float(value["sd"]))
            else:
                out[key] = (float(value[0]), float(value[1]))
    return out


@dataclass(frozen=True)
class MarginalLikelihoodResult:
    log_marginal_likelihood: float
    log_likelihood: float
    log_prior: float
    log_posterior_ordinate: float
    n_draws: int
    method: str


def _finite_1d(x) -> np.ndarray:
    arr = np.asarray(x, dtype=float).reshape(-1)
    return arr[np.isfinite(arr)]


def _draws(ds, var: str) -> np.ndarray:
    if var not in ds:
        raise KeyError(f"Missing posterior variable: {var}")
    return np.asarray(ds[var], dtype=float).reshape(-1)


def _lambda_ez_draws(ds) -> np.ndarray:
    """Return lambda_ez draws, reconstructing them from stored correlations if needed."""
    if "lambda_ez" in ds:
        return _draws(ds, "lambda_ez")
    sigma_e = _draws(ds, "sigma_e")
    sigma_zeta = np.maximum(_draws(ds, "sigma_zeta"), 1e-12)
    if "corr_e_zeta" in ds:
        return _draws(ds, "corr_e_zeta") * sigma_e / sigma_zeta
    if "rho" in ds:
        return _draws(ds, "rho") * sigma_e / sigma_zeta
    return np.zeros_like(sigma_e)


def _state_draws(ds, var: str) -> np.ndarray:
    if var not in ds:
        raise KeyError(f"Missing posterior state variable: {var}")
    arr = np.asarray(ds[var], dtype=float)
    return arr.reshape(-1, arr.shape[-1])


def _logmeanexp(values: np.ndarray) -> float:
    vals = _finite_1d(values)
    if vals.size == 0:
        return -np.inf
    m = float(np.max(vals))
    return m + float(np.log(np.mean(np.exp(vals - m))))


def _log_norm_pdf(x: np.ndarray | float, mu: np.ndarray | float, sd: float) -> np.ndarray | float:
    sd = float(sd)
    return -0.5 * np.log(2.0 * np.pi * sd * sd) - 0.5 * ((np.asarray(x) - mu) / sd) ** 2


def _log_ig_pdf_var(x: float, a: float, b: float) -> float:
    from math import lgamma

    x = float(x)
    if not np.isfinite(x) or x <= 0.0:
        return -np.inf
    return float(a * np.log(b) - lgamma(a) - (a + 1.0) * np.log(x) - b / x)


def _log_mvn_pdf(x: np.ndarray, mean: np.ndarray, cov: np.ndarray) -> float:
    x = np.asarray(x, dtype=float).reshape(-1)
    mean = np.asarray(mean, dtype=float).reshape(-1)
    cov = np.asarray(cov, dtype=float)
    cov = (cov + cov.T) / 2.0
    sign, logdet = np.linalg.slogdet(cov)
    if sign <= 0:
        vals, vecs = np.linalg.eigh(cov)
        vals = np.maximum(vals, 1e-10)
        cov = vecs @ np.diag(vals) @ vecs.T
        sign, logdet = np.linalg.slogdet(cov)
    diff = x - mean
    sol = np.linalg.solve(cov, diff)
    return float(-0.5 * (x.size * np.log(2.0 * np.pi) + logdet + diff @ sol))


def _log_gaussian_likelihood(resid: np.ndarray, variance: float) -> float:
    resid = np.asarray(resid, dtype=float).reshape(-1)
    variance = float(variance)
    if variance <= 0.0:
        return -np.inf
    return float(-0.5 * resid.size * np.log(2.0 * np.pi * variance) - 0.5 * np.sum(resid**2) / variance)


def _kalman_loglik_n_only(star, data) -> float:
    T = data["N"].size
    rho1, rho2 = star["rho_1"], star["rho_2"]
    sigma_u2, sigma_eps2 = star["sigma_u2"], star["sigma_eps2"]
    F = np.array([[rho1, rho2, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=float)
    c = np.array([0.0, 0.0, star["n"]], dtype=float)
    Q = np.diag([sigma_u2, 1e-10, sigma_eps2])
    m = np.array([0.0, 0.0, data["N"][0]], dtype=float)
    P = np.eye(3) * 10.0
    loglik = 0.0
    H = np.array([[1.0, 0.0, 1.0]], dtype=float)
    R = np.array([[float(star.get("sigma_N2", 1e-6))]], dtype=float)
    for t in range(T):
        if t > 0:
            m = F @ m + c
            P = F @ P @ F.T + Q
        y = np.array([data["N"][t]], dtype=float)
        S = H @ P @ H.T + R
        v = y - H @ m
        loglik += _log_mvn_pdf(v, np.zeros(1), S)
        K = P @ H.T @ np.linalg.inv(S)
        m = m + K @ v
        P = (np.eye(3) - K @ H) @ P
        P = (P + P.T) / 2.0
    return float(loglik)


def _sample_star(ds, vars_: list[str]) -> dict[str, float]:
    out: dict[str, float] = {}
    for var in vars_:
        if var == "lambda_ez":
            out[var] = float(np.nanmean(_lambda_ez_draws(ds)))
        else:
            out[var] = float(np.nanmean(_draws(ds, var)))
    if "sigma_e" in out and "lambda_ez" in out and "sigma_zeta" in out:
        sigma_eta2_draws = _sigma_eta2_draws(ds)
        out["sigma_eta2"] = float(np.nanmean(sigma_eta2_draws))
    if "sigma_zeta" in out:
        out["sigma_zeta2"] = float(np.nanmean(_draws(ds, "sigma_zeta") ** 2))
    if "sigma_u" in out:
        out["sigma_u2"] = float(np.nanmean(_draws(ds, "sigma_u") ** 2))
    if "sigma_eps" in out:
        out["sigma_eps2"] = float(np.nanmean(_draws(ds, "sigma_eps") ** 2))
    if "sigma_N" in out:
        out["sigma_N2"] = float(np.nanmean(_draws(ds, "sigma_N") ** 2))
    return out


def _star_vars_with_sigma_N(ds, vars_: list[str]) -> list[str]:
    if "sigma_N" in ds:
        return [*vars_, "sigma_N"]
    return list(vars_)


def _sigma_eta2_draws(ds) -> np.ndarray:
    sigma_e = _draws(ds, "sigma_e")
    sigma_zeta = _draws(ds, "sigma_zeta")
    lambda_ez = _lambda_ez_draws(ds)
    out = sigma_e**2 - (lambda_ez**2) * (sigma_zeta**2)
    return np.maximum(out, 1e-10)


def _ces_star(ds) -> dict[str, float]:
    return _sample_star(ds, ["alpha", "kappa", "phi_1", "lambda_ez", "sigma_e", "sigma_zeta"])


def _dynamic_star(ds) -> dict[str, float]:
    return _sample_star(
        ds,
        _star_vars_with_sigma_N(
            ds,
            ["alpha", "kappa", "theta", "phi_1", "lambda_ez", "rho_1", "rho_2", "n", "sigma_e", "sigma_zeta", "sigma_u", "sigma_eps"],
        ),
    )


def _steady_star(ds) -> dict[str, float]:
    return _sample_star(
        ds,
        _star_vars_with_sigma_N(
            ds,
            ["alpha", "kappa_0", "delta", "phi_1", "lambda_ez", "rho_1", "rho_2", "n", "sigma_e", "sigma_zeta", "sigma_u", "sigma_eps"],
        ),
    )


def _full_star(ds) -> dict[str, float]:
    return _sample_star(
        ds,
        [
            "alpha",
            "kappa_0",
            "delta",
            "theta_0",
            "gamma",
            "phi_1",
            "lambda_ez",
            "rho_1",
            "rho_2",
            "n",
            "sigma_e",
            "sigma_zeta",
            "sigma_u",
            "sigma_eps",
        ],
    )


def _ces_beta_cond_logpdf(star, data, lambda_ez, phi_1, sigma_eta2, pri) -> float:
    y = data["pi"] - data["pi_expect"]
    a_t = data["pi_prev"] - data["pi_expect"]
    zeta = data["x"] - phi_1 * data["x_prev"]
    X = np.column_stack([a_t, data["x"]])
    prior_mean = np.array([pri["alpha"][0], pri["kappa"][0]], dtype=float)
    prior_prec = np.diag([1.0 / pri["alpha"][1] ** 2, 1.0 / pri["kappa"][1] ** 2])
    cov = np.linalg.inv(X.T @ X / sigma_eta2 + prior_prec)
    mean = cov @ (X.T @ (y - lambda_ez * zeta) / sigma_eta2 + prior_prec @ prior_mean)
    return _log_mvn_pdf(np.array([star["alpha"], star["kappa"]]), mean, cov)


def _lambda_cond_logpdf(lambda_star, e_base, zeta, sigma_eta2, pri, *, orth: bool) -> float:
    if orth:
        return 0.0
    mu_lambda, sd_lambda = pri["lambda_ez"]
    prior_prec = 1.0 / sd_lambda**2
    var = 1.0 / (prior_prec + float(np.sum(zeta**2)) / sigma_eta2)
    mean = var * (mu_lambda * prior_prec + float(np.dot(zeta, e_base)) / sigma_eta2)
    return float(_log_norm_pdf(lambda_star, mean, np.sqrt(var)))


def _phi_cond_logpdf(phi_star, *, x, x_prev, y_tilde, lambda_ez, sigma_zeta2, sigma_eta2, pri) -> float:
    mu_phi, sd_phi = pri["phi_1"]
    prec = (
        1.0 / sd_phi**2
        + float(np.sum(x_prev**2)) / sigma_zeta2
        + (lambda_ez**2) * float(np.sum(x_prev**2)) / sigma_eta2
    )
    mean_num = (
        mu_phi / sd_phi**2
        + float(np.dot(x_prev, x)) / sigma_zeta2
        - lambda_ez * float(np.dot(x_prev, y_tilde - lambda_ez * x)) / sigma_eta2
    )
    return float(_log_norm_pdf(phi_star, mean_num / prec, np.sqrt(1.0 / prec)))


def _rho_cond_logpdf(rho_star: np.ndarray, Nhat: np.ndarray, sigma_u2: float, pri) -> float:
    y = Nhat[2:]
    X = np.column_stack([Nhat[1:-1], Nhat[:-2]])
    prior_mean = np.array([pri["rho_1"][0], pri["rho_2"][0]], dtype=float)
    prior_prec = np.diag([1.0 / pri["rho_1"][1] ** 2, 1.0 / pri["rho_2"][1] ** 2])
    cov = np.linalg.inv(X.T @ X / sigma_u2 + prior_prec)
    mean = cov @ (X.T @ y / sigma_u2 + prior_prec @ prior_mean)
    return _log_mvn_pdf(rho_star, mean, cov)


def _n_cond_logpdf(n_star: float, Nbar: np.ndarray, sigma_eps2: float, pri) -> float:
    mu_n, sd_n = pri["n"]
    dNbar = Nbar[1:] - Nbar[:-1]
    var = 1.0 / (1.0 / sd_n**2 + dNbar.size / sigma_eps2)
    mean = var * (mu_n / sd_n**2 + float(np.sum(dNbar)) / sigma_eps2)
    return float(_log_norm_pdf(n_star, mean, np.sqrt(var)))


def _kalman_loglik_hsa_dynamic(star, data, *, obs_var_n: float = 1e-6) -> float:
    T = data["pi"].size
    rho1, rho2 = star["rho_1"], star["rho_2"]
    sigma_u2, sigma_eps2, sigma_eta2 = star["sigma_u2"], star["sigma_eps2"], star["sigma_eta2"]
    F = np.array([[rho1, rho2, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=float)
    c = np.array([0.0, 0.0, star["n"]], dtype=float)
    Q = np.diag([sigma_u2, 1e-10, sigma_eps2])
    m = np.array([0.0, 0.0, data["N"][0]], dtype=float)
    P = np.eye(3) * 10.0
    loglik = 0.0
    zeta = data["x"] - star["phi_1"] * data["x_prev"]
    loglik += _log_gaussian_likelihood(zeta, star["sigma_zeta2"])
    for t in range(T):
        if t > 0:
            m = F @ m + c
            P = F @ P @ F.T + Q
        det_pi = (
            star["alpha"] * data["pi_prev"][t]
            + (1.0 - star["alpha"]) * data["pi_expect"][t]
            + star["kappa"] * data["x"][t]
            + star["lambda_ez"] * zeta[t]
        )
        H = np.array([[1.0, 0.0, 1.0], [-star["theta"], 0.0, 0.0]], dtype=float)
        y = np.array([data["N"][t], data["pi"][t] - det_pi], dtype=float)
        R = np.diag([float(star.get("sigma_N2", obs_var_n)), sigma_eta2])
        S = H @ P @ H.T + R
        v = y - H @ m
        loglik += _log_mvn_pdf(v, np.zeros(2), S)
        K = P @ H.T @ np.linalg.inv(S)
        m = m + K @ v
        P = (np.eye(3) - K @ H) @ P
        P = (P + P.T) / 2.0
    return float(loglik)


def _kalman_loglik_hsa_steady(star, data, *, obs_var_n: float = 1e-6) -> float:
    T = data["pi"].size
    rho1, rho2 = star["rho_1"], star["rho_2"]
    sigma_u2, sigma_eps2, sigma_eta2 = star["sigma_u2"], star["sigma_eps2"], star["sigma_eta2"]
    F = np.array([[rho1, rho2, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=float)
    c = np.array([0.0, 0.0, star["n"]], dtype=float)
    Q = np.diag([sigma_u2, 1e-10, sigma_eps2])
    m = np.array([0.0, 0.0, data["N"][0]], dtype=float)
    P = np.eye(3) * 10.0
    loglik = 0.0
    zeta = data["x"] - star["phi_1"] * data["x_prev"]
    loglik += _log_gaussian_likelihood(zeta, star["sigma_zeta2"])
    for t in range(T):
        if t > 0:
            m = F @ m + c
            P = F @ P @ F.T + Q
        det_pi = (
            star["alpha"] * data["pi_prev"][t]
            + (1.0 - star["alpha"]) * data["pi_expect"][t]
            + star["kappa_0"] * data["x"][t]
            + star["lambda_ez"] * zeta[t]
        )
        H = np.array([[1.0, 0.0, 1.0], [0.0, 0.0, star["delta"] * data["x"][t]]], dtype=float)
        y = np.array([data["N"][t], data["pi"][t] - det_pi], dtype=float)
        R = np.diag([float(star.get("sigma_N2", obs_var_n)), sigma_eta2])
        S = H @ P @ H.T + R
        v = y - H @ m
        loglik += _log_mvn_pdf(v, np.zeros(2), S)
        K = P @ H.T @ np.linalg.inv(S)
        m = m + K @ v
        P = (np.eye(3) - K @ H) @ P
        P = (P + P.T) / 2.0
    return float(loglik)


def _full_conditional_state_star(ds) -> dict[str, np.ndarray]:
    return {
        "Nhat": np.nanmean(_state_draws(ds, "Nhat"), axis=0),
        "Nbar": np.nanmean(_state_draws(ds, "Nbar"), axis=0),
    }


def _ces_log_likelihood(star, data) -> float:
    y = data["pi"] - data["pi_expect"]
    a_t = data["pi_prev"] - data["pi_expect"]
    zeta = data["x"] - star["phi_1"] * data["x_prev"]
    eta = y - star["alpha"] * a_t - star["kappa"] * data["x"] - star["lambda_ez"] * zeta
    return _log_gaussian_likelihood(zeta, star["sigma_zeta2"]) + _log_gaussian_likelihood(eta, star["sigma_eta2"])


def _ces_conditional_star(ds) -> dict[str, float]:
    return {
        "alpha": float(np.nanmean(_draws(ds, "alpha"))),
        "kappa": float(np.nanmean(_draws(ds, "kappa"))),
        "sigma_e2": float(np.nanmean(_draws(ds, "sigma_e") ** 2)),
    }


def _dynamic_conditional_star(ds) -> dict[str, float]:
    out = {
        "alpha": float(np.nanmean(_draws(ds, "alpha"))),
        "kappa": float(np.nanmean(_draws(ds, "kappa"))),
        "theta": float(np.nanmean(_draws(ds, "theta"))),
        "sigma_e2": float(np.nanmean(_draws(ds, "sigma_e") ** 2)),
    }
    out.update(_dynamic_star(ds))
    return out


def _steady_conditional_star(ds) -> dict[str, float]:
    out = {
        "alpha": float(np.nanmean(_draws(ds, "alpha"))),
        "kappa_0": float(np.nanmean(_draws(ds, "kappa_0"))),
        "delta": float(np.nanmean(_draws(ds, "delta"))),
        "sigma_e2": float(np.nanmean(_draws(ds, "sigma_e") ** 2)),
    }
    out.update(_steady_star(ds))
    return out


def _full_conditional_star(ds) -> dict[str, float | np.ndarray]:
    out: dict[str, float | np.ndarray] = {
        "alpha": float(np.nanmean(_draws(ds, "alpha"))),
        "kappa_0": float(np.nanmean(_draws(ds, "kappa_0"))),
        "delta": float(np.nanmean(_draws(ds, "delta"))),
        "theta_0": float(np.nanmean(_draws(ds, "theta_0"))),
        "gamma": float(np.nanmean(_draws(ds, "gamma"))),
        "sigma_e2": float(np.nanmean(_draws(ds, "sigma_e") ** 2)),
    }
    out.update(_full_star(ds))
    out.update(_full_conditional_state_star(ds))
    return out


def _ces_conditional_log_likelihood(star, data) -> float:
    resid = (
        data["pi"]
        - star["alpha"] * data["pi_prev"]
        - (1.0 - star["alpha"]) * data["pi_expect"]
        - star["kappa"] * data["x"]
    )
    return _log_gaussian_likelihood(resid, star["sigma_e2"])


def _dynamic_conditional_log_likelihood(star, data) -> float:
    # p(pi | x, N_obs, theta) = p(pi, N_obs | x, theta) - p(N_obs | theta_N).
    return _kalman_loglik_hsa_dynamic(star, data) - _kalman_loglik_n_only(star, data) - _log_gaussian_likelihood(
        data["x"] - star["phi_1"] * data["x_prev"],
        star["sigma_zeta2"],
    )


def _steady_conditional_log_likelihood(star, data) -> float:
    return _kalman_loglik_hsa_steady(star, data) - _kalman_loglik_n_only(star, data) - _log_gaussian_likelihood(
        data["x"] - star["phi_1"] * data["x_prev"],
        star["sigma_zeta2"],
    )


def _full_conditional_log_likelihood(star, data) -> float:
    # The full HSA observation equation is nonlinear in the two latent
    # competition states through gamma * Nbar_t * Nhat_t. For this fallback
    # Chib calculation we condition on posterior mean state paths and evaluate
    # the inflation equation likelihood.
    Nhat = np.asarray(star["Nhat"], dtype=float)
    Nbar = np.asarray(star["Nbar"], dtype=float)
    n = min(data["pi"].size, Nhat.size, Nbar.size)
    y = data["pi"][:n] - data["pi_expect"][:n]
    a_t = data["pi_prev"][:n] - data["pi_expect"][:n]
    x = data["x"][:n]
    kappa_t = star["kappa_0"] + star["delta"] * Nbar[:n]
    theta_t = star["theta_0"] + star["gamma"] * Nbar[:n]
    resid = y - star["alpha"] * a_t - kappa_t * x + theta_t * Nhat[:n]
    return _log_gaussian_likelihood(resid, star["sigma_e2"])


def _conditional_beta_logpdf(
    beta_star: np.ndarray,
    y: np.ndarray,
    X: np.ndarray,
    sigma2: float,
    prior_mean: np.ndarray,
    prior_sd: np.ndarray | None = None,
) -> float:
    if prior_sd is None:
        prior_prec = np.eye(prior_mean.size) / 0.2**2
    else:
        prior_prec = np.diag(1.0 / np.asarray(prior_sd, dtype=float) ** 2)
    cov = np.linalg.inv(X.T @ X / sigma2 + prior_prec)
    mean = cov @ (X.T @ y / sigma2 + prior_prec @ prior_mean)
    return _log_mvn_pdf(beta_star, mean, cov)


def _beta_prior_arrays(pri, names: list[str]) -> tuple[np.ndarray, np.ndarray]:
    means = np.array([pri[name][0] for name in names], dtype=float)
    sds = np.array([pri[name][1] for name in names], dtype=float)
    return means, sds


def _ces_conditional_log_ordinate(star, ds, data, pri) -> float:
    sigma_draws = _draws(ds, "sigma_e") ** 2
    y = data["pi"] - data["pi_expect"]
    X = np.column_stack([data["pi_prev"] - data["pi_expect"], data["x"]])
    beta_star = np.array([star["alpha"], star["kappa"]], dtype=float)
    prior_mean, prior_sd = _beta_prior_arrays(pri, ["alpha", "kappa"])
    beta_terms = [
        _conditional_beta_logpdf(beta_star, y, X, sigma2, prior_mean, prior_sd)
        for sigma2 in sigma_draws
    ]
    resid = data["pi"] - star["alpha"] * data["pi_prev"] - (1.0 - star["alpha"]) * data["pi_expect"] - star["kappa"] * data["x"]
    sigma_term = _log_ig_pdf_var(
        star["sigma_e2"], pri["a_e"] + 0.5 * resid.size, pri["b_e"] + 0.5 * float(np.sum(resid**2))
    )
    return float(_logmeanexp(np.array(beta_terms)) + sigma_term)


def _dynamic_conditional_log_ordinate(star, ds, data, pri) -> float:
    sigma_draws = _draws(ds, "sigma_e") ** 2
    Nhat_draws = _state_draws(ds, "Nhat")
    y = data["pi"] - data["pi_expect"]
    a_t = data["pi_prev"] - data["pi_expect"]
    beta_star = np.array([star["alpha"], star["kappa"], star["theta"]], dtype=float)
    prior_mean, prior_sd = _beta_prior_arrays(pri, ["alpha", "kappa", "theta"])
    beta_terms = []
    sigma_terms = []
    for sigma2, Nhat in zip(sigma_draws, Nhat_draws):
        X = np.column_stack([a_t, data["x"], -Nhat])
        beta_terms.append(_conditional_beta_logpdf(beta_star, y, X, sigma2, prior_mean, prior_sd))
        resid = y - star["alpha"] * a_t - star["kappa"] * data["x"] + star["theta"] * Nhat
        sigma_terms.append(
            _log_ig_pdf_var(star["sigma_e2"], pri["a_e"] + 0.5 * resid.size, pri["b_e"] + 0.5 * float(np.sum(resid**2)))
        )
    return float(_logmeanexp(np.array(beta_terms)) + _logmeanexp(np.array(sigma_terms)))


def _steady_conditional_log_ordinate(star, ds, data, pri) -> float:
    sigma_draws = _draws(ds, "sigma_e") ** 2
    Nbar_draws = _state_draws(ds, "Nbar")
    y = data["pi"] - data["pi_expect"]
    a_t = data["pi_prev"] - data["pi_expect"]
    beta_star = np.array([star["alpha"], star["kappa_0"], star["delta"]], dtype=float)
    prior_mean, prior_sd = _beta_prior_arrays(pri, ["alpha", "kappa_0", "delta"])
    beta_terms = []
    sigma_terms = []
    for sigma2, Nbar in zip(sigma_draws, Nbar_draws):
        X = np.column_stack([a_t, data["x"], data["x"] * Nbar])
        beta_terms.append(_conditional_beta_logpdf(beta_star, y, X, sigma2, prior_mean, prior_sd))
        resid = y - star["alpha"] * a_t - (star["kappa_0"] + star["delta"] * Nbar) * data["x"]
        sigma_terms.append(
            _log_ig_pdf_var(star["sigma_e2"], pri["a_e"] + 0.5 * resid.size, pri["b_e"] + 0.5 * float(np.sum(resid**2)))
        )
    return float(_logmeanexp(np.array(beta_terms)) + _logmeanexp(np.array(sigma_terms)))


def _full_conditional_log_ordinate(star, ds, data, pri) -> float:
    sigma_draws = _draws(ds, "sigma_e") ** 2
    Nhat_draws = _state_draws(ds, "Nhat")
    Nbar_draws = _state_draws(ds, "Nbar")
    y = data["pi"] - data["pi_expect"]
    a_t = data["pi_prev"] - data["pi_expect"]
    beta_star = np.array(
        [star["alpha"], star["kappa_0"], star["delta"], star["theta_0"], star["gamma"]],
        dtype=float,
    )
    beta_terms = []
    sigma_terms = []
    prior_mean, prior_sd = _beta_prior_arrays(pri, ["alpha", "kappa_0", "delta", "theta_0", "gamma"])
    for sigma2, Nhat, Nbar in zip(sigma_draws, Nhat_draws, Nbar_draws):
        n = min(y.size, Nhat.size, Nbar.size)
        X = np.column_stack(
            [
                a_t[:n],
                data["x"][:n],
                data["x"][:n] * Nbar[:n],
                -Nhat[:n],
                -(Nhat[:n] * Nbar[:n]),
            ]
        )
        beta_terms.append(_conditional_beta_logpdf(beta_star, y[:n], X, sigma2, prior_mean, prior_sd))
        kappa_t = star["kappa_0"] + star["delta"] * Nbar[:n]
        theta_t = star["theta_0"] + star["gamma"] * Nbar[:n]
        resid = y[:n] - star["alpha"] * a_t[:n] - kappa_t * data["x"][:n] + theta_t * Nhat[:n]
        sigma_terms.append(
            _log_ig_pdf_var(star["sigma_e2"], pri["a_e"] + 0.5 * resid.size, pri["b_e"] + 0.5 * float(np.sum(resid**2)))
        )
    return float(_logmeanexp(np.array(beta_terms)) + _logmeanexp(np.array(sigma_terms)))


def _ces_conditional_log_prior(star, pri) -> float:
    return float(
        _log_norm_pdf(star["alpha"], *pri["alpha"])
        + _log_norm_pdf(star["kappa"], *pri["kappa"])
        + _log_ig_pdf_var(star["sigma_e2"], pri["a_e"], pri["b_e"])
    )


def _dynamic_conditional_log_prior(star, pri) -> float:
    return float(
        _log_norm_pdf(star["alpha"], *pri["alpha"])
        + _log_norm_pdf(star["kappa"], *pri["kappa"])
        + _log_norm_pdf(star["theta"], *pri["theta"])
        + _log_ig_pdf_var(star["sigma_e2"], pri["a_e"], pri["b_e"])
    )


def _steady_conditional_log_prior(star, pri) -> float:
    return float(
        _log_norm_pdf(star["alpha"], *pri["alpha"])
        + _log_norm_pdf(star["kappa_0"], *pri["kappa_0"])
        + _log_norm_pdf(star["delta"], *pri["delta"])
        + _log_ig_pdf_var(star["sigma_e2"], pri["a_e"], pri["b_e"])
    )


def _full_conditional_log_prior(star, pri) -> float:
    return float(
        _log_norm_pdf(star["alpha"], *pri["alpha"])
        + _log_norm_pdf(star["kappa_0"], *pri["kappa_0"])
        + _log_norm_pdf(star["delta"], *pri["delta"])
        + _log_norm_pdf(star["theta_0"], *pri["theta_0"])
        + _log_norm_pdf(star["gamma"], *pri["gamma"])
        + _log_ig_pdf_var(star["sigma_e2"], pri["a_e"], pri["b_e"])
    )


def _log_prior_common(star, pri, *, orth: bool, hsa: bool) -> float:
    out = 0.0
    out += float(_log_norm_pdf(star["alpha"], *pri["alpha"]))
    out += float(_log_norm_pdf(star["phi_1"], *pri["phi_1"]))
    if not orth:
        out += float(_log_norm_pdf(star["lambda_ez"], *pri["lambda_ez"]))
    out += _log_ig_pdf_var(star["sigma_eta2"], pri["a_e"], pri["b_e"])
    out += _log_ig_pdf_var(star["sigma_zeta2"], pri["a_z"], pri["b_z"])
    if hsa:
        out += float(_log_norm_pdf(star["rho_1"], *pri["rho_1"]))
        out += float(_log_norm_pdf(star["rho_2"], *pri["rho_2"]))
        out += float(_log_norm_pdf(star["n"], *pri["n"]))
        out += _log_ig_pdf_var(star["sigma_u2"], pri["a_u"], pri["b_u"])
        out += _log_ig_pdf_var(star["sigma_eps2"], pri["a_eps"], pri["b_eps"])
        if "sigma_N2" in star:
            out += _log_ig_pdf_var(star["sigma_N2"], pri["a_N"], pri["b_N"])
    return float(out)


def _ces_log_prior(star, pri, *, orth: bool) -> float:
    return float(_log_prior_common(star, pri, orth=orth, hsa=False) + _log_norm_pdf(star["kappa"], *pri["kappa"]))


def _dynamic_log_prior(star, pri, *, orth: bool) -> float:
    out = _log_prior_common(star, pri, orth=orth, hsa=True)
    out += float(_log_norm_pdf(star["kappa"], *pri["kappa"]))
    out += float(_log_norm_pdf(star["theta"], *pri["theta"]))
    return float(out)


def _steady_log_prior(star, pri, *, orth: bool) -> float:
    out = _log_prior_common(star, pri, orth=orth, hsa=True)
    out += float(_log_norm_pdf(star["kappa_0"], *pri["kappa_0"]))
    out += float(_log_norm_pdf(star["delta"], *pri["delta"]))
    return float(out)


def _full_log_prior(star, pri, *, orth: bool) -> float:
    out = _log_prior_common(star, pri, orth=orth, hsa=True)
    out += float(_log_norm_pdf(star["kappa_0"], *pri["kappa_0"]))
    out += float(_log_norm_pdf(star["delta"], *pri["delta"]))
    out += float(_log_norm_pdf(star["theta_0"], *pri["theta_0"]))
    out += float(_log_norm_pdf(star["gamma"], *pri["gamma"]))
    return float(out)


def _sigma_N_ordinate_terms(star, ds, data, pri) -> np.ndarray | None:
    if "sigma_N2" not in star or "Nhat" not in ds or "Nbar" not in ds or "N" not in data:
        return None
    Nhat_draws = _state_draws(ds, "Nhat")
    Nbar_draws = _state_draws(ds, "Nbar")
    terms = []
    for Nhat, Nbar in zip(Nhat_draws, Nbar_draws):
        n = min(data["N"].size, Nhat.size, Nbar.size)
        resid_N = data["N"][:n] - Nhat[:n] - Nbar[:n]
        terms.append(
            _log_ig_pdf_var(star["sigma_N2"], pri["a_N"] + 0.5 * n, pri["b_N"] + 0.5 * float(np.sum(resid_N**2)))
        )
    return np.array(terms)


def _ces_log_ordinate(star, ds, data, pri, *, orth: bool) -> float:
    lambda_draws = _lambda_ez_draws(ds)
    phi_draws = _draws(ds, "phi_1")
    sigma_zeta2_draws = _draws(ds, "sigma_zeta") ** 2
    sigma_eta2_draws = _sigma_eta2_draws(ds)
    beta_terms = [
        _ces_beta_cond_logpdf(star, data, lmb, phi, sig_eta, pri)
        for lmb, phi, sig_eta in zip(lambda_draws, phi_draws, sigma_eta2_draws)
    ]
    y = data["pi"] - data["pi_expect"]
    a_t = data["pi_prev"] - data["pi_expect"]
    zeta_star = data["x"] - star["phi_1"] * data["x_prev"]
    e_base = y - star["alpha"] * a_t - star["kappa"] * data["x"]
    lambda_term = _lambda_cond_logpdf(star["lambda_ez"], e_base, zeta_star, star["sigma_eta2"], pri, orth=orth)
    phi_terms = [
        _phi_cond_logpdf(
            star["phi_1"],
            x=data["x"],
            x_prev=data["x_prev"],
            y_tilde=y - star["alpha"] * a_t - star["kappa"] * data["x"],
            lambda_ez=star["lambda_ez"],
            sigma_zeta2=sig_zeta,
            sigma_eta2=sig_eta,
            pri=pri,
        )
        for sig_zeta, sig_eta in zip(sigma_zeta2_draws, sigma_eta2_draws)
    ]
    eta_star = y - star["alpha"] * a_t - star["kappa"] * data["x"] - star["lambda_ez"] * zeta_star
    out = _logmeanexp(np.array(beta_terms))
    out += lambda_term
    out += _logmeanexp(np.array(phi_terms))
    out += _log_ig_pdf_var(
        star["sigma_zeta2"], pri["a_z"] + 0.5 * data["x"].size, pri["b_z"] + 0.5 * float(np.sum(zeta_star**2))
    )
    out += _log_ig_pdf_var(
        star["sigma_eta2"], pri["a_e"] + 0.5 * data["x"].size, pri["b_e"] + 0.5 * float(np.sum(eta_star**2))
    )
    return float(out)


def _dynamic_log_ordinate(star, ds, data, pri, *, orth: bool) -> float:
    y = data["pi"] - data["pi_expect"]
    a_t = data["pi_prev"] - data["pi_expect"]
    Nhat_draws = _state_draws(ds, "Nhat")
    Nbar_draws = _state_draws(ds, "Nbar")
    lambda_draws = _lambda_ez_draws(ds)
    phi_draws = _draws(ds, "phi_1")
    sigma_eta2_draws = _sigma_eta2_draws(ds)
    sigma_zeta2_draws = _draws(ds, "sigma_zeta") ** 2
    sigma_u2_draws = _draws(ds, "sigma_u") ** 2
    sigma_eps2_draws = _draws(ds, "sigma_eps") ** 2
    beta_star = np.array([star["alpha"], star["kappa"], star["theta"]], dtype=float)
    prior_mean, prior_sd = _beta_prior_arrays(pri, ["alpha", "kappa", "theta"])
    prior_prec = np.diag(1.0 / prior_sd**2)
    beta_terms = []
    for Nhat, lmb, phi, sig_eta in zip(Nhat_draws, lambda_draws, phi_draws, sigma_eta2_draws):
        zeta = data["x"] - phi * data["x_prev"]
        X = np.column_stack([a_t, data["x"], -Nhat])
        cov = np.linalg.inv(X.T @ X / sig_eta + prior_prec)
        mean = cov @ (X.T @ (y - lmb * zeta) / sig_eta + prior_prec @ prior_mean)
        beta_terms.append(_log_mvn_pdf(beta_star, mean, cov))
    zeta_star = data["x"] - star["phi_1"] * data["x_prev"]
    phi_terms = []
    lambda_terms = []
    sigma_eta_terms = []
    rho_terms = []
    sigma_u_terms = []
    n_terms = []
    sigma_eps_terms = []
    for Nhat, Nbar, sig_z, sig_eta, sig_u, sig_eps in zip(
        Nhat_draws, Nbar_draws, sigma_zeta2_draws, sigma_eta2_draws, sigma_u2_draws, sigma_eps2_draws
    ):
        e_base = y - star["alpha"] * a_t - star["kappa"] * data["x"] + star["theta"] * Nhat
        lambda_terms.append(_lambda_cond_logpdf(star["lambda_ez"], e_base, zeta_star, sig_eta, pri, orth=orth))
        y_tilde = y - star["alpha"] * a_t - star["kappa"] * data["x"] + star["theta"] * Nhat
        phi_terms.append(
            _phi_cond_logpdf(
                star["phi_1"],
                x=data["x"],
                x_prev=data["x_prev"],
                y_tilde=y_tilde,
                lambda_ez=star["lambda_ez"],
                sigma_zeta2=sig_z,
                sigma_eta2=sig_eta,
                pri=pri,
            )
        )
        eta = y - star["alpha"] * a_t - star["kappa"] * data["x"] + star["theta"] * Nhat - star["lambda_ez"] * zeta_star
        sigma_eta_terms.append(
            _log_ig_pdf_var(star["sigma_eta2"], pri["a_e"] + 0.5 * data["x"].size, pri["b_e"] + 0.5 * float(np.sum(eta**2)))
        )
        resid_u = Nhat[2:] - star["rho_1"] * Nhat[1:-1] - star["rho_2"] * Nhat[:-2]
        rho_terms.append(_rho_cond_logpdf(np.array([star["rho_1"], star["rho_2"]]), Nhat, sig_u, pri))
        sigma_u_terms.append(
            _log_ig_pdf_var(star["sigma_u2"], pri["a_u"] + 0.5 * resid_u.size, pri["b_u"] + 0.5 * float(np.sum(resid_u**2)))
        )
        resid_eps = Nbar[1:] - star["n"] - Nbar[:-1]
        n_terms.append(_n_cond_logpdf(star["n"], Nbar, sig_eps, pri))
        sigma_eps_terms.append(
            _log_ig_pdf_var(star["sigma_eps2"], pri["a_eps"] + 0.5 * resid_eps.size, pri["b_eps"] + 0.5 * float(np.sum(resid_eps**2)))
        )
    out = _logmeanexp(np.array(beta_terms))
    out += _logmeanexp(np.array(lambda_terms))
    out += _logmeanexp(np.array(phi_terms))
    out += _logmeanexp(np.array(rho_terms))
    out += _logmeanexp(np.array(n_terms))
    out += _log_ig_pdf_var(
        star["sigma_zeta2"], pri["a_z"] + 0.5 * data["x"].size, pri["b_z"] + 0.5 * float(np.sum(zeta_star**2))
    )
    out += _logmeanexp(np.array(sigma_eta_terms))
    out += _logmeanexp(np.array(sigma_u_terms))
    out += _logmeanexp(np.array(sigma_eps_terms))
    sigma_N_terms = _sigma_N_ordinate_terms(star, ds, data, pri)
    if sigma_N_terms is not None:
        out += _logmeanexp(sigma_N_terms)
    return float(out)


def _steady_log_ordinate(star, ds, data, pri, *, orth: bool) -> float:
    y = data["pi"] - data["pi_expect"]
    a_t = data["pi_prev"] - data["pi_expect"]
    Nhat_draws = _state_draws(ds, "Nhat")
    Nbar_draws = _state_draws(ds, "Nbar")
    lambda_draws = _lambda_ez_draws(ds)
    phi_draws = _draws(ds, "phi_1")
    sigma_eta2_draws = _sigma_eta2_draws(ds)
    sigma_zeta2_draws = _draws(ds, "sigma_zeta") ** 2
    sigma_u2_draws = _draws(ds, "sigma_u") ** 2
    sigma_eps2_draws = _draws(ds, "sigma_eps") ** 2
    beta_star = np.array([star["alpha"], star["kappa_0"], star["delta"]], dtype=float)
    prior_mean, prior_sd = _beta_prior_arrays(pri, ["alpha", "kappa_0", "delta"])
    prior_prec = np.diag(1.0 / prior_sd**2)
    beta_terms = []
    for Nbar, lmb, phi, sig_eta in zip(Nbar_draws, lambda_draws, phi_draws, sigma_eta2_draws):
        zeta = data["x"] - phi * data["x_prev"]
        X = np.column_stack([a_t, data["x"], data["x"] * Nbar])
        cov = np.linalg.inv(X.T @ X / sig_eta + prior_prec)
        mean = cov @ (X.T @ (y - lmb * zeta) / sig_eta + prior_prec @ prior_mean)
        beta_terms.append(_log_mvn_pdf(beta_star, mean, cov))
    zeta_star = data["x"] - star["phi_1"] * data["x_prev"]
    lambda_terms = []
    phi_terms = []
    sigma_eta_terms = []
    rho_terms = []
    sigma_u_terms = []
    n_terms = []
    sigma_eps_terms = []
    for Nhat, Nbar, sig_z, sig_eta, sig_u, sig_eps in zip(
        Nhat_draws, Nbar_draws, sigma_zeta2_draws, sigma_eta2_draws, sigma_u2_draws, sigma_eps2_draws
    ):
        kappa_t = star["kappa_0"] + star["delta"] * Nbar
        e_base = y - star["alpha"] * a_t - kappa_t * data["x"]
        lambda_terms.append(_lambda_cond_logpdf(star["lambda_ez"], e_base, zeta_star, sig_eta, pri, orth=orth))
        y_tilde = y - star["alpha"] * a_t - kappa_t * data["x"]
        phi_terms.append(
            _phi_cond_logpdf(
                star["phi_1"],
                x=data["x"],
                x_prev=data["x_prev"],
                y_tilde=y_tilde,
                lambda_ez=star["lambda_ez"],
                sigma_zeta2=sig_z,
                sigma_eta2=sig_eta,
                pri=pri,
            )
        )
        eta = y - star["alpha"] * a_t - kappa_t * data["x"] - star["lambda_ez"] * zeta_star
        sigma_eta_terms.append(
            _log_ig_pdf_var(star["sigma_eta2"], pri["a_e"] + 0.5 * data["x"].size, pri["b_e"] + 0.5 * float(np.sum(eta**2)))
        )
        resid_u = Nhat[2:] - star["rho_1"] * Nhat[1:-1] - star["rho_2"] * Nhat[:-2]
        rho_terms.append(_rho_cond_logpdf(np.array([star["rho_1"], star["rho_2"]]), Nhat, sig_u, pri))
        sigma_u_terms.append(
            _log_ig_pdf_var(star["sigma_u2"], pri["a_u"] + 0.5 * resid_u.size, pri["b_u"] + 0.5 * float(np.sum(resid_u**2)))
        )
        resid_eps = Nbar[1:] - star["n"] - Nbar[:-1]
        n_terms.append(_n_cond_logpdf(star["n"], Nbar, sig_eps, pri))
        sigma_eps_terms.append(
            _log_ig_pdf_var(star["sigma_eps2"], pri["a_eps"] + 0.5 * resid_eps.size, pri["b_eps"] + 0.5 * float(np.sum(resid_eps**2)))
        )
    out = _logmeanexp(np.array(beta_terms))
    out += _logmeanexp(np.array(lambda_terms))
    out += _logmeanexp(np.array(phi_terms))
    out += _logmeanexp(np.array(rho_terms))
    out += _logmeanexp(np.array(n_terms))
    out += _log_ig_pdf_var(
        star["sigma_zeta2"], pri["a_z"] + 0.5 * data["x"].size, pri["b_z"] + 0.5 * float(np.sum(zeta_star**2))
    )
    out += _logmeanexp(np.array(sigma_eta_terms))
    out += _logmeanexp(np.array(sigma_u_terms))
    out += _logmeanexp(np.array(sigma_eps_terms))
    sigma_N_terms = _sigma_N_ordinate_terms(star, ds, data, pri)
    if sigma_N_terms is not None:
        out += _logmeanexp(sigma_N_terms)
    return float(out)


def chib_marginal_likelihood(
    ds,
    data: dict[str, np.ndarray],
    *,
    family: Family,
    orth: bool,
    priors: dict | None = None,
) -> MarginalLikelihoodResult:
    data = {k: np.asarray(v, dtype=float).reshape(-1) for k, v in data.items()}
    pri = _resolve_priors(priors)
    if family == "ces":
        star = _ces_star(ds)
        log_lik = _ces_log_likelihood(star, data)
        log_prior = _ces_log_prior(star, pri, orth=orth)
        log_ord = _ces_log_ordinate(star, ds, data, pri, orth=orth)
    elif family == "dynamic":
        star = _dynamic_star(ds)
        log_lik = _kalman_loglik_hsa_dynamic(star, data)
        log_prior = _dynamic_log_prior(star, pri, orth=orth)
        log_ord = _dynamic_log_ordinate(star, ds, data, pri, orth=orth)
    elif family == "steady":
        star = _steady_star(ds)
        log_lik = _kalman_loglik_hsa_steady(star, data)
        log_prior = _steady_log_prior(star, pri, orth=orth)
        log_ord = _steady_log_ordinate(star, ds, data, pri, orth=orth)
    elif family == "full":
        raise ValueError(
            "Full HSA has a nonlinear latent-state observation equation. "
            "Use chib_conditional_marginal_likelihood(..., family='full') instead."
        )
    else:
        raise ValueError(f"Unknown family: {family}")
    return MarginalLikelihoodResult(
        log_marginal_likelihood=float(log_lik + log_prior - log_ord),
        log_likelihood=float(log_lik),
        log_prior=float(log_prior),
        log_posterior_ordinate=float(log_ord),
        n_draws=int(_draws(ds, "alpha").size),
        method="Chib 1995, Rao-Blackwellized Gibbs ordinate",
    )


def chib_conditional_marginal_likelihood(
    ds,
    data: dict[str, np.ndarray],
    *,
    family: Family,
    priors: dict | None = None,
) -> MarginalLikelihoodResult:
    data = {k: np.asarray(v, dtype=float).reshape(-1) for k, v in data.items()}
    pri = _resolve_priors(priors)
    if family == "ces":
        star = _ces_conditional_star(ds)
        log_lik = _ces_conditional_log_likelihood(star, data)
        log_prior = _ces_conditional_log_prior(star, pri)
        log_ord = _ces_conditional_log_ordinate(star, ds, data, pri)
    elif family == "dynamic":
        star = _dynamic_conditional_star(ds)
        log_lik = _dynamic_conditional_log_likelihood(star, data)
        log_prior = _dynamic_conditional_log_prior(star, pri)
        log_ord = _dynamic_conditional_log_ordinate(star, ds, data, pri)
    elif family == "steady":
        star = _steady_conditional_star(ds)
        log_lik = _steady_conditional_log_likelihood(star, data)
        log_prior = _steady_conditional_log_prior(star, pri)
        log_ord = _steady_conditional_log_ordinate(star, ds, data, pri)
    elif family == "full":
        star = _full_conditional_star(ds)
        log_lik = _full_conditional_log_likelihood(star, data)
        log_prior = _full_conditional_log_prior(star, pri)
        log_ord = _full_conditional_log_ordinate(star, ds, data, pri)
    else:
        raise ValueError(f"Unknown family: {family}")
    return MarginalLikelihoodResult(
        log_marginal_likelihood=float(log_lik + log_prior - log_ord),
        log_likelihood=float(log_lik),
        log_prior=float(log_prior),
        log_posterior_ordinate=float(log_ord),
        n_draws=int(_draws(ds, "alpha").size),
        method=(
            "Conditional Chib 1995 for HSA full inflation equation, conditioned on posterior mean latent states"
            if family == "full"
            else "Conditional Chib 1995 for NKPC inflation equation"
        ),
    )


def load_posterior_dataset(path: str | Path):
    import xarray as xr

    path = Path(path)
    for engine in ("h5netcdf", "netcdf4", None):
        try:
            return xr.open_dataset(path, group="posterior", engine=engine)
        except Exception:
            pass
    return xr.open_dataset(path, group="posterior")
