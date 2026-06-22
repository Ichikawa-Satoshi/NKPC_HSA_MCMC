from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np


Family = Literal["ces", "dynamic", "steady"]


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
    R = np.array([[1e-6]], dtype=float)
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
    return out


def _sigma_eta2_draws(ds) -> np.ndarray:
    sigma_e = _draws(ds, "sigma_e")
    sigma_zeta = _draws(ds, "sigma_zeta")
    lambda_ez = _draws(ds, "lambda_ez")
    out = sigma_e**2 - (lambda_ez**2) * (sigma_zeta**2)
    return np.maximum(out, 1e-10)


def _ces_star(ds) -> dict[str, float]:
    return _sample_star(ds, ["alpha", "kappa", "phi_1", "lambda_ez", "sigma_e", "sigma_zeta"])


def _dynamic_star(ds) -> dict[str, float]:
    return _sample_star(
        ds,
        ["alpha", "kappa", "theta", "phi_1", "lambda_ez", "rho_1", "rho_2", "n", "sigma_e", "sigma_zeta", "sigma_u", "sigma_eps"],
    )


def _steady_star(ds) -> dict[str, float]:
    return _sample_star(
        ds,
        ["alpha", "kappa_0", "delta", "phi_1", "lambda_ez", "rho_1", "rho_2", "n", "sigma_e", "sigma_zeta", "sigma_u", "sigma_eps"],
    )


def _ces_beta_cond_logpdf(star, data, lambda_ez, phi_1, sigma_eta2) -> float:
    y = data["pi"] - data["pi_expect"]
    a_t = data["pi_prev"] - data["pi_expect"]
    zeta = data["x"] - phi_1 * data["x_prev"]
    X = np.column_stack([a_t, data["x"]])
    prior_mean = np.array([0.5, 0.1], dtype=float)
    prior_prec = np.diag([1.0 / 0.2**2, 1.0 / 0.2**2])
    cov = np.linalg.inv(X.T @ X / sigma_eta2 + prior_prec)
    mean = cov @ (X.T @ (y - lambda_ez * zeta) / sigma_eta2 + prior_prec @ prior_mean)
    return _log_mvn_pdf(np.array([star["alpha"], star["kappa"]]), mean, cov)


def _lambda_cond_logpdf(lambda_star, e_base, zeta, sigma_eta2, *, orth: bool) -> float:
    if orth:
        return 0.0
    prior_prec = 1.0 / 0.5**2
    var = 1.0 / (prior_prec + float(np.sum(zeta**2)) / sigma_eta2)
    mean = var * (float(np.dot(zeta, e_base)) / sigma_eta2)
    return float(_log_norm_pdf(lambda_star, mean, np.sqrt(var)))


def _phi_cond_logpdf(phi_star, *, x, x_prev, y_tilde, lambda_ez, sigma_zeta2, sigma_eta2) -> float:
    prec = (
        1.0 / 0.2**2
        + float(np.sum(x_prev**2)) / sigma_zeta2
        + (lambda_ez**2) * float(np.sum(x_prev**2)) / sigma_eta2
    )
    mean_num = (
        0.7 / 0.2**2
        + float(np.dot(x_prev, x)) / sigma_zeta2
        - lambda_ez * float(np.dot(x_prev, y_tilde - lambda_ez * x)) / sigma_eta2
    )
    return float(_log_norm_pdf(phi_star, mean_num / prec, np.sqrt(1.0 / prec)))


def _rho_cond_logpdf(rho_star: np.ndarray, Nhat: np.ndarray, sigma_u2: float) -> float:
    y = Nhat[2:]
    X = np.column_stack([Nhat[1:-1], Nhat[:-2]])
    prior_mean = np.array([0.2, 0.2], dtype=float)
    prior_prec = np.diag([1.0 / 0.2**2, 1.0 / 0.2**2])
    cov = np.linalg.inv(X.T @ X / sigma_u2 + prior_prec)
    mean = cov @ (X.T @ y / sigma_u2 + prior_prec @ prior_mean)
    return _log_mvn_pdf(rho_star, mean, cov)


def _n_cond_logpdf(n_star: float, Nbar: np.ndarray, sigma_eps2: float) -> float:
    dNbar = Nbar[1:] - Nbar[:-1]
    var = 1.0 / (1.0 / 0.1**2 + dNbar.size / sigma_eps2)
    mean = var * (float(np.sum(dNbar)) / sigma_eps2)
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
        R = np.diag([obs_var_n, sigma_eta2])
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
        R = np.diag([obs_var_n, sigma_eta2])
        S = H @ P @ H.T + R
        v = y - H @ m
        loglik += _log_mvn_pdf(v, np.zeros(2), S)
        K = P @ H.T @ np.linalg.inv(S)
        m = m + K @ v
        P = (np.eye(3) - K @ H) @ P
        P = (P + P.T) / 2.0
    return float(loglik)


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


def _conditional_beta_logpdf(beta_star: np.ndarray, y: np.ndarray, X: np.ndarray, sigma2: float, prior_mean: np.ndarray) -> float:
    prior_prec = np.eye(prior_mean.size) / 0.2**2
    cov = np.linalg.inv(X.T @ X / sigma2 + prior_prec)
    mean = cov @ (X.T @ y / sigma2 + prior_prec @ prior_mean)
    return _log_mvn_pdf(beta_star, mean, cov)


def _ces_conditional_log_ordinate(star, ds, data) -> float:
    sigma_draws = _draws(ds, "sigma_e") ** 2
    y = data["pi"] - data["pi_expect"]
    X = np.column_stack([data["pi_prev"] - data["pi_expect"], data["x"]])
    beta_star = np.array([star["alpha"], star["kappa"]], dtype=float)
    beta_terms = [
        _conditional_beta_logpdf(beta_star, y, X, sigma2, np.array([0.5, 0.1], dtype=float))
        for sigma2 in sigma_draws
    ]
    resid = data["pi"] - star["alpha"] * data["pi_prev"] - (1.0 - star["alpha"]) * data["pi_expect"] - star["kappa"] * data["x"]
    sigma_term = _log_ig_pdf_var(star["sigma_e2"], 2.0 + 0.5 * resid.size, 2.0 + 0.5 * float(np.sum(resid**2)))
    return float(_logmeanexp(np.array(beta_terms)) + sigma_term)


def _dynamic_conditional_log_ordinate(star, ds, data) -> float:
    sigma_draws = _draws(ds, "sigma_e") ** 2
    Nhat_draws = _state_draws(ds, "Nhat")
    y = data["pi"] - data["pi_expect"]
    a_t = data["pi_prev"] - data["pi_expect"]
    beta_star = np.array([star["alpha"], star["kappa"], star["theta"]], dtype=float)
    beta_terms = []
    sigma_terms = []
    for sigma2, Nhat in zip(sigma_draws, Nhat_draws):
        X = np.column_stack([a_t, data["x"], -Nhat])
        beta_terms.append(_conditional_beta_logpdf(beta_star, y, X, sigma2, np.array([0.5, 0.1, 0.1], dtype=float)))
        resid = y - star["alpha"] * a_t - star["kappa"] * data["x"] + star["theta"] * Nhat
        sigma_terms.append(_log_ig_pdf_var(star["sigma_e2"], 2.0 + 0.5 * resid.size, 2.0 + 0.5 * float(np.sum(resid**2))))
    return float(_logmeanexp(np.array(beta_terms)) + _logmeanexp(np.array(sigma_terms)))


def _steady_conditional_log_ordinate(star, ds, data) -> float:
    sigma_draws = _draws(ds, "sigma_e") ** 2
    Nbar_draws = _state_draws(ds, "Nbar")
    y = data["pi"] - data["pi_expect"]
    a_t = data["pi_prev"] - data["pi_expect"]
    beta_star = np.array([star["alpha"], star["kappa_0"], star["delta"]], dtype=float)
    beta_terms = []
    sigma_terms = []
    for sigma2, Nbar in zip(sigma_draws, Nbar_draws):
        X = np.column_stack([a_t, data["x"], data["x"] * Nbar])
        beta_terms.append(_conditional_beta_logpdf(beta_star, y, X, sigma2, np.array([0.5, 0.1, 0.1], dtype=float)))
        resid = y - star["alpha"] * a_t - (star["kappa_0"] + star["delta"] * Nbar) * data["x"]
        sigma_terms.append(_log_ig_pdf_var(star["sigma_e2"], 2.0 + 0.5 * resid.size, 2.0 + 0.5 * float(np.sum(resid**2))))
    return float(_logmeanexp(np.array(beta_terms)) + _logmeanexp(np.array(sigma_terms)))


def _ces_conditional_log_prior(star) -> float:
    return float(
        _log_norm_pdf(star["alpha"], 0.5, 0.2)
        + _log_norm_pdf(star["kappa"], 0.1, 0.2)
        + _log_ig_pdf_var(star["sigma_e2"], 2.0, 2.0)
    )


def _dynamic_conditional_log_prior(star) -> float:
    return float(
        _log_norm_pdf(star["alpha"], 0.5, 0.2)
        + _log_norm_pdf(star["kappa"], 0.1, 0.2)
        + _log_norm_pdf(star["theta"], 0.1, 0.2)
        + _log_ig_pdf_var(star["sigma_e2"], 2.0, 2.0)
    )


def _steady_conditional_log_prior(star) -> float:
    return float(
        _log_norm_pdf(star["alpha"], 0.5, 0.2)
        + _log_norm_pdf(star["kappa_0"], 0.1, 0.2)
        + _log_norm_pdf(star["delta"], 0.1, 0.2)
        + _log_ig_pdf_var(star["sigma_e2"], 2.0, 2.0)
    )


def _log_prior_common(star, *, orth: bool, hsa: bool) -> float:
    out = 0.0
    out += float(_log_norm_pdf(star["alpha"], 0.5, 0.2))
    out += float(_log_norm_pdf(star["phi_1"], 0.7, 0.2))
    if not orth:
        out += float(_log_norm_pdf(star["lambda_ez"], 0.0, 0.5))
    out += _log_ig_pdf_var(star["sigma_eta2"], 2.0, 2.0)
    out += _log_ig_pdf_var(star["sigma_zeta2"], 0.001, 0.001)
    if hsa:
        out += float(_log_norm_pdf(star["rho_1"], 0.2, 0.2))
        out += float(_log_norm_pdf(star["rho_2"], 0.2, 0.2))
        out += float(_log_norm_pdf(star["n"], 0.0, 0.1))
        out += _log_ig_pdf_var(star["sigma_u2"], 2.0, 2.0)
        out += _log_ig_pdf_var(star["sigma_eps2"], 2.0, 2.0)
    return float(out)


def _ces_log_prior(star, *, orth: bool) -> float:
    return float(_log_prior_common(star, orth=orth, hsa=False) + _log_norm_pdf(star["kappa"], 0.1, 0.2))


def _dynamic_log_prior(star, *, orth: bool) -> float:
    out = _log_prior_common(star, orth=orth, hsa=True)
    out += float(_log_norm_pdf(star["kappa"], 0.1, 0.2))
    out += float(_log_norm_pdf(star["theta"], 0.1, 0.2))
    return float(out)


def _steady_log_prior(star, *, orth: bool) -> float:
    out = _log_prior_common(star, orth=orth, hsa=True)
    out += float(_log_norm_pdf(star["kappa_0"], 0.1, 0.2))
    out += float(_log_norm_pdf(star["delta"], 0.1, 0.2))
    return float(out)


def _ces_log_ordinate(star, ds, data, *, orth: bool) -> float:
    lambda_draws = _draws(ds, "lambda_ez")
    phi_draws = _draws(ds, "phi_1")
    sigma_zeta2_draws = _draws(ds, "sigma_zeta") ** 2
    sigma_eta2_draws = _sigma_eta2_draws(ds)
    beta_terms = [
        _ces_beta_cond_logpdf(star, data, lmb, phi, sig_eta)
        for lmb, phi, sig_eta in zip(lambda_draws, phi_draws, sigma_eta2_draws)
    ]
    y = data["pi"] - data["pi_expect"]
    a_t = data["pi_prev"] - data["pi_expect"]
    zeta_star = data["x"] - star["phi_1"] * data["x_prev"]
    e_base = y - star["alpha"] * a_t - star["kappa"] * data["x"]
    lambda_term = _lambda_cond_logpdf(star["lambda_ez"], e_base, zeta_star, star["sigma_eta2"], orth=orth)
    phi_terms = [
        _phi_cond_logpdf(
            star["phi_1"],
            x=data["x"],
            x_prev=data["x_prev"],
            y_tilde=y - star["alpha"] * a_t - star["kappa"] * data["x"],
            lambda_ez=star["lambda_ez"],
            sigma_zeta2=sig_zeta,
            sigma_eta2=sig_eta,
        )
        for sig_zeta, sig_eta in zip(sigma_zeta2_draws, sigma_eta2_draws)
    ]
    eta_star = y - star["alpha"] * a_t - star["kappa"] * data["x"] - star["lambda_ez"] * zeta_star
    out = _logmeanexp(np.array(beta_terms))
    out += lambda_term
    out += _logmeanexp(np.array(phi_terms))
    out += _log_ig_pdf_var(star["sigma_zeta2"], 0.001 + 0.5 * data["x"].size, 0.001 + 0.5 * float(np.sum(zeta_star**2)))
    out += _log_ig_pdf_var(star["sigma_eta2"], 2.0 + 0.5 * data["x"].size, 2.0 + 0.5 * float(np.sum(eta_star**2)))
    return float(out)


def _dynamic_log_ordinate(star, ds, data, *, orth: bool) -> float:
    y = data["pi"] - data["pi_expect"]
    a_t = data["pi_prev"] - data["pi_expect"]
    Nhat_draws = _state_draws(ds, "Nhat")
    Nbar_draws = _state_draws(ds, "Nbar")
    lambda_draws = _draws(ds, "lambda_ez")
    phi_draws = _draws(ds, "phi_1")
    sigma_eta2_draws = _sigma_eta2_draws(ds)
    sigma_zeta2_draws = _draws(ds, "sigma_zeta") ** 2
    sigma_u2_draws = _draws(ds, "sigma_u") ** 2
    sigma_eps2_draws = _draws(ds, "sigma_eps") ** 2
    beta_star = np.array([star["alpha"], star["kappa"], star["theta"]], dtype=float)
    beta_terms = []
    for Nhat, lmb, phi, sig_eta in zip(Nhat_draws, lambda_draws, phi_draws, sigma_eta2_draws):
        zeta = data["x"] - phi * data["x_prev"]
        X = np.column_stack([a_t, data["x"], -Nhat])
        prior_prec = np.diag([1 / 0.2**2, 1 / 0.2**2, 1 / 0.2**2])
        prior_mean = np.array([0.5, 0.1, 0.1], dtype=float)
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
        lambda_terms.append(_lambda_cond_logpdf(star["lambda_ez"], e_base, zeta_star, sig_eta, orth=orth))
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
            )
        )
        eta = y - star["alpha"] * a_t - star["kappa"] * data["x"] + star["theta"] * Nhat - star["lambda_ez"] * zeta_star
        sigma_eta_terms.append(_log_ig_pdf_var(star["sigma_eta2"], 2.0 + 0.5 * data["x"].size, 2.0 + 0.5 * float(np.sum(eta**2))))
        resid_u = Nhat[2:] - star["rho_1"] * Nhat[1:-1] - star["rho_2"] * Nhat[:-2]
        rho_terms.append(_rho_cond_logpdf(np.array([star["rho_1"], star["rho_2"]]), Nhat, sig_u))
        sigma_u_terms.append(_log_ig_pdf_var(star["sigma_u2"], 2.0 + 0.5 * resid_u.size, 2.0 + 0.5 * float(np.sum(resid_u**2))))
        resid_eps = Nbar[1:] - star["n"] - Nbar[:-1]
        n_terms.append(_n_cond_logpdf(star["n"], Nbar, sig_eps))
        sigma_eps_terms.append(_log_ig_pdf_var(star["sigma_eps2"], 2.0 + 0.5 * resid_eps.size, 2.0 + 0.5 * float(np.sum(resid_eps**2))))
    out = _logmeanexp(np.array(beta_terms))
    out += _logmeanexp(np.array(lambda_terms))
    out += _logmeanexp(np.array(phi_terms))
    out += _logmeanexp(np.array(rho_terms))
    out += _logmeanexp(np.array(n_terms))
    out += _log_ig_pdf_var(star["sigma_zeta2"], 0.001 + 0.5 * data["x"].size, 0.001 + 0.5 * float(np.sum(zeta_star**2)))
    out += _logmeanexp(np.array(sigma_eta_terms))
    out += _logmeanexp(np.array(sigma_u_terms))
    out += _logmeanexp(np.array(sigma_eps_terms))
    return float(out)


def _steady_log_ordinate(star, ds, data, *, orth: bool) -> float:
    y = data["pi"] - data["pi_expect"]
    a_t = data["pi_prev"] - data["pi_expect"]
    Nhat_draws = _state_draws(ds, "Nhat")
    Nbar_draws = _state_draws(ds, "Nbar")
    lambda_draws = _draws(ds, "lambda_ez")
    phi_draws = _draws(ds, "phi_1")
    sigma_eta2_draws = _sigma_eta2_draws(ds)
    sigma_zeta2_draws = _draws(ds, "sigma_zeta") ** 2
    sigma_u2_draws = _draws(ds, "sigma_u") ** 2
    sigma_eps2_draws = _draws(ds, "sigma_eps") ** 2
    beta_star = np.array([star["alpha"], star["kappa_0"], star["delta"]], dtype=float)
    beta_terms = []
    for Nbar, lmb, phi, sig_eta in zip(Nbar_draws, lambda_draws, phi_draws, sigma_eta2_draws):
        zeta = data["x"] - phi * data["x_prev"]
        X = np.column_stack([a_t, data["x"], data["x"] * Nbar])
        prior_prec = np.diag([1 / 0.2**2, 1 / 0.2**2, 1 / 0.2**2])
        prior_mean = np.array([0.5, 0.1, 0.1], dtype=float)
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
        lambda_terms.append(_lambda_cond_logpdf(star["lambda_ez"], e_base, zeta_star, sig_eta, orth=orth))
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
            )
        )
        eta = y - star["alpha"] * a_t - kappa_t * data["x"] - star["lambda_ez"] * zeta_star
        sigma_eta_terms.append(_log_ig_pdf_var(star["sigma_eta2"], 2.0 + 0.5 * data["x"].size, 2.0 + 0.5 * float(np.sum(eta**2))))
        resid_u = Nhat[2:] - star["rho_1"] * Nhat[1:-1] - star["rho_2"] * Nhat[:-2]
        rho_terms.append(_rho_cond_logpdf(np.array([star["rho_1"], star["rho_2"]]), Nhat, sig_u))
        sigma_u_terms.append(_log_ig_pdf_var(star["sigma_u2"], 2.0 + 0.5 * resid_u.size, 2.0 + 0.5 * float(np.sum(resid_u**2))))
        resid_eps = Nbar[1:] - star["n"] - Nbar[:-1]
        n_terms.append(_n_cond_logpdf(star["n"], Nbar, sig_eps))
        sigma_eps_terms.append(_log_ig_pdf_var(star["sigma_eps2"], 2.0 + 0.5 * resid_eps.size, 2.0 + 0.5 * float(np.sum(resid_eps**2))))
    out = _logmeanexp(np.array(beta_terms))
    out += _logmeanexp(np.array(lambda_terms))
    out += _logmeanexp(np.array(phi_terms))
    out += _logmeanexp(np.array(rho_terms))
    out += _logmeanexp(np.array(n_terms))
    out += _log_ig_pdf_var(star["sigma_zeta2"], 0.001 + 0.5 * data["x"].size, 0.001 + 0.5 * float(np.sum(zeta_star**2)))
    out += _logmeanexp(np.array(sigma_eta_terms))
    out += _logmeanexp(np.array(sigma_u_terms))
    out += _logmeanexp(np.array(sigma_eps_terms))
    return float(out)


def chib_marginal_likelihood(ds, data: dict[str, np.ndarray], *, family: Family, orth: bool) -> MarginalLikelihoodResult:
    data = {k: np.asarray(v, dtype=float).reshape(-1) for k, v in data.items()}
    if family == "ces":
        star = _ces_star(ds)
        log_lik = _ces_log_likelihood(star, data)
        log_prior = _ces_log_prior(star, orth=orth)
        log_ord = _ces_log_ordinate(star, ds, data, orth=orth)
    elif family == "dynamic":
        star = _dynamic_star(ds)
        log_lik = _kalman_loglik_hsa_dynamic(star, data)
        log_prior = _dynamic_log_prior(star, orth=orth)
        log_ord = _dynamic_log_ordinate(star, ds, data, orth=orth)
    elif family == "steady":
        star = _steady_star(ds)
        log_lik = _kalman_loglik_hsa_steady(star, data)
        log_prior = _steady_log_prior(star, orth=orth)
        log_ord = _steady_log_ordinate(star, ds, data, orth=orth)
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


def chib_conditional_marginal_likelihood(ds, data: dict[str, np.ndarray], *, family: Family) -> MarginalLikelihoodResult:
    data = {k: np.asarray(v, dtype=float).reshape(-1) for k, v in data.items()}
    if family == "ces":
        star = _ces_conditional_star(ds)
        log_lik = _ces_conditional_log_likelihood(star, data)
        log_prior = _ces_conditional_log_prior(star)
        log_ord = _ces_conditional_log_ordinate(star, ds, data)
    elif family == "dynamic":
        star = _dynamic_conditional_star(ds)
        log_lik = _dynamic_conditional_log_likelihood(star, data)
        log_prior = _dynamic_conditional_log_prior(star)
        log_ord = _dynamic_conditional_log_ordinate(star, ds, data)
    elif family == "steady":
        star = _steady_conditional_star(ds)
        log_lik = _steady_conditional_log_likelihood(star, data)
        log_prior = _steady_conditional_log_prior(star)
        log_ord = _steady_conditional_log_ordinate(star, ds, data)
    else:
        raise ValueError(f"Unknown family: {family}")
    return MarginalLikelihoodResult(
        log_marginal_likelihood=float(log_lik + log_prior - log_ord),
        log_likelihood=float(log_lik),
        log_prior=float(log_prior),
        log_posterior_ordinate=float(log_ord),
        n_draws=int(_draws(ds, "alpha").size),
        method="Conditional Chib 1995 for NKPC inflation equation",
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
