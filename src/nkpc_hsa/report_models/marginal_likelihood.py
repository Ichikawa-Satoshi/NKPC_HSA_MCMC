"""Marginal likelihood for the four-case, five-model design.

The integrated (state-marginalised) likelihood p(y | theta) is:
  * exact, via the Kalman filter, for the linear-Gaussian Models 0-2;
  * estimated by a bootstrap particle filter for the bilinear Models 3-4.

The marginal likelihood m(y) = integral p(y|theta) p(theta) dtheta is estimated
by the Laplace--Metropolis estimator (Raftery 1996) evaluated in an
unconstrained parameter space (log for variances, atanh for rho):

  log m(y) ~= 0.5 d log(2 pi) + 0.5 log|Sigma_t|
             + log p(y|theta*) + log p(theta*) + log|J(theta*)|,

with theta* the posterior mean and Sigma_t the posterior covariance of the
transformed draws. This complements the WAIC reported alongside it.
"""

from __future__ import annotations

import numpy as np
from scipy.special import gammaln
from scipy.stats import norm

from nkpc_hsa.gibbs.common.joint_ffbs import force_pd
from nkpc_hsa.report_models.engine import (
    ANCHOR_REL_VAR, Priors, ar2_stationary_cov, build_priors, coeff_names, is_stationary_ar2,
)
from nkpc_hsa.report_models.cases import CaseData


# ---- parameter vector layout ------------------------------------------------
def param_names(model: int, case: int, hybrid: bool = False, ar_order: int = 1) -> list[str]:
    names = list(coeff_names(model, hybrid))
    names += ["rho"]
    if ar_order == 2:
        names += ["rho2"]
    names += ["sigma_pi2", "sigma_bar2", "sigma_hat2"]
    if case == 4:
        names.append("lambda_E")
    else:
        names.append("sigma_nu2")
    return names


_VAR = {"sigma_pi2", "sigma_bar2", "sigma_hat2", "sigma_nu2"}


def _transform(name: str, value: np.ndarray, ar2: bool = False) -> np.ndarray:
    if name in _VAR:
        return np.log(value)
    # AR(1) rho is mapped to the real line via atanh; the AR(2) pair (rho, rho2)
    # is left untransformed (its support is the stationarity triangle, not a box).
    if name == "rho" and not ar2:
        return np.arctanh(np.clip(value, -0.999999, 0.999999))
    return value


def _inverse(name: str, t: float, ar2: bool = False) -> float:
    if name in _VAR:
        return float(np.exp(t))
    if name == "rho" and not ar2:
        return float(np.tanh(t))
    return float(t)


def _log_jac(name: str, value: float, ar2: bool = False) -> float:
    if name in _VAR:            # value = exp(t) -> d value/dt = value
        return float(np.log(value))
    if name == "rho" and not ar2:   # value = tanh(t) -> 1 - value^2
        return float(np.log(1.0 - value ** 2))
    return 0.0


# ---- integrated likelihood --------------------------------------------------
def _obs_rows(t: int, model: int, theta: dict, data: CaseData, var_nu: float, ar2: bool = False):
    """Return (values, H, R) for the stacked observations at time t (pre state-mean removal)."""
    pi, epi, x, n_obs = data.pi, data.epi, data.x, data.n_obs
    base = pi[t] - theta["alpha"] * epi[t] - theta["kappa_0"] * x[t]
    if "intercept" in theta:
        base -= theta["intercept"] + theta.get("alpha_b", 0.0) * data.pi_lag[t]
    lag_pad = [0.0] if ar2 else []
    h_pi = [0.0, 0.0]
    if model == 1:
        h_pi = [theta.get("delta", 0.0) * x[t], 0.0]
    elif model == 2:
        h_pi = [0.0, -theta.get("theta_0", 0.0)]   # inflation loads -theta_0 * Nhat
    vals, rows, rvar = [base], [h_pi + lag_pad], [theta["sigma_pi2"]]
    if np.isfinite(n_obs[t]):
        vals.append(n_obs[t]); rows.append([1.0, 1.0] + lag_pad); rvar.append(var_nu)
    return np.asarray(vals), np.asarray(rows), np.diag(rvar)


def kalman_loglik(theta: dict, data: CaseData, model: int) -> float:
    """Exact integrated log-likelihood for the linear-Gaussian Models 0-2.

    Supports an AR(1) fast state [Ntilde, Nhat] or, when ``theta`` carries a
    ``rho2`` entry, an AR(2) fast state in companion form
    [Ntilde, Nhat, Nhat_lag]. The filter never inverts the (possibly singular
    companion) state covariance -- only the innovation covariance S, which always
    carries measurement noise -- so the integrated likelihood stays exact.
    """
    T = data.n_periods
    rho = theta["rho"]
    var_hat = theta["sigma_hat2"]
    var_bar = theta["sigma_bar2"]
    var_nu = ANCHOR_REL_VAR * data.s_N ** 2 if data.case == 4 else theta["sigma_nu2"]
    gE = data.gE if data.case == 4 else np.zeros(T)
    lam = theta.get("lambda_E", 0.0)
    ar2 = "rho2" in theta
    if ar2:
        rho2 = theta["rho2"]
        dim = 3
        F = np.array([[1.0, 0.0, 0.0], [0.0, rho, rho2], [0.0, 1.0, 0.0]])
        Q = np.diag([var_bar, var_hat, 0.0])
        P = np.zeros((3, 3))
        P[0, 0] = (2.0 * data.s_N) ** 2
        P[1:, 1:] = ar2_stationary_cov(rho, rho2, var_hat)
        c_hat_idx = 1
    else:
        dim = 2
        F = np.array([[1.0, 0.0], [0.0, rho]])
        Q = np.diag([var_bar, var_hat])
        P = np.diag([(2.0 * data.s_N) ** 2, var_hat / max(1e-6, 1 - rho ** 2)])
        c_hat_idx = 1
    m = np.zeros(dim)
    eye = np.eye(dim)
    ll = 0.0
    for t in range(T):
        if t > 0:
            c = np.zeros(dim); c[c_hat_idx] = lam * gE[t]
            m = c + F @ m
            P = force_pd(F @ P @ F.T + Q)
        values, H, R = _obs_rows(t, model, theta, data, var_nu, ar2)
        # Remove the known state-independent part already folded into `values`.
        innov = values - H @ m
        S = force_pd(H @ P @ H.T + R)
        sign, logdet = np.linalg.slogdet(S)
        sol = np.linalg.solve(S, innov)
        ll += -0.5 * (len(values) * np.log(2 * np.pi) + logdet + innov @ sol)
        K = np.linalg.solve(S, H @ P).T
        m = m + K @ innov
        P = force_pd((eye - K @ H) @ P @ (eye - K @ H).T + K @ R @ K.T)
    return float(ll)


def particle_loglik(theta: dict, data: CaseData, model: int, rng: np.random.Generator,
                    n_particles: int = 4000) -> float:
    """Bootstrap particle-filter integrated log-likelihood for the bilinear Models 3-4."""
    T = data.n_periods
    pi, epi, x, n_obs = data.pi, data.epi, data.x, data.n_obs
    rho = theta["rho"]
    var_hat = theta["sigma_hat2"]
    var_bar = theta["sigma_bar2"]
    var_pi = theta["sigma_pi2"]
    var_nu = ANCHOR_REL_VAR * data.s_N ** 2 if data.case == 4 else theta["sigma_nu2"]
    gE = data.gE if data.case == 4 else np.zeros(T)
    lam = theta.get("lambda_E", 0.0)
    th0 = theta.get("theta_0", 0.0)
    de = theta.get("delta", 0.0)
    ga = theta.get("gamma", 0.0)
    ar2 = "rho2" in theta
    rho2 = theta.get("rho2", 0.0)
    N = n_particles
    ntil = rng.normal(0.0, 2.0 * data.s_N, N)
    if ar2:
        cov = ar2_stationary_cov(rho, rho2, var_hat)
        L = np.linalg.cholesky(force_pd(cov))
        z = rng.normal(size=(N, 2)) @ L.T
        nhat, nhat_lag = z[:, 0], z[:, 1]
    else:
        nhat = rng.normal(0.0, np.sqrt(var_hat / max(1e-6, 1 - rho ** 2)), N)
    ll = 0.0
    for t in range(T):
        if t > 0:
            ntil = ntil + rng.normal(0.0, np.sqrt(var_bar), N)
            if ar2:
                nhat_new = rho * nhat + rho2 * nhat_lag + lam * gE[t] + rng.normal(0.0, np.sqrt(var_hat), N)
                nhat_lag = nhat
                nhat = nhat_new
            else:
                nhat = rho * nhat + lam * gE[t] + rng.normal(0.0, np.sqrt(var_hat), N)
        mu = (theta["alpha"] * epi[t] + theta["kappa_0"] * x[t]
              + de * x[t] * ntil - th0 * nhat + ga * ntil * nhat)
        if "intercept" in theta:
            mu = mu + theta["intercept"] + theta.get("alpha_b", 0.0) * data.pi_lag[t]
        logw = -0.5 * (np.log(2 * np.pi * var_pi) + (pi[t] - mu) ** 2 / var_pi)
        if np.isfinite(n_obs[t]):
            logw += -0.5 * (np.log(2 * np.pi * var_nu) + (n_obs[t] - (ntil + nhat)) ** 2 / var_nu)
        mx = logw.max()
        w = np.exp(logw - mx)
        sw = w.sum()
        ll += mx + np.log(sw / N)
        w /= sw
        idx = rng.choice(N, size=N, p=w)
        ntil, nhat = ntil[idx], nhat[idx]
        if ar2:
            nhat_lag = nhat_lag[idx]
    return float(ll)


def integrated_loglik(theta: dict, data: CaseData, model: int,
                      rng: np.random.Generator | None = None) -> float:
    if model in (0, 1, 2):
        return kalman_loglik(theta, data, model)
    rng = rng or np.random.default_rng(0)
    return particle_loglik(theta, data, model, rng)


# ---- prior density ----------------------------------------------------------
def _ig_logpdf(x: float, shape: float, scale: float) -> float:
    return shape * np.log(scale) - gammaln(shape) - (shape + 1) * np.log(x) - scale / x


_AR2_NORM_CACHE: dict[tuple[float, float], float] = {}


def _ar2_log_trunc_norm(rho_mean: float, rho_sd: float) -> float:
    """log P((rho1, rho2) in AR(2) stationarity triangle) under the independent
    priors rho1 ~ N(rho_mean, rho_sd), rho2 ~ N(0, rho_sd). Deterministic MC so
    the AR(2) prior integrates to one over its stationary support -- this is the
    Occam term that keeps the AR(1) vs AR(2) marginal-likelihood comparison fair.
    """
    key = (round(rho_mean, 6), round(rho_sd, 6))
    if key not in _AR2_NORM_CACHE:
        g = np.random.default_rng(12345)
        n = 2_000_000
        r1 = g.normal(rho_mean, rho_sd, n)
        r2 = g.normal(0.0, rho_sd, n)
        inside = (r2 > -1.0) & (r1 + r2 < 1.0) & (r2 - r1 < 1.0)
        frac = max(inside.mean(), 1e-9)
        _AR2_NORM_CACHE[key] = float(np.log(frac))
    return _AR2_NORM_CACHE[key]


def log_prior(theta: dict, data: CaseData, model: int, priors: Priors) -> float:
    lp = norm.logpdf(theta["alpha"], priors.alpha_mean, priors.alpha_sd)
    lp += norm.logpdf(theta["kappa_0"], 0.0, priors.kappa0_sd)
    if "intercept" in theta:
        lp += norm.logpdf(theta["intercept"], 0.0, priors.intercept_sd)
        lp += norm.logpdf(theta["alpha_b"], priors.alpha_b_mean, priors.alpha_b_sd)
    if "delta" in theta:
        lp += norm.logpdf(theta["delta"], 0.0, priors.delta_sd)
    if "theta_0" in theta:
        lp += norm.logpdf(theta["theta_0"], 0.0, priors.theta0_sd)
    if "gamma" in theta:
        lp += norm.logpdf(theta["gamma"], 0.0, priors.gamma_sd)
    if "rho2" in theta:
        # AR(2): rho1 ~ N(rho_mean, rho_sd), rho2 ~ N(0, rho_sd), jointly
        # truncated to the stationarity triangle.
        if not is_stationary_ar2(theta["rho"], theta["rho2"]):
            return float("-inf")
        lp += norm.logpdf(theta["rho"], priors.rho_mean, priors.rho_sd)
        lp += norm.logpdf(theta["rho2"], 0.0, priors.rho_sd)
        lp -= _ar2_log_trunc_norm(priors.rho_mean, priors.rho_sd)
    else:
        # rho ~ TN(0.5, 0.25) on (-1, 1)
        z = norm.cdf((1 - priors.rho_mean) / priors.rho_sd) - norm.cdf((-1 - priors.rho_mean) / priors.rho_sd)
        lp += norm.logpdf(theta["rho"], priors.rho_mean, priors.rho_sd) - np.log(z)
    lp += _ig_logpdf(theta["sigma_pi2"], priors.ig_shape, priors.sigma_pi_b)
    lp += _ig_logpdf(theta["sigma_bar2"], priors.ig_shape, priors.sigma_bar_b)
    lp += _ig_logpdf(theta["sigma_hat2"], priors.ig_shape, priors.sigma_hat_b)
    if data.case == 4:
        lp += norm.logpdf(theta["lambda_E"], 0.0, priors.lambda_sd)
    else:
        lp += _ig_logpdf(theta["sigma_nu2"], priors.ig_shape, priors.sigma_nu_b)
    return float(lp)


# ---- Laplace-Metropolis marginal likelihood ---------------------------------
def _draws_matrix(draws: dict, names: list[str], model: int, case: int) -> np.ndarray:
    coeff_names = list(draws["coeff_names"])
    cols = {}
    for c in coeff_names:
        cols[c] = draws["coeffs"][:, :, coeff_names.index(c)].reshape(-1)
    cols["rho"] = draws["rho"].reshape(-1)
    if "rho2" in names:
        cols["rho2"] = draws["rho2"].reshape(-1)
    cols["sigma_pi2"] = draws["sigma_pi"].reshape(-1) ** 2
    cols["sigma_bar2"] = draws["sigma_bar"].reshape(-1) ** 2
    cols["sigma_hat2"] = draws["sigma_hat"].reshape(-1) ** 2
    if case == 4:
        cols["lambda_E"] = draws["lambda_E"].reshape(-1)
    else:
        cols["sigma_nu2"] = draws["sigma_nu"].reshape(-1) ** 2
    return np.column_stack([cols[n] for n in names])


def _unnorm_log_post(theta: dict, data: CaseData, model: int, priors: Priors,
                     rng: np.random.Generator, ar_order: int = 1) -> float:
    ar2 = ar_order == 2
    ll = integrated_loglik(theta, data, model, rng)
    lp = log_prior(theta, data, model, priors)
    jac = sum(_log_jac(n, theta[n], ar2) for n in param_names(model, data.case, "intercept" in theta, ar_order))
    return ll + lp + jac, ll, lp, jac


def laplace_metropolis_logml(draws: dict, data: CaseData, model: int,
                             priors: Priors | None = None, seed: int = 0,
                             n_mode_search: int = 40) -> float:
    """Laplace--Metropolis estimator with a robust high-density evaluation point.

    theta* is the draw (from an evenly-spaced subsample) with the highest
    unnormalised posterior ordinate, which avoids the posterior-mean landing in a
    low-density valley when the chains are imperfectly mixed. The covariance term
    uses the full transformed-draw covariance (ridge-regularised).

    Detects an AR(2) fast state from a non-empty ``rho2`` entry in ``draws``.
    """
    hybrid = "intercept" in list(draws["coeff_names"])
    ar_order = 2 if ("rho2" in draws and np.asarray(draws["rho2"]).size > 0) else 1
    ar2 = ar_order == 2
    priors = priors or build_priors(data, hybrid=hybrid)
    names = param_names(model, data.case, hybrid, ar_order)
    raw = _draws_matrix(draws, names, model, data.case)
    trans = np.column_stack([_transform(n, raw[:, j], ar2) for j, n in enumerate(names)])
    d = len(names)
    Sigma = np.atleast_2d(np.cov(trans, rowvar=False))
    Sigma = Sigma + 1e-8 * np.trace(Sigma) / d * np.eye(d)  # ridge for stability

    rng = np.random.default_rng(seed + 5)
    n = raw.shape[0]
    take = np.linspace(0, n - 1, min(n_mode_search, n)).astype(int)
    best_lp, best_theta = -np.inf, None
    for i in take:
        theta = {nm: float(raw[i, j]) for j, nm in enumerate(names)}
        lp_total, *_ = _unnorm_log_post(theta, data, model, priors, np.random.default_rng(seed + int(i)), ar_order)
        if lp_total > best_lp:
            best_lp, best_theta = lp_total, theta
    sign, logdet = np.linalg.slogdet(force_pd(Sigma))
    return float(0.5 * d * np.log(2 * np.pi) + 0.5 * logdet + best_lp)
