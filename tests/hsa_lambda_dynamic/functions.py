"""Data construction and MCMC for the estimated-lambda HSA experiment.

The sampler is a compact extension of ``nkpc_hsa.report_models.engine``.  It
reuses the same scale-derived priors and state variance updates, while adding:

* a genuine AR(1) inflation disturbance;
* quadratic kappa(N);
* an estimated, sign-unrestricted HSA multiplier lambda;
* elliptical-slice state draws, needed by the nonlinear dynamic specifications.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import arviz as az
import numpy as np
import pandas as pd
from scipy.special import gammaln
from scipy.stats import norm, truncnorm

from nkpc_hsa.gibbs.common.joint_ffbs import force_pd
from nkpc_hsa.phillips.state import _draw_ig
from nkpc_hsa.report_models.cases import CaseData, GUSTAVO_ANNUAL_COL, _load_frame
from nkpc_hsa.report_models.engine import Priors, build_priors


MODEL_LABELS = {
    "ces": "CES baseline",
    "slope": "Slope channel",
    "direct": "Direct channel",
    "free_static": "Free static combined",
    "hsa_static": "HSA-restricted static",
    "free_dynamic": "Free dynamic combined",
    "hsa_dynamic": "HSA-restricted dynamic",
}

BASE_NAMES = ("intercept", "alpha_b", "alpha_f", "kappa_0")
COEFF_NAMES = {
    "ces": BASE_NAMES,
    "slope": BASE_NAMES + ("delta",),
    "direct": BASE_NAMES + ("theta_0",),
    "free_static": BASE_NAMES + ("delta", "theta_0"),
    "hsa_static": BASE_NAMES + ("theta_0",),
    "free_dynamic": BASE_NAMES + ("delta_1", "delta_2", "theta_0", "gamma"),
    "hsa_dynamic": BASE_NAMES + ("theta_0", "gamma"),
}
HSA_MODELS = {"hsa_static", "hsa_dynamic"}
DYNAMIC_MODELS = {"free_dynamic", "hsa_dynamic"}


@dataclass(frozen=True)
class AllocationResult:
    quarterly: pd.Series
    annual: pd.Series
    average_weights: np.ndarray
    raw_weights: dict[int, np.ndarray]
    used_weights: dict[int, np.ndarray]
    coherence: dict[int, float]
    source: dict[int, str]
    max_anchor_error: float


@dataclass(frozen=True)
class ExperimentData:
    case: CaseData
    frame: pd.DataFrame
    allocation: AllocationResult


@dataclass
class FitResult:
    model: str
    label: str
    names: tuple[str, ...]
    draws: np.ndarray
    sigma_pi: np.ndarray
    phi: np.ndarray
    rho: np.ndarray
    sigma_bar: np.ndarray
    sigma_hat: np.ndarray
    sigma_nu: np.ndarray
    nbar: np.ndarray
    nhat: np.ndarray
    periods: tuple[str, ...]
    prior_mean: dict[str, float]
    prior_sd: dict[str, float]
    diagnostics: dict


def robust_scale(values) -> float:
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    out = float(np.subtract(*np.quantile(v, [0.75, 0.25])) / 1.349)
    return out if np.isfinite(out) and out > 0 else 1.0


def build_quarterly_competition(
    frame: pd.DataFrame,
    competition_col: str,
    stable_raw_weight_max: float = 3.0,
) -> AllocationResult:
    """Allocate Gustavo annual changes using guarded Capital IQ quarterly shares.

    The robust fallback is the componentwise median of stable annual Capital IQ
    ratios, renormalized to sum to one.  In observed years the raw share receives
    weight c=|sum(dq)|/sum(|dq|), so cancellation-driven explosions are shrunk
    continuously rather than handled by an arbitrary hard replacement.
    """
    num = lambda c: pd.to_numeric(frame[c], errors="coerce")
    annual_level = num(GUSTAVO_ANNUAL_COL).dropna()
    annual = pd.Series({ix.year: 10.0 * np.log(v) for ix, v in annual_level.items()}, dtype=float)
    dciq = (10.0 * np.log(num(competition_col)).dropna()).diff()

    raw: dict[int, np.ndarray] = {}
    changes: dict[int, np.ndarray] = {}
    for year in annual.index:
        periods = [pd.Period(f"{year}Q{q}", freq="Q") for q in range(1, 5)]
        if all(p in dciq.index and np.isfinite(dciq.get(p, np.nan)) for p in periods):
            dq = np.array([dciq[p] for p in periods], float)
            total = float(dq.sum())
            if abs(total) > 1e-10:
                changes[int(year)] = dq
                raw[int(year)] = dq / total

    stable = np.array([w for w in raw.values() if np.max(np.abs(w)) <= stable_raw_weight_max])
    if stable.size == 0:
        raise ValueError("No stable Capital IQ annual profiles are available.")
    average = np.median(stable, axis=0)
    average = average / average.sum()

    first, last = int(min(annual.index)), int(max(annual.index))
    index = pd.period_range(f"{first}Q1", f"{last}Q4", freq="Q")
    quarterly = pd.Series(index=index, dtype=float)
    used: dict[int, np.ndarray] = {}
    coherence: dict[int, float] = {}
    source: dict[int, str] = {}
    for year in annual.index:
        year = int(year)
        previous = float(annual.get(year - 1, annual[year]))
        change = float(annual[year] - previous)
        if year in raw:
            dq = changes[year]
            c = float(abs(dq.sum()) / max(np.abs(dq).sum(), 1e-12))
            weights = c * raw[year] + (1.0 - c) * average
            source[year] = "capital_iq_shrunk" if c < 0.999 else "capital_iq"
            coherence[year] = c
        else:
            weights = average.copy()
            source[year] = "average_missing"
            coherence[year] = 0.0
        weights = weights / weights.sum()
        used[year] = weights
        cumulative = 0.0
        for quarter in range(1, 5):
            cumulative += float(weights[quarter - 1] * change)
            quarterly[pd.Period(f"{year}Q{quarter}", freq="Q")] = previous + cumulative

    anchor_error = max(
        abs(float(quarterly[pd.Period(f"{int(y)}Q4", freq="Q")]) - float(annual[y]))
        for y in annual.index
    )
    return AllocationResult(
        quarterly=quarterly.dropna(), annual=annual, average_weights=average,
        raw_weights=raw, used_weights=used, coherence=coherence, source=source,
        max_anchor_error=float(anchor_error),
    )


def load_experiment_data(config: dict) -> ExperimentData:
    frame = _load_frame()
    data_cfg = config["data"]
    alloc_cfg = config["competition_allocation"]
    allocation = build_quarterly_competition(
        frame, data_cfg["competition"],
        stable_raw_weight_max=float(alloc_cfg["stable_raw_weight_max"]),
    )
    num = lambda c: pd.to_numeric(frame[c], errors="coerce")
    d = pd.concat({
        "pi": num(data_cfg["inflation"]),
        "pi_lag": num(data_cfg["inflation_lag"]),
        "epi": num(data_cfg["expectation"]),
        "x": num(data_cfg["activity"]),
        "n": allocation.quarterly,
    }, axis=1).dropna()
    d = d[d.index >= pd.Period(data_cfg["sample_start"], freq="Q")]
    n_obs = (d["n"] - d["n"].mean()).to_numpy(float)
    case = CaseData(
        case=1, label="ppi__neg_unemp_gap__gustavo_capiq", periods=d.index,
        pi=d["pi"].to_numpy(float), epi=d["epi"].to_numpy(float),
        x=d["x"].to_numpy(float), n_obs=n_obs, exact_anchor=False, gE=None,
        s_x=robust_scale(d["x"]), s_N=robust_scale(n_obs), s_pi=robust_scale(d["pi"]),
        s_E=None, pi_lag=d["pi_lag"].to_numpy(float),
    )
    return ExperimentData(case=case, frame=d, allocation=allocation)


def _prior_maps(model: str, priors: Priors, lambda_mean: float, lambda_sd: float):
    means = {
        "intercept": 0.0, "alpha_b": priors.alpha_b_mean, "alpha_f": priors.alpha_mean,
        "kappa_0": 0.0, "delta": 0.0, "delta_1": 0.0, "delta_2": 0.0,
        "theta_0": 0.0, "gamma": 0.0, "lambda": lambda_mean,
    }
    sds = {
        "intercept": priors.intercept_sd, "alpha_b": priors.alpha_b_sd,
        "alpha_f": priors.alpha_sd, "kappa_0": priors.kappa0_sd,
        "delta": priors.delta_sd, "delta_1": priors.delta_sd,
        "delta_2": priors.gamma_sd / max(priors.theta0_sd, 1e-8) * priors.delta_sd,
        "theta_0": priors.theta0_sd, "gamma": priors.gamma_sd, "lambda": lambda_sd,
    }
    names = COEFF_NAMES[model]
    return names, means, sds


def _design(model: str, data: CaseData, nbar, nhat, lam: float) -> np.ndarray:
    x = data.x
    columns = {
        "intercept": np.ones(data.n_periods),
        "alpha_b": data.pi_lag,
        "alpha_f": data.epi,
        "kappa_0": x,
        "delta": x * nbar,
        "delta_1": x * nbar,
        "delta_2": x * nbar ** 2,
        "theta_0": -nhat,
        "gamma": -nbar * nhat,
    }
    if model in HSA_MODELS:
        columns["theta_0"] = lam * x * nbar - nhat
        if model == "hsa_dynamic":
            columns["gamma"] = 0.5 * lam * x * nbar ** 2 - nbar * nhat
    return np.column_stack([columns[name] for name in COEFF_NAMES[model]])


def _mu(model: str, data: CaseData, beta: dict, nbar, nhat, lam: float) -> np.ndarray:
    return _design(model, data, nbar, nhat, lam) @ np.array([beta[n] for n in COEFF_NAMES[model]])


def _whiten(y, X, phi: float):
    scale0 = np.sqrt(max(1e-8, 1.0 - phi ** 2))
    yw = np.empty_like(y, dtype=float)
    Xw = np.empty_like(X, dtype=float)
    yw[0] = scale0 * y[0]
    Xw[0] = scale0 * X[0]
    yw[1:] = y[1:] - phi * y[:-1]
    Xw[1:] = X[1:] - phi * X[:-1]
    return yw, Xw


def _inflation_loglik(y, mu, sigma2: float, phi: float, include_constants: bool = False) -> float:
    resid = y - mu
    innovation = np.empty_like(resid)
    innovation[0] = np.sqrt(max(1e-8, 1.0 - phi ** 2)) * resid[0]
    innovation[1:] = resid[1:] - phi * resid[:-1]
    out = -0.5 * float(innovation @ innovation) / sigma2 + 0.5 * np.log(max(1e-8, 1.0 - phi ** 2))
    if include_constants:
        out -= 0.5 * len(y) * np.log(2.0 * np.pi * sigma2)
    return float(out)


def _tridiagonal_cholesky(diag, off):
    diag = np.asarray(diag, float)
    off = np.asarray(off, float)
    ld = np.empty_like(diag)
    ls = np.empty_like(off)
    ld[0] = np.sqrt(max(diag[0], 1e-12))
    for i in range(1, len(diag)):
        ls[i - 1] = off[i - 1] / ld[i - 1]
        ld[i] = np.sqrt(max(diag[i] - ls[i - 1] ** 2, 1e-12))
    return ld, ls


def _solve_precision(ld, ls, rhs):
    rhs = np.asarray(rhs, float)
    y = np.empty_like(rhs)
    y[0] = rhs[0] / ld[0]
    for i in range(1, len(rhs)):
        y[i] = (rhs[i] - ls[i - 1] * y[i - 1]) / ld[i]
    x = np.empty_like(rhs)
    x[-1] = y[-1] / ld[-1]
    for i in range(len(rhs) - 2, -1, -1):
        x[i] = (y[i] - ls[i] * x[i + 1]) / ld[i]
    return x


def _sample_zero_precision(ld, ls, rng):
    z = rng.normal(size=len(ld))
    x = np.empty_like(z)
    x[-1] = z[-1] / ld[-1]
    for i in range(len(z) - 2, -1, -1):
        x[i] = (z[i] - ls[i] * x[i + 1]) / ld[i]
    return x


def _conditional_nbar(n_obs, nhat, var_bar, var_nu, init_sd, rng):
    T = len(n_obs)
    diag = np.full(T, 2.0 / var_bar + 1.0 / var_nu)
    diag[0] = 1.0 / init_sd ** 2 + 1.0 / var_bar + 1.0 / var_nu
    diag[-1] = 1.0 / var_bar + 1.0 / var_nu
    off = np.full(T - 1, -1.0 / var_bar)
    rhs = (n_obs - nhat) / var_nu
    ld, ls = _tridiagonal_cholesky(diag, off)
    mean = _solve_precision(ld, ls, rhs)
    return mean, lambda: _sample_zero_precision(ld, ls, rng)


def _conditional_nhat(n_obs, nbar, rho, var_hat, var_nu, rng):
    T = len(n_obs)
    diag = np.full(T, (1.0 + rho ** 2) / var_hat + 1.0 / var_nu)
    diag[0] = 1.0 / var_hat + 1.0 / var_nu
    diag[-1] = 1.0 / var_hat + 1.0 / var_nu
    off = np.full(T - 1, -rho / var_hat)
    rhs = (n_obs - nbar) / var_nu
    ld, ls = _tridiagonal_cholesky(diag, off)
    mean = _solve_precision(ld, ls, rhs)
    return mean, lambda: _sample_zero_precision(ld, ls, rng)


def _elliptical_slice(current, mean, sample_zero: Callable[[], np.ndarray], loglik, rng):
    centered = current - mean
    direction = sample_zero()
    threshold = loglik(current) + np.log(rng.uniform())
    angle = rng.uniform(0.0, 2.0 * np.pi)
    low, high = angle - 2.0 * np.pi, angle
    for evaluations in range(1, 101):
        proposal = mean + centered * np.cos(angle) + direction * np.sin(angle)
        if loglik(proposal) >= threshold:
            return proposal, evaluations
        if angle < 0:
            low = angle
        else:
            high = angle
        angle = rng.uniform(low, high)
    return current, 100


def _draw_coefficients(rng, y, X, phi, prior_mean, prior_sd, sigma2, ig_shape, ig_scale):
    yw, Xw = _whiten(y, X, phi)
    precision = np.diag(1.0 / prior_sd ** 2) + Xw.T @ Xw / sigma2
    covariance = np.linalg.inv(force_pd(precision))
    mean = covariance @ (prior_mean / prior_sd ** 2 + Xw.T @ yw / sigma2)
    beta = rng.multivariate_normal(mean, force_pd(covariance))
    resid = yw - Xw @ beta
    sigma2 = _draw_ig(rng, ig_shape + y.size / 2.0, ig_scale + 0.5 * float(resid @ resid))
    return beta, sigma2


def _draw_lambda(rng, model, data, beta, nbar, nhat, phi, sigma2, prior_mean, prior_sd):
    theta = beta["theta_0"]
    gamma = beta.get("gamma", 0.0)
    without = (beta["intercept"] + beta["alpha_b"] * data.pi_lag + beta["alpha_f"] * data.epi
               + beta["kappa_0"] * data.x - theta * nhat - gamma * nbar * nhat)
    loading = theta * data.x * nbar
    if model == "hsa_dynamic":
        loading = loading + 0.5 * gamma * data.x * nbar ** 2
    yw, Xw = _whiten(data.pi - without, loading[:, None], phi)
    precision = 1.0 / prior_sd ** 2 + float(Xw[:, 0] @ Xw[:, 0]) / sigma2
    variance = 1.0 / precision
    mean = variance * (prior_mean / prior_sd ** 2 + float(Xw[:, 0] @ yw) / sigma2)
    return float(rng.normal(mean, np.sqrt(variance)))


def _draw_phi(rng, y, mu, sigma2, current, prior_mean, prior_sd, proposal_sd=0.045):
    proposal = float(current + rng.normal(0.0, proposal_sd))
    if abs(proposal) >= 0.98:
        return current, False
    def lp(value):
        return (_inflation_loglik(y, mu, sigma2, value, include_constants=False)
                + norm.logpdf(value, prior_mean, prior_sd))
    if np.log(rng.uniform()) < lp(proposal) - lp(current):
        return proposal, True
    return current, False


def _bounded_regression_draw(rng, y, x, variance, prior_mean, prior_sd, lower, upper):
    precision = 1.0 / prior_sd ** 2 + float(x @ x) / variance
    post_sd = np.sqrt(1.0 / precision)
    post_mean = (prior_mean / prior_sd ** 2 + float(x @ y) / variance) / precision
    a, b = (lower - post_mean) / post_sd, (upper - post_mean) / post_sd
    return float(truncnorm.rvs(a, b, loc=post_mean, scale=post_sd, random_state=rng))


def fit_model(
    data: CaseData,
    model: str,
    *, iterations: int, warmup: int, thin: int, chains: int, seed: int,
    coefficient_scale: float, lambda_mean: float, lambda_sd: float,
    phi_mean: float, phi_sd: float, rho_lower: float, rho_upper: float,
) -> FitResult:
    if model not in COEFF_NAMES:
        raise ValueError(f"Unknown model {model}")
    priors = build_priors(data, coef_scale=coefficient_scale, hybrid=True)
    names, means, sds = _prior_maps(model, priors, lambda_mean, lambda_sd)
    stored_names = names + (("lambda",) if model in HSA_MODELS else ())
    nsave = (iterations - warmup + thin - 1) // thin
    shape = (chains, nsave)
    draws = np.zeros(shape + (len(stored_names),))
    sigma_pi = np.zeros(shape); phi_out = np.zeros(shape); rho_out = np.zeros(shape)
    sbar = np.zeros(shape); shat = np.zeros(shape); snu = np.zeros(shape)
    nbar_out = np.zeros(shape + (data.n_periods,)); nhat_out = np.zeros_like(nbar_out)
    phi_accept = np.zeros(chains); ess_eval_n = np.zeros(chains); ess_eval_h = np.zeros(chains)

    pmean = np.array([means[n] for n in names]); psd = np.array([sds[n] for n in names])
    for chain in range(chains):
        rng = np.random.default_rng(seed + 7919 * chain + 104729 * list(COEFF_NAMES).index(model))
        beta_vec = pmean.copy(); beta = dict(zip(names, beta_vec))
        lam = lambda_mean
        phi = phi_mean; rho = 0.5
        sigma2 = priors.sigma_pi_b / (priors.ig_shape - 1.0)
        var_bar = priors.sigma_bar_b / (priors.ig_shape - 1.0)
        var_hat = priors.sigma_hat_b / (priors.ig_shape - 1.0)
        var_nu = priors.sigma_nu_b / (priors.ig_shape - 1.0)
        nbar = pd.Series(data.n_obs).ewm(halflife=8, adjust=False).mean().to_numpy(copy=True)
        nbar -= nbar.mean(); nhat = data.n_obs - nbar
        save = 0
        for iteration in range(iterations):
            # Slow state conditional base: RW prior + competition measurement.
            mean_n, sample_n = _conditional_nbar(
                data.n_obs, nhat, var_bar, var_nu, priors.ntilde0_sd, rng
            )
            def ll_n(candidate):
                return _inflation_loglik(data.pi, _mu(model, data, beta, candidate, nhat, lam), sigma2, phi)
            nbar, ne = _elliptical_slice(nbar, mean_n, sample_n, ll_n, rng)
            ess_eval_n[chain] += ne

            # Cyclical state conditional base: AR(1) prior + competition measurement.
            mean_h, sample_h = _conditional_nhat(data.n_obs, nbar, rho, var_hat, var_nu, rng)
            def ll_h(candidate):
                return _inflation_loglik(data.pi, _mu(model, data, beta, nbar, candidate, lam), sigma2, phi)
            nhat, he = _elliptical_slice(nhat, mean_h, sample_h, ll_h, rng)
            ess_eval_h[chain] += he

            X = _design(model, data, nbar, nhat, lam)
            beta_vec, sigma2 = _draw_coefficients(
                rng, data.pi, X, phi, pmean, psd, sigma2, priors.ig_shape, priors.sigma_pi_b
            )
            beta = dict(zip(names, beta_vec))
            if model in HSA_MODELS:
                lam = _draw_lambda(
                    rng, model, data, beta, nbar, nhat, phi, sigma2, lambda_mean, lambda_sd
                )
            mu = _mu(model, data, beta, nbar, nhat, lam)
            phi, accepted = _draw_phi(rng, data.pi, mu, sigma2, phi, phi_mean, phi_sd)
            phi_accept[chain] += float(accepted)
            rho = _bounded_regression_draw(
                rng, nhat[1:], nhat[:-1], var_hat, 0.5, 0.25, rho_lower, rho_upper
            )

            rb = np.diff(nbar)
            var_bar = _draw_ig(rng, priors.ig_shape + rb.size / 2,
                               priors.sigma_bar_b + 0.5 * float(rb @ rb))
            rh = nhat[1:] - rho * nhat[:-1]
            var_hat = _draw_ig(rng, priors.ig_shape + rh.size / 2,
                               priors.sigma_hat_b + 0.5 * float(rh @ rh))
            rn = data.n_obs - nbar - nhat
            var_nu = _draw_ig(rng, priors.ig_shape + rn.size / 2,
                              priors.sigma_nu_b + 0.5 * float(rn @ rn))

            if iteration >= warmup and (iteration - warmup) % thin == 0:
                values = list(beta_vec) + ([lam] if model in HSA_MODELS else [])
                draws[chain, save] = values
                sigma_pi[chain, save] = np.sqrt(sigma2); phi_out[chain, save] = phi
                rho_out[chain, save] = rho; sbar[chain, save] = np.sqrt(var_bar)
                shat[chain, save] = np.sqrt(var_hat); snu[chain, save] = np.sqrt(var_nu)
                nbar_out[chain, save] = nbar; nhat_out[chain, save] = nhat
                save += 1

    rhats = {}
    for index, name in enumerate(stored_names):
        rhats[name] = float(np.asarray(az.rhat(draws[:, :, index], method="rank")))
    for name, values in (("sigma_pi", sigma_pi), ("phi", phi_out), ("rho", rho_out),
                         ("sigma_bar", sbar), ("sigma_hat", shat), ("sigma_nu", snu)):
        rhats[name] = float(np.asarray(az.rhat(values, method="rank")))
    diagnostics = {
        "rhat": rhats,
        "max_rhat": float(max(rhats.values())),
        "phi_acceptance": (phi_accept / iterations).tolist(),
        "mean_ess_bracket_evals_nbar": (ess_eval_n / iterations).tolist(),
        "mean_ess_bracket_evals_nhat": (ess_eval_h / iterations).tolist(),
    }
    prior_mean_map = {n: means[n] for n in stored_names}
    prior_sd_map = {n: sds[n] for n in stored_names}
    return FitResult(
        model=model, label=MODEL_LABELS[model], names=stored_names, draws=draws,
        sigma_pi=sigma_pi, phi=phi_out, rho=rho_out, sigma_bar=sbar,
        sigma_hat=shat, sigma_nu=snu, nbar=nbar_out, nhat=nhat_out,
        periods=tuple(map(str, data.periods)), prior_mean=prior_mean_map,
        prior_sd=prior_sd_map, diagnostics=diagnostics,
    )


def derived_paths(fit: FitResult, data: CaseData):
    flat = fit.draws.reshape(-1, fit.draws.shape[-1])
    nbar = fit.nbar.reshape(-1, fit.nbar.shape[-1])
    params = {name: flat[:, i] for i, name in enumerate(fit.names)}
    kappa0 = params["kappa_0"][:, None]
    if fit.model == "ces" or fit.model == "direct":
        kappa = np.broadcast_to(kappa0, nbar.shape)
    elif fit.model in {"slope", "free_static"}:
        kappa = kappa0 + params["delta"][:, None] * nbar
    elif fit.model == "hsa_static":
        kappa = kappa0 + (params["lambda"] * params["theta_0"])[:, None] * nbar
    elif fit.model == "free_dynamic":
        kappa = (kappa0 + params["delta_1"][:, None] * nbar
                 + params["delta_2"][:, None] * nbar ** 2)
    else:
        kappa = (kappa0 + (params["lambda"] * params["theta_0"])[:, None] * nbar
                 + (0.5 * params["lambda"] * params["gamma"])[:, None] * nbar ** 2)

    if fit.model in {"ces", "slope"}:
        theta = np.zeros_like(nbar)
    elif fit.model in {"direct", "free_static", "hsa_static"}:
        theta = np.broadcast_to(params["theta_0"][:, None], nbar.shape)
    else:
        theta = params["theta_0"][:, None] + params["gamma"][:, None] * nbar
    return kappa, theta


def pointwise_loglik(fit: FitResult, data: CaseData, max_draws: int = 2000):
    flat = fit.draws.reshape(-1, fit.draws.shape[-1])
    nbar = fit.nbar.reshape(-1, fit.nbar.shape[-1]); nhat = fit.nhat.reshape(-1, fit.nhat.shape[-1])
    sigma = fit.sigma_pi.reshape(-1); phi = fit.phi.reshape(-1)
    indices = np.linspace(0, len(flat) - 1, min(max_draws, len(flat))).astype(int)
    out = np.zeros((len(indices), data.n_periods))
    predictions = np.zeros_like(out)
    for row, idx in enumerate(indices):
        beta = {name: flat[idx, j] for j, name in enumerate(fit.names) if name != "lambda"}
        lam = float(flat[idx, fit.names.index("lambda")]) if "lambda" in fit.names else 0.0
        mu = _mu(fit.model, data, beta, nbar[idx], nhat[idx], lam)
        resid = data.pi - mu
        pred = mu.copy()
        pred[1:] += phi[idx] * resid[:-1]
        innovations = data.pi - pred
        innovations[0] *= np.sqrt(max(1e-8, 1.0 - phi[idx] ** 2))
        variance = sigma[idx] ** 2
        out[row] = -0.5 * (np.log(2 * np.pi * variance) + innovations ** 2 / variance)
        out[row, 0] += 0.5 * np.log(max(1e-8, 1.0 - phi[idx] ** 2))
        predictions[row] = pred
    return out, predictions


def _logmeanexp(values, axis=0):
    maximum = np.max(values, axis=axis, keepdims=True)
    return np.squeeze(maximum, axis=axis) + np.log(np.mean(np.exp(values - maximum), axis=axis))


def comparison_metrics(fit: FitResult, data: CaseData):
    loglik, predictions = pointwise_loglik(fit, data)
    lppd = float(np.sum(_logmeanexp(loglik, axis=0)))
    pwaic = float(np.sum(np.var(loglik, axis=0, ddof=1)))
    waic = -2.0 * (lppd - pwaic)
    pred_mean = predictions.mean(axis=0)
    rmse = float(np.sqrt(np.mean((data.pi - pred_mean) ** 2)))
    rng = np.random.default_rng(90210)
    sigma = fit.sigma_pi.reshape(-1)
    phi = fit.phi.reshape(-1)
    indices = np.linspace(0, len(sigma) - 1, len(predictions)).astype(int)
    noise = rng.normal(size=predictions.shape) * sigma[indices, None]
    noise[:, 0] /= np.sqrt(np.maximum(1e-8, 1.0 - phi[indices] ** 2))
    predictive_draws = predictions + noise
    coverage = float(np.mean((data.pi >= np.percentile(predictive_draws, 2.5, axis=0))
                             & (data.pi <= np.percentile(predictive_draws, 97.5, axis=0))))
    return {"waic": waic, "lppd": lppd, "p_waic": pwaic,
            "predictive_rmse": rmse, "predictive_coverage_95": coverage}


def particle_loglik(fit: FitResult, data: CaseData, params: dict, particles: int, seed: int):
    """Integrated likelihood at one global-parameter point (bootstrap PF)."""
    rng = np.random.default_rng(seed)
    T = data.n_periods
    n = rng.normal(0.0, params["init_sd"], particles)
    h = rng.normal(0.0, params["sigma_hat"] / np.sqrt(max(1e-6, 1 - params["rho"] ** 2)), particles)
    previous_resid = np.zeros(particles)
    loglik = 0.0
    for t in range(T):
        if t > 0:
            n += rng.normal(0.0, params["sigma_bar"], particles)
            h = params["rho"] * h + rng.normal(0.0, params["sigma_hat"], particles)
        beta = params["beta"]
        common = (beta["intercept"] + beta["alpha_b"] * data.pi_lag[t]
                  + beta["alpha_f"] * data.epi[t] + beta["kappa_0"] * data.x[t])
        if fit.model == "ces":
            mu = np.full(particles, common)
        elif fit.model == "slope":
            mu = common + beta["delta"] * data.x[t] * n
        elif fit.model == "direct":
            mu = common - beta["theta_0"] * h
        elif fit.model == "free_static":
            mu = common + beta["delta"] * data.x[t] * n - beta["theta_0"] * h
        elif fit.model == "hsa_static":
            mu = common + params["lambda"] * beta["theta_0"] * data.x[t] * n - beta["theta_0"] * h
        elif fit.model == "free_dynamic":
            mu = (common + beta["delta_1"] * data.x[t] * n + beta["delta_2"] * data.x[t] * n ** 2
                  - (beta["theta_0"] + beta["gamma"] * n) * h)
        else:
            theta = beta["theta_0"] + beta["gamma"] * n
            kappa_extra = (params["lambda"] * beta["theta_0"] * n
                           + 0.5 * params["lambda"] * beta["gamma"] * n ** 2)
            mu = common + data.x[t] * kappa_extra - theta * h
        conditional_mean = mu + params["phi"] * previous_resid
        logw = -0.5 * (np.log(2 * np.pi * params["sigma_pi"] ** 2)
                       + (data.pi[t] - conditional_mean) ** 2 / params["sigma_pi"] ** 2)
        logw += -0.5 * (np.log(2 * np.pi * params["sigma_nu"] ** 2)
                        + (data.n_obs[t] - n - h) ** 2 / params["sigma_nu"] ** 2)
        maximum = float(logw.max()); weights = np.exp(logw - maximum); total = float(weights.sum())
        loglik += maximum + np.log(total / particles)
        weights /= total
        idx = rng.choice(particles, size=particles, p=weights)
        n, h, mu = n[idx], h[idx], mu[idx]
        previous_resid = data.pi[t] - mu
    return float(loglik)


def approximate_log_marginal(fit: FitResult, data: CaseData, particles: int, seed: int):
    """Laplace-Metropolis approximation using a PF-integrated likelihood."""
    blocks = [fit.draws]
    scalar_names = list(fit.names)
    for name, values in (("phi", fit.phi), ("rho", fit.rho), ("log_sigma_pi", np.log(fit.sigma_pi)),
                         ("log_sigma_bar", np.log(fit.sigma_bar)),
                         ("log_sigma_hat", np.log(fit.sigma_hat)),
                         ("log_sigma_nu", np.log(fit.sigma_nu))):
        blocks.append(values[:, :, None]); scalar_names.append(name)
    transformed = np.concatenate(blocks, axis=2).reshape(-1, len(scalar_names))
    center = transformed.mean(axis=0)
    covariance = np.cov(transformed, rowvar=False)
    sign, logdet = np.linalg.slogdet(force_pd(covariance))
    if sign <= 0:
        return float("nan")
    values = dict(zip(scalar_names, center))
    beta = {name: values[name] for name in fit.names if name != "lambda"}
    params = {
        "beta": beta, "lambda": values.get("lambda", 0.0), "phi": values["phi"], "rho": values["rho"],
        "sigma_pi": np.exp(values["log_sigma_pi"]), "sigma_bar": np.exp(values["log_sigma_bar"]),
        "sigma_hat": np.exp(values["log_sigma_hat"]), "sigma_nu": np.exp(values["log_sigma_nu"]),
        "init_sd": 2.0 * data.s_N,
    }
    ll = particle_loglik(fit, data, params, particles=particles, seed=seed)
    lp = 0.0
    for name in fit.names:
        lp += norm.logpdf(values[name], fit.prior_mean[name], fit.prior_sd[name])
    lp += norm.logpdf(values["phi"], 0.0, 0.35) + norm.logpdf(values["rho"], 0.5, 0.25)
    priors = build_priors(data, coef_scale=0.20, hybrid=True)
    for key, scale in (("log_sigma_pi", priors.sigma_pi_b), ("log_sigma_bar", priors.sigma_bar_b),
                       ("log_sigma_hat", priors.sigma_hat_b), ("log_sigma_nu", priors.sigma_nu_b)):
        variance = np.exp(2.0 * values[key])
        lp += (priors.ig_shape * np.log(scale) - gammaln(priors.ig_shape)
               - (priors.ig_shape + 1.0) * np.log(variance) - scale / variance)
        lp += np.log(2.0) + 2.0 * values[key]  # d variance / d log(sigma)
    dimension = len(center)
    return float(ll + lp + 0.5 * dimension * np.log(2.0 * np.pi) + 0.5 * logdet)


def save_fit(path: Path, fit: FitResult):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path, model=fit.model, label=fit.label, names=np.asarray(fit.names), draws=fit.draws,
        sigma_pi=fit.sigma_pi, phi=fit.phi, rho=fit.rho, sigma_bar=fit.sigma_bar,
        sigma_hat=fit.sigma_hat, sigma_nu=fit.sigma_nu, nbar=fit.nbar, nhat=fit.nhat,
        periods=np.asarray(fit.periods), prior_names=np.asarray(list(fit.prior_mean)),
        prior_mean=np.asarray(list(fit.prior_mean.values())), prior_sd=np.asarray(list(fit.prior_sd.values())),
    )


def load_fit(path: Path, diagnostics: dict | None = None) -> FitResult:
    z = np.load(path, allow_pickle=False)
    pnames = [str(x) for x in z["prior_names"]]
    return FitResult(
        model=str(z["model"]), label=str(z["label"]), names=tuple(map(str, z["names"])),
        draws=z["draws"], sigma_pi=z["sigma_pi"], phi=z["phi"], rho=z["rho"],
        sigma_bar=z["sigma_bar"], sigma_hat=z["sigma_hat"], sigma_nu=z["sigma_nu"],
        nbar=z["nbar"], nhat=z["nhat"], periods=tuple(map(str, z["periods"])),
        prior_mean=dict(zip(pnames, map(float, z["prior_mean"]))),
        prior_sd=dict(zip(pnames, map(float, z["prior_sd"]))), diagnostics=diagnostics or {},
    )
