"""Core data, exact-state, nested-design, and MCMC functions.

This module implements the static confirmatory ladder in SPECIFICATION.md.  The
Capital-IQ allocation distribution is an external measurement-only cut.  Given
each allocation draw, the slow/cycle split is sampled either jointly with the
inflation equation or from the state laws alone.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import arviz as az
import numpy as np
import pandas as pd
from scipy.special import expit, logit
from scipy.stats import beta as beta_dist, norm

from nkpc_hsa.phillips.state import _draw_ig
from nkpc_hsa.report_models.cases import CaseData, GUSTAVO_ANNUAL_COL, _load_frame
from nkpc_hsa.report_models.engine import build_priors
from tests.hsa_exact_n_decomposition.functions import (
    AllocationPosterior,
    build_allocation_posterior,
)
from tests.hsa_lambda_dynamic.functions import (
    _draw_coefficients,
    _draw_phi,
    _inflation_loglik,
    _whiten,
    robust_scale,
)


BASE_NAMES = ("intercept", "alpha_b", "alpha_f", "kappa_0")
PRIMARY_MODELS = ("ces", "slow_slope", "direct", "free_static_combined")
BENCHMARK_BASE_MODELS = (
    "ces", "slow_slope", "direct", "free_static_combined",
)


@dataclass(frozen=True)
class CellData:
    label: str
    role: str
    price: str
    activity_role: str
    periods: pd.PeriodIndex
    pi: np.ndarray
    pi_lag: np.ndarray
    epi: np.ndarray
    x: np.ndarray
    allocation_positions: np.ndarray
    s_pi: float
    s_x: float
    s_q: float

    @property
    def n_periods(self) -> int:
        return len(self.pi)

    def case_data(self, q_reference: np.ndarray) -> CaseData:
        return CaseData(
            case=1, label=self.label, periods=self.periods,
            pi=self.pi, epi=self.epi, x=self.x, n_obs=q_reference,
            exact_anchor=True, gE=None, s_x=self.s_x, s_N=self.s_q,
            s_pi=self.s_pi, s_E=None, pi_lag=self.pi_lag,
        )


@dataclass(frozen=True)
class ExperimentData:
    allocation: AllocationPosterior
    q0: float
    allocation_mean_raw: np.ndarray
    cells: dict[str, CellData]
    allocation_summary: dict[str, Any]

    def draw_q(self, rng: np.random.Generator, cell: CellData) -> np.ndarray:
        raw = self.allocation.draw_path(rng)
        return raw[cell.allocation_positions] - self.q0

    def mean_q(self, cell: CellData) -> np.ndarray:
        return self.allocation_mean_raw[cell.allocation_positions] - self.q0


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    role: str
    coefficient_names: tuple[str, ...]
    lambda_fixed: float | None = None
    free_lambda: bool = False


@dataclass
class ModelFit:
    spec: ModelSpec
    periods: tuple[str, ...]
    names: tuple[str, ...]
    draws: np.ndarray
    sigma_pi: np.ndarray
    phi: np.ndarray
    n_total: np.ndarray
    nbar: np.ndarray
    nhat: np.ndarray
    omega: np.ndarray
    tau: np.ndarray
    cycle_damping: np.ndarray
    cycle_period: np.ndarray
    prior_mean: dict[str, float]
    prior_sd: dict[str, float]
    diagnostics: dict[str, Any]


def _allocation_mean_path(allocation: AllocationPosterior) -> np.ndarray:
    values = pd.Series(index=allocation.periods, dtype=float)
    for year in allocation.annual.index:
        year = int(year)
        previous = float(allocation.annual.get(year - 1, allocation.annual[year]))
        change = float(allocation.annual[year] - previous)
        cumulative = 0.0
        for quarter in range(1, 5):
            cumulative += allocation.mean_weights[year][quarter - 1] * change
            values[pd.Period(f"{year}Q{quarter}", freq="Q")] = previous + cumulative
    return values.to_numpy(float)


def _build_cell(frame: pd.DataFrame, cfg: dict, allocation: AllocationPosterior,
                q0: float, mean_raw: np.ndarray, price: str,
                activity_role: str, activity_col: str) -> CellData:
    price_cfg = cfg["data"]["prices"][price]
    role = f"{price}_{activity_role}"
    num = lambda c: pd.to_numeric(frame[c], errors="coerce")
    marker = pd.Series(1.0, index=allocation.periods)
    d = pd.concat({
        "pi": num(price_cfg["inflation"]), "lag": num(price_cfg["inflation_lag"]),
        "epi": num(price_cfg["expectation"]), "x": num(activity_col), "marker": marker,
    }, axis=1).dropna()
    start, end = pd.Period(cfg["samples"]["start"], freq="Q"), pd.Period(cfg["samples"]["end"], freq="Q")
    d = d[(d.index >= start) & (d.index <= end)]
    positions = allocation.periods.get_indexer(d.index)
    if np.any(positions < 0):
        raise ValueError(f"{role}: sample dates fall outside the allocation index")
    qref = mean_raw[positions] - q0
    return CellData(
        label=role, role=role, price=price, activity_role=activity_role, periods=d.index,
        pi=d.pi.to_numpy(float), pi_lag=d.lag.to_numpy(float),
        epi=d.epi.to_numpy(float), x=d.x.to_numpy(float),
        allocation_positions=positions, s_pi=robust_scale(d.pi),
        s_x=robust_scale(d.x), s_q=robust_scale(qref),
    )


def load_experiment(config: dict) -> ExperimentData:
    frame = _load_frame()
    ac, dc = config["allocation"], config["data"]
    allocation = build_allocation_posterior(
        frame, dc["quarterly_indicator"], float(ac["stable_raw_weight_max"]),
        float(ac["covariance_scale"]),
    )
    mean_raw = _allocation_mean_path(allocation)
    start, end = pd.Period(config["samples"]["start"], freq="Q"), pd.Period(config["samples"]["end"], freq="Q")
    center_mask = (allocation.periods >= start) & (allocation.periods <= end)
    q0 = float(np.mean(mean_raw[center_mask]))
    cells = {}
    for price in config["data"]["prices"]:
        for activity_role, activity_col in config["data"]["activities"].items():
            cell = _build_cell(
                frame, config, allocation, q0, mean_raw,
                price, activity_role, activity_col,
            )
            cells[cell.role] = cell
    anchor_errors = []
    for year, value in allocation.annual.items():
        pos = allocation.periods.get_loc(pd.Period(f"{int(year)}Q4", freq="Q"))
        anchor_errors.append(abs(mean_raw[pos] - float(value)))
    summary = {
        "average_weights": allocation.average_weights.tolist(),
        "q0": q0,
        "max_mean_path_anchor_error": float(max(anchor_errors)),
        "mean_weights": {str(k): v.tolist() for k, v in allocation.mean_weights.items()},
        "coherence": {str(k): float(v) for k, v in allocation.coherence.items()},
    }
    return ExperimentData(allocation, q0, mean_raw, cells, summary)


def build_model_specs(config: dict) -> tuple[list[ModelSpec], list[ModelSpec]]:
    primary = [
        ModelSpec("ces", "primary", BASE_NAMES),
        ModelSpec("slow_slope", "primary", BASE_NAMES + ("delta_s",)),
        ModelSpec("direct", "primary", BASE_NAMES + ("theta",)),
        ModelSpec("free_static_combined", "primary", BASE_NAMES + ("delta_s", "theta")),
    ]
    benchmark = [
        ModelSpec("ces", "benchmark", BASE_NAMES),
        ModelSpec("slow_slope", "benchmark", BASE_NAMES + ("delta_s",)),
        ModelSpec("direct", "benchmark", BASE_NAMES + ("theta",)),
        ModelSpec("free_static_combined", "benchmark", BASE_NAMES + ("delta_s", "theta")),
    ]
    for value in config["models"]["hsa_lambda_grid"]:
        benchmark.append(ModelSpec(f"hsa_fixed_lambda_{value:g}", "benchmark", BASE_NAMES + ("theta",), float(value)))
    if config["models"].get("free_lambda_diagnostic", False):
        benchmark.append(ModelSpec("free_lambda_diagnostic", "benchmark", BASE_NAMES + ("theta",), None, True))
    return primary, benchmark


def _design(spec: ModelSpec, cell: CellData, q: np.ndarray, h: np.ndarray,
            lam: float | None = None) -> np.ndarray:
    bar = q - h
    columns: dict[str, np.ndarray] = {
        "intercept": np.ones(cell.n_periods), "alpha_b": cell.pi_lag,
        "alpha_f": cell.epi, "kappa_0": cell.x,
        "delta_s": bar * cell.x, "theta": -h,
    }
    if spec.lambda_fixed is not None or spec.free_lambda:
        use_lam = float(spec.lambda_fixed if spec.lambda_fixed is not None else lam)
        columns["theta"] = use_lam * bar * cell.x - h
    return np.column_stack([columns[n] for n in spec.coefficient_names])


def _mu(spec: ModelSpec, cell: CellData, q: np.ndarray, h: np.ndarray,
        beta: np.ndarray, lam: float | None = None) -> np.ndarray:
    return _design(spec, cell, q, h, lam) @ beta


def _cycle_coefficients(damping: float, period: float) -> tuple[float, float]:
    angle = 2.0 * np.pi / period
    return float(2.0 * damping * np.cos(angle)), float(-(damping**2))


def _cycle_unit_cov(damping: float, period: float) -> np.ndarray:
    """Stationary covariance of [h_0,h_1] when cycle innovation variance is one."""
    phi1, phi2 = _cycle_coefficients(damping, period)
    denominator = (1.0 + phi2) * ((1.0 - phi2) ** 2 - phi1**2)
    if denominator <= 1e-12:
        raise ValueError("AR(2) stochastic-cycle covariance is not positive definite")
    gamma0 = (1.0 - phi2) / denominator
    gamma1 = phi1 * gamma0 / (1.0 - phi2)
    covariance = np.array([[gamma0, gamma1], [gamma1, gamma0]])
    if np.linalg.det(covariance) <= 1e-12:
        raise ValueError("AR(2) stochastic-cycle covariance is singular")
    return covariance


def _cycle_sumsq(h: np.ndarray, damping: float, period: float) -> tuple[float, float]:
    covariance = _cycle_unit_cov(damping, period)
    initial = float(h[:2] @ np.linalg.solve(covariance, h[:2]))
    phi1, phi2 = _cycle_coefficients(damping, period)
    innovations = h[2:] - phi1 * h[1:-1] - phi2 * h[:-2]
    return initial + float(innovations @ innovations), float(np.linalg.slogdet(covariance)[1])


def _band2_cholesky(diag: np.ndarray, off1: np.ndarray, off2: np.ndarray):
    """Cholesky factor of a symmetric positive-definite bandwidth-two matrix."""
    n = len(diag); ld = np.empty(n); l1 = np.empty(n - 1); l2 = np.empty(n - 2)
    ld[0] = np.sqrt(max(float(diag[0]), 1e-12))
    if n > 1:
        l1[0] = off1[0] / ld[0]
        ld[1] = np.sqrt(max(float(diag[1] - l1[0] ** 2), 1e-12))
    for i in range(2, n):
        l2[i - 2] = off2[i - 2] / ld[i - 2]
        l1[i - 1] = (off1[i - 1] - l2[i - 2] * l1[i - 2]) / ld[i - 1]
        ld[i] = np.sqrt(max(float(diag[i] - l1[i - 1] ** 2 - l2[i - 2] ** 2), 1e-12))
    return ld, l1, l2


def _band2_solve(ld: np.ndarray, l1: np.ndarray, l2: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    n = len(ld); y = np.empty(n); y[0] = rhs[0] / ld[0]
    if n > 1: y[1] = (rhs[1] - l1[0] * y[0]) / ld[1]
    for i in range(2, n): y[i] = (rhs[i] - l1[i - 1] * y[i - 1] - l2[i - 2] * y[i - 2]) / ld[i]
    out = np.empty(n); out[-1] = y[-1] / ld[-1]
    if n > 1: out[-2] = (y[-2] - l1[-1] * out[-1]) / ld[-2]
    for i in range(n - 3, -1, -1):
        out[i] = (y[i] - l1[i] * out[i + 1] - l2[i] * out[i + 2]) / ld[i]
    return out


def _band2_zero_draw(ld: np.ndarray, l1: np.ndarray, l2: np.ndarray,
                     rng: np.random.Generator) -> np.ndarray:
    z = rng.normal(size=len(ld)); out = np.empty(len(ld)); out[-1] = z[-1] / ld[-1]
    if len(ld) > 1: out[-2] = (z[-2] - l1[-1] * out[-1]) / ld[-2]
    for i in range(len(ld) - 3, -1, -1):
        out[i] = (z[i] - l1[i] * out[i + 1] - l2[i] * out[i + 2]) / ld[i]
    return out


def _state_precision(q: np.ndarray, damping: float, period: float,
                     tau2: float, omega: float):
    T = len(q); vb = max(1e-10, omega * tau2); vh = max(1e-10, (1.0 - omega) * tau2)
    diag = np.zeros(T); off1 = np.zeros(T - 1); off2 = np.zeros(T - 2); rhs = np.zeros(T)
    dq = np.diff(q)
    for t in range(1, T):
        diag[t - 1] += 1.0 / vb; diag[t] += 1.0 / vb; off1[t - 1] -= 1.0 / vb
        rhs[t - 1] -= dq[t - 1] / vb; rhs[t] += dq[t - 1] / vb
    initial_precision = np.linalg.inv(_cycle_unit_cov(damping, period)) / vh
    diag[0] += initial_precision[0, 0]; diag[1] += initial_precision[1, 1]
    off1[0] += initial_precision[0, 1]
    phi1, phi2 = _cycle_coefficients(damping, period)
    for t in range(2, T):
        diag[t - 2] += phi2**2 / vh; diag[t - 1] += phi1**2 / vh; diag[t] += 1.0 / vh
        off1[t - 2] += phi1 * phi2 / vh; off1[t - 1] -= phi1 / vh; off2[t - 2] -= phi2 / vh
    return diag, off1, off2, rhs


def _sample_h(rng: np.random.Generator, q: np.ndarray, damping: float, period: float,
              tau2: float, omega: float, y: np.ndarray | None = None,
              constant: np.ndarray | None = None, loading: np.ndarray | None = None,
              sigma2: float | None = None, phi: float | None = None) -> np.ndarray:
    diag, off1, off2, rhs = _state_precision(q, damping, period, tau2, omega)
    if y is not None:
        assert constant is not None and loading is not None and sigma2 is not None and phi is not None
        scale0 = np.sqrt(max(1e-8, 1.0 - phi**2))
        y0 = scale0 * (y[0] - constant[0]); g0 = scale0 * loading[0]
        diag[0] += g0**2 / sigma2; rhs[0] += g0 * y0 / sigma2
        for t in range(1, len(q)):
            yt = (y[t] - constant[t]) - phi * (y[t - 1] - constant[t - 1])
            gp, gc = -phi * loading[t - 1], loading[t]
            diag[t - 1] += gp**2 / sigma2; diag[t] += gc**2 / sigma2
            off1[t - 1] += gp * gc / sigma2
            rhs[t - 1] += gp * yt / sigma2; rhs[t] += gc * yt / sigma2
    ld, l1, l2 = _band2_cholesky(diag, off1, off2)
    mean = _band2_solve(ld, l1, l2, rhs)
    return mean + _band2_zero_draw(ld, l1, l2, rng)


def _prior_maps(spec: ModelSpec, cell: CellData, config: dict):
    priors = build_priors(cell.case_data(np.zeros(cell.n_periods)), coef_scale=float(config["priors"]["coefficient_scale"]), hybrid=True)
    means = {
        "intercept": 0.0, "alpha_b": priors.alpha_b_mean, "alpha_f": priors.alpha_mean,
        "kappa_0": 0.0, "delta_s": 0.0, "theta": 0.0,
    }
    sds = {
        "intercept": priors.intercept_sd, "alpha_b": priors.alpha_b_sd,
        "alpha_f": priors.alpha_sd, "kappa_0": priors.kappa0_sd,
        "delta_s": priors.delta_sd, "theta": priors.theta0_sd,
    }
    if spec.lambda_fixed is not None:
        sds["theta"] = float((1.0 / priors.theta0_sd**2 + spec.lambda_fixed**2 / priors.delta_sd**2) ** -0.5)
    return priors, means, sds


def _draw_lambda_diagnostic(rng, cell, q, h, beta, phi, sigma2, mean, sd):
    b = dict(zip(BASE_NAMES + ("theta",), beta))
    common = b["intercept"] + b["alpha_b"] * cell.pi_lag + b["alpha_f"] * cell.epi + b["kappa_0"] * cell.x - b["theta"] * h
    loading = b["theta"] * (q - h) * cell.x
    yw, Xw = _whiten(cell.pi - common, loading[:, None], phi)
    precision = 1.0 / sd**2 + float(Xw[:, 0] @ Xw[:, 0]) / sigma2
    variance = 1.0 / precision
    post_mean = variance * (mean / sd**2 + float(Xw[:, 0] @ yw) / sigma2)
    return float(rng.normal(post_mean, np.sqrt(variance)))


def _bounded_value(z: float, lower: float, upper: float) -> float:
    return float(lower + (upper - lower) * expit(z))


def _bounded_z(value: float, lower: float, upper: float) -> float:
    return float(logit((value - lower) / (upper - lower)))


def _cycle_logtarget(z_damping: float, z_period: float, h: np.ndarray,
                     variance: float, config: dict) -> float:
    d_lo, d_hi = map(float, config["cycle_damping_bounds"])
    p_lo, p_hi = map(float, config["cycle_period_bounds_quarters"])
    damping = _bounded_value(z_damping, d_lo, d_hi); period = _bounded_value(z_period, p_lo, p_hi)
    ss, logdet = _cycle_sumsq(h, damping, period)
    u_d, u_p = expit(z_damping), expit(z_period)
    a, b = map(float, config["cycle_damping_prior"])
    target = -0.5 * logdet - 0.5 * ss / variance + beta_dist.logpdf(u_d, a, b)
    target += norm.logpdf(period, float(config["cycle_period_prior_mean"]), float(config["cycle_period_prior_sd"]))
    target += np.log(u_d) + np.log1p(-u_d) + np.log(u_p) + np.log1p(-u_p)
    return float(target)


def _draw_cycle_params(rng, h, damping, period, variance, config):
    d_lo, d_hi = map(float, config["cycle_damping_bounds"])
    p_lo, p_hi = map(float, config["cycle_period_bounds_quarters"])
    zd, zp = _bounded_z(damping, d_lo, d_hi), _bounded_z(period, p_lo, p_hi)
    accepted_damping = accepted_period = False
    candidate = zd + rng.normal(0.0, float(config["cycle_damping_mh_sd"]))
    if np.log(rng.uniform()) < _cycle_logtarget(candidate, zp, h, variance, config) - _cycle_logtarget(zd, zp, h, variance, config):
        zd = candidate; accepted_damping = True
    candidate = zp + rng.normal(0.0, float(config["cycle_period_mh_sd"]))
    if np.log(rng.uniform()) < _cycle_logtarget(zd, candidate, h, variance, config) - _cycle_logtarget(zd, zp, h, variance, config):
        zp = candidate; accepted_period = True
    return (_bounded_value(zd, d_lo, d_hi), _bounded_value(zp, p_lo, p_hi),
            accepted_damping, accepted_period)


def _omega_logtarget_ar2(z, rb, cycle_ss, tau2, a, b):
    omega = float(expit(z)); T = len(rb) + 1
    out = -(T - 1) / 2 * np.log(omega) - float(rb @ rb) / (2 * omega * tau2)
    out += -T / 2 * np.log1p(-omega) - cycle_ss / (2 * (1 - omega) * tau2)
    out += beta_dist.logpdf(omega, a, b) + np.log(omega) + np.log1p(-omega)
    return float(out)


def _update_state_params(rng, q, h, damping, period, omega, tau2, s_q, config):
    sp = config["state"]
    bar = q - h
    damping, period, accepted_damping, accepted_period = _draw_cycle_params(
        rng, h, damping, period, (1.0 - omega) * tau2, sp,
    )
    rb = np.diff(bar); cycle_ss, _ = _cycle_sumsq(h, damping, period)
    proposal = logit(omega) + rng.normal(0.0, float(sp["omega_mh_sd"]))
    a, b = map(float, sp["omega_prior"])
    old = _omega_logtarget_ar2(logit(omega), rb, cycle_ss, tau2, a, b)
    new = _omega_logtarget_ar2(proposal, rb, cycle_ss, tau2, a, b)
    accepted_omega = False
    if np.log(rng.uniform()) < new - old:
        omega = float(expit(proposal)); accepted_omega = True
    tau_scale = 2.0 * (float(sp["tau2_scale_fraction"]) * float(s_q)) ** 2
    scaled = float(rb @ rb) / omega + cycle_ss / (1.0 - omega)
    tau2 = _draw_ig(rng, float(sp["tau2_prior_shape"]) + (2 * len(q) - 1) / 2,
                    tau_scale + 0.5 * scaled)
    return bar, damping, period, omega, tau2, accepted_damping, accepted_period, accepted_omega


def _state_diagnostics(nt, nb, nh, om, tau, damping, period):
    rhat = {
        "omega": float(az.rhat(om, method="rank")), "tau": float(az.rhat(tau, method="rank")),
        "cycle_damping": float(az.rhat(damping, method="rank")),
        "cycle_period": float(az.rhat(period, method="rank")),
    }
    return {"rhat": rhat, "max_rhat": max(rhat.values()),
            "exact_identity_error": float(np.max(np.abs(nt - nb - nh)))}


def fit_model(experiment: ExperimentData, cell: CellData, spec: ModelSpec, config: dict,
              sampling: dict, seed: int) -> ModelFit:
    priors, means, sds = _prior_maps(spec, cell, config)
    names = spec.coefficient_names + (("lambda",) if spec.free_lambda else ())
    pmean = np.array([means[n] for n in spec.coefficient_names]); psd = np.array([sds[n] for n in spec.coefficient_names])
    chains, iterations = int(sampling["chains"]), int(sampling["iterations"])
    warmup, thin = int(sampling["warmup"]), int(sampling["thin"])
    nsave = (iterations - warmup + thin - 1) // thin; T = cell.n_periods; shape = (chains, nsave)
    draws = np.zeros(shape + (len(names),)); sig = np.zeros(shape); phis = np.zeros(shape)
    nt = np.zeros(shape + (T,)); nb = np.zeros_like(nt); nh = np.zeros_like(nt)
    om = np.zeros(shape); taus = np.zeros(shape); damping_out = np.zeros(shape); period_out = np.zeros(shape)
    phi_acc = np.zeros(chains); omega_acc = np.zeros(chains)
    damping_acc = np.zeros(chains); period_acc = np.zeros(chains)
    sp, ep, diagnostic = config["state"], config["inflation_error"], config["diagnostic"]
    for ch in range(chains):
        rng = np.random.default_rng(seed + 7919 * ch)
        beta = pmean.copy(); phi = float(ep["phi_prior_mean"])
        sigma2 = priors.sigma_pi_b / (priors.ig_shape - 1.0)
        lam = float(diagnostic["lambda_prior_mean"])
        omega = float(sp["omega_prior"][0]) / sum(map(float, sp["omega_prior"]))
        qref = experiment.mean_q(cell); tau_scale = 2.0 * (float(sp["tau2_scale_fraction"]) * robust_scale(qref)) ** 2
        tau2 = tau_scale / (float(sp["tau2_prior_shape"]) - 1.0)
        damping = float(sp["cycle_damping_initial"]); period = float(sp["cycle_period_initial"])
        save = 0
        for it in range(iterations):
            q = experiment.draw_q(rng, cell)
            zero = np.zeros(T); one = np.ones(T)
            constant = _mu(spec, cell, q, zero, beta, lam)
            loading = _mu(spec, cell, q, one, beta, lam) - constant
            h = _sample_h(rng, q, damping, period, tau2, omega, cell.pi, constant, loading, sigma2, phi)
            bar, damping, period, omega, tau2, ok_d, ok_p, ok_w = _update_state_params(
                rng, q, h, damping, period, omega, tau2, cell.s_q, config,
            )
            damping_acc[ch] += ok_d; period_acc[ch] += ok_p; omega_acc[ch] += ok_w
            X = _design(spec, cell, q, h, lam)
            beta, sigma2 = _draw_coefficients(rng, cell.pi, X, phi, pmean, psd, sigma2,
                                              priors.ig_shape, priors.sigma_pi_b)
            if spec.free_lambda:
                lam = _draw_lambda_diagnostic(
                    rng, cell, q, h, beta, phi, sigma2,
                    float(diagnostic["lambda_prior_mean"]), float(diagnostic["lambda_prior_sd"]),
                )
            mu = _mu(spec, cell, q, h, beta, lam)
            phi, ok = _draw_phi(rng, cell.pi, mu, sigma2, phi,
                                float(ep["phi_prior_mean"]), float(ep["phi_prior_sd"]))
            phi_acc[ch] += ok
            if it >= warmup and (it - warmup) % thin == 0:
                draws[ch, save] = list(beta) + ([lam] if spec.free_lambda else [])
                sig[ch, save] = np.sqrt(sigma2); phis[ch, save] = phi
                nt[ch, save] = q; nb[ch, save] = bar; nh[ch, save] = h
                om[ch, save] = omega; taus[ch, save] = np.sqrt(tau2)
                damping_out[ch, save] = damping; period_out[ch, save] = period; save += 1
    diagnostics = _model_diagnostics(names, draws, sig, phis, nt, nb, nh, om, taus, damping_out, period_out)
    diagnostics["phi_acceptance"] = (phi_acc / iterations).tolist()
    diagnostics["omega_acceptance"] = (omega_acc / iterations).tolist()
    diagnostics["cycle_damping_acceptance"] = (damping_acc / iterations).tolist()
    diagnostics["cycle_period_acceptance"] = (period_acc / iterations).tolist()
    prior_mean = {n: (float(diagnostic["lambda_prior_mean"]) if n == "lambda" else float(means[n])) for n in names}
    prior_sd = {n: (float(diagnostic["lambda_prior_sd"]) if n == "lambda" else float(sds[n])) for n in names}
    return ModelFit(spec, tuple(map(str, cell.periods)), names, draws, sig, phis, nt, nb, nh,
                    om, taus, damping_out, period_out, prior_mean, prior_sd, diagnostics)


def _model_diagnostics(names, draws, sig, phi, nt, nb, nh, om, tau, damping, period):
    rhat = {n: float(az.rhat(draws[:, :, i], method="rank")) for i, n in enumerate(names)}
    rhat.update(sigma_pi=float(az.rhat(sig, method="rank")), phi=float(az.rhat(phi, method="rank")),
                omega=float(az.rhat(om, method="rank")), tau=float(az.rhat(tau, method="rank")),
                cycle_damping=float(az.rhat(damping, method="rank")),
                cycle_period=float(az.rhat(period, method="rank")))
    ess_bulk = {n: float(az.ess(draws[:, :, i], method="bulk")) for i, n in enumerate(names)}
    ess_tail = {n: float(az.ess(draws[:, :, i], method="tail", prob=(0.05, 0.95))) for i, n in enumerate(names)}
    return {"rhat": rhat, "max_rhat": max(rhat.values()), "ess_bulk": ess_bulk, "ess_tail": ess_tail,
            "exact_identity_error": float(np.max(np.abs(nt - nb - nh)))}


def pointwise_loglik(fit: ModelFit, cell: CellData, max_draws: int = 1200) -> np.ndarray:
    C, D = fit.draws.shape[:2]; total = C * D
    take = np.linspace(0, total - 1, min(total, max_draws)).astype(int)
    coeff = fit.draws.reshape(total, -1); sig = fit.sigma_pi.reshape(total)
    phi = fit.phi.reshape(total); q = fit.n_total.reshape(total, cell.n_periods); h = fit.nhat.reshape(total, cell.n_periods)
    out = np.zeros((len(take), cell.n_periods))
    for row, j in enumerate(take):
        beta = coeff[j, :len(fit.spec.coefficient_names)]
        lam = coeff[j, fit.names.index("lambda")] if fit.spec.free_lambda else fit.spec.lambda_fixed
        mu = _mu(fit.spec, cell, q[j], h[j], beta, lam)
        e = cell.pi - mu; innovation = np.empty_like(e)
        innovation[0] = np.sqrt(max(1e-8, 1.0 - phi[j] ** 2)) * e[0]
        innovation[1:] = e[1:] - phi[j] * e[:-1]
        out[row] = -0.5 * np.log(2 * np.pi * sig[j] ** 2) - 0.5 * innovation**2 / sig[j] ** 2
        # Jacobian of the stationary AR(1) first-observation transform.  The
        # sampler already includes this term in _inflation_loglik; saved-draw
        # predictive metrics must use the identical density.
        out[row, 0] += 0.5 * np.log(max(1e-8, 1.0 - phi[j] ** 2))
    return out


def comparison_metrics(fit: ModelFit, cell: CellData):
    ll = pointwise_loglik(fit, cell)
    mx = np.max(ll, axis=0); lppd_i = mx + np.log(np.mean(np.exp(ll - mx), axis=0))
    pwaic_i = np.var(ll, axis=0, ddof=1); waic = -2.0 * float(np.sum(lppd_i - pwaic_i))
    coeff = fit.draws.reshape(-1, fit.draws.shape[-1]); q = fit.n_total.reshape(-1, cell.n_periods); h = fit.nhat.reshape(-1, cell.n_periods)
    idx = np.linspace(0, len(coeff) - 1, min(1000, len(coeff))).astype(int); mus = []
    for j in idx:
        lam = coeff[j, fit.names.index("lambda")] if fit.spec.free_lambda else fit.spec.lambda_fixed
        mus.append(_mu(fit.spec, cell, q[j], h[j], coeff[j, :len(fit.spec.coefficient_names)], lam))
    pred = np.mean(mus, axis=0)
    # Four-quarter block log likelihood for later PSIS/refit processing.
    nblock = int(np.ceil(cell.n_periods / 4)); block_ll = np.zeros((ll.shape[0], nblock))
    for b in range(nblock): block_ll[:, b] = ll[:, 4*b:min(4*(b+1), cell.n_periods)].sum(axis=1)
    return {"waic_secondary": waic, "lppd": float(np.sum(lppd_i)),
            "predictive_rmse_in_sample": float(np.sqrt(np.mean((cell.pi - pred) ** 2))),
            "block_loglik": block_ll}


def summarize_fit(fit: ModelFit):
    flat = fit.draws.reshape(-1, fit.draws.shape[-1]); out = {}
    for i, name in enumerate(fit.names):
        v = flat[:, i]
        out[name] = {"mean": float(v.mean()), "sd": float(v.std(ddof=1)),
                     "q2.5": float(np.percentile(v, 2.5)), "q97.5": float(np.percentile(v, 97.5)),
                     "p_positive": float(np.mean(v > 0)), "prior_mean": fit.prior_mean[name],
                     "prior_sd": fit.prior_sd[name], "rhat": fit.diagnostics["rhat"][name],
                     "ess_bulk": fit.diagnostics["ess_bulk"][name], "ess_tail": fit.diagnostics["ess_tail"][name]}
    return out


def restriction_diagnostics(fit: ModelFit, cell: CellData, lambda_grid: list[float], tolerance: float):
    if fit.spec.model_id != "free_static_combined":
        raise ValueError("restriction diagnostics require free_static_combined")
    flat = fit.draws.reshape(-1, fit.draws.shape[-1]); delta_s = flat[:, fit.names.index("delta_s")]; theta = flat[:, fit.names.index("theta")]
    bar = fit.nbar.reshape(-1, cell.n_periods); x = cell.x[None, :]; out = {}
    for lam in lambda_grid:
        r = delta_s - float(lam) * theta; impact = r[:, None] * bar * x
        rms = np.sqrt(np.mean(impact**2, axis=1))
        out[f"{float(lam):g}"] = {
            "r_mean": float(r.mean()), "r_q2.5": float(np.percentile(r, 2.5)),
            "r_q97.5": float(np.percentile(r, 97.5)), "p_r_positive": float(np.mean(r > 0)),
            "rms_mean": float(rms.mean()), "equivalence_probability": float(np.mean(rms < tolerance)),
        }
    return out


def save_fit(path: Path, fit: ModelFit):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, model_id=fit.spec.model_id, role=fit.spec.role,
                        lambda_fixed=np.nan if fit.spec.lambda_fixed is None else fit.spec.lambda_fixed,
                        free_lambda=fit.spec.free_lambda, periods=fit.periods, names=fit.names,
                        draws=fit.draws, sigma_pi=fit.sigma_pi, phi=fit.phi,
                        n_total=fit.n_total, nbar=fit.nbar, nhat=fit.nhat,
                        omega=fit.omega, tau=fit.tau,
                        cycle_damping=fit.cycle_damping, cycle_period=fit.cycle_period)


def load_fit(path: Path, diagnostics: dict, prior_mean: dict, prior_sd: dict) -> ModelFit:
    z = np.load(path, allow_pickle=False); lf = float(z["lambda_fixed"])
    spec = ModelSpec(str(z["model_id"]), str(z["role"]), tuple(map(str, z["names"])),
                     None if np.isnan(lf) else lf, bool(z["free_lambda"]))
    # coefficient_names excludes the separately stored diagnostic lambda.
    if spec.free_lambda:
        spec = ModelSpec(spec.model_id, spec.role, spec.coefficient_names[:-1], None, True)
    return ModelFit(spec, tuple(map(str, z["periods"])), tuple(map(str, z["names"])), z["draws"],
                    z["sigma_pi"], z["phi"], z["n_total"], z["nbar"], z["nhat"],
                    z["omega"], z["tau"], z["cycle_damping"], z["cycle_period"],
                    prior_mean, prior_sd, diagnostics)
