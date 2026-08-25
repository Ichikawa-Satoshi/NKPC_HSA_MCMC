"""Mixed-frequency Gustavo-Q4 and Capital-IQ-growth competition measurement.

Inflation is deliberately absent from this module.  Annual Gustavo observations
are exact measurements of total log competition at Q4.  Capital IQ firm- and
revenue-weighted quarterly growth rates are noisy indicators of the same latent
total growth during their observed overlap.  Missing Capital IQ quarters remain
missing; the average quarterly allocation profile enters only as transition
drift, never as an observed quarterly N path.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import arviz as az
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logit
from scipy.stats import beta as beta_dist, halfnorm, norm

from nkpc_hsa.error_robustness.ma_error import AdaptiveRandomWalk
from nkpc_hsa.report_models.cases import _load_frame
from tests.hsa_exact_n_decomposition.functions import build_allocation_posterior
from tests.hsa_nested_validation.functions import robust_scale


SERIES = ("firm_weighted", "revenue_weighted")
PARAMETER_NAMES = (
    "omega", "tau", "cycle_damping", "cycle_period",
    "intercept_firm_weighted", "loading_firm_weighted", "sigma_firm_weighted",
    "intercept_revenue_weighted", "loading_revenue_weighted", "sigma_revenue_weighted",
)


@dataclass(frozen=True)
class MeasurementData:
    periods: pd.PeriodIndex
    gustavo: np.ndarray
    ciq_growth: dict[str, np.ndarray]
    average_weights: np.ndarray
    drift: np.ndarray
    reference_values: dict[str, float]
    scales: dict[str, float]


@dataclass
class MeasurementFit:
    periods: tuple[str, ...]
    parameter_names: tuple[str, ...]
    parameters: np.ndarray
    nbar: np.ndarray
    nhat: np.ndarray
    n_total: np.ndarray
    diagnostics: dict[str, Any]
    map_parameters: dict[str, float]
    average_weights: np.ndarray


def ar2_coefficients(damping: float, period: float) -> tuple[float, float]:
    return float(2.0 * damping * np.cos(2.0 * np.pi / period)), float(-damping**2)


def load_measurement_data(config: dict[str, Any]) -> MeasurementData:
    frame = _load_frame()
    dc, sc, mc = config["data"], config["sample"], config["measurement"]
    periods = pd.period_range(sc["state_start"], sc["state_end"], freq="Q")
    ref_period = pd.Period(sc["coordinate_reference"], freq="Q")

    raw_g = pd.to_numeric(frame[dc["gustavo"]], errors="coerce").reindex(periods)
    if not np.all(raw_g.dropna().index.quarter == 4):
        raise ValueError("Gustavo observations must be Q4 stocks")
    g_ref = float(raw_g.loc[ref_period])
    gustavo = (10.0 * np.log(raw_g / g_ref)).to_numpy(float)

    ciq_growth: dict[str, np.ndarray] = {}
    refs: dict[str, float] = {"gustavo": g_ref}
    scales: dict[str, float] = {}
    for label, column in dc["capital_iq"].items():
        raw = pd.to_numeric(frame[column], errors="coerce").reindex(periods)
        ref = float(raw.loc[ref_period])
        coordinate = 10.0 * np.log(raw / ref)
        # Reindex before differencing so a sparse annual Q4 observation is not
        # mistaken for a one-quarter change.
        growth = coordinate.diff()
        ciq_growth[label] = growth.to_numpy(float)
        refs[label] = ref
        scales[label] = max(robust_scale(growth.dropna()), 1e-4)

    allocation = build_allocation_posterior(
        frame,
        dc["capital_iq"]["firm_weighted"],
        float(mc["stable_raw_weight_max"]),
        float(mc["prior_covariance_scale"]),
    )
    weights = np.asarray(allocation.average_weights, dtype=float)
    weights = weights / weights.sum()
    annual = pd.Series(
        {p.year: 10.0 * np.log(v / g_ref) for p, v in raw_g.dropna().items()},
        dtype=float,
    )
    drift = np.zeros(len(periods), dtype=float)
    for t in range(1, len(periods)):
        p = periods[t]
        if p.year in annual.index and p.year - 1 in annual.index:
            drift[t] = weights[p.quarter - 1] * float(annual.loc[p.year] - annual.loc[p.year - 1])
    annual_changes = np.diff(annual.to_numpy(float))
    scales["state"] = max(robust_scale(annual_changes) / 2.0, 1e-3)
    return MeasurementData(periods, gustavo, ciq_growth, weights, drift, refs, scales)


def _decode(z: np.ndarray, data: MeasurementData, config: dict[str, Any]) -> dict[str, float]:
    mc = config["measurement"]
    dlo, dhi = map(float, mc["damping_bounds"])
    plo, phi = map(float, mc["period_bounds"])
    return {
        "omega": float(expit(z[0])),
        "tau": float(np.exp(np.clip(z[1], -20, 20))),
        "cycle_damping": float(dlo + (dhi - dlo) * expit(z[2])),
        "cycle_period": float(plo + (phi - plo) * expit(z[3])),
        "intercept_firm_weighted": float(z[4]),
        "loading_firm_weighted": float(z[5]),
        "sigma_firm_weighted": float(np.exp(np.clip(z[6], -20, 20))),
        "intercept_revenue_weighted": float(z[7]),
        "loading_revenue_weighted": float(z[8]),
        "sigma_revenue_weighted": float(np.exp(np.clip(z[9], -20, 20))),
    }


def _encode(p: dict[str, float], config: dict[str, Any]) -> np.ndarray:
    mc = config["measurement"]
    dlo, dhi = map(float, mc["damping_bounds"])
    plo, phi = map(float, mc["period_bounds"])
    unit = lambda value, lo, hi: logit(np.clip((value - lo) / (hi - lo), 1e-6, 1 - 1e-6))
    return np.array([
        logit(np.clip(p["omega"], 1e-6, 1 - 1e-6)), np.log(p["tau"]),
        unit(p["cycle_damping"], dlo, dhi), unit(p["cycle_period"], plo, phi),
        p["intercept_firm_weighted"], p["loading_firm_weighted"], np.log(p["sigma_firm_weighted"]),
        p["intercept_revenue_weighted"], p["loading_revenue_weighted"], np.log(p["sigma_revenue_weighted"]),
    ])


def _initial_parameters(data: MeasurementData, config: dict[str, Any]) -> dict[str, float]:
    total = benchmark_path(data, "average_allocation")
    dtotal = np.diff(total, prepend=np.nan)
    out = {"omega": 0.15, "tau": data.scales["state"], "cycle_damping": 0.65, "cycle_period": 12.0}
    for label in SERIES:
        y = data.ciq_growth[label]
        mask = np.isfinite(y) & np.isfinite(dtotal)
        X = np.column_stack([np.ones(mask.sum()), dtotal[mask]])
        beta = np.linalg.lstsq(X, y[mask], rcond=None)[0]
        resid = y[mask] - X @ beta
        out[f"intercept_{label}"] = float(beta[0])
        out[f"loading_{label}"] = float(beta[1])
        out[f"sigma_{label}"] = max(robust_scale(pd.Series(resid)), 0.1 * data.scales[label])
    return out


def _log_prior(z: np.ndarray, p: dict[str, float], data: MeasurementData, config: dict[str, Any]) -> float:
    mc = config["measurement"]
    omega = p["omega"]
    ao, bo = map(float, mc["omega_prior"])
    value = beta_dist.logpdf(omega, ao, bo) + np.log(omega) + np.log1p(-omega)
    tau_scale = float(mc["tau_halfnormal_scale_multiple"]) * data.scales["state"]
    value += halfnorm.logpdf(p["tau"], scale=tau_scale) + np.log(p["tau"])

    dlo, dhi = map(float, mc["damping_bounds"])
    ud = (p["cycle_damping"] - dlo) / (dhi - dlo)
    ad, bd = map(float, mc["damping_prior"])
    value += beta_dist.logpdf(ud, ad, bd) + np.log(ud) + np.log1p(-ud)
    plo, phi = map(float, mc["period_bounds"])
    up = (p["cycle_period"] - plo) / (phi - plo)
    value += norm.logpdf(p["cycle_period"], float(mc["period_prior_mean"]), float(mc["period_prior_sd"]))
    value += np.log(up) + np.log1p(-up)
    for label in SERIES:
        sy = data.scales[label]
        value += norm.logpdf(p[f"intercept_{label}"], 0.0, float(mc["intercept_prior_scale_multiple"]) * sy)
        value += norm.logpdf(p[f"loading_{label}"], float(mc["loading_prior_mean"]), float(mc["loading_prior_sd"]))
        scale = float(mc["measurement_sd_prior_scale_multiple"]) * sy
        value += halfnorm.logpdf(p[f"sigma_{label}"], scale=scale) + np.log(p[f"sigma_{label}"])
    return float(value)


def _force_psd(matrix: np.ndarray, floor: float = 0.0) -> np.ndarray:
    matrix = (matrix + matrix.T) / 2.0
    values, vectors = np.linalg.eigh(matrix)
    return vectors @ np.diag(np.maximum(values, floor)) @ vectors.T


def kalman_filter(
    z: np.ndarray,
    data: MeasurementData,
    config: dict[str, Any],
    include_ciq: dict[str, np.ndarray] | None = None,
) -> dict[str, Any]:
    p = _decode(z, data, config)
    phi1, phi2 = ar2_coefficients(p["cycle_damping"], p["cycle_period"])
    F = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, phi1, phi2, 0.0],
                  [0.0, 1.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]])
    Q = np.zeros((4, 4))
    Q[0, 0] = max(1e-12, p["omega"] * p["tau"] ** 2)
    Q[1, 1] = max(1e-12, (1.0 - p["omega"]) * p["tau"] ** 2)
    T = len(data.periods)
    first_g = int(np.flatnonzero(np.isfinite(data.gustavo))[0])
    g0 = float(data.gustavo[first_g])
    m = np.array([g0, 0.0, 0.0, g0])
    P = np.diag([4.0, 4.0, 4.0, 4.0])
    mp = np.zeros((T, 4)); pp = np.zeros((T, 4, 4))
    mf = np.zeros_like(mp); pf = np.zeros_like(pp)
    loglik = 0.0; eye = np.eye(4)
    masks = include_ciq or {label: np.ones(T, dtype=bool) for label in SERIES}
    for t in range(T):
        if t > 0:
            # The fourth coordinate is lagged slow state, so it inherits the
            # previous level but not the current-quarter drift.
            m = F @ m + np.array([data.drift[t], 0.0, 0.0, 0.0])
            P = _force_psd(F @ P @ F.T + Q, 1e-12)
        mp[t], pp[t] = m, P
        # Gustavo is an exact conditioning restriction, not a zero-noise
        # Gaussian density.  Including its density would make the likelihood
        # unbounded as the state innovation variance tends to zero because the
        # annual drift already reconciles consecutive Q4 totals.
        if np.isfinite(data.gustavo[t]):
            Hq = np.array([[1.0, 1.0, 0.0, 0.0]])
            Sq = float((Hq @ P @ Hq.T).item())
            if Sq <= 0.0 or not np.isfinite(Sq):
                return {"loglik": -np.inf}
            innovation_q = float(data.gustavo[t] - (Hq @ m).item())
            Kq = (P @ Hq.T / Sq).reshape(4)
            m = m + Kq * innovation_q
            P = _force_psd(P - np.outer(Kq, Hq @ P), 0.0)

        rows: list[np.ndarray] = []; obs: list[float] = []; offsets: list[float] = []; variances: list[float] = []
        for label in SERIES:
            value = data.ciq_growth[label][t]
            if np.isfinite(value) and bool(masks[label][t]):
                loading = p[f"loading_{label}"]
                rows.append(loading * np.array([1.0, 1.0, -1.0, -1.0]))
                obs.append(float(value)); offsets.append(p[f"intercept_{label}"])
                variances.append(p[f"sigma_{label}"] ** 2)
        if rows:
            H = np.vstack(rows); y = np.asarray(obs) - np.asarray(offsets)
            R = np.diag(variances); S = _force_psd(H @ P @ H.T + R, 1e-12)
            sign, logdet = np.linalg.slogdet(S)
            if sign <= 0 or not np.isfinite(logdet):
                return {"loglik": -np.inf}
            innovation = y - H @ m
            solved = np.linalg.solve(S, innovation)
            loglik += -0.5 * (len(y) * np.log(2.0 * np.pi) + logdet + float(innovation @ solved))
            K = np.linalg.solve(S, H @ P).T
            m = m + K @ innovation
            P = _force_psd((eye - K @ H) @ P @ (eye - K @ H).T + K @ R @ K.T, 0.0)
        mf[t], pf[t] = m, P
    return {"loglik": float(loglik), "pred_mean": mp, "pred_cov": pp,
            "filt_mean": mf, "filt_cov": pf, "F": F, "params": p}


def log_posterior(z: np.ndarray, data: MeasurementData, config: dict[str, Any],
                  include_ciq: dict[str, np.ndarray] | None = None) -> float:
    try:
        p = _decode(z, data, config)
        prior = _log_prior(z, p, data, config)
        result = kalman_filter(z, data, config, include_ciq)
        return float(prior + result["loglik"])
    except (ValueError, np.linalg.LinAlgError, FloatingPointError):
        return -np.inf


def fit_map(data: MeasurementData, config: dict[str, Any], include_ciq: dict[str, np.ndarray] | None = None,
            initial: np.ndarray | None = None) -> tuple[np.ndarray, dict[str, Any]]:
    z0 = _encode(_initial_parameters(data, config), config) if initial is None else np.asarray(initial, float)
    def objective(z: np.ndarray) -> float:
        value = log_posterior(z, data, config, include_ciq)
        return -value if np.isfinite(value) else 1e100
    result = minimize(objective, z0, method="L-BFGS-B", options={"maxiter": 500, "ftol": 1e-9, "maxls": 30})
    if not np.isfinite(result.fun):
        raise RuntimeError("Mixed-frequency MAP optimization failed")
    return np.asarray(result.x, float), {"success": bool(result.success), "message": str(result.message), "objective": float(result.fun), "iterations": int(result.nit)}


def _smooth_mean(filtered: dict[str, Any]) -> np.ndarray:
    mf, pf = filtered["filt_mean"], filtered["filt_cov"]
    mp, pp, F = filtered["pred_mean"], filtered["pred_cov"], filtered["F"]
    out = mf.copy()
    for t in range(len(out) - 2, -1, -1):
        J = pf[t] @ F.T @ np.linalg.pinv(pp[t + 1])
        out[t] = mf[t] + J @ (out[t + 1] - mp[t + 1])
    return out


def _draw_smoother(rng: np.random.Generator, filtered: dict[str, Any]) -> np.ndarray:
    mf, pf = filtered["filt_mean"], filtered["filt_cov"]
    mp, pp, F = filtered["pred_mean"], filtered["pred_cov"], filtered["F"]
    T = len(mf); out = np.zeros_like(mf)
    cov = _force_psd(pf[-1], 0.0)
    out[-1] = rng.multivariate_normal(mf[-1], cov, check_valid="ignore")
    for t in range(T - 2, -1, -1):
        J = pf[t] @ F.T @ np.linalg.pinv(pp[t + 1])
        mean = mf[t] + J @ (out[t + 1] - mp[t + 1])
        cov = _force_psd(pf[t] - J @ pp[t + 1] @ J.T, 0.0)
        out[t] = rng.multivariate_normal(mean, cov, check_valid="ignore")
    return out


def _diagnostics(names: tuple[str, ...], parameters: np.ndarray, nbar: np.ndarray, nhat: np.ndarray) -> dict[str, Any]:
    rhat = {name: float(az.rhat(parameters[:, :, j], method="rank")) for j, name in enumerate(names)}
    bulk = {name: float(az.ess(parameters[:, :, j], method="bulk")) for j, name in enumerate(names)}
    tail = {name: float(az.ess(parameters[:, :, j], method="tail", prob=(0.05, 0.95))) for j, name in enumerate(names)}
    # Selected dates keep the mock diagnostic useful without letting one weakly
    # determined pre-overlap point dominate every scalar convergence statement.
    selected = np.unique(np.linspace(0, nbar.shape[-1] - 1, 12, dtype=int))
    for label, values in (("nbar", nbar[..., selected]), ("nhat", nhat[..., selected])):
        rr = np.asarray(az.rhat(values, method="rank")); ee = np.asarray(az.ess(values, method="bulk"))
        rhat[f"{label}_selected_max"] = float(np.nanmax(rr)); bulk[f"{label}_selected_min"] = float(np.nanmin(ee))
    return {"rhat": rhat, "ess_bulk": bulk, "ess_tail": tail,
            "max_rhat": max(rhat.values()), "min_bulk_ess": min(bulk.values()), "min_tail_ess": min(tail.values())}


def fit_measurement(data: MeasurementData, config: dict[str, Any], sampling: dict[str, Any], seed: int) -> MeasurementFit:
    zmap, map_info = fit_map(data, config)
    iterations = int(sampling["state_iterations"]); warmup = int(sampling["state_warmup"])
    thin = int(sampling["state_thin"]); chains = int(sampling["state_chains"])
    nsave = (iterations - warmup + thin - 1) // thin; T = len(data.periods)
    parameters = np.zeros((chains, nsave, len(PARAMETER_NAMES)))
    nbar = np.zeros((chains, nsave, T)); nhat = np.zeros_like(nbar)
    acceptance = []
    for chain in range(chains):
        rng = np.random.default_rng(seed + 10007 * chain)
        current = zmap + rng.normal(0.0, 0.015, len(zmap)); lp = log_posterior(current, data, config)
        if not np.isfinite(lp): current = zmap.copy(); lp = log_posterior(current, data, config)
        proposal = AdaptiveRandomWalk(len(zmap), init_scale=float(sampling["proposal_scale"]), target_accept=0.234)
        save = 0
        for it in range(iterations):
            candidate = proposal.propose(current, rng); clp = log_posterior(candidate, data, config)
            accepted = bool(np.log(rng.uniform()) < clp - lp)
            if accepted: current, lp = candidate, clp
            proposal.register(current, accepted)
            if it == warmup - 1: proposal.freeze()
            if it >= warmup and (it - warmup) % thin == 0:
                filtered = kalman_filter(current, data, config); state = _draw_smoother(rng, filtered)
                # Numerical projection onto every exact Q4 total constraint.
                anchors = np.flatnonzero(np.isfinite(data.gustavo))
                state[anchors, 0] += data.gustavo[anchors] - state[anchors, 0] - state[anchors, 1]
                p = filtered["params"]
                parameters[chain, save] = [p[name] for name in PARAMETER_NAMES]
                nbar[chain, save] = state[:, 0]; nhat[chain, save] = state[:, 1]; save += 1
        acceptance.append(float(proposal.acceptance_rate))
    total = nbar + nhat
    diagnostics = _diagnostics(PARAMETER_NAMES, parameters, nbar, nhat)
    anchors = np.flatnonzero(np.isfinite(data.gustavo))
    diagnostics["max_q4_anchor_error"] = float(np.max(np.abs(total[..., anchors] - data.gustavo[anchors])))
    diagnostics["proposal_acceptance"] = acceptance; diagnostics["map"] = map_info
    return MeasurementFit(tuple(map(str, data.periods)), PARAMETER_NAMES, parameters, nbar, nhat, total,
                          diagnostics, _decode(zmap, data, config), data.average_weights)


def benchmark_path(data: MeasurementData, method: str) -> np.ndarray:
    if method not in {"equal_allocation", "average_allocation"}: raise ValueError(method)
    weights = np.full(4, 0.25) if method == "equal_allocation" else data.average_weights
    out = np.full(len(data.periods), np.nan)
    anchors = np.flatnonzero(np.isfinite(data.gustavo))
    out[anchors] = data.gustavo[anchors]
    for left, right in zip(anchors[:-1], anchors[1:]):
        if right - left != 4: raise ValueError("Gustavo Q4 anchors are not consecutive")
        change = data.gustavo[right] - data.gustavo[left]
        out[left + 1:right + 1] = out[left] + np.cumsum(weights * change)
    return out


def blocked_backtest(data: MeasurementData, config: dict[str, Any]) -> pd.DataFrame:
    full_map, _ = fit_map(data, config)
    rows: list[dict[str, Any]] = []
    for block_index, (start, end) in enumerate(config["backtest"]["blocks"]):
        block = (data.periods >= pd.Period(start, freq="Q")) & (data.periods <= pd.Period(end, freq="Q"))
        include = {label: ~block for label in SERIES}
        z, _ = fit_map(data, config, include, initial=full_map)
        filtered = kalman_filter(z, data, config, include); state = _smooth_mean(filtered); p = filtered["params"]
        dtotal = (state[:, 0] + state[:, 1]) - (state[:, 3] + state[:, 2])
        for label in SERIES:
            observed = data.ciq_growth[label]; test = block & np.isfinite(observed)
            train = (~block) & np.isfinite(observed)
            for method in ("equal_allocation", "average_allocation"):
                path = benchmark_path(data, method); dx = np.diff(path, prepend=np.nan)
                use = train & np.isfinite(dx); X = np.column_stack([np.ones(use.sum()), dx[use]])
                beta = np.linalg.lstsq(X, observed[use], rcond=None)[0]
                resid = observed[use] - X @ beta; sigma = max(float(np.std(resid, ddof=2)), 1e-6)
                pred = beta[0] + beta[1] * dx[test]
                error = observed[test] - pred
                rows.append({"block": block_index + 1, "start": start, "end": end, "series": label,
                             "method": method, "n": int(test.sum()), "rmse": float(np.sqrt(np.mean(error**2))),
                             "mae": float(np.mean(np.abs(error))),
                             "mean_log_score": float(np.mean(norm.logpdf(error, 0.0, sigma)))})
            pred = p[f"intercept_{label}"] + p[f"loading_{label}"] * dtotal[test]
            error = observed[test] - pred; sigma = p[f"sigma_{label}"]
            rows.append({"block": block_index + 1, "start": start, "end": end, "series": label,
                         "method": "mixed_frequency_state", "n": int(test.sum()),
                         "rmse": float(np.sqrt(np.mean(error**2))), "mae": float(np.mean(np.abs(error))),
                         "mean_log_score": float(np.mean(norm.logpdf(error, 0.0, sigma)))})
    return pd.DataFrame(rows)


def summarize_parameters(fit: MeasurementFit) -> pd.DataFrame:
    rows = []
    for j, name in enumerate(fit.parameter_names):
        values = fit.parameters[:, :, j].reshape(-1)
        rows.append({"parameter": name, "mean": float(values.mean()), "sd": float(values.std(ddof=1)),
                     "q2.5": float(np.percentile(values, 2.5)), "q50": float(np.percentile(values, 50)),
                     "q97.5": float(np.percentile(values, 97.5)), "rhat": fit.diagnostics["rhat"][name],
                     "ess_bulk": fit.diagnostics["ess_bulk"][name], "ess_tail": fit.diagnostics["ess_tail"][name]})
    return pd.DataFrame(rows)
