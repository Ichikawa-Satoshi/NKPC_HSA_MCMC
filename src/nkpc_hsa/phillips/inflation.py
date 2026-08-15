from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
import pandas as pd

from nkpc_hsa.gibbs.common.joint_ffbs import force_pd

from .data import CELL_SPECS, DesignData, robust_scale
from .state import MeasurementPosterior
from .temporal import whiten_aggregated


MODELS = ("E0", "E1", "E2", "SLOW")
TRANSFORMATIONS = ("qoq", "qoq_matched", "yoy")


@dataclass(frozen=True)
class CellFit:
    cell: int
    inflation: str
    activity: str
    model: str
    transformation: str
    coefficient_names: tuple[str, ...]
    coefficients: np.ndarray
    sigma: np.ndarray
    q0: float
    x_scale: float
    prior_sds: dict[str, float]
    n_endpoints: int
    expectation_status: str
    estimator: str = "modular_cut"
    auxiliary_draws: dict[str, np.ndarray] = field(default_factory=dict)


@dataclass(frozen=True)
class BenchmarkResult:
    status: str
    reason: str
    fit: CellFit | None
    delta_hsa: np.ndarray | None
    theta_reference: np.ndarray | None
    equivalence_band: float
    equivalence_probability: float | None
    local_probability: np.ndarray
    local_endpoints: np.ndarray
    q_reference_draws: np.ndarray
    zeta_reference: float
    b_x: float


def coefficient_names(
    model: str,
    *,
    single_coordinate: bool = False,
    no_lag: bool = False,
    include_slow_level: bool = True,
) -> tuple[str, ...]:
    if model not in MODELS:
        raise ValueError(f"Unknown empirical hierarchy model {model!r}.")
    common = ["a"]
    if not no_lag:
        common.append("beta_b")
    common.append("beta_f")
    if include_slow_level:
        common.append("psi")
    common.append("kappa_0")
    if model == "SLOW":
        common.append("kappa_1")
        return tuple(common)
    if model in {"E1", "E2"}:
        common.extend(("kappa_1", "theta_0"))
    if model == "E2":
        common.append("gamma")
    return tuple(common)


def _prior_sds(names: Iterable[str], *, q_scale: float, x_scale: float) -> dict[str, float]:
    all_sds = {
        "a": 5.0,
        "beta_b": 1.0,
        "beta_f": 1.0,
        "psi": 1.5 / q_scale,
        "kappa_0": 2.0 / x_scale,
        "kappa_1": 1.0 / (q_scale * x_scale),
        "theta_0": 1.0 / q_scale,
        "gamma": 1.0 / q_scale**2,
        "lambda_qx": 1.0 / (q_scale * x_scale),
        "lambda_u": 1.0 / q_scale,
        "lambda_eta": 1.0 / q_scale,
    }
    return {name: all_sds[name] for name in names}


def _quarterly_design(
    *,
    pi_lag: np.ndarray,
    expectation: np.ndarray,
    x: np.ndarray,
    qbar: np.ndarray,
    qhat: np.ndarray,
    q0: float,
    model: str,
    single_coordinate: bool = False,
    no_lag: bool = False,
    include_slow_level: bool = True,
    extra: str | None = None,
    d_q: float = 0.0,
    phi_q: float = 0.0,
) -> tuple[np.ndarray, tuple[str, ...]]:
    centered = qbar - q0
    columns: list[np.ndarray] = [np.ones_like(x)]
    names = list(
        coefficient_names(
            model,
            single_coordinate=single_coordinate,
            no_lag=no_lag,
            include_slow_level=include_slow_level,
        )
    )
    if not no_lag:
        columns.append(pi_lag)
    columns.append(expectation)
    if include_slow_level:
        columns.append(centered)
    columns.append(x)
    if model == "SLOW":
        columns.append(centered * x)
    elif model in {"E1", "E2"}:
        slope_coordinate = centered + qhat if single_coordinate else centered
        columns.extend((slope_coordinate * x, -qhat))
    if model == "E2":
        columns.append(-centered * qhat)
    if extra == "fast_by_fast":
        columns.append(qhat * x)
        names.append("lambda_qx")
    elif extra in {"cs1", "cs2"}:
        u = np.r_[0.0, qhat[1:] - phi_q * qhat[:-1]]
        columns.append(u)
        names.append("lambda_u")
        if extra == "cs2":
            eta = np.r_[0.0, qbar[1:] - d_q - qbar[:-1]]
            columns.append(eta)
            names.append("lambda_eta")
    return np.column_stack(columns), tuple(names)


def _ar1_whiten(y: np.ndarray, X: np.ndarray, rho: float) -> tuple[np.ndarray, np.ndarray]:
    """Apply the exact stationary AR(1) innovation transform."""
    if not (-1.0 < rho < 1.0):
        raise ValueError("Persistent-error rho must be in (-1, 1).")
    scale0 = np.sqrt(max(1.0 - rho**2, 1e-12))
    yw = np.empty_like(y)
    Xw = np.empty_like(X)
    yw[0] = scale0 * y[0]
    Xw[0] = scale0 * X[0]
    yw[1:] = y[1:] - rho * y[:-1]
    Xw[1:] = X[1:] - rho * X[:-1]
    return yw, Xw


def _rho_log_posterior(rho: float, residual: np.ndarray, sigma2: float) -> float:
    if not (-1.0 < rho < 1.0):
        return -np.inf
    innovations = residual[1:] - rho * residual[:-1]
    innovation_ss = (1.0 - rho**2) * residual[0] ** 2 + float(
        np.dot(innovations, innovations)
    )
    prior = -0.5 * ((rho - 0.5) / 0.3) ** 2
    return 0.5 * np.log1p(-rho**2) - innovation_ss / (2.0 * sigma2) + prior


def _mh_rho(
    rng: np.random.Generator,
    current: float,
    residual: np.ndarray,
    sigma2: float,
    *,
    steps: int = 2,
) -> float:
    for _ in range(steps):
        proposal = float(current + rng.normal(0.0, 0.08))
        log_ratio = _rho_log_posterior(proposal, residual, sigma2) - _rho_log_posterior(
            current, residual, sigma2
        )
        if np.log(rng.uniform()) < log_ratio:
            current = proposal
    return current


def _low_frequency_draw(
    rng: np.random.Generator,
    residual: np.ndarray,
    *,
    rho: float,
    state_variance: float,
    observation_variance: float,
) -> np.ndarray:
    """Scalar FFBS draw for the declared low-frequency inflation nuisance."""
    T = residual.size
    m_pred = np.zeros(T)
    p_pred = np.zeros(T)
    m_filt = np.zeros(T)
    p_filt = np.zeros(T)
    p_pred[0] = state_variance / max(1.0 - rho**2, 1e-6)
    for t in range(T):
        if t:
            m_pred[t] = rho * m_filt[t - 1]
            p_pred[t] = rho**2 * p_filt[t - 1] + state_variance
        gain = p_pred[t] / (p_pred[t] + observation_variance)
        m_filt[t] = m_pred[t] + gain * (residual[t] - m_pred[t])
        p_filt[t] = max((1.0 - gain) * p_pred[t], 1e-12)
    out = np.zeros(T)
    out[-1] = rng.normal(m_filt[-1], np.sqrt(p_filt[-1]))
    for t in range(T - 2, -1, -1):
        next_variance = max(rho**2 * p_filt[t] + state_variance, 1e-12)
        gain = p_filt[t] * rho / next_variance
        mean = m_filt[t] + gain * (out[t + 1] - rho * m_filt[t])
        variance = max(p_filt[t] - gain**2 * next_variance, 1e-12)
        out[t] = rng.normal(mean, np.sqrt(variance))
    return out


def _truncated_unit_regression_draw(
    rng: np.random.Generator,
    y: np.ndarray,
    x: np.ndarray,
    variance: float,
    *,
    prior_mean: float,
    prior_sd: float,
) -> float:
    precision = 1.0 / prior_sd**2 + float(np.dot(x, x)) / variance
    post_sd = np.sqrt(1.0 / precision)
    post_mean = (prior_mean / prior_sd**2 + float(np.dot(x, y)) / variance) / precision
    from scipy.stats import truncnorm

    return float(
        truncnorm.rvs(
            (0.0 - post_mean) / post_sd,
            (1.0 - post_mean) / post_sd,
            loc=post_mean,
            scale=post_sd,
            random_state=rng,
        )
    )


def _transformed_regression(
    y: np.ndarray,
    X: np.ndarray,
    transformation: str,
    endpoint_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if transformation == "qoq":
        mask = np.ones(y.size, dtype=bool) if endpoint_mask is None else endpoint_mask
        return y[mask], X[mask]
    if transformation == "qoq_matched":
        mask = np.arange(y.size) >= 3
        if endpoint_mask is not None:
            mask &= endpoint_mask
        return y[mask], X[mask]
    if transformation != "yoy":
        raise ValueError(f"Unknown inflation transformation {transformation!r}.")
    if endpoint_mask is not None:
        valid_endpoint = np.flatnonzero(endpoint_mask)
        # A local YoY endpoint is admissible only when all contributing QoQ
        # endpoints are in the frozen local set.
        valid = [t for t in valid_endpoint if t >= 3 and endpoint_mask[t - 3 : t + 1].all()]
        if not valid:
            return np.empty(0), np.empty((0, X.shape[1]))
        y_w, X_w = whiten_aggregated(y, X)
        rows = np.asarray(valid, dtype=int) - 3
        # Selecting rows after global whitening would not equal whitening the
        # selected covariance. Rebuild the selected aggregate explicitly.
        A = np.zeros((len(valid), y.size))
        for row, t in enumerate(valid):
            A[row, t - 3 : t + 1] = 0.25
        G = A @ A.T
        L = np.linalg.cholesky(G)
        return np.linalg.solve(L, A @ y), np.linalg.solve(L, A @ X)
    return whiten_aggregated(y, X)


def _draw_regression(
    rng: np.random.Generator,
    y: np.ndarray,
    X: np.ndarray,
    names: tuple[str, ...],
    prior_sds: dict[str, float],
    sigma2: float,
) -> tuple[np.ndarray, float]:
    if y.size <= X.shape[1]:
        raise ValueError("The regression has no residual degrees of freedom.")
    prior_precision = np.diag([1.0 / prior_sds[name] ** 2 for name in names])
    precision = force_pd(prior_precision + X.T @ X / sigma2)
    covariance = force_pd(np.linalg.inv(precision))
    mean = covariance @ (X.T @ y / sigma2)
    beta = rng.multivariate_normal(mean, covariance)
    residual = y - X @ beta
    sigma2 = float(1.0 / rng.gamma(3.0 + y.size / 2.0, 1.0 / (8.0 + np.dot(residual, residual) / 2.0)))
    return beta, sigma2


def fit_cut_model(
    data: DesignData,
    measurement: MeasurementPosterior,
    *,
    cell: int,
    model: str,
    transformation: str,
    q0: float,
    seed: int,
    endpoint_mask: np.ndarray | None = None,
    single_coordinate: bool = False,
    no_lag: bool = False,
    extra: str | None = None,
    error_model: str | None = None,
    price_override: str | None = None,
    activity_override: str | None = None,
    include_slow_level: bool = True,
) -> CellFit:
    spec = next(item for item in CELL_SPECS if int(item["cell"]) == cell)
    price = str(price_override or spec["inflation"])
    activity = str(activity_override or spec["activity"])
    price_column = f"pi_{price}"
    lag_column = f"pi_{price}_lag1"
    activity_column = f"x_{activity}"
    missing = [
        column
        for column in (price_column, lag_column, activity_column)
        if column not in data.quarterly
    ]
    if missing:
        raise KeyError(f"Requested price/activity override columns are missing: {missing}")
    y = data.quarterly[price_column].to_numpy(dtype=float)
    pi_lag = data.quarterly[lag_column].to_numpy(dtype=float)
    expectation = data.quarterly["expectation"].to_numpy(dtype=float)
    x = data.quarterly[activity_column].to_numpy(dtype=float)
    x_scale = robust_scale(x)
    qbar_draws = measurement.draws["qbar"]
    qhat_draws = measurement.draws["qhat"]
    chains, draws, _ = qbar_draws.shape
    names = coefficient_names(
        model,
        single_coordinate=single_coordinate,
        no_lag=no_lag,
        include_slow_level=include_slow_level,
    )
    if extra == "fast_by_fast":
        names = names + ("lambda_qx",)
    elif extra == "cs1":
        names = names + ("lambda_u",)
    elif extra == "cs2":
        names = names + ("lambda_u", "lambda_eta")
    priors = _prior_sds(names, q_scale=data.q_scale, x_scale=x_scale)
    coefficients = np.zeros((chains, draws, len(names)))
    sigma = np.zeros((chains, draws))
    if error_model not in {None, "persistent_ar1", "low_frequency"}:
        raise ValueError(f"Unknown inflation error robustness {error_model!r}.")
    if error_model is not None and transformation != "qoq":
        raise ValueError("Inflation-error robustness is defined for the primary QoQ equation.")
    auxiliary = (
        {"rho_pi": np.zeros((chains, draws))}
        if error_model == "persistent_ar1"
        else {
            "rho_low_frequency": np.zeros((chains, draws)),
            "sigma_low_frequency": np.zeros((chains, draws)),
        }
        if error_model == "low_frequency"
        else {}
    )
    n_endpoints = 0
    for chain in range(chains):
        model_seed = {"E0": 11, "E1": 23, "E2": 37, "SLOW": 43}[model]
        transform_seed = {"qoq": 101, "qoq_matched": 211, "yoy": 307}[transformation]
        extra_seed = {None: 0, "fast_by_fast": 401, "cs1": 503, "cs2": 601}[extra]
        rng = np.random.default_rng(seed + cell * 10007 + chain * 1009 + model_seed + transform_seed + extra_seed)
        sigma2 = 4.0
        rho_pi = 0.5
        low_frequency = np.zeros(y.size)
        rho_low_frequency = 0.95
        variance_low_frequency = 0.25**2
        for draw in range(draws):
            kwargs = {}
            if extra in {"cs1", "cs2"}:
                kwargs = {
                    "d_q": float(measurement.draws["d_q"][chain, draw]),
                    "phi_q": float(measurement.draws["phi_q"][chain, draw]),
                }
            X, built_names = _quarterly_design(
                pi_lag=pi_lag,
                expectation=expectation,
                x=x,
                qbar=qbar_draws[chain, draw],
                qhat=qhat_draws[chain, draw],
                q0=q0,
                model=model,
                single_coordinate=single_coordinate,
                no_lag=no_lag,
                include_slow_level=include_slow_level,
                extra=extra,
                **kwargs,
            )
            if built_names != names:
                raise RuntimeError("Coefficient-name and design-column order diverged.")
            yt, Xt = _transformed_regression(y, X, transformation, endpoint_mask)
            n_endpoints = int(yt.size)
            if error_model == "persistent_ar1":
                yw, Xw = _ar1_whiten(yt, Xt, rho_pi)
                coefficients[chain, draw], sigma2 = _draw_regression(
                    rng, yw, Xw, names, priors, sigma2
                )
                residual = yt - Xt @ coefficients[chain, draw]
                rho_pi = _mh_rho(rng, rho_pi, residual, sigma2)
                auxiliary["rho_pi"][chain, draw] = rho_pi
            elif error_model == "low_frequency":
                coefficients[chain, draw], sigma2 = _draw_regression(
                    rng, yt - low_frequency, Xt, names, priors, sigma2
                )
                residual = yt - Xt @ coefficients[chain, draw]
                low_frequency = _low_frequency_draw(
                    rng,
                    residual,
                    rho=rho_low_frequency,
                    state_variance=variance_low_frequency,
                    observation_variance=sigma2,
                )
                rho_low_frequency = _truncated_unit_regression_draw(
                    rng,
                    low_frequency[1:],
                    low_frequency[:-1],
                    variance_low_frequency,
                    prior_mean=0.95,
                    prior_sd=0.04,
                )
                innovation = low_frequency[1:] - rho_low_frequency * low_frequency[:-1]
                variance_low_frequency = float(
                    1.0
                    / rng.gamma(
                        3.0 + innovation.size / 2.0,
                        1.0 / (2.0 * 0.25**2 + np.dot(innovation, innovation) / 2.0),
                    )
                )
                auxiliary["rho_low_frequency"][chain, draw] = rho_low_frequency
                auxiliary["sigma_low_frequency"][chain, draw] = np.sqrt(variance_low_frequency)
            else:
                coefficients[chain, draw], sigma2 = _draw_regression(
                    rng, yt, Xt, names, priors, sigma2
                )
            sigma[chain, draw] = np.sqrt(sigma2)
    price_config = data.config.get("cells", {}).get("inflation", {}).get(price, {})
    expectation_status = str(price_config.get("expectation_status", "proxy_conditioned"))
    return CellFit(
        cell=cell,
        inflation=price,
        activity=activity,
        model=model,
        transformation=transformation,
        coefficient_names=names,
        coefficients=coefficients,
        sigma=sigma,
        q0=q0,
        x_scale=x_scale,
        prior_sds=priors,
        n_endpoints=n_endpoints,
        expectation_status=expectation_status,
        estimator="modular_cut" if error_model is None else f"modular_cut_{error_model}",
        auxiliary_draws=auxiliary,
    )


def reference_draws(
    data: DesignData,
    measurement: MeasurementPosterior,
) -> tuple[np.ndarray, float, np.ndarray]:
    # Period comparisons are clearer and do not depend on timestamp conventions.
    ref_start = pd.Period(str(data.config["measurement"]["reference_window"][0]), freq="Q")
    ref_end = pd.Period(str(data.config["measurement"]["reference_window"][1]), freq="Q")
    mask = (data.periods >= ref_start) & (data.periods <= ref_end)
    qref = measurement.draws["qbar"][:, :, mask].mean(axis=2)
    return qref, float(qref.mean()), mask


def fit_hsa_benchmark(
    data: DesignData,
    measurement: MeasurementPosterior,
    *,
    seed: int,
) -> BenchmarkResult:
    qref, q0, _ = reference_draws(data, measurement)
    qbar = measurement.draws["qbar"]
    bandwidth = float(data.config["benchmark"]["local_bandwidth_iqr"]) * (
        float(np.subtract(*np.quantile(data.annual_values, [0.75, 0.25])))
    )
    local_probability = np.mean(np.abs(qbar - qref[:, :, None]) <= bandwidth, axis=(0, 1))
    threshold = float(data.config["benchmark"]["local_probability"])
    local = local_probability >= threshold
    minimum = int(data.config["benchmark"]["minimum_qoq_endpoints"])
    x_scale = robust_scale(data.quarterly["x_inverse_markup"])
    equivalence_band = float(data.config["benchmark"]["equivalence_contribution_pp"]) / (
        data.q_scale * x_scale
    )
    mu = float(data.config["benchmark"]["mu_reference"])
    zeta = mu / (mu - 1.0)
    b_x = float(data.config["benchmark"]["b_x"])
    reason_parts = [
        "unit-scale b_x calibration",
        "activity orthogonality maintained, not identified",
        "GDP-deflator expectation proxy used for PPI",
        "reference markup is a calibration",
    ]
    if int(local.sum()) < minimum:
        return BenchmarkResult(
            status="unsupported",
            reason=f"locality gate failed ({int(local.sum())} < {minimum}); " + "; ".join(reason_parts),
            fit=None,
            delta_hsa=None,
            theta_reference=None,
            equivalence_band=equivalence_band,
            equivalence_probability=None,
            local_probability=local_probability,
            local_endpoints=local,
            q_reference_draws=qref,
            zeta_reference=zeta,
            b_x=b_x,
        )
    fit = fit_cut_model(
        data,
        measurement,
        cell=1,
        model="E2",
        transformation="qoq",
        q0=q0,
        seed=seed,
        endpoint_mask=local,
        single_coordinate=True,
        no_lag=True,
    )
    index = {name: i for i, name in enumerate(fit.coefficient_names)}
    theta_ref = fit.coefficients[:, :, index["theta_0"]] + fit.coefficients[:, :, index["gamma"]] * (qref - q0)
    delta = fit.coefficients[:, :, index["kappa_1"]] - b_x * zeta * theta_ref
    probability = float(np.mean(np.abs(delta) < equivalence_band))
    return BenchmarkResult(
        status="conditional_benchmark",
        reason="; ".join(reason_parts),
        fit=fit,
        delta_hsa=delta,
        theta_reference=theta_ref,
        equivalence_band=equivalence_band,
        equivalence_probability=probability,
        local_probability=local_probability,
        local_endpoints=local,
        q_reference_draws=qref,
        zeta_reference=zeta,
        b_x=b_x,
    )
