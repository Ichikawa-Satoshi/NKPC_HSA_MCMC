"""Observed inverse-HHI model experiments.

The module deliberately avoids the annual-firm/QCEW common-factor block.  A
quarterly inverse HHI is transformed to the repository's ten-log-point scale
and enters the inflation equation as an observed competition coordinate.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

import arviz as az
import numpy as np
import pandas as pd
from scipy.stats import norm

from nkpc_hsa.config import load_yaml
from nkpc_hsa.gibbs.common.joint_ffbs import force_pd
from nkpc_hsa.paths import data_root

from nkpc_hsa.phillips.data import robust_scale
from nkpc_hsa.phillips.inflation import (
    _ar1_whiten,
    _draw_regression,
    _low_frequency_draw,
    _mh_rho,
    _truncated_unit_regression_draw,
)


@dataclass(frozen=True)
class ObservedHHISample:
    periods: pd.PeriodIndex
    y: np.ndarray
    pi_lag: np.ndarray
    expectation: np.ndarray
    activity: np.ndarray
    q: np.ndarray
    inflation: str
    activity_name: str
    hhi_variant: str


@dataclass(frozen=True)
class ObservedHHIFit:
    coefficients: np.ndarray
    sigma: np.ndarray
    names: tuple[str, ...]
    prior_sds: dict[str, float]
    periods: tuple[str, ...]
    cell: int
    inflation: str
    activity: str
    hhi_variant: str
    fast_definition: str
    environment_definition: str
    timing: str
    model_variant: str
    error_model: str
    design_condition_number: float
    theta_orthogonal_share: float
    auxiliary: dict[str, np.ndarray]


CELL_SPECS: tuple[tuple[int, str, str], ...] = tuple(
    (index + 1, inflation, activity)
    for index, (inflation, activity) in enumerate(
        (p, a)
        for p in ("ppi", "cpi", "core_cpi")
        for a in ("inverse_markup", "bn_output_gap", "negative_unemployment_gap")
    )
)


def transform_inverse_hhi(values: np.ndarray) -> np.ndarray:
    """Convert effective firm counts to centered ten-log-point units."""
    values = np.asarray(values, dtype=float)
    if np.any(~np.isfinite(values)) or np.any(values <= 0.0):
        raise ValueError("Inverse HHI must be finite and positive.")
    transformed = 10.0 * np.log(values)
    return transformed - np.mean(transformed)


def _ewma_innovation(q: np.ndarray, half_life: float) -> np.ndarray:
    """One-sided forecast errors from a fixed-gain local-level filter."""
    if half_life <= 0.0:
        raise ValueError("half_life must be positive.")
    gain = 1.0 - np.exp(np.log(0.5) / half_life)
    level = float(q[0])
    innovation = np.full(q.size, np.nan)
    for t in range(1, q.size):
        innovation[t] = q[t] - level
        level += gain * innovation[t]
    return innovation


def fast_component(q: np.ndarray, definition: str) -> np.ndarray:
    """Construct a transparent fast HHI movement without a common factor."""
    q = np.asarray(q, dtype=float)
    if definition.startswith("ewma_hl"):
        return _ewma_innovation(q, float(definition.removeprefix("ewma_hl")))
    if definition == "first_difference":
        return np.r_[np.nan, np.diff(q)]
    if definition == "ar1_innovation":
        x = np.column_stack((np.ones(q.size - 1), q[:-1]))
        beta = np.linalg.lstsq(x, q[1:], rcond=None)[0]
        return np.r_[np.nan, q[1:] - x @ beta]
    if definition == "ar2_innovation":
        x = np.column_stack((np.ones(q.size - 2), q[1:-1], q[:-2]))
        beta = np.linalg.lstsq(x, q[2:], rcond=None)[0]
        return np.r_[np.nan, np.nan, q[2:] - x @ beta]
    raise ValueError(f"Unknown fast-component definition: {definition}")


def timed_fast_component(fast: np.ndarray, timing: str) -> np.ndarray:
    fast = np.asarray(fast, dtype=float)
    if timing == "current":
        return fast.copy()
    if timing.startswith("lag"):
        lag = int(timing.removeprefix("lag"))
        if lag < 1:
            raise ValueError("A named lag must be at least one quarter.")
        return np.r_[np.full(lag, np.nan), fast[:-lag]]
    if timing == "distributed4":
        out = np.full(fast.size, np.nan)
        weights = np.full(4, 0.25)
        for t in range(3, fast.size):
            window = fast[t - 3 : t + 1]
            if np.isfinite(window).all():
                out[t] = float(weights @ window[::-1])
        return out
    raise ValueError(f"Unknown fast timing: {timing}")


def load_observed_hhi_frame(config_path: str | Path | None = None) -> tuple[pd.DataFrame, Mapping]:
    path = Path(config_path or Path(__file__).resolve().parent / "config.yaml")
    config = load_yaml(path)
    frame = pd.read_csv(data_root() / "processed" / str(config["data"]["file"]), parse_dates=["DATE"])
    frame.index = pd.PeriodIndex(frame.pop("DATE"), freq="Q")
    return frame.sort_index(), config


def load_cell_sample(
    frame: pd.DataFrame,
    config: Mapping,
    *,
    cell: int,
    hhi_variant: str,
) -> ObservedHHISample:
    try:
        _, inflation, activity = next(item for item in CELL_SPECS if item[0] == cell)
    except StopIteration as exc:
        raise ValueError(f"Unknown cell {cell}.") from exc
    y_col, lag_col = config["cells"]["inflation"][inflation]
    x_col = str(config["cells"]["activity"][activity])
    hhi_col = str(config["data"]["hhi_variants"][hhi_variant])
    expectation_col = str(config["data"]["expectation"])
    columns = [y_col, lag_col, expectation_col, x_col, hhi_col]
    selected = frame[columns].apply(pd.to_numeric, errors="coerce").dropna()
    if len(selected) < 16:
        raise ValueError(f"Cell {cell} / {hhi_variant} has only {len(selected)} complete observations.")
    return ObservedHHISample(
        periods=selected.index,
        y=selected[y_col].to_numpy(float),
        pi_lag=selected[lag_col].to_numpy(float),
        expectation=selected[expectation_col].to_numpy(float),
        activity=selected[x_col].to_numpy(float),
        q=transform_inverse_hhi(selected[hhi_col].to_numpy(float)),
        inflation=inflation,
        activity_name=activity,
        hhi_variant=hhi_variant,
    )


def _prior_sds(names: Iterable[str], q_scale: float, x_scale: float) -> dict[str, float]:
    values = {
        "a": 5.0,
        "beta_b": 1.0,
        "beta_f": 1.0,
        "psi": 1.5 / q_scale,
        "kappa_0": 2.0 / x_scale,
        "kappa_1": 1.0 / (q_scale * x_scale),
        "kappa_2": 1.0 / (q_scale**2 * x_scale),
        "theta_0": 1.0 / q_scale,
        "gamma": 1.0 / q_scale**2,
        "theta_hsa": 1.0 / q_scale,
    }
    return {name: values[name] for name in names}


def build_observed_design(
    sample: ObservedHHISample,
    *,
    fast_definition: str,
    environment_definition: str = "total",
    timing: str,
    model_variant: str,
    no_lag: bool = False,
    include_level: bool = True,
    zeta_reference: float = 6.0,
    b_x: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...], np.ndarray]:
    """Return y, X, names, and retained-row mask for an observed-HHI model.

    ``include_level`` controls the standalone competition-level term (``psi``).
    It is an empirical control, NOT part of the structural HSA NKPC (which enters
    competition only through the slope interaction, the direct fast channel, and
    the bilinear term); set it False for the theory-faithful specification.
    """
    raw_fast = fast_component(sample.q, fast_definition)
    fast = timed_fast_component(raw_fast, timing)
    if environment_definition == "total":
        z = sample.q.copy()
    elif environment_definition == "predicted_level":
        # For an EWMA innovation this is the one-step-ahead level based only on
        # information through t-1.  The same identity yields q[t-1] for first
        # differences and the fitted AR(1) level for AR(1) innovations.
        z = sample.q - raw_fast
    else:
        raise ValueError(f"Unknown HHI environment definition: {environment_definition}")
    z = z - np.nanmean(z)
    columns: list[np.ndarray] = [np.ones_like(z)]
    names = ["a"]
    if not no_lag:
        columns.append(sample.pi_lag)
        names.append("beta_b")
    columns.append(sample.expectation); names.append("beta_f")
    if include_level:
        columns.append(z); names.append("psi")
    columns.append(sample.activity); names.append("kappa_0")
    if model_variant == "no_theta":
        columns.append(z * sample.activity)
        names.append("kappa_1")
    elif model_variant == "constant_theta":
        columns.extend((z * sample.activity, -fast))
        names.extend(("kappa_1", "theta_0"))
    elif model_variant == "varying_theta":
        columns.extend((z * sample.activity, -fast, -z * fast))
        names.extend(("kappa_1", "theta_0", "gamma"))
    elif model_variant == "quadratic_theta":
        # Nonlinear competition-dependent slope: kappa(q) = kappa_0 + kappa_1 q + kappa_2 q^2.
        # The marginal effect of competition on the slope is itself competition-dependent.
        columns.extend((z * sample.activity, z * z * sample.activity, -fast))
        names.extend(("kappa_1", "kappa_2", "theta_0"))
    elif model_variant == "hsa_restricted":
        columns.append(b_x * zeta_reference * z * sample.activity - fast)
        names.append("theta_hsa")
    else:
        raise ValueError(f"Unknown observed-HHI model variant: {model_variant}")
    X = np.column_stack(columns)
    mask = np.isfinite(sample.y) & np.isfinite(X).all(axis=1)
    if int(mask.sum()) <= X.shape[1] + 3:
        raise ValueError("Observed-HHI regression has too few residual degrees of freedom.")
    return sample.y[mask], X[mask], tuple(names), mask


def _design_diagnostics(X: np.ndarray, names: tuple[str, ...]) -> tuple[float, float]:
    standardized = X[:, 1:].copy()
    standardized -= standardized.mean(axis=0)
    scales = standardized.std(axis=0, ddof=1)
    standardized /= np.where(scales > 1e-12, scales, 1.0)
    condition = float(np.linalg.cond(standardized))
    theta_name = "theta_hsa" if "theta_hsa" in names else "theta_0" if "theta_0" in names else None
    if theta_name is None:
        return condition, np.nan
    index = names.index(theta_name)
    target = X[:, index]
    other = np.delete(X, index, axis=1)
    fitted = other @ np.linalg.lstsq(other, target, rcond=None)[0]
    total = float(np.dot(target - target.mean(), target - target.mean()))
    residual = float(np.dot(target - fitted, target - fitted))
    return condition, residual / total if total > 1e-12 else 0.0


def fit_observed_hhi_model(
    sample: ObservedHHISample,
    *,
    cell: int,
    fast_definition: str,
    environment_definition: str = "total",
    timing: str,
    model_variant: str,
    error_model: str,
    iterations: int,
    warmup: int,
    thin: int,
    chains: int,
    seed: int,
    no_lag: bool = False,
    include_level: bool = True,
    zeta_reference: float = 6.0,
    b_x: float = 1.0,
) -> ObservedHHIFit:
    if error_model not in {"iid", "persistent_ar1", "low_frequency"}:
        raise ValueError(f"Unknown error model: {error_model}")
    if iterations <= warmup or thin < 1 or chains < 2:
        raise ValueError("Invalid sampling configuration.")
    y, X, names, mask = build_observed_design(
        sample,
        fast_definition=fast_definition,
        environment_definition=environment_definition,
        timing=timing,
        model_variant=model_variant,
        no_lag=no_lag,
        include_level=include_level,
        zeta_reference=zeta_reference,
        b_x=b_x,
    )
    q_scale = robust_scale(sample.q[mask])
    x_scale = robust_scale(sample.activity[mask])
    priors = _prior_sds(names, q_scale, x_scale)
    draws_per_chain = (iterations - warmup - 1) // thin + 1
    coefficients = np.zeros((chains, draws_per_chain, len(names)))
    sigma = np.zeros((chains, draws_per_chain))
    auxiliary: dict[str, np.ndarray] = {}
    if error_model == "persistent_ar1":
        auxiliary["rho_pi"] = np.zeros((chains, draws_per_chain))
    if error_model == "low_frequency":
        auxiliary["rho_low_frequency"] = np.zeros((chains, draws_per_chain))
        auxiliary["sigma_low_frequency"] = np.zeros((chains, draws_per_chain))
    for chain in range(chains):
        rng = np.random.default_rng(seed + cell * 10007 + chain * 1009)
        sigma2 = max(float(np.var(y)), 1.0)
        rho_pi = 0.3
        low_frequency = np.zeros(y.size)
        rho_low = 0.95
        var_low = 0.25**2
        saved = 0
        for iteration in range(iterations):
            if error_model == "persistent_ar1":
                yw, Xw = _ar1_whiten(y, X, rho_pi)
                beta, sigma2 = _draw_regression(rng, yw, Xw, names, priors, sigma2)
                rho_pi = _mh_rho(rng, rho_pi, y - X @ beta, sigma2)
            elif error_model == "low_frequency":
                beta, sigma2 = _draw_regression(rng, y - low_frequency, X, names, priors, sigma2)
                residual = y - X @ beta
                low_frequency = _low_frequency_draw(
                    rng,
                    residual,
                    rho=rho_low,
                    state_variance=var_low,
                    observation_variance=sigma2,
                )
                rho_low = _truncated_unit_regression_draw(
                    rng,
                    low_frequency[1:],
                    low_frequency[:-1],
                    var_low,
                    prior_mean=0.95,
                    prior_sd=0.04,
                )
                innovation = low_frequency[1:] - rho_low * low_frequency[:-1]
                var_low = float(
                    1.0
                    / rng.gamma(
                        3.0 + innovation.size / 2.0,
                        1.0 / (2.0 * 0.25**2 + np.dot(innovation, innovation) / 2.0),
                    )
                )
            else:
                beta, sigma2 = _draw_regression(rng, y, X, names, priors, sigma2)
            if iteration >= warmup and (iteration - warmup) % thin == 0:
                coefficients[chain, saved] = beta
                sigma[chain, saved] = np.sqrt(sigma2)
                if error_model == "persistent_ar1":
                    auxiliary["rho_pi"][chain, saved] = rho_pi
                elif error_model == "low_frequency":
                    auxiliary["rho_low_frequency"][chain, saved] = rho_low
                    auxiliary["sigma_low_frequency"][chain, saved] = np.sqrt(var_low)
                saved += 1
    condition, orthogonal = _design_diagnostics(X, names)
    return ObservedHHIFit(
        coefficients=coefficients,
        sigma=sigma,
        names=names,
        prior_sds=priors,
        periods=tuple(map(str, sample.periods[mask])),
        cell=cell,
        inflation=sample.inflation,
        activity=sample.activity_name,
        hhi_variant=sample.hhi_variant,
        fast_definition=fast_definition,
        environment_definition=environment_definition,
        timing=timing,
        model_variant=model_variant,
        error_model=error_model,
        design_condition_number=condition,
        theta_orthogonal_share=orthogonal,
        auxiliary=auxiliary,
    )


def _diagnostics(values: np.ndarray) -> tuple[float, float]:
    rhat = float(np.asarray(az.rhat(values, method="rank")))
    ess = float(np.asarray(az.ess(values, method="bulk")))
    return rhat, ess


def summarize_observed_fit(fit: ObservedHHIFit) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for index, name in enumerate(fit.names):
        values = fit.coefficients[:, :, index]
        flat = values.reshape(-1)
        rhat, ess = _diagnostics(values)
        probability_positive = float(np.mean(flat > 0.0))
        rows.append(
            {
                "cell": fit.cell,
                "inflation": fit.inflation,
                "activity": fit.activity,
                "hhi_variant": fit.hhi_variant,
                "fast_definition": fit.fast_definition,
                "environment_definition": fit.environment_definition,
                "timing": fit.timing,
                "model_variant": fit.model_variant,
                "error_model": fit.error_model,
                "n": len(fit.periods),
                "sample_start": fit.periods[0],
                "sample_end": fit.periods[-1],
                "parameter": name,
                "mean": float(np.mean(flat)),
                "sd": float(np.std(flat, ddof=1)),
                "ci_2.5": float(np.quantile(flat, 0.025)),
                "ci_97.5": float(np.quantile(flat, 0.975)),
                "probability_positive": probability_positive,
                "sign_probability": max(probability_positive, 1.0 - probability_positive),
                "prior_sd": fit.prior_sds[name],
                "posterior_prior_sd_ratio": float(np.std(flat, ddof=1) / fit.prior_sds[name]),
                "rhat": rhat,
                "bulk_ess": ess,
                "convergence_gate": bool(rhat <= 1.01 and ess >= 400),
                "design_condition_number": fit.design_condition_number,
                "theta_orthogonal_share": fit.theta_orthogonal_share,
            }
        )
    return pd.DataFrame(rows)


def posterior_theta(fit: ObservedHHIFit) -> np.ndarray | None:
    name = "theta_hsa" if "theta_hsa" in fit.names else "theta_0" if "theta_0" in fit.names else None
    return None if name is None else fit.coefficients[:, :, fit.names.index(name)]


def simulate_theta_recovery(
    sample: ObservedHHISample,
    *,
    fast_definition: str,
    timing: str,
    sample_sizes: Iterable[int],
    effect_sizes_sd: Iterable[float],
    replications: int,
    seed: int,
) -> pd.DataFrame:
    """Fixed-design recovery experiment on standardized theta contributions."""
    y, X, names, _ = build_observed_design(
        sample,
        fast_definition=fast_definition,
        timing=timing,
        model_variant="constant_theta",
    )
    theta_index = names.index("theta_0")
    rng = np.random.default_rng(seed)
    residual_sd = max(float(np.std(y, ddof=1)), 1e-6)
    rows: list[dict[str, object]] = []
    for n in sample_sizes:
        repeats = int(np.ceil(n / len(y)))
        Xn = np.tile(X, (repeats, 1))[:n].copy()
        theta_regressor = Xn[:, theta_index]
        theta_scale = float(np.std(theta_regressor, ddof=1))
        if theta_scale <= 1e-12:
            raise ValueError("Theta regressor has no simulation variation.")
        prior_sds = _prior_sds(names, robust_scale(np.tile(sample.q, repeats)[:n]), robust_scale(np.tile(sample.activity, repeats)[:n]))
        prior_precision = np.diag([1.0 / prior_sds[name] ** 2 for name in names])
        precision = force_pd(prior_precision + Xn.T @ Xn / residual_sd**2)
        covariance = force_pd(np.linalg.inv(precision))
        post_sd = float(np.sqrt(covariance[theta_index, theta_index]))
        for effect_sd in effect_sizes_sd:
            theta_true = float(effect_sd * residual_sd / theta_scale)
            detected = 0
            covered = 0
            means: list[float] = []
            for _ in range(replications):
                beta_true = np.zeros(len(names))
                beta_true[theta_index] = theta_true
                simulated = Xn @ beta_true + rng.normal(0.0, residual_sd, n)
                mean = covariance @ (Xn.T @ simulated / residual_sd**2)
                theta_mean = float(mean[theta_index])
                probability_positive = float(norm.cdf(theta_mean / post_sd))
                sign_probability = probability_positive if theta_true >= 0 else 1.0 - probability_positive
                detected += int(sign_probability >= 0.8)
                covered += int(theta_mean - 1.96 * post_sd <= theta_true <= theta_mean + 1.96 * post_sd)
                means.append(theta_mean)
            rows.append(
                {
                    "n": n,
                    "effect_sd": effect_sd,
                    "theta_true": theta_true,
                    "posterior_sd": post_sd,
                    "posterior_prior_sd_ratio": post_sd / prior_sds["theta_0"],
                    "sign_probability_ge_0.8_rate": detected / replications,
                    "coverage_95": covered / replications,
                    "mean_estimate": float(np.mean(means)),
                    "bias": float(np.mean(means) - theta_true),
                    "replications": replications,
                }
            )
    return pd.DataFrame(rows)
