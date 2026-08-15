"""Q4-anchored competition state with a quarterly inverse-markup bridge.

This module is deliberately separate from the QCEW measurement experiment.  The
annual Q4 coordinate hard-anchors ``q = qbar + qhat`` and inverse-markup changes
only inform the path between annual anchors::

    delta(qbar_t + qhat_t) = alpha_markup * delta(inverse_markup_t) + error_t,
    alpha_markup > 0.

The fast component follows the historical AR(2) law so the experiment directly
tests whether a quarterly proxy resolves the annual-sampling ambiguity discussed
in the technical audit.  ``markup_error="ar1"`` adds a stationary markup-specific
state and is the required conservative sensitivity.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from scipy.stats import truncnorm

from .state import MeasurementPosterior, _draw_ig, sample_linear_state_path


def _is_stationary_ar2(rho_1: float, rho_2: float) -> bool:
    return (
        abs(rho_2) < 1.0
        and rho_1 + rho_2 < 1.0
        and rho_2 - rho_1 < 1.0
    )


def _normal_regression_draw(
    rng: np.random.Generator,
    y: np.ndarray,
    X: np.ndarray,
    variance: float,
    prior_mean: np.ndarray,
    prior_sd: np.ndarray,
) -> np.ndarray:
    prior_precision = np.diag(1.0 / np.square(prior_sd))
    precision = prior_precision + X.T @ X / variance
    covariance = np.linalg.inv(precision)
    mean = covariance @ (prior_precision @ prior_mean + X.T @ y / variance)
    return rng.multivariate_normal(mean, covariance)


def _stationary_ar2_draw(
    rng: np.random.Generator,
    qhat: np.ndarray,
    initial_lag: float,
    variance: float,
    current: tuple[float, float],
    *,
    max_tries: int = 2000,
) -> tuple[float, float]:
    y = qhat[1:]
    second_lag = np.concatenate([[initial_lag], qhat[:-2]])
    X = np.column_stack([qhat[:-1], second_lag])
    prior_mean = np.array([0.5, -0.5])
    prior_sd = np.array([0.3, 0.3])
    prior_precision = np.diag(1.0 / np.square(prior_sd))
    precision = prior_precision + X.T @ X / variance
    covariance = np.linalg.inv(precision)
    mean = covariance @ (prior_precision @ prior_mean + X.T @ y / variance)
    for _ in range(max_tries):
        draw = rng.multivariate_normal(mean, covariance)
        if _is_stationary_ar2(float(draw[0]), float(draw[1])):
            return float(draw[0]), float(draw[1])
    if _is_stationary_ar2(*current):
        return current
    return 0.5, -0.5


def _positive_loading_draw(
    rng: np.random.Generator,
    y: np.ndarray,
    x: np.ndarray,
    variance: float,
    prior_mean: float,
    prior_sd: float,
) -> float:
    precision = 1.0 / prior_sd**2 + float(np.dot(x, x)) / variance
    post_sd = np.sqrt(1.0 / precision)
    post_mean = (
        prior_mean / prior_sd**2 + float(np.dot(x, y)) / variance
    ) / precision
    return float(
        truncnorm.rvs(
            (0.0 - post_mean) / post_sd,
            np.inf,
            loc=post_mean,
            scale=post_sd,
            random_state=rng,
        )
    )


def _unit_ar_draw(
    rng: np.random.Generator,
    y: np.ndarray,
    x: np.ndarray,
    variance: float,
    current: float,
) -> float:
    precision = 1.0 / 0.3**2 + float(np.dot(x, x)) / variance
    post_sd = np.sqrt(1.0 / precision)
    post_mean = (0.5 / 0.3**2 + float(np.dot(x, y)) / variance) / precision
    try:
        return float(
            truncnorm.rvs(
                (-1.0 - post_mean) / post_sd,
                (1.0 - post_mean) / post_sd,
                loc=post_mean,
                scale=post_sd,
                random_state=rng,
            )
        )
    except ValueError:
        return current


def _ffbs(
    rng: np.random.Generator,
    *,
    annual: np.ndarray,
    proxy: np.ndarray,
    include_proxy: bool,
    markup_error: str,
    drift: float,
    rho_1: float,
    rho_2: float,
    rho_markup: float,
    alpha_markup: float,
    variances: dict[str, float],
    q_scale: float,
) -> np.ndarray:
    if markup_error not in {"iid", "ar1"}:
        raise ValueError("markup_error must be 'iid' or 'ar1'.")
    include_markup_state = include_proxy and markup_error == "ar1"
    # State order is qbar_t, qhat_t, qhat_(t-1), q_(t-1), [r_markup_t].
    # Carrying q_(t-1) makes the change equation an ordinary observation row.
    dim = 5 if include_markup_state else 4
    F = np.zeros((dim, dim))
    F[0, 0] = 1.0
    F[1, 1:3] = [rho_1, rho_2]
    F[2, 1] = 1.0
    F[3, 0:2] = 1.0
    if include_markup_state:
        F[4, 4] = rho_markup
    c = np.zeros(dim)
    c[0] = drift
    innovation = np.array(
        [variances["qbar"], variances["qhat"], 1e-10, 1e-10]
        + ([variances["r_markup"]] if include_markup_state else [])
    )
    initial_sd = np.array(
        [2.0 * q_scale, 2.0 * q_scale, 2.0 * q_scale, 2.0 * q_scale]
        + ([2.0 * q_scale] if include_markup_state else [])
    )
    observations: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    for t in range(annual.size):
        values: list[float] = []
        rows: list[np.ndarray] = []
        obs_variances: list[float] = []
        if np.isfinite(annual[t]):
            row = np.zeros(dim)
            row[:2] = 1.0
            rows.append(row)
            values.append(float(annual[t]))
            # This is a numerical approximation to the exact Q4 constraint.
            obs_variances.append((1e-6 * q_scale) ** 2)
        if include_proxy and t > 0 and np.isfinite(proxy[t - 1 : t + 1]).all():
            row = np.zeros(dim)
            row[0] = 1.0
            row[1] = 1.0
            row[3] = -1.0
            if include_markup_state:
                # Delta q_t = alpha * Delta proxy_t + r_t + e_t.
                row[4] = -1.0
            rows.append(row)
            values.append(float(alpha_markup * (proxy[t] - proxy[t - 1])))
            obs_variances.append(variances["markup"])
        if rows:
            observations.append(
                (
                    np.asarray(values),
                    np.vstack(rows),
                    np.diag(obs_variances),
                )
            )
        else:
            observations.append((np.empty(0), np.empty((0, dim)), np.empty((0, 0))))
    return sample_linear_state_path(
        rng,
        F=F,
        c=c,
        Q=np.diag(innovation),
        m0=np.zeros(dim),
        P0=np.diag(np.square(initial_sd)),
        observations=observations,
    )


def _run_chain(
    *,
    annual: np.ndarray,
    proxy: np.ndarray,
    include_proxy: bool,
    markup_error: str,
    q_scale: float,
    proxy_scale: float,
    iterations: int,
    warmup: int,
    thin: int,
    seed: int,
    progress_tick: Callable[[], None] | None,
) -> dict[str, np.ndarray]:
    if iterations <= warmup or thin < 1:
        raise ValueError("iterations must exceed warmup and thin must be positive.")
    rng = np.random.default_rng(seed)
    prior_shape = 3.0
    drift, rho_1, rho_2 = 0.0, 0.5, -0.5
    rho_markup = 0.5
    loading_scale = q_scale / proxy_scale
    anchor_index = np.flatnonzero(np.isfinite(annual))
    anchor_delta = np.diff(annual[anchor_index])
    proxy_anchor_delta = np.diff(proxy[anchor_index])
    initial_denominator = float(np.dot(proxy_anchor_delta, proxy_anchor_delta))
    initial_slope = (
        float(np.dot(proxy_anchor_delta, anchor_delta)) / initial_denominator
        if initial_denominator > 0.0
        else 0.0
    )
    alpha_markup = max(initial_slope, 0.05 * loading_scale)
    variances = {
        "qbar": (0.05 * q_scale) ** 2,
        "qhat": (0.25 * q_scale) ** 2,
        "markup": (0.35 * q_scale) ** 2,
        "r_markup": (0.15 * q_scale) ** 2,
    }
    saved: dict[str, list[np.ndarray | float]] = {
        name: []
        for name in (
            "qbar",
            "qhat",
            "d_q",
            "rho_1",
            "rho_2",
            "sigma_qbar",
            "sigma_qhat",
            "max_anchor_error",
        )
    }
    if include_proxy:
        saved.update({"alpha_markup": [], "sigma_markup": []})
        if markup_error == "ar1":
            saved.update({"r_markup": [], "rho_markup": [], "sigma_r_markup": []})

    state = np.zeros((annual.size, 5 if include_proxy and markup_error == "ar1" else 4))
    initial_lag = 0.0
    for iteration in range(iterations):
        state = _ffbs(
            rng,
            annual=annual,
            proxy=proxy,
            include_proxy=include_proxy,
            markup_error=markup_error,
            drift=drift,
            rho_1=rho_1,
            rho_2=rho_2,
            rho_markup=rho_markup,
            alpha_markup=alpha_markup,
            variances=variances,
            q_scale=q_scale,
        )
        qbar, qhat = state[:, 0], state[:, 1]
        initial_lag = float(state[0, 2])
        drift_draw = _normal_regression_draw(
            rng,
            np.diff(qbar),
            np.ones((qbar.size - 1, 1)),
            variances["qbar"],
            np.array([0.0]),
            np.array([0.05 * q_scale]),
        )
        drift = float(drift_draw[0])
        rho_1, rho_2 = _stationary_ar2_draw(
            rng,
            qhat,
            initial_lag,
            variances["qhat"],
            (rho_1, rho_2),
        )
        second_lag = np.concatenate([[initial_lag], qhat[:-2]])
        res_qbar = np.diff(qbar) - drift
        res_qhat = qhat[1:] - rho_1 * qhat[:-1] - rho_2 * second_lag
        variances["qbar"] = _draw_ig(
            rng,
            prior_shape + res_qbar.size / 2,
            2 * (0.05 * q_scale) ** 2 + np.dot(res_qbar, res_qbar) / 2,
        )
        variances["qhat"] = _draw_ig(
            rng,
            prior_shape + res_qhat.size / 2,
            2 * (0.25 * q_scale) ** 2 + np.dot(res_qhat, res_qhat) / 2,
        )
        annual_mask = np.isfinite(annual)
        max_anchor_error = float(
            np.max(np.abs(annual[annual_mask] - qbar[annual_mask] - qhat[annual_mask]))
        )

        if include_proxy:
            qtotal = qbar + qhat
            delta_q = np.diff(qtotal)
            delta_proxy = np.diff(proxy)
            r_markup = state[:, 4] if markup_error == "ar1" else np.zeros_like(qtotal)
            alpha_markup = _positive_loading_draw(
                rng,
                delta_q - r_markup[1:],
                delta_proxy,
                variances["markup"],
                0.0,
                2.0 * loading_scale,
            )
            res_markup = delta_q - alpha_markup * delta_proxy - r_markup[1:]
            variances["markup"] = _draw_ig(
                rng,
                prior_shape + res_markup.size / 2,
                2 * (0.35 * q_scale) ** 2 + np.dot(res_markup, res_markup) / 2,
            )
            if markup_error == "ar1":
                rho_markup = _unit_ar_draw(
                    rng,
                    r_markup[1:],
                    r_markup[:-1],
                    variances["r_markup"],
                    rho_markup,
                )
                innovation = r_markup[1:] - rho_markup * r_markup[:-1]
                variances["r_markup"] = _draw_ig(
                    rng,
                    prior_shape + innovation.size / 2,
                    2 * (0.15 * q_scale) ** 2 + np.dot(innovation, innovation) / 2,
                )

        if iteration >= warmup and (iteration - warmup) % thin == 0:
            values: dict[str, np.ndarray | float] = {
                "qbar": qbar.copy(),
                "qhat": qhat.copy(),
                "d_q": drift,
                "rho_1": rho_1,
                "rho_2": rho_2,
                "sigma_qbar": np.sqrt(variances["qbar"]),
                "sigma_qhat": np.sqrt(variances["qhat"]),
                "max_anchor_error": max_anchor_error,
            }
            if include_proxy:
                values.update(
                    alpha_markup=alpha_markup,
                    sigma_markup=np.sqrt(variances["markup"]),
                )
                if markup_error == "ar1":
                    values.update(
                        r_markup=r_markup.copy(),
                        rho_markup=rho_markup,
                        sigma_r_markup=np.sqrt(variances["r_markup"]),
                    )
            for name, value in values.items():
                saved[name].append(value)
        if progress_tick is not None:
            progress_tick()
    return {name: np.asarray(values, dtype=float) for name, values in saved.items()}


def sample_markup_measurement_posterior(
    annual: np.ndarray,
    inverse_markup: np.ndarray,
    *,
    q_scale: float,
    proxy_scale: float,
    periods: tuple[str, ...],
    markup_error: str = "iid",
    iterations: int,
    warmup: int,
    thin: int,
    chains: int,
    seed: int,
    progress_tick: Callable[[], None] | None = None,
) -> MeasurementPosterior:
    """Fit hard-anchored annual-only and markup-change bridge AR(2) systems."""
    annual = np.asarray(annual, dtype=float)
    proxy = np.asarray(inverse_markup, dtype=float)
    if annual.shape != proxy.shape:
        raise ValueError("annual and inverse_markup must have the same quarterly length.")
    if not np.all(np.isfinite(proxy)):
        raise ValueError("The quarterly inverse-markup proxy must be complete.")
    if chains < 2:
        raise ValueError("At least two chains are required for diagnostics.")
    outputs: dict[bool, dict[str, list[np.ndarray]]] = {False: {}, True: {}}
    for include_proxy in (False, True):
        for chain in range(chains):
            result = _run_chain(
                annual=annual,
                proxy=proxy,
                include_proxy=include_proxy,
                markup_error=markup_error,
                q_scale=q_scale,
                proxy_scale=proxy_scale,
                iterations=iterations,
                warmup=warmup,
                thin=thin,
                seed=seed + (100003 if include_proxy else 0) + chain * 1009,
                progress_tick=progress_tick,
            )
            for name, values in result.items():
                outputs[include_proxy].setdefault(name, []).append(values)
    annual_draws = {
        name: np.stack(values, axis=0) for name, values in outputs[False].items()
    }
    augmented_draws = {
        name: np.stack(values, axis=0) for name, values in outputs[True].items()
    }
    annual_sd = np.std(annual_draws["qhat"], axis=(0, 1), ddof=1)
    augmented_sd = np.std(augmented_draws["qhat"], axis=(0, 1), ddof=1)
    information_ratio = float(np.median(augmented_sd) / np.median(annual_sd))
    return MeasurementPosterior(
        draws=augmented_draws,
        annual_only_draws=annual_draws,
        information_ratio=information_ratio,
        periods=periods,
    )
