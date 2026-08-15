"""Q4-anchored interpolation with zero-sum within-year markup timing.

For each complete Q4-to-Q4 interval, quarterly changes are constructed as

    delta q[y, j] = delta N[y] / 4 + lambda * s_N * z[y, j],

where the standardized within-year markup signal ``z`` sums to zero.  Hence the
four quarterly changes always sum exactly to the observed annual N change.
``lambda`` is a declared sensitivity weight, not an estimated parameter.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from nkpc_hsa.phillips.data import robust_scale
from nkpc_hsa.phillips.markup_measurement import _run_chain
from nkpc_hsa.phillips.state import MeasurementPosterior


def _stacked_centered_changes(proxy: np.ndarray, anchors: np.ndarray) -> np.ndarray:
    pieces: list[np.ndarray] = []
    first = np.diff(proxy[: anchors[0] + 1])
    if first.size:
        pieces.append(first - np.mean(first))
    for left, right in zip(anchors[:-1], anchors[1:]):
        change = np.diff(proxy[left : right + 1])
        if change.size != 4:
            raise ValueError("Every complete Q4-to-Q4 interval must contain four changes.")
        pieces.append(change - np.mean(change))
    return np.concatenate(pieces)


def build_q4_anchored_path(
    annual: np.ndarray,
    inverse_markup: np.ndarray,
    *,
    lambda_weight: float,
) -> tuple[np.ndarray, dict[str, float]]:
    """Construct a quarterly total-q path whose Q4 changes equal annual N changes."""
    annual = np.asarray(annual, dtype=float)
    proxy = np.asarray(inverse_markup, dtype=float)
    if annual.shape != proxy.shape or annual.ndim != 1:
        raise ValueError("annual and inverse_markup must be same-length vectors.")
    if not np.all(np.isfinite(proxy)):
        raise ValueError("inverse_markup must be complete.")
    if lambda_weight < 0.0:
        raise ValueError("lambda_weight must be nonnegative.")
    anchors = np.flatnonzero(np.isfinite(annual))
    if anchors.size < 2 or np.any(np.diff(anchors) != 4):
        raise ValueError("Annual anchors must be consecutive Q4 observations.")

    centered_changes = _stacked_centered_changes(proxy, anchors)
    within_markup_scale = robust_scale(centered_changes)
    annual_change_scale = robust_scale(np.diff(annual[anchors]))
    quarterly_n_scale = annual_change_scale / 2.0

    path = np.full(annual.size, np.nan)
    first_anchor = int(anchors[0])
    path[first_anchor] = annual[first_anchor]
    if first_anchor:
        first_change = np.diff(proxy[: first_anchor + 1])
        z = (first_change - np.mean(first_change)) / within_markup_scale
        increments = lambda_weight * quarterly_n_scale * z
        for t in range(first_anchor - 1, -1, -1):
            path[t] = path[t + 1] - increments[t]

    for left, right in zip(anchors[:-1], anchors[1:]):
        left, right = int(left), int(right)
        annual_change = float(annual[right] - annual[left])
        proxy_change = np.diff(proxy[left : right + 1])
        z = (proxy_change - np.mean(proxy_change)) / within_markup_scale
        increments = annual_change / 4.0 + lambda_weight * quarterly_n_scale * z
        for offset, t in enumerate(range(left + 1, right + 1)):
            path[t] = path[t - 1] + increments[offset]
        path[right] = annual[right]

    if not np.all(np.isfinite(path)):
        raise ValueError("The interpolation did not cover the full quarterly sample.")
    anchor_error = float(np.max(np.abs(path[anchors] - annual[anchors])))
    if anchor_error > 1e-10:
        raise RuntimeError("The interpolation failed its exact Q4 anchor invariant.")
    return path, {
        "lambda_weight": float(lambda_weight),
        "within_markup_scale": float(within_markup_scale),
        "annual_change_scale": float(annual_change_scale),
        "quarterly_n_scale": float(quarterly_n_scale),
        "anchor_error": anchor_error,
    }


def sample_hard_anchor_draws(
    observations: np.ndarray,
    *,
    q_scale: float,
    proxy_scale: float,
    iterations: int,
    warmup: int,
    thin: int,
    chains: int,
    seed: int,
    progress_tick: Callable[[], None] | None = None,
) -> dict[str, np.ndarray]:
    """Decompose hard-anchored total-q observations into RW slow and AR(2) fast."""
    observations = np.asarray(observations, dtype=float)
    missing_proxy = np.zeros_like(observations)
    outputs: dict[str, list[np.ndarray]] = {}
    for chain in range(chains):
        result = _run_chain(
            annual=observations,
            proxy=missing_proxy,
            include_proxy=False,
            markup_error="iid",
            q_scale=q_scale,
            proxy_scale=proxy_scale,
            iterations=iterations,
            warmup=warmup,
            thin=thin,
            seed=seed + chain * 1009,
            progress_tick=progress_tick,
        )
        for name, values in result.items():
            outputs.setdefault(name, []).append(values)
    return {name: np.stack(values, axis=0) for name, values in outputs.items()}


def as_measurement_posterior(
    draws: dict[str, np.ndarray],
    baseline_draws: dict[str, np.ndarray],
    *,
    periods: tuple[str, ...],
) -> MeasurementPosterior:
    baseline_sd = np.std(baseline_draws["qhat"], axis=(0, 1), ddof=1)
    augmented_sd = np.std(draws["qhat"], axis=(0, 1), ddof=1)
    ratio = float(np.median(augmented_sd) / np.median(baseline_sd))
    return MeasurementPosterior(
        draws=draws,
        annual_only_draws=baseline_draws,
        information_ratio=ratio,
        periods=periods,
    )
