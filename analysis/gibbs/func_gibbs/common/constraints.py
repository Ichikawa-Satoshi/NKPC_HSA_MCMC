from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import numpy as np


def constraint_enabled(constraints: Mapping[str, Any] | None) -> bool:
    return bool(isinstance(constraints, Mapping) and constraints.get("enabled"))


def draw_with_constraints(
    draw_fn: Callable[[], np.ndarray],
    names: list[str] | tuple[str, ...],
    constraints: Mapping[str, Any] | None,
    *,
    stats: dict[str, int] | None = None,
) -> np.ndarray:
    """Draw from ``draw_fn`` and reject draws violating hard bounds.

    ``constraints`` must use sampler-internal units. The public wrapper converts
    kappa-like physical-unit bounds before reaching this helper.
    """
    if not constraint_enabled(constraints):
        draw = np.asarray(draw_fn(), dtype=float)
        if stats is not None:
            stats["attempts"] = stats.get("attempts", 0) + 1
        return draw

    bounds = dict(constraints.get("bounds", {}) or {})
    max_tries = int(constraints.get("max_tries", 1000))
    if max_tries <= 0:
        raise ValueError("coefficient_constraints.max_tries must be positive.")

    relevant = {
        name: bounds[name]
        for name in names
        if name in bounds
    }
    if not relevant:
        draw = np.asarray(draw_fn(), dtype=float)
        if stats is not None:
            stats["attempts"] = stats.get("attempts", 0) + 1
        return draw

    name_to_idx = {name: i for i, name in enumerate(names)}
    for attempt in range(1, max_tries + 1):
        draw = np.asarray(draw_fn(), dtype=float)
        ok = True
        for name, pair in relevant.items():
            lower, upper = pair
            value = float(draw[name_to_idx[name]])
            if lower is not None and value < float(lower):
                ok = False
                break
            if upper is not None and value > float(upper):
                ok = False
                break
        if ok:
            if stats is not None:
                stats["attempts"] = stats.get("attempts", 0) + attempt
                stats["rejections"] = stats.get("rejections", 0) + attempt - 1
            return draw

    raise RuntimeError(
        f"Failed to draw coefficients satisfying hard constraints after {max_tries} tries. "
        "Use weaker priors, wider bounds, or disable coefficient_constraints."
    )


def constraint_stats_summary(stats: Mapping[str, int] | None) -> dict[str, float | int]:
    attempts = int((stats or {}).get("attempts", 0))
    rejections = int((stats or {}).get("rejections", 0))
    return {
        "attempts": attempts,
        "rejections": rejections,
        "rejection_rate": 0.0 if attempts <= 0 else rejections / attempts,
    }
