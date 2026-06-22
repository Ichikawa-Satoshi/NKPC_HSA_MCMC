from __future__ import annotations

from typing import Literal

import numpy as np

NTransform = Literal["log100", "log", "identity"]


def transform_competition_series(N: np.ndarray, transform: NTransform = "log100") -> np.ndarray:
    """Transform the competition/aggregator series used by HSA models.

    The canonical HSA convention is ``100 * log(N)`` for strictly positive level
    series. If a model-ready series is already transformed, pass
    ``transform="identity"`` explicitly.
    """
    arr = np.asarray(N, dtype=float).reshape(-1)
    if np.any(~np.isfinite(arr)):
        raise ValueError("N contains NaN or infinite values.")
    if transform == "identity":
        return arr.copy()
    if np.any(arr <= 0.0):
        raise ValueError("N must be strictly positive before log transformation.")
    if transform == "log100":
        return 100.0 * np.log(arr)
    if transform == "log":
        return np.log(arr)
    raise ValueError(f"Unknown N transform: {transform}")
