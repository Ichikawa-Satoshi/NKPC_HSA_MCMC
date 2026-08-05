from __future__ import annotations

from typing import Literal

import numpy as np

NTransform = Literal["log100_centered10", "log100", "log", "identity"]
DEFAULT_N_TRANSFORM: NTransform = "log100_centered10"


def competition_transform_note(transform: str) -> str:
    if transform == "log100_centered10":
        return (
            "N is transformed as (100*log(N_level) - sample mean)/10. "
            "One transformed unit is a ten log-point movement around the sample mean."
        )
    if transform == "log100":
        return "N is transformed as 100*log(N_level); coefficients are per one log-point unit."
    if transform == "log":
        return "N is transformed as log(N_level); coefficients are per one natural-log unit."
    if transform == "identity":
        return "N is passed through unchanged; metadata should document the supplied units."
    return f"Unknown N transform: {transform}"


def transform_competition_series(N: np.ndarray, transform: NTransform = DEFAULT_N_TRANSFORM) -> np.ndarray:
    """Transform the competition/aggregator series used by HSA models.

    The default convention estimates HSA coefficients using
    ``(100 * log(N) - sample mean) / 10``. This keeps the decomposition
    ``N_t = Nhat_t + Nbar_t`` but makes the competition components mean-centered
    and measured in ten log-point units. If a model-ready series is already
    transformed, pass ``transform="identity"`` explicitly.
    """
    arr = np.asarray(N, dtype=float).reshape(-1)
    if np.any(~np.isfinite(arr)):
        raise ValueError("N contains NaN or infinite values.")
    if transform == "identity":
        return arr.copy()
    if np.any(arr <= 0.0):
        raise ValueError("N must be strictly positive before log transformation.")
    if transform == "log100_centered10":
        raw = 100.0 * np.log(arr)
        return (raw - float(np.mean(raw))) / 10.0
    if transform == "log100":
        return 100.0 * np.log(arr)
    if transform == "log":
        return np.log(arr)
    raise ValueError(f"Unknown N transform: {transform}")
