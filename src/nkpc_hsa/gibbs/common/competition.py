from __future__ import annotations

from typing import Any

import numpy as np


def as_1d_float(x: Any) -> np.ndarray:
    return np.asarray(x, dtype=float).reshape(-1)


def finite_N_residuals(N_obs: Any, Nhat: Any, Nbar: Any) -> np.ndarray:
    """Residuals for observed competition measurement quarters only."""
    n_obs = as_1d_float(N_obs)
    nhat = as_1d_float(Nhat)
    nbar = as_1d_float(Nbar)
    if not (n_obs.size == nhat.size == nbar.size):
        raise ValueError("N_obs, Nhat, and Nbar must have the same length.")
    mask = np.isfinite(n_obs)
    return n_obs[mask] - nhat[mask] - nbar[mask]


def initial_competition_path(N_obs: Any) -> np.ndarray:
    """Fill missing competition observations for sampler initialization only."""
    n_obs = as_1d_float(N_obs)
    if n_obs.size == 0:
        return n_obs.copy()
    finite = np.isfinite(n_obs)
    if not finite.any():
        raise ValueError("At least one finite N observation is required.")
    if finite.all():
        return n_obs.copy()
    idx = np.arange(n_obs.size, dtype=float)
    out = n_obs.copy()
    out[~finite] = np.interp(idx[~finite], idx[finite], n_obs[finite])
    return out
