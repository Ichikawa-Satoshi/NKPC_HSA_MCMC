from __future__ import annotations

import numpy as np
from scipy.linalg import solve_triangular


def aggregation_matrix(n_quarters: int, width: int = 4) -> np.ndarray:
    """Return the exact moving-average operator used by the YoY likelihood.

    Rows are endpoint ordered.  For width four, row zero averages quarters
    0--3 and corresponds to the fourth quarterly endpoint.
    """
    if width < 1 or n_quarters < width:
        raise ValueError("n_quarters must be at least the positive aggregation width.")
    out = np.zeros((n_quarters - width + 1, n_quarters), dtype=float)
    for row in range(out.shape[0]):
        out[row, row : row + width] = 1.0 / width
    return out


def overlap_covariance(n_quarters: int, width: int = 4) -> np.ndarray:
    A = aggregation_matrix(n_quarters, width)
    return A @ A.T


def whiten_aggregated(y_quarterly: np.ndarray, design_quarterly: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Aggregate the entire quarterly equation and whiten its exact covariance."""
    y = np.asarray(y_quarterly, dtype=float).reshape(-1)
    X = np.asarray(design_quarterly, dtype=float)
    if X.ndim != 2 or X.shape[0] != y.size:
        raise ValueError("Quarterly outcome and design must have matching rows.")
    A = aggregation_matrix(y.size)
    G = A @ A.T
    L = np.linalg.cholesky(G)
    return (
        solve_triangular(L, A @ y, lower=True, check_finite=False),
        solve_triangular(L, A @ X, lower=True, check_finite=False),
    )


def ar1_quarterly_covariance(n: int, rho: float, innovation_variance: float) -> np.ndarray:
    """Stationary quarterly AR(1) covariance used by the declared robustness."""
    if n < 1 or not (-1.0 < rho < 1.0) or innovation_variance <= 0.0:
        raise ValueError("Invalid stationary AR(1) covariance inputs.")
    variance = innovation_variance / (1.0 - rho**2)
    lag = np.abs(np.subtract.outer(np.arange(n), np.arange(n)))
    return variance * rho**lag


def aggregated_ar1_covariance(n: int, rho: float, innovation_variance: float) -> np.ndarray:
    A = aggregation_matrix(n)
    omega = ar1_quarterly_covariance(n, rho, innovation_variance)
    return A @ omega @ A.T
