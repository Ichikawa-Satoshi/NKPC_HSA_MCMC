from __future__ import annotations

from typing import Any, Optional

import numpy as np
from numpy.linalg import inv


def getd(d: Optional[dict[str, Any]], key: str, default: Any) -> Any:
    if isinstance(d, dict) and key in d and d[key] is not None:
        return d[key]
    return default


def assert_all_pos(arr: Any, msg: str) -> None:
    values = np.asarray(arr, dtype=float)
    if np.any(~np.isfinite(values)) or np.any(values <= 0.0):
        raise ValueError(msg)


def _as_1d(x: Any) -> np.ndarray:
    return np.asarray(x, dtype=float).reshape(-1)


def is_stationary_ar2(r1: float, r2: float) -> bool:
    return (abs(r2) < 1.0) and ((r1 + r2) < 1.0) and ((r2 - r1) < 1.0)


def force_pd(S: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    S = (np.asarray(S, dtype=float) + np.asarray(S, dtype=float).T) / 2.0
    vals, vecs = np.linalg.eigh(S)
    vals = np.maximum(vals, eps)
    repaired = (vecs * vals) @ vecs.T
    return (repaired + repaired.T) / 2.0


def mvnrnd(mean: np.ndarray, cov: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    return rng.multivariate_normal(np.asarray(mean, dtype=float), force_pd(cov))


def sample_invgamma(a_post: float, b_post: float, rng: np.random.Generator) -> float:
    return 1.0 / rng.gamma(shape=a_post, scale=1.0 / b_post)


def sample_beta_gaussian(
    y: np.ndarray,
    X: np.ndarray,
    sigma2: float,
    prior_mean: np.ndarray,
    prior_var: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    y = _as_1d(y)
    X = np.asarray(X, dtype=float)
    prior_mean = _as_1d(prior_mean)
    prior_var = _as_1d(prior_var)

    V0_inv = np.diag(1.0 / prior_var)
    precision = X.T @ X / sigma2 + V0_inv
    rhs = X.T @ y / sigma2 + V0_inv @ prior_mean
    Vn = inv(precision)
    mn = np.linalg.solve(precision, rhs)
    return mvnrnd(mn, Vn, rng)


def sample_ar1_x_draws(
    x: np.ndarray,
    x_prev: np.ndarray,
    *,
    prior_mu: float,
    prior_sd: float,
    n_draws: int,
    rng: np.random.Generator,
    prior_a: float = 0.001,
    prior_b: float = 0.001,
) -> tuple[np.ndarray, np.ndarray]:
    y = _as_1d(x)
    X = _as_1d(x_prev)[:, None]

    phi = np.empty(n_draws, dtype=float)
    sigma2 = np.empty(n_draws, dtype=float)

    phi_curr = float(prior_mu)
    sigma_curr = float(np.var(y - phi_curr * X[:, 0]) + 1e-3)

    for i in range(n_draws):
        phi_curr = sample_beta_gaussian(
            y,
            X,
            sigma2=sigma_curr,
            prior_mean=np.array([prior_mu], dtype=float),
            prior_var=np.array([prior_sd**2], dtype=float),
            rng=rng,
        )[0]
        resid = y - X[:, 0] * phi_curr
        a_post = prior_a + 0.5 * y.size
        b_post = prior_b + 0.5 * float(np.sum(resid**2))
        sigma_curr = sample_invgamma(a_post, b_post, rng)
        phi[i] = phi_curr
        sigma2[i] = sigma_curr

    return phi, sigma2
