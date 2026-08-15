"""Semi-modular feedback paths for the quarterly inflation equation.

The measurement posterior is used as the proposal distribution.  Integrating
the inflation-regression coefficients and variance gives ``m_pi(y | q)``.  A
feedback path then has state marginal

    p_lambda(q | D) proportional to p(q | annual N, markup) m_pi(y | q)^lambda.

Thus lambda zero is the modular cut and lambda one is the full-joint state
marginal, provided importance sampling has adequate overlap.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import gammaln, logsumexp, roots_genlaguerre
from scipy.stats import genpareto


@dataclass(frozen=True)
class FeedbackWeights:
    weights: np.ndarray
    raw_ess: float
    entropy_ess: float
    max_weight: float


def log_marginal_regression(
    y: np.ndarray,
    X: np.ndarray,
    prior_sds: np.ndarray,
    *,
    variance_shape: float = 3.0,
    variance_scale: float = 8.0,
    quadrature_nodes: int = 64,
) -> float:
    """Integrate beta and sigma2 under the priors used by the NKPC sampler.

    ``beta ~ N(0, diag(prior_sds**2))`` and
    ``sigma2 ~ inverse-gamma(variance_shape, variance_scale)``.  After beta is
    integrated analytically, generalized Gauss-Laguerre quadrature integrates
    precision ``tau = 1 / sigma2`` under its gamma prior.
    """
    y = np.asarray(y, dtype=float)
    X = np.asarray(X, dtype=float)
    prior_sds = np.asarray(prior_sds, dtype=float)
    if y.ndim != 1 or X.ndim != 2 or X.shape[0] != y.size:
        raise ValueError("y and X have incompatible shapes.")
    if X.shape[1] != prior_sds.size or np.any(prior_sds <= 0.0):
        raise ValueError("prior_sds must be positive and match the columns of X.")
    if variance_shape <= 0.0 or variance_scale <= 0.0 or quadrature_nodes < 8:
        raise ValueError("Invalid inverse-gamma or quadrature settings.")

    prior_variances = prior_sds**2
    prior_precision = np.diag(1.0 / prior_variances)
    xtx = X.T @ X
    xty = X.T @ y
    yty = float(y @ y)
    logdet_prior_variance = float(np.log(prior_variances).sum())
    n = y.size

    nodes, weights = roots_genlaguerre(quadrature_nodes, variance_shape - 1.0)
    taus = nodes / variance_scale
    precisions = prior_precision[None, :, :] + taus[:, None, None] * xtx[None, :, :]
    signs, logdet_precisions = np.linalg.slogdet(precisions)
    if np.any(signs <= 0):
        raise np.linalg.LinAlgError("Regression precision is not positive definite.")
    right_hand_sides = np.broadcast_to(xty, (quadrature_nodes, xty.size))[..., None]
    solved = np.linalg.solve(precisions, right_hand_sides)[..., 0]
    quadratic = taus * yty - taus**2 * np.einsum("ij,j->i", solved, xty)
    logdet_covariance = (
        -n * np.log(taus) + logdet_prior_variance + logdet_precisions
    )
    constant = n * np.log(2.0 * np.pi)
    log_likelihood = -0.5 * (constant + logdet_covariance + quadratic)
    log_terms = np.log(weights) + log_likelihood
    return float(logsumexp(log_terms) - gammaln(variance_shape))


def feedback_weights(log_marginal: np.ndarray, feedback: float) -> FeedbackWeights:
    """Normalize raw importance weights for a requested feedback strength."""
    values = np.asarray(log_marginal, dtype=float).reshape(-1)
    if not 0.0 <= feedback <= 1.0:
        raise ValueError("feedback must lie between zero and one.")
    log_weights = feedback * (values - float(np.max(values)))
    weights = np.exp(log_weights - logsumexp(log_weights))
    raw_ess = float(1.0 / np.sum(weights**2))
    positive = weights > 0.0
    entropy_ess = float(np.exp(-np.sum(weights[positive] * np.log(weights[positive]))))
    return FeedbackWeights(
        weights=weights,
        raw_ess=raw_ess,
        entropy_ess=entropy_ess,
        max_weight=float(np.max(weights)),
    )


def pareto_k_diagnostic(log_weights: np.ndarray) -> float:
    """Estimate the generalized-Pareto tail shape of importance ratios.

    The largest ``min(S/5, 3*sqrt(S))`` ratios are used, matching the usual
    PSIS tail-length rule.  This is a diagnostic only; reported estimates use
    the unmodified raw weights.
    """
    values = np.asarray(log_weights, dtype=float).reshape(-1)
    if values.size < 20 or not np.isfinite(values).all():
        return float("nan")
    ratios = np.exp(values - float(np.max(values)))
    tail_size = int(min(values.size // 5, 3.0 * np.sqrt(values.size)))
    if tail_size < 5:
        return float("nan")
    ordered = np.sort(ratios)
    threshold = float(ordered[-tail_size - 1])
    excess = ordered[-tail_size:] - threshold
    if np.max(excess) <= np.finfo(float).eps:
        return 0.0
    shape, _, _ = genpareto.fit(excess, floc=0.0)
    return float(shape)


def weighted_quantile(
    values: np.ndarray, weights: np.ndarray, probabilities: np.ndarray
) -> np.ndarray:
    """Return linearly interpolated weighted quantiles for one-dimensional data."""
    values = np.asarray(values, dtype=float).reshape(-1)
    weights = np.asarray(weights, dtype=float).reshape(-1)
    probabilities = np.asarray(probabilities, dtype=float)
    if values.size != weights.size or values.size == 0:
        raise ValueError("values and weights must have the same nonzero length.")
    if np.any(weights < 0.0) or not np.isfinite(weights).all():
        raise ValueError("weights must be finite and nonnegative.")
    total = float(weights.sum())
    if total <= 0.0:
        raise ValueError("weights must have positive total mass.")
    order = np.argsort(values)
    sorted_values = values[order]
    sorted_weights = weights[order] / total
    positions = np.cumsum(sorted_weights) - 0.5 * sorted_weights
    return np.interp(probabilities, positions, sorted_values)
