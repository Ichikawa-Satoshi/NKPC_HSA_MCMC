"""Per-period log predictive densities for saved runs, so LOO and WAIC can be formed.

The samplers store only the posterior, and ``az.loo`` / ``az.waic`` need
``log p(obs_t | theta^(s))`` for every draw and period. This module recomputes it
offline from the stored draws -- no re-sampling -- by running the model's own
Kalman filter over the estimation sample and taking the one-step-ahead predictive
density at each period:

    log p(obs_t | obs_{1:t-1}, theta^(s)).

That factorisation, not the state-conditional likelihood, is what makes the
per-period terms a genuine decomposition of the model's marginal likelihood for
the observed data. ``obs_t`` is the vector of rows active at t: the inflation row
always, the firm-count row only where the annual-Q4 design observes it, and the
establishment row in a joint N/E run. The activity innovation
``zeta_t = x_t - phi_1 x_{t-1}`` contributes at the same period, because the AR(1)
for x is part of the estimated model and x is observed data.

The system matrices are built to match ``gibbs/common/joint_ffbs.py`` exactly,
including the initial prior ``(m0, P0)`` taken from the run's own saved priors.

Coverage. ``ces``, ``hsa_steady`` and ``hsa_const_theta`` are exact, and the
joint N/E variants of the latter two are handled by the same six-state filter.
``hsa_dynamic`` (correlated transition and measurement shocks) and ``hsa_full``
(bilinear ``gamma * Nbar_t * Nhat_t``, which is not linear-Gaussian in the state)
are not implemented here; ``pointwise_log_likelihood`` raises
``UnsupportedModel`` for them so a caller reports a gap instead of a wrong number.
"""
from __future__ import annotations

import numpy as np

KAPPA_SCALE = 100.0
_LOG_2PI = float(np.log(2.0 * np.pi))

EXACT_MODELS = ("ces", "hsa_steady", "hsa_const_theta")


class UnsupportedModel(NotImplementedError):
    """Raised for a model whose predictive density this module cannot form."""


def _gaussian_logpdf(value: np.ndarray, variance: float) -> np.ndarray:
    return -0.5 * (_LOG_2PI + np.log(variance) + value**2 / variance)


def _mvn_logpdf(innovation: np.ndarray, covariance: np.ndarray) -> float:
    k = innovation.size
    chol = np.linalg.cholesky(covariance)
    solved = np.linalg.solve(chol, innovation)
    log_det = 2.0 * float(np.sum(np.log(np.diag(chol))))
    return -0.5 * (k * _LOG_2PI + log_det + float(solved @ solved))


def _force_pd(matrix: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    symmetric = 0.5 * (matrix + matrix.T)
    return symmetric + eps * np.eye(symmetric.shape[0])


def _sigma_eta2(draw: dict) -> float:
    """sigma_e^2 = lambda_ez^2 sigma_zeta^2 + sigma_eta^2, as the samplers store it."""
    if "sigma_eta" in draw:
        return float(draw["sigma_eta"]) ** 2
    residual = float(draw["sigma_e"]) ** 2 - float(draw.get("lambda_ez", 0.0)) ** 2 * float(draw["sigma_zeta"]) ** 2
    return max(residual, 1e-12)


def _initial_prior(priors: dict, size: int) -> tuple[np.ndarray, np.ndarray]:
    """(m0, P0) for the state block, in the sampler's own ordering."""
    names = ["Nhat", "Nhat_lag", "Nbar"]
    if size == 6:
        names += ["Ehat", "Ehat_lag", "Ebar"]
    m0 = np.array([float(priors.get(f"m0_{name}", 0.0)) for name in names], dtype=float)
    P0 = np.diag([float(priors.get(f"P0_{name}", 10.0)) for name in names])
    return m0, P0


def _ces_pointwise(draw: dict, data: dict) -> np.ndarray:
    y = data["pi"] - data["pi_expect"]
    a_t = data["pi_prev"] - data["pi_expect"]
    zeta = data["x"] - float(draw["phi_1"]) * data["x_prev"]
    eta = (
        y
        - float(draw["alpha"]) * a_t
        - (float(draw["kappa"]) / KAPPA_SCALE) * data["x"]
        - float(draw.get("lambda_ez", 0.0)) * zeta
    )
    return _gaussian_logpdf(eta, _sigma_eta2(draw)) + _gaussian_logpdf(zeta, float(draw["sigma_zeta"]) ** 2)


def _competition_pointwise(draw: dict, data: dict, priors: dict, *, h_nhat: np.ndarray) -> np.ndarray:
    """Three-state filter shared by hsa_steady (h_nhat = 0) and hsa_const_theta."""
    pi = data["pi"]
    T = pi.size
    x_t = data["x"]
    zeta = x_t - float(draw["phi_1"]) * data["x_prev"]
    sigma_eta2 = _sigma_eta2(draw)
    y_tilde = (
        pi
        - data["pi_expect"]
        - float(draw["alpha"]) * (data["pi_prev"] - data["pi_expect"])
        - (float(draw["kappa_0"]) / KAPPA_SCALE) * x_t
        - float(draw.get("lambda_ez", 0.0)) * zeta
    )
    h_nbar = (float(draw["delta"]) / KAPPA_SCALE) * x_t

    rho1, rho2 = float(draw["rho_1"]), float(draw["rho_2"])
    F = np.array([[rho1, rho2, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=float)
    c = np.array([0.0, 0.0, float(draw["n"])], dtype=float)
    Q = np.diag([float(draw["sigma_u"]) ** 2, 0.0, float(draw["sigma_eps"]) ** 2])
    sigma_N2 = float(draw["sigma_N"]) ** 2

    m, P = _initial_prior(priors, 3)
    N_obs = data["N_obs"]
    out = np.zeros(T, dtype=float)
    identity = np.eye(3)
    for t in range(T):
        if t > 0:
            m = c + F @ m
            P = _force_pd(F @ P @ F.T + Q)
        else:
            P = _force_pd(P)
        pi_row = np.array([h_nhat[t], 0.0, h_nbar[t]], dtype=float)
        if np.isfinite(N_obs[t]):
            H = np.vstack([[1.0, 0.0, 1.0], pi_row])
            values = np.array([N_obs[t], y_tilde[t]], dtype=float)
            R = np.diag([sigma_N2, sigma_eta2])
        else:
            H = pi_row.reshape(1, 3)
            values = np.array([y_tilde[t]], dtype=float)
            R = np.array([[sigma_eta2]], dtype=float)
        S = _force_pd(H @ P @ H.T + R)
        innovation = values - H @ m
        out[t] = _mvn_logpdf(innovation, S)
        K = P @ H.T @ np.linalg.inv(S)
        m = m + K @ innovation
        KH = K @ H
        P = _force_pd((identity - KH) @ P @ (identity - KH).T + K @ R @ K.T)
    return out + _gaussian_logpdf(zeta, float(draw["sigma_zeta"]) ** 2)


def _draw_dicts(posterior) -> tuple[list[dict], int, int]:
    """Flatten the posterior into one dict of scalars per draw."""
    scalars = {name: np.asarray(posterior[name]) for name in posterior.data_vars if posterior[name].ndim == 2}
    if not scalars:
        raise ValueError("The posterior contains no scalar parameters.")
    n_chain, n_draw = next(iter(scalars.values())).shape
    draws = [
        {name: float(values[chain, draw]) for name, values in scalars.items()}
        for chain in range(n_chain)
        for draw in range(n_draw)
    ]
    return draws, n_chain, n_draw


def waic_from_pointwise(log_lik: np.ndarray) -> dict[str, float]:
    """WAIC-2 from a (chain, draw, T) array.

    ArviZ 1.0 removed ``az.waic``, so the standard estimator is computed here:
    ``lppd_t = log mean_s exp(l_st)``, ``p_t = var_s(l_st)``, and
    ``elpd_waic = sum_t (lppd_t - p_t)`` with the usual sqrt(T * var_t) standard
    error.
    """
    flat = np.asarray(log_lik, dtype=float).reshape(-1, np.asarray(log_lik).shape[-1])
    max_per_period = np.max(flat, axis=0)
    lppd = max_per_period + np.log(np.mean(np.exp(flat - max_per_period), axis=0))
    penalty = np.var(flat, axis=0, ddof=1)
    pointwise = lppd - penalty
    n_periods = pointwise.size
    return {
        "elpd_waic": float(np.sum(pointwise)),
        "p_waic": float(np.sum(penalty)),
        "se": float(np.sqrt(n_periods * np.var(pointwise, ddof=1))) if n_periods > 1 else float("nan"),
    }


def pointwise_log_likelihood(model: str, posterior, data: dict, priors: dict) -> np.ndarray:
    """Return a (chain, draw, T) array of per-period log predictive densities.

    ``data`` needs ``pi``, ``pi_prev``, ``pi_expect``, ``x``, ``x_prev`` and
    ``N_obs`` (nan where the firm count is unobserved), i.e. exactly what the
    sampler consumed.
    """
    if model not in EXACT_MODELS:
        raise UnsupportedModel(
            f"{model}: no exact per-period predictive density is implemented "
            "(hsa_dynamic has correlated transition/measurement shocks; hsa_full is "
            "bilinear in the competition states and needs a particle filter)."
        )
    draws, n_chain, n_draw = _draw_dicts(posterior)
    T = np.asarray(data["pi"]).size
    out = np.zeros((len(draws), T), dtype=float)
    for index, draw in enumerate(draws):
        if model == "ces":
            out[index] = _ces_pointwise(draw, data)
        elif model == "hsa_steady":
            out[index] = _competition_pointwise(draw, data, priors, h_nhat=np.zeros(T))
        else:
            theta = float(draw["theta"])
            out[index] = _competition_pointwise(draw, data, priors, h_nhat=np.full(T, -theta))
    return out.reshape(n_chain, n_draw, T)
