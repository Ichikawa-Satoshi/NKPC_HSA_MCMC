"""Joint Kalman/FFBS for the firm-count states with MA(q) inflation-equation errors.

Every inflation series in this project is a four-quarter change sampled quarterly, so
``pi_t`` and ``pi_{t-1}`` share three of four quarters. Even if quarterly inflation were
white noise the overlap alone would put ``corr(pi_t, pi_{t-1})`` at 0.75; in the data it
is 0.97. The baseline specification absorbs that into a lagged-inflation coefficient
``alpha``, which is therefore not cleanly interpretable as economic inertia, and the
inflation-equation residual is left serially correlated (Ljung-Box(8) p = 0.00015)
against a likelihood that assumes it is i.i.d.

This module models the overlap directly. The inflation residual is

    eta_t = eps_t + psi_1 eps_{t-1} + ... + psi_q eps_{t-q},    eps ~ N(0, sigma^2) iid,

and the innovations ``eps`` are carried **in the state vector**:

    s_t = (Nhat_t, Nhat_{t-1}, Nbar_t, eps_t, eps_{t-1}, ..., eps_{t-q})'

so the entire MA structure sits in the observation matrix and the measurement error on
the inflation row is (numerically) zero. That keeps the system linear-Gaussian with
*independent* measurement errors, so the ordinary Kalman/FFBS recursion applies and no
state/measurement cross-covariance term is needed.

Because ``eps`` is in the state, the inflation residual is determined once the state is
drawn. The coefficient block must therefore marginalise ``eps`` analytically rather than
condition on it -- that is the GLS step in the caller, using the banded MA covariance.
The two blocks (coefficients with ``eps`` integrated out; states including ``eps``) form
a valid Gibbs partition.
"""
from __future__ import annotations

import numpy as np
from numpy.linalg import inv

__all__ = ["ma_covariance", "sample_joint_competition_states_ffbs_ma"]

# Measurement variance placed on the inflation row. The row is exact given the state, so
# this is only a numerical floor to keep the innovation covariance invertible.
_EXACT_ROW_JITTER = 1e-10


def ma_covariance(psi: np.ndarray, sigma2: float, T: int) -> np.ndarray:
    """Covariance of an MA(q) process with unit leading coefficient.

    ``Omega[i, j] = sigma2 * sum_k c_k c_{k + |i-j|}`` with ``c = (1, psi_1, ..., psi_q)``.
    Banded and Toeplitz, so it is built directly rather than by multiplying filters.
    """
    c = np.concatenate([[1.0], np.asarray(psi, dtype=float).reshape(-1)])
    q = c.size - 1
    gamma = np.array([float(np.dot(c[: c.size - h], c[h:])) for h in range(q + 1)])
    idx = np.abs(np.subtract.outer(np.arange(T), np.arange(T)))
    out = np.zeros((T, T), dtype=float)
    for h in range(q + 1):
        out[idx == h] = gamma[h]
    return sigma2 * out


def _force_pd(S: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    S = (np.asarray(S, dtype=float) + np.asarray(S, dtype=float).T) / 2.0
    vals, vecs = np.linalg.eigh(S)
    vals = np.maximum(vals, eps)
    out = (vecs * vals) @ vecs.T
    return (out + out.T) / 2.0


def sample_joint_competition_states_ffbs_ma(
    *,
    N_obs: np.ndarray,
    y_tilde: np.ndarray,
    h_nbar: np.ndarray,
    psi: np.ndarray,
    n_drift: float,
    rho1: float,
    rho2: float,
    sigma2: float,
    sigma_u2: float,
    sigma_eps2: float,
    sigma_N2: float,
    m0: np.ndarray,
    P0: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Draw ``(Nbar, Nhat, eps)`` jointly with MA(q) inflation-equation errors.

    ``y_tilde`` is the inflation observation net of everything that does not load on the
    state, and ``h_nbar`` its loading on ``Nbar_t`` (that is ``delta * x_t / KAPPA_SCALE``).
    ``sigma2`` is the variance of the MA *innovation*, not of the residual.

    ``m0``/``P0`` describe the three firm-count states only; the MA lags are initialised
    from their stationary distribution ``N(0, sigma2)``.
    """
    N_obs = np.asarray(N_obs, dtype=float).reshape(-1)
    y_tilde = np.asarray(y_tilde, dtype=float).reshape(-1)
    h_nbar = np.asarray(h_nbar, dtype=float).reshape(-1)
    psi = np.asarray(psi, dtype=float).reshape(-1)
    T = N_obs.size
    q = psi.size
    k = 3 + q + 1  # three firm-count states plus eps_t ... eps_{t-q}

    F = np.zeros((k, k))
    F[0, 0], F[0, 1] = rho1, rho2
    F[1, 0] = 1.0
    F[2, 2] = 1.0
    for j in range(q):
        F[4 + j, 3 + j] = 1.0  # shift register for the MA lags

    c = np.zeros(k)
    c[2] = n_drift

    Q = np.zeros((k, k))
    Q[0, 0] = sigma_u2
    Q[2, 2] = sigma_eps2
    Q[3, 3] = sigma2  # eps_t is the fresh innovation at t

    ma_row = np.concatenate([[1.0], psi])  # loading on (eps_t, ..., eps_{t-q})

    m_pred = np.zeros((T, k))
    P_pred = np.zeros((T, k, k))
    m_filt = np.zeros((T, k))
    P_filt = np.zeros((T, k, k))
    Ik = np.eye(k)

    m0_full = np.zeros(k)
    m0_full[:3] = np.asarray(m0, dtype=float).reshape(-1)
    P0_full = np.zeros((k, k))
    P0_full[:3, :3] = np.asarray(P0, dtype=float)
    for j in range(q + 1):
        P0_full[3 + j, 3 + j] = sigma2

    for t in range(T):
        if t == 0:
            m_pred[t] = m0_full
            P_pred[t] = _force_pd(P0_full)
        else:
            m_pred[t] = c + F @ m_filt[t - 1]
            P_pred[t] = _force_pd(F @ P_filt[t - 1] @ F.T + Q)

        pi_row = np.zeros(k)
        pi_row[2] = h_nbar[t]
        pi_row[3:] = ma_row

        if np.isfinite(N_obs[t]):
            n_row = np.zeros(k)
            n_row[0], n_row[2] = 1.0, 1.0
            H = np.vstack([n_row, pi_row])
            z = np.array([N_obs[t], y_tilde[t]])
            R = np.diag([sigma_N2, _EXACT_ROW_JITTER])
        else:
            H = pi_row.reshape(1, k)
            z = np.array([y_tilde[t]])
            R = np.array([[_EXACT_ROW_JITTER]])

        S = _force_pd(H @ P_pred[t] @ H.T + R)
        K = P_pred[t] @ H.T @ inv(S)
        innov = z - H @ m_pred[t]
        m_filt[t] = m_pred[t] + K @ innov
        KH = K @ H
        P_filt[t] = _force_pd((Ik - KH) @ P_pred[t] @ (Ik - KH).T + K @ R @ K.T)

    states = np.zeros((T, k))
    states[-1] = rng.multivariate_normal(m_filt[-1], _force_pd(P_filt[-1]))
    for t in range(T - 2, -1, -1):
        Ptp1 = _force_pd(P_pred[t + 1])
        A = P_filt[t] @ F.T @ inv(Ptp1)
        mean_s = m_filt[t] + A @ (states[t + 1] - c - F @ m_filt[t])
        cov_s = _force_pd(P_filt[t] - A @ Ptp1 @ A.T)
        states[t] = rng.multivariate_normal(mean_s, cov_s)

    return states[:, 2], states[:, 0], states
