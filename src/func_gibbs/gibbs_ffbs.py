from __future__ import annotations

import numpy as np
from numpy.linalg import inv

try:
    from .gibbs_utils import _as_1d, force_pd, mvnrnd
except ImportError:  # pragma: no cover
    from gibbs_utils import _as_1d, force_pd, mvnrnd


def _coerce_kappa_series(
    T: int,
    *,
    kappa: float | None = None,
    kappa_t: np.ndarray | None = None,
) -> np.ndarray:
    if kappa_t is not None:
        arr = _as_1d(kappa_t)
        if arr.size != T:
            raise ValueError(f"kappa_t must have length T={T}, got {arr.size}.")
        return arr
    if kappa is None:
        raise ValueError("Either kappa or kappa_t must be provided.")
    return np.full(T, float(kappa), dtype=float)


def sample_ar2_states_ffbs(
    y_target: np.ndarray,
    rho1: float,
    rho2: float,
    sigma_state2: float,
    pi_t: np.ndarray,
    alpha: float,
    pi_tm1: np.ndarray,
    pi_expect: np.ndarray,
    x_t: np.ndarray,
    theta: float,
    sigma_obs2: float,
    target_scale: float,
    rng: np.random.Generator,
    *,
    kappa: float | None = None,
    kappa_t: np.ndarray | None = None,
    obs_offset: np.ndarray | None = None,
) -> np.ndarray:
    y_target = _as_1d(y_target)
    pi_t = _as_1d(pi_t)
    pi_tm1 = _as_1d(pi_tm1)
    pi_expect = _as_1d(pi_expect)
    x_t = _as_1d(x_t)
    T = y_target.size
    obs_offset_arr = np.zeros(T, dtype=float) if obs_offset is None else _as_1d(obs_offset)

    if T < 3:
        return y_target.copy()

    kappa_series = _coerce_kappa_series(T, kappa=kappa, kappa_t=kappa_t)

    F = np.array([[rho1, rho2], [1.0, 0.0]], dtype=float)
    Q = np.array([[sigma_state2, 0.0], [0.0, 0.0]], dtype=float)

    m = np.zeros((2, T), dtype=float)
    P = np.zeros((2, 2, T), dtype=float)
    m_pred = np.zeros((2, T), dtype=float)
    P_pred = np.zeros((2, 2, T), dtype=float)

    m[:, 0] = np.array([y_target[0], 0.0], dtype=float)
    P[:, :, 0] = np.eye(2) * 10.0

    for t in range(1, T):
        m_pred[:, t] = F @ m[:, t - 1]
        P_pred[:, :, t] = F @ P[:, :, t - 1] @ F.T + Q

        H = np.array([[1.0, 0.0], [theta, 0.0]], dtype=float)
        y = np.array(
            [
                y_target[t],
                alpha * pi_tm1[t]
                + (1.0 - alpha) * pi_expect[t]
                + kappa_series[t] * x_t[t]
                + obs_offset_arr[t]
                - pi_t[t],
            ],
            dtype=float,
        )
        R = np.diag([max(sigma_state2 * target_scale, 1e-10), max(sigma_obs2, 1e-10)])

        S = H @ P_pred[:, :, t] @ H.T + R
        K = P_pred[:, :, t] @ H.T @ inv(S)

        m[:, t] = m_pred[:, t] + K @ (y - H @ m_pred[:, t])
        P[:, :, t] = force_pd(P_pred[:, :, t] - K @ H @ P_pred[:, :, t])

    states = np.zeros((2, T), dtype=float)
    states[:, -1] = mvnrnd(m[:, -1], P[:, :, -1], rng)

    for t in range(T - 2, -1, -1):
        Ptp1_pred = F @ P[:, :, t] @ F.T + Q
        A = P[:, :, t] @ F.T @ inv(Ptp1_pred)
        m_s = m[:, t] + A @ (states[:, t + 1] - F @ m[:, t])
        P_s = force_pd(P[:, :, t] - A @ (Ptp1_pred - P[:, :, t + 1]) @ A.T)
        states[:, t] = mvnrnd(m_s, P_s, rng)

    return states[0]


def sample_rw_states_ffbs(
    y_target: np.ndarray,
    n_drift: float,
    sigma_state2: float,
    target_scale: float,
    rng: np.random.Generator,
) -> np.ndarray:
    y_target = _as_1d(y_target)
    T = y_target.size

    if T < 2:
        return y_target.copy()

    m = np.zeros(T, dtype=float)
    P = np.zeros(T, dtype=float)
    m_pred = np.zeros(T, dtype=float)
    P_pred = np.zeros(T, dtype=float)

    m[0] = y_target[0]
    P[0] = 10.0

    for t in range(1, T):
        m_pred[t] = n_drift + m[t - 1]
        P_pred[t] = P[t - 1] + sigma_state2
        R = max(sigma_state2 * target_scale, 1e-10)
        K = P_pred[t] / (P_pred[t] + R)
        m[t] = m_pred[t] + K * (y_target[t] - m_pred[t])
        P[t] = (1.0 - K) * P_pred[t]

    states = np.zeros(T, dtype=float)
    states[-1] = m[-1] + np.sqrt(max(P[-1], 1e-8)) * rng.standard_normal()

    for t in range(T - 2, -1, -1):
        Ptp1_pred = P[t] + sigma_state2
        A = P[t] / Ptp1_pred
        m_s = m[t] + A * (states[t + 1] - (n_drift + m[t]))
        P_s = max(P[t] - A * Ptp1_pred * A, 1e-10)
        states[t] = m_s + np.sqrt(P_s) * rng.standard_normal()

    return states
