"""Exact joint Kalman/FFBS draw of the firm-count state block.

Shared by every HSA specification whose inflation observation is *linear* in the
joint state

    s_t = (Nhat_t, Nhat_{t-1}, Nbar_t)'.

That covers

    hsa_steady       inflation row loading  [0,        0, delta_eff * x_t]
    hsa_const_theta  inflation row loading  [-theta_0, 0, delta_eff * x_t]

and excludes ``hsa_full``, whose bilinear ``-gamma * Nbar_t * Nhat_t`` term makes
the joint state non-linear-Gaussian (that model needs alternating conditional
blocks or a Particle-Gibbs sweep instead).

Why this matters. ``N_obs_t = Nbar_t + Nhat_t + nu_t`` with a small sigma_N pins
the *sum* of the two states almost exactly while leaving the split nearly
unidentified: the posterior correlation between Nbar_0 and Nhat_0 is about
-0.999 in the estimated cells. A two-block Gibbs that alternates
``Nhat | Nbar`` and ``Nbar | Nhat`` moves the shared level with autocorrelation
rho^2 per sweep, which at rho = -0.999 is an integrated autocorrelation time of
order 10^3 sweeps. Drawing the whole state jointly sidesteps that geometry
entirely -- it is the reason ``hsa_steady`` mixes and the alternating blocks do
not.

The numerics here are a verbatim generalisation of the routine that
``hsa_steady`` has always used (Joseph-form covariance update, ``_force_pd``
repair with eps = 1e-10, and the same order of random draws), so delegating
``hsa_steady`` to this module reproduces its previous draws bit for bit given
the same seed. ``tests/test_joint_ffbs.py`` pins that equivalence.
"""

from __future__ import annotations

import numpy as np
from numpy.linalg import inv

__all__ = ["force_pd", "sample_joint_competition_states_ffbs"]


def force_pd(S: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    """Symmetrise and clip eigenvalues, matching the historical hsa_steady form."""
    S = (np.asarray(S, dtype=float) + np.asarray(S, dtype=float).T) / 2.0
    vals, vecs = np.linalg.eigh(S)
    vals = np.maximum(vals, eps)
    return vecs @ np.diag(vals) @ vecs.T


def _mvnrnd(mean: np.ndarray, cov: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    return rng.multivariate_normal(np.asarray(mean, dtype=float), force_pd(cov))


def sample_joint_competition_states_ffbs(
    *,
    N_obs: np.ndarray,
    y_tilde: np.ndarray,
    h_nhat: np.ndarray,
    h_nbar: np.ndarray,
    n_drift: float,
    rho1: float,
    rho2: float,
    sigma_eta2: float,
    sigma_u2: float,
    sigma_eps2: float,
    sigma_N2: float,
    m0: np.ndarray,
    P0: np.ndarray,
    rng: np.random.Generator,
    Ehat_obs: np.ndarray | None = None,
    lambda_E: float = 0.0,
    sigma_E2: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Draw ``s_{0:T-1}`` jointly from its exact smoothing posterior.

    Parameters
    ----------
    N_obs
        Firm-count observation, ``nan`` where unobserved (the annual-Q4 design
        leaves Q1-Q3 missing). Missing quarters simply drop the firm-count row;
        prediction and the inflation row still run at every quarter.
    y_tilde
        Inflation observation net of every term that does not load on the state,
        i.e. ``y_t - alpha*a_t - kappa0_eff*x_t - lambda_ez*zeta_t``.
    h_nhat, h_nbar
        Per-period loadings of the inflation row on ``Nhat_t`` and ``Nbar_t``.
        ``hsa_steady`` passes ``h_nhat = 0``; ``hsa_const_theta`` passes
        ``h_nhat = -theta_0``. Both pass ``h_nbar = delta_eff * x_t``.

    Returns
    -------
    (Nbar, Nhat, states) where ``states[:, 1]`` carries the sampled ``Nhat_{t-1}``
    so callers can use ``states[0, 1]`` as the drawn ``Nhat_{-1}`` initial lag.

    State equation
    --------------
        Nhat_t = rho1*Nhat_{t-1} + rho2*Nhat_{t-2} + u_t
        Nbar_t = n + Nbar_{t-1} + epsilon_t

    Observation rows
    ----------------
        N_obs_t   = [1, 0, 1] s_t + nu_t                     var sigma_N^2
        y_tilde_t = [h_nhat_t, 0, h_nbar_t] s_t + eta_t      var sigma_eta^2
        Ehat_obs_t = [lambda_E, 0, 0] s_t + omega_t           var sigma_E^2

    The optional third row is the experimental establishment-cycle measurement
    equation.  Omitting ``Ehat_obs`` preserves the production two-row filter
    exactly, including its random-number stream.
    """
    N_obs = np.asarray(N_obs, dtype=float).reshape(-1)
    y_tilde = np.asarray(y_tilde, dtype=float).reshape(-1)
    h_nhat = np.broadcast_to(np.asarray(h_nhat, dtype=float), N_obs.shape)
    h_nbar = np.broadcast_to(np.asarray(h_nbar, dtype=float), N_obs.shape)
    Ehat = None if Ehat_obs is None else np.asarray(Ehat_obs, dtype=float).reshape(-1)

    T = N_obs.size
    if not (y_tilde.size == T):
        raise ValueError("All input series must have the same length.")
    if Ehat is not None and Ehat.size != T:
        raise ValueError("Ehat_obs must have the same length as N_obs.")
    if Ehat is not None and (not np.isfinite(sigma_E2) or sigma_E2 <= 0.0):
        raise ValueError("sigma_E2 must be finite and positive when Ehat_obs is supplied.")

    F = np.array(
        [
            [rho1, rho2, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    c = np.array([0.0, 0.0, n_drift], dtype=float)
    Q = np.diag([sigma_u2, 0.0, sigma_eps2])

    m_pred = np.zeros((T, 3), dtype=float)
    P_pred = np.zeros((T, 3, 3), dtype=float)
    m_filt = np.zeros((T, 3), dtype=float)
    P_filt = np.zeros((T, 3, 3), dtype=float)
    I3 = np.eye(3)

    # ---------- Forward Kalman filter ----------
    for t in range(T):
        if t == 0:
            m_pred[t] = np.asarray(m0, dtype=float).reshape(-1)
            P_pred[t] = force_pd(P0)
        else:
            m_pred[t] = c + F @ m_filt[t - 1]
            P_pred[t] = force_pd(F @ P_filt[t - 1] @ F.T + Q)

        pi_row = np.array([h_nhat[t], 0.0, h_nbar[t]], dtype=float)

        if Ehat is None:
            # Keep the production path byte-for-byte identical when the
            # experimental establishment row is absent.
            if np.isfinite(N_obs[t]):
                y_obs = np.array([N_obs[t], y_tilde[t]], dtype=float)
                H = np.vstack([[1.0, 0.0, 1.0], pi_row])
                R = np.diag([sigma_N2, sigma_eta2])
            else:
                y_obs = np.array([y_tilde[t]], dtype=float)
                H = pi_row.reshape(1, 3)
                R = np.array([[sigma_eta2]], dtype=float)
        elif np.isfinite(N_obs[t]) and np.isfinite(Ehat[t]):
            y_obs = np.array([N_obs[t], y_tilde[t], Ehat[t]], dtype=float)
            H = np.vstack([[1.0, 0.0, 1.0], pi_row, [lambda_E, 0.0, 0.0]])
            R = np.diag([sigma_N2, sigma_eta2, sigma_E2])
        elif np.isfinite(N_obs[t]):
            y_obs = np.array([N_obs[t], y_tilde[t]], dtype=float)
            H = np.vstack([[1.0, 0.0, 1.0], pi_row])
            R = np.diag([sigma_N2, sigma_eta2])
        elif np.isfinite(Ehat[t]):
            y_obs = np.array([y_tilde[t], Ehat[t]], dtype=float)
            H = np.vstack([pi_row, [lambda_E, 0.0, 0.0]])
            R = np.diag([sigma_eta2, sigma_E2])
        else:
            y_obs = np.array([y_tilde[t]], dtype=float)
            H = pi_row.reshape(1, 3)
            R = np.array([[sigma_eta2]], dtype=float)

        S = force_pd(H @ P_pred[t] @ H.T + R)
        K = P_pred[t] @ H.T @ inv(S)
        innov = y_obs - H @ m_pred[t]
        m_filt[t] = m_pred[t] + K @ innov

        # Joseph form for numerical stability
        KH = K @ H
        P_filt[t] = force_pd((I3 - KH) @ P_pred[t] @ (I3 - KH).T + K @ R @ K.T)

    # ---------- Backward sampling ----------
    states = np.zeros((T, 3), dtype=float)
    states[-1] = _mvnrnd(m_filt[-1], P_filt[-1], rng)

    for t in range(T - 2, -1, -1):
        Ptp1 = force_pd(P_pred[t + 1])
        A = P_filt[t] @ F.T @ inv(Ptp1)
        mean_s = m_filt[t] + A @ (states[t + 1] - c - F @ m_filt[t])
        cov_s = force_pd(P_filt[t] - A @ Ptp1 @ A.T)
        states[t] = _mvnrnd(mean_s, cov_s, rng)

    return states[:, 2], states[:, 0], states
