"""Joint Kalman/FFBS state draw with an MA(q) inflation disturbance.

The additive counterpart of ``nkpc_hsa.gibbs.common.joint_ffbs``, which is left
untouched. The difference is the state vector. Production carries

    s_t = (Nhat_t, Nhat_{t-1}, Nbar_t)

and puts the inflation disturbance in the observation noise. That only works
when the disturbance is serially independent. With ``xi_t = psi(L) v_t`` the
disturbance is *not* independent across periods, so it cannot stay in the
measurement equation -- it has to move into the state:

    s_t = (Nhat_t, Nhat_{t-1}, Nbar_t, v_t, v_{t-1}, ..., v_{t-q})

The inflation row then loads ``[h_nhat, 0, h_nbar, 1, psi_1, ..., psi_q]`` and
carries **no** measurement noise of its own; the whole disturbance is in the
state. The firm-count row is unchanged.

Two properties worth stating, because they are what make the augmentation exact
rather than approximate:

* ``P0`` for the ``v`` block is ``sigma_v^2 * I``. The ``v`` are i.i.d., so that
  is the exact stationary initial covariance -- unlike the AR blocks there is no
  initialisation to approximate and no pre-sample to condition away.
* The inflation row's innovation variance is ``h_v' P_v h_v >= sigma_v^2 > 0``
  even with zero measurement noise, so the Kalman gain is always well defined.

The ``v``-lag components of the state are deterministic copies of earlier
``v_t`` draws, which makes the backward-sampling covariance singular by
construction. ``force_pd`` clips it exactly as the production routine does; the
resulting O(1e-10) jitter on the redundant coordinates is harmless and does not
touch the ``(Nhat, Nbar)`` block that callers actually use.

With ``psi = []`` the v block collapses to a single state carrying an i.i.d.
disturbance, which is distributionally identical to the production routine
(though not draw-for-draw: a 4-dimensional multivariate normal consumes a
different number of random numbers than a 3-dimensional one).
"""

from __future__ import annotations

import numpy as np
from numpy.linalg import inv

from nkpc_hsa.error_robustness.ma_error import state_augmentation

__all__ = [
    "joint_loglik_ma",
    "force_pd",
    "sample_joint_states_ffbs_ma",
    "sample_joint_states_ffbs_dynamic_ma",
]


def force_pd(S: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    """Symmetrise and clip eigenvalues, matching the production hsa_steady form."""
    S = (np.asarray(S, dtype=float) + np.asarray(S, dtype=float).T) / 2.0
    vals, vecs = np.linalg.eigh(S)
    vals = np.maximum(vals, eps)
    return vecs @ np.diag(vals) @ vecs.T


def _mvnrnd(mean: np.ndarray, cov: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    return rng.multivariate_normal(np.asarray(mean, dtype=float), force_pd(cov))


def sample_joint_states_ffbs_ma(
    *,
    N_obs: np.ndarray,
    y_tilde: np.ndarray,
    h_nhat: np.ndarray,
    h_nbar: np.ndarray,
    n_drift: float,
    rho1: float,
    rho2: float,
    psi: np.ndarray,
    sigma_v2: float,
    sigma_u2: float,
    sigma_eps2: float,
    sigma_N2: float,
    m0: np.ndarray,
    P0: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Draw the joint firm-count and MA-innovation state from its exact smoothing posterior.

    Parameters
    ----------
    y_tilde
        Inflation observation net of every term that does not load on the
        state, i.e. ``y - alpha*a_t - kappa0_eff*x_t - lambda_ez*zeta_t``.
    h_nhat, h_nbar
        Per-period loadings of the inflation row on ``Nhat_t`` and ``Nbar_t``.
        ``hsa_steady`` passes ``h_nhat = 0`` and ``h_nbar = delta_eff * x_t``.
    psi
        MA coefficients; may be empty for the i.i.d. case.
    m0, P0
        Prior mean and covariance for the *firm-count* block only (length 3 and
        3x3). The ``v`` block's exact stationary initialisation is appended here.

    Returns
    -------
    ``(Nbar, Nhat, states, v_path)`` where ``states`` is the full augmented
    state array and ``v_path = states[:, 3]`` is the drawn innovation sequence,
    which the caller needs for the ``sigma_v^2`` and ``psi`` blocks.
    """
    N_obs = np.asarray(N_obs, dtype=float).reshape(-1)
    y_tilde = np.asarray(y_tilde, dtype=float).reshape(-1)
    h_nhat = np.broadcast_to(np.asarray(h_nhat, dtype=float), N_obs.shape)
    h_nbar = np.broadcast_to(np.asarray(h_nbar, dtype=float), N_obs.shape)

    T = N_obs.size
    if y_tilde.size != T:
        raise ValueError("All input series must have the same length.")

    F_v, Q_v, h_v, P0_v = state_augmentation(psi, sigma_v2)
    dim_v = h_v.size
    dim = 3 + dim_v

    F = np.zeros((dim, dim), dtype=float)
    F[:3, :3] = [[rho1, rho2, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    F[3:, 3:] = F_v

    c = np.zeros(dim, dtype=float)
    c[2] = n_drift

    Q = np.zeros((dim, dim), dtype=float)
    Q[0, 0] = sigma_u2
    Q[2, 2] = sigma_eps2
    Q[3:, 3:] = Q_v

    m_init = np.zeros(dim, dtype=float)
    m_init[:3] = np.asarray(m0, dtype=float).reshape(-1)
    P_init = np.zeros((dim, dim), dtype=float)
    P_init[:3, :3] = np.asarray(P0, dtype=float)
    P_init[3:, 3:] = P0_v

    n_row = np.zeros(dim, dtype=float)
    n_row[0] = 1.0
    n_row[2] = 1.0

    m_pred = np.zeros((T, dim), dtype=float)
    P_pred = np.zeros((T, dim, dim), dtype=float)
    m_filt = np.zeros((T, dim), dtype=float)
    P_filt = np.zeros((T, dim, dim), dtype=float)
    Id = np.eye(dim)

    # ---------- Forward Kalman filter ----------
    for t in range(T):
        if t == 0:
            m_pred[t] = m_init
            P_pred[t] = force_pd(P_init)
        else:
            m_pred[t] = c + F @ m_filt[t - 1]
            P_pred[t] = force_pd(F @ P_filt[t - 1] @ F.T + Q)

        pi_row = np.zeros(dim, dtype=float)
        pi_row[0] = h_nhat[t]
        pi_row[2] = h_nbar[t]
        pi_row[3:] = h_v

        if np.isfinite(N_obs[t]):
            y_obs = np.array([N_obs[t], y_tilde[t]], dtype=float)
            H = np.vstack([n_row, pi_row])
            # Zero measurement noise on the inflation row: the disturbance is in
            # the state now. S stays invertible because h_v' P_v h_v >= sigma_v2.
            R = np.diag([sigma_N2, 0.0])
        else:
            y_obs = np.array([y_tilde[t]], dtype=float)
            H = pi_row.reshape(1, dim)
            R = np.zeros((1, 1), dtype=float)

        S = force_pd(H @ P_pred[t] @ H.T + R)
        K = P_pred[t] @ H.T @ inv(S)
        innov = y_obs - H @ m_pred[t]
        m_filt[t] = m_pred[t] + K @ innov

        KH = K @ H
        P_filt[t] = force_pd((Id - KH) @ P_pred[t] @ (Id - KH).T + K @ R @ K.T)

    # ---------- Backward sampling ----------
    states = np.zeros((T, dim), dtype=float)
    states[-1] = _mvnrnd(m_filt[-1], P_filt[-1], rng)

    for t in range(T - 2, -1, -1):
        Ptp1 = force_pd(P_pred[t + 1])
        A = P_filt[t] @ F.T @ inv(Ptp1)
        mean_s = m_filt[t] + A @ (states[t + 1] - c - F @ m_filt[t])
        cov_s = force_pd(P_filt[t] - A @ Ptp1 @ A.T)
        states[t] = _mvnrnd(mean_s, cov_s, rng)

    return states[:, 2], states[:, 0], states, states[:, 3]


def sample_joint_states_ffbs_dynamic_ma(
    *,
    N_obs: np.ndarray,
    pi_t: np.ndarray,
    pi_tm1: np.ndarray,
    pi_expect: np.ndarray,
    x_t: np.ndarray,
    zeta: np.ndarray,
    alpha: float,
    kappa: float,
    theta: float,
    rho1: float,
    rho2: float,
    n_drift: float,
    Sigma: np.ndarray,
    psi: np.ndarray,
    sigma_N2: float,
    kappa_scale: float,
    m0: np.ndarray,
    P0: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """hsa_dynamic's joint FFBS with the MA(q) innovation carried in the state.

    Production (``gibbs.hsa_dynamic.model._sample_states_joint_ffbs_fullSigma``)
    keeps the inflation disturbance ``e_t`` in the measurement equation. Because
    ``e_t`` is correlated with the state innovations ``u_t`` and ``epsilon_t``
    through the 4x4 ``Sigma``, that forces a Kalman recursion with a non-zero
    cross-covariance between measurement error and state innovation -- the
    ``C_base`` term.

    With ``e_t = psi(L) v_t`` the disturbance cannot stay in the measurement
    equation at all, so ``v`` moves into the state:

        s_t = (Nhat_t, Nhat_{t-1}, Nbar_t, v_t, v_{t-1}, ..., v_{t-q})

    That makes the recursion *simpler* than production's, not harder: once
    ``v_t`` is a state innovation alongside ``u_t`` and ``epsilon_t``, their
    correlation is just an off-diagonal block of ``Q`` and the cross-covariance
    machinery disappears. The inflation row carries no measurement noise.

    Conditioning matches production exactly. ``x_t`` is observed, so ``zeta_t``
    is known given ``phi``; the recursion therefore uses
    ``[v_t, u_t, epsilon_t] | zeta_t``, whose mean shifts the state intercept
    period by period and whose covariance is ``Q``.

    Returns ``(Nbar, Nhat, states, v_path)``.
    """
    N_obs = np.asarray(N_obs, dtype=float).reshape(-1)
    pi_t = np.asarray(pi_t, dtype=float).reshape(-1)
    pi_tm1 = np.asarray(pi_tm1, dtype=float).reshape(-1)
    pi_expect = np.asarray(pi_expect, dtype=float).reshape(-1)
    x_t = np.asarray(x_t, dtype=float).reshape(-1)
    zeta = np.asarray(zeta, dtype=float).reshape(-1)
    Sigma = np.asarray(Sigma, dtype=float)

    T = N_obs.size
    if not (pi_t.size == pi_tm1.size == pi_expect.size == x_t.size == zeta.size == T):
        raise ValueError("All series must have the same length.")

    # [v_t, u_t, epsilon_t] | zeta_t -- identical algebra to production, with
    # Sigma's first coordinate now read as the MA innovation rather than as e_t.
    idx_r, idx_z = [0, 2, 3], [1]
    Szz = float(Sigma[1, 1])
    if Szz <= 0.0:
        raise ValueError("Sigma[1,1] must be positive.")
    B = Sigma[np.ix_(idx_r, idx_z)] / Szz
    S_r = force_pd(
        Sigma[np.ix_(idx_r, idx_r)]
        - Sigma[np.ix_(idx_r, idx_z)] @ Sigma[np.ix_(idx_z, idx_r)] / Szz
    )
    means_r = zeta.reshape(-1, 1) @ B.T
    mu_v, mu_u, mu_eps = means_r[:, 0], means_r[:, 1], means_r[:, 2]

    _F_v, _Q_v, h_v, _P0_v = state_augmentation(psi, 1.0)
    dim_v = h_v.size
    dim = 3 + dim_v

    F = np.zeros((dim, dim), dtype=float)
    F[:3, :3] = [[rho1, rho2, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    if dim_v > 1:
        F[4:, 3:-1] = np.eye(dim_v - 1)

    # State innovation covariance. Positions 0, 2, 3 carry u_t, epsilon_t, v_t;
    # S_r is ordered [v, u, eps], so map it in rather than copying blindly.
    Q = np.zeros((dim, dim), dtype=float)
    order = {0: 1, 2: 2, 3: 0}  # state index -> row of S_r
    for si, ri in order.items():
        for sj, rj in order.items():
            Q[si, sj] = S_r[ri, rj]

    y_pi = pi_t - pi_expect - alpha * (pi_tm1 - pi_expect) - (kappa / kappa_scale) * x_t

    n_row = np.zeros(dim, dtype=float)
    n_row[0] = 1.0
    n_row[2] = 1.0
    pi_row = np.zeros(dim, dtype=float)
    pi_row[0] = -theta
    pi_row[3:] = h_v

    m_pred = np.zeros((T, dim), dtype=float)
    P_pred = np.zeros((T, dim, dim), dtype=float)
    m_filt = np.zeros((T, dim), dtype=float)
    P_filt = np.zeros((T, dim, dim), dtype=float)
    Id = np.eye(dim)

    # ---------- Forward filter ----------
    for t in range(T):
        if t == 0:
            m_pred[t] = np.zeros(dim)
            m_pred[t][:3] = np.asarray(m0, dtype=float).reshape(-1)
            # v_0 given zeta_0 has the same conditional moments as any period.
            # The pre-sample lags are conditioned on nothing, so they carry the
            # unconditional innovation variance Sigma[0, 0].
            m_pred[t][3] = mu_v[0]
            P_init = np.zeros((dim, dim), dtype=float)
            P_init[:3, :3] = np.asarray(P0, dtype=float)
            P_init[3, 3] = S_r[0, 0]
            for j in range(4, dim):
                P_init[j, j] = float(Sigma[0, 0])
            P_pred[t] = force_pd(P_init)
        else:
            c_t = np.zeros(dim, dtype=float)
            c_t[0] = mu_u[t]
            c_t[2] = n_drift + mu_eps[t]
            c_t[3] = mu_v[t]
            m_pred[t] = c_t + F @ m_filt[t - 1]
            P_pred[t] = force_pd(F @ P_filt[t - 1] @ F.T + Q)

        if np.isfinite(N_obs[t]):
            y_obs = np.array([N_obs[t], y_pi[t]], dtype=float)
            H = np.vstack([n_row, pi_row])
            R = np.diag([sigma_N2, 0.0])
        else:
            y_obs = np.array([y_pi[t]], dtype=float)
            H = pi_row.reshape(1, dim)
            R = np.zeros((1, 1), dtype=float)

        S = force_pd(H @ P_pred[t] @ H.T + R)
        K = P_pred[t] @ H.T @ inv(S)
        m_filt[t] = m_pred[t] + K @ (y_obs - H @ m_pred[t])
        KH = K @ H
        P_filt[t] = force_pd((Id - KH) @ P_pred[t] @ (Id - KH).T + K @ R @ K.T)

    # ---------- Backward sampling ----------
    states = np.zeros((T, dim), dtype=float)
    states[-1] = _mvnrnd(m_filt[-1], P_filt[-1], rng)
    for t in range(T - 2, -1, -1):
        Ptp1 = force_pd(P_pred[t + 1])
        A = P_filt[t] @ F.T @ inv(Ptp1)
        mean_s = m_filt[t] + A @ (states[t + 1] - m_pred[t + 1])
        cov_s = force_pd(P_filt[t] - A @ Ptp1 @ A.T)
        states[t] = _mvnrnd(mean_s, cov_s, rng)

    return states[:, 2], states[:, 0], states, states[:, 3]


def joint_loglik_ma(
    *,
    N_obs: np.ndarray,
    y_tilde: np.ndarray,
    h_nhat: np.ndarray,
    h_nbar: np.ndarray,
    n_drift: float,
    rho1: float,
    rho2: float,
    psi: np.ndarray,
    sigma_v2: float,
    sigma_u2: float,
    sigma_eps2: float,
    sigma_N2: float,
    m0: np.ndarray,
    P0: np.ndarray,
) -> float:
    """``log p(pi, N_obs | theta)`` with the states integrated out.

    The forward pass of :func:`sample_joint_states_ffbs_ma` with the innovation
    log-densities accumulated instead of a backward draw. Chib's marginal
    likelihood needs exactly this: the likelihood with the latent path
    integrated away rather than conditioned on.

    Both observation rows contribute, and quarters with a missing firm count
    contribute the inflation row alone -- the same treatment the sampler uses,
    so likelihood and posterior refer to the same model.
    """
    N_obs = np.asarray(N_obs, dtype=float).reshape(-1)
    y_tilde = np.asarray(y_tilde, dtype=float).reshape(-1)
    h_nhat = np.broadcast_to(np.asarray(h_nhat, dtype=float), N_obs.shape)
    h_nbar = np.broadcast_to(np.asarray(h_nbar, dtype=float), N_obs.shape)

    T = N_obs.size
    if y_tilde.size != T:
        raise ValueError("All input series must have the same length.")

    F_v, Q_v, h_v, P0_v = state_augmentation(psi, sigma_v2)
    dim = 3 + h_v.size

    F = np.zeros((dim, dim), dtype=float)
    F[:3, :3] = [[rho1, rho2, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    F[3:, 3:] = F_v
    c = np.zeros(dim, dtype=float)
    c[2] = n_drift
    Q = np.zeros((dim, dim), dtype=float)
    Q[0, 0] = sigma_u2
    Q[2, 2] = sigma_eps2
    Q[3:, 3:] = Q_v

    m = np.zeros(dim, dtype=float)
    m[:3] = np.asarray(m0, dtype=float).reshape(-1)
    P = np.zeros((dim, dim), dtype=float)
    P[:3, :3] = np.asarray(P0, dtype=float)
    P[3:, 3:] = P0_v
    P = force_pd(P)

    n_row = np.zeros(dim, dtype=float)
    n_row[0] = 1.0
    n_row[2] = 1.0

    total = 0.0
    Id = np.eye(dim)
    for t in range(T):
        if t > 0:
            m = c + F @ m
            P = force_pd(F @ P @ F.T + Q)

        pi_row = np.zeros(dim, dtype=float)
        pi_row[0] = h_nhat[t]
        pi_row[2] = h_nbar[t]
        pi_row[3:] = h_v

        if np.isfinite(N_obs[t]):
            y_obs = np.array([N_obs[t], y_tilde[t]], dtype=float)
            H = np.vstack([n_row, pi_row])
            R = np.diag([sigma_N2, 0.0])
        else:
            y_obs = np.array([y_tilde[t]], dtype=float)
            H = pi_row.reshape(1, dim)
            R = np.zeros((1, 1), dtype=float)

        S = force_pd(H @ P @ H.T + R)
        innov = y_obs - H @ m
        sign, logdet = np.linalg.slogdet(S)
        if sign <= 0:
            raise ValueError("Non-positive-definite innovation covariance.")
        solved = np.linalg.solve(S, innov)
        total += -0.5 * (y_obs.size * np.log(2.0 * np.pi) + logdet + float(innov @ solved))

        K = P @ H.T @ np.linalg.inv(S)
        KH = K @ H
        m = m + K @ innov
        P = force_pd((Id - KH) @ P @ (Id - KH).T + K @ R @ K.T)

    return float(total)
