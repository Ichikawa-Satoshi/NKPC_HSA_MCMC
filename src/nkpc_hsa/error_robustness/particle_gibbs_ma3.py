"""Conditional-SMC state update for ``hsa_full`` with an MA(q) disturbance.

The additive counterpart of
``nkpc_hsa.gibbs.hsa_full_pg.model.sample_states_particle_gibbs``, which is left
untouched. ``hsa_full`` needs a particle method rather than an FFBS because the
bilinear ``-gamma * Nbar_t * Nhat_t`` term makes the joint state
non-linear-Gaussian.

Why a naive augmentation would break
------------------------------------
The obvious move -- append ``(v_t, ..., v_{t-q})`` to the particle state and
propagate ``v_t`` from its prior -- turns the inflation observation into a point
mass given the state, so every particle would get weight zero. A bootstrap
filter cannot survive a degenerate observation density.

The fix is that ``v_t`` does not have to be *proposed* at all. Given the
particle's ``(Nhat_t, Nbar_t)`` and its own ``v`` history, the inflation
equation **determines** it:

    v_t = [y_t - mu_t(Nhat_t, Nbar_t)] - psi_1 v_{t-1} - ... - psi_q v_{t-q}

so the particle carries only the ``q`` lags, computes ``v_t`` exactly, and takes
its weight from the *prior* density of the implied innovation,
``N(v_t; 0, sigma_v^2)``. The transformation from ``y_t`` to ``v_t`` has unit
Jacobian, so this is exact, not an approximation. Note what it does and does not
buy: the ``v`` dimension is absorbed rather than proposed, which is what makes
the augmentation possible at all, but ``Nhat`` and ``Nbar`` are still propagated
from the bootstrap proposal exactly as in production -- so with ``psi = []`` the
routine is bit-for-bit the production sampler, not a better-conditioned one.

The pre-sample innovations ``v_{-1..-q}`` are part of what is being sampled:
non-reference particles draw them from the stationary ``N(0, sigma_v^2)``, and
the reference trajectory carries its own from the previous sweep.

With ``psi = []`` the innovation is the disturbance itself and the weight
reduces to the production inflation term, so the routine collapses to
``sample_states_particle_gibbs``.
"""

from __future__ import annotations

from typing import Any

import numpy as np

__all__ = ["sample_states_particle_gibbs_ma"]


def _normalise_logw(logw: np.ndarray) -> tuple[np.ndarray, float]:
    m = float(np.max(logw))
    w = np.exp(logw - m)
    s = float(np.sum(w))
    weights = w / s
    ess = 1.0 / float(np.sum(weights * weights))
    return weights, ess


def sample_states_particle_gibbs_ma(
    *,
    y: np.ndarray,
    a_t: np.ndarray,
    x_t: np.ndarray,
    zeta: np.ndarray,
    N_obs: np.ndarray,
    alpha: float,
    kappa0_eff: float,
    delta_eff: float,
    theta0: float,
    gamma: float,
    lambda_ez: float,
    rho1: float,
    rho2: float,
    n_drift: float,
    psi: np.ndarray,
    sigma_v2: float,
    sigma_u2: float,
    sigma_eps2: float,
    sigma_N2: float,
    Nbar_ref: np.ndarray,
    Nhat_ref: np.ndarray,
    Nhat_ref_lag: float,
    v_presample_ref: np.ndarray,
    m0_Nhat: float,
    P0_Nhat: float,
    m0_Nhat_lag: float,
    P0_Nhat_lag: float,
    m0_Nbar: float,
    P0_Nbar: float,
    n_particles: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    """One Particle Gibbs sweep drawing ``(Nhat, Nbar, v_{-1..-q})`` jointly.

    ``v_presample_ref`` is ``[v_{-1}, ..., v_{-q}]`` for the reference
    trajectory. Returns the drawn paths plus the drawn pre-sample, which the
    caller feeds back as the next sweep's reference.
    """
    y = np.asarray(y, dtype=float).reshape(-1)
    psi = np.asarray(psi, dtype=float).reshape(-1)
    q = psi.size
    T = y.size
    P = int(n_particles)
    su = float(np.sqrt(sigma_u2))
    se = float(np.sqrt(sigma_eps2))
    sv = float(np.sqrt(sigma_v2))

    Nhat_store = np.empty((T, P))
    Nlag_store = np.empty((T, P))
    Nbar_store = np.empty((T, P))
    v_store = np.empty((T, P))
    # v_lags[:, j] is v_{t-1-j} for the current t, carried per particle.
    v_lags = np.zeros((P, q))
    vpre_store = np.empty((P, q))
    ancestors = np.empty((T, P), dtype=np.int64)
    ess = np.empty(T)

    def _step(t: int, Nhat_p: np.ndarray, Nbar_p: np.ndarray, lags: np.ndarray):
        """Solve for v_t, then weight by its prior density and the firm-count row."""
        mu = (
            alpha * a_t[t]
            + kappa0_eff * x_t[t]
            + delta_eff * x_t[t] * Nbar_p
            - theta0 * Nhat_p
            - gamma * Nbar_p * Nhat_p
            + lambda_ez * zeta[t]
        )
        resid = y[t] - mu
        v_t = resid - (lags @ psi if q else 0.0)
        # Additive constants are common across particles and cancel.
        logw = -0.5 * v_t**2 / sigma_v2
        if np.isfinite(N_obs[t]):
            logw = logw - 0.5 * (N_obs[t] - Nbar_p - Nhat_p) ** 2 / sigma_N2
        return v_t, logw

    # ---- t = 0: initial distribution, reference pinned to slot 0 ----
    Nhat0 = np.empty(P)
    Nlag0 = np.empty(P)
    Nbar0 = np.empty(P)
    Nhat0[0] = Nhat_ref[0]
    Nlag0[0] = Nhat_ref_lag
    Nbar0[0] = Nbar_ref[0]
    if q:
        v_lags[0] = np.asarray(v_presample_ref, dtype=float).reshape(-1)
    if P > 1:
        Nhat0[1:] = m0_Nhat + np.sqrt(P0_Nhat) * rng.standard_normal(P - 1)
        Nlag0[1:] = m0_Nhat_lag + np.sqrt(P0_Nhat_lag) * rng.standard_normal(P - 1)
        Nbar0[1:] = m0_Nbar + np.sqrt(P0_Nbar) * rng.standard_normal(P - 1)
        if q:
            # Pre-sample innovations are stationary N(0, sigma_v^2).
            v_lags[1:] = sv * rng.standard_normal((P - 1, q))
    vpre_store[:] = v_lags

    Nhat_store[0] = Nhat0
    Nlag_store[0] = Nlag0
    Nbar_store[0] = Nbar0
    ancestors[0] = np.arange(P)
    v0, logw = _step(0, Nhat0, Nbar0, v_lags)
    v_store[0] = v0
    if q:
        v_lags = np.column_stack([v0, v_lags[:, : q - 1]]) if q > 1 else v0.reshape(-1, 1)
    W, ess[0] = _normalise_logw(logw)

    # ---- t = 1 .. T-1 ----
    for t in range(1, T):
        anc = rng.choice(P, size=P, p=W)
        anc[0] = 0  # the reference keeps its lineage
        parent_Nhat = Nhat_store[t - 1, anc]
        parent_Nlag = Nlag_store[t - 1, anc]
        parent_Nbar = Nbar_store[t - 1, anc]
        lags = v_lags[anc] if q else v_lags

        new_Nhat = rho1 * parent_Nhat + rho2 * parent_Nlag + su * rng.standard_normal(P)
        new_Nlag = parent_Nhat
        new_Nbar = n_drift + parent_Nbar + se * rng.standard_normal(P)

        new_Nhat[0] = Nhat_ref[t]
        new_Nlag[0] = Nhat_ref[t - 1]
        new_Nbar[0] = Nbar_ref[t]

        Nhat_store[t] = new_Nhat
        Nlag_store[t] = new_Nlag
        Nbar_store[t] = new_Nbar
        ancestors[t] = anc
        vpre_store = vpre_store[anc]

        v_t, logw = _step(t, new_Nhat, new_Nbar, lags)
        v_store[t] = v_t
        if q:
            v_lags = np.column_stack([v_t, lags[:, : q - 1]]) if q > 1 else v_t.reshape(-1, 1)
        W, ess[t] = _normalise_logw(logw)

    # ---- sample a terminal particle and trace its ancestry ----
    b = np.empty(T, dtype=np.int64)
    b[T - 1] = int(rng.choice(P, p=W))
    for t in range(T - 2, -1, -1):
        b[t] = ancestors[t + 1, b[t + 1]]

    idx = np.arange(T)
    return {
        "Nhat": Nhat_store[idx, b],
        "Nbar": Nbar_store[idx, b],
        "Nhat_lag": float(Nlag_store[0, b[0]]),
        "v": v_store[idx, b],
        "v_presample": vpre_store[b[T - 1]].copy() if q else np.zeros(0),
        "ess_mean": float(np.mean(ess)),
        "ess_min": float(np.min(ess)),
        "moved_frac": float(np.mean(b != 0)),
    }
