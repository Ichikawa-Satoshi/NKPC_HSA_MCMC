"""Particle Gibbs (conditional SMC) state update for the PCHIP ``hsa_full`` model.

The model is identical to :func:`nkpc_hsa.gibbs.hsa_full.model.func_nkpc_hsa_full`:

    y_t = alpha*a_t + kappa0*x_t + delta*x_t*Nbar_t
          - theta0*Nhat_t - gamma*Nbar_t*Nhat_t + lambda_ez*zeta_t + eta_t
    Nobs_t = Nbar_t + Nhat_t + nu_t
    Nhat_t = rho1*Nhat_{t-1} + rho2*Nhat_{t-2} + u_t
    Nbar_t = n + Nbar_{t-1} + epsilon_t

with kappa_t = kappa0 + delta*Nbar_t and theta_t = theta0 + gamma*Nbar_t.

The bilinear term ``-gamma*Nbar_t*Nhat_t`` makes the joint (Nbar, Nhat) state
non-linear-Gaussian, which is why the production estimator alternates two exact
conditional FFBS blocks. Here we instead sample the *joint* path with a
conditional bootstrap particle filter (Particle Gibbs), conditioning on the
previous MCMC trajectory. Every other Gibbs block is imported unchanged from
``hsa_full`` so priors, scaling, parameter updates and output format match.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

# Reuse the production estimator's routines verbatim -- do not reimplement.
from nkpc_hsa.gibbs.common.competition import finite_N_residuals
from nkpc_hsa.gibbs.common.constraints import constraint_stats_summary, draw_with_constraints
from nkpc_hsa.gibbs.hsa_full.model import (
    KAPPA_SCALE,
    _ar2_stats_summary,
    _common_priors,
    _getd,
    _init_states,
    _kappa_t_constraint_validators,
    _sample_ar2_coeffs,
    _sample_beta_gaussian,
    _sample_invgamma,
    _sample_phi_joint,
    _summary,
)

# Default particle count if not supplied via opts["n_particles"].
DEFAULT_N_PARTICLES = 256


# ---------------------------------------------------------------------------
# Observation log-likelihood (inflation row + PCHIP firm-count row)
# ---------------------------------------------------------------------------
def _obs_loglik(
    Nhat: np.ndarray,
    Nbar: np.ndarray,
    *,
    y_t: float,
    a_t: float,
    x_t: float,
    zeta_t: float,
    N_obs_t: float,
    alpha: float,
    kappa0_eff: float,
    delta_eff: float,
    theta0: float,
    gamma: float,
    lambda_ez: float,
    sigma_eta2: float,
    sigma_N2: float,
) -> np.ndarray:
    """Unnormalised per-particle log weight at a single time t.

    Uses BOTH the inflation likelihood and the firm-count measurement
    likelihood. Additive constants (the -0.5*log(2*pi*sigma^2) terms) are common
    across particles at a given t and cancel under weight normalisation, so they
    are omitted for numerical economy.
    """
    mu = (
        alpha * a_t
        + kappa0_eff * x_t
        + delta_eff * x_t * Nbar
        - theta0 * Nhat
        - gamma * Nbar * Nhat
        + lambda_ez * zeta_t
    )
    ll = -0.5 * (y_t - mu) ** 2 / sigma_eta2
    if np.isfinite(N_obs_t):
        ll = ll - 0.5 * (N_obs_t - Nbar - Nhat) ** 2 / sigma_N2
    return ll


def _normalise_logw(logw: np.ndarray) -> tuple[np.ndarray, float]:
    m = float(np.max(logw))
    w = np.exp(logw - m)
    s = float(np.sum(w))
    W = w / s
    ess = 1.0 / float(np.sum(W * W))
    return W, ess


# ---------------------------------------------------------------------------
# Joint Particle Gibbs state update (conditional bootstrap SMC)
# ---------------------------------------------------------------------------
def sample_states_particle_gibbs(
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
    sigma_eta2: float,
    sigma_u2: float,
    sigma_eps2: float,
    sigma_N2: float,
    Nbar_ref: np.ndarray,
    Nhat_ref: np.ndarray,
    Nhat_ref_lag: float,
    m0_Nhat: float,
    P0_Nhat: float,
    m0_Nhat_lag: float,
    P0_Nhat_lag: float,
    m0_Nbar: float,
    P0_Nbar: float,
    n_particles: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    """One Particle Gibbs sweep: draw a joint (Nhat, Nbar) path.

    Conditional bootstrap particle filter with the reference trajectory pinned to
    particle slot 0 (its ancestor is always slot 0). Particles are propagated
    from the state transition (bootstrap proposal), the AR(2) genealogy for Nhat
    is preserved by carrying (Nhat_t, Nhat_{t-1}), and weights use both the
    inflation and firm-count measurement likelihoods with stable log-sum-exp
    normalisation. A terminal particle is sampled and its ancestry traced back to
    yield one full joint path.
    """
    T = y.size
    P = int(n_particles)
    su = float(np.sqrt(sigma_u2))
    se = float(np.sqrt(sigma_eps2))

    Nhat_store = np.empty((T, P))
    Nlag_store = np.empty((T, P))
    Nbar_store = np.empty((T, P))
    ancestors = np.empty((T, P), dtype=np.int64)
    ess = np.empty(T)

    def _weights(t: int, Nhat_p: np.ndarray, Nbar_p: np.ndarray) -> tuple[np.ndarray, float]:
        logw = _obs_loglik(
            Nhat_p,
            Nbar_p,
            y_t=float(y[t]),
            a_t=float(a_t[t]),
            x_t=float(x_t[t]),
            zeta_t=float(zeta[t]),
            N_obs_t=float(N_obs[t]),
            alpha=alpha,
            kappa0_eff=kappa0_eff,
            delta_eff=delta_eff,
            theta0=theta0,
            gamma=gamma,
            lambda_ez=lambda_ez,
            sigma_eta2=sigma_eta2,
            sigma_N2=sigma_N2,
        )
        return _normalise_logw(logw)

    # ---- t = 0: initial distribution, reference in slot 0 ----
    Nhat0 = np.empty(P)
    Nlag0 = np.empty(P)
    Nbar0 = np.empty(P)
    Nhat0[0] = Nhat_ref[0]
    Nlag0[0] = Nhat_ref_lag
    Nbar0[0] = Nbar_ref[0]
    if P > 1:
        Nhat0[1:] = m0_Nhat + np.sqrt(P0_Nhat) * rng.standard_normal(P - 1)
        Nlag0[1:] = m0_Nhat_lag + np.sqrt(P0_Nhat_lag) * rng.standard_normal(P - 1)
        Nbar0[1:] = m0_Nbar + np.sqrt(P0_Nbar) * rng.standard_normal(P - 1)
    Nhat_store[0] = Nhat0
    Nlag_store[0] = Nlag0
    Nbar_store[0] = Nbar0
    ancestors[0] = np.arange(P)
    W, ess[0] = _weights(0, Nhat0, Nbar0)

    # ---- t = 1 .. T-1 ----
    for t in range(1, T):
        a = rng.choice(P, size=P, p=W)  # conditional multinomial resampling
        a[0] = 0                        # reference keeps its lineage (slot 0)
        parent_Nhat = Nhat_store[t - 1, a]
        parent_Nlag = Nlag_store[t - 1, a]
        parent_Nbar = Nbar_store[t - 1, a]

        new_Nhat = rho1 * parent_Nhat + rho2 * parent_Nlag + su * rng.standard_normal(P)
        new_Nlag = parent_Nhat
        new_Nbar = n_drift + parent_Nbar + se * rng.standard_normal(P)

        # pin the reference trajectory into slot 0
        new_Nhat[0] = Nhat_ref[t]
        new_Nlag[0] = Nhat_ref[t - 1]
        new_Nbar[0] = Nbar_ref[t]

        Nhat_store[t] = new_Nhat
        Nlag_store[t] = new_Nlag
        Nbar_store[t] = new_Nbar
        ancestors[t] = a
        W, ess[t] = _weights(t, new_Nhat, new_Nbar)

    # ---- sample a terminal particle and trace ancestry backward ----
    b = np.empty(T, dtype=np.int64)
    b[T - 1] = int(rng.choice(P, p=W))
    for t in range(T - 2, -1, -1):
        b[t] = ancestors[t + 1, b[t + 1]]

    idx = np.arange(T)
    Nhat_new = Nhat_store[idx, b]
    Nbar_new = Nbar_store[idx, b]
    Nhat_lag_new = float(Nlag_store[0, b[0]])
    moved_frac = float(np.mean(b != 0))  # fraction of periods the path leaves the reference

    return {
        "Nhat": Nhat_new,
        "Nbar": Nbar_new,
        "Nhat_lag": Nhat_lag_new,
        "ess_mean": float(np.mean(ess)),
        "ess_min": float(np.min(ess)),
        "moved_frac": moved_frac,
    }


# ---------------------------------------------------------------------------
# Exact joint FFBS for the gamma == 0 linear-Gaussian special case (validation)
# ---------------------------------------------------------------------------
def sample_states_joint_ffbs_gamma0(
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
    lambda_ez: float,
    rho1: float,
    rho2: float,
    n_drift: float,
    sigma_eta2: float,
    sigma_u2: float,
    sigma_eps2: float,
    sigma_N2: float,
    m0_Nhat: float,
    P0_Nhat: float,
    m0_Nhat_lag: float,
    P0_Nhat_lag: float,
    m0_Nbar: float,
    P0_Nbar: float,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    """Exact joint 3-state Kalman/FFBS draw for gamma = 0.

    With gamma = 0 the inflation observation is linear in the state
    s_t = (Nhat_t, Nhat_{t-1}, Nbar_t), so the whole model is linear-Gaussian and
    the joint state path can be sampled exactly. Used only to validate the
    Particle Gibbs sampler. Observation rows:
        N_obs_t   = [1, 0, 1] s_t + nu_t,                     var sigma_N2
        ytilde_t  = [-theta0, 0, delta_eff*x_t] s_t + eta_t,  var sigma_eta2
    where ytilde_t = y_t - alpha*a_t - kappa0_eff*x_t - lambda_ez*zeta_t.
    """
    T = y.size
    F = np.array([[rho1, rho2, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    c = np.array([0.0, 0.0, n_drift])
    Q = np.diag([sigma_u2, 0.0, sigma_eps2])
    m0 = np.array([m0_Nhat, m0_Nhat_lag, m0_Nbar])
    P0 = np.diag([P0_Nhat, P0_Nhat_lag, P0_Nbar])

    ytilde = y - alpha * a_t - kappa0_eff * x_t - lambda_ez * zeta

    m_pred = np.empty((T, 3))
    P_pred = np.empty((T, 3, 3))
    m_filt = np.empty((T, 3))
    P_filt = np.empty((T, 3, 3))
    I3 = np.eye(3)

    for t in range(T):
        if t == 0:
            m_pred[t] = m0
            P_pred[t] = P0
        else:
            m_pred[t] = c + F @ m_filt[t - 1]
            P_pred[t] = F @ P_filt[t - 1] @ F.T + Q

        rows_H = []
        rows_z = []
        rows_R = []
        if np.isfinite(N_obs[t]):
            rows_H.append([1.0, 0.0, 1.0])
            rows_z.append(N_obs[t])
            rows_R.append(sigma_N2)
        rows_H.append([-theta0, 0.0, delta_eff * x_t[t]])
        rows_z.append(ytilde[t])
        rows_R.append(sigma_eta2)
        H = np.array(rows_H)
        z = np.array(rows_z)
        R = np.diag(rows_R)

        S = H @ P_pred[t] @ H.T + R
        K = P_pred[t] @ H.T @ np.linalg.inv(S)
        innov = z - H @ m_pred[t]
        m_filt[t] = m_pred[t] + K @ innov
        KH = K @ H
        P_filt[t] = (I3 - KH) @ P_pred[t] @ (I3 - KH).T + K @ R @ K.T

    states = np.empty((T, 3))
    states[-1] = rng.multivariate_normal(m_filt[-1], _sym(P_filt[-1]))
    for t in range(T - 2, -1, -1):
        Ptp1 = P_pred[t + 1]
        A = P_filt[t] @ F.T @ np.linalg.inv(Ptp1)
        mean_s = m_filt[t] + A @ (states[t + 1] - c - F @ m_filt[t])
        cov_s = _sym(P_filt[t] - A @ Ptp1 @ A.T)
        states[t] = rng.multivariate_normal(mean_s, cov_s)

    return {"Nhat": states[:, 0], "Nbar": states[:, 2]}


def _sym(S: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    S = 0.5 * (S + S.T)
    return S + eps * np.eye(S.shape[0])


# ---------------------------------------------------------------------------
# Full Gibbs sampler with the Particle Gibbs state block
# ---------------------------------------------------------------------------
def func_nkpc_hsa_full_pg(
    pi_data,
    pi_prev_data,
    Epi_data,
    x_data,
    x_prev_data,
    N_data,
    n_burn: int,
    n_keep: int,
    priors: Optional[dict[str, Any]] = None,
    opts: Optional[dict[str, Any]] = None,
    *,
    orth: bool = False,
) -> dict[str, Any]:
    """``hsa_full`` Gibbs sampler with a joint Particle Gibbs state update.

    Signature and output are identical to
    :func:`nkpc_hsa.gibbs.hsa_full.model.func_nkpc_hsa_full`, so this can be
    dispatched through the existing ``run_model`` pipeline. The only difference
    is that the alternating FFBS state block is replaced by a conditional-SMC
    joint draw. Particle count is taken from ``opts['n_particles']`` (default
    :data:`DEFAULT_N_PARTICLES`).
    """
    pi_t = np.asarray(pi_data, dtype=float).reshape(-1)
    pi_tm1 = np.asarray(pi_prev_data, dtype=float).reshape(-1)
    pi_expect = np.asarray(Epi_data, dtype=float).reshape(-1)
    x_t = np.asarray(x_data, dtype=float).reshape(-1)
    x_tm1 = np.asarray(x_prev_data, dtype=float).reshape(-1)
    N_obs = np.asarray(N_data, dtype=float).reshape(-1)
    T = pi_t.size
    if not (pi_tm1.size == pi_expect.size == x_t.size == x_tm1.size == N_obs.size == T):
        raise ValueError("All input series must have the same length.")

    pri = _common_priors(priors or {})
    opts = opts or {}
    alpha = float(_getd(opts, "alpha0", pri["mu_alpha"]))
    kappa0 = float(_getd(opts, "kappa00", pri["mu_kappa0"]))
    delta = float(_getd(opts, "delta0", pri["mu_delta"]))
    theta0 = float(_getd(opts, "theta00", pri["mu_theta"]))
    gamma = float(_getd(opts, "gamma0", pri["mu_gamma"]))
    phi_1 = float(_getd(opts, "phi10", pri["mu_phi"]))
    lambda_ez = 0.0 if orth else float(_getd(opts, "lambda0", 0.0))
    rho1 = float(_getd(opts, "rho10", 0.5))
    rho2 = float(_getd(opts, "rho20", -0.5))
    n_drift = float(_getd(opts, "n0", 0.01))
    sigma_eta2 = float(_getd(opts, "sigma_e20", _getd(opts, "sigma_v20", 1.0)))
    sigma_zeta2 = float(_getd(opts, "sigma_zeta20", 1.0))
    sigma_u2 = float(_getd(opts, "sigma_u20", _getd(opts, "sigma_eps20", 0.5)))
    sigma_eps2 = float(_getd(opts, "sigma_eps20", _getd(opts, "sigma_eta20", 0.1)))
    sigma_N2 = float(_getd(opts, "sigma_N20", _getd(opts, "sigma_m20", 1.0)))
    enforce_stationary = bool(_getd(opts, "enforce_stationary", True))
    ar2_max_tries = int(max(1, _getd(opts, "ar2_max_tries", 2000)))
    static_theta = bool(_getd(opts, "static_theta", False))
    if static_theta:
        gamma = 0.0
    n_particles = int(_getd(opts, "n_particles", DEFAULT_N_PARTICLES))
    store_every = int(max(1, _getd(opts, "store_every", 1)))
    verbose = bool(_getd(opts, "verbose", False))
    coefficient_constraints = _getd(opts, "coefficient_constraints", {})
    constraint_stats: dict[str, int] = {}
    ar2_stats: dict[str, int] = {}
    rng = np.random.default_rng(_getd(opts, "seed", None))

    Nbar, Nhat = _init_states(N_obs)
    Nhat_initial_lag = float(_getd(opts, "m0_Nhat_lag", pri["m0_Nhat_lag"]))
    a_t = pi_tm1 - pi_expect
    lambda_prec0 = 0.0 if orth else 1.0 / pri["sigma_lambda"] ** 2

    m0_Nhat = float(_getd(opts, "m0_Nhat", pri["m0_Nhat"]))
    P0_Nhat = float(_getd(opts, "P0_Nhat", pri["P0_Nhat"]))
    m0_Nhat_lag = float(_getd(opts, "m0_Nhat_lag", pri["m0_Nhat_lag"]))
    P0_Nhat_lag = float(_getd(opts, "P0_Nhat_lag", pri["P0_Nhat_lag"]))
    m0_Nbar = float(_getd(opts, "m0_Nbar", pri["m0_Nbar"]))
    P0_Nbar = float(_getd(opts, "P0_Nbar", pri["P0_Nbar"]))

    n_store = int(n_keep // store_every)
    alpha_draws = np.zeros(n_store)
    kappa0_draws = np.zeros(n_store)
    delta_draws = np.zeros(n_store)
    theta0_draws = np.zeros(n_store)
    gamma_draws = np.zeros(n_store)
    phi_draws = np.zeros(n_store)
    lambda_draws = np.zeros(n_store)
    rho1_draws = np.zeros(n_store)
    rho2_draws = np.zeros(n_store)
    n_draws = np.zeros(n_store)
    sigma_e_draws = np.zeros(n_store)
    sigma_zeta_draws = np.zeros(n_store)
    sigma_u_draws = np.zeros(n_store)
    sigma_eps_draws = np.zeros(n_store)
    sigma_N_draws = np.zeros(n_store)
    rho_ez_draws = np.zeros(n_store)
    Nbar_draws = np.zeros((n_store, T))
    Nhat_draws = np.zeros((n_store, T))
    kappa_t_draws = np.zeros((n_store, T))
    theta_t_draws = np.zeros((n_store, T))
    pg_ess_mean_draws = np.zeros(n_store)
    pg_ess_min_draws = np.zeros(n_store)
    pg_moved_draws = np.zeros(n_store)

    total_iter = n_burn + n_keep
    store_idx = 0

    for it in range(1, total_iter + 1):
        kappa_t = kappa0 + delta * Nbar
        kappa_t_eff = kappa_t / KAPPA_SCALE
        theta_t = theta0 + gamma * Nbar
        zeta = x_t - phi_1 * x_tm1
        y = pi_t - pi_expect
        y_adj = y - lambda_ez * zeta

        columns = [a_t, x_t / KAPPA_SCALE, (x_t * Nbar) / KAPPA_SCALE, -Nhat]
        prior_means = [pri["mu_alpha"], pri["mu_kappa0"], pri["mu_delta"], pri["mu_theta"]]
        prior_vars = [pri["sigma_alpha"] ** 2, pri["sigma_kappa0"] ** 2, pri["sigma_delta"] ** 2, pri["sigma_theta"] ** 2]
        beta_names = ["alpha", "kappa_0", "delta", "theta" if static_theta else "theta_0"]
        if not static_theta:
            columns.append(-(Nhat * Nbar))
            prior_means.append(pri["mu_gamma"])
            prior_vars.append(pri["sigma_gamma"] ** 2)
            beta_names.append("gamma")
        X = np.column_stack(columns)
        beta = draw_with_constraints(
            lambda: _sample_beta_gaussian(
                y_adj, X, sigma2=sigma_eta2,
                prior_mean=np.array(prior_means, dtype=float),
                prior_var=np.array(prior_vars, dtype=float),
                rng=rng,
            ),
            tuple(beta_names),
            coefficient_constraints,
            validators=_kappa_t_constraint_validators(Nbar, coefficient_constraints),
            stats=constraint_stats,
        )
        alpha = float(beta[0])
        kappa0 = float(beta[1])
        delta = float(beta[2])
        theta0 = float(beta[3])
        if not static_theta:
            gamma = float(beta[4])
        kappa_t = kappa0 + delta * Nbar
        kappa_t_eff = kappa_t / KAPPA_SCALE
        theta_t = theta0 + gamma * Nbar

        if not orth:
            e_base = y - alpha * a_t - kappa_t_eff * x_t + theta_t * Nhat
            post_var_lambda = 1.0 / (lambda_prec0 + float(np.sum(zeta ** 2)) / sigma_eta2)
            post_mean_lambda = post_var_lambda * (
                pri["mu_lambda"] * lambda_prec0 + float(np.dot(zeta, e_base)) / sigma_eta2
            )
            lambda_ez = float(post_mean_lambda + np.sqrt(post_var_lambda) * rng.standard_normal())
        else:
            lambda_ez = 0.0

        y_tilde_phi = y - alpha * a_t - kappa_t_eff * x_t + theta_t * Nhat
        phi_1 = _sample_phi_joint(
            x_t=x_t, x_tm1=x_tm1, y_tilde=y_tilde_phi, lambda_ez=lambda_ez,
            sigma_zeta2=sigma_zeta2, sigma_eta2=sigma_eta2,
            mu_phi=pri["mu_phi"], sigma_phi=pri["sigma_phi"], rng=rng,
        )

        zeta = x_t - phi_1 * x_tm1
        eta = y - alpha * a_t - kappa_t_eff * x_t + theta_t * Nhat - lambda_ez * zeta
        sigma_zeta2 = _sample_invgamma(pri["a_z"] + 0.5 * T, pri["b_z"] + 0.5 * float(np.sum(zeta ** 2)), rng)
        sigma_eta2 = _sample_invgamma(pri["a_e"] + 0.5 * T, pri["b_e"] + 0.5 * float(np.sum(eta ** 2)), rng)

        if T >= 2:
            rho1, rho2 = _sample_ar2_coeffs(
                Nhat, sigma_u2, pri["mu_rho1"], pri["sigma_rho1"], pri["mu_rho2"], pri["sigma_rho2"],
                enforce_stationary, rng, max_tries=ar2_max_tries, current=(rho1, rho2),
                stats=ar2_stats, initial_lag=Nhat_initial_lag,
            )
            second_lag = np.concatenate([[Nhat_initial_lag], Nhat[:-2]])
            resid_u = Nhat[1:] - rho1 * Nhat[:-1] - rho2 * second_lag
            sigma_u2 = _sample_invgamma(pri["a_u"] + 0.5 * resid_u.size, pri["b_u"] + 0.5 * float(np.sum(resid_u ** 2)), rng)

        if T >= 2:
            dNbar = Nbar[1:] - Nbar[:-1]
            post_var_n = 1.0 / (1.0 / pri["sigma_n"] ** 2 + dNbar.size / sigma_eps2)
            post_mean_n = post_var_n * (pri["mu_n"] / pri["sigma_n"] ** 2 + float(np.sum(dNbar)) / sigma_eps2)
            n_drift = float(post_mean_n + np.sqrt(post_var_n) * rng.standard_normal())
            resid_eps = Nbar[1:] - n_drift - Nbar[:-1]
            sigma_eps2 = _sample_invgamma(pri["a_eps"] + 0.5 * resid_eps.size, pri["b_eps"] + 0.5 * float(np.sum(resid_eps ** 2)), rng)

        resid_N = finite_N_residuals(N_obs, Nhat, Nbar)
        sigma_N2 = _sample_invgamma(pri["a_N"] + 0.5 * resid_N.size, pri["b_N"] + 0.5 * float(np.sum(resid_N ** 2)), rng)

        # ---------- JOINT Particle Gibbs state update (replaces alternating FFBS) ----------
        pg = sample_states_particle_gibbs(
            y=y, a_t=a_t, x_t=x_t, zeta=zeta, N_obs=N_obs,
            alpha=alpha, kappa0_eff=kappa0 / KAPPA_SCALE, delta_eff=delta / KAPPA_SCALE,
            theta0=theta0, gamma=gamma, lambda_ez=lambda_ez,
            rho1=rho1, rho2=rho2, n_drift=n_drift,
            sigma_eta2=sigma_eta2, sigma_u2=sigma_u2, sigma_eps2=sigma_eps2, sigma_N2=sigma_N2,
            Nbar_ref=Nbar, Nhat_ref=Nhat, Nhat_ref_lag=Nhat_initial_lag,
            m0_Nhat=m0_Nhat, P0_Nhat=P0_Nhat, m0_Nhat_lag=m0_Nhat_lag, P0_Nhat_lag=P0_Nhat_lag,
            m0_Nbar=m0_Nbar, P0_Nbar=P0_Nbar, n_particles=n_particles, rng=rng,
        )
        Nhat = pg["Nhat"]
        Nbar = pg["Nbar"]
        Nhat_initial_lag = pg["Nhat_lag"]
        # ---------------------------------------------------------------------------------

        kappa_t = kappa0 + delta * Nbar
        kappa_t_eff = kappa_t / KAPPA_SCALE
        theta_t = theta0 + gamma * Nbar

        sigma_e = float(np.sqrt(lambda_ez ** 2 * sigma_zeta2 + sigma_eta2))
        rho_ez = 0.0 if orth else float((lambda_ez * np.sqrt(sigma_zeta2)) / max(sigma_e, 1e-12))

        if it > n_burn and (it - n_burn) % store_every == 0:
            alpha_draws[store_idx] = alpha
            kappa0_draws[store_idx] = kappa0 / KAPPA_SCALE
            delta_draws[store_idx] = delta / KAPPA_SCALE
            theta0_draws[store_idx] = theta0
            gamma_draws[store_idx] = gamma
            phi_draws[store_idx] = phi_1
            lambda_draws[store_idx] = lambda_ez
            rho1_draws[store_idx] = rho1
            rho2_draws[store_idx] = rho2
            n_draws[store_idx] = n_drift
            sigma_e_draws[store_idx] = sigma_e
            sigma_zeta_draws[store_idx] = np.sqrt(sigma_zeta2)
            sigma_u_draws[store_idx] = np.sqrt(sigma_u2)
            sigma_eps_draws[store_idx] = np.sqrt(sigma_eps2)
            sigma_N_draws[store_idx] = np.sqrt(sigma_N2)
            rho_ez_draws[store_idx] = rho_ez
            Nbar_draws[store_idx] = Nbar
            Nhat_draws[store_idx] = Nhat
            kappa_t_draws[store_idx] = kappa_t_eff
            theta_t_draws[store_idx] = theta_t
            pg_ess_mean_draws[store_idx] = pg["ess_mean"]
            pg_ess_min_draws[store_idx] = pg["ess_min"]
            pg_moved_draws[store_idx] = pg["moved_frac"]
            store_idx += 1

        if verbose and it % 2000 == 0:
            print(
                f"Iter {it}/{total_iter}: alpha={alpha:.3f}, kappa0={kappa0:.3f}, delta={delta:.3f}, "
                f"theta0={theta0:.3f}, gamma={gamma:.3f}, pg_ess={pg['ess_mean']:.0f}, moved={pg['moved_frac']:.2f}"
            )

    if static_theta:
        theta_keys = {"theta": _summary(theta0_draws)}
    else:
        theta_keys = {"theta_0": _summary(theta0_draws), "gamma": _summary(gamma_draws)}

    return {
        "alpha": _summary(alpha_draws),
        "kappa_0": _summary(kappa0_draws),
        "delta": _summary(delta_draws),
        **theta_keys,
        "phi_1": _summary(phi_draws),
        "lambda_ez": _summary(lambda_draws),
        "rho": _summary(rho_ez_draws),
        "rho1": _summary(rho1_draws),
        "rho2": _summary(rho2_draws),
        "n": _summary(n_draws),
        "sigma_e": _summary(sigma_e_draws),
        "sigma_zeta": _summary(sigma_zeta_draws),
        "sigma_u": _summary(sigma_u_draws),
        "sigma_eps": _summary(sigma_eps_draws),
        "sigma_N": _summary(sigma_N_draws),
        "state_draws": {
            "Nbar": Nbar_draws,
            "Nhat": Nhat_draws,
            "kappa_t": kappa_t_draws,
            "theta_t": theta_t_draws,
        },
        "pg_diagnostics": {
            "n_particles": n_particles,
            "ess_mean": pg_ess_mean_draws,
            "ess_min": pg_ess_min_draws,
            "moved_frac": pg_moved_draws,
        },
        "priors": priors or {},
        "opts": opts,
        "model": {
            "N_measurement_error": True,
            "N_measurement_equation": "N_obs_t = Nhat_t + Nbar_t + nu_t, nu_t ~ N(0, sigma_N^2)",
            "state_blocks": (
                f"JOINT Particle Gibbs (conditional bootstrap SMC, {n_particles} particles) "
                "replacing the alternating Nhat|Nbar and Nbar|Nhat FFBS blocks."
            ),
            "state_sampler": "particle_gibbs",
            "n_particles": n_particles,
            "kappa_scale": KAPPA_SCALE,
            "kappa_internal": "stored kappa_0, delta, and kappa_t multiplied by KAPPA_SCALE",
            "stored_units": "physical",
            "theta_specification": "static (theta_t = theta, gamma fixed at 0)" if static_theta else "time-varying (theta_t = theta_0 + gamma * Nbar_t)",
            "coefficient_constraints": coefficient_constraints,
            "coefficient_constraint_stats": constraint_stats_summary(constraint_stats),
            "ar2_stationarity": {
                "enforce_stationary": enforce_stationary,
                "max_tries": ar2_max_tries,
                **_ar2_stats_summary(ar2_stats),
            },
        },
    }


__all__ = [
    "func_nkpc_hsa_full_pg",
    "sample_states_particle_gibbs",
    "sample_states_joint_ffbs_gamma0",
    "DEFAULT_N_PARTICLES",
]
