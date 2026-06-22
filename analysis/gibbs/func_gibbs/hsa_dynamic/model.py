from __future__ import annotations

from typing import Any, Optional

import numpy as np
from numpy.linalg import inv

from analysis.gibbs.func_gibbs.common.constraints import constraint_stats_summary, draw_with_constraints


# Kappa is sampled internally on a KAPPA_SCALE-multiplied scale because the
# regression column is x / KAPPA_SCALE. Stored draws are physical units.
KAPPA_SCALE = 100.0


# ============================================================
# Basic helpers
# ============================================================

def _getd(d: Optional[dict[str, Any]], key: str, default: Any) -> Any:
    if isinstance(d, dict) and key in d and d[key] is not None:
        return d[key]
    return default


def _as_1d(x: Any) -> np.ndarray:
    return np.asarray(x, dtype=float).reshape(-1)


def _assert_all_pos(arr: Any, msg: str) -> None:
    values = np.asarray(arr, dtype=float)
    if np.any(~np.isfinite(values)) or np.any(values <= 0.0):
        raise ValueError(msg)


def _force_pd(S: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    S = (np.asarray(S, dtype=float) + np.asarray(S, dtype=float).T) / 2.0
    vals, vecs = np.linalg.eigh(S)
    vals = np.maximum(vals, eps)
    repaired = (vecs * vals) @ vecs.T
    return (repaired + repaired.T) / 2.0


def _restrict_sigma_structure(Sigma: np.ndarray, structure: str) -> np.ndarray:
    """Apply the intended HSA shock-covariance restrictions.

    ``e_zeta_only`` matches the empirical specification that allows correlation
    between the NKPC shock e_t and output-gap shock zeta_t only. ``diagonal``
    imposes orthogonal shocks. ``full`` preserves the historical unrestricted
    sampler and is kept only as an explicit opt-in.
    """
    S = _force_pd(Sigma)
    if structure == "full":
        return S
    if structure == "diagonal":
        return _force_pd(np.diag(np.diag(S)))
    if structure == "e_zeta_only":
        out = np.diag(np.diag(S))
        out[0, 1] = out[1, 0] = S[0, 1]
        return _force_pd(out)
    raise ValueError("covariance_structure must be one of: e_zeta_only, diagonal, full.")


def _mvnrnd(mean: np.ndarray, cov: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    return rng.multivariate_normal(np.asarray(mean, dtype=float), _force_pd(cov))


def _sample_invgamma(a_post: float, b_post: float, rng: np.random.Generator) -> float:
    if a_post <= 0.0 or b_post <= 0.0:
        raise ValueError("Inverse-gamma posterior parameters must be positive.")
    return 1.0 / rng.gamma(shape=a_post, scale=1.0 / b_post)


def _is_stationary_ar2(r1: float, r2: float) -> bool:
    return (abs(r2) < 1.0) and ((r1 + r2) < 1.0) and ((r2 - r1) < 1.0)


def _summary(draws: np.ndarray) -> dict[str, Any]:
    arr = np.asarray(draws, dtype=float)
    if arr.shape[0] == 0:
        raise ValueError("No posterior draws were stored. Check n_keep and store_every.")

    qs = np.quantile(
        arr,
        [0.025, 0.05, 0.25, 0.5, 0.75, 0.95, 0.975],
        axis=0,
    )
    return {
        "draws": arr,
        "mean": np.mean(arr, axis=0),
        "std": np.std(arr, axis=0, ddof=1)
        if arr.shape[0] > 1
        else np.zeros_like(np.mean(arr, axis=0)),
        "quantiles": qs,
    }


# ============================================================
# Gaussian regression helpers
# ============================================================

def _sample_beta_gaussian_weighted(
    y: np.ndarray,
    X: np.ndarray,
    var: np.ndarray,
    prior_mean: np.ndarray,
    prior_var: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Weighted Gaussian regression:

        y_t = X_t beta + error_t
        error_t ~ N(0, var_t)

    prior:

        beta ~ N(prior_mean, diag(prior_var))
    """
    y = _as_1d(y)
    X = np.asarray(X, dtype=float)
    var = _as_1d(var)
    prior_mean = _as_1d(prior_mean)
    prior_var = _as_1d(prior_var)

    if X.ndim != 2:
        raise ValueError("X must be 2D.")
    if X.shape[0] != y.size or var.size != y.size:
        raise ValueError("y, X, and var have incompatible lengths.")
    if X.shape[1] != prior_mean.size:
        raise ValueError("prior_mean dimension does not match X.")
    if prior_mean.size != prior_var.size:
        raise ValueError("prior_mean and prior_var dimensions do not match.")

    _assert_all_pos(var, "Regression variances must be positive.")
    _assert_all_pos(prior_var, "Prior variances must be positive.")

    w = 1.0 / var
    V0_inv = np.diag(1.0 / prior_var)

    XtW = X.T * w
    Vn = inv(XtW @ X + V0_inv)
    mn = Vn @ (XtW @ y + V0_inv @ prior_mean)

    return _mvnrnd(mn, Vn, rng)


# ============================================================
# Inverse-Wishart sampler for full Sigma
# ============================================================

def _sample_invwishart(
    df: float,
    scale: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Draw Sigma ~ IW(df, scale).

    Implementation:
        If Sigma ~ IW(nu, Psi), then inv(Sigma) ~ W(nu, inv(Psi)).
    """
    scale = _force_pd(np.asarray(scale, dtype=float))
    p = scale.shape[0]

    if scale.shape != (p, p):
        raise ValueError("scale must be square.")
    if df <= p - 1:
        raise ValueError("Inverse-Wishart df must be greater than p - 1.")

    wishart_scale = _force_pd(inv(scale))
    L = np.linalg.cholesky(wishart_scale)

    A = np.zeros((p, p), dtype=float)
    for i in range(p):
        A[i, i] = np.sqrt(rng.chisquare(df - i))
        for j in range(i):
            A[i, j] = rng.standard_normal()

    W = L @ A @ A.T @ L.T
    Sigma = inv(_force_pd(W))

    return _force_pd(Sigma)


def _sample_Sigma_full(
    e: np.ndarray,
    zeta: np.ndarray,
    u: np.ndarray,
    eps: np.ndarray,
    nu0: float,
    S0: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Sample full Sigma for:

        [e_t, zeta_t, u_t, epsilon_t]'

    We use t >= 1 because u_t and epsilon_t are transition shocks.
    """
    e = _as_1d(e)
    zeta = _as_1d(zeta)
    u = _as_1d(u)
    eps = _as_1d(eps)

    if not (e.size == zeta.size == u.size == eps.size):
        raise ValueError("Residual arrays must have the same length.")

    resid = np.column_stack([e[1:], zeta[1:], u[1:], eps[1:]])
    resid = resid[np.all(np.isfinite(resid), axis=1)]

    if resid.shape[0] == 0:
        raise ValueError("No valid residual rows for Sigma sampling.")

    S_post = np.asarray(S0, dtype=float) + resid.T @ resid
    nu_post = float(nu0) + resid.shape[0]

    return _sample_invwishart(nu_post, S_post, rng)


# ============================================================
# Conditional normal moments from full Sigma
# ============================================================

def _conditional_scalar_many(
    Sigma: np.ndarray,
    target_idx: int,
    cond_idx: list[int],
    cond_values: np.ndarray,
) -> tuple[np.ndarray, float]:
    """
    For zero-mean Gaussian vector r ~ N(0, Sigma),
    compute:

        r_target | r_cond = q

    for many rows q.

    Returns:
        conditional means, conditional variance
    """
    Sigma = _force_pd(Sigma)

    cond_values = np.asarray(cond_values, dtype=float)
    if cond_values.ndim == 1:
        cond_values = cond_values.reshape(-1, 1)

    n = cond_values.shape[0]

    if len(cond_idx) == 0:
        return np.zeros(n, dtype=float), float(Sigma[target_idx, target_idx])

    Scc = _force_pd(Sigma[np.ix_(cond_idx, cond_idx)])
    Stc = Sigma[target_idx, cond_idx]

    coeff = Stc @ inv(Scc)
    means = cond_values @ coeff.T

    var = Sigma[target_idx, target_idx] - coeff @ Sigma[cond_idx, target_idx]
    var = float(max(var, 1e-12))

    return means.reshape(-1), var


def _conditional_e_all(
    Sigma: np.ndarray,
    zeta: np.ndarray,
    u: np.ndarray,
    eps: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    e_t | available shocks.

    t = 0:
        condition only on zeta_t

    t >= 1:
        condition on zeta_t, u_t, epsilon_t
    """
    zeta = _as_1d(zeta)
    u = _as_1d(u)
    eps = _as_1d(eps)
    T = zeta.size

    mean = np.zeros(T, dtype=float)
    var = np.zeros(T, dtype=float)

    m0, v0 = _conditional_scalar_many(
        Sigma,
        target_idx=0,
        cond_idx=[1],
        cond_values=np.array([[zeta[0]]]),
    )
    mean[0] = m0[0]
    var[0] = v0

    if T > 1:
        q = np.column_stack([zeta[1:], u[1:], eps[1:]])
        m, v = _conditional_scalar_many(
            Sigma,
            target_idx=0,
            cond_idx=[1, 2, 3],
            cond_values=q,
        )
        mean[1:] = m
        var[1:] = v

    return mean, var


def _conditional_zeta_all(
    Sigma: np.ndarray,
    e: np.ndarray,
    u: np.ndarray,
    eps: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    zeta_t | available shocks.

    t = 0:
        condition only on e_t

    t >= 1:
        condition on e_t, u_t, epsilon_t
    """
    e = _as_1d(e)
    u = _as_1d(u)
    eps = _as_1d(eps)
    T = e.size

    mean = np.zeros(T, dtype=float)
    var = np.zeros(T, dtype=float)

    m0, v0 = _conditional_scalar_many(
        Sigma,
        target_idx=1,
        cond_idx=[0],
        cond_values=np.array([[e[0]]]),
    )
    mean[0] = m0[0]
    var[0] = v0

    if T > 1:
        q = np.column_stack([e[1:], u[1:], eps[1:]])
        m, v = _conditional_scalar_many(
            Sigma,
            target_idx=1,
            cond_idx=[0, 2, 3],
            cond_values=q,
        )
        mean[1:] = m
        var[1:] = v

    return mean, var


# ============================================================
# Priors and initialization
# ============================================================

def _common_priors(priors: dict[str, Any]) -> dict[str, Any]:
    S_default = np.eye(4)

    return {
        "mu_alpha": _getd(priors, "mu_alpha", 0.5),
        "sigma_alpha": _getd(priors, "sigma_alpha", 0.2),

        "mu_kappa": _getd(priors, "mu_kappa", 10.0),
        "sigma_kappa": _getd(priors, "sigma_kappa", 20.0),

        "mu_theta": _getd(priors, "mu_theta", 0.1),
        "sigma_theta": _getd(priors, "sigma_theta", 0.2),

        "mu_phi": _getd(priors, "mu_phi_1", 0.7),
        "sigma_phi": _getd(priors, "sigma_phi_1", 0.2),

        "mu_rho1": _getd(priors, "mu_rho1", 0.5),
        "sigma_rho1": _getd(priors, "sigma_rho1", 0.2),

        "mu_rho2": _getd(priors, "mu_rho2", -0.5),
        "sigma_rho2": _getd(priors, "sigma_rho2", 0.2),

        "mu_n": _getd(priors, "mu_n", 0.0),
        "sigma_n": _getd(priors, "sigma_n", 0.1),

        # Measurement error:
        # N_obs_t = Nhat_t + Nbar_t + nu_t
        "a_N": _getd(priors, "a_N", 2.0),
        "b_N": _getd(priors, "b_N", 2.0),

        # Full covariance prior:
        # [e_t, zeta_t, u_t, epsilon_t]' ~ N(0, Sigma)
        # Sigma ~ IW(nu_Sigma, S_Sigma)
        "nu_Sigma": _getd(priors, "nu_Sigma", 8.0),
        "S_Sigma": np.asarray(_getd(priors, "S_Sigma", S_default), dtype=float),

        # Initial state prior for:
        # s_0 = [Nhat_0, Nhat_{-1}, Nbar_0]'
        "m0_Nhat": _getd(priors, "m0_Nhat", 0.0),
        "m0_Nhat_lag": _getd(priors, "m0_Nhat_lag", 0.0),
        "m0_Nbar": _getd(priors, "m0_Nbar", 0.0),

        "P0_Nhat": _getd(priors, "P0_Nhat", 10.0),
        "P0_Nhat_lag": _getd(priors, "P0_Nhat_lag", 10.0),
        "P0_Nbar": _getd(priors, "P0_Nbar", 10.0),
    }


def _init_states(N_obs: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    N_obs = _as_1d(N_obs)
    T = N_obs.size

    Nbar = np.zeros(T, dtype=float)
    k0 = min(2, T)
    Nbar[:k0] = N_obs[:k0]

    for t in range(2, T):
        Nbar[t] = 0.7 * Nbar[t - 1] + 0.3 * N_obs[t]

    Nhat = N_obs - Nbar

    states = np.zeros((T, 3), dtype=float)
    states[:, 0] = Nhat
    states[:, 2] = Nbar
    states[0, 1] = 0.0
    if T > 1:
        states[1:, 1] = Nhat[:-1]

    return Nbar, Nhat, states


def _compute_state_residuals(
    states: np.ndarray,
    rho1: float,
    rho2: float,
    n_drift: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute u_t and epsilon_t from sampled states.

    states_t = [Nhat_t, Nhat_{t-1}, Nbar_t]'

    For t >= 1:

        u_t = Nhat_t - rho1*Nhat_{t-1} - rho2*Nhat_{t-2}
        epsilon_t = Nbar_t - n - Nbar_{t-1}

    t = 0 is treated as initial condition, so u_0 and epsilon_0 are NaN.
    """
    states = np.asarray(states, dtype=float)
    T = states.shape[0]

    u = np.full(T, np.nan, dtype=float)
    eps = np.full(T, np.nan, dtype=float)

    for t in range(1, T):
        u[t] = states[t, 0] - rho1 * states[t - 1, 0] - rho2 * states[t - 1, 1]
        eps[t] = states[t, 2] - n_drift - states[t - 1, 2]

    return u, eps


# ============================================================
# Parameter samplers under full Sigma
# ============================================================

def _sample_alpha_kappa_theta_full(
    *,
    y: np.ndarray,
    a_t: np.ndarray,
    x_t: np.ndarray,
    Nhat: np.ndarray,
    zeta: np.ndarray,
    u: np.ndarray,
    eps: np.ndarray,
    Sigma: np.ndarray,
    pri: dict[str, Any],
    rng: np.random.Generator,
    coefficient_constraints: dict[str, Any] | None = None,
    constraint_stats: dict[str, int] | None = None,
) -> tuple[float, float, float]:
    """
    Inflation equation:

        y_t = alpha*a_t + (kappa/KAPPA_SCALE)*x_t - theta*Nhat_t + e_t

    where:

        y_t = pi_t - E_t pi_{t+1}
        a_t = pi_{t-1} - E_t pi_{t+1}

    With full Sigma, use:

        e_t | zeta_t, u_t, epsilon_t
    """
    y = _as_1d(y)
    a_t = _as_1d(a_t)
    x_t = _as_1d(x_t)
    Nhat = _as_1d(Nhat)

    mean_e, var_e = _conditional_e_all(Sigma, zeta, u, eps)

    X = np.column_stack(
        [
            a_t,
            x_t / KAPPA_SCALE,
            -Nhat,
        ]
    )

    beta_prior_mean = np.array(
        [pri["mu_alpha"], pri["mu_kappa"], pri["mu_theta"]],
        dtype=float,
    )
    beta_prior_var = np.array(
        [
            pri["sigma_alpha"] ** 2,
            pri["sigma_kappa"] ** 2,
            pri["sigma_theta"] ** 2,
        ],
        dtype=float,
    )
    beta = draw_with_constraints(
        lambda: _sample_beta_gaussian_weighted(
            y=y - mean_e,
            X=X,
            var=var_e,
            prior_mean=beta_prior_mean,
            prior_var=beta_prior_var,
            rng=rng,
        ),
        ("alpha", "kappa", "theta"),
        coefficient_constraints,
        stats=constraint_stats,
    )

    return float(beta[0]), float(beta[1]), float(beta[2])


def _sample_phi_full(
    *,
    x_t: np.ndarray,
    x_tm1: np.ndarray,
    e: np.ndarray,
    u: np.ndarray,
    eps: np.ndarray,
    Sigma: np.ndarray,
    mu_phi: float,
    sigma_phi: float,
    rng: np.random.Generator,
) -> float:
    """
    x equation:

        x_t = phi_1*x_{t-1} + zeta_t

    With full Sigma, use:

        zeta_t | e_t, u_t, epsilon_t
    """
    x_t = _as_1d(x_t)
    x_tm1 = _as_1d(x_tm1)

    mean_zeta, var_zeta = _conditional_zeta_all(Sigma, e, u, eps)

    y_phi = x_t - mean_zeta
    X_phi = x_tm1

    w = 1.0 / var_zeta

    prec = 1.0 / sigma_phi**2 + float(np.sum(w * X_phi**2))
    mean_num = mu_phi / sigma_phi**2 + float(np.sum(w * X_phi * y_phi))

    return float(mean_num / prec + rng.standard_normal() / np.sqrt(prec))


def _sample_ar2_coeffs_full(
    *,
    states: np.ndarray,
    e: np.ndarray,
    zeta: np.ndarray,
    eps: np.ndarray,
    Sigma: np.ndarray,
    mu_rho1: float,
    sigma_rho1: float,
    mu_rho2: float,
    sigma_rho2: float,
    enforce_stationary: bool,
    rng: np.random.Generator,
    max_tries: int = 2000,
) -> tuple[float, float]:
    """
    AR(2) equation:

        Nhat_t = rho1*Nhat_{t-1} + rho2*Nhat_{t-2} + u_t

    With full Sigma, use:

        u_t | e_t, zeta_t, epsilon_t

    for t >= 1, because state vector contains Nhat_{t-1}.
    """
    states = np.asarray(states, dtype=float)
    T = states.shape[0]

    if T < 2:
        raise ValueError("Need T >= 2 to sample AR(2) coefficients in state-vector form.")

    q = np.column_stack([e[1:], zeta[1:], eps[1:]])

    mean_u, var_u = _conditional_scalar_many(
        Sigma,
        target_idx=2,
        cond_idx=[0, 1, 3],
        cond_values=q,
    )

    y_reg = states[1:, 0] - mean_u
    X_reg = np.column_stack([states[:-1, 0], states[:-1, 1]])

    prior_prec = np.diag([1.0 / sigma_rho1**2, 1.0 / sigma_rho2**2])

    post_cov = inv(X_reg.T @ X_reg / var_u + prior_prec)
    post_mean = post_cov @ (
        X_reg.T @ y_reg / var_u
        + prior_prec @ np.array([mu_rho1, mu_rho2], dtype=float)
    )

    if not enforce_stationary:
        draw = _mvnrnd(post_mean, post_cov, rng)
        return float(draw[0]), float(draw[1])

    for _ in range(max_tries):
        draw = _mvnrnd(post_mean, post_cov, rng)
        r1, r2 = float(draw[0]), float(draw[1])
        if _is_stationary_ar2(r1, r2):
            return r1, r2

    raise RuntimeError(
        "Failed to draw stationary AR(2) coefficients after max_tries. "
        "Try weaker priors or enforce_stationary=False."
    )


def _sample_n_full(
    *,
    states: np.ndarray,
    e: np.ndarray,
    zeta: np.ndarray,
    u: np.ndarray,
    Sigma: np.ndarray,
    mu_n: float,
    sigma_n: float,
    rng: np.random.Generator,
) -> float:
    """
    Trend equation:

        Nbar_t = n + Nbar_{t-1} + epsilon_t

    With full Sigma, use:

        epsilon_t | e_t, zeta_t, u_t
    """
    states = np.asarray(states, dtype=float)
    T = states.shape[0]

    if T < 2:
        raise ValueError("Need T >= 2 to sample n.")

    q = np.column_stack([e[1:], zeta[1:], u[1:]])

    mean_eps, var_eps = _conditional_scalar_many(
        Sigma,
        target_idx=3,
        cond_idx=[0, 1, 2],
        cond_values=q,
    )

    dNbar = states[1:, 2] - states[:-1, 2]
    y_n = dNbar - mean_eps

    prec = 1.0 / sigma_n**2 + y_n.size / var_eps
    mean_num = mu_n / sigma_n**2 + float(np.sum(y_n)) / var_eps

    return float(mean_num / prec + rng.standard_normal() / np.sqrt(prec))


# ============================================================
# Joint Kalman filter + FFBS
# ============================================================

def _sample_states_joint_ffbs_fullSigma(
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
    sigma_N2: float,
    m0: np.ndarray,
    P0: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Joint FFBS for:

        s_t = [Nhat_t, Nhat_{t-1}, Nbar_t]'

    State equations:

        Nhat_t = rho1*Nhat_{t-1} + rho2*Nhat_{t-2} + u_t
        Nbar_t = n + Nbar_{t-1} + epsilon_t

    Measurement equations:

        N_obs_t = Nhat_t + Nbar_t + nu_t

        pi_t - Epi_t - alpha*(pi_{t-1}-Epi_t) - kappa*x_t
            = -theta*Nhat_t + e_t

    Full covariance:

        [e_t, zeta_t, u_t, epsilon_t]' ~ N(0, Sigma)

    Since x_t is observed, zeta_t is known conditional on phi.
    Therefore the Kalman system uses:

        [e_t, u_t, epsilon_t] | zeta_t
    """
    N_obs = _as_1d(N_obs)
    pi_t = _as_1d(pi_t)
    pi_tm1 = _as_1d(pi_tm1)
    pi_expect = _as_1d(pi_expect)
    x_t = _as_1d(x_t)
    zeta = _as_1d(zeta)

    T = N_obs.size

    if not (pi_t.size == pi_tm1.size == pi_expect.size == x_t.size == zeta.size == T):
        raise ValueError("All series must have the same length.")

    # Conditional distribution of r_t = [e_t, u_t, epsilon_t]' given zeta_t.
    idx_r = [0, 2, 3]
    idx_z = [1]

    Szz = float(Sigma[1, 1])
    if Szz <= 0.0:
        raise ValueError("Sigma[1,1] must be positive.")

    B = Sigma[np.ix_(idx_r, idx_z)] / Szz
    S_r_given_z = Sigma[np.ix_(idx_r, idx_r)] - Sigma[np.ix_(idx_r, idx_z)] @ Sigma[np.ix_(idx_z, idx_r)] / Szz
    S_r_given_z = _force_pd(S_r_given_z)

    means_r = zeta.reshape(-1, 1) @ B.T

    mu_e = means_r[:, 0]
    mu_u = means_r[:, 1]
    mu_eps = means_r[:, 2]

    var_e = float(S_r_given_z[0, 0])
    var_u = float(S_r_given_z[1, 1])
    var_eps = float(S_r_given_z[2, 2])

    cov_eu = float(S_r_given_z[0, 1])
    cov_eeps = float(S_r_given_z[0, 2])
    cov_u_eps = float(S_r_given_z[1, 2])

    F = np.array(
        [
            [rho1, rho2, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )

    c = np.array([0.0, 0.0, n_drift], dtype=float)

    # State innovation covariance for [u_t, 0, epsilon_t]'
    Q = np.array(
        [
            [var_u, 0.0, cov_u_eps],
            [0.0, 0.0, 0.0],
            [cov_u_eps, 0.0, var_eps],
        ],
        dtype=float,
    )

    # Measurement error covariance for [nu_t, e_t]'
    R = np.array(
        [
            [sigma_N2, 0.0],
            [0.0, var_e],
        ],
        dtype=float,
    )

    # Cross covariance:
    # Cov(state innovation [u_t,0,epsilon_t]', measurement error [nu_t,e_t]')
    C_base = np.array(
        [
            [0.0, cov_eu],
            [0.0, 0.0],
            [0.0, cov_eeps],
        ],
        dtype=float,
    )

    m_pred = np.zeros((T, 3), dtype=float)
    P_pred = np.zeros((T, 3, 3), dtype=float)

    m_filt = np.zeros((T, 3), dtype=float)
    P_filt = np.zeros((T, 3, 3), dtype=float)

    y_pi = (
        pi_t
        - pi_expect
        - alpha * (pi_tm1 - pi_expect)
        - (kappa / KAPPA_SCALE) * x_t
    )

    H = np.array(
        [
            [1.0, 0.0, 1.0],
            [-theta, 0.0, 0.0],
        ],
        dtype=float,
    )

    # ---------- Forward filter ----------
    for t in range(T):
        if t == 0:
            m_pred[t] = _as_1d(m0)
            P_pred[t] = _force_pd(P0)
            C = np.zeros((3, 2), dtype=float)
        else:
            mu_w_t = np.array([mu_u[t], 0.0, mu_eps[t]], dtype=float)
            m_pred[t] = c + mu_w_t + F @ m_filt[t - 1]
            P_pred[t] = _force_pd(F @ P_filt[t - 1] @ F.T + Q)
            C = C_base

        y_obs = np.array([N_obs[t], y_pi[t]], dtype=float)
        v_mean = np.array([0.0, mu_e[t]], dtype=float)

        innov = y_obs - H @ m_pred[t] - v_mean

        S = H @ P_pred[t] @ H.T + R + H @ C + C.T @ H.T
        S = _force_pd(S)

        cross = P_pred[t] @ H.T + C
        K = cross @ inv(S)

        m_filt[t] = m_pred[t] + K @ innov
        P_filt[t] = _force_pd(P_pred[t] - K @ S @ K.T)

    # ---------- Backward sampling ----------
    states = np.zeros((T, 3), dtype=float)
    states[-1] = _mvnrnd(m_filt[-1], P_filt[-1], rng)

    for t in range(T - 2, -1, -1):
        A = P_filt[t] @ F.T @ inv(_force_pd(P_pred[t + 1]))

        mean_s = m_filt[t] + A @ (states[t + 1] - m_pred[t + 1])
        cov_s = _force_pd(P_filt[t] - A @ P_pred[t + 1] @ A.T)

        states[t] = _mvnrnd(mean_s, cov_s, rng)

    Nhat = states[:, 0]
    Nbar = states[:, 2]

    return Nbar, Nhat, states


# ============================================================
# Main Gibbs sampler
# ============================================================

def func_nkpc_hsa_decomp_joint_fullSigma(
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
) -> dict[str, Any]:
    """
    Gibbs sampler for:

        pi_t = alpha*pi_{t-1}
               + (1-alpha)*E_t*pi_{t+1}
               + kappa*x_t
               - theta*Nhat_t
               + e_t

        x_t = phi_1*x_{t-1} + zeta_t

        N_obs_t = Nhat_t + Nbar_t + nu_t

        Nhat_t = rho_1*Nhat_{t-1} + rho_2*Nhat_{t-2} + u_t

        Nbar_t = n + Nbar_{t-1} + epsilon_t

        [e_t, zeta_t, u_t, epsilon_t]' ~ N(0, Sigma)

    Measurement error:

        nu_t ~ N(0, sigma_N^2)

    nu_t is assumed independent of [e_t, zeta_t, u_t, epsilon_t].
    """
    pi_t = _as_1d(pi_data)
    pi_tm1 = _as_1d(pi_prev_data)
    pi_expect = _as_1d(Epi_data)
    x_t = _as_1d(x_data)
    x_tm1 = _as_1d(x_prev_data)
    N_obs = _as_1d(N_data)

    T = pi_t.size

    if not (pi_tm1.size == pi_expect.size == x_t.size == x_tm1.size == N_obs.size == T):
        raise ValueError("All input series must have the same length.")
    if T < 3:
        raise ValueError("Need T >= 3.")
    if n_burn < 0:
        raise ValueError("n_burn must be nonnegative.")
    if n_keep <= 0:
        raise ValueError("n_keep must be positive.")

    pri = _common_priors(priors or {})
    opts = opts or {}

    _assert_all_pos(
        [
            pri["sigma_alpha"],
            pri["sigma_kappa"],
            pri["sigma_theta"],
            pri["sigma_phi"],
            pri["sigma_rho1"],
            pri["sigma_rho2"],
            pri["sigma_n"],
            pri["a_N"],
            pri["b_N"],
            pri["P0_Nhat"],
            pri["P0_Nhat_lag"],
            pri["P0_Nbar"],
        ],
        "Prior scales and inverse-gamma hyperparameters must be positive.",
    )

    if pri["nu_Sigma"] <= 3:
        raise ValueError("nu_Sigma must be greater than 3 for a 4x4 inverse-Wishart prior.")

    covariance_structure = str(_getd(opts, "covariance_structure", "e_zeta_only"))
    S_Sigma = _restrict_sigma_structure(np.asarray(pri["S_Sigma"], dtype=float), covariance_structure)
    if S_Sigma.shape != (4, 4):
        raise ValueError("S_Sigma must be a 4x4 matrix.")

    rng = np.random.default_rng(_getd(opts, "seed", None))

    alpha = float(_getd(opts, "alpha0", pri["mu_alpha"]))
    kappa = float(_getd(opts, "kappa0", pri["mu_kappa"]))
    theta = float(_getd(opts, "theta0", pri["mu_theta"]))
    phi_1 = float(_getd(opts, "phi10", pri["mu_phi"]))

    rho1 = float(_getd(opts, "rho10", pri["mu_rho1"]))
    rho2 = float(_getd(opts, "rho20", pri["mu_rho2"]))
    n_drift = float(_getd(opts, "n0", pri["mu_n"]))

    sigma_N2 = float(_getd(opts, "sigma_N20", 1.0))
    _assert_all_pos([sigma_N2], "Initial sigma_N2 must be positive.")

    Sigma0 = _getd(opts, "Sigma0", None)
    if Sigma0 is None:
        Sigma = np.diag(
            [
                float(_getd(opts, "sigma_e20", 1.0)),
                float(_getd(opts, "sigma_zeta20", 1.0)),
                float(_getd(opts, "sigma_u20", 0.5)),
                float(_getd(opts, "sigma_eps20", 0.1)),
            ]
        )
    else:
        Sigma = np.asarray(Sigma0, dtype=float)

    Sigma = _restrict_sigma_structure(Sigma, covariance_structure)
    if Sigma.shape != (4, 4):
        raise ValueError("Sigma0 must be 4x4.")

    enforce_stationary = bool(_getd(opts, "enforce_stationary", True))
    store_every = int(max(1, _getd(opts, "store_every", 1)))
    verbose = bool(_getd(opts, "verbose", False))
    coefficient_constraints = _getd(opts, "coefficient_constraints", {})
    constraint_stats: dict[str, int] = {}

    n_store = int(n_keep // store_every)
    if n_store <= 0:
        raise ValueError("No draws would be stored. Use n_keep >= store_every.")

    Nbar, Nhat, states = _init_states(N_obs)

    a_t = pi_tm1 - pi_expect
    y = pi_t - pi_expect

    m0 = np.array(
        [
            float(_getd(opts, "m0_Nhat", pri["m0_Nhat"])),
            float(_getd(opts, "m0_Nhat_lag", pri["m0_Nhat_lag"])),
            float(_getd(opts, "m0_Nbar", N_obs[0])),
        ],
        dtype=float,
    )

    P0 = np.diag(
        [
            float(_getd(opts, "P0_Nhat", pri["P0_Nhat"])),
            float(_getd(opts, "P0_Nhat_lag", pri["P0_Nhat_lag"])),
            float(_getd(opts, "P0_Nbar", pri["P0_Nbar"])),
        ]
    )

    alpha_draws = np.zeros(n_store)
    kappa_draws = np.zeros(n_store)
    theta_draws = np.zeros(n_store)
    phi_draws = np.zeros(n_store)
    rho1_draws = np.zeros(n_store)
    rho2_draws = np.zeros(n_store)
    n_draws = np.zeros(n_store)

    sigma_N_draws = np.zeros(n_store)
    sigma_e_draws = np.zeros(n_store)
    sigma_zeta_draws = np.zeros(n_store)
    sigma_u_draws = np.zeros(n_store)
    sigma_eps_draws = np.zeros(n_store)

    corr_e_zeta_draws = np.zeros(n_store)
    corr_e_u_draws = np.zeros(n_store)
    corr_e_eps_draws = np.zeros(n_store)
    corr_u_eps_draws = np.zeros(n_store)

    Sigma_draws = np.zeros((n_store, 4, 4))
    Nbar_draws = np.zeros((n_store, T))
    Nhat_draws = np.zeros((n_store, T))

    total_iter = n_burn + n_keep
    store_idx = 0

    for it in range(1, total_iter + 1):
        # ------------------------------------------------------------
        # Current residuals
        # ------------------------------------------------------------
        zeta = x_t - phi_1 * x_tm1
        u, eps = _compute_state_residuals(states, rho1, rho2, n_drift)

        kappa_eff = kappa / KAPPA_SCALE
        e = y - alpha * a_t - kappa_eff * x_t + theta * Nhat

        # ------------------------------------------------------------
        # 1. Draw alpha, kappa, theta
        # ------------------------------------------------------------
        alpha, kappa, theta = _sample_alpha_kappa_theta_full(
            y=y,
            a_t=a_t,
            x_t=x_t,
            Nhat=Nhat,
            zeta=zeta,
            u=u,
            eps=eps,
            Sigma=Sigma,
            pri=pri,
            rng=rng,
            coefficient_constraints=coefficient_constraints,
            constraint_stats=constraint_stats,
        )

        kappa_eff = kappa / KAPPA_SCALE
        e = y - alpha * a_t - kappa_eff * x_t + theta * Nhat

        # ------------------------------------------------------------
        # 2. Draw phi_1
        # ------------------------------------------------------------
        phi_1 = _sample_phi_full(
            x_t=x_t,
            x_tm1=x_tm1,
            e=e,
            u=u,
            eps=eps,
            Sigma=Sigma,
            mu_phi=pri["mu_phi"],
            sigma_phi=pri["sigma_phi"],
            rng=rng,
        )

        zeta = x_t - phi_1 * x_tm1

        # ------------------------------------------------------------
        # 3. Draw rho1, rho2
        # ------------------------------------------------------------
        rho1, rho2 = _sample_ar2_coeffs_full(
            states=states,
            e=e,
            zeta=zeta,
            eps=eps,
            Sigma=Sigma,
            mu_rho1=pri["mu_rho1"],
            sigma_rho1=pri["sigma_rho1"],
            mu_rho2=pri["mu_rho2"],
            sigma_rho2=pri["sigma_rho2"],
            enforce_stationary=enforce_stationary,
            rng=rng,
        )

        u, eps = _compute_state_residuals(states, rho1, rho2, n_drift)

        # ------------------------------------------------------------
        # 4. Draw n
        # ------------------------------------------------------------
        n_drift = _sample_n_full(
            states=states,
            e=e,
            zeta=zeta,
            u=u,
            Sigma=Sigma,
            mu_n=pri["mu_n"],
            sigma_n=pri["sigma_n"],
            rng=rng,
        )

        u, eps = _compute_state_residuals(states, rho1, rho2, n_drift)

        # ------------------------------------------------------------
        # 5. Draw full Sigma
        # ------------------------------------------------------------
        Sigma = _sample_Sigma_full(
            e=e,
            zeta=zeta,
            u=u,
            eps=eps,
            nu0=pri["nu_Sigma"],
            S0=S_Sigma,
            rng=rng,
        )
        Sigma = _restrict_sigma_structure(Sigma, covariance_structure)

        # ------------------------------------------------------------
        # 6. Draw measurement error variance sigma_N2
        # ------------------------------------------------------------
        resid_N = N_obs - Nhat - Nbar

        sigma_N2 = _sample_invgamma(
            pri["a_N"] + 0.5 * T,
            pri["b_N"] + 0.5 * float(np.sum(resid_N**2)),
            rng,
        )

        # ------------------------------------------------------------
        # 7. Draw latent states jointly by Kalman filter + FFBS
        # ------------------------------------------------------------
        states_old = states

        Nbar, Nhat, states = _sample_states_joint_ffbs_fullSigma(
            N_obs=N_obs,
            pi_t=pi_t,
            pi_tm1=pi_tm1,
            pi_expect=pi_expect,
            x_t=x_t,
            zeta=zeta,
            alpha=alpha,
            kappa=kappa,
            theta=theta,
            rho1=rho1,
            rho2=rho2,
            n_drift=n_drift,
            Sigma=Sigma,
            sigma_N2=sigma_N2,
            m0=m0,
            P0=P0,
            rng=rng,
        )

        # ------------------------------------------------------------
        # 8. Store
        # ------------------------------------------------------------
        if it > n_burn and (it - n_burn) % store_every == 0:
            alpha_draws[store_idx] = alpha
            kappa_draws[store_idx] = kappa / KAPPA_SCALE
            theta_draws[store_idx] = theta
            phi_draws[store_idx] = phi_1
            rho1_draws[store_idx] = rho1
            rho2_draws[store_idx] = rho2
            n_draws[store_idx] = n_drift

            sigma_N_draws[store_idx] = np.sqrt(sigma_N2)
            sigma_e_draws[store_idx] = np.sqrt(Sigma[0, 0])
            sigma_zeta_draws[store_idx] = np.sqrt(Sigma[1, 1])
            sigma_u_draws[store_idx] = np.sqrt(Sigma[2, 2])
            sigma_eps_draws[store_idx] = np.sqrt(Sigma[3, 3])

            corr_e_zeta_draws[store_idx] = Sigma[0, 1] / np.sqrt(Sigma[0, 0] * Sigma[1, 1])
            corr_e_u_draws[store_idx] = Sigma[0, 2] / np.sqrt(Sigma[0, 0] * Sigma[2, 2])
            corr_e_eps_draws[store_idx] = Sigma[0, 3] / np.sqrt(Sigma[0, 0] * Sigma[3, 3])
            corr_u_eps_draws[store_idx] = Sigma[2, 3] / np.sqrt(Sigma[2, 2] * Sigma[3, 3])

            Sigma_draws[store_idx] = Sigma
            Nbar_draws[store_idx] = Nbar
            Nhat_draws[store_idx] = Nhat

            store_idx += 1

        if verbose and it % 5000 == 0:
            print(
                f"Iter {it}/{total_iter}: "
                f"alpha={alpha:.3f}, "
                f"kappa={kappa:.3f}, "
                f"theta={theta:.3f}, "
                f"rho1={rho1:.3f}, "
                f"rho2={rho2:.3f}, "
                f"n={n_drift:.3f}, "
                f"sigma_N={np.sqrt(sigma_N2):.3f}"
            )

    return {
        "alpha": _summary(alpha_draws),
        "kappa": _summary(kappa_draws),
        "theta": _summary(theta_draws),
        "phi_1": _summary(phi_draws),
        "rho1": _summary(rho1_draws),
        "rho2": _summary(rho2_draws),
        "n": _summary(n_draws),

        "sigma_N": _summary(sigma_N_draws),
        "sigma_e": _summary(sigma_e_draws),
        "sigma_zeta": _summary(sigma_zeta_draws),
        "sigma_u": _summary(sigma_u_draws),
        "sigma_eps": _summary(sigma_eps_draws),

        "corr_e_zeta": _summary(corr_e_zeta_draws),
        "corr_e_u": _summary(corr_e_u_draws),
        "corr_e_eps": _summary(corr_e_eps_draws),
        "corr_u_eps": _summary(corr_u_eps_draws),

        "Sigma": _summary(Sigma_draws),

        "state_draws": {
            "Nbar": Nbar_draws,
            "Nhat": Nhat_draws,
        },

        "priors": priors or {},
        "opts": opts,
        "model": {
            "inflation": "pi_t = alpha*pi_{t-1} + (1-alpha)*E_t*pi_{t+1} + kappa*x_t - theta*Nhat_t + e_t",
            "x": "x_t = phi_1*x_{t-1} + zeta_t",
            "measurement": "N_obs_t = Nhat_t + Nbar_t + nu_t",
            "gap": "Nhat_t = rho1*Nhat_{t-1} + rho2*Nhat_{t-2} + u_t",
            "trend": "Nbar_t = n + Nbar_{t-1} + epsilon_t",
            "Sigma_order": "[e_t, zeta_t, u_t, epsilon_t]",
            "covariance_structure": covariance_structure,
            "nu_independent": True,
            "state_vector": "[Nhat_t, Nhat_{t-1}, Nbar_t]'",
            "kappa_scale": KAPPA_SCALE,
            "kappa_internal": "stored kappa * KAPPA_SCALE",
            "stored_units": "physical",
            "coefficient_constraints": coefficient_constraints,
            "coefficient_constraint_stats": constraint_stats_summary(constraint_stats),
        },
    }

func_nkpc_hsa_decomp = func_nkpc_hsa_decomp_joint_fullSigma

__all__ = [
    "func_nkpc_hsa_decomp_joint_fullSigma",
    "func_nkpc_hsa_decomp",
]
