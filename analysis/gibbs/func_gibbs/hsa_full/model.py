from __future__ import annotations

from typing import Any, Optional

import numpy as np
from numpy.linalg import inv

from analysis.gibbs.func_gibbs.common.constraints import constraint_stats_summary, draw_with_constraints


KAPPA_SCALE = 100.0


def _getd(d: Optional[dict[str, Any]], key: str, default: Any) -> Any:
    if isinstance(d, dict) and key in d and d[key] is not None:
        return d[key]
    return default


def _assert_all_pos(arr: Any, msg: str) -> None:
    values = np.asarray(arr, dtype=float)
    if np.any(~np.isfinite(values)) or np.any(values <= 0.0):
        raise ValueError(msg)


def _as_1d(x: Any) -> np.ndarray:
    return np.asarray(x, dtype=float).reshape(-1)


def _is_stationary_ar2(r1: float, r2: float) -> bool:
    return (abs(r2) < 1.0) and ((r1 + r2) < 1.0) and ((r2 - r1) < 1.0)


def _force_pd(S: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    S = (np.asarray(S, dtype=float) + np.asarray(S, dtype=float).T) / 2.0
    vals, vecs = np.linalg.eigh(S)
    vals = np.maximum(vals, eps)
    repaired = (vecs * vals) @ vecs.T
    return (repaired + repaired.T) / 2.0


def _mvnrnd(mean: np.ndarray, cov: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    return rng.multivariate_normal(np.asarray(mean, dtype=float), _force_pd(cov))


def _sample_invgamma(a_post: float, b_post: float, rng: np.random.Generator) -> float:
    return 1.0 / rng.gamma(shape=a_post, scale=1.0 / b_post)


def _sample_beta_gaussian(
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
    Vn = inv(X.T @ X / sigma2 + V0_inv)
    mn = Vn @ (X.T @ y / sigma2 + V0_inv @ prior_mean)
    return _mvnrnd(mn, Vn, rng)


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


def _sample_ar2_states_ffbs(
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
        P[:, :, t] = _force_pd(P_pred[:, :, t] - K @ H @ P_pred[:, :, t])

    states = np.zeros((2, T), dtype=float)
    states[:, -1] = _mvnrnd(m[:, -1], P[:, :, -1], rng)

    for t in range(T - 2, -1, -1):
        Ptp1_pred = F @ P[:, :, t] @ F.T + Q
        A = P[:, :, t] @ F.T @ inv(Ptp1_pred)
        m_s = m[:, t] + A @ (states[:, t + 1] - F @ m[:, t])
        P_s = _force_pd(P[:, :, t] - A @ Ptp1_pred @ A.T)
        states[:, t] = _mvnrnd(m_s, P_s, rng)

    return states


def _sample_ar2_states_ffbs_tv_theta(
    y_target: np.ndarray,
    rho1: float,
    rho2: float,
    sigma_state2: float,
    pi_t: np.ndarray,
    alpha: float,
    pi_tm1: np.ndarray,
    pi_expect: np.ndarray,
    x_t: np.ndarray,
    sigma_obs2: float,
    target_scale: float,
    rng: np.random.Generator,
    *,
    kappa: float | None = None,
    kappa_t: np.ndarray | None = None,
    theta: float | None = None,
    theta_t: np.ndarray | None = None,
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
    if theta_t is None:
        if theta is None:
            raise ValueError("Either theta or theta_t must be provided.")
        theta_series = np.full(T, float(theta), dtype=float)
    else:
        theta_series = _as_1d(theta_t)
        if theta_series.size != T:
            raise ValueError(f"theta_t must have length T={T}, got {theta_series.size}.")

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

        H = np.array([[1.0, 0.0], [theta_series[t], 0.0]], dtype=float)
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
        P[:, :, t] = _force_pd(P_pred[:, :, t] - K @ H @ P_pred[:, :, t])

    states = np.zeros((2, T), dtype=float)
    states[:, -1] = _mvnrnd(m[:, -1], P[:, :, -1], rng)

    for t in range(T - 2, -1, -1):
        Ptp1_pred = F @ P[:, :, t] @ F.T + Q
        A = P[:, :, t] @ F.T @ inv(Ptp1_pred)
        m_s = m[:, t] + A @ (states[:, t + 1] - F @ m[:, t])
        P_s = _force_pd(P[:, :, t] - A @ Ptp1_pred @ A.T)
        states[:, t] = _mvnrnd(m_s, P_s, rng)

    return states


def _sample_rw_states_ffbs(
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


def _sample_rw_states_ffbs_tv_theta_kappa(
    y_target: np.ndarray,
    n_drift: float,
    sigma_state2: float,
    target_scale: float,
    pi_t: np.ndarray,
    alpha: float,
    pi_tm1: np.ndarray,
    pi_expect: np.ndarray,
    x_t: np.ndarray,
    Nhat: np.ndarray,
    kappa0: float,
    delta: float,
    theta0: float,
    gamma: float,
    sigma_obs2: float,
    obs_offset: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    y_target = _as_1d(y_target)
    pi_t = _as_1d(pi_t)
    pi_tm1 = _as_1d(pi_tm1)
    pi_expect = _as_1d(pi_expect)
    x_t = _as_1d(x_t)
    Nhat = _as_1d(Nhat)
    obs_offset = _as_1d(obs_offset)

    T = y_target.size
    if not (pi_t.size == pi_tm1.size == pi_expect.size == x_t.size == Nhat.size == obs_offset.size == T):
        raise ValueError("All input series must have the same length.")

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

        y1 = y_target[t]
        h1 = 1.0
        r1 = max(sigma_state2 * target_scale, 1e-10)

        y2 = (
            pi_t[t]
            - pi_expect[t]
            - alpha * (pi_tm1[t] - pi_expect[t])
            - kappa0 * x_t[t]
            + theta0 * Nhat[t]
            - obs_offset[t]
        )
        h2 = delta * x_t[t] - gamma * Nhat[t]
        r2 = max(sigma_obs2, 1e-10)

        H = np.array([[1.0], [h2]], dtype=float)
        y = np.array([y1, y2], dtype=float)
        R = np.diag([r1, r2])

        S = P_pred[t] * (H @ H.T) + R
        K = (P_pred[t] * H.T) @ inv(S)

        innov = y - (H[:, 0] * m_pred[t])
        m[t] = m_pred[t] + float((K @ innov)[0])
        P[t] = max(P_pred[t] - float((K @ H)[0, 0]) * P_pred[t], 1e-10)

    states = np.zeros(T, dtype=float)
    states[-1] = m[-1] + np.sqrt(max(P[-1], 1e-8)) * rng.standard_normal()

    for t in range(T - 2, -1, -1):
        Ptp1_pred = P[t] + sigma_state2
        A = P[t] / Ptp1_pred

        m_s = m[t] + A * (states[t + 1] - (n_drift + m[t]))
        P_s = max(P[t] - A * Ptp1_pred * A, 1e-10)

        states[t] = m_s + np.sqrt(P_s) * rng.standard_normal()

    return states


def _summary(draws: np.ndarray) -> dict[str, Any]:
    arr = np.asarray(draws, dtype=float)
    if arr.shape[0] == 0:
        raise ValueError("No posterior draws were stored. Check n_keep and store_every.")
    qs = np.quantile(arr, [0.025, 0.05, 0.25, 0.5, 0.75, 0.95, 0.975], axis=0)
    return {
        "draws": arr,
        "mean": np.mean(arr, axis=0),
        "std": np.std(arr, axis=0, ddof=1) if arr.shape[0] > 1 else np.zeros_like(np.mean(arr, axis=0)),
        "quantiles": qs,
    }


def _init_states(N_obs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    T = N_obs.size
    Nbar = np.zeros(T, dtype=float)
    k0 = min(2, T)
    Nbar[:k0] = N_obs[:k0]
    for t in range(2, T):
        Nbar[t] = 0.7 * Nbar[t - 1] + 0.3 * N_obs[t]
    return Nbar, N_obs - Nbar


def _sample_ar2_coeffs(
    Nhat: np.ndarray,
    sigma_state2: float,
    mu_rho1: float,
    sigma_rho1: float,
    mu_rho2: float,
    sigma_rho2: float,
    enforce_stationary: bool,
    rng: np.random.Generator,
) -> tuple[float, float]:
    y = Nhat[2:]
    X = np.column_stack([Nhat[1:-1], Nhat[:-2]])
    prior_prec = np.diag([1.0 / sigma_rho1**2, 1.0 / sigma_rho2**2])
    post_cov = inv(X.T @ X / sigma_state2 + prior_prec)
    post_mean = post_cov @ (
        X.T @ y / sigma_state2 + prior_prec @ np.array([mu_rho1, mu_rho2], dtype=float)
    )
    for _ in range(2000):
        draw = _mvnrnd(post_mean, post_cov, rng)
        if (not enforce_stationary) or _is_stationary_ar2(float(draw[0]), float(draw[1])):
            return float(draw[0]), float(draw[1])
    return float(post_mean[0]), float(post_mean[1])


def _common_priors(priors: dict[str, Any]) -> dict[str, float]:
    return {
        "mu_alpha": _getd(priors, "mu_alpha", 0.5),
        "sigma_alpha": _getd(priors, "sigma_alpha", 0.2),
        "mu_kappa0": _getd(priors, "mu_kappa", _getd(priors, "mu_kappa_0", 10.0)),
        "sigma_kappa0": _getd(priors, "sigma_kappa", _getd(priors, "sigma_kappa_0", 20.0)),
        "mu_delta": _getd(priors, "mu_delta", 10.0),
        "sigma_delta": _getd(priors, "sigma_delta", 20.0),
        "mu_theta": _getd(priors, "mu_theta", 0.1),
        "sigma_theta": _getd(priors, "sigma_theta", 0.2),
        "mu_gamma": _getd(priors, "mu_gamma", 0.1),
        "sigma_gamma": _getd(priors, "sigma_gamma", 0.2),
        "mu_phi": _getd(priors, "mu_phi_1", 0.7),
        "sigma_phi": _getd(priors, "sigma_phi_1", 0.2),
        "mu_lambda": _getd(priors, "mu_lambda", 0.0),
        "sigma_lambda": _getd(priors, "sigma_lambda", 0.5),
        "mu_rho1": _getd(priors, "mu_rho1", 0.5),
        "sigma_rho1": _getd(priors, "sigma_rho1", 0.2),
        "mu_rho2": _getd(priors, "mu_rho2", -0.5),
        "sigma_rho2": _getd(priors, "sigma_rho2", 0.2),
        "mu_n": _getd(priors, "mu_n", 0.0),
        "sigma_n": _getd(priors, "sigma_n", 0.1),
        "a_e": _getd(priors, "a_e", _getd(priors, "a_v", 2.0)),
        "b_e": _getd(priors, "b_e", _getd(priors, "b_v", 2.0)),
        "a_u": _getd(priors, "a_u", 2.0),
        "b_u": _getd(priors, "b_u", 2.0),
        "a_eps": _getd(priors, "a_eps", 2.0),
        "b_eps": _getd(priors, "b_eps", 2.0),
        "a_z": _getd(priors, "a_z", 0.001),
        "b_z": _getd(priors, "b_z", 0.001),
    }


def _sample_phi_joint(
    *,
    x_t: np.ndarray,
    x_tm1: np.ndarray,
    y_tilde: np.ndarray,
    lambda_ez: float,
    sigma_zeta2: float,
    sigma_eta2: float,
    mu_phi: float,
    sigma_phi: float,
    rng: np.random.Generator,
) -> float:
    prec = (
        1.0 / sigma_phi**2
        + float(np.sum(x_tm1**2)) / sigma_zeta2
        + (lambda_ez**2) * float(np.sum(x_tm1**2)) / sigma_eta2
    )
    mean_num = (
        mu_phi / sigma_phi**2
        + float(np.dot(x_tm1, x_t)) / sigma_zeta2
        - lambda_ez * float(np.dot(x_tm1, y_tilde - lambda_ez * x_t)) / sigma_eta2
    )
    return float(mean_num / prec + rng.standard_normal() / np.sqrt(prec))


def func_nkpc_hsa_full(
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
    _assert_all_pos(
        [
            pri["sigma_alpha"],
            pri["sigma_kappa0"],
            pri["sigma_delta"],
            pri["sigma_theta"],
            pri["sigma_gamma"],
            pri["sigma_phi"],
            pri["sigma_lambda"],
            pri["sigma_rho1"],
            pri["sigma_rho2"],
            pri["sigma_n"],
            pri["a_e"],
            pri["b_e"],
            pri["a_u"],
            pri["b_u"],
            pri["a_eps"],
            pri["b_eps"],
            pri["a_z"],
            pri["b_z"],
        ],
        "Full HSA prior scales must be positive.",
    )

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
    target_scale = float(_getd(opts, "target_scale", _getd(opts, "r_target_scale", 0.1)))
    rw_scale = float(_getd(opts, "rw_scale", _getd(opts, "r_rw_scale", 0.1)))
    enforce_stationary = bool(_getd(opts, "enforce_stationary", True))
    store_every = int(max(1, _getd(opts, "store_every", 1)))
    verbose = bool(_getd(opts, "verbose", False))
    coefficient_constraints = _getd(opts, "coefficient_constraints", {})
    constraint_stats: dict[str, int] = {}
    rng = np.random.default_rng(_getd(opts, "seed", None))

    Nbar, Nhat = _init_states(N_obs)
    a_t = pi_tm1 - pi_expect
    lambda_prec0 = 0.0 if orth else 1.0 / pri["sigma_lambda"]**2

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
    rho_ez_draws = np.zeros(n_store)
    Nbar_draws = np.zeros((n_store, T))
    Nhat_draws = np.zeros((n_store, T))
    kappa_t_draws = np.zeros((n_store, T))
    theta_t_draws = np.zeros((n_store, T))

    total_iter = n_burn + n_keep
    store_idx = 0

    for it in range(1, total_iter + 1):
        kappa_t = kappa0 + delta * Nbar
        kappa_t_eff = kappa_t / KAPPA_SCALE
        theta_t = theta0 + gamma * Nbar
        zeta = x_t - phi_1 * x_tm1
        y = pi_t - pi_expect
        y_adj = y - lambda_ez * zeta

        X = np.column_stack([
            a_t,
            x_t / KAPPA_SCALE,
            (x_t * Nbar) / KAPPA_SCALE,
            -Nhat,
            -(Nhat * Nbar),
        ])
        beta_prior_mean = np.array(
            [
                pri["mu_alpha"],
                pri["mu_kappa0"],
                pri["mu_delta"],
                pri["mu_theta"],
                pri["mu_gamma"],
            ],
            dtype=float,
        )
        beta_prior_var = np.array(
            [
                pri["sigma_alpha"]**2,
                pri["sigma_kappa0"]**2,
                pri["sigma_delta"]**2,
                pri["sigma_theta"]**2,
                pri["sigma_gamma"]**2,
            ],
            dtype=float,
        )
        beta = draw_with_constraints(
            lambda: _sample_beta_gaussian(
                y_adj,
                X,
                sigma2=sigma_eta2,
                prior_mean=beta_prior_mean,
                prior_var=beta_prior_var,
                rng=rng,
            ),
            ("alpha", "kappa_0", "delta", "theta_0", "gamma"),
            coefficient_constraints,
            stats=constraint_stats,
        )
        alpha = float(beta[0])
        kappa0 = float(beta[1])
        delta = float(beta[2])
        theta0 = float(beta[3])
        gamma = float(beta[4])
        kappa_t = kappa0 + delta * Nbar
        kappa_t_eff = kappa_t / KAPPA_SCALE
        theta_t = theta0 + gamma * Nbar

        if not orth:
            e_base = y - alpha * a_t - kappa_t_eff * x_t + theta_t * Nhat
            post_var_lambda = 1.0 / (lambda_prec0 + float(np.sum(zeta**2)) / sigma_eta2)
            post_mean_lambda = post_var_lambda * (
                pri["mu_lambda"] * lambda_prec0 + float(np.dot(zeta, e_base)) / sigma_eta2
            )
            lambda_ez = float(post_mean_lambda + np.sqrt(post_var_lambda) * rng.standard_normal())
        else:
            lambda_ez = 0.0

        y_tilde_phi = y - alpha * a_t - kappa_t_eff * x_t + theta_t * Nhat
        phi_1 = _sample_phi_joint(
            x_t=x_t,
            x_tm1=x_tm1,
            y_tilde=y_tilde_phi,
            lambda_ez=lambda_ez,
            sigma_zeta2=sigma_zeta2,
            sigma_eta2=sigma_eta2,
            mu_phi=pri["mu_phi"],
            sigma_phi=pri["sigma_phi"],
            rng=rng,
        )

        zeta = x_t - phi_1 * x_tm1
        eta = y - alpha * a_t - kappa_t_eff * x_t + theta_t * Nhat - lambda_ez * zeta
        sigma_zeta2 = _sample_invgamma(pri["a_z"] + 0.5 * T, pri["b_z"] + 0.5 * float(np.sum(zeta**2)), rng)
        sigma_eta2 = _sample_invgamma(pri["a_e"] + 0.5 * T, pri["b_e"] + 0.5 * float(np.sum(eta**2)), rng)

        if T >= 3:
            rho1, rho2 = _sample_ar2_coeffs(
                Nhat,
                sigma_u2,
                pri["mu_rho1"],
                pri["sigma_rho1"],
                pri["mu_rho2"],
                pri["sigma_rho2"],
                enforce_stationary,
                rng,
            )
            resid_u = Nhat[2:] - rho1 * Nhat[1:-1] - rho2 * Nhat[:-2]
            sigma_u2 = _sample_invgamma(pri["a_u"] + 0.5 * resid_u.size, pri["b_u"] + 0.5 * float(np.sum(resid_u**2)), rng)

        if T >= 2:
            dNbar = Nbar[1:] - Nbar[:-1]
            post_var_n = 1.0 / (1.0 / pri["sigma_n"]**2 + dNbar.size / sigma_eps2)
            post_mean_n = post_var_n * (pri["mu_n"] / pri["sigma_n"]**2 + float(np.sum(dNbar)) / sigma_eps2)
            n_drift = float(post_mean_n + np.sqrt(post_var_n) * rng.standard_normal())
            resid_eps = Nbar[1:] - n_drift - Nbar[:-1]
            sigma_eps2 = _sample_invgamma(pri["a_eps"] + 0.5 * resid_eps.size, pri["b_eps"] + 0.5 * float(np.sum(resid_eps**2)), rng)

        obs_offset = lambda_ez * zeta
        Nhat_states = _sample_ar2_states_ffbs_tv_theta(
            y_target=N_obs - Nbar,
            rho1=rho1,
            rho2=rho2,
            sigma_state2=sigma_u2,
            pi_t=pi_t,
            alpha=alpha,
            pi_tm1=pi_tm1,
            pi_expect=pi_expect,
            x_t=x_t,
            sigma_obs2=sigma_eta2,
            target_scale=target_scale,
            rng=rng,
            kappa_t=kappa_t_eff,
            theta_t=theta_t,
            obs_offset=obs_offset,
        )
        Nhat = Nhat_states[0]
        Nbar = _sample_rw_states_ffbs_tv_theta_kappa(
            y_target=N_obs - Nhat,
            n_drift=n_drift,
            sigma_state2=sigma_eps2,
            target_scale=rw_scale,
            pi_t=pi_t,
            alpha=alpha,
            pi_tm1=pi_tm1,
            pi_expect=pi_expect,
            x_t=x_t,
            Nhat=Nhat,
            kappa0=kappa0 / KAPPA_SCALE,
            delta=delta / KAPPA_SCALE,
            theta0=theta0,
            gamma=gamma,
            sigma_obs2=sigma_eta2,
            obs_offset=obs_offset,
            rng=rng,
        )
        kappa_t = kappa0 + delta * Nbar
        kappa_t_eff = kappa_t / KAPPA_SCALE
        theta_t = theta0 + gamma * Nbar

        sigma_e = float(np.sqrt(lambda_ez**2 * sigma_zeta2 + sigma_eta2))
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
            rho_ez_draws[store_idx] = rho_ez
            Nbar_draws[store_idx] = Nbar
            Nhat_draws[store_idx] = Nhat
            kappa_t_draws[store_idx] = kappa_t_eff
            theta_t_draws[store_idx] = theta_t
            store_idx += 1

        if verbose and it % 5000 == 0:
            print(
                f"Iter {it}/{total_iter}: alpha={alpha:.3f}, kappa0={kappa0:.3f}, delta={delta:.3f}, theta0={theta0:.3f}, gamma={gamma:.3f}"
            )

    return {
        "alpha": _summary(alpha_draws),
        "kappa_0": _summary(kappa0_draws),
        "delta": _summary(delta_draws),
        "theta_0": _summary(theta0_draws),
        "gamma": _summary(gamma_draws),
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
        "state_draws": {
            "Nbar": Nbar_draws,
            "Nhat": Nhat_draws,
            "kappa_t": kappa_t_draws,
            "theta_t": theta_t_draws,
        },
        "priors": priors or {},
        "opts": opts,
        "model": {
            "kappa_scale": KAPPA_SCALE,
            "kappa_internal": "stored kappa_0, delta, and kappa_t multiplied by KAPPA_SCALE",
            "stored_units": "physical",
            "coefficient_constraints": coefficient_constraints,
            "coefficient_constraint_stats": constraint_stats_summary(constraint_stats),
        },
    }


__all__ = ["func_nkpc_hsa_full"]
