from __future__ import annotations

from typing import Any, Optional

import numpy as np
from numpy.linalg import inv

try:
    from .gibbs_ffbs import sample_ar2_states_ffbs, sample_rw_states_ffbs
    from .gibbs_utils import (
        assert_all_pos,
        getd,
        is_stationary_ar2,
        mvnrnd,
        sample_beta_gaussian,
        sample_invgamma,
    )
except ImportError:  # pragma: no cover
    from gibbs_ffbs import sample_ar2_states_ffbs, sample_rw_states_ffbs
    from gibbs_utils import (
        assert_all_pos,
        getd,
        is_stationary_ar2,
        mvnrnd,
        sample_beta_gaussian,
        sample_invgamma,
    )


def _summary(draws: np.ndarray) -> dict[str, Any]:
    arr = np.asarray(draws, dtype=float)
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
        draw = mvnrnd(post_mean, post_cov, rng)
        if (not enforce_stationary) or is_stationary_ar2(float(draw[0]), float(draw[1])):
            return float(draw[0]), float(draw[1])
    return float(post_mean[0]), float(post_mean[1])


def _common_priors(priors: dict[str, Any], *, steady: bool) -> dict[str, float]:
    out = {
        "mu_alpha": getd(priors, "mu_alpha", 0.5),
        "sigma_alpha": getd(priors, "sigma_alpha", 0.2),
        "mu_phi": getd(priors, "mu_phi_1", 0.7),
        "sigma_phi": getd(priors, "sigma_phi_1", 0.2),
        "mu_lambda": getd(priors, "mu_lambda", 0.0),
        "sigma_lambda": getd(priors, "sigma_lambda", 0.5),
        "mu_rho1": getd(priors, "mu_rho1", 0.5),
        "sigma_rho1": getd(priors, "sigma_rho1", 0.2),
        "mu_rho2": getd(priors, "mu_rho2", -0.5),
        "sigma_rho2": getd(priors, "sigma_rho2", 0.2),
        "mu_n": getd(priors, "mu_n", 0.0),
        "sigma_n": getd(priors, "sigma_n", 0.1),
        "a_e": getd(priors, "a_e", getd(priors, "a_v", 2.0)),
        "b_e": getd(priors, "b_e", getd(priors, "b_v", 2.0)),
        "a_u": getd(priors, "a_u", getd(priors, "a_eps", 2.0)),
        "b_u": getd(priors, "b_u", getd(priors, "b_eps", 2.0)),
        "a_eps": getd(priors, "a_eps", getd(priors, "a_eta", 2.0)),
        "b_eps": getd(priors, "b_eps", getd(priors, "b_eta", 2.0)),
        "a_z": getd(priors, "a_z", 0.001),
        "b_z": getd(priors, "b_z", 0.001),
    }
    if steady:
        out["mu_kappa0"] = getd(priors, "mu_kappa", getd(priors, "mu_kappa_0", 0.1))
        out["sigma_kappa0"] = getd(priors, "sigma_kappa", getd(priors, "sigma_kappa_0", 0.2))
        out["mu_delta"] = getd(priors, "mu_delta", getd(priors, "mu_gamma", 0.1))
        out["sigma_delta"] = getd(priors, "sigma_delta", getd(priors, "sigma_gamma", 0.2))
    else:
        out["mu_kappa"] = getd(priors, "mu_kappa", 0.1)
        out["sigma_kappa"] = getd(priors, "sigma_kappa", 0.2)
        out["mu_theta"] = getd(priors, "mu_theta", 0.1)
        out["sigma_theta"] = getd(priors, "sigma_theta", 0.2)
    return out


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


def func_nkpc_hsa_decomp(
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

    pri = _common_priors(priors or {}, steady=False)
    assert_all_pos(
        [
            pri["sigma_alpha"], pri["sigma_kappa"], pri["sigma_theta"], pri["sigma_phi"], pri["sigma_lambda"],
            pri["sigma_rho1"], pri["sigma_rho2"], pri["sigma_n"], pri["a_e"], pri["b_e"], pri["a_u"], pri["b_u"],
            pri["a_eps"], pri["b_eps"], pri["a_z"], pri["b_z"],
        ],
        "Dynamic HSA prior scales must be positive.",
    )

    opts = opts or {}
    alpha = float(getd(opts, "alpha0", pri["mu_alpha"]))
    kappa = float(getd(opts, "kappa0", pri["mu_kappa"]))
    theta = float(getd(opts, "theta0", pri["mu_theta"]))
    phi_1 = float(getd(opts, "phi10", pri["mu_phi"]))
    lambda_ez = 0.0 if orth else float(getd(opts, "lambda0", 0.0))
    rho1 = float(getd(opts, "rho10", 0.5))
    rho2 = float(getd(opts, "rho20", -0.5))
    n_drift = float(getd(opts, "n0", 0.01))
    sigma_eta2 = float(getd(opts, "sigma_e20", getd(opts, "sigma_v20", 1.0)))
    sigma_zeta2 = float(getd(opts, "sigma_zeta20", 1.0))
    sigma_u2 = float(getd(opts, "sigma_u20", getd(opts, "sigma_eps20", 0.5)))
    sigma_eps2 = float(getd(opts, "sigma_eps20", getd(opts, "sigma_eta20", 0.1)))
    target_scale = float(getd(opts, "target_scale", getd(opts, "r_target_scale", 0.1)))
    rw_scale = float(getd(opts, "rw_scale", getd(opts, "r_rw_scale", 0.1)))
    enforce_stationary = bool(getd(opts, "enforce_stationary", True))
    store_every = int(max(1, getd(opts, "store_every", 1)))
    verbose = bool(getd(opts, "verbose", False))
    rng = np.random.default_rng(getd(opts, "seed", None))

    Nbar, Nhat = _init_states(N_obs)
    a_t = pi_tm1 - pi_expect
    lambda_prec0 = 0.0 if orth else 1.0 / pri["sigma_lambda"]**2

    n_store = int(np.ceil(n_keep / store_every))
    alpha_draws = np.zeros(n_store)
    kappa_draws = np.zeros(n_store)
    theta_draws = np.zeros(n_store)
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

    total_iter = n_burn + n_keep
    store_idx = 0

    for it in range(1, total_iter + 1):
        zeta = x_t - phi_1 * x_tm1
        y = pi_t - pi_expect
        y_adj = y - lambda_ez * zeta
        X = np.column_stack([a_t, x_t, -Nhat])
        beta = sample_beta_gaussian(
            y_adj,
            X,
            sigma2=sigma_eta2,
            prior_mean=np.array([pri["mu_alpha"], pri["mu_kappa"], pri["mu_theta"]], dtype=float),
            prior_var=np.array([pri["sigma_alpha"]**2, pri["sigma_kappa"]**2, pri["sigma_theta"]**2], dtype=float),
            rng=rng,
        )
        alpha = float(np.clip(beta[0], 0.0, 0.999))
        kappa = float(beta[1])
        theta = float(beta[2])

        if not orth:
            e_base = y - alpha * a_t - kappa * x_t + theta * Nhat
            post_var_lambda = 1.0 / (lambda_prec0 + float(np.sum(zeta**2)) / sigma_eta2)
            post_mean_lambda = post_var_lambda * (
                pri["mu_lambda"] * lambda_prec0 + float(np.dot(zeta, e_base)) / sigma_eta2
            )
            lambda_ez = float(post_mean_lambda + np.sqrt(post_var_lambda) * rng.standard_normal())
        else:
            lambda_ez = 0.0

        y_tilde_phi = y - alpha * a_t - kappa * x_t + theta * Nhat
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
        eta = y - alpha * a_t - kappa * x_t + theta * Nhat - lambda_ez * zeta
        sigma_zeta2 = sample_invgamma(pri["a_z"] + 0.5 * T, pri["b_z"] + 0.5 * float(np.sum(zeta**2)), rng)
        sigma_eta2 = sample_invgamma(pri["a_e"] + 0.5 * T, pri["b_e"] + 0.5 * float(np.sum(eta**2)), rng)

        if T >= 3:
            rho1, rho2 = _sample_ar2_coeffs(
                Nhat, sigma_u2, pri["mu_rho1"], pri["sigma_rho1"], pri["mu_rho2"], pri["sigma_rho2"], enforce_stationary, rng
            )
            resid_u = Nhat[2:] - rho1 * Nhat[1:-1] - rho2 * Nhat[:-2]
            sigma_u2 = sample_invgamma(pri["a_u"] + 0.5 * resid_u.size, pri["b_u"] + 0.5 * float(np.sum(resid_u**2)), rng)

        if T >= 2:
            dNbar = Nbar[1:] - Nbar[:-1]
            post_var_n = 1.0 / (1.0 / pri["sigma_n"]**2 + dNbar.size / sigma_eps2)
            post_mean_n = post_var_n * (pri["mu_n"] / pri["sigma_n"]**2 + float(np.sum(dNbar)) / sigma_eps2)
            n_drift = abs(float(post_mean_n + np.sqrt(post_var_n) * rng.standard_normal()))
            resid_eps = Nbar[1:] - n_drift - Nbar[:-1]
            sigma_eps2 = sample_invgamma(pri["a_eps"] + 0.5 * resid_eps.size, pri["b_eps"] + 0.5 * float(np.sum(resid_eps**2)), rng)

        obs_offset = lambda_ez * zeta
        Nhat = sample_ar2_states_ffbs(
            y_target=N_obs - Nbar,
            rho1=rho1,
            rho2=rho2,
            sigma_state2=sigma_u2,
            pi_t=pi_t,
            alpha=alpha,
            pi_tm1=pi_tm1,
            pi_expect=pi_expect,
            x_t=x_t,
            theta=theta,
            sigma_obs2=sigma_eta2,
            target_scale=target_scale,
            rng=rng,
            kappa=kappa,
            obs_offset=obs_offset,
        )
        Nbar = sample_rw_states_ffbs(
            y_target=N_obs - Nhat,
            n_drift=n_drift,
            sigma_state2=sigma_eps2,
            target_scale=rw_scale,
            rng=rng,
        )

        sigma_e = float(np.sqrt(lambda_ez**2 * sigma_zeta2 + sigma_eta2))
        rho_ez = 0.0 if orth else float((lambda_ez * np.sqrt(sigma_zeta2)) / max(sigma_e, 1e-12))

        if it > n_burn and (it - n_burn) % store_every == 0:
            alpha_draws[store_idx] = alpha
            kappa_draws[store_idx] = kappa
            theta_draws[store_idx] = theta
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
            store_idx += 1

        if verbose and it % 5000 == 0:
            print(f"Iter {it}/{total_iter}: alpha={alpha:.3f}, kappa={kappa:.3f}, theta={theta:.3f}")

    return {
        "alpha": _summary(alpha_draws),
        "kappa": _summary(kappa_draws),
        "theta": _summary(theta_draws),
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
        "state_draws": {"Nbar": Nbar_draws, "Nhat": Nhat_draws},
        "priors": priors or {},
        "opts": opts,
    }


def func_nkpc_hsa_decomp_tv_kappa_noerror(
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

    pri = _common_priors(priors or {}, steady=True)
    assert_all_pos(
        [
            pri["sigma_alpha"], pri["sigma_kappa0"], pri["sigma_delta"], pri["sigma_phi"], pri["sigma_lambda"],
            pri["sigma_rho1"], pri["sigma_rho2"], pri["sigma_n"], pri["a_e"], pri["b_e"], pri["a_u"], pri["b_u"],
            pri["a_eps"], pri["b_eps"], pri["a_z"], pri["b_z"],
        ],
        "Steady HSA prior scales must be positive.",
    )

    opts = opts or {}
    alpha = float(getd(opts, "alpha0", pri["mu_alpha"]))
    kappa0 = float(getd(opts, "kappa0", pri["mu_kappa0"]))
    delta = float(getd(opts, "delta0", pri["mu_delta"]))
    phi_1 = float(getd(opts, "phi10", pri["mu_phi"]))
    lambda_ez = 0.0 if orth else float(getd(opts, "lambda0", 0.0))
    rho1 = float(getd(opts, "rho10", 0.5))
    rho2 = float(getd(opts, "rho20", -0.5))
    n_drift = float(getd(opts, "n0", 0.01))
    sigma_eta2 = float(getd(opts, "sigma_e20", getd(opts, "sigma_v20", 1.0)))
    sigma_zeta2 = float(getd(opts, "sigma_zeta20", 1.0))
    sigma_u2 = float(getd(opts, "sigma_u20", getd(opts, "sigma_eps20", 0.5)))
    sigma_eps2 = float(getd(opts, "sigma_eps20", getd(opts, "sigma_eta20", 0.1)))
    target_scale = float(getd(opts, "target_scale", getd(opts, "r_target_scale", 0.1)))
    rw_scale = float(getd(opts, "rw_scale", getd(opts, "r_rw_scale", 0.1)))
    enforce_stationary = bool(getd(opts, "enforce_stationary", True))
    store_every = int(max(1, getd(opts, "store_every", 1)))
    verbose = bool(getd(opts, "verbose", False))
    rng = np.random.default_rng(getd(opts, "seed", None))

    Nbar, Nhat = _init_states(N_obs)
    a_t = pi_tm1 - pi_expect
    lambda_prec0 = 0.0 if orth else 1.0 / pri["sigma_lambda"]**2

    n_store = int(np.ceil(n_keep / store_every))
    alpha_draws = np.zeros(n_store)
    kappa0_draws = np.zeros(n_store)
    delta_draws = np.zeros(n_store)
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

    total_iter = n_burn + n_keep
    store_idx = 0

    for it in range(1, total_iter + 1):
        kappa_t = kappa0 + delta * Nbar
        zeta = x_t - phi_1 * x_tm1
        y = pi_t - pi_expect
        y_adj = y - lambda_ez * zeta
        X = np.column_stack([a_t, x_t, x_t * Nbar])
        beta = sample_beta_gaussian(
            y_adj,
            X,
            sigma2=sigma_eta2,
            prior_mean=np.array([pri["mu_alpha"], pri["mu_kappa0"], pri["mu_delta"]], dtype=float),
            prior_var=np.array([pri["sigma_alpha"]**2, pri["sigma_kappa0"]**2, pri["sigma_delta"]**2], dtype=float),
            rng=rng,
        )
        alpha = float(np.clip(beta[0], 0.0, 0.999))
        kappa0 = float(beta[1])
        delta = float(beta[2])
        kappa_t = kappa0 + delta * Nbar

        if not orth:
            e_base = y - alpha * a_t - kappa_t * x_t
            post_var_lambda = 1.0 / (lambda_prec0 + float(np.sum(zeta**2)) / sigma_eta2)
            post_mean_lambda = post_var_lambda * (
                pri["mu_lambda"] * lambda_prec0 + float(np.dot(zeta, e_base)) / sigma_eta2
            )
            lambda_ez = float(post_mean_lambda + np.sqrt(post_var_lambda) * rng.standard_normal())
        else:
            lambda_ez = 0.0

        y_tilde_phi = y - alpha * a_t - kappa_t * x_t
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
        eta = y - alpha * a_t - kappa_t * x_t - lambda_ez * zeta
        sigma_zeta2 = sample_invgamma(pri["a_z"] + 0.5 * T, pri["b_z"] + 0.5 * float(np.sum(zeta**2)), rng)
        sigma_eta2 = sample_invgamma(pri["a_e"] + 0.5 * T, pri["b_e"] + 0.5 * float(np.sum(eta**2)), rng)

        if T >= 3:
            rho1, rho2 = _sample_ar2_coeffs(
                Nhat, sigma_u2, pri["mu_rho1"], pri["sigma_rho1"], pri["mu_rho2"], pri["sigma_rho2"], enforce_stationary, rng
            )
            resid_u = Nhat[2:] - rho1 * Nhat[1:-1] - rho2 * Nhat[:-2]
            sigma_u2 = sample_invgamma(pri["a_u"] + 0.5 * resid_u.size, pri["b_u"] + 0.5 * float(np.sum(resid_u**2)), rng)

        if T >= 2:
            dNbar = Nbar[1:] - Nbar[:-1]
            post_var_n = 1.0 / (1.0 / pri["sigma_n"]**2 + dNbar.size / sigma_eps2)
            post_mean_n = post_var_n * (pri["mu_n"] / pri["sigma_n"]**2 + float(np.sum(dNbar)) / sigma_eps2)
            n_drift = abs(float(post_mean_n + np.sqrt(post_var_n) * rng.standard_normal()))
            resid_eps = Nbar[1:] - n_drift - Nbar[:-1]
            sigma_eps2 = sample_invgamma(pri["a_eps"] + 0.5 * resid_eps.size, pri["b_eps"] + 0.5 * float(np.sum(resid_eps**2)), rng)

        obs_offset = lambda_ez * zeta
        Nhat = sample_ar2_states_ffbs(
            y_target=N_obs - Nbar,
            rho1=rho1,
            rho2=rho2,
            sigma_state2=sigma_u2,
            pi_t=pi_t,
            alpha=alpha,
            pi_tm1=pi_tm1,
            pi_expect=pi_expect,
            x_t=x_t,
            theta=0.0,
            sigma_obs2=sigma_eta2,
            target_scale=target_scale,
            rng=rng,
            kappa_t=kappa_t,
            obs_offset=obs_offset,
        )
        Nbar = sample_rw_states_ffbs(
            y_target=N_obs - Nhat,
            n_drift=n_drift,
            sigma_state2=sigma_eps2,
            target_scale=rw_scale,
            rng=rng,
        )
        kappa_t = kappa0 + delta * Nbar

        sigma_e = float(np.sqrt(lambda_ez**2 * sigma_zeta2 + sigma_eta2))
        rho_ez = 0.0 if orth else float((lambda_ez * np.sqrt(sigma_zeta2)) / max(sigma_e, 1e-12))

        if it > n_burn and (it - n_burn) % store_every == 0:
            alpha_draws[store_idx] = alpha
            kappa0_draws[store_idx] = kappa0
            delta_draws[store_idx] = delta
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
            kappa_t_draws[store_idx] = kappa_t
            store_idx += 1

        if verbose and it % 5000 == 0:
            print(f"Iter {it}/{total_iter}: alpha={alpha:.3f}, kappa0={kappa0:.3f}, delta={delta:.3f}")

    return {
        "alpha": _summary(alpha_draws),
        "kappa_0": _summary(kappa0_draws),
        "delta": _summary(delta_draws),
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
        "state_draws": {"Nbar": Nbar_draws, "Nhat": Nhat_draws, "kappa_t": kappa_t_draws},
        "priors": priors or {},
        "opts": opts,
    }
