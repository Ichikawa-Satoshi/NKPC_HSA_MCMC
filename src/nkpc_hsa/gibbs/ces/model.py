from __future__ import annotations

from typing import Any, Optional

import numpy as np

from nkpc_hsa.gibbs.common.constraints import constraint_stats_summary, draw_with_constraints

KAPPA_SCALE = 100


def _getd(d: Optional[dict[str, Any]], key: str, default: Any) -> Any:
    if isinstance(d, dict) and key in d and d[key] is not None:
        return d[key]
    return default


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


def _mvnrnd(mean: np.ndarray, cov: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    return rng.multivariate_normal(np.asarray(mean, dtype=float), _force_pd(cov))


def _sample_invgamma(a_post: float, b_post: float, rng: np.random.Generator) -> float:
    if a_post <= 0.0 or b_post <= 0.0:
        raise ValueError("Inverse-gamma posterior parameters must be positive.")
    return 1.0 / rng.gamma(shape=a_post, scale=1.0 / b_post)


def _summary(draws: np.ndarray) -> dict[str, Any]:
    arr = np.asarray(draws, dtype=float)
    if arr.shape[0] == 0:
        raise ValueError("No posterior draws were stored. Check n_keep and store_every.")
    qs = np.quantile(arr, [0.025, 0.05, 0.25, 0.5, 0.75, 0.95, 0.975], axis=0)

    return {
        "draws": arr,
        "mean": np.mean(arr, axis=0),
        "std": np.std(arr, axis=0, ddof=1)
        if arr.shape[0] > 1
        else np.zeros_like(np.mean(arr, axis=0)),
        "quantiles": qs,
    }


def func_nkpc_ces(
    pi_data,
    pi_prev_data,
    Epi_data,
    x_data,
    x_prev_data,
    n_burn: int,
    n_keep: int,
    priors: Optional[dict[str, Any]] = None,
    opts: Optional[dict[str, Any]] = None,
    *,
    orth: bool = False,
) -> dict[str, Any]:
    """
    Estimate a CES-style NKPC model using a Gibbs sampler.

    If `orth=True`, the cross-equation shock loading `lambda_ez` is fixed at zero.
    """

    pi_t = np.asarray(pi_data, dtype=float).reshape(-1)
    pi_tm1 = np.asarray(pi_prev_data, dtype=float).reshape(-1)
    pi_expect = np.asarray(Epi_data, dtype=float).reshape(-1)
    x_t = np.asarray(x_data, dtype=float).reshape(-1)
    x_tm1 = np.asarray(x_prev_data, dtype=float).reshape(-1)

    T = pi_t.size
    if not (pi_tm1.size == pi_expect.size == x_t.size == x_tm1.size == T):
        raise ValueError("All input series must have the same length.")

    priors = priors or {}
    mu_alpha = _getd(priors, "mu_alpha", 0.5)
    sigma_alpha = _getd(priors, "sigma_alpha", 0.2)
    mu_kappa = _getd(priors, "mu_kappa", 10.0)
    sigma_kappa = _getd(priors, "sigma_kappa", 20.0)
    mu_phi = _getd(priors, "mu_phi_1", 0.7)
    sigma_phi = _getd(priors, "sigma_phi_1", 0.2)
    mu_lambda = _getd(priors, "mu_lambda", 0.0)
    sigma_lambda = _getd(priors, "sigma_lambda", 0.5)

    a_e = _getd(priors, "a_e", _getd(priors, "a_v", 2.0))
    b_e = _getd(priors, "b_e", _getd(priors, "b_v", 2.0))
    a_z = _getd(priors, "a_z", 0.001)
    b_z = _getd(priors, "b_z", 0.001)

    _assert_all_pos(
        [sigma_alpha, sigma_kappa, sigma_phi, sigma_lambda, a_e, b_e, a_z, b_z],
        "CES prior scales must be positive.",
    )

    opts = opts or {}
    alpha = float(_getd(opts, "alpha0", mu_alpha))
    kappa = float(_getd(opts, "kappa0", mu_kappa))
    phi_1 = float(_getd(opts, "phi10", mu_phi))
    lambda_ez = 0.0 if orth else float(_getd(opts, "lambda0", 0.0))
    sigma_eta2 = float(_getd(opts, "sigma_e20", _getd(opts, "sigma_v20", 1.0)))
    sigma_zeta2 = float(_getd(opts, "sigma_zeta20", 1.0))
    seed = _getd(opts, "seed", None)
    store_every = int(max(1, _getd(opts, "store_every", 1)))
    verbose = bool(_getd(opts, "verbose", False))
    coefficient_constraints = _getd(opts, "coefficient_constraints", {})
    constraint_stats: dict[str, int] = {}
    rng = np.random.default_rng(seed)

    n_store = int(n_keep // store_every)
    alpha_draws = np.zeros(n_store)
    kappa_draws = np.zeros(n_store)
    phi_draws = np.zeros(n_store)
    lambda_draws = np.zeros(n_store)
    sigma_e2_draws = np.zeros(n_store)
    sigma_zeta2_draws = np.zeros(n_store)
    rho_draws = np.zeros(n_store)

    total_iter = n_burn + n_keep
    store_idx = 0

    a_t = pi_tm1 - pi_expect
    prior_mean = np.array([mu_alpha, mu_kappa], dtype=float)
    prior_prec = np.diag([1.0 / sigma_alpha**2, 1.0 / sigma_kappa**2])
    phi_prec0 = 1.0 / sigma_phi**2
    lambda_prec0 = 0.0 if orth else 1.0 / sigma_lambda**2

    for it in range(1, total_iter + 1):
        y = pi_t - pi_expect
        zeta = x_t - phi_1 * x_tm1

        X = np.column_stack([a_t, x_t / KAPPA_SCALE])
        y_adj = y - lambda_ez * zeta
        post_cov = np.linalg.inv(X.T @ X / sigma_eta2 + prior_prec)
        post_mean = post_cov @ (X.T @ y_adj / sigma_eta2 + prior_prec @ prior_mean)
        beta = draw_with_constraints(
            lambda: _mvnrnd(post_mean, post_cov, rng),
            ("alpha", "kappa"),
            coefficient_constraints,
            stats=constraint_stats,
        )
        alpha = float(beta[0])
        kappa = float(beta[1])
        kappa_eff = kappa / KAPPA_SCALE

        if not orth:
            e_base = y - alpha * a_t - kappa_eff * x_t
            post_var_lambda = 1.0 / (
                lambda_prec0 + float(np.sum(zeta**2)) / sigma_eta2
            )
            post_mean_lambda = post_var_lambda * (
                mu_lambda * lambda_prec0 + float(np.dot(zeta, e_base)) / sigma_eta2
            )
            lambda_ez = float(
                post_mean_lambda + np.sqrt(post_var_lambda) * rng.standard_normal()
            )
        else:
            lambda_ez = 0.0

        prec_phi = (
            phi_prec0
            + float(np.sum(x_tm1**2)) / sigma_zeta2
            + (lambda_ez**2) * float(np.sum(x_tm1**2)) / sigma_eta2
        )
        mean_num_phi = (
            mu_phi * phi_prec0
            + float(np.dot(x_tm1, x_t)) / sigma_zeta2
            - lambda_ez
            * float(np.dot(x_tm1, y - alpha * a_t - kappa_eff * x_t - lambda_ez * x_t))
            / sigma_eta2
        )
        phi_1 = float(
            mean_num_phi / prec_phi + rng.standard_normal() / np.sqrt(prec_phi)
        )

        zeta = x_t - phi_1 * x_tm1
        eta = y - alpha * a_t - kappa_eff * x_t - lambda_ez * zeta

        sigma_zeta2 = _sample_invgamma(
            a_z + 0.5 * T,
            b_z + 0.5 * float(np.sum(zeta**2)),
            rng,
        )
        sigma_eta2 = _sample_invgamma(
            a_e + 0.5 * T,
            b_e + 0.5 * float(np.sum(eta**2)),
            rng,
        )

        sigma_e2 = lambda_ez**2 * sigma_zeta2 + sigma_eta2
        rho_corr = 0.0 if orth else float(
            (lambda_ez * np.sqrt(sigma_zeta2)) / max(np.sqrt(sigma_e2), 1e-12)
        )

        if it > n_burn and (it - n_burn) % store_every == 0:
            alpha_draws[store_idx] = alpha
            kappa_draws[store_idx] = kappa_eff
            phi_draws[store_idx] = phi_1
            lambda_draws[store_idx] = lambda_ez
            sigma_e2_draws[store_idx] = sigma_e2
            sigma_zeta2_draws[store_idx] = sigma_zeta2
            rho_draws[store_idx] = rho_corr
            store_idx += 1

        if verbose and it % 5000 == 0:
            print(f"Iter {it}/{total_iter}: alpha={alpha:.3f}, kappa={kappa:.3f}")

    return {
        "alpha": _summary(alpha_draws),
        "kappa": _summary(kappa_draws),
        "phi_1": _summary(phi_draws),
        "lambda_ez": _summary(lambda_draws),
        "sigma_e2": _summary(sigma_e2_draws),
        "sigma_zeta2": _summary(sigma_zeta2_draws),
        "rho": _summary(rho_draws),
        "priors": priors,
        "opts": opts,
        "model": {
            "kappa_scale": KAPPA_SCALE,
            "kappa_internal": "stored kappa * KAPPA_SCALE",
            "stored_units": "physical",
            "coefficient_constraints": coefficient_constraints,
            "coefficient_constraint_stats": constraint_stats_summary(constraint_stats),
        },
    }


__all__ = ["func_nkpc_ces"]
