from __future__ import annotations

from typing import Any, Optional

import numpy as np
try:
    from .gibbs_utils import assert_all_pos, getd, mvnrnd, sample_invgamma
except ImportError:
    from nkpc_hsa.gibbs.gibbs_utils import assert_all_pos, getd, mvnrnd, sample_invgamma

def _summary(draws: np.ndarray) -> dict[str, Any]:
    """Return posterior draws and basic posterior summaries."""
    arr = np.asarray(draws, dtype=float)
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

    # Convert all input series to one-dimensional floating-point arrays.
    pi_t = np.asarray(pi_data, dtype=float).reshape(-1)
    pi_tm1 = np.asarray(pi_prev_data, dtype=float).reshape(-1)
    pi_expect = np.asarray(Epi_data, dtype=float).reshape(-1)
    x_t = np.asarray(x_data, dtype=float).reshape(-1)
    x_tm1 = np.asarray(x_prev_data, dtype=float).reshape(-1)

    # All series must be aligned observation by observation.
    T = pi_t.size
    if not (pi_tm1.size == pi_expect.size == x_t.size == x_tm1.size == T):
        raise ValueError("All input series must have the same length.")

    # Prior hyperparameters.
    priors = priors or {}
    mu_alpha = getd(priors, "mu_alpha", 0.5)
    sigma_alpha = getd(priors, "sigma_alpha", 0.2)
    mu_kappa = getd(priors, "mu_kappa", 0.1)
    sigma_kappa = getd(priors, "sigma_kappa", 0.2)
    mu_phi = getd(priors, "mu_phi_1", 0.7)
    sigma_phi = getd(priors, "sigma_phi_1", 0.2)
    mu_lambda = getd(priors, "mu_lambda", 0.0)
    sigma_lambda = getd(priors, "sigma_lambda", 0.5)

    # Inverse-gamma priors for the innovation variances.
    a_e = getd(priors, "a_e", getd(priors, "a_v", 2.0))
    b_e = getd(priors, "b_e", getd(priors, "b_v", 2.0))
    a_z = getd(priors, "a_z", 0.001)
    b_z = getd(priors, "b_z", 0.001)

    # Validate prior scales and inverse-gamma parameters.
    assert_all_pos(
        [sigma_alpha, sigma_kappa, sigma_phi, sigma_lambda, a_e, b_e, a_z, b_z],
        "CES prior scales must be positive.",
    )

    # Initial values and sampler options.
    opts = opts or {}
    alpha = float(getd(opts, "alpha0", mu_alpha))
    kappa = float(getd(opts, "kappa0", mu_kappa))
    phi_1 = float(getd(opts, "phi10", mu_phi))
    lambda_ez = 0.0 if orth else float(getd(opts, "lambda0", 0.0))
    sigma_eta2 = float(getd(opts, "sigma_e20", getd(opts, "sigma_v20", 1.0)))
    sigma_zeta2 = float(getd(opts, "sigma_zeta20", 1.0))
    seed = getd(opts, "seed", None)
    store_every = int(max(1, getd(opts, "store_every", 1)))
    verbose = bool(getd(opts, "verbose", False))
    rng = np.random.default_rng(seed)

    # Allocate arrays for retained posterior draws.
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

    # Precompute fixed regressors and prior precision matrices.
    a_t = pi_tm1 - pi_expect
    prior_mean = np.array([mu_alpha, mu_kappa], dtype=float)
    prior_prec = np.diag([1.0 / sigma_alpha**2, 1.0 / sigma_kappa**2])
    phi_prec0 = 1.0 / sigma_phi**2
    lambda_prec0 = 0.0 if orth else 1.0 / sigma_lambda**2

    for it in range(1, total_iter + 1):
        # Inflation equation after subtracting expected inflation.
        y = pi_t - pi_expect

        # Current innovation in the x process.
        zeta = x_t - phi_1 * x_tm1

        # Sample alpha and kappa jointly from their Gaussian conditional posterior.
        X = np.column_stack([a_t, x_t])
        y_adj = y - lambda_ez * zeta
        post_cov = np.linalg.inv(X.T @ X / sigma_eta2 + prior_prec)
        post_mean = post_cov @ (X.T @ y_adj / sigma_eta2 + prior_prec @ prior_mean)
        beta = mvnrnd(post_mean, post_cov, rng)
        alpha = float(beta[0])
        kappa = float(beta[1])

        # Sample the cross-equation shock loading unless orthogonality is imposed.
        if not orth:
            e_base = y - alpha * a_t - kappa * x_t
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

        # Sample the AR(1) coefficient of the x process.
        prec_phi = (
            phi_prec0
            + float(np.sum(x_tm1**2)) / sigma_zeta2
            + (lambda_ez**2) * float(np.sum(x_tm1**2)) / sigma_eta2
        )
        mean_num_phi = (
            mu_phi * phi_prec0
            + float(np.dot(x_tm1, x_t)) / sigma_zeta2
            - lambda_ez
            * float(np.dot(x_tm1, y - alpha * a_t - kappa * x_t - lambda_ez * x_t))
            / sigma_eta2
        )
        phi_1 = float(mean_num_phi / prec_phi + rng.standard_normal() / np.sqrt(prec_phi))

        # Recompute innovations using the updated phi_1.
        zeta = x_t - phi_1 * x_tm1
        eta = y - alpha * a_t - kappa * x_t - lambda_ez * zeta

        # Sample innovation variances from inverse-gamma conditional posteriors.
        sigma_zeta2 = sample_invgamma(
            a_z + 0.5 * T,
            b_z + 0.5 * float(np.sum(zeta**2)),
            rng,
        )
        sigma_eta2 = sample_invgamma(
            a_e + 0.5 * T,
            b_e + 0.5 * float(np.sum(eta**2)),
            rng,
        )

        # Implied variance of the reduced-form inflation error and its correlation
        # with the x-process innovation.
        sigma_e2 = lambda_ez**2 * sigma_zeta2 + sigma_eta2
        rho_corr = 0.0 if orth else float(
            (lambda_ez * np.sqrt(sigma_zeta2)) / max(np.sqrt(sigma_e2), 1e-12)
        )

        # Store retained draws after burn-in according to `store_every`.
        if it > n_burn and (it - n_burn) % store_every == 0:
            alpha_draws[store_idx] = alpha
            kappa_draws[store_idx] = kappa
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
        "deprecated": (
            "nkpc_hsa.gibbs.gibbs_ces is legacy. "
            "Use nkpc_hsa.gibbs.ces.model or nkpc_hsa.inference.wrappers instead."
        ),
    }
