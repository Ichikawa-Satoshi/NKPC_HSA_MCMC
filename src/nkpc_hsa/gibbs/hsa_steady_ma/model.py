"""HSA steady with MA(q) inflation-equation errors.

Motivation is in ``nkpc_hsa.gibbs.common.joint_ffbs_ma``: the inflation series are
four-quarter changes sampled quarterly, so the inflation-equation residual is serially
correlated by construction while the baseline likelihood assumes it is i.i.d. The
baseline absorbs the overlap into the lagged-inflation coefficient ``alpha``; dropping
that term instead (``no_inertia``) leaves the equation badly misspecified. This variant
models the overlap where it belongs, in the error process:

    y_t   = alpha*a_t + (kappa_0 + delta*Nbar_t)*x_t + lambda_ez*zeta_t + eta_t
    eta_t = eps_t + psi_1 eps_{t-1} + ... + psi_q eps_{t-q},   eps ~ N(0, sigma^2) iid

Everything else -- priors, KAPPA_SCALE conventions, the firm-count state block, the
AR(2)/random-walk transitions, missing-N handling -- is imported unchanged from
``hsa_steady`` so the two are comparable coefficient by coefficient.

Gibbs partition (see the module docstring of ``joint_ffbs_ma`` for why):

  1. ``(alpha, kappa_0, delta) | psi, sigma^2, states``  -- GLS with the banded MA
     covariance, i.e. ``eps`` integrated out analytically.
  2. ``lambda_ez``, ``phi_1``                            -- as in hsa_steady, GLS-weighted.
  3. ``psi | beta, sigma^2, states``                     -- random-walk Metropolis on the
     exact Gaussian likelihood, restricted to invertible MA roots.
  4. ``sigma^2 | psi, beta, states``                     -- inverse gamma on the whitened
     residual.
  5. ``rho, sigma_u^2, n, sigma_eps^2, sigma_N^2``       -- as in hsa_steady.
  6. ``(Nbar, Nhat, eps) | rest``                        -- joint FFBS on the augmented
     state, which carries the MA lags.
"""
from __future__ import annotations

from typing import Any, Optional

import numpy as np
from numpy.linalg import cholesky, inv, solve

from nkpc_hsa.gibbs.common.competition import finite_N_residuals
from nkpc_hsa.gibbs.common.joint_ffbs_ma import (
    ma_covariance,
    sample_joint_competition_states_ffbs_ma,
)
from nkpc_hsa.gibbs.hsa_steady.model import (
    KAPPA_SCALE,
    _ar2_stats_summary,
    _common_priors,
    _getd,
    _init_states,
    _is_stationary_ar2,
    _sample_ar2_coeffs,
    _sample_invgamma,
    _summary,
)

DEFAULT_MA_ORDER = 3


def _invertible(psi: np.ndarray) -> bool:
    """Invertibility of Psi(L) = 1 + psi_1 L + ... + psi_q L^q.

    The MA polynomial's roots must lie outside the unit circle. Substituting w = 1/z
    turns that into "the roots of w^q + psi_1 w^{q-1} + ... + psi_q lie inside", which is
    what ``np.roots`` returns for the coefficients in the order they are written.
    """
    psi = np.asarray(psi, dtype=float).reshape(-1)
    if psi.size == 0:
        return True
    return bool(np.all(np.abs(np.roots(np.concatenate([[1.0], psi]))) < 1.0))


def _gls_beta(y, X, Omega_chol, prior_mean, prior_var, rng):
    """Conjugate Gaussian draw for beta under Cov(eta) = Omega."""
    # Whiten: L z = r  =>  z = L^{-1} r
    Xw = solve(Omega_chol, X)
    yw = solve(Omega_chol, y)
    V0_inv = np.diag(1.0 / np.asarray(prior_var, dtype=float))
    prec = Xw.T @ Xw + V0_inv
    cov = inv(prec)
    mean = cov @ (Xw.T @ yw + V0_inv @ np.asarray(prior_mean, dtype=float))
    cov = (cov + cov.T) / 2.0
    vals, vecs = np.linalg.eigh(cov)
    cov = (vecs * np.maximum(vals, 1e-12)) @ vecs.T
    return rng.multivariate_normal(mean, cov)


def _log_gauss_ma(resid: np.ndarray, psi: np.ndarray, sigma2: float) -> float:
    T = resid.size
    Om = ma_covariance(psi, sigma2, T)
    L = cholesky(Om)
    z = solve(L, resid)
    return float(-np.sum(np.log(np.diag(L))) - 0.5 * float(z @ z))


def func_nkpc_hsa_steady_ma(
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

    pri = _common_priors(priors or {})
    opts = opts or {}
    rng = np.random.default_rng(_getd(opts, "seed", None))
    q = int(_getd(opts, "ma_order", DEFAULT_MA_ORDER))
    psi_sd = float(_getd(opts, "psi_prior_sd", 1.0))
    psi_step = float(_getd(opts, "psi_step", 0.08))

    alpha = float(pri["mu_alpha"])
    kappa0 = float(pri["mu_kappa0"])
    delta = float(pri["mu_delta"])
    phi_1 = float(pri["mu_phi"])
    lambda_ez = 0.0
    rho1, rho2 = float(pri["mu_rho1"]), float(pri["mu_rho2"])
    n_drift = 0.0
    sigma2 = 1.0
    sigma_zeta2 = 1.0
    sigma_u2, sigma_eps2, sigma_N2 = 1.0, 1.0, 1.0
    psi = np.zeros(q)

    enforce_stationary = bool(_getd(opts, "enforce_stationary", True))
    ar2_max_tries = int(max(1, _getd(opts, "ar2_max_tries", 2000)))
    store_every = int(max(1, _getd(opts, "store_every", 1)))
    ar2_stats: dict[str, int] = {}

    Nbar, Nhat = _init_states(N_obs)
    states = np.zeros((T, 3))
    states[:, 0], states[:, 2] = Nhat, Nbar
    if T > 1:
        states[1:, 1] = Nhat[:-1]
    a_t = pi_tm1 - pi_expect
    y = pi_t - pi_expect
    lambda_prec0 = 0.0 if orth else 1.0 / pri["sigma_lambda"] ** 2

    m0 = np.array([pri["m0_Nhat"], pri["m0_Nhat_lag"], pri["m0_Nbar"]])
    P0 = np.diag([pri["P0_Nhat"], pri["P0_Nhat_lag"], pri["P0_Nbar"]])

    n_store = int(n_keep // store_every)
    store = {k: np.zeros(n_store) for k in
             ["alpha", "kappa_0", "delta", "phi_1", "lambda_ez", "rho_1", "rho_2", "n",
              "sigma_e", "sigma_zeta", "sigma_u", "sigma_eps", "sigma_N"]}
    psi_draws = np.zeros((n_store, q))
    Nbar_draws = np.zeros((n_store, T))
    Nhat_draws = np.zeros((n_store, T))
    kappa_t_draws = np.zeros((n_store, T))
    psi_accept = 0

    total_iter = n_burn + n_keep
    idx = 0
    for it in range(1, total_iter + 1):
        zeta = x_t - phi_1 * x_tm1
        y_adj = y - lambda_ez * zeta
        Om = ma_covariance(psi, sigma2, T)
        L = cholesky(Om)

        # --- 1. (alpha, kappa_0, delta) by GLS ---
        X = np.column_stack([a_t, x_t / KAPPA_SCALE, (x_t * Nbar) / KAPPA_SCALE])
        beta = _gls_beta(
            y_adj, X, L,
            [pri["mu_alpha"], pri["mu_kappa0"], pri["mu_delta"]],
            [pri["sigma_alpha"] ** 2, pri["sigma_kappa0"] ** 2, pri["sigma_delta"] ** 2],
            rng,
        )
        alpha, kappa0, delta = (float(v) for v in beta)
        kappa_t_eff = (kappa0 + delta * Nbar) / KAPPA_SCALE

        # --- 2. lambda_ez and phi_1, both GLS-weighted ---
        if not orth:
            e_base = y - alpha * a_t - kappa_t_eff * x_t
            zw = solve(L, zeta)
            ew = solve(L, e_base)
            v = 1.0 / (lambda_prec0 + float(zw @ zw))
            mn = v * (pri["mu_lambda"] * lambda_prec0 + float(zw @ ew))
            lambda_ez = float(mn + np.sqrt(v) * rng.standard_normal())

        y_tilde_phi = y - alpha * a_t - kappa_t_eff * x_t
        xw = solve(L, lambda_ez * x_tm1)
        rw = solve(L, y_tilde_phi - lambda_ez * x_t)
        prec = 1.0 / pri["sigma_phi"] ** 2 + float(np.sum(x_tm1 ** 2)) / sigma_zeta2 + float(xw @ xw)
        num = (pri["mu_phi"] / pri["sigma_phi"] ** 2
               + float(np.dot(x_tm1, x_t)) / sigma_zeta2
               - float(xw @ rw))
        phi_1 = float(num / prec + rng.standard_normal() / np.sqrt(prec))

        zeta = x_t - phi_1 * x_tm1
        eta = y - alpha * a_t - kappa_t_eff * x_t - lambda_ez * zeta

        # --- 3. psi by random-walk Metropolis, restricted to invertible roots ---
        prop = psi + psi_step * rng.standard_normal(q)
        if _invertible(prop):
            cur_lp = _log_gauss_ma(eta, psi, sigma2) - 0.5 * float(psi @ psi) / psi_sd ** 2
            new_lp = _log_gauss_ma(eta, prop, sigma2) - 0.5 * float(prop @ prop) / psi_sd ** 2
            if np.log(rng.random()) < new_lp - cur_lp:
                psi = prop
                psi_accept += 1

        # --- 4. sigma^2 from the whitened residual ---
        Lu = cholesky(ma_covariance(psi, 1.0, T))
        z = solve(Lu, eta)
        sigma2 = _sample_invgamma(pri["a_e"] + 0.5 * T, pri["b_e"] + 0.5 * float(z @ z), rng)
        sigma_zeta2 = _sample_invgamma(
            pri["a_z"] + 0.5 * T, pri["b_z"] + 0.5 * float(np.sum(zeta ** 2)), rng
        )

        # --- 5. firm-count state parameters, exactly as in hsa_steady ---
        rho1, rho2 = _sample_ar2_coeffs(
            Nhat=Nhat, sigma_state2=sigma_u2,
            mu_rho1=pri["mu_rho1"], sigma_rho1=pri["sigma_rho1"],
            mu_rho2=pri["mu_rho2"], sigma_rho2=pri["sigma_rho2"],
            enforce_stationary=enforce_stationary, rng=rng,
            max_tries=ar2_max_tries, current=(rho1, rho2), stats=ar2_stats,
            initial_lag=float(states[0, 1]),
        )
        resid_u = states[1:, 0] - rho1 * states[:-1, 0] - rho2 * states[:-1, 1]
        sigma_u2 = _sample_invgamma(
            pri["a_u"] + 0.5 * resid_u.size, pri["b_u"] + 0.5 * float(np.sum(resid_u ** 2)), rng
        )
        dNbar = Nbar[1:] - Nbar[:-1]
        pv = 1.0 / (1.0 / pri["sigma_n"] ** 2 + dNbar.size / sigma_eps2)
        pm = pv * (pri["mu_n"] / pri["sigma_n"] ** 2 + float(np.sum(dNbar)) / sigma_eps2)
        n_drift = float(pm + np.sqrt(pv) * rng.standard_normal())
        resid_eps = Nbar[1:] - n_drift - Nbar[:-1]
        sigma_eps2 = _sample_invgamma(
            pri["a_eps"] + 0.5 * resid_eps.size,
            pri["b_eps"] + 0.5 * float(np.sum(resid_eps ** 2)), rng,
        )
        resid_N = finite_N_residuals(N_obs, Nhat, Nbar)
        sigma_N2 = _sample_invgamma(
            pri["a_N"] + 0.5 * resid_N.size, pri["b_N"] + 0.5 * float(np.sum(resid_N ** 2)), rng
        )

        # --- 6. joint state draw on the MA-augmented state ---
        y_tilde_state = y - alpha * a_t - (kappa0 / KAPPA_SCALE) * x_t - lambda_ez * zeta
        Nbar, Nhat, aug = sample_joint_competition_states_ffbs_ma(
            N_obs=N_obs, y_tilde=y_tilde_state, h_nbar=(delta / KAPPA_SCALE) * x_t,
            psi=psi, n_drift=n_drift, rho1=rho1, rho2=rho2, sigma2=sigma2,
            sigma_u2=sigma_u2, sigma_eps2=sigma_eps2, sigma_N2=sigma_N2,
            m0=m0, P0=P0, rng=rng,
        )
        states = aug[:, :3]

        if it > n_burn and (it - n_burn) % store_every == 0:
            kappa_t = kappa0 + delta * Nbar
            store["alpha"][idx] = alpha
            store["kappa_0"][idx] = kappa0 / KAPPA_SCALE
            store["delta"][idx] = delta / KAPPA_SCALE
            store["phi_1"][idx] = phi_1
            store["lambda_ez"][idx] = lambda_ez
            store["rho_1"][idx] = rho1
            store["rho_2"][idx] = rho2
            store["n"][idx] = n_drift
            store["sigma_e"][idx] = np.sqrt(sigma2)
            store["sigma_zeta"][idx] = np.sqrt(sigma_zeta2)
            store["sigma_u"][idx] = np.sqrt(sigma_u2)
            store["sigma_eps"][idx] = np.sqrt(sigma_eps2)
            store["sigma_N"][idx] = np.sqrt(sigma_N2)
            psi_draws[idx] = psi
            Nbar_draws[idx] = Nbar
            Nhat_draws[idx] = Nhat
            kappa_t_draws[idx] = kappa_t / KAPPA_SCALE
            idx += 1

    out = {k: _summary(v) for k, v in store.items()}
    for j in range(q):
        out[f"psi_{j + 1}"] = _summary(psi_draws[:, j])
    out["state_draws"] = {"Nbar": Nbar_draws, "Nhat": Nhat_draws, "kappa_t": kappa_t_draws}
    out["priors"] = priors or {}
    out["opts"] = opts
    out["model"] = {
        "N_measurement_error": True,
        "state_sampler": "joint_ffbs_ma",
        "ma_order": q,
        "psi_acceptance_rate": psi_accept / total_iter,
        "inflation_equation": (
            "y_t = alpha*a_t + kappa_t*x_t + lambda_ez*zeta_t + eta_t, "
            "eta_t = eps_t + sum_j psi_j eps_{t-j}"
        ),
        "state_vector": "[Nhat_t, Nhat_{t-1}, Nbar_t, eps_t, ..., eps_{t-q}]'",
        "kappa_scale": KAPPA_SCALE,
        "stored_units": "physical",
        "ar2_stationarity": {"enforce_stationary": enforce_stationary,
                             "max_tries": ar2_max_tries, **_ar2_stats_summary(ar2_stats)},
    }
    return out


__all__ = ["func_nkpc_hsa_steady_ma", "DEFAULT_MA_ORDER"]
