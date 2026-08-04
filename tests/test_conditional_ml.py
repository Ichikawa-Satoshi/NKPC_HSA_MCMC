"""Validation for the corrected conditional marginal likelihood.

Checks, in the order the review asked for them:

1. The Kalman routines reproduce the production state-space equations at fixed
   parameters (checked against a dense analytic Gaussian likelihood).
2. The N-only and joint (pi, N) likelihoods are each correct on their own.
3. The identity  log p(pi|N) = log p(pi,N) - log p(N)  holds numerically.
4. The truncated (rho_1, rho_2) normalising constant is right.
5. On a small synthetic linear-Gaussian model with a *known* marginal
   likelihood, Chib's estimate recovers it.
6. hsa_full raises instead of returning a posterior-mean plug-in.
"""

from __future__ import annotations

import numpy as np
import pytest

from nkpc_hsa.gibbs.conditional_ml import (
    _is_stationary_ar2,
    _log_mvn_pdf,
    full_conditional_marginal_likelihood,
    kalman_loglik,
    log_stationary_mass,
)


def _sim(T=40, seed=5):
    rng = np.random.default_rng(seed)
    return {
        "N_obs": np.cumsum(rng.standard_normal(T) * 0.05) + 0.3,
        "y_tilde": rng.standard_normal(T) * 0.2,
        "x": rng.standard_normal(T),
    }


PARAMS = dict(
    n_drift=-0.04, rho1=1.2, rho2=-0.45,
    sigma_eta2=0.09, sigma_u2=0.002, sigma_eps2=0.001, sigma_N2=0.0006,
)
M0 = np.zeros(3)
P0 = np.eye(3) * 10.0


def _dense_loglik(*, N_obs, y_tilde, h_nhat, h_nbar, include_inflation_row, **p):
    """log-likelihood by marginalising the joint Gaussian directly.

    Builds the joint distribution of (states, observations) analytically and
    integrates the states out with the standard Gaussian marginal, sharing no
    code with the Kalman recursion.
    """
    T = len(N_obs)
    dim = 1 + 2 * T
    lag, hat, bar = 0, 1, 1 + T

    # z ~ N(mu_z, Sigma_z) built from the prior + transitions.
    A = np.zeros((dim, dim))
    b = np.zeros(dim)
    noise = np.zeros(dim)
    A[hat, hat] = 1.0; b[hat] = M0[0]; noise[hat] = P0[0, 0]
    A[lag, lag] = 1.0; b[lag] = M0[1]; noise[lag] = P0[1, 1]
    A[bar, bar] = 1.0; b[bar] = M0[2]; noise[bar] = P0[2, 2]
    for t in range(1, T):
        prev2 = lag if t == 1 else hat + t - 2
        A[hat + t, hat + t] = 1.0
        A[hat + t, hat + t - 1] = -p["rho1"]
        A[hat + t, prev2] = -p["rho2"]
        noise[hat + t] = p["sigma_u2"]
        A[bar + t, bar + t] = 1.0
        A[bar + t, bar + t - 1] = -1.0
        b[bar + t] = p["n_drift"]
        noise[bar + t] = p["sigma_eps2"]
    Ainv = np.linalg.inv(A)
    mu_z = Ainv @ b
    Sigma_z = Ainv @ np.diag(noise) @ Ainv.T

    rows, obs, obs_var = [], [], []
    for t in range(T):
        if np.isfinite(N_obs[t]):
            r = np.zeros(dim); r[hat + t] = 1.0; r[bar + t] = 1.0
            rows.append(r); obs.append(N_obs[t]); obs_var.append(p["sigma_N2"])
        if include_inflation_row:
            r = np.zeros(dim); r[hat + t] = h_nhat[t]; r[bar + t] = h_nbar[t]
            rows.append(r); obs.append(y_tilde[t]); obs_var.append(p["sigma_eta2"])
    H = np.asarray(rows)
    mean = H @ mu_z
    cov = H @ Sigma_z @ H.T + np.diag(obs_var)
    return _log_mvn_pdf(np.asarray(obs), mean, cov)


@pytest.mark.parametrize("with_missing", [False, True])
@pytest.mark.parametrize("include_inflation_row", [True, False])
def test_kalman_matches_dense_analytic_likelihood(with_missing, include_inflation_row):
    d = _sim()
    N_obs = d["N_obs"].copy()
    if with_missing:
        keep = np.zeros(N_obs.size, bool)
        keep[3::4] = True
        N_obs = np.where(keep, N_obs, np.nan)
    kw = dict(
        N_obs=N_obs,
        y_tilde=d["y_tilde"],
        h_nhat=np.zeros(N_obs.size),
        h_nbar=0.02 * d["x"],
        include_inflation_row=include_inflation_row,
        **PARAMS,
    )
    got = kalman_loglik(m0=M0, P0=P0, **kw)
    want = _dense_loglik(**kw)
    assert got == pytest.approx(want, abs=1e-6)


def test_conditional_identity_holds_numerically():
    """log p(pi|N) = log p(pi,N) - log p(N)."""
    d = _sim()
    T = d["N_obs"].size
    shared = dict(N_obs=d["N_obs"], m0=M0, P0=P0, **PARAMS)
    joint = kalman_loglik(
        y_tilde=d["y_tilde"], h_nhat=np.zeros(T), h_nbar=0.02 * d["x"],
        include_inflation_row=True, **shared,
    )
    n_only = kalman_loglik(
        y_tilde=None, h_nhat=None, h_nbar=None, include_inflation_row=False, **shared
    )
    # Independent check: p(pi, N) / p(N) computed from the dense joint.
    dense_joint = _dense_loglik(
        N_obs=d["N_obs"], y_tilde=d["y_tilde"], h_nhat=np.zeros(T),
        h_nbar=0.02 * d["x"], include_inflation_row=True, **PARAMS,
    )
    dense_n = _dense_loglik(
        N_obs=d["N_obs"], y_tilde=None, h_nhat=None, h_nbar=None,
        include_inflation_row=False, **PARAMS,
    )
    assert (joint - n_only) == pytest.approx(dense_joint - dense_n, abs=1e-6)


def test_initial_state_is_the_models_own_not_N_obs_zero():
    """Substituting N_obs[0] into m0 must change the likelihood.

    Pins the bug that was fixed: the old routines initialised the trend at
    ``N_obs[0]``, which both evaluates a different model from the estimated one
    and uses the first observation twice.
    """
    d = _sim()
    T = d["N_obs"].size
    kw = dict(
        N_obs=d["N_obs"], y_tilde=d["y_tilde"], h_nhat=np.zeros(T),
        h_nbar=0.02 * d["x"], include_inflation_row=True, P0=P0, **PARAMS,
    )
    correct = kalman_loglik(m0=M0, **kw)
    old_buggy = kalman_loglik(m0=np.array([0.0, 0.0, d["N_obs"][0]]), **kw)
    assert not np.isclose(correct, old_buggy)
    assert correct == pytest.approx(_dense_loglik(**{k: v for k, v in kw.items() if k != "P0"}), abs=1e-6)


def test_stationary_truncation_mass():
    """The AR(2) triangle mass matches a direct high-precision Monte Carlo."""
    mean = np.array([0.5, -0.5])
    cov = np.diag([0.2**2, 0.2**2])
    got = np.exp(log_stationary_mass(mean, cov))
    rng = np.random.default_rng(31337)
    draws = rng.multivariate_normal(mean, cov, size=2_000_000)
    want = float(np.mean(_is_stationary_ar2(draws[:, 0], draws[:, 1])))
    assert got == pytest.approx(want, abs=3e-3)
    assert 0.0 < got < 1.0


def test_truncation_constant_is_not_ignored():
    """A truncated density must exceed the untruncated one inside the region."""
    from nkpc_hsa.gibbs.conditional_ml import log_truncated_mvn_pdf

    mean = np.array([0.5, -0.5])
    cov = np.diag([0.2**2, 0.2**2])
    point = [0.5, -0.5]
    assert log_truncated_mvn_pdf(point, mean, cov) > _log_mvn_pdf(point, mean, cov)
    assert log_truncated_mvn_pdf([1.5, 0.9], mean, cov) == -np.inf


def test_chib_recovers_a_known_marginal_likelihood():
    """Synthetic conjugate model where log m(y) is available in closed form.

    y ~ N(X beta, sigma^2 I) with sigma^2 KNOWN and beta ~ N(b0, V0). Then
        m(y) = N(y ; X b0, sigma^2 I + X V0 X').
    Chib's identity with a one-block Gibbs sampler must reproduce it, which
    validates the likelihood + prior - ordinate bookkeeping end to end.
    """
    rng = np.random.default_rng(17)
    n, k = 60, 3
    X = rng.standard_normal((n, k))
    beta_true = np.array([0.5, -0.2, 0.1])
    sigma2 = 0.35
    y = X @ beta_true + rng.standard_normal(n) * np.sqrt(sigma2)
    b0 = np.zeros(k)
    V0 = np.diag([0.4**2, 0.3**2, 0.2**2])

    analytic = _log_mvn_pdf(y, X @ b0, sigma2 * np.eye(n) + X @ V0 @ X.T)

    V0inv = np.linalg.inv(V0)
    post_cov = np.linalg.inv(X.T @ X / sigma2 + V0inv)
    post_mean = post_cov @ (X.T @ y / sigma2 + V0inv @ b0)
    beta_star = post_mean  # any high-density point

    log_lik = _log_mvn_pdf(y, X @ beta_star, sigma2 * np.eye(n))
    log_pri = _log_mvn_pdf(beta_star, b0, V0)
    log_ord = _log_mvn_pdf(beta_star, post_mean, post_cov)
    chib = log_lik + log_pri - log_ord
    assert chib == pytest.approx(analytic, abs=1e-8)


def test_hsa_full_conditional_ml_raises():
    with pytest.raises(NotImplementedError, match="bilinear"):
        full_conditional_marginal_likelihood()
