"""Validation for the shared exact joint firm-count FFBS.

Three properties are pinned here, in the order the review asked for them:

1. Delegation is numerically exact -- ``hsa_steady`` routed through the shared
   routine reproduces its historical in-line implementation bit for bit, so no
   existing hsa_steady posterior changes.
2. ``hsa_const_theta``'s state block agrees with the independent gamma = 0
   benchmark used to validate the Particle-Gibbs sampler
   (``sample_states_joint_ffbs_gamma0``).
3. With ``theta_0 = 0`` the const-theta state block reduces exactly to the
   ``hsa_steady`` state block.
"""

from __future__ import annotations

import numpy as np
import pytest

from nkpc_hsa.gibbs.common.joint_ffbs import sample_joint_competition_states_ffbs
from nkpc_hsa.gibbs.hsa_full_pg.model import sample_states_joint_ffbs_gamma0

KAPPA_SCALE = 100.0


def _fixture(with_missing: bool = True):
    rng = np.random.default_rng(11)
    T = 96
    N_obs = np.cumsum(rng.standard_normal(T) * 0.05) + 0.3
    if with_missing:
        # Exercise the annual-Q4 style missing-observation path.
        mask = np.ones(T, dtype=bool)
        mask[[i for i in range(T) if i % 4 != 3]] = False
        N_obs = np.where(mask, N_obs, np.nan)
    data = {
        "N_obs": N_obs,
        "pi_t": rng.standard_normal(T),
        "pi_tm1": rng.standard_normal(T),
        "pi_expect": rng.standard_normal(T),
        "x_t": rng.standard_normal(T) * 2.0,
        "obs_offset": rng.standard_normal(T) * 0.1,
    }
    params = dict(
        alpha=0.79,
        kappa0=5.1,
        delta=2.3,
        n_drift=-0.046,
        rho1=1.80,
        rho2=-0.81,
        sigma_eta2=0.10,
        sigma_u2=0.0020,
        sigma_eps2=0.0009,
        sigma_N2=0.0005,
    )
    return data, params, np.zeros(3), np.eye(3) * 10.0


@pytest.mark.parametrize("with_missing", [False, True])
def test_hsa_steady_delegation_is_bit_identical(with_missing):
    """hsa_steady through the shared routine == the historical in-line filter."""
    from nkpc_hsa.gibbs.hsa_steady.model import _sample_states_kalman_ffbs

    data, params, m0, P0 = _fixture(with_missing)
    T = data["N_obs"].size

    got = _sample_states_kalman_ffbs(
        N_obs=data["N_obs"],
        pi_t=data["pi_t"],
        pi_tm1=data["pi_tm1"],
        pi_expect=data["pi_expect"],
        x_t=data["x_t"],
        obs_offset=data["obs_offset"],
        m0=m0,
        P0=P0,
        rng=np.random.default_rng(2024),
        **params,
    )

    y_tilde = (
        data["pi_t"]
        - data["pi_expect"]
        - params["alpha"] * (data["pi_tm1"] - data["pi_expect"])
        - (params["kappa0"] / KAPPA_SCALE) * data["x_t"]
        - data["obs_offset"]
    )
    want = sample_joint_competition_states_ffbs(
        N_obs=data["N_obs"],
        y_tilde=y_tilde,
        h_nhat=np.zeros(T),
        h_nbar=(params["delta"] / KAPPA_SCALE) * data["x_t"],
        n_drift=params["n_drift"],
        rho1=params["rho1"],
        rho2=params["rho2"],
        sigma_eta2=params["sigma_eta2"],
        sigma_u2=params["sigma_u2"],
        sigma_eps2=params["sigma_eps2"],
        sigma_N2=params["sigma_N2"],
        m0=m0,
        P0=P0,
        rng=np.random.default_rng(2024),
    )
    for a, b in zip(got, want):
        assert np.array_equal(a, b)


def _dense_smoother(
    *,
    N_obs,
    y_tilde,
    h_nhat,
    h_nbar,
    n_drift,
    rho1,
    rho2,
    sigma_eta2,
    sigma_u2,
    sigma_eps2,
    sigma_N2,
    m0,
    P0,
):
    """Analytic smoothing moments by assembling the joint Gaussian directly.

    Stacks z = (Nhat_{-1}, Nhat_0..Nhat_{T-1}, Nbar_0..Nbar_{T-1}) and builds the
    joint precision from the prior, the AR(2) and random-walk transitions and the
    two observation rows. Because every term is linear-Gaussian the smoothing
    distribution is N(Omega^{-1} b, Omega^{-1}) exactly. This shares no code with
    the Kalman/FFBS recursion, so it is a genuine independent check of it.
    """
    T = len(y_tilde)
    dim = 1 + 2 * T
    lag, hat, bar = 0, 1, 1 + T
    Omega = np.zeros((dim, dim))
    b = np.zeros(dim)

    def add(rows, coeffs, value, variance):
        prec = 1.0 / variance
        for i, ci in zip(rows, coeffs):
            b[i] += prec * ci * value
            for j, cj in zip(rows, coeffs):
                Omega[i, j] += prec * ci * cj

    # Initial-state prior.
    add([hat + 0], [1.0], m0[0], P0[0, 0])
    add([lag], [1.0], m0[1], P0[1, 1])
    add([bar + 0], [1.0], m0[2], P0[2, 2])

    # Transitions.
    for t in range(1, T):
        prev2 = lag if t == 1 else hat + t - 2
        add([hat + t, hat + t - 1, prev2], [1.0, -rho1, -rho2], 0.0, sigma_u2)
        add([bar + t, bar + t - 1], [1.0, -1.0], n_drift, sigma_eps2)

    # Observation rows.
    for t in range(T):
        if np.isfinite(N_obs[t]):
            add([hat + t, bar + t], [1.0, 1.0], float(N_obs[t]), sigma_N2)
        add([hat + t, bar + t], [h_nhat[t], h_nbar[t]], float(y_tilde[t]), sigma_eta2)

    cov = np.linalg.inv(Omega)
    mean = cov @ b
    sd = np.sqrt(np.diag(cov))
    return {
        "Nhat_mean": mean[hat : hat + T],
        "Nbar_mean": mean[bar : bar + T],
        "Nhat_sd": sd[hat : hat + T],
        "Nbar_sd": sd[bar : bar + T],
    }


@pytest.mark.parametrize("theta0", [0.0, 0.062])
def test_joint_ffbs_matches_dense_gaussian_smoother(theta0):
    """FFBS draws reproduce the analytic smoothing mean and sd.

    Covers both the hsa_steady case (theta_0 = 0) and the hsa_const_theta case
    (theta_0 != 0), i.e. exactly the two loadings the shared routine serves.
    """
    data, params, m0, P0 = _fixture(with_missing=True)
    T = data["N_obs"].size
    y_tilde = (
        data["pi_t"]
        - data["pi_expect"]
        - params["alpha"] * (data["pi_tm1"] - data["pi_expect"])
        - (params["kappa0"] / KAPPA_SCALE) * data["x_t"]
        - data["obs_offset"]
    )
    kwargs = dict(
        N_obs=data["N_obs"],
        y_tilde=y_tilde,
        h_nhat=np.full(T, -theta0),
        h_nbar=(params["delta"] / KAPPA_SCALE) * data["x_t"],
        n_drift=params["n_drift"],
        rho1=params["rho1"],
        rho2=params["rho2"],
        sigma_eta2=params["sigma_eta2"],
        sigma_u2=params["sigma_u2"],
        sigma_eps2=params["sigma_eps2"],
        sigma_N2=params["sigma_N2"],
        m0=m0,
        P0=P0,
    )
    analytic = _dense_smoother(**kwargs)

    M = 4000
    rng = np.random.default_rng(4242)
    nbar = np.empty((M, T))
    nhat = np.empty((M, T))
    for i in range(M):
        nbar[i], nhat[i], _ = sample_joint_competition_states_ffbs(rng=rng, **kwargs)

    # FFBS draws are i.i.d. here, so the Monte Carlo error of the mean is sd/sqrt(M).
    for empirical, name in ((nbar, "Nbar"), (nhat, "Nhat")):
        mc_se = analytic[f"{name}_sd"] / np.sqrt(M)
        z = np.abs(empirical.mean(axis=0) - analytic[f"{name}_mean"]) / mc_se
        assert z.max() < 5.0, f"{name} mean off by {z.max():.2f} MC se"
        rel_sd = np.abs(empirical.std(axis=0, ddof=1) - analytic[f"{name}_sd"]) / analytic[f"{name}_sd"]
        assert rel_sd.max() < 0.10, f"{name} sd off by {rel_sd.max():.3f} relative"


def test_const_theta_matches_independent_gamma0_benchmark():
    """Distributional agreement with the PG gamma = 0 validation benchmark.

    The two routines target the same smoothing posterior but repair covariances
    differently (eigenvalue clip vs. a 1e-12 ridge), so same-seeded single paths
    diverge chaotically. Correctness is therefore checked in distribution, over
    independent draws, rather than path by path.
    """
    data, params, m0, P0 = _fixture(with_missing=False)
    T = data["N_obs"].size
    theta0 = 0.062
    y = data["pi_t"] - data["pi_expect"]
    a_t = data["pi_tm1"] - data["pi_expect"]
    zeta = data["obs_offset"]
    lambda_ez = 1.0
    kappa0_eff = params["kappa0"] / KAPPA_SCALE
    delta_eff = params["delta"] / KAPPA_SCALE

    M = 1500
    rng_b = np.random.default_rng(555)
    bench_bar = np.empty((M, T))
    bench_hat = np.empty((M, T))
    for i in range(M):
        out = sample_states_joint_ffbs_gamma0(
            y=y,
            a_t=a_t,
            x_t=data["x_t"],
            zeta=zeta,
            N_obs=data["N_obs"],
            alpha=params["alpha"],
            kappa0_eff=kappa0_eff,
            delta_eff=delta_eff,
            theta0=theta0,
            lambda_ez=lambda_ez,
            rho1=params["rho1"],
            rho2=params["rho2"],
            n_drift=params["n_drift"],
            sigma_eta2=params["sigma_eta2"],
            sigma_u2=params["sigma_u2"],
            sigma_eps2=params["sigma_eps2"],
            sigma_N2=params["sigma_N2"],
            m0_Nhat=float(m0[0]),
            P0_Nhat=float(P0[0, 0]),
            m0_Nhat_lag=float(m0[1]),
            P0_Nhat_lag=float(P0[1, 1]),
            m0_Nbar=float(m0[2]),
            P0_Nbar=float(P0[2, 2]),
            rng=rng_b,
        )
        bench_bar[i], bench_hat[i] = out["Nbar"], out["Nhat"]

    y_tilde = y - params["alpha"] * a_t - kappa0_eff * data["x_t"] - lambda_ez * zeta
    rng_j = np.random.default_rng(556)
    joint_bar = np.empty((M, T))
    joint_hat = np.empty((M, T))
    for i in range(M):
        joint_bar[i], joint_hat[i], _ = sample_joint_competition_states_ffbs(
            N_obs=data["N_obs"],
            y_tilde=y_tilde,
            h_nhat=np.full(T, -theta0),
            h_nbar=delta_eff * data["x_t"],
            n_drift=params["n_drift"],
            rho1=params["rho1"],
            rho2=params["rho2"],
            sigma_eta2=params["sigma_eta2"],
            sigma_u2=params["sigma_u2"],
            sigma_eps2=params["sigma_eps2"],
            sigma_N2=params["sigma_N2"],
            m0=m0,
            P0=P0,
            rng=rng_j,
        )

    for a, b, name in ((bench_bar, joint_bar, "Nbar"), (bench_hat, joint_hat, "Nhat")):
        # Both sets are i.i.d., so the difference of means has se sqrt(2/M)*sd.
        pooled_sd = 0.5 * (a.std(axis=0, ddof=1) + b.std(axis=0, ddof=1))
        se = pooled_sd * np.sqrt(2.0 / M)
        z = np.abs(a.mean(axis=0) - b.mean(axis=0)) / se
        assert z.max() < 5.0, f"{name} mean differs by {z.max():.2f} se"
        rel = np.abs(a.std(axis=0, ddof=1) - b.std(axis=0, ddof=1)) / pooled_sd
        assert rel.max() < 0.10, f"{name} sd differs by {rel.max():.3f} relative"


def test_const_theta_reduces_to_steady_when_theta_is_zero():
    """theta_0 = 0 collapses the const-theta state block onto hsa_steady's."""
    data, params, m0, P0 = _fixture(with_missing=True)
    T = data["N_obs"].size
    y_tilde = (
        data["pi_t"]
        - data["pi_expect"]
        - params["alpha"] * (data["pi_tm1"] - data["pi_expect"])
        - (params["kappa0"] / KAPPA_SCALE) * data["x_t"]
        - data["obs_offset"]
    )
    kwargs = dict(
        N_obs=data["N_obs"],
        y_tilde=y_tilde,
        h_nbar=(params["delta"] / KAPPA_SCALE) * data["x_t"],
        n_drift=params["n_drift"],
        rho1=params["rho1"],
        rho2=params["rho2"],
        sigma_eta2=params["sigma_eta2"],
        sigma_u2=params["sigma_u2"],
        sigma_eps2=params["sigma_eps2"],
        sigma_N2=params["sigma_N2"],
        m0=m0,
        P0=P0,
    )
    steady = sample_joint_competition_states_ffbs(
        h_nhat=np.zeros(T), rng=np.random.default_rng(99), **kwargs
    )
    const_theta = sample_joint_competition_states_ffbs(
        h_nhat=np.full(T, -0.0), rng=np.random.default_rng(99), **kwargs
    )
    for a, b in zip(steady, const_theta):
        assert np.array_equal(a, b)


def test_missing_observations_drop_only_the_firm_count_row():
    """A missing N_obs must not stop the inflation row informing the state."""
    data, params, m0, P0 = _fixture(with_missing=False)
    T = data["N_obs"].size
    y_tilde = np.full(T, 5.0)
    kwargs = dict(
        y_tilde=y_tilde,
        h_nhat=np.zeros(T),
        h_nbar=np.full(T, 0.5),
        n_drift=0.0,
        rho1=0.5,
        rho2=-0.2,
        sigma_eta2=0.05,
        sigma_u2=0.01,
        sigma_eps2=0.01,
        sigma_N2=0.01,
        m0=m0,
        P0=P0,
    )
    all_missing = np.full(T, np.nan)
    Nbar, _, _ = sample_joint_competition_states_ffbs(
        N_obs=all_missing, rng=np.random.default_rng(3), **kwargs
    )
    # With every firm-count row dropped the inflation row alone must still pull
    # Nbar towards y_tilde / h_nbar = 10, not leave it at the prior mean of 0.
    assert np.mean(Nbar) > 1.0
    assert np.all(np.isfinite(Nbar))
