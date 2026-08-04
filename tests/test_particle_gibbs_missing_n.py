"""Particle Gibbs must handle the annual-Q4 missing-N pattern correctly.

Particle Gibbs became the production ``hsa_full`` state sampler for both
observation designs, but until now it had only ever been run on PCHIP data
where every quarter carries a firm-count observation. These tests pin the
missing-N behaviour before the annual-Q4 production runs.

The strong test is a one-step invariance check against the exact joint FFBS: at
``gamma = 0`` the model is linear-Gaussian, so the exact conditional posterior
is available in closed form. Each Particle-Gibbs step is started from an
*independent* exact draw, so the outputs are i.i.d. and must be distributed as
the exact posterior. This isolates correctness from the sampler's mixing.
"""
from __future__ import annotations

import numpy as np
import pytest

from nkpc_hsa.gibbs.hsa_full_pg.model import (
    _obs_loglik,
    sample_states_joint_ffbs_gamma0,
    sample_states_particle_gibbs,
)

GEOM = dict(
    m0_Nhat=0.0, P0_Nhat=10.0,
    m0_Nhat_lag=0.0, P0_Nhat_lag=10.0,
    m0_Nbar=0.0, P0_Nbar=10.0,
)
PARAMS = dict(
    alpha=0.5, kappa0_eff=0.06, delta_eff=0.02, theta0=0.05,
    lambda_ez=0.0, rho1=0.6, rho2=-0.2, n_drift=-0.05,
    sigma_eta2=0.10, sigma_u2=0.02, sigma_eps2=0.01, sigma_N2=0.01,
)


def _annual_q4_data(T: int = 124, seed: int = 7):
    """Synthetic series with the annual-Q4 firm-count observation pattern."""
    rng = np.random.default_rng(seed)
    x = rng.normal(0.0, 1.0, T)
    y = rng.normal(0.0, 0.5, T)
    a_t = rng.normal(0.0, 0.5, T)
    zeta = rng.normal(0.0, 0.5, T)
    N_full = np.cumsum(rng.normal(-0.02, 0.05, T))
    N_obs = np.full(T, np.nan)
    q4 = np.arange(3, T, 4)          # every 4th quarter, as in the annual-Q4 design
    N_obs[q4] = N_full[q4]
    return x, y, a_t, zeta, N_obs, q4


def test_missing_quarters_drop_only_the_firm_count_term():
    """A missing N_obs must remove the firm-count likelihood, not the inflation one."""
    Nhat = np.array([0.3, -0.2])
    Nbar = np.array([1.0, 1.1])
    common = dict(
        y_t=0.4, a_t=0.1, x_t=1.2, zeta_t=0.0, alpha=0.5,
        kappa0_eff=0.06, delta_eff=0.02, theta0=0.05, gamma=0.0,
        lambda_ez=0.0, sigma_eta2=0.10, sigma_N2=0.01,
    )
    observed = _obs_loglik(Nhat, Nbar, N_obs_t=1.25, **common)
    missing = _obs_loglik(Nhat, Nbar, N_obs_t=np.nan, **common)

    inflation_only = -0.5 * (
        common["y_t"]
        - (common["alpha"] * common["a_t"]
           + common["kappa0_eff"] * common["x_t"]
           + common["delta_eff"] * common["x_t"] * Nbar
           - common["theta0"] * Nhat)
    ) ** 2 / common["sigma_eta2"]

    assert np.allclose(missing, inflation_only)
    assert np.all(np.isfinite(missing))
    # The observed case must be strictly more informative (adds a penalty term).
    assert np.all(observed < missing)


def test_all_quarters_missing_still_returns_a_finite_path():
    """With no firm-count data at all the transition + inflation row must still work."""
    T = 40
    x, y, a_t, zeta, _, _ = _annual_q4_data(T=T)
    N_obs = np.full(T, np.nan)
    rng = np.random.default_rng(0)
    ref_Nhat = np.zeros(T)
    ref_Nbar = np.linspace(0.0, -1.0, T)
    out = sample_states_particle_gibbs(
        y=y, a_t=a_t, x_t=x, zeta=zeta, N_obs=N_obs,
        gamma=0.0, Nbar_ref=ref_Nbar, Nhat_ref=ref_Nhat, Nhat_ref_lag=0.0,
        n_particles=64, rng=rng, **PARAMS, **GEOM,
    )
    assert np.all(np.isfinite(out["Nhat"]))
    assert np.all(np.isfinite(out["Nbar"]))
    assert out["ess_min"] > 0.0


@pytest.mark.parametrize("n_particles", [256])
def test_one_step_invariance_against_exact_ffbs_with_annual_q4_gaps(n_particles):
    """gamma = 0 + annual-Q4 gaps: PG output must match the exact posterior.

    Each PG step starts from an independent exact FFBS draw, so the M outputs are
    i.i.d. draws from the PG kernel applied to the exact posterior. If the kernel
    is invariant, their distribution equals the exact posterior's.

    The tolerance is **calibrated from the data**, not fixed: comparing two
    independent halves of the exact sample gives the null distribution of the
    same statistics at this M and T. A fixed threshold would either be vacuous or
    fail on Monte Carlo noise, since the statistics are a max over T = 124
    strongly correlated time points.
    """
    T = 124
    x, y, a_t, zeta, N_obs, q4 = _annual_q4_data(T=T)
    assert np.isfinite(N_obs).sum() == len(q4) == 31

    M = 400
    rng = np.random.default_rng(20260804)

    def exact_draw():
        return sample_states_joint_ffbs_gamma0(
            y=y, a_t=a_t, x_t=x, zeta=zeta, N_obs=N_obs, rng=rng, **PARAMS, **GEOM
        )

    # 2M exact draws: the first M are the reference sample, the second M give the
    # null distribution of the comparison statistics.
    exact = {"Nbar": np.empty((2 * M, T)), "Nhat": np.empty((2 * M, T))}
    for m in range(2 * M):
        r = exact_draw()
        exact["Nbar"][m], exact["Nhat"][m] = r["Nbar"], r["Nhat"]

    pg = {"Nbar": np.empty((M, T)), "Nhat": np.empty((M, T))}
    for m in range(M):
        ref = exact_draw()
        out = sample_states_particle_gibbs(
            y=y, a_t=a_t, x_t=x, zeta=zeta, N_obs=N_obs,
            gamma=0.0, Nbar_ref=ref["Nbar"], Nhat_ref=ref["Nhat"], Nhat_ref_lag=0.0,
            n_particles=n_particles, rng=rng, **PARAMS, **GEOM,
        )
        pg["Nbar"][m], pg["Nhat"][m] = out["Nbar"], out["Nhat"]

    def compare(a, b):
        se = np.sqrt(a.var(0) / a.shape[0] + b.var(0) / b.shape[0])
        return {
            "max_z": float((np.abs(a.mean(0) - b.mean(0)) / (se + 1e-12)).max()),
            "sd_spread": float(np.abs(b.std(0) / (a.std(0) + 1e-12) - 1.0).max()),
        }

    for name in ("Nbar", "Nhat"):
        ref_half, null_half = exact[name][:M], exact[name][M:]
        null = compare(ref_half, null_half)          # exact vs exact
        test = compare(ref_half, pg[name])           # exact vs Particle Gibbs
        # PG must not look worse than two independent exact samples do, up to a
        # 50% slack on the null statistic.
        assert test["max_z"] <= max(1.5 * null["max_z"], 4.0), (
            f"{name}: PG max|z| {test['max_z']:.2f} vs null {null['max_z']:.2f}"
        )
        assert test["sd_spread"] <= max(1.5 * null["sd_spread"], 0.20), (
            f"{name}: PG sd spread {test['sd_spread']:.3f} vs null {null['sd_spread']:.3f}"
        )


def test_unobserved_quarters_are_inferred_not_interpolated():
    """Q1-Q3 states must carry more posterior spread than the pinned Q4 anchors."""
    T = 124
    x, y, a_t, zeta, N_obs, q4 = _annual_q4_data(T=T)
    rng = np.random.default_rng(11)
    draws = np.empty((200, T))
    for m in range(200):
        r = sample_states_joint_ffbs_gamma0(
            y=y, a_t=a_t, x_t=x, zeta=zeta, N_obs=N_obs, rng=rng, **PARAMS, **GEOM
        )
        draws[m] = r["Nbar"] + r["Nhat"]
    sd = draws.std(0)
    off_q4 = np.setdiff1d(np.arange(T), q4)
    # The total state is pinned at the anchors and free in between.
    assert sd[q4].mean() < sd[off_q4].mean()
