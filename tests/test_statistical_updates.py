from __future__ import annotations

import numpy as np

from nkpc_hsa.gibbs.hsa_dynamic.model import _sample_sigma_restricted
from nkpc_hsa.gibbs.hsa_full.model import _common_priors as full_priors
from nkpc_hsa.gibbs.hsa_full.model import _sample_ar2_states_ffbs
from nkpc_hsa.gibbs.hsa_steady.model import _common_priors as steady_priors


def test_restricted_covariance_is_sampled_on_restricted_space() -> None:
    rng = np.random.default_rng(123)
    residuals = rng.normal(size=(4, 12))
    for structure in ("e_zeta_only", "diagonal"):
        sigma = _sample_sigma_restricted(
            *residuals,
            nu0=8.0,
            S0=np.eye(4),
            structure=structure,
            rng=rng,
        )
        assert np.all(np.linalg.eigvalsh(sigma) > 0.0)
        if structure == "e_zeta_only":
            allowed = {(0, 0), (0, 1), (1, 0), (1, 1), (2, 2), (3, 3)}
        else:
            allowed = {(i, i) for i in range(4)}
        for i in range(4):
            for j in range(4):
                if (i, j) not in allowed:
                    assert sigma[i, j] == 0.0


def test_ar2_ffbs_assimilates_period_zero_inflation_observation() -> None:
    states = _sample_ar2_states_ffbs(
        y_target=np.zeros(3),
        rho1=0.0,
        rho2=0.0,
        sigma_state2=1.0,
        pi_t=np.array([-50.0, 0.0, 0.0]),
        alpha=0.0,
        pi_tm1=np.zeros(3),
        pi_expect=np.zeros(3),
        x_t=np.zeros(3),
        theta=1.0,
        sigma_obs2=1e-6,
        sigma_target2=1e8,
        rng=np.random.default_rng(44),
        kappa=0.0,
    )
    assert states[0, 0] > 49.0


def test_kappa_zero_specific_prior_takes_precedence() -> None:
    priors = {
        "mu_kappa": 99.0,
        "sigma_kappa": 88.0,
        "mu_kappa_0": 1.25,
        "sigma_kappa_0": 2.5,
    }
    for resolver in (steady_priors, full_priors):
        resolved = resolver(priors)
        assert resolved["mu_kappa0"] == 1.25
        assert resolved["sigma_kappa0"] == 2.5
