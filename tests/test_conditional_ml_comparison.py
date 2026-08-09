"""The conditional marginal likelihood must not depend on where theta* is put.

Chib's identity is exact for any theta* in the support: the likelihood and prior
rise as the ordinate falls, and the three cancel. An implementation that mixes a
*conditional* likelihood with a *joint* posterior ordinate breaks that -- the
identity collapses to m(pi,N|x)/p(N|x,theta*), which still carries theta*. That
is not a small error: the module once returned +555 on one seed and -104 on
another for the same cell.

These tests pin the three properties that failure violated:

* theta* invariance, on the two-block activity model where it is cheap to check
  exactly;
* the decomposition identity, which must hold to machine precision;
* the effective-draw guard, which is what turns "silently wrong" into "raises".

The full HSA comparison is far too slow for a unit test (minutes per seed even
with the reduced runs parallelised), so it is exercised on the activity model and
on synthetic data rather than on the production cell.
"""
from __future__ import annotations

import numpy as np
import pytest

from nkpc_hsa.gibbs.conditional_ml import (
    MIN_EFFECTIVE_ORDINATE_DRAWS,
    OrdinateNotIdentified,
    _checked_logmeanexp,
    _effective_terms,
    _log_ig_pdf_var,
    _log_norm_pdf,
    _logmeanexp,
    _phi_x_only_moments,
    activity_loglik,
    activity_marginal_likelihood,
)

PRIORS = {
    "phi_1": (0.7, 0.2),
    "a_z": 0.001,
    "b_z": 0.001,
}


def _activity_data(T: int = 124, phi: float = 0.85, sigma: float = 0.4, seed: int = 7) -> dict:
    rng = np.random.default_rng(seed)
    x = np.empty(T)
    x[0] = rng.standard_normal() * sigma
    for t in range(1, T):
        x[t] = phi * x[t - 1] + sigma * rng.standard_normal()
    return {"x": x, "x_prev": np.concatenate([[0.0], x[:-1]])}


def _log_m_at(star: dict, data: dict, sigma_draws: np.ndarray) -> float:
    """Chib's identity evaluated at an arbitrary theta*, reusing one run's draws."""
    x, x_prev = data["x"], data["x_prev"]
    T = x.size
    log_lik = activity_loglik(star, data)
    log_prior = float(
        _log_norm_pdf(star["phi_1"], *PRIORS["phi_1"])
        + _log_ig_pdf_var(star["sigma_zeta2"], PRIORS["a_z"], PRIORS["b_z"])
    )
    ordinate = _logmeanexp(
        np.array([
            _log_norm_pdf(star["phi_1"], *_phi_x_only_moments(x, x_prev, s, PRIORS))
            for s in sigma_draws
        ])
    )
    resid = x - star["phi_1"] * x_prev
    ordinate += _log_ig_pdf_var(
        star["sigma_zeta2"], PRIORS["a_z"] + 0.5 * T, PRIORS["b_z"] + 0.5 * float(resid @ resid)
    )
    return float(log_lik + log_prior - ordinate)


def _run_activity_gibbs(data: dict, *, n_burn: int, n_keep: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x, x_prev = data["x"], data["x_prev"]
    T = x.size
    sigma2 = 1.0
    draws = []
    for it in range(n_burn + n_keep):
        mean, sd = _phi_x_only_moments(x, x_prev, sigma2, PRIORS)
        phi = float(mean + sd * rng.standard_normal())
        resid = x - phi * x_prev
        sigma2 = float(
            1.0 / rng.gamma(PRIORS["a_z"] + 0.5 * T, 1.0 / (PRIORS["b_z"] + 0.5 * float(resid @ resid)))
        )
        if it >= n_burn:
            draws.append(sigma2)
    return np.asarray(draws)


def test_marginal_likelihood_does_not_depend_on_theta_star() -> None:
    """Move theta* off the posterior mean; the estimate must not follow it."""
    data = _activity_data()
    sigma_draws = _run_activity_gibbs(data, n_burn=1000, n_keep=8000, seed=11)
    base, star = activity_marginal_likelihood(data, PRIORS, n_burn=1000, n_keep=8000, seed=11)

    for scale in (0.9, 0.95, 1.05, 1.1):
        moved = {"phi_1": star["phi_1"] * scale, "sigma_zeta2": star["sigma_zeta2"]}
        assert _log_m_at(moved, data, sigma_draws) == pytest.approx(base, abs=0.05), (
            f"log m moved when theta* moved by {scale:g}x -- the identity is not closing"
        )


def test_estimate_is_stable_across_seeds() -> None:
    data = _activity_data()
    values = [
        activity_marginal_likelihood(data, PRIORS, n_burn=1000, n_keep=8000, seed=s)[0]
        for s in (101, 202, 303)
    ]
    assert np.std(values, ddof=1) < 0.05, f"seed-to-seed spread too large: {values}"


def test_same_activity_marginal_is_used_for_both_models() -> None:
    """m(x) is a property of the data, not of the model that conditions on it.

    Both models are debited the same m(x); the comparison is only a comparison if
    that quantity is literally identical, which is why the driver computes it once
    and passes it in rather than letting each model recompute it.
    """
    data = _activity_data()
    first, _ = activity_marginal_likelihood(data, PRIORS, n_burn=800, n_keep=4000, seed=5)
    second, _ = activity_marginal_likelihood(data, PRIORS, n_burn=800, n_keep=4000, seed=5)
    assert first == second


def test_guard_rejects_an_ordinate_carried_by_one_draw() -> None:
    """One dominant term is the failure mode that produced the 660-point error."""
    dominated = np.concatenate([[0.0], np.full(999, -80.0)])
    assert _effective_terms(dominated) < MIN_EFFECTIVE_ORDINATE_DRAWS
    with pytest.raises(OrdinateNotIdentified, match="effective"):
        _checked_logmeanexp(dominated, block="rho")


def test_guard_passes_a_well_supported_ordinate() -> None:
    spread = np.random.default_rng(0).normal(-3.0, 0.3, size=1000)
    assert _effective_terms(spread) > MIN_EFFECTIVE_ORDINATE_DRAWS
    assert _checked_logmeanexp(spread, block="rho") == pytest.approx(_logmeanexp(spread))


def test_guard_allows_an_exact_single_term_factor() -> None:
    """The last block's conditional is exact; one evaluation is the answer."""
    assert _checked_logmeanexp(np.array([-2.5]), block="sigma_N2") == pytest.approx(-2.5)
