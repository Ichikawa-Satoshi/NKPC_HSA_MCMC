"""The per-period predictive densities must factorize the model's joint density.

``pointwise_log_likelihood`` exists so LOO and WAIC can be formed from saved
runs. That is only meaningful if the per-period terms sum to the log density the
model assigns to the whole observed sample, so the tests below build the stacked
Gaussian implied by the state-space form and compare against it directly rather
than against another implementation of the same recursion.
"""
from __future__ import annotations

import numpy as np
import pytest

from nkpc_hsa.inference.pointwise_loglik import (
    KAPPA_SCALE,
    UnsupportedModel,
    pointwise_log_likelihood,
    waic_from_pointwise,
)


def _data(T: int, *, missing: tuple[int, ...] = ()) -> dict:
    rng = np.random.default_rng(11)
    x = rng.normal(size=T)
    N_obs = rng.normal(size=T) * 0.2
    for index in missing:
        N_obs[index] = np.nan
    return {
        "pi": rng.normal(size=T) + 2.0,
        "pi_prev": rng.normal(size=T) + 2.0,
        "pi_expect": rng.normal(size=T) + 2.0,
        "x": x,
        "x_prev": np.concatenate([[0.0], x[:-1]]),
        "N_obs": N_obs,
    }


DRAW = {
    "alpha": 0.7,
    "kappa_0": 0.15,
    "delta": 0.03,
    "theta": 0.25,
    "n": -0.02,
    "rho_1": 0.4,
    "rho_2": -0.3,
    "phi_1": 0.8,
    "sigma_u": 0.15,
    "sigma_eps": 0.10,
    "sigma_N": 0.07,
    "sigma_eta": 0.6,
    "sigma_zeta": 0.5,
    "lambda_ez": 0.2,
}
PRIORS = {"m0_Nhat": 0.0, "m0_Nhat_lag": 0.0, "m0_Nbar": 0.1, "P0_Nhat": 1.5, "P0_Nhat_lag": 1.5, "P0_Nbar": 2.0}


def _dense_reference(draw: dict, data: dict, priors: dict, *, theta: float | None) -> float:
    """log density of the stacked observations under the same state-space model."""
    T = data["pi"].size
    zeta = data["x"] - draw["phi_1"] * data["x_prev"]
    sigma_eta2 = draw["sigma_eta"] ** 2
    y_tilde = (
        data["pi"]
        - data["pi_expect"]
        - draw["alpha"] * (data["pi_prev"] - data["pi_expect"])
        - (draw["kappa_0"] / KAPPA_SCALE) * data["x"]
        - draw["lambda_ez"] * zeta
    )
    h_nbar = (draw["delta"] / KAPPA_SCALE) * data["x"]
    h_nhat = np.zeros(T) if theta is None else np.full(T, -theta)

    F = np.array([[draw["rho_1"], draw["rho_2"], 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    c = np.array([0.0, 0.0, draw["n"]])
    Q = np.diag([draw["sigma_u"] ** 2, 0.0, draw["sigma_eps"] ** 2])
    m0 = np.array([priors["m0_Nhat"], priors["m0_Nhat_lag"], priors["m0_Nbar"]])
    P0 = np.diag([priors["P0_Nhat"], priors["P0_Nhat_lag"], priors["P0_Nbar"]])

    # Marginal mean and autocovariance of the whole state path.
    means = [m0]
    variances = [P0]
    for _ in range(1, T):
        means.append(c + F @ means[-1])
        variances.append(F @ variances[-1] @ F.T + Q)
    cov = np.zeros((3 * T, 3 * T))
    for i in range(T):
        cov[3 * i : 3 * i + 3, 3 * i : 3 * i + 3] = variances[i]
        propagated = variances[i]
        for j in range(i + 1, T):
            propagated = propagated @ F.T
            cov[3 * i : 3 * i + 3, 3 * j : 3 * j + 3] = propagated
            cov[3 * j : 3 * j + 3, 3 * i : 3 * i + 3] = propagated.T
    state_mean = np.concatenate(means)

    rows, values, obs_var = [], [], []
    for t in range(T):
        block = np.zeros(3 * T)
        block[3 * t : 3 * t + 3] = [h_nhat[t], 0.0, h_nbar[t]]
        rows.append(block)
        values.append(y_tilde[t])
        obs_var.append(sigma_eta2)
        if np.isfinite(data["N_obs"][t]):
            block = np.zeros(3 * T)
            block[3 * t : 3 * t + 3] = [1.0, 0.0, 1.0]
            rows.append(block)
            values.append(data["N_obs"][t])
            obs_var.append(draw["sigma_N"] ** 2)
    H = np.vstack(rows)
    observation_cov = H @ cov @ H.T + np.diag(obs_var)
    innovation = np.asarray(values) - H @ state_mean
    sign, logdet = np.linalg.slogdet(observation_cov)
    assert sign > 0
    quadratic = innovation @ np.linalg.solve(observation_cov, innovation)
    dense = -0.5 * (len(values) * np.log(2.0 * np.pi) + logdet + quadratic)

    activity = -0.5 * np.sum(
        np.log(2.0 * np.pi) + np.log(draw["sigma_zeta"] ** 2) + zeta**2 / draw["sigma_zeta"] ** 2
    )
    return float(dense + activity)


def _single_draw_posterior(draw: dict):
    import xarray as xr

    return xr.Dataset({name: (("chain", "draw"), np.array([[value]])) for name, value in draw.items()})


@pytest.mark.parametrize("missing", [(), (1, 2, 4)])
def test_hsa_steady_periods_sum_to_the_dense_gaussian(missing) -> None:
    data = _data(7, missing=missing)

    log_lik = pointwise_log_likelihood("hsa_steady", _single_draw_posterior(DRAW), data, PRIORS)

    assert log_lik.shape == (1, 1, 7)
    expected = _dense_reference(DRAW, data, PRIORS, theta=None)
    # The filter adds the sampler's own 1e-10 positive-definiteness jitter to each
    # covariance, which the dense reference does not; that floors the agreement at
    # ~1e-8 in absolute terms. A genuinely wrong recursion is off by orders of magnitude.
    assert np.isclose(float(np.sum(log_lik)), expected, rtol=1e-6, atol=1e-6)


def test_hsa_const_theta_periods_sum_to_the_dense_gaussian() -> None:
    data = _data(7, missing=(3,))

    log_lik = pointwise_log_likelihood("hsa_const_theta", _single_draw_posterior(DRAW), data, PRIORS)

    expected = _dense_reference(DRAW, data, PRIORS, theta=DRAW["theta"])
    assert np.isclose(float(np.sum(log_lik)), expected, rtol=1e-6, atol=1e-6)


def test_missing_firm_counts_drop_only_that_row() -> None:
    """A quarter with no firm count still contributes its inflation row."""
    observed = _data(6)
    partly_missing = {**observed, "N_obs": observed["N_obs"].copy()}
    partly_missing["N_obs"][2] = np.nan

    full = pointwise_log_likelihood("hsa_steady", _single_draw_posterior(DRAW), observed, PRIORS)
    dropped = pointwise_log_likelihood("hsa_steady", _single_draw_posterior(DRAW), partly_missing, PRIORS)

    assert np.isfinite(dropped).all()
    # One fewer observed variable at t=2, so that period's density must rise.
    assert float(dropped[0, 0, 2]) > float(full[0, 0, 2])


def test_ces_matches_its_closed_form() -> None:
    draw = {k: DRAW[k] for k in ("alpha", "phi_1", "lambda_ez", "sigma_zeta")}
    draw["kappa"] = 0.2
    draw["sigma_eta"] = 0.6
    data = _data(5)

    log_lik = pointwise_log_likelihood("ces", _single_draw_posterior(draw), data, {})

    zeta = data["x"] - draw["phi_1"] * data["x_prev"]
    eta = (
        (data["pi"] - data["pi_expect"])
        - draw["alpha"] * (data["pi_prev"] - data["pi_expect"])
        - (draw["kappa"] / KAPPA_SCALE) * data["x"]
        - draw["lambda_ez"] * zeta
    )
    def normal(value, sd):
        return -0.5 * (np.log(2.0 * np.pi) + 2.0 * np.log(sd) + (value / sd) ** 2)

    expected = normal(eta, draw["sigma_eta"]) + normal(zeta, draw["sigma_zeta"])
    np.testing.assert_allclose(log_lik[0, 0], expected, rtol=1e-12, atol=1e-12)


def test_sigma_eta_is_recovered_when_only_sigma_e_is_stored() -> None:
    """hsa_full stores sigma_e, not sigma_eta; the identity must be inverted."""
    draw = dict(DRAW)
    lambda_ez, sigma_zeta, sigma_eta = draw["lambda_ez"], draw["sigma_zeta"], draw["sigma_eta"]
    del draw["sigma_eta"]
    draw["sigma_e"] = float(np.sqrt(lambda_ez**2 * sigma_zeta**2 + sigma_eta**2))
    data = _data(5)

    with_sigma_e = pointwise_log_likelihood("hsa_steady", _single_draw_posterior(draw), data, PRIORS)
    with_sigma_eta = pointwise_log_likelihood("hsa_steady", _single_draw_posterior(DRAW), data, PRIORS)

    np.testing.assert_allclose(with_sigma_e, with_sigma_eta, rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize("model", ["hsa_dynamic", "hsa_full"])
def test_models_without_an_exact_filter_are_refused_not_approximated(model) -> None:
    with pytest.raises(UnsupportedModel):
        pointwise_log_likelihood(model, _single_draw_posterior(DRAW), _data(4), PRIORS)


def test_waic_matches_its_definition() -> None:
    rng = np.random.default_rng(3)
    log_lik = rng.normal(size=(2, 50, 6)) - 1.0

    result = waic_from_pointwise(log_lik)

    flat = log_lik.reshape(-1, 6)
    lppd = np.log(np.mean(np.exp(flat), axis=0))
    penalty = np.var(flat, axis=0, ddof=1)
    assert np.isclose(result["elpd_waic"], float(np.sum(lppd - penalty)))
    assert np.isclose(result["p_waic"], float(np.sum(penalty)))
