"""Full-joint annual-N, inverse-markup, and quarterly-inflation estimator.

The measurement block is the Q4-hard-anchored change bridge used by
``markup_measurement.py``.  Unlike the modular experiment, inflation is also
allowed to update both competition states.  Conditional on either ``qbar`` or
``qhat``, the E2 inflation equation is linear Gaussian, so alternating exact
conditional FFBS steps yield a Gibbs sampler for the full posterior.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from nkpc_hsa.phillips.data import DesignData, robust_scale
from nkpc_hsa.phillips.inflation import CellFit, _draw_regression, _prior_sds, _quarterly_design, coefficient_names
from nkpc_hsa.phillips.markup_measurement import (
    _normal_regression_draw,
    _positive_loading_draw,
    _stationary_ar2_draw,
    _unit_ar_draw,
)
from nkpc_hsa.phillips.state import MeasurementPosterior, _draw_ig, sample_linear_state_path


@dataclass(frozen=True)
class MarkupJointFit:
    fit: CellFit
    draws: dict[str, np.ndarray]


def _level_shift_log_kernel(
    *,
    qbar: np.ndarray,
    qhat: np.ndarray,
    initial_lag: float,
    beta: dict[str, float],
    rho_1: float,
    rho_2: float,
    variance_qhat: float,
    q_scale: float,
    priors: dict[str, float],
) -> float:
    """Terms that change under a total-q-preserving level shift."""
    second_lag = np.concatenate([[initial_lag], qhat[:-2]])
    residual = qhat[1:] - rho_1 * qhat[:-1] - rho_2 * second_lag
    state_prior = -0.5 * (
        (qbar[0] / (2.0 * q_scale)) ** 2
        + (qhat[0] / (2.0 * q_scale)) ** 2
        + (initial_lag / (2.0 * q_scale)) ** 2
        + float(np.dot(residual, residual)) / variance_qhat
    )
    coefficient_prior = -0.5 * sum(
        (float(beta[name]) / float(priors[name])) ** 2 for name in beta
    )
    return float(state_prior + coefficient_prior)


def _global_level_shift_move(
    rng: np.random.Generator,
    *,
    qbar: np.ndarray,
    qhat: np.ndarray,
    initial_lag: float,
    beta: dict[str, float],
    rho_1: float,
    rho_2: float,
    variance_qhat: float,
    q_scale: float,
    priors: dict[str, float],
    proposal_sd: float,
    steps: int = 4,
) -> tuple[np.ndarray, np.ndarray, float, dict[str, float], int]:
    """Move along the slow/fast level ridge while preserving total q and pi fit.

    For ``qbar' = qbar + c`` and ``qhat' = qhat - c``, the displayed E2
    coefficient transformation leaves the inflation conditional mean exactly
    unchanged.  Annual-N and markup likelihoods depend on total q and are also
    unchanged, leaving only state and coefficient priors in the MH ratio.
    """
    accepted = 0
    current = _level_shift_log_kernel(
        qbar=qbar,
        qhat=qhat,
        initial_lag=initial_lag,
        beta=beta,
        rho_1=rho_1,
        rho_2=rho_2,
        variance_qhat=variance_qhat,
        q_scale=q_scale,
        priors=priors,
    )
    for _ in range(steps):
        shift = float(rng.normal(0.0, proposal_sd))
        proposal_beta = dict(beta)
        proposal_beta["a"] = (
            beta["a"] - beta["psi"] * shift - beta["theta_0"] * shift
            + beta["gamma"] * shift**2
        )
        proposal_beta["psi"] = beta["psi"] - beta["gamma"] * shift
        proposal_beta["kappa_0"] = beta["kappa_0"] - beta["kappa_1"] * shift
        proposal_beta["theta_0"] = beta["theta_0"] - beta["gamma"] * shift
        proposal_qbar = qbar + shift
        proposal_qhat = qhat - shift
        proposal_lag = initial_lag - shift
        proposed = _level_shift_log_kernel(
            qbar=proposal_qbar,
            qhat=proposal_qhat,
            initial_lag=proposal_lag,
            beta=proposal_beta,
            rho_1=rho_1,
            rho_2=rho_2,
            variance_qhat=variance_qhat,
            q_scale=q_scale,
            priors=priors,
        )
        if np.log(rng.uniform()) < proposed - current:
            qbar, qhat, initial_lag, beta = (
                proposal_qbar,
                proposal_qhat,
                proposal_lag,
                proposal_beta,
            )
            current = proposed
            accepted += 1
    return qbar, qhat, initial_lag, beta, accepted


def _draw_qhat_block(
    rng: np.random.Generator,
    *,
    annual: np.ndarray,
    proxy: np.ndarray,
    y: np.ndarray,
    lag: np.ndarray,
    expectation: np.ndarray,
    x: np.ndarray,
    qbar: np.ndarray,
    beta: dict[str, float],
    q0: float,
    rho_1: float,
    rho_2: float,
    rho_markup: float,
    alpha_markup: float,
    variances: dict[str, float],
    q_scale: float,
) -> np.ndarray:
    # State order: qhat_t, qhat_(t-1), r_markup_t.
    F = np.array(
        [
            [rho_1, rho_2, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, rho_markup],
        ]
    )
    Q = np.diag([variances["qhat"], 1e-10, variances["r_markup"]])
    centered = qbar - q0
    base = (
        beta["a"]
        + beta["beta_b"] * lag
        + beta["beta_f"] * expectation
        + beta["psi"] * centered
        + beta["kappa_0"] * x
        + beta["kappa_1"] * centered * x
    )
    loading = -(beta["theta_0"] + beta["gamma"] * centered)
    delta_qbar = np.diff(qbar)
    delta_proxy = np.diff(proxy)
    annual_variance = (1e-6 * q_scale) ** 2
    observations = []
    for t in range(y.size):
        values = [y[t] - base[t]]
        rows = [[loading[t], 0.0, 0.0]]
        obsvars = [variances["pi"]]
        if np.isfinite(annual[t]):
            values.append(annual[t] - qbar[t])
            rows.append([1.0, 0.0, 0.0])
            obsvars.append(annual_variance)
        if t > 0:
            # Delta qhat_t - r_t = alpha Delta proxy_t - Delta qbar_t + omega_t.
            values.append(alpha_markup * delta_proxy[t - 1] - delta_qbar[t - 1])
            rows.append([1.0, -1.0, -1.0])
            obsvars.append(variances["markup"])
        observations.append((np.asarray(values), np.asarray(rows), np.diag(obsvars)))
    return sample_linear_state_path(
        rng,
        F=F,
        c=np.zeros(3),
        Q=Q,
        m0=np.zeros(3),
        P0=np.diag([(2.0 * q_scale) ** 2] * 3),
        observations=observations,
    )


def _draw_qbar_block(
    rng: np.random.Generator,
    *,
    annual: np.ndarray,
    proxy: np.ndarray,
    y: np.ndarray,
    lag: np.ndarray,
    expectation: np.ndarray,
    x: np.ndarray,
    qhat: np.ndarray,
    r_markup: np.ndarray,
    beta: dict[str, float],
    q0: float,
    drift: float,
    alpha_markup: float,
    variances: dict[str, float],
    q_scale: float,
) -> np.ndarray:
    # Carry qbar_(t-1) so the change bridge is an ordinary observation row.
    F = np.array([[1.0, 0.0], [1.0, 0.0]])
    Q = np.diag([variances["qbar"], 1e-10])
    loading = beta["psi"] + beta["kappa_1"] * x - beta["gamma"] * qhat
    constant = (
        beta["a"]
        + beta["beta_b"] * lag
        + beta["beta_f"] * expectation
        + beta["kappa_0"] * x
        - beta["psi"] * q0
        - beta["kappa_1"] * q0 * x
        - beta["theta_0"] * qhat
        + beta["gamma"] * q0 * qhat
    )
    delta_qhat = np.diff(qhat)
    delta_proxy = np.diff(proxy)
    annual_variance = (1e-6 * q_scale) ** 2
    observations = []
    for t in range(y.size):
        values = [y[t] - constant[t]]
        rows = [[loading[t], 0.0]]
        obsvars = [variances["pi"]]
        if np.isfinite(annual[t]):
            values.append(annual[t] - qhat[t])
            rows.append([1.0, 0.0])
            obsvars.append(annual_variance)
        if t > 0:
            # Delta qbar_t = alpha Delta proxy_t + r_t - Delta qhat_t + omega_t.
            values.append(
                alpha_markup * delta_proxy[t - 1]
                + r_markup[t]
                - delta_qhat[t - 1]
            )
            rows.append([1.0, -1.0])
            obsvars.append(variances["markup"])
        observations.append((np.asarray(values), np.asarray(rows), np.diag(obsvars)))
    states = sample_linear_state_path(
        rng,
        F=F,
        c=np.array([drift, 0.0]),
        Q=Q,
        m0=np.zeros(2),
        P0=np.diag([(2.0 * q_scale) ** 2] * 2),
        observations=observations,
    )
    return states[:, 0]


def fit_markup_full_joint_qoq_e2(
    data: DesignData,
    measurement: MeasurementPosterior,
    *,
    q0: float,
    iterations: int,
    warmup: int,
    thin: int,
    chains: int,
    seed: int,
    price: str = "core_cpi",
    activity: str = "negative_unemployment_gap",
    progress_tick: Callable[[], None] | None = None,
) -> MarkupJointFit:
    """Draw the full posterior for the AR(1)-error markup bridge and E2 NKPC."""
    if iterations <= warmup or thin < 1:
        raise ValueError("iterations must exceed warmup and thin must be positive.")
    if chains < 2:
        raise ValueError("At least two chains are required for diagnostics.")
    required_init = {
        "qbar",
        "qhat",
        "d_q",
        "rho_1",
        "rho_2",
        "sigma_qbar",
        "sigma_qhat",
        "alpha_markup",
        "sigma_markup",
        "r_markup",
        "rho_markup",
        "sigma_r_markup",
    }
    missing = required_init.difference(measurement.draws)
    if missing:
        raise KeyError(f"AR(1) markup initialization is missing: {sorted(missing)}")

    annual = np.asarray(data.annual_observation, float)
    markup = data.quarterly["markup"].to_numpy(float)
    # log(mu_ref / mu) differs from -log(mu) only by a constant, which cancels
    # exactly in the bridge's first difference.
    proxy = -np.log(markup)
    proxy -= float(np.mean(proxy))
    proxy_scale = robust_scale(proxy)
    y = data.quarterly[f"pi_{price}"].to_numpy(float)
    lag = data.quarterly[f"pi_{price}_lag1"].to_numpy(float)
    expectation = data.quarterly["expectation"].to_numpy(float)
    x = data.quarterly[f"x_{activity}"].to_numpy(float)
    x_scale = robust_scale(x)
    names = coefficient_names("E2")
    priors = _prior_sds(names, q_scale=data.q_scale, x_scale=x_scale)
    n_save = (iterations - warmup + thin - 1) // thin

    saved: dict[str, np.ndarray] = {
        "coefficients": np.zeros((chains, n_save, len(names))),
        "sigma_pi": np.zeros((chains, n_save)),
        "qbar": np.zeros((chains, n_save, y.size)),
        "qhat": np.zeros((chains, n_save, y.size)),
        "r_markup": np.zeros((chains, n_save, y.size)),
    }
    for scalar in (
        "d_q",
        "rho_1",
        "rho_2",
        "sigma_qbar",
        "sigma_qhat",
        "alpha_markup",
        "sigma_markup",
        "rho_markup",
        "sigma_r_markup",
        "max_anchor_error",
        "level_shift_acceptance",
    ):
        saved[scalar] = np.zeros((chains, n_save))

    prior_shape = 3.0
    loading_scale = data.q_scale / proxy_scale
    for chain in range(chains):
        rng = np.random.default_rng(seed + chain * 2029)
        init_draw = min(chain, measurement.draws["qbar"].shape[0] - 1)
        qbar = measurement.draws["qbar"][init_draw, 0].copy()
        qhat = measurement.draws["qhat"][init_draw, 0].copy()
        r_markup = measurement.draws["r_markup"][init_draw, 0].copy()
        drift = float(measurement.draws["d_q"][init_draw, 0])
        rho_1 = float(measurement.draws["rho_1"][init_draw, 0])
        rho_2 = float(measurement.draws["rho_2"][init_draw, 0])
        rho_markup = float(measurement.draws["rho_markup"][init_draw, 0])
        alpha_markup = float(measurement.draws["alpha_markup"][init_draw, 0])
        variances = {
            "qbar": float(measurement.draws["sigma_qbar"][init_draw, 0] ** 2),
            "qhat": float(measurement.draws["sigma_qhat"][init_draw, 0] ** 2),
            "markup": float(measurement.draws["sigma_markup"][init_draw, 0] ** 2),
            "r_markup": float(measurement.draws["sigma_r_markup"][init_draw, 0] ** 2),
            "pi": 4.0,
        }
        beta_array = np.zeros(len(names))
        beta = dict(zip(names, beta_array))
        save = 0
        for iteration in range(iterations):
            fast_state = _draw_qhat_block(
                rng,
                annual=annual,
                proxy=proxy,
                y=y,
                lag=lag,
                expectation=expectation,
                x=x,
                qbar=qbar,
                beta=beta,
                q0=q0,
                rho_1=rho_1,
                rho_2=rho_2,
                rho_markup=rho_markup,
                alpha_markup=alpha_markup,
                variances=variances,
                q_scale=data.q_scale,
            )
            qhat = fast_state[:, 0]
            initial_lag = float(fast_state[0, 1])
            r_markup = fast_state[:, 2]
            qbar = _draw_qbar_block(
                rng,
                annual=annual,
                proxy=proxy,
                y=y,
                lag=lag,
                expectation=expectation,
                x=x,
                qhat=qhat,
                r_markup=r_markup,
                beta=beta,
                q0=q0,
                drift=drift,
                alpha_markup=alpha_markup,
                variances=variances,
                q_scale=data.q_scale,
            )

            drift = float(
                _normal_regression_draw(
                    rng,
                    np.diff(qbar),
                    np.ones((y.size - 1, 1)),
                    variances["qbar"],
                    np.array([0.0]),
                    np.array([0.05 * data.q_scale]),
                )[0]
            )
            rho_1, rho_2 = _stationary_ar2_draw(
                rng,
                qhat,
                initial_lag,
                variances["qhat"],
                (rho_1, rho_2),
            )
            rho_markup = _unit_ar_draw(
                rng,
                r_markup[1:],
                r_markup[:-1],
                variances["r_markup"],
                rho_markup,
            )
            delta_total = np.diff(qbar + qhat)
            delta_proxy = np.diff(proxy)
            alpha_markup = _positive_loading_draw(
                rng,
                delta_total - r_markup[1:],
                delta_proxy,
                variances["markup"],
                0.0,
                2.0 * loading_scale,
            )

            second_lag = np.concatenate([[initial_lag], qhat[:-2]])
            residuals = {
                "qbar": np.diff(qbar) - drift,
                "qhat": qhat[1:] - rho_1 * qhat[:-1] - rho_2 * second_lag,
                "markup": delta_total - alpha_markup * delta_proxy - r_markup[1:],
                "r_markup": r_markup[1:] - rho_markup * r_markup[:-1],
            }
            scales = {
                "qbar": 0.05 * data.q_scale,
                "qhat": 0.25 * data.q_scale,
                "markup": 0.35 * data.q_scale,
                "r_markup": 0.15 * data.q_scale,
            }
            for key, residual in residuals.items():
                variances[key] = _draw_ig(
                    rng,
                    prior_shape + residual.size / 2,
                    2.0 * scales[key] ** 2 + float(np.dot(residual, residual)) / 2.0,
                )

            X, built_names = _quarterly_design(
                pi_lag=lag,
                expectation=expectation,
                x=x,
                qbar=qbar,
                qhat=qhat,
                q0=q0,
                model="E2",
            )
            if built_names != names:
                raise RuntimeError("Full-joint E2 design order changed.")
            beta_array, variances["pi"] = _draw_regression(
                rng, y, X, names, priors, variances["pi"]
            )
            beta = dict(zip(names, beta_array))
            qbar, qhat, initial_lag, beta, shift_accepts = _global_level_shift_move(
                rng,
                qbar=qbar,
                qhat=qhat,
                initial_lag=initial_lag,
                beta=beta,
                rho_1=rho_1,
                rho_2=rho_2,
                variance_qhat=variances["qhat"],
                q_scale=data.q_scale,
                priors=priors,
                proposal_sd=0.20 * data.q_scale,
            )
            beta_array = np.asarray([beta[name] for name in names])

            if iteration >= warmup and (iteration - warmup) % thin == 0:
                saved["coefficients"][chain, save] = beta_array
                saved["sigma_pi"][chain, save] = np.sqrt(variances["pi"])
                saved["qbar"][chain, save] = qbar
                saved["qhat"][chain, save] = qhat
                saved["r_markup"][chain, save] = r_markup
                saved["d_q"][chain, save] = drift
                saved["rho_1"][chain, save] = rho_1
                saved["rho_2"][chain, save] = rho_2
                saved["sigma_qbar"][chain, save] = np.sqrt(variances["qbar"])
                saved["sigma_qhat"][chain, save] = np.sqrt(variances["qhat"])
                saved["alpha_markup"][chain, save] = alpha_markup
                saved["sigma_markup"][chain, save] = np.sqrt(variances["markup"])
                saved["rho_markup"][chain, save] = rho_markup
                saved["sigma_r_markup"][chain, save] = np.sqrt(variances["r_markup"])
                mask = np.isfinite(annual)
                saved["max_anchor_error"][chain, save] = float(
                    np.max(np.abs(annual[mask] - qbar[mask] - qhat[mask]))
                )
                saved["level_shift_acceptance"][chain, save] = shift_accepts / 4.0
                save += 1
            if progress_tick is not None:
                progress_tick()

    fit = CellFit(
        cell=9,
        inflation=price,
        activity=activity,
        model="E2",
        transformation="qoq",
        coefficient_names=names,
        coefficients=saved["coefficients"],
        sigma=saved["sigma_pi"],
        q0=q0,
        x_scale=x_scale,
        prior_sds=priors,
        n_endpoints=y.size,
        expectation_status="headline_cpi_proxy_one_quarter_ahead",
        estimator="full_joint_markup_bridge",
    )
    return MarkupJointFit(fit=fit, draws=saved)
