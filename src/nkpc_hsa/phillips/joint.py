from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .data import CELL_SPECS, DesignData, robust_scale
from .inflation import CellFit, _draw_regression, _prior_sds, _quarterly_design, coefficient_names
from .state import (
    MeasurementPosterior,
    _draw_ig,
    _normal_regression_draw,
    _truncated_regression_draw,
    sample_linear_state_path,
)


@dataclass(frozen=True)
class JointCellFit:
    fit: CellFit
    qbar: np.ndarray
    qhat: np.ndarray
    importance_status: str = "alternating_conditional_ffbs"


def _fast_draw(
    rng: np.random.Generator,
    *,
    annual: np.ndarray,
    quarterly: np.ndarray,
    y: np.ndarray,
    pi_lag: np.ndarray,
    expectation: np.ndarray,
    x: np.ndarray,
    qbar: np.ndarray,
    beta: dict[str, float],
    q0: float,
    phi_q: float,
    d_e: float,
    b_e: float,
    variances: dict[str, float],
    q_scale: float,
    e_scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    cq = qbar - q0
    base = (
        beta["a"] + beta["beta_b"] * pi_lag + beta["beta_f"] * expectation
        + beta["psi"] * cq + beta["kappa_0"] * x + beta["kappa_1"] * cq * x
    )
    loading = -(beta["theta_0"] + beta["gamma"] * cq)
    observations = []
    for t in range(y.size):
        values, rows, obsvars = [y[t] - base[t]], [[loading[t], 0.0]], [variances["pi"]]
        if np.isfinite(annual[t]):
            values.append(annual[t] - qbar[t])
            rows.append([1.0, 0.0])
            obsvars.append(variances["annual"])
        values.append(quarterly[t])
        rows.append([b_e, 1.0])
        obsvars.append(variances["quarterly"])
        observations.append((np.asarray(values), np.asarray(rows), np.diag(obsvars)))
    states = sample_linear_state_path(
        rng,
        F=np.diag([phi_q, 1.0]),
        c=np.array([0.0, d_e]),
        Q=np.diag([variances["qhat"], variances["ebar"]]),
        m0=np.zeros(2),
        P0=np.diag([(2 * q_scale) ** 2, (2 * e_scale) ** 2]),
        observations=observations,
    )
    return states[:, 0], states[:, 1]


def _slow_draw(
    rng: np.random.Generator,
    *,
    annual: np.ndarray,
    y: np.ndarray,
    pi_lag: np.ndarray,
    expectation: np.ndarray,
    x: np.ndarray,
    qhat: np.ndarray,
    beta: dict[str, float],
    q0: float,
    d_q: float,
    variances: dict[str, float],
    q_scale: float,
) -> np.ndarray:
    loading = beta["psi"] + beta["kappa_1"] * x - beta["gamma"] * qhat
    constant = (
        beta["a"] + beta["beta_b"] * pi_lag + beta["beta_f"] * expectation
        - beta["psi"] * q0 + beta["kappa_0"] * x - beta["kappa_1"] * q0 * x
        - beta["theta_0"] * qhat + beta["gamma"] * q0 * qhat
    )
    observations = []
    for t in range(y.size):
        values, rows, obsvars = [y[t] - constant[t]], [[loading[t]]], [variances["pi"]]
        if np.isfinite(annual[t]):
            values.append(annual[t] - qhat[t])
            rows.append([1.0])
            obsvars.append(variances["annual"])
        observations.append((np.asarray(values), np.asarray(rows), np.diag(obsvars)))
    return sample_linear_state_path(
        rng,
        F=np.ones((1, 1)),
        c=np.array([d_q]),
        Q=np.array([[variances["qbar"]]]),
        m0=np.zeros(1),
        P0=np.array([[(2 * q_scale) ** 2]]),
        observations=observations,
    )[:, 0]


def fit_joint_qoq_e2(
    data: DesignData,
    measurement: MeasurementPosterior,
    *,
    cell: int,
    q0: float,
    iterations: int,
    warmup: int,
    thin: int,
    chains: int,
    seed: int,
    progress_tick: Callable[[], None] | None = None,
) -> JointCellFit:
    """Secondary full-joint E2 estimator using the design's two conditional FFBS blocks."""
    spec = next(item for item in CELL_SPECS if int(item["cell"]) == cell)
    price, activity = spec["inflation"], spec["activity"]
    y = data.quarterly[f"pi_{price}"].to_numpy(float)
    lag = data.quarterly[f"pi_{price}_lag1"].to_numpy(float)
    expectation = data.quarterly["expectation"].to_numpy(float)
    x = data.quarterly[f"x_{activity}"].to_numpy(float)
    x_scale = robust_scale(x)
    names = coefficient_names("E2")
    priors = _prior_sds(names, q_scale=data.q_scale, x_scale=x_scale)
    n_save = (iterations - warmup + thin - 1) // thin
    coef = np.zeros((chains, n_save, len(names)))
    sigma = np.zeros((chains, n_save))
    qbar_saved = np.zeros((chains, n_save, y.size))
    qhat_saved = np.zeros_like(qbar_saved)

    for chain in range(chains):
        rng = np.random.default_rng(seed + cell * 30011 + chain * 2029)
        qbar = measurement.draws["qbar"][chain, 0].copy()
        qhat = measurement.draws["qhat"][chain, 0].copy()
        ebar = measurement.draws["ebar"][chain, 0].copy()
        d_q = float(measurement.draws["d_q"][chain, 0])
        phi_q = float(measurement.draws["phi_q"][chain, 0])
        d_e = float(measurement.draws["d_e"][chain, 0])
        b_e = float(measurement.draws["b_e"][chain, 0])
        variances = {
            "qbar": float(measurement.draws["sigma_qbar"][chain, 0] ** 2),
            "qhat": float(measurement.draws["sigma_qhat"][chain, 0] ** 2),
            "ebar": float(measurement.draws["sigma_ebar"][chain, 0] ** 2),
            "annual": float(measurement.draws["sigma_annual"][chain, 0] ** 2),
            "quarterly": float(measurement.draws["sigma_quarterly"][chain, 0] ** 2),
            "pi": 4.0,
        }
        beta_array = np.zeros(len(names))
        beta = dict(zip(names, beta_array))
        save = 0
        for iteration in range(iterations):
            qhat, ebar = _fast_draw(
                rng, annual=data.annual_observation, quarterly=data.quarterly_indicator,
                y=y, pi_lag=lag, expectation=expectation, x=x, qbar=qbar, beta=beta,
                q0=q0, phi_q=phi_q, d_e=d_e, b_e=b_e, variances=variances,
                q_scale=data.q_scale, e_scale=data.e_scale,
            )
            qbar = _slow_draw(
                rng, annual=data.annual_observation, y=y, pi_lag=lag,
                expectation=expectation, x=x, qhat=qhat, beta=beta, q0=q0,
                d_q=d_q, variances=variances, q_scale=data.q_scale,
            )
            d_q = _normal_regression_draw(rng, np.diff(qbar), np.ones(y.size - 1), variances["qbar"], 0.0, 0.05 * data.q_scale)
            phi_q = _truncated_regression_draw(rng, qhat[1:], qhat[:-1], variances["qhat"], 0.5, 0.3)
            d_e = _normal_regression_draw(rng, np.diff(ebar), np.ones(y.size - 1), variances["ebar"], 0.0, 0.05 * data.e_scale)
            b_e = _normal_regression_draw(rng, data.quarterly_indicator - ebar, qhat, variances["quarterly"], data.e_scale / data.q_scale, 0.5 * data.e_scale / data.q_scale)
            res_qbar = np.diff(qbar) - d_q
            res_qhat = qhat[1:] - phi_q * qhat[:-1]
            res_ebar = np.diff(ebar) - d_e
            res_e = data.quarterly_indicator - ebar - b_e * qhat
            mask = np.isfinite(data.annual_observation)
            res_n = data.annual_observation[mask] - qbar[mask] - qhat[mask]
            for key, residual, s0 in (
                ("qbar", res_qbar, 0.05 * data.q_scale),
                ("qhat", res_qhat, 0.25 * data.q_scale),
                ("ebar", res_ebar, 0.05 * data.e_scale),
                ("quarterly", res_e, 0.50 * data.e_scale),
                ("annual", res_n, 0.25 * data.q_scale),
            ):
                variances[key] = _draw_ig(rng, 3.0 + residual.size / 2, 2 * s0**2 + np.dot(residual, residual) / 2)
            X, built_names = _quarterly_design(
                pi_lag=lag, expectation=expectation, x=x, qbar=qbar, qhat=qhat,
                q0=q0, model="E2",
            )
            if built_names != names:
                raise RuntimeError("Joint E2 design order changed.")
            beta_array, variances["pi"] = _draw_regression(rng, y, X, names, priors, variances["pi"])
            beta = dict(zip(names, beta_array))
            if iteration >= warmup and (iteration - warmup) % thin == 0:
                coef[chain, save] = beta_array
                sigma[chain, save] = np.sqrt(variances["pi"])
                qbar_saved[chain, save] = qbar
                qhat_saved[chain, save] = qhat
                save += 1
            if progress_tick is not None:
                progress_tick()
    fit = CellFit(
        cell=cell,
        inflation=price,
        activity=activity,
        model="E2",
        transformation="qoq",
        coefficient_names=names,
        coefficients=coef,
        sigma=sigma,
        q0=q0,
        x_scale=x_scale,
        prior_sds=priors,
        n_endpoints=y.size,
        expectation_status=str(data.config["cells"]["inflation"][price]["expectation_status"]),
        estimator="full_joint",
    )
    return JointCellFit(fit=fit, qbar=qbar_saved, qhat=qhat_saved)
