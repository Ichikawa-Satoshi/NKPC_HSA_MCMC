from __future__ import annotations

from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Callable

import numpy as np
from scipy.stats import truncnorm

from nkpc_hsa.gibbs.common.joint_ffbs import force_pd


@dataclass(frozen=True)
class MeasurementPosterior:
    draws: dict[str, np.ndarray]
    annual_only_draws: dict[str, np.ndarray]
    information_ratio: float
    periods: tuple[str, ...]


@dataclass(frozen=True)
class MeasurementSpec:
    """Frozen identification sensitivity for the mixed-frequency state block."""

    name: str
    q_drift: bool = True
    e_drift: bool = True
    phi_mean: float = 0.5
    phi_sd: float = 0.3
    phi_fixed: float | None = None
    idiosyncratic_cycle: bool = False


def sample_linear_state_path(
    rng: np.random.Generator,
    *,
    F: np.ndarray,
    c: np.ndarray,
    Q: np.ndarray,
    m0: np.ndarray,
    P0: np.ndarray,
    observations: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
) -> np.ndarray:
    """Generic missing-row Kalman/FFBS draw used by conditional state blocks."""
    F, c, Q = np.asarray(F, float), np.asarray(c, float), np.asarray(Q, float)
    m0, P0 = np.asarray(m0, float), np.asarray(P0, float)
    T, dim = len(observations), c.size
    if F.shape != (dim, dim) or Q.shape != (dim, dim) or m0.shape != (dim,) or P0.shape != (dim, dim):
        raise ValueError("Linear-state dimensions do not agree.")
    m_pred = np.zeros((T, dim))
    P_pred = np.zeros((T, dim, dim))
    m_filt = np.zeros((T, dim))
    P_filt = np.zeros((T, dim, dim))
    I = np.eye(dim)
    for t, (values, H, R) in enumerate(observations):
        if t == 0:
            m_pred[t], P_pred[t] = m0, force_pd(P0)
        else:
            m_pred[t] = c + F @ m_filt[t - 1]
            P_pred[t] = force_pd(F @ P_filt[t - 1] @ F.T + Q)
        values, H, R = np.asarray(values, float), np.asarray(H, float), np.asarray(R, float)
        if values.size == 0:
            m_filt[t], P_filt[t] = m_pred[t], P_pred[t]
            continue
        S = force_pd(H @ P_pred[t] @ H.T + R)
        K = np.linalg.solve(S, H @ P_pred[t]).T
        m_filt[t] = m_pred[t] + K @ (values - H @ m_pred[t])
        P_filt[t] = force_pd((I - K @ H) @ P_pred[t] @ (I - K @ H).T + K @ R @ K.T)
    states = np.zeros((T, dim))
    states[-1] = rng.multivariate_normal(m_filt[-1], force_pd(P_filt[-1]))
    for t in range(T - 2, -1, -1):
        Pnext = force_pd(P_pred[t + 1])
        J = np.linalg.solve(Pnext, F @ P_filt[t]).T
        mean = m_filt[t] + J @ (states[t + 1] - c - F @ m_filt[t])
        cov = force_pd(P_filt[t] - J @ Pnext @ J.T)
        states[t] = rng.multivariate_normal(mean, cov)
    return states


def _draw_ig(rng: np.random.Generator, shape: float, scale: float) -> float:
    return float(1.0 / rng.gamma(shape, 1.0 / scale))


def _normal_regression_draw(
    rng: np.random.Generator,
    y: np.ndarray,
    x: np.ndarray,
    variance: float,
    prior_mean: float,
    prior_sd: float,
) -> float:
    precision = 1.0 / prior_sd**2 + float(np.dot(x, x)) / variance
    post_variance = 1.0 / precision
    post_mean = post_variance * (
        prior_mean / prior_sd**2 + float(np.dot(x, y)) / variance
    )
    return float(rng.normal(post_mean, np.sqrt(post_variance)))


def _truncated_regression_draw(
    rng: np.random.Generator,
    y: np.ndarray,
    x: np.ndarray,
    variance: float,
    prior_mean: float,
    prior_sd: float,
) -> float:
    precision = 1.0 / prior_sd**2 + float(np.dot(x, x)) / variance
    post_sd = np.sqrt(1.0 / precision)
    post_mean = (prior_mean / prior_sd**2 + float(np.dot(x, y)) / variance) / precision
    a = (-1.0 - post_mean) / post_sd
    b = (1.0 - post_mean) / post_sd
    return float(truncnorm.rvs(a, b, loc=post_mean, scale=post_sd, random_state=rng))


def _ffbs(
    rng: np.random.Generator,
    *,
    annual: np.ndarray,
    quarterly: np.ndarray,
    include_quarterly: bool,
    d_q: float,
    phi_q: float,
    d_e: float,
    b_e: float,
    var_qbar: float,
    var_qhat: float,
    var_ebar: float,
    var_annual: float,
    var_quarterly: float,
    q_scale: float,
    e_scale: float,
) -> np.ndarray:
    T = annual.size
    dim = 3 if include_quarterly else 2
    F = np.diag(([1.0, phi_q, 1.0] if include_quarterly else [1.0, phi_q])).astype(float)
    c = np.array(([d_q, 0.0, d_e] if include_quarterly else [d_q, 0.0]), dtype=float)
    Q = np.diag(([var_qbar, var_qhat, var_ebar] if include_quarterly else [var_qbar, var_qhat]))
    m_pred = np.zeros((T, dim))
    P_pred = np.zeros((T, dim, dim))
    m_filt = np.zeros((T, dim))
    P_filt = np.zeros((T, dim, dim))
    initial_sd = np.array(([2 * q_scale, 2 * q_scale, 2 * e_scale] if include_quarterly else [2 * q_scale, 2 * q_scale]))
    I = np.eye(dim)
    for t in range(T):
        if t == 0:
            m_pred[t] = 0.0
            P_pred[t] = np.diag(initial_sd**2)
        else:
            m_pred[t] = c + F @ m_filt[t - 1]
            P_pred[t] = force_pd(F @ P_filt[t - 1] @ F.T + Q)
        rows: list[np.ndarray] = []
        values: list[float] = []
        variances: list[float] = []
        if np.isfinite(annual[t]):
            rows.append(np.array(([1.0, 1.0, 0.0] if include_quarterly else [1.0, 1.0])))
            values.append(float(annual[t]))
            variances.append(var_annual)
        if include_quarterly and np.isfinite(quarterly[t]):
            rows.append(np.array([0.0, b_e, 1.0]))
            values.append(float(quarterly[t]))
            variances.append(var_quarterly)
        if not rows:
            m_filt[t], P_filt[t] = m_pred[t], P_pred[t]
            continue
        H = np.vstack(rows)
        R = np.diag(variances)
        S = force_pd(H @ P_pred[t] @ H.T + R)
        K = np.linalg.solve(S, H @ P_pred[t]).T
        m_filt[t] = m_pred[t] + K @ (np.asarray(values) - H @ m_pred[t])
        P_filt[t] = force_pd((I - K @ H) @ P_pred[t] @ (I - K @ H).T + K @ R @ K.T)

    states = np.zeros((T, dim))
    states[-1] = rng.multivariate_normal(m_filt[-1], force_pd(P_filt[-1]))
    for t in range(T - 2, -1, -1):
        Pnext = force_pd(P_pred[t + 1])
        J = np.linalg.solve(Pnext, F @ P_filt[t]).T
        mean = m_filt[t] + J @ (states[t + 1] - c - F @ m_filt[t])
        cov = force_pd(P_filt[t] - J @ Pnext @ J.T)
        states[t] = rng.multivariate_normal(mean, cov)
    return states


def _run_chain(
    *,
    annual: np.ndarray,
    quarterly: np.ndarray,
    q_scale: float,
    e_scale: float,
    include_quarterly: bool,
    iterations: int,
    warmup: int,
    thin: int,
    seed: int,
    progress_tick: Callable[[], None] | None = None,
) -> dict[str, np.ndarray]:
    if iterations <= warmup or thin < 1:
        raise ValueError("iterations must exceed warmup and thin must be positive.")
    rng = np.random.default_rng(seed)
    pri_shape = 3.0
    d_q, phi_q, d_e = 0.0, 0.5, 0.0
    b_e = e_scale / q_scale
    var_qbar = (0.05 * q_scale) ** 2
    var_qhat = (0.25 * q_scale) ** 2
    var_annual = (0.25 * q_scale) ** 2
    var_ebar = (0.05 * e_scale) ** 2
    var_quarterly = (0.50 * e_scale) ** 2
    state = np.zeros((annual.size, 3 if include_quarterly else 2))
    saved: dict[str, list[np.ndarray | float]] = {
        name: []
        for name in (
            "qbar", "qhat", "d_q", "phi_q", "sigma_qbar", "sigma_qhat", "sigma_annual"
        )
    }
    if include_quarterly:
        saved.update({name: [] for name in ("ebar", "d_e", "b_e", "sigma_ebar", "sigma_quarterly")})

    for iteration in range(iterations):
        state = _ffbs(
            rng,
            annual=annual,
            quarterly=quarterly,
            include_quarterly=include_quarterly,
            d_q=d_q,
            phi_q=phi_q,
            d_e=d_e,
            b_e=b_e,
            var_qbar=var_qbar,
            var_qhat=var_qhat,
            var_ebar=var_ebar,
            var_annual=var_annual,
            var_quarterly=var_quarterly,
            q_scale=q_scale,
            e_scale=e_scale,
        )
        qbar, qhat = state[:, 0], state[:, 1]
        d_q = _normal_regression_draw(
            rng, np.diff(qbar), np.ones(qbar.size - 1), var_qbar, 0.0, 0.05 * q_scale
        )
        phi_q = _truncated_regression_draw(
            rng, qhat[1:], qhat[:-1], var_qhat, 0.5, 0.3
        )
        res_qbar = np.diff(qbar) - d_q
        res_qhat = qhat[1:] - phi_q * qhat[:-1]
        var_qbar = _draw_ig(rng, pri_shape + res_qbar.size / 2, 2 * (0.05 * q_scale) ** 2 + np.dot(res_qbar, res_qbar) / 2)
        var_qhat = _draw_ig(rng, pri_shape + res_qhat.size / 2, 2 * (0.25 * q_scale) ** 2 + np.dot(res_qhat, res_qhat) / 2)
        mask_n = np.isfinite(annual)
        res_n = annual[mask_n] - qbar[mask_n] - qhat[mask_n]
        var_annual = _draw_ig(rng, pri_shape + res_n.size / 2, 2 * (0.25 * q_scale) ** 2 + np.dot(res_n, res_n) / 2)

        if include_quarterly:
            ebar = state[:, 2]
            d_e = _normal_regression_draw(
                rng, np.diff(ebar), np.ones(ebar.size - 1), var_ebar, 0.0, 0.05 * e_scale
            )
            b_e = _normal_regression_draw(
                rng,
                quarterly - ebar,
                qhat,
                var_quarterly,
                e_scale / q_scale,
                0.5 * e_scale / q_scale,
            )
            res_ebar = np.diff(ebar) - d_e
            res_e = quarterly - ebar - b_e * qhat
            var_ebar = _draw_ig(rng, pri_shape + res_ebar.size / 2, 2 * (0.05 * e_scale) ** 2 + np.dot(res_ebar, res_ebar) / 2)
            var_quarterly = _draw_ig(rng, pri_shape + res_e.size / 2, 2 * (0.50 * e_scale) ** 2 + np.dot(res_e, res_e) / 2)

        if iteration >= warmup and (iteration - warmup) % thin == 0:
            values: dict[str, np.ndarray | float] = {
                "qbar": qbar.copy(),
                "qhat": qhat.copy(),
                "d_q": d_q,
                "phi_q": phi_q,
                "sigma_qbar": np.sqrt(var_qbar),
                "sigma_qhat": np.sqrt(var_qhat),
                "sigma_annual": np.sqrt(var_annual),
            }
            if include_quarterly:
                values.update(
                    ebar=ebar.copy(),
                    d_e=d_e,
                    b_e=b_e,
                    sigma_ebar=np.sqrt(var_ebar),
                    sigma_quarterly=np.sqrt(var_quarterly),
                )
            for name, value in values.items():
                saved[name].append(value)
        if progress_tick is not None:
            progress_tick()
    return {name: np.asarray(values, dtype=float) for name, values in saved.items()}


def sample_measurement_posterior(
    annual: np.ndarray,
    quarterly: np.ndarray,
    *,
    q_scale: float,
    e_scale: float,
    periods: tuple[str, ...],
    iterations: int,
    warmup: int,
    thin: int,
    chains: int,
    seed: int,
    progress_tick: Callable[[], None] | None = None,
) -> MeasurementPosterior:
    """Estimate the annual-only N module and quarterly-augmented C module."""
    if chains < 2:
        raise ValueError("At least two chains are required for the declared diagnostics.")
    outputs: dict[bool, dict[str, list[np.ndarray]]] = {False: {}, True: {}}
    for include_quarterly in (False, True):
        for chain in range(chains):
            result = _run_chain(
                annual=np.asarray(annual, dtype=float),
                quarterly=np.asarray(quarterly, dtype=float),
                q_scale=q_scale,
                e_scale=e_scale,
                include_quarterly=include_quarterly,
                iterations=iterations,
                warmup=warmup,
                thin=thin,
                seed=seed + (100003 if include_quarterly else 0) + chain * 1009,
                progress_tick=progress_tick,
            )
            for name, values in result.items():
                outputs[include_quarterly].setdefault(name, []).append(values)
    annual_draws = {name: np.stack(values, axis=0) for name, values in outputs[False].items()}
    augmented_draws = {name: np.stack(values, axis=0) for name, values in outputs[True].items()}
    sd_n = np.std(annual_draws["qhat"], axis=(0, 1), ddof=1)
    sd_c = np.std(augmented_draws["qhat"], axis=(0, 1), ddof=1)
    ratio = float(np.median(sd_c) / np.median(sd_n))
    return MeasurementPosterior(
        draws=augmented_draws,
        annual_only_draws=annual_draws,
        information_ratio=ratio,
        periods=periods,
    )


def sample_annual_only_posterior(
    annual: np.ndarray,
    *,
    q_scale: float,
    periods: tuple[str, ...],
    iterations: int,
    warmup: int,
    thin: int,
    chains: int,
    seed: int,
    progress_tick: Callable[[], None] | None = None,
) -> MeasurementPosterior:
    """Infer quarterly states from annual N observations and the state law only."""
    if chains < 2:
        raise ValueError("At least two chains are required for diagnostics.")
    outputs: dict[str, list[np.ndarray]] = {}
    missing_quarterly = np.full(np.asarray(annual).size, np.nan)
    for chain in range(chains):
        result = _run_chain(
            annual=np.asarray(annual, dtype=float),
            quarterly=missing_quarterly,
            q_scale=q_scale,
            e_scale=q_scale,
            include_quarterly=False,
            iterations=iterations,
            warmup=warmup,
            thin=thin,
            seed=seed + chain * 1009,
            progress_tick=progress_tick,
        )
        for name, values in result.items():
            outputs.setdefault(name, []).append(values)
    draws = {name: np.stack(values, axis=0) for name, values in outputs.items()}
    return MeasurementPosterior(
        draws=draws,
        annual_only_draws=draws,
        information_ratio=float("nan"),
        periods=periods,
    )


def _variant_ffbs(
    rng: np.random.Generator,
    *,
    annual: np.ndarray,
    quarterly: np.ndarray,
    include_quarterly: bool,
    spec: MeasurementSpec,
    d_q: float,
    phi_q: float,
    d_e: float,
    b_e: float,
    rho_e: float,
    variances: dict[str, float],
    q_scale: float,
    e_scale: float,
) -> np.ndarray:
    dim = 2
    if include_quarterly:
        dim += 1 + int(spec.idiosyncratic_cycle)
    diagonal = [1.0, phi_q]
    drift = [d_q if spec.q_drift else 0.0, 0.0]
    innovation = [variances["qbar"], variances["qhat"]]
    initial_sd = [2.0 * q_scale, 2.0 * q_scale]
    if include_quarterly:
        diagonal.append(1.0)
        drift.append(d_e if spec.e_drift else 0.0)
        innovation.append(variances["ebar"])
        initial_sd.append(2.0 * e_scale)
        if spec.idiosyncratic_cycle:
            diagonal.append(rho_e)
            drift.append(0.0)
            innovation.append(variances["r_e"])
            initial_sd.append(2.0 * e_scale)
    observations: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    for t in range(annual.size):
        values: list[float] = []
        rows: list[np.ndarray] = []
        measurement_variances: list[float] = []
        if np.isfinite(annual[t]):
            row = np.zeros(dim)
            row[:2] = 1.0
            rows.append(row)
            values.append(float(annual[t]))
            measurement_variances.append(variances["annual"])
        if include_quarterly and np.isfinite(quarterly[t]):
            row = np.zeros(dim)
            row[1] = b_e
            row[2] = 1.0
            if spec.idiosyncratic_cycle:
                row[3] = 1.0
            rows.append(row)
            values.append(float(quarterly[t]))
            measurement_variances.append(variances["quarterly"])
        if rows:
            observations.append(
                (
                    np.asarray(values),
                    np.vstack(rows),
                    np.diag(measurement_variances),
                )
            )
        else:
            observations.append((np.empty(0), np.empty((0, dim)), np.empty((0, 0))))
    return sample_linear_state_path(
        rng,
        F=np.diag(diagonal),
        c=np.asarray(drift),
        Q=np.diag(innovation),
        m0=np.zeros(dim),
        P0=np.diag(np.square(initial_sd)),
        observations=observations,
    )


def _run_variant_chain(
    *,
    annual: np.ndarray,
    quarterly: np.ndarray,
    q_scale: float,
    e_scale: float,
    include_quarterly: bool,
    spec: MeasurementSpec,
    iterations: int,
    warmup: int,
    thin: int,
    seed: int,
    progress_tick: Callable[[], None] | None,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    prior_shape = 3.0
    d_q, phi_q, d_e = 0.0, spec.phi_mean, 0.0
    b_e = e_scale / q_scale
    rho_e = 0.5
    variances = {
        "qbar": (0.05 * q_scale) ** 2,
        "qhat": (0.25 * q_scale) ** 2,
        "annual": (0.25 * q_scale) ** 2,
        "ebar": (0.05 * e_scale) ** 2,
        "quarterly": (0.50 * e_scale) ** 2,
        "r_e": (0.25 * e_scale) ** 2,
    }
    saved: dict[str, list[np.ndarray | float]] = {
        name: []
        for name in (
            "qbar",
            "qhat",
            "d_q",
            "phi_q",
            "sigma_qbar",
            "sigma_qhat",
            "sigma_annual",
        )
    }
    if include_quarterly:
        saved.update(
            {
                name: []
                for name in (
                    "ebar",
                    "d_e",
                    "b_e",
                    "sigma_ebar",
                    "sigma_quarterly",
                )
            }
        )
        if spec.idiosyncratic_cycle:
            saved.update({"r_e": [], "rho_e": [], "sigma_r_e": []})

    for iteration in range(iterations):
        state = _variant_ffbs(
            rng,
            annual=annual,
            quarterly=quarterly,
            include_quarterly=include_quarterly,
            spec=spec,
            d_q=d_q,
            phi_q=phi_q,
            d_e=d_e,
            b_e=b_e,
            rho_e=rho_e,
            variances=variances,
            q_scale=q_scale,
            e_scale=e_scale,
        )
        qbar, qhat = state[:, 0], state[:, 1]
        if spec.q_drift:
            d_q = _normal_regression_draw(
                rng,
                np.diff(qbar),
                np.ones(qbar.size - 1),
                variances["qbar"],
                0.0,
                0.05 * q_scale,
            )
        else:
            d_q = 0.0
        if spec.phi_fixed is None:
            phi_q = _truncated_regression_draw(
                rng,
                qhat[1:],
                qhat[:-1],
                variances["qhat"],
                spec.phi_mean,
                spec.phi_sd,
            )
        else:
            if not -1.0 < spec.phi_fixed < 1.0:
                raise ValueError("A fixed phi_q must lie strictly inside (-1, 1).")
            phi_q = float(spec.phi_fixed)
        res_qbar = np.diff(qbar) - d_q
        res_qhat = qhat[1:] - phi_q * qhat[:-1]
        variances["qbar"] = _draw_ig(
            rng,
            prior_shape + res_qbar.size / 2,
            2 * (0.05 * q_scale) ** 2 + np.dot(res_qbar, res_qbar) / 2,
        )
        variances["qhat"] = _draw_ig(
            rng,
            prior_shape + res_qhat.size / 2,
            2 * (0.25 * q_scale) ** 2 + np.dot(res_qhat, res_qhat) / 2,
        )
        annual_mask = np.isfinite(annual)
        res_annual = annual[annual_mask] - qbar[annual_mask] - qhat[annual_mask]
        variances["annual"] = _draw_ig(
            rng,
            prior_shape + res_annual.size / 2,
            2 * (0.25 * q_scale) ** 2 + np.dot(res_annual, res_annual) / 2,
        )

        if include_quarterly:
            ebar = state[:, 2]
            r_e = state[:, 3] if spec.idiosyncratic_cycle else np.zeros_like(ebar)
            if spec.e_drift:
                d_e = _normal_regression_draw(
                    rng,
                    np.diff(ebar),
                    np.ones(ebar.size - 1),
                    variances["ebar"],
                    0.0,
                    0.05 * e_scale,
                )
            else:
                d_e = 0.0
            b_e = _normal_regression_draw(
                rng,
                quarterly - ebar - r_e,
                qhat,
                variances["quarterly"],
                e_scale / q_scale,
                0.5 * e_scale / q_scale,
            )
            res_ebar = np.diff(ebar) - d_e
            res_quarterly = quarterly - ebar - b_e * qhat - r_e
            variances["ebar"] = _draw_ig(
                rng,
                prior_shape + res_ebar.size / 2,
                2 * (0.05 * e_scale) ** 2 + np.dot(res_ebar, res_ebar) / 2,
            )
            variances["quarterly"] = _draw_ig(
                rng,
                prior_shape + res_quarterly.size / 2,
                2 * (0.50 * e_scale) ** 2 + np.dot(res_quarterly, res_quarterly) / 2,
            )
            if spec.idiosyncratic_cycle:
                rho_e = _truncated_regression_draw(
                    rng,
                    r_e[1:],
                    r_e[:-1],
                    variances["r_e"],
                    0.5,
                    0.3,
                )
                res_r = r_e[1:] - rho_e * r_e[:-1]
                variances["r_e"] = _draw_ig(
                    rng,
                    prior_shape + res_r.size / 2,
                    2 * (0.25 * e_scale) ** 2 + np.dot(res_r, res_r) / 2,
                )

        if iteration >= warmup and (iteration - warmup) % thin == 0:
            values: dict[str, np.ndarray | float] = {
                "qbar": qbar.copy(),
                "qhat": qhat.copy(),
                "d_q": d_q,
                "phi_q": phi_q,
                "sigma_qbar": np.sqrt(variances["qbar"]),
                "sigma_qhat": np.sqrt(variances["qhat"]),
                "sigma_annual": np.sqrt(variances["annual"]),
            }
            if include_quarterly:
                values.update(
                    ebar=ebar.copy(),
                    d_e=d_e,
                    b_e=b_e,
                    sigma_ebar=np.sqrt(variances["ebar"]),
                    sigma_quarterly=np.sqrt(variances["quarterly"]),
                )
                if spec.idiosyncratic_cycle:
                    values.update(
                        r_e=r_e.copy(),
                        rho_e=rho_e,
                        sigma_r_e=np.sqrt(variances["r_e"]),
                    )
            for name, value in values.items():
                saved[name].append(value)
        if progress_tick is not None:
            progress_tick()
    return {name: np.asarray(values, dtype=float) for name, values in saved.items()}


def sample_measurement_variant(
    annual: np.ndarray,
    quarterly: np.ndarray,
    *,
    q_scale: float,
    e_scale: float,
    periods: tuple[str, ...],
    spec: MeasurementSpec,
    iterations: int,
    warmup: int,
    thin: int,
    chains: int,
    seed: int,
    progress_tick: Callable[[], None] | None = None,
    progress_update: Callable[[int], None] | None = None,
    parallel_chains: bool = False,
) -> MeasurementPosterior:
    """Estimate an alternative annual-only/augmented pair under one frozen state law."""
    outputs: dict[bool, dict[str, list[np.ndarray]]] = {False: {}, True: {}}
    if parallel_chains:
        tasks: list[tuple[bool, int]] = [
            (include_quarterly, chain)
            for include_quarterly in (False, True)
            for chain in range(chains)
        ]
        ordered: dict[tuple[bool, int], dict[str, np.ndarray]] = {}
        completed_iterations = 0
        with ProcessPoolExecutor(max_workers=chains) as executor:
            futures = {
                executor.submit(
                    _run_variant_chain,
                    annual=np.asarray(annual, dtype=float),
                    quarterly=np.asarray(quarterly, dtype=float),
                    q_scale=q_scale,
                    e_scale=e_scale,
                    include_quarterly=include_quarterly,
                    spec=spec,
                    iterations=iterations,
                    warmup=warmup,
                    thin=thin,
                    seed=seed
                    + (100003 if include_quarterly else 0)
                    + chain * 1009,
                    progress_tick=None,
                ): (include_quarterly, chain)
                for include_quarterly, chain in tasks
            }
            for future in as_completed(futures):
                ordered[futures[future]] = future.result()
                completed_iterations += iterations
                if progress_update is not None:
                    progress_update(completed_iterations)
        for include_quarterly, chain in tasks:
            result = ordered[(include_quarterly, chain)]
            for name, values in result.items():
                outputs[include_quarterly].setdefault(name, []).append(values)
    else:
        for include_quarterly in (False, True):
            for chain in range(chains):
                result = _run_variant_chain(
                    annual=np.asarray(annual, dtype=float),
                    quarterly=np.asarray(quarterly, dtype=float),
                    q_scale=q_scale,
                    e_scale=e_scale,
                    include_quarterly=include_quarterly,
                    spec=spec,
                    iterations=iterations,
                    warmup=warmup,
                    thin=thin,
                    seed=seed + (100003 if include_quarterly else 0) + chain * 1009,
                    progress_tick=progress_tick,
                )
                for name, values in result.items():
                    outputs[include_quarterly].setdefault(name, []).append(values)
    annual_draws = {name: np.stack(values) for name, values in outputs[False].items()}
    augmented_draws = {name: np.stack(values) for name, values in outputs[True].items()}
    sd_n = np.std(annual_draws["qhat"], axis=(0, 1), ddof=1)
    sd_c = np.std(augmented_draws["qhat"], axis=(0, 1), ddof=1)
    return MeasurementPosterior(
        draws=augmented_draws,
        annual_only_draws=annual_draws,
        information_ratio=float(np.median(sd_c) / np.median(sd_n)),
        periods=periods,
    )


def sample_annual_only_variant(
    annual: np.ndarray,
    *,
    q_scale: float,
    periods: tuple[str, ...],
    spec: MeasurementSpec,
    iterations: int,
    warmup: int,
    thin: int,
    chains: int,
    seed: int,
    parallel_chains: bool = True,
) -> MeasurementPosterior:
    """Annual-only quarterly state posterior under a frozen alternative law."""
    missing = np.full(np.asarray(annual).size, np.nan)
    ordered: dict[int, dict[str, np.ndarray]] = {}
    if parallel_chains:
        with ProcessPoolExecutor(max_workers=chains) as executor:
            futures = {
                executor.submit(
                    _run_variant_chain,
                    annual=np.asarray(annual, dtype=float),
                    quarterly=missing,
                    q_scale=q_scale,
                    e_scale=q_scale,
                    include_quarterly=False,
                    spec=spec,
                    iterations=iterations,
                    warmup=warmup,
                    thin=thin,
                    seed=seed + chain * 1009,
                    progress_tick=None,
                ): chain
                for chain in range(chains)
            }
            for future in as_completed(futures):
                ordered[futures[future]] = future.result()
    else:
        for chain in range(chains):
            ordered[chain] = _run_variant_chain(
                annual=np.asarray(annual, dtype=float),
                quarterly=missing,
                q_scale=q_scale,
                e_scale=q_scale,
                include_quarterly=False,
                spec=spec,
                iterations=iterations,
                warmup=warmup,
                thin=thin,
                seed=seed + chain * 1009,
                progress_tick=None,
            )
    names = ordered[0]
    draws = {name: np.stack([ordered[chain][name] for chain in range(chains)]) for name in names}
    return MeasurementPosterior(
        draws=draws,
        annual_only_draws=draws,
        information_ratio=float("nan"),
        periods=periods,
    )
