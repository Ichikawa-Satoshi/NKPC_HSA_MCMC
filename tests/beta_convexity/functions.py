"""Hybrid-NKPC adding-up / convexity restriction on (beta_b, beta_f).

This experiment addresses review item §4.2: the shared inflation regression puts
independent, zero-centred priors on the lagged-inflation weight ``beta_b`` and the
expectation weight ``beta_f`` — it imposes no hybrid-NKPC discipline. Here we
re-estimate the same modular-cut cells under two restrictions and measure how the
HSA slope ``delta`` and ``kappa_1`` move:

* ``convexity``  — both weights in [0, 1] (drawn by rejection from the exact
  conditional posterior);
* ``adding_up``  — ``beta_b + beta_f = 1`` imposed exactly by reparameterising
  ``beta_b = 1 - beta_f`` (and keeping ``beta_f`` in [0, 1]).

The shared design, priors, transforms and ``CellFit`` container are imported from
``nkpc_hsa.phillips`` — nothing here changes the unconstrained default estimator.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from nkpc_hsa.gibbs.common.joint_ffbs import force_pd
from nkpc_hsa.phillips.data import CELL_SPECS, DesignData, robust_scale
from nkpc_hsa.phillips.inflation import (
    CellFit,
    _prior_sds,
    _quarterly_design,
    _transformed_regression,
    coefficient_names,
)
from nkpc_hsa.phillips.state import MeasurementPosterior

CONSTRAINTS = ("baseline", "convexity", "adding_up")


def _posterior_moments(
    y: np.ndarray, X: np.ndarray, names: tuple[str, ...], prior_sds: dict[str, float], sigma2: float
) -> tuple[np.ndarray, np.ndarray]:
    prior_precision = np.diag([1.0 / prior_sds[name] ** 2 for name in names])
    precision = force_pd(prior_precision + X.T @ X / sigma2)
    covariance = force_pd(np.linalg.inv(precision))
    mean = covariance @ (X.T @ y / sigma2)
    return mean, covariance


def _draw_sigma2(rng: np.random.Generator, y: np.ndarray, X: np.ndarray, beta: np.ndarray) -> float:
    residual = y - X @ beta
    return float(1.0 / rng.gamma(3.0 + y.size / 2.0, 1.0 / (8.0 + np.dot(residual, residual) / 2.0)))


def _draw_box(
    rng: np.random.Generator,
    y: np.ndarray,
    X: np.ndarray,
    names: tuple[str, ...],
    prior_sds: dict[str, float],
    sigma2: float,
    box: dict[str, tuple[float, float]],
    *,
    max_tries: int = 400,
) -> tuple[np.ndarray, float, bool]:
    """Draw beta from the exact Gaussian conditional, rejecting box violations."""
    if y.size <= X.shape[1]:
        raise ValueError("The regression has no residual degrees of freedom.")
    mean, covariance = _posterior_moments(y, X, names, prior_sds, sigma2)
    index = {name: i for i, name in enumerate(names)}
    accepted = False
    beta = mean.copy()
    for _ in range(max_tries):
        candidate = rng.multivariate_normal(mean, covariance)
        if all(lo <= candidate[index[n]] <= hi for n, (lo, hi) in box.items() if n in index):
            beta, accepted = candidate, True
            break
    if not accepted:
        # Keep the chain moving on the rare all-reject step: clip the constrained
        # coordinates of the posterior mean. Tracked as a rejection so the report
        # can flag a binding restriction.
        for n, (lo, hi) in box.items():
            if n in index:
                beta[index[n]] = min(max(mean[index[n]], lo), hi)
    return beta, _draw_sigma2(rng, y, X, beta), accepted


def _draw_adding_up(
    rng: np.random.Generator,
    y: np.ndarray,
    X: np.ndarray,
    names: tuple[str, ...],
    prior_sds: dict[str, float],
    sigma2: float,
    *,
    max_tries: int = 400,
) -> tuple[np.ndarray, float, bool]:
    """Impose beta_b + beta_f = 1 exactly by substituting beta_b = 1 - beta_f.

    ``y - pi_lag = a + beta_f * (expectation - pi_lag) + (other terms)``, so the
    lag column is dropped and the expectation column becomes ``expectation - pi_lag``.
    ``beta_f`` is kept in [0, 1] (hence ``beta_b`` in [0, 1] too).
    """
    index = {name: i for i, name in enumerate(names)}
    if "beta_b" not in index or "beta_f" not in index:
        raise ValueError("adding_up needs both beta_b and beta_f in the design.")
    ib, if_ = index["beta_b"], index["beta_f"]
    y2 = y - X[:, ib]
    keep = [i for i in range(len(names)) if i != ib]
    reduced_names = tuple(names[i] for i in keep)
    X2 = X[:, keep].copy()
    jf = reduced_names.index("beta_f")
    X2[:, jf] = X[:, if_] - X[:, ib]
    reduced_priors = {n: prior_sds[n] for n in reduced_names}
    beta2, sigma2, accepted = _draw_box(
        rng, y2, X2, reduced_names, reduced_priors, sigma2, {"beta_f": (0.0, 1.0)}, max_tries=max_tries
    )
    full = np.zeros(len(names))
    for j, n in enumerate(reduced_names):
        full[index[n]] = beta2[j]
    full[ib] = 1.0 - full[if_]
    return full, sigma2, accepted


def fit_hybrid_restricted(
    data: DesignData,
    measurement: MeasurementPosterior,
    *,
    cell: int,
    model: str,
    q0: float,
    seed: int,
    constraint: str,
    price_override: str | None = None,
    activity_override: str | None = None,
    include_slow_level: bool = True,
) -> CellFit:
    """Fit one QoQ cell of the modular cut under a (beta_b, beta_f) restriction.

    ``constraint='baseline'`` reproduces the unconstrained draw; ``'convexity'`` and
    ``'adding_up'`` apply §4.2. Only the hybrid QoQ equation (E1/E2, with a lag) is
    supported, since the restriction is about the backward/forward split.
    """
    if constraint not in CONSTRAINTS:
        raise ValueError(f"Unknown constraint {constraint!r}.")
    if model not in {"E1", "E2"}:
        raise ValueError("The (beta_b, beta_f) restriction needs the hybrid E1/E2 equation.")
    spec = next(item for item in CELL_SPECS if int(item["cell"]) == cell)
    price = str(price_override or spec["inflation"])
    activity = str(activity_override or spec["activity"])
    y = data.quarterly[f"pi_{price}"].to_numpy(dtype=float)
    pi_lag = data.quarterly[f"pi_{price}_lag1"].to_numpy(dtype=float)
    expectation = data.quarterly["expectation"].to_numpy(dtype=float)
    x = data.quarterly[f"x_{activity}"].to_numpy(dtype=float)
    x_scale = robust_scale(x)
    qbar_draws = measurement.draws["qbar"]
    qhat_draws = measurement.draws["qhat"]
    chains, draws, _ = qbar_draws.shape
    names = coefficient_names(model, no_lag=False, include_slow_level=include_slow_level)
    priors = _prior_sds(names, q_scale=data.q_scale, x_scale=x_scale)
    coefficients = np.zeros((chains, draws, len(names)))
    sigma = np.zeros((chains, draws))
    acceptance = np.zeros((chains, draws))
    constraint_seed = {"baseline": 0, "convexity": 71, "adding_up": 131}[constraint]
    model_seed = {"E1": 23, "E2": 37}[model]
    n_endpoints = 0
    for chain in range(chains):
        rng = np.random.default_rng(seed + cell * 10007 + chain * 1009 + model_seed + constraint_seed)
        sigma2 = 4.0
        for draw in range(draws):
            X, built_names = _quarterly_design(
                pi_lag=pi_lag,
                expectation=expectation,
                x=x,
                qbar=qbar_draws[chain, draw],
                qhat=qhat_draws[chain, draw],
                q0=q0,
                model=model,
                no_lag=False,
                include_slow_level=include_slow_level,
            )
            if built_names != names:
                raise RuntimeError("Coefficient-name and design-column order diverged.")
            yt, Xt = _transformed_regression(y, X, "qoq", None)
            n_endpoints = int(yt.size)
            if constraint == "convexity":
                beta, sigma2, accepted = _draw_box(
                    rng, yt, Xt, names, priors, sigma2, {"beta_b": (0.0, 1.0), "beta_f": (0.0, 1.0)}
                )
            elif constraint == "adding_up":
                beta, sigma2, accepted = _draw_adding_up(rng, yt, Xt, names, priors, sigma2)
            else:  # baseline
                beta, sigma2, accepted = _draw_box(rng, yt, Xt, names, priors, sigma2, {})
            coefficients[chain, draw] = beta
            sigma[chain, draw] = np.sqrt(sigma2)
            acceptance[chain, draw] = float(accepted)
    return CellFit(
        cell=cell,
        inflation=price,
        activity=activity,
        model=model,
        transformation="qoq",
        coefficient_names=names,
        coefficients=coefficients,
        sigma=sigma,
        q0=q0,
        x_scale=x_scale,
        prior_sds=priors,
        n_endpoints=n_endpoints,
        expectation_status="proxy_conditioned",
        estimator=f"modular_cut_{constraint}",
        auxiliary_draws={"restriction_accepted": acceptance},
    )


def delta_summary(fit: CellFit, *, b_x: float, zeta_reference: float, qref_minus_q0: float = 0.0) -> dict[str, float]:
    """HSA slope delta = kappa_1 - b_x * zeta * theta_ref, summarised over draws."""
    index = {name: i for i, name in enumerate(fit.coefficient_names)}
    kappa = fit.coefficients[:, :, index["kappa_1"]]
    if "theta_0" in index:
        theta = fit.coefficients[:, :, index["theta_0"]]
        if "gamma" in index:
            theta = theta + fit.coefficients[:, :, index["gamma"]] * qref_minus_q0
        delta = kappa - b_x * zeta_reference * theta
    else:
        delta = kappa
    beta_b = fit.coefficients[:, :, index["beta_b"]]
    beta_f = fit.coefficients[:, :, index["beta_f"]]
    return {
        "beta_b_mean": float(np.mean(beta_b)),
        "beta_f_mean": float(np.mean(beta_f)),
        "beta_sum_mean": float(np.mean(beta_b + beta_f)),
        "kappa_1_mean": float(np.mean(kappa)),
        "delta_mean": float(np.mean(delta)),
        "delta_ci_2_5": float(np.quantile(delta, 0.025)),
        "delta_ci_97_5": float(np.quantile(delta, 0.975)),
        "restriction_binding_share": float(1.0 - np.mean(fit.auxiliary_draws["restriction_accepted"])),
    }
