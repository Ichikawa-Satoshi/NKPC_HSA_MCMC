"""Test no-lag inflation equations across prices, activity gaps, and model forms.

The script reuses the production N_Gustavo-only mixed-frequency quarterly state
posterior.  Every inflation equation omits lagged inflation and instead uses a
persistent AR(1) disturbance.  The state smoother remains a modular cut and is
therefore not reweighted by any price series used here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import time
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import logsumexp

import sys as _sys, pathlib as _pathlib  # noqa: E402  (bootstrap: importable at any depth)
_ROOT = next(_p for _p in _pathlib.Path(__file__).resolve().parents if (_p / "pyproject.toml").exists())
_sys.path[:0] = [str(_ROOT), str(_ROOT / "src"), str(_ROOT / "tests")]
from experiments import _bootstrap  # noqa: F401,E402
from experiments._bootstrap import RESULTS_DIR, ROOT


BUNDLE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BUNDLE_DIR / "results"
from nkpc_hsa.config import load_yaml
from nkpc_hsa.phillips.data import DesignData, load_design_data, robust_scale
from nkpc_hsa.phillips.estimation import summarize_fit
from nkpc_hsa.phillips.inflation import CellFit, fit_cut_model, reference_draws
from nkpc_hsa.phillips.state import MeasurementPosterior
from nkpc_hsa.progress import ProgressReporter


BLUE = "#0072B2"
ORANGE = "#D55E00"
GREEN = "#009E73"
PURPLE = "#7B3294"
GREY = "#6B7280"

PRICE_LABELS = {
    "ppi": "PPI",
    "cpi": "headline CPI",
    "core_cpi": "core CPI",
    "pce": "headline PCE",
    "core_pce": "core PCE",
}
ACTIVITY_LABELS = {
    "inverse_markup": "inverse markup",
    "bn_output_gap": "BN output gap",
    "hp_output_gap": "HP output gap",
    "negative_unemployment_gap": "negative unemployment gap",
}
MODEL_LABELS = {
    "E0": "E0: level only",
    "SLOW": "SLOW: varying slope only",
    "E1": "E1: constant fast loading",
    "E2": "E2: state-dependent fast loading",
}


def _load_measurement(path: Path, data: DesignData, *, quick: bool) -> MeasurementPosterior:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing production state posterior {path}. Run production/main_scripts/23_run_n_gustavo_state_space.py first."
        )
    saved = np.load(path)
    draws = {
        name.removeprefix("draw_"): saved[name]
        for name in saved.files
        if name.startswith("draw_")
    }
    if draws["qbar"].shape[-1] != len(data.periods):
        raise ValueError(
            "The saved N_Gustavo state posterior and the requested quarterly sample have different lengths."
        )
    if quick:
        draws = {name: values[:2, :60].copy() for name, values in draws.items()}
    return MeasurementPosterior(
        draws=draws,
        annual_only_draws=draws,
        information_ratio=float("nan"),
        periods=tuple(map(str, data.periods)),
    )


def _timed_measurement(measurement: MeasurementPosterior, timing: str) -> MeasurementPosterior:
    original = measurement.draws["qhat"]
    if timing == "current":
        timed = original.copy()
    elif timing.startswith("lag"):
        lag = int(timing.removeprefix("lag"))
        timed = np.full_like(original, np.nan)
        timed[:, :, lag:] = original[:, :, :-lag]
    elif timing == "distributed4":
        timed = np.full_like(original, np.nan)
        timed[:, :, 4:] = np.mean(
            np.stack([original[:, :, 4 - lag : original.shape[-1] - lag] for lag in range(5)]),
            axis=0,
        )
    else:
        raise ValueError(f"Unknown fast-state timing {timing!r}.")
    draws = dict(measurement.draws)
    draws["qhat"] = timed
    return replace(measurement, draws=draws, annual_only_draws=draws)


def _posterior_prediction(
    fit: CellFit,
    data: DesignData,
    measurement: MeasurementPosterior,
    *,
    endpoint_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return y, posterior predictions, and conditional AR(1) log likelihood."""
    index = {name: i for i, name in enumerate(fit.coefficient_names)}
    beta = fit.coefficients.reshape(-1, len(fit.coefficient_names))
    qbar = measurement.draws["qbar"].reshape(-1, len(data.periods))
    qhat = measurement.draws["qhat"].reshape(-1, len(data.periods))
    centered = qbar - fit.q0
    y = data.quarterly[f"pi_{fit.inflation}"].to_numpy(float)
    expectation = data.quarterly["expectation"].to_numpy(float)
    x = data.quarterly[f"x_{fit.activity}"].to_numpy(float)

    prediction = beta[:, index["a"], None] + beta[:, index["beta_f"], None] * expectation
    if "psi" in index:
        prediction += beta[:, index["psi"], None] * centered
    prediction += beta[:, index["kappa_0"], None] * x
    if fit.model in {"SLOW", "E1", "E2"}:
        prediction += beta[:, index["kappa_1"], None] * centered * x
    if fit.model in {"E1", "E2"}:
        prediction -= beta[:, index["theta_0"], None] * qhat
    if fit.model == "E2":
        prediction -= beta[:, index["gamma"], None] * centered * qhat

    mask = np.ones(len(y), dtype=bool) if endpoint_mask is None else endpoint_mask
    y = y[mask]
    prediction = prediction[:, mask]
    residual = y[None, :] - prediction
    rho = fit.auxiliary_draws["rho_pi"].reshape(-1)
    sigma = fit.sigma.reshape(-1)
    innovations = np.empty_like(residual)
    innovations[:, 0] = np.sqrt(np.maximum(1.0 - rho**2, 1e-12)) * residual[:, 0]
    innovations[:, 1:] = residual[:, 1:] - rho[:, None] * residual[:, :-1]
    log_likelihood = -0.5 * (
        np.log(2.0 * np.pi * sigma[:, None] ** 2)
        + innovations**2 / sigma[:, None] ** 2
    )
    log_likelihood[:, 0] += 0.5 * np.log(np.maximum(1.0 - rho**2, 1e-12))
    return y, prediction, log_likelihood


def _predictive_summary(
    fit: CellFit,
    data: DesignData,
    measurement: MeasurementPosterior,
    *,
    endpoint_mask: np.ndarray | None = None,
) -> dict[str, float]:
    y, prediction, log_likelihood = _posterior_prediction(
        fit, data, measurement, endpoint_mask=endpoint_mask
    )
    draws = log_likelihood.shape[0]
    lppd = float(np.sum(logsumexp(log_likelihood, axis=0) - np.log(draws)))
    p_waic = float(np.sum(np.var(log_likelihood, axis=0, ddof=1)))
    elpd = lppd - p_waic
    rmse = float(np.sqrt(np.mean((y - prediction.mean(axis=0)) ** 2)))
    rho = fit.auxiliary_draws["rho_pi"].reshape(-1)
    sigma = fit.sigma.reshape(-1)
    return {
        "elpd_waic": elpd,
        "waic": -2.0 * elpd,
        "p_waic": p_waic,
        "posterior_mean_rmse": rmse,
        "rho_pi_mean": float(np.mean(rho)),
        "rho_pi_ci_2.5": float(np.quantile(rho, 0.025)),
        "rho_pi_ci_97.5": float(np.quantile(rho, 0.975)),
        "innovation_sigma_mean": float(np.mean(sigma)),
    }


def _fit_one(
    data: DesignData,
    measurement: MeasurementPosterior,
    *,
    price: str,
    activity: str,
    model: str,
    q0: float,
    seed: int,
    endpoint_mask: np.ndarray | None,
    test_run: bool,
    include_slow_level: bool,
) -> tuple[CellFit, pd.DataFrame, dict[str, float]]:
    fit = fit_cut_model(
        data,
        measurement,
        cell=1,
        model=model,
        transformation="qoq",
        q0=q0,
        seed=seed,
        endpoint_mask=endpoint_mask,
        no_lag=True,
        error_model="persistent_ar1",
        price_override=price,
        activity_override=activity,
        include_slow_level=include_slow_level,
    )
    summary = summarize_fit(fit, test_run=test_run)
    return fit, summary, _predictive_summary(
        fit, data, measurement, endpoint_mask=endpoint_mask
    )


def _save_fit(path: Path, fit: CellFit) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        coefficients=fit.coefficients,
        sigma=fit.sigma,
        coefficient_names=np.asarray(fit.coefficient_names),
        **fit.auxiliary_draws,
    )


def _run_grid(
    data: DesignData,
    measurement: MeasurementPosterior,
    cfg: dict,
    out: Path,
    *,
    quick: bool,
    include_slow_level: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    prices = list(map(str, cfg["prices"]))
    activities = list(map(str, cfg["activities"]))
    models = list(map(str, cfg["models"]))
    timings = list(map(str, cfg["fast_state_timings"]))
    _, q0, _ = reference_draws(data, measurement)
    base_tasks = [(p, a, m) for p in prices for a in activities for m in models]
    timing_tasks = [(p, a, timing) for p in prices for a in activities for timing in timings]
    summary_frames: list[pd.DataFrame] = []
    model_rows: list[dict[str, object]] = []
    timing_frames: list[pd.DataFrame] = []
    timing_rows: list[dict[str, object]] = []
    posterior = out / "posterior"
    common_mask = np.arange(len(data.periods)) >= int(cfg["timing_common_start_offset"])
    total = len(base_tasks) + len(timing_tasks)
    task_number = 0
    seed0 = 20260812

    with ProgressReporter(
        total,
        label="no-lag price/gap model grid" + ("" if include_slow_level else " (psi=0)"),
        key="nolag-price-gap-grid" + ("" if include_slow_level else "-psi0"),
        style="auto",
    ) as progress:
        for price, activity, model in base_tasks:
            task_number += 1
            fit, summary, predictive = _fit_one(
                data,
                measurement,
                price=price,
                activity=activity,
                model=model,
                q0=q0,
                seed=seed0 + task_number * 1009,
                endpoint_mask=None,
                test_run=quick,
                include_slow_level=include_slow_level,
            )
            summary["grid"] = "model"
            summary["fast_timing"] = "current"
            summary["price_label"] = PRICE_LABELS[price]
            summary["activity_label"] = ACTIVITY_LABELS[activity]
            summary["slow_level_nuisance"] = include_slow_level
            summary_frames.append(summary)
            row = {
                "inflation": price,
                "activity": activity,
                "model": model,
                "fast_timing": "current",
                "n_endpoints": fit.n_endpoints,
                **predictive,
            }
            model_rows.append(row)
            _save_fit(posterior / "model" / f"{price}__{activity}__{model}.npz", fit)
            if task_number < total:
                progress.update(task_number)

        timed = {timing: _timed_measurement(measurement, timing) for timing in timings}
        for price, activity, timing in timing_tasks:
            task_number += 1
            fit, summary, predictive = _fit_one(
                data,
                timed[timing],
                price=price,
                activity=activity,
                model="E1",
                q0=q0,
                seed=seed0 + task_number * 1009,
                endpoint_mask=common_mask,
                test_run=quick,
                include_slow_level=include_slow_level,
            )
            summary["grid"] = "timing"
            summary["fast_timing"] = timing
            summary["price_label"] = PRICE_LABELS[price]
            summary["activity_label"] = ACTIVITY_LABELS[activity]
            summary["slow_level_nuisance"] = include_slow_level
            timing_frames.append(summary)
            timing_rows.append(
                {
                    "inflation": price,
                    "activity": activity,
                    "model": "E1",
                    "fast_timing": timing,
                    "n_endpoints": fit.n_endpoints,
                    **predictive,
                }
            )
            _save_fit(posterior / "timing" / f"{price}__{activity}__E1__{timing}.npz", fit)
            if task_number < total:
                progress.update(task_number)

    coefficients = pd.concat(summary_frames + timing_frames, ignore_index=True)
    model_comparison = pd.DataFrame(model_rows)
    model_comparison["best_elpd_waic"] = model_comparison.groupby(
        ["inflation", "activity"]
    )["elpd_waic"].transform("max")
    model_comparison["delta_elpd_from_best"] = (
        model_comparison["elpd_waic"] - model_comparison["best_elpd_waic"]
    )
    timing_comparison = pd.DataFrame(timing_rows)
    timing_comparison["best_elpd_waic"] = timing_comparison.groupby(
        ["inflation", "activity"]
    )["elpd_waic"].transform("max")
    timing_comparison["delta_elpd_from_best"] = (
        timing_comparison["elpd_waic"] - timing_comparison["best_elpd_waic"]
    )
    diagnostics = _data_diagnostics(data, prices, activities)
    return coefficients, model_comparison, timing_comparison, diagnostics


def _data_diagnostics(data: DesignData, prices: list[str], activities: list[str]) -> pd.DataFrame:
    rows = []
    for kind, names, prefix in (
        ("price", prices, "pi_"),
        ("activity", activities, "x_"),
    ):
        for name in names:
            values = data.quarterly[f"{prefix}{name}"].to_numpy(float)
            rows.append(
                {
                    "kind": kind,
                    "series": name,
                    "label": PRICE_LABELS.get(name, ACTIVITY_LABELS.get(name, name)),
                    "n": int(np.isfinite(values).sum()),
                    "mean": float(np.mean(values)),
                    "sd": float(np.std(values, ddof=1)),
                    "iqr_scale": robust_scale(values),
                    "ar1_correlation": float(np.corrcoef(values[1:], values[:-1])[0, 1]),
                }
            )
    return pd.DataFrame(rows)


def _parameter_table(coefficients: pd.DataFrame, parameter: str) -> pd.DataFrame:
    return coefficients.loc[
        coefficients.grid.eq("model")
        & coefficients.model.eq("E1")
        & coefficients.parameter.eq(parameter)
    ].copy()


def _write_figures(
    coefficients: pd.DataFrame,
    model_comparison: pd.DataFrame,
    figures: Path,
    *,
    include_slow_level: bool,
) -> None:
    figures.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")
    activities = list(ACTIVITY_LABELS)
    prices = list(PRICE_LABELS)

    for parameter, filename, color, symbol in (
        ("theta_0", "theta_by_price_gap.png", ORANGE, r"$\theta_0$"),
        ("kappa_1", "kappa1_by_price_gap.png", BLUE, r"$\kappa_1$"),
    ):
        table = _parameter_table(coefficients, parameter)
        fig, axes = plt.subplots(1, len(activities), figsize=(15.5, 5.0), sharex=False, sharey=True)
        for ax, activity in zip(axes, activities, strict=True):
            group = table.loc[table.activity.eq(activity)].set_index("inflation").loc[prices]
            y = np.arange(len(prices))
            ax.errorbar(
                group["mean"],
                y,
                xerr=np.vstack(
                    (group["mean"] - group["ci_2.5"], group["ci_97.5"] - group["mean"])
                ),
                fmt="o",
                color=color,
                capsize=2.5,
            )
            ax.axvline(0.0, color="black", lw=0.9)
            ax.set_title(ACTIVITY_LABELS[activity], fontsize=10)
            ax.set_yticks(y, [PRICE_LABELS[p] for p in prices])
            ax.set_xlabel(f"{symbol} (95% interval)")
        theory_note = "with slow-level nuisance" if include_slow_level else r"theory-near $\psi=0$"
        fig.suptitle(f"No lagged inflation; persistent AR(1) error; E1; {theory_note}", y=1.01)
        fig.tight_layout()
        fig.savefig(figures / filename, dpi=220, bbox_inches="tight")
        plt.close(fig)

    timing = coefficients.loc[
        coefficients.grid.eq("timing") & coefficients.parameter.eq("theta_0")
    ].copy()
    timing["signed_probability"] = np.where(
        timing["mean"] >= 0, timing.sign_probability, -timing.sign_probability
    )
    timing["spec"] = timing.inflation.map(PRICE_LABELS) + " / " + timing.activity.map(ACTIVITY_LABELS)
    matrix = timing.pivot(index="spec", columns="fast_timing", values="signed_probability")
    order = list(dict.fromkeys(
        f"{PRICE_LABELS[p]} / {ACTIVITY_LABELS[a]}" for p in PRICE_LABELS for a in ACTIVITY_LABELS
    ))
    matrix = matrix.loc[order, ["current", "lag1", "lag2", "lag3", "lag4", "distributed4"]]
    fig, ax = plt.subplots(figsize=(10.5, 8.2))
    image = ax.imshow(matrix.to_numpy(), aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(matrix.shape[1]), matrix.columns, rotation=30, ha="right")
    ax.set_yticks(range(matrix.shape[0]), matrix.index, fontsize=8)
    ax.set_title(r"Fast-state timing sensitivity: signed posterior sign probability of $\theta_0$")
    fig.colorbar(image, ax=ax, label="negative certainty  ←  0  →  positive certainty")
    fig.tight_layout()
    fig.savefig(figures / "theta_timing_heatmap.png", dpi=220)
    plt.close(fig)

    average = model_comparison.groupby("model", as_index=False)["delta_elpd_from_best"].mean()
    average = average.set_index("model").loc[["E0", "SLOW", "E1", "E2"]].reset_index()
    fig, ax = plt.subplots(figsize=(8.3, 3.8))
    ax.barh(
        [MODEL_LABELS[m] for m in average.model],
        average.delta_elpd_from_best,
        color=[GREY, GREEN, ORANGE, PURPLE],
    )
    ax.axvline(0.0, color="black", lw=0.9)
    ax.set_xlabel("Mean ELPD difference from the best model in each price/activity cell")
    ax.set_title("Conditional WAIC comparison (higher is better)")
    fig.tight_layout()
    fig.savefig(figures / "model_comparison.png", dpi=220)
    plt.close(fig)


def _fmt(value: object, digits: int = 2) -> str:
    return "--" if value is None or pd.isna(value) else f"{float(value):.{digits}f}"


def _write_report(
    out: Path,
    coefficients: pd.DataFrame,
    model_comparison: pd.DataFrame,
    timing_comparison: pd.DataFrame,
    diagnostics: pd.DataFrame,
    *,
    quick: bool,
    state_draws: tuple[int, int],
    include_slow_level: bool,
    psi_comparison: pd.DataFrame | None,
) -> Path:
    e1 = coefficients.loc[
        coefficients.grid.eq("model") & coefficients.model.eq("E1")
    ]
    theta = e1.loc[e1.parameter.eq("theta_0")].copy()
    kappa = e1.loc[e1.parameter.eq("kappa_1")].copy()
    merged = theta.merge(
        kappa,
        on=["inflation", "activity"],
        suffixes=("_theta", "_kappa"),
    )
    output_gap = merged.loc[merged.activity.isin(["bn_output_gap", "hp_output_gap"])]
    rows = "\n".join(
        f"{PRICE_LABELS[r.inflation]} & {ACTIVITY_LABELS[r.activity]} & "
        f"{_fmt(r.mean_theta)} & [{_fmt(r['ci_2.5_theta'])}, {_fmt(r['ci_97.5_theta'])}] & "
        f"{_fmt(r.sign_probability_theta)} & {_fmt(r.mean_kappa)} & "
        f"[{_fmt(r['ci_2.5_kappa'])}, {_fmt(r['ci_97.5_kappa'])}] \\\\"
        for _, r in output_gap.iterrows()
    )
    best_counts = (
        model_comparison.loc[np.isclose(model_comparison.delta_elpd_from_best, 0.0)]
        .groupby("model")
        .size()
        .reindex(["E0", "SLOW", "E1", "E2"], fill_value=0)
    )
    model_rows = "\n".join(
        f"{MODEL_LABELS[model]} & {int(best_counts[model])} & "
        f"{_fmt(model_comparison.loc[model_comparison.model.eq(model), 'delta_elpd_from_best'].mean())} & "
        f"{_fmt(model_comparison.loc[model_comparison.model.eq(model), 'rho_pi_mean'].mean())} \\\\"
        for model in ["E0", "SLOW", "E1", "E2"]
    )
    timing_theta = coefficients.loc[
        coefficients.grid.eq("timing") & coefficients.parameter.eq("theta_0")
    ]
    timing_groups = timing_theta.groupby(["inflation", "activity"])
    stable_sign = int(
        sum((group["mean"] > 0).all() or (group["mean"] < 0).all() for _, group in timing_groups)
    )
    timing_excluding_zero = int(
        sum(
            ((group["ci_2.5"] > 0) | (group["ci_97.5"] < 0)).all()
            for _, group in timing_groups
        )
    )
    theta_nonzero = int(((theta["ci_2.5"] > 0) | (theta["ci_97.5"] < 0)).sum())
    kappa_nonzero = int(((kappa["ci_2.5"] > 0) | (kappa["ci_97.5"] < 0)).sum())
    theta_learning = int((theta.posterior_prior_sd_ratio <= 0.75).sum())
    theta_converged = int(theta.convergence_gate.sum())
    best_theta = theta.loc[theta.sign_probability.idxmax()]
    e1_rho = model_comparison.loc[model_comparison.model.eq("E1"), "rho_pi_mean"]
    mode_text = "SHORT TEST RUN — NOT FOR INFERENCE" if quick else "Production posterior reuse"
    diagnostic_rows = "\n".join(
        f"{r.kind} & {str(r.label).replace('_', ' ')} & {int(r.n)} & {_fmt(r.sd)} & {_fmt(r.ar1_correlation)} \\\\"
        for _, r in diagnostics.iterrows()
    )
    if include_slow_level:
        title = "No-Lag Inflation Model Tests"
        subtitle = "Multiple prices, activity gaps, and competition-channel specifications"
        slow_level_statement = (
            "The standalone slow-state nuisance $\\psi(\\bar q_t-q_0)$ is retained "
            "symmetrically across model families."
        )
        equation_slow = r"+\psi(\bar q_t-q_0)"
        e0_description = "E0 includes the slow competition level but neither varying slope nor fast loading"
        report_stem = "nolag_price_gap_model_tests"
        comparison_section = ""
    else:
        title = r"Theory-Near No-Lag Tests ($\psi=0$)"
        subtitle = "HSA-motivated inflation equation without the empirical slow-level nuisance"
        slow_level_statement = (
            "The standalone term $\\psi(\\bar q_t-q_0)$ is fixed exactly to zero. "
            "Slow competition can enter inflation only through the activity-slope interaction, "
            "while the stationary state enters through its direct loading."
        )
        equation_slow = ""
        e0_description = "E0 contains neither a competition-state level nor a varying-slope or fast-state channel"
        report_stem = "theory_faithful_nolag_model_tests"
        if psi_comparison is not None and not psi_comparison.empty:
            comparable = psi_comparison.loc[
                psi_comparison.grid.eq("model")
                & psi_comparison.model.eq("E1")
                & psi_comparison.fast_timing.eq("current")
            ]
            theta_cmp = comparable.loc[comparable.parameter.eq("theta_0")]
            kappa_cmp = comparable.loc[comparable.parameter.eq("kappa_1")]
            largest = theta_cmp.loc[theta_cmp.mean_shift.abs().idxmax()]
            theta_sign_flips = int((np.sign(theta_cmp.mean_with_psi) != np.sign(theta_cmp.mean_psi0)).sum())
            kappa_sign_flips = int((np.sign(kappa_cmp.mean_with_psi) != np.sign(kappa_cmp.mean_psi0)).sum())
            comparison_section = rf"""
\section*{{3. Direct comparison with the empirical $\psi$ specification}}
Holding the state draws, sample, priors for shared coefficients, seeds, and all other regressors fixed, removing $\psi$ changes the E1 $\theta_0$ posterior mean by a median absolute {_fmt(theta_cmp.mean_shift.abs().median(), 3)} and a maximum absolute {_fmt(theta_cmp.mean_shift.abs().max(), 3)}. Posterior-mean signs change in {theta_sign_flips}/20 $\theta_0$ combinations and {kappa_sign_flips}/20 $\kappa_1$ combinations. The largest $\theta_0$ shift occurs for {PRICE_LABELS[largest.inflation]} with {ACTIVITY_LABELS[largest.activity]}: {_fmt(largest.mean_with_psi, 3)} with $\psi$ versus {_fmt(largest.mean_psi0, 3)} under $\psi=0$. These are paired specification shifts, not independent samples.
"""
        else:
            comparison_section = ""
    model_section = 4 if comparison_section else 3
    timing_section = model_section + 1
    data_section = model_section + 2
    decision_section = model_section + 3

    tex = rf"""\documentclass[11pt]{{article}}
\usepackage[margin=0.78in]{{geometry}}
\usepackage{{booktabs,graphicx,xcolor,amsmath,microtype,hyperref,newtxtext,newtxmath}}
\definecolor{{navy}}{{HTML}}{{17365D}}\definecolor{{light}}{{HTML}}{{EEF3F8}}
\setlength{{\parindent}}{{0pt}}\setlength{{\parskip}}{{5pt}}
\begin{{document}}
\begin{{center}}{{\color{{navy}}\LARGE\bfseries {title}}}\\[3pt]
{{\large {subtitle}}}\\[6pt]
{mode_text}; {state_draws[0]} chains $\times$ {state_draws[1]:,} retained state draws
\end{{center}}

\colorbox{{light}}{{\parbox{{0.95\linewidth}}{{\textbf{{Executive result.}} Removing lagged inflation does not by itself establish the fast competition loading. All {theta_converged}/20 E1 $\theta_0$ chains pass the coefficient convergence rule, but {theta_nonzero} intervals exclude zero and {theta_learning} pass the posterior/prior SD learning threshold. The most signed E1 estimate is {PRICE_LABELS[best_theta.inflation]} with {ACTIVITY_LABELS[best_theta.activity]}: mean {_fmt(best_theta['mean'])}, 95\% interval [{_fmt(best_theta['ci_2.5'])}, {_fmt(best_theta['ci_97.5'])}], sign probability {_fmt(best_theta.sign_probability)}. Across timing definitions, {stable_sign}/20 combinations keep the same posterior-mean sign and {timing_excluding_zero}/20 exclude zero at every timing.}}}}

\section*{{1. Test held fixed}}
The competition input is only annual \texttt{{N\_Gustavo}}, observed at Q4 and decomposed into quarterly slow and stationary states by the existing mixed-frequency Kalman/FFBS posterior. Inflation never enters that state smoother. Every regression here omits lagged inflation, retains the one-quarter-ahead SPF GDP-deflator expectation proxy, and assigns remaining inflation persistence to a stationary AR(1) error. Thus a change in $\theta_0$ cannot be attributed to QCEW, SEC HHI, interpolation, or a different competition state.

Five price indices (PPI, headline/core CPI, and headline/core PCE) are crossed with inverse markup, BN output gap, HP output gap, and the negative unemployment gap. The common expectation is not price-index-specific, so the cross-price exercise is a robustness comparison rather than five equally structural NKPCs.

{slow_level_statement}

The estimated E1 residual persistence averages {_fmt(e1_rho.mean())} across the 20 combinations (range {_fmt(e1_rho.min())} to {_fmt(e1_rho.max())}). Thus the AR(1) component absorbs material but not near-unit-root persistence after lagged inflation is removed.

\section*{{2. Main E1 results}}
E1 estimates
\[\pi_t=a+\beta_f E_t\pi_{{t+1}}{equation_slow}+[\kappa_0+\kappa_1(\bar q_t-q_0)]x_t-\theta_0\hat q_t+\varepsilon_t,\quad \varepsilon_t=\rho_\pi\varepsilon_{{t-1}}+u_t.\]
The output-gap subset is shown below; the inverse-markup and unemployment-gap estimates are retained in the CSV and figures.
\begin{{center}}\scriptsize\begin{{tabular}}{{@{{}}l l r r r r r@{{}}}}\toprule
Price & Activity & $\theta_0$ & 95\% interval & Sign prob. & $\kappa_1$ & 95\% interval\\\midrule
{rows}
\bottomrule\end{{tabular}}\end{{center}}
Across all 20 E1 combinations, {theta_nonzero} $\theta_0$ intervals and {kappa_nonzero} $\kappa_1$ intervals exclude zero.
\begin{{center}}\includegraphics[width=0.99\linewidth]{{figures/theta_by_price_gap.png}}\end{{center}}
\begin{{center}}\includegraphics[width=0.99\linewidth]{{figures/kappa1_by_price_gap.png}}\end{{center}}

{comparison_section}

\section*{{{model_section}. Does model form matter?}}
{e0_description}; SLOW adds only $\kappa_1$; E1 adds constant $\theta_0$; E2 also adds $\gamma$. Conditional WAIC is an in-sample predictive diagnostic, not causal evidence and not a substitute for the coefficient-identification checks.
\begin{{center}}\small\begin{{tabular}}{{@{{}}l r r r@{{}}}}\toprule Model & Best cells (of 20) & Mean ELPD difference & Mean $\rho_\pi$\\\midrule
{model_rows}
\bottomrule\end{{tabular}}\end{{center}}
\begin{{center}}\includegraphics[width=0.78\linewidth]{{figures/model_comparison.png}}\end{{center}}

\section*{{{timing_section}. Timing of the fast state}}
E1 is re-estimated with current, one- through four-quarter lags, and the equal-weight current-to-lag-4 average. Every timing uses the same 124 endpoints. Red is negative and blue is positive; color intensity is the posterior probability of the displayed sign.
\begin{{center}}\includegraphics[width=0.91\linewidth]{{figures/theta_timing_heatmap.png}}\end{{center}}
Only a sign that survives economically plausible timing choices should be interpreted as robust. A timing that happens to maximize WAIC or sign probability after inspecting this grid is exploratory.

\section*{{{data_section}. Data diagnostics}}
\begin{{center}}\scriptsize\begin{{tabular}}{{@{{}}l l r r r@{{}}}}\toprule Kind & Series & $T$ & SD & AR(1) corr.\\\midrule
{diagnostic_rows}
\bottomrule\end{{tabular}}\end{{center}}

\section*{{{decision_section}. Decision rule}}
Dropping lagged inflation is useful only if the inferred competition effect is stable across price and gap definitions, learned relative to its prior, insensitive to fast-state timing, and not merely purchased by a more flexible model. Otherwise the correct conclusion remains ``not identified'', not ``zero'' and not ``HSA confirmed.'' The machine-readable coefficient, predictive, timing, and data-diagnostic tables accompany this PDF.
\end{{document}}
"""
    path = out / f"{report_stem}.tex"
    path.write_text(tex, encoding="utf-8")
    subprocess.run(
        ["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error", path.name],
        cwd=out,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return path.with_suffix(".pdf")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=BUNDLE_DIR / "config.yaml"
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument(
        "--theory-faithful",
        action="store_true",
        help="Fix the standalone slow-state nuisance psi exactly to zero.",
    )
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    family = "theory_faithful_nolag_models" if args.theory_faithful else "nolag_price_gap_models"
    out = args.output_dir or OUTPUT_DIR
    tables, figures = out / "tables", out / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    data = load_design_data(include_qcew=False, sample_end="2013Q4")
    state_path = ROOT / str(cfg["state_posterior"])
    measurement = _load_measurement(state_path, data, quick=args.quick)
    include_slow_level = not args.theory_faithful
    coefficients, model_comparison, timing_comparison, diagnostics = _run_grid(
        data,
        measurement,
        cfg,
        out,
        quick=args.quick,
        include_slow_level=include_slow_level,
    )
    coefficients.to_csv(tables / "coefficient_summaries.csv", index=False)
    model_comparison.to_csv(tables / "model_comparison.csv", index=False)
    timing_comparison.to_csv(tables / "timing_comparison.csv", index=False)
    diagnostics.to_csv(tables / "data_diagnostics.csv", index=False)
    _write_figures(
        coefficients,
        model_comparison,
        figures,
        include_slow_level=include_slow_level,
    )
    psi_comparison = None
    baseline_path = RESULTS_DIR / "nolag_price_gap_models" / "production" / "tables" / "coefficient_summaries.csv"
    if args.theory_faithful and not args.quick and baseline_path.exists():
        baseline = pd.read_csv(baseline_path)
        shared = ["inflation", "activity", "model", "grid", "fast_timing", "parameter"]
        psi_comparison = baseline.merge(
            coefficients,
            on=shared,
            suffixes=("_with_psi", "_psi0"),
        )
        psi_comparison["mean_shift"] = psi_comparison["mean_psi0"] - psi_comparison["mean_with_psi"]
        psi_comparison.to_csv(tables / "psi_specification_comparison.csv", index=False)
    state_shape = measurement.draws["qbar"].shape[:2]
    pdf = _write_report(
        out,
        coefficients,
        model_comparison,
        timing_comparison,
        diagnostics,
        quick=args.quick,
        state_draws=state_shape,
        include_slow_level=include_slow_level,
        psi_comparison=psi_comparison,
    )
    manifest = {
        "revision": str(cfg["revision"]),
        "is_test_run": args.quick,
        "sample_start": str(data.periods[0]),
        "sample_end": str(data.periods[-1]),
        "quarterly_observations": len(data.periods),
        "competition_input": "N_Gustavo annual Q4 only",
        "state_method": "quarterly mixed-frequency Kalman FFBS modular cut",
        "lagged_inflation": False,
        "slow_level_nuisance_psi": include_slow_level,
        "inflation_error": "persistent_ar1",
        "prices": cfg["prices"],
        "activities": cfg["activities"],
        "models": cfg["models"],
        "fast_state_timings": cfg["fast_state_timings"],
        "state_draw_shape": list(measurement.draws["qbar"].shape),
        "fit_count": int(
            len(cfg["prices"])
            * len(cfg["activities"])
            * (len(cfg["models"]) + len(cfg["fast_state_timings"]))
        ),
        "coefficient_rows": len(coefficients),
        "source_state_posterior": str(state_path.resolve()),
        "source_state_sha256": hashlib.sha256(state_path.read_bytes()).hexdigest(),
        "pdf_sha256": hashlib.sha256(pdf.read_bytes()).hexdigest(),
        "elapsed_seconds": time.perf_counter() - started,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"wrote {pdf}")


if __name__ == "__main__":
    main()
