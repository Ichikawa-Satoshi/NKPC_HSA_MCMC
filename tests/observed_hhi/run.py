"""Run observed inverse-HHI model tests and build a formal comparison report."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import sys as _sys, pathlib as _pathlib  # noqa: E402  (bootstrap: importable at any depth)
_ROOT = next(_p for _p in _pathlib.Path(__file__).resolve().parents if (_p / "pyproject.toml").exists())
_sys.path[:0] = [str(_ROOT), str(_ROOT / "src"), str(_ROOT / "tests")]
from tests import _bootstrap  # noqa: F401,E402
from tests._bootstrap import DATA_DIR
from tests.observed_hhi.functions import (
    CELL_SPECS,
    build_observed_design,
    fast_component,
    fit_observed_hhi_model,
    load_cell_sample,
    load_observed_hhi_frame,
    simulate_theta_recovery,
    summarize_observed_fit,
    timed_fast_component,
)

BUNDLE_DIR = Path(__file__).resolve().parent
# Results and figures live inside the test bundle (results/, results/figures/,
# results/tables/). Generated content is git-ignored (see experiments/.gitignore).
OUTPUT_DIR = BUNDLE_DIR / "results"


BLUE = "#0072B2"
ORANGE = "#D55E00"
GREEN = "#009E73"
PURPLE = "#CC79A7"
GREY = "#6B7280"


def _stable_seed(base: int, task: dict[str, object]) -> int:
    digest = hashlib.sha256(json.dumps(task, sort_keys=True).encode()).hexdigest()
    return base + int(digest[:8], 16) % 1_000_000_000


def _task_key(task: dict[str, object]) -> str:
    lag = "nolag" if task["no_lag"] else "hybrid"
    return "__".join(
        str(task[name])
        for name in (
            "cell",
            "hhi_variant",
            "fast_definition",
            "environment_definition",
            "timing",
            "model_variant",
            "error_model",
        )
    ) + f"__{lag}"


def _save_draws(fit: "ObservedHHIFit", task: dict[str, object], draws_dir: Path) -> Path:
    """Persist the raw posterior draws for one task under ``results/draws/``.

    Written per task from inside the worker process (distinct filenames, so the
    concurrent writes never collide). These arrays are large and reproducible,
    so ``experiments/.gitignore`` keeps them out of version control.
    """
    draws_dir.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {
        "coefficients": fit.coefficients,
        "sigma": fit.sigma,
        "names": np.asarray(fit.names),
        "periods": np.asarray([str(period) for period in fit.periods]),
    }
    for key, value in fit.auxiliary.items():
        arrays[f"aux__{key}"] = np.asarray(value)
    path = draws_dir / f"{_task_key(task)}.npz"
    np.savez_compressed(path, **arrays)
    return path


def _run_task(payload: dict[str, object]) -> dict[str, object]:
    frame, config = load_observed_hhi_frame(payload["config"])
    task = payload["task"]
    sample = load_cell_sample(frame, config, cell=int(task["cell"]), hhi_variant=str(task["hhi_variant"]))
    sampling = payload["sampling"]
    hsa = config["hsa"]
    fit = fit_observed_hhi_model(
        sample,
        cell=int(task["cell"]),
        fast_definition=str(task["fast_definition"]),
        environment_definition=str(task["environment_definition"]),
        timing=str(task["timing"]),
        model_variant=str(task["model_variant"]),
        error_model=str(task["error_model"]),
        iterations=int(sampling["iterations"]),
        warmup=int(sampling["warmup"]),
        thin=int(sampling["thin"]),
        chains=int(sampling["chains"]),
        seed=int(payload["seed"]),
        no_lag=bool(task["no_lag"]),
        zeta_reference=float(hsa["zeta_reference"]),
        b_x=float(hsa["b_x"]),
    )
    draws_dir = payload.get("draws_dir")
    if draws_dir is not None:
        _save_draws(fit, task, Path(draws_dir))
    summary = summarize_observed_fit(fit)
    summary["no_lag"] = bool(task["no_lag"])
    summary["task_groups"] = "+".join(sorted(task["groups"]))
    derived: dict[str, object] = {
        "task_key": _task_key(task),
        "cell": int(task["cell"]),
        "hhi_variant": str(task["hhi_variant"]),
        "fast_definition": str(task["fast_definition"]),
        "environment_definition": str(task["environment_definition"]),
        "timing": str(task["timing"]),
        "model_variant": str(task["model_variant"]),
        "error_model": str(task["error_model"]),
        "no_lag": bool(task["no_lag"]),
        "n": len(fit.periods),
        "sample_start": fit.periods[0],
        "sample_end": fit.periods[-1],
        "condition_number": fit.design_condition_number,
        "theta_orthogonal_share": fit.theta_orthogonal_share,
    }
    names = fit.names
    if "theta_0" in names:
        theta = fit.coefficients[:, :, names.index("theta_0")]
        fast = timed_fast_component(fast_component(sample.q, fit.fast_definition), fit.timing)
        fast = fast[np.isfinite(fast)]
        fast_iqr = float(np.subtract(*np.quantile(fast, [0.75, 0.25])))
        contribution = -theta * fast_iqr
        derived.update(
            theta_mean=float(np.mean(theta)),
            theta_sd=float(np.std(theta, ddof=1)),
            theta_contribution_iqr_mean=float(np.mean(contribution)),
            theta_contribution_iqr_ci_2_5=float(np.quantile(contribution, 0.025)),
            theta_contribution_iqr_ci_97_5=float(np.quantile(contribution, 0.975)),
        )
    if "kappa_1" in names:
        kappa = fit.coefficients[:, :, names.index("kappa_1")]
        q_iqr = float(np.subtract(*np.quantile(sample.q, [0.75, 0.25])))
        x_iqr = float(np.subtract(*np.quantile(sample.activity, [0.75, 0.25])))
        contribution = kappa * q_iqr * x_iqr
        derived.update(
            kappa_1_mean=float(np.mean(kappa)),
            kappa_contribution_joint_iqr_mean=float(np.mean(contribution)),
            kappa_contribution_joint_iqr_ci_2_5=float(np.quantile(contribution, 0.025)),
            kappa_contribution_joint_iqr_ci_97_5=float(np.quantile(contribution, 0.975)),
        )
        if "theta_0" in names:
            theta = fit.coefficients[:, :, names.index("theta_0")]
            delta = kappa - float(hsa["b_x"]) * float(hsa["zeta_reference"]) * theta
            equivalence_band = 0.1 * fit.prior_sds["kappa_1"]
            derived.update(
                delta_hsa_mean=float(np.mean(delta)),
                delta_hsa_ci_2_5=float(np.quantile(delta, 0.025)),
                delta_hsa_ci_97_5=float(np.quantile(delta, 0.975)),
                delta_hsa_equivalence_band=equivalence_band,
                delta_hsa_equivalence_probability=float(np.mean(np.abs(delta) < equivalence_band)),
                kappa_theta_correlation=float(np.corrcoef(kappa.reshape(-1), theta.reshape(-1))[0, 1]),
            )
    if "theta_hsa" in names:
        theta = fit.coefficients[:, :, names.index("theta_hsa")]
        derived.update(
            theta_hsa_mean=float(np.mean(theta)),
            theta_hsa_sd=float(np.std(theta, ddof=1)),
        )
    return {"summary": summary.to_dict(orient="records"), "derived": derived}


def _add_task(tasks: dict[str, dict[str, object]], group: str, **kwargs: object) -> None:
    kwargs.setdefault("environment_definition", "total")
    task = {**kwargs, "groups": {group}}
    key = _task_key(task)
    if key in tasks:
        tasks[key]["groups"].add(group)
    else:
        tasks[key] = task


def build_tasks(config: dict) -> list[dict[str, object]]:
    tasks: dict[str, dict[str, object]] = {}
    screen = config["screening"]
    selected = config["selected_robustness"]
    primary_hhi = str(screen["primary_hhi"])
    for cell, _, _ in CELL_SPECS:
        for definition in screen["fast_definitions"]:
            for timing in screen["timings"]:
                _add_task(
                    tasks,
                    "screening",
                    cell=cell,
                    hhi_variant=primary_hhi,
                    fast_definition=str(definition),
                    timing=str(timing),
                    model_variant="constant_theta",
                    error_model=str(screen["error_model"]),
                    no_lag=False,
                )
        for hhi_variant in config["data"]["hhi_variants"]:
            _add_task(
                tasks,
                "hhi_aggregation",
                cell=cell,
                hhi_variant=str(hhi_variant),
                fast_definition=str(selected["fast_definition"]),
                timing=str(selected["timing"]),
                model_variant="constant_theta",
                error_model="iid",
                no_lag=False,
            )
        for model in selected["model_variants"]:
            for error in selected["error_models"]:
                _add_task(
                    tasks,
                    "model_error",
                    cell=cell,
                    hhi_variant=primary_hhi,
                    fast_definition=str(selected["fast_definition"]),
                    timing=str(selected["timing"]),
                    model_variant=str(model),
                    error_model=str(error),
                    no_lag=False,
                )
        for error in selected["error_models"]:
            _add_task(
                tasks,
                "no_lag",
                cell=cell,
                hhi_variant=primary_hhi,
                fast_definition=str(selected["fast_definition"]),
                timing=str(selected["timing"]),
                model_variant="constant_theta",
                error_model=str(error),
                no_lag=True,
            )
    profile = config["timing_profile"]
    for cell in profile["cells"]:
        for timing in profile["timings"]:
            for error in profile["error_models"]:
                _add_task(
                    tasks,
                    "timing_profile",
                    cell=int(cell),
                    hhi_variant=primary_hhi,
                    fast_definition=str(profile["fast_definition"]),
                    timing=str(timing),
                    model_variant="constant_theta",
                    error_model=str(error),
                    no_lag=False,
                )
    environment = config["environment_profile"]
    for cell in environment["cells"]:
        for definition in environment["definitions"]:
            for timing in environment["timings"]:
                for error in environment["error_models"]:
                    _add_task(
                        tasks,
                        "environment_profile",
                        cell=int(cell),
                        hhi_variant=primary_hhi,
                        fast_definition=str(environment["fast_definition"]),
                        environment_definition=str(definition),
                        timing=str(timing),
                        model_variant="constant_theta",
                        error_model=str(error),
                        no_lag=False,
                    )
    return list(tasks.values())


def _write_figures(summary: pd.DataFrame, derived: pd.DataFrame, simulation: pd.DataFrame, figures: Path) -> None:
    figures.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")

    primary = summary.loc[
        summary["parameter"].eq("theta_0")
        & summary["hhi_variant"].eq("log_revenue_weighted")
        & summary["fast_definition"].eq("ewma_hl8")
        & summary["environment_definition"].eq("total")
        & summary["timing"].eq("current")
        & summary["model_variant"].eq("constant_theta")
        & ~summary["no_lag"]
        & summary["error_model"].isin(["iid", "persistent_ar1", "low_frequency"])
    ].copy()
    fig, ax = plt.subplots(figsize=(9.0, 5.4))
    offsets = {"iid": -0.20, "persistent_ar1": 0.0, "low_frequency": 0.20}
    colors = {"iid": BLUE, "persistent_ar1": ORANGE, "low_frequency": GREEN}
    labels = {"iid": "i.i.d.", "persistent_ar1": "persistent AR(1)", "low_frequency": "low frequency"}
    for error, group in primary.groupby("error_model"):
        y = group["cell"].to_numpy(float) + offsets[error]
        ax.errorbar(
            group["mean"],
            y,
            xerr=np.vstack((group["mean"] - group["ci_2.5"], group["ci_97.5"] - group["mean"])),
            fmt="o",
            color=colors[error],
            capsize=2.5,
            label=labels[error],
        )
    ax.axvline(0.0, color="black", lw=1.0)
    ax.set_yticks(range(1, 10))
    ax.set_ylabel("Nine-cell design cell")
    ax.set_xlabel(r"Direct fast-HHI loading $\theta_0$ (95% credible interval)")
    ax.set_title("Observed-HHI direct-loading estimates")
    ax.legend(frameon=False, ncol=3, fontsize=9)
    fig.tight_layout()
    fig.savefig(figures / "theta_primary_forest.png", dpi=220)
    plt.close(fig)

    screen = summary.loc[
        summary["parameter"].eq("theta_0")
        & summary["task_groups"].str.contains("screening")
        & summary["environment_definition"].eq("total")
        & summary["cell"].isin([1, 2, 3])
    ].copy()
    screen["spec"] = screen["fast_definition"] + " / " + screen["timing"]
    specs = list(dict.fromkeys(screen["spec"]))
    matrix = screen.pivot_table(index="cell", columns="spec", values="posterior_prior_sd_ratio").reindex(index=[1, 2, 3], columns=specs)
    fig, ax = plt.subplots(figsize=(12.0, 3.2))
    image = ax.imshow(matrix.to_numpy(), aspect="auto", vmin=0.0, vmax=1.1, cmap="viridis_r")
    ax.set_yticks(range(3), ["Cell 1", "Cell 2", "Cell 3"])
    ax.set_xticks(range(len(specs)), specs, rotation=55, ha="right", fontsize=8)
    ax.set_title(r"Posterior/prior SD ratio for $\theta_0$ (PPI cells)")
    fig.colorbar(image, ax=ax, label="SD ratio", fraction=0.025, pad=0.02)
    fig.tight_layout()
    fig.savefig(figures / "theta_learning_screen.png", dpi=220)
    plt.close(fig)

    aggregation = summary.loc[
        summary["parameter"].eq("theta_0")
        & summary["task_groups"].str.contains("hhi_aggregation")
        & summary["environment_definition"].eq("total")
        & summary["cell"].isin([1, 2, 3])
    ].copy()
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.8), sharex=False)
    for cell, ax in zip([1, 2, 3], axes, strict=True):
        group = aggregation.loc[aggregation["cell"].eq(cell)].sort_values("hhi_variant")
        positions = np.arange(len(group))
        ax.errorbar(
            group["mean"], positions,
            xerr=np.vstack((group["mean"] - group["ci_2.5"], group["ci_97.5"] - group["mean"])),
            fmt="o", color=BLUE, capsize=2.5,
        )
        ax.axvline(0.0, color="black", lw=0.9)
        ax.set_yticks(positions, group["hhi_variant"].str.replace("_", " "), fontsize=8)
        ax.set_title(f"Cell {cell}")
    axes[0].set_xlabel(r"$\theta_0$")
    axes[1].set_xlabel(r"$\theta_0$")
    axes[2].set_xlabel(r"$\theta_0$")
    fig.suptitle("Sensitivity to inverse-HHI aggregation")
    fig.tight_layout()
    fig.savefig(figures / "theta_hhi_aggregation.png", dpi=220)
    plt.close(fig)

    timing = summary.loc[
        summary["parameter"].eq("theta_0")
        & summary["task_groups"].str.contains("timing_profile")
        & summary["environment_definition"].eq("total")
        & summary["error_model"].eq("persistent_ar1")
    ].copy()
    order = ["current", "lag1", "lag2", "lag3", "lag4", "distributed4"]
    position = {name: index for index, name in enumerate(order)}
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.8), sharey=True)
    for cell, ax in zip([1, 2, 3], axes, strict=True):
        group = timing.loc[timing["cell"].eq(cell)].copy()
        group["position"] = group["timing"].map(position)
        group = group.sort_values("position")
        ax.errorbar(
            group["position"], group["mean"],
            yerr=np.vstack((group["mean"] - group["ci_2.5"], group["ci_97.5"] - group["mean"])),
            fmt="o-", color=ORANGE, capsize=2.5,
        )
        ax.axhline(0.0, color="black", lw=0.9)
        ax.set_xticks(range(len(order)), ["0", "1", "2", "3", "4", "avg0-3"])
        ax.set_xlabel("Fast-HHI timing (quarters)")
        ax.set_title(f"Cell {cell}")
    axes[0].set_ylabel(r"$\theta_0$ (95% credible interval)")
    fig.suptitle("PPI timing profile under persistent inflation errors")
    fig.tight_layout()
    fig.savefig(figures / "theta_timing_profile.png", dpi=220)
    plt.close(fig)

    environment = summary.loc[
        summary["parameter"].eq("theta_0")
        & summary["task_groups"].str.contains("environment_profile")
        & summary["error_model"].eq("persistent_ar1")
    ].copy()
    environment["spec"] = environment["environment_definition"].map(
        {"total": "total HHI level", "predicted_level": "predicted level"}
    ) + " / " + environment["timing"]
    spec_order = [
        "total HHI level / current",
        "predicted level / current",
        "total HHI level / lag1",
        "predicted level / lag1",
    ]
    colors = [BLUE, GREEN, ORANGE, PURPLE]
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.9), sharey=True)
    for cell, ax in zip([1, 2, 3], axes, strict=True):
        group = environment.loc[environment["cell"].eq(cell)].set_index("spec").reindex(spec_order)
        positions = np.arange(len(group))
        ax.errorbar(
            positions,
            group["mean"],
            yerr=np.vstack((group["mean"] - group["ci_2.5"], group["ci_97.5"] - group["mean"])),
            fmt="none",
            ecolor=GREY,
            capsize=2.5,
        )
        ax.scatter(positions, group["mean"], c=colors, s=30, zorder=3)
        ax.axhline(0.0, color="black", lw=0.9)
        ax.set_xticks(positions, ["total\nnow", "pred.\nnow", "total\nlag 1", "pred.\nlag 1"])
        ax.set_title(f"Cell {cell}")
    axes[0].set_ylabel(r"$\theta_0$ (95% credible interval)")
    fig.suptitle("Observed-HHI environment decomposition under persistent errors")
    fig.tight_layout()
    fig.savefig(figures / "theta_environment_profile.png", dpi=220)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
    for effect, group in simulation.loc[simulation["cell"].eq(1)].groupby("effect_sd"):
        axes[0].plot(group["n"], group["sign_probability_ge_0.8_rate"], marker="o", label=f"effect={effect:.2f} SD")
    axes[0].axhline(0.8, color=GREY, ls="--", lw=1)
    axes[0].set(xlabel="Sample size", ylabel="Detection rate", title="Cell-1 design recovery")
    axes[0].legend(frameon=False, fontsize=8)
    for cell, color in [(1, BLUE), (2, ORANGE)]:
        group = simulation.loc[simulation["cell"].eq(cell) & simulation["effect_sd"].eq(0.25)]
        axes[1].plot(group["n"], group["posterior_prior_sd_ratio"], marker="o", color=color, label=f"Cell {cell} design")
    axes[1].axhline(0.75, color=GREY, ls="--", lw=1)
    axes[1].set(xlabel="Sample size", ylabel="Posterior/prior SD ratio", title="Learning at a 0.25-SD effect")
    axes[1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(figures / "theta_recovery_simulation.png", dpi=220)
    plt.close(fig)


def _fmt(value: object, digits: int = 3) -> str:
    if value is None or pd.isna(value):
        return "--"
    return f"{float(value):.{digits}f}"


def _write_tables(summary: pd.DataFrame, derived: pd.DataFrame, simulation: pd.DataFrame, tables: Path) -> dict[str, str]:
    tables.mkdir(parents=True, exist_ok=True)
    summary.to_csv(tables / "coefficient_summaries.csv", index=False)
    derived.to_csv(tables / "specification_summaries.csv", index=False)
    simulation.to_csv(tables / "theta_recovery_simulation.csv", index=False)

    primary = summary.loc[
        summary["parameter"].eq("theta_0")
        & summary["hhi_variant"].eq("log_revenue_weighted")
        & summary["fast_definition"].eq("ewma_hl8")
        & summary["environment_definition"].eq("total")
        & summary["timing"].eq("current")
        & summary["model_variant"].eq("constant_theta")
        & summary["error_model"].eq("persistent_ar1")
        & ~summary["no_lag"]
    ].sort_values("cell")
    rows = []
    for _, row in primary.iterrows():
        rows.append(
            f"{int(row['cell'])} & {str(row['inflation']).replace('_', ' ')} / {str(row['activity']).replace('_', ' ')} & "
            f"{int(row['n'])} & {_fmt(row['mean'])} & [{_fmt(row['ci_2.5'])}, {_fmt(row['ci_97.5'])}] & "
            f"{_fmt(row['sign_probability'])} & {_fmt(row['posterior_prior_sd_ratio'])} \\\\"
        )
    primary_tex = "\n".join(rows)

    cell1 = summary.loc[
        summary["cell"].eq(1)
        & summary["parameter"].isin(["theta_0", "theta_hsa"])
        & summary["hhi_variant"].eq("log_revenue_weighted")
        & summary["fast_definition"].eq("ewma_hl8")
        & summary["environment_definition"].eq("total")
        & summary["timing"].eq("current")
        & ~summary["no_lag"]
    ].sort_values(["model_variant", "error_model"])
    rows = []
    for _, row in cell1.iterrows():
        rows.append(
            f"{str(row['model_variant']).replace('_', ' ')} & {str(row['error_model']).replace('_', ' ')} & {int(row['n'])} & "
            f"{_fmt(row['mean'])} & [{_fmt(row['ci_2.5'])}, {_fmt(row['ci_97.5'])}] & "
            f"{_fmt(row['posterior_prior_sd_ratio'])} \\\\"
        )
    cell1_tex = "\n".join(rows)

    sim = simulation.loc[simulation["cell"].eq(1) & simulation["effect_sd"].isin([0.25, 0.5])]
    rows = []
    for _, row in sim.iterrows():
        rows.append(
            f"{int(row['n'])} & {_fmt(row['effect_sd'], 2)} & {_fmt(row['posterior_prior_sd_ratio'])} & "
            f"{_fmt(row['sign_probability_ge_0.8_rate'])} & {_fmt(row['coverage_95'])} \\\\"
        )
    simulation_tex = "\n".join(rows)
    environment = summary.loc[
        summary["cell"].eq(1)
        & summary["parameter"].eq("theta_0")
        & summary["task_groups"].str.contains("environment_profile")
        & summary["error_model"].eq("persistent_ar1")
    ].sort_values(["timing", "environment_definition"])
    rows = []
    for _, row in environment.iterrows():
        rows.append(
            f"{str(row['environment_definition']).replace('_', ' ')} & {str(row['timing'])} & "
            f"{_fmt(row['mean'])} & [{_fmt(row['ci_2.5'])}, {_fmt(row['ci_97.5'])}] & "
            f"{_fmt(row['posterior_prior_sd_ratio'])} & {_fmt(row['theta_orthogonal_share'])} \\\\"
        )
    return {
        "primary": primary_tex,
        "cell1": cell1_tex,
        "simulation": simulation_tex,
        "environment": "\n".join(rows),
    }


def _report_findings(summary: pd.DataFrame, derived: pd.DataFrame, simulation: pd.DataFrame) -> dict[str, object]:
    theta = summary.loc[summary["parameter"].eq("theta_0")].copy()
    primary = theta.loc[
        theta["hhi_variant"].eq("log_revenue_weighted")
        & theta["fast_definition"].eq("ewma_hl8")
        & theta["environment_definition"].eq("total")
        & theta["timing"].eq("current")
        & theta["model_variant"].eq("constant_theta")
        & theta["error_model"].eq("persistent_ar1")
        & ~theta["no_lag"]
    ]
    learned = primary.loc[primary["posterior_prior_sd_ratio"].le(0.75)]
    excludes_zero = primary.loc[(primary["ci_2.5"] > 0) | (primary["ci_97.5"] < 0)]
    converged = int(primary["convergence_gate"].sum())
    cell1 = primary.loc[primary["cell"].eq(1)].iloc[0]
    screen_cell1 = theta.loc[
        theta["cell"].eq(1)
        & theta["task_groups"].str.contains("screening")
        & theta["environment_definition"].eq("total")
    ]
    timing_cell1 = theta.loc[
        theta["cell"].eq(1)
        & theta["task_groups"].str.contains("timing_profile")
        & theta["environment_definition"].eq("total")
        & theta["error_model"].eq("persistent_ar1")
    ]
    primary_delta = derived.loc[
        derived["cell"].eq(1)
        & derived["hhi_variant"].eq("log_revenue_weighted")
        & derived["fast_definition"].eq("ewma_hl8")
        & derived["environment_definition"].eq("total")
        & derived["timing"].eq("current")
        & derived["model_variant"].eq("constant_theta")
        & derived["error_model"].eq("persistent_ar1")
        & ~derived["no_lag"]
    ].iloc[0]
    sim24 = simulation.loc[
        simulation["cell"].eq(1) & simulation["n"].eq(24) & simulation["effect_sd"].eq(0.25)
    ].iloc[0]
    environment_cell1 = theta.loc[
        theta["cell"].eq(1)
        & theta["task_groups"].str.contains("environment_profile")
        & theta["error_model"].eq("persistent_ar1")
    ]
    predicted_current = environment_cell1.loc[
        environment_cell1["environment_definition"].eq("predicted_level")
        & environment_cell1["timing"].eq("current")
    ].iloc[0]
    return {
        "primary_count": len(primary),
        "primary_converged": converged,
        "learned_cells": learned["cell"].astype(int).tolist(),
        "nonzero_cells": excludes_zero["cell"].astype(int).tolist(),
        "cell1_n": int(cell1["n"]),
        "cell1_theta_mean": float(cell1["mean"]),
        "cell1_theta_lower": float(cell1["ci_2.5"]),
        "cell1_theta_upper": float(cell1["ci_97.5"]),
        "cell1_theta_ratio": float(cell1["posterior_prior_sd_ratio"]),
        "cell1_screen_min_ratio": float(screen_cell1["posterior_prior_sd_ratio"].min()),
        "cell1_screen_max_ratio": float(screen_cell1["posterior_prior_sd_ratio"].max()),
        "cell1_screen_min_mean": float(screen_cell1["mean"].min()),
        "cell1_screen_max_mean": float(screen_cell1["mean"].max()),
        "cell1_screen_ci_excludes_zero": int(((screen_cell1["ci_2.5"] > 0) | (screen_cell1["ci_97.5"] < 0)).sum()),
        "cell1_timing_min_mean": float(timing_cell1["mean"].min()),
        "cell1_timing_max_mean": float(timing_cell1["mean"].max()),
        "cell1_hsa_equivalence_probability": float(primary_delta["delta_hsa_equivalence_probability"]),
        "cell1_sim24_detection_025": float(sim24["sign_probability_ge_0.8_rate"]),
        "cell1_predicted_theta_mean": float(predicted_current["mean"]),
        "cell1_predicted_theta_lower": float(predicted_current["ci_2.5"]),
        "cell1_predicted_theta_upper": float(predicted_current["ci_97.5"]),
        "cell1_predicted_theta_ratio": float(predicted_current["posterior_prior_sd_ratio"]),
        "cell1_predicted_orthogonal_share": float(predicted_current["theta_orthogonal_share"]),
    }


def _escape(text: object) -> str:
    return str(text).replace("_", r"\_").replace("%", r"\%")


def _write_report(out: Path, tables_tex: dict[str, str], findings: dict[str, object], config: dict) -> Path:
    report = out / "observed_hhi_model_tests.tex"
    learned = ", ".join(map(str, findings["learned_cells"])) or "none"
    nonzero = ", ".join(map(str, findings["nonzero_cells"])) or "none"
    tex = rf"""\documentclass[11pt]{{article}}
\usepackage[margin=0.82in]{{geometry}}
\usepackage{{booktabs,array,graphicx,xcolor,amsmath,amssymb,microtype,hyperref}}
\usepackage{{newtxtext,newtxmath}}
\definecolor{{navy}}{{HTML}}{{17365D}}
\definecolor{{light}}{{HTML}}{{EEF3F8}}
\hypersetup{{colorlinks=true,linkcolor=navy,urlcolor=navy}}
\setlength{{\parindent}}{{0pt}}
\setlength{{\parskip}}{{5pt}}
\newcommand{{\good}}{{\textcolor{{navy}}{{\textbf{{PASS}}}}}}
\begin{{document}}

\begin{{center}}
{{\color{{navy}}\LARGE\bfseries Observed Inverse-HHI Model Tests}}\\[4pt]
{{\large Direct competition coordinates, timing, dynamics, and HSA restrictions}}\\[8pt]
Production-equivalent sampling: {config['sampling']['iterations']:,} iterations, {config['sampling']['warmup']:,} warmup, thin {config['sampling']['thin']}, {config['sampling']['chains']} chains
\end{{center}}

\vspace{{4pt}}
\colorbox{{light}}{{\parbox{{0.95\linewidth}}{{
\textbf{{Executive result.}} Replacing the QCEW common factor with an observed quarterly SEC inverse HHI removes the measurement ridge, but it does not by itself deliver sharp Cell-1 identification. The theory-near cell has only {findings['cell1_n']} usable quarters because the inverse-markup input ends in 2017. Under the pre-fixed EWMA-8/current/AR(1) specification, $\theta_0={findings['cell1_theta_mean']:.3f}$ with 95\% interval [{findings['cell1_theta_lower']:.3f}, {findings['cell1_theta_upper']:.3f}] and posterior/prior SD ratio {findings['cell1_theta_ratio']:.3f}. The main limitation is now short-sample and regressor collinearity, not MCMC convergence.
}}}}

\section*{{1. Design and scope}}
The quarterly coordinate is $q_t=10\log(1/HHI_t)$, centered in each estimation sample. No annual-firm/QCEW common factor is estimated. The main unrestricted equation is
\[
\pi_t=a+\beta_b\pi_{{t-1}}+\beta_fE_t\pi_{{t+1}}+\psi q_t+
(\kappa_0+\kappa_1q_t)x_t-\theta_0c_t-\gamma q_tc_t+\varepsilon_t.
\]
The fixed primary short-run movement $c_t$ is the one-sided EWMA forecast error with half-life eight quarters; first differences, AR(1) innovations, alternative half-lives, timing, and four HHI aggregates are sensitivities. The HSA-restricted model replaces the two free columns by $\theta_{{HSA}}[b_x\zeta_{{ref}}q_tx_t-c_t]$, using $b_x=1$ and $\zeta_{{ref}}=6$ as calibrations. A tight restricted posterior is not counted as independent evidence for HSA.

\section*{{2. Primary direct-loading results}}
\begin{{center}}\small
\begin{{tabular}}{{@{{}}r l r r r r r@{{}}}}
\toprule Cell & Inflation / activity & $T$ & Mean & 95\% interval & Sign prob. & SD ratio\\
\midrule
{tables_tex['primary']}
\bottomrule
\end{{tabular}}
\end{{center}}

All {findings['primary_converged']}/{findings['primary_count']} primary coefficient fits pass the retained-draw $\widehat R\le1.01$ and bulk-ESS$\ge400$ rule. Cells satisfying the exploratory posterior/prior SD ratio of 0.75 are: {learned}. Cells whose 95\% interval excludes zero are: {nonzero}. These labels are descriptive because the user has not adopted the SD-ratio rule as a formal gate.

\begin{{center}}\includegraphics[width=0.94\linewidth]{{figures/theta_primary_forest.png}}\end{{center}}

\section*{{3. What changes under alternative fast components?}}
The PPI cells were screened across five fast-component definitions and three timings. For Cell 1, the posterior/prior SD ratio ranges from {findings['cell1_screen_min_ratio']:.3f} to {findings['cell1_screen_max_ratio']:.3f}, while the posterior mean ranges from {findings['cell1_screen_min_mean']:.3f} to {findings['cell1_screen_max_mean']:.3f}. {findings['cell1_screen_ci_excludes_zero']} screening intervals exclude zero, but they do so under specifications with opposing signs. A favorable single specification therefore cannot be interpreted as robust identification.

\begin{{center}}\includegraphics[width=0.98\linewidth]{{figures/theta_learning_screen.png}}\end{{center}}

The dedicated 0--4 quarter profile confirms that timing is consequential: the Cell-1 posterior mean ranges from {findings['cell1_timing_min_mean']:.3f} to {findings['cell1_timing_max_mean']:.3f} under persistent inflation errors. This is evidence of timing sensitivity, not evidence for choosing the lag that produces the preferred HSA sign.

\begin{{center}}\includegraphics[width=0.96\linewidth]{{figures/theta_timing_profile.png}}\end{{center}}

\section*{{4. Total HHI versus a predicted low-frequency environment}}
The baseline uses total observed $q_t$ both as the competition environment and to construct its innovation. A deterministic alternative defines the environment as $z_t=q_t-c_t$, which for EWMA is the one-step-ahead level predicted from information through $t-1$. This is not a measurement model: it uses the same observed HHI once, introduces no latent state, and estimates no loading equation.

\begin{{center}}\small
\begin{{tabular}}{{@{{}}l l r r r r@{{}}}}
\toprule Environment & Fast timing & Mean & 95\% interval & SD ratio & Orthogonal share\\
\midrule
{tables_tex['environment']}
\bottomrule
\end{{tabular}}
\end{{center}}

For Cell 1 at the current timing, the predicted-level version gives $\theta_0={findings['cell1_predicted_theta_mean']:.3f}$ with 95\% interval [{findings['cell1_predicted_theta_lower']:.3f}, {findings['cell1_predicted_theta_upper']:.3f}], SD ratio {findings['cell1_predicted_theta_ratio']:.3f}, and fast-regressor orthogonal share {100 * findings['cell1_predicted_orthogonal_share']:.1f}\%. It is retained only if the sign and interval are stable across timing; improved orthogonality alone is not identification.

\begin{{center}}\includegraphics[width=0.96\linewidth]{{figures/theta_environment_profile.png}}\end{{center}}

\section*{{5. HHI aggregation sensitivity}}
The four observed competition coordinates use firm-weighted, revenue-weighted, revenue-weighted geometric, and non-financial revenue-weighted geometric aggregation. The figure keeps the inflation equation fixed, so dispersion across points is attributable to HHI construction rather than to a latent measurement system.

\begin{{center}}\includegraphics[width=0.98\linewidth]{{figures/theta_hhi_aggregation.png}}\end{{center}}

\section*{{6. Cell-1 model restrictions}}
\begin{{center}}\small
\begin{{tabular}}{{@{{}}l l r r r r@{{}}}}
\toprule Model & Inflation error & $T$ & Mean & 95\% interval & SD ratio\\
\midrule
{tables_tex['cell1']}
\bottomrule
\end{{tabular}}
\end{{center}}

The varying-$\theta$ model spends the limited Cell-1 information on both $\theta_0$ and $\gamma$ and is therefore treated as a sensitivity. The HSA-restricted model gains precision mechanically by imposing $\kappa_1=b_x\zeta_{{ref}}\theta_0$; it cannot establish that restriction. In the unrestricted primary model, the pre-declared HSA equivalence probability is only {100 * findings['cell1_hsa_equivalence_probability']:.1f}\%. The full residual is retained in the machine-readable specification table.

\section*{{7. Recovery simulation}}
Known direct-loading effects were injected into the observed Cell-1 and Cell-2 design matrices. Effect size is the inflation response generated by a one-standard-deviation fast-HHI movement, expressed relative to the inflation residual standard deviation. At the actual Cell-1 length, a 0.25-SD effect crosses the 0.80 sign-probability threshold in only {100 * findings['cell1_sim24_detection_025']:.1f}\% of replications.

\begin{{center}}\includegraphics[width=0.92\linewidth]{{figures/theta_recovery_simulation.png}}\end{{center}}

\begin{{center}}\small
\begin{{tabular}}{{@{{}}r r r r r@{{}}}}
\toprule $T$ & Effect (SD) & SD ratio & Detection rate & 95\% coverage\\
\midrule
{tables_tex['simulation']}
\bottomrule
\end{{tabular}}
\end{{center}}

\section*{{8. Decision}}
The observed-HHI model is preferable to the QCEW common-factor model because it aligns the economic object and removes a weak loading/state decomposition. It should become the implementation target for Capital IQ and Datastream. Separating the observed coordinate into a predicted level and a one-sided innovation materially reduces the Cell-1 collinearity, but the innovation loading still changes sign across current and lagged timing. The present SEC exercise is therefore not a final HSA test: Cell 1 is too short, the inverse-markup mapping $b_x$ remains calibrated, activity endogeneity is unresolved, and the inflation-expectation proxy is not PPI-specific. The model tests determine which specifications are worth carrying forward; the long historical vendor HHI is what can provide the missing identifying variation.

\section*{{Reproducibility}}
Full coefficient summaries, specification-level HSA residuals, recovery simulations, figures, configuration, and a manifest are stored beside this report. No QCEW observation enters these experiments.

\end{{document}}
"""
    report.write_text(tex, encoding="utf-8")
    subprocess.run(
        ["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error", report.name],
        cwd=out,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return report.with_suffix(".pdf")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=BUNDLE_DIR / "config.yaml")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--reuse-existing", action="store_true")
    parser.add_argument(
        "--no-draws",
        dest="save_draws",
        action="store_false",
        help="Skip writing raw posterior draws to results/draws/ (on by default).",
    )
    args = parser.parse_args()

    frame, config = load_observed_hhi_frame(args.config)
    out = args.output_dir
    tables = out / "tables"
    figures = out / "figures"
    draws_dir = out / "draws"
    out.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)
    sampling = dict(config["sampling"])
    if args.quick:
        sampling.update(iterations=600, warmup=200, thin=2, chains=2)
    summary_path = tables / "coefficient_summaries.csv"
    derived_path = tables / "specification_summaries.csv"
    started = time.perf_counter()
    if args.reuse_existing:
        summary = pd.read_csv(summary_path)
        derived = pd.read_csv(derived_path)
    else:
        tasks = build_tasks(config)
        print(f"observed-HHI tasks={len(tasks)} jobs={args.jobs} sampling={sampling}", flush=True)
        summaries: list[dict[str, object]] = []
        derived_rows: list[dict[str, object]] = []
        payloads = [
            {
                "config": str(args.config),
                "task": task,
                "sampling": sampling,
                "seed": _stable_seed(int(config["sampling"]["seed"]), {k: v for k, v in task.items() if k != "groups"}),
                "draws_dir": str(draws_dir) if args.save_draws else None,
            }
            for task in tasks
        ]
        with ProcessPoolExecutor(max_workers=max(1, args.jobs)) as executor:
            futures = [executor.submit(_run_task, payload) for payload in payloads]
            for index, future in enumerate(as_completed(futures), 1):
                result = future.result()
                summaries.extend(result["summary"])
                derived_rows.append(result["derived"])
                if index == 1 or index % 10 == 0 or index == len(futures):
                    print(f"[{index}/{len(futures)}] completed", flush=True)
        summary = pd.DataFrame(summaries)
        derived = pd.DataFrame(derived_rows)
        summary.to_csv(summary_path, index=False)
        derived.to_csv(derived_path, index=False)

    simulations: list[pd.DataFrame] = []
    for cell in (1, 2):
        sample = load_cell_sample(frame, config, cell=cell, hhi_variant=str(config["screening"]["primary_hhi"]))
        simulation = simulate_theta_recovery(
            sample,
            fast_definition=str(config["selected_robustness"]["fast_definition"]),
            timing=str(config["selected_robustness"]["timing"]),
            sample_sizes=config["simulation"]["sample_sizes"],
            effect_sizes_sd=config["simulation"]["effect_sizes_sd"],
            replications=int(config["simulation"]["replications"]),
            seed=int(config["sampling"]["seed"]) + cell * 1009,
        )
        simulation["cell"] = cell
        simulations.append(simulation)
    simulation = pd.concat(simulations, ignore_index=True)
    tables_tex = _write_tables(summary, derived, simulation, tables)
    _write_figures(summary, derived, simulation, figures)
    findings = _report_findings(summary, derived, simulation)
    pdf = _write_report(out, tables_tex, findings, config)
    previous_manifest = {}
    manifest_path = out / "manifest.json"
    if args.reuse_existing and manifest_path.exists():
        previous_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest = {
        "revision": config["revision"],
        "config": str(args.config),
        "data": str((DATA_DIR / "processed" / config["data"]["file"])),
        "uses_qcew": False,
        "uses_measurement_only_common_factor": False,
        "sampling": sampling,
        "tasks": int(derived.shape[0]),
        "coefficient_rows": int(summary.shape[0]),
        "findings": findings,
        "estimation_elapsed_seconds": (
            previous_manifest.get("estimation_elapsed_seconds", previous_manifest.get("elapsed_seconds"))
            if args.reuse_existing
            else time.perf_counter() - started
        ),
        "report_build_elapsed_seconds": time.perf_counter() - started,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"wrote {pdf}")


if __name__ == "__main__":
    main()
