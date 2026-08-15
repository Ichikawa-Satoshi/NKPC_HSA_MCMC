from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import arviz as az
import numpy as np
import pandas as pd

from nkpc_hsa.paths import results_root
from nkpc_hsa.progress import ProgressReporter
from nkpc_hsa.provenance import stamp_artifact_metadata

from .data import CELL_SPECS, DesignData, load_design_data
from .inflation import BenchmarkResult, CellFit, fit_cut_model, fit_hsa_benchmark, reference_draws
from .joint import JointCellFit, fit_joint_qoq_e2
from .state import MeasurementPosterior, sample_measurement_posterior


@dataclass(frozen=True)
class DesignRun:
    data: DesignData
    measurement: MeasurementPosterior
    cut_fits: dict[str, CellFit]
    joint_fits: dict[str, JointCellFit]
    robustness_fits: dict[str, CellFit]
    benchmark: BenchmarkResult
    q0: float
    output_dir: Path
    is_test: bool


def fit_key(fit: CellFit) -> str:
    return f"cell{fit.cell}_{fit.transformation}_{fit.model}_{fit.estimator}"


def _parameter_diagnostics(values: np.ndarray) -> tuple[float, float]:
    try:
        idata = az.from_dict({"posterior": {"value": values}})
        rhat = float(np.asarray(az.rhat(idata, var_names=["value"], method="rank")["value"]))
        ess = float(np.asarray(az.ess(idata, var_names=["value"], method="bulk")["value"]))
        return rhat, ess
    except Exception:
        return float("nan"), float("nan")


def summarize_fit(fit: CellFit, *, test_run: bool) -> pd.DataFrame:
    rows = []
    for index, name in enumerate(fit.coefficient_names):
        values = fit.coefficients[:, :, index]
        flat = values.reshape(-1)
        positive = float(np.mean(flat > 0.0))
        negative = float(np.mean(flat < 0.0))
        rhat, ess = _parameter_diagnostics(values)
        post_sd = float(np.std(flat, ddof=1))
        ratio = post_sd / fit.prior_sds[name]
        rows.append(
            {
                "cell": fit.cell,
                "inflation": fit.inflation,
                "activity": fit.activity,
                "model": fit.model,
                "transformation": fit.transformation,
                "estimator": fit.estimator,
                "parameter": name,
                "mean": float(np.mean(flat)),
                "median": float(np.median(flat)),
                "sd": post_sd,
                "ci_2.5": float(np.quantile(flat, 0.025)),
                "ci_97.5": float(np.quantile(flat, 0.975)),
                "p_positive": positive,
                "p_negative": negative,
                "sign_probability": max(positive, negative),
                "posterior_prior_sd_ratio": ratio,
                "learning_gate": bool(ratio <= 0.75 and not test_run),
                "rhat": rhat,
                "bulk_ess": ess,
                "convergence_gate": bool(rhat <= 1.01 and ess >= 400 and not test_run),
                "n_endpoints": fit.n_endpoints,
                "expectation_status": fit.expectation_status,
                "test_run": test_run,
            }
        )
    return pd.DataFrame(rows)


def _save_fit(path: Path, fit: CellFit, *, extra_arrays: dict[str, np.ndarray] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, Any] = {
        "coefficients": fit.coefficients,
        "sigma": fit.sigma,
        "coefficient_names": np.asarray(fit.coefficient_names),
    }
    arrays.update(fit.auxiliary_draws)
    arrays.update(extra_arrays or {})
    np.savez_compressed(path, **arrays)


def _benchmark_summary(benchmark: BenchmarkResult) -> dict[str, Any]:
    out: dict[str, Any] = {
        "status": benchmark.status,
        "reason": benchmark.reason,
        "local_endpoints": int(benchmark.local_endpoints.sum()),
        "equivalence_band": benchmark.equivalence_band,
        "equivalence_probability": benchmark.equivalence_probability,
        "zeta_reference": benchmark.zeta_reference,
        "b_x": benchmark.b_x,
    }
    if benchmark.delta_hsa is not None:
        values = benchmark.delta_hsa.reshape(-1)
        out.update(
            delta_mean=float(np.mean(values)),
            delta_median=float(np.median(values)),
            delta_ci_2_5=float(np.quantile(values, 0.025)),
            delta_ci_97_5=float(np.quantile(values, 0.975)),
        )
    return out


def _effect_table(data: DesignData, fits: dict[str, CellFit]) -> pd.DataFrame:
    q_iqr = float(np.subtract(*np.quantile(data.annual_values, [0.75, 0.25])))
    rows = []
    for fit in fits.values():
        if not (fit.model == "E2" and fit.transformation == "qoq" and fit.estimator == "modular_cut"):
            continue
        names = {name: i for i, name in enumerate(fit.coefficient_names)}
        x = data.quarterly[f"x_{fit.activity}"].to_numpy(float)
        x_iqr = float(np.subtract(*np.quantile(x, [0.75, 0.25])))
        values = fit.coefficients[:, :, names["kappa_1"]] * q_iqr * x_iqr
        rows.append(
            {
                "cell": fit.cell,
                "inflation": fit.inflation,
                "activity": fit.activity,
                "delta_q_iqr": q_iqr,
                "delta_x_iqr": x_iqr,
                "slope_effect_mean_pp": float(values.mean()),
                "slope_effect_ci_2.5_pp": float(np.quantile(values, 0.025)),
                "slope_effect_ci_97.5_pp": float(np.quantile(values, 0.975)),
            }
        )
    return pd.DataFrame(rows)


def _cut_joint_table(cut: dict[str, CellFit], joint: dict[str, JointCellFit]) -> pd.DataFrame:
    rows = []
    for cell in range(1, 10):
        cut_fit = cut[f"cell{cell}_qoq_E2_modular_cut"]
        joint_fit = joint[f"cell{cell}_qoq_E2_full_joint"].fit
        ci = {name: i for i, name in enumerate(cut_fit.coefficient_names)}
        ji = {name: i for i, name in enumerate(joint_fit.coefficient_names)}
        for parameter in ("kappa_1", "theta_0", "gamma"):
            c = cut_fit.coefficients[:, :, ci[parameter]].reshape(-1)
            j = joint_fit.coefficients[:, :, ji[parameter]].reshape(-1)
            shift = abs(float(j.mean() - c.mean())) / float(np.std(c, ddof=1))
            rows.append(
                {
                    "cell": cell,
                    "parameter": parameter,
                    "cut_mean": float(c.mean()),
                    "joint_mean": float(j.mean()),
                    "cut_sd_shift": shift,
                    "sign_crosses_half": bool((np.mean(c > 0) - 0.5) * (np.mean(j > 0) - 0.5) < 0),
                    "conflict": bool(shift > 0.5 or (np.mean(c > 0) - 0.5) * (np.mean(j > 0) - 0.5) < 0),
                }
            )
    return pd.DataFrame(rows)


def save_design_run(run: DesignRun) -> None:
    out = run.output_dir
    (out / "posterior").mkdir(parents=True, exist_ok=True)
    (out / "tables").mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out / "posterior" / "measurement_cut.npz",
        periods=np.asarray(run.measurement.periods),
        information_ratio=run.measurement.information_ratio,
        **{f"C_{k}": v for k, v in run.measurement.draws.items()},
        **{f"N_{k}": v for k, v in run.measurement.annual_only_draws.items()},
    )
    summaries = []
    for key, fit in run.cut_fits.items():
        _save_fit(out / "posterior" / f"{key}.npz", fit)
        summaries.append(summarize_fit(fit, test_run=run.is_test))
    for key, result in run.joint_fits.items():
        _save_fit(
            out / "posterior" / f"{key}.npz",
            result.fit,
            extra_arrays={"qbar": result.qbar, "qhat": result.qhat},
        )
        summaries.append(summarize_fit(result.fit, test_run=run.is_test))
    for key, fit in run.robustness_fits.items():
        _save_fit(out / "posterior" / f"{key}.npz", fit)
        frame = summarize_fit(fit, test_run=run.is_test)
        frame["robustness"] = key.rsplit("_", 1)[-1]
        summaries.append(frame)
    pd.concat(summaries, ignore_index=True).to_csv(out / "tables" / "coefficient_summaries.csv", index=False)
    _effect_table(run.data, run.cut_fits).to_csv(out / "tables" / "economic_effects.csv", index=False)
    _cut_joint_table(run.cut_fits, run.joint_fits).to_csv(out / "tables" / "cut_joint_comparison.csv", index=False)
    np.savez_compressed(
        out / "posterior" / "hsa_benchmark.npz",
        local_probability=run.benchmark.local_probability,
        local_endpoints=run.benchmark.local_endpoints,
        q_reference_draws=run.benchmark.q_reference_draws,
        delta_hsa=np.asarray([]) if run.benchmark.delta_hsa is None else run.benchmark.delta_hsa,
        theta_reference=np.asarray([]) if run.benchmark.theta_reference is None else run.benchmark.theta_reference,
    )
    (out / "tables" / "hsa_benchmark.json").write_text(
        json.dumps(_benchmark_summary(run.benchmark), indent=2), encoding="utf-8"
    )
    skipped = "test_skipped" if run.is_test else "not_implemented"
    compliance = pd.DataFrame(
        [
            ("nine_cells", "Nine frozen price-activity cells", "executed", "Common 1982Q1--2012Q4 sample"),
            ("qoq_yoy", "QoQ and exact A4/G4 YoY for every cell", "executed", "QoQ endpoint match also executed"),
            ("cut", "Primary measurement-only modular cut", "executed", "Inflation cannot reweight the primary state"),
            ("joint", "Secondary full joint", "partial", "QoQ E2 implemented; YoY J-Q remains outstanding"),
            ("models", "E0/E1/E2 hierarchy", "executed", "Cut QoQ and YoY"),
            ("measurement_n_c", "Annual-only N versus quarterly-augmented C", "executed", f"R_q={run.measurement.information_ratio:.3f}"),
            ("fast_by_fast", "Free fast-by-fast interaction", "executed", "All nine QoQ E2 cells"),
            ("cs1_cs2", "Correlated competition-shock sensitivities", "partial", "Cut regressions executed; joint state feedback outstanding"),
            ("single_coordinate", "Single-coordinate local HSA benchmark", "executed", run.benchmark.status),
            ("measurement_error_corr", "Correlated quarterly measurement errors", "not_applicable", "J_E=1"),
            ("activity_endogeneity", "Identified activity-endogeneity correction", "data_blocked", "No defensible external instrument supplied"),
            ("ppi_expectation", "PPI-consistent frozen-date expectation", "data_blocked", "GDP-deflator t+1 proxy used"),
            ("markup_scale", "Externally identified b_x", "data_blocked", "Unit-scale calibration only"),
            ("realtime_lfo", "Vintage-consistent LFO", "data_blocked", "Required real-time vintages/release rules unavailable"),
            ("persistent_error", "Persistent quarterly inflation error", skipped, "Covariance primitive implemented; estimator integration outstanding"),
            ("low_frequency", "Low-frequency inflation nuisance", skipped, "Outstanding"),
            ("state_laws", "Alternative slow/cycle laws and idiosyncratic E cycle", skipped, "Outstanding"),
            ("reference_grid", "Reference-window, markup, b_x and bandwidth grids", skipped, "Baseline uncertainty propagated; sensitivity grid outstanding"),
            ("generated_gaps", "Generated-gap uncertainty and vintages", "data_blocked", "External BN construction/vintages unavailable"),
            ("chib", "Full generative Chib marginal likelihood", skipped, "No Bayes factor reported"),
            ("simulation_suite", "All pre-declared simulation validations", "partial", "A4/G4, hierarchy, and covariance tests implemented"),
        ],
        columns=["design_item", "requirement", "status", "note"],
    )
    compliance.to_csv(out / "tables" / "design_compliance.csv", index=False)
    manifest = stamp_artifact_metadata(
        {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "code_revision": str(run.data.config["revision"]),
            "estimation_revision": str(run.data.config["revision"]),
            "model": "nine_cell_E0_E1_E2",
            "model_hierarchy": ["E0", "E1", "E2"],
            "model_definition": "report/design.tex",
            "restriction_taxonomy": "main_unrestricted_auxiliary_hsa_benchmark",
            "exact_restrictions": [],
            "data_transformation": "quarterly annualized log and exact A4 YoY",
            "inflation_observation": "QoQ primary; exact aggregated YoY paired",
            "structural_frequency": "quarterly",
            "sample_start": str(run.data.periods[0]),
            "sample_end": str(run.data.periods[-1]),
            "n_obs": len(run.data.periods),
            "competition_proxy": "annual N_Gustavo plus quarterly QCEW establishments",
            "activity_proxy": "inverse markup / BN output gap / negative unemployment gap",
            "expectation_series": "SPF GDP-price t+1 (proxy outside GDP-price object)",
            "expectation_horizon": "one quarter ahead",
            "expectation_information_date": "survey quarter; release-date audit incomplete",
            "is_test_run": run.is_test,
            "measurement_information_ratio": run.measurement.information_ratio,
            "measurement_gate": bool(run.measurement.information_ratio <= 0.80 and not run.is_test),
            "q0": run.q0,
            "source_paths": run.data.source_paths,
            "implemented_robustness": ["fast_by_fast", "cs1", "cs2"],
            "incomplete_design_modules": [
                "persistent inflation disturbance", "low-frequency nuisance state",
                "alternative state laws", "idiosyncratic quarterly cycle",
                "prior scale grid", "Chib marginal likelihood", "real-time LFO",
            ],
        }
    )
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def run_nine_cell_design(
    *,
    config_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    test_run: bool = False,
    include_robustness: bool = True,
    progress: str | None = "auto",
) -> DesignRun:
    data = load_design_data(config_path)
    mode = data.config["test" if test_run else "production"]
    iterations, warmup = int(mode["iterations"]), int(mode["warmup"])
    thin, chains, seed = int(mode["thin"]), int(mode["chains"]), int(mode["seed"])
    out = Path(output_dir or results_root() / "nine_cell_design" / ("test_run" if test_run else "production"))
    with ProgressReporter(
        2 * chains * iterations,
        label="measurement N/C",
        key="nine-cell-measurement",
        style=progress,
    ) as measurement_progress:
        measurement_total = 2 * chains * iterations
        measurement_done = [0]

        def measurement_tick() -> None:
            measurement_done[0] += 1
            if measurement_done[0] < measurement_total:
                measurement_progress.update(measurement_done[0])

        measurement = sample_measurement_posterior(
            data.annual_observation,
            data.quarterly_indicator,
            q_scale=data.q_scale,
            e_scale=data.e_scale,
            periods=tuple(map(str, data.periods)),
            iterations=iterations,
            warmup=warmup,
            thin=thin,
            chains=chains,
            seed=seed,
            progress_tick=measurement_tick,
        )
    _, q0, _ = reference_draws(data, measurement)
    cut: dict[str, CellFit] = {}
    with ProgressReporter(9 * 7, label="nine-cell cut fits", key="nine-cell-cut", style=progress) as cut_progress:
        completed_fits = 0
        for cell in range(1, 10):
            for transformation in ("qoq", "yoy"):
                for model in ("E0", "E1", "E2"):
                    fit = fit_cut_model(
                        data, measurement, cell=cell, model=model,
                        transformation=transformation, q0=q0, seed=seed,
                    )
                    cut[fit_key(fit)] = fit
                    completed_fits += 1
                    if completed_fits < 9 * 7:
                        cut_progress.update(completed_fits)
            fit = fit_cut_model(
                data, measurement, cell=cell, model="E2",
                transformation="qoq_matched", q0=q0, seed=seed,
            )
            cut[fit_key(fit)] = fit
            completed_fits += 1
            if completed_fits < 9 * 7:
                cut_progress.update(completed_fits)

    joint_iterations = int(mode.get("joint_iterations", iterations))
    joint_warmup = int(mode.get("joint_warmup", warmup))
    joint: dict[str, JointCellFit] = {}
    with ProgressReporter(
        9 * chains * joint_iterations,
        label="full-joint QoQ E2",
        key="nine-cell-joint",
        style=progress,
    ) as joint_progress:
        joint_total = 9 * chains * joint_iterations
        joint_done = [0]

        def joint_tick() -> None:
            joint_done[0] += 1
            if joint_done[0] < joint_total:
                joint_progress.update(joint_done[0])

        for cell in range(1, 10):
            result = fit_joint_qoq_e2(
                data, measurement, cell=cell, q0=q0,
                iterations=joint_iterations, warmup=joint_warmup,
                thin=thin, chains=chains, seed=seed + 700001,
                progress_tick=joint_tick,
            )
            joint[fit_key(result.fit)] = result

    robustness: dict[str, CellFit] = {}
    if include_robustness:
        with ProgressReporter(9 * 3, label="robustness fits", key="nine-cell-robustness", style=progress) as robustness_progress:
            completed_robustness = 0
            for cell in range(1, 10):
                for extra in ("fast_by_fast", "cs1", "cs2"):
                    fit = fit_cut_model(
                        data, measurement, cell=cell, model="E2", transformation="qoq",
                        q0=q0, seed=seed + 900001, extra=extra,
                    )
                    robustness[f"{fit_key(fit)}_{extra}"] = fit
                    completed_robustness += 1
                    if completed_robustness < 9 * 3:
                        robustness_progress.update(completed_robustness)
    benchmark = fit_hsa_benchmark(data, measurement, seed=seed + 500001)
    run = DesignRun(
        data=data, measurement=measurement, cut_fits=cut, joint_fits=joint,
        robustness_fits=robustness, benchmark=benchmark, q0=q0,
        output_dir=out, is_test=test_run,
    )
    save_design_run(run)
    return run
