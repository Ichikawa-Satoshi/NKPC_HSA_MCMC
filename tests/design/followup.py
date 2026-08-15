from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from nkpc_hsa.paths import project_root, results_root
from nkpc_hsa.progress import ProgressReporter

from nkpc_hsa.phillips.data import CELL_SPECS, DesignData, load_design_data
from nkpc_hsa.phillips.estimation import summarize_fit
from nkpc_hsa.phillips.inflation import CellFit, fit_cut_model, reference_draws
from nkpc_hsa.phillips.state import (
    MeasurementPosterior,
    MeasurementSpec,
    sample_measurement_variant,
)


PRICE_LABEL = {"ppi": "PPI", "cpi": "CPI", "core_cpi": "Core CPI"}
ACTIVITY_LABEL = {
    "inverse_markup": "Inverse markup",
    "bn_output_gap": "BN output gap",
    "negative_unemployment_gap": "Negative unemployment gap",
}


def load_measurement_posterior(path: str | Path) -> MeasurementPosterior:
    archive = np.load(Path(path), allow_pickle=True)
    draws = {name[2:]: archive[name] for name in archive.files if name.startswith("C_")}
    annual = {name[2:]: archive[name] for name in archive.files if name.startswith("N_")}
    return MeasurementPosterior(
        draws=draws,
        annual_only_draws=annual,
        information_ratio=float(archive["information_ratio"]),
        periods=tuple(map(str, archive["periods"])),
    )


def _save_measurement(path: Path, posterior: MeasurementPosterior, spec: MeasurementSpec) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        periods=np.asarray(posterior.periods),
        information_ratio=posterior.information_ratio,
        specification=np.asarray(json.dumps(asdict(spec))),
        **{f"C_{name}": values for name, values in posterior.draws.items()},
        **{f"N_{name}": values for name, values in posterior.annual_only_draws.items()},
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


def _fix_copied_production_text(output_dir: Path, *, test_run: bool) -> None:
    if test_run:
        return
    table = output_dir / "tables" / "nine_cell_coefficients.tex"
    text = table.read_text(encoding="utf-8")
    text = text.replace(
        "The test run is a software validation only; its intervals are not inferential results.",
        "The production run uses retained post-warmup draws; convergence and identification gates are reported separately.",
    )
    table.write_text(text, encoding="utf-8")


def _diagnostic(values: np.ndarray) -> tuple[float, float]:
    data = np.asarray(values, float)
    if float(np.std(data)) == 0.0:
        return float("nan"), float("inf")
    idata = az.from_dict({"posterior": {"value": data}})
    rhat = float(np.asarray(az.rhat(idata, var_names=["value"], method="rank")["value"]))
    ess = float(np.asarray(az.ess(idata, var_names=["value"], method="bulk")["value"]))
    return rhat, ess


def _measurement_summary(
    name: str,
    posterior: MeasurementPosterior,
    *,
    threshold: float,
) -> dict[str, Any]:
    diagnostics: list[tuple[float, float]] = []
    for key, values in posterior.draws.items():
        if values.ndim == 2:
            diagnostics.append(_diagnostic(values))
    diagnostics.append(_diagnostic(posterior.draws["qhat"].mean(axis=2)))
    b_e = posterior.draws.get("b_e")
    phi = posterior.draws["phi_q"]
    sd_n = np.std(posterior.annual_only_draws["qhat"], axis=(0, 1), ddof=1)
    sd_c = np.std(posterior.draws["qhat"], axis=(0, 1), ddof=1)
    finite_rhat = [value[0] for value in diagnostics if np.isfinite(value[0])]
    finite_ess = [value[1] for value in diagnostics if np.isfinite(value[1])]
    max_rhat = max(finite_rhat)
    min_bulk_ess = min(finite_ess)
    return {
        "specification": name,
        "R_q": posterior.information_ratio,
        "uncertainty_reduction": 1.0 - posterior.information_ratio,
        "measurement_gate": bool(posterior.information_ratio <= threshold),
        "median_sd_annual_only": float(np.median(sd_n)),
        "median_sd_augmented": float(np.median(sd_c)),
        "phi_q_mean": float(phi.mean()),
        "b_e_mean": float(b_e.mean()) if b_e is not None else np.nan,
        "b_e_ci_2.5": float(np.quantile(b_e, 0.025)) if b_e is not None else np.nan,
        "b_e_ci_97.5": float(np.quantile(b_e, 0.975)) if b_e is not None else np.nan,
        "max_rhat": max_rhat,
        "min_bulk_ess": min_bulk_ess,
        "convergence_gate": bool(max_rhat <= 1.01 and min_bulk_ess >= 400),
    }


def _measurement_data_audit(data: DesignData) -> pd.DataFrame:
    annual_mask = np.isfinite(data.annual_observation)
    annual = data.annual_observation[annual_mask]
    quarterly_q4 = data.quarterly_indicator[annual_mask]
    corr = float(np.corrcoef(annual, quarterly_q4)[0, 1])
    rank_corr = float(pd.Series(annual).corr(pd.Series(quarterly_q4), method="spearman"))
    metrics: list[tuple[str, Any, str]] = [
        ("sample_start", str(data.periods[0]), "Frozen common sample"),
        ("sample_end", str(data.periods[-1]), "Frozen common sample"),
        ("annual_observations", int(annual_mask.sum()), "One annual observation at Q4"),
        ("quarterly_observations", len(data.quarterly_indicator), "QCEW establishments"),
        ("annual_robust_scale", data.q_scale, "Transformed coordinate"),
        ("quarterly_robust_scale", data.e_scale, "Transformed coordinate"),
        ("q4_pearson_correlation", corr, "Annual coordinate versus QCEW at Q4"),
        ("q4_spearman_correlation", rank_corr, "Annual coordinate versus QCEW at Q4"),
        (
            "annual_linear_trend",
            float(np.polyfit(np.arange(annual.size), annual, 1)[0]),
            "Coordinate units per year",
        ),
        (
            "quarterly_linear_trend",
            float(np.polyfit(np.arange(len(data.quarterly_indicator)), data.quarterly_indicator, 1)[0]),
            "Coordinate units per quarter",
        ),
    ]
    return pd.DataFrame(metrics, columns=["metric", "value", "interpretation"])


def _write_measurement_artifacts(
    data: DesignData,
    baseline: MeasurementPosterior,
    variants: dict[str, MeasurementPosterior],
    output_dir: Path,
    *,
    threshold: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows = [_measurement_summary("baseline", baseline, threshold=threshold)]
    rows.extend(
        _measurement_summary(name, posterior, threshold=threshold)
        for name, posterior in variants.items()
    )
    summary = pd.DataFrame(rows)
    summary.to_csv(output_dir / "tables" / "measurement_sensitivity.csv", index=False)
    _measurement_data_audit(data).to_csv(
        output_dir / "tables" / "measurement_data_audit.csv", index=False
    )
    fast_retained = bool(summary["measurement_gate"].all())
    decision = {
        "decision": "retain" if fast_retained else "demote",
        "fast_state_retained_for_substantive_interpretation": fast_retained,
        "rule": "retain only when baseline and every frozen measurement sensitivity satisfy R_q <= 0.80",
        "threshold": threshold,
        "failed_specifications": summary.loc[
            ~summary["measurement_gate"], "specification"
        ].tolist(),
    }
    (output_dir / "tables" / "fast_state_decision.json").write_text(
        json.dumps(decision, indent=2), encoding="utf-8"
    )

    tex = [
        r"\begin{tabular}{@{}lrrrrll@{}}",
        r"\toprule",
        r"Specification & $R_q$ & Reduction & $\phi_q$ & $b_E$ & Info gate & MCMC \\",
        r"\midrule",
    ]
    for row in summary.itertuples(index=False):
        gate = "Pass" if row.measurement_gate else "Fail"
        diagnostic = "Pass" if row.convergence_gate else "Flag"
        tex.append(
            f"{str(row.specification).replace('_', ' ')} & {row.R_q:.3f} & "
            f"{100.0 * row.uncertainty_reduction:.1f}\\% & {row.phi_q_mean:.3f} & "
            f"{row.b_e_mean:.3f} & {gate} & {diagnostic} " + r"\\"
        )
    tex.extend([r"\bottomrule", r"\end{tabular}"])
    (output_dir / "tables" / "measurement_sensitivity.tex").write_text(
        "\n".join(tex), encoding="utf-8"
    )

    periods = pd.PeriodIndex(baseline.periods, freq="Q").to_timestamp()
    sd_n = np.std(baseline.annual_only_draws["qhat"], axis=(0, 1), ddof=1)
    sd_c = np.std(baseline.draws["qhat"], axis=(0, 1), ddof=1)
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 7.0))
    annual_mask = np.isfinite(data.annual_observation)
    axes[0, 0].plot(periods, data.quarterly_indicator, color="#0072B2", label="QCEW")
    axes[0, 0].scatter(
        periods[annual_mask],
        data.annual_observation[annual_mask],
        s=18,
        color="#222222",
        label="Annual Q4",
        zorder=3,
    )
    axes[0, 0].set_title("Transformed business-demography measurements")
    axes[0, 0].legend(frameon=False)
    axes[0, 1].plot(periods, sd_n, color="#777777", label="Annual only")
    axes[0, 1].plot(periods, sd_c, color="#D55E00", label="Quarterly augmented")
    axes[0, 1].set_title(r"Posterior SD of $\hat q_t$")
    axes[0, 1].legend(frameon=False)
    axes[1, 0].plot(periods, sd_c / sd_n, color="#009E73")
    axes[1, 0].axhline(threshold, color="#D55E00", ls="--", label="Gate")
    axes[1, 0].axhline(1.0, color="#555555", ls=":", label="No gain")
    axes[1, 0].set_title("Pointwise uncertainty ratio")
    axes[1, 0].legend(frameon=False)
    colors = ["#D55E00" if not passed else "#009E73" for passed in summary.measurement_gate]
    axes[1, 1].barh(summary.specification.str.replace("_", " "), summary["R_q"], color=colors)
    axes[1, 1].axvline(threshold, color="#222222", ls="--")
    axes[1, 1].set_title(r"Measurement sensitivity: $R_q$")
    for ax in axes.flat:
        ax.grid(alpha=0.18)
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(output_dir / "figures" / "measurement_diagnostics.png", dpi=220)
    plt.close(fig)
    return summary, decision


def _fit_robustness(
    data: DesignData,
    measurement: MeasurementPosterior,
    output_dir: Path,
    *,
    seed: int,
    test_run: bool,
    focus_cells: tuple[int, ...],
    progress: str | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    fits: dict[str, CellFit] = {}
    tasks = [
        (cell, model, error_model)
        for error_model in ("persistent_ar1", "low_frequency")
        for model in ("E0", "E1", "E2")
        for cell in range(1, 10)
    ]
    q0 = reference_draws(data, measurement)[1]
    with ProgressReporter(
        len(tasks) + 9 + 2 * len(focus_cells),
        label="follow-up inflation robustness",
        key="nine-cell-followup-inflation",
        style=progress,
    ) as reporter:
        done = 0
        for cell, model, error_model in tasks:
            fit = fit_cut_model(
                data,
                measurement,
                cell=cell,
                model=model,
                transformation="qoq",
                q0=q0,
                seed=seed + 1100003,
                error_model=error_model,
            )
            key = f"cell{cell}_qoq_{model}_{fit.estimator}"
            fits[key] = fit
            _save_fit(output_dir / "posterior" / "followup" / f"{key}.npz", fit)
            done += 1
            reporter.update(done)
        for cell in range(1, 10):
            fit = fit_cut_model(
                data,
                measurement,
                cell=cell,
                model="SLOW",
                transformation="qoq",
                q0=q0,
                seed=seed + 1300003,
            )
            key = f"cell{cell}_qoq_SLOW_modular_cut"
            fits[key] = fit
            _save_fit(output_dir / "posterior" / "followup" / f"{key}.npz", fit)
            done += 1
            reporter.update(done)
        for cell in focus_cells:
            for transformation in ("qoq_matched", "yoy"):
                fit = fit_cut_model(
                    data,
                    measurement,
                    cell=cell,
                    model="SLOW",
                    transformation=transformation,
                    q0=q0,
                    seed=seed + 1500003,
                )
                key = f"cell{cell}_{transformation}_SLOW_modular_cut"
                fits[key] = fit
                _save_fit(output_dir / "posterior" / "followup" / f"{key}.npz", fit)
                done += 1
                reporter.update(done)

    frames: list[pd.DataFrame] = []
    for key, fit in fits.items():
        frame = summarize_fit(fit, test_run=test_run)
        if fit.model == "SLOW":
            frame["robustness"] = "slow_only"
        elif "persistent_ar1" in fit.estimator:
            frame["robustness"] = "persistent_ar1"
        else:
            frame["robustness"] = "low_frequency"
        frame["fit_key"] = key
        frames.append(frame)
    summaries = pd.concat(frames, ignore_index=True)
    summaries.to_csv(
        output_dir / "tables" / "followup_coefficient_summaries.csv", index=False
    )

    baseline = pd.read_csv(output_dir / "tables" / "coefficient_summaries.csv")
    baseline = baseline[
        (baseline["model"] == "E2")
        & (baseline["transformation"] == "qoq")
        & (baseline["estimator"] == "modular_cut")
        & baseline["robustness"].isna()
    ]
    baseline = baseline[baseline["parameter"].isin(["kappa_1", "theta_0", "gamma"])]
    robust = summaries[
        (summaries["model"].isin(["E2", "SLOW"]))
        & (summaries["transformation"] == "qoq")
        & summaries["parameter"].isin(["kappa_1", "theta_0", "gamma"])
    ]
    rows: list[dict[str, Any]] = []
    threshold = float(data.config["gates"]["qualitative_sign_probability"])
    for row in robust.itertuples(index=False):
        base = baseline[
            (baseline["cell"] == row.cell) & (baseline["parameter"] == row.parameter)
        ]
        if base.empty:
            continue
        b = base.iloc[0]
        shift = abs(float(row.mean) - float(b["mean"])) / float(b["sd"])
        sign_cross = (float(b["sign_probability"]) >= threshold) != (
            float(row.sign_probability) >= threshold
        )
        learning_fail = bool(b["learning_gate"]) and not bool(row.learning_gate)
        rows.append(
            {
                "cell": row.cell,
                "parameter": row.parameter,
                "robustness": row.robustness,
                "baseline_mean": b["mean"],
                "robust_mean": row.mean,
                "baseline_sd_shift": shift,
                "sign_threshold_crossed": sign_cross,
                "learning_gate_failed": learning_fail,
                "material": bool(
                    shift > float(data.config["gates"]["cut_joint_shift_sd"])
                    or sign_cross
                    or learning_fail
                ),
            }
        )
    shifts = pd.DataFrame(rows)
    shifts.to_csv(output_dir / "tables" / "followup_robustness_shifts.csv", index=False)
    return summaries, shifts


def _write_robustness_artifacts(
    output_dir: Path,
    summaries: pd.DataFrame,
    shifts: pd.DataFrame,
    *,
    focus_cells: tuple[int, ...],
) -> None:
    baseline = pd.read_csv(output_dir / "tables" / "coefficient_summaries.csv")
    baseline = baseline[
        (baseline.model == "E2")
        & (baseline.transformation == "qoq")
        & (baseline.estimator == "modular_cut")
        & baseline.robustness.isna()
        & baseline.parameter.eq("kappa_1")
    ].copy()
    baseline["robustness"] = "baseline"
    robust = summaries[
        summaries.parameter.eq("kappa_1")
        & summaries.transformation.eq("qoq")
        & summaries.model.isin(["E2", "SLOW"])
    ].copy()
    combined = pd.concat([baseline, robust], ignore_index=True)
    combined = combined[combined.cell.isin(focus_cells)]

    order = ["baseline", "persistent_ar1", "low_frequency", "slow_only"]
    tex = [
        r"\begin{tabular}{@{}clrrrr@{}}",
        r"\toprule",
        r"Cell & Specification & Mean & 95\% lower & 95\% upper & Sign probability \\",
        r"\midrule",
    ]
    for cell in focus_cells:
        for robustness in order:
            row = combined[
                (combined.cell == cell) & (combined.robustness == robustness)
            ]
            if row.empty:
                continue
            value = row.iloc[0]
            tex.append(
                f"{cell} & {robustness.replace('_', ' ')} & {value['mean']:.3f} & "
                f"{value['ci_2.5']:.3f} & {value['ci_97.5']:.3f} & "
                f"{value['sign_probability']:.3f} " + r"\\"
            )
    tex.extend([r"\bottomrule", r"\end{tabular}"])
    (output_dir / "tables" / "focus_robustness.tex").write_text(
        "\n".join(tex), encoding="utf-8"
    )

    shift_tex = [
        r"\begin{tabular}{@{}clrrl@{}}",
        r"\toprule",
        r"Cell & Robustness & $\kappa_1$ shift (baseline SD) & Material & Decision \\",
        r"\midrule",
    ]
    kappa = shifts[shifts.parameter.eq("kappa_1")]
    for cell in range(1, 10):
        for robustness in ("persistent_ar1", "low_frequency", "slow_only"):
            row = kappa[(kappa.cell == cell) & (kappa.robustness == robustness)]
            if row.empty:
                continue
            value = row.iloc[0]
            decision = "Flag" if value.material else "Pass"
            shift_tex.append(
                f"{cell} & {robustness.replace('_', ' ')} & "
                f"{value.baseline_sd_shift:.2f} & "
                f"{'Yes' if value.material else 'No'} & {decision} " + r"\\"
            )
    shift_tex.extend([r"\bottomrule", r"\end{tabular}"])
    (output_dir / "tables" / "inflation_robustness.tex").write_text(
        "\n".join(shift_tex), encoding="utf-8"
    )

    fig, axes = plt.subplots(1, len(focus_cells), figsize=(11.2, 4.2), sharey=True)
    colors = {
        "baseline": "#222222",
        "persistent_ar1": "#0072B2",
        "low_frequency": "#D55E00",
        "slow_only": "#009E73",
    }
    display_labels = {
        "baseline": "baseline",
        "persistent_ar1": "persistent AR(1)",
        "low_frequency": "low frequency",
        "slow_only": "slow only",
    }
    for ax, cell in zip(axes, focus_cells):
        frame = combined[combined.cell.eq(cell)].set_index("robustness")
        for y, robustness in enumerate(order[::-1]):
            if robustness not in frame.index:
                continue
            row = frame.loc[robustness]
            ax.errorbar(
                row["mean"],
                y,
                xerr=[
                    [row["mean"] - row["ci_2.5"]],
                    [row["ci_97.5"] - row["mean"]],
                ],
                fmt="o",
                color=colors[robustness],
                ms=5,
            )
        ax.axvline(0.0, color="#777777", lw=0.8)
        ax.set_title(f"Cell {cell}")
        ax.grid(axis="x", alpha=0.18)
        ax.spines[["top", "right", "left"]].set_visible(False)
    axes[0].set_yticks(
        range(len(order)), labels=[display_labels[value] for value in order[::-1]]
    )
    fig.suptitle(r"Focused $\kappa_1$ robustness after the measurement-gate decision")
    fig.tight_layout()
    fig.savefig(output_dir / "figures" / "focus_robustness.png", dpi=220)
    plt.close(fig)


def _rewrite_evidence(
    output_dir: Path,
    measurement: pd.DataFrame,
    decision: dict[str, Any],
    summaries: pd.DataFrame,
) -> None:
    baseline = pd.read_csv(output_dir / "tables" / "coefficient_summaries.csv")
    main = baseline[
        (baseline.model == "E2")
        & (baseline.transformation == "qoq")
        & (baseline.estimator == "modular_cut")
        & baseline.robustness.isna()
    ]
    robust_pass = int(summaries["convergence_gate"].sum())
    mechanism = summaries[
        summaries["parameter"].isin(["kappa_1", "theta_0", "gamma"])
    ]
    mechanism_pass = int(mechanism["convergence_gate"].sum())
    lines = [
        r"\begin{tabular}{@{}>{\raggedright\arraybackslash}p{3.8cm}>{\raggedright\arraybackslash}p{2.6cm}>{\raggedright\arraybackslash}p{7.5cm}@{}}",
        r"\toprule Gate / evidence & Production status & Interpretation \\",
        r"\midrule",
        f"Baseline quarterly information $R_q\\le0.80$ & {measurement.iloc[0]['R_q']:.3f} & "
        + ("Pass." if measurement.iloc[0].measurement_gate else "Fail; the stationary state is not substantively interpreted.")
        + r"\\",
        f"Alternative measurement specifications & {int(measurement.measurement_gate.sum())}/{len(measurement)} pass & "
        + "The fast state is retained only if every frozen specification passes."
        + r"\\",
        f"Fast-state decision & {decision['decision'].capitalize()} & "
        + (
            "Fast-state coefficients remain interpretable."
            if decision["fast_state_retained_for_substantive_interpretation"]
            else r"$\theta_0$ and $\gamma$ are reported but demoted; slow-only results are the fallback."
        )
        + r"\\",
        f"Baseline coefficient convergence & {int(main.convergence_gate.sum())}/{len(main)} pass & "
        + r"Production $\widehat R$/ESS gates evaluated on retained draws.\\",
        f"Inflation-error robustness & {mechanism_pass}/{len(mechanism)} mechanism pass & "
        + f"AR(1) and low-frequency specifications are executed across E0--E2; {len(summaries) - robust_pass} non-mechanism intercept draws are flagged."
        + r"\\",
        r"Activity endogeneity & Not identified & Coefficients remain conditional associations.\\",
        r"Chib Bayes factors & Not run & No Bayes-factor claim is made from the modular cut.\\",
        r"Real-time LFO score & Inadmissible & Required vintages and release rules remain unavailable.\\",
        r"\bottomrule\end{tabular}",
    ]
    (output_dir / "tables" / "evidence.tex").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def _update_metadata(
    output_dir: Path,
    measurement: pd.DataFrame,
    decision: dict[str, Any],
    shifts: pd.DataFrame,
) -> None:
    manifest_path = output_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update(
        {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "report_status": "final_followup_production",
            "measurement_sensitivity": measurement.to_dict(orient="records"),
            "fast_state_decision": decision,
            "persistent_inflation_disturbance": "executed_E0_E1_E2_all_cells_qoq",
            "low_frequency_inflation_nuisance": "executed_E0_E1_E2_all_cells_qoq",
            "slow_only_fallback": "executed_all_cells_qoq_and_focus_cells_qoq_matched_yoy",
            "focus_cells": [2, 5, 6, 9],
            "material_followup_shifts": int(shifts.material.sum()),
        }
    )
    incomplete = [
        item
        for item in manifest.get("incomplete_design_modules", [])
        if item
        not in {
            "persistent inflation disturbance",
            "low-frequency nuisance state",
            "alternative state laws",
            "idiosyncratic quarterly cycle",
        }
    ]
    manifest["incomplete_design_modules"] = incomplete
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    compliance_path = output_dir / "tables" / "design_compliance.csv"
    compliance = pd.read_csv(compliance_path)
    replacements = {
        "persistent_error": (
            "executed",
            "QoQ AR(1) disturbance across E0--E2 and all nine cells",
        ),
        "low_frequency": (
            "executed",
            "Latent AR(1) low-frequency nuisance across E0--E2 and all nine cells",
        ),
        "state_laws": (
            "executed",
            "No-drift, persistent-cycle, and idiosyncratic-QCEW-cycle sensitivities",
        ),
    }
    for item, (status, note) in replacements.items():
        mask = compliance.design_item.eq(item)
        compliance.loc[mask, ["status", "note"]] = [status, note]
    compliance.to_csv(compliance_path, index=False)


def compile_final_report(
    output_dir: Path, *, fast_retained: bool, test_run: bool = False
) -> Path:
    root = project_root()
    report_dir = root / "report"
    source = report_dir / "nine_cell_design_report.tex"
    relative = Path("..") / "results" / "nine_cell_design" / output_dir.name
    label = "FOLLOW-UP TEST RUN: NOT FOR INFERENCE" if test_run else (
        "FINAL PRODUCTION RUN: FAST STATE RETAINED"
        if fast_retained
        else "FINAL PRODUCTION RUN: FAST STATE DEMOTED"
    )
    command = (
        rf"\def\ResultRoot{{{relative.as_posix()}}}"
        rf"\def\RunLabel{{{label}}}"
        rf"\def\RunAbstractNotice{{{'This short follow-up run validates the new estimators and report path; it is not an empirical result.' if test_run else 'This final production report combines the converged core run with pre-declared measurement, persistent-error, low-frequency, and slow-only follow-up analyses.'}}}"
        rf"\def\PriorCaptionNotice{{{'Short test chains are shown only to verify the reporting path.' if test_run else 'Production posterior summaries use retained post-warmup draws.'}}}"
        rf"\def\ScopeNotice{{Failed identification gates remain binding; unavailable external instruments, real-time vintages, and PPI-specific expectations are not treated as passed.}}"
        rf"\input{{{source.name}}}"
    )
    for _ in range(2):
        result = subprocess.run(
            ["xelatex", "-interaction=nonstopmode", "-jobname=nine_cell_design_final_report", command],
            cwd=report_dir,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError("\n".join(result.stdout.splitlines()[-60:]))
    pdf = report_dir / "nine_cell_design_final_report.pdf"
    delivered = output_dir / "nine_cell_design_final_report.pdf"
    shutil.copy2(pdf, delivered)
    return delivered


def run_followup(
    *,
    config_path: str | Path | None = None,
    baseline_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
    test_run: bool = False,
    progress: str | None = "auto",
    compile_report: bool = True,
    reuse_existing: bool = False,
) -> Path | None:
    data = load_design_data(config_path)
    mode = data.config["test" if test_run else "production"]
    source = Path(
        baseline_dir
        or results_root()
        / "nine_cell_design"
        / ("test_run" if test_run else "production")
    )
    output = Path(
        output_dir
        or results_root()
        / "nine_cell_design"
        / ("followup_test" if test_run else "production_final")
    )
    output.mkdir(parents=True, exist_ok=True)
    if not reuse_existing:
        for item in source.iterdir():
            target = output / item.name
            if item.is_dir():
                shutil.copytree(item, target, dirs_exist_ok=True)
            else:
                shutil.copy2(item, target)
    (output / "tables").mkdir(parents=True, exist_ok=True)
    (output / "figures").mkdir(parents=True, exist_ok=True)
    _fix_copied_production_text(output, test_run=test_run)

    baseline = load_measurement_posterior(source / "posterior" / "measurement_cut.npz")
    sensitivity_specs = [
        MeasurementSpec(**spec) for spec in data.config["measurement"]["sensitivity_specs"]
    ]
    variants: dict[str, MeasurementPosterior] = {}
    iterations = int(mode["iterations"])
    warmup = int(mode["warmup"])
    thin = int(mode["thin"])
    chains = int(mode["chains"])
    seed = int(mode["seed"])
    for number, spec in enumerate(sensitivity_specs):
        saved_path = output / "posterior" / "measurement_sensitivity" / f"{spec.name}.npz"
        if reuse_existing:
            variants[spec.name] = load_measurement_posterior(saved_path)
            continue
        total = 2 * chains * iterations
        with ProgressReporter(
            total,
            label=f"measurement sensitivity: {spec.name}",
            key=f"measurement-{spec.name}",
            style=progress,
        ) as reporter:
            posterior = sample_measurement_variant(
                data.annual_observation,
                data.quarterly_indicator,
                q_scale=data.q_scale,
                e_scale=data.e_scale,
                periods=tuple(map(str, data.periods)),
                spec=spec,
                iterations=iterations,
                warmup=warmup,
                thin=thin,
                chains=chains,
                seed=seed + (number + 1) * 2000003,
                progress_update=reporter.update,
                parallel_chains=not test_run,
            )
        variants[spec.name] = posterior
        _save_measurement(saved_path, posterior, spec)

    threshold = float(data.config["gates"]["measurement_information_ratio"])
    measurement_summary, decision = _write_measurement_artifacts(
        data,
        baseline,
        variants,
        output,
        threshold=threshold,
    )
    focus_cells = tuple(map(int, data.config["followup"]["focus_cells"]))
    if reuse_existing:
        summaries = pd.read_csv(
            output / "tables" / "followup_coefficient_summaries.csv"
        )
        shifts = pd.read_csv(output / "tables" / "followup_robustness_shifts.csv")
    else:
        summaries, shifts = _fit_robustness(
            data,
            baseline,
            output,
            seed=seed,
            test_run=test_run,
            focus_cells=focus_cells,
            progress=progress,
        )
    _write_robustness_artifacts(
        output, summaries, shifts, focus_cells=focus_cells
    )
    _rewrite_evidence(output, measurement_summary, decision, summaries)
    _update_metadata(output, measurement_summary, decision, shifts)
    if compile_report:
        return compile_final_report(
            output,
            fast_retained=decision["fast_state_retained_for_substantive_interpretation"],
            test_run=test_run,
        )
    return None
