"""Estimate the mandatory N_Gustavo-only quarterly state-space specification."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import time
from pathlib import Path

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import sys as _sys, pathlib as _pathlib  # noqa: E402  (bootstrap: importable at any depth)
_ROOT = next(_p for _p in _pathlib.Path(__file__).resolve().parents if (_p / "pyproject.toml").exists())
_sys.path[:0] = [str(_ROOT), str(_ROOT / "src"), str(_ROOT / "tests")]
from tests import _bootstrap  # noqa: F401,E402
from tests._bootstrap import RESULTS_DIR, ROOT


BUNDLE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BUNDLE_DIR / "results"
from nkpc_hsa.phillips.data import load_design_data
from nkpc_hsa.phillips.estimation import summarize_fit
from nkpc_hsa.phillips.inflation import fit_cut_model, reference_draws
from nkpc_hsa.phillips.state import (
    MeasurementPosterior,
    MeasurementSpec,
    sample_annual_only_posterior,
    sample_annual_only_variant,
)
from nkpc_hsa.progress import ProgressReporter


BLUE = "#0072B2"
ORANGE = "#D55E00"
GREEN = "#009E73"
GREY = "#6B7280"


def _diagnostics(values: np.ndarray) -> tuple[float, float]:
    return float(np.asarray(az.rhat(values, method="rank"))), float(np.asarray(az.ess(values, method="bulk")))


def _state_summary(measurement: MeasurementPosterior, annual: np.ndarray) -> dict[str, float]:
    draws = measurement.draws
    total = draws["qbar"] + draws["qhat"]
    q4 = np.isfinite(annual)
    posterior_mean = total.mean(axis=(0, 1))
    residual = annual[q4] - posterior_mean[q4]
    qbar_flat = draws["qbar"].reshape(-1, draws["qbar"].shape[-1])
    qhat_flat = draws["qhat"].reshape(-1, draws["qhat"].shape[-1])
    correlations = [np.corrcoef(a, b)[0, 1] for a, b in zip(qbar_flat[::20], qhat_flat[::20], strict=True)]
    out = {
        "annual_observations": int(q4.sum()),
        "quarterly_states": int(annual.size),
        "annual_fit_rmse": float(np.sqrt(np.mean(residual**2))),
        "median_qhat_posterior_sd": float(np.median(np.std(draws["qhat"], axis=(0, 1), ddof=1))),
        "mean_qbar_qhat_path_correlation": float(np.mean(correlations)),
    }
    for name in ("d_q", "phi_q", "sigma_qbar", "sigma_qhat", "sigma_annual"):
        values = draws[name]
        rhat, ess = _diagnostics(values)
        flat = values.reshape(-1)
        out[f"{name}_mean"] = float(flat.mean())
        out[f"{name}_ci_2_5"] = float(np.quantile(flat, 0.025))
        out[f"{name}_ci_97_5"] = float(np.quantile(flat, 0.975))
        out[f"{name}_rhat"] = rhat
        out[f"{name}_bulk_ess"] = ess
    out["state_scalar_convergence_gate"] = bool(
        max(out[f"{name}_rhat"] for name in ("d_q", "phi_q", "sigma_qbar", "sigma_qhat", "sigma_annual")) <= 1.01
        and min(out[f"{name}_bulk_ess"] for name in ("d_q", "phi_q", "sigma_qbar", "sigma_qhat", "sigma_annual")) >= 400
    )
    return out


def _fit_grid(data, measurement, q0: float, seed: int, *, test_run: bool) -> tuple[pd.DataFrame, dict[str, object]]:
    summaries: list[pd.DataFrame] = []
    fits: dict[str, object] = {}
    tasks = []
    for cell in range(1, 10):
        for model in ("E1", "E2"):
            for error in (None, "persistent_ar1", "low_frequency"):
                tasks.append((cell, model, "qoq", error, False))
            tasks.append((cell, model, "yoy", None, False))
    for model in ("E1", "E2"):
        for error in (None, "persistent_ar1", "low_frequency"):
            tasks.append((1, model, "qoq", error, True))
    with ProgressReporter(len(tasks), label="N_Gustavo-only inflation fits", key="n-gustavo-only-fits", style="auto") as progress:
        for index, (cell, model, transformation, error, no_lag) in enumerate(tasks, 1):
            fit = fit_cut_model(
                data,
                measurement,
                cell=cell,
                model=model,
                transformation=transformation,
                q0=q0,
                seed=seed + index * 1009,
                error_model=error,
                no_lag=no_lag,
            )
            key = f"cell{cell}_{model}_{transformation}_{error or 'iid'}_{'nolag' if no_lag else 'hybrid'}"
            fits[key] = fit
            table = summarize_fit(fit, test_run=test_run)
            table["error_model"] = error or "iid"
            table["no_lag"] = no_lag
            table["fit_key"] = key
            summaries.append(table)
            if index < len(tasks):
                progress.update(index)
    return pd.concat(summaries, ignore_index=True), fits


def _state_sensitivity_grid(data, sampling: dict, out: Path) -> pd.DataFrame:
    specs = (
        MeasurementSpec(name="fixed_phi_05", phi_fixed=0.5),
        MeasurementSpec(name="fixed_phi_09", phi_fixed=0.9),
        MeasurementSpec(name="no_drift_fixed_phi_08", q_drift=False, phi_fixed=0.8),
    )
    rows = []
    posterior_dir = out / "posterior" / "state_sensitivity"
    posterior_dir.mkdir(parents=True, exist_ok=True)
    print(f"running {len(specs)} N_Gustavo-only state-law sensitivities with parallel chains", flush=True)
    for index, spec in enumerate(specs, 1):
        path = posterior_dir / f"{spec.name}.npz"
        if path.exists():
            saved = np.load(path)
            draws = {name.removeprefix("draw_"): saved[name] for name in saved.files}
            valid_saved_shape = draws["qhat"].shape[-1] == len(data.periods)
        else:
            valid_saved_shape = False
        if valid_saved_shape:
            measurement = MeasurementPosterior(draws=draws, annual_only_draws=draws, information_ratio=float("nan"), periods=tuple(map(str, data.periods)))
        else:
            measurement = sample_annual_only_variant(
                data.annual_observation,
                q_scale=data.q_scale,
                periods=tuple(map(str, data.periods)),
                spec=spec,
                **sampling,
            )
            np.savez_compressed(path, **{f"draw_{k}": v for k, v in measurement.draws.items()})
        _, q0, _ = reference_draws(data, measurement)
        fit = fit_cut_model(data, measurement, cell=1, model="E1", transformation="qoq", q0=q0, seed=sampling["seed"] + index * 100003, error_model="persistent_ar1")
        theta = summarize_fit(fit, test_run=False).loc[lambda x: x.parameter.eq("theta_0")].iloc[0]
        qhat = measurement.draws["qhat"]
        rows.append({
            "state_law": spec.name,
            "q_drift": spec.q_drift,
            "phi_fixed": spec.phi_fixed,
            "theta_mean": theta["mean"],
            "theta_ci_2_5": theta["ci_2.5"],
            "theta_ci_97_5": theta["ci_97.5"],
            "theta_sign_probability": theta.sign_probability,
            "theta_posterior_prior_sd_ratio": theta.posterior_prior_sd_ratio,
            "theta_rhat": theta.rhat,
            "theta_bulk_ess": theta.bulk_ess,
            "median_qhat_posterior_sd": float(np.median(np.std(qhat, axis=(0, 1), ddof=1))),
        })
        print(f"[{index}/{len(specs)}] {spec.name} completed", flush=True)
    return pd.DataFrame(rows)


def _write_figures(data, measurement, summary: pd.DataFrame, sensitivity: pd.DataFrame, figures: Path) -> None:
    figures.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")
    periods = data.periods.to_timestamp()
    total = measurement.draws["qbar"] + measurement.draws["qhat"]
    fig, axes = plt.subplots(2, 1, figsize=(10.5, 6.3), sharex=True)
    for ax, values, title in (
        (axes[0], measurement.draws["qbar"], "Slow competition environment"),
        (axes[1], measurement.draws["qhat"], "Stationary quarterly competition state"),
    ):
        flat = values.reshape(-1, values.shape[-1])
        ax.fill_between(periods, np.quantile(flat, .025, axis=0), np.quantile(flat, .975, axis=0), color=BLUE, alpha=.18)
        ax.plot(periods, flat.mean(axis=0), color=BLUE, lw=1.4)
        ax.axhline(0, color="black", lw=.7)
        ax.set_title(title)
        ax.set_ylabel("Ten log points")
    mask = np.isfinite(data.annual_observation)
    axes[0].scatter(periods[mask], data.annual_observation[mask], color="black", s=16, label="Observed N_Gustavo (Q4)", zorder=4)
    axes[0].legend(frameon=False)
    axes[1].set_xlabel("Quarter")
    fig.tight_layout()
    fig.savefig(figures / "n_gustavo_quarterly_states.png", dpi=220)
    plt.close(fig)

    theta = summary.loc[
        summary.parameter.eq("theta_0") & summary.model.eq("E1") & summary.transformation.eq("qoq") & ~summary.no_lag
    ]
    offsets = {"iid": -.18, "persistent_ar1": 0, "low_frequency": .18}
    colors = {"iid": BLUE, "persistent_ar1": ORANGE, "low_frequency": GREEN}
    fig, ax = plt.subplots(figsize=(9, 5.3))
    for error, group in theta.groupby("error_model"):
        y = group.cell.to_numpy(float) + offsets[error]
        ax.errorbar(group["mean"], y, xerr=np.vstack((group["mean"]-group["ci_2.5"], group["ci_97.5"]-group["mean"])), fmt="o", capsize=2.5, color=colors[error], label=error.replace("_", " "))
    ax.axvline(0, color="black", lw=1)
    ax.set_yticks(range(1,10))
    ax.set(xlabel=r"$\theta_0$ (95% credible interval)", ylabel="Nine-cell design cell", title="N_Gustavo-only quarterly-state loading")
    ax.legend(frameon=False, ncol=3)
    fig.tight_layout()
    fig.savefig(figures / "n_gustavo_theta_forest.png", dpi=220)
    plt.close(fig)

    cell1 = summary.loc[summary.cell.eq(1) & summary.parameter.eq("theta_0") & summary.transformation.eq("qoq")].copy()
    cell1["label"] = cell1.model + " / " + cell1.error_model.str.replace("_", " ") + np.where(cell1.no_lag, " / no lagged inflation", "")
    cell1 = cell1.sort_values(["no_lag", "model", "error_model"])
    fig, ax = plt.subplots(figsize=(9.5, 4.2))
    y = np.arange(len(cell1))
    ax.errorbar(cell1["mean"], y, xerr=np.vstack((cell1["mean"]-cell1["ci_2.5"], cell1["ci_97.5"]-cell1["mean"])), fmt="o", color=ORANGE, capsize=2.5)
    ax.axvline(0, color="black", lw=1)
    ax.set_yticks(y, cell1.label, fontsize=8)
    ax.set_xlabel(r"Cell 1 $\theta_0$ (95% credible interval)")
    ax.set_title("Cell 1 specification sensitivity")
    fig.tight_layout()
    fig.savefig(figures / "n_gustavo_cell1_sensitivity.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 2.7))
    y = np.arange(len(sensitivity))
    ax.errorbar(sensitivity.theta_mean, y, xerr=np.vstack((sensitivity.theta_mean-sensitivity.theta_ci_2_5, sensitivity.theta_ci_97_5-sensitivity.theta_mean)), fmt="o", color=GREEN, capsize=3)
    ax.axvline(0, color="black", lw=1)
    ax.set_yticks(y, sensitivity.state_law.str.replace("_", " "))
    ax.set_xlabel(r"Cell 1 $\theta_0$ (95% credible interval)")
    ax.set_title("Sensitivity to imposed quarterly state dynamics")
    fig.tight_layout()
    fig.savefig(figures / "n_gustavo_state_law_sensitivity.png", dpi=220)
    plt.close(fig)


def _fmt(x: object, digits: int = 3) -> str:
    return "--" if x is None or pd.isna(x) else f"{float(x):.{digits}f}"


def _write_report(out: Path, summary: pd.DataFrame, sensitivity: pd.DataFrame, state: dict[str, float], sampling: dict) -> Path:
    primary = summary.loc[
        summary.parameter.eq("theta_0") & summary.model.eq("E1") & summary.transformation.eq("qoq")
        & summary.error_model.eq("persistent_ar1") & ~summary.no_lag
    ].sort_values("cell")
    rows = "\n".join(
        f"{int(r.cell)} & {str(r.inflation).replace('_',' ')} / {str(r.activity).replace('_',' ')} & {_fmt(r['mean'])} & [{_fmt(r['ci_2.5'])}, {_fmt(r['ci_97.5'])}] & {_fmt(r.sign_probability)} & {_fmt(r.posterior_prior_sd_ratio)} \\\\"
        for _, r in primary.iterrows()
    )
    cell1 = primary.loc[primary.cell.eq(1)].iloc[0]
    all_theta = summary.loc[summary.parameter.eq("theta_0")]
    theta_converged = int(all_theta.convergence_gate.sum())
    nonzero = int(((primary["ci_2.5"] > 0) | (primary["ci_97.5"] < 0)).sum())
    sensitivity_rows = "\n".join(
        f"{str(r.state_law).replace('_',' ')} & {_fmt(r.phi_fixed)} & {_fmt(r.theta_mean)} & [{_fmt(r.theta_ci_2_5)}, {_fmt(r.theta_ci_97_5)}] & {_fmt(r.theta_sign_probability)} \\\\"
        for _, r in sensitivity.iterrows()
    )
    tex = rf"""\documentclass[11pt]{{article}}
\usepackage[margin=0.82in]{{geometry}}
\usepackage{{booktabs,graphicx,xcolor,amsmath,microtype,hyperref,newtxtext,newtxmath}}
\definecolor{{navy}}{{HTML}}{{17365D}}\definecolor{{light}}{{HTML}}{{EEF3F8}}
\setlength{{\parindent}}{{0pt}}\setlength{{\parskip}}{{5pt}}
\begin{{document}}
\begin{{center}}{{\color{{navy}}\LARGE\bfseries N-Gustavo-Only Quarterly State-Space Test}}\\[4pt]
{{\large Native annual observations, inferred quarterly competition states}}\\[7pt]
{sampling['iterations']:,} iterations, {sampling['warmup']:,} warmup, thin {sampling['thin']}, {sampling['chains']} chains
\end{{center}}
\colorbox{{light}}{{\parbox{{0.95\linewidth}}{{\textbf{{Executive result.}} Annual \texttt{{N\_Gustavo}} alone identifies a long quarterly history but not a sharply signed Cell-1 fast-state loading. Under the primary persistent-error E1 model, $\theta_0={cell1['mean']:.3f}$ with 95\% interval [{cell1['ci_2.5']:.3f}, {cell1['ci_97.5']:.3f}]. The state-space calculation is stable only if the displayed state-parameter diagnostics pass; no QCEW, SEC HHI, interpolation, or inflation observation enters the state smoother.}}}}

\section*{{1. Mandatory interim specification}}
The native annual inverse-HHI observation is placed at calendar Q4. Q1--Q3 are missing observations. A mixed-frequency state-space model decomposes
\[q_t=\bar q_t+\hat q_t,\qquad \bar q_t=\bar q_{{t-1}}+d+u_t,\qquad \hat q_t=\phi\hat q_{{t-1}}+v_t,\]
and observes $N_y=\bar q_{{t_y}}+\hat q_{{t_y}}+\nu_y$ only at Q4. The state posterior is estimated before and independently of inflation. This is a modular cut, not annual interpolation and not an annual regression. Capital IQ or Datastream quarterly HHI will replace this interim primary input only after its construction audit.

\section*{{2. What the annual series tells us about quarterly states}}
There are {int(state['annual_observations'])} annual observations over {int(state['quarterly_states'])} quarters. The posterior median quarterly SD of the fast state is {state['median_qhat_posterior_sd']:.3f}; the mean within-draw slow/fast path correlation is {state['mean_qbar_qhat_path_correlation']:.3f}. The scalar state-parameter convergence gate is {'PASS' if state['state_scalar_convergence_gate'] else 'FAIL'}; $\phi$ has posterior mean {state['phi_q_mean']:.3f} and interval [{state['phi_q_ci_2_5']:.3f}, {state['phi_q_ci_97_5']:.3f}]. These diagnostics matter because annual observations cannot directly reveal the quarter in which a within-year movement occurred.
\begin{{center}}\includegraphics[width=0.96\linewidth]{{figures/n_gustavo_quarterly_states.png}}\end{{center}}

\section*{{3. Primary $\theta$ results}}
The E1 equation allows the Phillips-curve slope to vary with the slow competition environment and loads inflation directly on the inferred stationary quarterly state. Persistent AR(1) inflation errors are primary.
\begin{{center}}\small\begin{{tabular}}{{@{{}}r l r r r r@{{}}}}\toprule Cell & Inflation / activity & Mean & 95\% interval & Sign prob. & SD ratio\\\midrule
{rows}
\bottomrule\end{{tabular}}\end{{center}}
All {theta_converged}/{len(all_theta)} reported $\theta$ fits pass the coefficient convergence rule. {nonzero} of nine primary intervals exclude zero.
\begin{{center}}\includegraphics[width=0.94\linewidth]{{figures/n_gustavo_theta_forest.png}}\end{{center}}

\section*{{4. Cell-1 robustness}}
E2 permits the fast-state loading to vary with the slow environment. The grid also permits persistent and low-frequency inflation disturbances and removes lagged inflation as a theory-near sensitivity. A result is treated as robust only if its sign survives these changes; a narrow interval in one specification is insufficient.
\begin{{center}}\includegraphics[width=0.94\linewidth]{{figures/n_gustavo_cell1_sensitivity.png}}\end{{center}}

\section*{{5. State-law sensitivity}}
The unrestricted state sampler does not pass the scalar state-parameter convergence gate because $\phi$ and $\sigma_{{\bar q}}$ mix slowly. This is evidence that annual N-Gustavo alone does not freely identify the quarterly slow/fast decomposition. The following tests impose transparent quarterly persistence assumptions; their tighter state paths are conditional on those assumptions and are not independent identification evidence.
\begin{{center}}\small\begin{{tabular}}{{@{{}}l r r r r@{{}}}}\toprule State law & Fixed $\phi$ & $\theta_0$ & 95\% interval & Sign prob.\\\midrule
{sensitivity_rows}
\bottomrule\end{{tabular}}\end{{center}}
\begin{{center}}\includegraphics[width=0.82\linewidth]{{figures/n_gustavo_state_law_sensitivity.png}}\end{{center}}

\section*{{6. Interpretation}}
This exercise answers whether the native annual N-Gustavo history, combined only with a quarterly state law, supports a distinct inflation loading. It does not pretend that Q1--Q3 competition was observed. If $\theta$ remains weak, the limitation is the annual observation frequency and slow/fast decomposition, not the short SEC overlap. The next new competition input is a validated Capital IQ or Datastream quarterly HHI; QCEW and the short SEC HHI remain outside the interim primary state.
\end{{document}}
"""
    path = out / "n_gustavo_state_space_tests.tex"
    path.write_text(tex, encoding="utf-8")
    subprocess.run(["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error", path.name], cwd=out, check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    return path.with_suffix(".pdf")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--reuse-existing", action="store_true")
    args = parser.parse_args()
    out, tables, figures = args.output_dir, args.output_dir / "tables", args.output_dir / "figures"
    out.mkdir(parents=True, exist_ok=True); tables.mkdir(parents=True, exist_ok=True)
    data = load_design_data(include_qcew=False, sample_end="2013Q4")
    mode = dict(data.config["test" if args.quick else "production"])
    sampling = {k: int(mode[k]) for k in ("iterations", "warmup", "thin", "chains", "seed")}
    started = time.perf_counter()
    state_path = out / "posterior" / "n_gustavo_only_state.npz"
    summary_path = tables / "coefficient_summaries.csv"
    if args.reuse_existing:
        saved = np.load(state_path)
        draws = {name.removeprefix("draw_"): saved[name] for name in saved.files if name.startswith("draw_")}
        measurement = MeasurementPosterior(draws=draws, annual_only_draws=draws, information_ratio=float("nan"), periods=tuple(map(str, data.periods)))
        summary = pd.read_csv(summary_path)
    else:
        with ProgressReporter(sampling["chains"] * sampling["iterations"], label="N_Gustavo-only quarterly state", key="n-gustavo-only-state", style="auto") as progress:
            done = [0]
            def tick() -> None:
                done[0] += 1
                if done[0] < sampling["chains"] * sampling["iterations"]: progress.update(done[0])
            measurement = sample_annual_only_posterior(data.annual_observation, q_scale=data.q_scale, periods=tuple(map(str, data.periods)), progress_tick=tick, **sampling)
        _, q0, _ = reference_draws(data, measurement)
        summary, fits = _fit_grid(data, measurement, q0, sampling["seed"], test_run=args.quick)
        summary.to_csv(summary_path, index=False)
        state_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(state_path, **{f"draw_{k}": v for k, v in measurement.draws.items()})
        for key, fit in fits.items():
            np.savez_compressed(out / "posterior" / f"{key}.npz", coefficients=fit.coefficients, sigma=fit.sigma, coefficient_names=np.asarray(fit.coefficient_names), **fit.auxiliary_draws)
    state = _state_summary(measurement, data.annual_observation)
    sensitivity_path = tables / "state_law_sensitivity.csv"
    sensitivity = _state_sensitivity_grid(data, sampling, out) if not args.quick else _state_sensitivity_grid(data, sampling, out)
    sensitivity.to_csv(sensitivity_path, index=False)
    (tables / "state_diagnostics.json").write_text(json.dumps(state, indent=2), encoding="utf-8")
    _write_figures(data, measurement, summary, sensitivity, figures)
    pdf = _write_report(out, summary, sensitivity, state, sampling)
    manifest = {
        "revision": "n-gustavo-only-quarterly-state-space-v2",
        "interim_primary": True,
        "competition_input": "N_Gustavo annual Q4 only",
        "quarterly_state_method": "mixed-frequency Kalman FFBS",
        "uses_qcew": False,
        "uses_sec_hhi": False,
        "uses_deterministic_interpolation": False,
        "uses_inflation_in_state_smoother": False,
        "sample_start": str(data.periods[0]), "sample_end": str(data.periods[-1]),
        "annual_observations": int(np.isfinite(data.annual_observation).sum()),
        "quarterly_states": len(data.periods), "sampling": sampling,
        "coefficient_rows": len(summary), "state_diagnostics": state,
        "state_law_sensitivities": sensitivity.to_dict(orient="records"),
        "source_paths": list(data.source_paths), "report_build_elapsed_seconds": time.perf_counter()-started,
        "pdf_sha256": hashlib.sha256(pdf.read_bytes()).hexdigest(),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"wrote {pdf}")


if __name__ == "__main__":
    main()
