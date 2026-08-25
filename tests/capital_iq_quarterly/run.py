"""Test observed quarterly Capital IQ N and build a formal result PDF."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import sys as _sys, pathlib as _pathlib  # noqa: E402
_ROOT = next(_p for _p in _pathlib.Path(__file__).resolve().parents if (_p / "pyproject.toml").exists())
_sys.path[:0] = [str(_ROOT), str(_ROOT / "src"), str(_ROOT / "tests")]
from tests import _bootstrap  # noqa: F401,E402
from tests._bootstrap import DATA_DIR  # noqa: E402

from nkpc_hsa.config import load_yaml  # noqa: E402
from tests.observed_hhi.functions import (  # noqa: E402
    ObservedHHISample,
    fit_observed_hhi_model,
    summarize_observed_fit,
    transform_inverse_hhi,
)

BUNDLE_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT = BUNDLE_DIR / "results"

BLUE = "#0072B2"
ORANGE = "#D55E00"
GREEN = "#009E73"
GREY = "#6B7280"


def _load_frame(config: dict) -> pd.DataFrame:
    path = DATA_DIR / "processed" / str(config["data"]["file"])
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run /Applications/StataNow/StataSE.app/Contents/MacOS/"
            "stata-se -b do build_data/run.do first."
        )
    frame = pd.read_csv(path, parse_dates=["DATE"])
    frame.index = pd.PeriodIndex(frame.pop("DATE"), freq="Q")
    required = {str(config["data"]["expectation"])}
    required.update(map(str, config["data"]["n_variants"].values()))
    for pair in config["columns"]["inflation"].values():
        required.update(map(str, pair))
    required.update(map(str, config["columns"]["activity"].values()))
    missing = sorted(required - set(frame.columns))
    if missing:
        raise KeyError(f"Capital IQ test input is missing columns: {missing}")
    return frame.sort_index()


def _cell(config: dict, cell_id: int) -> dict:
    for cell in config["cells"]:
        if int(cell["id"]) == cell_id:
            return dict(cell)
    raise ValueError(f"Unknown cell {cell_id}")


def _sample(frame: pd.DataFrame, config: dict, cell_id: int, variant: str) -> ObservedHHISample:
    cell = _cell(config, cell_id)
    y_col, lag_col = config["columns"]["inflation"][cell["inflation"]]
    x_col = str(config["columns"]["activity"][cell["activity"]])
    n_col = str(config["data"]["n_variants"][variant])
    expectation_col = str(config["data"]["expectation"])
    columns = [y_col, lag_col, expectation_col, x_col, n_col]
    selected = frame[columns].apply(pd.to_numeric, errors="coerce").dropna()
    if len(selected) < 32:
        raise ValueError(f"Cell {cell_id} / {variant} has only {len(selected)} complete quarters")
    return ObservedHHISample(
        periods=selected.index,
        y=selected[y_col].to_numpy(float),
        pi_lag=selected[lag_col].to_numpy(float),
        expectation=selected[expectation_col].to_numpy(float),
        activity=selected[x_col].to_numpy(float),
        q=transform_inverse_hhi(selected[n_col].to_numpy(float)),
        inflation=str(cell["inflation"]),
        activity_name=str(cell["activity"]),
        hhi_variant=variant,
    )


def _task_key(task: dict) -> str:
    return f"cell{task['cell']}__{task['variant']}__{task['error_model']}"


def _stable_seed(base: int, task: dict) -> int:
    digest = hashlib.sha256(json.dumps(task, sort_keys=True).encode()).hexdigest()
    return base + int(digest[:8], 16) % 1_000_000_000


def _save_fit(path: Path, fit) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {
        "coefficients": fit.coefficients,
        "sigma": fit.sigma,
        "names": np.asarray(fit.names),
        "periods": np.asarray(fit.periods),
    }
    arrays.update({f"aux__{key}": np.asarray(value) for key, value in fit.auxiliary.items()})
    np.savez_compressed(path, **arrays)


def _run_task(payload: dict) -> dict:
    config = load_yaml(payload["config_path"])
    frame = _load_frame(config)
    task = payload["task"]
    sample = _sample(frame, config, int(task["cell"]), str(task["variant"]))
    design = config["design"]
    sampling = payload["sampling"]
    fit = fit_observed_hhi_model(
        sample,
        cell=int(task["cell"]),
        fast_definition=str(design["fast_definition"]),
        environment_definition=str(design["environment_definition"]),
        timing=str(design["timing"]),
        model_variant=str(design["model_variant"]),
        error_model=str(task["error_model"]),
        iterations=int(sampling["iterations"]),
        warmup=int(sampling["warmup"]),
        thin=int(sampling["thin"]),
        chains=int(sampling["chains"]),
        seed=int(payload["seed"]),
        zeta_reference=float(design["zeta_reference"]),
        b_x=float(design["b_x"]),
    )
    if payload["draws_dir"]:
        _save_fit(Path(payload["draws_dir"]) / f"{_task_key(task)}.npz", fit)
    summary = summarize_observed_fit(fit)
    summary["cell_label"] = str(_cell(config, int(task["cell"]))["label"])
    names = fit.names
    kappa = fit.coefficients[:, :, names.index("kappa_1")]
    theta = fit.coefficients[:, :, names.index("theta_0")]
    delta = kappa - float(design["b_x"]) * float(design["zeta_reference"]) * theta
    derived = {
        "task_key": _task_key(task),
        "cell": int(task["cell"]),
        "cell_label": str(_cell(config, int(task["cell"]))["label"]),
        "variant": str(task["variant"]),
        "error_model": str(task["error_model"]),
        "n": len(fit.periods),
        "sample_start": fit.periods[0],
        "sample_end": fit.periods[-1],
        "condition_number": fit.design_condition_number,
        "theta_orthogonal_share": fit.theta_orthogonal_share,
        "delta_hsa_mean": float(np.mean(delta)),
        "delta_hsa_ci_2_5": float(np.quantile(delta, 0.025)),
        "delta_hsa_ci_97_5": float(np.quantile(delta, 0.975)),
        "delta_hsa_probability_positive": float(np.mean(delta > 0)),
    }
    return {"summary": summary.to_dict(orient="records"), "derived": derived}


def _tasks(config: dict) -> list[dict]:
    return [
        {"cell": int(cell["id"]), "variant": variant, "error_model": error}
        for cell in config["cells"]
        for variant in config["data"]["n_variants"]
        for error in config["design"]["error_models"]
    ]


def _input_audit(frame: pd.DataFrame, config: dict) -> pd.DataFrame:
    rows = []
    for variant, column in config["data"]["n_variants"].items():
        values = pd.to_numeric(frame[column], errors="coerce").dropna()
        q = transform_inverse_hhi(values.to_numpy(float))
        rows.append({
            "variant": variant,
            "column": column,
            "quarters": len(values),
            "first_quarter": str(values.index[0]),
            "last_quarter": str(values.index[-1]),
            "min_level": float(values.min()),
            "max_level": float(values.max()),
            "mean_level": float(values.mean()),
            "q_sd": float(np.std(q, ddof=1)),
        })
    return pd.DataFrame(rows)


def _gate(summary: pd.DataFrame, config: dict, quick: bool) -> tuple[bool, dict]:
    gate = config["gates"]["smoke" if quick else "production"]
    finite_columns = ["mean", "sd", "ci_2.5", "ci_97.5", "rhat", "bulk_ess"]
    finite = bool(np.isfinite(summary[finite_columns].to_numpy(float)).all())
    max_rhat = float(summary["rhat"].max())
    min_ess = float(summary["bulk_ess"].min())
    passed = finite and max_rhat <= float(gate["max_rhat"]) and min_ess >= float(gate["min_bulk_ess"])
    return passed, {
        "finite_outputs": finite,
        "max_rhat": max_rhat,
        "min_bulk_ess": min_ess,
        "required_max_rhat": float(gate["max_rhat"]),
        "required_min_bulk_ess": float(gate["min_bulk_ess"]),
    }


def _plots(frame: pd.DataFrame, config: dict, summary: pd.DataFrame, figures: Path) -> None:
    figures.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(9.2, 6.2), sharex=True)
    colors = {"firm_weighted": BLUE, "revenue_weighted": ORANGE}
    for variant, column in config["data"]["n_variants"].items():
        values = pd.to_numeric(frame[column], errors="coerce").dropna()
        axes[0].plot(values.index.to_timestamp(), values, lw=1.7, color=colors[variant], label=variant.replace("_", " "))
        q = pd.Series(transform_inverse_hhi(values.to_numpy(float)), index=values.index)
        axes[1].plot(q.index.to_timestamp(), q, lw=1.5, color=colors[variant], label=variant.replace("_", " "))
    axes[0].set_ylabel("Effective firms N")
    axes[0].set_title("Capital IQ quarterly competition aggregates")
    axes[0].legend(frameon=False, ncol=2)
    axes[1].axhline(0, color="black", lw=0.7)
    axes[1].set_ylabel("Centered 10 log(N)")
    axes[1].set_xlabel("Quarter")
    fig.tight_layout()
    fig.savefig(figures / "capital_iq_paths.png", dpi=220)
    plt.close(fig)

    primary_error = str(config["design"]["primary_error_model"])
    theta = summary[(summary["parameter"] == "theta_0") & (summary["error_model"] == primary_error)].copy()
    theta = theta.sort_values(["cell", "hhi_variant"])
    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    labels = []
    for index, (_, row) in enumerate(theta.iterrows()):
        color = colors[str(row["hhi_variant"])]
        ax.errorbar(
            row["mean"], index,
            xerr=[[row["mean"] - row["ci_2.5"]], [row["ci_97.5"] - row["mean"]]],
            fmt="o", capsize=3, color=color,
        )
        labels.append(f"{row['cell_label']} | {str(row['hhi_variant']).replace('_', ' ')}")
    ax.axvline(0, color="black", lw=0.8, ls="--")
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("Direct fast-N loading theta_0 (posterior mean and 95% interval)")
    ax.set_title("Primary persistent-AR(1) specifications")
    fig.tight_layout()
    fig.savefig(figures / "theta_primary_forest.png", dpi=220)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.8))
    axes[0].scatter(np.arange(len(summary)), summary["rhat"], s=10, color=BLUE)
    axes[0].axhline(float(config["gates"]["production"]["max_rhat"]), color=ORANGE, ls="--", lw=1)
    axes[0].set_ylabel("Rank-normalized R-hat")
    axes[0].set_xlabel("Coefficient fit")
    axes[1].scatter(np.arange(len(summary)), summary["bulk_ess"], s=10, color=GREEN)
    axes[1].axhline(float(config["gates"]["production"]["min_bulk_ess"]), color=ORANGE, ls="--", lw=1)
    axes[1].set_ylabel("Bulk ESS")
    axes[1].set_xlabel("Coefficient fit")
    fig.suptitle("MCMC convergence diagnostics")
    fig.tight_layout()
    fig.savefig(figures / "convergence_diagnostics.png", dpi=220)
    plt.close(fig)


def _tex_escape(value: object) -> str:
    return str(value).replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")


def _result_rows(summary: pd.DataFrame, config: dict) -> str:
    primary = str(config["design"]["primary_error_model"])
    selected = summary[(summary["error_model"] == primary) & (summary["parameter"] == "theta_0")]
    lines = []
    for _, row in selected.sort_values(["cell", "hhi_variant"]).iterrows():
        lines.append(
            f"{int(row['cell'])} & {_tex_escape(row['hhi_variant'])} & {int(row['n'])} & "
            f"{row['mean']:.3f} & [{row['ci_2.5']:.3f}, {row['ci_97.5']:.3f}] & "
            f"{row['sign_probability']:.3f} & {row['posterior_prior_sd_ratio']:.3f} \\\\"
        )
    return "\n".join(lines)


def _diagnostic_rows(summary: pd.DataFrame) -> str:
    grouped = summary.groupby(["cell", "hhi_variant", "error_model"], sort=True)
    lines = []
    for (cell, variant, error), rows in grouped:
        lines.append(
            f"{int(cell)} & {_tex_escape(variant)} & {_tex_escape(error)} & "
            f"{rows['rhat'].max():.3f} & {rows['bulk_ess'].min():.0f} & {rows['design_condition_number'].max():.1f} \\\\"
        )
    return "\n".join(lines)


def _write_report(out: Path, config: dict, audit: pd.DataFrame, summary: pd.DataFrame, derived: pd.DataFrame, gate: dict, quick: bool) -> Path:
    report = out / "capital_iq_quarterly_test_report.tex"
    status = "PASS" if gate["passed"] else "FAIL"
    primary = str(config["design"]["primary_error_model"])
    theta = summary[(summary["error_model"] == primary) & (summary["parameter"] == "theta_0")]
    strongest = theta.loc[theta["sign_probability"].idxmax()]
    delta_excludes = int(((derived["delta_hsa_ci_2_5"] > 0) | (derived["delta_hsa_ci_97_5"] < 0)).sum())
    audit_lines = "\n".join(
        f"{_tex_escape(row.variant)} & {int(row.quarters)} & {_tex_escape(row.first_quarter)} & {_tex_escape(row.last_quarter)} & "
        f"{row.min_level:.2f} & {row.max_level:.2f} \\\\"
        for row in audit.itertuples()
    )
    mode = "SMOKE - NOT FOR INFERENCE" if quick else "PRODUCTION"
    fast_def = str(config["design"]["fast_definition"])
    if fast_def.startswith("ewma_hl"):
        fast_desc = (
            f"the current one-sided EWMA forecast error with a "
            f"{_tex_escape(fast_def.removeprefix('ewma_hl'))}-quarter half-life"
        )
    elif fast_def == "ar2_innovation":
        fast_desc = (
            "the current AR(2) innovation, i.e. the residual of $q_t$ regressed on "
            "a constant, $q_{t-1}$, and $q_{t-2}$"
        )
    elif fast_def == "ar1_innovation":
        fast_desc = (
            "the current AR(1) innovation, i.e. the residual of $q_t$ regressed on "
            "a constant and $q_{t-1}$"
        )
    elif fast_def == "first_difference":
        fast_desc = "the current first difference $q_t-q_{t-1}$"
    else:
        fast_desc = _tex_escape(fast_def)
    tex = rf"""\documentclass[11pt]{{article}}
\usepackage[margin=0.80in]{{geometry}}
\usepackage{{booktabs,array,graphicx,xcolor,amsmath,microtype,hyperref,needspace}}
\usepackage{{newtxtext,newtxmath}}
\definecolor{{navy}}{{HTML}}{{17365D}}
\definecolor{{light}}{{HTML}}{{EEF3F8}}
\definecolor{{pass}}{{HTML}}{{007A5E}}
\hypersetup{{colorlinks=true,linkcolor=navy,urlcolor=navy}}
\setlength{{\parindent}}{{0pt}}
\setlength{{\parskip}}{{5pt}}
\begin{{document}}
\begin{{center}}
{{\color{{navy}}\LARGE\bfseries Capital IQ Observed Quarterly N Test}}\\[4pt]
{{\large Firm-weighted and revenue-weighted effective-firm aggregates}}\\[6pt]
\textbf{{{mode}}} \quad Revision: \texttt{{{_tex_escape(config['revision'])}}}
\end{{center}}

\colorbox{{light}}{{\parbox{{0.95\linewidth}}{{
\textbf{{Execution status: \textcolor{{pass}}{{{status}}}.}}
All outputs are finite; maximum rank-normalized $\widehat R={gate['max_rhat']:.4f}$
(limit {gate['required_max_rhat']:.2f}) and minimum bulk ESS is {gate['min_bulk_ess']:.0f}
(minimum {gate['required_min_bulk_ess']:.0f}). The largest primary sign probability is
{strongest['sign_probability']:.3f} for Cell {int(strongest['cell'])},
{_tex_escape(strongest['hhi_variant'])}; this is descriptive evidence, not a causal conclusion.
}}}}

\section*{{1. Data audit}}
Capital IQ company-quarter revenues are collapsed to market revenue-share HHIs,
seasonally adjusted within market, and aggregated before inversion. The test uses
the resulting observed quarters directly; it applies neither annual interpolation
nor the QCEW measurement block.

\begin{{center}}\small
\begin{{tabular}}{{lrrrrr}}
\toprule Aggregate & Quarters & First & Last & Min $N$ & Max $N$\\
\midrule
{audit_lines}
\bottomrule
\end{{tabular}}
\end{{center}}

\begin{{center}}\includegraphics[width=0.96\linewidth]{{figures/capital_iq_paths.png}}\end{{center}}

\Needspace{{12\baselineskip}}
\section*{{2. Estimating equation}}
Following the observed-HHI test bundle, the competition coordinate is
$q_t=10\log N_t$, centered in each complete-case sample. Its fast component
$c_t$ is {fast_desc}. Each cell estimates
\[
\pi_t=a+\beta_b\pi_{{t-1}}+\beta_fE_t\pi_{{t+1}}+\psi q_t
+(\kappa_0+\kappa_1q_t)x_t-\theta_0c_t+\varepsilon_t.
\]
The primary error is persistent AR(1); iid errors are retained as a sensitivity.
The three cells are PPI/inverse markup, CPI/HP output gap, and core CPI/negative
unemployment gap. The first cell is shorter because the inverse-markup input ends earlier.

\section*{{3. Primary direct-loading results}}
\begin{{center}}\small
\begin{{tabular}}{{r l r r r r r}}
\toprule Cell & Aggregate & $T$ & Mean & 95\% interval & Sign prob. & SD ratio\\
\midrule
{_result_rows(summary, config)}
\bottomrule
\end{{tabular}}
\end{{center}}

\begin{{center}}\includegraphics[width=0.97\linewidth]{{figures/theta_primary_forest.png}}\end{{center}}

Firm and revenue weighting change both the level and short-run variation of $N_t$.
Differences between paired estimates should therefore be read as aggregation
sensitivity. They do not select a preferred weighting ex post. The HSA residual
$\delta=\kappa_1-b_x\zeta_{{ref}}\theta_0$ has a 95\% interval excluding zero in
{delta_excludes} of {len(derived)} fitted specifications; the unrestricted fits do
not impose that cross-equation restriction.

\section*{{4. Convergence and numerical checks}}
\begin{{center}}\scriptsize
\begin{{tabular}}{{r l l r r r}}
\toprule Cell & Aggregate & Error & Max $\widehat R$ & Min bulk ESS & Condition no.\\
\midrule
{_diagnostic_rows(summary)}
\bottomrule
\end{{tabular}}
\end{{center}}

\begin{{center}}\includegraphics[width=0.92\linewidth]{{figures/convergence_diagnostics.png}}\end{{center}}

\section*{{5. Interpretation and reproducibility}}
This test establishes that both Capital IQ quarterly aggregates enter the
observed-competition pipeline without interpolation, produce finite posteriors,
and satisfy the declared convergence gate. It does not resolve aggregation choice,
activity endogeneity, expectation-measure mismatch, or causal identification.
Machine-readable coefficient summaries, HSA residuals, raw draws, figures, the
configuration, and a manifest are stored beside this report under
\texttt{{tests/capital\_iq\_quarterly/results/}}.

\end{{document}}
"""
    report.write_text(tex, encoding="utf-8")
    completed = subprocess.run(
        ["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error", report.name],
        cwd=out, check=False, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    if completed.returncode:
        raise RuntimeError(f"LaTeX report build failed:\n{completed.stdout[-4000:]}")
    return report.with_suffix(".pdf")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=BUNDLE_DIR / "config.yaml")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--no-draws", dest="save_draws", action="store_false")
    args = parser.parse_args()

    config = load_yaml(args.config)
    out = args.output_dir or (DEFAULT_OUTPUT / "smoke" if args.quick else DEFAULT_OUTPUT)
    tables, figures, draws = out / "tables", out / "figures", out / "draws"
    for directory in (tables, figures, draws):
        directory.mkdir(parents=True, exist_ok=True)
    frame = _load_frame(config)
    audit = _input_audit(frame, config)
    audit.to_csv(tables / "input_audit.csv", index=False)
    sampling = dict(config["sampling"])
    if args.quick:
        sampling.update(dict(config["quick_sampling"]))
    tasks = _tasks(config)
    print(f"Capital IQ tasks={len(tasks)} jobs={args.jobs} sampling={sampling}", flush=True)
    started = time.perf_counter()
    summaries = []
    derived_rows = []
    payloads = [
        {
            "config_path": str(args.config),
            "task": task,
            "sampling": sampling,
            "seed": _stable_seed(int(sampling["seed"]), task),
            "draws_dir": str(draws) if args.save_draws else None,
        }
        for task in tasks
    ]
    with ProcessPoolExecutor(max_workers=max(1, args.jobs)) as executor:
        futures = [executor.submit(_run_task, payload) for payload in payloads]
        for index, future in enumerate(as_completed(futures), 1):
            result = future.result()
            summaries.extend(result["summary"])
            derived_rows.append(result["derived"])
            print(f"[{index}/{len(futures)}] completed", flush=True)
    summary = pd.DataFrame(summaries).sort_values(["cell", "hhi_variant", "error_model", "parameter"])
    derived = pd.DataFrame(derived_rows).sort_values(["cell", "variant", "error_model"])
    summary.to_csv(tables / "coefficient_summaries.csv", index=False)
    derived.to_csv(tables / "specification_summaries.csv", index=False)
    passed, gate = _gate(summary, config, args.quick)
    gate["passed"] = passed
    _plots(frame, config, summary, figures)
    pdf = _write_report(out, config, audit, summary, derived, gate, args.quick)
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "revision": config["revision"],
        "mode": "smoke" if args.quick else "production",
        "config": str(args.config),
        "data": str(DATA_DIR / "processed" / config["data"]["file"]),
        "sampling": sampling,
        "tasks": len(tasks),
        "coefficient_rows": len(summary),
        "gate": gate,
        "elapsed_seconds": time.perf_counter() - started,
        "report": str(pdf),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"wrote {pdf}", flush=True)
    print(f"gate={'PASS' if passed else 'FAIL'} max_rhat={gate['max_rhat']:.4f} min_bulk_ess={gate['min_bulk_ess']:.0f}", flush=True)
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
