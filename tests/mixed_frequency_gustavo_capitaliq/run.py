"""Run the cut mixed-frequency Gustavo x Capital-IQ diagnostic bundle."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logsumexp

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists())
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT / "tests")]
from tests import _bootstrap  # noqa: F401,E402
from nkpc_hsa.config import load_yaml  # noqa: E402
from nkpc_hsa.paths import data_root  # noqa: E402
from tests.gustavo_state_capitaliq_cycle.functions import (  # noqa: E402
    CycleFit,
    fit_qoq_theta,
    load_nkpc_cells,
    load_oil_controls,
    qoq_pointwise_loglik,
    save_qoq,
    summarize_qoq,
)
from tests.mixed_frequency_gustavo_capitaliq.functions import (  # noqa: E402
    blocked_backtest,
    fit_measurement,
    load_measurement_data,
    summarize_parameters,
)

BUNDLE = Path(__file__).resolve().parent
BASE_NKPC_CONFIG = ROOT / "tests" / "gustavo_state_capitaliq_cycle" / "config.yaml"


def _json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=ROOT, text=True, stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL, check=False,
    ).stdout.strip()


def _sampling_override(profile: dict) -> dict[str, int]:
    return {
        "iterations": int(profile["nkpc_iterations"]),
        "warmup": int(profile["nkpc_warmup"]),
        "thin": int(profile["nkpc_thin"]),
        "chains": int(profile["nkpc_chains"]),
    }


def _waic(loglik: np.ndarray) -> dict[str, float]:
    flat = np.asarray(loglik).reshape(-1, loglik.shape[-1])
    point_lppd = logsumexp(flat, axis=0) - np.log(len(flat))
    point_penalty = np.var(flat, axis=0, ddof=1)
    point_elpd = point_lppd - point_penalty
    return {
        "elpd_waic": float(point_elpd.sum()),
        "waic": float(-2.0 * point_elpd.sum()),
        "p_waic": float(point_penalty.sum()),
        "elpd_se": float(np.sqrt(len(point_elpd) * np.var(point_elpd, ddof=1))),
    }


def _write_results_markdown(out: Path, manifest: dict, coefficient: pd.DataFrame,
                            backtest: pd.DataFrame, comparison: pd.DataFrame) -> None:
    structural = coefficient[coefficient.parameter.isin(["theta_N", "delta"])]
    lines = [
        "# Saved results", "",
        "> **NOT FOR INFERENCE.** This is a mock code-path and identification diagnostic.", "",
        f"- Profile: `{manifest['profile']}`",
        f"- Created: `{manifest['created_utc']}`",
        f"- Completed NKPC cells: `{manifest['completed_models']}`",
        f"- Overall mock gate: `{'PASS' if manifest['gate']['passed'] else 'FAIL'}`", "",
        "## Structural coefficients", "",
        "| Model | Oil | Parameter | Mean | 95% interval | P(>0) | Post/prior SD | R-hat | Bulk ESS |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in structural.iterrows():
        lines.append(
            f"| {row['model']} | {row['oil']} | {row['parameter']} | {row['mean']:.3f} | "
            f"[{row['q2.5']:.3f}, {row['q97.5']:.3f}] | {row['p_positive']:.3f} | "
            f"{row['posterior_prior_sd_ratio']:.3f} | {row['rhat']:.3f} | {row['ess_bulk']:.0f} |"
        )
    aggregate = backtest.groupby("method", as_index=False).agg(rmse=("rmse", "mean"), mae=("mae", "mean"))
    lines.extend(["", "## Capital IQ blocked backtest", "", "| Method | Mean RMSE | Mean MAE |", "|---|---:|---:|"])
    for row in aggregate.itertuples():
        lines.append(f"| {row.method} | {row.rmse:.3f} | {row.mae:.3f} |")
    lines.extend(["", "## Predictive comparison", "", "| Model | Oil | ELPD-WAIC | WAIC |", "|---|---|---:|---:|"])
    for row in comparison.itertuples():
        lines.append(f"| {row.model} | {row.oil} | {row.elpd_waic:.2f} | {row.waic:.2f} |")
    lines.extend([
        "", "## Interpretation", "",
        "1. The run establishes whether the new mixed-frequency likelihood is computationally coherent and whether it predicts held-out Capital IQ growth better than mechanical annual allocation.",
        "2. It does not establish an HSA channel: a mock profile is too short for inference, and no HSA restriction or lambda is estimated here.",
        "3. The model can replace the mechanically allocated quarterly N only if the blocked measurement backtest and later quick/full identification gates pass.", "",
    ])
    (out / "RESULTS.md").write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("mock", "quick", "full"), default="mock")
    parser.add_argument("--no-report", action="store_true")
    args = parser.parse_args()
    started = time.time()
    cfg = load_yaml(BUNDLE / "config.yaml")
    sampling = cfg["sampling"][args.profile]
    seed = int(cfg["sampling"]["seed"])
    out = BUNDLE / "results" / args.profile
    for folder in (out, out / "draws", out / "tables", out / "report"):
        folder.mkdir(parents=True, exist_ok=True)

    data = load_measurement_data(cfg)
    print("Fitting measurement-only mixed-frequency state...", flush=True)
    state = fit_measurement(data, cfg, sampling, seed)
    np.savez_compressed(
        out / "draws" / "state.npz", periods=state.periods,
        parameter_names=state.parameter_names, parameters=state.parameters,
        nbar=state.nbar, nhat=state.nhat, n_total=state.n_total,
        average_weights=state.average_weights,
    )
    _json(out / "draws" / "state.json", {
        "diagnostics": state.diagnostics,
        "map_parameters": state.map_parameters,
    })
    state_parameters = summarize_parameters(state)
    state_parameters.to_csv(out / "tables" / "state_parameters.csv", index=False)

    state_paths = pd.DataFrame({"period": state.periods, "gustavo": data.gustavo})
    for name, values in (("nbar", state.nbar), ("nhat", state.nhat), ("n_total", state.n_total)):
        state_paths[f"{name}_mean"] = values.mean(axis=(0, 1))
        state_paths[f"{name}_q2.5"] = np.percentile(values, 2.5, axis=(0, 1))
        state_paths[f"{name}_q97.5"] = np.percentile(values, 97.5, axis=(0, 1))
    state_paths.to_csv(out / "tables" / "state_paths.csv", index=False)

    print("Running predeclared blocked Capital IQ backtest...", flush=True)
    backtest = blocked_backtest(data, cfg)
    backtest.to_csv(out / "tables" / "backtest.csv", index=False)
    backtest_summary = backtest.groupby(["method", "series"], as_index=False).agg(
        n=("n", "sum"), rmse=("rmse", "mean"), mae=("mae", "mean"),
        mean_log_score=("mean_log_score", "mean"),
    )
    backtest_summary.to_csv(out / "tables" / "backtest_summary.csv", index=False)

    base_cfg = load_yaml(BASE_NKPC_CONFIG)
    cell = load_nkpc_cells(base_cfg)["ppi_inverse_markup"]
    oil_controls, oil_metadata = load_oil_controls(cell.periods)
    propagated = CycleFit(
        "mixed_frequency", state.periods, state.parameter_names, state.parameters,
        state.nbar, state.nhat, state.diagnostics,
        np.nanmean(np.vstack([data.ciq_growth[name] for name in data.ciq_growth]), axis=0),
    )
    coefficient_rows: list[dict] = []
    convergence_rows: list[dict] = []
    prior_rows: list[dict] = []
    comparison_rows: list[dict] = []
    summaries: list[dict] = []
    for model_index, model in enumerate(cfg["nkpc"]["models"]):
        for oil_index, oil in enumerate(cfg["nkpc"]["oil"]):
            controls = oil_controls if oil == "with_oil" else None
            print(f"Fitting NKPC: {model}, {oil}...", flush=True)
            fit = fit_qoq_theta(
                cell, propagated, base_cfg,
                seed + 100003 * (1 + 2 * model_index + oil_index),
                error_model="iid", include_delta=model == "free_static_combined",
                sampling_override=_sampling_override(sampling), controls=controls,
            )
            path = out / "draws" / f"{model}_{oil}.npz"
            save_qoq(path, fit)
            summary = summarize_qoq(fit)
            summaries.append(summary)
            _json(path.with_suffix(".json"), summary)
            for parameter, values in summary["coefficients"].items():
                display = "theta_N" if parameter == "theta_CIQ" else parameter
                coefficient_rows.append({"model": model, "oil": oil, "parameter": display, **values})
                prior_rows.append({
                    "model": model, "oil": oil, "parameter": display,
                    "prior_mean": values["prior_mean"], "prior_sd": values["prior_sd"],
                    "posterior_mean": values["mean"], "posterior_sd": values["sd"],
                    "posterior_prior_sd_ratio": values["posterior_prior_sd_ratio"],
                })
                convergence_rows.append({
                    "block": "nkpc", "model": model, "oil": oil, "parameter": display,
                    "rhat": values["rhat"], "ess_bulk": values["ess_bulk"], "ess_tail": values["ess_tail"],
                })
            score = _waic(qoq_pointwise_loglik(cell, fit, controls))
            comparison_rows.append({"model": model, "oil": oil, **score})

    coefficient = pd.DataFrame(coefficient_rows)
    prior = pd.DataFrame(prior_rows)
    convergence = pd.DataFrame(convergence_rows)
    for row in state_parameters.itertuples():
        convergence.loc[len(convergence)] = {
            "block": "measurement", "model": "mixed_frequency_state", "oil": "none",
            "parameter": row.parameter, "rhat": row.rhat,
            "ess_bulk": row.ess_bulk, "ess_tail": row.ess_tail,
        }
    comparison = pd.DataFrame(comparison_rows)
    best = float(comparison.elpd_waic.max())
    comparison["delta_elpd_from_best"] = comparison.elpd_waic - best
    coefficient.to_csv(out / "tables" / "coefficients.csv", index=False)
    prior.to_csv(out / "tables" / "prior_posterior.csv", index=False)
    convergence.to_csv(out / "tables" / "convergence.csv", index=False)
    comparison.to_csv(out / "tables" / "model_comparison.csv", index=False)

    gate_cfg = cfg["gates"]
    rhat_limit = float(gate_cfg[f"{args.profile}_max_rhat"])
    ess_limit = float(gate_cfg[f"{args.profile}_min_bulk_ess"])
    observed_max_rhat = float(convergence.rhat.max())
    observed_min_ess = float(convergence.ess_bulk.min())
    aggregate_bt = backtest.groupby("method").rmse.mean()
    relative_improvement = float(
        (aggregate_bt["average_allocation"] - aggregate_bt["mixed_frequency_state"])
        / aggregate_bt["average_allocation"]
    )
    gate = {
        "rhat_limit": rhat_limit,
        "observed_max_rhat": observed_max_rhat,
        "ess_bulk_limit": ess_limit,
        "observed_min_bulk_ess": observed_min_ess,
        "q4_anchor_error_limit": float(gate_cfg["max_q4_anchor_error"]),
        "observed_q4_anchor_error": state.diagnostics["max_q4_anchor_error"],
        "backtest_relative_rmse_improvement_required": float(gate_cfg["backtest_required_relative_rmse_improvement"]),
        "backtest_relative_rmse_improvement": relative_improvement,
    }
    gate["convergence_passed"] = observed_max_rhat <= rhat_limit and observed_min_ess >= ess_limit
    gate["anchor_passed"] = state.diagnostics["max_q4_anchor_error"] <= gate["q4_anchor_error_limit"]
    gate["backtest_passed"] = relative_improvement >= gate["backtest_relative_rmse_improvement_required"]
    gate["passed"] = bool(gate["convergence_passed"] and gate["anchor_passed"] and gate["backtest_passed"])

    model_ready = data_root() / "processed" / "model_ready.csv"
    manifest = {
        "revision": cfg["revision"], "profile": args.profile,
        "is_test_run": True, "not_for_inference": args.profile != "full",
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "elapsed_seconds": time.time() - started, "seed": seed,
        "sampling": sampling, "completed_models": len(summaries),
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
        "config_sha256": _sha(BUNDLE / "config.yaml"),
        "data_path": str(model_ready), "data_sha256": _sha(model_ready),
        "sample": {
            "state": [str(data.periods[0]), str(data.periods[-1]), len(data.periods)],
            "gustavo_q4_observations": int(np.isfinite(data.gustavo).sum()),
            "capital_iq_growth_observations": {k: int(np.isfinite(v).sum()) for k, v in data.ciq_growth.items()},
            "nkpc": [str(cell.periods[0]), str(cell.periods[-1]), len(cell.periods)],
        },
        "oil": oil_metadata, "gate": gate,
        "report": str(out / "report" / f"mixed_frequency_gustavo_capitaliq_{args.profile}.pdf"),
        "interpretation_rule": "The measurement backtest and free theta_N identification must pass before any HSA restriction is estimated.",
    }
    _json(out / "manifest.json", manifest)
    _write_results_markdown(out, manifest, coefficient, backtest, comparison)
    print(json.dumps(gate, indent=2), flush=True)
    if not args.no_report:
        from tests.mixed_frequency_gustavo_capitaliq.build_report import build
        build(args.profile)


if __name__ == "__main__":
    main()
