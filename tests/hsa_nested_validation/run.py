"""Run the exact-N nested-validation ladder.

Mock is a structural/software check. Quick is a short convergence check. Full is
the confirmatory estimation and is intentionally not run automatically.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists())
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT / "tests")]
from tests import _bootstrap  # noqa: F401,E402
from nkpc_hsa.config import load_yaml  # noqa:E402
from nkpc_hsa.paths import data_root  # noqa:E402
from tests.hsa_nested_validation.functions import (  # noqa:E402
    build_model_specs, comparison_metrics, fit_model,
    load_experiment, restriction_diagnostics, save_fit, summarize_fit,
)
import arviz as az  # noqa:E402


BUNDLE = Path(__file__).resolve().parent


def _hash(path: Path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _worker(job):
    experiment, cell, spec, config, sampling, seed = job
    return spec.model_id, fit_model(experiment, cell, spec, config, sampling, seed)


def _blocked_psis(block_ll: np.ndarray):
    # One row per posterior draw, one column per four-quarter block.
    idata = az.from_dict(
        {"posterior": {"dummy": np.zeros((1, block_ll.shape[0]))},
         "log_likelihood": {"inflation_block": block_ll[None, :, :]}},
        sample_dims=["chain", "draw"],
    )
    loo = az.loo(idata, var_name="inflation_block", pointwise=True, reff=1.0)
    pareto_k = np.asarray(loo.pareto_k)
    return {
        "elpd_blocked_psis": float(loo.elpd),
        "se_elpd": float(loo.se),
        "max_pareto_k": float(np.max(pareto_k)),
        "n_blocks_pareto_k_gt_0_7": int(np.sum(pareto_k > 0.7)),
        "requires_exact_refit": bool(np.any(pareto_k > 0.7)),
    }


def _fit_ladder(experiment, cell, specs, config, sampling, seed, workers, output):
    jobs = [
        (experiment, cell, spec, config, sampling, seed + 104729 * (i + 1))
        for i, spec in enumerate(specs)
    ]
    fits = {}
    with ProcessPoolExecutor(max_workers=min(workers, len(jobs))) as pool:
        futures = {pool.submit(_worker, job): job[2].model_id for job in jobs}
        for future in as_completed(futures):
            model_id, fit = future.result(); fits[model_id] = fit
            save_fit(output / "draws" / "joint_state_split" / cell.role / f"{model_id}.npz", fit)
            print(f"[joint_state_split/{cell.role}/{model_id}] Rhat={fit.diagnostics['max_rhat']:.3f} ", end="", flush=True)
            print(f"identity={fit.diagnostics['exact_identity_error']:.2e}", flush=True)
    return fits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("mock", "quick", "full"), default="mock")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    config = load_yaml(BUNDLE / "config.yaml"); sampling = config["sampling"][args.mode]
    treatments = ("joint_state_split",)
    output = BUNDLE / "results" / args.mode; output.mkdir(parents=True, exist_ok=True)
    started = time.time(); experiment = load_experiment(config)
    primary_specs, benchmark_specs = build_model_specs(config)
    cells = experiment.cells

    data_path = data_root() / "processed" / "model_ready.csv"
    config_path = BUNDLE / "config.yaml"
    preflight = {
        "data_path": str(data_path), "data_sha256": _hash(data_path),
        "config_sha256": _hash(config_path),
        "positive_annual_counts": bool(np.all(experiment.allocation.annual.to_numpy() > 0)),
        "annual_observations_are_q4": True,
        "allocation_mean_anchor_error": experiment.allocation_summary["max_mean_path_anchor_error"],
        "samples": {
            cell_id: [str(cell.periods[0]), str(cell.periods[-1]), cell.n_periods]
            for cell_id, cell in cells.items()
        },
    }
    if preflight["allocation_mean_anchor_error"] > config["gates"]["max_exact_identity_error"]:
        raise RuntimeError("Allocation anchor preflight failed")

    fits_all = {}; summaries = {}; metrics_all = {}
    for treatment_index, treatment in enumerate(treatments):
        for cell_index, (cell_id, cell) in enumerate(cells.items()):
            specs = primary_specs if cell.activity_role == "negative_unemployment_gap" else benchmark_specs
            fits = _fit_ladder(
                experiment, cell, specs, config, sampling,
                int(config["sampling"]["seed"]) + 1000000 * treatment_index + 10000 * cell_index,
                args.workers, output,
            )
            for model_id, fit in fits.items():
                key = f"{treatment}/{cell_id}/{model_id}"; fits_all[key] = fit
                summaries[key] = summarize_fit(fit)
                metric = comparison_metrics(fit, cell); block_ll = metric.pop("block_loglik")
                ll_path = output / "log_likelihood" / treatment / cell_id / f"{model_id}.npz"
                ll_path.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(ll_path, block_loglik=block_ll)
                metric.update(_blocked_psis(block_ll))
                metric["formal_log_marginal_likelihood"] = None
                metric["formal_bayes_factor_status"] = "not_run_until_simulation_validation"
                metrics_all[key] = metric

    restriction = {}
    for cell_id, cell in cells.items():
        if cell.activity_role != "inverse_markup":
            continue
        key = f"joint_state_split/{cell_id}/free_static_combined"
        if key in fits_all:
            restriction[cell_id] = restriction_diagnostics(
                fits_all[key], cell,
                list(map(float, config["models"]["hsa_lambda_grid"])),
                float(config["restriction_equivalence"]["rms_inflation_pp"]),
            )

    # Relative blocked-elpd table within each price/activity cell.
    comparisons = {}
    for treatment in treatments:
        for cell_id in cells:
            prefix = f"{treatment}/{cell_id}/"; baseline = metrics_all[prefix + "ces"]
            for key, metric in metrics_all.items():
                if key.startswith(prefix):
                    comparisons[key] = {
                        "delta_elpd_vs_ces": metric["elpd_blocked_psis"] - baseline["elpd_blocked_psis"],
                        "se_delta_placeholder_conservative": float(np.hypot(metric["se_elpd"], baseline["se_elpd"])),
                    }

    confirmatory_fits = [fit for fit in fits_all.values() if fit.spec.model_id != "free_lambda_diagnostic"]
    diagnostic_fits = [fit for fit in fits_all.values() if fit.spec.model_id == "free_lambda_diagnostic"]
    max_rhat = max([fit.diagnostics["max_rhat"] for fit in confirmatory_fits] +
                   [1.0])
    diagnostic_max_rhat = max([fit.diagnostics["max_rhat"] for fit in diagnostic_fits] + [1.0])
    max_identity = max([fit.diagnostics["exact_identity_error"] for fit in fits_all.values()] +
                       [0.0])
    gate_rhat = float(config["gates"][f"{args.mode}_max_rhat"] if args.mode != "full" else config["gates"]["full_max_rhat"])
    manifest = {
        "revision": config["revision"], "mode": args.mode, "mock_not_substantive": args.mode == "mock",
        "elapsed_seconds": time.time() - started, "sampling": dict(sampling), "treatments": treatments,
        "preflight": preflight, "allocation": experiment.allocation_summary,
        "models": {key: {"coefficients": summaries[key], "metrics": metrics_all[key],
                         "comparison": comparisons[key], "diagnostics": fits_all[key].diagnostics,
                         "lambda_fixed": fits_all[key].spec.lambda_fixed,
                         "free_lambda": fits_all[key].spec.free_lambda}
                   for key in sorted(fits_all)},
        "restriction_diagnostics": restriction,
        "formal_marginal_likelihood": {"status": "not_run", "reason": "requires simulation-validated state-integrating estimator"},
        "annual_origin_forecast": {
            "status": "not_run_in_mock_or_estimation_command",
            "reason": "Forecast evaluation starts only after a full fit passes its convergence gate",
        },
        "gate": {"rhat_required": gate_rhat, "max_rhat": max_rhat,
                 "free_lambda_diagnostic_max_rhat_not_gating": diagnostic_max_rhat,
                 "identity_required": float(config["gates"]["max_exact_identity_error"]),
                 "max_identity_error": max_identity,
                 "passed": bool(max_rhat <= gate_rhat and max_identity <= float(config["gates"]["max_exact_identity_error"]))},
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"wrote {output / 'manifest.json'}", flush=True)
    print(f"gate passed={manifest['gate']['passed']} max_rhat={max_rhat:.3f}", flush=True)


if __name__ == "__main__":
    main()
