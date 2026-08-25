"""Targeted extension for one fit in an existing nested-validation run."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists())
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT / "tests")]
from tests import _bootstrap  # noqa: F401,E402
from nkpc_hsa.config import load_yaml  # noqa:E402
from tests.hsa_nested_validation.functions import (  # noqa:E402
    build_model_specs,
    comparison_metrics,
    fit_model,
    load_experiment,
    save_fit,
    summarize_fit,
)
from tests.hsa_nested_validation.run import _blocked_psis  # noqa:E402


BUNDLE = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("quick", "full"), default="full")
    parser.add_argument("--cell", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--iterations", type=int, required=True)
    parser.add_argument("--warmup", type=int, required=True)
    parser.add_argument("--thin", type=int, required=True)
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--force", action="store_true", help="accept even when max R-hat does not improve")
    args = parser.parse_args()

    result = BUNDLE / "results" / args.mode
    manifest_path = result / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Run the base estimation first: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    key = f"joint_state_split/{args.cell}/{args.model}"
    if key not in manifest["models"]:
        raise KeyError(f"Unknown saved fit: {key}")

    config = load_yaml(BUNDLE / "config.yaml")
    experiment = load_experiment(config)
    if args.cell not in experiment.cells:
        raise KeyError(f"Unknown cell: {args.cell}")
    cell = experiment.cells[args.cell]
    primary, benchmark = build_model_specs(config)
    candidates = primary if cell.activity_role == "negative_unemployment_gap" else benchmark
    spec = next((item for item in candidates if item.model_id == args.model), None)
    if spec is None:
        raise KeyError(f"Model {args.model} is not defined for {args.cell}")

    sampling = {
        "iterations": args.iterations,
        "warmup": args.warmup,
        "thin": args.thin,
        "chains": args.chains,
    }
    started = time.time()
    fit = fit_model(experiment, cell, spec, config, sampling, args.seed)
    old = manifest["models"][key]
    improved = fit.diagnostics["max_rhat"] < old["diagnostics"]["max_rhat"]
    if not improved and not args.force:
        raise RuntimeError(
            f"Refit max R-hat {fit.diagnostics['max_rhat']:.6f} did not improve "
            f"on {old['diagnostics']['max_rhat']:.6f}; saved result was not replaced"
        )

    metric = comparison_metrics(fit, cell)
    block_ll = metric.pop("block_loglik")
    metric.update(_blocked_psis(block_ll))
    metric["formal_log_marginal_likelihood"] = None
    metric["formal_bayes_factor_status"] = "not_run_until_simulation_validation"
    baseline = manifest["models"][f"joint_state_split/{args.cell}/ces"]["metrics"]
    comparison = {
        "delta_elpd_vs_ces": metric["elpd_blocked_psis"] - baseline["elpd_blocked_psis"],
        "se_delta_placeholder_conservative": float(np.hypot(metric["se_elpd"], baseline["se_elpd"])),
    }

    stamp = time.strftime("%Y%m%dT%H%M%S")
    backup = result / "refits" / f"backup_{stamp}_{args.cell}_{args.model}"
    backup.mkdir(parents=True, exist_ok=False)
    target = result / "draws" / "joint_state_split" / args.cell / f"{args.model}.npz"
    ll_target = result / "log_likelihood" / "joint_state_split" / args.cell / f"{args.model}.npz"
    shutil.copy2(target, backup / "fit.npz")
    shutil.copy2(ll_target, backup / "block_loglik.npz")
    shutil.copy2(manifest_path, backup / "manifest.json")
    save_fit(target, fit)
    np.savez_compressed(ll_target, block_loglik=block_ll)

    manifest["models"][key] = {
        "coefficients": summarize_fit(fit),
        "metrics": metric,
        "comparison": comparison,
        "diagnostics": fit.diagnostics,
        "lambda_fixed": fit.spec.lambda_fixed,
        "free_lambda": fit.spec.free_lambda,
        "sampling_override": sampling,
    }
    manifest.setdefault("refits", []).append({
        "fit": key,
        "reason": "targeted convergence extension",
        "sampling": sampling,
        "seed": args.seed,
        "elapsed_seconds": time.time() - started,
        "initial_max_rhat": old["diagnostics"]["max_rhat"],
        "replacement_max_rhat": fit.diagnostics["max_rhat"],
        "backup": str(backup.relative_to(result)),
    })
    confirmatory = [
        value for saved_key, value in manifest["models"].items()
        if not saved_key.endswith("/free_lambda_diagnostic")
    ]
    manifest["gate"]["max_rhat"] = max(value["diagnostics"]["max_rhat"] for value in confirmatory)
    manifest["gate"]["max_identity_error"] = max(
        value["diagnostics"]["exact_identity_error"] for value in manifest["models"].values()
    )
    gate = manifest["gate"]
    gate["passed"] = bool(
        gate["max_rhat"] <= gate["rhat_required"]
        and gate["max_identity_error"] <= gate["identity_required"]
    )
    temporary = manifest_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    temporary.replace(manifest_path)
    print(json.dumps({"fit": key, "diagnostics": fit.diagnostics, "gate": gate}, indent=2))


if __name__ == "__main__":
    main()
