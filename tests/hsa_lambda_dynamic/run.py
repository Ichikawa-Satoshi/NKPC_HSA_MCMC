"""Run the estimated-lambda HSA model comparison.

Usage:
  python tests/hsa_lambda_dynamic/run.py --quick
  python tests/hsa_lambda_dynamic/run.py
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import time

import numpy as np

import sys as _sys, pathlib as _pathlib
_ROOT = next(p for p in _pathlib.Path(__file__).resolve().parents if (p / "pyproject.toml").exists())
_sys.path[:0] = [str(_ROOT), str(_ROOT / "src"), str(_ROOT / "tests")]
from tests import _bootstrap  # noqa: F401,E402
from nkpc_hsa.config import load_yaml  # noqa: E402
from tests.hsa_lambda_dynamic.functions import (  # noqa: E402
    MODEL_LABELS, approximate_log_marginal, comparison_metrics, fit_model,
    load_experiment_data, load_fit, save_fit,
)

BUNDLE = Path(__file__).resolve().parent


def _summary(fit):
    flat = fit.draws.reshape(-1, fit.draws.shape[-1])
    rows = {}
    for i, name in enumerate(fit.names):
        d = flat[:, i]
        rows[name] = {
            "mean": float(d.mean()), "sd": float(d.std(ddof=1)),
            "q2.5": float(np.percentile(d, 2.5)), "q97.5": float(np.percentile(d, 97.5)),
            "p_positive": float(np.mean(d > 0)), "rhat": float(fit.diagnostics["rhat"][name]),
        }
    if fit.model == "hsa_static":
        lam, th = flat[:, fit.names.index("lambda")], flat[:, fit.names.index("theta_0")]
        derived = {"delta": lam * th}
    elif fit.model == "hsa_dynamic":
        lam = flat[:, fit.names.index("lambda")]
        th = flat[:, fit.names.index("theta_0")]
        ga = flat[:, fit.names.index("gamma")]
        derived = {"delta_1": lam * th, "delta_2": 0.5 * lam * ga}
    else:
        derived = {}
    for name, d in derived.items():
        rows[name + "_derived"] = {
            "mean": float(d.mean()), "sd": float(d.std(ddof=1)),
            "q2.5": float(np.percentile(d, 2.5)), "q97.5": float(np.percentile(d, 97.5)),
            "p_positive": float(np.mean(d > 0)), "rhat": None,
        }
    return rows


def _fit_one(args):
    model, data, sampling, cfg, seed = args
    fit = fit_model(
        data, model, iterations=int(sampling["iterations"]), warmup=int(sampling["warmup"]),
        thin=int(sampling["thin"]), chains=int(sampling["chains"]), seed=seed,
        coefficient_scale=float(cfg["priors"]["coefficient_scale"]),
        lambda_mean=float(cfg["priors"]["lambda_mean"]), lambda_sd=float(cfg["priors"]["lambda_sd"]),
        phi_mean=float(cfg["priors"]["phi_mean"]), phi_sd=float(cfg["priors"]["phi_sd"]),
        rho_lower=float(cfg["priors"]["rho_lower"]), rho_upper=float(cfg["priors"]["rho_upper"]),
    )
    return model, fit


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--summarize-only", action="store_true",
                        help="reuse saved draws and recompute summaries/comparison metrics")
    args = parser.parse_args()
    cfg = load_yaml(BUNDLE / "config.yaml")
    mode = "quick" if args.quick else "full"
    sampling = cfg["sampling"][mode]
    out = BUNDLE / "results" / ("smoke" if args.quick else "full")
    draws_dir = out / "draws"; draws_dir.mkdir(parents=True, exist_ok=True)
    exp = load_experiment_data(cfg); data = exp.case
    started = time.time()
    fits = {}
    if args.summarize_only:
        old = json.loads((out / "manifest.json").read_text())
        fits = {m: load_fit(draws_dir / f"{m}.npz", old["results"][m]["diagnostics"])
                for m in cfg["models"]}
    else:
        jobs = [(m, data, sampling, cfg, int(cfg["sampling"]["seed"]) + 1000 * i)
                for i, m in enumerate(cfg["models"])]
        with ProcessPoolExecutor(max_workers=min(args.workers, len(jobs))) as pool:
            futures = {pool.submit(_fit_one, job): job[0] for job in jobs}
            for future in as_completed(futures):
                model, fit = future.result(); fits[model] = fit
                save_fit(draws_dir / f"{model}.npz", fit)
                print(f"[{model}] max Rhat={fit.diagnostics['max_rhat']:.3f}", flush=True)

    results = {}
    for index, model in enumerate(cfg["models"]):
        fit = fits[model]
        metrics = comparison_metrics(fit, data)
        metrics["log_marginal_laplace_pf"] = approximate_log_marginal(
            fit, data, particles=int(sampling["particles"]),
            seed=int(cfg["sampling"]["seed"]) + 70000 + index,
        )
        results[model] = {
            "label": MODEL_LABELS[model], "coefficients": _summary(fit),
            "metrics": metrics, "diagnostics": fit.diagnostics,
        }
        print(f"[{model}] WAIC={metrics['waic']:.1f} logML={metrics['log_marginal_laplace_pf']:.1f} "
              f"RMSE={metrics['predictive_rmse']:.3f}", flush=True)

    allocation = exp.allocation
    manifest = {
        "revision": cfg["revision"], "mode": mode,
        "sample": {"first": str(data.periods[0]), "last": str(data.periods[-1]), "n": data.n_periods},
        "sampling": dict(sampling), "elapsed_seconds": time.time() - started,
        "allocation": {
            "average_weights": allocation.average_weights.tolist(),
            "max_anchor_error": allocation.max_anchor_error,
            "source_counts": {s: list(allocation.source.values()).count(s) for s in sorted(set(allocation.source.values()))},
            "coherence": {str(k): v for k, v in allocation.coherence.items()},
            "raw_weights": {str(k): v.tolist() for k, v in allocation.raw_weights.items()},
            "used_weights": {str(k): v.tolist() for k, v in allocation.used_weights.items()},
        },
        "results": results,
        "gate": {
            "required_max_rhat": float(cfg["gates"]["max_rhat"]),
            "passed": bool(all(v["diagnostics"]["max_rhat"] <= float(cfg["gates"]["max_rhat"])
                                for v in results.values())),
        },
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"wrote {out / 'manifest.json'}", flush=True)


if __name__ == "__main__":
    main()
