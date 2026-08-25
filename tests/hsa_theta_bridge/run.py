"""Run the controlled 2x2 HSA theta bridge validation."""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
import pathlib
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

ROOT = next(p for p in pathlib.Path(__file__).resolve().parents if (p / "pyproject.toml").exists())
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT / "tests")]
from tests import _bootstrap  # noqa: F401,E402
from nkpc_hsa.config import load_yaml  # noqa:E402
from tests.hsa_exact_n_decomposition.functions import fit_states, load_exact_data  # noqa:E402
from tests.hsa_theta_bridge.functions import CELL_SPECS, fit_bridge, save_fit, summarize_fit  # noqa:E402


BUNDLE = pathlib.Path(__file__).resolve().parent


def _worker(args):
    return args[2], fit_bridge(*args)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    cfg = load_yaml(BUNDLE / "config.yaml")
    mode = "quick" if args.quick else "full"
    sampling = cfg["sampling"][mode]
    output = BUNDLE / "results" / ("smoke" if args.quick else "full")
    draw_dir = output / "draws"
    draw_dir.mkdir(parents=True, exist_ok=True)
    exact = load_exact_data(cfg)
    # Hold the allocation posterior at its mean so the 2x2 cells differ only in
    # lambda treatment and whether inflation conditions the exact state split.
    zero_chols = {year: value * 0.0 for year, value in exact.allocation.chol_weights.items()}
    exact = replace(exact, allocation=replace(exact.allocation, chol_weights=zero_chols))
    started = time.time()

    # One common cut-state posterior is shared by both cut cells.
    states = fit_states(exact, cfg, sampling, int(cfg["sampling"]["seed"]))
    print(
        f"[common cut states] Rhat={states.diagnostics['max_rhat']:.3f} "
        f"identity={states.diagnostics['exact_identity_error']:.2e}", flush=True,
    )
    jobs = []
    for index, cell in enumerate(CELL_SPECS):
        jobs.append((exact, states, cell, cfg, sampling, int(cfg["sampling"]["seed"]) + 1009 * (index + 1)))
    fits = {}
    with ProcessPoolExecutor(max_workers=min(args.workers, len(jobs))) as pool:
        futures = {pool.submit(_worker, job): job[2] for job in jobs}
        for future in as_completed(futures):
            cell, fit = future.result()
            fits[cell] = fit
            save_fit(draw_dir / f"{cell}.npz", fit)
            s = summarize_fit(fit)
            print(
                f"[{cell}] theta={s['theta']['mean']:+.3f} "
                f"[{s['theta']['q2.5']:+.3f},{s['theta']['q97.5']:+.3f}] "
                f"P+={s['theta']['p_positive']:.3f} "
                f"sd(Nhat)={s['state']['nhat_path_sd']:.3f} "
                f"Rhat={s['diagnostics']['max_rhat']:.3f}", flush=True,
            )

    results = {cell: summarize_fit(fits[cell]) for cell in CELL_SPECS}
    legacy = {
        "same_cell_fixed_decomposition_fixed6": {
            "theta_mean": 0.054, "q2.5": -0.008, "q97.5": 0.113, "p_positive": 0.96,
        },
        "same_cell_joint_measurement_error_fixed6": {
            "theta_mean": 0.159, "q2.5": -0.028, "q97.5": 0.346, "p_positive": 0.98,
        },
    }
    max_rhat = max([states.diagnostics["max_rhat"]] + [r["diagnostics"]["max_rhat"] for r in results.values()])
    manifest = {
        "revision": cfg["revision"], "mode": mode,
        "sample": {"first": str(exact.case.periods[0]), "last": str(exact.case.periods[-1]), "n": exact.case.n_periods},
        "controlled_features": {
            "exact_identity": "N=Nbar+Nhat without measurement error",
            "annual_constraint": "Gustavo Q4 exact",
            "allocation": "common Capital IQ-updated allocation posterior mean",
            "inflation": "PPI", "activity": "negative unemployment gap", "expectations": "SPF GDP",
        },
        "sampling": dict(sampling), "elapsed_seconds": time.time() - started,
        "common_cut_state_diagnostics": states.diagnostics,
        "results": results, "legacy_reference": legacy,
        "gate": {"required": float(cfg["gates"]["max_rhat"]), "max_rhat": max_rhat,
                 "passed": bool(max_rhat <= float(cfg["gates"]["max_rhat"]))},
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"wrote {output / 'manifest.json'}", flush=True)


if __name__ == "__main__":
    main()
