from __future__ import annotations

import argparse
import json
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import pandas as pd

from _bootstrap import ROOT
from nkpc_hsa.config import load_model_config
from nkpc_hsa.data.transforms import DEFAULT_N_TRANSFORM
from nkpc_hsa.inference.wrappers import ESTIMATION_REVISION, run_model
from nkpc_hsa.report.cpi_ppi_spec import annual_q4_run_keys, report_run_keys


PRIOR_FILES = {
    "baseline": ROOT / "configs" / "priors_baseline.yaml",
    "weak": ROOT / "configs" / "priors_weak.yaml",
    "tight": ROOT / "configs" / "priors_tight.yaml",
}


def _existing_keys(
    runs_dir: Path,
    *,
    min_iter: int,
    competition_frequency: str,
) -> set[tuple[str, str, str]]:
    keys: set[tuple[str, str, str]] = set()
    for metadata_path in runs_dir.glob("*/metadata.json"):
        posterior = metadata_path.parent / "posterior.nc"
        if not posterior.exists():
            continue
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        if str(metadata.get("estimation_revision", "")) != ESTIMATION_REVISION:
            continue
        if int(metadata.get("n_iter", 0) or 0) < min_iter:
            continue
        if str(metadata.get("period", "full") or "full") != "full":
            continue
        if str(metadata.get("constraint_spec", "unrestricted") or "unrestricted") != "unrestricted":
            continue
        if str(metadata.get("competition_measurement_frequency", "quarterly_interpolated")) != competition_frequency:
            continue
        keys.add(
            (
                str(metadata.get("model", "")),
                str(metadata.get("data_spec", "")),
                str(metadata.get("prior_spec", "baseline") or "baseline"),
            )
        )
    return keys


def _estimate_one(job: dict[str, object]) -> tuple[tuple[str, str, str], float, str]:
    started = time.perf_counter()
    model, data_spec_name, prior = job["key"]
    config = load_model_config(job["config"])
    spec = {**config["data_specs"][data_spec_name], "name": data_spec_name}
    data = pd.read_csv(job["data"], parse_dates=["DATE"]).set_index("DATE")
    run_id = str(job["run_id"])
    run_dir = Path(str(job["runs_dir"])) / f"{model}_{data_spec_name}_{prior}_{run_id}"
    run_model(
        model,
        data=data,
        data_spec=spec,
        prior_specs=PRIOR_FILES[prior],
        n_iter=int(job["n_iter"]),
        burn=int(job["burn"]),
        thin=int(job["thin"]),
        chains=int(job["chains"]),
        seed=int(job["seed"]),
        n_transform=str(job["n_transform"]),
        covariance_structure=str(job["covariance_structure"]),
        coefficient_constraints=None,
        competition_measurement={"frequency": str(job["competition_frequency"]), "annual_timing": "q4"},
        enforce_stationary=True,
        ar2_max_tries=int(job["ar2_max_tries"]),
        run_id=run_id,
        run_dir=run_dir,
    )
    return (model, data_spec_name, prior), time.perf_counter() - started, run_id


def main() -> None:
    parser = argparse.ArgumentParser(description="Estimate every current-revision run required by the CPI/PPI report.")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "models.yaml")
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "processed" / "model_ready.csv")
    parser.add_argument("--runs-dir", type=Path, default=ROOT / "results" / "runs")
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--force", action="store_true", help="Re-estimate cells even when a complete current-revision run exists.")
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument("--quick", action="store_true", help="Run 80 iterations per cell for pipeline testing only.")
    parser.add_argument(
        "--competition-frequency",
        choices=["quarterly_interpolated", "annual_q4"],
        default="quarterly_interpolated",
        help="Estimate the standard PCHIP cells or the mixed-frequency annual-Q4 HSA cells.",
    )
    args = parser.parse_args()

    config = load_model_config(args.config)
    defaults = config.get("defaults", {})
    n_iter = 80 if args.quick else int(defaults.get("n_iter", 12000))
    burn = 40 if args.quick else int(defaults.get("burn", 4000))
    thin = 2 if args.quick else int(defaults.get("thin", 5))
    chains = 2 if args.quick else int(defaults.get("chains", 2))
    seed = int(defaults.get("seed", 12345))
    required = report_run_keys() if args.competition_frequency == "quarterly_interpolated" else annual_q4_run_keys()
    existing = set() if args.force else _existing_keys(
        args.runs_dir,
        min_iter=n_iter,
        competition_frequency=args.competition_frequency,
    )
    missing = [key for key in required if key not in existing]
    print(
        f"revision={ESTIMATION_REVISION} required={len(required)} existing={len(required) - len(missing)} "
        f"to_estimate={len(missing)} jobs={args.jobs} frequency={args.competition_frequency}",
        flush=True,
    )

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    jobs = []
    for index, key in enumerate(missing):
        model, data_spec, prior = key
        jobs.append(
            {
                "key": key,
                "config": str(args.config),
                "data": str(args.data),
                "runs_dir": str(args.runs_dir),
                "n_iter": n_iter,
                "burn": burn,
                "thin": thin,
                "chains": chains,
                "seed": seed + index * 1009,
                "n_transform": defaults.get("n_transform", DEFAULT_N_TRANSFORM),
                "covariance_structure": defaults.get("covariance_structure", "e_zeta_only"),
                "ar2_max_tries": defaults.get("ar2_max_tries", 2000),
                "competition_frequency": args.competition_frequency,
                "run_id": f"{stamp}_{index:03d}",
            }
        )

    completed = 0
    if jobs:
        with ProcessPoolExecutor(max_workers=max(1, args.jobs)) as executor:
            futures = {executor.submit(_estimate_one, job): job["key"] for job in jobs}
            for future in as_completed(futures):
                key, elapsed, run_id = future.result()
                completed += 1
                print(
                    f"[{completed}/{len(jobs)}] {'/'.join(key)} completed in {elapsed:.1f}s ({run_id})",
                    flush=True,
                )

    command = [
        "python",
        str(ROOT / "scripts" / "12_build_cpi_ppi_report.py"),
        "--runs-dir",
        str(args.runs_dir),
        "--min-iter",
        str(n_iter),
    ]
    if not args.no_compile:
        command.append("--compile")
    subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
