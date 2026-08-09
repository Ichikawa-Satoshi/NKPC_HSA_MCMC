from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import pandas as pd

from _bootstrap import DATA_DIR, RESULTS_DIR, ROOT
from nkpc_hsa.config import configured_data_specs, load_model_config
from nkpc_hsa.dataprep.transforms import DEFAULT_N_TRANSFORM
from nkpc_hsa.inference.wrappers import ESTIMATION_REVISION, model_sample_index, run_model
from nkpc_hsa.reporting.cpi_ppi_spec import annual_q4_run_keys, report_run_keys


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
    expected_samples: dict[str, tuple[int, str, str]],
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
        data_spec = str(metadata.get("data_spec", ""))
        expected = expected_samples.get(data_spec)
        if expected is None:
            continue
        expected_n, expected_start, expected_end = expected
        if int(metadata.get("n_obs", 0) or 0) != expected_n:
            continue
        if str(metadata.get("sample_start", "")) != expected_start:
            continue
        if str(metadata.get("sample_end", "")) != expected_end:
            continue
        keys.add(
            (
                str(metadata.get("model", "")),
                data_spec,
                str(metadata.get("prior_spec", "baseline") or "baseline"),
            )
        )
    return keys


def _resolved_data_spec(config: dict[str, object], data_spec_name: str) -> dict[str, object]:
    """Resolve one cell through the same helper used by every other production entry point."""
    return configured_data_specs(config, [data_spec_name])[data_spec_name]


def _sample_signature(data: pd.DataFrame, spec: dict[str, object]) -> tuple[int, str, str]:
    sample_index = model_sample_index(data, spec)
    if not isinstance(sample_index, pd.DatetimeIndex) or not len(sample_index):
        raise ValueError(f"Could not resolve a dated sample for {spec.get('name', '')!r}.")
    return (
        len(sample_index),
        sample_index.min().date().isoformat(),
        sample_index.max().date().isoformat(),
    )


def _estimate_one(job: dict[str, object]) -> tuple[tuple[str, str, str], float, str]:
    started = time.perf_counter()
    model, data_spec_name, prior = job["key"]
    config = load_model_config(job["config"])
    # Resolve through the canonical helper so the study-wide sample window in
    # defaults.sample_start / defaults.sample_end is injected into every cell.
    # Building the dict directly used to bypass that window and, after the PCHIP
    # date fix extended model_ready.csv, silently changed T from 124 to 128.
    spec = _resolved_data_spec(config, str(data_spec_name))
    data = pd.read_csv(job["data"], parse_dates=["DATE"]).set_index("DATE")
    run_id = str(job["run_id"])
    # One directory per (model, data spec, prior, observation design). The name
    # deliberately carries no timestamp: a cell is re-estimated in place, so the
    # directory is the cell rather than one attempt at it, and the estimation
    # timestamp lives in metadata.json as run_id. The observation design MUST be
    # in the name -- without it the interpolated and mixed-frequency runs of the
    # same cell collide, which is what the old timestamped scheme was hiding.
    frequency = str(job["competition_frequency"])
    run_dir = Path(str(job["runs_dir"])) / f"{model}_{data_spec_name}_{prior}_{frequency}"
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
        n_particles=int(job.get("n_particles", 512)),
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
    parser.add_argument("--data", type=Path, default=DATA_DIR / "processed" / "model_ready.csv")
    parser.add_argument("--runs-dir", type=Path, default=RESULTS_DIR / "runs")
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--force", action="store_true", help="Re-estimate cells even when a complete current-revision run exists.")
    # The old --no-compile passed a --compile flag that scripts/12 does not accept,
    # so every run without it died at the last line. Replaced by two honest flags:
    # --no-build skips the report build entirely, --compile additionally runs xelatex.
    parser.add_argument("--no-build", action="store_true", help="Estimate only; skip the report build.")
    parser.add_argument("--compile", action="store_true", help="Also run xelatex after building the report.")
    parser.add_argument("--quick", action="store_true", help="Run 80 iterations per cell for pipeline testing only.")
    parser.add_argument(
        "--competition-frequency",
        choices=["quarterly_interpolated", "annual_q4"],
        default=None,
        help="Observation design. Defaults to configs/models.yaml "
             "defaults.competition_measurement.frequency.",
    )
    args = parser.parse_args()

    config = load_model_config(args.config)
    defaults = config.get("defaults", {})
    n_iter = 80 if args.quick else int(defaults.get("n_iter", 12000))
    burn = 40 if args.quick else int(defaults.get("burn", 4000))
    thin = 2 if args.quick else int(defaults.get("thin", 5))
    chains = 2 if args.quick else int(defaults.get("chains", 2))
    seed = int(defaults.get("seed", 12345))
    if args.competition_frequency is None:
        args.competition_frequency = str(
            (defaults.get("competition_measurement", {}) or {}).get("frequency", "annual_q4")
        )
    required = report_run_keys() if args.competition_frequency == "quarterly_interpolated" else annual_q4_run_keys()
    data = pd.read_csv(args.data, parse_dates=["DATE"]).set_index("DATE")
    expected_samples = {
        name: _sample_signature(data, _resolved_data_spec(config, name))
        for name in sorted({data_spec for _, data_spec, _ in required})
    }
    existing = set() if args.force else _existing_keys(
        args.runs_dir,
        min_iter=n_iter,
        competition_frequency=args.competition_frequency,
        expected_samples=expected_samples,
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
                "n_particles": int(defaults.get("n_particles", 512)),
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

    # Rebuild every report artifact, not just the ones script 12 owns. Calling
    # 12 directly here is what left the headline table, the fit comparison and
    # the data figure at their previous vintage after a re-estimation.
    if args.no_build:
        print("Skipping the report build (--no-build); run scripts/build_report.py when ready.")
        return
    command = [
        sys.executable,
        str(ROOT / "scripts" / "build_report.py"),
        "--runs-dir",
        str(args.runs_dir),
        "--min-iter",
        str(n_iter),
    ]
    if args.compile:
        command.append("--compile")
    subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
