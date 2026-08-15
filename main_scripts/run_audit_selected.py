"""Run the small, pre-declared production subset used by the independent audit.

This deliberately does not dispatch the full report grid.  It re-estimates the main
core-CPI CES and mixed-frequency HSA-steady cells, plus the corrected SEC aggregate
that matches the log-linear empirical equation (the revenue-weighted geometric mean
of inverse market HHIs).
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

import _bootstrap  # noqa: F401
from _bootstrap import DATA_DIR, RESULTS_DIR, ROOT
from nkpc_hsa.config import configured_data_specs, load_model_config
from nkpc_hsa.inference.wrappers import ESTIMATION_REVISION, run_model


def _run(job: dict) -> str:
    config = load_model_config(ROOT / "configs" / "models.yaml")
    spec = configured_data_specs(config)["unemployment_gap_core"]
    data_path = Path(job["data"])
    data = pd.read_csv(data_path, parse_dates=["DATE"]).set_index("DATE")
    if job["kind"] == "sec":
        spec = {
            **spec,
            "name": "unemployment_gap_core__sec_inverse_hhi_logrevw",
            "n_col": "N_SEC_inverse_HHI_logrevw",
            "sample_start": "2012Q1",
        }
        spec.pop("sample_end", None)
    run_model(
        job["model"],
        data=data,
        data_spec=spec,
        prior_specs=ROOT / "configs" / "priors_baseline.yaml",
        n_iter=job["n_iter"],
        burn=job["burn"],
        thin=job["thin"],
        chains=2,
        seed=job["seed"],
        n_transform="log100_centered10",
        competition_measurement={"frequency": job["frequency"], "annual_timing": "q4"},
        covariance_structure="e_zeta_only",
        enforce_stationary=True,
        run_id=f"audit_{ESTIMATION_REVISION}",
        run_dir=Path(job["run_dir"]),
        save=True,
    )
    return str(job["run_dir"])


def main() -> None:
    config = load_model_config(ROOT / "configs" / "models.yaml")
    defaults = config["defaults"]
    out = RESULTS_DIR / "audit" / ESTIMATION_REVISION / "selected_runs"
    jobs = [
        {
            "kind": "main",
            "model": "ces",
            "frequency": "quarterly_interpolated",
            "data": DATA_DIR / "processed" / "model_ready.csv",
            "run_dir": out / "ces_core",
            "seed": 32101,
        },
        {
            "kind": "main",
            "model": "hsa_steady",
            "frequency": "annual_q4",
            "data": DATA_DIR / "processed" / "model_ready.csv",
            "run_dir": out / "hsa_steady_core_annual_q4",
            "seed": 32102,
        },
        {
            "kind": "sec",
            "model": "hsa_steady",
            "frequency": "quarterly_observed",
            "data": DATA_DIR / "processed" / "model_ready_sec_inverse_hhi.csv",
            "run_dir": out / "hsa_steady_core_sec_logrevw",
            "seed": 32103,
        },
    ]
    for job in jobs:
        job.update(
            n_iter=int(defaults["n_iter"]),
            burn=int(defaults["burn"]),
            thin=int(defaults["thin"]),
        )
    print(f"revision={ESTIMATION_REVISION}; selected production cells={len(jobs)}")
    with ProcessPoolExecutor(max_workers=len(jobs)) as executor:
        futures = [executor.submit(_run, job) for job in jobs]
        for future in as_completed(futures):
            print(f"completed {future.result()}", flush=True)


if __name__ == "__main__":
    main()
