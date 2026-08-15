"""Short Particle-Gibbs particle-count audit on the main core-CPI cell.

These are diagnostics, not production posterior estimates: the short chains are used to
verify that PG path-mixing statistics survive the wrapper and to compare 128 with 512
particles before any expensive full-model re-estimation is contemplated.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

import _bootstrap  # noqa: F401
from _bootstrap import DATA_DIR, RESULTS_DIR, ROOT
from nkpc_hsa.config import configured_data_specs, load_model_config
from nkpc_hsa.inference.wrappers import ESTIMATION_REVISION, run_model


def _run(n_particles: int) -> str:
    config = load_model_config(ROOT / "configs" / "models.yaml")
    spec = configured_data_specs(config)["unemployment_gap_core"]
    data = pd.read_csv(
        DATA_DIR / "processed" / "model_ready.csv", parse_dates=["DATE"]
    ).set_index("DATE")
    target = (
        RESULTS_DIR
        / "audit"
        / ESTIMATION_REVISION
        / "pg_pilot"
        / f"hsa_full_core_annual_q4_particles{n_particles}"
    )
    run_model(
        "hsa_full",
        data=data,
        data_spec=spec,
        prior_specs=ROOT / "configs" / "priors_baseline.yaml",
        n_iter=1200,
        burn=400,
        thin=2,
        chains=2,
        seed=45100 + n_particles,
        n_transform="log100_centered10",
        competition_measurement={"frequency": "annual_q4", "annual_timing": "q4"},
        covariance_structure="e_zeta_only",
        enforce_stationary=True,
        n_particles=n_particles,
        run_id=f"audit_pg_{n_particles}_{ESTIMATION_REVISION}",
        run_dir=target,
        save=True,
    )
    return str(target)


def main() -> None:
    with ProcessPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(_run, particles) for particles in (128, 512)]
        for future in as_completed(futures):
            print(f"completed {future.result()}", flush=True)


if __name__ == "__main__":
    main()
