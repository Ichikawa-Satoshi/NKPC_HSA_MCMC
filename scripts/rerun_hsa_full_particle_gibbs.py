"""Re-estimate every production ``hsa_full`` cell with the Particle-Gibbs sampler.

Particle Gibbs is now the ``run_model("hsa_full")`` state sampler for BOTH
observation designs (see ``src/nkpc_hsa/models/hsa_full.py``). Previously only
the 15 PCHIP cells had Particle-Gibbs runs, produced out-of-band by a monkeypatch
in ``scripts/appendix_pg_full_runs.py``; the 15 annual-Q4 cells were still
alternating FFBS, so the PCHIP-vs-annual-Q4 comparison confounded the observation
scheme with the sampler.

This script re-runs all 30 report cells (15 PCHIP + 15 annual-Q4) through the
ordinary pipeline. Existing runs are never overwritten: each new run goes to its
own directory with a ``_pg`` run-id suffix, and the report's run selector keeps
the newest run-id per (model, data spec, prior) key.

    python scripts/rerun_hsa_full_particle_gibbs.py [--quick]
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

import _bootstrap  # noqa: F401
from _bootstrap import ROOT

from nkpc_hsa.config import configured_data_specs, load_model_config
from nkpc_hsa.data.transforms import DEFAULT_N_TRANSFORM
from nkpc_hsa.inference.wrappers import run_model
from nkpc_hsa.report.cpi_ppi_spec import report_run_keys


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="80 iterations, for a smoke run.")
    args = ap.parse_args()

    config = load_model_config()
    defaults = config.get("defaults", {})
    specs = configured_data_specs(config, list(config.get("data_specs", {})))
    data = pd.read_csv(
        ROOT / "data" / "processed" / "model_ready.csv", parse_dates=["DATE"]
    ).set_index("DATE")

    n_iter = 80 if args.quick else int(defaults.get("n_iter", 12000))
    burn = 40 if args.quick else int(defaults.get("burn", 4000))
    thin = 2 if args.quick else int(defaults.get("thin", 5))
    chains = 2 if args.quick else int(defaults.get("chains", 2))
    seed = int(defaults.get("seed", 12345))
    n_particles = 64 if args.quick else int(defaults.get("n_particles", 512))

    cells = [key for key in report_run_keys() if key[0] == "hsa_full"]
    jobs = [
        (spec, prior, freq)
        for (_, spec, prior) in cells
        for freq in ("quarterly_interpolated", "annual_q4")
    ]
    stamp = time.strftime("%Y%m%d_%H%M%S")
    print(f"{len(jobs)} hsa_full cells, Particle Gibbs, P={n_particles}, {n_iter} iters")

    t0 = time.time()
    for i, (spec, prior, freq) in enumerate(jobs, 1):
        run_id = f"{stamp}_pg"
        parts = ["hsa_full", spec, prior]
        if freq != "quarterly_interpolated":
            parts.append(freq)
        parts.append(run_id)
        run_dir = ROOT / "results" / "runs" / "_".join(parts)
        print(f"[{i}/{len(jobs)}] {spec} / {prior} / {freq}", flush=True)
        run_model(
            "hsa_full",
            data=data,
            data_spec=specs[spec],
            prior_specs=str(ROOT / "configs" / f"priors_{prior}.yaml"),
            prior_name=prior,
            n_iter=n_iter,
            burn=burn,
            thin=thin,
            chains=chains,
            seed=seed,
            n_transform=defaults.get("n_transform", DEFAULT_N_TRANSFORM),
            covariance_structure=defaults.get("covariance_structure", "e_zeta_only"),
            ar2_max_tries=int(defaults.get("ar2_max_tries", 2000)),
            n_particles=n_particles,
            competition_measurement={"frequency": freq, "annual_timing": "q4"},
            run_dir=run_dir,
            run_id=run_id,
            save=True,
        )
        print(f"    done ({time.time() - t0:.0f}s cumulative)", flush=True)
    print(f"ALL {len(jobs)} Particle-Gibbs hsa_full runs saved in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
