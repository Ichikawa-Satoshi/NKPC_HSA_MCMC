"""Save Particle-Gibbs hsa_full runs as posterior.nc (pipeline format).

Routes the existing run_model pipeline's ``hsa_full`` slot to the appendix
Particle-Gibbs sampler (monkeypatch in THIS process only; no files changed) and
saves each run to results/appendix_particle_gibbs/runs/ -- existing results are
never touched. These posterior.nc files are then consumed by the table-override
step so the report's HSA full rows reflect Particle Gibbs.

Cells: baseline 9 (3 price x {unemployment, HP, BN}) + weak/tight 6
(3 price x unemployment), all PCHIP, P=512, same MCMC config as production.
"""
from __future__ import annotations

import time
from pathlib import Path

import pandas as pd

import _bootstrap  # noqa: F401
from _bootstrap import ROOT

import nkpc_hsa.models.hsa_full as hsa_full_mod
import nkpc_hsa.gibbs.hsa_full_pg.model as pg_mod
from nkpc_hsa.gibbs.hsa_full_pg.model import func_nkpc_hsa_full_pg
from nkpc_hsa.config import configured_data_specs, load_model_config
from nkpc_hsa.inference.wrappers import run_model

# --- monkeypatch: hsa_full -> Particle Gibbs (this process only) ---
pg_mod.DEFAULT_N_PARTICLES = 512
hsa_full_mod.func_nkpc_hsa_full = func_nkpc_hsa_full_pg

OUT_RUNS = ROOT / "results" / "appendix_particle_gibbs" / "runs"
OUT_RUNS.mkdir(parents=True, exist_ok=True)

UNEMP = ["unemployment_gap", "unemployment_gap_core", "unemployment_gap_ppi"]
OG = ["output_gap_hp", "output_gap_hp_core", "output_gap_hp_ppi",
      "output_gap_bn", "output_gap_bn_core", "output_gap_bn_ppi"]
CELLS = ([(s, "baseline") for s in UNEMP + OG]
         + [(s, p) for s in UNEMP for p in ("weak", "tight")])


def main():
    specs = configured_data_specs(load_model_config())
    df = pd.read_csv(ROOT / "data" / "processed" / "model_ready.csv")
    t0 = time.time()
    for i, (spec_name, prior) in enumerate(CELLS, 1):
        run_dir = OUT_RUNS / f"pg_hsa_full_{spec_name}_{prior}"
        print(f"[{i}/{len(CELLS)}] {spec_name} / {prior} -> {run_dir.name}", flush=True)
        run_model(
            "hsa_full",
            data=df,
            data_spec=specs[spec_name],
            prior_specs=str(ROOT / "configs" / f"priors_{prior}.yaml"),
            prior_name=prior,
            n_iter=12000, burn=4000, thin=5, chains=2, seed=12345,
            run_dir=run_dir, run_id=f"pg_{spec_name}_{prior}", save=True,
        )
        print(f"    done ({time.time()-t0:.0f}s cumulative)", flush=True)
    print(f"ALL {len(CELLS)} PG hsa_full runs saved in {time.time()-t0:.0f}s -> {OUT_RUNS}", flush=True)


if __name__ == "__main__":
    main()
