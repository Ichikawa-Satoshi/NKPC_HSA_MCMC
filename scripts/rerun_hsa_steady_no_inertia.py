"""Estimate HSA steady without the lagged-inflation term, mixed-frequency design.

The baseline estimating equation adds a backward-looking term that the theoretical
Phillips curve does not have:

    pi_t = alpha*pi_{t-1} + (1-alpha)*E_t pi_{t+1} + kappa_t*x_t + e_t

Two things make that term worth testing rather than assuming. It is absent from the
theory (report section 1.4 lists it as a departure), and every inflation series here is a
four-quarter change sampled quarterly, so pi_t and pi_{t-1} share three of four quarters:
even if quarterly inflation were white noise the overlap alone would put
corr(pi_t, pi_{t-1}) at 0.75, against 0.97 in the data and an estimated alpha of 0.79.

This script re-estimates the purely forward-looking restriction alpha == 0,

    pi_t = E_t pi_{t+1} + kappa_t*x_t + e_t,

for all nine price-index x activity-measure cells under the main mixed-frequency design.
The runs are written with ``constraint_spec = "alpha_zero"``, so the report's run selector
(which requires ``unrestricted``) cannot pick them up by accident.

    python scripts/rerun_hsa_steady_no_inertia.py [--quick]
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

import _bootstrap  # noqa: F401
from _bootstrap import ROOT

from nkpc_hsa.config import configured_data_specs, load_model_config
from nkpc_hsa.dataprep.transforms import DEFAULT_N_TRANSFORM
from nkpc_hsa.inference.wrappers import run_model
from nkpc_hsa.reporting.cpi_ppi_spec import INFLATION_SPECS

FREQ = "annual_q4"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    config = load_model_config()
    defaults = config.get("defaults", {})
    specs = configured_data_specs(config, list(config.get("data_specs", {})))
    data = pd.read_csv(
        ROOT / "data" / "processed" / "model_ready.csv", parse_dates=["DATE"]
    ).set_index("DATE")

    n_iter = 200 if args.quick else int(defaults.get("n_iter", 12000))
    burn = 100 if args.quick else int(defaults.get("burn", 4000))
    thin = 2 if args.quick else int(defaults.get("thin", 5))
    chains = 2 if args.quick else int(defaults.get("chains", 2))

    cells = [spec for activity in INFLATION_SPECS.values() for spec in activity.values()]
    cells = sorted(set(cells))
    stamp = time.strftime("%Y%m%d_%H%M%S")
    print(f"{len(cells)} cells, HSA steady, alpha == 0, {FREQ}, {n_iter} iters")

    t0 = time.time()
    for i, spec in enumerate(cells, 1):
        run_id = f"{stamp}_alphazero"
        run_dir = ROOT / "results" / "runs" / f"hsa_steady_{spec}_baseline_alpha_zero_{FREQ}_{run_id}"
        print(f"[{i}/{len(cells)}] {spec}", flush=True)
        run_model(
            "hsa_steady",
            data=data,
            data_spec=specs[spec],
            prior_specs=str(ROOT / "configs" / "priors_baseline.yaml"),
            prior_name="baseline",
            n_iter=n_iter, burn=burn, thin=thin, chains=chains,
            seed=int(defaults.get("seed", 12345)),
            n_transform=defaults.get("n_transform", DEFAULT_N_TRANSFORM),
            ar2_max_tries=int(defaults.get("ar2_max_tries", 2000)),
            competition_measurement={"frequency": FREQ, "annual_timing": "q4"},
            no_inertia=True,
            run_dir=run_dir, run_id=run_id, save=True,
        )
        print(f"    done ({time.time() - t0:.0f}s cumulative)", flush=True)
    print(f"ALL {len(cells)} alpha==0 runs saved in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
