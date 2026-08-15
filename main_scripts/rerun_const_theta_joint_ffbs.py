"""Re-estimate every production hsa_const_theta cell with the exact joint FFBS.

Only hsa_const_theta changed sampler, so only hsa_const_theta is re-run; every
other model's production runs are left untouched.

Old run directories are NOT overwritten. Each re-run gets a fresh timestamped
run directory and its model metadata declares ``state_sampler = "joint_ffbs"``,
so provenance is explicit and the previous alternating-FFBS runs remain on disk
for comparison. The report's run selector keeps the newest run per cell, so the
tables pick these up automatically.

Covers both observation designs (PCHIP quarterly and annual-Q4), because the
state block is a property of the model, not of the observation scheme.
"""
from __future__ import annotations

import argparse
import time
from datetime import datetime

import pandas as pd

import _bootstrap  # noqa: F401
from _bootstrap import DATA_DIR, ROOT

from nkpc_hsa.config import configured_data_specs, load_model_config
from nkpc_hsa.inference.wrappers import run_model
from nkpc_hsa.reporting.cpi_ppi_spec import report_run_keys

MODEL = "hsa_const_theta"


def cells() -> list[tuple[str, str]]:
    return [(spec, prior) for model, spec, prior in report_run_keys() if model == MODEL]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--frequency",
        choices=["quarterly_interpolated", "annual_q4", "both"],
        default="both",
    )
    args = ap.parse_args()

    freqs = (
        ["quarterly_interpolated", "annual_q4"]
        if args.frequency == "both"
        else [args.frequency]
    )
    specs = configured_data_specs(load_model_config())
    df = pd.read_csv(DATA_DIR / "processed" / "model_ready.csv", parse_dates=["DATE"]).set_index("DATE")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    todo = [(f, s, p) for f in freqs for s, p in cells()]
    t0 = time.time()
    for i, (freq, spec_name, prior) in enumerate(todo, 1):
        print(f"[{i}/{len(todo)}] {MODEL} / {spec_name} / {prior} / {freq}", flush=True)
        run_model(
            MODEL,
            data=df,
            data_spec=specs[spec_name],
            prior_specs=str(ROOT / "configs" / f"priors_{prior}.yaml"),
            prior_name=prior,
            n_iter=12000,
            burn=4000,
            thin=5,
            chains=2,
            seed=12345,
            competition_measurement={"frequency": freq, "annual_timing": "q4"},
            run_id=f"{stamp}_jointffbs",
            save=True,
        )
        print(f"    done ({time.time() - t0:.0f}s cumulative)", flush=True)
    print(f"ALL {len(todo)} hsa_const_theta joint-FFBS runs saved in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
