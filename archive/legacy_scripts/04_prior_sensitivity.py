from __future__ import annotations

import argparse

import pandas as pd
import yaml

from _bootstrap import ROOT
from nkpc_hsa.inference.prior_sensitivity import (
    prior_sensitivity_table,
    run_prior_sensitivity,
    save_prior_sensitivity_overlays,
)
from nkpc_hsa.report.tables import write_latex_fragment


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="hsa_steady")
    parser.add_argument("--config", default=str(ROOT / "configs" / "models.yaml"))
    parser.add_argument("--data", default=str(ROOT / "data" / "processed" / "model_ready.csv"))
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load(open(args.config, encoding="utf-8")) or {}
    data_spec = {"name": "default", **config.get("data_specs", {}).get("default", {})}
    data = pd.read_csv(args.data, parse_dates=["DATE"]).set_index("DATE")
    idata_map = run_prior_sensitivity(
        args.model,
        data=data,
        data_spec=data_spec,
        n_iter=80 if args.quick else int(config.get("defaults", {}).get("n_iter", 12000)),
        burn=40 if args.quick else int(config.get("defaults", {}).get("burn", 4000)),
        thin=2 if args.quick else int(config.get("defaults", {}).get("thin", 5)),
        chains=2,
    )
    table = prior_sensitivity_table(idata_map)
    table.to_csv(ROOT / "results" / "tables" / "prior_sensitivity.csv", index=False)
    write_latex_fragment(table if not table.empty else pd.DataFrame({"note": ["No prior sensitivity output."]}), ROOT / "results" / "tables" / "prior_sensitivity.tex")
    save_prior_sensitivity_overlays(idata_map, ROOT / "results" / "figures")


if __name__ == "__main__":
    main()
