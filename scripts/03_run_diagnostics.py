from __future__ import annotations

import argparse
from pathlib import Path

import arviz as az
import pandas as pd

from _bootstrap import ROOT
from nkpc_hsa.config import configured_data_specs, load_model_config
from nkpc_hsa.inference.diagnostics import save_diagnostics
from nkpc_hsa.report.tables import write_latex_fragment


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", default=str(ROOT / "results" / "runs"))
    parser.add_argument("--out-dir", default=str(ROOT / "results" / "diagnostics"))
    parser.add_argument("--config", default=str(ROOT / "configs" / "models.yaml"))
    parser.add_argument(
        "--data-spec",
        action="append",
        dest="data_specs",
        help="Data spec name to diagnose. Repeat for multiple. Defaults to config run_data_specs.",
    )
    parser.add_argument("--all-runs", action="store_true", help="Diagnose every run in runs-dir, including robustness period runs.")
    args = parser.parse_args()
    selected_specs = None
    if not args.all_runs:
        selected_specs = set(configured_data_specs(load_model_config(args.config), args.data_specs))
    rows = []
    for posterior in sorted(Path(args.runs_dir).glob("*/posterior.nc")):
        idata = az.from_netcdf(posterior)
        data_spec = str(getattr(idata, "attrs", {}).get("data_spec", ""))
        if selected_specs is not None and data_spec not in selected_specs:
            continue
        run_name = posterior.parent.name
        summary = save_diagnostics(idata, Path(args.out_dir) / run_name)
        if not summary.empty:
            summary.insert(0, "run", run_name)
            summary.insert(1, "model", getattr(idata, "attrs", {}).get("model", ""))
            summary.insert(2, "data_spec", data_spec)
            summary.insert(3, "prior_spec", getattr(idata, "attrs", {}).get("prior_spec", ""))
            summary.insert(4, "constraint_spec", getattr(idata, "attrs", {}).get("constraint_spec", "unrestricted"))
            rows.append(summary)
    table = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    table.to_csv(ROOT / "results" / "tables" / "mcmc_diagnostics.csv", index=False)
    write_latex_fragment(table if not table.empty else pd.DataFrame({"note": ["No diagnostics available."]}), ROOT / "results" / "tables" / "mcmc_diagnostics.tex")


if __name__ == "__main__":
    main()
