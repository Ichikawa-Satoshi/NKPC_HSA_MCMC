from __future__ import annotations

import argparse
from pathlib import Path

import arviz as az
import pandas as pd

from _bootstrap import ROOT
from nkpc_hsa.report.figures import save_posterior_predictive_placeholder
from nkpc_hsa.report.tables import posterior_summary_table, write_latex_fragment


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", default=str(ROOT / "results" / "runs"))
    args = parser.parse_args()
    rows = []
    for posterior in sorted(Path(args.runs_dir).glob("*/posterior.nc")):
        idata = az.from_netcdf(posterior)
        table = posterior_summary_table(idata)
        if not table.empty:
            table.insert(0, "run", posterior.parent.name)
            rows.append(table)
    summary = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame({"note": ["No posterior summaries available."]})
    summary.to_csv(ROOT / "results" / "tables" / "posterior_summary.csv", index=False)
    write_latex_fragment(summary, ROOT / "results" / "tables" / "posterior_summary.tex")
    save_posterior_predictive_placeholder(ROOT / "results" / "figures" / "posterior_predictive_placeholder.png")


if __name__ == "__main__":
    main()
