from __future__ import annotations

import argparse
from pathlib import Path

import arviz as az

from _bootstrap import ROOT
from nkpc_hsa.inference.model_comparison import model_comparison_table, save_model_comparison


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", default=str(ROOT / "results" / "runs"))
    args = parser.parse_args()
    results = {path.parent.name: az.from_netcdf(path) for path in sorted(Path(args.runs_dir).glob("*/posterior.nc"))}
    table = model_comparison_table(results)
    save_model_comparison(table, ROOT / "results" / "model_comparison")


if __name__ == "__main__":
    main()
