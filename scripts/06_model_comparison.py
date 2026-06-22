from __future__ import annotations

import argparse
from pathlib import Path

import arviz as az
import pandas as pd

from _bootstrap import ROOT
from nkpc_hsa.config import configured_data_specs, load_model_config
from nkpc_hsa.data.transforms import transform_competition_series
from nkpc_hsa.inference.model_comparison import model_comparison_table, save_model_comparison


def _comparison_data(data_path: Path, config: dict, spec: dict) -> dict[str, object]:
    defaults = config.get("defaults", {})
    df = pd.read_csv(data_path, parse_dates=["DATE"]).set_index("DATE")
    sample = df[
        [
            spec.get("pi_col", "pi"),
            spec.get("pi_prev_col", "pi_prev"),
            spec.get("pi_expect_col", "pi_expect"),
            spec.get("x_col", "x"),
            spec.get("x_prev_col", "x_prev"),
            spec.get("n_col", spec.get("N_col", "N")),
        ]
    ].dropna()
    n_raw = sample[spec.get("n_col", spec.get("N_col", "N"))].to_numpy(dtype=float)
    return {
        "pi": sample[spec.get("pi_col", "pi")].to_numpy(dtype=float),
        "pi_prev": sample[spec.get("pi_prev_col", "pi_prev")].to_numpy(dtype=float),
        "pi_expect": sample[spec.get("pi_expect_col", "pi_expect")].to_numpy(dtype=float),
        "x": sample[spec.get("x_col", "x")].to_numpy(dtype=float),
        "x_prev": sample[spec.get("x_prev_col", "x_prev")].to_numpy(dtype=float),
        "N": transform_competition_series(n_raw, transform=str(defaults.get("n_transform", "log100"))),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", default=str(ROOT / "results" / "runs"))
    parser.add_argument("--config", default=str(ROOT / "configs" / "models.yaml"))
    parser.add_argument("--data", default=str(ROOT / "data" / "processed" / "model_ready.csv"))
    parser.add_argument(
        "--data-spec",
        action="append",
        dest="data_specs",
        help="Data spec name to compare. Repeat for multiple. Defaults to config run_data_specs.",
    )
    args = parser.parse_args()

    config = load_model_config(args.config)
    data_specs = configured_data_specs(config, args.data_specs)
    results = {path.parent.name: az.from_netcdf(path) for path in sorted(Path(args.runs_dir).glob("*/posterior.nc"))}
    data_by_spec = {
        name: _comparison_data(Path(args.data), config, spec)
        for name, spec in data_specs.items()
        if Path(args.data).exists()
    }
    data_by_run = {
        run: data_by_spec.get(str(getattr(idata, "attrs", {}).get("data_spec", "")))
        for run, idata in results.items()
    }
    table = model_comparison_table(results, data_by_model=data_by_run)
    save_model_comparison(table, ROOT / "results" / "model_comparison")
    for data_spec_name in data_specs:
        spec_results = {
            run: idata
            for run, idata in results.items()
            if str(getattr(idata, "attrs", {}).get("data_spec", "")) == data_spec_name
        }
        spec_data = data_by_spec.get(data_spec_name)
        spec_table = model_comparison_table(
            spec_results,
            data_by_model={run: spec_data for run in spec_results} if spec_data else None,
        )
        save_model_comparison(spec_table, ROOT / "results" / "model_comparison" / data_spec_name)


if __name__ == "__main__":
    main()
