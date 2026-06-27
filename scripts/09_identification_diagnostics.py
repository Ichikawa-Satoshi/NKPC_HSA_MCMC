from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from _bootstrap import ROOT
from nkpc_hsa.config import configured_data_specs, load_model_config
from nkpc_hsa.data.transforms import DEFAULT_N_TRANSFORM
from nkpc_hsa.inference.identification import load_posterior_runs, write_identification_outputs
from nkpc_hsa.inference.period_robustness import load_periods


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Check whether weak or unexpected kappa/delta results are driven by "
            "code conventions, scale, sample periods, or weak HSA interaction design."
        )
    )
    parser.add_argument("--config", default=str(ROOT / "configs" / "models.yaml"))
    parser.add_argument("--periods", default=str(ROOT / "configs" / "periods.yaml"))
    parser.add_argument("--data", default=str(ROOT / "data" / "processed" / "model_ready.csv"))
    parser.add_argument("--runs-dir", default=str(ROOT / "results" / "runs"))
    parser.add_argument("--out-dir", default=str(ROOT / "results" / "diagnostics" / "identification"))
    parser.add_argument("--prior", default="baseline", help="Prior spec to diagnose. Use 'all' to include all priors.")
    parser.add_argument(
        "--constraint",
        default="unrestricted",
        help="Constraint spec to diagnose. Use 'all' to include all constraints.",
    )
    parser.add_argument(
        "--data-spec",
        action="append",
        dest="data_specs",
        help="Data spec to diagnose. Repeat for multiple. Defaults to config run_data_specs.",
    )
    parser.add_argument(
        "--model",
        action="append",
        dest="models",
        default=None,
        help="Model to diagnose. Defaults to hsa_steady and hsa_full.",
    )
    args = parser.parse_args()

    config = load_model_config(args.config)
    defaults = config.get("defaults", {})
    n_transform = str(defaults.get("n_transform", DEFAULT_N_TRANSFORM))
    data_specs = configured_data_specs(config, args.data_specs)
    models = args.models or ["hsa_steady", "hsa_full"]
    prior = None if args.prior == "all" else args.prior
    constraint = None if args.constraint == "all" else args.constraint

    data = pd.read_csv(args.data, parse_dates=["DATE"])
    periods = load_periods(args.periods)
    runs = load_posterior_runs(
        args.runs_dir,
        data_specs=list(data_specs),
        models=models,
        prior_spec=prior,
        constraint_spec=constraint,
        n_transform=n_transform,
        latest_only=True,
    )
    outputs = write_identification_outputs(
        out_dir=Path(args.out_dir),
        data=data,
        config=config,
        runs=runs,
        periods=periods,
        requested_data_specs=list(data_specs),
    )

    for name, table in outputs.items():
        path = Path(args.out_dir) / f"{name}.csv"
        print(f"Saved {path} ({len(table)} rows)")


if __name__ == "__main__":
    main()
