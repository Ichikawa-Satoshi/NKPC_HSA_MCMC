from __future__ import annotations

import argparse

import pandas as pd
import yaml

from _bootstrap import ROOT
from nkpc_hsa.config import coefficient_constraints_from_config, configured_data_specs, load_model_config
from nkpc_hsa.inference.period_robustness import (
    load_periods,
    run_period_robustness,
    save_period_robustness_table,
)
from nkpc_hsa.report.tables import write_latex_fragment


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="hsa_steady")
    parser.add_argument("--config", default=str(ROOT / "configs" / "models.yaml"))
    parser.add_argument("--periods", default=str(ROOT / "configs" / "periods.yaml"))
    parser.add_argument("--priors", default=str(ROOT / "configs" / "priors_baseline.yaml"))
    parser.add_argument("--data", default=str(ROOT / "data" / "processed" / "model_ready.csv"))
    parser.add_argument(
        "--data-spec",
        action="append",
        dest="data_specs",
        help="Data spec name to run. Repeat to run multiple. Defaults to config run_data_specs.",
    )
    parser.add_argument("--quick", action="store_true")
    parser.add_argument(
        "--positive",
        action="append",
        default=[],
        help="Force coefficients nonnegative. Repeat or comma-separate, e.g. --positive kappa,theta.",
    )
    parser.add_argument("--no-coefficient-constraints", action="store_true")
    args = parser.parse_args()

    config = load_model_config(args.config)
    period_config = yaml.safe_load(open(args.periods, encoding="utf-8")) or {}
    defaults = config.get("defaults", {})
    data_specs = configured_data_specs(config, args.data_specs)
    data = pd.read_csv(args.data, parse_dates=["DATE"]).set_index("DATE")
    coefficient_constraints = coefficient_constraints_from_config(
        defaults,
        positive=args.positive,
        disabled=args.no_coefficient_constraints,
    )

    out_dir = ROOT / "results" / "period_robustness"
    all_tables = []
    for data_spec_name, data_spec in data_specs.items():
        _, table = run_period_robustness(
            args.model,
            data=data,
            periods=load_periods(args.periods),
            data_spec=data_spec,
            prior_specs=args.priors,
            n_iter=80 if args.quick else int(defaults.get("n_iter", 12000)),
            burn=40 if args.quick else int(defaults.get("burn", 4000)),
            thin=2 if args.quick else int(defaults.get("thin", 5)),
            chains=2 if args.quick else int(defaults.get("chains", 2)),
            seed=int(defaults.get("seed", 12345)),
            min_obs=int(period_config.get("min_obs", 40)),
            n_transform=str(defaults.get("n_transform", "log100")),
            covariance_structure=str(defaults.get("covariance_structure", "e_zeta_only")),
            coefficient_constraints=coefficient_constraints,
        )
        if not table.empty:
            table.insert(0, "data_spec", data_spec_name)
            all_tables.append(table)
        save_period_robustness_table(table, out_dir / data_spec_name)
        write_latex_fragment(
            table if not table.empty else pd.DataFrame({"note": ["No period robustness output."]}),
            ROOT / "results" / "tables" / data_spec_name / "period_robustness.tex",
        )
    table = pd.concat(all_tables, ignore_index=True) if all_tables else pd.DataFrame()
    save_period_robustness_table(table, out_dir)
    write_latex_fragment(
        table if not table.empty else pd.DataFrame({"note": ["No period robustness output."]}),
        ROOT / "results" / "tables" / "period_robustness.tex",
    )


if __name__ == "__main__":
    main()
