from __future__ import annotations

import argparse

import pandas as pd

from _bootstrap import ROOT
from nkpc_hsa.config import coefficient_constraints_from_config, configured_data_specs, load_model_config
from nkpc_hsa.inference.prior_robustness import (
    prior_robustness_table,
    run_prior_robustness,
    save_prior_robustness_overlays,
)
from nkpc_hsa.report.tables import write_latex_fragment


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="hsa_steady")
    parser.add_argument("--config", default=str(ROOT / "configs" / "models.yaml"))
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
        help="Force coefficients nonnegative. Repeat or comma-separate, e.g. --positive kappa_0,delta.",
    )
    parser.add_argument("--no-coefficient-constraints", action="store_true")
    args = parser.parse_args()

    config = load_model_config(args.config)
    defaults = config.get("defaults", {})
    data_specs = configured_data_specs(config, args.data_specs)
    data = pd.read_csv(args.data, parse_dates=["DATE"]).set_index("DATE")
    n_iter = 80 if args.quick else int(defaults.get("n_iter", 12000))
    burn = 40 if args.quick else int(defaults.get("burn", 4000))
    thin = 2 if args.quick else int(defaults.get("thin", 5))
    chains = 2 if args.quick else int(defaults.get("chains", 2))
    coefficient_constraints = coefficient_constraints_from_config(
        defaults,
        positive=args.positive,
        disabled=args.no_coefficient_constraints,
    )

    out_dir = ROOT / "results" / "prior_robustness"
    out_dir.mkdir(parents=True, exist_ok=True)
    all_tables = []
    for data_spec_name, data_spec in data_specs.items():
        idata_map = run_prior_robustness(
            args.model,
            data=data,
            data_spec=data_spec,
            n_iter=n_iter,
            burn=burn,
            thin=thin,
            chains=chains,
            seed=int(defaults.get("seed", 12345)),
            n_transform=str(defaults.get("n_transform", "log100")),
            covariance_structure=str(defaults.get("covariance_structure", "e_zeta_only")),
            coefficient_constraints=coefficient_constraints,
        )
        table = prior_robustness_table(idata_map)
        if not table.empty:
            table.insert(0, "data_spec", data_spec_name)
            all_tables.append(table)
        spec_dir = out_dir / data_spec_name
        spec_dir.mkdir(parents=True, exist_ok=True)
        spec_table = table if not table.empty else pd.DataFrame({"note": ["No prior robustness output."]})
        spec_table.to_csv(spec_dir / "prior_robustness.csv", index=False)
        write_latex_fragment(spec_table, ROOT / "results" / "tables" / data_spec_name / "prior_robustness.tex")
        save_prior_robustness_overlays(idata_map, ROOT / "results" / "figures" / data_spec_name)
    table = pd.concat(all_tables, ignore_index=True) if all_tables else pd.DataFrame()
    table.to_csv(out_dir / "prior_robustness.csv", index=False)
    table.to_csv(ROOT / "results" / "tables" / "prior_robustness.csv", index=False)
    write_latex_fragment(
        table if not table.empty else pd.DataFrame({"note": ["No prior robustness output."]}),
        ROOT / "results" / "tables" / "prior_robustness.tex",
    )
    write_latex_fragment(
        table if not table.empty else pd.DataFrame({"note": ["No prior robustness output."]}),
        ROOT / "results" / "tables" / "prior_sensitivity.tex",
    )


if __name__ == "__main__":
    main()
