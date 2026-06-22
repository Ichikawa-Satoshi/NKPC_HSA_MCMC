from __future__ import annotations

import argparse

import pandas as pd

from _bootstrap import ROOT
from nkpc_hsa.config import coefficient_constraints_from_config, configured_data_specs, load_model_config
from nkpc_hsa.inference.wrappers import run_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / "models.yaml"))
    parser.add_argument("--priors", default=str(ROOT / "configs" / "priors_baseline.yaml"))
    parser.add_argument("--data", default=str(ROOT / "data" / "processed" / "model_ready.csv"))
    parser.add_argument(
        "--data-spec",
        action="append",
        dest="data_specs",
        help="Data spec name to estimate. Repeat to estimate multiple. Defaults to config run_data_specs.",
    )
    parser.add_argument("--quick", action="store_true", help="Use a tiny run for smoke testing.")
    parser.add_argument(
        "--positive",
        action="append",
        default=[],
        help="Force coefficients nonnegative for this run. Repeat or comma-separate, e.g. --positive kappa,theta.",
    )
    parser.add_argument(
        "--no-coefficient-constraints",
        action="store_true",
        help="Disable coefficient hard constraints even if configured in models.yaml.",
    )
    args = parser.parse_args()

    config = load_model_config(args.config)
    defaults = config.get("defaults", {})
    data_specs = configured_data_specs(config, args.data_specs)
    data = pd.read_csv(args.data, parse_dates=["DATE"]).set_index("DATE")

    n_iter = 80 if args.quick else int(defaults.get("n_iter", 12000))
    burn = 40 if args.quick else int(defaults.get("burn", 4000))
    thin = 2 if args.quick else int(defaults.get("thin", 5))
    chains = 2 if args.quick else int(defaults.get("chains", 2))
    seed = int(defaults.get("seed", 12345))
    n_transform = defaults.get("n_transform", "log100")
    covariance_structure = defaults.get("covariance_structure", "e_zeta_only")
    coefficient_constraints = coefficient_constraints_from_config(
        defaults,
        positive=args.positive,
        disabled=args.no_coefficient_constraints,
    )

    for data_spec_name, data_spec in data_specs.items():
        for model in config.get("models", ["ces", "hsa_steady", "hsa_dynamic", "hsa_full"]):
            print(f"Estimating {model} [{data_spec_name}]...")
            run_model(
                model,
                data=data,
                data_spec=data_spec,
                prior_specs=args.priors,
                n_iter=n_iter,
                burn=burn,
                thin=thin,
                chains=chains,
                seed=seed,
                n_transform=n_transform,
                covariance_structure=covariance_structure,
                coefficient_constraints=coefficient_constraints,
            )


if __name__ == "__main__":
    main()
