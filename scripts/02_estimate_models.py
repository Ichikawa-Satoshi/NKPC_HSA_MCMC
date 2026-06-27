from __future__ import annotations

import argparse

import pandas as pd

from _bootstrap import ROOT
from nkpc_hsa.config import coefficient_constraints_from_config, configured_data_specs, load_model_config
from nkpc_hsa.data.transforms import DEFAULT_N_TRANSFORM
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
        "--model",
        action="append",
        dest="models",
        help="Model name to estimate. Repeat to estimate multiple. Defaults to configs/models.yaml.",
    )
    parser.add_argument(
        "--positive",
        action="append",
        default=[],
        help=(
            "Force coefficients nonnegative for this run. Repeat or comma-separate, e.g. --positive kappa,theta. "
            "Use kappa_t to require the whole time-varying kappa path to be nonnegative."
        ),
    )
    parser.add_argument(
        "--no-coefficient-constraints",
        action="store_true",
        help="Disable coefficient hard constraints even if configured in models.yaml.",
    )
    parser.add_argument(
        "--no-ar2-stationarity",
        action="store_true",
        help="Do not enforce AR(2) stationarity for HSA Nhat dynamics.",
    )
    parser.add_argument(
        "--ar2-max-tries",
        type=int,
        default=None,
        help="Maximum rejection-sampling proposals for stationary AR(2) coefficients before keeping the previous value.",
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
    n_transform = defaults.get("n_transform", DEFAULT_N_TRANSFORM)
    covariance_structure = defaults.get("covariance_structure", "e_zeta_only")
    enforce_stationary = not args.no_ar2_stationarity and bool(defaults.get("enforce_stationary", True))
    ar2_max_tries = int(args.ar2_max_tries or defaults.get("ar2_max_tries", 2000))
    coefficient_constraints = coefficient_constraints_from_config(
        defaults,
        positive=args.positive,
        disabled=args.no_coefficient_constraints,
    )
    models = list(args.models or config.get("models", ["ces", "hsa_steady", "hsa_dynamic", "hsa_full"]))

    for data_spec_name, data_spec in data_specs.items():
        for model in models:
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
                enforce_stationary=enforce_stationary,
                ar2_max_tries=ar2_max_tries,
            )


if __name__ == "__main__":
    main()
