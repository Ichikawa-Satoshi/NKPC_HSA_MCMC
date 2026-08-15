"""Estimate the F0/U/R1/R2/R3 theory hierarchy in its own result namespace."""

from __future__ import annotations

import argparse

import pandas as pd

from _bootstrap import DATA_DIR, ROOT
from nkpc_hsa.config import configured_theory_data_specs, load_model_config
from nkpc_hsa.dataprep.transforms import DEFAULT_N_TRANSFORM
from nkpc_hsa.inference.wrappers import run_model
from nkpc_hsa.progress import STYLES as PROGRESS_STYLES
from nkpc_hsa.theory_models import THEORY_MODELS


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / "models.yaml"))
    parser.add_argument("--priors", default=str(ROOT / "configs" / "priors_baseline.yaml"))
    parser.add_argument("--data", default=str(DATA_DIR / "processed" / "model_ready.csv"))
    parser.add_argument("--data-spec", action="append", dest="data_specs")
    parser.add_argument("--model", action="append", dest="models")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--competition-frequency", choices=["quarterly_interpolated", "annual_q4"], default=None)
    parser.add_argument(
        "--progress",
        choices=list(PROGRESS_STYLES),
        default=None,
        help=(
            "Sampler progress display. 'auto' (default) draws a bar when stderr is a terminal "
            "and stays silent otherwise; 'stream' emits the machine-readable events the "
            "production driver aggregates."
        ),
    )
    args = parser.parse_args()

    config = load_model_config(args.config)
    defaults = dict(config.get("defaults", {}) or {})
    theory_defaults = dict(config.get("theory_defaults", {}) or {})
    specs = configured_theory_data_specs(config, args.data_specs)
    models = list(args.models or config.get("theory_models", THEORY_MODELS))
    unknown = sorted(set(models) - set(THEORY_MODELS))
    if unknown:
        raise ValueError(f"Theory driver received non-theory slugs: {unknown}")
    data = pd.read_csv(args.data, parse_dates=["DATE"]).set_index("DATE")
    measurement = dict(defaults.get("competition_measurement", {}) or {})
    if args.competition_frequency:
        measurement["frequency"] = args.competition_frequency

    sampling_by_model = dict(theory_defaults.get("sampling_by_model", {}) or {})
    for spec_name, spec in specs.items():
        allowed = set(spec.get("models", models) or models)
        models_for_spec = [model for model in models if model in allowed]
        for model in models_for_spec:
            sampling = dict(sampling_by_model.get(model, {}) or {})
            n_iter = 80 if args.quick else int(sampling.get("n_iter", defaults.get("n_iter", 12000)))
            burn = 40 if args.quick else int(sampling.get("burn", defaults.get("burn", 4000)))
            thin = 2 if args.quick else int(sampling.get("thin", defaults.get("thin", 5)))
            chains = 2 if args.quick else int(sampling.get("chains", defaults.get("chains", 2)))
            # Flushed because the production driver reads this through a pipe.
            print(f"Estimating {model} [{spec_name}]...", flush=True)
            run_model(
                model,
                data=data,
                data_spec=spec,
                prior_specs=args.priors,
                n_iter=n_iter,
                burn=burn,
                thin=thin,
                chains=chains,
                seed=int(defaults.get("seed", 12345)),
                n_transform=str(defaults.get("n_transform", DEFAULT_N_TRANSFORM)),
                competition_measurement=measurement,
                enforce_stationary=bool(defaults.get("enforce_stationary", True)),
                ar2_max_tries=int(defaults.get("ar2_max_tries", 2000)),
                n_particles=int(defaults.get("n_particles", 512)),
                inflation_observation=str(spec["inflation_observation"]),
                zeta0=float(theory_defaults.get("zeta0", 6.0)),
                zeta0_treatment=str(theory_defaults.get("zeta0_treatment", "fixed")),
                require_path_positivity=bool(theory_defaults.get("require_path_positivity", True)),
                progress=args.progress,
            )


if __name__ == "__main__":
    main()
