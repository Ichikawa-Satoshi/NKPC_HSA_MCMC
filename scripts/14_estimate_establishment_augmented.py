"""Estimate the experimental quarterly-establishment HSA const-theta specification.

This run is deliberately kept outside ``results/runs`` so it cannot enter the
production report.  The specification uses the 79 quarters from 1993Q2 through
2012Q4 and estimates the loading in

    Ehat_obs_t = lambda_E * Nhat_t + omega_t.
"""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from _bootstrap import DATA_DIR, RESULTS_DIR, ROOT
from nkpc_hsa.config import configured_data_specs, load_model_config, load_yaml
from nkpc_hsa.inference.wrappers import run_model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "models.yaml")
    parser.add_argument("--data", type=Path, default=DATA_DIR / "processed" / "model_ready.csv")
    parser.add_argument("--prior", type=Path, default=ROOT / "configs" / "priors_baseline.yaml")
    parser.add_argument("--chains", type=int, default=2)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--quick", action="store_true", help="Use 400 iterations and 200 burn-in for a smoke run.")
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()

    config = load_model_config(args.config)
    defaults = dict(config.get("defaults", {}) or {})
    data_spec = configured_data_specs(
        config, ["unemployment_gap_core_establishment"]
    )["unemployment_gap_core_establishment"]
    data = pd.read_csv(args.data, parse_dates=["DATE"]).set_index("DATE")
    priors = load_yaml(args.prior)

    n_iter = 400 if args.quick else int(defaults.get("n_iter", 12000))
    burn = 200 if args.quick else int(defaults.get("burn", 4000))
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = RESULTS_DIR / "experiments" / "establishment_augmented" / run_id

    idata = run_model(
        "hsa_const_theta",
        data=data,
        data_spec=data_spec,
        prior_specs=priors,
        prior_name=args.prior.stem.replace("priors_", ""),
        n_iter=n_iter,
        burn=burn,
        thin=int(defaults.get("thin", 5)),
        chains=args.chains,
        seed=args.seed,
        n_transform=str(defaults.get("n_transform", "log100_centered10")),
        competition_measurement={"frequency": "annual_q4", "annual_timing": "q4"},
        enforce_stationary=bool(defaults.get("enforce_stationary", True)),
        ar2_max_tries=int(defaults.get("ar2_max_tries", 2000)),
        run_id=run_id,
        run_dir=run_dir,
        save=not args.no_save,
    )

    rows = []
    for name in ("theta", "delta", "lambda_E", "rho_1", "rho_2", "sigma_E"):
        draws = np.asarray(idata.posterior[name], dtype=float).reshape(-1)
        rows.append(
            {
                "parameter": name,
                "mean": float(np.mean(draws)),
                "sd": float(np.std(draws, ddof=1)),
                "q025": float(np.quantile(draws, 0.025)),
                "q500": float(np.quantile(draws, 0.5)),
                "q975": float(np.quantile(draws, 0.975)),
            }
        )
    print(pd.DataFrame(rows).set_index("parameter").to_string())
    if not args.no_save:
        print(f"Saved experimental run to {run_dir}")


if __name__ == "__main__":
    main()
