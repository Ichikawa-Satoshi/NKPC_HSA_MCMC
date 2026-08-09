"""Estimate the MA(3) error-structure runs.

The error_robustness counterpart of ``scripts/02_estimate_models.py``. It reads
data specs from ``configs/models.yaml`` (via ``configs/error_robustness.yaml``),
merges the psi prior into the chosen ``priors_*.yaml``, and writes run
directories under ``results/error_robustness/runs/`` using the *same* directory
names ``results/runs/`` uses, so the two trees can be compared cell by cell.

Nothing in ``results/runs/`` is read for writing or modified.

    python scripts/er_02_estimate.py --quick                  # smoke test
    python scripts/er_02_estimate.py                          # every configured cell
    python scripts/er_02_estimate.py --model hsa_steady --data-spec unemployment_gap_core
    python scripts/er_02_estimate.py --ma-order 0             # the nested iid control
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

import _bootstrap  # noqa: F401
from _bootstrap import DATA_DIR, ROOT
from nkpc_hsa.config import configured_data_specs, load_yaml
from nkpc_hsa.error_robustness.runner import ERROR_ROBUSTNESS_RUNS, run_model_ma3

CONFIG = ROOT / "configs" / "error_robustness.yaml"


def _resolve_priors(prior_name: str, config: dict) -> dict:
    """Load priors_<name>.yaml and merge the psi block from the robustness config."""
    prior_path = ROOT / "configs" / f"priors_{prior_name}.yaml"
    if not prior_path.exists():
        raise SystemExit(f"missing prior file {prior_path}")
    priors = dict(load_yaml(prior_path))
    priors.update(dict(config.get("psi_prior", {}) or {}))
    return priors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(CONFIG))
    parser.add_argument("--data", default=str(DATA_DIR / "processed" / "model_ready.csv"))
    parser.add_argument("--model", action="append", dest="models")
    parser.add_argument("--data-spec", action="append", dest="data_specs")
    parser.add_argument("--prior", action="append", dest="priors")
    parser.add_argument("--ma-order", type=int, default=None,
                        help="Override the MA order. 0 estimates the nested iid control.")
    parser.add_argument("--runs-dir", type=Path, default=ERROR_ROBUSTNESS_RUNS)
    parser.add_argument("--quick", action="store_true", help="Tiny run for smoke testing.")
    args = parser.parse_args()

    config = load_yaml(args.config)
    base_config_path = ROOT / "configs" / str(config.get("base_config", "models.yaml"))
    defaults = dict(config.get("defaults", {}) or {})

    models = args.models or list(config.get("run_models", []))
    prior_names = args.priors or list(config.get("run_priors", ["baseline"]))
    spec_names = args.data_specs or list(config.get("run_data_specs", []))
    if not (models and prior_names and spec_names):
        raise SystemExit("nothing to estimate: check run_models / run_priors / run_data_specs")

    data = pd.read_csv(args.data, parse_dates=["DATE"]).set_index("DATE")
    specs = configured_data_specs(load_yaml(base_config_path), spec_names)

    ma_order = args.ma_order if args.ma_order is not None else int(defaults.get("ma_order", 3))
    n_iter = 900 if args.quick else int(defaults.get("n_iter", 12000))
    burn = 300 if args.quick else int(defaults.get("burn", 4000))
    thin = 1 if args.quick else int(defaults.get("thin", 5))
    chains = 1 if args.quick else int(defaults.get("chains", 2))

    total = len(models) * len(prior_names) * len(specs)
    print(f"{total} cells -> {args.runs_dir}  (ma_order={ma_order}, "
          f"n_iter={n_iter}, burn={burn}, thin={thin}, chains={chains})")

    done = 0
    for prior_name in prior_names:
        priors = _resolve_priors(prior_name, config)
        for model in models:
            for spec_name, spec in specs.items():
                done += 1
                label = f"[{done}/{total}] {model} / {spec_name} / {prior_name}"
                started = time.time()
                try:
                    run_model_ma3(
                        model,
                        data=data,
                        data_spec=spec,
                        prior_specs=priors,
                        prior_name=prior_name,
                        n_iter=n_iter,
                        burn=burn,
                        thin=thin,
                        chains=chains,
                        seed=int(defaults.get("seed", 12345)),
                        n_transform=str(defaults.get("n_transform", "log100_centered10")),
                        competition_measurement=defaults.get("competition_measurement"),
                        coefficient_constraints=defaults.get("coefficient_constraints"),
                        enforce_stationary=bool(defaults.get("enforce_stationary", True)),
                        ar2_max_tries=int(defaults.get("ar2_max_tries", 2000)),
                        ma_order=ma_order,
                        n_psi_steps=int(defaults.get("n_psi_steps", 2)),
                        psi_init_scale=float(defaults.get("psi_init_scale", 0.08)),
                        runs_root=args.runs_dir,
                    )
                except Exception as exc:  # keep the sweep going; report at the end
                    print(f"{label}: FAILED after {time.time() - started:.0f}s -- {exc}")
                    continue
                print(f"{label}: done in {time.time() - started:.0f}s")

    print(f"\nruns written under {args.runs_dir}")


if __name__ == "__main__":
    main()
