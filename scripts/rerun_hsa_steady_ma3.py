"""HSA steady with MA(3) inflation-equation errors, mixed-frequency design.

Answers the question the alpha == 0 experiment raised: delta's sign turned out to depend
on how the overlapping-YoY construction of the inflation series is handled, and neither
the baseline (absorb it into a lagged-inflation coefficient) nor the restriction
alpha == 0 (ignore it, and leave a badly misspecified equation) handles it properly.
This variant models the overlap in the error process instead.

Runs are marked ``constraint_spec = "ma3_errors"`` so the report's run selector, which
requires ``unrestricted``, cannot pick them up as ordinary hsa_steady cells.

    python scripts/rerun_hsa_steady_ma3.py [--quick]
"""
from __future__ import annotations

import argparse, json, time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

import _bootstrap  # noqa: F401
from _bootstrap import ROOT

from nkpc_hsa.config import configured_data_specs, load_model_config
from nkpc_hsa.data.competition import build_competition_observation, load_raw_annual_competition_series
from nkpc_hsa.gibbs.hsa_steady_ma.model import func_nkpc_hsa_steady_ma
from nkpc_hsa.inference.wrappers import _coerce_model_data, _transform_annual_competition_like_quarterly
from nkpc_hsa.models.common import prior_specs_to_internal
from nkpc_hsa.report.cpi_ppi_spec import INFLATION_SPECS

OUT = ROOT / "results" / "ma3_errors"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    cfg = load_model_config(); defaults = cfg.get("defaults", {})
    specs = configured_data_specs(cfg, list(cfg.get("data_specs", {})))
    df = pd.read_csv(ROOT / "data" / "processed" / "model_ready.csv", parse_dates=["DATE"]).set_index("DATE")
    pri = prior_specs_to_internal(yaml.safe_load((ROOT / "configs" / "priors_baseline.yaml").read_text()))

    n_iter = 400 if args.quick else int(defaults.get("n_iter", 12000))
    burn = 200 if args.quick else int(defaults.get("burn", 4000))
    thin = 2 if args.quick else int(defaults.get("thin", 5))
    chains = 2 if args.quick else int(defaults.get("chains", 2))

    cells = sorted({s for a in INFLATION_SPECS.values() for s in a.values()})
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    t0 = time.time()
    for i, spec in enumerate(cells, 1):
        sd = specs[spec]
        md = _coerce_model_data(df, data_spec=sd)
        sample_idx = df[[sd["pi_col"], sd["pi_prev_col"], sd["pi_expect_col"],
                         sd["x_col"], sd["x_prev_col"], sd["n_col"]]].dropna().index.to_period("Q")
        ann = load_raw_annual_competition_series(sd["n_col"])
        ann = ann.loc[ann.index.isin(sorted({int(p.year) for p in sample_idx}))]
        annT = pd.Series(_transform_annual_competition_like_quarterly(ann, md["N"], "log100_centered10"),
                         index=ann.index)
        N_obs = build_competition_observation(annT, sample_idx, frequency="annual_q4").N_obs

        print(f"[{i}/{len(cells)}] {spec}", flush=True)
        per_chain = []
        for ch in range(chains):
            r = func_nkpc_hsa_steady_ma(
                md["pi"], md["pi_prev"], md["pi_expect"], md["x"], md["x_prev"], N_obs,
                n_burn=burn, n_keep=n_iter - burn, priors=pri,
                opts={"seed": 12345 + 1000 * ch, "store_every": thin, "psi_step": 0.10},
            )
            per_chain.append(r)
        get = lambda k: np.concatenate([c[k]["draws"] for c in per_chain])
        d = get("delta")
        kt = np.concatenate([c["state_draws"]["kappa_t"] for c in per_chain], axis=0)
        rows.append({
            "spec": spec,
            "delta_mean": float(d.mean()),
            "delta_lo": float(np.quantile(d, .025)), "delta_hi": float(np.quantile(d, .975)),
            "alpha_mean": float(get("alpha").mean()),
            "psi1": float(get("psi_1").mean()), "psi2": float(get("psi_2").mean()),
            "psi3": float(get("psi_3").mean()),
            "sigma_e": float(get("sigma_e").mean()),
            "kappa_start": float(kt[:, 0].mean()), "kappa_end": float(kt[:, -1].mean()),
            "psi_accept": float(np.mean([c["model"]["psi_acceptance_rate"] for c in per_chain])),
        })
        print(f"    delta={rows[-1]['delta_mean']:+.4f} [{rows[-1]['delta_lo']:+.4f},{rows[-1]['delta_hi']:+.4f}]"
              f"  psi=({rows[-1]['psi1']:.2f},{rows[-1]['psi2']:.2f},{rows[-1]['psi3']:.2f})"
              f"  ({time.time()-t0:.0f}s)", flush=True)
    tab = pd.DataFrame(rows)
    tab.to_csv(OUT / "ma3_summary.csv", index=False)
    print(f"\n{tab.to_string(index=False)}\nsaved -> {OUT/'ma3_summary.csv'}")


if __name__ == "__main__":
    main()
