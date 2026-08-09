"""Conditional marginal likelihood  p(pi | x, N_obs, M)  for CES vs HSA steady.

The estimand is the inflation mechanism, so the two models must be compared as
densities of the same data given the same conditioning set:

    log p(pi | x, N_obs, M) = log m(pi, N_obs, x | M) - log m(N_obs | M) - log m(x)

Every term is a Chib (1995) marginal likelihood. The Gibbs posterior conditions
on pi, N_obs *and* x -- ``sigma_zeta2`` is drawn from the activity-equation
residuals and both ``phi_1`` and ``sigma_zeta2`` are ordinate blocks -- so the
conditioning set is (x, N_obs) and both Occam factors have to be refunded. They
factorise because p(N|theta) touches only the firm-count block and
p(x|phi_1,sigma_zeta2) only the activity block, and the two are a priori
independent. m(x) is identical for the two models; it is computed rather than
cancelled by assertion, so that the run itself shows both models conditioned on
the same quantity. CES has no firm-count block, so its m(N) term is absent.

An earlier revision of this script stopped short of a Bayes factor for a good
reason: it fed the *conditional* likelihood into Chib's identity while taking the
ordinate from the *joint* posterior, which does not estimate anything -- the
identity collapses to m(pi,N|x)/p(N|x,theta*) and still depends on theta*. On one
seed that produced a 660-log-point error. Both defects are fixed here and pinned
by tests: theta* invariance, seed-to-seed stability, and an effective-draw guard
that raises rather than returning an ordinate factor carried by a handful of
draws.

    python scripts/chib_marginal_likelihood.py            # main cell, 3 seeds
    python scripts/chib_marginal_likelihood.py --all-cells
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

import _bootstrap  # noqa: F401
from _bootstrap import DATA_DIR, RESULTS_DIR, ROOT

from nkpc_hsa.config import configured_data_specs, load_model_config
from nkpc_hsa.dataprep.transforms import DEFAULT_N_TRANSFORM, transform_competition_series
from nkpc_hsa.gibbs.conditional_ml import (
    OrdinateNotIdentified,
    activity_marginal_likelihood,
    conditional_comparison,
)
from nkpc_hsa.reporting.cpi_ppi_spec import INFLATION_SPECS
from nkpc_hsa.inference.wrappers import (
    _coerce_model_data,
    _prepare_competition_measurement,
    model_sample_index,
)
from nkpc_hsa.models.common import prior_specs_to_internal

TAB = RESULTS_DIR / "evidence" / "tables"
TAB.mkdir(parents=True, exist_ok=True)

UNEMP = {
    "Headline CPI": "unemployment_gap",
    "Core CPI": "unemployment_gap_core",
    "PPI": "unemployment_gap_ppi",
}

# Physical-unit prior set expected by the ordinate/prior evaluation.
_PAIR_KEYS = ["alpha", "kappa", "kappa_0", "delta", "theta", "theta_0", "gamma",
              "phi_1", "rho_1", "rho_2", "n"]
_SCALAR_KEYS = ["a_e", "b_e", "a_z", "b_z", "a_u", "b_u", "a_eps", "b_eps", "a_N", "b_N"]


def physical_priors(spec: dict) -> dict:
    out: dict = {}
    for key in _PAIR_KEYS:
        if key in spec:
            out[key] = (float(spec[key][0]), float(spec[key][1]))
    for key in _SCALAR_KEYS:
        if key in spec:
            out[key] = float(spec[key])
    # lambda_ez is not in the YAML files; it is a sampler default.
    out.setdefault("lambda_ez", (0.0, 0.5))
    return out


def build_data(spec_dict, df, *, frequency: str):
    """Model-ready arrays with the firm count on the requested observation scheme.

    The competition series must be built the same way the estimation built it,
    through ``_prepare_competition_measurement``. Calling
    ``transform_competition_series`` directly -- which is what this script used
    to do -- silently produces the PCHIP-interpolated series with all 124
    quarters observed, so the marginal likelihood was computed under the
    comparison design no matter which design was asked for. The Kalman routine
    itself has always handled the missing quarters; nothing upstream was giving
    it any.
    """
    md = _coerce_model_data(df, data_spec=spec_dict)
    sample_index = model_sample_index(df, spec_dict)
    context = _prepare_competition_measurement(
        model="hsa_steady",
        data=df,
        data_spec=spec_dict,
        model_data=md,
        sample_index=sample_index,
        n_transform=DEFAULT_N_TRANSFORM,
        competition_measurement={"frequency": frequency, "annual_timing": "q4"},
    )
    n_obs = context.get("N_obs_used")
    if n_obs is None:
        raise SystemExit(
            f"no firm-count observation could be built for frequency={frequency!r}"
        )
    return {
        "pi": md["pi"],
        "pi_prev": md["pi_prev"],
        "pi_expect": md["pi_expect"],
        "x": md["x"],
        "x_prev": md["x_prev"],
        "N": np.asarray(n_obs, dtype=float),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-burn", type=int, default=3000)
    ap.add_argument("--n-keep", type=int, default=30000,
                    help="Reduced-run length. The AR(2)/state block mixes slowly; "
                         "shorter runs make the rho ordinate factor unreliable and "
                         "the guard will say so.")
    ap.add_argument("--all-cells", action="store_true",
                    help="All nine baseline cells. Without it, the main cell only.")
    ap.add_argument("--seeds", type=int, nargs="+", default=[90210, 11311, 24601],
                    help="Repeat seeds for the main cell; other cells use the first.")
    ap.add_argument("--frequency", default="annual_q4",
                    choices=["annual_q4", "quarterly_interpolated"])
    args = ap.parse_args()

    specs = configured_data_specs(load_model_config())
    # DATE must be the index: the annual-Q4 scheme needs a quarterly period
    # index to place each annual observation in its own Q4.
    df = pd.read_csv(
        DATA_DIR / "processed" / "model_ready.csv", parse_dates=["DATE"]
    ).set_index("DATE")
    prior_spec = yaml.safe_load((ROOT / "configs" / "priors_baseline.yaml").read_text())
    priors_internal = prior_specs_to_internal(prior_spec)
    pri = physical_priors(prior_spec)

    cells = (
        [(price, activity, activity_specs[activity])
         for price, activity_specs in INFLATION_SPECS.items()
         for activity in ("Unemployment gap", "HP output gap", "BN output gap")]
        if args.all_cells else
        [("Core CPI", "Unemployment gap", INFLATION_SPECS["Core CPI"]["Unemployment gap"])]
    )
    main_cell = ("Core CPI", "Unemployment gap")

    print("Conditional marginal likelihood  log p(pi | x, N_obs, M)")
    print("Chib (1995), states integrated out exactly, reduced runs in parallel.")
    print(f"design={args.frequency}  reduced runs: burn={args.n_burn} keep={args.n_keep}\n", flush=True)

    rows = []
    for price, activity, spec_name in cells:
        seeds = args.seeds if (price, activity) == main_cell else args.seeds[:1]
        data = build_data(specs[spec_name], df, frequency=args.frequency)
        for seed in seeds:
            started = time.perf_counter()
            entry = {"price": price, "activity": activity, "spec": spec_name, "seed": seed}
            try:
                # One m(x) per cell, handed to both models, so the run shows them
                # conditioned on the identical quantity rather than assuming it.
                log_m_x, _ = activity_marginal_likelihood(
                    data, pri, seed=seed, n_burn=args.n_burn, n_keep=args.n_keep)
                hsa = conditional_comparison(
                    data, priors_internal, pri, family="steady", seed=seed,
                    n_burn=args.n_burn, n_keep=args.n_keep, log_m_activity=log_m_x)
                ces = conditional_comparison(
                    data, priors_internal, pri, family="ces", seed=seed,
                    n_burn=args.n_burn, n_keep=args.n_keep, log_m_activity=log_m_x)
                delta = hsa.log_m_conditional - ces.log_m_conditional
                entry.update({
                    "log_m_joint_HSA": hsa.log_m_joint,
                    "log_m_N_HSA": hsa.log_m_firm_count,
                    "log_m_joint_CES": ces.log_m_joint,
                    "log_m_x": log_m_x,
                    "log_m_HSA_cond": hsa.log_m_conditional,
                    "log_m_CES_cond": ces.log_m_conditional,
                    "Delta_log_m": delta,
                    "BF_HSA_CES": float(np.exp(delta)),
                    "error": None,
                })
                print(f"  {price:13s} {activity:18s} seed {seed}: "
                      f"Delta={delta:+7.3f}  BF={np.exp(delta):8.1f}  "
                      f"({time.perf_counter()-started:.0f}s)", flush=True)
            except OrdinateNotIdentified as exc:
                # Report the cell as not identified rather than emitting a number
                # the estimator cannot support.
                entry["error"] = str(exc)
                print(f"  {price:13s} {activity:18s} seed {seed}: NOT IDENTIFIED "
                      f"-- {exc}", flush=True)
            rows.append(entry)
            pd.DataFrame(rows).to_csv(TAB / "conditional_ml.csv", index=False)

    table = pd.DataFrame(rows)
    done = table[table["error"].isna()] if "error" in table else table
    print("\n" + done.to_string(index=False))
    if len(args.seeds) > 1:
        main = done[(done.price == main_cell[0]) & (done.activity == main_cell[1])]
        if len(main) > 1:
            delta = main["Delta_log_m"].to_numpy(float)
            print(f"\nmain cell over {len(delta)} seeds: Delta = {delta.mean():+.3f}  "
                  f"sd {delta.std(ddof=1):.3f}  MCSE {delta.std(ddof=1)/np.sqrt(len(delta)):.3f}  "
                  f"BF = {np.exp(delta.mean()):.1f} (range {np.exp(delta.min()):.0f}-{np.exp(delta.max()):.0f})")
    not_identified = int(table["error"].notna().sum()) if "error" in table else 0
    print(f"cells reported as not identified: {not_identified}")

    json.dump({"rows": rows, "n_burn": args.n_burn, "n_keep": args.n_keep,
               "frequency": args.frequency, "seeds": args.seeds},
              open(TAB / "conditional_ml.json", "w"), indent=2, default=float)
    print(f"saved -> {TAB / 'conditional_ml.csv'}")


if __name__ == "__main__":
    main()
