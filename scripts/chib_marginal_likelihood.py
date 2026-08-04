"""Conditional marginal likelihood  p(pi | x, N_obs, M)  for CES vs HSA steady.

This replaces the previous implementation, which had five defects (all fixed
here, all documented in the module it now calls,
``nkpc_hsa.gibbs.conditional_ml``):

  1. the Kalman routines initialised the trend at ``N_obs[0]`` instead of the
     estimating model's own ``m0 = (0,0,0)``, evaluating a different model from
     the one estimated and using the first observation twice;
  2. the AR(2) posterior ordinate used the no-initial-lag regression, not the
     initial-lag conditional the production sampler actually draws from;
  3. the truncated ``(rho_1, rho_2)`` prior and conditional carried no
     normalising constant;
  4. ``family="full"`` conditioned on posterior-MEAN latent states and reported
     the result as a marginal likelihood -- a plug-in, now raising;
  5. differences between ``p(pi)`` and ``p(pi, N_obs)`` were written out under
     ``lnBF_*`` column names; those are densities of different data and are no
     longer produced at all.

Additionally the posterior ordinate is now Chib's sequential factorisation with
explicit reduced Gibbs runs, rather than evaluating every block's conditional on
draws from the full run.

The numbers this prints are NOT promoted into the paper by this script. It
writes them next to the previous values so the change can be reviewed first.

    python scripts/chib_marginal_likelihood.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

import _bootstrap  # noqa: F401
from _bootstrap import ROOT

from nkpc_hsa.config import configured_data_specs, load_model_config
from nkpc_hsa.data.transforms import DEFAULT_N_TRANSFORM, transform_competition_series
from nkpc_hsa.gibbs.conditional_ml import conditional_marginal_likelihood
from nkpc_hsa.inference.wrappers import _coerce_model_data
from nkpc_hsa.models.common import prior_specs_to_internal

TAB = ROOT / "results" / "appendix_particle_gibbs" / "tables"
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


def build_data(spec_dict, df):
    md = _coerce_model_data(df, data_spec=spec_dict)
    return {
        "pi": md["pi"],
        "pi_prev": md["pi_prev"],
        "pi_expect": md["pi_expect"],
        "x": md["x"],
        "x_prev": md["x_prev"],
        "N": np.asarray(transform_competition_series(md["N"], transform=DEFAULT_N_TRANSFORM), float),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-burn", type=int, default=1500)
    ap.add_argument("--n-keep", type=int, default=3000)
    args = ap.parse_args()

    specs = configured_data_specs(load_model_config())
    df = pd.read_csv(ROOT / "data" / "processed" / "model_ready.csv")
    prior_spec = yaml.safe_load((ROOT / "configs" / "priors_baseline.yaml").read_text())
    priors_internal = prior_specs_to_internal(prior_spec)
    pri = physical_priors(prior_spec)

    print("Conditional marginal likelihood  log p(pi | x, N_obs, M)")
    print("Chib (1995) with explicit reduced Gibbs runs; states integrated out exactly.")
    print(f"reduced-run length: burn={args.n_burn} keep={args.n_keep}\n")

    rows = []
    for price, spec_name in UNEMP.items():
        data = build_data(specs[spec_name], df)
        entry = {"price": price, "spec": spec_name}
        for label, family in (("ces", "ces"), ("hsa_steady", "steady")):
            res = conditional_marginal_likelihood(
                data, priors_internal, pri, family=family,
                n_burn=args.n_burn, n_keep=args.n_keep,
            )
            entry[f"logCML_{label}"] = round(res.log_conditional_marginal_likelihood, 2)
            entry[f"loglik_{label}"] = round(res.log_likelihood, 2)
            entry[f"logprior_{label}"] = round(res.log_prior, 2)
            entry[f"logord_{label}"] = round(res.log_posterior_ordinate, 2)
            print(f"  {price:>13s} / {label:<11s} "
                  f"logCML={res.log_conditional_marginal_likelihood:9.2f}  "
                  f"(lik {res.log_likelihood:8.2f}, prior {res.log_prior:7.2f}, "
                  f"ord {res.log_posterior_ordinate:8.2f})", flush=True)
        rows.append(entry)

    table = pd.DataFrame(rows)
    print("\n" + table.to_string(index=False))
    table.to_csv(TAB / "conditional_ml_corrected.csv", index=False)

    print("""
================================================================================
NO BAYES FACTOR IS REPORTED, AND THE COMPONENTS ABOVE MUST NOT BE DIFFERENCED.
================================================================================
Every enumerated defect is fixed and unit-tested, but fixing them exposed a
problem with the ESTIMAND, not the implementation, and the difference of the two
logCML columns is not a valid Bayes factor.

What is computed above is

    log integral p(pi | x, N_obs, theta) p(theta) d theta

i.e. the parameters are weighted by their PRIOR. For CES that is exactly
p(pi | x, CES). For HSA steady it is not what we want: the firm-count block
(rho_1, rho_2, sigma_u^2, n, sigma_eps^2, sigma_N^2) is pinned almost entirely by
N_obs, which we are conditioning on -- yet the prior-weighted integral still
charges HSA steady a full Occam penalty for those six parameters. In the core-CPI
cell that penalty is about 50 log points, which swamps the roughly 9 log points of
extra inflation fit, and would make CES look decisively better for a reason that
has nothing to do with the inflation mechanism.

The coherent object weights the parameters by their posterior GIVEN the
conditioning data:

    p(pi | x, N_obs, M) = m(pi, N_obs | x, M) / m(N_obs | M)

with both terms full marginal likelihoods. The denominator carries the Occam
factor for the firm-count block, so it cancels instead of being double-charged.
CES has no firm-count block, so its denominator is 1 and
p(pi | x, N_obs, CES) = m(pi | x, CES).

Computing m(N_obs | M) needs a second marginal likelihood over the six-parameter
firm-count block. That is a well-posed and tractable extension of the machinery
here, but it is NOT implemented, so no model comparison is emitted.

What IS trustworthy and reusable from this module:
  * kalman_loglik  -- exact states-integrated likelihood, validated against a
    dense analytic Gaussian likelihood, with the estimating model's own initial
    state and correct missing-N handling;
  * the identity log p(pi|N) = log p(pi,N) - log p(N), verified numerically;
  * the Chib ordinate machinery with explicit reduced Gibbs runs, using the
    production samplers' own conditionals including the sampled Nhat_{-1} lag;
  * the normalised truncated (rho_1, rho_2) prior and conditional.
================================================================================
""")

    json.dump(
        {"rows": rows, "n_burn": args.n_burn, "n_keep": args.n_keep,
         "bayes_factor_emitted": False,
         "reason": "prior-weighted conditional ML double-charges the Occam factor "
                   "for the firm-count block; needs m(pi,N)/m(N) instead."},
        open(TAB / "conditional_ml_corrected.json", "w"),
        indent=2,
    )
    print(f"saved components -> {TAB / 'conditional_ml_corrected.csv'}")


if __name__ == "__main__":
    main()
