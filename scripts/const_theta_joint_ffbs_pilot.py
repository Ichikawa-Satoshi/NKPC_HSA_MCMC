"""Pilot: hsa_const_theta, alternating FFBS vs exact joint FFBS.

Runs the SAME cell (core CPI x negative unemployment gap x baseline priors,
PCHIP) under both state blocks with identical data, priors, seeds and MCMC
settings, and reports the diagnostics and posterior summaries side by side.

This is the gate that must pass before any production hsa_const_theta run is
replaced. What we require:

  * mixing improves materially (that is the point of the change);
  * posterior summaries agree within Monte Carlo error *where the old chains
    were converged enough for that comparison to mean anything*. Where the old
    chains were not converged (Nbar path Rhat ~ 2.7, bulk ESS ~ 2) a changed
    summary is evidence the old number was wrong, not that the model changed.

Writes results/const_theta_pilot/pilot.json and prints a comparison table.
Nothing under results/runs is touched.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd

import _bootstrap  # noqa: F401
from _bootstrap import ROOT

from nkpc_hsa.config import configured_data_specs, load_model_config
from nkpc_hsa.gibbs.hsa_const_theta import func_nkpc_hsa_const_theta
from nkpc_hsa.gibbs.hsa_full import func_nkpc_hsa_full_static_theta
from nkpc_hsa.inference.wrappers import _coerce_model_data
from nkpc_hsa.data.transforms import DEFAULT_N_TRANSFORM, transform_competition_series
from nkpc_hsa.models.common import prior_specs_to_internal

import yaml

OUT = ROOT / "results" / "const_theta_pilot"
OUT.mkdir(parents=True, exist_ok=True)

SPEC = "unemployment_gap_core"
PRIOR = "baseline"
N_ITER, BURN, THIN, CHAINS, SEED = 12000, 4000, 5, 2, 12345

SCALARS = ["alpha", "kappa_0", "delta", "theta", "rho1", "rho2", "n", "phi_1",
           "sigma_e", "sigma_u", "sigma_eps", "sigma_N"]
PATHS = ["Nbar", "Nhat", "kappa_t"]


def _run(sampler, data, priors_internal, label):
    seeds = np.random.SeedSequence(SEED).spawn(CHAINS)
    chains = []
    t0 = time.time()
    for child in seeds:
        result = sampler(
            pi_data=data["pi"],
            pi_prev_data=data["pi_prev"],
            Epi_data=data["pi_expect"],
            x_data=data["x"],
            x_prev_data=data["x_prev"],
            N_data=data["N"],
            n_burn=BURN,
            n_keep=N_ITER - BURN,
            priors=priors_internal,
            opts={
                "seed": int(child.generate_state(1)[0]),
                "store_every": THIN,
                "verbose": False,
                "enforce_stationary": True,
                "ar2_max_tries": 2000,
            },
            orth=False,
        )
        draws = {}
        for key, value in result.items():
            if key in {"priors", "opts", "model"}:
                continue
            if key == "state_draws":
                draws.update({k: np.asarray(v, float) for k, v in value.items()})
            elif isinstance(value, dict) and "draws" in value:
                draws[key] = np.asarray(value["draws"], float)
        chains.append(draws)
    seconds = time.time() - t0
    stacked = {k: np.stack([c[k] for c in chains], axis=0) for k in chains[0]}
    print(f"  {label}: {seconds:.0f}s")
    return stacked, seconds


def _diag(stacked, name):
    if name not in stacked:
        return None
    arr = stacked[name]
    if float(np.nanstd(arr)) <= 0:
        return None
    rhat = float(np.nanmax(np.asarray(az.rhat(az.convert_to_dataset(arr)).x)))
    ess = float(np.nanmin(np.asarray(az.ess(az.convert_to_dataset(arr), method="bulk").x)))
    return rhat, ess


def _summary(stacked, name):
    if name not in stacked:
        return None
    flat = np.asarray(stacked[name]).reshape(-1)
    return float(np.mean(flat)), float(np.quantile(flat, 0.025)), float(np.quantile(flat, 0.975))


def main():
    specs = configured_data_specs(load_model_config())
    df = pd.read_csv(ROOT / "data" / "processed" / "model_ready.csv")
    md = _coerce_model_data(df, data_spec=specs[SPEC])
    data = dict(md)
    data["N"] = transform_competition_series(md["N"], transform=DEFAULT_N_TRANSFORM)

    prior_dict = yaml.safe_load((ROOT / "configs" / f"priors_{PRIOR}.yaml").read_text())
    priors_internal = prior_specs_to_internal(prior_dict)

    print(f"Pilot cell: {SPEC} / {PRIOR} / PCHIP, T={len(data['pi'])}")
    old, old_s = _run(func_nkpc_hsa_full_static_theta, data, priors_internal, "alternating FFBS")
    new, new_s = _run(func_nkpc_hsa_const_theta, data, priors_internal, "joint FFBS")

    rows = []
    for name in SCALARS + PATHS:
        do, dn = _diag(old, name), _diag(new, name)
        so, sn = _summary(old, name), _summary(new, name)
        if do is None and dn is None:
            continue
        row = {"quantity": name}
        row["old Rhat"] = None if do is None else round(do[0], 3)
        row["new Rhat"] = None if dn is None else round(dn[0], 3)
        row["old ESS"] = None if do is None else round(do[1], 1)
        row["new ESS"] = None if dn is None else round(dn[1], 1)
        if name in SCALARS:
            row["old mean"] = None if so is None else round(so[0], 4)
            row["new mean"] = None if sn is None else round(sn[0], 4)
            row["old 95%"] = None if so is None else [round(so[1], 4), round(so[2], 4)]
            row["new 95%"] = None if sn is None else [round(sn[1], 4), round(sn[2], 4)]
        rows.append(row)

    table = pd.DataFrame(rows)
    print()
    print(table.to_string(index=False))

    # Posterior-invariance check on the scalars the old chains could actually
    # resolve: compare means in units of the pooled Monte Carlo standard error.
    invariance = []
    for name in SCALARS:
        do, dn = _diag(old, name), _diag(new, name)
        so, sn = _summary(old, name), _summary(new, name)
        if None in (do, dn, so, sn):
            continue
        sd_o = float(np.std(np.asarray(old[name]).reshape(-1), ddof=1))
        sd_n = float(np.std(np.asarray(new[name]).reshape(-1), ddof=1))
        mcse = np.sqrt(sd_o**2 / max(do[1], 1.0) + sd_n**2 / max(dn[1], 1.0))
        invariance.append(
            {
                "parameter": name,
                "old_ess": round(do[1], 1),
                "z_mean_diff": round(abs(so[0] - sn[0]) / max(mcse, 1e-12), 2),
                "old_chains_usable": bool(do[0] <= 1.01 and do[1] >= 400),
            }
        )
    inv = pd.DataFrame(invariance)
    print("\nPosterior invariance (|mean difference| in pooled MC se):")
    print(inv.to_string(index=False))

    payload = {
        "cell": SPEC,
        "prior": PRIOR,
        "mcmc": {"n_iter": N_ITER, "burn": BURN, "thin": THIN, "chains": CHAINS, "seed": SEED},
        "seconds": {"alternating_ffbs": old_s, "joint_ffbs": new_s},
        "comparison": json.loads(table.to_json(orient="records")),
        "invariance": json.loads(inv.to_json(orient="records")),
    }
    (OUT / "pilot.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    table.to_csv(OUT / "pilot.csv", index=False)
    print(f"\nSaved {OUT / 'pilot.json'}")


if __name__ == "__main__":
    main()
