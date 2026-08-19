"""Separate delta's own prior from the AR(2) prior in the prior-sensitivity result.

The reported prior sweep moves every prior at once, so the shrinkage of delta from
+0.023 (baseline) to +0.013 (tight) mixes two channels:

  * delta's own prior, N(0, sigma_delta^2), acting directly on the coefficient;
  * the AR(2) prior on (rho_1, rho_2), acting indirectly by changing how the
    firm count is split into trend and cycle, hence changing the regressor
    x_t * Nbar_t that identifies delta.

The second channel matters because the AR(2) prior is in strong tension with the
data: it is centred on a ~5-quarter oscillation with a 2-quarter half-life, while
the PCHIP posterior wants an ~82-quarter quasi-trend and the annual-Q4 posterior a
~4.3-quarter oscillation.

This script crosses the two priors while holding everything else at baseline, so
the two channels can be read off separately. The runs are diagnostic and are
written outside ``results/runs`` so they cannot enter the report run-set.

    python main_scripts/prior_decomposition_rho_delta.py [--spec unemployment_gap_core] [--quick]
"""
from __future__ import annotations

import argparse
import itertools
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

import _bootstrap  # noqa: F401
from _bootstrap import DATA_DIR, RESULTS_DIR, ROOT

from nkpc_hsa.config import configured_data_specs, load_model_config
from nkpc_hsa.inference.wrappers import ESTIMATION_REVISION, run_model

OUT = RESULTS_DIR / "prior_decomposition"
DELTA_SD = {"baseline": 0.02, "tight": 0.01}
RHO_SD = {"weak": 0.5, "baseline": 0.2, "tight": 0.1}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", default="unemployment_gap_core")
    ap.add_argument("--model", default="hsa_steady")
    ap.add_argument("--freq", default="quarterly_interpolated")
    ap.add_argument("--quick", action="store_true")
    # Rebuild the macros from the CSVs already on disk without re-estimating.
    # build_report.py uses this: the macro file is a report input and must survive
    # a clean rebuild, but the 12 cells behind it take ~45 minutes and are not
    # part of the report build.
    ap.add_argument("--macros-only", action="store_true",
                    help="Rewrite the macro file from existing CSVs; estimate nothing.")
    args = ap.parse_args()

    if args.macros_only:
        if write_decomposition_macros(model=args.model) is None:
            raise SystemExit(
                f"no decomposition CSV under {OUT}; run this script without "
                "--macros-only for --spec unemployment_gap and unemployment_gap_core first"
            )
        return

    config = load_model_config()
    defaults = config.get("defaults", {})
    specs = configured_data_specs(config, list(config.get("data_specs", {})))
    data = pd.read_csv(
        DATA_DIR / "processed" / "model_ready.csv", parse_dates=["DATE"]
    ).set_index("DATE")
    base_priors = yaml.safe_load((ROOT / "configs" / "priors_baseline.yaml").read_text())

    n_iter = 200 if args.quick else int(defaults.get("n_iter", 12000))
    burn = 100 if args.quick else int(defaults.get("burn", 4000))
    thin = 2 if args.quick else int(defaults.get("thin", 5))
    chains = 2 if args.quick else int(defaults.get("chains", 2))

    runs_dir = OUT / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    prior_dir = OUT / "priors"
    prior_dir.mkdir(parents=True, exist_ok=True)

    cells = list(itertools.product(DELTA_SD, RHO_SD))
    print(f"{len(cells)} cells: delta prior x AR(2) prior, everything else at baseline")
    t0 = time.time()
    for i, (dlab, rlab) in enumerate(cells, 1):
        priors = dict(base_priors)
        priors["delta"] = [0.0, DELTA_SD[dlab]]
        priors["rho_1"] = [0.5, RHO_SD[rlab]]
        priors["rho_2"] = [-0.5, RHO_SD[rlab]]
        name = f"delta{dlab}_rho{rlab}"
        prior_path = prior_dir / f"priors_{name}.yaml"
        prior_path.write_text(yaml.safe_dump(priors, sort_keys=False), encoding="utf-8")

        run_dir = runs_dir / f"{args.model}_{args.spec}_{name}"
        print(f"[{i}/{len(cells)}] delta sd={DELTA_SD[dlab]}  rho sd={RHO_SD[rlab]}", flush=True)
        run_model(
            args.model,
            data=data,
            data_spec=specs[args.spec],
            prior_specs=str(prior_path),
            prior_name=name,
            n_iter=n_iter, burn=burn, thin=thin, chains=chains,
            seed=int(defaults.get("seed", 12345)),
            n_transform=defaults.get("n_transform", "log100_centered10"),
            competition_measurement={"frequency": args.freq, "annual_timing": "q4"},
            run_dir=run_dir, run_id=name, save=True,
        )
        print(f"    done ({time.time() - t0:.0f}s)", flush=True)

    # ---- collect ----
    import arviz as az

    rows = []
    for dlab, rlab in cells:
        name = f"delta{dlab}_rho{rlab}"
        idata = az.from_netcdf(runs_dir / f"{args.model}_{args.spec}_{name}" / "posterior.nc")
        get = lambda v: np.asarray(idata.posterior[v], dtype=float).reshape(-1)
        d, r1 = get("delta"), get("rho_1")
        kt = np.asarray(idata.posterior["kappa_t"], dtype=float)
        kt = kt.reshape(-1, kt.shape[-1])
        rows.append({
            "estimation_revision": ESTIMATION_REVISION,
            "competition_measurement_frequency": args.freq,
            "n_iter": n_iter,
            "delta_prior_sd": DELTA_SD[dlab], "rho_prior_sd": RHO_SD[rlab],
            "delta_mean": d.mean(), "delta_lo": np.quantile(d, 0.025), "delta_hi": np.quantile(d, 0.975),
            "rho1_mean": r1.mean(),
            "kappa_start": kt[:, 0].mean(), "kappa_end": kt[:, -1].mean(),
            "delta_rhat": float(np.nanmax(np.asarray(az.rhat(idata.posterior["delta"])))),
            "delta_ess": float(np.nanmin(np.asarray(az.ess(idata.posterior["delta"], method="bulk")))),
        })
    table = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT / f"decomposition_{args.model}_{args.spec}.csv", index=False)
    print(f"\n{table.to_string(index=False)}")
    print(f"\nsaved -> {OUT / f'decomposition_{args.model}_{args.spec}.csv'}")
    write_decomposition_macros(model=args.model)


def _channel_shares(table: pd.DataFrame) -> dict[str, float] | None:
    """Split the baseline-to-tight shrinkage of delta into its two prior channels.

    A 2x2 factorial read off the (delta prior) x (AR(2) prior) grid:

        base      delta prior baseline, AR(2) prior baseline
        delta     delta prior tight,    AR(2) prior baseline
        rho       delta prior baseline, AR(2) prior tight
        both      both tight

    The delta-channel share is (base - delta)/(base - both), the AR(2)-channel
    share is (base - rho)/(base - both), and what is left over is the interaction.
    Shares can exceed one or go negative when the two channels offset, which is
    why the interaction is reported rather than hidden.
    """
    def cell(dsd: float, rsd: float) -> float | None:
        hit = table[(table.delta_prior_sd == dsd) & (table.rho_prior_sd == rsd)]
        return float(hit.delta_mean.iloc[0]) if len(hit) else None

    base = cell(DELTA_SD["baseline"], RHO_SD["baseline"])
    only_delta = cell(DELTA_SD["tight"], RHO_SD["baseline"])
    only_rho = cell(DELTA_SD["baseline"], RHO_SD["tight"])
    both = cell(DELTA_SD["tight"], RHO_SD["tight"])
    if None in (base, only_delta, only_rho, both):
        return None
    total = base - both
    if abs(total) < 1e-12:
        return None
    delta_share = (base - only_delta) / total
    rho_share = (base - only_rho) / total
    return {
        "delta": 100.0 * delta_share,
        "rho": 100.0 * rho_share,
        "interaction": 100.0 * (1.0 - delta_share - rho_share),
    }


def write_decomposition_macros(*, model: str = "hsa_steady") -> Path | None:
    """Emit the shares the report quotes, for whichever specs have been run.

    The report states these numbers in prose, so they must come from here rather
    than being typed in: see the macro note at the top of the .tex.
    """
    labels = {"unemployment_gap": "Headline", "unemployment_gap_core": "Core"}
    lines = ["% Generated by main_scripts/prior_decomposition_rho_delta.py; do not edit by hand."]
    found = False
    for spec, label in labels.items():
        path = OUT / f"decomposition_{model}_{spec}.csv"
        if not path.exists():
            continue
        table = pd.read_csv(path)
        required = {"estimation_revision", "competition_measurement_frequency", "n_iter"}
        missing = required.difference(table.columns)
        if missing:
            raise RuntimeError(
                f"{path} lacks provenance columns {sorted(missing)}; rerun the decomposition"
            )
        if set(table["estimation_revision"].dropna()) != {ESTIMATION_REVISION}:
            raise RuntimeError(f"{path} is stale for current revision {ESTIMATION_REVISION}")
        shares = _channel_shares(table)
        if shares is None:
            continue
        found = True
        for channel, name in (("delta", "DeltaPriorShare"), ("rho", "ArTwoPriorShare"), ("interaction", "InteractionShare")):
            lines.append(rf"\providecommand{{\PriorDecomp{label}{name}}}{{{shares[channel]:.0f}}}")
    if not found:
        return None
    target = RESULTS_DIR / "tables" / "shared" / "prior_decomposition_macros.tex"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"saved -> {target}")
    return target


if __name__ == "__main__":
    main()
