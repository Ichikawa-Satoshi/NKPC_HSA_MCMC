"""Backfill Laplace--Metropolis marginal likelihoods into an existing results tree.

Scans results_dir/caseX/<spec>/model*.npz produced by report_estimate.py, computes
log m(y) for each fit, and writes the log_ml column into model_comparison.csv.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import sys as _sys, pathlib as _pathlib  # noqa: E402
_ROOT = next(_p for _p in _pathlib.Path(__file__).resolve().parents if (_p / "pyproject.toml").exists())
_sys.path[:0] = [str(_ROOT), str(_ROOT / "src")]

from nkpc_hsa.paths import results_root  # noqa: E402
from nkpc_hsa.report_models import build_priors, load_case  # noqa: E402
from nkpc_hsa.report_models.cases import _load_frame  # noqa: E402
from nkpc_hsa.report_models.marginal_likelihood import laplace_metropolis_logml  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-dir", type=Path, default=results_root() / "report_estimation" / "pilot")
    ap.add_argument("--seed", type=int, default=20260819)
    args = ap.parse_args()

    frame = _load_frame()
    cmp_path = args.results_dir / "model_comparison.csv"
    cmp = pd.read_csv(cmp_path)
    log_ml = {}
    for case_dir in sorted(args.results_dir.glob("case*")):
        case = int(case_dir.name.removeprefix("case"))
        for spec_dir in sorted(p for p in case_dir.iterdir() if p.is_dir()):
            infl, forc, variant = spec_dir.name.split("__")
            data = load_case(case, infl, forc, variant, frame=frame)
            priors = build_priors(data)
            for npz in sorted(spec_dir.glob("model*.npz")):
                model = int(npz.stem.removeprefix("model"))
                z = np.load(npz, allow_pickle=True)
                draws = {k: z[k] for k in z.files}
                ml = laplace_metropolis_logml(draws, data, model, priors=priors, seed=args.seed)
                log_ml[(case, infl, forc, variant, model)] = ml
                print(f"case{case} {infl}/{forc}/{variant} M{model} log_ml={ml:.2f}", flush=True)

    cmp["log_ml"] = [
        log_ml.get((int(r.case), r.inflation, r.forcing, r.variant, int(r.model)), float("nan"))
        for r in cmp.itertuples()
    ]
    cmp.to_csv(cmp_path, index=False)
    print(f"\nupdated {cmp_path} with log_ml for {len(log_ml)} fits", flush=True)


if __name__ == "__main__":
    main()
