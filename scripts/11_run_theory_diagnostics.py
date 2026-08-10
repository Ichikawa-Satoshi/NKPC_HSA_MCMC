"""Diagnose theory runs and finalize convergence/admissibility metadata."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import arviz as az

from _bootstrap import RESULTS_DIR
from nkpc_hsa.inference.diagnostics import (
    check_estimation_success,
    local_admissibility_diagnostics,
    save_diagnostics,
)
from nkpc_hsa.provenance import stamp_artifact_metadata
from nkpc_hsa.theory_models import THEORY_ESTIMATION_REVISION


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--runs-dir",
        default=str(RESULTS_DIR / "theory_runs" / THEORY_ESTIMATION_REVISION),
    )
    parser.add_argument(
        "--out-dir",
        default=str(RESULTS_DIR / "theory_diagnostics" / THEORY_ESTIMATION_REVISION),
    )
    args = parser.parse_args()
    runs_dir, out_dir = Path(args.runs_dir), Path(args.out_dir)
    for posterior in sorted(runs_dir.glob("*/posterior.nc")):
        run_dir = posterior.parent
        metadata_path = run_dir / "metadata.json"
        if not metadata_path.exists():
            raise RuntimeError(f"Theory run lacks metadata: {run_dir}")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("estimation_revision") != THEORY_ESTIMATION_REVISION:
            raise RuntimeError(f"STALE / HISTORICAL theory run: {run_dir}")
        idata = az.from_netcdf(posterior)
        target = out_dir / run_dir.name
        save_diagnostics(idata, target)
        warnings = check_estimation_success(idata)
        admissibility = local_admissibility_diagnostics(idata)
        rejection_share = float(
            (metadata.get("admissibility_sampling", {}) or {}).get("state_path_rejection_share", 0.0)
        )
        admissibility["state_path_rejection_share"] = rejection_share
        if rejection_share >= 0.05:
            admissibility["local_approximation_range_warning"] = True
            admissibility["interpretation"] = (
                "State-path positivity proposals fail frequently; the local linear approximation may cover too wide an observed N range."
            )
        status = "converged" if warnings.empty else "failed"
        diagnostics = {
            "estimation_revision": THEORY_ESTIMATION_REVISION,
            "model": metadata.get("model"),
            "run_id": metadata.get("run_id"),
            "convergence_status": status,
            "warning_count": int(len(warnings)),
            "local_admissibility": admissibility,
        }
        target.mkdir(parents=True, exist_ok=True)
        (target / "diagnostics.json").write_text(
            json.dumps(diagnostics, indent=2, sort_keys=True), encoding="utf-8"
        )
        metadata["convergence_status"] = status
        metadata["local_admissibility_diagnostics"] = admissibility
        metadata = stamp_artifact_metadata(metadata)
        metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
