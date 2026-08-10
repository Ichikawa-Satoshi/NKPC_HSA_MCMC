"""Run the complete production F0/U/R1/R2/R3 pipeline and build its PDF.

The command intentionally has no ``--quick`` mode: quick/smoke runs can never
be promoted into the restriction report.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from _bootstrap import DATA_DIR, ROOT


def _run(script: str, args: list[str]) -> None:
    command = [sys.executable, str(ROOT / "scripts" / script), *args]
    print("\n=== " + " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def _require_clean_revision() -> None:
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if status:
        raise SystemExit(
            "Restriction production runs require a clean committed revision. "
            "Commit the model/report migration first; dirty-code runs are rejected by the report builder."
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs" / "models.yaml"))
    parser.add_argument("--priors", default=str(ROOT / "configs" / "priors_baseline.yaml"))
    parser.add_argument("--data", default=str(DATA_DIR / "processed" / "model_ready.csv"))
    parser.add_argument(
        "--competition-frequency",
        choices=["annual_q4", "quarterly_interpolated"],
        default="annual_q4",
    )
    parser.add_argument("--rebuild-data", action="store_true")
    args = parser.parse_args()

    _require_clean_revision()
    if args.rebuild_data:
        _run("01_build_data.py", [])
    _run(
        "10_estimate_theory_models.py",
        [
            "--config", args.config,
            "--priors", args.priors,
            "--data", args.data,
            "--competition-frequency", args.competition_frequency,
        ],
    )
    _run("11_run_theory_diagnostics.py", [])
    _run("build_restriction_report.py", [])


if __name__ == "__main__":
    main()
