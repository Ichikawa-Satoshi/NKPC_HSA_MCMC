"""Run the complete production F0/U/R1/R2/R3 pipeline and build its PDF.

The command intentionally has no ``--quick`` mode: quick/smoke runs can never
be promoted into the restriction report.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from _bootstrap import DATA_DIR, ROOT
from nkpc_hsa.config import configured_theory_data_specs, load_model_config
from nkpc_hsa.theory_models import THEORY_MODELS


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
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Independent model/spec subprocesses to run concurrently.",
    )
    args = parser.parse_args()
    if args.workers <= 0:
        raise SystemExit("--workers must be positive")

    _require_clean_revision()
    if args.rebuild_data:
        _run("01_build_data.py", [])
    config = load_model_config(args.config)
    models = list(config.get("theory_models", THEORY_MODELS))
    tasks: list[list[str]] = []
    for spec_name, spec in configured_theory_data_specs(config).items():
        allowed = set(spec.get("models", models) or models)
        for model in models:
            if model not in allowed:
                continue
            tasks.append(
                [
                    "--config", args.config,
                    "--priors", args.priors,
                    "--data", args.data,
                    "--competition-frequency", args.competition_frequency,
                    "--data-spec", spec_name,
                    "--model", model,
                ]
            )
    with ThreadPoolExecutor(max_workers=min(args.workers, len(tasks))) as executor:
        futures = [executor.submit(_run, "10_estimate_theory_models.py", task) for task in tasks]
        for future in futures:
            future.result()
    _run("11_run_theory_diagnostics.py", [])
    _run("build_restriction_report.py", [])


if __name__ == "__main__":
    main()
