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
from nkpc_hsa.progress import STYLES as PROGRESS_STYLES
from nkpc_hsa.progress import ProgressBoard, parse_event, resolve_style
from nkpc_hsa.theory_models import THEORY_MODELS

ESTIMATION_SCRIPT = "10_estimate_theory_models.py"


def _run(script: str, args: list[str]) -> None:
    command = [sys.executable, str(ROOT / "main_scripts" / script), *args]
    print("\n=== " + " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def _run_estimation(
    args: list[str],
    *,
    board: ProgressBoard | None,
    key: str,
) -> None:
    """Run one estimation cell, folding its progress events into the board.

    The child's stdout is read line by line: progress events update the board
    and everything else is printed above it, so ordinary output and the live
    display do not overwrite each other.
    """
    command = [sys.executable, str(ROOT / "main_scripts" / ESTIMATION_SCRIPT), *args]
    header = "=== " + " ".join(command)
    if board is None:
        print("\n" + header, flush=True)
        subprocess.run(command, cwd=ROOT, check=True)
        return

    board.write_line(header)
    board.start_cell(key)
    process = subprocess.Popen(
        command,
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=None,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None
    try:
        for line in process.stdout:
            event = parse_event(line.rstrip("\n"))
            if event is None:
                if line.strip():
                    board.write_line(line.rstrip("\n"))
                continue
            board.apply_event(event)
    finally:
        process.stdout.close()
        returncode = process.wait()
    board.finish_cell(key, ok=returncode == 0, note="" if returncode == 0 else f"exit {returncode}")
    if returncode != 0:
        raise subprocess.CalledProcessError(returncode, command)


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
    parser.add_argument(
        "--progress",
        choices=list(PROGRESS_STYLES),
        default=None,
        help=(
            "Live progress display. 'auto' (default) shows a per-cell board with an overall "
            "bar on a terminal and prints nothing extra when the output is redirected; "
            "'plain' prints one periodic summary line, which is what a log file wants."
        ),
    )
    args = parser.parse_args()
    if args.workers <= 0:
        raise SystemExit("--workers must be positive")

    _require_clean_revision()
    if args.rebuild_data:
        _run("01_build_data.py", [])
    config = load_model_config(args.config)
    models = list(config.get("theory_models", THEORY_MODELS))
    style = resolve_style(args.progress)
    # The children only emit events when this parent is going to render them.
    child_style = "stream" if style != "off" else "off"
    tasks: list[tuple[str, list[str]]] = []
    for spec_name, spec in configured_theory_data_specs(config).items():
        allowed = set(spec.get("models", models) or models)
        for model in models:
            if model not in allowed:
                continue
            tasks.append(
                (
                    f"{model}:{spec_name}",
                    [
                        "--config", args.config,
                        "--priors", args.priors,
                        "--data", args.data,
                        "--competition-frequency", args.competition_frequency,
                        "--data-spec", spec_name,
                        "--model", model,
                        # The child reports machine-readable events; this parent owns the display.
                        "--progress", child_style,
                    ],
                )
            )
    board = (
        ProgressBoard([key for key, _ in tasks], style=style, title="theory estimation")
        if style != "off"
        else None
    )
    if board is not None:
        board.render(force=True)
    try:
        with ThreadPoolExecutor(max_workers=min(args.workers, len(tasks))) as executor:
            futures = [
                executor.submit(_run_estimation, task, board=board, key=key) for key, task in tasks
            ]
            for future in futures:
                future.result()
    finally:
        if board is not None:
            board.close()
    _run("11_run_theory_diagnostics.py", [])
    _run("build_restriction_report.py", [])


if __name__ == "__main__":
    main()
