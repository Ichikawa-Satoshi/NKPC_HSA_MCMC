"""Run the executable nine-cell design and build its report.

Examples
--------
Software-validation run (short chains, explicitly non-inferential):

    python experiments/design/run.py --test-run --compile

Long-chain core run (still blocked from a complete design claim until every
mandatory robustness and evidence module in the manifest is implemented):

    python experiments/design/run.py --compile
"""
from __future__ import annotations

import argparse
from pathlib import Path

import sys as _sys, pathlib as _pathlib  # noqa: E402  (bootstrap: importable at any depth)
_ROOT = next(_p for _p in _pathlib.Path(__file__).resolve().parents if (_p / "pyproject.toml").exists())
_sys.path[:0] = [str(_ROOT), str(_ROOT / "src"), str(_ROOT / "tests")]
from experiments import _bootstrap  # noqa: F401,E402
from experiments._bootstrap import ROOT


BUNDLE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BUNDLE_DIR / "results"
from nkpc_hsa.phillips.estimation import run_nine_cell_design
from experiments.design.reporting import build_design_report_artifacts, compile_design_report
from nkpc_hsa.progress import STYLES as PROGRESS_STYLES


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "nine_cell_design.yaml")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--test-run", action="store_true", help="Use short chains and label every output non-inferential.")
    parser.add_argument("--no-robustness", action="store_true", help="Skip fast-by-fast and CS1/CS2 smoke sensitivities.")
    parser.add_argument("--compile", action="store_true", help="Compile the polished PDF after building tables and figures.")
    parser.add_argument(
        "--progress",
        choices=PROGRESS_STYLES,
        default="auto",
        help="Progress display: auto shows bars in a terminal; plain is suitable for logs.",
    )
    args = parser.parse_args()

    label = "TEST RUN -- NOT FOR INFERENCE" if args.test_run else "LONG CORE RUN -- DESIGN INCOMPLETE"
    print(f"[{label}] loading the frozen nine-cell data and estimating the measurement cut", flush=True)
    run = run_nine_cell_design(
        config_path=args.config,
        output_dir=args.output_dir,
        test_run=args.test_run,
        include_robustness=not args.no_robustness,
        progress=args.progress,
    )
    print(f"measurement R_q={run.measurement.information_ratio:.3f}; building report artifacts", flush=True)
    build_design_report_artifacts(run)
    print(f"wrote {run.output_dir}", flush=True)
    if args.compile:
        pdf = compile_design_report(run)
        print(f"wrote {pdf}", flush=True)


if __name__ == "__main__":
    main()
