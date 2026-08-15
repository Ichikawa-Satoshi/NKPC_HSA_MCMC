"""Run the measurement and inflation follow-up, then build the final production PDF."""

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
from experiments.design.followup import run_followup
from nkpc_hsa.progress import STYLES as PROGRESS_STYLES


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "nine_cell_design.yaml",
    )
    parser.add_argument("--baseline-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--test-run", action="store_true")
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument(
        "--reuse-existing",
        action="store_true",
        help="Rebuild tables, figures, and PDF from already saved follow-up draws.",
    )
    parser.add_argument(
        "--progress",
        choices=PROGRESS_STYLES,
        default="auto",
    )
    args = parser.parse_args()
    pdf = run_followup(
        config_path=args.config,
        baseline_dir=args.baseline_dir,
        output_dir=args.output_dir,
        test_run=args.test_run,
        progress=args.progress,
        compile_report=not args.no_compile,
        reuse_existing=args.reuse_existing,
    )
    if pdf is not None:
        print(f"wrote {pdf}", flush=True)


if __name__ == "__main__":
    main()
