from __future__ import annotations

import argparse

from _bootstrap import RESULTS_DIR
from nkpc_hsa.reporting.theory_report import build_theory_report_inputs
from nkpc_hsa.theory_models import THEORY_ESTIMATION_REVISION


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--runs-dir",
        default=str(RESULTS_DIR / "theory_runs" / THEORY_ESTIMATION_REVISION),
    )
    parser.add_argument("--out-dir", default=str(RESULTS_DIR / "tables" / "theory"))
    parser.add_argument("--allow-missing", action="store_true")
    args = parser.parse_args()
    build_theory_report_inputs(args.runs_dir, args.out_dir, allow_missing=args.allow_missing)


if __name__ == "__main__":
    main()
