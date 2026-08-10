"""Build the separate F0/U/R1/R2/R3 restriction-version report.

This builder never invokes the historical report pipeline and never scans
``results/runs``. It validates and renders only the signed theory-run namespace.
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from _bootstrap import RESULTS_DIR, ROOT
from nkpc_hsa.reporting.theory_report import build_theory_report_inputs
from nkpc_hsa.theory_models import THEORY_ESTIMATION_REVISION


REPORT = ROOT / "report" / "nkpc_hsa_restriction_report.tex"


def report_builder_revision() -> tuple[str, bool]:
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    return revision, dirty


def ensure_report_results_link() -> None:
    repo_results = ROOT / "results"
    target = RESULTS_DIR.resolve()
    if repo_results.is_symlink():
        if repo_results.resolve() != target:
            raise SystemExit(f"{repo_results} points to {repo_results.resolve()}, expected {target}")
        return
    if repo_results.exists():
        if repo_results.resolve() == target:
            return
        raise SystemExit(f"Cannot expose external results: {repo_results} already exists")
    repo_results.symlink_to(target, target_is_directory=True)


def compile_pdf() -> Path:
    for pass_number in (1, 2):
        result = subprocess.run(
            ["xelatex", "-interaction=nonstopmode", REPORT.name],
            cwd=REPORT.parent,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            tail = "\n".join(result.stdout.splitlines()[-30:])
            raise SystemExit(f"restriction report XeLaTeX pass {pass_number} failed:\n{tail}")
    log = REPORT.with_suffix(".log").read_text(encoding="utf-8", errors="replace")
    failures = [
        line for line in log.splitlines()
        if "Undefined control sequence" in line or "LaTeX Error" in line
    ]
    if failures:
        raise SystemExit("Restriction report contains LaTeX failures: " + "; ".join(failures[:8]))
    return REPORT.with_suffix(".pdf")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=RESULTS_DIR / "theory_runs" / THEORY_ESTIMATION_REVISION,
    )
    parser.add_argument(
        "--tables-dir",
        type=Path,
        default=RESULTS_DIR / "tables" / "theory",
    )
    parser.add_argument(
        "--allow-missing-runs",
        action="store_true",
        help="Build an explicit pending preview. Production builds must omit this flag.",
    )
    parser.add_argument("--no-compile", action="store_true")
    args = parser.parse_args()

    ensure_report_results_link()
    builder_revision, builder_dirty = report_builder_revision()
    build_theory_report_inputs(
        args.runs_dir,
        args.tables_dir,
        allow_missing=args.allow_missing_runs,
        report_builder_revision=builder_revision,
        report_builder_dirty=builder_dirty,
    )
    if args.no_compile:
        print(f"wrote theory report inputs under {args.tables_dir}")
    else:
        print(f"wrote {compile_pdf()}")


if __name__ == "__main__":
    main()
