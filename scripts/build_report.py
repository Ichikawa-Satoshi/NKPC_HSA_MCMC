"""Build every report input from the saved runs, in one command.

This exists because the report's artifacts are produced by five separate
scripts, and until now only one of them was wired into the estimation pipeline.
Re-estimating therefore refreshed most of the tables but silently left the
headline table, the fit comparison and the data figure at their previous
vintage -- no error, no warning, a PDF that compiles cleanly and disagrees with
its own run set.

The fix is to have exactly one place that knows the order. That place is here.
Do not add a report artifact without adding it to STEPS.

Order matters in one place: ``make_fit_comparison_table`` reads the CSV that
``predictive_comparison`` writes, so it must come after it.

    python scripts/build_report.py               # tables and figures only
    python scripts/build_report.py --compile     # ... then run xelatex twice
    python scripts/build_report.py --skip-predictive   # reuse the existing scores
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import _bootstrap  # noqa: F401
from _bootstrap import ROOT

REPORT = ROOT / "report" / "nkpc_hsa_report.tex"

# (script, extra args, what it produces, whether --skip-predictive skips it)
STEPS: list[tuple[str, list[str], str, bool]] = [
    (
        "12_build_cpi_ppi_report.py",
        ["--runs-dir", "{runs_dir}", "--min-iter", "{min_iter}"],
        "most tables, all result macros, most figures; itself chains "
        "make_spec_tables.py and 11_additional_report_evidence.py",
        False,
    ),
    (
        "make_headline_results_table.py",
        [],
        "headline_results.tex, model_comparison_unemp.tex, ppi_results.tex",
        False,
    ),
    (
        "predictive_comparison.py",
        [],
        "results/evidence/tables/predictive_comparison.csv (prequential LPD, WAIC, PSIS-LOO)",
        True,
    ),
    (
        "make_fit_comparison_table.py",
        [],
        "fit_comparison.tex, fit_comparison_macros.tex -- reads the CSV above",
        False,
    ),
    (
        "make_data_series_figure.py",
        [],
        "data_series.png (reads data/processed/model_ready.csv, not the runs)",
        False,
    ),
    (
        "prior_decomposition_rho_delta.py",
        ["--macros-only"],
        "prior_decomposition_macros.tex, rebuilt from the existing factorial CSVs. "
        "The 12 estimation cells behind them are NOT re-run here.",
        False,
    ),
    (
        "make_conditional_ml_table.py",
        [],
        "conditional_ml.tex and its macros, from the Chib run's CSV. Like the prior "
        "decomposition, the Chib run itself is estimation and is NOT re-run here.",
        False,
    ),
]

# Not a report-build step: it estimates 12 extra diagnostic cells of its own and
# takes about 45 minutes. Run it directly when the prior sweep is re-done.
NOT_BUILT_HERE = "prior_decomposition_rho_delta.py"


def _run(script: str, args: list[str]) -> float:
    started = time.perf_counter()
    print(f"\n=== {script} {' '.join(args)}", flush=True)
    subprocess.run([sys.executable, str(ROOT / "scripts" / script), *args], cwd=ROOT, check=True)
    return time.perf_counter() - started


def compile_pdf() -> None:
    """Two xelatex passes, so the table of contents and \\ref numbering settle."""
    if not REPORT.exists():
        raise SystemExit(f"missing {REPORT}")
    for pass_number in (1, 2):
        print(f"\n=== xelatex (pass {pass_number})", flush=True)
        result = subprocess.run(
            ["xelatex", "-interaction=nonstopmode", REPORT.name],
            cwd=REPORT.parent,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0 and pass_number == 2:
            tail = "\n".join(result.stdout.strip().splitlines()[-25:])
            raise SystemExit(f"xelatex failed:\n{tail}")
    log = REPORT.with_suffix(".log").read_text(encoding="utf-8", errors="replace")
    undefined = [line for line in log.splitlines() if "Undefined control sequence" in line]
    if undefined:
        raise SystemExit(
            f"{len(undefined)} undefined control sequence(s) in {REPORT.with_suffix('.log')}. "
            "A macro the .tex uses was not generated -- check the STEPS above."
        )
    print(f"\nwrote {REPORT.with_suffix('.pdf').relative_to(ROOT)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs-dir", type=Path, default=ROOT / "results" / "runs")
    parser.add_argument("--min-iter", type=int, default=12000)
    parser.add_argument("--compile", action="store_true", help="Run xelatex twice after building.")
    parser.add_argument(
        "--skip-predictive",
        action="store_true",
        help="Reuse the existing predictive_comparison.csv instead of recomputing the scores.",
    )
    args = parser.parse_args()

    timings = []
    for script, extra, _, skippable in STEPS:
        if skippable and args.skip_predictive:
            print(f"\n=== {script} SKIPPED (--skip-predictive)", flush=True)
            continue
        formatted = [a.format(runs_dir=args.runs_dir, min_iter=args.min_iter) for a in extra]
        timings.append((script, _run(script, formatted)))

    print("\n" + "=" * 60)
    for script, elapsed in timings:
        print(f"  {script:38s} {elapsed:7.1f}s")
    if args.skip_predictive:
        print("  (predictive scores reused; they are NOT guaranteed to match the current runs)")
    print(f"  ({NOT_BUILT_HERE} is estimation, not a build step -- run it separately)")

    if args.compile:
        compile_pdf()
    else:
        print("\nTables and figures built. Add --compile to produce the PDF.")


if __name__ == "__main__":
    main()
