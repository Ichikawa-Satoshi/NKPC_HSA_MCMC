from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from _bootstrap import ROOT
from nkpc_hsa.report.figures import save_posterior_predictive_placeholder
from nkpc_hsa.report.latex import compile_report, write_default_report
from nkpc_hsa.report.tables import write_latex_fragment


def _ensure_fragment(path: Path, text: str) -> None:
    if not path.exists() or "Run scripts/" in path.read_text(encoding="utf-8"):
        write_latex_fragment(pd.DataFrame({"note": [text]}), path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tex", default=str(ROOT / "reports" / "main.tex"))
    parser.add_argument("--skip-compile", action="store_true")
    args = parser.parse_args()
    write_default_report(args.tex)
    _ensure_fragment(ROOT / "results" / "tables" / "posterior_summary.tex", "Run scripts/06_make_tables_figures.py first.")
    _ensure_fragment(ROOT / "results" / "tables" / "mcmc_diagnostics.tex", "Run scripts/03_run_diagnostics.py first.")
    _ensure_fragment(ROOT / "results" / "tables" / "prior_sensitivity.tex", "Run scripts/04_prior_sensitivity.py first.")
    _ensure_fragment(ROOT / "results" / "model_comparison" / "model_comparison.tex", "Run scripts/05_model_comparison.py first.")
    save_posterior_predictive_placeholder(ROOT / "results" / "figures" / "posterior_predictive_placeholder.png")
    if args.skip_compile:
        print(f"Wrote {args.tex}; skipped PDF compilation.")
        return
    pdf = compile_report(args.tex, ROOT / "results" / "report")
    print(f"Saved {pdf}")


if __name__ == "__main__":
    main()
