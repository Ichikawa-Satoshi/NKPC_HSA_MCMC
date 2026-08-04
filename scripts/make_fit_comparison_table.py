"""Build the fit-comparison table that shows the plug-in score next to proper scores.

The report has always described an in-sample plug-in fit score but never tabulated it,
and it listed a proper predictive comparison as future work even though
``scripts/predictive_comparison.py`` computes one. This table reports both, side by side,
for the main mixed-frequency design, so the reader can see how far the crude criterion is
from the proper ones rather than taking either on faith.

Reads results/appendix_particle_gibbs/tables/predictive_comparison.csv (written by
scripts/predictive_comparison.py) and writes the LaTeX fragment the report inputs.

    python scripts/predictive_comparison.py      # first
    python scripts/make_fit_comparison_table.py  # then
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

import _bootstrap  # noqa: F401
from _bootstrap import ROOT

SRC = ROOT / "results" / "appendix_particle_gibbs" / "tables" / "predictive_comparison.csv"
OUT = ROOT / "results" / "tables" / "cpi_ppi_report" / "annual_q4"
LABELS = {"ces": "CES", "hsa_steady": "HSA steady", "hsa_dynamic": "HSA dynamic",
          "hsa_full": "HSA full"}
ORDER = ["ces", "hsa_steady", "hsa_dynamic", "hsa_full"]
PRICES = ["Core CPI", "Headline CPI", "PPI"]


def main() -> None:
    if not SRC.exists():
        raise SystemExit(f"missing {SRC}; run scripts/predictive_comparison.py first")
    table = pd.read_csv(SRC)
    main_design = table[table["design"] == "annual_q4"]
    if main_design.empty:
        raise SystemExit("predictive_comparison.csv has no annual_q4 rows")

    lines = [
        r"\begin{table}[H]", r"\centering",
        (r"\caption{Fit comparison for the negative unemployment gap, mixed-frequency "
         r"design. All four columns are log scores, so higher is better, and all are "
         r"differences against nothing---only comparisons within a price index are "
         r"meaningful. \emph{Plug-in} is the in-sample score of the text: the inflation "
         r"equation evaluated at posterior-mean parameters and states, scored by a "
         r"Gaussian log-likelihood at the posterior-mean shock variance. "
         r"\emph{LPD}$_1$ is the prequential one-step-ahead log predictive density, "
         r"$\sum_t \log p(\pi_t\mid\pi_{1:t-1},x,N)$, integrated over the posterior. "
         r"WAIC and PSIS-LOO are computed on the pointwise inflation log-likelihood. "
         r"$\hat k$ is the maximum Pareto shape for the LOO importance weights; "
         r"$\hat k>0.7$ marks a cell where LOO is unreliable.}"),
        r"\label{tab:fit-comparison}", r"\small",
        r"\begin{tabular}{llccccc}", r"\toprule",
        (r"Price & Model & Plug-in & LPD$_1$ & WAIC & PSIS-LOO & $\hat k$ \\"),
        r"\midrule",
    ]
    for price in PRICES:
        sub = main_design[main_design["price"] == price].set_index("model")
        for i, model in enumerate(ORDER):
            if model not in sub.index:
                continue
            r = sub.loc[model]
            flag = r"\textsuperscript{$\ast$}" if float(r["max_pareto_k"]) > 0.7 else ""
            cells = [
                price if i == 0 else "",
                LABELS[model],
                f"{r['plugin_score']:.2f}",
                f"{r['LPD_1step']:.2f}",
                f"{r['WAIC']:.2f}",
                f"{r['LOO']:.2f}" + flag,
                f"{r['max_pareto_k']:.2f}",
            ]
            if model == "hsa_steady":
                cells = [r"\textbf{" + c + "}" if c else c for c in cells]
            lines.append(" & ".join(cells) + r" \\")
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    lines += [r"\end{tabular}", r"\end{table}", ""]

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "fit_comparison.tex").write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {(OUT / 'fit_comparison.tex').relative_to(ROOT)}")

    # The number the report quotes: how much the plug-in score overstates the gain.
    for price in PRICES:
        sub = main_design[main_design["price"] == price].set_index("model")
        if "ces" not in sub.index or "hsa_steady" not in sub.index:
            continue
        c, h = sub.loc["ces"], sub.loc["hsa_steady"]
        print(f"  {price:13s} HSA steady - CES:  plug-in {h['plugin_score']-c['plugin_score']:+6.2f}"
              f"   LPD1 {h['LPD_1step']-c['LPD_1step']:+6.2f}"
              f"   WAIC {h['WAIC']-c['WAIC']:+6.2f}   LOO {h['LOO']-c['LOO']:+6.2f}")


if __name__ == "__main__":
    main()
