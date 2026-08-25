"""Rebuild the Capital IQ economy-wide effective-firm-count N at 4-digit SIC.

The default model input (``N_capitaliq_firmw``) uses coarse SIC divisions
(~10 markets), which makes the effective firm count N=1/HHI track Capital IQ
*coverage* (corr ~0.7 with the raw reporting-firm count): the 1993-96 surge is
the EDGAR / Capital IQ historical-coverage transition, not real competition.

A 4-digit SIC market definition (~400 codes, ~60 surviving the >=10-firm
filter) decouples N from coverage (corr ~ -0.2) and yields a stable
"effective competitors within a narrow industry" measure (N ~ 4-6).

This script replicates parts (B)+(C) of ``build_data/load_capital_iq.do`` with
``$MARKET = sic`` and writes ``capital_iq_N_sic4_quarterly.csv`` next to the
other processed Capital IQ files.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import sys as _sys, pathlib as _pathlib  # noqa: E402
_ROOT = next(_p for _p in _pathlib.Path(__file__).resolve().parents if (_p / "pyproject.toml").exists())
_sys.path[:0] = [str(_ROOT), str(_ROOT / "src")]
from nkpc_hsa.paths import data_root  # noqa: E402

MIN_FIRMS = 10


def build_sic4_N(panel: pd.DataFrame) -> pd.DataFrame:
    """Firm- and revenue-weighted N=1/HHI at 4-digit SIC, industry-SA'd."""
    d = panel[panel["total_revenue_usd"] > 0].copy()
    d["sic"] = d["sic"].astype("Int64").astype(str)
    d = d[d["sic"] != "<NA>"]
    d["q"] = pd.PeriodIndex(d["tq"].astype(str).str.upper(), freq="Q")
    d["qtr"] = d["q"].dt.quarter

    mrev = d.groupby(["q", "sic"])["total_revenue_usd"].sum().rename("mrev")
    d = d.join(mrev, on=["q", "sic"])
    d["sh2"] = (d["total_revenue_usd"] / d["mrev"]) ** 2
    h = (d.groupby(["q", "sic"])
         .agg(hhi=("sh2", "sum"), n_firms=("entity_id", "count"),
              rev=("total_revenue_usd", "sum"), fy=("fiscal_year", "first"),
              qtr=("qtr", "first")).reset_index())
    h = h[h["n_firms"] >= MIN_FIRMS].copy()

    # Multiplicative seasonal adjustment per industry (log two-way fit).
    h["lh"] = np.log(h["hhi"])
    h["ym"] = h.groupby(["sic", "fy"])["lh"].transform("mean")
    h["dev"] = h["lh"] - h["ym"]
    qe = h.groupby(["sic", "qtr"])["dev"].mean().rename("qe").reset_index()
    qm = qe.groupby("sic")["qe"].mean().rename("qm").reset_index()
    qe = qe.merge(qm, on="sic")
    qe["seas"] = qe["qe"] - qe["qm"]
    h = h.merge(qe[["sic", "qtr", "seas"]], on=["sic", "qtr"])
    h["hhi_sa"] = np.exp(h["lh"] - h["seas"])

    tot = h.groupby("q").agg(tf=("n_firms", "sum"), tr=("rev", "sum"))
    h = h.join(tot, on="q")
    h["fw"] = h["hhi_sa"] * h["n_firms"] / h["tf"]
    h["rw"] = h["hhi_sa"] * h["rev"] / h["tr"]
    agg = h.groupby("q").agg(mfw=("fw", "sum"), mrw=("rw", "sum"),
                             n_markets=("sic", "nunique"), covered_firms=("n_firms", "sum"))
    agg["N_capitaliq_sic4_firmw"] = 1.0 / agg["mfw"]
    agg["N_capitaliq_sic4_revw"] = 1.0 / agg["mrw"]
    return agg.sort_index()


def main() -> None:
    processed = data_root() / "processed"
    panel = pd.read_csv(processed / "capital_iq_company_quarter_revenues.csv",
                        usecols=["entity_id", "tq", "fiscal_year", "sic", "total_revenue_usd"])
    agg = build_sic4_N(panel)
    out = agg.reset_index()
    out["tq"] = out["q"].astype(str)
    out = out[["tq", "N_capitaliq_sic4_firmw", "N_capitaliq_sic4_revw", "n_markets", "covered_firms"]]
    path = processed / "capital_iq_N_sic4_quarterly.csv"
    out.to_csv(path, index=False)
    n = agg["N_capitaliq_sic4_firmw"]
    print(f"wrote {path}")
    print(f"  quarters={len(out)}  N_firmw range {n.min():.2f}..{n.max():.2f}  "
          f"markets(last)={int(agg['n_markets'].iloc[-1])}  covered_firms(last)={int(agg['covered_firms'].iloc[-1])}")


if __name__ == "__main__":
    main()
