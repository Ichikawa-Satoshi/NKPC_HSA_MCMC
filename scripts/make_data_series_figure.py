"""Regenerate Figure 1 (results/figures/report/data_series.png).

Panel (a) inflation: headline CPI, core CPI, and PPI (no crisis shading).
Panel (b) real-activity gaps: negative unemployment gap, BN and HP output gaps.
Panel (c) competition measure: the estimation transform of the firm count.

Sample matches the estimation window (1982Q1-2012Q4, T=124).
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import _bootstrap  # noqa: F401
from _bootstrap import ROOT
from nkpc_hsa.data import transform_competition_series
from nkpc_hsa.data.transforms import DEFAULT_N_TRANSFORM

OUT = ROOT / "results" / "figures" / "report" / "data_series.png"


def main():
    df = pd.read_csv(ROOT / "data" / "processed" / "model_ready.csv")
    df["DATE"] = pd.to_datetime(df["DATE"])
    # estimation sample: 1982Q1-2012Q4
    m = (df["DATE"].dt.year >= 1982) & (df["DATE"].dt.year <= 2012)
    d = df.loc[m].copy()
    yr = d["DATE"].dt.year + (d["DATE"].dt.quarter - 1) / 4.0
    N = transform_competition_series(d["N_Gustavo"].to_numpy(), transform=DEFAULT_N_TRANSFORM)

    fig, ax = plt.subplots(3, 1, figsize=(9, 10), sharex=True)

    # (a) inflation: headline, core, PPI
    ax[0].plot(yr, d["pi_cpi"], color="#CC3311", lw=1.4, label="Headline CPI inflation")
    ax[0].plot(yr, d["pi_cpi_core"], color="#33415C", lw=1.4, label="Core CPI inflation")
    ax[0].plot(yr, d["pi_ppi"], color="#0077BB", lw=1.2, alpha=0.9, label="PPI inflation")
    ax[0].axhline(0, color="k", lw=0.6)
    ax[0].set_ylabel("inflation (annualized, %)")
    ax[0].set_title("(a) Inflation: headline, core, and PPI")
    ax[0].legend(fontsize=8, ncol=3, frameon=False, loc="upper right")

    # (b) real-activity gaps
    ax[1].plot(yr, d["unemp_gap"], color="#4477AA", lw=1.3, label="Negative unemployment gap")
    ax[1].plot(yr, d["output_gap_BN"], color="#228833", lw=1.3, label="Output gap (BN)")
    ax[1].plot(yr, d["output_gap_HP"], color="#AA3377", lw=1.3, label="Output gap (HP)")
    ax[1].axhline(0, color="k", lw=0.6)
    ax[1].set_ylabel("gap")
    ax[1].set_title("(b) Real-activity gaps")
    ax[1].legend(fontsize=8, ncol=3, frameon=False, loc="lower right")

    # (c) competition measure (estimation transform)
    ax[2].plot(yr, N, color="#009988", lw=1.8, label="N (firm-count)")
    ax[2].axhline(0, color="k", lw=0.6)
    ax[2].set_ylabel("transformed N\n(ten-log-points, centered)")
    ax[2].set_title("(c) Competition measure (estimation transform)")
    ax[2].set_xlabel("year")
    ax[2].legend(fontsize=8, frameon=False, loc="upper right")

    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT.relative_to(ROOT)}  (T={len(d)})")


if __name__ == "__main__":
    main()
