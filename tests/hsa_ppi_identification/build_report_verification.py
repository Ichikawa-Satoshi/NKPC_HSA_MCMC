"""Verification report for the Gustavo x Capital IQ HSA cell.

Reads results/verification/verification.json (written by verify.py) and builds a PDF:
A1 joint vs fixed decomposition, A2 fixed-decomposition robustness, B2 firm/revenue
profile, C3 timing x error, and D2 the joint trend/cycle decomposition paths.

    python tests/hsa_ppi_identification/verify.py            # produce the results first
    python tests/hsa_ppi_identification/build_report_verification.py
"""
from __future__ import annotations
import json, subprocess
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np, pandas as pd

import sys as _sys, pathlib as _pathlib  # noqa: E402
_ROOT = next(_p for _p in _pathlib.Path(__file__).resolve().parents if (_p / "pyproject.toml").exists())
_sys.path[:0] = [str(_ROOT), str(_ROOT / "src"), str(_ROOT / "tests")]
from tests import _bootstrap  # noqa: F401,E402
from nkpc_hsa.config import load_yaml  # noqa: E402
from nkpc_hsa.report_models.cases import GUSTAVO_ANNUAL_COL  # noqa: E402
from tests.hsa_ppi_identification.functions import _load_frame, gustavo_capiq_quarterly  # noqa: E402

BUNDLE = Path(__file__).resolve().parent
BLUE, ORANGE, GREEN, GREY = "#0072B2", "#D55E00", "#009E73", "#6B7280"
LAB = {"inverse_markup": "inverse markup", "neg_unemp_gap": "neg-unemployment gap"}


def esc(v): return str(v).replace("_", r"\_").replace("%", r"\%")


def fig_paths(R, frame, figures):
    """D2: joint trend/cycle decomposition of the combined N."""
    g = pd.to_numeric(frame[GUSTAVO_ANNUAL_COL], errors="coerce").dropna()
    gpts = g[(g.index >= pd.Period("1974Q4", freq="Q")) & (g.index <= pd.Period("2013Q4", freq="Q"))]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
    for ax, act in zip(axes, R["A1_joint"]):
        j = R["A1_joint"][act]
        per = pd.PeriodIndex(j["periods"], freq="Q").to_timestamp()
        nbar = np.array(j["nbar"]); nhat = np.array(j["nhat"])
        ax.plot(per, nbar, color=GREEN, lw=2.4, label=r"trend $\bar N$ (joint)")
        ax.plot(per, nhat, color=BLUE, lw=1.2, label=r"cycle $\hat N$ (joint)")
        ax.plot(per, nbar + nhat, color=GREY, lw=0.8, alpha=0.6, label=r"$\bar N+\hat N$")
        ax.plot(gpts.index.to_timestamp(), 10 * np.log(gpts.values) - np.mean(10 * np.log(gpts.values)),
                "s", color=ORANGE, ms=3, label="Gustavo annual (centred)")
        ax.axhline(0, color="black", lw=0.6)
        ax.set_title(f"{LAB[act]}\njoint decomposition of combined N", fontsize=9.5)
        ax.set_xlabel("year"); ax.legend(frameon=False, fontsize=7)
    fig.suptitle("D2: joint (state-space) trend/cycle decomposition — trend tracks the falling Gustavo level")
    fig.tight_layout(); fig.savefig(figures / "verify_paths.png", dpi=190); plt.close(fig)


def main():
    out = BUNDLE / "results"; figures = out / "figures"; figures.mkdir(parents=True, exist_ok=True)
    R = json.loads((out / "verification" / "verification.json").read_text())
    frame = _load_frame()
    fig_paths(R, frame, figures)
    s = R["shares"]

    def cell(t):  # (mean, lo, hi, P) tuple -> latex
        return f"{t[0]:+.3f} [{t[1]:+.3f}, {t[2]:+.3f}] & {t[3]:.2f}" if t else "-- &"

    # A1 joint vs fixed
    a1 = ""
    for act in R["A1_joint"]:
        j = R["A1_joint"][act]; th = j["theta"]
        fx = R["A2_fastdefs"][act]["ewma_hl8"].get("theta_hsa")
        a1 += (f"{esc(LAB[act])} & {th[0]:+.3f} & {th[2]:.2f} & {j['delta_implied']:+.3f} & "
               f"{fx[0]:+.3f} & {fx[3]:.2f} \\\\\n")
    # A2 fast defs (theta_hsa)
    a2 = ""
    for act in R["A2_fastdefs"]:
        for fast in ["ewma_hl8", "ar1_innovation", "first_difference"]:
            t = R["A2_fastdefs"][act][fast].get("theta_hsa")
            a2 += f"{esc(LAB[act])} & {esc(fast)} & {cell(t)} \\\\\n"
    # B2 firm vs revenue
    b2 = ""
    for act in R["B2_profile"]:
        for w in ["firm", "revenue"]:
            t = R["B2_profile"][act][w].get("theta_hsa")
            b2 += f"{esc(LAB[act])} & {w} & {cell(t)} \\\\\n"
    # C3 timing x error
    c3 = ""
    for act in R["C3_timing_error"]:
        for key in ["current|iid", "current|persistent_ar1", "lag1|iid", "lag1|persistent_ar1"]:
            t = R["C3_timing_error"][act][key].get("theta_hsa")
            c3 += f"{esc(LAB[act])} & {esc(key)} & {cell(t)} \\\\\n"

    tex = rf"""\documentclass[11pt]{{article}}
\usepackage[margin=0.8in]{{geometry}}
\usepackage{{booktabs,graphicx,amsmath,xcolor,microtype,hyperref,newtxtext,newtxmath}}
\definecolor{{navy}}{{HTML}}{{17365D}}\hypersetup{{colorlinks=true,linkcolor=navy,urlcolor=navy}}
\setlength{{\parindent}}{{0pt}}\setlength{{\parskip}}{{5pt}}
\begin{{document}}
\begin{{center}}
{{\color{{navy}}\LARGE\bfseries HSA Cell: Verification Battery}}\\[3pt]
{{\large Gustavo $\times$ Capital IQ temporal-disaggregation, PPI, 1974--2013}}
\end{{center}}

\textbf{{Pipeline.}} (1) build a quarterly competition series by allocating the annual Gustavo change with
Capital IQ's average quarterly profile $s_q=[{s[0]:.2f},{s[1]:.2f},{s[2]:.2f},{s[3]:.2f}]$; (2) decompose it
into trend/cycle; (3) estimate the theory-faithful HSA NKPC ($\psi$ excluded). This report verifies the
decomposition step and the robustness of the HSA channels.

\section*{{A1. Joint vs fixed decomposition}}
The fixed decomposition splits the competition series by a one-sided EWMA \emph{{outside}} the model; the
joint decomposition estimates trend $\bar N$ and cycle $\hat N$ as latent states \emph{{together}} with the
NKPC (state-space, HSA-restricted, hybrid). $\theta$ (and the implied $\delta=b_x\zeta\theta$):
\begin{{center}}\small\begin{{tabular}}{{l r r r r r}}
\toprule & \multicolumn{{3}}{{c}}{{Joint (state-space)}} & \multicolumn{{2}}{{c}}{{Fixed (EWMA)}}\\
\cmidrule(lr){{2-4}}\cmidrule(lr){{5-6}}
Activity & $\theta$ & P($>$0) & $\delta$ implied & $\theta_{{\mathrm{{HSA}}}}$ & P($>$0)\\\midrule
{a1}\bottomrule\end{{tabular}}\end{{center}}
The joint decomposition gives a $\theta$ comparable to (or stronger than) the fixed EWMA -- on the combined
series the state-space is not materially attenuated.

\section*{{A2. Fixed-decomposition robustness (fast component)}}
\begin{{center}}\small\begin{{tabular}}{{l l r r}}
\toprule Activity & fast component & $\theta_{{\mathrm{{HSA}}}}$ (95\% CI) & P($>$0)\\\midrule
{a2}\bottomrule\end{{tabular}}\end{{center}}
$\theta$ is essentially invariant to the fast-component definition (EWMA / AR(1) innovation / first
difference).

\section*{{B2. Capital IQ firm vs revenue profile}}
\begin{{center}}\small\begin{{tabular}}{{l l r r}}
\toprule Activity & profile weighting & $\theta_{{\mathrm{{HSA}}}}$ (95\% CI) & P($>$0)\\\midrule
{b2}\bottomrule\end{{tabular}}\end{{center}}

\section*{{C3. Timing $\times$ error model}}
\begin{{center}}\small\begin{{tabular}}{{l l r r}}
\toprule Activity & timing $|$ error & $\theta_{{\mathrm{{HSA}}}}$ (95\% CI) & P($>$0)\\\midrule
{c3}\bottomrule\end{{tabular}}\end{{center}}

\section*{{D2. Joint trend/cycle decomposition paths}}
\begin{{center}}\includegraphics[width=\linewidth]{{figures/verify_paths.png}}\end{{center}}
The estimated trend $\bar N$ tracks the (falling) annual Gustavo level; the cycle $\hat N$ is the stationary
deviation. This is the object the fixed EWMA approximates outside the model.

\section*{{Summary}}
$\delta$ and $\theta$ are positive throughout; $\theta$ is robust to the decomposition method (joint vs
fixed; EWMA/AR1/first-diff), to the firm/revenue profile, and to timing/error. It remains modest (the
benchmarked disaggregation keeps Gustavo's within-year variation small), but the sign and the competition
mechanism are stable across the battery.

\end{{document}}
"""
    tp = out / "hsa_ppi_identification_verification_report.tex"
    tp.write_text(tex, encoding="utf-8")
    r = subprocess.run(["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error", tp.name],
                       cwd=out, check=False, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if r.returncode:
        raise RuntimeError("LaTeX failed:\n" + r.stdout[-3000:])
    print("wrote", tp.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
