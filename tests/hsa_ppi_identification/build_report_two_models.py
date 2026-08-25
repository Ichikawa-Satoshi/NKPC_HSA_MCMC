"""Report for the two HSA-NKPC estimators on the indicator-allocated Gustavo x
Capital IQ competition series. Reads results/two_models/two_models.json.

    python tests/hsa_ppi_identification/two_models.py
    python tests/hsa_ppi_identification/build_report_two_models.py
"""
from __future__ import annotations
import json, subprocess
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np, pandas as pd
from scipy.stats import norm

import sys as _sys, pathlib as _pathlib  # noqa: E402
_ROOT = next(_p for _p in _pathlib.Path(__file__).resolve().parents if (_p / "pyproject.toml").exists())
_sys.path[:0] = [str(_ROOT), str(_ROOT / "src"), str(_ROOT / "tests")]
from tests import _bootstrap  # noqa: F401,E402

BUNDLE = Path(__file__).resolve().parent
BLUE, ORANGE, GREEN = "#0072B2", "#D55E00", "#009E73"
ZETA = 6.0
CELL_LAB = {"ppi|inverse_markup": "PPI / inverse markup", "ppi|neg_unemp_gap": "PPI / neg-unemp gap",
            "core_cpi|inverse_markup": "core CPI / inverse markup", "core_cpi|neg_unemp_gap": "core CPI / neg-unemp gap"}


def esc(v): return str(v).replace("_", r"\_").replace("%", r"\%")


def fig_paths(R, figures):
    cells = list(R["cells"])
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7))
    for ax, key in zip(axes.ravel(), cells):
        j = R["cells"][key]["model2_joint"]
        per = pd.PeriodIndex(j["periods"], freq="Q").to_timestamp()
        nbar = np.array(j["nbar"]); nhat = np.array(j["nhat"])
        ax.plot(per, nbar, color=GREEN, lw=2.2, label=r"$\bar N_t$ (trend)")
        ax.plot(per, nhat, color=BLUE, lw=1.1, label=r"$\hat N_t$ (cycle)")
        ax.axhline(0, color="black", lw=0.6)
        ax.set_title(CELL_LAB[key], fontsize=9); ax.legend(frameon=False, fontsize=7)
    fig.suptitle(r"Joint (state-space) decomposition: $\bar N_t$ (trend) and $\hat N_t$ (cycle)")
    fig.tight_layout(); fig.savefig(figures / "tm_paths.png", dpi=185); plt.close(fig)


def fig_tvp(R, figures):
    """Time-varying kappa_t and theta_t: fixed vs joint decomposition, 4 cells."""
    cells = list(R["cells"])
    for coef, fname, ttl in [("kappa", "tm_tvp_kappa.png", r"$\kappa_t=\kappa_0+\delta\bar N_t$"),
                             ("theta", "tm_tvp_theta.png", r"$\theta_t=\theta_0+\gamma\bar N_t$")]:
        fig, axes = plt.subplots(2, 2, figsize=(11, 7))
        for ax, key in zip(axes.ravel(), cells):
            c = R["cells"][key]; per = pd.PeriodIndex(c["model2_joint"]["periods"], freq="Q").to_timestamp()
            mf, lof, hif = [np.array(x) for x in c["tvp_fixed"][coef]]
            mj, loj, hij = [np.array(x) for x in c["tvp_joint"][coef]]
            ax.plot(per, mf, color=BLUE, lw=2, label="fixed (EWMA)")
            ax.fill_between(per, lof, hif, color=BLUE, alpha=0.12)
            ax.plot(per, mj, color=ORANGE, lw=2, ls="--", label="joint (state-space)")
            ax.fill_between(per, loj, hij, color=ORANGE, alpha=0.12)
            ax.axhline(0, color="black", lw=0.6); ax.set_title(CELL_LAB[key], fontsize=9)
        axes.ravel()[0].legend(frameon=False, fontsize=7)
        fig.suptitle(f"Time-varying {ttl}: fixed vs joint decomposition")
        fig.tight_layout(); fig.savefig(figures / fname, dpi=175); plt.close(fig)


def fig_ppd(R, figures, key):
    """Prior vs posterior for the varying-theta coefficients of one cell."""
    c = R["cells"][key]; ppd = c["ppd"]; psd = c["prior_sds"]
    disp = {"kappa_0": r"$\kappa_0$", "kappa_1": r"$\delta=\kappa_1$", "theta_0": r"$\theta_0$", "gamma": r"$\gamma$"}
    fig, axes = plt.subplots(2, 2, figsize=(9, 6))
    for ax, p in zip(axes.ravel(), ["kappa_0", "kappa_1", "theta_0", "gamma"]):
        d = np.array(ppd[p]); sd = psd[p]
        ax.hist(d, bins=35, density=True, color=BLUE, alpha=0.6, label="posterior")
        xs = np.linspace(min(d.min(), -3 * sd), max(d.max(), 3 * sd), 300)
        ax.plot(xs, norm.pdf(xs, 0, sd), color=ORANGE, lw=2, label="prior N(0,sd)")
        ax.axvline(0, color="black", lw=0.7, ls="--"); ax.axvline(d.mean(), color=GREEN, lw=1.5)
        ax.set_title(f"{disp[p]}  (P(>0)={np.mean(d>0):.2f})", fontsize=9); ax.set_yticks([])
    axes.ravel()[0].legend(frameon=False, fontsize=8)
    fig.suptitle(f"Prior vs posterior — {CELL_LAB[key]} (fixed, varying-$\\theta$)")
    fig.tight_layout(); fig.savefig(figures / "tm_ppd.png", dpi=185); plt.close(fig)


def main():
    out = BUNDLE / "results"; figures = out / "figures"; figures.mkdir(parents=True, exist_ok=True)
    R = json.loads((out / "two_models" / "two_models.json").read_text())
    fig_paths(R, figures)
    fig_tvp(R, figures)
    ppd_key = "ppi|neg_unemp_gap" if "ppi|neg_unemp_gap" in R["cells"] else list(R["cells"])[0]
    fig_ppd(R, figures, ppd_key)
    w = R["wbar"]

    def trow(key):
        c = R["cells"][key]; m1 = c["model1_fixed"]; m2 = c["model2_joint"]
        t1, t2 = m1["theta"], m2["theta"]
        return (f"{esc(CELL_LAB[key])} & {c['n']} & "
                f"{t1[0]:+.3f} [{t1[1]:+.3f}, {t1[2]:+.3f}] & {t1[3]:.2f} & "
                f"{t2[0]:+.3f} [{t2[1]:+.3f}, {t2[2]:+.3f}] & {t2[3]:.2f} \\\\")
    rows = "\n".join(trow(k) for k in R["cells"])
    ppd_lab = esc(CELL_LAB[ppd_key])

    tex = rf"""\documentclass[11pt]{{article}}
\usepackage[margin=0.85in]{{geometry}}
\usepackage{{booktabs,graphicx,amsmath,xcolor,microtype,hyperref,newtxtext,newtxmath}}
\definecolor{{navy}}{{HTML}}{{17365D}}\hypersetup{{colorlinks=true,linkcolor=navy,urlcolor=navy}}
\setlength{{\parindent}}{{0pt}}\setlength{{\parskip}}{{5pt}}
\begin{{document}}
\begin{{center}}
{{\color{{navy}}\LARGE\bfseries HSA NKPC: Fixed vs Joint Decomposition}}\\[3pt]
{{\large Gustavo $\times$ Capital IQ competition, PPI \& core CPI, 1974--2013}}
\end{{center}}

\section*{{1. The competition series $N_t$ (indicator-based disaggregation)}}
The latent competitive environment is $N_t=\bar N_t+\hat N_t$: a slow trend $\bar N_t$ and a stationary
cycle $\hat N_t$ (all in ten-log-point units, $10\log$ effective firms). Annual Gustavo pins the trend;
Capital IQ supplies the within-year allocation. Writing $\Delta N^{{G}}_{{t}}$ for the annual (Q4-to-Q4)
Gustavo change, the quarterly increments are
\[
\widehat{{\Delta N}}^{{G}}_{{tq}}=
\begin{{cases}}
\hat w^{{CIQ}}_{{tq}}\,\Delta N^{{G}}_{{t}}, & \text{{Capital IQ observed in year }}t,\\[3pt]
\bar w_{{q}}\,\Delta N^{{G}}_{{t}}, & \text{{Capital IQ missing}},
\end{{cases}}
\qquad \sum_{{q=1}}^{{4}} w_{{tq}}=1,
\]
where $\hat w^{{CIQ}}_{{tq}}=\Delta \mathrm{{CIQ}}_{{tq}}/\sum_q \Delta \mathrm{{CIQ}}_{{tq}}$ is that year's own
Capital IQ quarterly share and $\bar w_q$ is the average share
$[{w[0]:.2f},{w[1]:.2f},{w[2]:.2f},{w[3]:.2f}]$ (used before Capital IQ begins, or when the annual
Capital IQ change is too small for stable shares). Cumulating reproduces each annual Gustavo benchmark
at Q4 exactly, giving a quarterly $N_t$ over the full 1974--2013 span (steep-Phillips era included) without
a spline-interpolation artefact.

\section*{{2. The estimated model (HSA NKPC)}}
Both estimators share the theory-faithful HSA New Keynesian Phillips curve (competition level term
$\psi N_t$ excluded, as it is not part of the structural equation):
\[
\pi_t=a+\alpha_b\pi_{{t-1}}+\alpha\,E_t\pi_{{t+1}}
+\big(\kappa_0+\delta\,\bar N_t\big)x_t-\theta\,\hat N_t+\varepsilon_t,
\qquad \varepsilon_t\sim\mathrm{{AR}}(1),
\]
with $x_t$ the activity proxy (inverse markup or negative unemployment gap) and $E_t\pi_{{t+1}}$ the SPF
GDP-deflator forecast. The Phillips slope is competition-dependent, $\kappa_t=\kappa_0+\delta\bar N_t$, and
$\theta$ is the direct loading of the cyclical competition state $\hat N_t$.

\textbf{{HSA restriction.}} Under the Rotemberg--HSA producer-pricing model the theoretical slope and direct
loading are $\kappa^{{th}}(N)=(\zeta(N)-1)/\chi$ and $\Theta^{{th}}(N)$, and they obey the identity
\[
\frac{{d\kappa(N)}}{{dN}}=\zeta(N)\,\theta(N)\;\Longrightarrow\; \boxed{{\;\delta=b_x\,\zeta_{{\mathrm{{ref}}}}\,\theta\;}}
\quad(b_x=1,\ \zeta_{{\mathrm{{ref}}}}={ZETA:.0f}),
\]
i.e.\ the competition-sensitivity of the slope ($\delta$) and the direct channel ($\theta$) are \emph{{not}}
free parameters: both arise from the \emph{{same}} mechanism -- competition (the number of firms/varieties)
shifting the demand elasticity $\zeta$. \textbf{{This is why the restriction is justified}}: it is the
structural cross-equation implication of the model, not an estimation convenience. Imposing it lets $\theta$
be identified through the well-measured slope--trend interaction $\delta\bar N_t x_t$ (the trend $\bar N_t$
has ample low-frequency variation) rather than only through the weak cyclical channel $\hat N_t$. Operationally
a single $\theta$ loads on $b_x\zeta_{{\mathrm{{ref}}}}\bar N_t x_t-\hat N_t$.

\textbf{{Two estimators.}}
\emph{{Model 1 (fixed decomposition)}}: $\bar N_t,\hat N_t$ are extracted from $N_t$ by a one-sided EWMA
(8-quarter half-life) \emph{{before}} estimation; the NKPC is then a Bayesian AR(1)-error regression.
\emph{{Model 2 (joint decomposition)}}: $\bar N_t$ (random walk) and $\hat N_t$ (AR(1)) are latent states
estimated \emph{{jointly}} with the NKPC by a Gibbs/FFBS state-space sampler (measurement
$N_t=\bar N_t+\hat N_t+\nu_t$). Both are MCMC; both impose the HSA restriction and a hybrid ($\pi_{{t-1}}$) term.

\section*{{3. Results: $\theta$ (direct HSA channel), both estimators}}
\begin{{center}}\small\begin{{tabular}}{{l r c c c c}}
\toprule & & \multicolumn{{2}}{{c}}{{Model 1: fixed (EWMA)}} & \multicolumn{{2}}{{c}}{{Model 2: joint (state-space)}}\\
\cmidrule(lr){{3-4}}\cmidrule(lr){{5-6}}
Cell & $n$ & $\theta$ [95\% CI] & P($>$0) & $\theta$ [95\% CI] & P($>$0)\\\midrule
{rows}
\bottomrule\end{{tabular}}\end{{center}}
$\theta>0$ across all four cells and both estimators (the implied $\delta=b_x\zeta\theta>0$). The joint
decomposition is comparable to (or stronger than) the fixed one, so the result does not hinge on
pre-filtering the competition series.

\section*{{4. Time-varying coefficients $\kappa_t$ and $\theta_t$}}
With a varying-$\theta$ specification the slope and direct loading move with the trend competition,
$\kappa_t=\kappa_0+\delta\bar N_t$ and $\theta_t=\theta_0+\gamma\bar N_t$. We show both the fixed (EWMA)
and joint (state-space) decompositions.
\begin{{center}}\includegraphics[width=\linewidth]{{figures/tm_tvp_kappa.png}}\end{{center}}
\begin{{center}}\includegraphics[width=\linewidth]{{figures/tm_tvp_theta.png}}\end{{center}}
In the PPI cells the fixed decomposition shows a clear flattening of $\kappa_t$ and a rising $\theta_t$ as
competition $\bar N_t$ declines; the joint decomposition gives the same direction but muted, because
estimating $\bar N_t,\hat N_t$ with uncertainty attenuates $\delta,\gamma$ (errors-in-variables). The core
CPI cells are flat ($\delta,\gamma\approx0$).

\section*{{5. Prior vs.\ posterior (varying-$\theta$ coefficients, {ppd_lab})}}
\begin{{center}}\includegraphics[width=0.9\linewidth]{{figures/tm_ppd.png}}\end{{center}}
The posteriors of $\delta=\kappa_1$ and $\theta_0$ concentrate away from the prior toward positive values;
$\kappa_0$ updates less (the marginal-cost puzzle) and $\gamma$ is small.

\section*{{6. Joint trend/cycle decomposition}}
\begin{{center}}\includegraphics[width=\linewidth]{{figures/tm_paths.png}}\end{{center}}
$\bar N_t$ tracks the falling annual Gustavo level (rising concentration); $\hat N_t$ is the stationary
cycle. Because $\delta>0$ and $\bar N_t$ declines, the effective slope $\kappa_t=\kappa_0+\delta\bar N_t$
flattens over the sample -- an HSA-consistent account of the flattening Phillips curve.

\end{{document}}
"""
    tp = out / "hsa_ppi_identification_two_models_report.tex"
    tp.write_text(tex, encoding="utf-8")
    r = subprocess.run(["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error", tp.name],
                       cwd=out, check=False, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if r.returncode:
        raise RuntimeError("LaTeX failed:\n" + r.stdout[-3000:])
    print("wrote", tp.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
