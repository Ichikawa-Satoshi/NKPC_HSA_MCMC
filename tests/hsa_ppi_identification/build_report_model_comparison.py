"""Report for the nested-model comparison (CES/Slope/Direct/Dynamic/Joint) on the
v2 Gustavo x Capital IQ competition series. Reads results/model_comparison/model_comparison.json.

    python tests/hsa_ppi_identification/model_comparison.py
    python tests/hsa_ppi_identification/build_report_model_comparison.py
"""
from __future__ import annotations
import json, subprocess
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np, pandas as pd
from scipy.stats import norm, gaussian_kde

import sys as _sys, pathlib as _pathlib  # noqa: E402
_ROOT = next(_p for _p in _pathlib.Path(__file__).resolve().parents if (_p / "pyproject.toml").exists())
_sys.path[:0] = [str(_ROOT), str(_ROOT / "src"), str(_ROOT / "tests")]
from tests import _bootstrap  # noqa: F401,E402

BUNDLE = Path(__file__).resolve().parent
BLUE, ORANGE, GREEN, GREY = "#0072B2", "#D55E00", "#009E73", "#888888"
CELL_LAB = {"ppi|inverse_markup": "PPI / inverse markup", "ppi|neg_unemp_gap": "PPI / neg-unemp gap",
            "core_cpi|inverse_markup": "core CPI / inverse markup", "core_cpi|neg_unemp_gap": "core CPI / neg-unemp gap"}
MORDER = [0, 1, 2, 3, 4]
MDEF = {0: r"CES: $\delta=\theta=\gamma=0$", 1: r"Slope: $\delta$ free",
        2: r"Direct: $\theta_0$ free", 3: r"Dynamic: $\theta_0,\gamma$ free",
        4: r"Joint: $\delta,\theta_0,\gamma$ free"}
COEF_ROWS = [("intercept", r"$a$"), ("alpha", r"$\alpha$ (fwd)"), ("alpha_b", r"$\alpha_b$ (lag)"),
             ("kappa_0", r"$\kappa_0$"), ("delta", r"$\delta$"), ("theta_0", r"$\theta_0$"), ("gamma", r"$\gamma$")]
TVP_LAB = {"kappa_0": r"$\kappa_0$", "delta": r"$\delta$", "theta_0": r"$\theta_0$", "gamma": r"$\gamma$"}


def esc(v): return str(v).replace("_", r"\_").replace("%", r"\%")


def _fmt(c):
    if c is None:
        return "--"
    m, p, _ = c
    pp = p if m > 0 else 1 - p
    return f"${m:+.3f}$ (P{pp:.2f})"


def _best(cell, metric):
    vals = {}
    for m in MORDER:
        v = cell["models"][str(m)][metric]
        if v is not None and v == v:
            vals[m] = v
    if not vals:
        return set()
    tgt = min(vals.values()) if metric == "waic" else max(vals.values())
    return {m for m, v in vals.items() if abs(v - tgt) < 1e-9}


def _maxrhat(d):
    rs = [c[2] for c in (d["delta"], d["theta_0"], d["gamma"], d["kappa_0"]) if c is not None]
    return max(rs) if rs else 1.0


def _conv(d):
    return r"$^{\dagger}$" if _maxrhat(d) > 1.1 else ""


# ---------------------------------------------------------------- tables
def cmp_table(R, key):
    cell = R[key]; bw = _best(cell, "waic"); bl = _best(cell, "log_ml")
    lines = []
    for m in MORDER:
        d = cell["models"][str(m)]
        waic = d["waic"]; lml = d["log_ml"]; cv = _conv(d)
        ws = f"\\textbf{{{waic:.1f}}}{cv}" if m in bw else f"{waic:.1f}{cv}"
        ls = "--" if lml != lml else (f"\\textbf{{{lml:.1f}}}" if m in bl else f"{lml:.1f}")
        lines.append(f"{m} & {MDEF[m]} & {ws} & {ls} & {_fmt(d['delta'])} & "
                     f"{_fmt(d['theta_0'])} & {_fmt(d['gamma'])} \\\\")
    body = "\n".join(lines)
    return rf"""\subsection*{{{esc(CELL_LAB[key])} ($n={cell['n']}$)}}
\begin{{center}}\small\begin{{tabular}}{{c l r r c c c}}
\toprule
M & specification & WAIC & $\log\mathrm{{ML}}$ & $\delta$ & $\theta_0$ & $\gamma$\\\midrule
{body}
\bottomrule\end{{tabular}}\end{{center}}"""


def coeff_table(R, key):
    ct = R[key]["coeff_table"]

    def cellval(m, name):
        e = ct[str(m)].get(name)
        if e is None:
            return "--"
        mean, lo, hi, p, rhat = e
        pp = p if mean > 0 else 1 - p
        dag = r"$^{\dagger}$" if rhat > 1.1 else ""
        return f"${mean:+.3f}$ (P{pp:.2f}){dag}"

    rows = []
    for name, lab in COEF_ROWS:
        cells = " & ".join(cellval(m, name) for m in MORDER)
        rows.append(f"{lab} & {cells} \\\\")
    body = "\n".join(rows)
    return rf"""\subsection*{{{esc(CELL_LAB[key])}}}
\begin{{center}}\footnotesize\begin{{tabular}}{{l c c c c c}}
\toprule
coef & M0 CES & M1 Slope & M2 Direct & M3 Dynamic & M4 Joint\\\midrule
{body}
\bottomrule\end{{tabular}}\end{{center}}"""


# ---------------------------------------------------------------- figures
def fig_paths(R, figures):
    cells = list(CELL_LAB)
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7))
    for ax, key in zip(axes.ravel(), cells):
        j = R[key]["joint"]; per = pd.PeriodIndex(R[key]["periods"], freq="Q").to_timestamp()
        nb, nbl, nbh = [np.array(x) for x in j["nbar"]]
        nh, nhl, nhh = [np.array(x) for x in j["nhat"]]
        ax.plot(per, nb, color=GREEN, lw=2.2, label=r"$\bar N_t$ (trend)")
        ax.fill_between(per, nbl, nbh, color=GREEN, alpha=0.15)
        ax.plot(per, nh, color=BLUE, lw=1.1, label=r"$\hat N_t$ (cycle)")
        ax.fill_between(per, nhl, nhh, color=BLUE, alpha=0.12)
        ax.axhline(0, color="black", lw=0.6); ax.set_title(CELL_LAB[key], fontsize=9)
        ax.legend(frameon=False, fontsize=7)
    fig.suptitle(r"Joint (state-space) decomposition, Model 4: $\bar N_t$ (trend) and $\hat N_t$ (cycle)")
    fig.tight_layout(); fig.savefig(figures / "mc_paths.png", dpi=185); plt.close(fig)


def fig_tvp(R, figures):
    cells = list(CELL_LAB)
    for coef, fname, ttl in [("kappa", "mc_tvp_kappa.png", r"$\kappa_t=\kappa_0+\delta\bar N_t$"),
                             ("theta", "mc_tvp_theta.png", r"$\theta_t=\theta_0+\gamma\bar N_t$")]:
        fig, axes = plt.subplots(2, 2, figsize=(11, 7))
        for ax, key in zip(axes.ravel(), cells):
            j = R[key]["joint"]; per = pd.PeriodIndex(R[key]["periods"], freq="Q").to_timestamp()
            m, lo, hi = [np.array(x) for x in j[coef]]
            ax.plot(per, m, color=ORANGE, lw=2)
            ax.fill_between(per, lo, hi, color=ORANGE, alpha=0.15)
            ax.axhline(0, color="black", lw=0.6); ax.set_title(CELL_LAB[key], fontsize=9)
        fig.suptitle(f"Time-varying {ttl} (Joint, Model 4)")
        fig.tight_layout(); fig.savefig(figures / fname, dpi=185); plt.close(fig)


def fig_ppd(R, figures):
    """Prior vs posterior of the four HSA coefficients (Joint, Model 4), all cells overlaid."""
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    colors = {"ppi|inverse_markup": BLUE, "ppi|neg_unemp_gap": ORANGE,
              "core_cpi|inverse_markup": GREEN, "core_cpi|neg_unemp_gap": "#9467BD"}
    for ax, coef in zip(axes.ravel(), ["kappa_0", "delta", "theta_0", "gamma"]):
        xmax = 0.0
        for key in CELL_LAB:
            d = np.array(R[key]["joint"]["ppd"][coef])
            if d.std() < 1e-9:
                continue
            kde = gaussian_kde(d); xs = np.linspace(d.min(), d.max(), 200)
            ax.plot(xs, kde(xs), color=colors[key], lw=1.6, label=CELL_LAB[key])
            xmax = max(xmax, np.abs([d.min(), d.max()]).max())
        # prior (use the mean sd across cells; mean 0 for all four)
        sd = np.mean([R[key]["joint"]["prior"][coef][1] for key in CELL_LAB])
        xs = np.linspace(-max(xmax, 3 * sd), max(xmax, 3 * sd), 300)
        ax.plot(xs, norm.pdf(xs, 0.0, sd), color=GREY, lw=1.6, ls="--", label="prior")
        ax.axvline(0, color="black", lw=0.6); ax.set_title(TVP_LAB[coef], fontsize=11)
    axes.ravel()[0].legend(frameon=False, fontsize=7)
    fig.suptitle("Prior vs posterior of the HSA coefficients (Joint, Model 4)")
    fig.tight_layout(); fig.savefig(figures / "mc_ppd.png", dpi=185); plt.close(fig)


# ---------------------------------------------------------------- main
def main():
    out = BUNDLE / "results"; figures = out / "figures"; figures.mkdir(parents=True, exist_ok=True)
    R = json.loads((out / "model_comparison" / "model_comparison.json").read_text())
    order = [k for k in CELL_LAB if k in R]
    has_joint = all("joint" in R[k] for k in order)
    if has_joint:
        fig_paths(R, figures); fig_tvp(R, figures); fig_ppd(R, figures)
    cmp_tables = "\n\n".join(cmp_table(R, k) for k in order)
    coef_tables = "\n\n".join(coeff_table(R, k) for k in order)

    def winner(key, metric):
        b = _best(R[key], metric)
        return ", ".join(MDEF[m].split(":")[0] for m in sorted(b)) if b else "--"
    wl = "\n".join(
        f"{esc(CELL_LAB[k])} & {winner(k,'waic')} & {winner(k,'log_ml')} \\\\" for k in order)

    fig_block = ""
    if has_joint:
        fig_block = r"""
\section*{5. Full coefficient tables}
Posterior mean (sign probability P) for every coefficient in each model; a dagger ($\dagger$) marks
$\hat R>1.1$. ``--'' means the coefficient is switched off in that model.
""" + coef_tables + r"""

\section*{6. Prior vs.\ posterior (Joint model, all four cells)}
\begin{center}\includegraphics[width=\linewidth]{figures/mc_ppd.png}\end{center}
The posteriors of $\delta$ and $\theta_0$ are shifted only modestly off the (dashed) prior and stay
wide: with all channels free the data update the competition coefficients weakly. $\kappa_0$ updates
more strongly (positive with the unemployment gap). This is the graphical face of the identification
problem that the HSA restriction resolves.

\section*{7. Time-varying $\kappa_t$ and $\theta_t$ (Joint model)}
With $\delta,\gamma$ free the slope and direct loading move with trend competition,
$\kappa_t=\kappa_0+\delta\bar N_t$ and $\theta_t=\theta_0+\gamma\bar N_t$ (posterior mean and 95\% band).
\begin{center}\includegraphics[width=\linewidth]{figures/mc_tvp_kappa.png}\end{center}
\begin{center}\includegraphics[width=\linewidth]{figures/mc_tvp_theta.png}\end{center}
Because $\delta>0$ and $\bar N_t$ declines over the sample, $\kappa_t$ flattens -- the HSA-consistent
account of a flattening Phillips slope -- but the band is wide (the coefficients are weakly identified
when unrestricted). $\theta_t$ is close to $\theta_0$ throughout ($\gamma\approx0$).

\section*{8. Joint trend/cycle decomposition (Model 4)}
\begin{center}\includegraphics[width=\linewidth]{figures/mc_paths.png}\end{center}
$\bar N_t$ tracks the falling annual Gustavo level (rising concentration); $\hat N_t$ is the stationary
within-year cycle supplied by Capital IQ. The two enter the NKPC through the slope tilt
$\delta\bar N_t x_t$ and the direct term $-\theta_t\hat N_t$ respectively.
"""

    tex = rf"""\documentclass[11pt]{{article}}
\usepackage[margin=0.85in]{{geometry}}
\usepackage{{booktabs,graphicx,amsmath,xcolor,microtype,hyperref,newtxtext,newtxmath}}
\definecolor{{navy}}{{HTML}}{{17365D}}\hypersetup{{colorlinks=true,linkcolor=navy,urlcolor=navy}}
\setlength{{\parindent}}{{0pt}}\setlength{{\parskip}}{{5pt}}
\begin{{document}}
\begin{{center}}
{{\color{{navy}}\LARGE\bfseries HSA NKPC: Nested-Model Comparison}}\\[3pt]
{{\large CES vs Slope vs Direct vs Dynamic vs Joint --- Gustavo $\times$ Capital IQ $N$, 1974--2013}}
\end{{center}}

\section*{{1. The five nested models}}
All five share the HSA New Keynesian Phillips curve and differ only in which competition
channels are switched on. With $N_t=\bar N_t+\hat N_t$ (trend $+$ cycle, ten-log-point units),
\[
\pi_t=a+\alpha_b\pi_{{t-1}}+\alpha\,E_t\pi_{{t+1}}
+\big(\kappa_0+\delta\,\bar N_t\big)x_t-\big(\theta_0+\gamma\,\bar N_t\big)\hat N_t+\varepsilon_t .
\]
\begin{{center}}\small\begin{{tabular}}{{c l l}}
\toprule
M & name & free competition parameters\\\midrule
0 & CES (baseline) & none ($\delta=\theta_0=\gamma=0$); constant slope $\kappa_0$\\
1 & Slope only & $\delta$ --- competition tilts the slope, no direct channel\\
2 & Direct only & $\theta_0$ --- direct cyclical channel, constant slope\\
3 & Dynamic direct & $\theta_0,\gamma$ --- direct channel varies with trend competition\\
4 & Joint (full HSA) & $\delta,\theta_0,\gamma$ --- slope tilt \emph{{and}} varying direct channel\\
\bottomrule\end{{tabular}}\end{{center}}
$\bar N_t$ (random walk) and $\hat N_t$ (AR(1)) are latent states estimated \emph{{jointly}} with the
NKPC by a Gibbs/FFBS state-space sampler (measurement $N_t=\bar N_t+\hat N_t+\nu_t$), so every model
here uses the joint decomposition. Models 0--2 are linear-Gaussian (exact Kalman marginal likelihood);
Models 3--4 have the $\gamma\,\bar N_t\hat N_t$ bilinearity (particle-filter likelihood).

\textbf{{Why compare these and not just the HSA-restricted single-$\theta$ run?}} The HSA restriction
$\delta=b_x\zeta\theta$ ties the slope tilt to the direct channel; it is the right \emph{{structural}}
prior, but it cannot by itself tell us \emph{{which channel the data actually want}}. The nested ladder
does: if the data prefer M1 (Slope) the identification comes through $\delta\bar N_t x_t$; if they prefer
M2/M3 it comes through the direct $\hat N_t$ term; if M4 wins both channels are present (consistent with
the HSA restriction). WAIC and the marginal likelihood penalise the extra parameters, so a channel only
``wins'' if it earns its complexity.

\section*{{2. Model comparison (WAIC $\downarrow$, $\log$ marginal likelihood $\uparrow$)}}
Lower WAIC and higher $\log\mathrm{{ML}}$ are better; the best in each column is \textbf{{bold}}.
$\delta,\theta_0,\gamma$ are posterior means with sign probability P. A dagger ($\dagger$) flags a
model whose competition coefficients did \emph{{not}} mix (posterior split-$\hat R>1.1$): its fit
statistics reflect a non-converged mode and must be read with caution, not as evidence.
{cmp_tables}

\section*{{3. Which model wins each cell}}
\begin{{center}}\small\begin{{tabular}}{{l c c}}
\toprule
Cell & best by WAIC & best by $\log\mathrm{{ML}}$\\\midrule
{wl}
\bottomrule\end{{tabular}}\end{{center}}

\section*{{4. Reading}}
\textbf{{No competition channel decisively beats the CES baseline in a well-mixed cell.}}
Across the three cells that converge cleanly the story is consistent:
\begin{{itemize}}
\item \emph{{PPI cells.}} All five models sit within $\Delta\mathrm{{WAIC}}<4$ and
$\Delta\log\mathrm{{ML}}<3$ of the CES baseline --- differences inside Monte-Carlo noise. The direct
($\theta_0$) and slope ($\delta$) coefficients are positive on average (e.g.\ PPI/neg-unemp gap
$\theta_0=+0.10$, P$=0.85$; $\delta=+0.04$--$0.07$) but none earns its extra parameter: the data do
not by themselves separate the individual channels.
\item \emph{{core CPI / inverse markup.}} CES is best on both criteria; every added channel is flat
($\theta_0\approx0$, $\delta$ small). This is the inverse-markup cell where the competition channels
were never identified.
\item \emph{{core CPI / neg-unemp gap.}} The Slope and Joint models post a large apparent gain
($\Delta\mathrm{{WAIC}}\approx-25$), \emph{{but both are daggered}}: $\delta$ and $\kappa_0$ fail to mix
($\hat R=1.3$--$1.6$) and the Slope $\delta$ mean is essentially zero ($+0.007$, P$=0.57$). The gain
comes from the latent decomposition jumping to an alternative, non-converged mode when $\delta$ is
freed, not from an identified slope channel. It is \emph{{not}} evidence for the slope channel.
\end{{itemize}}

\textbf{{Why this is the expected result --- and why it justifies the HSA restriction.}} When
$\delta,\theta_0,\gamma$ are left free, the direct channel is identified only through the weak cyclical
state $\hat N_t$ and the slope tilt only through $\bar N_t x_t$; jointly they are near-collinear, so
each is individually weak and the sampler wanders (the daggered cells). This is exactly the
identification problem the structural HSA restriction $\delta=b_x\zeta\theta$ is designed to solve: it
ties the two channels to the \emph{{same}} mechanism (competition shifting the demand elasticity
$\zeta$), collapsing the collinear pair to a single well-identified $\theta$ that loads on
$b_x\zeta\bar N_t x_t-\hat N_t$. The nested comparison therefore does not overturn the HSA-restricted
finding --- it explains why the restriction is needed: the free channels are individually
under-identified, and the restriction is the structural prior that makes $\theta$ estimable while
keeping $\theta>0$ (and hence $\delta=b_x\zeta\theta>0$) intact.
{fig_block}
\end{{document}}
"""
    tp = out / "hsa_ppi_identification_model_comparison_report.tex"
    tp.write_text(tex, encoding="utf-8")
    r = subprocess.run(["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error", tp.name],
                       cwd=out, check=False, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if r.returncode:
        raise RuntimeError("LaTeX failed:\n" + r.stdout[-3000:])
    print("wrote", tp.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
