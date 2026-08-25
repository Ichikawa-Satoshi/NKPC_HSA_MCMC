"""Formal PDF report for the Gustavo x Capital IQ temporal-disaggregation cell.

Competition = annual Gustavo level carrying Capital IQ's within-year allocation
(a Chow-Lin/Denton-style temporal disaggregation), over the full Gustavo span
1974-2013. Sections: 0 model/data, 1 coefficient table (both activities x
variants), 2 prior-vs-posterior, 3 decomposition (series construction + slope),
4 convergence.

    python tests/hsa_ppi_identification/build_report_gustavo_capiq.py
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import sys as _sys, pathlib as _pathlib  # noqa: E402
_ROOT = next(_p for _p in _pathlib.Path(__file__).resolve().parents if (_p / "pyproject.toml").exists())
_sys.path[:0] = [str(_ROOT), str(_ROOT / "src"), str(_ROOT / "tests")]
from tests import _bootstrap  # noqa: F401,E402
from nkpc_hsa.config import load_yaml  # noqa: E402
from nkpc_hsa.report_models.cases import GUSTAVO_ANNUAL_COL  # noqa: E402
from tests.observed_hhi.functions import summarize_observed_fit  # noqa: E402
from tests.hsa_ppi_identification.functions import (  # noqa: E402
    _load_frame, build_gustavo_capiq_sample, gustavo_capiq_quarterly)
from tests.hsa_ppi_identification.build_report import (  # noqa: E402
    _fit, _draws, esc, DISP, fig_prior_post, fig_decomposition, fig_convergence, BLUE, ORANGE, GREEN)

BUNDLE = Path(__file__).resolve().parent


def fig_series(frame, config, figures):
    """Show the disaggregated competition series and the Capital IQ contribution shares."""
    Gq, info = gustavo_capiq_quarterly(frame, config["cell"]["competition"])
    s = info["s"]
    g = pd.to_numeric(frame[GUSTAVO_ANNUAL_COL], errors="coerce").dropna()
    gpts = g[(g.index >= pd.Period("1974Q4", freq="Q")) & (g.index <= pd.Period("2013Q4", freq="Q"))]
    per = pd.PeriodIndex(Gq.index).to_timestamp()
    fig, (ax, axb) = plt.subplots(1, 2, figsize=(11, 4.0), gridspec_kw={"width_ratios": [3, 1]})
    ax.plot(per, Gq.values, color=BLUE, lw=1.1, alpha=0.85, label="quarterly $Gq_t$ (disaggregated)")
    ax.plot(gpts.index.to_timestamp(), 10 * np.log(gpts.values), "s-", color=GREEN, ms=4, lw=1.2,
            label="Gustavo annual benchmark (Q4)")
    ax.axvline(pd.Timestamp("1989-01-01"), color=ORANGE, ls=":", lw=1)
    ax.annotate("Capital IQ starts 1989", (pd.Timestamp("1989-06-01"), ax.get_ylim()[0]), fontsize=8, color=ORANGE)
    ax.set_title("Disaggregated Gustavo (annual level, avg. Capital IQ quarterly profile)")
    ax.set_ylabel("competition $10\\log N$"); ax.set_xlabel("year"); ax.legend(frameon=False, fontsize=8)
    axb.bar(range(1, 5), s, color=BLUE)
    axb.axhline(0, color="black", lw=0.7)
    axb.set_xticks(range(1, 5), [f"Q{q}" for q in range(1, 5)])
    axb.set_title("avg quarterly\ncontribution $s_q$", fontsize=9); axb.set_ylabel("share of annual change")
    fig.tight_layout(); fig.savefig(figures / "gc_series.png", dpi=190); plt.close(fig)
    return s


def main():
    config = load_yaml(BUNDLE / "config.yaml")
    design, sampling = config["design"], config["sampling"]
    out = BUNDLE / "results"; figures = out / "figures"; figures.mkdir(parents=True, exist_ok=True)
    frame = _load_frame()
    start = config["gustavo_capiq"]["samples"]["long"]

    shares = fig_series(frame, config, figures)

    fits_by_act, summ_by_act, meta = {}, {}, {}
    for act in config["activities"]:
        smp = build_gustavo_capiq_sample(frame, config, activity=act, sample_start=start)
        fits = {v: _fit(smp, v, design, sampling) for v in design["model_variants"]}
        summ = {v: summarize_observed_fit(f) for v, f in fits.items()}
        for v in summ:
            summ[v]["P"] = summ[v].apply(lambda r: r["sign_probability"] if r["mean"] > 0 else 1 - r["sign_probability"], axis=1)
        fits_by_act[act] = fits; summ_by_act[act] = summ
        meta[act] = (str(smp.periods.min()), str(smp.periods.max()), len(smp.y), smp)

    # figures from the inverse_markup cell (headline)
    hb = config.get("primary_activity", "inverse_markup")
    fig_prior_post(fits_by_act[hb], figures, "_gc")
    fig_decomposition(meta[hb][3], design, fits_by_act[hb]["constant_theta"], fits_by_act[hb]["quadratic_theta"], figures, "_gc")
    all_summ = pd.concat([s for d in summ_by_act.values() for s in d.values()], ignore_index=True)
    fig_convergence(all_summ, figures, "_gc")

    def table(act):
        rows = []
        summ = summ_by_act[act]
        blocks = [("Constant-$\\theta$", "constant_theta", ["kappa_0", "kappa_1", "theta_0"]),
                  ("Varying-$\\theta$", "varying_theta", ["kappa_0", "kappa_1", "theta_0", "gamma"]),
                  ("Quadratic slope", "quadratic_theta", ["kappa_0", "kappa_1", "kappa_2", "theta_0"]),
                  ("HSA-restricted", "hsa_restricted", ["kappa_0", "theta_hsa"])]
        for title, v, ps in blocks:
            rows.append(f"\\multicolumn{{5}}{{l}}{{\\emph{{{title}}}}}\\\\")
            for p in ps:
                m = summ[v][summ[v].parameter == p]
                if len(m):
                    m = m.iloc[0]
                    rows.append(f"{DISP.get(p,p)} & {m['mean']:+.3f} & [{m['ci_2.5']:+.3f}, {m['ci_97.5']:+.3f}] "
                                f"& {m['P']:.2f} & {m['rhat']:.3f} \\\\")
            rows.append("\\midrule")
        return "\n".join(rows[:-1])

    max_rhat = float(all_summ["rhat"].max())
    im, ug = "inverse_markup", "neg_unemp_gap"
    imn, ugn = meta[im][2], meta[ug][2]
    tex = rf"""\documentclass[11pt]{{article}}
\usepackage[margin=0.8in]{{geometry}}
\usepackage{{booktabs,graphicx,amsmath,xcolor,microtype,hyperref,newtxtext,newtxmath}}
\definecolor{{navy}}{{HTML}}{{17365D}}\hypersetup{{colorlinks=true,linkcolor=navy,urlcolor=navy}}
\setlength{{\parindent}}{{0pt}}\setlength{{\parskip}}{{5pt}}
\begin{{document}}
\begin{{center}}
{{\color{{navy}}\LARGE\bfseries HSA Phillips Curve: Gustavo $\times$ Capital IQ}}\\[3pt]
{{\large Temporal-disaggregation competition, PPI, 1974--2013}}
\end{{center}}

\section*{{0. Model and data}}
\textbf{{Competition series (temporal disaggregation).}} The annual Gustavo effective-firm count is the
benchmark (its trend is consistent with the rising-concentration literature); Capital IQ supplies the
average within-year allocation. Estimate each quarter's average share of the annual change from Capital
IQ, robustly,
\[
s_q=\frac{{\langle \Delta \mathrm{{CIQ}}_q,\ \Delta \mathrm{{CIQ}}_{{\text{{annual}}}}\rangle}}
        {{\langle \Delta \mathrm{{CIQ}}_{{\text{{annual}}}},\ \Delta \mathrm{{CIQ}}_{{\text{{annual}}}}\rangle}},
\qquad \sum_{{q=1}}^{{4}} s_q=1
\quad\big(\hat s=[{shares[0]:.2f},\,{shares[1]:.2f},\,{shares[2]:.2f},\,{shares[3]:.2f}]\big),
\]
then allocate every annual Gustavo change (Q4-to-Q4) across quarters by $s_q$ and cumulate, so each Q4
matches the annual Gustavo benchmark \emph{{exactly}}. This is a Chow--Lin/Denton-style disaggregation
(annual level from Gustavo, average quarterly profile from Capital IQ). It avoids the spline-interpolation
artefact of a purely interpolated quarterly Gustavo and, because Gustavo runs from 1974, the quarterly
estimation covers the \textbf{{pre-1985 steep-Phillips era}}.

\textbf{{Model (theory-faithful HSA NKPC, $\psi$ excluded).}}
\[
\pi_t=a+\beta_b\pi_{{t-1}}+\beta_f E_t\pi_{{t+1}}+(\kappa_0+\kappa_1 q_t)x_t-\theta_0 c_t+\varepsilon_t,
\qquad \varepsilon_t\sim\mathrm{{AR}}(1),
\]
with $q_t$ the slow competition (EWMA level of $Gq_t$), $c_t$ the cyclical part (EWMA innovation, lag 1),
$x_t$ the activity. Two activity cells: inverse markup (HSA-restriction cell, $b_x\zeta\theta=\kappa_1$)
and the negative unemployment gap. Bayesian Gibbs, AR(1) error, {sampling['iterations']} iters $\times$
{sampling['chains']} chains.

\begin{{center}}\includegraphics[width=0.95\linewidth]{{figures/gc_series.png}}\end{{center}}

\section*{{1. Coefficients}}
\textbf{{Inverse-markup cell}} (n={imn}, {esc(meta[im][0])}--{esc(meta[im][1])}):
\begin{{center}}\small\begin{{tabular}}{{l r c c r}}
\toprule Parameter & Mean & 95\% CI & P($>$0) & $\widehat R$\\\midrule
{table(im)}
\bottomrule\end{{tabular}}\end{{center}}

\textbf{{Negative-unemployment-gap cell}} (n={ugn}):
\begin{{center}}\small\begin{{tabular}}{{l r c c r}}
\toprule Parameter & Mean & 95\% CI & P($>$0) & $\widehat R$\\\midrule
{table(ug)}
\bottomrule\end{{tabular}}\end{{center}}
Over the full Gustavo span the HSA channels are theory-consistent: $\delta=\kappa_1>0$ and the
HSA-restricted $\theta>0$ with the interval excluding zero; the base slope $\kappa_0$ is now positive
(strongly so under the unemployment gap) because the sample includes the steep-Phillips period.

\section*{{2. Prior vs.\ posterior (inverse-markup cell)}}
\begin{{center}}\includegraphics[width=\linewidth]{{figures/prior_vs_posterior_gc.png}}\end{{center}}

\section*{{3. Decomposition (inverse-markup cell)}}
\begin{{center}}\includegraphics[width=\linewidth]{{figures/decomposition_gc.png}}\end{{center}}

\section*{{4. Convergence}}
\begin{{center}}\includegraphics[width=0.95\linewidth]{{figures/convergence_gc.png}}\end{{center}}
Max $\widehat R={max_rhat:.3f}$ across both cells and all variants.

\end{{document}}
"""
    tex_path = out / "hsa_ppi_identification_gustavo_capiq_report.tex"
    tex_path.write_text(tex, encoding="utf-8")
    r = subprocess.run(["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error", tex_path.name],
                       cwd=out, check=False, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if r.returncode:
        raise RuntimeError("LaTeX build failed:\n" + r.stdout[-3000:])
    print("wrote", tex_path.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
