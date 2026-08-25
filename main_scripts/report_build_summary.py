"""Build tables and figures for docs/report/summary_results and compile the PDF.

Consumes the posterior draws written by main_scripts/report_estimate.py and
populates summary_results.tex following the pre-declared skeleton, per case:

    \\subsection{Coefficient Table}
        \\subsubsection{Marginal Likelihoods}
        \\subsubsection{Predictive Distributions}
    \\subsection{Coefficient: alpha}
    \\subsection{Coefficient: kappa}
        \\subsubsection{Coefficient: delta and Time-Varying kappa_t}
    \\subsection{Coefficient: theta}
        \\subsubsection{Coefficient: gamma and Time-Varying theta_t}
    \\subsubsection{Decomposed Competition}
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import sys as _sys, pathlib as _pathlib  # noqa: E402
_ROOT = next(_p for _p in _pathlib.Path(__file__).resolve().parents if (_p / "pyproject.toml").exists())
_sys.path[:0] = [str(_ROOT), str(_ROOT / "src")]
from nkpc_hsa.paths import results_root  # noqa: E402

REPORT_DIR = _ROOT / "docs" / "report" / "summary_results"
OUT_TABLES = _ROOT / "docs" / "report" / "results" / "tables"
OUT_FIGS = _ROOT / "docs" / "report" / "results" / "figures"

BLUE, ORANGE, GREEN, GREY = "#0072B2", "#D55E00", "#009E73", "#6B7280"
MODEL_LABELS = {0: "CES", 1: "Slope", 2: "Direct", 3: "Dynamic", 4: "Joint"}
CASE_TITLES = {
    1: "Case 1: Quarterly Joint Estimation with Capital IQ Competition Data",
    2: "Case 2: Quarterly Joint Estimation with Interpolated Gustavo Competition Data",
    3: "Case 3: Mixed Frequency Joint Estimation with Gustavo Competition Data",
    4: "Case 4: Mixed Frequency Joint Estimation with Gustavo Competition Data and Number of Establishments",
}
GREEK = {"intercept": r"$c$", "alpha": r"$\alpha$", "alpha_b": r"$\alpha_b$",
         "kappa_0": r"$\kappa_0$", "delta": r"$\delta$",
         "theta_0": r"$\theta_0$", "gamma": r"$\gamma$"}
COEFF_ORDER = ("intercept", "alpha", "alpha_b", "kappa_0", "delta", "theta_0", "gamma")
# In which models each coefficient is a free parameter.
COEFF_MODELS = {"intercept": (0, 1, 2, 3, 4), "alpha": (0, 1, 2, 3, 4),
                "alpha_b": (0, 1, 2, 3, 4), "kappa_0": (0, 1, 2, 3, 4),
                "delta": (1, 4), "theta_0": (2, 3, 4), "gamma": (3, 4)}


# --------------------------------------------------------------------------- #
# draw loading
# --------------------------------------------------------------------------- #
def _load_model(spec_dir: Path, model: int) -> dict | None:
    path = spec_dir / f"model{model}.npz"
    if not path.exists():
        return None
    z = np.load(path, allow_pickle=True)
    return {k: z[k] for k in z.files}


def _flat(draws: dict, name: str) -> np.ndarray | None:
    names = list(draws["coeff_names"])
    if name not in names:
        return None
    return draws["coeffs"][:, :, names.index(name)].reshape(-1)


def _primary_spec_dir(case_dir: Path) -> Path | None:
    for d in sorted(case_dir.iterdir()):
        if d.is_dir() and d.name.startswith("ppi__inverse_markup__"):
            return d
    dirs = [d for d in sorted(case_dir.iterdir()) if d.is_dir()]
    return dirs[0] if dirs else None


# --------------------------------------------------------------------------- #
# tables
# --------------------------------------------------------------------------- #
def _coeff_table(summary: pd.DataFrame, case: int, infl: str, forc: str, variant: str) -> str:
    sub = summary[(summary.case == case) & (summary.variant == variant)
                  & (summary.inflation == infl) & (summary.forcing == forc)]
    spec = sub.iloc[0]
    present = [p for p in COEFF_ORDER if not sub[sub.parameter == p].empty]
    rows = []
    for model in range(5):
        m = sub[sub.model == model]
        if m.empty:
            continue
        cells = [MODEL_LABELS[model]]
        for p in present:
            r = m[m.parameter == p]
            if r.empty:
                cells.append("--")
            else:
                r = r.iloc[0]
                cells.append(f"\\makecell{{{r['mean']:.3f}\\\\[-1pt]\\footnotesize[{r['ci_2.5']:.3f}, {r['ci_97.5']:.3f}]}}")
        rows.append(" & ".join(cells) + r" \\[3pt]")
    body = "\n".join(rows)
    header = "Model & " + " & ".join(GREEK[p] for p in present) + r" \\"
    return (
        "\\begin{table}[H]\\centering\\footnotesize\n"
        f"\\caption{{Case {case}: posterior mean and 95\\% credible interval across "
        f"Models 0--4 ({spec['inflation']}, {spec['forcing'].replace('_',' ')}).}}\n"
        f"\\label{{tab:case{case}_coeff}}\n"
        "\\renewcommand{\\arraystretch}{1.1}\\setlength{\\tabcolsep}{3pt}\n"
        f"\\begin{{tabular}}{{@{{}}l{'c' * len(present)}@{{}}}}\\toprule\n"
        f"{header}\\midrule\n"
        f"{body}\n"
        "\\bottomrule\\end{tabular}\\end{table}\n"
    )


def _p(v) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "--"
    return f"{float(v):.3f}"


def _ml(v) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "--"
    return f"{float(v):.1f}"


def _rhat(v) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "--"
    star = r"$^{\dagger}$" if float(v) > 1.1 else ""
    return f"{float(v):.2f}{star}"


def _modelcmp_table(cmp: pd.DataFrame, case: int, infl: str, forc: str, variant: str) -> str:
    sub = cmp[(cmp.case == case) & (cmp.variant == variant)
              & (cmp.inflation == infl) & (cmp.forcing == forc)].sort_values("model")
    rows = []
    for _, r in sub.iterrows():
        rows.append(
            f"{MODEL_LABELS[int(r.model)]} & {_rhat(r.get('rhat_max'))} & {r['waic']:.1f} & {_ml(r.get('log_ml'))} & "
            f"{_p(r.get('p_delta_pos'))} & {_p(r.get('p_theta0_pos'))} & "
            f"{_p(r.get('p_gamma_pos'))} & {_p(r.get('p_joint_hsa'))} \\\\"
        )
    body = "\n".join(rows)
    return (
        "\\begin{table}[H]\\centering\\small\n"
        f"\\caption{{Case {case}: model comparison by WAIC (lower is better) and log "
        "marginal likelihood $\\log m(y)$ (higher is better), with the posterior "
        "directional probabilities addressing RQ1--RQ4: $P(\\delta>0)$, $P(\\theta_0>0)$, "
        "$P(\\gamma>0)$, and the joint HSA probability $P(\\delta>0,\\theta_0>0,\\gamma>0)$. "
        "$\\hat R$ is the largest structural-coefficient convergence statistic; "
        "$\\dagger$ marks $\\hat R>1.1$ (not fully converged).}\n"
        f"\\label{{tab:case{case}_cmp}}\n"
        "\\begin{tabular}{lrrrrrrr}\\toprule\n"
        "Model & $\\hat R$ & WAIC & $\\log m(y)$ & $P(\\delta{>}0)$ & $P(\\theta_0{>}0)$ & $P(\\gamma{>}0)$ & $P(\\text{joint})$ \\\\\\midrule\n"
        f"{body}\n"
        "\\bottomrule\\end{tabular}\\end{table}\n"
    )


INFL_LABEL = {"ppi": "PPI", "cpi": "Headline CPI", "core_cpi": "Core CPI"}
FORC_LABEL = {"inverse_markup": "inv.\\ markup", "negative_unemployment_gap": "neg.\\ unemp.\\ gap",
              "bn_output_gap": "BN gap", "hp_output_gap": "HP gap"}


def _robustness_table(cmp: pd.DataFrame, case: int) -> str:
    """Joint-model (M4) directional probabilities across every estimated specification."""
    sub = cmp[(cmp.case == case) & (cmp.model == 4)].copy()
    sub = sub.sort_values(["variant", "inflation", "forcing"])
    rows = []
    for _, r in sub.iterrows():
        spec = f"{INFL_LABEL.get(r.inflation, r.inflation)} / {FORC_LABEL.get(r.forcing, r.forcing)}"
        if len(sub.variant.unique()) > 1:
            spec += f" ({r.variant.replace('_', ' ')})"
        rows.append(
            f"{spec} & {_rhat(r.get('rhat_max'))} & {_ml(r.get('log_ml'))} & {_p(r.get('p_delta_pos'))} & "
            f"{_p(r.get('p_theta0_pos'))} & {_p(r.get('p_gamma_pos'))} & {_p(r.get('p_joint_hsa'))} \\\\"
        )
    body = "\n".join(rows)
    return (
        "\\begin{table}[H]\\centering\\small\n"
        f"\\caption{{Case {case}: robustness of the joint HSA directional evidence (Model 4) "
        "across all estimated inflation/forcing specifications. $\\hat R$ is the largest "
        "structural-coefficient convergence statistic; $\\dagger$ marks $\\hat R>1.1$ "
        "(directional probabilities for those rows are not fully converged).}\n"
        f"\\label{{tab:case{case}_robust}}\n"
        "\\footnotesize\\setlength{\\tabcolsep}{4.5pt}\n"
        "\\begin{tabular}{@{}lrrrrrr@{}}\\toprule\n"
        "Specification & $\\hat R$ & $\\log m(y)$ & $P(\\delta{>}0)$ & $P(\\theta_0{>}0)$ & $P(\\gamma{>}0)$ & $P(\\text{joint})$ \\\\\\midrule\n"
        f"{body}\n"
        "\\bottomrule\\end{tabular}\\end{table}\n"
    )


# --------------------------------------------------------------------------- #
# figures
# --------------------------------------------------------------------------- #
def _forest(ax, entries, xlabel):
    for i, (lab, mean, lo, hi, color) in enumerate(entries):
        ax.errorbar(mean, i, xerr=[[mean - lo], [hi - mean]], fmt="o", color=color, capsize=3)
    ax.axvline(0, color="black", lw=1, ls="--")
    ax.set_yticks(range(len(entries)), [e[0] for e in entries])
    ax.set_ylim(-0.6, len(entries) - 0.4)
    ax.set_xlabel(xlabel)


def _coeff_entries(spec_dir: Path, name: str, color: str):
    entries = []
    for model in COEFF_MODELS[name]:
        d = _load_model(spec_dir, model)
        if d is None:
            continue
        f = _flat(d, name)
        if f is None:
            continue
        entries.append((f"M{model} ({MODEL_LABELS[model]})", f.mean(),
                        np.quantile(f, .025), np.quantile(f, .975), color))
    return entries


def _single_forest(spec_dir: Path, case: int, name: str, color: str) -> Path | None:
    entries = _coeff_entries(spec_dir, name, color)
    if not entries:
        return None
    fig, ax = plt.subplots(figsize=(6.6, 0.5 * len(entries) + 1.0))
    _forest(ax, entries, f"{GREEK[name]} posterior mean and 95% CI")
    ax.set_title(f"Case {case}: {GREEK[name]} across models")
    fig.tight_layout()
    path = OUT_FIGS / f"case{case}_{name}_forest.png"
    fig.savefig(path, dpi=200); plt.close(fig)
    return path


def _forest_plus_path(spec_dir: Path, case: int, coef: str, color: str,
                      path_kind: str) -> Path | None:
    """Left: forest of `coef`; right: time-varying kappa_t or theta_t (Model 4)."""
    d4 = _load_model(spec_dir, 4)
    entries = _coeff_entries(spec_dir, coef, color)
    if d4 is None or not entries:
        return None
    periods = pd.PeriodIndex(d4["periods"].astype(str), freq="Q").to_timestamp()
    ntilde = d4["ntilde"].reshape(-1, d4["ntilde"].shape[-1])
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.2, 3.4), gridspec_kw={"width_ratios": [1, 1.6]})
    _forest(axL, entries, f"{GREEK[coef]} (mean, 95% CI)")
    axL.set_title(f"{GREEK[coef]} across models")
    if path_kind == "kappa":
        series = _flat(d4, "kappa_0")[:, None] + (_flat(d4, "delta")[:, None] * ntilde)
        title, ylab = r"Time-varying slope $\kappa_t$ (Model 4)", r"$\kappa_t$"
    else:
        series = _flat(d4, "theta_0")[:, None] + (_flat(d4, "gamma")[:, None] * ntilde)
        title, ylab = r"Time-varying direct loading $\theta_t$ (Model 4)", r"$\theta_t$"
    axR.fill_between(periods, np.quantile(series, .025, 0), np.quantile(series, .975, 0), color=color, alpha=.18)
    axR.plot(periods, series.mean(0), color=color, lw=1.5)
    axR.axhline(0, color="black", lw=.7)
    axR.set_title(title); axR.set_ylabel(ylab); axR.set_xlabel("Quarter")
    fig.suptitle(f"Case {case}", y=1.02, fontsize=11)
    fig.tight_layout()
    path = OUT_FIGS / f"case{case}_{coef}_path.png"
    fig.savefig(path, dpi=200, bbox_inches="tight"); plt.close(fig)
    return path


def _predictive_fig(spec_dir: Path, case: int) -> Path | None:
    d = _load_model(spec_dir, 4)
    if d is None:
        return None
    periods = pd.PeriodIndex(d["periods"].astype(str), freq="Q").to_timestamp()
    pi, epi, x = d["pi"], d["epi"], d["x"]
    nt = d["ntilde"].reshape(-1, d["ntilde"].shape[-1])
    nh = d["nhat"].reshape(-1, d["nhat"].shape[-1])
    zero = np.zeros((nt.shape[0], 1))
    gf = lambda n: (_flat(d, n)[:, None] if n in list(d["coeff_names"]) else zero)  # noqa: E731
    a, k0 = gf("alpha"), gf("kappa_0")
    de, th, ga = gf("delta"), gf("theta_0"), gf("gamma")
    mu = a * epi[None, :] + k0 * x[None, :] + de * x[None, :] * nt - th * nh + ga * nt * nh
    if "intercept" in list(d["coeff_names"]):        # hybrid NKPC: add c + alpha_b * pi_{t-1}
        pi_lag = np.concatenate([[pi[0]], pi[:-1]])
        mu = mu + gf("intercept") + gf("alpha_b") * pi_lag[None, :]
    fig, ax = plt.subplots(figsize=(11.2, 3.6))
    ax.fill_between(periods, np.quantile(mu, .05, 0), np.quantile(mu, .95, 0), color=BLUE, alpha=.20, label="90% predictive band")
    ax.plot(periods, mu.mean(0), color=BLUE, lw=1.4, label="Posterior mean fit")
    ax.plot(periods, pi, color="black", lw=1.0, alpha=.8, label="Observed inflation")
    ax.set_title(f"Case {case}: posterior predictive inflation (Model 4)")
    ax.set_ylabel("Annualized inflation"); ax.set_xlabel("Quarter")
    ax.legend(frameon=False, ncol=3, fontsize=8)
    fig.tight_layout()
    path = OUT_FIGS / f"case{case}_predictive.png"
    fig.savefig(path, dpi=200); plt.close(fig)
    return path


def _decomposition_fig(spec_dir: Path, case: int) -> Path | None:
    d = _load_model(spec_dir, 4)
    if d is None:
        return None
    periods = pd.PeriodIndex(d["periods"].astype(str), freq="Q").to_timestamp()
    nt = d["ntilde"].reshape(-1, d["ntilde"].shape[-1])
    nh = d["nhat"].reshape(-1, d["nhat"].shape[-1])
    x = d["x"]
    de, th, ga = _flat(d, "delta")[:, None], _flat(d, "theta_0")[:, None], _flat(d, "gamma")[:, None]
    c_slope = (de * nt) * x[None, :]
    c_direct = -th * nh                # inflation contribution is -theta_0 * Nhat
    c_inter = ga * (nt * nh)
    fig, ax = plt.subplots(figsize=(11.2, 3.6))
    for series, color, lab in ((c_slope, BLUE, r"slope $\delta\tilde N_t x_t$"),
                               (c_direct, ORANGE, r"direct $-\theta_0\hat N_t$"),
                               (c_inter, GREEN, r"interaction $\gamma\tilde N_t\hat N_t$")):
        ax.plot(periods, series.mean(0), color=color, lw=1.4, label=lab)
    ax.axhline(0, color="black", lw=.7)
    ax.set_title(f"Case {case}: competition contribution to inflation (Model 4)")
    ax.set_ylabel("Annualized pp"); ax.set_xlabel("Quarter")
    ax.legend(frameon=False, ncol=3, fontsize=8)
    fig.tight_layout()
    path = OUT_FIGS / f"case{case}_decomposition.png"
    fig.savefig(path, dpi=200); plt.close(fig)
    return path


# --------------------------------------------------------------------------- #
# document assembly
# --------------------------------------------------------------------------- #
DATA_SPEC_SECTION = r"""\section{Data Specifications}

The empirical specifications vary along two dimensions: the
inflation--expectation pair and the measure of the Phillips-curve forcing
variable. Table~\ref{tab:data_specs} summarizes the alternatives considered
throughout the estimation.

\begin{table}[H]
\centering
\caption{Empirical Data Specifications}
\label{tab:data_specs}
\small
\renewcommand{\arraystretch}{1.3}
\begin{tabular}{@{}p{0.30\textwidth}p{0.55\textwidth}@{}}
\toprule
\multicolumn{2}{@{}l}{\textbf{Panel A. Inflation--Expectation Pairs}} \\
\textbf{Inflation Measure} & \textbf{Expected Inflation Measure} \\
\midrule
PPI & GDP price index inflation forecast \\
Headline CPI & CPI inflation forecast \\
Core CPI & CPI inflation forecast \\
\bottomrule
\end{tabular}

\vspace{1.2em}

\begin{tabular}{@{}p{0.34\textwidth}p{0.50\textwidth}@{}}
\toprule
\multicolumn{2}{@{}l}{\textbf{Panel B. Phillips-Curve Forcing Variables}} \\
\textbf{Measure} & \textbf{Interpretation} \\
\midrule
Inverse markup & Proxy for real marginal cost \\
Negative unemployment gap & Labor-market slack measure \\
Beveridge--Nelson output gap & BN-decomposition output gap \\
Hodrick--Prescott output gap & HP-filter output gap \\
\bottomrule
\end{tabular}
\end{table}

The results below report the primary specification (PPI, inverse markup) for
each case. The remaining inflation and forcing combinations enter the full
robustness grid.
"""

PREAMBLE = r"""\documentclass[11pt]{article}
\usepackage[margin=0.9in]{geometry}
\usepackage{amsmath,amssymb}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{makecell}
\usepackage{caption}
\usepackage{float}
\usepackage{xcolor}
\usepackage[hidelinks]{hyperref}
\usepackage{etoolbox}
\pretocmd{\section}{\clearpage}{}{}
\captionsetup{font=small}
\newcommand{\Nbar}{\bar N}
\newcommand{\Nhat}{\hat N}
\graphicspath{{../results/figures/}}
\title{\textbf{Market Concentration and the Flattening of the Phillips Curve}\\[2pt]
\large A Theory-Motivated, Semi-Structural State-Space Assessment of the HSA Mechanism\\[4pt]
\normalsize Estimation Results Summary}
\author{Satoshi Ichikawa}
\date{Draft --- August 2026}
\begin{document}
\maketitle
"""


def _fig(name: str, width: float = 0.9) -> str:
    return f"\\begin{{center}}\\includegraphics[width={width}\\linewidth]{{{name}}}\\end{{center}}\n"


def build(results_dir: Path) -> Path:
    summary = pd.read_csv(results_dir / "coefficient_summaries.csv")
    cmp = pd.read_csv(results_dir / "model_comparison.csv")
    # Merge the largest structural-coefficient R-hat per fit for convergence flagging.
    struct = summary[summary.parameter.isin(["alpha", "kappa_0", "delta", "theta_0", "gamma"])]
    conv = (struct.groupby(["case", "variant", "inflation", "forcing", "model"])
            .agg(rhat_max=("rhat", "max"), ess_min=("bulk_ess", "min")).reset_index())
    cmp = cmp.merge(conv, on=["case", "variant", "inflation", "forcing", "model"], how="left")
    OUT_TABLES.mkdir(parents=True, exist_ok=True)
    OUT_FIGS.mkdir(parents=True, exist_ok=True)

    hybrid = "alpha_b" in set(summary.parameter)
    body = []
    if hybrid:
        body.append("\\begin{center}\\textit{Hybrid NKPC specification: the inflation "
                    "equation adds an intercept $c$ and lagged inflation $\\alpha_b\\pi_{t-1}$ "
                    "to the forward-looking term $\\alpha E_t\\pi_{t+1}$.}\\end{center}\n")
    body.append(DATA_SPEC_SECTION)
    for case in (1, 2, 3, 4):
        case_dir = results_dir / f"case{case}"
        if not case_dir.exists():
            continue
        spec_dir = _primary_spec_dir(case_dir)
        infl, forc, variant = spec_dir.name.split("__")
        body.append(f"\\section{{{CASE_TITLES[case]}}}\n")

        # ---- Coefficient Table ----
        (OUT_TABLES / f"case{case}_coeff.tex").write_text(_coeff_table(summary, case, infl, forc, variant), encoding="utf-8")
        body.append("\\subsection{Coefficient Table}\n")
        body.append(f"\\input{{../results/tables/case{case}_coeff.tex}}\n")

        # ---- Marginal Likelihoods ----
        (OUT_TABLES / f"case{case}_cmp.tex").write_text(_modelcmp_table(cmp, case, infl, forc, variant), encoding="utf-8")
        body.append("\\subsubsection{Marginal Likelihoods}\n")
        body.append("Models are compared by the log marginal likelihood $\\log m(y)$ "
                    "(higher is better; Laplace--Metropolis estimator with the exact Kalman "
                    "integrated likelihood for the linear Models 0--2 and a particle-filter "
                    "likelihood for the bilinear Models 3--4) and by the Widely Applicable "
                    "Information Criterion (WAIC; lower is better). The directional "
                    "probabilities in the same table summarize the RQ1--RQ4 posterior evidence.\n")
        body.append(f"\\input{{../results/tables/case{case}_cmp.tex}}\n")

        # ---- Predictive Distributions ----
        pred = _predictive_fig(spec_dir, case)
        body.append("\\subsubsection{Predictive Distributions}\n")
        if pred:
            body.append(_fig(f"case{case}_predictive.png"))

        # ---- Coefficient: alpha ----
        a = _single_forest(spec_dir, case, "alpha", GREY)
        body.append("\\subsection{Coefficient: $\\alpha$}\n")
        if a:
            body.append(_fig(f"case{case}_alpha_forest.png", 0.6))

        # ---- Coefficient: kappa ----
        k = _single_forest(spec_dir, case, "kappa_0", BLUE)
        body.append("\\subsection{Coefficient: $\\kappa$}\n")
        if k:
            body.append(_fig(f"case{case}_kappa_0_forest.png", 0.6))
        # delta and time-varying kappa_t
        dk = _forest_plus_path(spec_dir, case, "delta", BLUE, "kappa")
        body.append("\\subsubsection{Coefficient: $\\delta$ and Time-Varying $\\kappa_t$}\n")
        if dk:
            body.append(_fig(f"case{case}_delta_path.png", 0.98))

        # ---- Coefficient: theta ----
        th = _single_forest(spec_dir, case, "theta_0", ORANGE)
        body.append("\\subsection{Coefficient: $\\theta$}\n")
        if th:
            body.append(_fig(f"case{case}_theta_0_forest.png", 0.6))
        # gamma and time-varying theta_t
        gt = _forest_plus_path(spec_dir, case, "gamma", ORANGE, "theta")
        body.append("\\subsubsection{Coefficient: $\\gamma$ and Time-Varying $\\theta_t$}\n")
        if gt:
            body.append(_fig(f"case{case}_gamma_path.png", 0.98))

        # ---- Decomposed Competition ----
        dec = _decomposition_fig(spec_dir, case)
        body.append("\\subsubsection{Decomposed Competition}\n")
        if dec:
            body.append(_fig(f"case{case}_decomposition.png"))

        # ---- Robustness across specifications ----
        (OUT_TABLES / f"case{case}_robust.tex").write_text(_robustness_table(cmp, case), encoding="utf-8")
        body.append("\\subsection{Robustness Across Specifications}\n")
        body.append("The detailed results above use the primary specification "
                    "(PPI, inverse markup). Table~\\ref{tab:case" f"{case}"
                    "_robust} summarizes the joint-model directional evidence across "
                    "every estimated inflation and forcing combination.\n")
        body.append(f"\\input{{../results/tables/case{case}_robust.tex}}\n")

    tex = PREAMBLE + "\n".join(body) + "\n\\end{document}\n"
    out_tex = REPORT_DIR / "summary_results.tex"
    out_tex.write_text(tex, encoding="utf-8")
    completed = subprocess.run(
        ["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error", out_tex.name],
        cwd=REPORT_DIR, check=False, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    if completed.returncode:
        raise RuntimeError(f"LaTeX build failed:\n{completed.stdout[-4000:]}")
    print(f"wrote {out_tex.with_suffix('.pdf')}", flush=True)
    return out_tex.with_suffix(".pdf")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-dir", type=Path,
                    default=results_root() / "report_estimation" / "pilot")
    args = ap.parse_args()
    build(args.results_dir)


if __name__ == "__main__":
    main()
