"""Build the PDF report for the full estimated-lambda HSA experiment."""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm

import sys as _sys, pathlib as _pathlib
_ROOT = next(p for p in _pathlib.Path(__file__).resolve().parents if (p / "pyproject.toml").exists())
_sys.path[:0] = [str(_ROOT), str(_ROOT / "src"), str(_ROOT / "tests")]
from tests import _bootstrap  # noqa: F401,E402
from nkpc_hsa.config import load_yaml  # noqa: E402
from tests.hsa_lambda_dynamic.functions import (  # noqa: E402
    MODEL_LABELS, derived_paths, load_experiment_data, load_fit,
)

BUNDLE = Path(__file__).resolve().parent
BLUE, ORANGE, GREEN, PURPLE, GREY = "#0072B2", "#D55E00", "#009E73", "#CC79A7", "#666666"


def esc(value):
    return str(value).replace("_", r"\_").replace("%", r"\%")


def band(values):
    return values.mean(axis=0), np.percentile(values, 2.5, axis=0), np.percentile(values, 97.5, axis=0)


def make_allocation_figure(exp, manifest, figures):
    raw = {int(k): np.asarray(v) for k, v in manifest["allocation"]["raw_weights"].items()}
    used = {int(k): np.asarray(v) for k, v in manifest["allocation"]["used_weights"].items()}
    avg = np.asarray(manifest["allocation"]["average_weights"])
    years = sorted(raw)
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.2))
    for q in range(4):
        axes[0].plot(years, [raw[y][q] for y in years], marker="o", ms=2.8, lw=0.9, alpha=0.65,
                     label=f"Q{q+1} raw")
        axes[1].plot(years, [used[y][q] for y in years], marker="o", ms=2.8, lw=1.0,
                     label=f"Q{q+1} used")
        axes[1].axhline(avg[q], color=[BLUE, ORANGE, GREEN, PURPLE][q], ls=":", lw=0.9)
    axes[0].axhline(0, color="black", lw=0.6); axes[1].axhline(0, color="black", lw=0.6)
    axes[0].set_title("Raw Capital IQ annual ratios")
    axes[1].set_title("Coherence-shrunk ratios used in allocation")
    axes[0].set_ylabel("share of annual change")
    for ax in axes:
        ax.set_xlabel("year"); ax.legend(frameon=False, fontsize=7, ncol=2)
    fig.suptitle("Quarterly allocation diagnostics (dotted lines: robust missing-year profile)")
    fig.tight_layout(); fig.savefig(figures / "allocation_weights.png", dpi=200); plt.close(fig)


def make_state_figure(exp, fit, figures):
    periods = pd.PeriodIndex(fit.periods, freq="Q").to_timestamp()
    nbar = fit.nbar.reshape(-1, fit.nbar.shape[-1]); nhat = fit.nhat.reshape(-1, fit.nhat.shape[-1])
    nb, nbl, nbh = band(nbar); nh, nhl, nhh = band(nhat)
    observed = exp.case.n_obs
    fig, axes = plt.subplots(2, 1, figsize=(11.2, 6.7), sharex=True)
    axes[0].plot(periods, observed, color=GREY, lw=0.9, alpha=0.75, label=r"observed $N_t$")
    axes[0].plot(periods, nb, color=GREEN, lw=2.0, label=r"slow state $\bar N_t$")
    axes[0].fill_between(periods, nbl, nbh, color=GREEN, alpha=0.16)
    axes[0].plot(periods, nb + nh, color=BLUE, lw=1.0, ls="--", label=r"$\bar N_t+\hat N_t$")
    axes[0].axhline(0, color="black", lw=0.5); axes[0].legend(frameon=False, fontsize=8, ncol=3)
    axes[0].set_ylabel("ten-log-point units")
    axes[1].plot(periods, nh, color=BLUE, lw=1.4, label=r"cyclical state $\hat N_t$")
    axes[1].fill_between(periods, nhl, nhh, color=BLUE, alpha=0.16)
    axes[1].axhline(0, color="black", lw=0.6); axes[1].set_ylabel("ten-log-point units")
    axes[1].set_xlabel("year"); axes[1].legend(frameon=False, fontsize=8)
    fig.suptitle("Joint state-space decomposition under HSA-restricted dynamic")
    fig.tight_layout(); fig.savefig(figures / "n_decomposition.png", dpi=200); plt.close(fig)


def make_tvp_figure(data, fits, figures):
    periods = pd.PeriodIndex(fits["hsa_dynamic"].periods, freq="Q").to_timestamp()
    fig, axes = plt.subplots(2, 1, figsize=(11.2, 7.0), sharex=True)
    for model, color, style in (("hsa_static", BLUE, "--"), ("hsa_dynamic", ORANGE, "-")):
        kappa, theta = derived_paths(fits[model], data)
        for ax, values in zip(axes, (kappa, theta)):
            mean, low, high = band(values)
            ax.plot(periods, mean, color=color, ls=style, lw=2.0, label=MODEL_LABELS[model])
            ax.fill_between(periods, low, high, color=color, alpha=0.13)
    axes[0].axhline(0, color="black", lw=0.6); axes[1].axhline(0, color="black", lw=0.6)
    axes[0].set_ylabel(r"$\kappa_t$"); axes[1].set_ylabel(r"$\theta_t$")
    axes[1].set_xlabel("year"); axes[0].legend(frameon=False, fontsize=8, ncol=2)
    fig.suptitle(r"Time-varying $\kappa_t$ and $\theta_t$: static and dynamic HSA restrictions")
    fig.tight_layout(); fig.savefig(figures / "kappa_theta_paths.png", dpi=200); plt.close(fig)


def make_prior_posterior_figure(fit, figures):
    flat = fit.draws.reshape(-1, fit.draws.shape[-1])
    params = ["kappa_0", "theta_0", "gamma", "lambda"]
    labels = {"kappa_0": r"$\kappa_0$", "theta_0": r"$\theta_0$", "gamma": r"$\gamma$", "lambda": r"$\lambda$"}
    fig, axes = plt.subplots(2, 2, figsize=(9.5, 6.5))
    for ax, name in zip(axes.ravel(), params):
        values = flat[:, fit.names.index(name)]
        pm, ps = fit.prior_mean[name], fit.prior_sd[name]
        lo = min(np.percentile(values, 0.5), pm - 3.5 * ps)
        hi = max(np.percentile(values, 99.5), pm + 3.5 * ps)
        xs = np.linspace(lo, hi, 400)
        ax.plot(xs, norm.pdf(xs, pm, ps), color=ORANGE, lw=2.0, label="prior")
        ax.hist(values, bins=45, density=True, color=BLUE, alpha=0.62, label="posterior")
        ax.axvline(0, color="black", lw=0.7, ls="--")
        ax.axvline(values.mean(), color=GREEN, lw=1.4)
        ax.set_title(f"{labels[name]}  P(>0)={np.mean(values > 0):.2f}")
        ax.set_yticks([])
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.suptitle("Prior versus posterior: HSA-restricted dynamic")
    fig.tight_layout(); fig.savefig(figures / "prior_posterior.png", dpi=200); plt.close(fig)


def make_diagnostics_figure(manifest, figures):
    models = list(manifest["results"])
    values = [manifest["results"][m]["diagnostics"]["max_rhat"] for m in models]
    fig, ax = plt.subplots(figsize=(10.5, 3.5))
    colors = [GREEN if v <= 1.05 else ORANGE for v in values]
    ax.bar(range(len(models)), values, color=colors)
    ax.axhline(1.05, color="black", lw=1.0, ls="--", label="gate 1.05")
    ax.set_xticks(range(len(models)), [MODEL_LABELS[m].replace(" combined", "\ncombined") for m in models], fontsize=7)
    ax.set_ylim(0.99, max(1.08, max(values) * 1.01)); ax.set_ylabel("maximum rank R-hat")
    ax.legend(frameon=False, fontsize=8); ax.set_title("MCMC convergence gate by specification")
    fig.tight_layout(); fig.savefig(figures / "convergence.png", dpi=200); plt.close(fig)


def coefficient_rows(manifest):
    wanted = {
        "ces": ["kappa_0"], "slope": ["kappa_0", "delta"],
        "direct": ["kappa_0", "theta_0"], "free_static": ["kappa_0", "delta", "theta_0"],
        "hsa_static": ["kappa_0", "theta_0", "lambda", "delta_derived"],
        "free_dynamic": ["kappa_0", "delta_1", "delta_2", "theta_0", "gamma"],
        "hsa_dynamic": ["kappa_0", "theta_0", "gamma", "lambda", "delta_1_derived", "delta_2_derived"],
    }
    display = {
        "kappa_0": r"$\kappa_0$", "delta": r"$\delta$", "delta_1": r"$\delta_1$",
        "delta_2": r"$\delta_2$", "theta_0": r"$\theta_0$", "gamma": r"$\gamma$",
        "lambda": r"$\lambda$", "delta_derived": r"$\lambda\theta$",
        "delta_1_derived": r"$\lambda\theta_0$", "delta_2_derived": r"$\lambda\gamma/2$",
    }
    rows = []
    for model in manifest["results"]:
        rows.append(rf"\multicolumn{{6}}{{l}}{{\emph{{{esc(MODEL_LABELS[model])}}}}}\\")
        coefficients = manifest["results"][model]["coefficients"]
        for name in wanted[model]:
            value = coefficients[name]
            rhat = "--" if value["rhat"] is None else f"{value['rhat']:.3f}"
            rows.append(
                f"{display[name]} & {value['mean']:+.3f} & [{value['q2.5']:+.3f}, {value['q97.5']:+.3f}] "
                f"& {value['p_positive']:.2f} & {rhat} & {value['sd']:.3f} \\\\"
            )
        rows.append(r"\midrule")
    return "\n".join(rows[:-1])


def comparison_rows(manifest):
    rows = []
    for model, result in manifest["results"].items():
        m = result["metrics"]; rhat = result["diagnostics"]["max_rhat"]
        flag = "" if rhat <= 1.05 else r"$^{\dagger}$"
        rows.append(
            f"{esc(MODEL_LABELS[model])}{flag} & {m['waic']:.1f} & {m['log_marginal_laplace_pf']:.1f} "
            f"& {m['predictive_rmse']:.3f} & {m['predictive_coverage_95']:.2f} & {rhat:.3f} \\\\"
        )
    return "\n".join(rows)


def main():
    cfg = load_yaml(BUNDLE / "config.yaml")
    out = BUNDLE / "results" / "full"
    manifest_path = out / "manifest.json"
    if not manifest_path.exists():
        raise SystemExit("Full results missing. Run: python tests/hsa_lambda_dynamic/run.py")
    manifest = json.loads(manifest_path.read_text())
    exp = load_experiment_data(cfg)
    fits = {m: load_fit(out / "draws" / f"{m}.npz", manifest["results"][m]["diagnostics"])
            for m in manifest["results"]}
    figures = out / "figures"; figures.mkdir(parents=True, exist_ok=True)
    make_allocation_figure(exp, manifest, figures)
    make_state_figure(exp, fits["hsa_dynamic"], figures)
    make_tvp_figure(exp.case, fits, figures)
    make_prior_posterior_figure(fits["hsa_dynamic"], figures)
    make_diagnostics_figure(manifest, figures)

    w = manifest["allocation"]["average_weights"]
    gate = "PASS" if manifest["gate"]["passed"] else "FAIL"
    best_waic = min(manifest["results"], key=lambda m: manifest["results"][m]["metrics"]["waic"])
    best_ml = max(manifest["results"], key=lambda m: manifest["results"][m]["metrics"]["log_marginal_laplace_pf"])
    tex = rf"""\documentclass[10.5pt]{{article}}
\usepackage[margin=0.72in]{{geometry}}
\usepackage{{booktabs,graphicx,amsmath,xcolor,microtype,hyperref,newtxtext,newtxmath}}
\definecolor{{navy}}{{HTML}}{{17365D}}\hypersetup{{colorlinks=true,linkcolor=navy,urlcolor=navy}}
\setlength{{\parindent}}{{0pt}}\setlength{{\parskip}}{{4pt}}
\begin{{document}}
\begin{{center}}
{{\color{{navy}}\LARGE\bfseries HSA NKPC with Estimated $\lambda$}}\\[3pt]
{{\large Gustavo $\times$ Capital IQ competition, PPI / negative unemployment gap}}
\end{{center}}

\section*{{1. Data and quarterly competition allocation}}
Annual Gustavo effective-firm counts provide the exact Q4 benchmarks. When Capital IQ is observed,
its year-specific quarterly change ratios allocate the annual Gustavo change. A cancellation diagnostic
$c_t=|\sum_q\Delta CIQ_{{tq}}|/\sum_q|\Delta CIQ_{{tq}}|$ shrinks unstable ratios continuously toward a
robust missing-year profile. Missing years use the componentwise-median stable profile
$\bar w=[{w[0]:.3f},{w[1]:.3f},{w[2]:.3f},{w[3]:.3f}]$. The largest Q4 benchmark error is
{manifest['allocation']['max_anchor_error']:.1e}. This replaces the pooled profile that put roughly 95\%
of the annual change in Q1 and prevents near-zero annual Capital IQ changes from generating explosive ratios.
\begin{{center}}\includegraphics[width=0.98\linewidth]{{figures/allocation_weights.png}}\end{{center}}

The centered quarterly series is decomposed jointly with inflation,
$N_t=\bar N_t+\hat N_t$, with a random-walk slow state and stationary AR(1) cycle. The sample is
{esc(manifest['sample']['first'])}--{esc(manifest['sample']['last'])}, $n={manifest['sample']['n']}$. The common equation is
\[
\pi_t=a+\alpha_b\pi_{{t-1}}+\alpha_f E_t\pi_{{t+1}}+\kappa_t x_t-\theta_t\hat N_t+\varepsilon_t,
\qquad \varepsilon_t=\phi\varepsilon_{{t-1}}+u_t.
\]
There is no standalone competition-level control $\psi N_t$.
To keep the cyclical state distinct from the random-walk slow state, its persistence is restricted to
${cfg['priors']['rho_lower']:.2f}\le\rho\le{cfg['priors']['rho_upper']:.2f}$.

\section*{{2. Specifications and estimated HSA multiplier}}
The static HSA model uses $\kappa_t=\kappa_0+\lambda\theta\bar N_t$ and $\theta_t=\theta$.
The dynamic HSA model estimates
\[
\theta_t=\theta_0+\gamma\bar N_t,\qquad
\kappa_t=\kappa_0+\lambda\theta_0\bar N_t+\frac{{\lambda\gamma}}{{2}}\bar N_t^2,
\]
which imposes $d\kappa(N)/dN=\lambda\theta(N)$. The multiplier $\lambda$ has a sign-unrestricted
$N(0,{cfg['priors']['lambda_sd']:.0f}^2)$ prior; it is not fixed at six. Free static and dynamic counterparts
estimate the slope and direct coefficients independently. Because a free $\lambda$ makes
$\delta=\lambda\theta$ a reparameterization in the static case, the substantive restriction is tested most
sharply by the dynamic coefficient relation $\delta_1=\lambda\theta_0$ and
$\delta_2=\lambda\gamma/2$.

\section*{{3. Coefficient posteriors}}
\begin{{center}}\scriptsize
\begin{{tabular}}{{l r c c c r}}
\toprule Parameter & Mean & 95\% interval & P($>$0) & $\widehat R$ & SD\\\midrule
{coefficient_rows(manifest)}
\bottomrule
\end{{tabular}}
\end{{center}}

\section*{{4. Model comparison and prediction}}
Lower WAIC and RMSE are better; higher log marginal likelihood is better. The marginal likelihood is a
Laplace--Metropolis approximation using a bootstrap particle-filter integrated likelihood. A dagger marks
a specification that fails the predeclared $\widehat R\le1.05$ gate.
\begin{{center}}\small
\begin{{tabular}}{{l r r r r r}}
\toprule Specification & WAIC & log ML & RMSE & 95\% coverage & max $\widehat R$\\\midrule
{comparison_rows(manifest)}
\bottomrule
\end{{tabular}}
\end{{center}}
Best WAIC: \emph{{{esc(MODEL_LABELS[best_waic])}}}. Best approximate marginal likelihood:
\emph{{{esc(MODEL_LABELS[best_ml])}}}. Overall convergence gate: \textbf{{{gate}}}.
Because the overall gate fails, these rankings are diagnostic and are not promoted as substantive evidence.
\begin{{center}}\includegraphics[width=0.95\linewidth]{{figures/convergence.png}}\end{{center}}

\section*{{5. Prior versus posterior}}
\begin{{center}}\includegraphics[width=0.92\linewidth]{{figures/prior_posterior.png}}\end{{center}}

\section*{{6. Time-varying $\kappa_t$ and $\theta_t$}}
\begin{{center}}\includegraphics[width=\linewidth]{{figures/kappa_theta_paths.png}}\end{{center}}

\section*{{7. Jointly estimated competition states}}
\begin{{center}}\includegraphics[width=\linewidth]{{figures/n_decomposition.png}}\end{{center}}

\end{{document}}
"""
    tex_path = out / "hsa_lambda_dynamic_report.tex"
    tex_path.write_text(tex, encoding="utf-8")
    result = subprocess.run(
        ["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error", tex_path.name],
        cwd=out, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    if result.returncode:
        raise RuntimeError("LaTeX failed:\n" + result.stdout[-5000:])
    final_dir = _ROOT / "output" / "pdf"; final_dir.mkdir(parents=True, exist_ok=True)
    final_path = final_dir / "hsa_lambda_dynamic_report.pdf"
    shutil.copy2(tex_path.with_suffix(".pdf"), final_path)
    print("wrote", tex_path.with_suffix(".pdf"))
    print("copied", final_path)


if __name__ == "__main__":
    main()
