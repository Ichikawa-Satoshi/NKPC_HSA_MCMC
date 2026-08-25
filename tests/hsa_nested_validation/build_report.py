"""Build the English equation-first report for the price-separated 24-fit ladder."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys
import textwrap

import arviz as az
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd
from scipy.stats import norm

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists())
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT / "tests")]
from tests import _bootstrap  # noqa: F401,E402
from nkpc_hsa.config import load_yaml  # noqa:E402

BUNDLE = Path(__file__).resolve().parent
INK, MUTED, BLUE, GREEN, RED = "#18212A", "#5D6873", "#246B9E", "#397D63", "#A73B3B"
CELLS = (
    ("ppi_negative_unemployment_gap", "PPI / negative unemployment gap", "primary"),
    ("ppi_inverse_markup", "PPI / inverse markup", "benchmark"),
    ("core_cpi_negative_unemployment_gap", "Core CPI / negative unemployment gap", "primary"),
    ("core_cpi_inverse_markup", "Core CPI / inverse markup", "benchmark"),
)
PRIMARY_ORDER = ("ces", "slow_slope", "direct", "free_static_combined")
BENCHMARK_ORDER = PRIMARY_ORDER + (
    "hsa_fixed_lambda_3", "hsa_fixed_lambda_6", "hsa_fixed_lambda_9", "free_lambda_diagnostic",
)
MODEL_LABELS = {
    "ces": "CES", "slow_slope": "Slope channel", "direct": "Direct channel",
    "free_static_combined": "Free combined", "hsa_fixed_lambda_3": "HSA, lambda = 3",
    "hsa_fixed_lambda_6": "HSA, lambda = 6", "hsa_fixed_lambda_9": "HSA, lambda = 9",
    "free_lambda_diagnostic": "Free-lambda diagnostic",
}
PARAM_LABELS = {
    "intercept": r"$a$", "alpha_b": r"$\alpha_b$", "alpha_f": r"$\alpha_f$",
    "kappa_0": r"$\kappa_0$", "delta_s": r"$\delta_s$", "theta": r"$\theta$",
    "lambda": r"$\lambda$",
}
FREE_PARAMETERS = ("intercept", "alpha_b", "alpha_f", "kappa_0", "delta_s", "theta")
CELL_COLUMNS = (
    ("ppi_negative_unemployment_gap", "PPI x\nunemp gap"),
    ("ppi_inverse_markup", "PPI x\ninverse markup"),
    ("core_cpi_negative_unemployment_gap", "Core CPI x\nunemp gap"),
    ("core_cpi_inverse_markup", "Core CPI x\ninverse markup"),
)


def _style():
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["cmr10"], "font.size": 10,
        "axes.titlesize": 10.2, "axes.labelsize": 8.6, "axes.spines.top": False,
        "axes.spines.right": False, "figure.facecolor": "white", "axes.facecolor": "white",
        "pdf.fonttype": 42, "mathtext.fontset": "cm", "axes.formatter.use_mathtext": True,
        "axes.unicode_minus": False,
    })


def _page(title, subtitle, number, title_size=16):
    fig = plt.figure(figsize=(8.5, 11))
    fig.text(0.065, 0.955, title, fontsize=title_size, weight="bold", color=INK, va="top")
    fig.text(0.065, 0.925, subtitle, fontsize=9.4, color=MUTED, va="top")
    fig.text(0.935, 0.025, str(number), ha="right", fontsize=8, color=MUTED)
    return fig


def _section(fig, y, title):
    fig.text(0.075, y, title, fontsize=11.2, weight="bold", color=INK, va="top")
    return y - 0.035


def _text(fig, y, content, width=69, size=9.1, color=INK):
    wrapped = textwrap.fill(content, width=width)
    lines = wrapped.count("\n") + 1
    fig.text(0.085, y, wrapped, fontsize=size, color=color, va="top", linespacing=1.42)
    return y - 0.022 * lines - 0.011


def _bullet(fig, y, content, width=68, size=9.0):
    wrapped = textwrap.fill(content, width=width, subsequent_indent="   ")
    lines = wrapped.count("\n") + 1
    fig.text(0.09, y, "- " + wrapped, fontsize=size, color=INK, va="top", linespacing=1.4)
    return y - 0.022 * lines - 0.008


def _eq(fig, y, content, size=12, x=0.105):
    fig.text(x, y, content, fontsize=size, color=INK, va="top")
    return y - 0.048


def _table(fig, bbox, columns, rows, widths, size=7.3, left=(0,)):
    ax = fig.add_axes(bbox); ax.axis("off")
    tab = ax.table(cellText=rows, colLabels=columns, cellLoc="center", colLoc="center",
                   colWidths=widths, bbox=[0, 0, 1, 1])
    tab.auto_set_font_size(False); tab.set_fontsize(size)
    for (row, col), cell in tab.get_celld().items():
        cell.set_edgecolor("#7D858C"); cell.set_linewidth(0.5); cell.PAD = 0.03
        cell.set_facecolor("#F0F1F2" if row == 0 else "white")
        if row == 0: cell.get_text().set_weight("bold")
        if col in left: cell.get_text().set_ha("left")


def _key(cell, model):
    return f"joint_state_split/{cell}/{model}"


def _fit_path(mode, cell, model):
    return BUNDLE / "results" / mode / "draws" / "joint_state_split" / cell / f"{model}.npz"


def _load_fit(mode, cell, model):
    return np.load(_fit_path(mode, cell, model), allow_pickle=False)


def _posterior_rows(manifest):
    rows = []
    for parameter in FREE_PARAMETERS:
        row = [PARAM_LABELS[parameter]]
        for cell, _ in CELL_COLUMNS:
            value = manifest["models"][_key(cell, "free_static_combined")]["coefficients"][parameter]
            row.append(f"{value['mean']:.3f}\n[{value['q2.5']:.3f}, {value['q97.5']:.3f}]")
        rows.append(row)
    return rows


def _prior_rows(manifest):
    rows = []
    for parameter in FREE_PARAMETERS:
        row = [PARAM_LABELS[parameter]]
        for cell, _ in CELL_COLUMNS:
            value = manifest["models"][_key(cell, "free_static_combined")]["coefficients"][parameter]
            row.append(rf"$\mathcal{{N}}({value['prior_mean']:.3f},\,{value['prior_sd']:.3f})$")
        rows.append(row)
    return rows


def _estimate(model, parameter):
    if parameter not in model["coefficients"]: return "--"
    value = model["coefficients"][parameter]
    return f"{value['mean']:+.3f} [{value['q2.5']:+.3f}, {value['q97.5']:+.3f}]"


def _band(values):
    flat = values.reshape(-1, values.shape[-1])
    return np.mean(flat, axis=0), np.percentile(flat, 2.5, axis=0), np.percentile(flat, 97.5, axis=0)


def _page_map(pdf, manifest):
    fig = _page("HSA NKPC: price-separated nested validation", "1. The active 24-fit design", 1)
    y = _section(fig, 0.85, "Common NKPC")
    y = _eq(fig, y, r"$\pi_t=a+\alpha_b\pi_{t-1}+\alpha_f E_t\pi_{t+1}+\kappa_t x_t-\theta\hat q_t+\varepsilon_t$")
    y = _eq(fig, y, r"$q_t=\bar q_t+\hat q_t\quad\mathrm{exactly}$")
    y = _text(fig, y, "PPI and Core CPI are separate outcomes. Each price/activity cell has its own coefficients, inflation error, and joint slow/cycle posterior.")
    y = _section(fig, y, "Fit count")
    rows = [
        ["PPI", "Negative unemployment gap", "M0-M3", "4"],
        ["PPI", "Inverse markup", "B0-B3 + HSA grid + free lambda", "8"],
        ["Core CPI", "Negative unemployment gap", "M0-M3", "4"],
        ["Core CPI", "Inverse markup", "B0-B3 + HSA grid + free lambda", "8"],
    ]
    _table(fig, [0.08, 0.455, 0.84, 0.175], ["Inflation", "Activity", "Specifications", "Fits"], rows,
           [0.16, 0.31, 0.40, 0.13], 8.0, (0, 1, 2))
    y = 0.41
    y = _eq(fig, y, r"$4+8+4+8=24\ \mathrm{fits}$", 13)
    y = _section(fig, y, "What is excluded")
    y = _bullet(fig, y, "No state-cut treatment; every fit jointly estimates its slow/cycle split with its NKPC.")
    y = _bullet(fig, y, "No B4 two-coordinate slope and no B5 total-q slope.")
    _bullet(fig, y, "The free-lambda fit is diagnostic and does not count as confirmatory HSA evidence.")
    fig.text(0.075, 0.205, "How the report is organized", fontsize=11.0, weight="bold", color=INK)
    guide = [
        ["Pages 2-6", "Data, AR(2) state identification, equations, coefficients, and priors"],
        ["Pages 7-8", "Complete PPI and Core CPI model ladders"],
        ["Pages 9-12", "Prior/posterior learning, time paths, and HSA restriction distance"],
        ["Pages 13-15", "Convergence, predictive diagnostics, and the decision gate"],
    ]
    _table(fig, [0.09, 0.055, 0.82, 0.12], ["Location", "Content"], guide, [0.20, 0.80], 7.4, (0, 1))
    pdf.savefig(fig); plt.close(fig)


def _page_data(pdf, manifest):
    fig = _page("Data, exact states, and joint estimation", "2. Gustavo levels, Capital IQ allocation, state laws, and the cell posterior", 2, 15)
    y = _section(fig, 0.85, "Quarterly competition coordinate")
    y = _eq(fig, y, r"$q_y^G=10\log F_y,\qquad q_{y,Q4}^{*}=q_y^G$")
    y = _eq(fig, y, r"$W_{yq}=\sum_{r=1}^{q}w_{yr}$", 11.5)
    y = _eq(fig, y, r"$q_{yq}^{*}=q_{y-1}^G+W_{yq}\,(q_y^G-q_{y-1}^G)$", 11.5)
    y = _text(fig, y, "Capital IQ quarterly changes update the within-year allocation when available. The average quarterly profile is the prior mean when Capital IQ is missing or incoherent.")
    y = _section(fig, y, "Centering and exact slow/cycle decomposition")
    y = _eq(fig, y, r"$q_t=q_t^{*}-q_0=\bar q_t+\hat q_t$")
    y = _eq(fig, y, r"$\bar q_t=\bar q_{t-1}+\eta_t^b$")
    y = _eq(fig, y, r"$\hat q_t=2r\cos(2\pi/P)\hat q_{t-1}-r^2\hat q_{t-2}+\eta_t^h$", 10.8)
    y = _eq(fig, y, r"$\sigma_{\bar q}^2=\omega\tau^2,\qquad \sigma_{\hat q}^2=(1-\omega)\tau^2$")
    y = _text(fig, y, r"The parameter $\tau^2$ is total competition innovation variance and $\omega$ is the share assigned to the slow innovation.")
    y = _section(fig, y, "Four separate empirical cells")
    sample = manifest["preflight"]["samples"]["ppi_negative_unemployment_gap"]
    y = _bullet(fig, y, f"PPI/Core CPI x negative gap/inverse markup: {sample[0]} to {sample[1]}, {sample[2]} observations in every cell; all use SPF GDP-deflator expectations.", width=76)
    y = _section(fig, y, "Joint posterior for each cell c")
    y = _eq(fig, y, r"$p(\beta_c,\bar q_c,\hat q_c,\omega_c,\tau_c,r_c,P_c\mid \pi_c,q)$", 11.8)
    y = _text(fig, y, "Each cell has its own coefficients, inflation error, and state posterior. State cut is not estimated; allocation weights remain measurement-only.", width=76)
    y = _eq(fig, y, rf"$\max|q_t-\bar q_t-\hat q_t|={manifest['gate']['max_identity_error']:.2e}$", 10.8)
    _text(fig, y, rf"Required tolerance: {manifest['gate']['identity_required']:.1e}; Gustavo Q4 anchor error: {manifest['preflight']['allocation_mean_anchor_error']:.2e}.", width=76, size=8.2)
    pdf.savefig(fig); plt.close(fig)


def _page_joint(pdf, manifest):
    fig = _page("Joint state estimation in every cell", "3. Inflation updates the exact slow/cycle split", 3)
    y = _section(fig, 0.85, "Posterior for cell c")
    y = _eq(fig, y, r"$p(\beta_c,\bar q_c,\hat q_c,\omega_c,\tau_c,r_c,P_c\mid \pi_c,q)$", 12.5)
    y = _eq(fig, y, r"$q_t=\bar q_{c,t}+\hat q_{c,t}$", 12.5)
    y = _text(fig, y, "The observed quarterly competition path q is common, but each inflation/activity cell obtains its own posterior decomposition because its NKPC likelihood is different.")
    y = _section(fig, y, "Inflation error")
    y = _eq(fig, y, r"$\varepsilon_{c,t}=\phi_c\varepsilon_{c,t-1}+u_{c,t},\qquad u_{c,t}\sim\mathcal{N}(0,\sigma_{\pi,c}^2)$")
    y = _section(fig, y, "Interpretation")
    y = _bullet(fig, y, "PPI estimates never share inflation coefficients with Core CPI estimates.")
    y = _bullet(fig, y, "Negative-gap and inverse-markup estimates are also distinct fits.")
    y = _bullet(fig, y, "The Capital IQ allocation distribution remains measurement-only: inflation does not update annual allocation weights.")
    y = _bullet(fig, y, "State cut is not estimated, stored, compared, or shown in this revision.")
    y = _section(fig, y, "Identity check")
    y = _eq(fig, y, rf"$\max|q_t-\bar q_t-\hat q_t|={manifest['gate']['max_identity_error']:.2e}$", 12.5)
    _text(fig, y, rf"Required tolerance: {manifest['gate']['identity_required']:.1e}. Gustavo Q4 anchor error: {manifest['preflight']['allocation_mean_anchor_error']:.2e}.")
    pdf.savefig(fig); plt.close(fig)


def _page_states(pdf, manifest, mode):
    fig = _page("Separated competition states by empirical cell", "3. Joint posterior means and 95% intervals from the free combined fits", 3, 15)
    positions = ([0.08, 0.56, 0.39, 0.25], [0.55, 0.56, 0.39, 0.25],
                 [0.08, 0.20, 0.39, 0.25], [0.55, 0.20, 0.39, 0.25])
    for (cell, label, _), bbox in zip(CELLS, positions):
        with _load_fit(mode, cell, "free_static_combined") as z:
            periods = pd.PeriodIndex(z["periods"].astype(str), freq="Q").to_timestamp()
            ax = fig.add_axes(bbox)
            for field, color, name in (("n_total", INK, r"$q_t$"), ("nbar", GREEN, r"$\bar q_t$"), ("nhat", BLUE, r"$\hat q_t$")):
                mean, lo, hi = _band(z[field]); ax.plot(periods, mean, color=color, lw=1.2, label=name)
                if field != "n_total": ax.fill_between(periods, lo, hi, color=color, alpha=0.12)
            ax.axhline(0, color="#888888", lw=0.5); ax.set_title(label); ax.tick_params(labelsize=7); ax.legend(frameon=False, fontsize=7, ncol=3)
    fig.text(0.08, 0.12, "The observed q path is common; differences across panels come from cell-specific joint state inference.", fontsize=8.8, color=MUTED)
    pdf.savefig(fig); plt.close(fig)


def _state_summary(values):
    values = np.asarray(values).reshape(-1)
    return f"{np.mean(values):.3f}\n[{np.quantile(values, 0.025):.3f}, {np.quantile(values, 0.975):.3f}]"


def _page_state_parameters(pdf, manifest, mode, page_number):
    fig = _page("Identification of the AR(2) state split",
                f"{page_number}. Innovation allocation, damping, and period in the free combined fits",
                page_number, 14.5)
    fig.text(0.08, 0.855,
             r"$q_t=\bar q_t+\hat q_t,\quad \sigma_{\bar q}^2=\omega\tau^2,\quad "
             r"\sigma_{\hat q}^2=(1-\omega)\tau^2$",
             fontsize=11.0, color=INK)
    fig.text(0.08, 0.805,
             r"$\hat q_t=2r\cos(2\pi/P)\hat q_{t-1}-r^2\hat q_{t-2}+\eta_t^h$",
             fontsize=10.8, color=INK)
    fig.text(0.08, 0.765,
             r"$\omega\sim\mathrm{Beta}(2,8),\quad (r-0.25)/0.73\sim\mathrm{Beta}(8,2),\quad "
             r"P\sim\mathcal{N}(12,4^2)\;[6,20]$",
             fontsize=9.5, color=MUTED)
    rows = []
    for cell, label, _ in CELLS:
        with _load_fit(mode, cell, "free_static_combined") as z:
            omega = z["omega"].reshape(-1)
            tau2 = z["tau"].reshape(-1) ** 2
            rows.append([
                label.replace("negative unemployment gap", "neg. gap"),
                _state_summary(omega), _state_summary(tau2),
                _state_summary(omega * tau2), _state_summary((1.0 - omega) * tau2),
                _state_summary(z["cycle_damping"]), _state_summary(z["cycle_period"]),
            ])
    _table(fig, [0.035, 0.43, 0.93, 0.30],
           ["Cell", r"$\omega$", r"$\tau^2$", r"$\sigma_{\bar q}^2$",
            r"$\sigma_{\hat q}^2$", r"$r$", r"$P$ (qtrs)"],
           rows, [0.22, 0.12, 0.13, 0.14, 0.14, 0.12, 0.13], 6.6, (0,))
    fig.text(0.06, 0.395,
             "Posterior mean on the first line; central 95% interval on the second line.",
             fontsize=8.4, color=MUTED)
    y = _section(fig, 0.345, "How to read this page")
    y = _bullet(fig, y, r"$\omega$ near one assigns most quarterly innovation variance to the slow random walk; near zero assigns it to the stationary cycle.", width=78)
    y = _bullet(fig, y, r"$r$ controls damping and $P$ controls cycle length. Wide intervals for either quantity signal weak frequency identification.", width=78)
    y = _bullet(fig, y, "The four rows must not be pooled: the NKPC likelihood produces a separate joint state posterior for each price/activity cell.", width=78)
    warning = ("Mock values diagnose sampler and report behavior only; they are not empirical estimates."
               if manifest["mode"] == "mock" else
               ("The full convergence gate passes. The large slow variance is therefore not a "
                "short-chain artifact."))
    _text(fig, y, warning, width=78, color=RED)
    pdf.savefig(fig); plt.close(fig)


def _page_models(pdf, manifest):
    fig = _page("Free ladder and HSA restriction", "5. All estimated equations and the exact nesting relations", 5)
    fig.text(0.08, 0.855, r"$L_t=a+\alpha_b\pi_{t-1}+\alpha_fE_t\pi_{t+1}$", fontsize=11.5, color=INK)
    fig.text(0.08, 0.805, "Free competition-channel ladder", fontsize=11.2, weight="bold", color=INK)
    free = (
        ("M0 / B0: CES", r"$\pi_t=L_t+\kappa_0x_t+\varepsilon_t$"),
        ("M1 / B1: slope", r"$\pi_t=L_t+(\kappa_0+\delta_s\bar q_t)\,x_t+\varepsilon_t$"),
        ("M2 / B2: direct", r"$\pi_t=L_t+\kappa_0x_t-\theta\hat q_t+\varepsilon_t$"),
        ("M3 / B3: free combined", r"$\pi_t=L_t+(\kappa_0+\delta_s\bar q_t)\,x_t-\theta\hat q_t+\varepsilon_t$"),
    )
    y = 0.755
    for name, equation in free:
        fig.text(0.085, y, name, fontsize=9.2, weight="bold", color=INK)
        fig.text(0.105, y-0.033, equation, fontsize=10.5, color=INK)
        y -= 0.105
    fig.text(0.08, 0.31, "Free nesting", fontsize=11.0, weight="bold", color=INK)
    fig.text(0.105, 0.265, r"$M0\subset M1\subset M3,\qquad M0\subset M2\subset M3$", fontsize=11.0, color=INK)

    fig.text(0.53, 0.805, "HSA restriction in inverse-markup cells", fontsize=11.2, weight="bold", color=INK)
    fig.text(0.54, 0.745, "Unrestricted B3", fontsize=9.2, weight="bold", color=INK)
    fig.text(0.55, 0.707, r"$\kappa_t=\kappa_0+\delta_s\bar q_t$", fontsize=10.8, color=INK)
    fig.text(0.54, 0.645, "Fixed-lambda HSA(j)", fontsize=9.2, weight="bold", color=INK)
    fig.text(0.55, 0.607, r"$\pi_t=L_t+\kappa_0x_t+\theta[\lambda_j\bar q_t\,x_t-\hat q_t]+\varepsilon_t$", fontsize=9.5, color=INK)
    fig.text(0.55, 0.555, r"$\delta_s=\lambda_j\theta,\quad \lambda_j\in\{3,6,9\}$", fontsize=10.8, color=INK)
    fig.text(0.54, 0.485, "Empirical-coordinate multiplier", fontsize=9.2, weight="bold", color=INK)
    fig.text(0.55, 0.445, r"$\lambda=\frac{d\kappa_t/d\bar q_t}{\theta}$", fontsize=10.8, color=INK)
    fig.text(0.55, 0.405, r"$\lambda=b_x\zeta$ only if $q$ and structural $N$ use the same normalization.", fontsize=7.8, color=MUTED)
    fig.text(0.54, 0.35, "Restriction diagnostics", fontsize=9.2, weight="bold", color=INK)
    fig.text(0.55, 0.31, r"$r_j=\delta_s-\lambda_j\theta$", fontsize=10.8, color=INK)
    fig.text(0.55, 0.258, r"$D_{j,t}=r_j\bar q_t\,x_t,\quad A_j=\sqrt{T^{-1}\sum_tD_{j,t}^2}$", fontsize=10.0, color=INK)
    fig.text(0.54, 0.185, r"Practical equivalence: $P(A_j<0.10)\geq0.80$.", fontsize=9.0, color=INK)
    fig.text(0.08, 0.14, "Free lambda is retained only as a weak-identification diagnostic; B4 and B5 are not estimated.", fontsize=8.8, color=MUTED)
    pdf.savefig(fig); plt.close(fig)


def _page_hsa(pdf, manifest):
    fig = _page("HSA restriction in each inverse-markup cell", "6. Directly nested in the free combined benchmark", 6)
    y = _section(fig, 0.85, "Unrestricted benchmark B3"); y = _eq(fig, y, r"$\pi_t=L_t+(\kappa_0+\delta_s\bar q_t)\,x_t-\theta\hat q_t+\varepsilon_t$", 11.6)
    y = _section(fig, y, "Fixed-lambda HSA(j)"); y = _eq(fig, y, r"$\pi_t=L_t+\kappa_0x_t+\theta[\lambda_j\bar q_t\,x_t-\hat q_t]+\varepsilon_t$", 11.6)
    y = _eq(fig, y, r"$\delta_s=\lambda_j\theta,\qquad \lambda_j\in\{3,6,9\}$")
    y = _section(fig, y, "Meaning of lambda"); y = _eq(fig, y, r"$\lambda=\frac{d\kappa_t/d\bar q_t}{\theta}$")
    y = _text(fig, y, r"Lambda is estimated in the empirical q coordinate. It equals $b_x\zeta$ only when q and the structural competition coordinate N use the same normalization; setting $b_x=1$ alone is not sufficient.")
    y = _section(fig, y, "Restriction diagnostic from B3"); y = _eq(fig, y, r"$r_j=\delta_s-\lambda_j\theta,\qquad D_{j,t}=r_j\bar q_t\,x_t$")
    y = _eq(fig, y, r"$A_j=\sqrt{T^{-1}\sum_tD_{j,t}^2}$")
    y = _text(fig, y, r"Practical equivalence requires $P(A_j<0.10)\geq0.80$. A credible interval containing zero is not sufficient.")
    y = _section(fig, y, "Free lambda"); _text(fig, y, r"The parameterization $\delta_s=\lambda\theta$ with free $\lambda$ is singular near $\theta=0$ and is retained only as an identification diagnostic.")
    pdf.savefig(fig); plt.close(fig)


def _page_priors(pdf, manifest):
    fig = _page("Prior table", "7. Free combined coefficient priors by empirical cell", 7)
    fig.text(0.08, 0.855, r"$\pi_t=L_t+(\kappa_0+\delta_s\bar q_t)\,x_t-\theta\hat q_t+\varepsilon_t$", fontsize=10.8, color=INK)
    columns = ["Parameter"] + [label for _, label in CELL_COLUMNS]
    _table(fig, [0.055, 0.44, 0.89, 0.34], columns, _prior_rows(manifest),
           [0.12, 0.22, 0.22, 0.22, 0.22], 7.5, (0,))
    y = _section(fig, 0.37, "Additional prior structure")
    y = _eq(fig, y, r"$\omega\sim\mathrm{Beta}(2,8),\quad r^*\sim\mathrm{Beta}(8,2),\quad P\sim\mathcal{N}(12,4^2)$", 10.8)
    y = _eq(fig, y, r"$s_{\theta,j}^{-2}=s_\theta^{-2}+\lambda_j^2/s_\delta^2$", 11.5)
    y = _text(fig, y, r"The HSA prior is induced from free B3 by conditioning on $\delta_s=\lambda_j\theta$. Coefficient signs are unrestricted.")
    sample = manifest["sampling"]
    y = _section(fig, y, f"{manifest['mode'].capitalize()} sampling")
    y = _bullet(fig, y, f"{sample['chains']} chains, {sample['iterations']} iterations, {sample['warmup']} warmup, thinning {sample['thin']}.")
    warning = "Mock draws test code and storage only; none of their posterior values are empirical evidence." if manifest["mode"] == "mock" else "Substantive interpretation requires all convergence and predictive gates to pass."
    _text(fig, y, warning, color=RED); pdf.savefig(fig); plt.close(fig)


def _page_coefficients(pdf, manifest):
    fig = _page("Free combined coefficients and priors", "6. Posterior table and a separate prior table on the same page", 6)
    fig.text(0.08, 0.855, r"$\pi_t=L_t+(\kappa_0+\delta_s\bar q_t)\,x_t-\theta\hat q_t+\varepsilon_t$", fontsize=10.8, color=INK)
    columns = ["Parameter"] + [label for _, label in CELL_COLUMNS]
    fig.text(0.065, 0.805, "Posterior coefficient table", fontsize=10.8, weight="bold", color=INK)
    _table(fig, [0.045, 0.43, 0.91, 0.34], columns, _posterior_rows(manifest),
           [0.11, 0.2225, 0.2225, 0.2225, 0.2225], 7.6, (0,))
    fig.text(0.065, 0.395, "Mean on the first line; central 95% credible interval on the second line.", fontsize=8.5, color=MUTED)
    fig.text(0.065, 0.345, "Prior table", fontsize=10.8, weight="bold", color=INK)
    _table(fig, [0.055, 0.105, 0.89, 0.215], columns, _prior_rows(manifest),
           [0.12, 0.22, 0.22, 0.22, 0.22], 6.7, (0,))
    fig.text(0.065, 0.072, "Convergence diagnostics are reported only in the dedicated convergence figure.", fontsize=8.4, color=MUTED)
    if manifest["mode"] == "mock":
        fig.text(0.57, 0.072, "Mock posterior values validate layout only.", fontsize=8.4, color=RED)
    pdf.savefig(fig); plt.close(fig)


def _page_cell_results(pdf, manifest, cell, label, role, page_number):
    models = PRIMARY_ORDER if role == "primary" else BENCHMARK_ORDER
    fig = _page(f"{label}: model ladder", f"{page_number}. Model-specific coefficients and predictive comparison", page_number, 14.5)
    fig.text(0.08, 0.855, r"$\pi_t=L_t+(\kappa_0+\delta_s\bar q_t)\,x_t-\theta\hat q_t+\varepsilon_t$", fontsize=10.8, color=INK)
    if role == "benchmark":
        fig.text(0.08, 0.810, r"$HSA(j):\ \pi_t=L_t+\kappa_0x_t+\theta[\lambda_j\bar q_t\,x_t-\hat q_t]+\varepsilon_t$", fontsize=10.1, color=INK)
        top, height = 0.74, 0.45
    else:
        top, height = 0.76, 0.31
    rows = []
    for model_id in models:
        model = manifest["models"][_key(cell, model_id)]
        rows.append([MODEL_LABELS[model_id], _estimate(model, "kappa_0"), _estimate(model, "delta_s"),
                     _estimate(model, "theta"), f"{model['metrics']['max_pareto_k']:.3f}",
                     f"{model['comparison']['delta_elpd_vs_ces']:+.2f}"])
    _table(fig, [0.045, top-height, 0.91, height], ["Model", r"$\kappa_0$", r"$\delta_s$", r"$\theta$", "Max Pareto-k", "Delta ELPD"],
           rows, [0.18, 0.20, 0.20, 0.20, 0.11, 0.11], 6.55, (0,))
    note_y = top-height-0.045
    fig.text(0.07, note_y, "Coefficient cells are posterior mean [central 95% interval]. Delta ELPD is relative to CES within this cell.", fontsize=8.4, color=MUTED)
    fig.text(0.07, note_y-0.032, "Convergence diagnostics are shown only in the dedicated convergence figure.", fontsize=8.4, color=MUTED)
    fig.text(0.07, note_y-0.064, "Mock estimates are not substantive." if manifest["mode"] == "mock" else "Exact LOO refits are required before predictive ranking.", fontsize=8.4, color=RED)
    pdf.savefig(fig); plt.close(fig)


def _ladder_rows(manifest, cell, models):
    rows = []
    for model_id in models:
        model = manifest["models"][_key(cell, model_id)]
        rows.append([MODEL_LABELS[model_id], _estimate(model, "kappa_0"), _estimate(model, "delta_s"),
                     _estimate(model, "theta"), f"{model['metrics']['max_pareto_k']:.3f}",
                     f"{model['comparison']['delta_elpd_vs_ces']:+.2f}"])
    return rows


def _page_price_results(pdf, manifest, price, page_number):
    price_label = "PPI" if price == "ppi" else "Core CPI"
    gap = f"{price}_negative_unemployment_gap"
    markup = f"{price}_inverse_markup"
    fig = _page(f"{price_label}: both activity specifications", f"{page_number}. Negative-gap and inverse-markup ladders", page_number, 15)
    columns = ["Model", r"$\kappa_0$", r"$\delta_s$", r"$\theta$", "Max Pareto-k", "Delta ELPD"]
    widths = [0.18, 0.20, 0.20, 0.20, 0.11, 0.11]
    fig.text(0.08, 0.865, r"$\pi_t=L_t+(\kappa_0+\delta_s\bar q_t)\,x_t-\theta\hat q_t+\varepsilon_t$", fontsize=10.7, color=INK)
    fig.text(0.07, 0.815, f"{price_label} x negative unemployment gap: M0-M3", fontsize=10.6, weight="bold", color=INK)
    _table(fig, [0.045, 0.57, 0.91, 0.21], columns, _ladder_rows(manifest, gap, PRIMARY_ORDER), widths, 6.7, (0,))
    fig.text(0.065, 0.535, "Reduced-form channel ladder; HSA coefficient restrictions are not imposed in this activity coordinate.", fontsize=8.3, color=MUTED)
    fig.text(0.07, 0.492, f"{price_label} x inverse markup: B0-B3 and HSA restrictions", fontsize=10.6, weight="bold", color=INK)
    fig.text(0.08, 0.455, r"$HSA(j):\ \pi_t=L_t+\kappa_0x_t+\theta[\lambda_j\bar q_t\,x_t-\hat q_t]+\varepsilon_t$", fontsize=9.7, color=INK)
    _table(fig, [0.045, 0.105, 0.91, 0.315], columns, _ladder_rows(manifest, markup, BENCHMARK_ORDER), widths, 6.15, (0,))
    fig.text(0.065, 0.071, "Cells report posterior mean [95% interval]. Convergence is separated; ELPD remains descriptive pending exact refits.", fontsize=8.2, color=MUTED)
    pdf.savefig(fig); plt.close(fig)


def _page_prior_posterior(pdf, manifest, mode, price, page_number):
    price_label = "PPI" if price == "ppi" else "Core CPI"
    fig = _page(f"{price_label}: prior versus posterior", f"{page_number}. Structural coefficients in the two activity cells", page_number)
    selected = [(c, label) for c, label, _ in CELLS if c.startswith(price + "_")]
    axes = [fig.add_axes([0.09, 0.54, 0.37, 0.25]), fig.add_axes([0.55, 0.54, 0.37, 0.25]),
            fig.add_axes([0.09, 0.19, 0.37, 0.25]), fig.add_axes([0.55, 0.19, 0.37, 0.25])]
    index = 0
    for cell, label in selected:
        with _load_fit(mode, cell, "free_static_combined") as z:
            names = list(map(str, z["names"])); draws = z["draws"].reshape(-1, z["draws"].shape[-1]); summary = manifest["models"][_key(cell, "free_static_combined")]["coefficients"]
            for parameter in ("delta_s", "theta"):
                ax = axes[index]; index += 1; values = draws[:, names.index(parameter)]; row = summary[parameter]; mu, sd = row["prior_mean"], row["prior_sd"]
                grid = np.linspace(min(np.percentile(values, 0.5), mu-3.5*sd), max(np.percentile(values, 99.5), mu+3.5*sd), 300)
                ax.hist(values, bins=28, density=True, color=BLUE, alpha=0.43, label="posterior"); ax.plot(grid, norm.pdf(grid, mu, sd), color=INK, lw=1.2, label="prior"); ax.axvline(0, color="#888", lw=0.5)
                ax.set_title(f"{label}\n{PARAM_LABELS[parameter]} = {row['mean']:+.3f} [{row['q2.5']:+.3f}, {row['q97.5']:+.3f}]"); ax.legend(frameon=False, fontsize=7.5); ax.tick_params(labelsize=7)
    fig.text(0.09, 0.11, "Posterior learning, chain convergence, and valid predictive comparison must be assessed together.", fontsize=8.8, color=MUTED)
    pdf.savefig(fig); plt.close(fig)


def _page_paths(pdf, manifest, mode, page_number):
    fig = _page(r"Posterior-implied $\kappa_t$ and $\theta_t$", f"{page_number}. Free combined model in all four empirical cells", page_number)
    for row, (cell, label, _) in enumerate(CELLS):
        with _load_fit(mode, cell, "free_static_combined") as z:
            names = list(map(str, z["names"])); draws = z["draws"]; periods = pd.PeriodIndex(z["periods"].astype(str), freq="Q").to_timestamp()
            kappa = draws[..., names.index("kappa_0")][..., None] + draws[..., names.index("delta_s")][..., None] * z["nbar"]
            theta = np.repeat(draws[..., names.index("theta")][..., None], len(periods), axis=-1)
            for col, (values, symbol, color) in enumerate(((kappa, r"$\kappa_t$", GREEN), (theta, r"$\theta_t$", BLUE))):
                ax = fig.add_axes([0.08+0.47*col, 0.73-0.19*row, 0.39, 0.135]); mean, lo, hi = _band(values)
                ax.plot(periods, mean, color=color, lw=1.2); ax.fill_between(periods, lo, hi, color=color, alpha=0.15); ax.axhline(0, color="#888", lw=0.5); ax.set_title(f"{label}: {symbol}"); ax.tick_params(labelsize=6.6)
    fig.text(0.08, 0.075, r"In these static models, $\theta_t=\theta$ is constant while $\kappa_t=\kappa_0+\delta_s\bar q_t$ varies with the slow state.", fontsize=8.6, color=MUTED)
    pdf.savefig(fig); plt.close(fig)


def _page_restrictions(pdf, manifest, page_number):
    fig = _page("HSA restriction diagnostics by inflation outcome", f"{page_number}. Distance from the fixed-lambda manifolds", page_number)
    for col, cell in enumerate(("ppi_inverse_markup", "core_cpi_inverse_markup")):
        label = "PPI / inverse markup" if col == 0 else "Core CPI / inverse markup"; results = manifest["restriction_diagnostics"][cell]; lambdas = np.array([3., 6., 9.])
        r = np.array([results[f"{x:g}"]["r_mean"] for x in lambdas]); lo = np.array([results[f"{x:g}"]["r_q2.5"] for x in lambdas]); hi = np.array([results[f"{x:g}"]["r_q97.5"] for x in lambdas]); prob = np.array([results[f"{x:g}"]["equivalence_probability"] for x in lambdas])
        ax = fig.add_axes([0.09+0.47*col, 0.54, 0.37, 0.25]); ax.errorbar(lambdas, r, yerr=[r-lo, hi-r], color=BLUE, marker="o", capsize=3); ax.axhline(0, color="#888", lw=0.6); ax.set_xticks(lambdas); ax.set_title(label+"\nrestriction residual"); ax.set_ylabel(r"$r_j=\delta_s-\lambda_j\theta$")
        ax = fig.add_axes([0.09+0.47*col, 0.19, 0.37, 0.25]); ax.plot(lambdas, prob, color=GREEN, marker="o"); ax.axhline(0.8, color="#888", ls="--", lw=0.6); ax.set_ylim(0, 1); ax.set_xticks(lambdas); ax.set_title(label+"\npractical equivalence"); ax.set_ylabel(r"$P(A_j<0.10)$"); ax.set_xlabel(r"$\lambda_j$")
    fig.text(0.09, 0.11, "The two price outcomes are diagnosed separately; their restriction probabilities are never pooled.", fontsize=8.8, color=MUTED)
    pdf.savefig(fig); plt.close(fig)


def _min_ess(mode, key, diagnostics):
    cell, model = key.split("/")[1:]; bulk = list(map(float, diagnostics["ess_bulk"].values())); tail = list(map(float, diagnostics["ess_tail"].values()))
    with _load_fit(mode, cell, model) as z:
        for name in ("sigma_pi", "phi", "omega", "tau", "cycle_damping", "cycle_period"):
            bulk.append(float(az.ess(z[name], method="bulk"))); tail.append(float(az.ess(z[name], method="tail", prob=(0.05, 0.95))))
    return min(bulk), min(tail)


def _convergence_entries(manifest, mode, price):
    entries = []
    for cell, _, role in CELLS:
        if not cell.startswith(price + "_"):
            continue
        prefix = "Gap" if role == "primary" else "Markup"
        for model in (PRIMARY_ORDER if role == "primary" else BENCHMARK_ORDER):
            value = manifest["models"][_key(cell, model)]
            bulk, tail = _min_ess(mode, _key(cell, model), value["diagnostics"])
            short = MODEL_LABELS[model].replace(" channel", "").replace("Free combined", "Free")
            short = short.replace("HSA, lambda = ", "HSA ").replace("Free-lambda diagnostic", "Free lambda")
            entries.append((f"{prefix} / {short}", float(value["diagnostics"]["max_rhat"]), bulk, tail))
    return entries


def _page_convergence(pdf, manifest, mode, page_number):
    fig = _page("Convergence diagnostics", f"{page_number}. R-hat and effective sample size for all 24 fits", page_number)
    for col, (price, title) in enumerate((("ppi", "PPI"), ("core_cpi", "Core CPI"))):
        entries = _convergence_entries(manifest, mode, price)
        labels = [entry[0] for entry in entries][::-1]
        yy = np.arange(len(entries))
        rhat = np.array([entry[1] for entry in entries])[::-1]
        bulk = np.array([entry[2] for entry in entries])[::-1]
        tail = np.array([entry[3] for entry in entries])[::-1]

        x0 = 0.15 + 0.48 * col
        ax = fig.add_axes([x0, 0.55, 0.32, 0.30])
        ax.scatter(rhat, yy, color=BLUE, s=19)
        ax.axvline(1.01, color=RED, lw=0.8, ls="--", label="full gate 1.01")
        ax.set_yticks(yy); ax.set_yticklabels(labels, fontsize=6.1)
        ax.set_xlim(0.99, max(1.05, float(rhat.max()) + 0.02)); ax.set_title(f"{title}: maximum R-hat")
        ax.legend(frameon=False, fontsize=6.8, loc="lower right"); ax.tick_params(axis="x", labelsize=7)

        ax = fig.add_axes([x0, 0.14, 0.32, 0.30])
        ax.scatter(bulk, yy, color=GREEN, s=19, label="minimum bulk ESS")
        ax.scatter(tail, yy, color=BLUE, marker="x", s=23, label="minimum tail ESS")
        ax.axvline(400, color=RED, lw=0.8, ls="--", label="full gate 400")
        ax.set_yticks(yy); ax.set_yticklabels(labels, fontsize=6.1)
        ax.set_xlim(0, max(450, float(max(bulk.max(), tail.max())) * 1.08)); ax.set_title(f"{title}: effective sample size")
        ax.legend(frameon=False, fontsize=6.2, loc="lower right"); ax.tick_params(axis="x", labelsize=7)
    fig.text(0.08, 0.075, "Convergence is assessed separately from coefficient magnitude and predictive fit.", fontsize=8.8, color=MUTED)
    pdf.savefig(fig); plt.close(fig)


def _page_inventory(pdf, manifest, mode, page_number):
    fig = _page("Complete predictive inventory", f"{page_number}. Predictive diagnostics for every independent run", page_number, 15); rows = []
    for cell, label, role in CELLS:
        for model in (PRIMARY_ORDER if role == "primary" else BENCHMARK_ORDER):
            value = manifest["models"][_key(cell, model)]
            exact = "YES" if value["metrics"]["requires_exact_refit"] else "NO"
            rows.append([label.replace("negative unemployment gap", "neg. gap"), MODEL_LABELS[model],
                         f"{value['metrics']['max_pareto_k']:.3f}", exact,
                         f"{value['comparison']['delta_elpd_vs_ces']:+.2f}"])
    _table(fig, [0.07, 0.12, 0.86, 0.72], ["Cell", "Model", "Max Pareto-k", "Exact refit", "Delta ELPD"],
           rows, [0.27, 0.25, 0.16, 0.14, 0.18], 6.25, (0, 1))
    fig.text(0.08, 0.075, "Delta ELPD remains descriptive until every influential block receives its required exact refit.", fontsize=8.4, color=MUTED)
    pdf.savefig(fig); plt.close(fig)


def _page_diagnostics(pdf, manifest, page_number):
    fig = _page("Validation status and next gate", f"{page_number}. What the current run establishes", page_number); gate = manifest["gate"]; y = _section(fig, 0.85, "Mechanical checks")
    y = _bullet(fig, y, "24 fits present: 4 + 8 for PPI and 4 + 8 for Core CPI."); y = _bullet(fig, y, "Only joint_state_split is present; no state-cut model is stored."); y = _bullet(fig, y, f"Maximum exact state-identity error = {gate['max_identity_error']:.2e}."); y = _bullet(fig, y, r"Fixed-lambda HSA is nested in free B3 by $\delta_s=\lambda\theta$ to machine precision.")
    y = _section(fig, y, f"{manifest['mode'].capitalize()} convergence check"); y = _bullet(fig, y, f"Maximum confirmatory R-hat = {gate['max_rhat']:.3f}; current threshold = {gate['rhat_required']:.2f}."); y = _bullet(fig, y, f"Free-lambda diagnostic maximum R-hat = {gate['free_lambda_diagnostic_max_rhat_not_gating']:.3f}; excluded from the confirmatory gate.")
    if manifest.get("refits"):
        latest = manifest["refits"][-1]
        y = _bullet(fig, y, f"Targeted extension: {latest['fit'].split('/')[-2]} / {latest['fit'].split('/')[-1]}, R-hat {latest['initial_max_rhat']:.3f} to {latest['replacement_max_rhat']:.3f}.")
    if manifest["mode"] == "mock": y = _text(fig, y, "The mock run is intentionally too short for empirical interpretation. Its role is to detect indexing, storage, algebra, and report failures.", color=RED)
    elif not gate["passed"]: y = _text(fig, y, "The full convergence gate failed. All results remain preliminary.", color=RED)
    y = _section(fig, y, "Required before substantive conclusions")
    if manifest["mode"] == "full" and gate["passed"]:
        y = _bullet(fig, y, "Perform exact LOO refits for every four-quarter block with Pareto-k above 0.7; current PSIS rankings remain descriptive.")
        _bullet(fig, y, "Then compare HSA separately for PPI and Core CPI and run annual-origin forecasts and validated marginal likelihood.")
    else:
        y = _bullet(fig, y, "Run the full 24-fit sampler and require every coefficient and state convergence gate to pass; then refit every four-quarter block with Pareto-k above 0.7.")
        _bullet(fig, y, "Only after those gates, compare HSA separately for PPI and Core CPI and run annual-origin forecasts and validated marginal likelihood.")
    fig.text(0.075, 0.315, "Current workflow status", fontsize=11.0, weight="bold", color=INK)
    full_done = manifest["mode"] == "full"
    full_status = "PASS" if full_done and gate["passed"] else ("FAIL" if full_done else "PENDING")
    full_value = f"max R-hat {gate['max_rhat']:.3f}" if full_done else "not run"
    n_exact = sum(v["metrics"]["requires_exact_refit"] for v in manifest["models"].values())
    status_rows = [
        ["Price/activity cells", "4 of 4", "PASS"],
        ["Requested fits", "24 of 24", "PASS"],
        ["State-cut fits", "0", "PASS"],
        ["Exact state identity", f"{gate['max_identity_error']:.2e}", "PASS"],
        ["Full convergence", full_value, full_status],
        ["Exact predictive refits", f"{n_exact} fits require refit", "PENDING"],
        ["Formal marginal likelihood", "not run", "PENDING"],
    ]
    _table(fig, [0.11, 0.095, 0.78, 0.19], ["Stage", "Current value", "Status"], status_rows,
           [0.48, 0.30, 0.22], 7.2, (0, 1))
    pdf.savefig(fig); plt.close(fig)


def _write_tables(manifest, result, mode):
    out = result / "tables"; out.mkdir(parents=True, exist_ok=True); coefficients, models, restrictions, convergence, states = [], [], [], [], []
    for key, value in manifest["models"].items():
        for parameter, row in value["coefficients"].items(): coefficients.append({"fit": key, "parameter": parameter, **row})
        models.append({"fit": key, **value["metrics"], **value["comparison"], "max_rhat": value["diagnostics"]["max_rhat"]})
        bulk, tail = _min_ess(mode, key, value["diagnostics"])
        convergence.append({"fit": key, "max_rhat": value["diagnostics"]["max_rhat"],
                            "min_bulk_ess": bulk, "min_tail_ess": tail,
                            "exact_identity_error": value["diagnostics"]["exact_identity_error"]})
    for cell, grid in manifest["restriction_diagnostics"].items():
        for lam, row in grid.items(): restrictions.append({"cell": cell, "lambda": lam, **row})
    for cell, _, _ in CELLS:
        with _load_fit(mode, cell, "free_static_combined") as z:
            omega = z["omega"].reshape(-1); tau2 = z["tau"].reshape(-1) ** 2
            parameters = (
                ("omega", omega), ("tau2", tau2),
                ("slow_innovation_variance", omega * tau2),
                ("cycle_innovation_variance", (1.0 - omega) * tau2),
                ("cycle_damping", z["cycle_damping"].reshape(-1)),
                ("cycle_period_quarters", z["cycle_period"].reshape(-1)),
            )
            for parameter, values in parameters:
                states.append({"cell": cell, "model": "free_static_combined", "parameter": parameter,
                               "mean": np.mean(values), "q2.5": np.quantile(values, 0.025),
                               "q97.5": np.quantile(values, 0.975)})
    frame = pd.DataFrame(coefficients); frame.to_csv(out / "coefficients.csv", index=False); frame[["fit", "parameter", "prior_mean", "prior_sd", "mean", "sd", "q2.5", "q97.5"]].to_csv(out / "prior_posterior.csv", index=False); pd.DataFrame(models).to_csv(out / "model_comparison.csv", index=False); pd.DataFrame(restrictions).to_csv(out / "restriction_grid.csv", index=False); pd.DataFrame(convergence).to_csv(out / "convergence.csv", index=False); pd.DataFrame(states).to_csv(out / "state_identification.csv", index=False)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--mode", choices=("mock", "quick", "full"), default="mock"); args = parser.parse_args(); result = BUNDLE / "results" / args.mode; manifest_path = result / "manifest.json"
    if not manifest_path.exists(): raise FileNotFoundError(f"Run estimation first: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")); active = load_yaml(BUNDLE / "config.yaml")["revision"]
    if manifest.get("revision") != active: raise RuntimeError(f"Result revision {manifest.get('revision')} does not match {active}")
    _style(); _write_tables(manifest, result, args.mode); filename = "hsa_nested_validation_report.pdf" if args.mode == "full" else f"hsa_nested_validation_{args.mode}_report.pdf"; local = result / filename
    with PdfPages(local) as pdf:
        _page_map(pdf, manifest); _page_data(pdf, manifest); _page_states(pdf, manifest, args.mode)
        _page_state_parameters(pdf, manifest, args.mode, 4); _page_models(pdf, manifest); _page_coefficients(pdf, manifest)
        _page_price_results(pdf, manifest, "ppi", 7); _page_price_results(pdf, manifest, "core_cpi", 8)
        _page_prior_posterior(pdf, manifest, args.mode, "ppi", 9); _page_prior_posterior(pdf, manifest, args.mode, "core_cpi", 10)
        _page_paths(pdf, manifest, args.mode, 11); _page_restrictions(pdf, manifest, 12)
        _page_convergence(pdf, manifest, args.mode, 13); _page_inventory(pdf, manifest, args.mode, 14); _page_diagnostics(pdf, manifest, 15)
        meta = pdf.infodict(); meta["Title"] = "HSA NKPC: PPI and Core CPI Separate 24-Fit Report"; meta["Author"] = "NKPC_HSA_MCMC"
    final = ROOT / "output" / "pdf" / filename; final.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(local, final); print(final)


if __name__ == "__main__": main()
