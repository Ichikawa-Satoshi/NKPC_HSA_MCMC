"""Build the English PDF report for a saved mixed-frequency test profile."""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import textwrap
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists())
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT / "tests")]
from tests import _bootstrap  # noqa: F401,E402
from nkpc_hsa.config import load_yaml  # noqa: E402
from tests.gustavo_state_capitaliq_cycle.functions import load_nkpc_cells  # noqa: E402

BUNDLE = Path(__file__).resolve().parent
BASE_CONFIG = ROOT / "tests" / "gustavo_state_capitaliq_cycle" / "config.yaml"


def _style() -> None:
    mpl.rcParams.update({
        # Matplotlib's bundled STIX files embed reliably across macOS PDF
        # stacks; the system STIX Two file exposes an unsupported style flag.
        "font.family": "STIXGeneral",
        "font.size": 9.2,
        "mathtext.fontset": "stix",
        "axes.titlesize": 12,
        "axes.labelsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "pdf.fonttype": 42,
    })


def _page(title: str, subtitle: str = ""):
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.subplots_adjust(left=.09, right=.94, top=.90, bottom=.07)
    fig.text(.09, .955, title, fontsize=18, fontweight="bold", va="top")
    if subtitle:
        fig.text(.09, .925, subtitle, fontsize=9.5, color="#444444", va="top")
    return fig


def _paragraph(fig, x: float, y: float, text: str, width: int = 105,
               size: float = 9.3, color: str = "#222222") -> float:
    wrapped = textwrap.fill(text, width=width)
    fig.text(x, y, wrapped, fontsize=size, color=color, va="top", linespacing=1.32)
    return y - .019 * (wrapped.count("\n") + 1)


def _table(ax, frame: pd.DataFrame, widths=None, font_size=8.3, yscale=1.35):
    ax.axis("off")
    table = ax.table(cellText=frame.values, colLabels=frame.columns, loc="upper left",
                     cellLoc="left", colLoc="left", colWidths=widths)
    table.auto_set_font_size(False)
    table.set_fontsize(font_size)
    table.scale(1, yscale)
    for (r, _), cell in table.get_celld().items():
        cell.set_edgecolor("#b7b7b7")
        cell.set_linewidth(.45)
        cell.set_facecolor("#eeeeee" if r == 0 else "white")
        if r == 0:
            cell.set_text_props(weight="bold")
    return table


def _fmt_interval(row) -> str:
    return f"{row['mean']:.3f}\n[{row['q2.5']:.3f}, {row['q97.5']:.3f}]"


def _load(profile: str):
    out = BUNDLE / "results" / profile
    required = [out / "manifest.json", out / "tables" / "coefficients.csv"]
    if any(not path.exists() for path in required):
        raise FileNotFoundError(f"Run profile {profile!r} before building its report")
    manifest = json.loads((out / "manifest.json").read_text())
    tables = {path.stem: pd.read_csv(path) for path in (out / "tables").glob("*.csv")}
    return out, manifest, tables


def build(profile: str = "mock") -> Path:
    _style()
    out, manifest, tables = _load(profile)
    destination = out / "report" / f"mixed_frequency_gustavo_capitaliq_{profile}.pdf"
    destination.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(destination) as pdf:
        # 1. Executive summary and research sequence
        fig = _page("Mixed-frequency Gustavo × Capital IQ competition state",
                    f"{profile.upper()} diagnostic — NOT FOR INFERENCE")
        y = .86
        y = _paragraph(fig, .09, y,
            "Objective. Use every genuinely observed timing signal without inventing a quarterly firm-stock series. "
            "Annual Gustavo observations constrain total competition at Q4; two Capital IQ growth series inform the "
            "within-year path. The entire state posterior is estimated without inflation feedback.", width=102, size=10)
        fig.text(.09, y-.02, "Admissible research sequence", fontsize=12, fontweight="bold")
        steps = [
            "1. Validate the measurement-only state against hidden Capital IQ quarters.",
            "2. Propagate state uncertainty into a free direct-channel coefficient, theta_N.",
            "3. Add the free slow-state slope coefficient, delta, and check whether theta_N survives.",
            "4. Run simulation recovery on the realized data geometry.",
            "5. Only after those gates, test a restricted HSA cross-equation relation.",
        ]
        for i, line in enumerate(steps):
            fig.text(.11, y-.065-.045*i, line, fontsize=10, va="top")
        fig.text(.09, .48, "Core decomposition", fontsize=12, fontweight="bold")
        fig.text(.13, .425, r"$n_t=\bar n_t+\hat n_t$", fontsize=20)
        fig.text(.09, .35, "What this run can decide", fontsize=12, fontweight="bold")
        _paragraph(fig, .09, .315,
            "The mock can reveal coding errors, exact-anchor failures, gross sampling failure, and whether the new "
            "measurement state improves a prespecified blocked prediction exercise. It cannot establish a structural "
            "HSA effect, and WAIC differences are descriptive until identification and convergence pass at a longer profile.")
        gate = manifest["gate"]
        fig.text(.09, .19, f"Mock gate: {'PASS' if gate['passed'] else 'FAIL'}", fontsize=13,
                 fontweight="bold", color="#236b2c" if gate["passed"] else "#9b2c2c")
        fig.text(.09, .145, f"Created {manifest['created_utc']}  |  Four NKPC cells completed  |  Seed {manifest['seed']}", fontsize=8.7)
        pdf.savefig(fig); plt.close(fig)

        # 2. Implemented measurement likelihood and data
        fig = _page("1. Measurement model and data",
                    "Exact annual totals; noisy quarterly growth; no inflation feedback")
        fig.text(.09, .865, r"$\bar n_t=\bar n_{t-1}+m_{q(t)}+\eta^b_t,\quad "
                 r"\hat n_t=\phi_1\hat n_{t-1}+\phi_2\hat n_{t-2}+u_t$", fontsize=15)
        fig.text(.09, .815, r"$\sigma_{\bar n}^2=\omega\tau^2,\quad "
                 r"\sigma_{\hat n}^2=(1-\omega)\tau^2$", fontsize=15)
        fig.text(.09, .755, r"$g_y=(\bar n_t+\hat n_t)_{yQ4}\ \mathrm{exactly}$", fontsize=15)
        fig.text(.09, .705, r"$\Delta c_{j,t}=a_j+b_j\Delta(\bar n_t+\hat n_t)+e_{j,t}$", fontsize=15)
        y = .64
        y = _paragraph(fig, .09, y,
            "Gustavo is an equality condition, not a zero-noise Gaussian contribution. This distinction prevents the "
            "annual-anchor density from becoming unbounded as the total innovation variance approaches zero. Average "
            "quarterly allocation enters m_q(t), the prior transition mean, and is never labeled as observed N.")
        data_table = pd.DataFrame([
            ["Gustavo effective firms", "Annual Q4", "1974Q4–2013Q4", "40", "10 log points, 1993Q4=0"],
            ["Capital IQ firm weighted", "Quarterly growth", "1993Q2–2013Q4", "84", "10 Delta log coordinate"],
            ["Capital IQ revenue weighted", "Quarterly growth", "1993Q2–2013Q4", "84", "10 Delta log coordinate"],
            ["PPI inflation", "Quarterly", "1993Q2–2013Q4", "83", "400 Delta log P"],
            ["Inverse markup", "Quarterly", "1993Q2–2013Q4", "83", "level used as x_t"],
            ["SPF expectation", "Quarterly", "1993Q2–2013Q4", "83", "genuine one-quarter ahead"],
        ], columns=["Series", "Frequency", "Sample", "N", "Transformation"])
        ax = fig.add_axes([.08, .18, .86, .36])
        _table(ax, data_table, [.25, .14, .18, .07, .30], 7.8, 1.55)
        fig.text(.09, .105,
                 "Capital IQ is differenced only after reindexing to a complete quarterly grid; sparse annual values are not treated as QoQ observations.", fontsize=8.7)
        pdf.savefig(fig); plt.close(fig)

        # 3. Observed series and posterior decomposition
        path = tables["state_paths"]
        x = np.arange(len(path))
        ticks = np.arange(0, len(path), 16)
        labels = path.period.iloc[ticks].str.slice(0, 4)
        fig = _page("2. Competition data and inferred decomposition",
                    "Posterior mean and 95% intervals; Gustavo anchors are exact")
        ax1 = fig.add_axes([.10, .56, .82, .29])
        ax1.fill_between(x, path["n_total_q2.5"], path["n_total_q97.5"], color="#6b8fb3", alpha=.18)
        ax1.plot(x, path.n_total_mean, color="#295f8a", lw=1.5, label="total n")
        anchors = path.gustavo.notna()
        ax1.scatter(x[anchors], path.loc[anchors, "gustavo"], s=18, color="black", zorder=3, label="Gustavo Q4")
        ax1.set_ylabel("10 log points")
        ax1.set_xticks(ticks, labels)
        ax1.legend(frameon=False, ncol=2, loc="upper left")
        ax1.set_title("Total competition coordinate")
        ax2 = fig.add_axes([.10, .18, .82, .29])
        ax2.fill_between(x, path["nbar_q2.5"], path["nbar_q97.5"], color="#437a59", alpha=.17)
        ax2.plot(x, path.nbar_mean, color="#27633d", lw=1.4, label=r"slow $\bar n_t$")
        ax2.fill_between(x, path["nhat_q2.5"], path["nhat_q97.5"], color="#b06c63", alpha=.14)
        ax2.plot(x, path.nhat_mean, color="#963f35", lw=1.2, label=r"cycle $\hat n_t$")
        ax2.axhline(0, color="#888888", lw=.6)
        ax2.set_ylabel("10 log points")
        ax2.set_xticks(ticks, labels)
        ax2.legend(frameon=False, ncol=2, loc="upper left")
        ax2.set_title("Slow and cyclical components")
        _paragraph(fig, .10, .105,
                   "Intervals include measurement and state-parameter uncertainty. Wide bands outside the Capital IQ overlap are an intended consequence of missing timing data.",
                   width=100, size=8.7)
        pdf.savefig(fig); plt.close(fig)

        # 4. State parameter table and backtest
        fig = _page("3. State identification and blocked validation",
                    "A measurement model should earn its complexity before entering the NKPC")
        state_par = tables["state_parameters"].copy()
        keep = ["omega", "tau", "cycle_damping", "cycle_period", "loading_firm_weighted",
                "sigma_firm_weighted", "loading_revenue_weighted", "sigma_revenue_weighted"]
        state_par = state_par[state_par.parameter.isin(keep)]
        state_par["Estimate [95% interval]"] = state_par.apply(_fmt_interval, axis=1)
        state_par["R-hat"] = state_par.rhat.map(lambda z: f"{z:.3f}")
        state_par["Bulk ESS"] = state_par.ess_bulk.map(lambda z: f"{z:.0f}")
        show = state_par[["parameter", "Estimate [95% interval]", "R-hat", "Bulk ESS"]]
        show.columns = ["Parameter", "Posterior mean [95% interval]", "R-hat", "Bulk ESS"]
        ax = fig.add_axes([.08, .49, .86, .37]); _table(ax, show, [.28, .40, .12, .14], 8.0, 1.40)
        back = tables["backtest_summary"].copy()
        back["RMSE"] = back.rmse.map(lambda z: f"{z:.3f}")
        back["MAE"] = back.mae.map(lambda z: f"{z:.3f}")
        back["Log score"] = back.mean_log_score.map(lambda z: f"{z:.3f}")
        back = back[["method", "series", "n", "RMSE", "MAE", "Log score"]]
        back.columns = ["Method", "Capital IQ measure", "N", "RMSE", "MAE", "Mean log score"]
        fig.text(.09, .43, "Prespecified held-out Capital IQ growth", fontsize=11.5, fontweight="bold")
        ax = fig.add_axes([.08, .10, .86, .29]); _table(ax, back, [.23, .22, .07, .12, .12, .17], 7.7, 1.35)
        pdf.savefig(fig); plt.close(fig)

        # 5. NKPC equations and coefficient results
        fig = _page("4. Free-channel NKPC estimates",
                    "PPI × inverse markup × genuine one-quarter-ahead SPF; IID disturbance")
        fig.text(.09, .865, "Direct only", fontsize=11.5, fontweight="bold")
        fig.text(.09, .82, r"$\pi_t^q=a+\alpha_b\pi_{t-1}^q+\alpha_fE_t\pi_{t+1}^q+\kappa_0x_t-\theta_N\hat n_t+\varepsilon_t$", fontsize=13)
        fig.text(.09, .745, "Free static combined", fontsize=11.5, fontweight="bold")
        fig.text(.09, .70, r"$\pi_t^q=a+\alpha_b\pi_{t-1}^q+\alpha_fE_t\pi_{t+1}^q+(\kappa_0+\delta\bar n_t^c)x_t-\theta_N\hat n_t+\varepsilon_t$", fontsize=13)
        fig.text(.09, .63, r"Oil cells add $\beta_{o0}\Delta o_t+\beta_{o1}\Delta o_{t-1}$.", fontsize=10)
        coef = tables["coefficients"].copy()
        coef = coef[coef.parameter.isin(["alpha_b", "alpha_f", "kappa_0", "delta", "theta_N", "beta_oil_0", "beta_oil_1"])]
        coef["Estimate [95% interval]"] = coef.apply(_fmt_interval, axis=1)
        coef["P(>0)"] = coef.p_positive.map(lambda z: f"{z:.3f}")
        coef["Post/prior SD"] = coef.posterior_prior_sd_ratio.map(lambda z: f"{z:.2f}")
        show = coef[["model", "oil", "parameter", "Estimate [95% interval]", "P(>0)", "Post/prior SD"]]
        show.columns = ["Model", "Oil", "Coefficient", "Posterior mean [95% interval]", "P(>0)", "Post/prior SD"]
        ax = fig.add_axes([.06, .08, .90, .49]); _table(ax, show, [.19, .12, .15, .28, .10, .13], 7.25, 1.25)
        pdf.savefig(fig); plt.close(fig)

        # 6. Prior/posterior learning and time-varying kappa
        fig = _page("5. Structural learning and slope path",
                    "Posterior contraction is distinct from a preferred-sign probability")
        prior = tables["prior_posterior"]
        structural = prior[prior.parameter.isin(["theta_N", "delta"])].reset_index(drop=True)
        ax = fig.add_axes([.20, .59, .70, .25])
        yv = np.arange(len(structural))[::-1]
        coef_source = tables["coefficients"].set_index(["model", "oil", "parameter"])
        for i, row in structural.iterrows():
            key = (row.model, row.oil, row.parameter)
            rr = coef_source.loc[key]
            y0 = yv[i]
            ax.plot([rr["q2.5"], rr["q97.5"]], [y0, y0], color="#295f8a", lw=2)
            ax.scatter(rr["mean"], y0, color="#295f8a", s=26, zorder=3)
            ax.plot([row.prior_mean-1.96*row.prior_sd, row.prior_mean+1.96*row.prior_sd], [y0-.16, y0-.16], color="#999999", lw=1)
        ax.axvline(0, color="black", lw=.7)
        label_model = {"direct_only": "Direct", "free_static_combined": "Combined"}
        label_oil = {"without_oil": "no oil", "with_oil": "oil"}
        ax.set_yticks(yv, [f"{label_model[r.model]}, {label_oil[r.oil]}: {r.parameter}" for r in structural.itertuples()])
        ax.set_title("Posterior interval (blue) and prior interval (gray, offset)", loc="left")

        combined = out / "draws" / "free_static_combined_without_oil.npz"
        z = np.load(combined, allow_pickle=False)
        names = list(map(str, z["names"])); draws = z["draws"]
        k0 = draws[:, :, names.index("kappa_0")]
        delta = draws[:, :, names.index("delta")]
        bars = z["nbar_used"]
        centered = bars - bars.mean(axis=2, keepdims=True)
        kappa = k0[:, :, None] + delta[:, :, None] * centered
        kp_mean = kappa.mean(axis=(0, 1)); kp_lo = np.percentile(kappa, 2.5, axis=(0, 1)); kp_hi = np.percentile(kappa, 97.5, axis=(0, 1))
        periods = list(map(str, z["periods"])); xx = np.arange(len(periods)); tt = np.arange(0, len(xx), 8)
        ax2 = fig.add_axes([.11, .17, .80, .27])
        ax2.fill_between(xx, kp_lo, kp_hi, color="#437a59", alpha=.18)
        ax2.plot(xx, kp_mean, color="#27633d", lw=1.4)
        ax2.axhline(0, color="#888888", lw=.6)
        ax2.set_xticks(tt, [periods[i][:4] for i in tt])
        ax2.set_ylabel(r"$\kappa_t=\kappa_0+\delta\bar n_t^c$")
        ax2.set_title("Free combined model, without oil", loc="left")
        fig.text(.11, .105, "The direct coefficient theta_N is constant in this diagnostic. No dynamic theta or HSA restriction is estimated.", fontsize=8.7)
        pdf.savefig(fig); plt.close(fig)

        # 7. Comparison, diagnostics, conclusion
        fig = _page("6. Model comparison, diagnostics, and decision",
                    "Descriptive model scores are not structural evidence")
        comp = tables["model_comparison"].copy()
        for col in ("elpd_waic", "elpd_se", "waic", "p_waic", "delta_elpd_from_best"):
            comp[col] = comp[col].map(lambda z: f"{z:.2f}")
        comp = comp[["model", "oil", "elpd_waic", "elpd_se", "p_waic", "delta_elpd_from_best"]]
        comp.columns = ["Model", "Oil", "ELPD-WAIC", "SE", "p-WAIC", "Delta from best"]
        ax = fig.add_axes([.08, .65, .86, .20]); _table(ax, comp, [.22, .14, .16, .11, .12, .18], 8.0, 1.45)
        gate = manifest["gate"]
        gate_table = pd.DataFrame([
            ["Convergence", f"R-hat <= {gate['rhat_limit']:.2f}; bulk ESS >= {gate['ess_bulk_limit']:.0f}",
             f"{gate['observed_max_rhat']:.3f}; {gate['observed_min_bulk_ess']:.0f}", "PASS" if gate["convergence_passed"] else "FAIL"],
            ["Exact Q4 identity", f"error <= {gate['q4_anchor_error_limit']:.0e}", f"{gate['observed_q4_anchor_error']:.1e}", "PASS" if gate["anchor_passed"] else "FAIL"],
            ["Blocked RMSE", "mixed-frequency no worse than average allocation", f"relative improvement {gate['backtest_relative_rmse_improvement']:.1%}", "PASS" if gate["backtest_passed"] else "FAIL"],
        ], columns=["Gate", "Threshold", "Observed", "Decision"])
        fig.text(.09, .59, "Predeclared mock gates", fontsize=11.5, fontweight="bold")
        ax = fig.add_axes([.08, .39, .86, .17]); _table(ax, gate_table, [.20, .39, .26, .10], 7.9, 1.45)
        fig.text(.09, .33, "Interpretation", fontsize=11.5, fontweight="bold")
        y = .295
        y = _paragraph(fig, .09, y,
            "Establishes: the code path, exact-accounting restriction, measurement-only filtering, blocked prediction, and four free-channel NKPC fits can be assessed together under one reproducible manifest.")
        y = _paragraph(fig, .09, y-.015,
            "Does not establish: a positive direct HSA effect, a valid lambda restriction, or superior marginal evidence. The Capital IQ overlap remains short and state/measurement variance separation may be weak.")
        _paragraph(fig, .09, y-.015,
            "Decision rule: promote to the quick profile only if the blocked measurement test is competitive and the mock reveals no boundary or convergence pathology. Add simulation recovery before any HSA-restricted model.")
        fig.text(.09, .09, "NOT FOR INFERENCE", fontsize=12, fontweight="bold", color="#9b2c2c")
        pdf.savefig(fig); plt.close(fig)

        # 8. Failure diagnosis and next admissible changes
        fig = _page("7. Why the mock is not promoted",
                    "The failed gate is retained as evidence, not tuned away")
        coef_all = tables["coefficients"].set_index(["model", "oil", "parameter"])
        d0 = coef_all.loc[("direct_only", "without_oil", "theta_N")]
        do = coef_all.loc[("direct_only", "with_oil", "theta_N")]
        c0t = coef_all.loc[("free_static_combined", "without_oil", "theta_N")]
        c0d = coef_all.loc[("free_static_combined", "without_oil", "delta")]
        fig.text(.09, .86, "Observed facts", fontsize=11.5, fontweight="bold")
        facts = [
            f"Direct theta_N, no oil: {d0['mean']:.2f} [{d0['q2.5']:.2f}, {d0['q97.5']:.2f}], P(>0)={d0['p_positive']:.3f}.",
            f"Direct theta_N, oil: {do['mean']:.2f} [{do['q2.5']:.2f}, {do['q97.5']:.2f}], P(>0)={do['p_positive']:.3f}.",
            f"Combined theta_N, no oil: {c0t['mean']:.2f} [{c0t['q2.5']:.2f}, {c0t['q97.5']:.2f}], P(>0)={c0t['p_positive']:.3f}.",
            f"Combined delta, no oil: {c0d['mean']:.2f} [{c0d['q2.5']:.2f}, {c0d['q97.5']:.2f}], P(>0)={c0d['p_positive']:.3f}.",
            f"Current oil growth is tightly positive and the oil cells gain about {abs(float(tables['model_comparison'].query("model == 'direct_only' and oil == 'without_oil'").delta_elpd_from_best.iloc[0])):.1f} ELPD-WAIC units relative to direct/no-oil.",
        ]
        for i, line in enumerate(facts):
            fig.text(.11, .815-.039*i, "• " + line, fontsize=9.5, va="top")
        fig.text(.09, .59, "Most likely failure mechanisms", fontsize=11.5, fontweight="bold")
        mechanisms = [
            "Only within-year deviations identify tau: the exact Q4 endpoints and annual-change drift already reconcile the annual total.",
            "Two unknown Capital IQ loadings, total state variance, omega, and two measurement variances create a scale/variance trade-off. The estimated loadings above two are a warning sign, not additional information.",
            "The AR(2) damping posterior mixes poorly and lies near the short-cycle boundary. High-frequency state movement can be exchanged for measurement noise.",
            "Capital IQ and Gustavo may differ in coverage and aggregation, so a common latent growth factor need not predict hidden Capital IQ quarters better than a mechanical annual bridge.",
            "The direct NKPC coefficient loads on the residual cycle after annual conditioning. Its sign is opposite to the maintained HSA orientation in every mock cell.",
        ]
        yy = .55
        for i, line in enumerate(mechanisms, 1):
            yy = _paragraph(fig, .11, yy, f"{i}. {line}", width=96, size=9.2) - .012
        fig.text(.09, .285, "Next admissible diagnostic", fontsize=11.5, fontweight="bold")
        yy = .25
        for line in [
            "Retain this mock unchanged; do not promote it to quick or add an HSA restriction.",
            "Before another MCMC run, estimate a measurement-only reduced version: fix one Capital IQ loading as the coordinate normalization, compare AR(1) and AR(2) cycles, and repeat the same blocked forecast.",
            "Run leave-one-measure-out checks to learn whether the two Capital IQ series identify a shared factor or merely duplicate noise.",
        ]:
            yy = _paragraph(fig, .11, yy, "• " + line, width=96, size=9.2) - .012
        fig.text(.09, .055,
                 "If no reduced measurement model beats the allocation benchmark, abandon this hybrid.\nNo HSA-restricted model is justified by this saved mock.", fontsize=10.0,
                 fontweight="bold", color="#9b2c2c")
        pdf.savefig(fig); plt.close(fig)

    final_dir = ROOT / "output" / "pdf"
    final_dir.mkdir(parents=True, exist_ok=True)
    final_copy = final_dir / destination.name
    shutil.copy2(destination, final_copy)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("mock", "quick", "full"), default="mock")
    args = parser.parse_args()
    print(build(args.profile))


if __name__ == "__main__":
    main()
