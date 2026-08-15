from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm

from nkpc_hsa.paths import project_root

from nkpc_hsa.phillips.estimation import DesignRun, summarize_fit


PARAM_LABEL = {
    "a": r"$a$", "beta_b": r"$\beta_b$", "beta_f": r"$\beta_f$",
    "psi": r"$\psi$", "kappa_0": r"$\kappa_0$", "kappa_1": r"$\kappa_1$",
    "theta_0": r"$\theta_0$", "gamma": r"$\gamma$",
}
PRICE_LABEL = {"ppi": "PPI", "cpi": "CPI", "core_cpi": "Core CPI"}
ACTIVITY_LABEL = {
    "inverse_markup": "Inverse markup",
    "bn_output_gap": "BN output gap",
    "negative_unemployment_gap": "Negative unemployment gap",
}


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.18,
            "figure.dpi": 150,
            "savefig.dpi": 220,
        }
    )


def _summary(values: np.ndarray) -> tuple[float, float, float]:
    flat = np.asarray(values, float).reshape(-1)
    return float(flat.mean()), float(np.quantile(flat, 0.025)), float(np.quantile(flat, 0.975))


def _fmt(values: np.ndarray) -> str:
    mean, lo, hi = _summary(values)
    return f"{mean:.3f} [{lo:.3f}, {hi:.3f}]"


def _write_main_table(run: DesignRun, path: Path) -> None:
    lines = [
        r"\begin{longtable}{@{}clp{2.7cm}ccccc@{}}",
        r"\caption{Nine-cell QoQ modular-cut E2 coefficients}\label{tab:ninecell}\\",
        r"\toprule",
        r"Cell & Inflation & Activity & $\psi$ & $\kappa_0$ & $\kappa_1$ & $\theta_0$ & $\gamma$ \\",
        r"\midrule",
        r"\endfirsthead",
        r"\toprule Cell & Inflation & Activity & $\psi$ & $\kappa_0$ & $\kappa_1$ & $\theta_0$ & $\gamma$ \\",
        r"\midrule\endhead",
    ]
    for cell in range(1, 10):
        fit = run.cut_fits[f"cell{cell}_qoq_E2_modular_cut"]
        idx = {name: i for i, name in enumerate(fit.coefficient_names)}
        values = [_fmt(fit.coefficients[:, :, idx[p]]) for p in ("psi", "kappa_0", "kappa_1", "theta_0", "gamma")]
        lines.append(
            f"{cell} & {PRICE_LABEL[fit.inflation]} & {ACTIVITY_LABEL[fit.activity]} & "
            + " & ".join(values) + r" \\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\multicolumn{8}{p{0.97\linewidth}}{\footnotesize Notes: posterior mean and central 95\% credible interval. "
            + (
                r"The test run is a software validation only; its intervals are not inferential results.}\\"
                if run.is_test
                else r"The production run uses retained post-warmup draws and reports convergence separately.}\\"
            ),
            r"\end{longtable}",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_temporal_table(run: DesignRun, path: Path) -> None:
    lines = [
        r"\begin{longtable}{@{}clp{2.7cm}ccc@{}}",
        r"\caption{Endpoint-matched QoQ and exactly aggregated YoY comparison}\label{tab:temporal}\\",
        r"\toprule Cell & Inflation & Activity & $\kappa_1$ QoQ / YoY & $\theta_0$ QoQ / YoY & $\gamma$ QoQ / YoY \\",
        r"\midrule\endfirsthead",
        r"\toprule Cell & Inflation & Activity & $\kappa_1$ QoQ / YoY & $\theta_0$ QoQ / YoY & $\gamma$ QoQ / YoY \\",
        r"\midrule\endhead",
    ]
    for cell in range(1, 10):
        q = run.cut_fits[f"cell{cell}_qoq_matched_E2_modular_cut"]
        y = run.cut_fits[f"cell{cell}_yoy_E2_modular_cut"]
        qi, yi = ({n: i for i, n in enumerate(f.coefficient_names)} for f in (q, y))
        entries = [f"{_fmt(q.coefficients[:, :, qi[p]])} / {_fmt(y.coefficients[:, :, yi[p]])}" for p in ("kappa_1", "theta_0", "gamma")]
        lines.append(f"{cell} & {PRICE_LABEL[q.inflation]} & {ACTIVITY_LABEL[q.activity]} & " + " & ".join(entries) + r" \\")
    lines.extend(
        [
            r"\bottomrule",
            r"\multicolumn{6}{p{0.97\linewidth}}{\footnotesize Notes: QoQ uses the YoY endpoint dates. YoY aggregates the full quarterly design and uses $G_4=A_4A_4'$. A posterior for the QoQ-minus-YoY difference is deliberately not reported.}\\",
            r"\end{longtable}",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_small_tables(run: DesignRun, tables: Path) -> None:
    comparison = pd.read_csv(tables / "cut_joint_comparison.csv")
    lines = [r"\begin{tabular}{@{}crrc@{}}", "\\toprule Cell & Maximum shift (cut SD) & Conflicts & Status \\\\", r"\midrule"]
    for cell, frame in comparison.groupby("cell"):
        count = int(frame["conflict"].sum())
        status = "Flag" if count else "Pass"
        lines.append(f"{cell} & {frame['cut_sd_shift'].max():.2f} & {count} & {status} " + r"\\")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    (tables / "cut_joint.tex").write_text("\n".join(lines), encoding="utf-8")

    b = json.loads((tables / "hsa_benchmark.json").read_text(encoding="utf-8"))
    prob = "not reported" if b["equivalence_probability"] is None else f"{b['equivalence_probability']:.3f}"
    delta = "not estimated" if "delta_mean" not in b else f"{b['delta_mean']:.3f} [{b['delta_ci_2_5']:.3f}, {b['delta_ci_97_5']:.3f}]"
    benchmark_lines = [
        r"\begin{tabular}{@{}ll@{}}", "\\toprule Item & Result \\\\", r"\midrule",
        f"Status & {b['status'].replace('_', ' ')} " + r"\\",
        f"Local QoQ endpoints & {b['local_endpoints']} " + r"\\",
        f"$\\zeta_{{ref}}$ / $b_x$ & {b['zeta_reference']:.2f} / {b['b_x']:.2f} " + r"\\",
        f"$\\delta_{{HSA}}$ & {delta} " + r"\\",
        f"Equivalence band & $\\pm {b['equivalence_band']:.3f}$ " + r"\\",
        f"Equivalence probability & {prob} " + r"\\",
        r"\bottomrule\end{tabular}",
    ]
    (tables / "benchmark.tex").write_text("\n".join(benchmark_lines), encoding="utf-8")

    effects = pd.read_csv(tables / "economic_effects.csv")
    lines = [r"\begin{tabular}{@{}clp{3.2cm}c@{}}", "\\toprule Cell & Inflation & Activity & IQR-normalized slope contribution (pp) \\\\", r"\midrule"]
    for _, row in effects.iterrows():
        val = f"{row['slope_effect_mean_pp']:.3f} [{row['slope_effect_ci_2.5_pp']:.3f}, {row['slope_effect_ci_97.5_pp']:.3f}]"
        lines.append(f"{int(row.cell)} & {PRICE_LABEL[row.inflation]} & {ACTIVITY_LABEL[row.activity]} & {val} " + r"\\")
    lines.extend([r"\bottomrule\end{tabular}"])
    (tables / "effects.tex").write_text("\n".join(lines), encoding="utf-8")

    main = pd.concat(
        [
            summarize_fit(fit, test_run=run.is_test)
            for fit in run.cut_fits.values()
            if fit.model == "E2" and fit.transformation == "qoq"
        ],
        ignore_index=True,
    )
    convergence_count = int(main["convergence_gate"].sum())
    evidence_lines = [
        r"\begin{tabular}{@{}p{4.2cm}p{2.5cm}p{7.2cm}@{}}", "\\toprule Gate / evidence & Run status & Interpretation \\\\", r"\midrule",
        f"Quarterly measurement information $R_q\\le0.80$ & {run.measurement.information_ratio:.3f} & "
        + (
            "Numerical threshold met, but test-run diagnostics still block inference."
            if run.is_test and run.measurement.information_ratio <= .8
            else "Threshold met under the baseline measurement specification."
            if run.measurement.information_ratio <= .8
            else "Threshold failed; no stationary-state interpretation."
        ) + r"\\",
        (
            r"Coefficient learning and convergence & Not evaluated & Short chains intentionally fail the production ESS/$\widehat R$ requirements.\\"
            if run.is_test
            else f"Coefficient learning and convergence & {convergence_count}/{len(main)} pass & Production $\\widehat R$/ESS gates evaluated on retained draws. " + r"\\"
        ),
        r"Cut--joint information flow & Displayed & Conflicts are diagnostic flags, not efficiency gains to promote.\\",
        (
            r"Fast-by-fast and CS1/CS2 & Executed & Test-run sensitivity only. Activity endogeneity is not solved.\\"
            if run.is_test
            else r"Fast-by-fast and CS1/CS2 & Executed & Production sensitivity; activity endogeneity is not solved.\\"
        ),
        r"Persistent-error / low-frequency nuisance & Pending & Reported separately when the follow-up robustness module is run.\\",
        r"Chib Bayes factors & Not run & No Bayes-factor claim is made from a modular cut.\\",
        r"Real-time LFO score & Inadmissible & Vintage/release-date inputs are incomplete; no forecast claim is made.\\",
        r"\bottomrule\end{tabular}",
    ]
    (tables / "evidence.tex").write_text("\n".join(evidence_lines), encoding="utf-8")


def _plot_prior_posterior(run: DesignRun, path: Path) -> None:
    fit = run.cut_fits["cell1_qoq_E2_modular_cut"]
    index = {name: i for i, name in enumerate(fit.coefficient_names)}
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.4))
    for ax, parameter in zip(axes, ("kappa_1", "theta_0", "gamma")):
        values = fit.coefficients[:, :, index[parameter]].reshape(-1)
        lo, hi = np.quantile(values, [0.005, 0.995])
        prior_sd = fit.prior_sds[parameter]
        lo, hi = min(lo, -3 * prior_sd), max(hi, 3 * prior_sd)
        grid = np.linspace(lo, hi, 300)
        ax.plot(grid, norm.pdf(grid, 0.0, prior_sd), color="#666666", ls="--", label="Prior")
        if np.std(values) > 0:
            from scipy.stats import gaussian_kde
            ax.plot(grid, gaussian_kde(values)(grid), color="#0072B2", lw=2, label="Posterior")
        ratio = np.std(values, ddof=1) / prior_sd
        ax.set_title(f"{PARAM_LABEL[parameter]}  SD ratio={ratio:.2f}")
        ax.axvline(0, color="#222222", lw=.8)
    axes[0].legend(frameon=False)
    fig.suptitle("Cell 1: normalized prior and modular-cut posterior")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _plot_state(run: DesignRun, path: Path) -> None:
    periods = pd.PeriodIndex(run.measurement.periods, freq="Q").to_timestamp()
    fig, axes = plt.subplots(3, 1, figsize=(11.2, 7.2), sharex=True)
    for ax, name, color, label in zip(
        axes,
        ("qbar", "qhat", "total"),
        ("#0072B2", "#D55E00", "#009E73"),
        (r"Slow environment $\bar q_t$", r"Stationary state $\hat q_t$", r"Total coordinate $q_t$"),
    ):
        values = run.measurement.draws["qbar"] if name == "qbar" else run.measurement.draws["qhat"]
        if name == "total":
            values = run.measurement.draws["qbar"] + run.measurement.draws["qhat"]
        mean = values.mean(axis=(0, 1))
        lo, hi = np.quantile(values, [0.025, 0.975], axis=(0, 1))
        ax.fill_between(periods, lo, hi, color=color, alpha=.18)
        ax.plot(periods, mean, color=color, lw=1.7)
        ax.set_ylabel(label)
        if name == "total":
            mask = np.isfinite(run.data.annual_observation)
            ax.scatter(periods[mask], run.data.annual_observation[mask], color="#222222", s=13, zorder=3, label="Annual Q4 measurement")
            ax.legend(frameon=False, loc="best")
    axes[0].set_title(f"Measurement-only competition decomposition; $R_q={run.measurement.information_ratio:.3f}$")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _plot_slope(run: DesignRun, path: Path) -> None:
    fit = run.cut_fits["cell1_qoq_E2_modular_cut"]
    idx = {name: i for i, name in enumerate(fit.coefficient_names)}
    k0 = fit.coefficients[:, :, idx["kappa_0"]]
    k1 = fit.coefficients[:, :, idx["kappa_1"]]
    path_draws = k0[:, :, None] + k1[:, :, None] * (run.measurement.draws["qbar"] - run.q0)
    mean = path_draws.mean(axis=(0, 1))
    lo, hi = np.quantile(path_draws, [0.025, 0.975], axis=(0, 1))
    periods = pd.PeriodIndex(run.measurement.periods, freq="Q").to_timestamp()
    fig, ax = plt.subplots(figsize=(11.2, 4.2))
    ax.fill_between(periods, lo, hi, color="#0072B2", alpha=.18)
    ax.plot(periods, mean, color="#0072B2", lw=1.8)
    ax.axhline(float(k0.mean()), color="#333333", ls="--", lw=1, label=r"Slope at $q_0$")
    ax.set(title="Cell 1 Phillips-curve slope path", ylabel=r"$\kappa_t=\kappa_0+\kappa_1(\bar q_t-q_0)$")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _plot_cut_joint(run: DesignRun, path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 4.3), sharey=True)
    for ax, parameter in zip(axes, ("kappa_1", "theta_0", "gamma")):
        for cell in range(1, 10):
            cut = run.cut_fits[f"cell{cell}_qoq_E2_modular_cut"]
            joint = run.joint_fits[f"cell{cell}_qoq_E2_full_joint"].fit
            ci, ji = ({n: i for i, n in enumerate(f.coefficient_names)} for f in (cut, joint))
            cv, jv = cut.coefficients[:, :, ci[parameter]], joint.coefficients[:, :, ji[parameter]]
            cm, cl, ch = _summary(cv)
            jm, jl, jh = _summary(jv)
            y = 10 - cell
            ax.errorbar(cm, y + .12, xerr=[[cm - cl], [ch - cm]], fmt="o", color="#0072B2", ms=4)
            ax.errorbar(jm, y - .12, xerr=[[jm - jl], [jh - jm]], fmt="s", color="#D55E00", ms=3.5)
        ax.axvline(0, color="#333333", lw=.8)
        ax.set_title(PARAM_LABEL[parameter])
    axes[0].set_yticks(range(1, 10), labels=[f"Cell {c}" for c in range(9, 0, -1)])
    axes[0].plot([], [], "o", color="#0072B2", label="Cut")
    axes[0].plot([], [], "s", color="#D55E00", label="Full joint")
    axes[0].legend(frameon=False)
    fig.suptitle("Primary modular cut and secondary full-joint E2 posterior")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _plot_locality(run: DesignRun, path: Path) -> None:
    periods = pd.PeriodIndex(run.measurement.periods, freq="Q").to_timestamp()
    fig, ax = plt.subplots(figsize=(11.2, 3.8))
    ax.plot(periods, run.benchmark.local_probability, color="#0072B2", lw=1.8, label=r"$p_t^{loc}$")
    threshold = float(run.data.config["benchmark"]["local_probability"])
    ax.axhline(threshold, color="#D55E00", ls="--", label=f"Frozen threshold {threshold:.2f}")
    ax.fill_between(periods, 0, 1, where=run.benchmark.local_endpoints, color="#009E73", alpha=.10, label="Selected endpoint")
    ax.set_ylim(0, 1.03)
    ax.set(title=f"Cell 1 locality support ({int(run.benchmark.local_endpoints.sum())} QoQ endpoints)", ylabel="Cut posterior probability")
    ax.legend(frameon=False, ncol=3, loc="lower center")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def build_design_report_artifacts(run: DesignRun) -> None:
    _style()
    tables = run.output_dir / "tables"
    figures = run.output_dir / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    _write_main_table(run, tables / "nine_cell_coefficients.tex")
    _write_temporal_table(run, tables / "qoq_yoy.tex")
    _write_small_tables(run, tables)
    _plot_prior_posterior(run, figures / "prior_posterior_cell1.png")
    _plot_state(run, figures / "competition_state.png")
    _plot_slope(run, figures / "slope_path_cell1.png")
    _plot_cut_joint(run, figures / "cut_joint.png")
    _plot_locality(run, figures / "locality.png")


def compile_design_report(run: DesignRun) -> Path:
    root = project_root()
    report_dir = root / "report"
    source = report_dir / "nine_cell_design_report.tex"
    relative = Path("..") / "results" / "nine_cell_design" / run.output_dir.name
    command = (
        rf"\def\ResultRoot{{{relative.as_posix()}}}"
        rf"\def\RunLabel{{{'TEST RUN -- NOT FOR INFERENCE' if run.is_test else 'PRODUCTION RUN'}}}"
        rf"\def\RunAbstractNotice{{{'This short run is for software validation only.' if run.is_test else 'This production run uses four long chains; interpretation remains conditional on the displayed measurement and robustness gates.'}}}"
        rf"\def\PriorCaptionNotice{{{'Short test chains are shown only to verify the reporting path.' if run.is_test else 'Production posterior summaries use retained post-warmup draws.'}}}"
        rf"\def\ScopeNotice{{{'This validation run does not establish production convergence or robustness.' if run.is_test else 'Any failed or unavailable identification gate remains binding even when MCMC convergence passes.'}}}"
        rf"\input{{{source.name}}}"
    )
    for _ in range(2):
        result = subprocess.run(
            ["xelatex", "-interaction=nonstopmode", "-jobname=nine_cell_design_report", command],
            cwd=report_dir,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            tail = "\n".join(result.stdout.splitlines()[-40:])
            raise RuntimeError(f"xelatex failed:\n{tail}")
    pdf = report_dir / "nine_cell_design_report.pdf"
    delivered = run.output_dir / "nine_cell_design_report.pdf"
    shutil.copy2(pdf, delivered)
    return pdf
