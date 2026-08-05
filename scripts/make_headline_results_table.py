"""Build the headline coefficient tables for both firm-count observation designs.

Each table reports the MAIN model (HSA steady, baseline priors) so the reader sees, per
specification (price index x activity measure), the slope level kappa_0, the competition
dependence delta, its Savage-Dickey BF10, the implied kappa_t path, and the convergence
status. The main specification (core CPI x negative unemployment gap) is bold.

The tables also carry delta's OWN Rhat/ESS. That column exists because the blanket
convergence flag is driven by whichever scalar mixes worst, and under the mixed-frequency
design that is almost always the AR(2) block -- a nuisance object for HSA steady, where
theta = 0 keeps Nhat out of the inflation equation entirely. The reader should be able to
see the diagnostics for the parameter the conclusions actually rest on, rather than infer
them from a single aggregated dagger.

Outputs, for each design:
  .../headline_results.tex          (all 9 specs)
  .../ppi_results.tex               (3 PPI specs)
  .../model_comparison_unemp.tex    (5 models x 3 price indices)
where the mixed-frequency (annual-Q4) versions go in the annual_q4/ subdirectory.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import _bootstrap  # noqa: F401
from _bootstrap import ROOT

EN_ROOT = ROOT / "results" / "tables"


def _load12():
    p = ROOT / "scripts" / "12_build_cpi_ppi_report.py"
    spec = importlib.util.spec_from_file_location("build12", p)
    m = importlib.util.module_from_spec(spec)
    sys.modules["build12"] = m
    spec.loader.exec_module(m)
    return m


def build(m, *, frequency: str, out_dir: Path, design: str, label_suffix: str):
    """Write the three headline tables for one firm-count observation design."""
    out_dir.mkdir(parents=True, exist_ok=True)
    runs = m.load_report_runs(min_iter=1, competition_frequency=frequency, verbose=True)
    m.assert_single_sampler_per_cell(runs)
    full_sampler = " / ".join(sorted(
        {m._sampler_label(idata) for (mod, _, _), (_, idata) in runs.items() if mod == "hsa_full"}
    )) or "n/a"
    const_sampler = " / ".join(sorted(
        {m._sampler_label(idata) for (mod, _, _), (_, idata) in runs.items() if mod == "hsa_const_theta"}
    )) or "n/a"

    EN = out_dir  # local alias so the writer lines below stay unchanged
    ACT = [("Negative unemployment gap", "Unemployment gap"),
           ("HP output gap", "HP output gap"),
           ("BN output gap", "BN output gap")]
    PRICE = ["Core CPI", "Headline CPI", "PPI"]          # core first (main index)
    MAIN = ("Unemployment gap", "Core CPI")

    def cells(idata):
        diagnostics = m._diagnostics(idata)
        conv = bool(diagnostics["converged"])
        dag = "" if conv else r"\textsuperscript{$\dagger$}"
        delta = m._fmt(m._summary(idata, "delta")) + dag
        bf = m._fmt_num(m._bf10(idata, "delta"))
        kappa0 = m._fmt(m._summary(idata, "kappa_0"))
        path = m._path_summary(idata, "kappa_t")
        kpath = "--" if path is None else f"{path['start']:+.3f} $\\rightarrow$ {path['end']:+.3f}"
        dr, de = m._group_diagnostics(idata, ["delta"])["max_rhat"], m._group_diagnostics(idata, ["delta"])["min_ess"]
        ddiag = f"{dr:.3f} / {de:.0f}"
        return delta, bf, kappa0, kpath, ddiag, m._conv_status(diagnostics, japanese=False)

    def bold(x):
        return r"\textbf{" + x + "}"

    # ---- headline table: all 9 specs ----
    lines = [
        r"\begin{table}[H]", r"\centering",
        (r"\caption{Headline results (main model: HSA steady, baseline priors, " + design +
         r", $T=124$), by price index and activity measure. $\delta$ is the competition "
         r"dependence of the slope (a positive $\delta$ means the slope flattens as the "
         r"firm count falls); $\mathrm{BF}_{10}$ is the Savage--Dickey Bayes factor "
         r"against $\delta=0$; $\kappa_0$ is the slope at average competition; the last "
         r"column is the implied $\kappa_t$ from the start to the end of the sample. "
         r"The \textbf{main specification (core CPI, negative unemployment gap)} is in "
         r"bold. $\dagger$: fails the \emph{coefficient} convergence rule "
         r"($\hat R\le1.01$ and bulk ESS $\ge400$ over every scalar parameter, "
         r"including the trend drift $n$ and all variances). ``OK (coef)'' marks a "
         r"cell that passes on the coefficients but not on the latent-state paths; "
         r"the group-by-group diagnostics are in "
         r"Table~\ref{tab:group-convergence" + label_suffix + r"}. The "
         r"$\delta$ $\hat R$/ESS column is that coefficient's own mixing, which the "
         r"blanket flag does not show.}"),
        r"\label{tab:headline" + label_suffix + r"}", r"\small",
        r"\begin{tabular}{llcccccc}", r"\toprule",
        (r"Activity measure & Price & $\delta$ [95\% CI] & $\mathrm{BF}_{10}$ & "
         r"$\kappa_0$ [95\% CI] & $\kappa_t$ start$\to$end & $\delta$ $\hat R$/ESS & Conv. \\"),
        r"\midrule",
    ]
    for disp_act, key_act in ACT:
        for i, price in enumerate(PRICE):
            spec = m.INFLATION_SPECS[price][key_act]
            item = runs.get(("hsa_steady", spec, "baseline"))
            if item is None:
                continue
            d, bf, k0, kp, ddiag, conv = cells(item[1])
            act_label = disp_act if i == 0 else ""
            is_main = (key_act, price) == MAIN
            row = [act_label, price, d, bf, k0, kp, ddiag, conv]
            if is_main:
                row = [bold(c) if c else c for c in row]
            lines.append(" & ".join(row) + r" \\")
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"  # replace last midrule
    lines += [r"\end{tabular}", r"\end{table}", ""]
    (EN / "headline_results.tex").write_text("\n".join(lines), encoding="utf-8")

    # ---- PPI table: 3 PPI specs ----
    plines = [
        r"\begin{table}[H]", r"\centering",
        (r"\caption{PPI results (HSA steady, baseline priors, " + design + r"), by activity "
         r"measure. Columns as in Table~\ref{tab:headline" + label_suffix + r"}. "
         r"Every $\delta$ interval includes zero, "
         r"so the CPI relationship is not confirmed at the producer-price stage. "
         r"$\dagger$: outside the convergence criteria.}"),
        r"\label{tab:ppi-results" + label_suffix + r"}", r"\small",
        r"\begin{tabular}{lcccccc}", r"\toprule",
        (r"Activity measure & $\delta$ [95\% CI] & $\mathrm{BF}_{10}$ & "
         r"$\kappa_0$ [95\% CI] & $\kappa_t$ start$\to$end & $\delta$ $\hat R$/ESS & Conv. \\"),
        r"\midrule",
    ]
    for disp_act, key_act in ACT:
        spec = m.INFLATION_SPECS["PPI"][key_act]
        item = runs.get(("hsa_steady", spec, "baseline"))
        if item is None:
            continue
        d, bf, k0, kp, ddiag, conv = cells(item[1])
        plines.append(" & ".join([disp_act, d, bf, k0, kp, ddiag, conv]) + r" \\")
    plines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    (EN / "ppi_results.tex").write_text("\n".join(plines), encoding="utf-8")

    # ---- model-comparison table: all models, negative unemployment gap ----
    def slope_name(model):
        return "kappa" if model in {"ces", "hsa_dynamic"} else "kappa_0"

    clines = [
        r"\begin{table}[H]", r"\centering",
        (r"\caption{Model comparison for the negative unemployment gap (baseline priors, "
         + design + r"), by price index. Slope is $\kappa$ for CES / HSA dynamic and $\kappa_0$ "
         r"otherwise; the competition dependence $\delta$ (and its $\mathrm{BF}_{10}$) is "
         r"defined only for the HSA models with a firm-count-dependent slope. The main "
         r"model, \textbf{HSA steady}, is in bold; full coefficients ($\theta$, $\gamma$, "
         r"$\kappa_t$ path) for every activity measure are in "
         r"Appendix~\ref{app:model-tables}. State blocks: CES conjugate; HSA steady / "
         r"HSA dynamic exact joint FFBS; HSA const-theta " + const_sampler + r"; HSA full "
         + full_sampler + r". $\dagger$: fails the coefficient convergence rule; "
         r"``OK (coef)'' passes on coefficients but not on the latent-state paths.}"),
        r"\label{tab:model-comp" + label_suffix + r"}", r"\small",
        r"\begin{tabular}{llcccc}", r"\toprule",
        (r"Model & Price & slope [95\% CI] & $\delta$ [95\% CI] & $\mathrm{BF}_{10}(\delta)$ "
         r"& Conv. \\"),
        r"\midrule",
    ]
    for model in m.MODEL_ORDER:
        for i, price in enumerate(PRICE):
            spec = m.INFLATION_SPECS[price]["Unemployment gap"]
            item = runs.get((model, spec, "baseline"))
            if item is None:
                continue
            idata = item[1]
            diagnostics = m._diagnostics(idata)
            conv = bool(diagnostics["converged"])
            dag = "" if conv else r"\textsuperscript{$\dagger$}"
            slope = m._fmt(m._summary(idata, slope_name(model)))
            dsum = m._summary(idata, "delta")
            delta = m._fmt(dsum) + (dag if dsum is not None else "")
            bf = m._fmt_num(m._bf10(idata, "delta"))
            row = [m.MODEL_LABELS[model] if i == 0 else "", price, slope, delta, bf,
                   m._conv_status(diagnostics, japanese=False)]
            if model == "hsa_steady":
                row = [bold(c) if c else c for c in row]
            clines.append(" & ".join(row) + r" \\")
        clines.append(r"\midrule")
    clines[-1] = r"\bottomrule"
    clines += [r"\end{tabular}", r"\end{table}", ""]
    (EN / "model_comparison_unemp.tex").write_text("\n".join(clines), encoding="utf-8")

    print(f"  [{design}] wrote headline_results.tex, ppi_results.tex, model_comparison_unemp.tex "
          f"-> {out_dir.relative_to(ROOT)}")
    for line in (EN / "headline_results.tex").read_text().splitlines():
        if "&" in line and "textbf" in line:
            print("     " + line.strip()[:150])


def main():
    m = _load12()
    # Mixed-frequency (annual-Q4) is the main design: the firm count is annual, so the
    # 31 Q4 observations are the data and the 93 intervening quarters are inferred.
    # PCHIP is reported alongside it as the interpolate-then-estimate comparison.
    build(m, frequency="annual_q4", out_dir=EN_ROOT / "annual_q4",
          design="mixed-frequency annual-Q4", label_suffix="")
    build(m, frequency="quarterly_interpolated", out_dir=EN_ROOT / "quarterly_interpolated",
          design="PCHIP-interpolated", label_suffix="-pchip")


if __name__ == "__main__":
    main()
