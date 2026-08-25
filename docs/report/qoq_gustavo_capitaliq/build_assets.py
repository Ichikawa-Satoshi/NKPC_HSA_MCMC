"""Generate tables and figures for the comprehensive QoQ HSA report."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde, norm

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists())
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT / "tests")]

from nkpc_hsa.config import load_yaml  # noqa: E402
from tests.gustavo_state_capitaliq_cycle.dynamic_functions import dynamic_mu  # noqa: E402
from tests.gustavo_state_capitaliq_cycle.functions import (  # noqa: E402
    build_qoq_design,
    load_cycle,
    load_measurements,
    load_nkpc_cells,
    load_oil_controls,
    load_qoq,
)

BUNDLE = ROOT / "tests" / "gustavo_state_capitaliq_cycle"
BASE = BUNDLE / "results" / "mock_qoq"
STAGED = BUNDLE / "results" / "staged_validation"
DYNAMIC = BUNDLE / "results" / "dynamic_validation"
CORE = BUNDLE / "results" / "core_cpi_full"
OIL = BUNDLE / "results" / "oil_control_full"
HERE = Path(__file__).resolve().parent
FIG = HERE / "figures"
TAB = HERE / "tables"

INK = "#1f2933"
BLUE = "#24567a"
GREEN = "#2f7355"
RED = "#a33b32"
GOLD = "#9a6b1d"
PURPLE = "#6c4f8d"
GRID = "#c9d1d8"
mpl.rcParams.update(
    {
        "font.family": "STIXGeneral",
        "mathtext.fontset": "stix",
        "font.size": 10,
        "axes.titlesize": 11,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
    }
)

CELL_ORDER = [
    ("firm_weighted", "ppi_negative_unemployment_gap", "Firm / unemployment"),
    ("revenue_weighted", "ppi_negative_unemployment_gap", "Revenue / unemployment"),
    ("firm_weighted", "ppi_inverse_markup", "Firm / inverse markup"),
    ("revenue_weighted", "ppi_inverse_markup", "Revenue / inverse markup"),
]
CORE_CELL_ORDER = [
    ("firm_weighted", "core_cpi_negative_unemployment_gap", "Firm / unemployment"),
    ("revenue_weighted", "core_cpi_negative_unemployment_gap", "Revenue / unemployment"),
    ("firm_weighted", "core_cpi_inverse_markup", "Firm / inverse markup"),
    ("revenue_weighted", "core_cpi_inverse_markup", "Revenue / inverse markup"),
]
MODEL_LABEL = {
    "direct_only": "Constant theta",
    "free_combined": "Free static combined",
    "varying_theta": "Varying theta",
    "free_dynamic": "Free dynamic",
    "hsa_restricted_dynamic": "HSA-restricted dynamic",
}


def _save(fig: plt.Figure, name: str) -> None:
    fig.savefig(FIG / name, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _npz(path: Path):
    return np.load(path, allow_pickle=True)


def _summary_from_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _fit_path(model: str, cycle: str, cell: str, error: str = "iid") -> Path:
    if model in {"direct_only", "free_combined"}:
        return STAGED / "draws" / "full" / model / cycle / error / f"{cell}.npz"
    return DYNAMIC / "draws" / "full" / model / cycle / error / f"{cell}.npz"


def _fit(model: str, cycle: str, cell: str, error: str = "iid"):
    path = _fit_path(model, cycle, cell, error)
    meta = _summary_from_json(path.with_suffix(".json"))
    return load_qoq(path, meta["diagnostics"]), meta


def _core_config() -> dict:
    config = load_yaml(BUNDLE / "config.yaml")
    extension = load_yaml(BUNDLE / "core_cpi_config.yaml")
    price = extension["price"]
    config["data"]["prices"] = {
        price["name"]: {k: price[k] for k in ("inflation", "inflation_lag", "expectation")}
    }
    return config


def _core_fit(model: str, cycle: str, cell: str, error: str = "iid"):
    path = CORE / "draws" / "full" / model / cycle / error / f"{cell}.npz"
    meta = _summary_from_json(path.with_suffix(".json"))
    return load_qoq(path, meta["diagnostics"]), meta


def _oil_fit(price: str, model: str, cycle: str, cell: str, error: str = "iid"):
    path = OIL / "draws" / price / "full" / model / cycle / error / f"{cell}.npz"
    meta = _summary_from_json(path.with_suffix(".json"))
    return load_qoq(path, meta["diagnostics"]), meta


def _fmt(mean: float, lo: float, hi: float) -> str:
    return rf"\makecell{{{mean:.3f}\\{{\scriptsize [{lo:.3f}, {hi:.3f}]}}}}"


def _write_table(path: Path, columns: list[str], rows: list[list[str]], widths: str | None = None) -> None:
    spec = widths or ("l" + "c" * (len(columns) - 1))
    lines = [rf"\begin{{tabular}}{{@{{}}{spec}@{{}}}}", "\\toprule", " & ".join(columns) + r" \\", "\\midrule"]
    lines.extend(" & ".join(map(str, row)) + r" \\" for row in rows)
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    path.write_text("\n".join(lines))


def _coefficient_table(model: str, parameters: list[str], filename: str) -> None:
    rows = []
    for parameter in parameters:
        row = [parameter.replace("_", r"\_")]
        for cycle, cell, _ in CELL_ORDER:
            _, meta = _fit(model, cycle, cell)
            source = meta.get("derived", {}) if parameter.startswith("derived:") else meta["coefficients"]
            key = parameter.split(":", 1)[-1]
            z = source[key]
            row.append(_fmt(z["mean"], z["q2.5"], z["q97.5"]))
        rows.append(row)
    _write_table(
        TAB / filename,
        ["Parameter", *[x[2] for x in CELL_ORDER]],
        rows,
        "lcccc",
    )


def data_figures_and_table(config: dict) -> None:
    cells = load_nkpc_cells(config)
    cell = cells["ppi_negative_unemployment_gap"]
    markup = cells["ppi_inverse_markup"]
    periods = cell.periods
    xdate = periods.to_timestamp()

    fig, axes = plt.subplots(3, 1, figsize=(10.6, 7.2), sharex=True)
    axes[0].plot(xdate, cell.pi, color=BLUE, lw=1.35, label="PPI inflation")
    axes[0].plot(xdate, cell.epi, color=GOLD, lw=1.15, label="SPF GDP-price expectation")
    axes[0].axhline(0, color="#7b8794", lw=0.6)
    axes[0].set_ylabel("Annualized %")
    axes[0].legend(frameon=False, ncol=2)
    axes[1].plot(xdate, cell.x, color=GREEN, lw=1.3)
    axes[1].axhline(0, color="#7b8794", lw=0.6)
    axes[1].set_ylabel("Percentage points")
    axes[1].set_title("Negative unemployment gap (NROU - UNRATE)", loc="left")
    axes[2].plot(xdate, markup.x, color=PURPLE, lw=1.3)
    axes[2].axhline(0, color="#7b8794", lw=0.6)
    axes[2].set_ylabel("Log proxy")
    axes[2].set_title(r"Inverse-markup proxy $\log(1.2/\mu_t)$", loc="left")
    for ax in axes:
        ax.grid(axis="y", color=GRID, lw=0.5)
    axes[-1].set_xlabel("Quarter")
    fig.suptitle("QoQ NKPC observables, 1993Q2-2013Q4", x=0.08, ha="left", fontsize=13, weight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    _save(fig, "data_timeseries.png")

    periods_all, gustavo, capital = load_measurements(config)
    fig, axes = plt.subplots(2, 1, figsize=(10.6, 6.7), sharex=False)
    axes[0].plot(gustavo.index.astype(int), gustavo.values, marker="o", ms=3.5, color=BLUE)
    axes[0].axhline(0, color="#7b8794", lw=0.6)
    axes[0].set_title(r"Gustavo annual coordinate: $10\log(N_y^G/N_{1993}^G)$", loc="left")
    axes[0].set_ylabel("10-log-point units")
    cdates = capital["firm_weighted"].index.to_timestamp()
    axes[1].plot(cdates, capital["firm_weighted"], color=GREEN, lw=1.2, label="Firm-weighted")
    axes[1].plot(cdates, capital["revenue_weighted"], color=GOLD, lw=1.2, label="Revenue-weighted")
    axes[1].axhline(0, color="#7b8794", lw=0.6)
    axes[1].set_title(r"Capital IQ coordinates: $10\log(N_{j,t}^{CIQ}/N_{j,1993Q4}^{CIQ})$", loc="left")
    axes[1].set_ylabel("10-log-point units")
    axes[1].legend(frameon=False, ncol=2)
    for ax in axes:
        ax.grid(axis="y", color=GRID, lw=0.5)
    fig.suptitle("Competition inputs are related coordinates, not a common observed level", x=0.08, ha="left", fontsize=13, weight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    _save(fig, "competition_inputs.png")

    variables = [
        ("PPI inflation", cell.pi, "FRED PPIACO", r"$400\Delta\log P_t$"),
        ("SPF expectation", cell.epi, "Philadelphia Fed SPF", r"$100\log(1+r_t/100)$"),
        ("Negative unemployment gap", cell.x, "FRED NROU and UNRATE", r"$u_t^*-u_t$"),
        ("Inverse-markup proxy", markup.x, "Nekarda-Ramey markup", r"$\log(1.2/\mu_t)$"),
    ]
    rows = []
    for name, values, source, transform in variables:
        a = np.asarray(values, float)
        rows.append([name, source, transform, str(periods[0]), str(periods[-1]), str(len(a)), f"{a.mean():.3f}", f"{a.std(ddof=1):.3f}"])
    _write_table(
        TAB / "data_summary.tex",
        ["Series", "Source", "Transformation", "Start", "End", "$T$", "Mean", "SD"],
        rows,
        r"p{0.16\linewidth}p{0.19\linewidth}p{0.18\linewidth}cccrr",
    )


def state_figures_and_table() -> None:
    paths = pd.read_csv(BASE / "tables" / "state_paths.csv")
    periods = pd.PeriodIndex(paths.period, freq="Q")
    dates = periods.to_timestamp()
    fig, axes = plt.subplots(3, 1, figsize=(10.7, 8.0), sharex=True)
    axes[0].fill_between(dates, paths["gustavo_slow_q2.5"], paths["gustavo_slow_q97.5"], color=BLUE, alpha=0.16)
    axes[0].plot(dates, paths.gustavo_slow_mean, color=BLUE, lw=1.3, label="Slow-state median")
    anchor = np.isfinite(paths.gustavo_anchor)
    axes[0].scatter(dates[anchor], paths.loc[anchor, "gustavo_anchor"], s=14, color=INK, label="Exact annual Q4 anchor", zorder=3)
    axes[0].set_title(r"Gustavo slow state $\bar n_t$", loc="left")
    for ax, label, color in ((axes[1], "firm_weighted", GREEN), (axes[2], "revenue_weighted", GOLD)):
        mean = paths[f"{label}_cycle_mean"]
        lo = paths[f"{label}_cycle_q2.5"]
        hi = paths[f"{label}_cycle_q97.5"]
        ax.fill_between(dates, lo, hi, color=color, alpha=0.16)
        ax.plot(dates, mean, color=color, lw=1.25)
        ax.axhline(0, color="#7b8794", lw=0.6)
        ax.set_title(label.replace("_", " ").title() + r" Capital IQ cycle $\hat n_t$", loc="left")
    for ax in axes:
        ax.set_ylabel("Coordinate")
        ax.grid(axis="y", color=GRID, lw=0.5)
    axes[-1].set_xlabel("Quarter")
    fig.suptitle("Cut competition-state posterior", x=0.08, ha="left", fontsize=13, weight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    _save(fig, "state_decomposition.png")

    d = pd.read_csv(BASE / "tables" / "state_parameters.csv")
    rows = []
    for _, r in d.iterrows():
        rows.append([str(r.block).replace("_", r"\_"), str(r.variant).replace("_", r"\_"), str(r.parameter).replace("_", r"\_"), _fmt(r["mean"], r["q2.5"], r["q97.5"])])
    _write_table(TAB / "state_parameters.tex", ["Block", "Variant", "Parameter", "Posterior"], rows, "lllc")


def coefficient_tables() -> None:
    _coefficient_table("direct_only", ["alpha_b", "alpha_f", "kappa_0", "theta_CIQ"], "coeff_direct.tex")
    _coefficient_table("free_combined", ["kappa_0", "delta", "theta_CIQ"], "coeff_static.tex")
    _coefficient_table("varying_theta", ["kappa_0", "theta_0", "gamma"], "coeff_varying.tex")
    _coefficient_table("free_dynamic", ["kappa_0", "delta_1", "delta_2", "theta_0", "gamma"], "coeff_free_dynamic.tex")
    _coefficient_table("hsa_restricted_dynamic", ["kappa_0", "theta_0", "gamma", "lambda", "derived:delta_1", "derived:delta_2"], "coeff_hsa_dynamic.tex")


def _core_coefficient_table(model: str, parameters: list[str], filename: str) -> None:
    rows = []
    for parameter in parameters:
        row = [parameter.replace("_", r"\_")]
        for cycle, cell, _ in CORE_CELL_ORDER:
            _, meta = _core_fit(model, cycle, cell)
            source = meta.get("derived", {}) if parameter.startswith("derived:") else meta["coefficients"]
            key = parameter.split(":", 1)[-1]
            z = source[key]
            row.append(_fmt(z["mean"], z["q2.5"], z["q97.5"]))
        rows.append(row)
    _write_table(TAB / filename, ["Parameter", *[x[2] for x in CORE_CELL_ORDER]], rows, "lcccc")


def core_coefficient_tables() -> None:
    _core_coefficient_table("direct_only", ["alpha_b", "alpha_f", "kappa_0", "theta_CIQ"], "core_coeff_direct.tex")
    _core_coefficient_table("free_combined", ["kappa_0", "delta", "theta_CIQ"], "core_coeff_static.tex")
    _core_coefficient_table("varying_theta", ["kappa_0", "theta_0", "gamma"], "core_coeff_varying.tex")
    _core_coefficient_table("free_dynamic", ["kappa_0", "delta_1", "delta_2", "theta_0", "gamma"], "core_coeff_free_dynamic.tex")
    _core_coefficient_table("hsa_restricted_dynamic", ["kappa_0", "theta_0", "gamma", "lambda", "derived:delta_1", "derived:delta_2"], "core_coeff_hsa_dynamic.tex")

    price_order = [
        ("ppi_negative_unemployment_gap", "PPI / unemployment"),
        ("ppi_inverse_markup", "PPI / inverse markup"),
        ("core_cpi_negative_unemployment_gap", "Core CPI / unemployment"),
        ("core_cpi_inverse_markup", "Core CPI / inverse markup"),
    ]
    row_specs = [
        ("direct_only", "theta_CIQ", "M0 $\\theta$"),
        ("free_combined", "delta", "M1 $\\delta$"),
        ("free_combined", "theta_CIQ", "M1 $\\theta$"),
        ("varying_theta", "theta_0", "M2 $\\theta_0$"),
        ("varying_theta", "gamma", "M2 $\\gamma$"),
        ("free_dynamic", "delta_1", "M3 $\\delta_1$"),
        ("free_dynamic", "delta_2", "M3 $\\delta_2$"),
        ("free_dynamic", "theta_0", "M3 $\\theta_0$"),
        ("free_dynamic", "gamma", "M3 $\\gamma$"),
        ("hsa_restricted_dynamic", "theta_0", "M4 $\\theta_0$"),
        ("hsa_restricted_dynamic", "gamma", "M4 $\\gamma$"),
        ("hsa_restricted_dynamic", "lambda", "M4 $\\lambda$"),
        ("hsa_restricted_dynamic", "derived:delta_1", "M4 $\\delta_1$"),
        ("hsa_restricted_dynamic", "derived:delta_2", "M4 $\\delta_2$"),
    ]
    rows = []
    for model, parameter, label in row_specs:
        row = [label]
        for cell, _ in price_order:
            getter = _fit if cell.startswith("ppi_") else _core_fit
            _, meta = getter(model, "firm_weighted", cell)
            source = meta.get("derived", {}) if parameter.startswith("derived:") else meta["coefficients"]
            z = source[parameter.split(":", 1)[-1]]
            row.append(_fmt(z["mean"], z["q2.5"], z["q97.5"]))
        rows.append(row)
    _write_table(TAB / "cross_price_targets.tex", ["Target", *[x[1] for x in price_order]], rows, "lcccc")


def core_data_assets() -> None:
    ppi = load_nkpc_cells(load_yaml(BUNDLE / "config.yaml"))["ppi_negative_unemployment_gap"]
    cells = load_nkpc_cells(_core_config())
    core = cells["core_cpi_negative_unemployment_gap"]
    markup = cells["core_cpi_inverse_markup"]
    dates = core.periods.to_timestamp()
    fig, axes = plt.subplots(3, 1, figsize=(10.7, 7.2), sharex=True)
    axes[0].plot(dates, ppi.pi, color=BLUE, lw=1.1, label="PPI")
    axes[0].plot(dates, core.pi, color=GREEN, lw=1.4, label="Core CPI")
    axes[0].set_title("Annualized quarter-on-quarter inflation", loc="left")
    axes[0].legend(frameon=False, ncol=2)
    axes[1].plot(dates, ppi.epi, color=GOLD, lw=1.1, label="SPF GDP-price DPGDP3")
    axes[1].plot(dates, core.epi, color=PURPLE, lw=1.2, label="SPF headline-CPI CPI3")
    axes[1].set_title("Genuine one-quarter-ahead expectations", loc="left")
    axes[1].legend(frameon=False, ncol=2)
    axes[2].plot(dates, core.pi-core.epi, color=RED, lw=1.2)
    axes[2].set_title("Core CPI inflation minus headline-CPI expectation proxy", loc="left")
    for ax in axes:
        ax.axhline(0, color="#7b8794", lw=0.55);ax.grid(axis="y", color=GRID, lw=0.5);ax.set_ylabel("Annualized pp")
    axes[-1].set_xlabel("Quarter")
    fig.suptitle("PPI and Core CPI observables on the matched 1993Q2-2013Q4 sample", x=0.07, ha="left", fontsize=13, weight="bold")
    fig.tight_layout(rect=[0,0,1,.95]);_save(fig,"core_data_timeseries.png")

    rows = []
    for name, values, source, transform in [
        ("Core CPI inflation", core.pi, "FRED CPILFESL", r"$400\Delta\log P_t$"),
        ("SPF headline-CPI expectation", core.epi, "Philadelphia Fed CPI3", r"$100\log(1+r_t/100)$"),
        ("Negative unemployment gap", core.x, "FRED NROU and UNRATE", r"$u_t^*-u_t$"),
        ("Inverse-markup proxy", markup.x, "Nekarda-Ramey markup", r"$\log(1.2/\mu_t)$"),
    ]:
        a=np.asarray(values,float);rows.append([name,source,transform,str(core.periods[0]),str(core.periods[-1]),str(len(a)),f"{a.mean():.3f}",f"{a.std(ddof=1):.3f}"])
    _write_table(TAB/"core_data_summary.tex",["Series","Source","Transformation","Start","End","$T$","Mean","SD"],rows,r"p{0.18\linewidth}p{0.18\linewidth}p{0.18\linewidth}cccrr")


def core_prior_posterior() -> None:
    specs=[("direct_only","theta_CIQ","M0 direct loading",BLUE),("free_combined","delta","M1 slope interaction",GREEN),("free_combined","theta_CIQ","M1 direct loading",BLUE),("varying_theta","theta_0","M2 average loading",PURPLE),("varying_theta","gamma","M2 state interaction",RED),("hsa_restricted_dynamic","lambda","M4 proportionality",GOLD)]
    fig,axes=plt.subplots(2,3,figsize=(11.2,6.7))
    for ax,(model,parameter,title,color) in zip(axes.flat,specs):
        fit,_=_core_fit(model,"firm_weighted","core_cpi_negative_unemployment_gap");j=fit.names.index(parameter);post=fit.draws[:,:,j].reshape(-1);mean=fit.prior_mean[parameter];sd=fit.prior_sd[parameter];lo=min(np.percentile(post,.3),mean-3.2*sd);hi=max(np.percentile(post,99.7),mean+3.2*sd);x=np.linspace(lo,hi,500);ax.plot(x,norm.pdf(x,mean,sd),color="#7b8794",ls="--",lw=1.2,label="Prior");ax.plot(x,gaussian_kde(post)(x),color=color,lw=1.5,label="Posterior");ax.axvline(0,color=INK,lw=.55);ax.set_title(title);ax.grid(axis="y",color=GRID,lw=.45)
    axes[0,0].legend(frameon=False);fig.suptitle("Core CPI prior versus posterior: firm-weighted unemployment-gap cell",x=.06,ha="left",fontsize=13,weight="bold");fig.tight_layout(rect=[0,0,1,.95]);_save(fig,"core_prior_posterior.png")


def core_dynamic_paths_and_fit() -> None:
    config=_core_config();cell=load_nkpc_cells(config)["core_cpi_negative_unemployment_gap"];dates=cell.periods.to_timestamp();models=list(MODEL_LABEL);colors=["#5b6770",BLUE,PURPLE,GREEN,GOLD]
    fig,ax=plt.subplots(figsize=(10.6,4.7));ax.plot(dates,cell.pi,color=INK,lw=1.25,label="Observed Core CPI")
    for model,color in zip(models,colors):
        fit,_=_core_fit(model,"firm_weighted",cell.name);mus=[]
        for c in range(fit.draws.shape[0]):
            for d in range(0,fit.draws.shape[1],max(1,fit.draws.shape[1]//350)):
                if model in {"direct_only","free_combined"}:X,_=build_qoq_design(cell,fit.nhat_used[c,d],fit.nbar_used[c,d] if model=="free_combined" else None);mus.append(X@fit.draws[c,d])
                else:mus.append(dynamic_mu(cell,fit,c,d))
        ax.plot(dates,np.mean(mus,axis=0),color=color,lw=1.05,label=MODEL_LABEL[model])
    ax.axhline(0,color="#7b8794",lw=.55);ax.grid(axis="y",color=GRID,lw=.5);ax.set_ylabel("Annualized percentage points");ax.set_title("Core CPI: observed and posterior-mean fitted inflation",loc="left",fontsize=13,weight="bold");ax.legend(frameon=False,ncol=3,fontsize=8);fig.tight_layout();_save(fig,"core_posterior_fit.png")

    fig,axes=plt.subplots(2,1,figsize=(10.7,6.9),sharex=True)
    for model,color in zip(["varying_theta","free_dynamic","hsa_restricted_dynamic"],[PURPLE,GREEN,GOLD]):
        fit,_=_core_fit(model,"firm_weighted",cell.name);b={n:fit.draws[:,:,j] for j,n in enumerate(fit.names)};bar=fit.nbar_used;barc=bar-bar.mean(axis=2,keepdims=True);q2=barc**2-(barc**2).mean(axis=2,keepdims=True);theta=b["theta_0"][:,:,None]+b["gamma"][:,:,None]*barc
        if model=="varying_theta":kappa=np.broadcast_to(b["kappa_0"][:,:,None],bar.shape)
        elif model=="free_dynamic":kappa=b["kappa_0"][:,:,None]+b["delta_1"][:,:,None]*barc+b["delta_2"][:,:,None]*q2
        else:kappa=b["kappa_0"][:,:,None]+b["lambda"][:,:,None]*b["theta_0"][:,:,None]*barc+.5*b["lambda"][:,:,None]*b["gamma"][:,:,None]*q2
        for ax,values in zip(axes,[theta,kappa]):
            lo,mid,hi=np.percentile(values.reshape(-1,values.shape[-1]),[2.5,50,97.5],axis=0);ax.fill_between(dates,lo,hi,color=color,alpha=.10);ax.plot(dates,mid,color=color,lw=1.2,label=MODEL_LABEL[model])
    axes[0].set_title(r"Core CPI time-varying direct loading $\theta_t$",loc="left");axes[1].set_title(r"Core CPI time-varying slope $\kappa_t$",loc="left")
    for ax in axes:ax.axhline(0,color="#7b8794",lw=.6);ax.grid(axis="y",color=GRID,lw=.5);ax.set_ylabel("Coefficient");ax.legend(frameon=False,ncol=3,fontsize=8)
    axes[-1].set_xlabel("Quarter");fig.tight_layout();_save(fig,"core_dynamic_paths.png")


def core_comparison_recovery_convergence() -> None:
    comp=pd.read_csv(CORE/"tables"/"model_comparison.csv");rows=[];plot=[]
    for (cycle,cell),g in comp.groupby(["cycle","cell"]):
        base=g[g.model=="direct_only"].iloc[0]
        for model in ["free_combined","varying_theta","free_dynamic","hsa_restricted_dynamic"]:
            r=g[g.model==model].iloc[0];rows.append([cycle.replace("_",r"\_"),cell.replace("core_cpi_","").replace("_",r"\_"),MODEL_LABEL[model],f"{r.elpd_loo-base.elpd_loo:.3f}",f"{r.elpd_waic-base.elpd_waic:.3f}",f"{r.holdout_elpd-base.holdout_elpd:.3f}",f"{r.holdout_rmse-base.holdout_rmse:.3f}",f"{r.max_pareto_k:.2f}"]);plot.append((cycle,cell,model,r.holdout_elpd-base.holdout_elpd,r.elpd_loo-base.elpd_loo))
    _write_table(TAB/"core_model_comparison.tex",["Cycle","Activity","Model",r"$\Delta$LOO",r"$\Delta$WAIC",r"$\Delta$holdout",r"$\Delta$RMSE","max $k$"],rows,"lllrrrrr")
    p=pd.DataFrame(plot,columns=["cycle","cell","model","holdout","loo"]);labels=[f"{c.split('_')[0]} / {'unemp' if 'unemployment' in a else 'markup'} / {MODEL_LABEL[m]}" for c,a,m in zip(p.cycle,p.cell,p.model)];y=np.arange(len(p));fig,axes=plt.subplots(1,2,figsize=(11.2,8),sharey=True);axes[0].barh(y,p.holdout,color=np.where(p.holdout>=0,GREEN,RED),alpha=.85);axes[1].barh(y,p.loo,color=np.where(p.loo>=0,GREEN,RED),alpha=.85);axes[0].set_yticks(y,labels,fontsize=8);axes[0].invert_yaxis();axes[0].set_title("Holdout ELPD");axes[1].set_title("PSIS-LOO ELPD")
    for ax in axes:ax.axvline(0,color=INK,lw=.7);ax.grid(axis="x",color=GRID,lw=.5);ax.set_xlabel("Difference from M0")
    fig.suptitle("Core CPI model comparison",x=.06,ha="left",fontsize=13,weight="bold");fig.tight_layout(rect=[0,0,1,.95]);_save(fig,"core_model_comparison.png")

    power=pd.read_csv(CORE/"tables"/"recovery_power.csv",keep_default_na=False);static=power[(power.kind=="static")&(power["mode"]=="propagated_state")];dynamic=power[(power.kind=="dynamic")&(power.parameter=="gamma")];sorder=["null","direct_observed","slope_observed","both_observed","both_moderate","both_large"];dorder=["theta_only","gamma_small","gamma_observed_scale","gamma_moderate","gamma_large"];fig,axes=plt.subplots(1,2,figsize=(11.2,4.7));x=np.arange(len(sorder))
    for parameter,color,marker in (("delta",GREEN,"s"),("theta_CIQ",BLUE,"o")):
        z=static[static.parameter==parameter].set_index("scenario").loc[sorder];axes[0].plot(x,z.suggestive_rate,color=color,marker=marker,label=parameter)
    axes[0].set_xticks(x,[s.replace("_","\n") for s in sorder],fontsize=7);axes[0].set_title("Core CPI free-static recovery");x2=np.arange(len(dorder))
    for mode,color,marker in (("propagated_state",BLUE,"o"),("oracle_state",GREEN,"s")):
        z=dynamic[dynamic["mode"]==mode].set_index("scenario").loc[dorder];axes[1].plot(x2,z.suggestive_rate,color=color,marker=marker,label=mode.replace("_"," "))
    axes[1].set_xticks(x2,["0",".05",".10",".20",".40"]);axes[1].set_title(r"Core CPI recovery of $\gamma$")
    for ax in axes:ax.axhline(.8,color=RED,ls="--",lw=1,label="0.80 gate");ax.set_ylim(-.03,1.04);ax.set_ylabel("Suggestive recovery rate");ax.grid(axis="y",color=GRID,lw=.5);ax.legend(frameon=False,fontsize=8)
    fig.tight_layout();_save(fig,"core_recovery.png")
    rows=[]
    for scenario in ["both_observed","both_moderate","both_large"]:
        for parameter in ["delta","theta_CIQ"]:
            r=static[(static.scenario==scenario)&(static.parameter==parameter)].iloc[0];rows.append([scenario.replace("_",r"\_"),parameter.replace("_",r"\_"),f"{r.standardized_true:.2f}",f"{r.suggestive_rate:.3f}",f"{r.strong_rate:.3f}",f"{r.coverage:.3f}"])
    _write_table(TAB/"core_recovery_static.tex",["Scenario","Parameter","True effect","Suggestive","Strong","Coverage"],rows,"llrrrr")
    rows=[]
    for scenario in dorder:
        r=dynamic[(dynamic["mode"]=="propagated_state")&(dynamic.scenario==scenario)].iloc[0];rows.append([f"{r.standardized_true:.2f}",f"{r.suggestive_rate:.3f}",f"{r.strong_rate:.3f}",f"{r.coverage:.3f}",f"{r.mean_sd_ratio:.3f}"])
    _write_table(TAB/"core_recovery_dynamic.tex",[r"True $s_\gamma$","Suggestive","Strong","Coverage","Mean SD ratio"],rows,"rrrrr")

    conv=[]
    for model in MODEL_LABEL:
        for cycle,cell,label in CORE_CELL_ORDER:
            _,meta=_core_fit(model,cycle,cell);d=meta["diagnostics"];conv.append([MODEL_LABEL[model],label,f"{d['max_rhat']:.4f}",f"{d['min_bulk_ess']:.1f}"])
    _write_table(TAB/"core_convergence.tex",["Model","Cell","Max R-hat","Min bulk ESS"],conv,"llrr")
    rows=[]
    for model,params in [("direct_only",["theta_CIQ"]),("free_combined",["delta","theta_CIQ"]),("varying_theta",["theta_0","gamma"]),("free_dynamic",["delta_1","delta_2","theta_0","gamma"]),("hsa_restricted_dynamic",["theta_0","gamma","lambda"])]:
        _,iid=_core_fit(model,"firm_weighted","core_cpi_negative_unemployment_gap","iid");_,ar1=_core_fit(model,"firm_weighted","core_cpi_negative_unemployment_gap","persistent_ar1")
        for parameter in params:
            zi=iid["coefficients"][parameter];za=ar1["coefficients"][parameter];rows.append([MODEL_LABEL[model],parameter.replace("_",r"\_"),_fmt(zi["mean"],zi["q2.5"],zi["q97.5"]),_fmt(za["mean"],za["q2.5"],za["q97.5"])])
    _write_table(TAB/"core_ar1_robustness.tex",["Model","Parameter","IID","Persistent AR(1)"],rows,"llcc")


def prior_posterior_figures() -> None:
    specs = [
        ("direct_only", "theta_CIQ", "Constant theta: direct loading", BLUE),
        ("free_combined", "delta", "Free static: slope interaction", GREEN),
        ("free_combined", "theta_CIQ", "Free static: direct loading", BLUE),
        ("varying_theta", "theta_0", "Varying theta: average loading", PURPLE),
        ("varying_theta", "gamma", "Varying theta: state interaction", RED),
        ("hsa_restricted_dynamic", "lambda", "HSA dynamic: proportionality", GOLD),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(11.2, 6.7))
    for ax, (model, parameter, title, color) in zip(axes.flat, specs):
        fit, _ = _fit(model, "firm_weighted", "ppi_negative_unemployment_gap")
        j = fit.names.index(parameter)
        posterior = fit.draws[:, :, j].reshape(-1)
        mean = fit.prior_mean[parameter]
        sd = fit.prior_sd[parameter]
        lo = min(np.percentile(posterior, 0.3), mean - 3.2 * sd)
        hi = max(np.percentile(posterior, 99.7), mean + 3.2 * sd)
        x = np.linspace(lo, hi, 500)
        ax.plot(x, norm.pdf(x, mean, sd), color="#7b8794", ls="--", lw=1.2, label="Prior")
        kde = gaussian_kde(posterior)
        ax.plot(x, kde(x), color=color, lw=1.5, label="Posterior")
        ax.axvline(0, color=INK, lw=0.55)
        ax.set_title(title)
        ax.grid(axis="y", color=GRID, lw=0.45)
    axes[0, 0].legend(frameon=False)
    fig.suptitle("Prior versus posterior: primary firm-weighted unemployment-gap cell", x=0.06, ha="left", fontsize=13, weight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    _save(fig, "prior_posterior_primary.png")

    rows = []
    labels = []
    for model, parameters in [
        ("direct_only", ["theta_CIQ"]),
        ("free_combined", ["delta", "theta_CIQ"]),
        ("varying_theta", ["theta_0", "gamma"]),
        ("free_dynamic", ["delta_1", "delta_2", "theta_0", "gamma"]),
        ("hsa_restricted_dynamic", ["theta_0", "gamma", "lambda"]),
    ]:
        for parameter in parameters:
            values = []
            for cycle, cell, _ in CELL_ORDER:
                _, meta = _fit(model, cycle, cell)
                values.append(meta["coefficients"][parameter]["posterior_prior_sd_ratio"])
            rows.append(values)
            labels.append(f"{MODEL_LABEL[model]}: {parameter}")
    a = np.asarray(rows)
    fig, ax = plt.subplots(figsize=(9.8, 7.0))
    im = ax.imshow(a, aspect="auto", cmap="YlGnBu_r", vmin=0.2, vmax=1.05)
    ax.set_xticks(range(4), [x[2].replace(" / ", "\n") for x in CELL_ORDER])
    ax.set_yticks(range(len(labels)), labels)
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            ax.text(j, i, f"{a[i,j]:.2f}", ha="center", va="center", fontsize=8, color="white" if a[i,j] > 0.72 else INK)
    fig.colorbar(im, ax=ax, label="Posterior SD / prior SD")
    ax.set_title("How much each coefficient learns relative to its prior", loc="left", fontsize=13, weight="bold")
    fig.tight_layout()
    _save(fig, "learning_heatmap.png")


def dynamic_paths_and_fit() -> None:
    models = ["direct_only", "free_combined", "varying_theta", "free_dynamic", "hsa_restricted_dynamic"]
    colors = ["#5b6770", BLUE, PURPLE, GREEN, GOLD]
    cell_name = "ppi_negative_unemployment_gap"
    config = load_yaml(BUNDLE / "config.yaml")
    cell = load_nkpc_cells(config)[cell_name]
    dates = cell.periods.to_timestamp()
    fig, ax = plt.subplots(figsize=(10.6, 4.7))
    ax.plot(dates, cell.pi, color=INK, lw=1.25, label="Observed PPI inflation")
    for model, color in zip(models, colors):
        fit, _ = _fit(model, "firm_weighted", cell_name)
        mus = []
        for c in range(fit.draws.shape[0]):
            for d in range(0, fit.draws.shape[1], max(1, fit.draws.shape[1] // 350)):
                if model in {"direct_only", "free_combined"}:
                    use_bar = fit.nbar_used[c, d] if model == "free_combined" else None
                    X, _ = build_qoq_design(cell, fit.nhat_used[c, d], use_bar)
                    mus.append(X @ fit.draws[c, d])
                else:
                    mus.append(dynamic_mu(cell, fit, c, d))
        ax.plot(dates, np.mean(mus, axis=0), color=color, lw=1.05, label=MODEL_LABEL[model])
    ax.axhline(0, color="#7b8794", lw=0.55)
    ax.grid(axis="y", color=GRID, lw=0.5)
    ax.set_ylabel("Annualized percentage points")
    ax.set_title("Observed and posterior-mean fitted inflation", loc="left", fontsize=13, weight="bold")
    ax.legend(frameon=False, ncol=3, fontsize=8)
    fig.tight_layout()
    _save(fig, "posterior_fit.png")

    dynmodels = ["varying_theta", "free_dynamic", "hsa_restricted_dynamic"]
    dyncolors = [PURPLE, GREEN, GOLD]
    fig, axes = plt.subplots(2, 1, figsize=(10.7, 6.9), sharex=True)
    for model, color in zip(dynmodels, dyncolors):
        fit, _ = _fit(model, "firm_weighted", cell_name)
        b = {n: fit.draws[:, :, j] for j, n in enumerate(fit.names)}
        bar = fit.nbar_used
        barc = bar - bar.mean(axis=2, keepdims=True)
        q2 = barc**2 - (barc**2).mean(axis=2, keepdims=True)
        theta = b["theta_0"][:, :, None] + b["gamma"][:, :, None] * barc
        if model == "varying_theta":
            kappa = np.broadcast_to(b["kappa_0"][:, :, None], bar.shape)
        elif model == "free_dynamic":
            kappa = b["kappa_0"][:, :, None] + b["delta_1"][:, :, None] * barc + b["delta_2"][:, :, None] * q2
        else:
            kappa = b["kappa_0"][:, :, None] + b["lambda"][:, :, None] * b["theta_0"][:, :, None] * barc + 0.5 * b["lambda"][:, :, None] * b["gamma"][:, :, None] * q2
        for ax, values in zip(axes, [theta, kappa]):
            flat = values.reshape(-1, values.shape[-1])
            lo, mid, hi = np.percentile(flat, [2.5, 50, 97.5], axis=0)
            ax.fill_between(dates, lo, hi, color=color, alpha=0.10)
            ax.plot(dates, mid, color=color, lw=1.2, label=MODEL_LABEL[model])
    axes[0].set_title(r"Time-varying direct loading $\theta_t$", loc="left")
    axes[1].set_title(r"Time-varying Phillips-curve slope $\kappa_t$", loc="left")
    for ax in axes:
        ax.axhline(0, color="#7b8794", lw=0.6)
        ax.grid(axis="y", color=GRID, lw=0.5)
        ax.set_ylabel("Coefficient")
        ax.legend(frameon=False, ncol=3, fontsize=8)
    axes[-1].set_xlabel("Quarter")
    fig.suptitle("Dynamic coefficients: primary firm-weighted unemployment-gap cell", x=0.08, ha="left", fontsize=13, weight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    _save(fig, "dynamic_paths.png")


def comparison_and_recovery() -> None:
    comp = pd.read_csv(DYNAMIC / "tables" / "model_comparison.csv")
    rows = []
    plot_rows = []
    for (cycle, cell), g in comp.groupby(["cycle", "cell"]):
        base = g[g.model == "constant_theta"].iloc[0]
        for model in ["varying_theta", "free_dynamic", "hsa_restricted_dynamic"]:
            r = g[g.model == model].iloc[0]
            row = [
                cycle.replace("_", r"\_"),
                cell.replace("ppi_", "").replace("_", r"\_"),
                MODEL_LABEL[model],
                f"{r.elpd_loo-base.elpd_loo:.3f}",
                f"{r.elpd_waic-base.elpd_waic:.3f}",
                f"{r.holdout_elpd-base.holdout_elpd:.3f}",
                f"{r.holdout_rmse-base.holdout_rmse:.3f}",
                f"{r.max_pareto_k:.2f}",
            ]
            rows.append(row)
            plot_rows.append((cycle, cell, model, r.holdout_elpd - base.holdout_elpd, r.elpd_loo - base.elpd_loo))
    _write_table(TAB / "model_comparison.tex", ["Cycle", "Activity", "Model", r"$\Delta$LOO", r"$\Delta$WAIC", r"$\Delta$holdout", r"$\Delta$RMSE", "max $k$"], rows, "lllrrrrr")

    p = pd.DataFrame(plot_rows, columns=["cycle", "cell", "model", "holdout", "loo"])
    labels = [f"{c.split('_')[0]} / {'unemp' if 'unemployment' in a else 'markup'} / {MODEL_LABEL[m]}" for c, a, m in zip(p.cycle, p.cell, p.model)]
    y = np.arange(len(p))
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 7.2), sharey=True)
    axes[0].barh(y, p.holdout, color=np.where(p.holdout >= 0, GREEN, RED), alpha=0.85)
    axes[1].barh(y, p.loo, color=np.where(p.loo >= 0, GREEN, RED), alpha=0.85)
    axes[0].set_yticks(y, labels, fontsize=8)
    axes[0].invert_yaxis()
    axes[0].set_title("2010Q1-2013Q4 holdout ELPD")
    axes[1].set_title("PSIS-LOO ELPD (descriptive)")
    for ax in axes:
        ax.axvline(0, color=INK, lw=0.7)
        ax.grid(axis="x", color=GRID, lw=0.5)
        ax.set_xlabel("Difference from constant theta")
    fig.suptitle("Dynamic model comparison", x=0.06, ha="left", fontsize=13, weight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    _save(fig, "model_comparison.png")

    staged = pd.read_csv(STAGED / "tables" / "recovery_power.csv", keep_default_na=False)
    primary = staged[(staged.activity == "ppi_negative_unemployment_gap") & (staged.error_model == "iid") & (staged["mode"] == "propagated_state")]
    order = ["null", "direct_observed", "slope_observed", "both_observed", "both_moderate", "both_large"]
    dynamic = pd.read_csv(DYNAMIC / "tables" / "recovery_power.csv")
    gd = dynamic[(dynamic.parameter == "gamma")]
    dorder = ["theta_only", "gamma_small", "gamma_observed_scale", "gamma_moderate", "gamma_large"]
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.7))
    x = np.arange(len(order))
    for parameter, color, marker in (("delta", GREEN, "s"), ("theta_CIQ", BLUE, "o")):
        z = primary[primary.parameter == parameter].set_index("scenario").loc[order]
        axes[0].plot(x, z.suggestive_rate, color=color, marker=marker, label=parameter)
    axes[0].set_xticks(x, [s.replace("_", "\n") for s in order], fontsize=7)
    axes[0].set_title("Free-static joint recovery")
    x2 = np.arange(len(dorder))
    for mode, color, marker in (("propagated_state", BLUE, "o"), ("oracle_state", GREEN, "s")):
        z = gd[gd["mode"] == mode].set_index("scenario").loc[dorder]
        axes[1].plot(x2, z.suggestive_rate, color=color, marker=marker, label=mode.replace("_", " "))
    axes[1].set_xticks(x2, ["0", ".05", ".10", ".20", ".40"])
    axes[1].set_title(r"Varying-theta recovery by injected $s_\gamma$")
    for ax in axes:
        ax.axhline(0.8, color=RED, ls="--", lw=1, label="0.80 gate")
        ax.set_ylim(-0.03, 1.04)
        ax.set_ylabel("Suggestive recovery rate")
        ax.grid(axis="y", color=GRID, lw=0.5)
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle("Simulation recovery on the actual 83-quarter design", x=0.06, ha="left", fontsize=13, weight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    _save(fig, "recovery.png")

    rows = []
    for scenario in ["both_observed", "both_moderate", "both_large"]:
        for parameter in ["delta", "theta_CIQ"]:
            r = primary[(primary.scenario == scenario) & (primary.parameter == parameter)].iloc[0]
            rows.append([scenario.replace("_", r"\_"), parameter.replace("_", r"\_"), f"{r.standardized_true:.2f}", f"{r.suggestive_rate:.3f}", f"{r.strong_rate:.3f}", f"{r.coverage:.3f}"])
    _write_table(TAB / "recovery_static.tex", ["Scenario", "Parameter", "True effect", "Suggestive", "Strong", "Coverage"], rows, "llrrrr")
    rows = []
    for scenario in dorder:
        r = gd[(gd["mode"] == "propagated_state") & (gd.scenario == scenario)].iloc[0]
        rows.append([f"{r.standardized_true:.2f}", f"{r.suggestive_rate:.3f}", f"{r.strong_rate:.3f}", f"{r.coverage:.3f}", f"{r.mean_sd_ratio:.3f}"])
    _write_table(TAB / "recovery_dynamic.tex", [r"True $s_\gamma$", "Suggestive", "Strong", "Coverage", "Mean SD ratio"], rows, "rrrrr")


def identification_geometry(config: dict) -> None:
    cells = load_nkpc_cells(config)
    rows = []
    primary_matrix = None
    primary_labels = None
    posterior_rows = []
    for cycle, cell_name, label in CELL_ORDER:
        path = BASE / "draws" / "cycle" / f"{cycle}.npz"
        meta = _summary_from_json(path.with_suffix(".json"))
        state = load_cycle(path, meta)
        cell = cells[cell_name]
        pos = pd.PeriodIndex(state.periods, freq="Q").get_indexer(cell.periods)
        bar = np.median(state.nbar_used[:, :, pos], axis=(0, 1))
        hat = np.median(state.nhat[:, :, pos], axis=(0, 1))
        barc = bar - bar.mean()
        q2 = barc**2 - np.mean(barc**2)
        Z = np.column_stack([-hat, -barc * hat, barc * cell.x, q2 * cell.x])
        names = [r"$-\hat n$", r"$-\bar n^c\hat n$", r"$\bar n^c x$", r"$q^{(2)}x$"]
        Zs = (Z - Z.mean(axis=0)) / Z.std(axis=0, ddof=1)
        corr = np.corrcoef(Zs, rowvar=False)
        cond_varying = np.linalg.cond(Zs[:, :2])
        cond_free = np.linalg.cond(Zs)
        maxcorr = np.max(np.abs(corr - np.eye(4)))
        rows.append([label, f"{corr[0,1]:.3f}", f"{corr[2,3]:.3f}", f"{maxcorr:.3f}", f"{cond_varying:.2f}", f"{cond_free:.2f}"])
        if cycle == "firm_weighted" and cell_name == "ppi_negative_unemployment_gap":
            primary_matrix = corr
            primary_labels = names

        fit, _ = _fit("free_dynamic", cycle, cell_name)
        idx = [fit.names.index(p) for p in ["delta_1", "delta_2", "theta_0", "gamma"]]
        pcorr = np.corrcoef(fit.draws[:, :, idx].reshape(-1, 4), rowvar=False)
        posterior_rows.append([label, f"{np.max(np.abs(pcorr - np.eye(4))):.3f}"])

    _write_table(TAB / "identification_geometry.tex", ["Cell", r"Corr($\theta_0,\gamma$ regressors)", r"Corr($\delta_1,\delta_2$ regressors)", "Max regressor corr.", "Cond. varying", "Cond. free dyn."], rows, "lrrrrr")
    _write_table(TAB / "posterior_geometry.tex", ["Cell", "Max target posterior correlation"], posterior_rows, "lr")

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.4))
    im = axes[0].imshow(primary_matrix, cmap="RdBu_r", vmin=-1, vmax=1)
    axes[0].set_xticks(range(4), primary_labels)
    axes[0].set_yticks(range(4), primary_labels)
    for i in range(4):
        for j in range(4):
            axes[0].text(j, i, f"{primary_matrix[i,j]:.2f}", ha="center", va="center", fontsize=9)
    fig.colorbar(im, ax=axes[0], shrink=0.82)
    axes[0].set_title("Regressor correlation")

    fit, _ = _fit("free_dynamic", "firm_weighted", "ppi_negative_unemployment_gap")
    target = ["delta_1", "delta_2", "theta_0", "gamma"]
    idx = [fit.names.index(p) for p in target]
    pcorr = np.corrcoef(fit.draws[:, :, idx].reshape(-1, 4), rowvar=False)
    im2 = axes[1].imshow(pcorr, cmap="RdBu_r", vmin=-1, vmax=1)
    axes[1].set_xticks(range(4), target)
    axes[1].set_yticks(range(4), target)
    for i in range(4):
        for j in range(4):
            axes[1].text(j, i, f"{pcorr[i,j]:.2f}", ha="center", va="center", fontsize=9)
    fig.colorbar(im2, ax=axes[1], shrink=0.82)
    axes[1].set_title("Posterior coefficient correlation")
    fig.suptitle("Identification geometry: firm-weighted unemployment-gap cell", x=0.06, ha="left", fontsize=13, weight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    _save(fig, "identification_geometry.png")


def convergence_and_robustness() -> None:
    rows = []
    points = []
    for model in ["direct_only", "free_combined", "varying_theta", "free_dynamic", "hsa_restricted_dynamic"]:
        for cycle, cell, label in CELL_ORDER:
            _, meta = _fit(model, cycle, cell)
            d = meta["diagnostics"]
            rows.append([MODEL_LABEL[model], label, f"{d['max_rhat']:.4f}", f"{d['min_bulk_ess']:.1f}"])
            for name in d["rhat"]:
                points.append((MODEL_LABEL[model], d["rhat"][name], d["ess_bulk"][name]))
    _write_table(TAB / "convergence.tex", ["Model", "Cell", "Max R-hat", "Min bulk ESS"], rows, "llrr")
    p = pd.DataFrame(points, columns=["model", "rhat", "ess"])
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    for model, color in zip(MODEL_LABEL.values(), ["#5b6770", BLUE, PURPLE, GREEN, GOLD]):
        z = p[p.model == model]
        ax.scatter(z.rhat, z.ess, s=24, alpha=0.75, label=model, color=color)
    ax.axvline(1.05, color=RED, ls="--", lw=1)
    ax.axhline(400, color=RED, ls="--", lw=1)
    ax.set_xlim(0.998, 1.053)
    ax.set_yscale("log")
    ax.set_xlabel("Rank-normalized R-hat")
    ax.set_ylabel("Bulk ESS (log scale)")
    ax.grid(color=GRID, lw=0.45)
    ax.legend(frameon=False, ncol=3, fontsize=8)
    ax.set_title("Observed-fit convergence diagnostics", loc="left", fontsize=13, weight="bold")
    fig.tight_layout()
    _save(fig, "convergence.png")

    rows = []
    for model, params in [
        ("varying_theta", ["theta_0", "gamma"]),
        ("free_dynamic", ["delta_1", "delta_2", "theta_0", "gamma"]),
        ("hsa_restricted_dynamic", ["theta_0", "gamma", "lambda"]),
    ]:
        _, iid = _fit(model, "firm_weighted", "ppi_negative_unemployment_gap", "iid")
        _, ar1 = _fit(model, "firm_weighted", "ppi_negative_unemployment_gap", "persistent_ar1")
        for parameter in params:
            zi = iid["coefficients"][parameter]
            za = ar1["coefficients"][parameter]
            rows.append([MODEL_LABEL[model], parameter.replace("_", r"\_"), _fmt(zi["mean"], zi["q2.5"], zi["q97.5"]), _fmt(za["mean"], za["q2.5"], za["q97.5"])])
    _write_table(TAB / "ar1_robustness.tex", ["Model", "Parameter", "IID", "Persistent AR(1)"], rows, "llcc")


def oil_control_assets() -> None:
    """Build the prespecified oil-control extension assets from saved full fits."""
    ppi_config = load_yaml(BUNDLE / "config.yaml")
    ppi_cells = load_nkpc_cells(ppi_config)
    core_cells = load_nkpc_cells(_core_config())
    ppi = ppi_cells["ppi_negative_unemployment_gap"]
    core = core_cells["core_cpi_negative_unemployment_gap"]
    oil, oil_meta = load_oil_controls(ppi.periods)
    dates = ppi.periods.to_timestamp()

    fig, axes = plt.subplots(3, 1, figsize=(10.7, 7.4), sharex=True)
    axes[0].plot(dates, oil[:, 0], color=GOLD, lw=1.25)
    axes[0].set_title(r"Real WTI/CPI oil-price change $q_t^o=400\Delta\log R_t^o$", loc="left")
    axes[1].plot(dates, ppi.pi, color=BLUE, lw=1.25)
    axes[1].set_title(rf"PPI inflation; corr$(\pi_t,q_t^o)={np.corrcoef(ppi.pi,oil[:,0])[0,1]:.3f}$", loc="left")
    axes[2].plot(dates, core.pi, color=GREEN, lw=1.25)
    axes[2].set_title(rf"Core CPI inflation; corr$(\pi_t,q_t^o)={np.corrcoef(core.pi,oil[:,0])[0,1]:.3f}$", loc="left")
    for ax in axes:
        ax.axhline(0, color="#7b8794", lw=.55);ax.grid(axis="y", color=GRID, lw=.5);ax.set_ylabel("Annualized pp")
    axes[-1].set_xlabel("Quarter")
    fig.suptitle("Prespecified oil control and the two inflation outcomes", x=.07, ha="left", fontsize=13, weight="bold")
    fig.tight_layout(rect=[0,0,1,.95]);_save(fig,"oil_timeseries.png")

    # Prior/posterior comparison for the two oil coefficients in each outcome.
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 6.5))
    for row, (price, cell, label, color) in enumerate([
        ("ppi", "ppi_negative_unemployment_gap", "PPI", BLUE),
        ("core_cpi", "core_cpi_negative_unemployment_gap", "Core CPI", GREEN),
    ]):
        fit, _ = _oil_fit(price, "direct_only", "firm_weighted", cell)
        for col, parameter in enumerate(["beta_oil_0", "beta_oil_1"]):
            ax=axes[row,col];post=fit.draws[:,:,fit.names.index(parameter)].reshape(-1);pm=fit.prior_mean[parameter];ps=fit.prior_sd[parameter]
            lo=min(np.percentile(post,.2),pm-3.1*ps);hi=max(np.percentile(post,99.8),pm+3.1*ps);x=np.linspace(lo,hi,500)
            ax.plot(x,norm.pdf(x,pm,ps),color="#7b8794",ls="--",lw=1.1,label="Prior")
            ax.plot(x,gaussian_kde(post)(x),color=color,lw=1.45,label="Posterior")
            ax.axvline(0,color=INK,lw=.55);ax.grid(axis="y",color=GRID,lw=.45);ax.set_title(f"{label}: {'current' if col==0 else 'lag 1'} oil")
    axes[0,0].legend(frameon=False);fig.suptitle("Oil-control prior versus posterior in M0",x=.06,ha="left",fontsize=13,weight="bold");fig.tight_layout(rect=[0,0,1,.94]);_save(fig,"oil_prior_posterior.png")

    cells = [
        ("ppi", "ppi_negative_unemployment_gap", "PPI / unemployment"),
        ("ppi", "ppi_inverse_markup", "PPI / inverse markup"),
        ("core_cpi", "core_cpi_negative_unemployment_gap", "Core CPI / unemployment"),
        ("core_cpi", "core_cpi_inverse_markup", "Core CPI / inverse markup"),
    ]
    row_specs = [
        ("direct_only","beta_oil_0",r"M0 $\beta_{o,0}$"),("direct_only","beta_oil_1",r"M0 $\beta_{o,1}$"),
        ("direct_only","theta_CIQ",r"M0 $\theta$"),("free_combined","delta",r"M1 $\delta$"),
        ("free_combined","theta_CIQ",r"M1 $\theta$"),("varying_theta","theta_0",r"M2 $\theta_0$"),
        ("varying_theta","gamma",r"M2 $\gamma$"),("hsa_restricted_dynamic","lambda",r"M4 $\lambda$"),
    ]
    rows=[]
    for model,parameter,label in row_specs:
        row=[label]
        for price,cell,_ in cells:
            _,meta=_oil_fit(price,model,"firm_weighted",cell);z=meta["coefficients"][parameter];row.append(_fmt(z["mean"],z["q2.5"],z["q97.5"]))
        rows.append(row)
    _write_table(TAB/"oil_coefficients.tex",["Target",*[x[2] for x in cells]],rows,"lcccc")

    # Direct no-oil versus oil comparison in the primary unemployment-gap cells.
    compare_specs=[("direct_only","theta_CIQ",r"M0 $\theta$"),("free_combined","delta",r"M1 $\delta$"),("free_combined","theta_CIQ",r"M1 $\theta$"),("varying_theta","gamma",r"M2 $\gamma$"),("hsa_restricted_dynamic","lambda",r"M4 $\lambda$")]
    rows=[]
    for model,parameter,label in compare_specs:
        row=[label]
        for price,cell in [("ppi","ppi_negative_unemployment_gap"),("core_cpi","core_cpi_negative_unemployment_gap")]:
            getter=_fit if price=="ppi" else _core_fit;_,base=getter(model,"firm_weighted",cell);_,controlled=_oil_fit(price,model,"firm_weighted",cell)
            zb=base["coefficients"][parameter];zo=controlled["coefficients"][parameter]
            row.extend([_fmt(zb["mean"],zb["q2.5"],zb["q97.5"]),_fmt(zo["mean"],zo["q2.5"],zo["q97.5"])])
        rows.append(row)
    _write_table(TAB/"oil_no_oil_primary.tex",["Target","PPI: no oil","PPI: oil","Core: no oil","Core: oil"],rows,"lcccc")

    # Forest plots for the same comparison across four cells.
    panels=[("direct_only","theta_CIQ",r"M0 direct loading $\theta$"),("free_combined","delta",r"M1 slope interaction $\delta$"),("free_combined","theta_CIQ",r"M1 direct loading $\theta$"),("varying_theta","gamma",r"M2 time-variation $\gamma$")]
    fig,axes=plt.subplots(2,2,figsize=(11.2,7.2))
    for ax,(model,parameter,title) in zip(axes.flat,panels):
        y=np.arange(len(cells))
        for offset,use_oil,color,marker,label in [(-.12,False,"#7b8794","o","No oil"),(.12,True,BLUE,"s","Oil control")]:
            means=[];los=[];his=[]
            for price,cell,_ in cells:
                if use_oil:_,meta=_oil_fit(price,model,"firm_weighted",cell)
                else:_,meta=(_fit if price=="ppi" else _core_fit)(model,"firm_weighted",cell)
                z=meta["coefficients"][parameter];means.append(z["mean"]);los.append(z["mean"]-z["q2.5"]);his.append(z["q97.5"]-z["mean"])
            ax.errorbar(means,y+offset,xerr=np.vstack([los,his]),fmt=marker,color=color,ms=4,capsize=2,lw=1,label=label)
        ax.axvline(0,color=INK,lw=.65);ax.set_yticks(y,[x[2] for x in cells],fontsize=8);ax.invert_yaxis();ax.grid(axis="x",color=GRID,lw=.45);ax.set_title(title)
    axes[0,0].legend(frameon=False);fig.suptitle("Competition coefficients before and after oil control",x=.06,ha="left",fontsize=13,weight="bold");fig.tight_layout(rect=[0,0,1,.94]);_save(fig,"oil_parameter_comparison.png")

    comparison=pd.read_csv(OIL/"tables"/"oil_vs_no_oil.csv")
    comparison=comparison[(comparison.cycle=="firm_weighted")]
    rows=[]
    fig,axes=plt.subplots(2,2,figsize=(11.2,7.0),sharex=False)
    for ax,(price,cell,label) in zip(axes.flat,cells):
        g=comparison[(comparison.price==price)&(comparison.cell==cell)].set_index("model").loc[list(MODEL_LABEL)]
        x=np.arange(len(g));ax.bar(x-.18,g.delta_elpd_waic_oil_minus_baseline,width=.36,color=BLUE,label=r"$\Delta$WAIC ELPD");ax.bar(x+.18,g.delta_holdout_elpd_oil_minus_baseline,width=.36,color=GOLD,label=r"$\Delta$holdout ELPD")
        ax.axhline(0,color=INK,lw=.65);ax.set_xticks(x,[f"M{i}" for i in range(5)]);ax.grid(axis="y",color=GRID,lw=.45);ax.set_title(label)
        for model,r in g.iterrows():rows.append([label,MODEL_LABEL[model],f"{r.delta_elpd_waic_oil_minus_baseline:.3f}",f"{r.delta_holdout_elpd_oil_minus_baseline:.3f}",f"{r.delta_holdout_rmse_oil_minus_baseline:.3f}",f"{r.max_pareto_k_oil:.2f}"])
    axes[0,0].legend(frameon=False,fontsize=8);fig.suptitle("Oil-control predictive differences from the no-oil specification",x=.06,ha="left",fontsize=13,weight="bold");fig.tight_layout(rect=[0,0,1,.94]);_save(fig,"oil_prediction.png")
    _write_table(TAB/"oil_prediction.tex",["Cell","Model",r"$\Delta$WAIC",r"$\Delta$holdout",r"$\Delta$RMSE","max $k$"],rows,"llrrrr")

    power=pd.read_csv(OIL/"tables"/"recovery_power.csv",keep_default_na=False);rows=[]
    for price in ["ppi","core_cpi"]:
        for parameter in ["delta","theta_CIQ"]:
            r=power[(power.price==price)&(power.kind=="static")&(power["mode"]=="propagated_state")&(power.scenario=="both_observed")&(power.parameter==parameter)].iloc[0]
            rows.append([price.replace("core_cpi","Core CPI").upper() if price=="ppi" else "Core CPI",parameter.replace("theta_CIQ",r"$\theta$").replace("delta",r"$\delta$"),f"{r.standardized_true:.2f}",f"{r.suggestive_rate:.2f}",f"{r.strong_rate:.2f}",f"{r.coverage:.2f}"])
        r=power[(power.price==price)&(power.kind=="dynamic")&(power["mode"]=="propagated_state")&(power.scenario=="gamma_observed_scale")&(power.parameter=="gamma")].iloc[0]
        rows.append(["PPI" if price=="ppi" else "Core CPI",r"$\gamma$",f"{r.standardized_true:.2f}",f"{r.suggestive_rate:.2f}",f"{r.strong_rate:.2f}",f"{r.coverage:.2f}"])
    _write_table(TAB/"oil_recovery.tex",["Outcome","Parameter","True effect","Suggestive","Strong","Coverage"],rows,"llrrrr")

    rows=[]
    for price,cell in [("ppi","ppi_negative_unemployment_gap"),("core_cpi","core_cpi_negative_unemployment_gap")]:
        for model in MODEL_LABEL:
            _,meta=_oil_fit(price,model,"firm_weighted",cell);d=meta["diagnostics"];rows.append(["PPI" if price=="ppi" else "Core CPI",MODEL_LABEL[model],f"{d['max_rhat']:.4f}",f"{d['min_bulk_ess']:.1f}"])
    _write_table(TAB/"oil_convergence.tex",["Outcome","Model","Max R-hat","Min bulk ESS"],rows,"llrr")

    rows=[["Oil control","FRED-derived real WTI/CPI index",r"$400\Delta\log R_t^o$",str(ppi.periods[0]),str(ppi.periods[-1]),str(len(ppi.periods)),f"{oil[:,0].mean():.3f}",f"{oil[:,0].std(ddof=1):.3f}"],["Oil control, lag 1","Same series",r"$q_{t-1}^o$",str(ppi.periods[0]),str(ppi.periods[-1]),str(len(ppi.periods)),f"{oil[:,1].mean():.3f}",f"{oil[:,1].std(ddof=1):.3f}"]]
    _write_table(TAB/"oil_data_summary.tex",["Series","Source","Transformation","Start","End","$T$","Mean","SD"],rows,r"p{0.15\linewidth}p{0.25\linewidth}p{0.16\linewidth}cccrr")


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    TAB.mkdir(parents=True, exist_ok=True)
    config = load_yaml(BUNDLE / "config.yaml")
    data_figures_and_table(config)
    state_figures_and_table()
    coefficient_tables()
    core_data_assets()
    core_coefficient_tables()
    prior_posterior_figures()
    core_prior_posterior()
    dynamic_paths_and_fit()
    core_dynamic_paths_and_fit()
    comparison_and_recovery()
    core_comparison_recovery_convergence()
    identification_geometry(config)
    convergence_and_robustness()
    oil_control_assets()
    print(f"wrote report assets under {HERE}")


if __name__ == "__main__":
    main()
