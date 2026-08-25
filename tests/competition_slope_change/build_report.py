"""Build the English equation-first report from structured saved results."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys
import textwrap

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd

ROOT=next(p for p in Path(__file__).resolve().parents if (p/"pyproject.toml").exists())
sys.path[:0]=[str(ROOT),str(ROOT/"src"),str(ROOT/"tests")]
from tests import _bootstrap  # noqa:F401,E402

BUNDLE=Path(__file__).resolve().parent
INK="#172127"; MUTED="#5A646B"; BLUE="#235F85"; RED="#9A403D"; GREEN="#3A765B"; GRID="#CBD0D3"


def _style():
    for name in ("STIXGeneral.otf","STIXGeneralItalic.otf","STIXGeneralBol.otf"):
        path=Path("/System/Library/Fonts/Supplemental")/name
        if path.exists(): font_manager.fontManager.addfont(str(path))
    plt.rcParams.update({"font.family":"STIXGeneral","mathtext.fontset":"stix","font.size":9.5,
        "pdf.fonttype":42,"ps.fonttype":42,"axes.unicode_minus":False,"figure.facecolor":"white",
        "axes.facecolor":"white","axes.spines.top":False,"axes.spines.right":False})


def _page(pdf,title,subtitle,n):
    fig=plt.figure(figsize=(8.5,11));fig.text(.065,.958,title,fontsize=16,weight="bold",color=INK,va="top")
    fig.text(.065,.928,subtitle,fontsize=9.3,color=MUTED,va="top");fig.text(.935,.025,str(n),ha="right",fontsize=8,color=MUTED)
    return fig


def _section(fig,y,title):
    fig.text(.075,y,title,fontsize=11.4,weight="bold",color=INK,va="top");return y-.034


def _para(fig,y,text,width=80,size=9.2,color=INK,x=.085):
    wrapped=textwrap.fill(text,width=width);lines=wrapped.count("\n")+1
    fig.text(x,y,wrapped,fontsize=size,color=color,va="top",linespacing=1.35)
    return y-.0205*lines-.011


def _bullet(fig,y,text,width=75,color=INK):
    wrapped=textwrap.fill(text,width=width,subsequent_indent="   ");lines=wrapped.count("\n")+1
    fig.text(.09,y,"- "+wrapped,fontsize=9.1,color=color,va="top",linespacing=1.35)
    return y-.0205*lines-.006


def _eq(fig,y,text,size=12.4,x=.10):
    fig.text(x,y,text,fontsize=size,color=INK,va="top");return y-.048


def _table(fig,bbox,columns,rows,widths=None,size=8.0,left=(0,)):
    ax=fig.add_axes(bbox);ax.axis("off")
    tab=ax.table(cellText=rows,colLabels=columns,cellLoc="center",colLoc="center",colWidths=widths,bbox=[0,0,1,1])
    tab.auto_set_font_size(False);tab.set_fontsize(size)
    for (r,c),cell in tab.get_celld().items():
        cell.set_edgecolor("#858D92");cell.set_linewidth(.45);cell.PAD=.035
        cell.set_facecolor("#EEF0F1" if r==0 else "white")
        if r==0:cell.get_text().set_weight("bold")
        if c in left:cell.get_text().set_ha("left")
    return tab


def _fmt(row):
    return f"{row['mean']:.3f}\n[{row['q2.5']:.3f}, {row['q97.5']:.3f}]"


def _cell_label(cell):
    return {"ppi_inverse_markup":"PPI x inverse markup","ppi_negative_unemployment_gap":"PPI x unemployment gap",
            "core_cpi_inverse_markup":"Core CPI x inverse markup","core_cpi_negative_unemployment_gap":"Core CPI x unemployment gap"}.get(cell,cell)


def _load(profile):
    out=BUNDLE/"results"/profile
    return (out,json.loads((out/"manifest.json").read_text()),
            pd.read_csv(out/"tables"/"state_identification.csv"),
            pd.read_csv(out/"tables"/"coefficients.csv"),
            pd.read_csv(out/"tables"/"economic_quantities.csv"),
            pd.read_csv(out/"tables"/"competition_paths.csv"),
            pd.read_csv(out/"tables"/"kappa_paths.csv"),
            pd.read_csv(out/"tables"/"measurement_inputs.csv"))


def _results_markdown(profile,manifest,state,coef,econ):
    baseline=state[(state.variant=="ar2_baseline")].set_index("parameter")
    gate=manifest["gate"]; lines=["# Competition-linked structural slope change: recorded result","",
        f"Profile: `{profile}`  ",f"Revision: `{manifest['revision']}`  ",
        f"Inferential status: **{'NOT FOR INFERENCE (smoke)' if profile=='smoke' else ('computational gate passed' if gate['passed'] else 'full run failed its computational gate')}**","",
        "This file is generated from `manifest.json` and CSV results. Numerical values are not manually transcribed.","",
        "## Competition-only state","",
        "```math","c_t^{obs}=\\bar c_t+\\hat c_t,\\qquad \\sigma_{\\bar c}^2=\\omega\\tau^2,\\quad \\sigma_{\\hat c}^2=(1-\\omega)\\tau^2.","```","",
        "| Quantity | Posterior mean | 95% interval |","|---|---:|---:|"]
    for p in ("omega","tau","slow_innovation_variance","cycle_innovation_variance","damping_or_rho","cycle_period"):
        if p in baseline.index:
            r=baseline.loc[p];lines.append(f"| `{p}` | {r['mean']:.4f} | [{r['q2.5']:.4f}, {r['q97.5']:.4f}] |")
    lines.extend(["","### State-law and omega-prior sensitivity","",
        "| Variant | omega | Maximum R-hat | Minimum bulk ESS |","|---|---:|---:|---:|"])
    for variant in state.variant.unique():
        block=state[state.variant==variant].set_index("parameter")
        r=block.loc["omega"]
        lines.append(f"| `{variant}` | {r['mean']:.3f} [{r['q2.5']:.3f}, {r['q97.5']:.3f}] | {r.max_rhat:.3f} | {r.min_bulk_ess:.0f} |")
    lines.extend(["",
        "The baseline AR(2) sampler converges, but the slow/cycle variance allocation is not data-dominated: changing the omega prior materially changes its posterior. The short AR(1) sensitivity also fails the full convergence threshold and is not a competing headline estimate.",
        "","## Primary slope-only NKPC","","```math",
        "\\pi_t=a+\\alpha_b\\pi_{t-1}+\\alpha_fE_t\\pi_{t+1}+(\\kappa_0+\\delta\\bar c_t)x_t+\\varepsilon_t,",
        "\\qquad \\varepsilon_t=u_t+\\psi_1u_{t-1}+\\psi_2u_{t-2}+\\psi_3u_{t-3}.","```","",
        "| Cell | delta | P(delta>0) | Post/prior SD | R-hat | Bulk ESS |","|---|---:|---:|---:|---:|---:|"])
    slope=coef[(coef.model=="slope_only")&(coef.parameter=="delta")]
    for _,r in slope.iterrows():
        lines.append(f"| {_cell_label(r.cell)} | {r['mean']:.3f} [{r['q2.5']:.3f}, {r['q97.5']:.3f}] | {r.p_positive:.3f} | {r.posterior_prior_sd_ratio:.3f} | {r.rhat:.3f} | {r.ess_bulk:.0f} |")
    lines.extend(["","## Main economic estimand","","```math","\\Delta\\kappa_{comp}=\\delta(\\bar c_{t_1}-\\bar c_{t_0}).","```","",
        "| Cell | Window | Delta kappa | P(Delta kappa>0) |","|---|---|---:|---:|"])
    rows=econ[econ.quantity=="delta_kappa_comp"]
    for _,r in rows.iterrows():lines.append(f"| {_cell_label(r.cell)} | {r.window} | {r['mean']:.3f} [{r['q2.5']:.3f}, {r['q97.5']:.3f}] | {r.p_positive:.3f} |")
    lines.extend(["","## Direct competition-index diagnostic","","```math","\\pi_t=\\cdots+(\\kappa_0+\\delta\\bar c_t)x_t-\\theta_C\\hat c_{t-j}+\\varepsilon_t.","```","",
        "| Cell | Timing | theta_C | P(theta_C>0) | Post/prior SD |","|---|---|---:|---:|---:|"])
    direct=coef[coef.parameter=="theta_C"]
    for _,r in direct.iterrows():lines.append(f"| {_cell_label(r.cell)} | {r.timing} | {r['mean']:.3f} [{r['q2.5']:.3f}, {r['q97.5']:.3f}] | {r.p_positive:.3f} | {r.posterior_prior_sd_ratio:.3f} |")
    lines.extend(["","## Computational gate","",
        f"- Maximum primary R-hat: {gate['primary_max_rhat']:.4f} (required <= {gate['max_rhat_required']:.2f}).",
        f"- Minimum primary bulk ESS: {gate['primary_min_bulk_ess']:.1f} (required >= {gate['min_bulk_ess_required']:.0f}).",
        f"- Minimum primary tail ESS: {gate['primary_min_tail_ess']:.1f} (required >= {gate['min_tail_ess_required']:.0f}).",
        f"- Exact-identity error: {gate['exact_identity_error']:.3e}.","",
        "## What this test shows","",
        "- The state decomposition is estimated without inflation feedback and state uncertainty is propagated rather than plugged in.",
        "- The data can be evaluated directly in terms of historical competition-induced slope changes.",
        "- Computation is stable, but no delta interval excludes zero and the omega allocation is prior-sensitive. The result is therefore suggestive at most, not a structurally identified competition effect.","",
        "## What this test does NOT show","",
        "- A zero-crossing theta_C interval does not imply theta_C=0.",
        "- theta_C is not the active-firm coefficient theta_N.",
        "- A positive delta does not establish full HSA or a causal competition-policy counterfactual.",
        "- No free lambda or fixed-lambda HSA restriction is estimated here.",""])
    return "\n".join(lines)


def build(profile="full"):
    _style();out,manifest,state,coef,econ,cpath,kpath,measurement=_load(profile)
    report_dir=out/"report";report_dir.mkdir(parents=True,exist_ok=True)
    pdf_path=report_dir/"competition_slope_change_report.pdf"
    with PdfPages(pdf_path) as pdf:
        fig=_page(pdf,"Competition-linked structural slope change",f"Semi-structural modular NKPC | {profile} profile",1);y=.88
        y=_section(fig,y,"Research target")
        y=_para(fig,y,"The test asks whether long-run competition changes are associated with changes in the structural NKPC slope. It does not attempt to identify the full HSA system from the same aggregate time series.")
        y=_eq(fig,y,r"$\Delta\kappa_{\mathrm{comp}}(t_0,t_1)=\delta\,[\bar c_{t_1}-\bar c_{t_0}]$")
        y=_section(fig,y,"Two-block design")
        y=_eq(fig,y,r"$p_{\mathrm{cut}}(\bar c,\hat c,\psi_C,\beta_\pi\mid C^{obs},\pi)=p(\bar c,\hat c,\psi_C\mid C^{obs})\,p(\beta_\pi\mid\pi,\bar c,\hat c)$",10.8)
        for text in ("Inflation never updates the competition decomposition.","Competition-state draws, rather than a posterior-mean plug-in, enter the NKPC mixture.","The empirical coordinate is effective competition C, not automatically the theoretical active-firm stock N."):
            y=_bullet(fig,y,text)
        y=_section(fig,y,"Headline rule")
        y=_para(fig,y,"A converged sampler is not enough. The report separately displays credible intervals, sign probabilities, posterior/prior contraction, state allocation, and timing stability.",color=RED)
        pdf.savefig(fig);plt.close(fig)

        fig=_page(pdf,"Data and measurement","Source coverage and the empirical competition coordinate",2)
        gustavo_periods=measurement.loc[measurement.gustavo_level.notna(),"period"]
        ciq_periods=measurement.loc[measurement.capital_iq_level.notna(),"period"]
        rows=[["Gustavo effective firms","Annual / Q4","Level benchmark",f"{gustavo_periods.iloc[0]}--{gustavo_periods.iloc[-1]} ({len(gustavo_periods)} Q4s)"],
              ["Capital IQ effective firms","Quarterly","Within-year allocation",f"{ciq_periods.iloc[0]}--{ciq_periods.iloc[-1]} ({len(ciq_periods)} observed)"],
              ["PPI / Core CPI inflation","Quarterly YoY","NKPC outcomes","1974Q4--2013Q4"],
              ["SPF GDP inflation","Quarterly","Expected inflation","1974Q4--2013Q4"],
              ["Inverse markup / unemployment gap","Quarterly","Activity proxies","1974Q4--2013Q4"]]
        _table(fig,[.08,.66,.84,.19],["Series","Frequency","Role","Used coverage"],rows,[.29,.16,.26,.29],7.9)
        y=.60;y=_para(fig,y,"The two competition series are not treated as interchangeable levels. Gustavo fixes annual endpoints. Capital IQ supplies within-year movement when observed and coherent; missing or cancellation-dominated allocations shrink toward the robust quarterly profile. The constructed coordinate is measured relative to the predeclared 1984 Gustavo value.")
        ax=fig.add_axes([.10,.12,.82,.36]);p=pd.PeriodIndex(measurement.period,freq="Q").to_timestamp()
        ax.plot(p,measurement.constructed_coordinate,color=BLUE,label=r"constructed $c_t^{obs}$")
        g=measurement.gustavo_coordinate.notna();ax.scatter(p[g],measurement.loc[g,"gustavo_coordinate"],color=INK,s=15,label="Gustavo Q4 anchors",zorder=3)
        ci=measurement.capital_iq_log_index.notna();ax.plot(p[ci],measurement.loc[ci,"capital_iq_log_index"],color=RED,alpha=.75,label="Capital IQ log index (own origin)")
        ax.axhline(0,color=INK,lw=.6);ax.grid(axis="y",color=GRID,lw=.5);ax.legend(frameon=False,fontsize=8,ncol=2)
        ax.set_title("Competition inputs and coverage (source-specific log coordinates)")
        pdf.savefig(fig);plt.close(fig)

        fig=_page(pdf,"Competition-only state","Exact annual accounting with an annual-allocation transition mean",3);y=.88
        y=_eq(fig,y,r"$c_t^{obs}=10(\log C_t-\log C_{ref}),\qquad c_t^{obs}=\bar c_t+\hat c_t$")
        y=_eq(fig,y,r"$\bar c_t=\bar c_{t-1}+m_{q(t)}\Delta g_{y(t)}+\eta_t^{\bar c}$")
        y=_eq(fig,y,r"$\hat c_t=2r\cos(2\pi/P)\hat c_{t-1}-r^2\hat c_{t-2}+\eta_t^{\hat c}$")
        y=_eq(fig,y,r"$\sigma_{\bar c}^2=\omega\tau^2,\qquad\sigma_{\hat c}^2=(1-\omega)\tau^2$")
        y=_para(fig,y,"The coordinate origin is the predeclared 1984 Gustavo value; it is not interpreted as a steady state. Capital IQ movements update within-year allocation when coherent. Missing or cancellation-dominated years shrink toward the robust average quarterly profile.")
        ax=fig.add_axes([.10,.12,.82,.39]);p=pd.PeriodIndex(cpath.period,freq="Q").to_timestamp()
        ax.plot(p,cpath.cbar_mean,color=BLUE,label=r"slow $\bar c_t$");ax.fill_between(p,cpath["cbar_q2.5"],cpath["cbar_q97.5"],color=BLUE,alpha=.16)
        ax.plot(p,cpath.chat_mean,color=RED,label=r"cycle $\hat c_t$");ax.fill_between(p,cpath["chat_q2.5"],cpath["chat_q97.5"],color=RED,alpha=.12)
        ax.axhline(0,color=INK,lw=.6);ax.grid(axis="y",color=GRID,lw=.5);ax.legend(frameon=False,ncol=2);ax.set_title("Competition-only posterior decomposition")
        pdf.savefig(fig);plt.close(fig)

        fig=_page(pdf,"Competition-block identification","State allocation, law, and prior sensitivity",4)
        display=[]
        for variant in state.variant.unique():
            block=state[state.variant==variant].set_index("parameter")
            def f(name):
                if name not in block.index:return "n/r"
                r=block.loc[name];return f"{r['mean']:.3f}\n[{r['q2.5']:.3f}, {r['q97.5']:.3f}]"
            r0=block.iloc[0]
            display.append([variant,f("omega"),f("slow_innovation_variance"),f("cycle_innovation_variance"),f("damping_or_rho"),f"{r0.max_rhat:.3f}",f"{r0.min_bulk_ess:.0f}"])
        _table(fig,[.06,.50,.88,.35],["State variant","omega","slow var","cycle var","damping/rho","max R-hat","min ESS"],display,[.20,.13,.14,.14,.14,.12,.11],7.4)
        y=.44;y=_para(fig,y,"AR(2) is the primary law because a stochastic cycle separates cyclical frequency from a slow random walk. AR(1) and alternative omega priors are diagnostics, not opportunities to choose the most favorable inflation result.")
        prof=pd.read_csv(out/"tables"/"omega_conditional_likelihood.csv")
        ax=fig.add_axes([.12,.10,.76,.24]);ax.plot(prof.omega,prof.relative_conditional_loglik,color=BLUE)
        ax.axhline(-1.92,color=MUTED,ls="--",lw=.7);ax.set_xlabel("omega: slow innovation share");ax.set_ylabel("relative conditional log likelihood");ax.grid(color=GRID,lw=.4)
        pdf.savefig(fig);plt.close(fig)

        fig=_page(pdf,"Primary slope-only NKPC","The estimating equation is shown next to its coefficient table",5);y=.88
        y=_eq(fig,y,r"$\pi_t=a+\alpha_b\pi_{t-1}+\alpha_fE_t\pi_{t+1}+(\kappa_0+\delta\bar c_t)x_t+\varepsilon_t$")
        y=_eq(fig,y,r"$\varepsilon_t=u_t+\psi_1u_{t-1}+\psi_2u_{t-2}+\psi_3u_{t-3},\quad u_t\sim\mathcal{N}(0,\sigma_u^2)$")
        cells=["ppi_inverse_markup","ppi_negative_unemployment_gap","core_cpi_inverse_markup","core_cpi_negative_unemployment_gap"]
        params=["alpha_b","alpha_f","kappa_0","delta","psi_1","psi_2","psi_3"]
        rows=[]
        slope=coef[coef.model=="slope_only"]
        for parameter in params:
            row=[parameter]
            for cell in cells:
                r=slope[(slope.cell==cell)&(slope.parameter==parameter)]
                row.append(_fmt(r.iloc[0]) if len(r) else "n/r")
            rows.append(row)
        _table(fig,[.055,.30,.89,.48],["Parameter",*map(_cell_label,cells)],rows,[.13,.215,.215,.215,.215],7.35)
        y=.25;y=_para(fig,y,"Entries are posterior means with 95% intervals. PPI and core CPI are not pooled. PPI/inverse markup is the structural-proxy cell; PPI/unemployment gap is the reduced-form validation cell.")
        pdf.savefig(fig);plt.close(fig)

        fig=_page(pdf,"Economic estimand","Historical slope change and fixed-competition counterfactual",6);y=.88
        y=_eq(fig,y,r"$\kappa_t=\kappa_0+\delta\bar c_t,\qquad\Delta\kappa_{comp}=\delta(\bar c_{t_1}-\bar c_{t_0})$")
        full=econ[(econ.quantity=="delta_kappa_comp")&(econ.window=="full_sample")]
        rows=[]
        for _,r in full.iterrows():rows.append([_cell_label(r.cell),f"{r['mean']:.3f}\n[{r['q2.5']:.3f}, {r['q97.5']:.3f}]",f"{r.p_positive:.3f}"])
        _table(fig,[.12,.62,.76,.18],["Cell","Delta kappa, full sample","P(Delta kappa>0)"],rows,[.45,.34,.20],8.1)
        ax=fig.add_axes([.10,.12,.82,.40]);colors={"ppi_inverse_markup":BLUE,"ppi_negative_unemployment_gap":GREEN}
        for cell in colors:
            d=kpath[kpath.cell==cell];p=pd.PeriodIndex(d.period,freq="Q").to_timestamp();c=colors[cell]
            ax.plot(p,d.kappa_mean,color=c,label=_cell_label(cell));ax.fill_between(p,d["kappa_q2.5"],d["kappa_q97.5"],color=c,alpha=.13)
            ax.plot(p,d.counterfactual_kappa_mean,color=c,ls="--",lw=1)
        ax.axhline(0,color=INK,lw=.6);ax.grid(axis="y",color=GRID,lw=.5);ax.legend(frameon=False,fontsize=8)
        ax.set_title("Historical kappa (solid) and fixed early-competition counterfactual (dashed)")
        pdf.savefig(fig);plt.close(fig)

        fig=_page(pdf,"Posterior learning","Economic slope change and prior-to-posterior contraction",7)
        z=np.load(out/"draws"/"nkpc"/"ppi_inverse_markup"/"slope_only_none.npz",allow_pickle=False)
        names=list(map(str,z["names"]));draws=z["draws"].reshape(-1,len(names));delta=draws[:,names.index("delta")]
        delta_kappa=(z["kappa"][:,:,-1]-z["kappa"][:,:,0]).reshape(-1)
        ax=fig.add_axes([.10,.57,.82,.25]);ax.hist(delta_kappa,bins=55,density=True,color=BLUE,alpha=.28,edgecolor="none")
        ax.axvline(0,color=INK,lw=.9);ax.axvline(np.mean(delta_kappa),color=BLUE,lw=1.4)
        ax.set_title(r"PPI x inverse markup: posterior of $\Delta\kappa_{comp}$")
        ax.set_xlabel(r"$\Delta\kappa_{comp}$");ax.set_yticks([]);ax.grid(axis="x",color=GRID,lw=.5)
        slope_row=coef[(coef.cell=="ppi_inverse_markup")&(coef.model=="slope_only")&(coef.parameter=="delta")].iloc[0]
        direct_row=coef[(coef.cell=="ppi_inverse_markup")&(coef.timing=="current")&(coef.parameter=="theta_C")].iloc[0]
        panels=[("delta",slope_row,delta),
                ("theta_C",direct_row,np.load(out/"draws"/"nkpc"/"ppi_inverse_markup"/"slope_plus_competition_cycle_current.npz",allow_pickle=False)["draws"])]
        for j,(label,row,posterior) in enumerate(panels):
            ax=fig.add_axes([.10+.43*j,.17,.36,.25])
            if label=="theta_C":
                zp=np.load(out/"draws"/"nkpc"/"ppi_inverse_markup"/"slope_plus_competition_cycle_current.npz",allow_pickle=False)
                nn=list(map(str,zp["names"]));posterior=zp["draws"].reshape(-1,len(nn))[:,nn.index("theta_C")]
            lo=min(np.percentile(posterior,.5),row.prior_mean-3.5*row.prior_sd);hi=max(np.percentile(posterior,99.5),row.prior_mean+3.5*row.prior_sd)
            x=np.linspace(lo,hi,400);prior=np.exp(-.5*((x-row.prior_mean)/row.prior_sd)**2)/(row.prior_sd*np.sqrt(2*np.pi))
            ax.plot(x,prior,color=MUTED,ls="--",label="prior");ax.hist(posterior,bins=45,density=True,color=BLUE,alpha=.30,label="posterior")
            ax.axvline(0,color=INK,lw=.7);ax.set_yticks([]);ax.set_title(f"PPI x inverse markup: {label}")
            ax.set_xlabel(f"Post/prior SD = {row.posterior_prior_sd_ratio:.3f}");ax.legend(frameon=False,fontsize=8)
        fig.text(.10,.10,"The slope direction is suggestive but its 95% interval crosses zero. The direct competition-index loading is almost unchanged from its prior.",fontsize=9.2,color=RED)
        pdf.savefig(fig);plt.close(fig)

        fig=_page(pdf,"Direct competition-index diagnostic","theta_C is not the structural active-firm coefficient theta_N",8);y=.88
        y=_eq(fig,y,r"$\pi_t=\cdots+(\kappa_0+\delta\bar c_t)x_t-\theta_C\hat c_{t-j}+\varepsilon_t$")
        y=_para(fig,y,"Current timing is the benchmark, lag one is a timing robustness check, and lead one is a placebo. A narrow restricted HSA coefficient is not estimated in this bundle.")
        direct=coef[coef.parameter=="theta_C"];rows=[]
        for _,r in direct.sort_values(["cell","timing"]).iterrows():rows.append([_cell_label(r.cell),r.timing,_fmt(r),f"{r.p_positive:.3f}",f"{r.posterior_prior_sd_ratio:.3f}",f"{r.rhat:.3f}"])
        _table(fig,[.07,.39,.86,.34],["Cell","Timing","theta_C","P(>0)","Post/prior SD","R-hat"],rows,[.28,.12,.23,.12,.15,.10],7.8)
        y=.33
        for text in ("An interval that crosses zero means weak sign identification, not proof that theta_C is zero.","Even an identified theta_C would load on cyclical concentration, not on an externally anchored active-firm stock.","Structural versus observational flattening remains a future C-plus-N exercise."):
            y=_bullet(fig,y,text)
        pdf.savefig(fig);plt.close(fig)

        fig=_page(pdf,"Convergence and parameter learning","R-hat, ESS, and prior-to-posterior contraction answer different questions",9);y=.88
        gate=manifest["gate"]
        rows=[["Primary maximum R-hat",f"{gate['primary_max_rhat']:.4f}",f"<= {gate['max_rhat_required']:.2f}"],
              ["Primary minimum bulk ESS",f"{gate['primary_min_bulk_ess']:.1f}",f">= {gate['min_bulk_ess_required']:.0f}"],
              ["Primary minimum tail ESS",f"{gate['primary_min_tail_ess']:.1f}",f">= {gate['min_tail_ess_required']:.0f}"],
              ["Exact identity error",f"{gate['exact_identity_error']:.2e}",f"<= {gate['max_exact_identity_error_required']:.0e}"],
              ["Computational gate","PASS" if gate["passed"] else "FAIL","all conditions"]]
        _table(fig,[.16,.64,.68,.20],["Diagnostic","Observed","Required"],rows,[.46,.26,.28],8.3)
        y=.58;y=_section(fig,y,"Identification reading")
        slope=coef[(coef.model=="slope_only")&(coef.parameter=="delta")]
        for _,r in slope.iterrows():
            y=_bullet(fig,y,f"{_cell_label(r.cell)}: P(delta>0)={r.p_positive:.3f}, posterior/prior SD={r.posterior_prior_sd_ratio:.3f}, 95% interval [{r['q2.5']:.3f}, {r['q97.5']:.3f}].")
        y=_para(fig,y,"The full economic claim must follow these learning diagnostics. Model fit is not promoted merely because a restriction compresses a weak direction.",color=RED)
        pdf.savefig(fig);plt.close(fig)

        fig=_page(pdf,"Interpretation","What the test can and cannot establish",10);y=.88
        y=_section(fig,y,"What this test can show")
        for text in ("Whether a competition-only slow state is associated with the historical NKPC slope.","The posterior distribution of competition-induced slope changes and partial-equilibrium fixed-competition counterfactuals.","Whether the result survives price/activity cells, state laws, omega priors, and MA(3) residual treatment."):
            y=_bullet(fig,y,text)
        y-=.01;y=_section(fig,y,"What this test cannot show")
        for text in ("That theta_N is zero or positive; no active-firm stock is used.","That delta>0 establishes the full HSA cross-equation restriction.","That 1/HHI or effective-firm competition is identical to the active-firm mass in the theory.","A causal competition-policy counterfactual or a decomposition of observational flattening."):
            y=_bullet(fig,y,text,color=RED)
        y-=.01;y=_section(fig,y,"Next admissible extension")
        y=_para(fig,y,"Build an external active-firm stock from BDS levels and BED timing measurements, keep that block cut from inflation, identify theta_N freely, and only then test the HSA derivative restriction on the same structural N coordinate.")
        pdf.savefig(fig);plt.close(fig)
    markdown=_results_markdown(profile,manifest,state,coef,econ)
    (BUNDLE/"RESULTS.md").write_text(markdown,encoding="utf-8")
    if profile=="full":
        target=ROOT/"output"/"pdf"/"competition_slope_change_report.pdf";target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(pdf_path,target)
    print(f"wrote {pdf_path}");print(f"wrote {BUNDLE/'RESULTS.md'}")
    return pdf_path


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--profile",choices=("smoke","full"),default="full");args=parser.parse_args();build(args.profile)


if __name__=="__main__":main()
