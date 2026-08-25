"""Build the English mock report for the Gustavo x Capital IQ decomposition."""
from __future__ import annotations

import json,shutil,sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd

ROOT=next(p for p in Path(__file__).resolve().parents if (p/"pyproject.toml").exists())
sys.path[:0]=[str(ROOT),str(ROOT/"src")]
BUNDLE=Path(__file__).resolve().parent;OUT=BUNDLE/"results"/"mock_qoq";INK="#1f2933";BLUE="#24567a";GREEN="#2f7355";RED="#a33b32";GRID="#c9d1d8"
mpl.rcParams.update({"font.family":"STIXGeneral","mathtext.fontset":"stix","font.size":10,"axes.titlesize":13,"axes.labelsize":10,"figure.facecolor":"white"})


def _page(pdf,title,subtitle,number):
    fig=plt.figure(figsize=(8.5,11));fig.text(.065,.955,title,fontsize=19,weight="bold",color=INK);fig.text(.065,.925,subtitle,fontsize=10,color="#52606d");fig.text(.94,.025,str(number),ha="right",fontsize=8,color="#66737f");return fig


def _para(fig,y,text,color=INK,size=10):
    import textwrap
    lines=[]
    for paragraph in text.split("\n"):lines.extend(textwrap.wrap(paragraph,100) or [""])
    fig.text(.075,y,"\n".join(lines),va="top",fontsize=size,color=color,linespacing=1.25);return y-.024*max(1,len(lines))


def _table(ax,rows,columns,widths=None,font=8.5):
    ax.axis("off");table=ax.table(cellText=rows,colLabels=columns,cellLoc="center",colLoc="center",loc="center",colWidths=widths)
    table.auto_set_font_size(False);table.set_fontsize(font);table.scale(1,1.45)
    for (r,c),cell in table.get_celld().items():cell.set_edgecolor("#7b8794");cell.set_linewidth(.55);cell.set_facecolor("#edf1f4" if r==0 else "white");cell.set_text_props(weight="bold" if r==0 else "normal",color=INK)
    return table


def _periods(values):return pd.PeriodIndex(values,freq="Q").to_timestamp(how="end")


def _fmt(row):return f"{row['mean']:.3f}\n[{row['q2.5']:.3f}, {row['q97.5']:.3f}]"


def _write_results(manifest,state,coeff,power):
    theta=coeff[coeff.parameter=="theta_CIQ"].copy();lines=["# Gustavo state x Capital IQ cycle: recorded mock result","","Status: **MOCK - NOT FOR INFERENCE**","","## Measurement design","","```math","\\bar n_{y,Q4}=10\\log(N_y^G/N_{1993}^G),","\\qquad c_t^{CIQ}=a_C+b_C\\bar n_t+\\hat n_t^{CIQ}+e_t.","```","","## State parameters","","| Variant | Parameter | Mean and 95% interval |","|---|---|---:|"]
    for _,r in state.iterrows():lines.append(f"| {r['variant']} | `{r['parameter']}` | {r['mean']:.3f} [{r['q2.5']:.3f}, {r['q97.5']:.3f}] |")
    lines.extend(["","## Free cycle coefficient","","```math","\\pi_t=a+\\alpha_b\\pi_{t-1}+\\alpha_fE_t\\pi_{t+1}+\\kappa_0x_t-\\theta_{CIQ}\\hat n_t^{CIQ}+\\varepsilon_t.","```","","| Cycle | Cell | theta_CIQ | P(theta_CIQ>0) | Post/prior SD | R-hat |","|---|---|---:|---:|---:|---:|"])
    for _,r in theta.iterrows():lines.append(f"| {r['cycle']} / {r['error_model']} | {r['cell']} | {r['mean']:.3f} [{r['q2.5']:.3f}, {r['q97.5']:.3f}] | {r['p_positive']:.3f} | {r['posterior_prior_sd_ratio']:.3f} | {r['rhat']:.3f} |")
    lines.extend(["","## Recovery","","| Error | Mode | True theta_CIQ | Detection rate | Mean estimate |","|---|---|---:|---:|---:|"])
    for _,r in power.iterrows():lines.append(f"| {r['error_model']} | {r['mode']} | {r['theta_true']:.2f} | {r['detection_rate']:.3f} | {r['mean_estimate']:.3f} |")
    g=manifest["gate"];lines.extend(["","## Gate","",f"- Maximum R-hat: `{g['observed_max_rhat']:.4f}` (required <= `{g['max_rhat_required']}`).",f"- Minimum bulk ESS: `{g['observed_min_bulk_ess']:.1f}` (required >= `{g['min_bulk_ess_required']}`).",f"- Exact Gustavo anchor error: `{g['gustavo_anchor_error']:.2e}`.",f"- Computational mock gate passed: `{g['passed']}`.","- Mock recovery rates are pipeline diagnostics, not power estimates.","- No delta, lambda, HSA restriction, marginal likelihood, or causal interpretation is estimated.",""])
    lines[0]="# Gustavo state x Capital IQ cycle: recorded QoQ mock result";lines.insert(2,"Inflation: `400 * quarterly log difference`; expectation: genuine SPF one-quarter-ahead annualized-log forecast.  ");(BUNDLE/"RESULTS.md").write_text("\n".join(lines))


def build():
    manifest=json.loads((OUT/"manifest.json").read_text());state=pd.read_csv(OUT/"tables"/"state_parameters.csv");coeff=pd.read_csv(OUT/"tables"/"coefficients.csv");power=pd.read_csv(OUT/"tables"/"recovery_power.csv");paths=pd.read_csv(OUT/"tables"/"state_paths.csv");report_dir=OUT/"report";report_dir.mkdir(parents=True,exist_ok=True);pdf_path=report_dir/"gustavo_state_capitaliq_cycle_qoq_report.pdf"
    with PdfPages(pdf_path) as pdf:
        fig=_page(pdf,"Gustavo slow state and Capital IQ cycle","Annualized QoQ mock identification diagnostic",1);y=.855;fig.text(.075,y,"Frozen order",fontsize=14,weight="bold",color=INK);y-=.05
        for text in ("1. Reuse the quarterly slow state drawn from Gustavo annual Q4 anchors only.","2. Reuse the cut AR(2) Capital IQ cycle without inflation feedback.","3. Estimate free theta_CIQ using 400 times the quarterly log price change and genuine one-quarter-ahead SPF expectations."):y=_para(fig,y,text);y-=.012
        fig.text(.075,y-.01,"Cut factorization",fontsize=14,weight="bold",color=INK);fig.text(.11,y-.10,r"$p(\bar n,\hat n,\beta_\pi\mid G,CIQ,\pi)=p(\bar n\mid G)\,p(\hat n\mid CIQ,\bar n)\,p(\beta_\pi\mid\pi,\hat n)$",fontsize=15,color=INK)
        _para(fig,y-.19,"The saved competition measurement draws are identical to the archived YoY mock. QoQ changes only the NKPC observation equation. IID is primary and persistent AR(1) is robustness. This report is mock-only.",RED);pdf.savefig(fig);plt.close(fig)

        fig=_page(pdf,"Measurement equations","Distinct roles and distinct level scales",2);fig.text(.075,.86,r"$g_y=10\{\log N_y^G-\log N_{1993}^G\},\qquad \bar n_{y,Q4}=g_y$",fontsize=15,color=INK);fig.text(.075,.79,r"$\bar n_t=\bar n_{t-1}+\mu+\eta_t^{\bar n}\quad\mathrm{subject\ to\ exact\ annual\ endpoints}$",fontsize=15,color=INK);fig.text(.075,.70,r"$c_{j,t}=a_j+b_j\bar n_t+\hat n_{j,t}+e_{j,t}$",fontsize=16,color=INK);fig.text(.075,.63,r"$\hat n_{j,t}=2r_j\cos(2\pi/P_j)\hat n_{j,t-1}-r_j^2\hat n_{j,t-2}+\eta_{j,t}$",fontsize=15,color=INK)
        _para(fig,.54,"The estimated intercept and loading absorb the fact that Gustavo and Capital IQ effective-firm counts have different universes and levels. Only the residual AR(2) component is called the Capital IQ cycle.")
        ax=fig.add_axes([.10,.12,.82,.31]);p=_periods(paths.period);ax.plot(p,paths.gustavo_slow_mean,color=BLUE,label="Gustavo slow state");ax.fill_between(p,paths["gustavo_slow_q2.5"],paths["gustavo_slow_q97.5"],color=BLUE,alpha=.16);mask=paths.gustavo_anchor.notna();ax.scatter(p[mask],paths.loc[mask,"gustavo_anchor"],s=16,color=INK,label="Gustavo Q4 anchors",zorder=3);ax.axhline(0,color=GRID,lw=.8);ax.set_ylabel("10 log points from 1993");ax.legend(frameon=False,ncol=2);ax.grid(axis="y",color=GRID,lw=.5);pdf.savefig(fig);plt.close(fig)

        fig=_page(pdf,"Capital IQ cycle extraction","Firm-weighted and revenue-weighted measurements",3);ax=fig.add_axes([.09,.51,.83,.31]);p=_periods(paths.period)
        colors={"firm_weighted":BLUE,"revenue_weighted":GREEN}
        for label,color in colors.items():ax.plot(p,paths[f"{label}_cycle_mean"],color=color,label=label.replace("_"," "));ax.fill_between(p,paths[f"{label}_cycle_q2.5"],paths[f"{label}_cycle_q97.5"],color=color,alpha=.10)
        ax.axhline(0,color=GRID,lw=.8);ax.set_ylabel("Capital IQ cycle coordinate");ax.legend(frameon=False,ncol=2);ax.grid(axis="y",color=GRID,lw=.5)
        rows=[]
        for label in colors:
            z=state[(state.variant==label)&state.parameter.isin(["loading","damping","period","sigma_cycle","sigma_measurement"])]
            values={r.parameter:f"{r['mean']:.2f}\n[{r['q2.5']:.2f}, {r['q97.5']:.2f}]" for _,r in z.iterrows()};rows.append([label.replace("_"," "),values.get("loading",""),values.get("damping",""),values.get("period",""),values.get("sigma_cycle",""),values.get("sigma_measurement","")])
        tax=fig.add_axes([.07,.12,.87,.28]);_table(tax,rows,["Cycle","Loading","Damping","Period","SD cycle","SD error"],[.20,.16,.16,.16,.16,.16],7.8);pdf.savefig(fig);plt.close(fig)

        fig=_page(pdf,"Free direct-channel diagnostic","QoQ equation, coefficient table, and recovery",4);fig.text(.09,.87,r"$\pi_t^q=400(\log P_t-\log P_{t-1})$",fontsize=14,color=INK);fig.text(.09,.82,r"$\pi_t^q=a+\alpha_b\pi_{t-1}^q+\alpha_fE_t\pi_{t+1}^q+\kappa_0x_t-\theta_{CIQ}\hat n_{j,t}+\varepsilon_t$",fontsize=14,color=INK);fig.text(.09,.77,r"$\varepsilon_t=u_t\ \mathrm{(primary)},\qquad \varepsilon_t=\rho\varepsilon_{t-1}+u_t\ \mathrm{(robustness)}$",fontsize=13,color=INK)
        theta=coeff[coeff.parameter=="theta_CIQ"];rows=[]
        for _,r in theta.iterrows():rows.append([r["cycle"].replace("_"," "),r["error_model"].replace("persistent_",""),r["cell"].replace("ppi_","").replace("_"," "),_fmt(r),f"{r['p_positive']:.3f}",f"{r['rhat']:.3f}"])
        tax=fig.add_axes([.05,.45,.90,.29]);_table(tax,rows,["CIQ cycle","Error","PPI activity","theta_CIQ","P(theta>0)","R-hat"],[.18,.11,.22,.20,.16,.13],7.2)
        ticks=np.array([0,1,3,10]);colors={"propagated_state":BLUE,"oracle_state":GREEN}
        for j,error in enumerate(("iid","persistent_ar1")):
            ax=fig.add_axes([.08+.46*j,.105,.39,.23])
            for mode,color in colors.items():z=power[(power["error_model"]==error)&(power["mode"]==mode)];ax.plot(np.log1p(z.theta_true),z.detection_rate,"o-",color=color,label=mode.replace("_"," "))
            ax.set_xticks(np.log1p(ticks),[f"{x:g}" for x in ticks]);ax.axhline(.8,color=RED,ls="--",lw=1);ax.set_ylim(-.03,1.05);ax.set_title(error.replace("persistent_",""),fontsize=10);ax.set_xlabel(r"Injected $\theta_{CIQ}$");ax.grid(color=GRID,lw=.5)
            if j==0:ax.set_ylabel("Detection rate");ax.legend(frameon=False,fontsize=7)
        pdf.savefig(fig);plt.close(fig)

        fig=_page(pdf,"Mock gate and interpretation","What this experiment permits",5);g=manifest["gate"];rows=[["Maximum R-hat",f"{g['observed_max_rhat']:.3f}",f"<= {g['max_rhat_required']}"],["Minimum bulk ESS",f"{g['observed_min_bulk_ess']:.1f}",f">= {g['min_bulk_ess_required']}"],["Gustavo anchor error",f"{g['gustavo_anchor_error']:.2e}",f"<= {g['max_anchor_error_required']:.0e}"],["Computational mock gate","PASS" if g["passed"] else "FAIL","all rows"]];tax=fig.add_axes([.15,.58,.70,.24]);_table(tax,rows,["Diagnostic","Observed","Required"],[.48,.25,.27],9)
        fig.text(.075,.48,"Interpretation",fontsize=14,weight="bold",color=INK);y=.43;main=theta[(theta.cycle=="firm_weighted")&(theta.error_model=="iid")&(theta.cell=="ppi_inverse_markup")].iloc[0];ar1=theta[(theta.cycle=="firm_weighted")&(theta.error_model=="persistent_ar1")&(theta.cell=="ppi_inverse_markup")].iloc[0]
        for text in (f"Primary IID PPI x inverse-markup theta_CIQ is {main['mean']:.3f} [{main['q2.5']:.3f}, {main['q97.5']:.3f}], with positive probability {main['p_positive']:.3f}.",f"Persistent AR(1) gives {ar1['mean']:.3f} [{ar1['q2.5']:.3f}, {ar1['q97.5']:.3f}]. The direction survives, but neither interval excludes zero.","QoQ is more favorable than the archived YoY mock, but propagated recovery remains weak below theta_CIQ=3 and one AR(1) recovery chain misses the ESS gate.","No delta, structural lambda, HSA restriction, marginal likelihood, or policy counterfactual is admissible in this mock."):y=_para(fig,y,text,color=RED if "No delta" in text else INK);y-=.012
        pdf.savefig(fig);plt.close(fig)
    final=ROOT/"output"/"pdf"/"gustavo_state_capitaliq_cycle_qoq_report.pdf";final.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(pdf_path,final);_write_results(manifest,state,coeff,power);print(f"wrote {pdf_path}");print(f"wrote {final}")


if __name__=="__main__":build()
