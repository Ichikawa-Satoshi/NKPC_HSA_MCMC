"""Build the English varying-theta and dynamic-HSA diagnostic report."""
from __future__ import annotations

import json
import shutil
import textwrap
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd

ROOT=next(p for p in Path(__file__).resolve().parents if (p/"pyproject.toml").exists())
BUNDLE=Path(__file__).resolve().parent
OUT=BUNDLE/"results"/"dynamic_validation"
INK="#1f2933";BLUE="#24567a";GREEN="#2f7355";RED="#a33b32";GRID="#c9d1d8";GOLD="#9a6b1d"
mpl.rcParams.update({"font.family":"STIXGeneral","mathtext.fontset":"stix","font.size":10,"axes.titlesize":12,"figure.facecolor":"white"})


def _page(title,subtitle,n):
    fig=plt.figure(figsize=(8.5,11));fig.text(.065,.955,title,fontsize=19,weight="bold",color=INK);fig.text(.065,.925,subtitle,fontsize=10,color="#52606d");fig.text(.94,.025,str(n),ha="right",fontsize=8,color="#66737f");return fig


def _para(fig,y,text,color=INK,size=10,width=102):
    lines=[]
    for p in text.split("\n"):lines.extend(textwrap.wrap(p,width) or [""])
    fig.text(.075,y,"\n".join(lines),va="top",fontsize=size,color=color,linespacing=1.25);return y-.024*max(1,len(lines))


def _table(ax,rows,cols,widths=None,font=8,scale=1.5):
    ax.axis("off");t=ax.table(cellText=rows,colLabels=cols,cellLoc="center",colLoc="center",loc="center",colWidths=widths);t.auto_set_font_size(False);t.set_fontsize(font);t.scale(1,scale)
    for (r,c),cell in t.get_celld().items():
        cell.set_edgecolor("#7b8794");cell.set_linewidth(.55);cell.set_facecolor("#edf1f4" if r==0 else "white");cell.set_text_props(weight="bold" if r==0 else "normal",color=INK)
    return t


def _fmt(r):return f"{r['mean']:.3f}\n[{r['q2.5']:.3f}, {r['q97.5']:.3f}]"


def _full_iid(coeff,model,parameters):
    z=coeff[(coeff.error_model=="iid")&(coeff.sample_end=="2013Q4")&(coeff.model==model)&coeff.parameter.isin(parameters)].copy()
    return z


def _coefficient_rows(coeff,model,parameters):
    z=_full_iid(coeff,model,parameters);rows=[]
    cells=[("firm_weighted","ppi_negative_unemployment_gap","Firm / unemployment"),("revenue_weighted","ppi_negative_unemployment_gap","Revenue / unemployment"),("firm_weighted","ppi_inverse_markup","Firm / inverse markup"),("revenue_weighted","ppi_inverse_markup","Revenue / inverse markup")]
    for parameter in parameters:
        row=[parameter.replace("_derived"," (derived)")]
        for cycle,cell,_ in cells:
            r=z[(z.cycle==cycle)&(z.cell==cell)&(z.parameter==parameter)].iloc[0];row.append(_fmt(r))
        rows.append(row)
    return rows,[x[2] for x in cells]


def _path_draws(fit_path):
    a=np.load(fit_path,allow_pickle=True);draws=a["draws"];names=[str(x) for x in a["names"]];bar=a["nbar_used"];b={n:draws[:,:,j] for j,n in enumerate(names)};barc=bar-bar.mean(axis=2,keepdims=True);q2=barc**2-(barc**2).mean(axis=2,keepdims=True);theta=b["theta_0"][:,:,None]+b["gamma"][:,:,None]*barc
    model=fit_path.parts[-4]
    if model=="varying_theta":kappa=np.broadcast_to(b["kappa_0"][:,:,None],bar.shape)
    elif model=="free_dynamic":kappa=b["kappa_0"][:,:,None]+b["delta_1"][:,:,None]*barc+b["delta_2"][:,:,None]*q2
    else:kappa=b["kappa_0"][:,:,None]+b["lambda"][:,:,None]*b["theta_0"][:,:,None]*barc+.5*b["lambda"][:,:,None]*b["gamma"][:,:,None]*q2
    periods=pd.PeriodIndex([str(x) for x in a["periods"]],freq="Q");return periods,theta.reshape(-1,theta.shape[-1]),kappa.reshape(-1,kappa.shape[-1])


def _band(ax,periods,values,color,label):
    x=periods.to_timestamp();lo,mid,hi=np.percentile(values,[2.5,50,97.5],axis=0);ax.fill_between(x,lo,hi,color=color,alpha=.16,lw=0);ax.plot(x,mid,color=color,lw=1.3,label=label);ax.axhline(0,color="#7b8794",lw=.7);ax.grid(axis="y",color=GRID,lw=.45);return lo,mid,hi


def _write_results(manifest,coeff,comp,power):
    gate=manifest["gate"];primary=_full_iid(coeff,"varying_theta",["theta_0","gamma"]);hsa=_full_iid(coeff,"hsa_restricted_dynamic",["theta_0","gamma","lambda","delta_1_derived","delta_2_derived"]);lines=["# Varying-theta and HSA-restricted dynamic diagnostic","","Status: **COMPUTATIONAL PASS; DYNAMIC IDENTIFICATION FAIL; NOT FOR INFERENCE**","","This branch was estimated at the user's request after the staged recovery gate had failed. It is a weak-identification diagnostic, not a reversal of that stopping decision.","","## Models","","```math","\\theta_t=\\theta_0+\\gamma\\bar n_t^c,","```","","```math","\\kappa_t^{free}=\\kappa_0+\\delta_1\\bar n_t^c+\\delta_2 q_t^{(2)},","\\qquad q_t^{(2)}=(\\bar n_t^c)^2-\\overline{(\\bar n^c)^2},","```","","```math","\\kappa_t^{HSA}=\\kappa_0+\\lambda\\theta_0\\bar n_t^c+\\frac{\\lambda\\gamma}{2}q_t^{(2)}.","```","","Thus the HSA restrictions are `delta_1=lambda*theta_0` and `delta_2=lambda*gamma/2`. Centering the quadratic term changes only the intercept of `kappa_t`; it does not change the derivative restriction.","","The paired Gustavo slow-state and Capital IQ cycle draws are held fixed and cut from inflation in every model.","","## Full-sample primary IID coefficients","","Each entry is posterior mean followed by the 95% interval.",""]
    for model,params in (("varying_theta",["theta_0","gamma"]),("free_dynamic",["theta_0","gamma","delta_1","delta_2"]),("hsa_restricted_dynamic",["theta_0","gamma","lambda","delta_1_derived","delta_2_derived"])):
        lines.extend([f"### `{model}`","","| Parameter | Firm / unemployment | Revenue / unemployment | Firm / inverse markup | Revenue / inverse markup |","|---|---:|---:|---:|---:|"])
        rows,cols=_coefficient_rows(coeff,model,params)
        for row in rows:lines.append("| "+" | ".join(x.replace("\n","<br>") for x in row)+" |")
        lines.append("")
    lines.extend(["## Predictive comparison","","Differences below are relative to the constant-theta model. Positive ELPD favors the dynamic model; negative RMSE favors it. PSIS-LOO is descriptive because every cell has at least one Pareto-k above 0.7.","","| Cycle | Activity | Model | Delta LOO ELPD | Delta WAIC ELPD | Delta holdout ELPD | Delta holdout RMSE |","|---|---|---|---:|---:|---:|---:|"])
    for (cy,cell),g in comp.groupby(["cycle","cell"]):
        b=g[g.model=="constant_theta"].iloc[0]
        for model in ("varying_theta","free_dynamic","hsa_restricted_dynamic"):
            r=g[g.model==model].iloc[0];lines.append(f"| {cy} | {cell} | {model} | {r.elpd_loo-b.elpd_loo:.3f} | {r.elpd_waic-b.elpd_waic:.3f} | {r.holdout_elpd-b.holdout_elpd:.3f} | {r.holdout_rmse-b.holdout_rmse:.3f} |")
    lines.extend(["","All twelve dynamic holdout-ELPD differences are negative. No dynamic specification improves held-out predictive density relative to constant theta.","","## Varying-theta recovery","","The primary propagated-state recovery uses 30 replications at each standardized gamma. Suggestive detection requires `P(gamma>0)>=0.80` and posterior/prior SD at most 0.75; strong detection additionally requires a positive 95% interval.","","| Standardized gamma | Suggestive rate | Strong rate | Coverage |","|---:|---:|---:|---:|"])
    p=power[(power["mode"]=="propagated_state")&(power.parameter=="gamma")]
    for scenario in ("theta_only","gamma_small","gamma_observed_scale","gamma_moderate","gamma_large"):
        r=p[p.scenario==scenario].iloc[0];lines.append(f"| {r.standardized_true:.2f} | {r.suggestive_rate:.3f} | {r.strong_rate:.3f} | {r.coverage:.3f} |")
    lines.extend(["",f"Convergence passes: observed maximum R-hat `{gate['observed_max_rhat']:.4f}`, observed minimum bulk ESS `{gate['observed_min_bulk_ess']:.1f}`, recovery maximum R-hat `{gate['recovery_max_rhat']:.4f}`, and recovery minimum bulk ESS `{gate['recovery_min_bulk_ess']:.1f}`.","","## Persistent-AR(1) robustness","","For the firm-weighted PPI x negative unemployment-gap cell, free-dynamic `gamma=-0.231 [-1.338,0.866]`; HSA-restricted `gamma=-0.096 [-1.113,0.847]` and `lambda=0.401 [-5.975,6.292]`. Allowing persistent inflation errors does not change the identification conclusion.","","## Interpretation","","The actual-data `theta_0` posterior remains directionally positive in the varying-theta model, but every 95% interval includes zero. Every `gamma` posterior also includes zero and leans negative. At an observed-scale standardized gamma of 0.10, propagated-state strong recovery is only 0.067. The sample therefore cannot distinguish a modest time-varying direct coefficient from a constant one.","","The HSA restriction does not solve this problem. Every unrestricted `lambda` interval spans both signs and zero, and every derived slope interval spans zero. Because the unrestricted free-dynamic channels are themselves weak, posterior narrowing inside the HSA parameterization is not independent structural identification.","","The dynamic branch is computationally valid but empirically unsupported. Retain the constant-theta direct-channel result only as suggestive directional evidence; do not claim time-varying theta or the HSA cross-equation restrictions are identified.",""])
    (BUNDLE/"RESULTS_DYNAMIC.md").write_text("\n".join(lines))


def build():
    manifest=json.loads((OUT/"manifest.json").read_text());coeff=pd.read_csv(OUT/"tables"/"coefficients.csv");comp=pd.read_csv(OUT/"tables"/"model_comparison.csv");power=pd.read_csv(OUT/"tables"/"recovery_power.csv");report_dir=OUT/"report";report_dir.mkdir(parents=True,exist_ok=True);pdf_path=report_dir/"gustavo_state_capitaliq_cycle_dynamic_report.pdf"
    with PdfPages(pdf_path) as pdf:
        fig=_page("Dynamic competition channels","Varying theta, free dynamic, and HSA-restricted dynamic diagnostics",1);y=.855
        y=_para(fig,y,"This branch is run at the user's request after the staged unrestricted recovery gate failed. Its purpose is to measure what the dynamic restriction does, not to bypass the failed identification gate.",RED);y-=.025
        fig.text(.08,y,r"$\pi_t^q=a+\alpha_b\pi_{t-1}^q+\alpha_fE_t\pi_{t+1}^q+\kappa_t x_t-\theta_t\hat n_t+\varepsilon_t$",fontsize=15,color=INK);y-=.085
        fig.text(.08,y,r"$\theta_t=\theta_0+\gamma\bar n_t^c,\qquad q_t^{(2)}=(\bar n_t^c)^2-\overline{(\bar n^c)^2}$",fontsize=14,color=INK);y-=.085
        for heading,equation,text in (("Varying theta",r"$\kappa_t=\kappa_0$","Tests only whether the direct competition loading varies with the slow state."),("Free dynamic",r"$\kappa_t=\kappa_0+\delta_1\bar n_t^c+\delta_2q_t^{(2)}$","Estimates the linear and quadratic slope channels independently of theta."),("HSA-restricted dynamic",r"$\kappa_t=\kappa_0+\lambda\theta_0\bar n_t^c+(\lambda\gamma/2)q_t^{(2)}$","Imposes delta_1=lambda theta_0 and delta_2=lambda gamma/2 with sign-unrestricted lambda.")):
            fig.text(.08,y,heading,fontsize=12.5,weight="bold",color=INK);y-=.035;fig.text(.10,y,equation,fontsize=13,color=BLUE);y-=.045;y=_para(fig,y,text,size=9.5);y-=.016
        _para(fig,y,"All models reuse the same paired Gustavo slow-state and Capital IQ cycle draws. Inflation never updates the competition states.",GREEN);pdf.savefig(fig);plt.close(fig)

        fig=_page("Does theta vary?","Time-varying direct loading; full-sample IID posterior",2);fig.text(.075,.862,r"$\theta_t=\theta_0+\gamma\bar n_t^c,\qquad \kappa_t=\kappa_0$",fontsize=15,color=INK);rows,cols=_coefficient_rows(coeff,"varying_theta",["theta_0","gamma"]);_table(fig.add_axes([.055,.50,.89,.25]),rows,["Parameter",*cols],[.14,.215,.215,.215,.215],8.3,1.85);y=.43
        y=_para(fig,y,"theta_0 remains directionally positive: P(theta_0>0) ranges from 0.804 to 0.830 across the four cells. Every 95% interval nevertheless includes zero.");y-=.016
        y=_para(fig,y,"gamma is negative-leaning in all four cells: P(gamma>0) ranges from 0.308 to 0.384, and every interval includes zero.",RED);y-=.016
        _para(fig,y,"Allowing theta to vary does not reveal positive time variation. It converts the earlier constant-theta result into a positive average loading plus an unidentified state interaction.");pdf.savefig(fig);plt.close(fig)

        fig=_page("Can both dynamic channels be estimated freely?","Free-dynamic model; coefficients are not linked by HSA",3);fig.text(.07,.867,r"$\theta_t=\theta_0+\gamma\bar n_t^c,\quad \kappa_t=\kappa_0+\delta_1\bar n_t^c+\delta_2q_t^{(2)}$",fontsize=14,color=INK);rows,cols=_coefficient_rows(coeff,"free_dynamic",["theta_0","gamma","delta_1","delta_2"]);_table(fig.add_axes([.045,.35,.91,.43]),rows,["Parameter",*cols],[.14,.215,.215,.215,.215],7.7,1.85);y=.27
        y=_para(fig,y,"All sixteen target intervals include zero. theta_0 keeps a positive direction (P>0 from 0.750 to 0.847), but gamma remains negative-leaning.");y-=.012
        _para(fig,y,"Neither the linear slope change delta_1 nor the quadratic slope change delta_2 is independently identified. This is the relevant unrestricted benchmark for interpreting a restricted HSA fit.",RED);pdf.savefig(fig);plt.close(fig)

        fig=_page("What does the HSA restriction add?","HSA-restricted dynamic model; lambda is estimated with unrestricted sign",4);fig.text(.075,.866,r"$\delta_1=\lambda\theta_0,\qquad \delta_2=\lambda\gamma/2$",fontsize=16,color=INK);rows,cols=_coefficient_rows(coeff,"hsa_restricted_dynamic",["theta_0","gamma","lambda","delta_1_derived","delta_2_derived"]);_table(fig.add_axes([.04,.29,.92,.49]),rows,["Parameter",*cols],[.14,.215,.215,.215,.215],7.2,1.75);y=.22
        y=_para(fig,y,"Every lambda interval spans zero and both signs. Every derived delta_1 and delta_2 interval also spans zero.");y-=.012
        _para(fig,y,"The restriction reduces some posterior standard deviations because products share information, but this is not independent identification: the unrestricted free-dynamic coefficients that feed the restriction are weak.",RED);pdf.savefig(fig);plt.close(fig)

        fig=_page("Implied time paths","Firm-weighted cycle x negative unemployment gap, full-sample IID",5);models=[("varying_theta",BLUE),("free_dynamic",GREEN),("hsa_restricted_dynamic",GOLD)]
        ax1=fig.add_axes([.10,.56,.82,.27]);ax2=fig.add_axes([.10,.18,.82,.27])
        for model,color in models:
            path=OUT/"draws"/"full"/model/"firm_weighted"/"iid"/"ppi_negative_unemployment_gap.npz";periods,theta,kappa=_path_draws(path);_band(ax1,periods,theta,color,model.replace("_"," "));_band(ax2,periods,kappa,color,model.replace("_"," "))
        ax1.set_title(r"Direct loading $\theta_t$");ax2.set_title(r"Slope $\kappa_t$");ax1.legend(frameon=False,ncol=3,fontsize=8,loc="upper left");ax2.legend(frameon=False,ncol=3,fontsize=8,loc="upper left");ax1.set_ylabel("Coefficient");ax2.set_ylabel("Coefficient")
        fig.text(.10,.10,"Bands are pointwise 95% posterior intervals. They are wide and overlap zero throughout most or all of the sample; visual movement of the median is not evidence that gamma is identified.",fontsize=9,color=RED,wrap=True);pdf.savefig(fig);plt.close(fig)

        fig=_page("Recovery of time variation","Varying-theta model on the actual sample design",6);p=power[(power.parameter=="gamma")];order=["theta_only","gamma_small","gamma_observed_scale","gamma_moderate","gamma_large"];x=np.arange(len(order));ax=fig.add_axes([.10,.56,.82,.27])
        for mode,color,marker in (("propagated_state",BLUE,"o"),("oracle_state",GREEN,"s")):
            z=p[p["mode"]==mode].set_index("scenario").loc[order];ax.plot(x,z.suggestive_rate,marker=marker,color=color,label=mode.replace("_"," "))
        ax.axhline(.8,color=RED,ls="--",lw=1,label="0.80 threshold");ax.set_xticks(x,["0.00","0.05","0.10","0.20","0.40"]);ax.set_xlabel("Injected standardized gamma");ax.set_ylabel("Suggestive recovery rate");ax.set_ylim(-.03,1.04);ax.grid(axis="y",color=GRID,lw=.5);ax.legend(frameon=False,ncol=3,fontsize=8)
        rows=[]
        for scenario in order:
            r=p[(p["mode"]=="propagated_state")&(p.scenario==scenario)].iloc[0];rows.append([f"{r.standardized_true:.2f}",f"{r.suggestive_rate:.3f}",f"{r.strong_rate:.3f}",f"{r.coverage:.3f}",f"{r.mean_sd_ratio:.3f}"])
        _table(fig.add_axes([.14,.19,.72,.26]),rows,["True gamma","Suggestive","Strong","Coverage","Mean SD ratio"],[.20,.20,.20,.20,.20],8.2,1.45);fig.text(.10,.11,"At gamma=0.10, propagated-state strong recovery is 0.067. Reliable strong detection appears only for much larger time variation (0.40: 0.800).",fontsize=9.2,color=RED,wrap=True);pdf.savefig(fig);plt.close(fig)

        fig=_page("Predictive comparison and decision","All dynamic models are compared with constant theta",7);rows=[]
        for (cy,cell),g in comp.groupby(["cycle","cell"]):
            b=g[g.model=="constant_theta"].iloc[0]
            for model in ("varying_theta","free_dynamic","hsa_restricted_dynamic"):
                r=g[g.model==model].iloc[0];rows.append([cy.replace("_weighted",""),"unemp" if "unemployment" in cell else "markup",model.replace("_dynamic","").replace("_"," "),f"{r.elpd_loo-b.elpd_loo:.2f}",f"{r.holdout_elpd-b.holdout_elpd:.2f}",f"{r.holdout_rmse-b.holdout_rmse:.2f}"])
        _table(fig.add_axes([.05,.42,.90,.40]),rows,["Cycle","Activity","Model","Delta LOO","Delta holdout ELPD","Delta RMSE"],[.12,.12,.25,.15,.20,.16],7.1,1.34);y=.35
        gate=manifest["gate"];y=_para(fig,y,f"MCMC passes: observed max R-hat {gate['observed_max_rhat']:.4f}, observed min bulk ESS {gate['observed_min_bulk_ess']:.1f}; recovery max R-hat {gate['recovery_max_rhat']:.4f}, min bulk ESS {gate['recovery_min_bulk_ess']:.1f}.",GREEN);y-=.012
        y=_para(fig,y,"All twelve holdout-ELPD differences are negative. PSIS-LOO is descriptive because influential observations produce Pareto-k above 0.7 in every comparison.");y-=.012
        y=_para(fig,y,"The persistent-AR(1) robustness fits reach the same conclusion: for the firm/unemployment cell, free-dynamic gamma is -0.231 [-1.338, 0.866] and HSA lambda is 0.401 [-5.975, 6.292].",size=9);y-=.008
        _para(fig,y,"Decision: computation succeeds, but gamma, lambda, and the derived HSA slope restrictions are not identified. The data do not support replacing the constant-theta diagnostic with either dynamic specification.",RED);pdf.savefig(fig);plt.close(fig)
    final=ROOT/"output"/"pdf"/"gustavo_state_capitaliq_cycle_dynamic_report.pdf";final.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(pdf_path,final);_write_results(manifest,coeff,comp,power);print(f"wrote {pdf_path}");print(f"wrote {final}")


if __name__=="__main__":build()
