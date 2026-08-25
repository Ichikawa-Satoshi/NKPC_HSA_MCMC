"""Build the English equation-first BDS/BED active-firm report."""
from __future__ import annotations

import argparse,json,shutil,sys,textwrap
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd

ROOT=next(p for p in Path(__file__).resolve().parents if (p/"pyproject.toml").exists());sys.path[:0]=[str(ROOT),str(ROOT/"src"),str(ROOT/"tests")]
from tests import _bootstrap  # noqa:F401,E402

BUNDLE=Path(__file__).resolve().parent;INK="#172127";MUTED="#5A646B";BLUE="#235F85";RED="#9A403D";GREEN="#3A765B";GRID="#CBD0D3"


def _style():
    for name in ("STIXGeneral.otf","STIXGeneralItalic.otf","STIXGeneralBol.otf"):
        p=Path("/System/Library/Fonts/Supplemental")/name
        if p.exists():font_manager.fontManager.addfont(str(p))
    plt.rcParams.update({"font.family":"STIXGeneral","mathtext.fontset":"stix","font.size":9.5,"pdf.fonttype":42,"axes.unicode_minus":False,"axes.spines.top":False,"axes.spines.right":False})


def _page(pdf,title,subtitle,n):
    fig=plt.figure(figsize=(8.5,11));fig.text(.065,.958,title,fontsize=16,weight="bold",color=INK,va="top");fig.text(.065,.928,subtitle,fontsize=9.3,color=MUTED,va="top");fig.text(.935,.025,str(n),ha="right",fontsize=8,color=MUTED);return fig


def _para(fig,y,text,width=80,size=9.2,color=INK,x=.085):
    w=textwrap.fill(text,width=width);lines=w.count("\n")+1;fig.text(x,y,w,fontsize=size,color=color,va="top",linespacing=1.35);return y-.0205*lines-.012


def _section(fig,y,title):fig.text(.075,y,title,fontsize=11.4,weight="bold",color=INK,va="top");return y-.035


def _bullet(fig,y,text,color=INK):return _para(fig,y,"- "+text,width=76,size=9.1,color=color,x=.09)


def _eq(fig,y,text,size=12):fig.text(.10,y,text,fontsize=size,color=INK,va="top");return y-.052


def _table(fig,bbox,columns,rows,widths=None,size=7.9,left=(0,)):
    ax=fig.add_axes(bbox);ax.axis("off");tab=ax.table(cellText=rows,colLabels=columns,cellLoc="center",colLoc="center",colWidths=widths,bbox=[0,0,1,1]);tab.auto_set_font_size(False);tab.set_fontsize(size)
    for (r,c),cell in tab.get_celld().items():
        cell.set_edgecolor("#858D92");cell.set_linewidth(.45);cell.PAD=.035;cell.set_facecolor("#EEF0F1" if r==0 else "white")
        if r==0:cell.get_text().set_weight("bold")
        if c in left:cell.get_text().set_ha("left")
    return tab


def _fmt(r):return f"{r['mean']:.3f}\n[{r['q2.5']:.3f}, {r['q97.5']:.3f}]"


def _load(profile):
    out=BUNDLE/"results"/profile
    return out,json.loads((out/"manifest.json").read_text()),pd.read_csv(out/"tables"/"state_parameters.csv"),pd.read_csv(out/"tables"/"state_paths.csv"),pd.read_csv(out/"tables"/"coefficients.csv"),pd.read_csv(out/"tables"/"recovery_power.csv"),pd.read_csv(out/"tables"/"recovery_replications.csv")


def _markdown(profile,manifest,state,coef,power):
    lines=["# BDS/BED external active-firm test: recorded result","",f"Profile: `{profile}`  ",f"Revision: `{manifest['revision']}`  ",f"Status: **{'NOT FOR INFERENCE' if profile!='full' else ('computational gate passed' if manifest['gate']['passed'] else 'full computational gate failed')}**","",
        "## State model","","```math","y_y^{BDS}=\\bar n_{y,Q1}+\\hat n_{y,Q1},\\qquad z_t^{BED}=a_E+\\ell_E\\Delta n_t+e_t^E.","```","",
        "| Parameter | Mean and 95% interval | R-hat | Bulk ESS |","|---|---:|---:|---:|"]
    for _,r in state.iterrows():
        rh="derived" if not np.isfinite(r.rhat) else f"{r.rhat:.3f}";ess="derived" if not np.isfinite(r.ess_bulk) else f"{r.ess_bulk:.0f}"
        lines.append(f"| `{r.parameter}` | {_fmt(r)} | {rh} | {ess} |")
    lines.extend(["","## Free theta_N real-data diagnostic","","```math","\\pi_t=a+\\alpha_b\\pi_{t-1}+\\alpha_fE_t\\pi_{t+1}+\\kappa_0x_t-\\theta_N\\hat n_t+\\varepsilon_t.","```","",
        "| Cell | theta_N | P(theta_N>0) | Post/prior SD | R-hat |","|---|---:|---:|---:|---:|"])
    for _,r in coef[coef.parameter=="theta_N"].iterrows():lines.append(f"| {r.cell} | {_fmt(r)} | {r.p_positive:.3f} | {r.posterior_prior_sd_ratio:.3f} | {r.rhat:.3f} |")
    lines.extend(["","## Recovery","","| Mode | True theta_N | Replicates | Detection rate | Mean estimate | Mean sign probability | Mean post/prior SD |","|---|---:|---:|---:|---:|---:|---:|"])
    for _,r in power.iterrows():lines.append(f"| {r['mode']} | {r.theta_true:.2f} | {int(r.replicates)} | {r.detection_rate:.3f} | {r.mean_estimate:.3f} | {r.mean_sign_probability:.3f} | {r.mean_sd_ratio:.3f} |")
    lines.extend(["","## Interpretation","",f"- Computational gate passed: `{manifest['gate']['passed']}`.",f"- Minimum detectable theta recorded by this profile: `{manifest['minimum_detectable_theta']}`.","- Mock and smoke recovery rates are not inferential power estimates.","- No delta, lambda, HSA restriction, or model-evidence comparison is estimated here.",""])
    return "\n".join(lines)


def build(profile="smoke"):
    _style();out,manifest,state,path,coef,power,reps=_load(profile);report=out/"report";report.mkdir(parents=True,exist_ok=True);pdf_path=report/"active_firm_stock_bds_bed_report.pdf"
    with PdfPages(pdf_path) as pdf:
        fig=_page(pdf,"External active-firm stock and free direct-channel recovery",f"BDS levels x BED timing | {profile} profile",1);y=.87
        y=_section(fig,y,"Required order")
        for text in ("Estimate the active-firm state from BDS and BED only.","Test free theta_N recovery using the actual aggregate sample geometry.","Read the real-data free coefficient only after the recovery gate.","Return to delta and the HSA restriction only if both directions are independently identified."):y=_bullet(fig,y,text)
        y-=.01;y=_section(fig,y,"Cut posterior")
        y=_eq(fig,y,r"$p_{cut}(n,\psi_N,\beta_\pi\mid BDS,BED,\pi)=p(n,\psi_N\mid BDS,BED)\,p(\beta_\pi\mid\pi,n)$",10.8)
        y=_para(fig,y,"Inflation never changes the firm-state decomposition. This bundle does not reuse the effective-competition C state and does not search over C transformations.",color=RED)
        pdf.savefig(fig);plt.close(fig)

        fig=_page(pdf,"Measurement and timing","Firm levels are not establishment flows",2);y=.87
        y=_eq(fig,y,r"$n_t=10(\log N_t-\log N_{1993}^{BDS})$")
        y=_eq(fig,y,r"$y_y^{BDS}=\bar n_{y,Q1}+\hat n_{y,Q1}$")
        y=_eq(fig,y,r"$z_t^{BED}=a_E+\ell_E\Delta n_t+e_t^E$")
        y=_para(fig,y,"BDS FIRM is the annual level anchor. BED births and deaths refer to establishments and only supply a noisy quarterly timing measurement through an estimated loading. The deprecated one-anchor cumulative establishment_stock proxy is not used.")
        p=pd.PeriodIndex(path.period,freq="Q").to_timestamp();ax=fig.add_axes([.10,.15,.82,.38]);g=path.bds_firms.notna();ax.plot(p[g],path.loc[g,"bds_firms"]/1e6,"o-",color=BLUE,label="BDS firms (millions)")
        ax2=ax.twinx();b=path.bed_observed.astype(bool);ax2.plot(p[b],(path.loc[b,"bed_births"]-path.loc[b,"bed_deaths"])/1000,color=RED,alpha=.75,label="BED net establishments (thousands)")
        ax.set_ylabel("BDS firms, millions",color=BLUE);ax2.set_ylabel("BED net establishments, thousands",color=RED);ax.grid(axis="y",color=GRID,lw=.5);ax.set_title("External measurement coverage")
        pdf.savefig(fig);plt.close(fig)

        fig=_page(pdf,"Competition-independent firm state","Slow/cycle decomposition from BDS and BED only",3);y=.87
        y=_eq(fig,y,r"$\bar n_t=\bar n_{t-1}+\mu+\eta_t^{\bar n}$")
        y=_eq(fig,y,r"$\hat n_t=2r\cos(2\pi/P)\hat n_{t-1}-r^2\hat n_{t-2}+\eta_t^{\hat n}$")
        y=_eq(fig,y,r"$\sigma_{\bar n}^2=\omega\tau^2,\qquad\sigma_{\hat n}^2=(1-\omega)\tau^2$")
        ax=fig.add_axes([.10,.13,.82,.48]);ax.plot(p,path.nbar_mean,color=BLUE,label=r"slow $\bar n_t$");ax.fill_between(p,path["nbar_q2.5"],path["nbar_q97.5"],color=BLUE,alpha=.15);ax.plot(p,path.nhat_mean,color=RED,label=r"cycle $\hat n_t$");ax.fill_between(p,path["nhat_q2.5"],path["nhat_q97.5"],color=RED,alpha=.12);ax.scatter(p[g],path.loc[g,"bds_coordinate"],s=11,color=INK,label="BDS annual observation");ax.axhline(0,color=INK,lw=.6);ax.grid(axis="y",color=GRID,lw=.5);ax.legend(frameon=False,ncol=3,fontsize=8);ax.set_title("Posterior firm-state decomposition")
        pdf.savefig(fig);plt.close(fig)

        fig=_page(pdf,"Firm-state identification","State-law and measurement parameters",4)
        rows=[]
        for _,r in state.iterrows():rows.append([r.parameter,_fmt(r),f"{r.rhat:.3f}" if np.isfinite(r.rhat) else "derived",f"{r.ess_bulk:.0f}" if np.isfinite(r.ess_bulk) else "derived"])
        _table(fig,[.12,.30,.76,.55],["Parameter","Posterior mean and 95% interval","R-hat","Bulk ESS"],rows,[.30,.36,.16,.18],7.9)
        fig.text(.10,.23,"The BED loading is estimated rather than fixed. A converged loading does not make establishments equal to firms; it only measures how their timing co-moves with latent firm-stock changes.",fontsize=9.2,color=INK,wrap=True)
        pdf.savefig(fig);plt.close(fig)

        fig=_page(pdf,"Simulation recovery","Minimum detectable free direct effect",5);y=.87
        y=_eq(fig,y,r"$\pi_t=\cdots+\kappa_0x_t-\theta_N\hat n_t+\varepsilon_t$")
        y=_para(fig,y,"Each replicate uses actual dates, expectations, activity, a firm-state draw, recursive inflation persistence, and the estimated MA(3) law. Detection requires a sign-consistent 95% interval, sign probability at least .975, and posterior/prior SD at most .75.")
        pivot=power.pivot(index="theta_true",columns="mode",values=["detection_rate","mean_estimate"]);rows=[]
        for theta,r in pivot.iterrows():rows.append([f"{theta:.2f}",f"{r.get(('detection_rate','propagated_state'),np.nan):.2f}",f"{r.get(('detection_rate','oracle_state'),np.nan):.2f}",f"{r.get(('mean_estimate','propagated_state'),np.nan):.2f}",f"{r.get(('mean_estimate','oracle_state'),np.nan):.2f}"])
        _table(fig,[.11,.47,.78,.25],["True theta_N","Propagated detect","Oracle detect","Propagated mean","Oracle mean"],rows,[.17,.21,.19,.22,.21],7.7)
        ax=fig.add_axes([.13,.13,.72,.25])
        for mode,color,label in (("propagated_state",BLUE,"propagated N posterior"),("oracle_state",GREEN,"oracle known N")):
            d=power[power["mode"]==mode];ax.plot(np.log1p(d.theta_true),d.detection_rate,"o-",color=color,label=label)
        ticks=np.array([0,.1,1,10,30,50]);ax.set_xticks(np.log1p(ticks),["0","0.1","1","10","30","50"])
        ax.axhline(.8,color=RED,ls="--",lw=.9,label="full-profile 80% gate");ax.set_ylim(-.03,1.03);ax.set_xlabel(r"Injected $\theta_N$ (log-spaced display)");ax.set_ylabel("Detection rate");ax.grid(color=GRID,lw=.5);ax.legend(frameon=False)
        if profile!="full":fig.text(.10,.08,"Mock/smoke rates verify the pipeline; they are not a final power curve.",fontsize=9.2,color=RED)
        pdf.savefig(fig);plt.close(fig)

        fig=_page(pdf,"Real-data free theta_N diagnostic","Read only after recovery and convergence",6);y=.87
        y=_eq(fig,y,r"$\pi_t=a+\alpha_b\pi_{t-1}+\alpha_fE_t\pi_{t+1}+\kappa_0x_t-\theta_N\hat n_t+\varepsilon_t$")
        theta=coef[coef.parameter=="theta_N"];rows=[]
        for _,r in theta.iterrows():rows.append([r.cell,_fmt(r),f"{r.p_positive:.3f}",f"{r.posterior_prior_sd_ratio:.3f}",f"{r.rhat:.3f}"])
        _table(fig,[.10,.56,.80,.18],["Cell","theta_N","P(theta_N>0)","Post/prior SD","R-hat"],rows,[.28,.27,.17,.17,.11],8.0)
        y=.48;y=_section(fig,y,"Interpretation discipline")
        for text in ("A narrow interval is evidence only if simulation recovers economically relevant injected effects.","theta_N is free: no fixed lambda transfers slope information into it.","The unemployment-gap cell is validation, not a substitute selected after viewing the markup result."):y=_bullet(fig,y,text)
        pdf.savefig(fig);plt.close(fig)

        fig=_page(pdf,"Gate and next decision","What this run permits",7);y=.87;gate=manifest["gate"]
        rows=[["State maximum R-hat",f"{gate['state_max_rhat']:.3f}",f"<= {gate['max_rhat_required']:.2f}"],["State minimum bulk ESS",f"{gate['state_min_bulk_ess']:.1f}",f">= {gate['min_bulk_ess_required']:.0f}"],["NKPC maximum R-hat",f"{gate['observed_max_rhat']:.3f}",f"<= {gate['max_rhat_required']:.2f}"],["NKPC minimum bulk ESS",f"{gate['observed_min_bulk_ess']:.1f}",f">= {gate['min_bulk_ess_required']:.0f}"],["Computational gate","PASS" if gate["passed"] else "FAIL","all rows"]]
        _table(fig,[.17,.62,.66,.22],["Diagnostic","Observed","Required"],rows,[.48,.25,.27],8.2)
        y=.54;y=_section(fig,y,"Decision")
        if profile!="full":y=_para(fig,y,"This profile is not inferential. Use it to find implementation, mixing, and recovery-design failures before a full power run.",color=RED)
        elif not gate["passed"]:y=_para(fig,y,"The full computational gate failed. Do not interpret theta_N or proceed to an HSA restriction.",color=RED)
        elif manifest["minimum_detectable_theta"] is None:y=_para(fig,y,"No injected effect passes the predeclared 80% recovery gate. Aggregate direct-channel identification is inadequate for the tested range.",color=RED)
        else:y=_para(fig,y,f"The minimum detectable effect is {manifest['minimum_detectable_theta']}. The real-data free coefficient may now be compared with this power threshold before any unrestricted slope/direct model is attempted.")
        y=_section(fig,y,"Still excluded")
        for text in ("No effective-competition C retuning.","No delta or lambda factorization.","No HSA restriction or marginal-likelihood promotion.","No causal competition-policy counterfactual."):y=_bullet(fig,y,text)
        pdf.savefig(fig);plt.close(fig)
    (BUNDLE/"RESULTS.md").write_text(_markdown(profile,manifest,state,coef,power))
    target=ROOT/"output"/"pdf"/"active_firm_stock_bds_bed_report.pdf";target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(pdf_path,target)
    print(f"wrote {pdf_path}");print(f"wrote {BUNDLE/'RESULTS.md'}")
    return pdf_path


def main():
    p=argparse.ArgumentParser();p.add_argument("--profile",choices=("mock","smoke","full"),default="smoke");a=p.parse_args();build(a.profile)


if __name__=="__main__":main()
