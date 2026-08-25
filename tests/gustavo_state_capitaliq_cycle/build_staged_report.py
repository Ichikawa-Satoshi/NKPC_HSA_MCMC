"""Build the English staged-validation report."""
from __future__ import annotations

import json,shutil,textwrap
from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd

ROOT=next(p for p in Path(__file__).resolve().parents if (p/"pyproject.toml").exists());BUNDLE=Path(__file__).resolve().parent;OUT=BUNDLE/"results"/"staged_validation"
INK="#1f2933";BLUE="#24567a";GREEN="#2f7355";RED="#a33b32";GRID="#c9d1d8"
mpl.rcParams.update({"font.family":"STIXGeneral","mathtext.fontset":"stix","font.size":10,"axes.titlesize":12,"figure.facecolor":"white"})


def _page(title,subtitle,n):
    fig=plt.figure(figsize=(8.5,11));fig.text(.065,.955,title,fontsize=19,weight="bold",color=INK);fig.text(.065,.925,subtitle,fontsize=10,color="#52606d");fig.text(.94,.025,str(n),ha="right",fontsize=8,color="#66737f");return fig


def _para(fig,y,text,color=INK,size=10,width=100):
    lines=[]
    for p in text.split("\n"):lines.extend(textwrap.wrap(p,width) or [""])
    fig.text(.075,y,"\n".join(lines),va="top",fontsize=size,color=color,linespacing=1.25);return y-.024*max(1,len(lines))


def _table(ax,rows,cols,widths=None,font=8,scale=1.5):
    ax.axis("off");t=ax.table(cellText=rows,colLabels=cols,cellLoc="center",colLoc="center",loc="center",colWidths=widths);t.auto_set_font_size(False);t.set_fontsize(font);t.scale(1,scale)
    for (r,c),cell in t.get_celld().items():cell.set_edgecolor("#7b8794");cell.set_linewidth(.55);cell.set_facecolor("#edf1f4" if r==0 else "white");cell.set_text_props(weight="bold" if r==0 else "normal",color=INK)
    return t


def _write_results(manifest,power,diff):
    gate=manifest["promotion_gate"];checks=gate["coefficient_checks"];primary=power[(power.activity=="ppi_negative_unemployment_gap")&(power.error_model=="iid")&(power["mode"]=="propagated_state")];lines=["# Staged free-combined validation result","","Status: **STOPPED BY PREDECLARED RECOVERY GATE - DYNAMIC HSA NOT RUN**","","## Staged question","","```math","\\pi_t^q=a+\\alpha_b\\pi_{t-1}^q+\\alpha_fE_t\\pi_{t+1}^q","+[\\kappa_0+\\delta\\bar n_t^c]x_t-\\theta_{CIQ}\\hat n_t+\\varepsilon_t.","```","","Can `delta` and `theta_CIQ` be recovered jointly at empirically relevant standardized effects before imposing any HSA linkage?","","## Promotion-gate result","","| Parameter | Suggestive recovery | Required | Coverage | Null false positive |","|---|---:|---:|---:|---:|"]
    for parameter in ("delta","theta_CIQ"):
        z=checks[parameter];lines.append(f"| `{parameter}` | {z['suggestive_rate']:.3f} | {z['suggestive_required']:.3f} | {z['coverage']:.3f} | {z['null_false_positive_rate']:.3f} |")
    lines.extend(["",f"Recovery convergence: maximum R-hat `{gate['convergence']['max_rhat']:.4f}`, minimum bulk ESS `{gate['convergence']['min_bulk_ess']:.1f}`.","","## Primary recovery by injected standardized effect","","| Scenario | Parameter | Standardized true effect | Suggestive rate | Strong rate | Coverage |","|---|---|---:|---:|---:|---:|"])
    order=["null","direct_observed","slope_observed","both_observed","both_moderate","both_large"]
    for scenario in order:
        for parameter in ("delta","theta_CIQ"):
            z=primary[(primary.scenario==scenario)&(primary.parameter==parameter)]
            if len(z):r=z.iloc[0];lines.append(f"| `{scenario}` | `{parameter}` | {r['standardized_true']:.2f} | {r['suggestive_rate']:.3f} | {r['strong_rate']:.3f} | {r['coverage']:.3f} |")
    lines.extend(["","## Direct-only versus free-combined model comparison","","Positive ELPD differences favor free combined; `BF01 > 1` favors `delta=0`.","","| Cycle | Activity | Delta LOO ELPD | Delta holdout ELPD | Delta holdout RMSE | BF01(delta=0) | Max Pareto k |","|---|---|---:|---:|---:|---:|---:|"])
    for _,r in diff.iterrows():lines.append(f"| {r['cycle']} | {r['cell']} | {r['delta_elpd_loo_combined_minus_direct']:.3f} | {r['delta_holdout_elpd_combined_minus_direct']:.3f} | {r['delta_holdout_rmse_combined_minus_direct']:.3f} | {r['bf01_delta_zero']:.3f} | {r['max_pareto_k']:.3f} |")
    lines.extend(["","All PSIS comparisons have influential observations (`Pareto k > 1`), so LOO is descriptive. Holdout ELPD is lower for free combined in all four cells. The Savage-Dickey diagnostic mildly favors `delta=0`.","","## Decision","","The gate failed (`delta`: 0.10, `theta_CIQ`: 0.333 versus 0.80 required). Oracle-state recovery is similarly weak, so state uncertainty is not the main bottleneck. Dynamic free and HSA-restricted models were not estimated. This prevents a restriction from manufacturing precision that the unrestricted channels do not possess.",""])
    (BUNDLE/"RESULTS_STAGED.md").write_text("\n".join(lines))


def build():
    manifest=json.loads((OUT/"manifest.json").read_text());power=pd.read_csv(OUT/"tables"/"recovery_power.csv",keep_default_na=False);diff=pd.read_csv(OUT/"tables"/"model_comparison_differences.csv");gate=manifest["promotion_gate"];report_dir=OUT/"report";report_dir.mkdir(parents=True,exist_ok=True);pdf_path=report_dir/"gustavo_state_capitaliq_cycle_staged_validation_report.pdf"
    with PdfPages(pdf_path) as pdf:
        fig=_page("Staged HSA identification validation","Recovery and model comparison before any HSA restriction",1);y=.85
        for heading,text in (("Stage 1: unrestricted channels","Estimate the direct-only and free-combined QoQ models with competition states cut from inflation."),("Stage 2: recovery","Inject standardized slope and direct effects into the actual design; compare propagated and oracle states."),("Stage 3: model comparison","Use PSIS-LOO, a 2010Q1-2013Q4 holdout, and a nested Savage-Dickey diagnostic."),("Stage 4: promotion rule","Run dynamic free and HSA-restricted models only if both observed-size effects pass the recovery gate.")):
            fig.text(.075,y,heading,fontsize=13,weight="bold",color=INK);y-=.038;y=_para(fig,y,text);y-=.025
        fig.text(.075,y-.01,r"$\pi_t^q=a+\alpha_b\pi_{t-1}^q+\alpha_fE_t\pi_{t+1}^q+[\kappa_0+\delta\bar n_t^c]x_t-\theta_{CIQ}\hat n_t+\varepsilon_t$",fontsize=14,color=INK);_para(fig,y-.10,"The measurement posterior is unchanged and inflation never updates either competition state.",RED);pdf.savefig(fig);plt.close(fig)

        fig=_page("Injected effects and detection rules","Standardized effects make activity measures comparable",2);fig.text(.09,.84,r"$s_\theta=\theta\,SD(\hat n)/SD(\pi),\qquad s_\delta=\delta\,SD(\bar n^c x)/SD(\pi)$",fontsize=16,color=INK);y=.75
        for text in ("The observed unemployment-gap fit corresponds approximately to s_theta=0.11 and s_delta=0.06.","Suggestive detection requires posterior sign probability at least 0.80 and posterior/prior SD at most 0.75.","Strong detection additionally requires sign probability at least 0.975 and a 95% interval excluding zero.","The primary design uses firm-weighted Capital IQ, IID errors, 30 replications, and both propagated-state and oracle-state modes. AR(1) and inverse-markup checks use 10 replications."):y=_para(fig,y,text);y-=.018
        rows=[["null","0.00","0.00"],["direct observed","0.00","0.11"],["slope observed","0.06","0.00"],["both observed","0.06","0.11"],["both moderate","0.20","0.20"],["both large","0.40","0.40"]];_table(fig.add_axes([.20,.14,.60,.30]),rows,["Scenario","s_delta","s_theta"],[.48,.26,.26],9);pdf.savefig(fig);plt.close(fig)

        fig=_page("Primary propagated-state recovery","PPI x negative unemployment gap, IID, 30 replications",3);primary=power[(power.activity=="ppi_negative_unemployment_gap")&(power.error_model=="iid")&(power["mode"]=="propagated_state")];order=["null","direct_observed","slope_observed","both_observed","both_moderate","both_large"];x=np.arange(len(order));ax=fig.add_axes([.10,.54,.82,.27])
        for parameter,color,marker in (("delta",GREEN,"s"),("theta_CIQ",BLUE,"o")):
            vals=[]
            for s in order:
                z=primary[(primary.scenario==s)&(primary.parameter==parameter)];vals.append(float(z.suggestive_rate.iloc[0]) if len(z) else np.nan)
            ax.plot(x,vals,marker=marker,color=color,label=parameter)
        ax.axhline(.8,color=RED,ls="--",lw=1,label="promotion threshold");ax.set_xticks(x,[s.replace("_","\n") for s in order],fontsize=8);ax.set_ylim(-.03,1.04);ax.set_ylabel("Suggestive recovery rate");ax.grid(axis="y",color=GRID,lw=.5);ax.legend(frameon=False,ncol=3)
        rows=[]
        for scenario in ("both_observed","both_moderate","both_large"):
            for parameter in ("delta","theta_CIQ"):
                r=primary[(primary.scenario==scenario)&(primary.parameter==parameter)].iloc[0];rows.append([scenario.replace("_"," "),parameter,f"{r['standardized_true']:.2f}",f"{r['suggestive_rate']:.3f}",f"{r['strong_rate']:.3f}",f"{r['coverage']:.3f}"])
        _table(fig.add_axes([.10,.15,.82,.27]),rows,["Scenario","Parameter","True effect","Suggestive","Strong","Coverage"],[.24,.18,.15,.15,.14,.14],8,1.42);pdf.savefig(fig);plt.close(fig)

        fig=_page("Oracle-state diagnosis","Removing state uncertainty does not solve the problem",4);rows=[]
        for mode in ("propagated_state","oracle_state"):
            for scenario in ("both_observed","both_moderate","both_large"):
                for parameter in ("delta","theta_CIQ"):
                    r=power[(power.activity=="ppi_negative_unemployment_gap")&(power.error_model=="iid")&(power["mode"]==mode)&(power.scenario==scenario)&(power.parameter==parameter)].iloc[0];rows.append([mode.replace("_"," "),scenario.replace("_"," "),parameter,f"{r['suggestive_rate']:.3f}",f"{r['strong_rate']:.3f}",f"{r['mean_p_positive']:.3f}"])
        _table(fig.add_axes([.08,.31,.84,.51]),rows,["State","Scenario","Parameter","Suggestive","Strong","Mean P(>0)"],[.22,.22,.16,.14,.13,.13],7.6,1.38);y=.23;y=_para(fig,y,"At observed-size effects, oracle recovery is 0.133 for delta and 0.333 for theta_CIQ, compared with 0.100 and 0.333 under propagated states.",color=INK);_para(fig,y-.015,"The dominant limitation is the aggregate quarterly design and signal size, not uncertainty in the extracted competition state.",color=RED);pdf.savefig(fig);plt.close(fig)

        fig=_page("Nested model comparison","Direct-only versus free combined",5);rows=[]
        for _,r in diff.iterrows():rows.append([r["cycle"].replace("_"," "),r["cell"].replace("ppi_","").replace("_"," "),f"{r['delta_elpd_loo_combined_minus_direct']:.3f}",f"{r['delta_holdout_elpd_combined_minus_direct']:.3f}",f"{r['delta_holdout_rmse_combined_minus_direct']:.3f}",f"{r['bf01_delta_zero']:.3f}",f"{r['max_pareto_k']:.3f}"])
        _table(fig.add_axes([.06,.47,.88,.35]),rows,["CIQ cycle","Activity","Delta LOO ELPD","Delta holdout ELPD","Delta RMSE","BF01 delta=0","Max Pareto k"],[.18,.20,.14,.15,.12,.11,.10],7.4,1.55);y=.38
        for text in ("Positive ELPD differences favor free combined. All four holdout ELPD differences are negative, so the added slope channel does not improve held-out predictive density.","BF01 ranges from 1.38 to 1.57 and therefore mildly favors the direct-only restriction delta=0.","Every PSIS comparison has Pareto k above 1. LOO is reported as descriptive only; the holdout and nested-density diagnostics receive greater weight."):y=_para(fig,y,text,color=RED if text.startswith("Every") else INK);y-=.018
        pdf.savefig(fig);plt.close(fig)

        fig=_page("Promotion decision","Dynamic HSA stopped by the predeclared gate",6);checks=gate["coefficient_checks"];rows=[]
        for parameter in ("delta","theta_CIQ"):
            z=checks[parameter];rows.append([parameter,f"{z['suggestive_rate']:.3f}",f">= {z['suggestive_required']:.2f}",f"{z['coverage']:.3f}",f"{z['null_false_positive_rate']:.3f}","FAIL" if z['suggestive_rate']<z['suggestive_required'] else "PASS"])
        _table(fig.add_axes([.10,.60,.80,.24]),rows,["Parameter","Recovery","Required","Coverage","Null false positive","Decision"],[.20,.16,.16,.16,.18,.14],9);y=.51
        fig.text(.075,y,"Decision",fontsize=14,weight="bold",color=INK);y-=.05
        for text in (f"Recovery computation passed: maximum R-hat {gate['convergence']['max_rhat']:.4f}; minimum bulk ESS {gate['convergence']['min_bulk_ess']:.1f}.","The observed-size joint recovery gate failed for both delta and theta_CIQ. Dynamic free and HSA-restricted fits were therefore not estimated.",r"The unrun restriction would have required $\delta_1=\lambda\theta_0$ and $\delta_2=\lambda\gamma/2$. Imposing it now could transfer weak slope information into the direct coefficients.","The valid carried-forward result is narrower: theta_CIQ has an approximately 0.8 positive posterior direction in the actual data and survives adding delta, but the aggregate design cannot reliably recover both channels at those effect sizes."):y=_para(fig,y,text,color=RED if text.startswith("The observed") else INK);y-=.016
        pdf.savefig(fig);plt.close(fig)
    final=ROOT/"output"/"pdf"/"gustavo_state_capitaliq_cycle_staged_validation_report.pdf";final.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(pdf_path,final);_write_results(manifest,power,diff);print(f"wrote {pdf_path}");print(f"wrote {final}")


if __name__=="__main__":build()
