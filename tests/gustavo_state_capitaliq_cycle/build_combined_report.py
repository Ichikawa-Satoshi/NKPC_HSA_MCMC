"""Build the English report for the QoQ free-combined diagnostic."""
from __future__ import annotations

import json,shutil,textwrap
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd
from scipy.stats import norm

ROOT=next(p for p in Path(__file__).resolve().parents if (p/"pyproject.toml").exists())
BUNDLE=Path(__file__).resolve().parent;OUT=BUNDLE/"results"/"free_combined_qoq"
INK="#1f2933";BLUE="#24567a";GREEN="#2f7355";RED="#a33b32";GRID="#c9d1d8"
mpl.rcParams.update({"font.family":"STIXGeneral","mathtext.fontset":"stix","font.size":10,"axes.titlesize":12,"axes.labelsize":10,"figure.facecolor":"white"})


def _page(title,subtitle,number):
    fig=plt.figure(figsize=(8.5,11));fig.text(.065,.955,title,fontsize=19,weight="bold",color=INK);fig.text(.065,.925,subtitle,fontsize=10,color="#52606d");fig.text(.94,.025,str(number),ha="right",fontsize=8,color="#66737f");return fig


def _para(fig,y,text,color=INK,size=10,width=100):
    lines=[]
    for paragraph in text.split("\n"):lines.extend(textwrap.wrap(paragraph,width) or [""])
    fig.text(.075,y,"\n".join(lines),va="top",fontsize=size,color=color,linespacing=1.25);return y-.024*max(1,len(lines))


def _table(ax,rows,columns,widths=None,font=8.3,yscale=1.50):
    ax.axis("off");table=ax.table(cellText=rows,colLabels=columns,cellLoc="center",colLoc="center",loc="center",colWidths=widths)
    table.auto_set_font_size(False);table.set_fontsize(font);table.scale(1,yscale)
    for (r,c),cell in table.get_celld().items():
        cell.set_edgecolor("#7b8794");cell.set_linewidth(.55);cell.set_facecolor("#edf1f4" if r==0 else "white");cell.set_text_props(weight="bold" if r==0 else "normal",color=INK)
    return table


def _interval(mean,lo,hi):return f"{mean:.3f}\n[{lo:.3f}, {hi:.3f}]"


def _label(value):return value.replace("ppi_","").replace("negative_unemployment_gap","negative unemployment gap").replace("inverse_markup","inverse markup").replace("_"," ")


def _write_results(manifest: dict,comparison: pd.DataFrame,coeff: pd.DataFrame):
    lines=["# QoQ free-combined slope and direct-channel diagnostic","","Status: **MOCK DIAGNOSTIC - NOT A STRUCTURAL HSA ESTIMATE**","","## Estimated equation","","```math","\\pi_t^q=a+\\alpha_b\\pi_{t-1}^q+\\alpha_fE_t\\pi_{t+1}^q","+\\left[\\kappa_0+\\delta(\\bar n_t-\\overline{\\bar n})\\right]x_t","-\\theta_{CIQ}\\hat n_{j,t}+\\varepsilon_t.","```","","The saved Gustavo slow-state and Capital IQ AR(2) cycle draws are reused without inflation feedback. `delta` and `theta_CIQ` are free; no relation `delta=lambda*theta_CIQ` is imposed.","","## Direct-only versus free combined","","| Cycle | Error | PPI activity | Direct theta_CIQ | Combined theta_CIQ | P(theta>0) | delta | P(delta>0) | Corr(delta,theta) |","|---|---|---|---:|---:|---:|---:|---:|---:|"]
    for _,r in comparison.iterrows():
        lines.append(f"| {_label(r['cycle'])} | {_label(r['error_model'])} | {_label(r['cell'])} | {r['direct_theta_mean']:.3f} [{r['direct_theta_q2.5']:.3f}, {r['direct_theta_q97.5']:.3f}] | {r['combined_theta_mean']:.3f} [{r['combined_theta_q2.5']:.3f}, {r['combined_theta_q97.5']:.3f}] | {r['combined_theta_p_positive']:.3f} | {r['delta_mean']:.3f} [{r['delta_q2.5']:.3f}, {r['delta_q97.5']:.3f}] | {r['delta_p_positive']:.3f} | {r['delta_theta_correlation']:.3f} |")
    lines.extend(["","## Complete coefficient table: primary IID","","Each cell reports posterior mean and 95% equal-tail interval.","","| Parameter | Firm / inverse markup | Firm / unemployment gap | Revenue / inverse markup | Revenue / unemployment gap |","|---|---:|---:|---:|---:|"])
    iid=coeff[coeff.error_model=="iid"]
    columns=[("firm_weighted","ppi_inverse_markup"),("firm_weighted","ppi_negative_unemployment_gap"),("revenue_weighted","ppi_inverse_markup"),("revenue_weighted","ppi_negative_unemployment_gap")]
    for parameter in ("intercept","alpha_b","alpha_f","kappa_0","delta","theta_CIQ"):
        row=[parameter]
        for cycle,cell in columns:
            r=iid[(iid.cycle==cycle)&(iid.cell==cell)&(iid.parameter==parameter)].iloc[0];row.append(f"{r['mean']:.3f} [{r['q2.5']:.3f}, {r['q97.5']:.3f}]")
        lines.append("| "+" | ".join(row)+" |")
    g=manifest["gate"];lines.extend(["","## Diagnostics and conclusion","",f"- Maximum R-hat: `{g['observed_max_rhat']:.4f}` (required <= `{g['max_rhat_required']}`).",f"- Minimum bulk ESS: `{g['observed_min_bulk_ess']:.1f}` (required >= `{g['min_bulk_ess_required']}`).",f"- The predeclared theta-retention diagnostic passes in `{g['theta_retained_cells']}` of `{g['theta_tested_cells']}` cells.","- Adding the slow-state slope channel does not remove the positive direct-channel update. Under IID unemployment-gap specifications, `P(theta_CIQ>0)=0.829` for both weightings.","- `delta` is not sign-identified: every 95% interval includes zero. The free-combined run supports channel separability, not the HSA cross-equation restriction.","- With unrestricted real `lambda`, the static equality `delta=lambda*theta` is a reparameterization whenever `theta` is nonzero; it is not a fit restriction.",""])
    (BUNDLE/"RESULTS_COMBINED.md").write_text("\n".join(lines))


def _density(ax,path,parameter,title):
    z=np.load(path,allow_pickle=False);names=list(map(str,z["names"]));j=names.index(parameter);draws=z["draws"][:,:,j].reshape(-1);pm=float(z["prior_mean"][j]);ps=float(z["prior_sd"][j]);lo=min(np.percentile(draws,.5),pm-3*ps);hi=max(np.percentile(draws,99.5),pm+3*ps);x=np.linspace(lo,hi,400)
    ax.plot(x,norm.pdf(x,pm,ps),color="#7b8794",lw=1.5,label="prior");ax.hist(draws,bins=45,density=True,color=BLUE,alpha=.35,label="posterior");ax.axvline(0,color=RED,lw=1);ax.set_title(title);ax.grid(axis="y",color=GRID,lw=.5);ax.legend(frameon=False,fontsize=8)


def build():
    manifest=json.loads((OUT/"manifest.json").read_text());comparison=pd.read_csv(OUT/"tables"/"direct_vs_combined.csv");coeff=pd.read_csv(OUT/"tables"/"coefficients.csv");report_dir=OUT/"report";report_dir.mkdir(parents=True,exist_ok=True);pdf_path=report_dir/"gustavo_state_capitaliq_cycle_free_combined_qoq_report.pdf"
    with PdfPages(pdf_path) as pdf:
        fig=_page("Free combined QoQ diagnostic","Does the direct-channel update survive a free slow-state slope channel?",1);y=.855
        fig.text(.075,y,"Research question",fontsize=14,weight="bold",color=INK);y-=.05
        y=_para(fig,y,"The direct-only QoQ model favored a positive theta_CIQ with posterior probability around 0.8. This nested extension asks whether that update remains after allowing the Gustavo slow state to move the NKPC slope.");y-=.02
        fig.text(.075,y,"Frozen measurement order",fontsize=14,weight="bold",color=INK);y-=.05
        for text in ("1. Gustavo annual Q4 observations determine the slow-state bridge.","2. Capital IQ determines the conditional stationary AR(2) cycle.","3. The same paired measurement posterior draws enter both competition channels. Inflation does not update either state.","4. Only delta is added to the direct-only QoQ NKPC."):y=_para(fig,y,text);y-=.008
        fig.text(.075,y-.01,"Nested comparison",fontsize=14,weight="bold",color=INK);fig.text(.10,y-.09,r"$M_{direct}:\quad \kappa_t=\kappa_0,\qquad \theta_{CIQ}\ \mathrm{free}$",fontsize=15,color=INK);fig.text(.10,y-.15,r"$M_{free}:\quad \kappa_t=\kappa_0+\delta(\bar n_t-\overline{\bar n}),\qquad (\delta,\theta_{CIQ})\ \mathrm{free}$",fontsize=15,color=INK);_para(fig,y-.22,"This is a channel-coexistence and separability test. It is not yet an HSA restriction test.",RED);pdf.savefig(fig);plt.close(fig)

        fig=_page("Estimated model","Annualized QoQ PPI and one-quarter-ahead SPF expectations",2);fig.text(.075,.855,r"$\pi_t^q=400(\log P_t-\log P_{t-1})$",fontsize=15,color=INK);fig.text(.075,.785,r"$\pi_t^q=a+\alpha_b\pi_{t-1}^q+\alpha_fE_t\pi_{t+1}^q$",fontsize=16,color=INK);fig.text(.145,.725,r"$+\left[\kappa_0+\delta(\bar n_t-\overline{\bar n})\right]x_t-\theta_{CIQ}\hat n_{j,t}+\varepsilon_t.$",fontsize=16,color=INK);fig.text(.075,.64,r"$\varepsilon_t=u_t\ \mathrm{(primary)},\qquad \varepsilon_t=\rho\varepsilon_{t-1}+u_t\ \mathrm{(robustness)}$",fontsize=14,color=INK)
        y=.55
        for text in ("Centering is performed inside every propagated slow-state draw. Therefore kappa_0 is the slope at the sample-average Gustavo state, while delta is unchanged by the coordinate origin.","The theta_CIQ prior is identical to the direct-only run. The delta prior is zero-mean Gaussian and scaled so a one-standard-deviation interaction has the same prior effect scale as the other competition coefficient.","Four chains, 2,000 iterations, 600 warmup iterations, and thinning by two are used for each of eight cells. The competition measurement draws are byte-identical to the preceding QoQ mock."):y=_para(fig,y,text);y-=.018
        pdf.savefig(fig);plt.close(fig)

        fig=_page("Primary IID coefficient table","Posterior mean with 95% equal-tail interval",3);fig.text(.075,.87,r"$\pi_t^q=a+\alpha_b\pi_{t-1}^q+\alpha_fE_t\pi_{t+1}^q+[\kappa_0+\delta\bar n_t^c]x_t-\theta_{CIQ}\hat n_t+u_t$",fontsize=13.5,color=INK)
        iid=coeff[coeff.error_model=="iid"];columns=[("firm_weighted","ppi_inverse_markup"),("firm_weighted","ppi_negative_unemployment_gap"),("revenue_weighted","ppi_inverse_markup"),("revenue_weighted","ppi_negative_unemployment_gap")];rows=[]
        for parameter in ("intercept","alpha_b","alpha_f","kappa_0","delta","theta_CIQ"):
            row=[parameter]
            for cycle,cell in columns:
                r=iid[(iid.cycle==cycle)&(iid.cell==cell)&(iid.parameter==parameter)].iloc[0];row.append(_interval(r["mean"],r["q2.5"],r["q97.5"]))
            rows.append(row)
        ax=fig.add_axes([.045,.27,.91,.53]);_table(ax,rows,["Parameter","Firm x markup","Firm x unemp. gap","Revenue x markup","Revenue x unemp. gap"],[.16,.21,.21,.21,.21],7.7,1.75);_para(fig,.19,"alpha_b is sharply positive. alpha_f remains close to its prior width. theta_CIQ is positive in posterior mean in every cell, while all delta and theta_CIQ 95% intervals include zero.",color=INK);pdf.savefig(fig);plt.close(fig)

        fig=_page("Does theta_CIQ remain?","Direct-only versus free-combined comparison",4);rows=[]
        for _,r in comparison.iterrows():rows.append([_label(r["cycle"]),"IID" if r["error_model"]=="iid" else "AR(1)",_label(r["cell"]),_interval(r["direct_theta_mean"],r["direct_theta_q2.5"],r["direct_theta_q97.5"]),_interval(r["combined_theta_mean"],r["combined_theta_q2.5"],r["combined_theta_q97.5"]),f"{r['combined_theta_p_positive']:.3f}",f"{r['combined_theta_sd_ratio']:.3f}",f"{r['delta_theta_correlation']:.3f}"])
        ax=fig.add_axes([.035,.31,.93,.53]);_table(ax,rows,["CIQ cycle","Error","PPI activity","Direct theta","Combined theta","P(theta>0)","Post/prior SD","Corr(delta,theta)"],[.14,.08,.17,.16,.16,.11,.10,.10],6.6,1.62);_para(fig,.23,"Result: theta_CIQ passes the predeclared retention diagnostic in all eight cells. The largest absolute delta-theta correlation is 0.352, so the two regressors do not generate severe posterior confounding.",color=GREEN);pdf.savefig(fig);plt.close(fig)

        fig=_page("Prior-to-posterior learning","Primary IID, firm-weighted Capital IQ cycle",5)
        for row,(cell,title) in enumerate((("ppi_inverse_markup","Inverse markup"),("ppi_negative_unemployment_gap","Negative unemployment gap"))):
            path=OUT/"draws"/"firm_weighted"/"iid"/f"{cell}.npz";_density(fig.add_axes([.08,.56-.39*row,.38,.27]),path,"theta_CIQ",f"{title}: theta_CIQ");_density(fig.add_axes([.55,.56-.39*row,.38,.27]),path,"delta",f"{title}: delta")
        fig.text(.08,.91,"Zero-mean Gaussian priors are shown in gray; posterior draws are shown in blue.",fontsize=10,color=INK);pdf.savefig(fig);plt.close(fig)

        fig=_page("Interpretation and next restriction","What has and has not been tested",6);g=manifest["gate"];rows=[["Maximum R-hat",f"{g['observed_max_rhat']:.4f}",f"<= {g['max_rhat_required']}"],["Minimum bulk ESS",f"{g['observed_min_bulk_ess']:.1f}",f">= {g['min_bulk_ess_required']}"],["Theta-retention cells",f"{g['theta_retained_cells']} / {g['theta_tested_cells']}","diagnostic only"],["Computational gate","PASS" if g["computational_pass"] else "FAIL","both convergence rows"]];_table(fig.add_axes([.15,.61,.70,.23]),rows,["Diagnostic","Observed","Required"],[.47,.25,.28],9)
        y=.52;fig.text(.075,y,"Conclusion",fontsize=14,weight="bold",color=INK);y-=.05
        for text in ("Adding delta does not explain away the positive theta_CIQ update. Under primary IID unemployment-gap cells, P(theta_CIQ > 0) is 0.829 for both Capital IQ weightings.","Delta is not sign-identified. Its strongest IID unemployment-gap positive probability is 0.678, and every 95% interval includes zero.","This supports empirical separability of a slow slope regressor and a cyclical direct regressor. It does not establish the HSA equality d kappa(N)/dN = lambda theta(N).", "A static model with unrestricted real lambda cannot test delta = lambda theta, because lambda = delta/theta whenever theta is nonzero. The next genuine HSA test needs an externally disciplined lambda or overidentifying dynamic restrictions."):y=_para(fig,y,text,color=RED if text.startswith("A static") else INK);y-=.012
        pdf.savefig(fig);plt.close(fig)
    final=ROOT/"output"/"pdf"/"gustavo_state_capitaliq_cycle_free_combined_qoq_report.pdf";final.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(pdf_path,final);_write_results(manifest,comparison,coeff);print(f"wrote {pdf_path}");print(f"wrote {final}")


if __name__=="__main__":build()
