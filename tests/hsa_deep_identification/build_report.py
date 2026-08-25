"""Build the English equation-first deep-identification audit report."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import textwrap

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import font_manager
import numpy as np
import pandas as pd

ROOT=next(p for p in Path(__file__).resolve().parents if (p/"pyproject.toml").exists())
sys.path[:0]=[str(ROOT),str(ROOT/"src"),str(ROOT/"tests")]
from tests import _bootstrap  # noqa:F401,E402

BUNDLE=Path(__file__).resolve().parent
OUT=ROOT/"output"/"pdf"/"hsa_deep_identification_report.pdf"
INK="#192128"; MUTED="#59636D"; BLUE="#245F86"; RED="#9A3C38"; GREEN="#39745A"; GRID="#C9CED2"


def style():
    regular=Path("/System/Library/Fonts/Supplemental/STIXGeneral.otf")
    italic=Path("/System/Library/Fonts/Supplemental/STIXGeneralItalic.otf")
    bold=Path("/System/Library/Fonts/Supplemental/STIXGeneralBol.otf")
    for path in (regular,italic,bold):
        if path.exists(): font_manager.fontManager.addfont(str(path))
    plt.rcParams.update({"font.family":"STIXGeneral","font.size":9.5,"mathtext.fontset":"stix",
        "pdf.fonttype":42,"ps.fonttype":42,"axes.unicode_minus":False,"figure.facecolor":"white",
        "axes.facecolor":"white","axes.spines.top":False,"axes.spines.right":False})


def page(pdf,title,subtitle,number):
    fig=plt.figure(figsize=(8.5,11)); fig.text(.065,.958,title,fontsize=16,weight="bold",color=INK,va="top")
    fig.text(.065,.927,subtitle,fontsize=9.3,color=MUTED,va="top"); fig.text(.935,.025,str(number),ha="right",fontsize=8,color=MUTED)
    return fig


def section(fig,y,title):
    fig.text(.075,y,title,fontsize=11.3,weight="bold",color=INK,va="top"); return y-.034


def para(fig,y,text,width=76,size=9.15,color=INK,x=.085,spacing=1.38):
    wrapped=textwrap.fill(text,width=width); lines=wrapped.count("\n")+1
    fig.text(x,y,wrapped,fontsize=size,color=color,va="top",linespacing=spacing)
    return y-.0205*lines-.011


def bullet(fig,y,text,width=73,color=INK):
    wrapped=textwrap.fill(text,width=width,subsequent_indent="   "); lines=wrapped.count("\n")+1
    fig.text(.09,y,"- "+wrapped,fontsize=9.05,color=color,va="top",linespacing=1.36)
    return y-.0205*lines-.007


def eq(fig,y,text,size=12.2,x=.10):
    fig.text(x,y,text,fontsize=size,color=INK,va="top"); return y-.047


def table(fig,bbox,columns,rows,widths,size=7.8,left=(0,)):
    ax=fig.add_axes(bbox); ax.axis("off")
    tab=ax.table(cellText=rows,colLabels=columns,cellLoc="center",colLoc="center",colWidths=widths,bbox=[0,0,1,1])
    tab.auto_set_font_size(False);tab.set_fontsize(size)
    for (r,c),cell in tab.get_celld().items():
        cell.set_edgecolor("#818A91");cell.set_linewidth(.45);cell.PAD=.035
        cell.set_facecolor("#EEF0F1" if r==0 else "white")
        if r==0:cell.get_text().set_weight("bold")
        if c in left:cell.get_text().set_ha("left")
    return tab


def load_json(path): return json.loads(Path(path).read_text())


def fmt(c): return f"{c['mean']:.3f}\n[{c['q2.5']:.3f}, {c['q97.5']:.3f}]"


def forest(ax,labels,means,lows,highs,colors=None):
    y=np.arange(len(labels))[::-1]; colors=colors or [BLUE]*len(labels)
    for i,yy in enumerate(y): ax.errorbar(means[i],yy,xerr=[[means[i]-lows[i]],[highs[i]-means[i]]],fmt="o",color=colors[i],capsize=2.5,ms=4)
    ax.axvline(0,color=INK,lw=.8);ax.set_yticks(y,labels);ax.grid(axis="x",color=GRID,lw=.5);ax.set_xlabel("Posterior/MLE estimate and 95% interval")


def build():
    style(); OUT.parent.mkdir(parents=True,exist_ok=True)
    screen=pd.read_csv(BUNDLE/"results"/"screen"/"candidate_screen.csv")
    dynamic=pd.read_csv(BUNDLE/"results"/"screen"/"dynamic_screen.csv")
    nonoverlap=pd.read_csv(BUNDLE/"results"/"screen"/"nonoverlap_q4.csv")
    free_diag=pd.read_csv(BUNDLE/"results"/"screen"/"free_channel_diagnostic.csv")
    key_free=load_json(BUNDLE/"results"/"quick"/"joint_ma3"/"annual_allocation_ar2"/"ppi_inverse_markup"/"free.json")
    key_hsa=load_json(BUNDLE/"results"/"quick"/"joint_ma3"/"annual_allocation_ar2"/"ppi_inverse_markup"/"hsa6.json")
    state_ces=load_json(BUNDLE/"results"/"quick"/"joint_ma3"/"annual_allocation_ar2"/"ppi_negative_unemployment_gap"/"ces.json")
    recovery=load_json(BUNDLE/"results"/"mock"/"simulation_recovery.json")
    qoq_root=BUNDLE/"results"/"mock"/"joint_qoq_iid"/"annual_allocation_ar2"
    qoq={cell:load_json(qoq_root/cell/"free.json") for cell in ("ppi_negative_unemployment_gap","ppi_inverse_markup","core_cpi_negative_unemployment_gap","core_cpi_inverse_markup")}
    s1_npz=np.load(BUNDLE/"results"/"quick"/"joint_ma3"/"annual_allocation_ar2"/"ppi_negative_unemployment_gap"/"ces.npz",allow_pickle=False)
    s0_npz=np.load(ROOT/"tests"/"hsa_nested_validation"/"results"/"full"/"draws"/"joint_state_split"/"ppi_negative_unemployment_gap"/"ces.npz",allow_pickle=False)

    with PdfPages(OUT) as pdf:
        # 1
        fig=page(pdf,"HSA NKPC: deep identification audit","Exact Gustavo x Capital IQ N, AR(2) cycle, and overlap-correct inflation errors",1)
        y=section(fig,.85,"Decision")
        y=para(fig,y,"No HSA specification passed the frozen conjunction of parameter identification, theory-consistent unrestricted signs, convergence, and positive kappa paths. The search therefore does not produce a defensible HSA winner, and a formal marginal-likelihood contest is not activated.",68,10.2,RED)
        y=section(fig,y,"What did work")
        y=bullet(fig,y,"The annual-allocation slow-state law reduced the mean slow innovation variance from about 0.028 to 0.001 while preserving q = bar q + hat q exactly.")
        y=bullet(fig,y,"The state-only S1 run converged: maximum rank R-hat 1.007 and exact identity error 2.2e-16.")
        y=bullet(fig,y,"The MA(3) joint FFBS sampler matched the independently computed dense Gaussian conditional in numerical recovery tests.")
        y=section(fig,y,"Why HSA still fails")
        y=bullet(fig,y,"In the leading free-channel cell, theta = -0.007 [-0.204, 0.197], P(theta > 0) = 0.471, and posterior/prior SD = 0.971.")
        y=bullet(fig,y,"Fixed lambda transfers slow-slope information into a combined coefficient; it does not identify the direct term. At lambda = 6, P(theta > 0) is only 0.929 and the kappa-path gate fails.")
        y=bullet(fig,y,"QoQ inflation, no intercept, alpha_b + alpha_f = 1, non-overlapping Q4 data, current/lagged timing, and dynamic theta all fail to restore joint identification.")
        y=section(fig,y,"Interpretation")
        para(fig,y,"The remaining obstacle is not MCMC convergence. It is lack of independent information about the scale and inflation loading of the cyclical competition state. More chain length cannot turn a posterior/prior SD ratio near one into data identification.",72)
        pdf.savefig(fig);plt.close(fig)

        # 2
        fig=page(pdf,"Data and exact competition identity","What is observed, allocated, and estimated",2);y=section(fig,.85,"Quarterly total competition")
        y=eq(fig,y,r"$q_t=\bar q_t+\hat q_t\quad\mathrm{exactly}$")
        y=para(fig,y,"Gustavo effective-firm counts provide the annual Q4 level. Capital IQ within-year movements allocate each annual change when available; missing quarters use the externally estimated average quarterly allocation profile. Allocation uncertainty is sampled without inflation feedback.")
        y=section(fig,y,"S0: quarterly local level plus AR(2) cycle")
        y=eq(fig,y,r"$\bar q_t=\bar q_{t-1}+\eta^b_t$")
        y=eq(fig,y,r"$\hat q_t=2r\cos(2\pi/P)\hat q_{t-1}-r^2\hat q_{t-2}+\eta^h_t$")
        y=section(fig,y,"S1: average annual allocation as the slow transition mean")
        y=eq(fig,y,r"$\Delta\bar q_{yq}=m_q\,\Delta G_y+\eta^b_{yq}$")
        y=para(fig,y,"S1 is a conditional decomposition law, not an extra observation of the slow state. Quarterly deviations retain an estimated variance. They are not forced to sum to zero, which would mechanically equate adjacent Q4 cycle states.")
        y=section(fig,y,"Variance-share parameterization")
        y=eq(fig,y,r"$\sigma_{\bar N}^2=\omega\tau^2,\qquad \sigma_{\hat N}^2=(1-\omega)\tau^2$")
        y=bullet(fig,y,"tau^2 is total state innovation variance; omega is its slow-state share.")
        bullet(fig,y,"No measurement error is added to q = bar q + hat q.")
        pdf.savefig(fig);plt.close(fig)

        # 3
        fig=page(pdf,"Inflation equations and errors","The exact equations used in the comparison",3);y=section(fig,.85,"CES")
        y=eq(fig,y,r"$\pi_t=a+\alpha_b\pi_{t-1}+\alpha_fE_t\pi_{t+1}+\kappa_0x_t+\varepsilon_t$")
        y=section(fig,y,"Free static channels")
        y=eq(fig,y,r"$\pi_t=a+\alpha_b\pi_{t-1}+\alpha_fE_t\pi_{t+1}+(\kappa_0+\delta\bar q_t)x_t-\theta\hat q_{t-j}+\varepsilon_t$")
        y=section(fig,y,"HSA static, fixed lambda")
        y=eq(fig,y,r"$\pi_t=\cdots+(\kappa_0+\lambda\theta\bar q_t)x_t-\theta\hat q_{t-j}+\varepsilon_t$")
        y=section(fig,y,"HSA dynamic")
        y=eq(fig,y,r"$\theta_t=\theta_0+\gamma\bar q_t$")
        y=eq(fig,y,r"$\kappa_t=\kappa_0+\lambda\theta_0\bar q_t+\frac{\lambda\gamma}{2}\bar q_t^2$")
        y=section(fig,y,"Frequency-consistent disturbances")
        y=eq(fig,y,r"$\varepsilon_t=v_t+\psi_1v_{t-1}+\psi_2v_{t-2}+\psi_3v_{t-3}$")
        y=bullet(fig,y,"Year-over-year inflation uses an invertible MA(3), required by four-quarter overlap.")
        y=bullet(fig,y,"Annualized QoQ inflation uses the genuine one-quarter-ahead SPF forecast and an IID diagnostic likelihood; AR(1) was screened separately.")
        y=bullet(fig,y,"The YoY IID model is a rejected placebo, not evidence.")
        pdf.savefig(fig);plt.close(fig)

        # 4
        fig=page(pdf,"Identification geometry of lambda","Why free-lambda static HSA is not a restriction",4);y=section(fig,.85,"Static mapping")
        y=eq(fig,y,r"$\delta=\lambda\theta,\qquad \lambda=\delta/\theta\ \ (\theta\ne0)$")
        y=para(fig,y,"With unrestricted real lambda, every free pair (delta, theta) with theta nonzero can be written in HSA form. Therefore free-lambda static HSA is observationally equivalent to the free combined model, apart from its prior parameterization. It does not reduce likelihood dimension.")
        y=section(fig,y,"What must be identified first")
        y=bullet(fig,y,"The slow-slope coefficient delta must be learned from bar q_t x_t.")
        y=bullet(fig,y,"The direct coefficient theta must be learned independently from -hat q_t.")
        y=bullet(fig,y,"Only then can lambda = delta/theta have a stable economic interpretation.")
        y=section(fig,y,"Observed diagnostic")
        rows=[["Free lambda, S0 / PPI gap","0.006 [-0.125, 0.154]","0.30 [-13.56, 12.39]"],
              ["Free lambda, S0 / PPI markup","-0.010 [-0.157, 0.178]","-1.51 [-18.70, 14.51]"],
              ["Free lambda, S1 / PPI gap","-0.007 [-0.158, 0.126]","-0.32 [-14.33, 12.49]"],
              ["Free lambda, S1 / PPI markup","0.001 [-0.177, 0.169]","-0.66 [-17.57, 17.89]"]]
        table(fig,[.09,.35,.82,.20],["Diagnostic","theta","lambda"],rows,[.46,.27,.27],8.0,(0,))
        y=.30;y=section(fig,y,"Dynamic model")
        y=para(fig,y,"Dynamic HSA does impose a nonlinear cross-coefficient relation, but its extra theta-path and quadratic kappa-path parameters also failed the frozen path-sign gates in every screened cell.")
        para(fig,y,"A positive prior on lambda would encode theory. It cannot be reported as data-based sign identification, so it was not used to manufacture success.",72)
        pdf.savefig(fig);plt.close(fig)

        # 5
        fig=page(pdf,"Frozen search protocol","Every candidate remains in the manifest",5)
        rows=[
            ["S0 + YoY AR(1)","Existing exact-N benchmark","Converged; theta weak; slow variance too large"],
            ["S0/S1 + YoY MA(3)","Overlap-correct joint sampler","S1 state stable; free theta not learned"],
            ["QoQ + 1q SPF","Avoid overlap structurally","All four theta posteriors remain prior-like"],
            ["Q4-only YoY","Valid non-overlap IID","Too few observations; split instability"],
            ["No intercept","Structural sensitivity","No change in theta learning"],
            ["alpha_b + alpha_f = 1","Hybrid NKPC normalization","No change in theta learning"],
            ["Current vs lag 1","Timing sensitivity","Signs are not stable across samples"],
            ["Free/fixed lambda","Restriction diagnostic","Fixed version borrows slope information"],
            ["Static vs dynamic","Nested HSA test","No dynamic path-sign pass"],
            ["HP/EWMA paths","Screening only","Never eligible as final state posterior"],
        ]
        table(fig,[.075,.30,.85,.50],["Candidate","Purpose","Outcome"],rows,[.23,.31,.46],7.8,(0,1,2))
        y=.265;y=section(fig,y,"Frozen gates")
        y=bullet(fig,y,"R-hat <= 1.01; bulk and tail ESS >= 800; exact identity error <= 1e-10.")
        y=bullet(fig,y,"95% interval excludes zero; unrestricted sign probability >= 0.975; posterior/prior SD <= 0.75.")
        y=bullet(fig,y,"theta_t and kappa_t positive over at least 95% of dates with posterior probability >= 0.95.")
        bullet(fig,y,"Formal log Bayes factor > log(3) is evaluated only after all earlier gates pass.")
        pdf.savefig(fig);plt.close(fig)

        # 6
        fig=page(pdf,"Screening outcomes by cell","No deterministic or likelihood screen passes the full conjunction",6)
        cells=[("ppi","negative_unemployment_gap"),("ppi","inverse_markup"),("core_cpi","negative_unemployment_gap"),("core_cpi","inverse_markup")]
        rows=[]
        for method,subset in [
            ("YoY MA(3), S0 fixed state",free_diag[free_diag.state.eq("s0_quarterly_local_level_ar2")]),
            ("QoQ IID/AR(1), best diagnostic",screen[screen.sample_split.eq("full")]),
            ("Q4-only IID, best diagnostic",nonoverlap[nonoverlap.sample_split.eq("full")]),
            ("Dynamic, best path diagnostic",dynamic[dynamic.sample_split.eq("full")]),
        ]:
            row=[method]
            for price,activity in cells:
                z=subset[(subset.price==price)&(subset.activity==activity)]
                if z.empty: row.append("--"); continue
                if "theta_p_positive" in z: best=z.loc[z.theta_p_positive.idxmax()]; m=best.theta_mean; p=best.theta_p_positive
                elif "theta_p" in z: best=z.loc[z.theta_p.idxmax()]; m=best.theta; p=best.theta_p
                elif "theta0_p_positive" in z: best=z.loc[z.theta0_p_positive.idxmax()]; m=best.theta0_mean; p=best.theta0_p_positive
                else: m=p=np.nan
                row.append(f"{m:.3f}\nP+ {p:.2f}")
            rows.append(row)
        table(fig,[.055,.50,.89,.30],["Screen","PPI x gap","PPI x markup","Core x gap","Core x markup"],rows,[.29,.1775,.1775,.1775,.1775],7.45,(0,))
        y=.44;y=section(fig,y,"Read this table carefully")
        y=para(fig,y,"Each cell shows the most positive result within the declared diagnostic family, so it is deliberately generous. Even this best-case display does not imply selection: discovery/validation stability, kappa-path positivity, posterior learning, and convergence must also pass.")
        y=bullet(fig,y,f"The full candidate screen contains {len(screen):,} likelihood rows; screen_identified = {int(screen.screen_identified.sum())}.")
        y=bullet(fig,y,f"The dynamic screen contains {len(dynamic):,} rows; joint theta/kappa path passes = {int(((dynamic.theta_positive_95pct_dates>=.95)&(dynamic.kappa_positive_95pct_dates>=.95)).sum())}.")
        y=bullet(fig,y,"Full-sample-only positive results reverse or become imprecise in the discovery/validation split and are not promoted.")
        pdf.savefig(fig);plt.close(fig)

        # 7 state plot
        fig=page(pdf,"Competition-state decomposition","S1 fixes the variance-allocation problem but not theta identification",7)
        ax=fig.add_axes([.09,.51,.84,.30]);p=pd.PeriodIndex(s1_npz["periods"].astype(str),freq="Q");x=np.arange(len(p))
        for z,label,color,ls in [(s0_npz,"S0 slow mean",RED,"--"),(s1_npz,"S1 slow mean",BLUE,"-")]:
            b=z["nbar"].mean(axis=(0,1)); ax.plot(x,b,label=label,color=color,lw=1.2,ls=ls)
        total=s1_npz["n_total"].mean(axis=(0,1));ax.plot(x,total,label="Total q",color=INK,lw=.8,alpha=.55)
        ticks=np.arange(0,len(p),20);ax.set_xticks(ticks,[str(p[i].year) for i in ticks]);ax.set_ylabel("Centered competition coordinate");ax.legend(frameon=False,ncol=3,fontsize=8);ax.grid(axis="y",color=GRID,lw=.45)
        rows=[["S0 local level", "0.72", "0.028", "0.011", "large slow innovations"],
              ["S1 allocation mean", f"{state_ces['state']['omega_mean']:.3f}", f"{state_ces['state']['slow_innovation_variance_mean']:.4f}", f"{state_ces['state']['cycle_innovation_variance_mean']:.4f}", "slow movement mostly in mean"]]
        table(fig,[.09,.29,.82,.15],["State law","E[omega]","slow var","cycle var","Reading"],rows,[.23,.13,.15,.15,.34],7.8,(0,4))
        y=.255;y=section(fig,y,"Caution")
        para(fig,y,"S1 uses the observed annual Gustavo change to form the slow transition mean. It is an admissible conditional decomposition requested by the design, but its sharper split is partly supplied by that law. It is not independent evidence from a second cycle-specific measurement.")
        pdf.savefig(fig);plt.close(fig)

        # 8 coefficients
        fig=page(pdf,"Leading free-channel posterior","PPI x inverse markup, S1, YoY MA(3), quick joint run",8)
        y=.84;y=eq(fig,y,r"$\pi_t=a+\alpha_b\pi_{t-1}+\alpha_fE_t\pi_{t+1}+(\kappa_0+\delta\bar q_t)x_t-\theta\hat q_t+\varepsilon_t$")
        order=["intercept","alpha_b","alpha_f","kappa_0","delta_s","theta"]
        labels={"intercept":r"$a$","alpha_b":r"$\alpha_b$","alpha_f":r"$\alpha_f$","kappa_0":r"$\kappa_0$","delta_s":r"$\delta$","theta":r"$\theta$"}
        rows=[]
        for name in order:
            c=key_free["coefficients"][name]; rows.append([labels[name],fmt(c),f"{c['p_positive']:.3f}",f"{c['posterior_prior_sd_ratio']:.3f}",f"{c['rhat']:.4f}",f"{c['ess_bulk']:.0f}"])
        table(fig,[.09,.42,.82,.31],["Coefficient","mean\n[95% interval]","P(>0)","post/prior SD","R-hat","bulk ESS"],rows,[.13,.30,.13,.17,.13,.14],7.7,(0,))
        y=.37;y=section(fig,y,"Gate interpretation")
        y=bullet(fig,y,"alpha_b and alpha_f are identified and positive.")
        y=bullet(fig,y,"kappa_0 and delta do not reach the 0.975 sign gate; the implied kappa path is frequently negative.")
        y=bullet(fig,y,"theta has excellent Monte Carlo precision but no statistical learning: its posterior width is 97% of its prior width.",color=RED)
        para(fig,y,"This distinction is central: a well-converged posterior centered near zero is evidence of weak empirical identification, not a convergence problem.")
        pdf.savefig(fig);plt.close(fig)

        # 9 forest sensitivity
        fig=page(pdf,"Direct-channel sensitivity","Every admissible reformulation leaves theta near its prior",9)
        labels=[];means=[];lows=[];highs=[];colors=[]
        entries=[("S1 YoY MA(3), free",key_free["coefficients"]["theta"],BLUE),
                 ("S1 YoY MA(3), HSA lambda=6",key_hsa["coefficients"]["theta"],RED)]
        for cell,label in [("ppi_negative_unemployment_gap","QoQ PPI x gap"),("ppi_inverse_markup","QoQ PPI x markup"),("core_cpi_negative_unemployment_gap","QoQ Core x gap"),("core_cpi_inverse_markup","QoQ Core x markup")]:
            entries.append((label,qoq[cell]["coefficients"]["theta"],GREEN))
        for label,c,color in entries: labels.append(label);means.append(c["mean"]);lows.append(c["q2.5"]);highs.append(c["q97.5"]);colors.append(color)
        ax=fig.add_axes([.20,.43,.70,.38]);forest(ax,labels,means,lows,highs,colors);ax.set_xlim(-.25,.25)
        rows=[["No intercept, PPI gap","-0.023 [-0.238, 0.190]","1.04"],
              ["No intercept, PPI markup","-0.018 [-0.227, 0.176]","0.96"],
              ["alpha sum one, PPI gap","-0.010 [-0.183, 0.191]","0.98"],
              ["alpha sum one, PPI markup","-0.009 [-0.222, 0.207]","1.01"]]
        table(fig,[.12,.20,.76,.17],["Additional restriction","theta","post/prior SD"],rows,[.48,.32,.20],7.8,(0,))
        para(fig,.145,"The HSA lambda=6 interval is narrower because theta multiplies a combined slow-slope/direct regressor. The free-channel result shows that the direct loading itself is not identified.",96)
        pdf.savefig(fig);plt.close(fig)

        # 10 dynamic/nonoverlap
        fig=page(pdf,"Non-overlap and dynamic tests","Why isolated positive coefficients are not adopted",10)
        y=section(fig,.85,"Q4-only non-overlapping YoY")
        y=para(fig,y,"Using Q4 observations only makes the YoY error non-overlapping and therefore permits IID errors. The most favorable full-sample result is Core CPI x unemployment gap under S0: theta = 1.755 [0.372, 3.138] and delta = 0.078 [0.028, 0.128].")
        y=bullet(fig,y,"The discovery sample instead gives delta = -0.226 [-0.459, 0.006].")
        y=bullet(fig,y,"The validation segment has only 14 annual observations and cannot stabilize the state or coefficients.")
        y=bullet(fig,y,"This is a full-sample reversal, not a validated discovery; it is not a winner.")
        y=section(fig,y,"Dynamic theta")
        y=eq(fig,y,r"$\theta_t=\theta_0+\gamma\bar q_t,\qquad \kappa_t=\kappa_0+\lambda\theta_0\bar q_t+\lambda\gamma\bar q_t^2/2$")
        n_free=int(dynamic.model.eq("free_dynamic").sum()); n_hsa=int(dynamic.model.eq("hsa_dynamic").sum())
        rows=[["Free dynamic",str(n_free),"0","theta and kappa path gates"],
              ["HSA dynamic, lambda 3/6/9",str(n_hsa),"0","theta and kappa path gates"],
              ["All dynamic rows",str(len(dynamic)),"0","full conjunction"]]
        table(fig,[.11,.38,.78,.17],["Family","Rows","Passes","Binding failure"],rows,[.40,.15,.15,.30],8.0,(0,3))
        y=.33;y=section(fig,y,"Nested conclusion")
        para(fig,y,"Adding gamma and the quadratic slope term increases parameter dimension but does not supply an independent cycle measurement. Dynamic HSA is therefore screened out before expensive joint MCMC, as required by the predeclared protocol.")
        pdf.savefig(fig);plt.close(fig)

        # 11 convergence/recovery
        fig=page(pdf,"Convergence and numerical validation","Separating sampler quality from empirical identification",11)
        hsa_min_bulk=min(key_hsa['diagnostics']['ess_bulk'].values()); hsa_min_tail=min(key_hsa['diagnostics']['ess_tail'].values())
        rows=[["S1 CES state-only quick",f"{state_ces['diagnostics']['max_rhat']:.4f}","409","448","2.2e-16","state stable"],
              ["S1 free PPI markup quick",f"{key_free['diagnostics']['max_rhat']:.4f}","338","427","2.2e-16","theta ESS 1350"],
              ["S1 HSA lambda=6 quick",f"{key_hsa['diagnostics']['max_rhat']:.4f}",f"{hsa_min_bulk:.0f}",f"{hsa_min_tail:.0f}","2.2e-16","R-hat gate miss"],
              ["Conditional FFBS test","--","6000 draws","--","algebraic","dense moments match"]]
        table(fig,[.07,.59,.86,.22],["Run","max R-hat","min bulk ESS","min tail ESS","identity","Reading"],rows,[.25,.12,.14,.14,.13,.22],7.4,(0,5))
        y=.54;y=section(fig,y,"Simulation recovery")
        p=recovery["parameters"]
        rows=[]
        for name in ("alpha_b","alpha_f","kappa_0","delta_s","theta","omega","tau","cycle_damping","cycle_period","psi_1","psi_2","psi_3"):
            c=p[name];rows.append([name,f"{c['truth']:.3f}",f"{c['median']:.3f}",f"[{c['q2.5']:.3f}, {c['q97.5']:.3f}]","yes" if c["covered"] else "no"])
        table(fig,[.12,.17,.76,.33],["Parameter","Truth","Median","95% interval","Covered"],rows,[.24,.14,.16,.31,.15],6.9,(0,))
        para(fig,.14,"All truths are covered, validating implementation. But moderate true theta = 0.16 still does not exclude zero, documenting limited design power rather than an algorithmic failure.",72)
        pdf.savefig(fig);plt.close(fig)

        # 12 evidence
        fig=page(pdf,"Model evidence and the stopping rule","Why no formal Bayes-factor claim is reported",12);y=section(fig,.85,"Predeclared sequence")
        y=eq(fig,y,r"$\mathrm{convergence}\rightarrow\mathrm{identification}\rightarrow\mathrm{signs}\rightarrow\mathrm{prediction/evidence}$")
        y=para(fig,y,"Integrated model evidence is meaningful only for an economically admissible, identified model. A marginal likelihood can reward a tight restriction even when the restriction merely transfers information between unidentified components.")
        y=section(fig,y,"What the cheap screen says")
        y=bullet(fig,y,"Some full-sample PPI x inverse-markup fixed-state HSA regressions have lower BIC than CES.")
        y=bullet(fig,y,"The corresponding free model does not identify theta, and fixed lambda produces negative kappa paths.")
        y=bullet(fig,y,"Discovery/validation performance is unstable; dynamic models have no joint path-sign pass.")
        y=section(fig,y,"Formal evidence decision")
        y=para(fig,y,"Because zero HSA candidates pass the earlier gates, Chib/bridge marginal likelihood and leave-future-out winner selection are not run. Reporting a favorable approximate number from a failed model would violate the frozen protocol.",70,9.7,RED)
        y=section(fig,y,"What would change the conclusion")
        y=bullet(fig,y,"An independent quarterly measurement that loads specifically on cyclical competition, not another deterministic filter of total N.")
        y=bullet(fig,y,"A longer or higher-frequency competition series with stable coverage and a measurement model validated outside the inflation equation.")
        y=bullet(fig,y,"A theory-derived scale for lambda that is coherent with the transformed N coordinate; this would be calibration evidence, not estimated lambda identification.")
        bullet(fig,y,"A new sample that validates the direct sign out of sample. More iterations alone are not sufficient.")
        pdf.savefig(fig);plt.close(fig)

        # 13 commands
        fig=page(pdf,"Reproducibility appendix","Commands and saved artifacts",13);y=section(fig,.85,"Run the transparent screens")
        cmds=[
            "python tests/hsa_deep_identification/screen.py",
            "python tests/hsa_deep_identification/nonoverlap_screen.py",
            "python tests/hsa_deep_identification/dynamic_screen.py",
        ]
        for c in cmds:y=para(fig,y,c,100,8.6,INK,.095)
        y=section(fig,y,"Run joint models")
        cmds=[
            "python tests/hsa_deep_identification/run_joint.py --profile mock",
            "python tests/hsa_deep_identification/run_joint.py --profile quick --architectures annual_allocation_ar2 --cells ppi_inverse_markup --models free hsa6",
            "python tests/hsa_deep_identification/run_qoq.py --profile mock --architectures annual_allocation_ar2 --models ces free",
            "python tests/hsa_deep_identification/simulation_recovery.py --profile mock",
        ]
        for c in cmds:y=para(fig,y,c,105,8.3,INK,.095)
        y=section(fig,y,"Validate and rebuild")
        for c in ["pytest -q tests/hsa_deep_identification/test_joint_ma3.py","python tests/hsa_deep_identification/build_report.py"]:
            y=para(fig,y,c,105,8.5,INK,.095)
        y=section(fig,y,"Files")
        y=bullet(fig,y,"Specification: tests/hsa_deep_identification/SPECIFICATION.md")
        y=bullet(fig,y,"Frozen configuration: tests/hsa_deep_identification/config.yaml")
        y=bullet(fig,y,"All screen rows and joint draws: tests/hsa_deep_identification/results/")
        y=bullet(fig,y,"Final report: output/pdf/hsa_deep_identification_report.pdf")
        para(fig,y,"The full profile is intentionally not launched for models that fail the quick identification or sign gates. This is the planned stopping rule, not selective omission.",72)
        pdf.savefig(fig);plt.close(fig)

    print(OUT)


if __name__=="__main__": build()
