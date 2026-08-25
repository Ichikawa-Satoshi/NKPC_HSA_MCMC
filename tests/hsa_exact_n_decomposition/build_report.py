"""Build and visually-verifiable report for the exact-N decomposition experiment."""
from __future__ import annotations
import json,shutil,subprocess
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np,pandas as pd
from scipy.stats import norm

import sys as _sys,pathlib as _pathlib
_ROOT=next(p for p in _pathlib.Path(__file__).resolve().parents if (p/"pyproject.toml").exists())
_sys.path[:0]=[str(_ROOT),str(_ROOT/"src"),str(_ROOT/"tests")]
from tests import _bootstrap  # noqa:F401,E402
from nkpc_hsa.config import load_yaml  # noqa:E402
from tests.hsa_lambda_dynamic.functions import MODEL_LABELS,derived_paths,load_fit  # noqa:E402
from tests.hsa_exact_n_decomposition.functions import load_exact_data,load_states  # noqa:E402
BUNDLE=Path(__file__).resolve().parent;BLUE="#0072B2";ORANGE="#D55E00";GREEN="#009E73";PURPLE="#CC79A7"
def esc(x):return str(x).replace("_",r"\_").replace("%",r"\%")
def band(x):return x.mean(0),np.percentile(x,2.5,0),np.percentile(x,97.5,0)


def figures(exact,states,fits,manifest,out):
    out.mkdir(parents=True,exist_ok=True);per=pd.PeriodIndex(states.periods,freq="Q").to_timestamp()
    # Allocation mean and uncertainty examples.
    avg=np.array(manifest["allocation"]["average_weights"]);years=sorted(map(int,manifest["allocation"]["mean_weights"]))
    fig,ax=plt.subplots(figsize=(10.8,4));
    for q,c in enumerate([BLUE,ORANGE,GREEN,PURPLE]):ax.plot(years,[manifest["allocation"]["mean_weights"][str(y)][q] for y in years],color=c,lw=1.2,label=f"Q{q+1}");ax.axhline(avg[q],color=c,ls=":",lw=.8)
    ax.axhline(0,color="black",lw=.6);ax.set(title="Posterior-mean allocation weights (dotted: missing-year prior mean)",xlabel="year",ylabel="share of annual Gustavo change");ax.legend(frameon=False,ncol=4);fig.tight_layout();fig.savefig(out/"allocation.png",dpi=200);plt.close(fig)
    # Exact decomposition.
    ntm,ntl,nth=band(states.n_total.reshape(-1,len(per)));nbm,nbl,nbh=band(states.nbar.reshape(-1,len(per)));nhm,nhl,nhh=band(states.nhat.reshape(-1,len(per)))
    fig,axs=plt.subplots(2,1,figsize=(11,6.8),sharex=True);axs[0].plot(per,ntm,color="black",lw=1,label=r"$N_t$");axs[0].plot(per,nbm,color=GREEN,lw=2,label=r"$\bar N_t$");axs[0].fill_between(per,nbl,nbh,color=GREEN,alpha=.16);axs[0].legend(frameon=False,ncol=2);axs[1].plot(per,nhm,color=BLUE,lw=1.5,label=r"$\hat N_t$");axs[1].fill_between(per,nhl,nhh,color=BLUE,alpha=.16);axs[1].axhline(0,color="black",lw=.5);axs[1].legend(frameon=False);axs[1].set_xlabel("year");fig.suptitle(r"Exact decomposition: $N_t=\bar N_t+\hat N_t$");fig.tight_layout();fig.savefig(out/"n_decomposition.png",dpi=200);plt.close(fig)
    # State parameters prior/posterior.
    fig,axs=plt.subplots(1,3,figsize=(11,3.4));vals=[states.omega.ravel(),states.tau.ravel(),states.rho.ravel()];labs=[r"$\omega$",r"$\tau$",r"$\rho$"]
    for ax,v,l in zip(axs,vals,labs):ax.hist(v,bins=45,density=True,color=BLUE,alpha=.7);ax.axvline(v.mean(),color=GREEN);ax.set_title(f"{l}: mean={v.mean():.3f}");ax.set_yticks([])
    fig.suptitle("N-state posterior parameters");fig.tight_layout();fig.savefig(out/"state_parameters.png",dpi=200);plt.close(fig)
    # Prior posterior HSA dynamic.
    fit=fits["hsa_dynamic"];flat=fit.draws.reshape(-1,fit.draws.shape[-1]);pars=["kappa_0","theta_0","gamma","lambda"];fig,axs=plt.subplots(2,2,figsize=(9.2,6.2))
    for ax,n in zip(axs.ravel(),pars):v=flat[:,fit.names.index(n)];pm=fit.prior_mean[n];ps=fit.prior_sd[n];lo=min(v.min(),pm-3*ps);hi=max(v.max(),pm+3*ps);xx=np.linspace(lo,hi,300);ax.plot(xx,norm.pdf(xx,pm,ps),color=ORANGE,label="prior");ax.hist(v,bins=40,density=True,color=BLUE,alpha=.65,label="posterior");ax.axvline(0,color="black",ls="--",lw=.6);ax.set_title(f"{n}  P(>0)={np.mean(v>0):.2f}");ax.set_yticks([])
    axs[0,0].legend(frameon=False);fig.tight_layout();fig.savefig(out/"prior_posterior.png",dpi=200);plt.close(fig)
    # Kappa theta.
    fig,axs=plt.subplots(2,1,figsize=(11,6.7),sharex=True)
    for m,c,ls in [("hsa_static",BLUE,"--"),("hsa_dynamic",ORANGE,"-")]:
        k,t=derived_paths(fits[m],exact.case)
        for ax,v in zip(axs,[k,t]):me,lo,hi=band(v);ax.plot(per,me,color=c,ls=ls,lw=2,label=MODEL_LABELS[m]);ax.fill_between(per,lo,hi,color=c,alpha=.12)
    axs[0].axhline(0,color="black",lw=.5);axs[1].axhline(0,color="black",lw=.5);axs[0].set_ylabel(r"$\kappa_t$");axs[1].set_ylabel(r"$\theta_t$");axs[1].set_xlabel("year");axs[0].legend(frameon=False,ncol=2);fig.tight_layout();fig.savefig(out/"kappa_theta.png",dpi=200);plt.close(fig)


def coeff_rows(M):
    wanted={"ces":["kappa_0"],"slope":["kappa_0","delta"],"direct":["kappa_0","theta_0"],"free_static":["kappa_0","delta","theta_0"],"hsa_static":["kappa_0","theta_0","lambda","delta_derived"],"free_dynamic":["kappa_0","delta_1","delta_2","theta_0","gamma"],"hsa_dynamic":["kappa_0","theta_0","gamma","lambda","delta_1_derived","delta_2_derived"]};rows=[]
    for m,r in M["results"].items():
        rows.append(rf"\multicolumn{{5}}{{l}}{{\emph{{{esc(MODEL_LABELS[m])}}}}}\\")
        for n in wanted[m]:
            z=r["coefficients"][n];rh="--" if z["rhat"] is None else f"{z['rhat']:.3f}";rows.append(f"{esc(n)} & {z['mean']:+.3f} & [{z['q2.5']:+.3f},{z['q97.5']:+.3f}] & {z['p_positive']:.2f} & {rh} " + r"\\")
        rows.append(r"\midrule")
    return "\n".join(rows[:-1])
def comp_rows(M):
    return "\n".join(f"{esc(MODEL_LABELS[m])} & {r['metrics']['waic']:.1f} & {r['metrics']['log_marginal_cut_laplace']:.1f} & {r['metrics']['predictive_rmse']:.3f} & {r['diagnostics']['max_rhat']:.3f} " + r"\\" for m,r in M["results"].items())


def main():
    cfg=load_yaml(BUNDLE/"config.yaml");out=BUNDLE/"results"/"full";M=json.loads((out/"manifest.json").read_text());exact=load_exact_data(cfg);states=load_states(out/"n_states.npz",M["state_diagnostics"]);fits={m:load_fit(out/"draws"/f"{m}.npz",M["results"][m]["diagnostics"]) for m in M["results"]};figdir=out/"figures";figures(exact,states,fits,M,figdir)
    s=M["state_summary"];gate="PASS" if M["gate"]["passed"] else "FAIL";tex=rf"""\documentclass[10.5pt]{{article}}\usepackage[margin=.75in]{{geometry}}\usepackage{{booktabs,graphicx,amsmath,xcolor,microtype,hyperref,newtxtext,newtxmath}}\definecolor{{navy}}{{HTML}}{{17365D}}\setlength{{\parindent}}{{0pt}}\setlength{{\parskip}}{{4pt}}\begin{{document}}
\begin{{center}}{{\color{{navy}}\LARGE\bfseries Exact-$N$ HSA NKPC Decomposition}}\\Gustavo annual constraints and Capital IQ allocation priors\end{{center}}
\section*{{1. Exact quarterly $N$ construction}}
For year $y$, quarterly increments satisfy $\sum_{{q=1}}^4d_{{yq}}=\Delta G_y$ exactly. The allocation prior is centered on the robust Capital IQ profile $\bar w=[{','.join(f'{x:.3f}' for x in M['allocation']['average_weights'])}]$. Observed Capital IQ annual ratios update this prior with coherence weight $c_y=|\sum_q\Delta C_{{yq}}|/\sum_q|\Delta C_{{yq}}|$; missing years retain the prior. Allocation uncertainty is propagated rather than replaced by a fixed interpolated path.
\begin{{center}}\includegraphics[width=.96\linewidth]{{figures/allocation.png}}\end{{center}}
\section*{{2. Exact slow/cycle decomposition}}
\[N_t=\bar N_t+\hat N_t,\quad \bar N_t=\bar N_{{t-1}}+\eta_t^b,\quad \hat N_t=\rho\hat N_{{t-1}}+\eta_t^h,\]
\[\operatorname{{Var}}(\eta_t^b)=\omega\tau^2,\qquad\operatorname{{Var}}(\eta_t^h)=(1-\omega)\tau^2.\]
There is no measurement error in the identity. We use $\omega\sim\mathrm{{Beta}}({cfg['state_priors']['omega_a']:.0f},{cfg['state_priors']['omega_b']:.0f})$ and ${cfg['state_priors']['rho_lower']:.2f}\le\rho\le{cfg['state_priors']['rho_upper']:.2f}$. N is estimated without inflation and its posterior draws are passed to the NKPC. Posterior means: $\omega={s['omega_mean']:.3f}$, $\tau={s['tau_mean']:.3f}$, $\rho={s['rho_mean']:.3f}$. State max $\widehat R={M['state_diagnostics']['max_rhat']:.3f}$; exact-identity error $={M['state_diagnostics']['exact_identity_error']:.1e}$.
\begin{{center}}\includegraphics[width=\linewidth]{{figures/n_decomposition.png}}\end{{center}}\begin{{center}}\includegraphics[width=.95\linewidth]{{figures/state_parameters.png}}\end{{center}}
\section*{{3. Coefficient posteriors}}\begin{{center}}\scriptsize\begin{{tabular}}{{l r c c c}}\toprule Parameter&Mean&95\% interval&P($>$0)&$\widehat R$\\\midrule{coeff_rows(M)}\\\bottomrule\end{{tabular}}\end{{center}}
\section*{{4. Model comparison}}WAIC and the cut-state Laplace approximation integrate over the N posterior but do not allow inflation to feed back into N.\begin{{center}}\small\begin{{tabular}}{{lrrrr}}\toprule Specification&WAIC&log ML&RMSE&max $\widehat R$\\\midrule
{comp_rows(M)}\\\bottomrule\end{{tabular}}\end{{center}}Overall gate: \textbf{{{gate}}}.
\section*{{5. Prior versus posterior}}\begin{{center}}\includegraphics[width=.92\linewidth]{{figures/prior_posterior.png}}\end{{center}}
\section*{{6. Time-varying coefficients}}\begin{{center}}\includegraphics[width=\linewidth]{{figures/kappa_theta.png}}\end{{center}}\end{{document}}"""
    tp=out/"hsa_exact_n_decomposition_report.tex";tp.write_text(tex,encoding="utf-8");r=subprocess.run(["latexmk","-pdf","-interaction=nonstopmode","-halt-on-error",tp.name],cwd=out,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    if r.returncode:raise RuntimeError(r.stdout[-5000:])
    final=_ROOT/"output"/"pdf"/"hsa_exact_n_decomposition_report.pdf";final.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(tp.with_suffix(".pdf"),final);print("wrote",final)
if __name__=="__main__":main()
