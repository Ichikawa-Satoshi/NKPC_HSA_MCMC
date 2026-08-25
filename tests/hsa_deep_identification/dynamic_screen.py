"""Nested free and HSA-restricted dynamic MA(3) identification screen."""
from __future__ import annotations
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from scipy.stats import norm

ROOT=next(p for p in Path(__file__).resolve().parents if (p/"pyproject.toml").exists())
sys.path[:0]=[str(ROOT),str(ROOT/"src"),str(ROOT/"tests")]
from tests import _bootstrap  # noqa:F401,E402
from tests.hsa_deep_identification.screen import _analysis_frame,_competition_paths,_cell_frame,_fit  # noqa:E402


def path_gates(beta,cov,bar,lam,rng,restricted):
    draws=rng.multivariate_normal(beta,cov,size=20000)
    if restricted:
        k0=draws[:,3,None]; theta0=draws[:,4,None]; gamma=draws[:,5,None]
        theta=theta0+gamma*bar; kappa=k0+lam*theta0*bar+lam*gamma*bar**2/2
    else:
        k0=draws[:,3,None]; d1=draws[:,4,None]; d2=draws[:,5,None]
        theta0=draws[:,6,None]; gamma=draws[:,7,None]
        theta=theta0+gamma*bar; kappa=k0+d1*bar+d2*bar**2
    return {"theta_positive_95pct_dates":float(np.mean(np.mean(theta>0,axis=1)>=.95)),
            "kappa_positive_95pct_dates":float(np.mean(np.mean(kappa>0,axis=1)>=.95)),
            "min_theta_pointwise_p":float(np.min(np.mean(theta>0,axis=0))),
            "min_kappa_pointwise_p":float(np.min(np.mean(kappa>0,axis=0)))}


def main():
    frame=_analysis_frame(); paths=_competition_paths(); rng=np.random.default_rng(20260825); rows=[]
    samples={"discovery":("1974Q4","1999Q4"),"validation":("2000Q1","2013Q4"),"full":("1974Q4","2013Q4")}
    for split,bounds in samples.items():
      for state_name in ("s0_quarterly_local_level_ar2","s1_annual_allocation_proxy"):
       for price in ("ppi","core_cpi"):
        for activity in ("negative_unemployment_gap","inverse_markup"):
         for timing in ("current","lag1"):
          d=_cell_frame(frame,paths[state_name],price,activity,"yoy",timing,bounds)
          y=d.pi.to_numpy(float); base=np.column_stack((np.ones(len(d)),d.pi_lag,d.epi,d.x))
          ces,_,_=_fit(y,base,"ma3"); bar=d["bar"].to_numpy(float); h=d.hat_use.to_numpy(float); x=d.x.to_numpy(float)
          X=np.column_stack((base,bar*x,bar**2*x,-h,-bar*h))
          fit,b,V=_fit(y,X,"ma3"); sd=np.sqrt(np.maximum(np.diag(V),0)); gates=path_gates(b,V,bar[None,:],np.nan,rng,False)
          rows.append({"sample_split":split,"state":state_name,"price":price,"activity":activity,"timing":timing,
                       "model":"free_dynamic","lambda":np.nan,"n":len(d),"bic_minus_ces":fit.bic-ces.bic,
                       "theta0_mean":b[6],"theta0_q2.5":b[6]-1.96*sd[6],"theta0_q97.5":b[6]+1.96*sd[6],
                       "theta0_p_positive":float(norm.cdf(b[6]/sd[6])),"gamma_mean":b[7],"optimizer":fit.mle_retvals.get("converged"),**gates})
          for lam in (3.,6.,9.):
            theta_col=lam*bar*x-h; gamma_col=lam*bar**2*x/2-bar*h
            X=np.column_stack((base,theta_col,gamma_col)); fit,b,V=_fit(y,X,"ma3"); sd=np.sqrt(np.maximum(np.diag(V),0)); gates=path_gates(b,V,bar[None,:],lam,rng,True)
            rows.append({"sample_split":split,"state":state_name,"price":price,"activity":activity,"timing":timing,
                         "model":"hsa_dynamic","lambda":lam,"n":len(d),"bic_minus_ces":fit.bic-ces.bic,
                         "theta0_mean":b[4],"theta0_q2.5":b[4]-1.96*sd[4],"theta0_q97.5":b[4]+1.96*sd[4],
                         "theta0_p_positive":float(norm.cdf(b[4]/sd[4])),"gamma_mean":b[5],"optimizer":fit.mle_retvals.get("converged"),**gates})
    out=pd.DataFrame(rows); folder=Path(__file__).resolve().parent/"results"/"screen"; folder.mkdir(parents=True,exist_ok=True)
    out.to_csv(folder/"dynamic_screen.csv",index=False)
    passed=out[(out.theta_positive_95pct_dates>=.95)&(out.kappa_positive_95pct_dates>=.95)&out.optimizer]
    print("passes",len(passed)); print(passed.to_string(index=False))
    print("\nTop by minimum path probability")
    print(out.sort_values(["min_theta_pointwise_p","min_kappa_pointwise_p"],ascending=False).head(30).to_string(index=False))


if __name__=="__main__": main()
