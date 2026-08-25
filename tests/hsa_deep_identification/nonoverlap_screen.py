"""Valid non-overlapping YoY diagnostic using Q4 observations only."""
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


def main():
    frame=_analysis_frame(); paths=_competition_paths(); rows=[]
    samples={"discovery":("1974Q4","1999Q4"),"validation":("2000Q1","2013Q4"),"full":("1974Q4","2013Q4")}
    for split,bounds in samples.items():
      for state_name in ("s0_quarterly_local_level_ar2","s1_annual_allocation_proxy"):
       for price in ("ppi","core_cpi"):
        for activity in ("negative_unemployment_gap","inverse_markup"):
         for timing in ("current","lag1"):
          d=_cell_frame(frame,paths[state_name],price,activity,"yoy",timing,bounds)
          d=d[d.index.quarter==4]
          if len(d)<12: continue
          X=np.column_stack((np.ones(len(d)),d.pi_lag,d.epi,d.x,d["bar"]*d.x,-d.hat_use))
          fit,b,V=_fit(d.pi.to_numpy(float),X,"iid"); sd=np.sqrt(np.maximum(np.diag(V),0))
          rows.append({"sample_split":split,"state":state_name,"price":price,"activity":activity,
                       "timing":timing,"n":len(d),"theta_mean":b[5],"theta_q2.5":b[5]-1.96*sd[5],
                       "theta_q97.5":b[5]+1.96*sd[5],"theta_p_positive":float(norm.cdf(b[5]/sd[5])),
                       "delta_mean":b[4],"delta_q2.5":b[4]-1.96*sd[4],"delta_q97.5":b[4]+1.96*sd[4],
                       "delta_p_positive":float(norm.cdf(b[4]/sd[4])),"optimizer":fit.mle_retvals.get("converged")})
    out=pd.DataFrame(rows); folder=Path(__file__).resolve().parent/"results"/"screen"; folder.mkdir(parents=True,exist_ok=True)
    out.to_csv(folder/"nonoverlap_q4.csv",index=False)
    print(out.sort_values("theta_p_positive",ascending=False).head(30).to_string(index=False))


if __name__=="__main__": main()
