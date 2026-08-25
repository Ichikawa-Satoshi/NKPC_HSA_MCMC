"""Simulation recovery for the exact-N MA(3) free-channel sampler."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists())
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT / "tests")]
from tests import _bootstrap  # noqa: F401,E402
from nkpc_hsa.config import load_yaml  # noqa:E402
from tests.hsa_deep_identification.joint_ma3 import fit_joint_ma3  # noqa:E402
from tests.hsa_nested_validation.functions import (  # noqa:E402
    BASE_NAMES, CellData, ModelSpec, _cycle_coefficients, _cycle_unit_cov,
)

BUNDLE = Path(__file__).resolve().parent


class FixedCompetitionExperiment:
    def __init__(self, q): self.q = np.asarray(q, dtype=float)
    def mean_q(self, _cell): return self.q.copy()
    def draw_q(self, _rng, _cell): return self.q.copy()


def simulate(seed=88173, T=157):
    rng=np.random.default_rng(seed); burn=80; n=T+burn
    truth={"intercept":.20,"alpha_b":.42,"alpha_f":.38,"kappa_0":.35,
           "delta_s":.10,"theta":.16,"sigma_pi":.70,"psi_1":.38,"psi_2":.22,"psi_3":.10,
           "omega":.20,"tau":.30,"cycle_damping":.72,"cycle_period":11.0}
    phi1,phi2=_cycle_coefficients(truth["cycle_damping"],truth["cycle_period"])
    vb=truth["omega"]*truth["tau"]**2; vh=(1-truth["omega"])*truth["tau"]**2
    h=np.zeros(n); h[:2]=rng.multivariate_normal(np.zeros(2),vh*_cycle_unit_cov(truth["cycle_damping"],truth["cycle_period"]))
    for t in range(2,n): h[t]=phi1*h[t-1]+phi2*h[t-2]+rng.normal(0,np.sqrt(vh))
    bar=np.cumsum(rng.normal(0,np.sqrt(vb),n)); q=bar+h
    x=np.zeros(n); epi=np.zeros(n)
    for t in range(1,n):
        x[t]=.65*x[t-1]+rng.normal(0,.8); epi[t]=.72*epi[t-1]+rng.normal(0,.35)
    v=rng.normal(0,truth["sigma_pi"],n+3); psi=np.array([1.,truth["psi_1"],truth["psi_2"],truth["psi_3"]])
    xi=np.array([psi@v[t:t+4][::-1] for t in range(n)])
    pi=np.zeros(n)
    for t in range(1,n):
        pi[t]=(truth["intercept"]+truth["alpha_b"]*pi[t-1]+truth["alpha_f"]*epi[t]
               +(truth["kappa_0"]+truth["delta_s"]*bar[t])*x[t]-truth["theta"]*h[t]+xi[t])
    sl=slice(burn,n); pi=pi[sl]; x=x[sl]; epi=epi[sl]; q=q[sl]
    periods=pd.period_range("1974Q4",periods=T,freq="Q")
    cell=CellData("simulation","simulation","ppi","sim",periods,pi,np.r_[pi[0],pi[:-1]],epi,x,
                  np.arange(T),float(np.std(pi)),float(np.std(x)),float(np.std(q)))
    return FixedCompetitionExperiment(q),cell,truth


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--profile",choices=("mock","quick","full"),default="mock")
    args=parser.parse_args(); cfg=load_yaml(BUNDLE/"config.yaml")
    experiment,cell,truth=simulate(); spec=ModelSpec("free_static_combined","recovery",BASE_NAMES+("delta_s","theta"))
    result=fit_joint_ma3(experiment,cell,spec,cfg,cfg["sampling"][args.profile],"quarterly_local_level_ar2",20260825)
    fit=result.model_fit; flat=fit.draws.reshape(-1,fit.draws.shape[-1]); rows={}
    for j,name in enumerate(fit.names):
        lo,mean,hi=np.percentile(flat[:,j],[2.5,50,97.5]); rows[name]={"truth":truth[name],"median":float(mean),
            "q2.5":float(lo),"q97.5":float(hi),"covered":bool(lo<=truth[name]<=hi),"rhat":fit.diagnostics["rhat"][name]}
    state_truth={k:truth[k] for k in ("omega","tau","cycle_damping","cycle_period")}
    arrays={"omega":fit.omega,"tau":fit.tau,"cycle_damping":fit.cycle_damping,"cycle_period":fit.cycle_period}
    for name,values in arrays.items():
        lo,mean,hi=np.percentile(values,[2.5,50,97.5]); rows[name]={"truth":state_truth[name],"median":float(mean),
            "q2.5":float(lo),"q97.5":float(hi),"covered":bool(lo<=state_truth[name]<=hi),"rhat":fit.diagnostics["rhat"][name]}
    for j in range(3):
        name=f"psi_{j+1}"; values=result.psi[:,:,j]; lo,mean,hi=np.percentile(values,[2.5,50,97.5]); rows[name]={
            "truth":truth[name],"median":float(mean),"q2.5":float(lo),"q97.5":float(hi),
            "covered":bool(lo<=truth[name]<=hi),"rhat":fit.diagnostics["rhat"][name]}
    out=BUNDLE/"results"/args.profile/"simulation_recovery.json"; out.parent.mkdir(parents=True,exist_ok=True)
    payload={"profile":args.profile,"all_covered":all(v["covered"] for v in rows.values()),
             "max_rhat":fit.diagnostics["max_rhat"],"parameters":rows}
    out.write_text(json.dumps(payload,indent=2),encoding="utf-8"); print(json.dumps(payload,indent=2))


if __name__=="__main__": main()
