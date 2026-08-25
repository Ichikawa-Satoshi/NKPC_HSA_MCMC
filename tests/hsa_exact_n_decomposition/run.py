"""Run exact-N decomposition first, then the modular seven-model NKPC comparison."""
from __future__ import annotations
import argparse,json,time
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
import numpy as np

import sys as _sys,pathlib as _pathlib
_ROOT=next(p for p in _pathlib.Path(__file__).resolve().parents if (p/"pyproject.toml").exists())
_sys.path[:0]=[str(_ROOT),str(_ROOT/"src"),str(_ROOT/"tests")]
from tests import _bootstrap  # noqa:F401,E402
from nkpc_hsa.config import load_yaml  # noqa:E402
from tests.hsa_lambda_dynamic.functions import MODEL_LABELS,comparison_metrics,load_fit,save_fit  # noqa:E402
from tests.hsa_exact_n_decomposition.functions import (  # noqa:E402
    fit_model_cut,fit_states,load_exact_data,load_states,save_states,state_averaged_logml,
)
BUNDLE=Path(__file__).resolve().parent


def summarize(fit):
    flat=fit.draws.reshape(-1,fit.draws.shape[-1]);rows={}
    for i,n in enumerate(fit.names):
        d=flat[:,i];rows[n]={"mean":float(d.mean()),"sd":float(d.std(ddof=1)),"q2.5":float(np.percentile(d,2.5)),"q97.5":float(np.percentile(d,97.5)),"p_positive":float(np.mean(d>0)),"rhat":fit.diagnostics["rhat"][n]}
    if fit.model=="hsa_static": derived={"delta":flat[:,fit.names.index("lambda")]*flat[:,fit.names.index("theta_0")]}
    elif fit.model=="hsa_dynamic":
        la=flat[:,fit.names.index("lambda")];th=flat[:,fit.names.index("theta_0")];ga=flat[:,fit.names.index("gamma")]
        derived={"delta_1":la*th,"delta_2":.5*la*ga}
    else:derived={}
    for n,d in derived.items():rows[n+"_derived"]={"mean":float(d.mean()),"sd":float(d.std(ddof=1)),"q2.5":float(np.percentile(d,2.5)),"q97.5":float(np.percentile(d,97.5)),"p_positive":float(np.mean(d>0)),"rhat":None}
    return rows


def worker(args):
    data,states,model,cfg,sampling,seed=args
    return model,fit_model_cut(data,states,model,cfg,sampling,seed)


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--quick",action="store_true");ap.add_argument("--workers",type=int,default=4);ap.add_argument("--summarize-only",action="store_true");a=ap.parse_args()
    cfg=load_yaml(BUNDLE/"config.yaml");mode="quick" if a.quick else "full";sampling=cfg["sampling"][mode]
    out=BUNDLE/"results"/("smoke" if a.quick else "full");drawdir=out/"draws";drawdir.mkdir(parents=True,exist_ok=True)
    exact=load_exact_data(cfg);started=time.time();state_path=out/"n_states.npz"
    if a.summarize_only:
        old=json.loads((out/"manifest.json").read_text());states=load_states(state_path,old["state_diagnostics"])
        fits={m:load_fit(drawdir/f"{m}.npz",old["results"][m]["diagnostics"]) for m in cfg["models"]}
    else:
        states=fit_states(exact,cfg,sampling,int(cfg["sampling"]["seed"]));save_states(state_path,states)
        print(f"[N states] max Rhat={states.diagnostics['max_rhat']:.3f} identity={states.diagnostics['exact_identity_error']:.1e}",flush=True)
        jobs=[(exact.case,states,m,cfg,sampling,int(cfg["sampling"]["seed"])+1000*i) for i,m in enumerate(cfg["models"])]
        fits={}
        with ProcessPoolExecutor(max_workers=min(a.workers,len(jobs))) as pool:
            futures={pool.submit(worker,j):j[2] for j in jobs}
            for f in as_completed(futures):
                m,fit=f.result();fits[m]=fit;save_fit(drawdir/f"{m}.npz",fit);print(f"[{m}] max Rhat={fit.diagnostics['max_rhat']:.3f}",flush=True)
    results={}
    for m in cfg["models"]:
        fit=fits[m];metrics=comparison_metrics(fit,exact.case);metrics["log_marginal_cut_laplace"]=state_averaged_logml(fit,exact.case)
        results[m]={"label":MODEL_LABELS[m],"coefficients":summarize(fit),"metrics":metrics,"diagnostics":fit.diagnostics}
        print(f"[{m}] WAIC={metrics['waic']:.1f} logML={metrics['log_marginal_cut_laplace']:.1f} RMSE={metrics['predictive_rmse']:.3f}",flush=True)
    apst=exact.allocation
    manifest={"revision":cfg["revision"],"mode":mode,"sample":{"first":str(exact.case.periods[0]),"last":str(exact.case.periods[-1]),"n":exact.case.n_periods},"sampling":dict(sampling),"elapsed_seconds":time.time()-started,
      "allocation":{"average_weights":apst.average_weights.tolist(),"coherence":{str(k):v for k,v in apst.coherence.items()},"raw_weights":{str(k):v.tolist() for k,v in apst.raw_weights.items()},"mean_weights":{str(k):v.tolist() for k,v in apst.mean_weights.items()}},
      "state_diagnostics":states.diagnostics,"state_summary":{"omega_mean":float(states.omega.mean()),"omega_interval":np.percentile(states.omega,[2.5,97.5]).tolist(),"tau_mean":float(states.tau.mean()),"rho_mean":float(states.rho.mean()),"rho_interval":np.percentile(states.rho,[2.5,97.5]).tolist()},"results":results}
    gate=max([states.diagnostics["max_rhat"]]+[v["diagnostics"]["max_rhat"] for v in results.values()]);manifest["gate"]={"required":float(cfg["gates"]["max_rhat"]),"max_rhat":gate,"passed":bool(gate<=float(cfg["gates"]["max_rhat"]))}
    (out/"manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8");print("wrote",out/"manifest.json")

if __name__=="__main__":main()

