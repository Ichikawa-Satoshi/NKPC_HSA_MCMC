"""Run varying-theta, free-dynamic, and HSA-restricted dynamic diagnostics."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
from datetime import datetime,timezone
import hashlib,json,subprocess,sys,time
from pathlib import Path
import arviz as az
import numpy as np
import pandas as pd
from scipy.special import logsumexp

ROOT=next(p for p in Path(__file__).resolve().parents if (p/"pyproject.toml").exists());sys.path[:0]=[str(ROOT),str(ROOT/"src"),str(ROOT/"tests")]
from tests import _bootstrap  # noqa:F401,E402
from nkpc_hsa.config import load_yaml  # noqa:E402
from nkpc_hsa.paths import data_root  # noqa:E402
from tests.active_firm_stock_bds_bed.functions import ThetaCell,robust_scale  # noqa:E402
from tests.gustavo_state_capitaliq_cycle.functions import CycleFit,build_qoq_design,load_cycle,load_nkpc_cells,load_qoq,qoq_pointwise_loglik,save_qoq  # noqa:E402
from tests.gustavo_state_capitaliq_cycle.dynamic_functions import dynamic_loglik,dynamic_mu,dynamic_summary,fit_dynamic,simulate_varying_theta  # noqa:E402

BUNDLE=Path(__file__).resolve().parent;BASE=BUNDLE/"results"/"mock_qoq";STAGED=BUNDLE/"results"/"staged_validation";OUT=BUNDLE/"results"/"dynamic_validation"


def _json(path,payload):path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(payload,indent=2,ensure_ascii=False)+"\n")
def _sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def _git(args):return subprocess.run(["git",*args],cwd=ROOT,text=True,stdout=subprocess.PIPE,stderr=subprocess.DEVNULL).stdout.strip()
def _state(label):
    p=BASE/"draws"/"cycle"/f"{label}.npz";return load_cycle(p,json.loads(p.with_suffix(".json").read_text()))
def _subset(cell,mask):return ThetaCell(cell.name,cell.periods[mask],cell.pi[mask],cell.pi_lag[mask],cell.epi[mask],cell.x[mask],robust_scale(cell.pi[mask]),robust_scale(cell.x[mask]))


def _fit_job(args):
    label,error_model,cell_name,model,sample,seed,sampling,holdout,reuse=args;path=OUT/"draws"/sample/model/label/error_model/f"{cell_name}.npz"
    if reuse and path.exists() and path.with_suffix(".json").exists():return json.loads(path.with_suffix(".json").read_text())
    base=load_yaml(BUNDLE/"config.yaml");dcfg=load_yaml(BUNDLE/"dynamic_config.yaml");cell=load_nkpc_cells(base)[cell_name]
    if sample=="train":cell=_subset(cell,np.asarray(cell.periods<pd.Period(holdout,freq="Q")))
    fit=fit_dynamic(cell,_state(label),base,dcfg,seed,model=model,error_model=error_model,sampling_override=sampling);summary=dynamic_summary(fit,model);summary["retried"]=False
    gates=dcfg["gates"]
    if summary["diagnostics"]["max_rhat"]>float(gates["max_rhat"]) or summary["diagnostics"]["min_bulk_ess"]<float(gates["min_bulk_ess"]):
        long={"iterations":12000,"warmup":3500,"thin":3,"chains":4};fit=fit_dynamic(cell,_state(label),base,dcfg,seed+33000001,model=model,error_model=error_model,sampling_override=long);summary=dynamic_summary(fit,model);summary["retried"]=True
    save_qoq(path,fit);_json(path.with_suffix(".json"),summary);return summary


def _metrics(ll):
    idata=az.from_dict({"posterior":{"dummy":np.zeros((*ll.shape[:2],1))},"log_likelihood":{"inflation":ll}});loo=az.loo(idata,pointwise=True);pareto=np.asarray(loo.pareto_k);flat=ll.reshape(-1,ll.shape[-1]);lppd=logsumexp(flat,axis=0)-np.log(len(flat));p_i=np.var(flat,axis=0,ddof=1);w_i=lppd-p_i
    return {"elpd_loo":float(loo.elpd),"se_loo":float(loo.se),"p_loo":float(loo.p),"elpd_waic":float(w_i.sum()),"se_waic":float(np.sqrt(len(w_i)*np.var(w_i,ddof=1))),"p_waic":float(p_i.sum()),"max_pareto_k":float(pareto.max()),"pareto_k_over_0.7":int(np.sum(pareto>.7))}


def _holdout(cell,train,test,fit,state,model):
    sp=pd.PeriodIndex(state.periods,freq="Q");tr=sp.get_indexer(train.periods);te=sp.get_indexer(test.periods);mus=[];ll=[]
    for c in range(fit.draws.shape[0]):
        for d in range(fit.draws.shape[1]):
            cs=int(fit.state_chain[c,d]);ds=int(fit.state_draw[c,d]);hat=state.nhat[cs,ds,te];raw=state.nbar_used[cs,ds];center=(float(np.mean(raw[tr])),float(np.mean((raw[tr]-np.mean(raw[tr]))**2)))
            if model=="constant_theta":X,_=build_qoq_design(test,hat,None);mu=X@fit.draws[c,d]
            else:mu=dynamic_mu(test,fit,c,d,bar=raw[te],hat=hat,center=center)
            mus.append(mu);sig=float(fit.sigma_u[c,d]);ll.append(-.5*np.log(2*np.pi*sig**2)-.5*(test.pi-mu)**2/sig**2)
    mus=np.asarray(mus);ll=np.asarray(ll);return {"holdout_elpd":float(np.sum(logsumexp(ll,axis=0)-np.log(len(ll)))),"holdout_rmse":float(np.sqrt(np.mean((test.pi-mus.mean(0))**2))),"holdout_n":len(test.pi)}


def _oracle(periods,hat,bar):return CycleFit("oracle",tuple(map(str,periods)),tuple(),np.zeros((1,1,0)),bar[None,None,:],hat[None,None,:],{},hat)


def _recovery_group(args):
    mode,reps,scenarios,seed,sampling,generator_path=args;base=load_yaml(BUNDLE/"config.yaml");dcfg=load_yaml(BUNDLE/"dynamic_config.yaml");state=_state("firm_weighted");cell=load_nkpc_cells(base)[dcfg["varying_theta_recovery"]["activity"]];meta=json.loads(Path(generator_path).with_suffix(".json").read_text());observed=load_qoq(Path(generator_path),meta["diagnostics"]);rows=[];gates=dcfg["gates"]
    for j,(scenario,(stheta,sgamma)) in enumerate(scenarios.items()):
        for rep in range(reps):
            rseed=seed+1000003*j+1009*rep;rng=np.random.default_rng(rseed);synthetic,hat,bar,true=simulate_varying_theta(rng,cell,observed,state,float(stheta),float(sgamma));use=state if mode=="propagated_state" else _oracle(synthetic.periods,hat,bar);fit=fit_dynamic(synthetic,use,base,dcfg,rseed+7000001,model="varying_theta",error_model="iid",sampling_override=sampling);summary=dynamic_summary(fit,"varying_theta");retried=False
            if summary["diagnostics"]["max_rhat"]>float(gates["recovery_max_rhat"]) or summary["diagnostics"]["min_bulk_ess"]<float(gates["recovery_min_bulk_ess"]):
                fit=fit_dynamic(synthetic,use,base,dcfg,rseed+17000001,model="varying_theta",error_model="iid",sampling_override={"iterations":2200,"warmup":700,"thin":2,"chains":4});summary=dynamic_summary(fit,"varying_theta");retried=True
            for parameter,stdtrue in (("theta_0",float(stheta)),("gamma",float(sgamma))):
                z=summary["coefficients"][parameter];raw=true[parameter];positive=raw>0;rows.append({"mode":mode,"scenario":scenario,"replicate":rep,"parameter":parameter,"standardized_true":stdtrue,"raw_true":raw,"mean":z["mean"],"q2.5":z["q2.5"],"q97.5":z["q97.5"],"p_positive":z["p_positive"],"sd_ratio":z["posterior_prior_sd_ratio"],"coverage":bool(z["q2.5"]<=raw<=z["q97.5"]),"suggestive_detected":bool(positive and z["p_positive"]>=.80 and z["posterior_prior_sd_ratio"]<=.75),"strong_detected":bool(positive and z["p_positive"]>=.975 and z["q2.5"]>0 and z["posterior_prior_sd_ratio"]<=.75),"false_positive":bool(not positive and ((z["p_positive"]>=.975 and z["q2.5"]>0) or (z["p_positive"]<=.025 and z["q97.5"]<0))),"max_rhat":summary["diagnostics"]["max_rhat"],"min_bulk_ess":summary["diagnostics"]["min_bulk_ess"],"retried":retried})
    return rows


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument("--workers",type=int,default=4);ap.add_argument("--refit",action="store_true");ap.add_argument("--no-report",action="store_true");args=ap.parse_args();started=time.time();base=load_yaml(BUNDLE/"config.yaml");dcfg=load_yaml(BUNDLE/"dynamic_config.yaml");OUT.mkdir(parents=True,exist_ok=True);(OUT/"tables").mkdir(exist_ok=True);hashes={"base":_sha(BUNDLE/"config.yaml"),"dynamic":_sha(BUNDLE/"dynamic_config.yaml")};old=json.loads((OUT/"manifest.json").read_text()) if (OUT/"manifest.json").exists() else {};reuse=bool(not args.refit and old.get("config_hashes")==hashes);sampling={k:v for k,v in dcfg["sampling"].items() if k!="seed_offset"};seed=int(base["sampling"]["seed"])+int(dcfg["sampling"]["seed_offset"]);holdout=dcfg["comparison"]["holdout_start"];labels=list(base["data"]["capital_iq"]);cells=load_nkpc_cells(base);jobs=[];k=0
    for label in labels:
        for cell_name in cells:
            for model in dcfg["models"]:
                for sample in ("full","train"):
                    k+=1;jobs.append((label,"iid",cell_name,model,sample,seed+1000003*k,sampling,holdout,reuse))
    for model in dcfg["models"]:
        k+=1;jobs.append(("firm_weighted","persistent_ar1","ppi_negative_unemployment_gap",model,"full",seed+1000003*k,sampling,holdout,reuse))
    summaries=[]
    with ProcessPoolExecutor(max_workers=min(args.workers,len(jobs))) as pool:
        for future in as_completed([pool.submit(_fit_job,j) for j in jobs]):
            z=future.result();summaries.append(z);print(f"DYNAMIC {z['sample']} {z['model']} {z['cycle']} {z['error_model']} {z['cell']} Rhat={z['diagnostics']['max_rhat']:.4f}",flush=True)
    coeff=[]
    for z in summaries:
        if z["sample"][2] not in {67,83}:continue
        for name,v in z["coefficients"].items():coeff.append({"cycle":z["cycle"],"error_model":z["error_model"],"cell":z["cell"],"model":z["model"],"sample_start":z["sample"][0],"sample_end":z["sample"][1],"parameter":name,**v,"fit_max_rhat":z["diagnostics"]["max_rhat"],"fit_min_bulk_ess":z["diagnostics"]["min_bulk_ess"],"retried":z.get("retried",False)})
        for name,v in z.get("derived",{}).items():coeff.append({"cycle":z["cycle"],"error_model":z["error_model"],"cell":z["cell"],"model":z["model"],"sample_start":z["sample"][0],"sample_end":z["sample"][1],"parameter":name+"_derived",**v,"fit_max_rhat":z["diagnostics"]["max_rhat"],"fit_min_bulk_ess":z["diagnostics"]["min_bulk_ess"],"retried":z.get("retried",False)})
    pd.DataFrame(coeff).to_csv(OUT/"tables"/"coefficients.csv",index=False)
    comparisons=[]
    for label in labels:
        state=_state(label)
        for cell_name,cell in cells.items():
            mask=np.asarray(cell.periods<pd.Period(holdout,freq="Q"));train=_subset(cell,mask);test=_subset(cell,~mask)
            for model in ("constant_theta",*dcfg["models"]):
                if model=="constant_theta":fp=STAGED/"draws"/"full"/"direct_only"/label/"iid"/f"{cell_name}.npz";tp=STAGED/"draws"/"train"/"direct_only"/label/"iid"/f"{cell_name}.npz"
                else:fp=OUT/"draws"/"full"/model/label/"iid"/f"{cell_name}.npz";tp=OUT/"draws"/"train"/model/label/"iid"/f"{cell_name}.npz"
                fm=json.loads(fp.with_suffix(".json").read_text());tm=json.loads(tp.with_suffix(".json").read_text());full=load_qoq(fp,fm["diagnostics"]);trained=load_qoq(tp,tm["diagnostics"]);ll=qoq_pointwise_loglik(cell,full) if model=="constant_theta" else dynamic_loglik(cell,full);comparisons.append({"cycle":label,"cell":cell_name,"model":model,**_metrics(ll),**_holdout(cell,train,test,trained,state,model),"max_rhat":fm["diagnostics"]["max_rhat"],"min_bulk_ess":fm["diagnostics"]["min_bulk_ess"]})
    comp=pd.DataFrame(comparisons);comp.to_csv(OUT/"tables"/"model_comparison.csv",index=False)
    scenarios=dict(dcfg["varying_theta_recovery"]["scenarios"]);reps=int(dcfg["varying_theta_recovery"]["replicates"]);rsamp=dict(dcfg["varying_theta_recovery"]["sampling"]);gen=OUT/"draws"/"full"/"varying_theta"/"firm_weighted"/"iid"/"ppi_negative_unemployment_gap.npz";groups=[(mode,reps,scenarios,seed+51000001+10000019*j,rsamp,str(gen)) for j,mode in enumerate(dcfg["varying_theta_recovery"]["modes"])];rpath=OUT/"tables"/"recovery_replications.csv";expected=2*len(groups)*reps*len(scenarios)
    if reuse and rpath.exists() and len(pd.read_csv(rpath))==expected:rec=pd.read_csv(rpath);print(f"RECOVERY reused {len(rec)} rows",flush=True)
    else:
        rows=[]
        with ProcessPoolExecutor(max_workers=min(args.workers,len(groups))) as pool:
            for future in as_completed([pool.submit(_recovery_group,g) for g in groups]):rows.extend(future.result());print(f"RECOVERY {len(rows)}/{expected} coefficient rows",flush=True)
        rec=pd.DataFrame(rows);rec.to_csv(rpath,index=False)
    power=rec.groupby(["mode","scenario","parameter"]).agg(replicates=("replicate","size"),standardized_true=("standardized_true","first"),suggestive_rate=("suggestive_detected","mean"),strong_rate=("strong_detected","mean"),false_positive_rate=("false_positive","mean"),coverage=("coverage","mean"),mean_estimate=("mean","mean"),mean_p_positive=("p_positive","mean"),mean_sd_ratio=("sd_ratio","mean"),max_rhat=("max_rhat","max"),min_bulk_ess=("min_bulk_ess","min"),retry_rate=("retried","mean")).reset_index();power.to_csv(OUT/"tables"/"recovery_power.csv",index=False)
    observed=[z for z in summaries if z["sample"][2]==83];gate={"observed_max_rhat":max(z["diagnostics"]["max_rhat"] for z in observed),"observed_min_bulk_ess":min(z["diagnostics"]["min_bulk_ess"] for z in observed),"max_rhat_required":float(dcfg["gates"]["max_rhat"]),"min_bulk_ess_required":float(dcfg["gates"]["min_bulk_ess"]),"recovery_max_rhat":float(rec.max_rhat.max()),"recovery_min_bulk_ess":float(rec.min_bulk_ess.min())};gate["observed_computational_pass"]=bool(gate["observed_max_rhat"]<=gate["max_rhat_required"] and gate["observed_min_bulk_ess"]>=gate["min_bulk_ess_required"])
    data_path=data_root()/"processed"/"model_ready.csv";manifest={"revision":dcfg["revision"],"profile":"dynamic_validation","not_for_inference":True,"created_utc":datetime.now(timezone.utc).isoformat(timespec="seconds"),"elapsed_seconds":time.time()-started,"seed":seed,"config_hashes":hashes,"data_sha256":_sha(data_path),"measurement_hashes":{l:_sha(BASE/"draws"/"cycle"/f"{l}.npz") for l in labels},"observed_fit_count":len(jobs),"recovery_fit_count":int(len(rec)/2),"git_commit":_git(["rev-parse","HEAD"]),"git_dirty":bool(_git(["status","--porcelain"])),"gate":gate,"interpretation":"HSA dynamic is estimated at user request as a weak-identification diagnostic after the static recovery gate failed; restrictions must not be treated as independent identification."};_json(OUT/"manifest.json",manifest);print(json.dumps(gate,indent=2),flush=True)
    if not args.no_report:
        from tests.gustavo_state_capitaliq_cycle.build_dynamic_report import build
        build()


if __name__=="__main__":main()
