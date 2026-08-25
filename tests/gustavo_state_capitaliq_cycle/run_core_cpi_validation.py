"""Estimate the full QoQ Core-CPI M0--M4 ladder on the frozen competition states."""
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

ROOT=next(p for p in Path(__file__).resolve().parents if (p/"pyproject.toml").exists())
sys.path[:0]=[str(ROOT),str(ROOT/"src"),str(ROOT/"tests")]
from tests import _bootstrap  # noqa:F401,E402
from nkpc_hsa.config import load_yaml  # noqa:E402
from nkpc_hsa.paths import data_root  # noqa:E402
from tests.active_firm_stock_bds_bed.functions import ThetaCell,robust_scale  # noqa:E402
from tests.gustavo_state_capitaliq_cycle.dynamic_functions import dynamic_loglik,dynamic_mu,dynamic_summary,fit_dynamic,simulate_varying_theta  # noqa:E402
from tests.gustavo_state_capitaliq_cycle.functions import CycleFit,build_qoq_design,fit_qoq_theta,load_cycle,load_nkpc_cells,load_qoq,qoq_pointwise_loglik,save_qoq,simulate_qoq_combined,summarize_qoq  # noqa:E402

BUNDLE=Path(__file__).resolve().parent
BASE=BUNDLE/"results"/"mock_qoq"
MODELS=("direct_only","free_combined","varying_theta","free_dynamic","hsa_restricted_dynamic")


def _json(path,payload):path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(payload,indent=2,ensure_ascii=False)+"\n")
def _sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def _git(args):return subprocess.run(["git",*args],cwd=ROOT,text=True,stdout=subprocess.PIPE,stderr=subprocess.DEVNULL).stdout.strip()


def _configs():
    base=load_yaml(BUNDLE/"config.yaml");core=load_yaml(BUNDLE/"core_cpi_config.yaml");price=core["price"]
    base["data"]["prices"]={price["name"]:{k:price[k] for k in ("inflation","inflation_lag","expectation")}}
    dynamic=load_yaml(BUNDLE/"dynamic_config.yaml")
    return base,core,dynamic


def _state(label):
    path=BASE/"draws"/"cycle"/f"{label}.npz"
    return load_cycle(path,json.loads(path.with_suffix(".json").read_text()))


def _subset(cell,mask):
    return ThetaCell(cell.name,cell.periods[mask],cell.pi[mask],cell.pi_lag[mask],cell.epi[mask],cell.x[mask],robust_scale(cell.pi[mask]),robust_scale(cell.x[mask]))


def _sampling(core,model,mode):
    if mode=="full":return dict(core["sampling"]["static" if model in {"direct_only","free_combined"} else "dynamic"])
    return {"iterations":500 if model in {"direct_only","free_combined"} else 700,"warmup":150 if model in {"direct_only","free_combined"} else 200,"thin":2,"chains":2}


def _fit_job(args):
    out_s,label,error_model,cell_name,model,sample,seed,sampling,holdout,reuse,mode=args
    out=Path(out_s);path=out/"draws"/sample/model/label/error_model/f"{cell_name}.npz"
    if reuse and path.exists() and path.with_suffix(".json").exists():return json.loads(path.with_suffix(".json").read_text())
    base,core,dynamic=_configs();cell=load_nkpc_cells(base)[cell_name]
    if sample=="train":cell=_subset(cell,np.asarray(cell.periods<pd.Period(holdout,freq="Q")))
    if model in {"direct_only","free_combined"}:
        fit=fit_qoq_theta(cell,_state(label),base,seed,error_model=error_model,include_delta=model=="free_combined",sampling_override=sampling);summary=summarize_qoq(fit)
    else:
        fit=fit_dynamic(cell,_state(label),base,dynamic,seed,model=model,error_model=error_model,sampling_override=sampling);summary=dynamic_summary(fit,model)
    summary["retried"]=False
    gates=core["gates"]
    if mode=="full" and (summary["diagnostics"]["max_rhat"]>float(gates["observed_max_rhat"]) or summary["diagnostics"]["min_bulk_ess"]<float(gates["observed_min_bulk_ess"])):
        long={"iterations":7000 if model in {"direct_only","free_combined"} else 12000,"warmup":2100 if model in {"direct_only","free_combined"} else 3500,"thin":3,"chains":4}
        if model in {"direct_only","free_combined"}:fit=fit_qoq_theta(cell,_state(label),base,seed+33000001,error_model=error_model,include_delta=model=="free_combined",sampling_override=long);summary=summarize_qoq(fit)
        else:fit=fit_dynamic(cell,_state(label),base,dynamic,seed+33000001,model=model,error_model=error_model,sampling_override=long);summary=dynamic_summary(fit,model)
        summary["retried"]=True
    save_qoq(path,fit);_json(path.with_suffix(".json"),summary);return summary


def _metrics(ll):
    idata=az.from_dict({"posterior":{"dummy":np.zeros((*ll.shape[:2],1))},"log_likelihood":{"inflation":ll}});loo=az.loo(idata,pointwise=True);pareto=np.asarray(loo.pareto_k);flat=ll.reshape(-1,ll.shape[-1]);lppd=logsumexp(flat,axis=0)-np.log(len(flat));p_i=np.var(flat,axis=0,ddof=1);w_i=lppd-p_i
    return {"elpd_loo":float(loo.elpd),"se_loo":float(loo.se),"p_loo":float(loo.p),"elpd_waic":float(w_i.sum()),"se_waic":float(np.sqrt(len(w_i)*np.var(w_i,ddof=1))),"p_waic":float(p_i.sum()),"max_pareto_k":float(pareto.max()),"pareto_k_over_0.7":int(np.sum(pareto>.7))}


def _holdout(train,test,fit,state,model):
    sp=pd.PeriodIndex(state.periods,freq="Q");tr=sp.get_indexer(train.periods);te=sp.get_indexer(test.periods);mus=[];ll=[]
    for c in range(fit.draws.shape[0]):
        for d in range(fit.draws.shape[1]):
            cs=int(fit.state_chain[c,d]);ds=int(fit.state_draw[c,d]);hat=state.nhat[cs,ds,te];raw=state.nbar_used[cs,ds]
            if model=="direct_only":X,_=build_qoq_design(test,hat,None);mu=X@fit.draws[c,d]
            elif model=="free_combined":
                bar=raw[te]-float(np.mean(raw[tr]));X=np.column_stack([np.ones(len(test.periods)),test.pi_lag,test.epi,test.x,bar*test.x,-hat]);mu=X@fit.draws[c,d]
            else:
                center=(float(np.mean(raw[tr])),float(np.mean((raw[tr]-np.mean(raw[tr]))**2)));mu=dynamic_mu(test,fit,c,d,bar=raw[te],hat=hat,center=center)
            mus.append(mu);sig=float(fit.sigma_u[c,d]);ll.append(-.5*np.log(2*np.pi*sig**2)-.5*(test.pi-mu)**2/sig**2)
    mus=np.asarray(mus);ll=np.asarray(ll)
    return {"holdout_elpd":float(np.sum(logsumexp(ll,axis=0)-np.log(len(ll)))),"holdout_rmse":float(np.sqrt(np.mean((test.pi-mus.mean(0))**2))),"holdout_n":len(test.pi)}


def _oracle(periods,hat,bar):return CycleFit("oracle",tuple(map(str,periods)),tuple(),np.zeros((1,1,0)),bar[None,None,:],hat[None,None,:],{},hat)


def _recovery_group(args):
    kind,mode,reps,scenarios,seed,sampling,generator_path=args;base,core,dynamic=_configs();state=_state("firm_weighted");cell=load_nkpc_cells(base)["core_cpi_negative_unemployment_gap"];meta=json.loads(Path(generator_path).with_suffix(".json").read_text());observed=load_qoq(Path(generator_path),meta["diagnostics"]);rows=[];g=core["gates"]
    for j,(scenario,truths) in enumerate(scenarios.items()):
        for rep in range(reps):
            rseed=seed+1000003*j+1009*rep;rng=np.random.default_rng(rseed)
            if kind=="static":
                synthetic,hat,bar,true=simulate_qoq_combined(rng,cell,observed,state,float(truths[0]),float(truths[1]));use=state if mode=="propagated_state" else _oracle(synthetic.periods,hat,bar);fit=fit_qoq_theta(synthetic,use,base,rseed+7000001,error_model="iid",include_delta=True,recovery=True,sampling_override=sampling);summary=summarize_qoq(fit);pairs=(("delta",float(truths[0])),("theta_CIQ",float(truths[1])))
            else:
                synthetic,hat,bar,true=simulate_varying_theta(rng,cell,observed,state,float(truths[0]),float(truths[1]));use=state if mode=="propagated_state" else _oracle(synthetic.periods,hat,bar);fit=fit_dynamic(synthetic,use,base,dynamic,rseed+7000001,model="varying_theta",error_model="iid",sampling_override=sampling);summary=dynamic_summary(fit,"varying_theta");pairs=(("theta_0",float(truths[0])),("gamma",float(truths[1])))
            retried=False
            if summary["diagnostics"]["max_rhat"]>float(g["recovery_max_rhat"]) or summary["diagnostics"]["min_bulk_ess"]<float(g["recovery_min_bulk_ess"]):
                longer={"iterations":2200,"warmup":700,"thin":2,"chains":4}
                if kind=="static":fit=fit_qoq_theta(synthetic,use,base,rseed+17000001,error_model="iid",include_delta=True,recovery=True,sampling_override=longer);summary=summarize_qoq(fit)
                else:fit=fit_dynamic(synthetic,use,base,dynamic,rseed+17000001,model="varying_theta",error_model="iid",sampling_override=longer);summary=dynamic_summary(fit,"varying_theta")
                retried=True
            for parameter,stdtrue in pairs:
                z=summary["coefficients"][parameter];raw=true[parameter];positive=raw>0;learn=z["posterior_prior_sd_ratio"]<=float(g["posterior_prior_sd_ratio"])
                rows.append({"kind":kind,"mode":mode,"scenario":scenario,"replicate":rep,"parameter":parameter,"standardized_true":stdtrue,"raw_true":raw,"mean":z["mean"],"q2.5":z["q2.5"],"q97.5":z["q97.5"],"p_positive":z["p_positive"],"sd_ratio":z["posterior_prior_sd_ratio"],"coverage":bool(z["q2.5"]<=raw<=z["q97.5"]),"suggestive_detected":bool(positive and z["p_positive"]>=float(g["suggestive_sign_probability"]) and learn),"strong_detected":bool(positive and z["p_positive"]>=float(g["strong_sign_probability"]) and z["q2.5"]>0 and learn),"false_positive":bool(not positive and ((z["p_positive"]>=float(g["strong_sign_probability"]) and z["q2.5"]>0) or (z["p_positive"]<=1-float(g["strong_sign_probability"]) and z["q97.5"]<0))),"max_rhat":summary["diagnostics"]["max_rhat"],"min_bulk_ess":summary["diagnostics"]["min_bulk_ess"],"retried":retried})
    return rows


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument("--mode",choices=("smoke","full"),default="full");ap.add_argument("--workers",type=int,default=4);ap.add_argument("--refit",action="store_true");ap.add_argument("--skip-recovery",action="store_true");args=ap.parse_args();started=time.time();base,core,dynamic=_configs();out=BUNDLE/"results"/f"core_cpi_{args.mode}";out.mkdir(parents=True,exist_ok=True);(out/"tables").mkdir(exist_ok=True)
    hashes={"base":_sha(BUNDLE/"config.yaml"),"core":_sha(BUNDLE/"core_cpi_config.yaml"),"dynamic":_sha(BUNDLE/"dynamic_config.yaml")};old=json.loads((out/"manifest.json").read_text()) if (out/"manifest.json").exists() else {};reuse=bool(not args.refit and old.get("config_hashes")==hashes);seed=int(base["sampling"]["seed"])+int(core["sampling"]["seed_offset"]);holdout=core["comparison"]["holdout_start"];labels=list(base["data"]["capital_iq"]);cells=load_nkpc_cells(base);jobs=[];k=0
    for label in labels:
        for cell_name in cells:
            for model in MODELS:
                for sample in ("full","train"):
                    k+=1;jobs.append((str(out),label,"iid",cell_name,model,sample,seed+1000003*k,_sampling(core,model,args.mode),holdout,reuse,args.mode))
    for model in MODELS:
        k+=1;jobs.append((str(out),"firm_weighted","persistent_ar1","core_cpi_negative_unemployment_gap",model,"full",seed+1000003*k,_sampling(core,model,args.mode),holdout,reuse,args.mode))
    summaries=[]
    with ProcessPoolExecutor(max_workers=min(args.workers,len(jobs))) as pool:
        for future in as_completed([pool.submit(_fit_job,j) for j in jobs]):
            z=future.result();summaries.append(z);print(f"CORE {z['sample']} {z['model']} {z['cycle']} {z['error_model']} {z['cell']} Rhat={z['diagnostics']['max_rhat']:.4f}",flush=True)
    coeff=[]
    for z in summaries:
        if z["sample"][2] not in {67,83}:continue
        for name,v in z["coefficients"].items():coeff.append({"cycle":z["cycle"],"error_model":z["error_model"],"cell":z["cell"],"model":z["model"],"sample_start":z["sample"][0],"sample_end":z["sample"][1],"parameter":name,**v,"fit_max_rhat":z["diagnostics"]["max_rhat"],"fit_min_bulk_ess":z["diagnostics"]["min_bulk_ess"],"retried":z.get("retried",False)})
        for name,v in z.get("derived",{}).items():coeff.append({"cycle":z["cycle"],"error_model":z["error_model"],"cell":z["cell"],"model":z["model"],"sample_start":z["sample"][0],"sample_end":z["sample"][1],"parameter":name+"_derived",**v,"fit_max_rhat":z["diagnostics"]["max_rhat"],"fit_min_bulk_ess":z["diagnostics"]["min_bulk_ess"],"retried":z.get("retried",False)})
    pd.DataFrame(coeff).to_csv(out/"tables"/"coefficients.csv",index=False)
    comparisons=[]
    for label in labels:
        state=_state(label)
        for cell_name,cell in cells.items():
            mask=np.asarray(cell.periods<pd.Period(holdout,freq="Q"));train=_subset(cell,mask);test=_subset(cell,~mask)
            for model in MODELS:
                fp=out/"draws"/"full"/model/label/"iid"/f"{cell_name}.npz";tp=out/"draws"/"train"/model/label/"iid"/f"{cell_name}.npz";fm=json.loads(fp.with_suffix(".json").read_text());tm=json.loads(tp.with_suffix(".json").read_text());full=load_qoq(fp,fm["diagnostics"]);trained=load_qoq(tp,tm["diagnostics"]);ll=qoq_pointwise_loglik(cell,full) if model in {"direct_only","free_combined"} else dynamic_loglik(cell,full);comparisons.append({"cycle":label,"cell":cell_name,"model":model,**_metrics(ll),**_holdout(train,test,trained,state,model),"max_rhat":fm["diagnostics"]["max_rhat"],"min_bulk_ess":fm["diagnostics"]["min_bulk_ess"]})
    pd.DataFrame(comparisons).to_csv(out/"tables"/"model_comparison.csv",index=False)
    rec=pd.DataFrame()
    if not args.skip_recovery:
        reps=int(core["static_recovery"]["replicates"] if args.mode=="full" else 2);groups=[]
        for j,mode in enumerate(core["static_recovery"]["modes"]):groups.append(("static",mode,reps,core["static_recovery"]["scenarios"],seed+51000001+10000019*j,dict(core["sampling"]["static_recovery"]),str(out/"draws"/"full"/"free_combined"/"firm_weighted"/"iid"/"core_cpi_negative_unemployment_gap.npz")))
        repsd=int(core["dynamic_recovery"]["replicates"] if args.mode=="full" else 2)
        for j,mode in enumerate(core["dynamic_recovery"]["modes"]):groups.append(("dynamic",mode,repsd,core["dynamic_recovery"]["scenarios"],seed+91000001+10000019*j,dict(core["sampling"]["dynamic_recovery"]),str(out/"draws"/"full"/"varying_theta"/"firm_weighted"/"iid"/"core_cpi_negative_unemployment_gap.npz")))
        rows=[]
        with ProcessPoolExecutor(max_workers=min(args.workers,len(groups))) as pool:
            for future in as_completed([pool.submit(_recovery_group,g) for g in groups]):rows.extend(future.result());print(f"CORE RECOVERY {len(rows)} coefficient rows",flush=True)
        rec=pd.DataFrame(rows);rec.to_csv(out/"tables"/"recovery_replications.csv",index=False);power=rec.groupby(["kind","mode","scenario","parameter"]).agg(replicates=("replicate","size"),standardized_true=("standardized_true","first"),suggestive_rate=("suggestive_detected","mean"),strong_rate=("strong_detected","mean"),false_positive_rate=("false_positive","mean"),coverage=("coverage","mean"),mean_estimate=("mean","mean"),mean_p_positive=("p_positive","mean"),mean_sd_ratio=("sd_ratio","mean"),max_rhat=("max_rhat","max"),min_bulk_ess=("min_bulk_ess","min"),retry_rate=("retried","mean")).reset_index();power.to_csv(out/"tables"/"recovery_power.csv",index=False)
    observed=[z for z in summaries if z["sample"][2]==83];gate={"observed_max_rhat":max(z["diagnostics"]["max_rhat"] for z in observed),"observed_min_bulk_ess":min(z["diagnostics"]["min_bulk_ess"] for z in observed),"observed_computational_pass":bool(max(z["diagnostics"]["max_rhat"] for z in observed)<=float(core["gates"]["observed_max_rhat"]) and min(z["diagnostics"]["min_bulk_ess"] for z in observed)>=float(core["gates"]["observed_min_bulk_ess"]))}
    if len(rec):gate.update(recovery_max_rhat=float(rec.max_rhat.max()),recovery_min_bulk_ess=float(rec.min_bulk_ess.min()))
    data_path=data_root()/"processed"/"model_ready.csv";manifest={"revision":core["revision"],"profile":args.mode,"not_for_inference":args.mode!="full","created_utc":datetime.now(timezone.utc).isoformat(timespec="seconds"),"elapsed_seconds":time.time()-started,"seed":seed,"config_hashes":hashes,"data_sha256":_sha(data_path),"measurement_hashes":{l:_sha(BASE/"draws"/"cycle"/f"{l}.npz") for l in labels},"observed_fit_count":len(jobs),"recovery_fit_count":int(len(rec)/2),"sample":{k:[str(v.periods[0]),str(v.periods[-1]),len(v.periods)] for k,v in cells.items()},"expectation":core["price"]["expectation_interpretation"],"git_commit":_git(["rev-parse","HEAD"]),"git_dirty":bool(_git(["status","--porcelain"])),"gate":gate,"interpretation":"Core CPI is estimated separately with the same cut competition states and matched sample; headline-CPI SPF CPI3 is an explicit proxy for unavailable core-CPI expectations."};_json(out/"manifest.json",manifest);print(json.dumps(gate,indent=2),flush=True)


if __name__=="__main__":main()
