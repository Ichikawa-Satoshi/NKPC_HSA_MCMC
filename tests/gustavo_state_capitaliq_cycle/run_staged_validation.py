"""Run recovery, nested comparison, and the predeclared HSA promotion gate."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
from dataclasses import replace
from datetime import datetime,timezone
import hashlib,json,subprocess,sys,time
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd
from scipy.special import logsumexp
from scipy.stats import gaussian_kde

ROOT=next(p for p in Path(__file__).resolve().parents if (p/"pyproject.toml").exists())
sys.path[:0]=[str(ROOT),str(ROOT/"src"),str(ROOT/"tests")]
from tests import _bootstrap  # noqa:F401,E402
from nkpc_hsa.config import load_yaml  # noqa:E402
from nkpc_hsa.paths import data_root  # noqa:E402
from tests.active_firm_stock_bds_bed.functions import ThetaCell,robust_scale  # noqa:E402
from tests.gustavo_state_capitaliq_cycle.functions import (  # noqa:E402
    CycleFit,build_qoq_design,fit_qoq_theta,load_cycle,load_nkpc_cells,load_qoq,
    qoq_pointwise_loglik,save_qoq,simulate_qoq_combined,summarize_qoq,
)

BUNDLE=Path(__file__).resolve().parent;BASE=BUNDLE/"results"/"mock_qoq";OUT=BUNDLE/"results"/"staged_validation"


def _json(path: Path,payload):path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(payload,indent=2,ensure_ascii=False)+"\n")


def _sha(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda:handle.read(1<<20),b""):h.update(block)
    return h.hexdigest()


def _git(args):return subprocess.run(["git",*args],cwd=ROOT,text=True,stdout=subprocess.PIPE,stderr=subprocess.DEVNULL).stdout.strip()


def _state(label: str) -> CycleFit:
    path=BASE/"draws"/"cycle"/f"{label}.npz";return load_cycle(path,json.loads(path.with_suffix(".json").read_text()))


def _subset(cell: ThetaCell,mask: np.ndarray) -> ThetaCell:
    return ThetaCell(cell.name,cell.periods[mask],cell.pi[mask],cell.pi_lag[mask],cell.epi[mask],cell.x[mask],robust_scale(cell.pi[mask]),robust_scale(cell.x[mask]))


def _fit_job(args):
    label,error_model,cell_name,model,sample,seed,sampling,holdout_start,reuse=args;cfg=load_yaml(BUNDLE/"config.yaml");state=_state(label);cell=load_nkpc_cells(cfg)[cell_name]
    if sample=="train":cell=_subset(cell,np.asarray(cell.periods<pd.Period(holdout_start,freq="Q")))
    path=OUT/"draws"/sample/model/label/error_model/f"{cell_name}.npz"
    if reuse and path.exists() and path.with_suffix(".json").exists():summary=json.loads(path.with_suffix(".json").read_text());return {"path":str(path),"sample":sample,"model":model,**summary}
    fit=fit_qoq_theta(cell,state,cfg,seed,error_model=error_model,include_delta=model=="free_combined",sampling_override=sampling);save_qoq(path,fit);summary=summarize_qoq(fit);_json(path.with_suffix(".json"),summary);return {"path":str(path),"sample":sample,"model":model,**summary}


def _oracle_state(periods,hat,bar):
    return CycleFit("oracle",tuple(map(str,periods)),tuple(),np.zeros((1,1,0)),np.asarray(bar)[None,None,:],np.asarray(hat)[None,None,:],{},np.asarray(hat))


def _recovery_group(args):
    activity,error_model,mode,reps,seed,sampling,scenarios,full_path=args;cfg=load_yaml(BUNDLE/"config.yaml");state=_state("firm_weighted");cell=load_nkpc_cells(cfg)[activity];meta=json.loads(Path(full_path).with_suffix(".json").read_text());observed=load_qoq(Path(full_path),meta["diagnostics"]);rows=[]
    retry={"iterations":1800,"warmup":600,"thin":2,"chains":4}
    for sidx,(scenario,(sdelta,stheta)) in enumerate(scenarios.items()):
        for rep in range(reps):
            rseed=seed+1000003*sidx+1009*rep;rng=np.random.default_rng(rseed);synthetic,true_hat,true_bar,true=simulate_qoq_combined(rng,cell,observed,state,float(sdelta),float(stheta));use_state=state if mode=="propagated_state" else _oracle_state(synthetic.periods,true_hat,true_bar)
            fit=fit_qoq_theta(synthetic,use_state,cfg,rseed+7000001,error_model=error_model,include_delta=True,recovery=True,sampling_override=sampling);summary=summarize_qoq(fit);retried=False
            if summary["diagnostics"]["max_rhat"]>1.10 or summary["diagnostics"]["min_bulk_ess"]<100:
                fit=fit_qoq_theta(synthetic,use_state,cfg,rseed+17000001,error_model=error_model,include_delta=True,recovery=True,sampling_override=retry);summary=summarize_qoq(fit);retried=True
            for parameter,standardized_true in (("delta",float(sdelta)),("theta_CIQ",float(stheta))):
                z=summary["coefficients"][parameter];raw=true[parameter];positive=raw>0;suggestive=bool(positive and z["p_positive"]>=.80 and z["posterior_prior_sd_ratio"]<=.75);strong=bool(positive and z["p_positive"]>=.975 and z["q2.5"]>0 and z["posterior_prior_sd_ratio"]<=.75);false_positive=bool(not positive and ((z["p_positive"]>=.975 and z["q2.5"]>0) or (z["p_positive"]<=.025 and z["q97.5"]<0)))
                rows.append({"activity":activity,"error_model":error_model,"mode":mode,"scenario":scenario,"replicate":rep,"parameter":parameter,"standardized_true":standardized_true,"raw_true":raw,"mean":z["mean"],"q2.5":z["q2.5"],"q97.5":z["q97.5"],"p_positive":z["p_positive"],"sd_ratio":z["posterior_prior_sd_ratio"],"coverage":bool(z["q2.5"]<=raw<=z["q97.5"]),"suggestive_detected":suggestive,"strong_detected":strong,"false_positive":false_positive,"delta_theta_correlation":summary["delta_theta_correlation"],"max_rhat":summary["diagnostics"]["max_rhat"],"min_bulk_ess":summary["diagnostics"]["min_bulk_ess"],"retried":retried})
    return rows


def _loo_metrics(cell,fit):
    ll=qoq_pointwise_loglik(cell,fit);idata=az.from_dict({"posterior":{"beta":fit.draws},"log_likelihood":{"inflation":ll}});loo=az.loo(idata,pointwise=True);pareto=np.asarray(loo.pareto_k);flat=ll.reshape(-1,ll.shape[-1]);lppd=logsumexp(flat,axis=0)-np.log(len(flat));pwaic_i=np.var(flat,axis=0,ddof=1);waic_i=lppd-pwaic_i
    return {"elpd_loo":float(loo.elpd),"se_loo":float(loo.se),"p_loo":float(loo.p),"elpd_waic":float(np.sum(waic_i)),"se_waic":float(np.sqrt(len(waic_i)*np.var(waic_i,ddof=1))),"p_waic":float(np.sum(pwaic_i)),"max_pareto_k":float(np.max(pareto)),"pareto_k_over_0.7":int(np.sum(pareto>.7))}


def _holdout_metrics(train: ThetaCell,test: ThetaCell,fit: CycleFit | object,state: CycleFit) -> dict[str,float]:
    sp=pd.PeriodIndex(state.periods,freq="Q");train_pos=sp.get_indexer(train.periods);test_pos=sp.get_indexer(test.periods);loglik=[];means=[]
    for chain in range(fit.draws.shape[0]):
        for draw in range(fit.draws.shape[1]):
            cs=int(fit.state_chain[chain,draw]);ds=int(fit.state_draw[chain,draw]);hat=state.nhat[cs,ds,test_pos];bar=None
            if "delta" in fit.names:
                raw=state.nbar_used[cs,ds];bar=raw[test_pos]-float(np.mean(raw[train_pos]));X=np.column_stack([np.ones(len(test.periods)),test.pi_lag,test.epi,test.x,bar*test.x,-hat])
            else:X,_=build_qoq_design(test,hat,None)
            mu=X@fit.draws[chain,draw];means.append(mu);loglik.append(-.5*np.log(2*np.pi*fit.sigma_u[chain,draw]**2)-.5*(test.pi-mu)**2/fit.sigma_u[chain,draw]**2)
    loglik=np.asarray(loglik);means=np.asarray(means);elpd=float(np.sum(logsumexp(loglik,axis=0)-np.log(len(loglik))));prediction=means.mean(axis=0);return {"holdout_elpd":elpd,"holdout_rmse":float(np.sqrt(np.mean((test.pi-prediction)**2))),"holdout_n":len(test.pi)}


def _savage_dickey(fit) -> float:
    j=fit.names.index("delta");draws=fit.draws[:,:,j].reshape(-1);post=float(gaussian_kde(draws)([0.])[0]);prior=float(1/(np.sqrt(2*np.pi)*fit.prior_sd["delta"]));return post/prior


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--workers",type=int,default=4);parser.add_argument("--no-report",action="store_true");parser.add_argument("--refit",action="store_true");args=parser.parse_args();started=time.time();cfg=load_yaml(BUNDLE/"config.yaml");scfg=load_yaml(BUNDLE/"staged_config.yaml");ccfg=load_yaml(BUNDLE/"combined_config.yaml");OUT.mkdir(parents=True,exist_ok=True);(OUT/"tables").mkdir(exist_ok=True)
    current_hashes={"base":_sha(BUNDLE/"config.yaml"),"combined":_sha(BUNDLE/"combined_config.yaml"),"staged":_sha(BUNDLE/"staged_config.yaml")};old_manifest=json.loads((OUT/"manifest.json").read_text()) if (OUT/"manifest.json").exists() else {};reuse=bool(not args.refit and old_manifest.get("config_hashes")==current_hashes)
    labels=list(cfg["data"]["capital_iq"]);cells=load_nkpc_cells(cfg);sampling={k:v for k,v in ccfg["sampling"].items() if k!="seed_offset"};seed=int(cfg["sampling"]["seed"])+91000000;holdout=scfg["comparison"]["holdout_start"]
    fit_jobs=[];counter=0
    for label in labels:
        for cell_name in cells:
            for model in ("direct_only","free_combined"):
                for sample in ("full","train"):
                    counter+=1;fit_jobs.append((label,"iid",cell_name,model,sample,seed+1000003*counter,sampling,holdout,reuse))
    counter+=1;fit_jobs.append(("firm_weighted","persistent_ar1",scfg["primary"]["activity"],"free_combined","full",seed+1000003*counter,sampling,holdout,reuse))
    fit_results=[]
    with ProcessPoolExecutor(max_workers=min(args.workers,len(fit_jobs))) as pool:
        for future in as_completed([pool.submit(_fit_job,j) for j in fit_jobs]):
            result=future.result();fit_results.append(result);print(f"FIT {result['sample']} {result['model']} {result['cycle']} {result['error_model']} {result['cell']} Rhat={result['diagnostics']['max_rhat']:.4f}",flush=True)
    comparison=[]
    for label in labels:
        state=_state(label)
        for cell_name,cell in cells.items():
            mask=np.asarray(cell.periods<pd.Period(holdout,freq="Q"));train=_subset(cell,mask);test=_subset(cell,~mask);model_rows={}
            for model in ("direct_only","free_combined"):
                fp=OUT/"draws"/"full"/model/label/"iid"/f"{cell_name}.npz";fm=json.loads(fp.with_suffix(".json").read_text());full=load_qoq(fp,fm["diagnostics"]);tp=OUT/"draws"/"train"/model/label/"iid"/f"{cell_name}.npz";tm=json.loads(tp.with_suffix(".json").read_text());trained=load_qoq(tp,tm["diagnostics"]);row={"cycle":label,"cell":cell_name,"model":model,**_loo_metrics(cell,full),**_holdout_metrics(train,test,trained,state),"max_rhat":fm["diagnostics"]["max_rhat"],"min_bulk_ess":fm["diagnostics"]["min_bulk_ess"]}
                if model=="free_combined":row["bf01_delta_zero"]=_savage_dickey(full)
                comparison.append(row);model_rows[model]=row
    comparison=pd.DataFrame(comparison);comparison.to_csv(OUT/"tables"/"model_comparison.csv",index=False)
    differences=[]
    for (cycle,cell),z in comparison.groupby(["cycle","cell"]):
        d=z.set_index("model");differences.append({"cycle":cycle,"cell":cell,"delta_elpd_loo_combined_minus_direct":d.loc["free_combined","elpd_loo"]-d.loc["direct_only","elpd_loo"],"delta_holdout_elpd_combined_minus_direct":d.loc["free_combined","holdout_elpd"]-d.loc["direct_only","holdout_elpd"],"delta_holdout_rmse_combined_minus_direct":d.loc["free_combined","holdout_rmse"]-d.loc["direct_only","holdout_rmse"],"bf01_delta_zero":d.loc["free_combined","bf01_delta_zero"],"max_pareto_k":max(d.max_pareto_k)})
    pd.DataFrame(differences).to_csv(OUT/"tables"/"model_comparison_differences.csv",index=False)
    scenarios={"null" if key is None else str(key):value for key,value in scfg["recovery"]["scenarios"].items()};rsamp=dict(scfg["recovery"]["sampling"]);groups=[];primary=scfg["primary"]
    full_primary=OUT/"draws"/"full"/"free_combined"/primary["cycle"]/primary["error_model"]/f"{primary['activity']}.npz"
    for m,mode in enumerate(scfg["recovery"]["primary_modes"]):groups.append((primary["activity"],primary["error_model"],mode,int(scfg["recovery"]["primary_replicates"]),seed+30000001+10000019*m,rsamp,scenarios,str(full_primary)))
    for j,r in enumerate(scfg["recovery"]["robustness"]):
        fp=OUT/"draws"/"full"/"free_combined"/primary["cycle"]/r["error_model"]/f"{r['activity']}.npz";groups.append((r["activity"],r["error_model"],r["mode"],int(r["replicates"]),seed+50000001+10000019*j,rsamp,scenarios,str(fp)))
    recovery_path=OUT/"tables"/"recovery_replications.csv";expected=2*sum(int(g[3])*len(scenarios) for g in groups)
    if reuse and recovery_path.exists() and len(pd.read_csv(recovery_path))==expected:
        rec=pd.read_csv(recovery_path);rec["scenario"]=rec["scenario"].fillna("null");print(f"RECOVERY reused {len(rec)} coefficient rows",flush=True)
    else:
        recovery=[]
        with ProcessPoolExecutor(max_workers=min(args.workers,len(groups))) as pool:
            for future in as_completed([pool.submit(_recovery_group,g) for g in groups]):recovery.extend(future.result());print(f"RECOVERY GROUP {len(recovery)} coefficient rows",flush=True)
        rec=pd.DataFrame(recovery);rec["scenario"]=rec["scenario"].fillna("null");rec.to_csv(recovery_path,index=False)
    power=rec.groupby(["activity","error_model","mode","scenario","parameter"]).agg(replicates=("replicate","size"),standardized_true=("standardized_true","first"),suggestive_rate=("suggestive_detected","mean"),strong_rate=("strong_detected","mean"),false_positive_rate=("false_positive","mean"),coverage=("coverage","mean"),mean_estimate=("mean","mean"),mean_p_positive=("p_positive","mean"),mean_sd_ratio=("sd_ratio","mean"),max_rhat=("max_rhat","max"),min_bulk_ess=("min_bulk_ess","min"),retry_rate=("retried","mean")).reset_index();power.to_csv(OUT/"tables"/"recovery_power.csv",index=False)
    gatecfg=scfg["promotion_gate"];target=power[(power.activity==primary["activity"])&(power.error_model==primary["error_model"])&(power["mode"]==gatecfg["mode"])&(power.scenario==gatecfg["scenario"])].set_index("parameter");null=power[(power.activity==primary["activity"])&(power.error_model==primary["error_model"])&(power["mode"]==gatecfg["mode"])&(power.scenario=="null")].set_index("parameter");checks={}
    for parameter in ("delta","theta_CIQ"):
        checks[parameter]={"suggestive_rate":float(target.loc[parameter,"suggestive_rate"]),"coverage":float(target.loc[parameter,"coverage"]),"null_false_positive_rate":float(null.loc[parameter,"false_positive_rate"]),"suggestive_required":float(gatecfg["minimum_suggestive_rate_each_coefficient"]),"coverage_required":float(gatecfg["minimum_interval_coverage_each_coefficient"]),"false_positive_maximum":float(gatecfg["maximum_null_false_positive_rate_each_coefficient"])}
    conv={"max_rhat":float(rec.max_rhat.max()),"min_bulk_ess":float(rec.min_bulk_ess.min()),"max_rhat_required":float(gatecfg["maximum_rhat"]),"min_bulk_ess_required":float(gatecfg["minimum_bulk_ess"])}
    passed=all(v["suggestive_rate"]>=v["suggestive_required"] and v["coverage"]>=v["coverage_required"] and v["null_false_positive_rate"]<=v["false_positive_maximum"] for v in checks.values()) and conv["max_rhat"]<=conv["max_rhat_required"] and conv["min_bulk_ess"]>=conv["min_bulk_ess_required"]
    promotion={"passed":bool(passed),"coefficient_checks":checks,"convergence":conv,"dynamic_hsa_status":"promoted" if passed else "stopped_by_predeclared_recovery_gate"};_json(OUT/"promotion_gate.json",promotion)
    data_path=data_root()/"processed"/"model_ready.csv";manifest={"revision":scfg["revision"],"profile":"staged_validation","not_for_inference":True,"created_utc":datetime.now(timezone.utc).isoformat(timespec="seconds"),"elapsed_seconds":time.time()-started,"seed":seed,"git_commit":_git(["rev-parse","HEAD"]),"git_dirty":bool(_git(["status","--porcelain"])),"config_hashes":current_hashes,"data_sha256":_sha(data_path),"measurement_hashes":{label:_sha(BASE/"draws"/"cycle"/f"{label}.npz") for label in labels},"fit_count":len(fit_jobs),"recovery_fit_count":int(len(rec)/2),"promotion_gate":promotion,"dynamic_hsa_run":bool(passed),"interpretation":"Failure of the gate is an empirical stopping result, not permission to impose an HSA restriction."};_json(OUT/"manifest.json",manifest);print(json.dumps(promotion,indent=2),flush=True)
    if passed:raise NotImplementedError("Promotion gate passed: dynamic HSA stage must be run before reporting")
    if not args.no_report:
        from tests.gustavo_state_capitaliq_cycle.build_staged_report import build
        build()


if __name__=="__main__":main()
