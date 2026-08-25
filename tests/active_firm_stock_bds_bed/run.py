"""Run the external BDS/BED firm-state and free-theta recovery experiment."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
from datetime import datetime,timezone
import hashlib,json,subprocess,sys,time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT=next(p for p in Path(__file__).resolve().parents if (p/"pyproject.toml").exists())
sys.path[:0]=[str(ROOT),str(ROOT/"src"),str(ROOT/"tests")]
from tests import _bootstrap  # noqa:F401,E402
from nkpc_hsa.config import load_yaml  # noqa:E402
from nkpc_hsa.paths import data_root  # noqa:E402
from tests.active_firm_stock_bds_bed.functions import (  # noqa:E402
    FirmStateFit,detection_indicator,fit_firm_state,fit_theta,load_external_data,load_state,
    load_theta,save_state,save_theta,simulate_inflation,summarize_state,summarize_theta,
)

BUNDLE=Path(__file__).resolve().parent


def _json(path,payload):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(payload,indent=2,ensure_ascii=False)+"\n")


def _sha(path):
    h=hashlib.sha256();
    with Path(path).open("rb") as f:
        for block in iter(lambda:f.read(1<<20),b""):h.update(block)
    return h.hexdigest()


def _git(args):
    return subprocess.run(["git",*args],cwd=ROOT,text=True,stdout=subprocess.PIPE,stderr=subprocess.DEVNULL).stdout.strip()


def _theta_job(args):
    config_path,profile,state_path,state_json,cell_name,seed,out_path=args
    cfg=load_yaml(config_path);data=load_external_data(cfg);state_summary=json.loads(Path(state_json).read_text());state=load_state(Path(state_path),state_summary["diagnostics"])
    fit=fit_theta(data.cells[cell_name],state,cfg,cfg["sampling"][profile],seed);save_theta(Path(out_path),fit);s=summarize_theta(fit);_json(Path(out_path).with_suffix(".json"),s);return s


def _recovery_job(args):
    config_path,profile,state_path,state_json,observed_path,observed_json,mode,theta,replicate,seed,out_path=args
    cfg=load_yaml(config_path);data=load_external_data(cfg);ss=json.loads(Path(state_json).read_text());state=load_state(Path(state_path),ss["diagnostics"])
    os=json.loads(Path(observed_json).read_text());observed=load_theta(Path(observed_path),os["diagnostics"]);rng=np.random.default_rng(seed)
    synthetic,true_hat=simulate_inflation(rng,data.cells["ppi_inverse_markup"],observed,state,float(theta))
    recovery_state=state
    if mode=="oracle_state":
        zeros=np.zeros((1,1,len(true_hat)));recovery_state=FirmStateFit(tuple(map(str,synthetic.periods)),tuple(),np.zeros((1,1,0)),zeros,true_hat[None,None,:],true_hat[None,None,:],{}, {},[])
    fit=fit_theta(synthetic,recovery_state,cfg,cfg["sampling"][profile],seed+99991,recovery=True);save_theta(Path(out_path),fit)
    s=summarize_theta(fit);r=s["coefficients"]["theta_N"];recovery_cfg=cfg["recovery"]
    det=detection_indicator(r["q2.5"],r["q97.5"],r["p_positive"],r["posterior_prior_sd_ratio"],positive=True,
        sign_probability=float(recovery_cfg["sign_probability"]),sd_ratio_limit=float(recovery_cfg["posterior_prior_sd_ratio"]),
        require_interval=bool(recovery_cfg["require_interval_excludes_zero"]))
    row={"mode":mode,"theta_true":float(theta),"replicate":int(replicate),"mean":r["mean"],"sd":r["sd"],"q2.5":r["q2.5"],"q97.5":r["q97.5"],"p_positive":r["p_positive"],"posterior_prior_sd_ratio":r["posterior_prior_sd_ratio"],"detected":det,"max_rhat":s["diagnostics"]["max_rhat"],"min_bulk_ess":s["diagnostics"]["min_bulk_ess"],"seed":int(seed)}
    _json(Path(out_path).with_suffix(".json"),row);return row


def _coefficient_rows(summaries):
    rows=[]
    for s in summaries:
        for name,v in s["coefficients"].items():rows.append({"cell":s["cell"],"parameter":name,**v,"fit_max_rhat":s["diagnostics"]["max_rhat"],"fit_min_bulk_ess":s["diagnostics"]["min_bulk_ess"],"fit_min_tail_ess":s["diagnostics"]["min_tail_ess"]})
    return rows


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--profile",choices=("mock","smoke","full"),default="mock");parser.add_argument("--workers",type=int,default=4);parser.add_argument("--reuse-state",action="store_true");parser.add_argument("--no-report",action="store_true");args=parser.parse_args()
    started=time.time();cfg=load_yaml(BUNDLE/"config.yaml");sampling=cfg["sampling"][args.profile];seed=int(cfg["sampling"]["seed"]);out=BUNDLE/"results"/args.profile
    for p in (out,out/"draws"/"state",out/"draws"/"observed",out/"draws"/"recovery",out/"tables"):p.mkdir(parents=True,exist_ok=True)
    data=load_external_data(cfg);state_path=out/"draws"/"state"/"external_firm_state.npz";state_json=state_path.with_suffix(".json")
    if args.reuse_state and state_path.exists() and state_json.exists():
        state_summary=json.loads(state_json.read_text());state=load_state(state_path,state_summary["diagnostics"])
    else:
        state=fit_firm_state(data,cfg,sampling,seed);save_state(state_path,state);state_summary=summarize_state(state);_json(state_json,state_summary)
    print(f"STATE Rhat={state_summary['diagnostics']['max_rhat']:.4f} minESS={state_summary['diagnostics']['min_bulk_ess']:.1f}",flush=True)

    jobs=[]
    for j,cell in enumerate(data.cells):
        path=out/"draws"/"observed"/f"{cell}.npz";jobs.append((BUNDLE/"config.yaml",args.profile,state_path,state_json,cell,seed+1000003*(j+1),path))
    observed=[]
    with ProcessPoolExecutor(max_workers=min(args.workers,len(jobs))) as pool:
        for future in as_completed([pool.submit(_theta_job,j) for j in jobs]):
            s=future.result();observed.append(s);print(f"OBSERVED {s['cell']} Rhat={s['diagnostics']['max_rhat']:.4f}",flush=True)
    observed.sort(key=lambda x:x["cell"]);pd.DataFrame(_coefficient_rows(observed)).to_csv(out/"tables"/"coefficients.csv",index=False)

    primary_path=out/"draws"/"observed"/"ppi_inverse_markup.npz";primary_json=primary_path.with_suffix(".json");recovery_jobs=[];replicates=int(sampling["recovery_replicates"])
    for m,mode in enumerate(cfg["recovery"]["modes"]):
        for j,theta in enumerate(cfg["recovery"]["theta_grid"]):
            for rep in range(replicates):
                rseed=seed+2000003+10000019*m+10007*j+101*rep;path=out/"draws"/"recovery"/mode/f"theta_{float(theta):.2f}_rep_{rep:03d}.npz"
                recovery_jobs.append((BUNDLE/"config.yaml",args.profile,state_path,state_json,primary_path,primary_json,mode,float(theta),rep,rseed,path))
    recovery=[]
    with ProcessPoolExecutor(max_workers=min(args.workers,len(recovery_jobs))) as pool:
        futures=[pool.submit(_recovery_job,j) for j in recovery_jobs]
        for k,future in enumerate(as_completed(futures),1):
            recovery.append(future.result())
            if k%max(1,len(futures)//10)==0:print(f"RECOVERY {k}/{len(futures)}",flush=True)
    rec=pd.DataFrame(recovery).sort_values(["mode","theta_true","replicate"]);rec.to_csv(out/"tables"/"recovery_replications.csv",index=False)
    power=rec.groupby(["mode","theta_true"]).agg(replicates=("detected","size"),detection_rate=("detected","mean"),mean_estimate=("mean","mean"),mean_interval_low=("q2.5","mean"),mean_interval_high=("q97.5","mean"),mean_sign_probability=("p_positive","mean"),mean_sd_ratio=("posterior_prior_sd_ratio","mean"),max_rhat=("max_rhat","max"),min_bulk_ess=("min_bulk_ess","min")).reset_index();power.to_csv(out/"tables"/"recovery_power.csv",index=False)

    state_rows=[]
    for name in state.names:
        v=state_summary[name];state_rows.append({"parameter":name,**v,"rhat":state_summary["diagnostics"]["rhat"][name],"ess_bulk":state_summary["diagnostics"]["ess_bulk"][name],"ess_tail":state_summary["diagnostics"]["ess_tail"][name]})
    for name in ("slow_innovation_variance","cycle_innovation_variance"):state_rows.append({"parameter":name,**state_summary[name],"rhat":np.nan,"ess_bulk":np.nan,"ess_tail":np.nan})
    pd.DataFrame(state_rows).to_csv(out/"tables"/"state_parameters.csv",index=False)
    path_table=pd.DataFrame({"period":state.periods,"nbar_mean":state.nbar.mean((0,1)),"nbar_q2.5":np.percentile(state.nbar,2.5,axis=(0,1)),"nbar_q97.5":np.percentile(state.nbar,97.5,axis=(0,1)),"nhat_mean":state.nhat.mean((0,1)),"nhat_q2.5":np.percentile(state.nhat,2.5,axis=(0,1)),"nhat_q97.5":np.percentile(state.nhat,97.5,axis=(0,1)),"n_total_mean":state.n_total.mean((0,1)),"n_total_q2.5":np.percentile(state.n_total,2.5,axis=(0,1)),"n_total_q97.5":np.percentile(state.n_total,97.5,axis=(0,1)),"bds_coordinate":data.bds_coordinate,"bds_firms":data.bds_firms,"bed_births":data.bed_births,"bed_deaths":data.bed_deaths,"bed_net_standardized":data.bed_net_standardized,"bed_observed":data.bed_observed.astype(int)});path_table.to_csv(out/"tables"/"state_paths.csv",index=False)

    rhat_limit=float(cfg["gates"][f"{args.profile}_max_rhat"]);ess_limit=float(cfg["gates"][f"{args.profile}_min_bulk_ess"]);observed_max=max(s["diagnostics"]["max_rhat"] for s in observed);observed_min=min(s["diagnostics"]["min_bulk_ess"] for s in observed)
    gate={"max_rhat_required":rhat_limit,"min_bulk_ess_required":ess_limit,"state_max_rhat":state_summary["diagnostics"]["max_rhat"],"state_min_bulk_ess":state_summary["diagnostics"]["min_bulk_ess"],"observed_max_rhat":observed_max,"observed_min_bulk_ess":observed_min,"passed":bool(max(state_summary["diagnostics"]["max_rhat"],observed_max)<=rhat_limit and min(state_summary["diagnostics"]["min_bulk_ess"],observed_min)>=ess_limit)}
    detectable=power[(power["mode"]=="propagated_state")&(power.theta_true>0)&(power.detection_rate>=float(cfg["recovery"]["detection_probability"]))]
    minimum=None if detectable.empty else float(detectable.theta_true.min())
    manifest={"revision":cfg["revision"],"profile":args.profile,"mock_or_smoke_not_for_inference":args.profile!="full","created_utc":datetime.now(timezone.utc).isoformat(timespec="seconds"),"elapsed_seconds":time.time()-started,"sampling":sampling,"seed":seed,"git_commit":_git(["rev-parse","HEAD"]),"git_dirty":bool(_git(["status","--porcelain"])),"config_sha256":_sha(BUNDLE/"config.yaml"),"data_files":{k:{"path":str(data_root()/cfg["data"][k]),"sha256":_sha(data_root()/cfg["data"][k])} for k in ("bds_file","bed_births_file","bed_deaths_file")},"sample":{"state":[str(data.periods[0]),str(data.periods[-1]),len(data.periods)],"nkpc":{k:[str(v.periods[0]),str(v.periods[-1]),len(v.periods)] for k,v in data.cells.items()},"bds_observations":int(np.isfinite(data.bds_coordinate).sum()),"bed_complete_quarters":int(data.bed_observed.sum())},"gate":gate,"minimum_detectable_theta":minimum,"minimum_detectable_is_inferential":args.profile=="full" and gate["passed"] and minimum is not None,"interpretation_rule":"Free theta_N recovery precedes any delta, lambda, or HSA restriction."};_json(out/"manifest.json",manifest)
    print(json.dumps(gate,indent=2),flush=True)
    if not args.no_report:
        from tests.active_firm_stock_bds_bed.build_report import build
        build(args.profile)


if __name__=="__main__":main()
