"""Run the QoQ Gustavo-state x Capital-IQ-cycle mock experiment."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
from datetime import datetime,timezone
import hashlib,json,shutil,subprocess,sys,time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT=next(p for p in Path(__file__).resolve().parents if (p/"pyproject.toml").exists())
sys.path[:0]=[str(ROOT),str(ROOT/"src"),str(ROOT/"tests")]
from tests import _bootstrap  # noqa:F401,E402
from nkpc_hsa.config import load_yaml  # noqa:E402
from nkpc_hsa.paths import data_root  # noqa:E402
from tests.active_firm_stock_bds_bed.functions import detection_indicator  # noqa:E402
from tests.gustavo_state_capitaliq_cycle.functions import (  # noqa:E402
    CycleFit,fit_capital_iq_cycle,fit_gustavo_slow,fit_qoq_theta,load_cycle,load_measurements,
    load_nkpc_cells,load_qoq,load_slow,save_cycle,save_qoq,save_slow,simulate_qoq,
    summarize_cycle,summarize_qoq,summarize_slow,
)

BUNDLE=Path(__file__).resolve().parent;PROFILE="mock_qoq";LEGACY=BUNDLE/"results"/"mock_yoy_legacy"


def _json(path: Path,payload):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(payload,indent=2,ensure_ascii=False)+"\n")


def _sha(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(1<<20),b""):h.update(block)
    return h.hexdigest()


def _git(args):return subprocess.run(["git",*args],cwd=ROOT,text=True,stdout=subprocess.PIPE,stderr=subprocess.DEVNULL).stdout.strip()


def _oracle_state(periods,true_hat):
    zeros=np.zeros((1,1,len(true_hat)));return CycleFit("oracle",tuple(map(str,periods)),tuple(),np.zeros((1,1,0)),zeros,true_hat[None,None,:],{},true_hat)


def _prepare_measurement(cfg,out,seed,refit=False):
    slow_out=out/"draws"/"slow"/"gustavo_slow.npz";legacy_slow=LEGACY/"draws"/"slow"/"gustavo_slow.npz";cycle_summaries={}
    if legacy_slow.exists() and not refit:
        slow_out.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(legacy_slow,slow_out);shutil.copy2(legacy_slow.with_suffix(".json"),slow_out.with_suffix(".json"));slow_meta=json.loads(slow_out.with_suffix(".json").read_text());slow=load_slow(slow_out,slow_meta)
        for label in cfg["data"]["capital_iq"]:
            source=LEGACY/"draws"/"cycle"/f"{label}.npz";target=out/"draws"/"cycle"/f"{label}.npz";target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target);shutil.copy2(source.with_suffix(".json"),target.with_suffix(".json"));cycle_summaries[label]=json.loads(target.with_suffix(".json").read_text())
        source="mock_yoy_legacy_saved_measurement"
    else:
        slow=fit_gustavo_slow(cfg,seed);save_slow(slow_out,slow);slow_meta=summarize_slow(slow);_json(slow_out.with_suffix(".json"),slow_meta)
        for j,label in enumerate(cfg["data"]["capital_iq"]):
            fit=fit_capital_iq_cycle(label,slow,cfg,seed+100003*(j+1));target=out/"draws"/"cycle"/f"{label}.npz";save_cycle(target,fit);cycle_summaries[label]=summarize_cycle(fit);_json(target.with_suffix(".json"),cycle_summaries[label])
        source="refitted_competition_only"
    return slow,slow_meta,cycle_summaries,source


def _nkpc_job(args):
    label,error_model,cell_name,cycle_path,cycle_json,seed=args;cfg=load_yaml(BUNDLE/"config.yaml");meta=json.loads(Path(cycle_json).read_text());cycle=load_cycle(Path(cycle_path),meta);cell=load_nkpc_cells(cfg)[cell_name];fit=fit_qoq_theta(cell,cycle,cfg,seed,error_model=error_model)
    path=BUNDLE/"results"/PROFILE/"draws"/"nkpc"/label/error_model/f"{cell_name}.npz";save_qoq(path,fit);summary=summarize_qoq(fit);_json(path.with_suffix(".json"),summary);return summary


def _recovery_job(args):
    error_model,mode,theta,rep,cycle_path,cycle_json,observed_path,observed_json,seed=args;cfg=load_yaml(BUNDLE/"config.yaml");meta=json.loads(Path(cycle_json).read_text());cycle=load_cycle(Path(cycle_path),meta);ometa=json.loads(Path(observed_json).read_text());observed=load_qoq(Path(observed_path),ometa["diagnostics"]);cell=load_nkpc_cells(cfg)["ppi_inverse_markup"];rng=np.random.default_rng(seed);synthetic,true_hat=simulate_qoq(rng,cell,observed,cycle,float(theta));state=cycle if mode=="propagated_state" else _oracle_state(synthetic.periods,true_hat)
    fit=fit_qoq_theta(synthetic,state,cfg,seed+900001,error_model=error_model,recovery=True);summary=summarize_qoq(fit);r=summary["coefficients"]["theta_CIQ"];rcfg=cfg["recovery"]
    detected=detection_indicator(r["q2.5"],r["q97.5"],r["p_positive"],r["posterior_prior_sd_ratio"],sign_probability=float(rcfg["sign_probability"]),sd_ratio_limit=float(rcfg["posterior_prior_sd_ratio"]),require_interval=bool(rcfg["require_interval_excludes_zero"]))
    row={"error_model":error_model,"mode":mode,"theta_true":float(theta),"replicate":int(rep),"mean":r["mean"],"q2.5":r["q2.5"],"q97.5":r["q97.5"],"p_positive":r["p_positive"],"posterior_prior_sd_ratio":r["posterior_prior_sd_ratio"],"detected":detected,"max_rhat":summary["diagnostics"]["max_rhat"],"min_bulk_ess":summary["diagnostics"]["min_bulk_ess"]}
    path=BUNDLE/"results"/PROFILE/"draws"/"recovery"/error_model/mode/f"theta_{float(theta):.2f}_rep_{rep:02d}.npz";save_qoq(path,fit);_json(path.with_suffix(".json"),row);return row


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--workers",type=int,default=4);parser.add_argument("--refit-measurement",action="store_true");parser.add_argument("--no-report",action="store_true");args=parser.parse_args();started=time.time();cfg=load_yaml(BUNDLE/"config.yaml");seed=int(cfg["sampling"]["seed"]);out=BUNDLE/"results"/PROFILE
    for p in (out,out/"tables",out/"draws"/"slow",out/"draws"/"cycle",out/"draws"/"nkpc",out/"draws"/"recovery"):p.mkdir(parents=True,exist_ok=True)
    slow,slow_summary,cycle_summaries,measurement_source=_prepare_measurement(cfg,out,seed,args.refit_measurement);print(f"MEASUREMENT {measurement_source} slow Rhat={slow.diagnostics['max_rhat']:.4f}",flush=True)
    labels=list(cfg["data"]["capital_iq"]);errors=list(cfg["nkpc"]["error_models"]);cells=load_nkpc_cells(cfg);jobs=[]
    for j,(label,error_model,cell) in enumerate((l,e,c) for l in labels for e in errors for c in cells):
        cp=out/"draws"/"cycle"/f"{label}.npz";jobs.append((label,error_model,cell,cp,cp.with_suffix(".json"),seed+1000003*(j+1)))
    nkpc=[]
    with ProcessPoolExecutor(max_workers=min(args.workers,len(jobs))) as pool:
        for future in as_completed([pool.submit(_nkpc_job,j) for j in jobs]):
            s=future.result();nkpc.append(s);print(f"QOQ {s['cycle']} {s['error_model']} {s['cell']} Rhat={s['diagnostics']['max_rhat']:.4f}",flush=True)
    nkpc.sort(key=lambda x:(x["cycle"],x["error_model"],x["cell"]));rows=[]
    for s in nkpc:
        for parameter,value in s["coefficients"].items():rows.append({"cycle":s["cycle"],"error_model":s["error_model"],"cell":s["cell"],"parameter":parameter,**value,"fit_max_rhat":s["diagnostics"]["max_rhat"],"fit_min_bulk_ess":s["diagnostics"]["min_bulk_ess"]})
    pd.DataFrame(rows).to_csv(out/"tables"/"coefficients.csv",index=False)

    primary=str(cfg["recovery"]["primary_cycle"]);cp=out/"draws"/"cycle"/f"{primary}.npz";recovery_jobs=[]
    for ee,error_model in enumerate(errors):
        op=out/"draws"/"nkpc"/primary/error_model/"ppi_inverse_markup.npz"
        for m,mode in enumerate(cfg["recovery"]["modes"]):
            for j,theta in enumerate(cfg["recovery"]["theta_grid"]):
                for rep in range(int(cfg["recovery"]["replicates"])):recovery_jobs.append((error_model,mode,float(theta),rep,cp,cp.with_suffix(".json"),op,op.with_suffix(".json"),seed+3000001+100000007*ee+10000019*m+10007*j+101*rep))
    recovery=[]
    with ProcessPoolExecutor(max_workers=min(args.workers,len(recovery_jobs))) as pool:
        for k,future in enumerate(as_completed([pool.submit(_recovery_job,j) for j in recovery_jobs]),1):
            recovery.append(future.result())
            if k%12==0:print(f"RECOVERY {k}/{len(recovery_jobs)}",flush=True)
    rec=pd.DataFrame(recovery).sort_values(["error_model","mode","theta_true","replicate"]);rec.to_csv(out/"tables"/"recovery_replications.csv",index=False)
    power=rec.groupby(["error_model","mode","theta_true"]).agg(replicates=("detected","size"),detection_rate=("detected","mean"),mean_estimate=("mean","mean"),mean_sign_probability=("p_positive","mean"),mean_sd_ratio=("posterior_prior_sd_ratio","mean"),max_rhat=("max_rhat","max"),min_bulk_ess=("min_bulk_ess","min")).reset_index();power.to_csv(out/"tables"/"recovery_power.csv",index=False)

    state_rows=[{"block":"gustavo_slow","variant":"gustavo","parameter":p,**slow_summary[p]} for p in ("mu","sigma_bar")]
    for label,s in cycle_summaries.items():
        for parameter,value in s["parameters"].items():state_rows.append({"block":"capital_iq_cycle","variant":label,"parameter":parameter,**value})
    pd.DataFrame(state_rows).to_csv(out/"tables"/"state_parameters.csv",index=False)
    periods,g,_=load_measurements(cfg);path=pd.DataFrame({"period":periods.astype(str),"gustavo_anchor":g.reindex(periods).to_numpy(float),"gustavo_slow_mean":slow.nbar.mean((0,1)),"gustavo_slow_q2.5":np.percentile(slow.nbar,2.5,axis=(0,1)),"gustavo_slow_q97.5":np.percentile(slow.nbar,97.5,axis=(0,1))})
    for label in labels:
        cycle=load_cycle(out/"draws"/"cycle"/f"{label}.npz",cycle_summaries[label]);idx=pd.PeriodIndex(cycle.periods,freq="Q");path[f"{label}_observed"]=pd.Series(cycle.observed_coordinate,index=idx).reindex(periods).to_numpy();path[f"{label}_cycle_mean"]=pd.Series(cycle.nhat.mean((0,1)),index=idx).reindex(periods).to_numpy();path[f"{label}_cycle_q2.5"]=pd.Series(np.percentile(cycle.nhat,2.5,axis=(0,1)),index=idx).reindex(periods).to_numpy();path[f"{label}_cycle_q97.5"]=pd.Series(np.percentile(cycle.nhat,97.5,axis=(0,1)),index=idx).reindex(periods).to_numpy()
    path.to_csv(out/"tables"/"state_paths.csv",index=False)

    all_diag=[slow.diagnostics,*[s["diagnostics"] for s in cycle_summaries.values()],*[s["diagnostics"] for s in nkpc]];rhat_limit=float(cfg["gates"]["mock_max_rhat"]);ess_limit=float(cfg["gates"]["mock_min_bulk_ess"])
    gate={"max_rhat_required":rhat_limit,"min_bulk_ess_required":ess_limit,"max_anchor_error_required":float(cfg["gates"]["max_gustavo_anchor_error"]),"observed_max_rhat":max(max(s["max_rhat"] for s in all_diag),float(rec.max_rhat.max())),"observed_min_bulk_ess":min(min(s["min_bulk_ess"] for s in all_diag),float(rec.min_bulk_ess.min())),"gustavo_anchor_error":slow.diagnostics["max_anchor_error"],"recovery_max_rhat":float(rec.max_rhat.max()),"recovery_min_bulk_ess":float(rec.min_bulk_ess.min())};gate["passed"]=bool(gate["observed_max_rhat"]<=rhat_limit and gate["observed_min_bulk_ess"]>=ess_limit and gate["gustavo_anchor_error"]<=gate["max_anchor_error_required"])
    data_path=data_root()/"processed"/"model_ready.csv";manifest={"revision":cfg["revision"],"profile":PROFILE,"not_for_inference":True,"inflation_observation":"annualized QoQ 400*Delta log P","expectation":"genuine SPF one-quarter-ahead annualized-log forecast","measurement_source":measurement_source,"created_utc":datetime.now(timezone.utc).isoformat(timespec="seconds"),"elapsed_seconds":time.time()-started,"seed":seed,"sampling":cfg["sampling"]["mock"],"git_commit":_git(["rev-parse","HEAD"]),"git_dirty":bool(_git(["status","--porcelain"])),"config_sha256":_sha(BUNDLE/"config.yaml"),"data_path":str(data_path),"data_sha256":_sha(data_path),"measurement_hashes":{"slow":_sha(out/"draws"/"slow"/"gustavo_slow.npz"),**{label:_sha(out/"draws"/"cycle"/f"{label}.npz") for label in labels}},"sample":{"gustavo":[str(g.index[0]),str(g.index[-1]),len(g)],"nkpc":{k:[str(v.periods[0]),str(v.periods[-1]),len(v.periods)] for k,v in cells.items()}},"gate":gate,"interpretation_rule":"QoQ free theta_CIQ recovery precedes any HSA restriction; YoY results are archived separately."};_json(out/"manifest.json",manifest);print(json.dumps(gate,indent=2),flush=True)
    if not args.no_report:
        from tests.gustavo_state_capitaliq_cycle.build_report import build
        build()


if __name__=="__main__":main()
