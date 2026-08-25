"""Fit the QoQ free-combined slope and direct-channel diagnostic."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
from datetime import datetime,timezone
import hashlib,json,subprocess,sys,time
from pathlib import Path

import pandas as pd

ROOT=next(p for p in Path(__file__).resolve().parents if (p/"pyproject.toml").exists())
sys.path[:0]=[str(ROOT),str(ROOT/"src"),str(ROOT/"tests")]
from tests import _bootstrap  # noqa:F401,E402
from nkpc_hsa.config import load_yaml  # noqa:E402
from nkpc_hsa.paths import data_root  # noqa:E402
from tests.gustavo_state_capitaliq_cycle.functions import (  # noqa:E402
    fit_qoq_theta,load_cycle,load_nkpc_cells,save_qoq,summarize_qoq,
)

BUNDLE=Path(__file__).resolve().parent
DIRECT=BUNDLE/"results"/"mock_qoq"
OUT=BUNDLE/"results"/"free_combined_qoq"


def _json(path: Path,payload):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(payload,indent=2,ensure_ascii=False)+"\n")


def _sha(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda:handle.read(1<<20),b""):h.update(block)
    return h.hexdigest()


def _git(args):
    return subprocess.run(["git",*args],cwd=ROOT,text=True,stdout=subprocess.PIPE,stderr=subprocess.DEVNULL).stdout.strip()


def _job(args):
    label,error_model,cell_name,cycle_path,cycle_json,seed,sampling=args
    cfg=load_yaml(BUNDLE/"config.yaml");meta=json.loads(Path(cycle_json).read_text())
    cycle=load_cycle(Path(cycle_path),meta);cell=load_nkpc_cells(cfg)[cell_name]
    fit=fit_qoq_theta(cell,cycle,cfg,seed,error_model=error_model,include_delta=True,sampling_override=sampling)
    path=OUT/"draws"/label/error_model/f"{cell_name}.npz";save_qoq(path,fit)
    summary=summarize_qoq(fit);_json(path.with_suffix(".json"),summary);return summary


def _direct_summary(label,error_model,cell_name):
    path=DIRECT/"draws"/"nkpc"/label/error_model/f"{cell_name}.json"
    if not path.exists():raise FileNotFoundError(f"Run the direct-only QoQ mock first: {path}")
    return json.loads(path.read_text())


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--workers",type=int,default=4);parser.add_argument("--no-report",action="store_true");args=parser.parse_args()
    started=time.time();cfg=load_yaml(BUNDLE/"config.yaml");ccfg=load_yaml(BUNDLE/"combined_config.yaml");sampling=dict(ccfg["sampling"]);seed=int(cfg["sampling"]["seed"])+int(sampling.pop("seed_offset"))
    OUT.mkdir(parents=True,exist_ok=True);(OUT/"tables").mkdir(exist_ok=True);(OUT/"draws").mkdir(exist_ok=True)
    labels=list(cfg["data"]["capital_iq"]);errors=list(cfg["nkpc"]["error_models"]);cells=load_nkpc_cells(cfg);jobs=[]
    measurement_hashes={}
    for label in labels:
        cp=DIRECT/"draws"/"cycle"/f"{label}.npz";cj=cp.with_suffix(".json")
        if not cp.exists():raise FileNotFoundError(f"Missing frozen competition draw: {cp}")
        measurement_hashes[label]=_sha(cp)
    for j,(label,error_model,cell_name) in enumerate((l,e,c) for l in labels for e in errors for c in cells):
        cp=DIRECT/"draws"/"cycle"/f"{label}.npz";jobs.append((label,error_model,cell_name,cp,cp.with_suffix(".json"),seed+1000003*(j+1),sampling))
    fits=[]
    with ProcessPoolExecutor(max_workers=min(args.workers,len(jobs))) as pool:
        for future in as_completed([pool.submit(_job,job) for job in jobs]):
            result=future.result();fits.append(result);print(f"COMBINED {result['cycle']} {result['error_model']} {result['cell']} Rhat={result['diagnostics']['max_rhat']:.4f}",flush=True)
    fits.sort(key=lambda x:(x["cycle"],x["error_model"],x["cell"]))
    coefficient_rows=[];comparison_rows=[];gates=ccfg["gates"]
    for result in fits:
        for parameter,value in result["coefficients"].items():
            coefficient_rows.append({"cycle":result["cycle"],"error_model":result["error_model"],"cell":result["cell"],"parameter":parameter,**value,"fit_max_rhat":result["diagnostics"]["max_rhat"],"fit_min_bulk_ess":result["diagnostics"]["min_bulk_ess"]})
        direct=_direct_summary(result["cycle"],result["error_model"],result["cell"]);old=direct["coefficients"]["theta_CIQ"];theta=result["coefficients"]["theta_CIQ"];delta=result["coefficients"]["delta"];corr=float(result["delta_theta_correlation"])
        retained=bool(theta["p_positive"]>=float(gates["theta_persistence_probability_floor"]) and theta["posterior_prior_sd_ratio"]<=float(gates["theta_posterior_prior_sd_ratio_ceiling"]) and abs(corr)<float(gates["severe_delta_theta_correlation"]))
        comparison_rows.append({"cycle":result["cycle"],"error_model":result["error_model"],"cell":result["cell"],"direct_theta_mean":old["mean"],"direct_theta_q2.5":old["q2.5"],"direct_theta_q97.5":old["q97.5"],"direct_theta_p_positive":old["p_positive"],"direct_theta_sd_ratio":old["posterior_prior_sd_ratio"],"combined_theta_mean":theta["mean"],"combined_theta_q2.5":theta["q2.5"],"combined_theta_q97.5":theta["q97.5"],"combined_theta_p_positive":theta["p_positive"],"combined_theta_sd_ratio":theta["posterior_prior_sd_ratio"],"delta_mean":delta["mean"],"delta_q2.5":delta["q2.5"],"delta_q97.5":delta["q97.5"],"delta_p_positive":delta["p_positive"],"delta_sd_ratio":delta["posterior_prior_sd_ratio"],"delta_theta_correlation":corr,"theta_retained":retained,"max_rhat":result["diagnostics"]["max_rhat"],"min_bulk_ess":result["diagnostics"]["min_bulk_ess"]})
    coeff=pd.DataFrame(coefficient_rows);comparison=pd.DataFrame(comparison_rows);coeff.to_csv(OUT/"tables"/"coefficients.csv",index=False);comparison.to_csv(OUT/"tables"/"direct_vs_combined.csv",index=False)
    max_rhat=float(comparison.max_rhat.max());min_ess=float(comparison.min_bulk_ess.min());gate={"max_rhat_required":float(gates["max_rhat"]),"min_bulk_ess_required":float(gates["min_bulk_ess"]),"observed_max_rhat":max_rhat,"observed_min_bulk_ess":min_ess,"computational_pass":bool(max_rhat<=float(gates["max_rhat"]) and min_ess>=float(gates["min_bulk_ess"])),"theta_retained_cells":int(comparison.theta_retained.sum()),"theta_tested_cells":len(comparison)}
    data_path=data_root()/"processed"/"model_ready.csv";manifest={"revision":ccfg["revision"],"profile":"free_combined_qoq","not_for_inference":True,"created_utc":datetime.now(timezone.utc).isoformat(timespec="seconds"),"elapsed_seconds":time.time()-started,"seed":seed,"sampling":sampling,"git_commit":_git(["rev-parse","HEAD"]),"git_dirty":bool(_git(["status","--porcelain"])),"base_config_sha256":_sha(BUNDLE/"config.yaml"),"combined_config_sha256":_sha(BUNDLE/"combined_config.yaml"),"data_path":str(data_path),"data_sha256":_sha(data_path),"measurement_source":"byte-identical saved mock_qoq competition posterior draws","measurement_hashes":measurement_hashes,"sample":{k:[str(v.periods[0]),str(v.periods[-1]),len(v.periods)] for k,v in cells.items()},"gate":gate,"interpretation_rule":"This free-combined model tests coexistence and separability of slope and direct channels. It does not impose or test delta=lambda*theta."};_json(OUT/"manifest.json",manifest);print(json.dumps(gate,indent=2),flush=True)
    if not args.no_report:
        from tests.gustavo_state_capitaliq_cycle.build_combined_report import build
        build()


if __name__=="__main__":main()
