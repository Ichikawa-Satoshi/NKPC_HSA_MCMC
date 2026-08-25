"""Run competition-only state estimation and modular slope-NKPC mixtures."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pandas as pd

ROOT=next(p for p in Path(__file__).resolve().parents if (p/"pyproject.toml").exists())
sys.path[:0]=[str(ROOT),str(ROOT/"src"),str(ROOT/"tests")]
from tests import _bootstrap  # noqa:F401,E402
from nkpc_hsa.config import load_yaml  # noqa:E402
from nkpc_hsa.paths import data_root  # noqa:E402
from nkpc_hsa.report_models.cases import _load_frame  # noqa:E402
from tests.competition_slope_change.functions import (  # noqa:E402
    conditional_omega_likelihood,economic_quantities,fit_competition_state,
    fit_modular_nkpc,load_state,prepare_experiment,save_nkpc,save_state,
    summarize_nkpc,summarize_state,
)

BUNDLE=Path(__file__).resolve().parent


def _sha(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(1<<20),b""): h.update(block)
    return h.hexdigest()


def _git(args):
    r=subprocess.run(["git",*args],cwd=ROOT,text=True,stdout=subprocess.PIPE,stderr=subprocess.DEVNULL,check=False)
    return r.stdout.strip()


def _write_json(path: Path,payload):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(payload,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")


def _state_job(args):
    config_path,profile,law,label,prior,seed,sensitivity,out_path=args
    cfg=load_yaml(config_path); sampling=cfg["sampling"][profile]; experiment=prepare_experiment(cfg)
    fit=fit_competition_state(experiment,cfg,sampling,law=law,omega_prior=tuple(prior),
                              prior_label=label,seed=seed,sensitivity=sensitivity)
    save_state(out_path,fit); summary=summarize_state(fit,tuple(prior))
    _write_json(out_path.with_suffix(".json"),summary)
    return label,summary


def _nkpc_job(args):
    config_path,profile,state_path,state_summary_path,cell_name,model,timing,seed,out_path=args
    cfg=load_yaml(config_path); sampling=cfg["sampling"][profile]; experiment=prepare_experiment(cfg)
    state_summary=json.loads(Path(state_summary_path).read_text())
    states=load_state(Path(state_path),state_summary["diagnostics"]); cell=experiment.cells[cell_name]
    fit=fit_modular_nkpc(cell,states,cfg,sampling,model=model,timing=timing,seed=seed)
    save_nkpc(out_path,fit); summary=summarize_nkpc(fit)
    _write_json(out_path.with_suffix(".json"),summary)
    return summary,economic_quantities(fit,cell,cfg)


def _state_rows(summaries):
    rows=[]
    for label,s in summaries.items():
        for parameter in ("omega","tau","slow_innovation_variance","cycle_innovation_variance","damping_or_rho","cycle_period"):
            if parameter not in s: continue
            row={"variant":label,"law":s["law"],"parameter":parameter,**s[parameter]}
            row.update(max_rhat=s["diagnostics"]["max_rhat"],min_bulk_ess=s["diagnostics"]["min_bulk_ess"],
                       min_tail_ess=s["diagnostics"]["min_tail_ess"],
                       exact_identity_error=s["diagnostics"]["exact_identity_error"])
            rows.append(row)
    return rows


def _coefficient_rows(fits):
    rows=[]
    for summary in fits:
        for name,value in summary["coefficients"].items():
            rows.append({"cell":summary["cell"],"model":summary["model"],"timing":summary["timing"],
                         "sample_start":summary["sample"][0],"sample_end":summary["sample"][1],"n":summary["sample"][2],
                         "parameter":name,**value,"fit_max_rhat":summary["diagnostics"]["max_rhat"],
                         "fit_min_bulk_ess":summary["diagnostics"]["min_bulk_ess"],
                         "fit_min_tail_ess":summary["diagnostics"]["min_tail_ess"]})
    return rows


def _path_tables(out,experiment,state_path,state_summary,fits):
    states=load_state(state_path,state_summary["diagnostics"]); periods=np.array(states.periods)
    state=pd.DataFrame({"period":periods,"c_total_mean":states.n_total.mean((0,1)),
        "c_total_q2.5":np.percentile(states.n_total,2.5,axis=(0,1)),"c_total_q97.5":np.percentile(states.n_total,97.5,axis=(0,1)),
        "cbar_mean":states.nbar.mean((0,1)),"cbar_q2.5":np.percentile(states.nbar,2.5,axis=(0,1)),"cbar_q97.5":np.percentile(states.nbar,97.5,axis=(0,1)),
        "chat_mean":states.nhat.mean((0,1)),"chat_q2.5":np.percentile(states.nhat,2.5,axis=(0,1)),"chat_q97.5":np.percentile(states.nhat,97.5,axis=(0,1))})
    state.to_csv(out/"tables"/"competition_paths.csv",index=False)
    rows=[]
    for fit_info in fits:
        if fit_info["model"]!="slope_only": continue
        path=out/"draws"/"nkpc"/fit_info["cell"]/"slope_only_none.npz"
        z=np.load(path,allow_pickle=False); k=z["kappa"]; p=np.array(z["periods"],str)
        names=list(map(str,z["names"])); draws=z["draws"].reshape(-1,len(names)); bar=z["nbar"].reshape(-1,len(p))
        delta=draws[:,names.index("delta")]; k0=draws[:,names.index("kappa_0")]
        pp=pd.PeriodIndex(p,freq="Q"); r0=pd.Period(load_yaml(BUNDLE/"config.yaml")["sample"]["counterfactual_reference_start"],freq="Q")
        r1=pd.Period(load_yaml(BUNDLE/"config.yaml")["sample"]["counterfactual_reference_end"],freq="Q")
        cstar=bar[:,(pp>=r0)&(pp<=r1)].mean(1); kcf=k0+delta*cstar
        for j,period in enumerate(p):
            rows.append({"cell":fit_info["cell"],"period":period,"kappa_mean":float(k[:,:,j].mean()),
                         "kappa_q2.5":float(np.percentile(k[:,:,j],2.5)),"kappa_q97.5":float(np.percentile(k[:,:,j],97.5)),
                         "counterfactual_kappa_mean":float(kcf.mean()),"counterfactual_kappa_q2.5":float(np.percentile(kcf,2.5)),
                         "counterfactual_kappa_q97.5":float(np.percentile(kcf,97.5))})
    pd.DataFrame(rows).to_csv(out/"tables"/"kappa_paths.csv",index=False)


def _measurement_inputs(out, experiment, cfg):
    """Save the source series and constructed coordinate used by the report."""
    frame=_load_frame(); periods=experiment.allocation.periods
    start=pd.Period(cfg["sample"]["start"],freq="Q");end=pd.Period(cfg["sample"]["end"],freq="Q")
    mask=(periods>=start)&(periods<=end); periods=periods[mask]
    gustavo=pd.to_numeric(frame[cfg["data"]["annual_competition"]],errors="coerce").reindex(periods)
    capital_iq=pd.to_numeric(frame[cfg["data"]["quarterly_indicator"]],errors="coerce").reindex(periods)
    first_ciq=float(capital_iq.dropna().iloc[0])
    constructed=pd.Series(experiment.allocation_mean_raw-experiment.q0,index=experiment.allocation.periods).reindex(periods)
    coherence=pd.Series({pd.Period(f"{int(y)}Q4",freq="Q"):v for y,v in experiment.allocation.coherence.items()})
    table=pd.DataFrame({"period":periods.astype(str),"gustavo_level":gustavo.to_numpy(float),
        "capital_iq_level":capital_iq.to_numpy(float),
        "gustavo_coordinate":np.where(gustavo>0,10*np.log(gustavo)-experiment.q0,np.nan),
        "capital_iq_log_index":np.where(capital_iq>0,10*np.log(capital_iq/first_ciq),np.nan),
        "constructed_coordinate":constructed.to_numpy(float),
        "capital_iq_coverage":capital_iq.notna().to_numpy(int),
        "allocation_coherence":coherence.reindex(periods).to_numpy(float)})
    table.to_csv(out/"tables"/"measurement_inputs.csv",index=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile",choices=("smoke","full"),default="smoke")
    parser.add_argument("--workers",type=int,default=4)
    parser.add_argument("--reuse-state",action="store_true")
    parser.add_argument("--no-report",action="store_true")
    args=parser.parse_args(); started=time.time(); cfg=load_yaml(BUNDLE/"config.yaml")
    sampling=cfg["sampling"][args.profile]; out=BUNDLE/"results"/args.profile
    for folder in (out,out/"tables",out/"draws"/"state",out/"draws"/"nkpc"):
        folder.mkdir(parents=True,exist_ok=True)
    seed=int(cfg["sampling"]["seed"]); state_cfg=cfg["state"]
    variants=[("ar2_baseline","ar2",state_cfg["omega_prior"],False),
              ("ar1_baseline","ar1",state_cfg["omega_prior"],True),
              ("ar2_omega_uniform","ar2",state_cfg["omega_prior_sensitivity"]["uniform"],True),
              ("ar2_omega_balanced","ar2",state_cfg["omega_prior_sensitivity"]["balanced"],True)]
    state_summaries={}; state_jobs=[]
    for j,(label,law,prior,sensitivity) in enumerate(variants):
        path=out/"draws"/"state"/f"{label}.npz"
        if args.reuse_state and path.exists() and path.with_suffix(".json").exists():
            state_summaries[label]=json.loads(path.with_suffix(".json").read_text()); continue
        state_jobs.append((BUNDLE/"config.yaml",args.profile,law,label,prior,seed+100003*j,sensitivity,path))
    if state_jobs:
        with ProcessPoolExecutor(max_workers=min(args.workers,len(state_jobs))) as pool:
            futures={pool.submit(_state_job,job):job[3] for job in state_jobs}
            for future in as_completed(futures):
                label,summary=future.result(); state_summaries[label]=summary
                print(f"STATE {label}: Rhat={summary['diagnostics']['max_rhat']:.4f} omega={summary['omega']['mean']:.4f}",flush=True)
    baseline_path=out/"draws"/"state"/"ar2_baseline.npz"
    baseline_json=baseline_path.with_suffix(".json")
    baseline=load_state(baseline_path,state_summaries["ar2_baseline"]["diagnostics"])
    conditional_omega_likelihood(baseline,np.linspace(.005,.995,199)).to_csv(out/"tables"/"omega_conditional_likelihood.csv",index=False)

    experiment=prepare_experiment(cfg); jobs=[]; count=0
    for cell in [*cfg["nkpc"]["primary_cells"],*cfg["nkpc"]["robustness_cells"]]:
        path=out/"draws"/"nkpc"/cell/"slope_only_none.npz"; count+=1
        jobs.append((BUNDLE/"config.yaml",args.profile,baseline_path,baseline_json,cell,"slope_only","none",seed+1000009*count,path))
    for cell in cfg["nkpc"]["primary_cells"]:
        for timing in cfg["nkpc"]["direct_timing"]:
            path=out/"draws"/"nkpc"/cell/f"slope_plus_competition_cycle_{timing}.npz"; count+=1
            jobs.append((BUNDLE/"config.yaml",args.profile,baseline_path,baseline_json,cell,"slope_plus_competition_cycle",timing,seed+1000009*count,path))
    fit_summaries=[]; economic=[]
    with ProcessPoolExecutor(max_workers=min(args.workers,len(jobs))) as pool:
        futures={pool.submit(_nkpc_job,job):(job[4],job[5],job[6]) for job in jobs}
        for future in as_completed(futures):
            summary,quantities=future.result(); fit_summaries.append(summary); economic.extend(quantities)
            print(f"NKPC {summary['cell']} {summary['model']} {summary['timing']}: Rhat={summary['diagnostics']['max_rhat']:.4f}",flush=True)
    fit_summaries.sort(key=lambda x:(x["cell"],x["model"],x["timing"]))
    _write_json(out/"fit_summaries.json",fit_summaries)
    pd.DataFrame(_state_rows(state_summaries)).to_csv(out/"tables"/"state_identification.csv",index=False)
    coeff=pd.DataFrame(_coefficient_rows(fit_summaries)); coeff.to_csv(out/"tables"/"coefficients.csv",index=False)
    pd.DataFrame(economic).to_csv(out/"tables"/"economic_quantities.csv",index=False)
    coeff[[c for c in coeff.columns if c in {"cell","model","timing","parameter","prior_mean","prior_sd","mean","sd","q2.5","q97.5","posterior_prior_sd_ratio"}]].to_csv(out/"tables"/"prior_posterior.csv",index=False)
    convergence=[]
    for label,s in state_summaries.items():
        convergence.append({"block":"state","fit":label,**{k:s["diagnostics"][k] for k in ("max_rhat","min_bulk_ess","min_tail_ess","exact_identity_error")}})
    for s in fit_summaries:
        convergence.append({"block":"nkpc","fit":f"{s['cell']}/{s['model']}/{s['timing']}",
                            "max_rhat":s["diagnostics"]["max_rhat"],"min_bulk_ess":s["diagnostics"]["min_bulk_ess"],
                            "min_tail_ess":s["diagnostics"]["min_tail_ess"],"exact_identity_error":np.nan})
    pd.DataFrame(convergence).to_csv(out/"tables"/"convergence.csv",index=False)
    theta=coeff[coeff.parameter=="theta_C"].copy(); theta.to_csv(out/"tables"/"direct_timing.csv",index=False)
    input_rows=[{"item":"coordinate_reference_year","value":cfg["sample"]["coordinate_reference_year"]},
                {"item":"coordinate_reference_log_value","value":experiment.q0},
                {"item":"average_allocation_weights","value":json.dumps(experiment.allocation.average_weights.tolist())},
                {"item":"capital_iq_observed_allocation_years","value":len(experiment.allocation.raw_weights)},
                {"item":"annual_gustavo_years","value":len(experiment.allocation.annual)},
                {"item":"max_mean_anchor_error","value":experiment.allocation_summary["max_mean_path_anchor_error"]}]
    pd.DataFrame(input_rows).to_csv(out/"tables"/"input_audit.csv",index=False)
    _measurement_inputs(out,experiment,cfg)
    _path_tables(out,experiment,baseline_path,state_summaries["ar2_baseline"],fit_summaries)

    rhat_limit=float(cfg["gates"][f"{args.profile}_max_rhat"]); bulk_limit=float(cfg["gates"][f"{args.profile}_min_bulk_ess"])
    tail_limit=float(cfg["gates"].get(f"{args.profile}_min_tail_ess",0))
    primary_fits=[s for s in fit_summaries if s["model"]=="slope_only"]
    state_diag=state_summaries["ar2_baseline"]["diagnostics"]
    primary_max_rhat=max([state_diag["max_rhat"]]+[s["diagnostics"]["max_rhat"] for s in primary_fits])
    primary_min_bulk=min([state_diag["min_bulk_ess"]]+[s["diagnostics"]["min_bulk_ess"] for s in primary_fits])
    primary_min_tail=min([state_diag["min_tail_ess"]]+[s["diagnostics"]["min_tail_ess"] for s in primary_fits])
    gate={"max_rhat_required":rhat_limit,"min_bulk_ess_required":bulk_limit,"min_tail_ess_required":tail_limit,
          "max_exact_identity_error_required":float(cfg["gates"]["max_exact_identity_error"]),
          "primary_max_rhat":primary_max_rhat,"primary_min_bulk_ess":primary_min_bulk,"primary_min_tail_ess":primary_min_tail,
          "exact_identity_error":state_diag["exact_identity_error"],
          "passed":bool(primary_max_rhat<=rhat_limit and primary_min_bulk>=bulk_limit and primary_min_tail>=tail_limit and state_diag["exact_identity_error"]<=float(cfg["gates"]["max_exact_identity_error"]))}
    data_path=data_root()/"processed"/"model_ready.csv"
    manifest={"revision":cfg["revision"],"created_utc":datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "profile":args.profile,"smoke_not_for_inference":args.profile=="smoke","elapsed_seconds":time.time()-started,
              "sampling":sampling,"seed":seed,"git_commit":_git(["rev-parse","HEAD"]),"git_dirty":bool(_git(["status","--porcelain"])),
              "config_sha256":_sha(BUNDLE/"config.yaml"),"data_path":str(data_path),"data_sha256":_sha(data_path),
              "sample":{"start":cfg["sample"]["start"],"end":cfg["sample"]["end"],"cells":{k:[str(v.periods[0]),str(v.periods[-1]),v.n_periods] for k,v in experiment.cells.items()}},
              "state_variants":state_summaries,"fits":fit_summaries,"gate":gate,
              "interpretation_rule":"Convergence is separate from economic identification; fixed HSA restrictions are not estimated in this bundle."}
    _write_json(out/"manifest.json",manifest)
    print(json.dumps(gate,indent=2),flush=True)
    if not args.no_report:
        from tests.competition_slope_change.build_report import build
        build(args.profile)


if __name__=="__main__": main()
