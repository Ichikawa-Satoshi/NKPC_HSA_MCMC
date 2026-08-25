"""Exact-N joint QoQ NKPC runs with genuine one-quarter-ahead SPF forecasts."""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys

import numpy as np

ROOT=next(p for p in Path(__file__).resolve().parents if (p/"pyproject.toml").exists())
sys.path[:0]=[str(ROOT),str(ROOT/"src"),str(ROOT/"tests")]
from tests import _bootstrap  # noqa:F401,E402
from nkpc_hsa.config import load_yaml  # noqa:E402
from nkpc_hsa.phillips.data import load_design_data  # noqa:E402
from tests.hsa_deep_identification.joint_ma3 import fit_joint_ma3  # noqa:E402
from tests.hsa_deep_identification.run_joint import save,summary,specs  # noqa:E402
from tests.hsa_nested_validation.functions import CellData,load_experiment  # noqa:E402

BUNDLE=Path(__file__).resolve().parent; NESTED=ROOT/"tests"/"hsa_nested_validation"


def scale(x):
    x=np.asarray(x,float); value=float(np.subtract(*np.percentile(x,[75,25]))/1.349)
    return value if value>1e-8 else float(np.std(x))


def cells(experiment):
    q=load_design_data(include_qcew=False,sample_start="1982Q1",sample_end="2013Q4").quarterly
    out={}
    for price,column in {"ppi":"pi_ppi","core_cpi":"pi_core_cpi"}.items():
      for activity,xcol in {"negative_unemployment_gap":"x_negative_unemployment_gap","inverse_markup":"x_inverse_markup"}.items():
        periods=q.index; positions=experiment.allocation.periods.get_indexer(periods)
        if np.any(positions<0): raise ValueError("QoQ sample lies outside competition allocation")
        qref=experiment.allocation_mean_raw[positions]-experiment.q0
        out[f"{price}_{activity}"]=CellData(
            f"{price}_{activity}_qoq",f"{price}_{activity}_qoq",price,activity,periods,
            q[column].to_numpy(float),q[f"{column}_lag1"].to_numpy(float),q["expectation"].to_numpy(float),
            q[xcol].to_numpy(float),positions,scale(q[column]),scale(q[xcol]),scale(qref),
        )
    return out


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--profile",choices=("mock","quick","full"),default="mock")
    parser.add_argument("--architectures",nargs="+",default=["quarterly_local_level_ar2","annual_allocation_ar2"])
    parser.add_argument("--cells",nargs="+",default=["ppi_negative_unemployment_gap","ppi_inverse_markup","core_cpi_negative_unemployment_gap","core_cpi_inverse_markup"])
    parser.add_argument("--models",nargs="+",default=["ces","free","free_lambda","hsa6"]); args=parser.parse_args()
    cfg=deepcopy(load_yaml(BUNDLE/"config.yaml")); cfg["inflation"]["ma_order"]=0
    experiment=load_experiment(load_yaml(NESTED/"config.yaml")); cellmap=cells(experiment); modelmap=specs()
    root=BUNDLE/"results"/args.profile/"joint_qoq_iid"; manifest=[]
    for architecture in args.architectures:
      for cell_name in args.cells:
       for model_name in args.models:
        print("RUN",architecture,cell_name,model_name,flush=True)
        result=fit_joint_ma3(experiment,cellmap[cell_name],modelmap[model_name],cfg,cfg["sampling"][args.profile],architecture,20260825+1009*len(manifest))
        folder=root/architecture/cell_name; save(folder/f"{model_name}.npz",result); report=summary(result)
        folder.mkdir(parents=True,exist_ok=True); (folder/f"{model_name}.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
        item={"architecture":architecture,"cell":cell_name,"model":model_name,"max_rhat":report["diagnostics"]["max_rhat"],
              "theta":report["coefficients"].get("theta"),"kappa":report["kappa"]}; manifest.append(item)
        root.mkdir(parents=True,exist_ok=True); (root/"manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
        print(json.dumps(item,indent=2),flush=True)


if __name__=="__main__": main()
