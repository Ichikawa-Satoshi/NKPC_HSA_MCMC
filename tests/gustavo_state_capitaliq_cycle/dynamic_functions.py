"""Cut-state QoQ varying-theta and HSA-dynamic samplers."""
from __future__ import annotations

from typing import Any
import numpy as np
import pandas as pd
from scipy.stats import norm

from nkpc_hsa.error_robustness.ma_error import MAWeighting
from tests.active_firm_stock_bds_bed.functions import ThetaCell,robust_scale
from tests.hsa_deep_identification.joint_ma3 import _draw_beta_gls,_slice_scalar
from tests.gustavo_state_capitaliq_cycle.functions import CycleFit,QoqFit,_ar1_transform,_draw_ig,_qoq_diagnostics

MODELS={"varying_theta","free_dynamic","hsa_restricted_dynamic"}


def centered_states(bar: np.ndarray) -> tuple[np.ndarray,np.ndarray]:
    barc=np.asarray(bar,float)-float(np.mean(bar));q2=barc**2-float(np.mean(barc**2));return barc,q2


def _scales(cell: ThetaCell,state: CycleFit,pos: np.ndarray) -> dict[str,float]:
    bars=state.nbar_used[:,:,pos];bars=bars-bars.mean(axis=2,keepdims=True);q2=bars**2-(bars**2).mean(axis=2,keepdims=True);hat=state.nhat[:,:,pos]
    return {"kappa_0":robust_scale(cell.x),"delta_1":robust_scale(bars*cell.x[None,None,:]),"delta_2":robust_scale(q2*cell.x[None,None,:]),"theta_0":robust_scale(hat),"gamma":robust_scale(bars*hat)}


def _prior(model: str,cell: ThetaCell,state: CycleFit,pos: np.ndarray,config: dict,
           controls: np.ndarray|None=None):
    s=_scales(cell,state,pos);scale=float(config["priors"]["coefficient_scale"]);common={"intercept":(0.,2*cell.s_pi),"alpha_b":(.5,.5),"alpha_f":(.5,.5),"kappa_0":(0.,scale*cell.s_pi/s["kappa_0"]),"delta_1":(0.,scale*cell.s_pi/s["delta_1"]),"delta_2":(0.,scale*cell.s_pi/s["delta_2"]),"theta_0":(0.,scale*cell.s_pi/s["theta_0"]),"gamma":(0.,scale*cell.s_pi/s["gamma"]),"lambda":(float(config["priors"]["lambda_mean"]),float(config["priors"]["lambda_sd"]))}
    names=["intercept","alpha_b","alpha_f","kappa_0"]
    if controls is not None:
        controls=np.asarray(controls,float)
        if controls.shape!=(len(cell.periods),2):raise ValueError("Oil controls must have shape (T, 2)")
        control_scale=float(config["priors"].get("control_coefficient_scale",1.0))
        for j,name in enumerate(("beta_oil_0","beta_oil_1")):common[name]=(0.,control_scale*cell.s_pi/robust_scale(controls[:,j]))
        names.extend(["beta_oil_0","beta_oil_1"])
    if model=="varying_theta":names.extend(["theta_0","gamma"])
    elif model=="free_dynamic":names.extend(["delta_1","delta_2","theta_0","gamma"])
    else:names.extend(["theta_0","gamma","lambda"])
    names=tuple(names)
    return names,np.array([common[n][0] for n in names]),np.array([common[n][1] for n in names])


def dynamic_design(model: str,cell: ThetaCell,hat: np.ndarray,bar: np.ndarray,lam: float=0.,
                   controls: np.ndarray|None=None) -> np.ndarray:
    barc,q2=centered_states(bar);base=[np.ones(len(cell.periods)),cell.pi_lag,cell.epi,cell.x]
    if controls is not None:
        controls=np.asarray(controls,float)
        if controls.shape!=(len(cell.periods),2):raise ValueError("Oil controls must have shape (T, 2)")
        base.extend([controls[:,0],controls[:,1]])
    if model=="varying_theta":base.extend([-hat,-barc*hat])
    elif model=="free_dynamic":base.extend([barc*cell.x,q2*cell.x,-hat,-barc*hat])
    elif model=="hsa_restricted_dynamic":base.extend([lam*barc*cell.x-hat,.5*lam*q2*cell.x-barc*hat])
    else:raise ValueError(model)
    return np.column_stack(base)


def fit_dynamic(cell: ThetaCell,state: CycleFit,base_config: dict,dynamic_config: dict,seed: int,*,model: str,error_model: str,sampling_override: dict[str,Any]|None=None,controls: np.ndarray|None=None) -> QoqFit:
    if model not in MODELS:raise ValueError(model)
    if error_model not in {"iid","persistent_ar1"}:raise ValueError(error_model)
    sp=pd.PeriodIndex(state.periods,freq="Q");pos=sp.get_indexer(cell.periods)
    if np.any(pos<0):raise ValueError("State does not cover dynamic sample")
    names,pmean,psd=_prior(model,cell,state,pos,dynamic_config,controls);sampling=dict(sampling_override or dynamic_config["sampling"]);iterations=int(sampling["iterations"]);warmup=int(sampling["warmup"]);thin=int(sampling["thin"]);chains=int(sampling["chains"]);ns=(iterations-warmup+thin-1)//thin
    draws=np.zeros((chains,ns,len(names)));sigma=np.zeros((chains,ns));rho_out=np.zeros((chains,ns));nhat=np.zeros((chains,ns,len(cell.periods)));nbar=np.zeros_like(nhat);sc_out=np.zeros((chains,ns),int);sd_out=np.zeros((chains,ns),int);ig_shape=3.;ig_scale=2*cell.s_pi**2;cfg=base_config["nkpc"];lo,hi=map(float,cfg["ar1_bounds"]);rho_mean=float(cfg["ar1_prior_mean"]);rho_sd=float(cfg["ar1_prior_sd"]);lambda_mean=float(dynamic_config["priors"]["lambda_mean"]);lambda_sd=float(dynamic_config["priors"]["lambda_sd"])
    beta_names=names[:-1] if model=="hsa_restricted_dynamic" else names;bpmean=pmean[:-1] if model=="hsa_restricted_dynamic" else pmean;bpsd=psd[:-1] if model=="hsa_restricted_dynamic" else psd
    for chain in range(chains):
        rng=np.random.default_rng(seed+65537*chain);beta=bpmean.copy();lam=lambda_mean;sigma2=ig_scale/(ig_shape-1);rho=0.;save=0
        for it in range(iterations):
            cs=int(rng.integers(state.nhat.shape[0]));ds=int(rng.integers(state.nhat.shape[1]));hat=state.nhat[cs,ds,pos];bar=state.nbar_used[cs,ds,pos];X=dynamic_design(model,cell,hat,bar,lam,controls);yt,Xt=(cell.pi,X) if error_model=="iid" else _ar1_transform(cell.pi,X,rho);beta=_draw_beta_gls(rng,yt,Xt,MAWeighting(np.zeros(3),len(yt)),sigma2,bpmean,bpsd)
            if model=="hsa_restricted_dynamic":
                b=dict(zip(beta_names,beta));barc,q2=centered_states(bar);oil=0. if controls is None else b["beta_oil_0"]*controls[:,0]+b["beta_oil_1"]*controls[:,1];base=b["intercept"]+b["alpha_b"]*cell.pi_lag+b["alpha_f"]*cell.epi+b["kappa_0"]*cell.x+oil-b["theta_0"]*hat-b["gamma"]*barc*hat;zeta=(b["theta_0"]*barc+.5*b["gamma"]*q2)*cell.x;yl=cell.pi-base;yl,zl=(yl,zeta[:,None]) if error_model=="iid" else _ar1_transform(yl,zeta[:,None],rho);z=zl[:,0];precision=1/lambda_sd**2+float(z@z)/sigma2;variance=1/precision;mean=variance*(lambda_mean/lambda_sd**2+float(z@yl)/sigma2);lam=float(rng.normal(mean,np.sqrt(variance)));X=dynamic_design(model,cell,hat,bar,lam,controls)
            residual=cell.pi-X@beta
            if error_model=="persistent_ar1":
                def target(value):
                    if value<=lo or value>=hi:return -np.inf
                    innovation=np.r_[np.sqrt(max(1e-8,1-value*value))*residual[0],residual[1:]-value*residual[:-1]]
                    return float(-.5*(innovation@innovation)/sigma2+.5*np.log(max(1e-8,1-value*value))+norm.logpdf(value,rho_mean,rho_sd))
                rho=float(_slice_scalar(rng,rho,target,width=float(cfg["ar1_slice_width"]),max_steps=20));innovation=np.r_[np.sqrt(max(1e-8,1-rho*rho))*residual[0],residual[1:]-rho*residual[:-1]]
            else:innovation=residual
            sigma2=_draw_ig(rng,ig_shape+len(cell.periods)/2,ig_scale+.5*float(innovation@innovation))
            if it>=warmup and (it-warmup)%thin==0:
                vector=np.r_[beta,lam] if model=="hsa_restricted_dynamic" else beta;draws[chain,save]=vector;sigma[chain,save]=np.sqrt(sigma2);rho_out[chain,save]=rho;nhat[chain,save]=hat;nbar[chain,save]=bar;sc_out[chain,save]=cs;sd_out[chain,save]=ds;save+=1
    diag=_qoq_diagnostics(names,draws,sigma,rho_out,error_model);fit=QoqFit(state.label,cell.name,error_model,tuple(map(str,cell.periods)),names,draws,sigma,rho_out,nhat,nbar,sc_out,sd_out,dict(zip(names,pmean)),dict(zip(names,psd)),diag);fit.diagnostics["dynamic_model"]=model;return fit


def dynamic_summary(fit: QoqFit,model: str) -> dict[str,Any]:
    def s(x):
        a=np.asarray(x).reshape(-1);return {"mean":float(a.mean()),"sd":float(a.std(ddof=1)),"q2.5":float(np.percentile(a,2.5)),"q50":float(np.percentile(a,50)),"q97.5":float(np.percentile(a,97.5)),"p_positive":float(np.mean(a>0))}
    out={"cycle":fit.cycle,"cell":fit.cell,"error_model":fit.error_model,"model":model,"sample":[fit.periods[0],fit.periods[-1],len(fit.periods)],"diagnostics":fit.diagnostics,"coefficients":{}}
    for j,name in enumerate(fit.names):
        z=s(fit.draws[:,:,j]);z.update(prior_mean=fit.prior_mean[name],prior_sd=fit.prior_sd[name],posterior_prior_sd_ratio=z["sd"]/fit.prior_sd[name],rhat=fit.diagnostics["rhat"][name],ess_bulk=fit.diagnostics["ess_bulk"][name],ess_tail=fit.diagnostics["ess_tail"][name]);out["coefficients"][name]=z
    out["sigma_u"]=s(fit.sigma_u)
    if fit.error_model=="persistent_ar1":out["rho_error"]=s(fit.rho)
    if model=="hsa_restricted_dynamic":
        flat=fit.draws.reshape(-1,len(fit.names));la=flat[:,fit.names.index("lambda")];th=flat[:,fit.names.index("theta_0")];ga=flat[:,fit.names.index("gamma")];out["derived"]={"delta_1":s(la*th),"delta_2":s(.5*la*ga)}
    return out


def dynamic_mu(cell: ThetaCell,fit: QoqFit,chain: int,draw: int,bar: np.ndarray|None=None,hat: np.ndarray|None=None,center: tuple[float,float]|None=None,controls: np.ndarray|None=None) -> np.ndarray:
    model=fit.diagnostics["dynamic_model"];beta=fit.draws[chain,draw];raw_bar=fit.nbar_used[chain,draw] if bar is None else np.asarray(bar);use_hat=fit.nhat_used[chain,draw] if hat is None else np.asarray(hat)
    if center is None:barc,q2=centered_states(raw_bar)
    else:barc=raw_bar-center[0];q2=barc**2-center[1]
    b=dict(zip(fit.names,beta));oil=0.
    if "beta_oil_0" in b:
        if controls is None:raise ValueError("Oil controls are required for an oil-control fit")
        oil=b["beta_oil_0"]*controls[:,0]+b["beta_oil_1"]*controls[:,1]
    mu=b["intercept"]+b["alpha_b"]*cell.pi_lag+b["alpha_f"]*cell.epi+b["kappa_0"]*cell.x+oil-(b["theta_0"]+b["gamma"]*barc)*use_hat
    if model=="free_dynamic":mu+=(b["delta_1"]*barc+b["delta_2"]*q2)*cell.x
    elif model=="hsa_restricted_dynamic":mu+=(b["lambda"]*b["theta_0"]*barc+.5*b["lambda"]*b["gamma"]*q2)*cell.x
    return mu


def dynamic_loglik(cell: ThetaCell,fit: QoqFit,controls: np.ndarray|None=None) -> np.ndarray:
    out=np.zeros((*fit.sigma_u.shape,len(cell.periods)))
    for c in range(fit.draws.shape[0]):
        for d in range(fit.draws.shape[1]):
            residual=cell.pi-dynamic_mu(cell,fit,c,d,controls=controls);sig=float(fit.sigma_u[c,d])
            if fit.error_model=="iid":out[c,d]=norm.logpdf(residual,0,sig)
            else:
                rho=float(fit.rho[c,d]);out[c,d,0]=norm.logpdf(residual[0],0,sig/np.sqrt(max(1e-8,1-rho*rho)));out[c,d,1:]=norm.logpdf(residual[1:],rho*residual[:-1],sig)
    return out


def simulate_varying_theta(rng: np.random.Generator,cell: ThetaCell,fit: QoqFit,state: CycleFit,s_theta: float,s_gamma: float,controls: np.ndarray|None=None):
    flat=fit.draws.reshape(-1,len(fit.names));b=dict(zip(fit.names,flat.mean(0)));sp=pd.PeriodIndex(state.periods,freq="Q");pos=sp.get_indexer(cell.periods);cs=int(rng.integers(state.nhat.shape[0]));ds=int(rng.integers(state.nhat.shape[1]));hat=state.nhat[cs,ds,pos];bar=state.nbar_used[cs,ds,pos];barc,_=centered_states(bar);theta=float(s_theta*cell.s_pi/robust_scale(state.nhat[:,:,pos]));gamma=float(s_gamma*cell.s_pi/robust_scale((state.nbar_used[:,:,pos]-state.nbar_used[:,:,pos].mean(axis=2,keepdims=True))*state.nhat[:,:,pos]));sig=float(fit.sigma_u.mean());rho=float(fit.rho.mean()) if fit.error_model=="persistent_ar1" else 0.;u=rng.normal(0,sig,len(cell.periods));eps=np.zeros(len(u));eps[0]=u[0]/np.sqrt(max(1e-8,1-rho*rho))
    for t in range(1,len(u)):eps[t]=rho*eps[t-1]+u[t]
    y=np.zeros(len(cell.periods));previous=float(cell.pi_lag[0])
    for t in range(len(y)):
        oil=0. if controls is None else b.get("beta_oil_0",0.)*controls[t,0]+b.get("beta_oil_1",0.)*controls[t,1]
        y[t]=b["intercept"]+b["alpha_b"]*previous+b["alpha_f"]*cell.epi[t]+b["kappa_0"]*cell.x[t]+oil-(theta+gamma*barc[t])*hat[t]+eps[t];previous=y[t]
    return cell.with_inflation(y,float(cell.pi_lag[0])),hat,bar,{"theta_0":theta,"gamma":gamma}
