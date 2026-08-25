"""Gustavo-only slow state and Capital-IQ-only cycle measurement functions."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import arviz as az
import numpy as np
import pandas as pd
from scipy.linalg import solve_discrete_lyapunov
from scipy.optimize import minimize
from scipy.special import expit,logit,logsumexp
from scipy.stats import beta as beta_dist,halfnorm,norm

from nkpc_hsa.dataprep.func_data_build import load_spf_cpi_quarter_ahead_expectations
from nkpc_hsa.error_robustness.ma_error import AdaptiveRandomWalk,MAWeighting
from nkpc_hsa.paths import data_root
from nkpc_hsa.phillips.data import load_design_data
from nkpc_hsa.report_models.cases import _load_frame
from tests.active_firm_stock_bds_bed.functions import ThetaCell,robust_scale
from tests.hsa_deep_identification.joint_ma3 import _draw_beta_gls,_slice_scalar


@dataclass
class SlowFit:
    periods: tuple[str,...]
    nbar: np.ndarray
    mu: np.ndarray
    sigma: np.ndarray
    diagnostics: dict[str,Any]
    anchors: dict[str,float]


@dataclass
class CycleFit:
    label: str
    periods: tuple[str,...]
    names: tuple[str,...]
    parameters: np.ndarray
    nbar_used: np.ndarray
    nhat: np.ndarray
    diagnostics: dict[str,Any]
    observed_coordinate: np.ndarray


@dataclass
class QoqFit:
    cycle: str
    cell: str
    error_model: str
    periods: tuple[str,...]
    names: tuple[str,...]
    draws: np.ndarray
    sigma_u: np.ndarray
    rho: np.ndarray
    nhat_used: np.ndarray
    nbar_used: np.ndarray
    state_chain: np.ndarray
    state_draw: np.ndarray
    prior_mean: dict[str,float]
    prior_sd: dict[str,float]
    diagnostics: dict[str,Any]


def ar2_coefficients(damping: float,period: float) -> tuple[float,float]:
    return float(2*damping*np.cos(2*np.pi/period)),float(-damping*damping)


def _draw_ig(rng: np.random.Generator,shape: float,scale: float) -> float:
    return float(1.0/rng.gamma(shape,1.0/scale))


def _diagnostics(series: dict[str,np.ndarray],paths: dict[str,np.ndarray] | None=None) -> dict[str,Any]:
    rhat={k:float(az.rhat(v,method="rank")) for k,v in series.items()}
    bulk={k:float(az.ess(v,method="bulk")) for k,v in series.items()}
    tail={k:float(az.ess(v,method="tail",prob=(.05,.95))) for k,v in series.items()}
    path_rhat={};path_bulk={};path_tail={}
    for name,value in (paths or {}).items():
        path_rhat[name]=float(np.nanmax(az.rhat(value,method="rank")))
        path_bulk[name]=float(np.nanmin(az.ess(value,method="bulk")))
        path_tail[name]=float(np.nanmin(az.ess(value,method="tail",prob=(.05,.95))))
    return {"rhat":rhat,"ess_bulk":bulk,"ess_tail":tail,"path_rhat":path_rhat,
        "path_ess_bulk":path_bulk,"path_ess_tail":path_tail,
        "max_rhat":max([*rhat.values(),*path_rhat.values()]),
        "min_bulk_ess":min([*bulk.values(),*path_bulk.values()]),
        "min_tail_ess":min([*tail.values(),*path_tail.values()])}


def load_measurements(config: dict) -> tuple[pd.PeriodIndex,pd.Series,dict[str,pd.Series]]:
    frame=_load_frame();dc=config["data"];sc=config["sample"]
    start=pd.Period(sc["state_start"],freq="Q");end=pd.Period(sc["state_end"],freq="Q")
    periods=pd.period_range(start,end,freq="Q")
    gustavo=pd.to_numeric(frame[dc["gustavo"]],errors="coerce").reindex(periods).dropna()
    if not np.all(gustavo.index.quarter==4):raise ValueError("Gustavo anchors must be Q4 observations")
    ref_year=int(sc["coordinate_reference_year"]);ref=float(gustavo[gustavo.index.year==ref_year].iloc[0])
    g=10*np.log(gustavo/ref)
    cstart=pd.Period(sc["cycle_start"],freq="Q");cend=pd.Period(sc["cycle_end"],freq="Q")
    cp=pd.period_range(cstart,cend,freq="Q");capital={}
    for label,column in dc["capital_iq"].items():
        raw=pd.to_numeric(frame[column],errors="coerce").reindex(cp)
        if raw.isna().any():raise ValueError(f"{label} Capital IQ series is incomplete over the cycle sample")
        ref=float(raw.loc[pd.Period(f"{ref_year}Q4",freq="Q")]);capital[label]=10*np.log(raw/ref)
    return periods,g,capital


def draw_bridge_path(rng: np.random.Generator,periods: pd.PeriodIndex,anchors: pd.Series,
                     sigma: float) -> np.ndarray:
    path=np.full(len(periods),np.nan);locations={p:periods.get_loc(p) for p in anchors.index}
    for p,value in anchors.items():path[locations[p]]=float(value)
    ordered=list(anchors.index)
    for left,right in zip(ordered[:-1],ordered[1:]):
        i0,i1=locations[left],locations[right]
        if i1-i0!=4:raise ValueError("Gustavo anchors must be consecutive annual Q4 observations")
        change=float(anchors.loc[right]-anchors.loc[left]);z=rng.normal(size=4);increments=change/4+sigma*(z-z.mean())
        path[i0+1:i1+1]=path[i0]+np.cumsum(increments)
    if np.isnan(path).any():raise ValueError("Slow-state bridge did not cover the requested period")
    return path


def fit_gustavo_slow(config: dict,seed: int) -> SlowFit:
    periods,anchors,_=load_measurements(config);cfg=config["slow_state"];sampling=config["sampling"]["mock"]
    d=np.diff(anchors.to_numpy(float));n=len(d);qscale=max(robust_scale(d)/2,1e-4)
    a0=float(cfg["variance_prior_shape"]);b0=(float(cfg["variance_prior_scale_multiple"])*qscale)**2
    prior_var=float(cfg["mu_prior_sd"])**2;chains=int(sampling["slow_chains"]);draws=int(sampling["slow_draws_per_chain"])
    mu=np.zeros((chains,draws));sigma=np.zeros_like(mu);nbar=np.zeros((chains,draws,len(periods)))
    for chain in range(chains):
        rng=np.random.default_rng(seed+1009*chain);m=float(np.mean(d)/4);s2=max(float(np.var(d,ddof=1))/4,qscale*qscale)
        for j in range(draws+200):
            precision=4*n/s2+1/prior_var;variance=1/precision
            mean=variance*(np.sum(d)/s2);m=float(rng.normal(mean,np.sqrt(variance)))
            s2=_draw_ig(rng,a0+n/2,b0+float(np.sum((d-4*m)**2))/8)
            if j>=200:
                k=j-200;mu[chain,k]=m;sigma[chain,k]=np.sqrt(s2)
                nbar[chain,k]=draw_bridge_path(rng,periods,anchors,np.sqrt(s2))
    diag=_diagnostics({"mu":mu,"sigma_bar":sigma},{"nbar_path":nbar})
    anchor_error=max(float(np.max(np.abs(nbar[:,:,periods.get_loc(p)]-value))) for p,value in anchors.items())
    diag["max_anchor_error"]=anchor_error
    return SlowFit(tuple(map(str,periods)),nbar,mu,sigma,diag,{str(k):float(v) for k,v in anchors.items()})


def _cycle_params(z: np.ndarray,sy: float,cfg: dict) -> dict[str,float]:
    lo_d,hi_d=map(float,cfg["damping_bounds"]);lo_p,hi_p=map(float,cfg["period_bounds"])
    return {"intercept":float(z[0]*sy),"loading":float(z[1]),
        "damping":float(lo_d+(hi_d-lo_d)*expit(z[2])),
        "period":float(lo_p+(hi_p-lo_p)*expit(z[3])),
        "sigma_cycle":float(sy*np.exp(z[4])),"sigma_measurement":float(sy*np.exp(z[5]))}


def _cycle_loglik_each(y: np.ndarray,bars: np.ndarray,p: dict[str,float]) -> np.ndarray:
    phi1,phi2=ar2_coefficients(p["damping"],p["period"]);F=np.array([[phi1,phi2],[1.,0.]])
    Q=np.diag([p["sigma_cycle"]**2,0.]);P=solve_discrete_lyapunov(F,Q);m=np.zeros((len(bars),2));ll=np.zeros(len(bars))
    residual=y[None,:]-p["intercept"]-p["loading"]*bars
    for t in range(len(y)):
        if t>0:m=m@F.T;P=F@P@F.T+Q
        variance=float(P[0,0]+p["sigma_measurement"]**2)
        if variance<=0 or not np.isfinite(variance):return np.full(len(bars),-np.inf)
        innovation=residual[:,t]-m[:,0];ll+=-.5*(np.log(2*np.pi*variance)+innovation**2/variance)
        gain=P[:,0]/variance;m+=innovation[:,None]*gain[None,:]
        P=P-np.outer(gain,P[0]);P=(P+P.T)/2
    return ll


def _cycle_logpost(z: np.ndarray,y: np.ndarray,bars: np.ndarray,sy: float,cfg: dict) -> float:
    p=_cycle_params(z,sy,cfg);lls=_cycle_loglik_each(y,bars,p)
    if not np.all(np.isfinite(lls)):return -np.inf
    out=float(logsumexp(lls)-np.log(len(lls)))
    out+=norm.logpdf(p["intercept"],0,float(cfg["intercept_prior_scale"])*sy)+np.log(sy)
    out+=norm.logpdf(p["loading"],0,float(cfg["loading_prior_sd"]))
    lo_d,hi_d=map(float,cfg["damping_bounds"]);ud=(p["damping"]-lo_d)/(hi_d-lo_d);aa,bb=map(float,cfg["damping_prior"])
    out+=beta_dist.logpdf(ud,aa,bb)+np.log(ud)+np.log1p(-ud)
    lo_p,hi_p=map(float,cfg["period_bounds"]);up=(p["period"]-lo_p)/(hi_p-lo_p)
    out+=norm.logpdf(p["period"],float(cfg["period_prior_mean"]),float(cfg["period_prior_sd"]))+np.log(up)+np.log1p(-up)
    out+=halfnorm.logpdf(p["sigma_cycle"],scale=float(cfg["cycle_sd_prior_scale"])*sy)+np.log(p["sigma_cycle"])
    out+=halfnorm.logpdf(p["sigma_measurement"],scale=float(cfg["measurement_sd_prior_scale"])*sy)+np.log(p["sigma_measurement"])
    return float(out)


def _draw_cycle_path(rng: np.random.Generator,y: np.ndarray,bar: np.ndarray,p: dict[str,float]) -> np.ndarray:
    phi1,phi2=ar2_coefficients(p["damping"],p["period"]);F=np.array([[phi1,phi2],[1.,0.]])
    Q=np.diag([p["sigma_cycle"]**2,0.]);T=len(y);m=np.zeros(2);P=solve_discrete_lyapunov(F,Q)
    mp=np.zeros((T,2));Pp=np.zeros((T,2,2));mf=np.zeros_like(mp);Pf=np.zeros_like(Pp)
    residual=y-p["intercept"]-p["loading"]*bar
    for t in range(T):
        if t>0:m=F@m;P=F@P@F.T+Q
        mp[t]=m;Pp[t]=P;variance=float(P[0,0]+p["sigma_measurement"]**2);innovation=residual[t]-m[0]
        gain=P[:,0]/variance;m=m+gain*innovation;P=P-np.outer(gain,P[0]);P=(P+P.T)/2
        mf[t]=m;Pf[t]=P
    x=np.zeros((T,2));x[-1]=rng.multivariate_normal(mf[-1],(Pf[-1]+Pf[-1].T)/2,check_valid="ignore")
    for t in range(T-2,-1,-1):
        J=Pf[t]@F.T@np.linalg.pinv(Pp[t+1]);mean=mf[t]+J@(x[t+1]-mp[t+1]);cov=Pf[t]-J@Pp[t+1]@J.T
        vals,vecs=np.linalg.eigh((cov+cov.T)/2);cov=vecs@np.diag(np.maximum(vals,1e-12))@vecs.T
        x[t]=rng.multivariate_normal(mean,cov,check_valid="ignore")
    return x[:,0]


def fit_capital_iq_cycle(label: str,slow: SlowFit,config: dict,seed: int) -> CycleFit:
    _,_,capital=load_measurements(config);series=capital[label];periods=series.index;y=series.to_numpy(float);sy=robust_scale(y)
    sp=pd.PeriodIndex(slow.periods,freq="Q");pos=sp.get_indexer(periods);flat=slow.nbar.reshape(-1,len(sp))[:,pos]
    k=min(int(config["capital_iq_cycle"]["integration_draws"]),len(flat));indices=np.linspace(0,len(flat)-1,k,dtype=int);bars=flat[indices]
    cfg=config["capital_iq_cycle"];sampling=config["sampling"]["mock"]
    meanbar=bars.mean(0);X=np.column_stack([np.ones(len(y)),meanbar]);ols=np.linalg.lstsq(X,y,rcond=None)[0];resid=y-X@ols
    lo_d,hi_d=map(float,cfg["damping_bounds"]);lo_p,hi_p=map(float,cfg["period_bounds"])
    init=np.array([ols[0]/sy,ols[1],logit((.65-lo_d)/(hi_d-lo_d)),logit((12-lo_p)/(hi_p-lo_p)),np.log(max(robust_scale(resid)*.6/sy,.05)),np.log(max(robust_scale(resid)*.6/sy,.05))])
    objective=lambda z:-_cycle_logpost(z,y,bars,sy,cfg) if np.isfinite(_cycle_logpost(z,y,bars,sy,cfg)) else 1e100
    opt=minimize(objective,init,method="BFGS",options={"maxiter":600,"gtol":1e-5});zmap=opt.x if np.isfinite(opt.fun) else init
    try:
        cov=np.asarray(opt.hess_inv,float);cov=(cov+cov.T)/2;v,q=np.linalg.eigh(cov);chol=np.linalg.cholesky(q@np.diag(np.maximum(v,1e-5))@q.T)
    except (ValueError,np.linalg.LinAlgError):chol=np.eye(6)
    iterations=int(sampling["cycle_iterations"]);warmup=int(sampling["cycle_warmup"]);thin=int(sampling["cycle_thin"]);chains=int(sampling["cycle_chains"]);ns=(iterations-warmup+thin-1)//thin
    names=("intercept","loading","damping","period","sigma_cycle","sigma_measurement")
    parameters=np.zeros((chains,ns,6));nbar=np.zeros((chains,ns,len(y)));nhat=np.zeros_like(nbar);accept=[]
    for chain in range(chains):
        rng=np.random.default_rng(seed+10007*chain);current=zmap+rng.normal(0,.03,6);lp=_cycle_logpost(current,y,bars,sy,cfg)
        if not np.isfinite(lp):current=zmap.copy();lp=_cycle_logpost(current,y,bars,sy,cfg)
        proposal=AdaptiveRandomWalk(6,init_scale=float(cfg["proposal_initial_scale"]),target_accept=.25);proposal._chol=chol.copy();save=0
        for it in range(iterations):
            candidate=proposal.propose(current,rng);clp=_cycle_logpost(candidate,y,bars,sy,cfg);accepted=np.log(rng.uniform())<clp-lp
            if accepted:current,lp=candidate,clp
            proposal.register(current,accepted)
            if it%int(cfg["slice_stride"])==0:
                for index in map(int,cfg["slice_indices"]):
                    def target(value,index=index):
                        z=current.copy();z[index]=value;return _cycle_logpost(z,y,bars,sy,cfg)
                    current[index]=_slice_scalar(rng,current[index],target,width=float(cfg["slice_width"]),max_steps=15);lp=_cycle_logpost(current,y,bars,sy,cfg)
            if it==warmup-1:proposal.freeze()
            if it>=warmup and (it-warmup)%thin==0:
                p=_cycle_params(current,sy,cfg);lls=_cycle_loglik_each(y,bars,p);w=np.exp(lls-logsumexp(lls));j=int(rng.choice(len(bars),p=w))
                parameters[chain,save]=[p[n] for n in names];nbar[chain,save]=bars[j];nhat[chain,save]=_draw_cycle_path(rng,y,bars[j],p);save+=1
        accept.append(proposal.acceptance_rate)
    diag=_diagnostics({n:parameters[:,:,j] for j,n in enumerate(names)},{"nhat_path":nhat});diag["acceptance"]=accept
    return CycleFit(label,tuple(map(str,periods)),names,parameters,nbar,nhat,diag,y)


def load_nkpc_cells(config: dict) -> dict[str,ThetaCell]:
    dc=config["data"];sc=config["sample"];idx=pd.period_range(sc["nkpc_start"],sc["nkpc_end"],freq="Q")
    frame=load_design_data(include_qcew=False,sample_start=sc["nkpc_start"],sample_end=sc["nkpc_end"]).quarterly;cells={}
    if any(pcfg.get("expectation") == "expectation_cpi" for pcfg in dc["prices"].values()):
        cpi_spf=load_spf_cpi_quarter_ahead_expectations(data_root()/"raw")
        cpi_spf.index=cpi_spf.index.to_period("Q")
        frame["expectation_cpi"]=pd.to_numeric(
            cpi_spf["Epi_spf_cpi_1q_ahead_ann_log"],errors="coerce"
        ).reindex(frame.index)
    for price,pcfg in dc["prices"].items():
        for activity,column in dc["activities"].items():
            table=pd.DataFrame({"pi":pd.to_numeric(frame[pcfg["inflation"]],errors="coerce"),"lag":pd.to_numeric(frame[pcfg["inflation_lag"]],errors="coerce"),"epi":pd.to_numeric(frame[pcfg["expectation"]],errors="coerce"),"x":pd.to_numeric(frame[column],errors="coerce")}).reindex(idx).dropna()
            name=f"{price}_{activity}";cells[name]=ThetaCell(name,table.index,table.pi.to_numpy(),table.lag.to_numpy(),table.epi.to_numpy(),table.x.to_numpy(),robust_scale(table.pi),robust_scale(table.x))
    return cells


def load_oil_controls(periods: pd.PeriodIndex) -> tuple[np.ndarray,dict[str,Any]]:
    """Load the prespecified current and lagged real-oil-price QoQ controls.

    The repository input ``WTISPLC_CPIAUCSL`` is the existing FRED-derived
    real WTI/CPI index.  Its arbitrary level normalization drops out of the
    log difference.  Controls are annualized QoQ log changes, matching the
    inflation transformation used by the QoQ NKPC.
    """
    path=data_root()/"raw"/"others"/"WTISPLC_CPIAUCSL.csv"
    raw=pd.read_csv(path);date_col=next((c for c in raw if c.lower() in {"date","observation_date"}),None)
    if date_col is None:raise ValueError("Oil input has no date column")
    values=pd.to_numeric(raw["WTISPLC_CPIAUCSL"],errors="coerce")
    index=pd.to_datetime(raw[date_col],errors="coerce").dt.to_period("Q")
    level=pd.Series(values.to_numpy(),index=pd.PeriodIndex(index,freq="Q")).sort_index()
    if level.index.has_duplicates:level=level.groupby(level.index).mean()
    qoq=400*np.log(level).diff()
    table=pd.DataFrame({"beta_oil_0":qoq,"beta_oil_1":qoq.shift(1)}).reindex(periods)
    if table.isna().any().any():raise ValueError("Oil controls do not cover the NKPC sample")
    controls=table.to_numpy(float)
    metadata={"source":str(path),"series":"WTISPLC_CPIAUCSL","transformation":"400_log_quarterly_difference","names":list(table.columns),"periods":[str(periods[0]),str(periods[-1]),len(periods)],"mean":table.mean().to_dict(),"sd":table.std(ddof=1).to_dict(),"correlation":float(table.corr().iloc[0,1])}
    return controls,metadata


def _qoq_diagnostics(names,draws,sigma,rho,error_model):
    series={name:draws[:,:,j] for j,name in enumerate(names)};series["sigma_u"]=sigma
    if error_model=="persistent_ar1":series["rho_error"]=rho
    return _diagnostics(series)


def _ar1_transform(y: np.ndarray,X: np.ndarray,rho: float) -> tuple[np.ndarray,np.ndarray]:
    scale=np.sqrt(max(1e-8,1-rho*rho));yt=np.r_[scale*y[0],y[1:]-rho*y[:-1]];Xt=np.vstack([scale*X[0],X[1:]-rho*X[:-1]])
    return yt,Xt


def build_qoq_design(cell: ThetaCell,hat: np.ndarray,bar: np.ndarray | None=None,
                     controls: np.ndarray | None=None) -> tuple[np.ndarray,np.ndarray | None]:
    """Build the direct-only or free-combined QoQ design matrix.

    The slow state is centered within each propagated measurement draw.  Thus
    kappa_0 is the NKPC slope at the sample-average slow competition state and
    delta is invariant to the arbitrary level origin of the Gustavo coordinate.
    """
    base=[np.ones(len(cell.periods)),cell.pi_lag,cell.epi,cell.x]
    if controls is not None:
        controls=np.asarray(controls,float)
        if controls.shape!=(len(cell.periods),2):raise ValueError("Oil controls must have shape (T, 2)")
        base.extend([controls[:,0],controls[:,1]])
    centered=None
    if bar is not None:
        centered=np.asarray(bar,float)-float(np.mean(bar))
        base.append(centered*cell.x)
    base.append(-np.asarray(hat,float))
    return np.column_stack(base),centered


def fit_qoq_theta(cell: ThetaCell,state: CycleFit,config: dict,seed: int,*,error_model: str,
                  recovery: bool=False,include_delta: bool=False,
                  sampling_override: dict[str,Any] | None=None,
                  controls: np.ndarray | None=None) -> QoqFit:
    if error_model not in {"iid","persistent_ar1"}:raise ValueError(error_model)
    sampling=config["sampling"]["mock"];sp=pd.PeriodIndex(state.periods,freq="Q");pos=sp.get_indexer(cell.periods)
    if np.any(pos<0):raise ValueError("Capital IQ cycle does not cover the QoQ sample")
    scale=float(config["nkpc"]["coefficient_scale"]);s_n=robust_scale(state.nhat[:,:,pos])
    names=["intercept","alpha_b","alpha_f","kappa_0"];pmean=[0.,.5,.5,0.];psd=[2*cell.s_pi,.5,.5,scale*cell.s_pi/cell.s_x]
    if controls is not None:
        controls=np.asarray(controls,float)
        if controls.shape!=(len(cell.periods),2):raise ValueError("Oil controls must have shape (T, 2)")
        control_scale=float(config["nkpc"].get("control_coefficient_scale",1.0))
        names.extend(["beta_oil_0","beta_oil_1"]);pmean.extend([0.,0.]);psd.extend([control_scale*cell.s_pi/robust_scale(controls[:,j]) for j in range(2)])
    if include_delta:
        bars=state.nbar_used[:,:,pos];bars=bars-bars.mean(axis=2,keepdims=True);s_delta=robust_scale(bars*cell.x[None,None,:])
        names.append("delta");pmean.append(0.);psd.append(scale*cell.s_pi/s_delta)
    names.append("theta_CIQ");pmean.append(0.);psd.append(scale*cell.s_pi/s_n)
    names=tuple(names);pmean=np.asarray(pmean,float);psd=np.asarray(psd,float)
    prefix="recovery_" if recovery else "nkpc_";iterations=int(sampling[prefix+"iterations"]);warmup=int(sampling[prefix+"warmup"]);thin=int(sampling[prefix+"thin"]);chains=int(sampling[prefix+"chains"]);ns=(iterations-warmup+thin-1)//thin
    if sampling_override is not None:
        iterations=int(sampling_override["iterations"]);warmup=int(sampling_override["warmup"]);thin=int(sampling_override["thin"]);chains=int(sampling_override["chains"]);ns=(iterations-warmup+thin-1)//thin
    draws=np.zeros((chains,ns,len(names)));sigma=np.zeros((chains,ns));rho_out=np.zeros((chains,ns));nhat=np.zeros((chains,ns,len(cell.periods)));nbar=np.zeros_like(nhat);state_chain=np.zeros((chains,ns),dtype=int);state_draw=np.zeros((chains,ns),dtype=int);ig_shape=3.;ig_scale=2*cell.s_pi**2;cfg=config["nkpc"]
    lo,hi=map(float,cfg["ar1_bounds"]);rho_mean=float(cfg["ar1_prior_mean"]);rho_sd=float(cfg["ar1_prior_sd"])
    for chain in range(chains):
        rng=np.random.default_rng(seed+65537*chain);beta=pmean.copy();sigma2=ig_scale/(ig_shape-1);rho=0.;save=0
        for it in range(iterations):
            cs=int(rng.integers(state.nhat.shape[0]));ds=int(rng.integers(state.nhat.shape[1]));hat=state.nhat[cs,ds,pos]
            bar=state.nbar_used[cs,ds,pos] if include_delta else None;X,_=build_qoq_design(cell,hat,bar,controls)
            yt,Xt=(cell.pi,X) if error_model=="iid" else _ar1_transform(cell.pi,X,rho);weight=MAWeighting(np.zeros(3),len(yt));beta=_draw_beta_gls(rng,yt,Xt,weight,sigma2,pmean,psd)
            residual=cell.pi-X@beta
            if error_model=="persistent_ar1":
                def target(value):
                    if value<=lo or value>=hi:return -np.inf
                    innovation=np.r_[np.sqrt(max(1e-8,1-value*value))*residual[0],residual[1:]-value*residual[:-1]]
                    return float(-.5*(innovation@innovation)/sigma2+.5*np.log(max(1e-8,1-value*value))+norm.logpdf(value,rho_mean,rho_sd))
                rho=float(_slice_scalar(rng,rho,target,width=float(cfg["ar1_slice_width"]),max_steps=20))
                innovation=np.r_[np.sqrt(max(1e-8,1-rho*rho))*residual[0],residual[1:]-rho*residual[:-1]]
            else:innovation=residual
            sigma2=_draw_ig(rng,ig_shape+len(cell.periods)/2,ig_scale+.5*float(innovation@innovation))
            if it>=warmup and (it-warmup)%thin==0:
                draws[chain,save]=beta;sigma[chain,save]=np.sqrt(sigma2);rho_out[chain,save]=rho;nhat[chain,save]=hat;nbar[chain,save]=state.nbar_used[cs,ds,pos];state_chain[chain,save]=cs;state_draw[chain,save]=ds;save+=1
    diag=_qoq_diagnostics(names,draws,sigma,rho_out,error_model)
    return QoqFit(state.label,cell.name,error_model,tuple(map(str,cell.periods)),names,draws,sigma,rho_out,nhat,nbar,state_chain,state_draw,dict(zip(names,pmean)),dict(zip(names,psd)),diag)


def summarize_qoq(fit: QoqFit) -> dict[str,Any]:
    def s(x):
        a=np.asarray(x).reshape(-1);return {"mean":float(a.mean()),"sd":float(a.std(ddof=1)),"q2.5":float(np.percentile(a,2.5)),"q50":float(np.percentile(a,50)),"q97.5":float(np.percentile(a,97.5)),"p_positive":float(np.mean(a>0))}
    out={"cycle":fit.cycle,"cell":fit.cell,"error_model":fit.error_model,"model":"free_combined" if "delta" in fit.names else "direct_only","sample":[fit.periods[0],fit.periods[-1],len(fit.periods)],"diagnostics":fit.diagnostics,"coefficients":{}}
    for j,name in enumerate(fit.names):
        v=s(fit.draws[:,:,j]);v.update(prior_mean=fit.prior_mean[name],prior_sd=fit.prior_sd[name],posterior_prior_sd_ratio=v["sd"]/fit.prior_sd[name],rhat=fit.diagnostics["rhat"][name],ess_bulk=fit.diagnostics["ess_bulk"][name],ess_tail=fit.diagnostics["ess_tail"][name]);out["coefficients"][name]=v
    out["sigma_u"]=s(fit.sigma_u)
    if fit.error_model=="persistent_ar1":out["rho_error"]=s(fit.rho)
    if "delta" in fit.names:
        delta=fit.draws[:,:,fit.names.index("delta")].reshape(-1);theta=fit.draws[:,:,fit.names.index("theta_CIQ")].reshape(-1)
        out["delta_theta_correlation"]=float(np.corrcoef(delta,theta)[0,1])
    return out


def simulate_qoq(rng: np.random.Generator,cell: ThetaCell,fit: QoqFit,state: CycleFit,theta: float) -> tuple[ThetaCell,np.ndarray]:
    means=fit.draws.reshape(-1,5).mean(0);sigma=float(fit.sigma_u.mean());rho=float(fit.rho.mean()) if fit.error_model=="persistent_ar1" else 0.;sp=pd.PeriodIndex(state.periods,freq="Q");pos=sp.get_indexer(cell.periods);cs=int(rng.integers(state.nhat.shape[0]));ds=int(rng.integers(state.nhat.shape[1]));hat=state.nhat[cs,ds,pos]
    u=rng.normal(0,sigma,len(cell.periods));eps=np.zeros(len(u));eps[0]=u[0]/np.sqrt(max(1e-8,1-rho*rho))
    for t in range(1,len(u)):eps[t]=rho*eps[t-1]+u[t]
    y=np.zeros(len(cell.periods));previous=float(cell.pi_lag[0]);a,ab,af,kappa,_=means
    for t in range(len(y)):y[t]=a+ab*previous+af*cell.epi[t]+kappa*cell.x[t]-theta*hat[t]+eps[t];previous=y[t]
    return cell.with_inflation(y,float(cell.pi_lag[0])),hat


def simulate_qoq_combined(rng: np.random.Generator,cell: ThetaCell,fit: QoqFit,state: CycleFit,
                          standardized_delta: float,standardized_theta: float,
                          controls: np.ndarray | None=None) -> tuple[ThetaCell,np.ndarray,np.ndarray,dict[str,float]]:
    """Simulate the free-combined model at standardized competition effects."""
    flat=fit.draws.reshape(-1,fit.draws.shape[-1]);means=flat.mean(0);coef=dict(zip(fit.names,means));sigma=float(fit.sigma_u.mean());rho=float(fit.rho.mean()) if fit.error_model=="persistent_ar1" else 0.
    sp=pd.PeriodIndex(state.periods,freq="Q");pos=sp.get_indexer(cell.periods);cs=int(rng.integers(state.nhat.shape[0]));ds=int(rng.integers(state.nhat.shape[1]));hat=state.nhat[cs,ds,pos];bar=state.nbar_used[cs,ds,pos];bar_c=bar-bar.mean()
    theta=float(standardized_theta*cell.s_pi/robust_scale(state.nhat[:,:,pos]));delta=float(standardized_delta*cell.s_pi/robust_scale((state.nbar_used[:,:,pos]-state.nbar_used[:,:,pos].mean(axis=2,keepdims=True))*cell.x[None,None,:]))
    u=rng.normal(0,sigma,len(cell.periods));eps=np.zeros(len(u));eps[0]=u[0]/np.sqrt(max(1e-8,1-rho*rho))
    for t in range(1,len(u)):eps[t]=rho*eps[t-1]+u[t]
    y=np.zeros(len(cell.periods));previous=float(cell.pi_lag[0])
    for t in range(len(y)):
        oil=0. if controls is None else coef.get("beta_oil_0",0.)*controls[t,0]+coef.get("beta_oil_1",0.)*controls[t,1]
        y[t]=coef["intercept"]+coef["alpha_b"]*previous+coef["alpha_f"]*cell.epi[t]+oil+(coef["kappa_0"]+delta*bar_c[t])*cell.x[t]-theta*hat[t]+eps[t];previous=y[t]
    return cell.with_inflation(y,float(cell.pi_lag[0])),hat,bar,{"delta":delta,"theta_CIQ":theta}


def qoq_pointwise_loglik(cell: ThetaCell,fit: QoqFit,controls: np.ndarray | None=None) -> np.ndarray:
    """Pointwise log likelihood on the original observation scale."""
    out=np.zeros((*fit.sigma_u.shape,len(cell.periods)))
    for chain in range(fit.draws.shape[0]):
        for draw in range(fit.draws.shape[1]):
            beta=fit.draws[chain,draw];names=fit.names;bar=fit.nbar_used[chain,draw] if "delta" in names else None;X,_=build_qoq_design(cell,fit.nhat_used[chain,draw],bar,controls);residual=cell.pi-X@beta;sigma=float(fit.sigma_u[chain,draw])
            if fit.error_model=="iid":out[chain,draw]=norm.logpdf(residual,0,sigma)
            else:
                rho=float(fit.rho[chain,draw]);out[chain,draw,0]=norm.logpdf(residual[0],0,sigma/np.sqrt(max(1e-8,1-rho*rho)));out[chain,draw,1:]=norm.logpdf(residual[1:],rho*residual[:-1],sigma)
    return out


def save_qoq(path: Path,fit: QoqFit):
    path.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(path,cycle=fit.cycle,cell=fit.cell,error_model=fit.error_model,periods=fit.periods,names=fit.names,draws=fit.draws,sigma_u=fit.sigma_u,rho=fit.rho,nhat_used=fit.nhat_used,nbar_used=fit.nbar_used,state_chain=fit.state_chain,state_draw=fit.state_draw,prior_mean=np.array([fit.prior_mean[n] for n in fit.names]),prior_sd=np.array([fit.prior_sd[n] for n in fit.names]))


def load_qoq(path: Path,diagnostics: dict) -> QoqFit:
    z=np.load(path,allow_pickle=False);names=tuple(map(str,z["names"]));shape=z["nhat_used"].shape;nbar=z["nbar_used"] if "nbar_used" in z.files else np.zeros(shape);sc=z["state_chain"] if "state_chain" in z.files else np.zeros(shape[:2],dtype=int);sd=z["state_draw"] if "state_draw" in z.files else np.zeros(shape[:2],dtype=int);return QoqFit(str(z["cycle"]),str(z["cell"]),str(z["error_model"]),tuple(map(str,z["periods"])),names,z["draws"],z["sigma_u"],z["rho"],z["nhat_used"],nbar,sc,sd,dict(zip(names,map(float,z["prior_mean"]))),dict(zip(names,map(float,z["prior_sd"]))),diagnostics)


def save_slow(path: Path,fit: SlowFit):
    path.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(path,periods=fit.periods,nbar=fit.nbar,mu=fit.mu,sigma=fit.sigma)


def load_slow(path: Path,meta: dict) -> SlowFit:
    z=np.load(path,allow_pickle=False);return SlowFit(tuple(map(str,z["periods"])),z["nbar"],z["mu"],z["sigma"],meta["diagnostics"],meta["anchors"])


def save_cycle(path: Path,fit: CycleFit):
    path.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(path,label=fit.label,periods=fit.periods,names=fit.names,parameters=fit.parameters,nbar_used=fit.nbar_used,nhat=fit.nhat,observed_coordinate=fit.observed_coordinate)


def load_cycle(path: Path,meta: dict) -> CycleFit:
    z=np.load(path,allow_pickle=False);return CycleFit(str(z["label"]),tuple(map(str,z["periods"])),tuple(map(str,z["names"])),z["parameters"],z["nbar_used"],z["nhat"],meta["diagnostics"],z["observed_coordinate"])


def summarize_slow(fit: SlowFit) -> dict[str,Any]:
    def s(x):return {"mean":float(np.mean(x)),"q2.5":float(np.percentile(x,2.5)),"q97.5":float(np.percentile(x,97.5))}
    return {"mu":s(fit.mu),"sigma_bar":s(fit.sigma),"diagnostics":fit.diagnostics,"anchors":fit.anchors}


def summarize_cycle(fit: CycleFit) -> dict[str,Any]:
    def s(x):return {"mean":float(np.mean(x)),"q2.5":float(np.percentile(x,2.5)),"q97.5":float(np.percentile(x,97.5))}
    return {"label":fit.label,"parameters":{n:s(fit.parameters[:,:,j]) for j,n in enumerate(fit.names)},"diagnostics":fit.diagnostics}
