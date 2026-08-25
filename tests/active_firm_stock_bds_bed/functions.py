"""External BDS-firm state, BED timing, and free-theta recovery functions."""
from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import arviz as az
import numpy as np
import pandas as pd
from scipy.linalg import solve_discrete_lyapunov
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import beta as beta_dist, halfnorm, norm

from nkpc_hsa.error_robustness.ma_error import AdaptiveRandomWalk, MAWeighting, PsiPrior, sample_psi
from nkpc_hsa.paths import data_root
from nkpc_hsa.phillips.state import _draw_ig
from nkpc_hsa.report_models.cases import _load_frame
from tests.hsa_deep_identification.joint_ma3 import _draw_beta_gls,_slice_scalar


@dataclass(frozen=True)
class ThetaCell:
    name: str
    periods: pd.PeriodIndex
    pi: np.ndarray
    pi_lag: np.ndarray
    epi: np.ndarray
    x: np.ndarray
    s_pi: float
    s_x: float

    def with_inflation(self, pi: np.ndarray, initial_lag: float) -> "ThetaCell":
        lag=np.r_[initial_lag,np.asarray(pi[:-1],float)]
        return replace(self,pi=np.asarray(pi,float),pi_lag=lag,s_pi=robust_scale(pi))


@dataclass(frozen=True)
class ExternalData:
    periods: pd.PeriodIndex
    bds_coordinate: np.ndarray
    bds_firms: np.ndarray
    bed_births: np.ndarray
    bed_deaths: np.ndarray
    bed_net_standardized: np.ndarray
    bed_observed: np.ndarray
    reference_firms: float
    cells: dict[str,ThetaCell]


@dataclass
class FirmStateFit:
    periods: tuple[str,...]
    names: tuple[str,...]
    parameters: np.ndarray
    nbar: np.ndarray
    nhat: np.ndarray
    n_total: np.ndarray
    diagnostics: dict[str,Any]
    map_parameters: dict[str,float]
    acceptance: list[float]


@dataclass
class ThetaFit:
    cell: str
    periods: tuple[str,...]
    names: tuple[str,...]
    draws: np.ndarray
    sigma_u: np.ndarray
    psi: np.ndarray
    nhat_used: np.ndarray
    prior_mean: dict[str,float]
    prior_sd: dict[str,float]
    diagnostics: dict[str,Any]


def robust_scale(values) -> float:
    x=np.asarray(values,float);x=x[np.isfinite(x)]
    if len(x)==0:return 1.0
    med=np.median(x);mad=1.4826*np.median(np.abs(x-med))
    sd=np.std(x,ddof=1) if len(x)>1 else 0.0
    return float(max(mad,sd,1e-6))


def _count(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.astype(str).str.replace(",","",regex=False).str.replace("−","-",regex=False),errors="coerce")


def _bed(path: Path, name: str) -> pd.Series:
    d=pd.read_csv(path);d.columns=["period",name]
    d.index=pd.PeriodIndex(d.period.astype(str).str.replace("-","",regex=False),freq="Q")
    return pd.to_numeric(d[name],errors="coerce")*1000.0


def load_external_data(config: dict) -> ExternalData:
    root=data_root();dc=config["data"];sc=config["sample"]
    bds=pd.read_csv(root/dc["bds_file"]);years=pd.to_numeric(bds[dc["bds_year"]],errors="coerce")
    firms=_count(bds[dc["bds_firms"]]);annual=pd.Series(firms.to_numpy(float),index=years).dropna()
    annual.index=annual.index.astype(int);ref=float(annual.loc[int(sc["reference_year"])])
    periods=pd.period_range(sc["state_start"],sc["state_end"],freq="Q")
    bds_level=np.full(len(periods),np.nan);bds_coordinate=np.full(len(periods),np.nan)
    for year,value in annual.items():
        p=pd.Period(f"{year}Q1",freq="Q")
        if p in periods:
            j=periods.get_loc(p);bds_level[j]=value;bds_coordinate[j]=10*np.log(value/ref)
    births=_bed(root/dc["bed_births_file"],"births").reindex(periods)
    deaths=_bed(root/dc["bed_deaths_file"],"deaths").reindex(periods)
    observed=(births.notna()&deaths.notna()).to_numpy()
    net=(births-deaths);z=np.zeros(len(periods));z[observed]=(net[observed]-net[observed].mean())/robust_scale(net[observed])
    frame=_load_frame();cells={}
    start,end=pd.Period(sc["nkpc_start"],freq="Q"),pd.Period(sc["nkpc_end"],freq="Q")
    for price,pcfg in dc["prices"].items():
        for activity,acol in dc["activities"].items():
            idx=pd.period_range(start,end,freq="Q")
            table=pd.DataFrame({"pi":pd.to_numeric(frame[pcfg["inflation"]],errors="coerce"),
                "lag":pd.to_numeric(frame[pcfg["inflation_lag"]],errors="coerce"),
                "epi":pd.to_numeric(frame[pcfg["expectation"]],errors="coerce"),
                "x":pd.to_numeric(frame[acol],errors="coerce")}).reindex(idx).dropna()
            if len(table)!=len(idx):
                first,last=table.index.min(),table.index.max();table=table.reindex(pd.period_range(first,last,freq="Q")).dropna()
            name=f"{price}_{activity}";cells[name]=ThetaCell(name,table.index,table.pi.to_numpy(),table.lag.to_numpy(),
                table.epi.to_numpy(),table.x.to_numpy(),robust_scale(table.pi),robust_scale(table.x))
    return ExternalData(periods,bds_coordinate,bds_level,births.to_numpy(float),deaths.to_numpy(float),z,observed,ref,cells)


def ar2_coefficients(damping: float, period: float) -> tuple[float,float]:
    return float(2*damping*np.cos(2*np.pi/period)),float(-damping*damping)


def transform_parameters(z: np.ndarray, s_d: float, s_y: float, state_cfg: dict) -> dict[str,float]:
    z=np.asarray(z,float);lo_d,hi_d=map(float,state_cfg["damping_bounds"]);lo_p,hi_p=map(float,state_cfg["period_bounds"])
    return {"mu":float(z[0]*s_d),"tau":float(s_d*np.exp(z[1])),
        "omega":float(expit(z[2])),"damping":float(lo_d+(hi_d-lo_d)*expit(z[3])),
        "period":float(lo_p+(hi_p-lo_p)*expit(z[4])),"sigma_f":float(state_cfg["bds_error_fixed"]),
        "bed_intercept":float(z[5]),"bed_loading":float(z[6]/s_d),"sigma_bed":float(np.exp(z[7]))}


def _transition(p: dict[str,float]):
    phi1,phi2=ar2_coefficients(p["damping"],p["period"])
    F=np.array([[1.,0.,0.,0.],[0.,phi1,phi2,0.],[0.,1.,0.,0.],[1.,1.,0.,0.]])
    Q=np.diag([p["omega"]*p["tau"]**2,(1-p["omega"])*p["tau"]**2,0.,0.])
    return F,Q


def _initial_state(y: np.ndarray,p: dict[str,float]):
    first=float(y[np.isfinite(y)][0]);phi1,phi2=ar2_coefficients(p["damping"],p["period"])
    A=np.array([[phi1,phi2],[1.,0.]])
    cyc=solve_discrete_lyapunov(A,np.diag([(1-p["omega"])*p["tau"]**2,0.]))
    m=np.array([first,0.,0.,first]);P=np.zeros((4,4));P[0,0]=max(p["sigma_f"]**2,p["tau"]**2);P[1:3,1:3]=cyc;P[3,3]=P[0,0]+cyc[0,0]
    return m,P


def kalman_filter(y: np.ndarray,zbed: np.ndarray,bed_observed: np.ndarray,p: dict[str,float],store: bool=False):
    T=len(y);F,Q=_transition(p);Hf=np.array([1.,1.,0.,0.]);Hb=p["bed_loading"]*np.array([1.,1.,0.,-1.]);m,P=_initial_state(y,p);ll=0.0
    mp=np.zeros((T,4));Pp=np.zeros((T,4,4));mf=np.zeros_like(mp);Pf=np.zeros_like(Pp)
    for t in range(T):
        if t>0:
            m=F@m+np.array([p["mu"],0.,0.,0.]);P=F@P@F.T+Q
        mp[t]=m;Pp[t]=P
        if np.isfinite(y[t]):
            variance=float(Hf@P@Hf+p["sigma_f"]**2)
            if variance<=0 or not np.isfinite(variance):return (-np.inf,None) if store else -np.inf
            innovation=float(y[t]-Hf@m);ll+=-.5*(np.log(2*np.pi*variance)+innovation**2/variance)
            K=P@Hf/variance;m=m+K*innovation;P=P-np.outer(K,Hf@P);P=(P+P.T)/2
        if bed_observed[t] and t>0:
            variance=float(Hb@P@Hb+p["sigma_bed"]**2)
            if variance<=0 or not np.isfinite(variance):return (-np.inf,None) if store else -np.inf
            innovation=float(zbed[t]-p["bed_intercept"]-Hb@m);ll+=-.5*(np.log(2*np.pi*variance)+innovation**2/variance)
            K=P@Hb/variance;m=m+K*innovation;P=P-np.outer(K,Hb@P);P=(P+P.T)/2
        mf[t]=m;Pf[t]=P
    if store:return ll,(mp,Pp,mf,Pf,F)
    return ll


def _log_posterior(z,data: ExternalData,s_d: float,s_y: float,cfg: dict) -> float:
    p=transform_parameters(z,s_d,s_y,cfg);ll=kalman_filter(data.bds_coordinate,data.bed_net_standardized,data.bed_observed,p)
    if not np.isfinite(ll):return -np.inf
    out=ll+norm.logpdf(p["mu"],0,float(cfg["mu_prior_sd_multiple"])*s_d)
    out+=halfnorm.logpdf(p["tau"],scale=float(cfg["tau_prior_scale_multiple"])*s_d)+np.log(p["tau"])
    o=p["omega"];a,b=map(float,cfg["omega_prior"]);out+=beta_dist.logpdf(o,a,b)+np.log(o)+np.log1p(-o)
    ud=(p["damping"]-cfg["damping_bounds"][0])/(cfg["damping_bounds"][1]-cfg["damping_bounds"][0])
    a,b=map(float,cfg["damping_prior"]);out+=beta_dist.logpdf(ud,a,b)+np.log(ud)+np.log1p(-ud)
    up=(p["period"]-cfg["period_bounds"][0])/(cfg["period_bounds"][1]-cfg["period_bounds"][0])
    out+=norm.logpdf(p["period"],cfg["period_prior_mean"],cfg["period_prior_sd"])+np.log(up)+np.log1p(-up)
    out+=norm.logpdf(p["bed_intercept"],0,float(cfg["bed_intercept_prior_sd"]))
    out+=norm.logpdf(z[6],0,float(cfg["bed_loading_scaled_prior_sd"]))
    out+=halfnorm.logpdf(p["sigma_bed"],scale=float(cfg["bed_error_prior_scale"]))+np.log(p["sigma_bed"])
    return float(out)


def _mvn(rng,mean,cov):
    cov=(cov+cov.T)/2;vals,vecs=np.linalg.eigh(cov);vals=np.maximum(vals,1e-12)
    return mean+vecs@(np.sqrt(vals)*rng.normal(size=len(mean)))


def draw_state_path(rng,data: ExternalData,p: dict[str,float]) -> np.ndarray:
    _,stored=kalman_filter(data.bds_coordinate,data.bed_net_standardized,data.bed_observed,p,store=True)
    mp,Pp,mf,Pf,F=stored;T=len(data.periods);x=np.zeros((T,4));x[-1]=_mvn(rng,mf[-1],Pf[-1])
    for t in range(T-2,-1,-1):
        inv=np.linalg.pinv(Pp[t+1]);J=Pf[t]@F.T@inv
        mean=mf[t]+J@(x[t+1]-mp[t+1]);cov=Pf[t]-J@Pp[t+1]@J.T
        x[t]=_mvn(rng,mean,cov)
    return x


def _state_diagnostics(names,parameters,nbar,nhat):
    series={name:parameters[:,:,j] for j,name in enumerate(names)}
    rhat={k:float(az.rhat(v,method="rank")) for k,v in series.items()};bulk={k:float(az.ess(v,method="bulk")) for k,v in series.items()};tail={k:float(az.ess(v,method="tail",prob=(.05,.95))) for k,v in series.items()}
    path_rhat=max(float(np.nanmax(az.rhat(nbar,method="rank"))),float(np.nanmax(az.rhat(nhat,method="rank"))))
    path_bulk=min(float(np.nanmin(az.ess(nbar,method="bulk"))),float(np.nanmin(az.ess(nhat,method="bulk"))))
    return {"rhat":rhat,"ess_bulk":bulk,"ess_tail":tail,"max_rhat":max(max(rhat.values()),path_rhat),
        "min_bulk_ess":min(min(bulk.values()),path_bulk),"min_tail_ess":min(tail.values()),"path_max_rhat":path_rhat,"path_min_bulk_ess":path_bulk}


def fit_firm_state(data: ExternalData,config: dict,sampling: dict,seed: int) -> FirmStateFit:
    cfg=config["state"];obs=data.bds_coordinate[np.isfinite(data.bds_coordinate)];s_y=robust_scale(np.diff(obs));s_d=max(s_y/2,1e-3)
    init=np.array([np.mean(np.diff(obs))/4/s_d,np.log(.8),-2.,0.,0.,0.,1.,np.log(.7)])
    objective=lambda z:-_log_posterior(z,data,s_d,s_y,cfg) if np.isfinite(_log_posterior(z,data,s_d,s_y,cfg)) else 1e100
    opt=minimize(objective,init,method="BFGS",options={"maxiter":800,"gtol":1e-6});zmap=opt.x if np.isfinite(opt.fun) else init
    try:
        proposal_cov=np.asarray(opt.hess_inv,float);proposal_cov=(proposal_cov+proposal_cov.T)/2
        values,vectors=np.linalg.eigh(proposal_cov);proposal_cov=vectors@np.diag(np.maximum(values,1e-6))@vectors.T
        proposal_chol=np.linalg.cholesky(proposal_cov)
    except (ValueError,np.linalg.LinAlgError):
        proposal_chol=np.eye(8)
    iterations=int(sampling["state_iterations"]);warmup=int(sampling["state_warmup"]);thin=int(sampling["state_thin"]);chains=int(sampling["state_chains"])
    nsave=(iterations-warmup+thin-1)//thin;names=("mu","tau","omega","damping","period","bed_intercept","bed_loading","sigma_bed")
    parameters=np.zeros((chains,nsave,len(names)));nbar=np.zeros((chains,nsave,len(data.periods)));nhat=np.zeros_like(nbar);accept=[]
    for chain in range(chains):
        rng=np.random.default_rng(seed+104729*chain);current=zmap+rng.normal(0,.04,size=8);lp=_log_posterior(current,data,s_d,s_y,cfg)
        if not np.isfinite(lp):current=zmap.copy();lp=_log_posterior(current,data,s_d,s_y,cfg)
        proposal=AdaptiveRandomWalk(8,init_scale=float(cfg["proposal_initial_scale"]),target_accept=.25);proposal._chol=proposal_chol.copy();save=0
        for it in range(iterations):
            candidate=proposal.propose(current,rng);cand_lp=_log_posterior(candidate,data,s_d,s_y,cfg);accepted=np.log(rng.uniform())<cand_lp-lp
            if accepted:current,lp=candidate,cand_lp
            proposal.register(current,accepted)
            if it%int(cfg.get("slice_stride",1))==0:
                for index in map(int,cfg.get("slice_indices",())):
                    def target(value,index=index):
                        candidate=current.copy();candidate[index]=value;return _log_posterior(candidate,data,s_d,s_y,cfg)
                    current[index]=_slice_scalar(rng,current[index],target,width=float(cfg.get("slice_width",.45)),max_steps=20)
                    lp=_log_posterior(current,data,s_d,s_y,cfg)
            if it==warmup-1:proposal.freeze()
            if it>=warmup and (it-warmup)%thin==0:
                p=transform_parameters(current,s_d,s_y,cfg);parameters[chain,save]=[p[n] for n in names]
                state=draw_state_path(rng,data,p);nbar[chain,save]=state[:,0];nhat[chain,save]=state[:,1];save+=1
        accept.append(proposal.acceptance_rate)
    diagnostics=_state_diagnostics(names,parameters,nbar,nhat);pmap=transform_parameters(zmap,s_d,s_y,cfg)
    return FirmStateFit(tuple(map(str,data.periods)),names,parameters,nbar,nhat,nbar+nhat,diagnostics,pmap,accept)


def _theta_diagnostics(names,draws,sigma,psi):
    series={name:draws[:,:,j] for j,name in enumerate(names)};series["sigma_u"]=sigma
    series.update({f"psi_{j+1}":psi[:,:,j] for j in range(3)})
    rhat={k:float(az.rhat(v,method="rank")) for k,v in series.items()};bulk={k:float(az.ess(v,method="bulk")) for k,v in series.items()};tail={k:float(az.ess(v,method="tail",prob=(.05,.95))) for k,v in series.items()}
    return {"rhat":rhat,"ess_bulk":bulk,"ess_tail":tail,"max_rhat":max(rhat.values()),"min_bulk_ess":min(bulk.values()),"min_tail_ess":min(tail.values())}


def fit_theta(cell: ThetaCell,state: FirmStateFit,config: dict,sampling: dict,seed: int,*,recovery: bool=False) -> ThetaFit:
    sp=pd.PeriodIndex(state.periods,freq="Q");pos=sp.get_indexer(cell.periods)
    if np.any(pos<0):raise ValueError("Firm-state dates do not cover the NKPC sample")
    names=("intercept","alpha_b","alpha_f","kappa_0","theta_N");s_n=robust_scale(state.nhat[:,:,pos])
    scale=float(config["nkpc"]["coefficient_scale"]);pmean=np.array([0.,.5,.5,0.,0.]);psd=np.array([2*cell.s_pi,.5,.5,scale*cell.s_pi/cell.s_x,scale*cell.s_pi/s_n])
    iterations=int(sampling["recovery_iterations"] if recovery else sampling["nkpc_iterations"]);warmup=int(sampling["recovery_warmup"] if recovery else sampling["nkpc_warmup"]);thin=int(sampling["recovery_thin"] if recovery else sampling["nkpc_thin"]);chains=int(sampling["recovery_chains"] if recovery else sampling["nkpc_chains"])
    nsave=(iterations-warmup+thin-1)//thin;draws=np.zeros((chains,nsave,5));sigma=np.zeros((chains,nsave));psi_out=np.zeros((chains,nsave,3));nhat_out=np.zeros((chains,nsave,len(cell.periods)))
    ig_shape=3.;ig_scale=2*cell.s_pi**2;inf=config["nkpc"];psi_prior=PsiPrior(np.full(3,float(inf["psi_prior_mean"])),np.full(3,float(inf["psi_prior_sd"])))
    for chain in range(chains):
        rng=np.random.default_rng(seed+65537*chain);beta=pmean.copy();sigma2=ig_scale/(ig_shape-1);psi=np.zeros(3);weight=MAWeighting(psi,len(cell.periods));proposal=AdaptiveRandomWalk(3,init_scale=float(inf["psi_initial_scale"]));save=0
        for it in range(iterations):
            cs=int(rng.integers(state.nhat.shape[0]));ds=int(rng.integers(state.nhat.shape[1]));hat=state.nhat[cs,ds,pos]
            X=np.column_stack([np.ones(len(cell.periods)),cell.pi_lag,cell.epi,cell.x,-hat])
            beta=_draw_beta_gls(rng,cell.pi,X,weight,sigma2,pmean,psd);resid=cell.pi-X@beta
            sigma2=_draw_ig(rng,ig_shape+len(cell.periods)/2,ig_scale+.5*weight.quadratic_form(resid))
            psi,weight=sample_psi(psi,resid,sigma2,prior=psi_prior,proposal=proposal,rng=rng,n_steps=int(inf["psi_steps_per_sweep"]),weighting=weight)
            if it==warmup-1:proposal.freeze()
            if it>=warmup and (it-warmup)%thin==0:
                draws[chain,save]=beta;sigma[chain,save]=np.sqrt(sigma2);psi_out[chain,save]=psi;nhat_out[chain,save]=hat;save+=1
    diagnostics=_theta_diagnostics(names,draws,sigma,psi_out)
    return ThetaFit(cell.name,tuple(map(str,cell.periods)),names,draws,sigma,psi_out,nhat_out,dict(zip(names,pmean)),dict(zip(names,psd)),diagnostics)


def summary(values) -> dict[str,float]:
    x=np.asarray(values).reshape(-1);return {"mean":float(x.mean()),"sd":float(x.std(ddof=1)),"q2.5":float(np.percentile(x,2.5)),"q50":float(np.percentile(x,50)),"q97.5":float(np.percentile(x,97.5)),"p_positive":float(np.mean(x>0))}


def summarize_state(fit: FirmStateFit) -> dict[str,Any]:
    out={"diagnostics":fit.diagnostics,"acceptance":fit.acceptance,"map":fit.map_parameters}
    for j,name in enumerate(fit.names):out[name]=summary(fit.parameters[:,:,j])
    out["slow_innovation_variance"]=summary(fit.parameters[:,:,1]**2*fit.parameters[:,:,2]);out["cycle_innovation_variance"]=summary(fit.parameters[:,:,1]**2*(1-fit.parameters[:,:,2]))
    return out


def summarize_theta(fit: ThetaFit) -> dict[str,Any]:
    out={"cell":fit.cell,"sample":[fit.periods[0],fit.periods[-1],len(fit.periods)],"diagnostics":fit.diagnostics,"coefficients":{}}
    for j,name in enumerate(fit.names):
        s=summary(fit.draws[:,:,j]);s.update(prior_mean=fit.prior_mean[name],prior_sd=fit.prior_sd[name],posterior_prior_sd_ratio=s["sd"]/fit.prior_sd[name],rhat=fit.diagnostics["rhat"][name],ess_bulk=fit.diagnostics["ess_bulk"][name],ess_tail=fit.diagnostics["ess_tail"][name]);out["coefficients"][name]=s
    return out


def simulate_inflation(rng,cell: ThetaCell,fit: ThetaFit,state: FirmStateFit,theta: float):
    means=fit.draws.reshape(-1,5).mean(0);sigma=float(fit.sigma_u.mean());psi=fit.psi.reshape(-1,3).mean(0)
    sp=pd.PeriodIndex(state.periods,freq="Q");pos=sp.get_indexer(cell.periods);c=int(rng.integers(state.nhat.shape[0]));d=int(rng.integers(state.nhat.shape[1]));hat=state.nhat[c,d,pos]
    u=rng.normal(0,sigma,size=len(cell.periods));eps=u.copy()
    for j in range(1,4):eps[j:]+=psi[j-1]*u[:-j]
    y=np.zeros(len(cell.periods));previous=float(cell.pi_lag[0]);a,ab,af,kappa,_=means
    for t in range(len(y)):
        y[t]=a+ab*previous+af*cell.epi[t]+kappa*cell.x[t]-theta*hat[t]+eps[t];previous=y[t]
    return cell.with_inflation(y,float(cell.pi_lag[0])),hat


def detection_indicator(qlo: float,qhi: float,p_positive: float,sd_ratio: float,*,positive: bool=True,
                        sign_probability: float=.975,sd_ratio_limit: float=.75,
                        require_interval: bool=True) -> bool:
    interval=(qlo>0) if positive else (qhi<0)
    sign=(p_positive>=sign_probability) if positive else (p_positive<=1-sign_probability)
    return bool((interval or not require_interval) and sign and sd_ratio<=sd_ratio_limit)


def save_state(path: Path,fit: FirmStateFit):
    path.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(path,periods=fit.periods,names=fit.names,parameters=fit.parameters,nbar=fit.nbar,nhat=fit.nhat,n_total=fit.n_total)


def load_state(path: Path,diagnostics=None) -> FirmStateFit:
    z=np.load(path,allow_pickle=False);return FirmStateFit(tuple(map(str,z["periods"])),tuple(map(str,z["names"])),z["parameters"],z["nbar"],z["nhat"],z["n_total"],diagnostics or {},{},[])


def save_theta(path: Path,fit: ThetaFit):
    path.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(path,cell=fit.cell,periods=fit.periods,names=fit.names,draws=fit.draws,sigma_u=fit.sigma_u,psi=fit.psi,nhat_used=fit.nhat_used,prior_mean=np.array([fit.prior_mean[n] for n in fit.names]),prior_sd=np.array([fit.prior_sd[n] for n in fit.names]))


def load_theta(path: Path,diagnostics=None) -> ThetaFit:
    z=np.load(path,allow_pickle=False);names=tuple(map(str,z["names"]));means=z["prior_mean"];sds=z["prior_sd"]
    return ThetaFit(str(z["cell"]),tuple(map(str,z["periods"])),names,z["draws"],z["sigma_u"],z["psi"],z["nhat_used"],dict(zip(names,map(float,means))),dict(zip(names,map(float,sds))),diagnostics or {})
