"""Exact competition allocation/decomposition and modular HSA estimation."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd
from scipy.special import expit, gammaln, logit
from scipy.stats import beta as beta_dist, norm, truncnorm

from nkpc_hsa.phillips.state import _draw_ig
from nkpc_hsa.report_models.cases import CaseData, GUSTAVO_ANNUAL_COL, _load_frame
from nkpc_hsa.report_models.engine import build_priors
from tests.hsa_lambda_dynamic.functions import (
    COEFF_NAMES, HSA_MODELS, MODEL_LABELS, FitResult, _design, _draw_coefficients,
    _draw_lambda, _draw_phi, _inflation_loglik, _prior_maps, comparison_metrics,
    derived_paths, robust_scale,
)


@dataclass(frozen=True)
class AllocationPosterior:
    annual: pd.Series
    periods: pd.PeriodIndex
    mean_weights: dict[int, np.ndarray]
    chol_weights: dict[int, np.ndarray]
    average_weights: np.ndarray
    raw_weights: dict[int, np.ndarray]
    coherence: dict[int, float]

    def draw_path(self, rng: np.random.Generator) -> np.ndarray:
        values = pd.Series(index=self.periods, dtype=float)
        for year in self.annual.index:
            year = int(year)
            previous = float(self.annual.get(year - 1, self.annual[year]))
            annual_change = float(self.annual[year] - previous)
            deviation = self.chol_weights[year] @ rng.normal(size=4)
            deviation -= deviation.mean()
            weights = self.mean_weights[year] + deviation
            weights += (1.0 - weights.sum()) / 4.0
            cumulative = 0.0
            for quarter in range(1, 5):
                cumulative += weights[quarter - 1] * annual_change
                values[pd.Period(f"{year}Q{quarter}", freq="Q")] = previous + cumulative
        return values.to_numpy(float)


@dataclass(frozen=True)
class ExactData:
    case: CaseData
    allocation: AllocationPosterior
    allocation_positions: np.ndarray


@dataclass
class StateFit:
    periods: tuple[str, ...]
    n_total: np.ndarray
    nbar: np.ndarray
    nhat: np.ndarray
    omega: np.ndarray
    tau: np.ndarray
    rho: np.ndarray
    diagnostics: dict


def build_allocation_posterior(frame: pd.DataFrame, competition_col: str,
                               stable_raw_weight_max: float, covariance_scale: float):
    num = lambda c: pd.to_numeric(frame[c], errors="coerce")
    g = num(GUSTAVO_ANNUAL_COL).dropna()
    annual = pd.Series({ix.year: 10.0 * np.log(v) for ix, v in g.items()}, dtype=float)
    dciq = (10.0 * np.log(num(competition_col)).dropna()).diff()
    raw = {}; coherence = {}
    for year in annual.index:
        periods = [pd.Period(f"{int(year)}Q{q}", freq="Q") for q in range(1, 5)]
        if all(p in dciq.index and np.isfinite(dciq.get(p, np.nan)) for p in periods):
            dq = np.array([dciq[p] for p in periods])
            if abs(dq.sum()) > 1e-10:
                raw[int(year)] = dq / dq.sum()
                coherence[int(year)] = float(abs(dq.sum()) / max(np.abs(dq).sum(), 1e-12))
    stable = np.array([w for w in raw.values() if np.max(np.abs(w)) <= stable_raw_weight_max])
    average = np.median(stable, axis=0); average /= average.sum()
    projector = np.eye(4) - np.ones((4, 4)) / 4.0
    covariance = projector @ np.cov(stable, rowvar=False) @ projector * covariance_scale
    eigval, eigvec = np.linalg.eigh((covariance + covariance.T) / 2)
    eigval = np.maximum(eigval, 0.0)
    base_chol = eigvec @ np.diag(np.sqrt(eigval))
    means = {}; chols = {}
    for year in annual.index:
        year = int(year)
        if year in raw:
            c = coherence[year]
            means[year] = c * raw[year] + (1.0 - c) * average
            chols[year] = np.sqrt(max(1e-5, 1.0 - c)) * base_chol
        else:
            coherence[year] = 0.0
            means[year] = average.copy(); chols[year] = base_chol.copy()
    periods = pd.period_range(f"{int(annual.index.min())}Q1", f"{int(annual.index.max())}Q4", freq="Q")
    return AllocationPosterior(annual, periods, means, chols, average, raw, coherence)


def load_exact_data(config: dict) -> ExactData:
    frame = _load_frame(); dc = config["data"]; ac = config["allocation"]
    allocation = build_allocation_posterior(
        frame, dc["competition"], float(ac["stable_raw_weight_max"]),
        float(ac["prior_covariance_scale"]),
    )
    num = lambda c: pd.to_numeric(frame[c], errors="coerce")
    marker = pd.Series(1.0, index=allocation.periods)
    d = pd.concat({"pi": num(dc["inflation"]), "lag": num(dc["inflation_lag"]),
                   "epi": num(dc["expectation"]), "x": num(dc["activity"]),
                   "marker": marker}, axis=1).dropna()
    d = d[d.index >= pd.Period(dc["sample_start"], freq="Q")]
    positions = allocation.periods.get_indexer(d.index)
    reference = allocation.draw_path(np.random.default_rng(0))[positions]
    reference -= reference.mean()
    case = CaseData(case=1, label="exact_n__ppi__neg_unemp", periods=d.index,
                    pi=d.pi.to_numpy(), epi=d.epi.to_numpy(), x=d.x.to_numpy(),
                    n_obs=reference, exact_anchor=True, gE=None,
                    s_x=robust_scale(d.x), s_N=robust_scale(reference), s_pi=robust_scale(d.pi),
                    s_E=None, pi_lag=d.lag.to_numpy())
    return ExactData(case, allocation, positions)


def _precision_draw_nhat(rng, n_total, rho, tau2, omega):
    T = len(n_total); vb = max(1e-10, omega * tau2); vh = max(1e-10, (1.0 - omega) * tau2)
    diag = np.full(T, 2.0 / vb + (1.0 + rho ** 2) / vh)
    diag[0] = 1.0 / vb + 1.0 / vh
    diag[-1] = 1.0 / vb + 1.0 / vh
    off = np.full(T - 1, -1.0 / vb - rho / vh)
    dn = np.diff(n_total)
    rhs = np.zeros(T)
    rhs[0] = -dn[0] / vb; rhs[-1] = dn[-1] / vb
    if T > 2:
        rhs[1:-1] = (dn[:-1] - dn[1:]) / vb
    # Tridiagonal Cholesky and precision solve/sample.
    ld = np.empty(T); ls = np.empty(T - 1); ld[0] = np.sqrt(max(diag[0], 1e-12))
    for i in range(1, T):
        ls[i - 1] = off[i - 1] / ld[i - 1]
        ld[i] = np.sqrt(max(diag[i] - ls[i - 1] ** 2, 1e-12))
    y = np.empty(T); y[0] = rhs[0] / ld[0]
    for i in range(1, T): y[i] = (rhs[i] - ls[i - 1] * y[i - 1]) / ld[i]
    mean = np.empty(T); mean[-1] = y[-1] / ld[-1]
    for i in range(T - 2, -1, -1): mean[i] = (y[i] - ls[i] * mean[i + 1]) / ld[i]
    z = rng.normal(size=T); noise = np.empty(T); noise[-1] = z[-1] / ld[-1]
    for i in range(T - 2, -1, -1): noise[i] = (z[i] - ls[i] * noise[i + 1]) / ld[i]
    return mean + noise


def _draw_rho(rng, h, variance, mean, sd, lower, upper):
    x, y = h[:-1], h[1:]
    precision = 1.0 / sd ** 2 + float(x @ x) / variance
    psd = np.sqrt(1.0 / precision)
    pmean = (mean / sd ** 2 + float(x @ y) / variance) / precision
    return float(truncnorm.rvs((lower-pmean)/psd, (upper-pmean)/psd,
                               loc=pmean, scale=psd, random_state=rng))


def _omega_logtarget(z, rb, rh, h0, rho, tau2, a, b):
    w = expit(z); T = len(rb) + 1
    ssh = float(rh @ rh) + max(1e-8, 1.0-rho**2) * h0**2
    out = -(T-1)/2*np.log(w) - float(rb@rb)/(2*w*tau2)
    out += -T/2*np.log(1-w) - ssh/(2*(1-w)*tau2)
    out += beta_dist.logpdf(w, a, b) + np.log(w) + np.log(1-w)
    return float(out)


def fit_states(exact: ExactData, config: dict, sampling: dict, seed: int) -> StateFit:
    data=exact.case; sp=config["state_priors"]; chains=int(sampling["chains"])
    iterations=int(sampling["state_iterations"]); warmup=int(sampling["state_warmup"]); thin=int(sampling["state_thin"])
    nsave=(iterations-warmup+thin-1)//thin; shape=(chains,nsave); T=data.n_periods
    nt=np.zeros(shape+(T,)); nb=np.zeros_like(nt); nh=np.zeros_like(nt)
    omega_out=np.zeros(shape); tau_out=np.zeros(shape); rho_out=np.zeros(shape); acc=np.zeros(chains)
    tau_scale=2*(float(sp["tau_scale_fraction"])*data.s_N)**2
    for ch in range(chains):
        rng=np.random.default_rng(seed+ch*7919); omega=float(sp["omega_a"])/(float(sp["omega_a"])+float(sp["omega_b"]))
        tau2=tau_scale/(float(sp["tau_shape"])-1); rho=float(sp["rho_mean"]); save=0
        for it in range(iterations):
            total=exact.allocation.draw_path(rng)[exact.allocation_positions]; total-=total.mean()
            h=_precision_draw_nhat(rng,total,rho,tau2,omega); bar=total-h
            rho=_draw_rho(rng,h,(1-omega)*tau2,float(sp["rho_mean"]),float(sp["rho_sd"]),
                          float(sp["rho_lower"]),float(sp["rho_upper"]))
            rb=np.diff(bar); rh=h[1:]-rho*h[:-1]
            proposal=logit(omega)+rng.normal(0,0.16)
            if np.log(rng.uniform()) < _omega_logtarget(proposal,rb,rh,h[0],rho,tau2,float(sp["omega_a"]),float(sp["omega_b"]))-_omega_logtarget(logit(omega),rb,rh,h[0],rho,tau2,float(sp["omega_a"]),float(sp["omega_b"])):
                omega=float(expit(proposal)); acc[ch]+=1
            ssh=float(rh@rh)+max(1e-8,1-rho**2)*h[0]**2
            scaled=float(rb@rb)/omega+ssh/(1-omega)
            tau2=_draw_ig(rng,float(sp["tau_shape"])+(2*T-1)/2,tau_scale+0.5*scaled)
            if it>=warmup and (it-warmup)%thin==0:
                nt[ch,save]=total; nb[ch,save]=bar; nh[ch,save]=h
                omega_out[ch,save]=omega; tau_out[ch,save]=np.sqrt(tau2); rho_out[ch,save]=rho; save+=1
    rhats={"omega":float(az.rhat(omega_out,method="rank")),"tau":float(az.rhat(tau_out,method="rank")),"rho":float(az.rhat(rho_out,method="rank"))}
    path_rhat=np.asarray(az.rhat(nb,method="rank")); rhats["nbar_path_max"]=float(np.nanmax(path_rhat))
    diagnostics={"rhat":rhats,"max_rhat":max(rhats.values()),"omega_acceptance":(acc/iterations).tolist(),
                 "exact_identity_error":float(np.max(np.abs(nt-nb-nh)))}
    return StateFit(tuple(map(str,data.periods)),nt,nb,nh,omega_out,tau_out,rho_out,diagnostics)


def fit_model_cut(data: CaseData, states: StateFit, model: str, config: dict, sampling: dict, seed: int):
    cp=config["coefficient_priors"]; chains=int(sampling["chains"]); iterations=int(sampling["model_iterations"])
    warmup=int(sampling["model_warmup"]); thin=int(sampling["model_thin"]); nsave=(iterations-warmup+thin-1)//thin
    priors=build_priors(data,coef_scale=float(cp["coefficient_scale"]),hybrid=True)
    names,means,sds=_prior_maps(model,priors,float(cp["lambda_mean"]),float(cp["lambda_sd"])); stored=names+(("lambda",) if model in HSA_MODELS else ())
    draws=np.zeros((chains,nsave,len(stored))); sig=np.zeros((chains,nsave)); phi_out=np.zeros((chains,nsave))
    nbar_out=np.zeros((chains,nsave,data.n_periods)); nhat_out=np.zeros_like(nbar_out)
    pmean=np.array([means[n] for n in names]); psd=np.array([sds[n] for n in names]); accept=np.zeros(chains)
    for ch in range(chains):
        rng=np.random.default_rng(seed+ch*7919+list(COEFF_NAMES).index(model)*104729)
        beta_vec=pmean.copy(); beta=dict(zip(names,beta_vec)); lam=float(cp["lambda_mean"]); phi=float(cp["phi_mean"])
        sigma2=priors.sigma_pi_b/(priors.ig_shape-1); save=0
        for it in range(iterations):
            j=int(rng.integers(states.nbar.shape[1])); nbar=states.nbar[ch,j]; nhat=states.nhat[ch,j]
            X=_design(model,data,nbar,nhat,lam)
            beta_vec,sigma2=_draw_coefficients(rng,data.pi,X,phi,pmean,psd,sigma2,priors.ig_shape,priors.sigma_pi_b)
            beta=dict(zip(names,beta_vec))
            if model in HSA_MODELS:
                lam=_draw_lambda(rng,model,data,beta,nbar,nhat,phi,sigma2,float(cp["lambda_mean"]),float(cp["lambda_sd"]))
            mu=_design(model,data,nbar,nhat,lam)@beta_vec
            phi,ok=_draw_phi(rng,data.pi,mu,sigma2,phi,float(cp["phi_mean"]),float(cp["phi_sd"]));accept[ch]+=ok
            if it>=warmup and (it-warmup)%thin==0:
                draws[ch,save]=list(beta_vec)+([lam] if model in HSA_MODELS else [])
                sig[ch,save]=np.sqrt(sigma2);phi_out[ch,save]=phi;nbar_out[ch,save]=nbar;nhat_out[ch,save]=nhat;save+=1
    rhats={n:float(az.rhat(draws[:,:,i],method="rank")) for i,n in enumerate(stored)}
    rhats.update(sigma_pi=float(az.rhat(sig,method="rank")),phi=float(az.rhat(phi_out,method="rank")))
    diagnostics={"rhat":rhats,"max_rhat":max(rhats.values()),"phi_acceptance":(accept/iterations).tolist()}
    zeros=np.zeros_like(sig)
    return FitResult(model,MODEL_LABELS[model],stored,draws,sig,phi_out,zeros,zeros,zeros,zeros,
                     nbar_out,nhat_out,tuple(map(str,data.periods)),{n:means[n] for n in stored},{n:sds[n] for n in stored},diagnostics)


def state_averaged_logml(fit: FitResult, data: CaseData):
    blocks=[fit.draws,fit.phi[:,:,None],np.log(fit.sigma_pi)[:,:,None]]
    matrix=np.concatenate(blocks,axis=2).reshape(-1,fit.draws.shape[-1]+2); center=matrix.mean(0)
    cov=np.cov(matrix,rowvar=False); sign,logdet=np.linalg.slogdet(cov+np.eye(len(center))*1e-10)
    beta={n:center[i] for i,n in enumerate(fit.names) if n!="lambda"};lam=center[fit.names.index("lambda")] if "lambda" in fit.names else 0
    phi=center[-2];sigma2=np.exp(2*center[-1]); nbar=fit.nbar.reshape(-1,data.n_periods);nhat=fit.nhat.reshape(-1,data.n_periods)
    idx=np.linspace(0,len(nbar)-1,min(1500,len(nbar))).astype(int); ll=[]
    for j in idx:
        mu=_design(fit.model,data,nbar[j],nhat[j],lam)@np.array([beta[n] for n in COEFF_NAMES[fit.model]])
        ll.append(_inflation_loglik(data.pi,mu,sigma2,phi,include_constants=True))
    mx=max(ll); integrated=mx+np.log(np.mean(np.exp(np.array(ll)-mx)))
    lp=sum(norm.logpdf(center[i],fit.prior_mean[n],fit.prior_sd[n]) for i,n in enumerate(fit.names))
    lp+=norm.logpdf(phi,0,.35); pri=build_priors(data,coef_scale=.2,hybrid=True);v=sigma2
    lp+=pri.ig_shape*np.log(pri.sigma_pi_b)-gammaln(pri.ig_shape)-(pri.ig_shape+1)*np.log(v)-pri.sigma_pi_b/v+np.log(2)+2*center[-1]
    return float(integrated+lp+0.5*len(center)*np.log(2*np.pi)+0.5*logdet) if sign>0 else float("nan")


def save_states(path: Path, fit: StateFit):
    path.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(path,periods=fit.periods,n_total=fit.n_total,nbar=fit.nbar,nhat=fit.nhat,omega=fit.omega,tau=fit.tau,rho=fit.rho)


def load_states(path: Path, diagnostics=None):
    z=np.load(path,allow_pickle=False);return StateFit(tuple(map(str,z["periods"])),z["n_total"],z["nbar"],z["nhat"],z["omega"],z["tau"],z["rho"],diagnostics or {})

