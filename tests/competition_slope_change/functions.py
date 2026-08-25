"""Competition-only state and modular MA(3) slope-NKPC functions."""
from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import arviz as az
import numpy as np
import pandas as pd
from scipy.special import expit, logit
from scipy.stats import beta as beta_dist, norm

from nkpc_hsa.error_robustness.ma_error import (
    AdaptiveRandomWalk,
    MAWeighting,
    PsiPrior,
    sample_psi,
)
from nkpc_hsa.phillips.state import _draw_ig
from tests.hsa_deep_identification.joint_ma3 import _slice_scalar, _update_state_parameters
from tests.hsa_nested_validation.functions import (
    BASE_NAMES,
    CellData,
    ExperimentData,
    ModelSpec,
    _cycle_coefficients,
    _cycle_sumsq,
    _prior_maps,
    _sample_h,
    load_experiment,
    robust_scale,
)
from tests.hsa_exact_n_decomposition.functions import (
    _draw_rho,
    _omega_logtarget,
    _precision_draw_nhat,
)
from tests.hsa_deep_identification.joint_ma3 import _draw_beta_gls


@dataclass
class CompetitionStateFit:
    law: str
    prior_label: str
    periods: tuple[str, ...]
    n_total: np.ndarray
    nbar: np.ndarray
    nhat: np.ndarray
    omega: np.ndarray
    tau: np.ndarray
    damping_or_rho: np.ndarray
    period: np.ndarray
    drift: np.ndarray
    diagnostics: dict[str, Any]


@dataclass
class NKPCFit:
    cell: str
    model: str
    timing: str
    periods: tuple[str, ...]
    names: tuple[str, ...]
    draws: np.ndarray
    sigma_u: np.ndarray
    psi: np.ndarray
    nbar: np.ndarray
    nhat_used: np.ndarray
    kappa: np.ndarray
    prior_mean: dict[str, float]
    prior_sd: dict[str, float]
    diagnostics: dict[str, Any]


def prepare_experiment(config: dict) -> ExperimentData:
    """Load the shared data but use a predeclared coordinate origin."""
    nested = {
        "samples": config["sample"],
        "data": config["data"],
        "allocation": config["allocation"],
    }
    experiment = load_experiment(nested)
    reference_year = int(config["sample"]["coordinate_reference_year"])
    if reference_year not in experiment.allocation.annual.index:
        raise ValueError(f"Coordinate reference year {reference_year} is unavailable")
    q0 = float(experiment.allocation.annual.loc[reference_year])
    # Cell scales are translation invariant; only the coordinate origin changes.
    return replace(experiment, q0=q0)


def state_periods(experiment: ExperimentData, config: dict) -> tuple[pd.PeriodIndex, np.ndarray]:
    start = pd.Period(config["sample"]["start"], freq="Q")
    end = pd.Period(config["sample"]["end"], freq="Q")
    mask = (experiment.allocation.periods >= start) & (experiment.allocation.periods <= end)
    return experiment.allocation.periods[mask], np.flatnonzero(mask)


def annual_allocation_drift(experiment: ExperimentData, periods: pd.PeriodIndex) -> np.ndarray:
    drift = np.zeros(len(periods), dtype=float)
    annual = experiment.allocation.annual
    weights = np.asarray(experiment.allocation.average_weights, dtype=float)
    for t in range(1, len(periods)):
        period = periods[t]; year = int(period.year)
        if year in annual.index and year - 1 in annual.index:
            drift[t] = weights[period.quarter - 1] * float(annual.loc[year] - annual.loc[year - 1])
    return drift


def _state_loglik_ar2(q: np.ndarray, drift: np.ndarray, damping: float, period: float,
                      omega: float, tau2: float) -> float:
    """Marginal likelihood of competition changes with the AR(2) cycle integrated out."""
    from tests.hsa_nested_validation.functions import _cycle_unit_cov
    phi1, phi2 = _cycle_coefficients(damping, period)
    F = np.array([[phi1, phi2], [1.0, 0.0]])
    vh=max(1e-12,(1.0-omega)*tau2);vb=max(1e-12,omega*tau2)
    Q=np.array([[vh,0.0],[0.0,0.0]]);H=np.array([1.0,-1.0])
    mean=np.zeros(2);cov=vh*_cycle_unit_cov(damping,period);ll=0.0
    observed=np.diff(q)-drift[1:]
    for value in observed:
        mean=F@mean;cov=F@cov@F.T+Q
        variance=float(H@cov@H+vb)
        if not np.isfinite(variance) or variance<=0:return -np.inf
        innovation=float(value-H@mean)
        ll+=-0.5*(np.log(2*np.pi*variance)+innovation**2/variance)
        gain=cov@H/variance;mean=mean+gain*innovation
        cov=cov-np.outer(gain,H@cov);cov=(cov+cov.T)/2
    return float(ll)


def _collapsed_ar2_parameters(rng,q,drift,damping,period,omega,tau2,s_q,sp):
    """Slice updates using the state-marginal competition likelihood."""
    lo_d,hi_d=map(float,sp["cycle_damping_bounds"]);lo_p,hi_p=map(float,sp["cycle_period_bounds_quarters"])
    a_o,b_o=map(float,sp["omega_prior"]);shape=float(sp["tau2_prior_shape"])
    scale=2.0*(float(sp["tau2_scale_fraction"])*float(s_q))**2
    def ll(d,p,o,t2):
        try:return _state_loglik_ar2(q,drift,d,p,o,t2)
        except (ValueError,np.linalg.LinAlgError,FloatingPointError):return -np.inf
    def omega_target(z):
        o=float(expit(z));return float(ll(damping,period,o,tau2)+beta_dist.logpdf(o,a_o,b_o)+np.log(o)+np.log1p(-o))
    omega=float(expit(_slice_scalar(rng,logit(omega),omega_target,width=float(sp["omega_slice_width"]))))
    def tau_target(z):
        t2=float(np.exp(np.clip(z,-30,30)));return float(ll(damping,period,omega,t2)-shape*z-scale/t2)
    tau2=float(np.exp(_slice_scalar(rng,np.log(tau2),tau_target,width=.5)))
    def value(z,lo,hi):return float(lo+(hi-lo)*expit(z))
    def zvalue(v,lo,hi):return float(logit((v-lo)/(hi-lo)))
    zd,zp=zvalue(damping,lo_d,hi_d),zvalue(period,lo_p,hi_p)
    def damping_target(z):
        d=value(z,lo_d,hi_d);u=expit(z);aa,bb=map(float,sp["cycle_damping_prior"])
        return float(ll(d,period,omega,tau2)+beta_dist.logpdf(u,aa,bb)+np.log(u)+np.log1p(-u))
    damping=value(_slice_scalar(rng,zd,damping_target,width=.6),lo_d,hi_d)
    def period_target(z):
        p=value(z,lo_p,hi_p);u=expit(z)
        return float(ll(damping,p,omega,tau2)+norm.logpdf(p,float(sp["cycle_period_prior_mean"]),float(sp["cycle_period_prior_sd"]))+np.log(u)+np.log1p(-u))
    period=value(_slice_scalar(rng,zp,period_target,width=.8),lo_p,hi_p)
    return damping,period,omega,tau2


def _state_diagnostics(fit: CompetitionStateFit) -> dict[str, Any]:
    scalar = {
        "omega": fit.omega,
        "tau": fit.tau,
        "cycle_damping" if fit.law == "ar2" else "cycle_rho": fit.damping_or_rho,
    }
    if fit.law == "ar2":
        scalar["cycle_period"] = fit.period
    rhat = {name: float(az.rhat(values, method="rank")) for name, values in scalar.items()}
    bulk = {name: float(az.ess(values, method="bulk")) for name, values in scalar.items()}
    tail = {name: float(az.ess(values, method="tail", prob=(0.05, 0.95))) for name, values in scalar.items()}
    path_rhat = {
        "nbar_path_max": float(np.nanmax(np.asarray(az.rhat(fit.nbar, method="rank")))),
        "nhat_path_max": float(np.nanmax(np.asarray(az.rhat(fit.nhat, method="rank")))),
    }
    path_bulk = {
        "nbar_path_min": float(np.nanmin(np.asarray(az.ess(fit.nbar, method="bulk")))),
        "nhat_path_min": float(np.nanmin(np.asarray(az.ess(fit.nhat, method="bulk")))),
    }
    path_tail = {
        "nbar_path_min": float(np.nanmin(np.asarray(az.ess(fit.nbar, method="tail", prob=(0.05, 0.95))))),
        "nhat_path_min": float(np.nanmin(np.asarray(az.ess(fit.nhat, method="tail", prob=(0.05, 0.95))))),
    }
    return {
        "rhat": rhat,
        "ess_bulk": bulk,
        "ess_tail": tail,
        "path_rhat": path_rhat,
        "path_ess_bulk": path_bulk,
        "path_ess_tail": path_tail,
        "max_rhat": max([*rhat.values(), *path_rhat.values()]),
        "min_bulk_ess": min([*bulk.values(), *path_bulk.values()]),
        "min_tail_ess": min([*tail.values(), *path_tail.values()]),
        "exact_identity_error": float(np.max(np.abs(fit.n_total - fit.nbar - fit.nhat))),
    }


def fit_competition_state(
    experiment: ExperimentData,
    config: dict,
    sampling: dict,
    *,
    law: str,
    omega_prior: tuple[float, float],
    prior_label: str,
    seed: int,
    sensitivity: bool = False,
) -> CompetitionStateFit:
    periods, positions = state_periods(experiment, config)
    drift = annual_allocation_drift(experiment, periods)
    sp = dict(config["state"]); sp["omega_prior"] = list(map(float, omega_prior))
    prefix = "sensitivity_" if sensitivity else "state_"
    iterations = int(sampling[f"{prefix}iterations"])
    warmup = int(sampling[f"{prefix}warmup"])
    thin = int(sampling[f"{prefix}thin"])
    chains = int(sampling["state_chains"])
    nsave = (iterations - warmup + thin - 1) // thin
    shape = (chains, nsave); T = len(periods)
    nt = np.zeros(shape + (T,)); nb = np.zeros_like(nt); nh = np.zeros_like(nt)
    omega_out = np.zeros(shape); tau_out = np.zeros(shape)
    cycle_out = np.zeros(shape); period_out = np.full(shape, np.nan)
    acc_omega = np.zeros(chains); acc_cycle = np.zeros(chains); acc_period = np.zeros(chains)
    mean_q = experiment.allocation_mean_raw[positions] - experiment.q0
    s_q = robust_scale(mean_q)
    scale = 2.0 * (float(sp["tau2_scale_fraction"]) * s_q) ** 2
    for chain in range(chains):
        rng = np.random.default_rng(seed + 7919 * chain)
        omega = float(omega_prior[0]) / float(sum(omega_prior))
        tau2 = scale / (float(sp["tau2_prior_shape"]) - 1.0)
        damping = float(sp["cycle_damping_initial"])
        period = float(sp["cycle_period_initial"])
        rho = float(sp["ar1_rho_mean"])
        save = 0
        for it in range(iterations):
            q = experiment.allocation.draw_path(rng)[positions] - experiment.q0
            q_adjusted = q - np.cumsum(drift)
            if law == "ar2":
                if it % int(sp.get("collapsed_stride",5)) == 0:
                    damping,period,omega,tau2=_collapsed_ar2_parameters(
                        rng,q,drift,damping,period,omega,tau2,s_q,sp)
                    h=_sample_h(rng,q_adjusted,damping,period,tau2,omega);bar=q-h
                    ad=ap=ao=True
                else:
                    h = _sample_h(rng, q_adjusted, damping, period, tau2, omega)
                    bar, damping, period, omega, tau2, ad, ap, ao = _update_state_parameters(
                        rng, q, h, drift, damping, period, omega, tau2, s_q, sp,
                    )
                acc_cycle[chain] += ad; acc_period[chain] += ap; acc_omega[chain] += ao
                cycle_value = damping; period_value = period
            elif law == "ar1":
                h = _precision_draw_nhat(rng, q_adjusted, rho, tau2, omega)
                bar = q - h
                rho = _draw_rho(
                    rng, h, (1.0 - omega) * tau2,
                    float(sp["ar1_rho_mean"]), float(sp["ar1_rho_sd"]),
                    float(sp["ar1_rho_bounds"][0]), float(sp["ar1_rho_bounds"][1]),
                )
                rb = np.diff(bar) - drift[1:]
                rh = h[1:] - rho * h[:-1]
                proposal = logit(omega) + rng.normal(0.0, float(sp["omega_mh_sd"]))
                old = _omega_logtarget(proposal * 0 + logit(omega), rb, rh, h[0], rho, tau2, *map(float, omega_prior))
                new = _omega_logtarget(proposal, rb, rh, h[0], rho, tau2, *map(float, omega_prior))
                if np.log(rng.uniform()) < new - old:
                    omega = float(expit(proposal)); acc_omega[chain] += 1
                ssh = float(rh @ rh) + max(1e-8, 1.0 - rho**2) * h[0] ** 2
                scaled = float(rb @ rb) / omega + ssh / (1.0 - omega)
                tau2 = _draw_ig(
                    rng, float(sp["tau2_prior_shape"]) + (2*T-1)/2,
                    scale + 0.5 * scaled,
                )
                cycle_value = rho; period_value = np.nan
            else:
                raise ValueError(f"Unsupported state law: {law}")
            if it >= warmup and (it - warmup) % thin == 0:
                nt[chain, save] = q; nb[chain, save] = bar; nh[chain, save] = h
                omega_out[chain, save] = omega; tau_out[chain, save] = np.sqrt(tau2)
                cycle_out[chain, save] = cycle_value; period_out[chain, save] = period_value
                save += 1
    fit = CompetitionStateFit(
        law, prior_label, tuple(map(str, periods)), nt, nb, nh,
        omega_out, tau_out, cycle_out, period_out, drift, {},
    )
    diagnostics = _state_diagnostics(fit)
    diagnostics.update(
        omega_acceptance=(acc_omega / iterations).tolist(),
        cycle_acceptance=(acc_cycle / iterations).tolist(),
        period_acceptance=(acc_period / iterations).tolist(),
    )
    fit.diagnostics = diagnostics
    return fit


def conditional_omega_likelihood(fit: CompetitionStateFit, grid: np.ndarray) -> pd.DataFrame:
    """Conditional state-likelihood slice; not a profile over all parameters."""
    q = fit.n_total.mean(axis=(0, 1)); bar = fit.nbar.mean(axis=(0, 1)); h = fit.nhat.mean(axis=(0, 1))
    tau2 = float(np.mean(fit.tau**2)); rb = np.diff(bar) - fit.drift[1:]
    values = []
    if fit.law == "ar2":
        damping = float(fit.damping_or_rho.mean()); period = float(fit.period.mean())
        cycle_ss, logdet = _cycle_sumsq(h, damping, period)
        for omega in grid:
            vb, vh = omega*tau2, (1-omega)*tau2
            ll = -(len(rb)/2)*np.log(vb) - float(rb@rb)/(2*vb)
            ll += -(len(h)/2)*np.log(vh) - 0.5*logdet - cycle_ss/(2*vh)
            values.append(ll)
    else:
        rho = float(fit.damping_or_rho.mean()); rh = h[1:]-rho*h[:-1]
        cycle_ss = float(rh@rh)+max(1e-8,1-rho**2)*h[0]**2
        for omega in grid:
            vb, vh = omega*tau2, (1-omega)*tau2
            ll = -(len(rb)/2)*np.log(vb)-float(rb@rb)/(2*vb)
            ll += -(len(h)/2)*np.log(vh)-cycle_ss/(2*vh)
            values.append(ll)
    values = np.asarray(values); values -= np.max(values)
    return pd.DataFrame({"omega": grid, "relative_conditional_loglik": values})


def _timing_indices(T: int, timing: str) -> tuple[np.ndarray, np.ndarray]:
    if timing == "current":
        idx = np.arange(T); return idx, idx
    if timing == "lag1":
        return np.arange(1, T), np.arange(0, T-1)
    if timing == "lead1":
        return np.arange(0, T-1), np.arange(1, T)
    if timing == "none":
        idx = np.arange(T); return idx, idx
    raise ValueError(f"Unsupported timing: {timing}")


def _nkpc_diagnostics(names, draws, sigma, psi):
    series = {name: draws[:, :, j] for j, name in enumerate(names)}
    series["sigma_u"] = sigma
    series.update({f"psi_{j+1}": psi[:, :, j] for j in range(psi.shape[-1])})
    rhat = {name: float(az.rhat(values, method="rank")) for name, values in series.items()}
    bulk = {name: float(az.ess(values, method="bulk")) for name, values in series.items()}
    tail = {name: float(az.ess(values, method="tail", prob=(0.05, 0.95))) for name, values in series.items()}
    return {"rhat": rhat, "ess_bulk": bulk, "ess_tail": tail,
            "max_rhat": max(rhat.values()), "min_bulk_ess": min(bulk.values()), "min_tail_ess": min(tail.values())}


def fit_modular_nkpc(
    cell: CellData,
    states: CompetitionStateFit,
    config: dict,
    sampling: dict,
    *,
    model: str,
    timing: str,
    seed: int,
) -> NKPCFit:
    if model not in {"slope_only", "slope_plus_competition_cycle"}:
        raise ValueError(model)
    state_periods_index = pd.PeriodIndex(states.periods, freq="Q")
    state_positions = state_periods_index.get_indexer(cell.periods)
    if np.any(state_positions < 0):
        raise ValueError(f"{cell.role}: state dates do not cover the NKPC cell")
    idx, direct_idx = _timing_indices(cell.n_periods, timing)
    y = cell.pi[idx]
    T = len(idx)
    spec = ModelSpec(
        "free_static_combined" if model.endswith("cycle") else "slow_slope",
        "diagnostic" if model.endswith("cycle") else "primary",
        BASE_NAMES + (("delta_s", "theta") if model.endswith("cycle") else ("delta_s",)),
    )
    _, means, sds = _prior_maps(spec, cell, {"priors": config["priors"]})
    names = BASE_NAMES + ("delta",) + (("theta_C",) if model.endswith("cycle") else ())
    source_names = BASE_NAMES + ("delta_s",) + (("theta",) if model.endswith("cycle") else ())
    pmean = np.array([means[name] for name in source_names])
    psd = np.array([sds[name] for name in source_names])
    prior_mean = dict(zip(names, map(float, pmean))); prior_sd = dict(zip(names, map(float, psd)))
    iterations = int(sampling["nkpc_iterations"]); warmup = int(sampling["nkpc_warmup"])
    thin = int(sampling["nkpc_thin"]); chains = int(sampling["nkpc_chains"])
    nsave = (iterations-warmup+thin-1)//thin; shape=(chains, nsave)
    draws=np.zeros(shape+(len(names),)); sigma=np.zeros(shape); psi_out=np.zeros(shape+(3,))
    nbar_out=np.zeros(shape+(T,)); nhat_out=np.zeros_like(nbar_out); kappa_out=np.zeros_like(nbar_out)
    # Reuse the shared prior's innovation-variance hyperparameters.
    priors, _, _ = _prior_maps(spec, cell, {"priors": config["priors"]})
    inf = config["nkpc"]
    psi_prior = PsiPrior(np.full(3, float(inf["psi_prior_mean"])), np.full(3, float(inf["psi_prior_sd"])))
    psi_accept = np.zeros(chains)
    Cstate, Dstate = states.nbar.shape[:2]
    for chain in range(chains):
        rng=np.random.default_rng(seed+7919*chain)
        beta=pmean.copy(); sigma2=priors.sigma_pi_b/(priors.ig_shape-1.0)
        psi=np.zeros(3); weighting=MAWeighting(psi,T)
        proposal=AdaptiveRandomWalk(3,init_scale=float(inf["psi_initial_scale"])); save=0
        for it in range(iterations):
            sc=int(rng.integers(Cstate)); sd=int(rng.integers(Dstate))
            bar_full=states.nbar[sc,sd,state_positions]
            hat_full=states.nhat[sc,sd,state_positions]
            bar=bar_full[idx]; hat=hat_full[direct_idx]
            columns=[np.ones(T),cell.pi_lag[idx],cell.epi[idx],cell.x[idx],bar*cell.x[idx]]
            if model.endswith("cycle"):
                columns.append(-hat)
            X=np.column_stack(columns)
            beta=_draw_beta_gls(rng,y,X,weighting,sigma2,pmean,psd)
            resid=y-X@beta
            sigma2=_draw_ig(rng,priors.ig_shape+T/2,priors.sigma_pi_b+0.5*weighting.quadratic_form(resid))
            old_accept=proposal.n_accept
            psi,weighting=sample_psi(
                psi,resid,sigma2,prior=psi_prior,proposal=proposal,rng=rng,
                n_steps=int(inf["psi_steps_per_sweep"]),weighting=weighting,
            )
            psi_accept[chain]+=proposal.n_accept-old_accept
            if it==warmup-1:
                proposal.freeze()
            if it>=warmup and (it-warmup)%thin==0:
                draws[chain,save]=beta; sigma[chain,save]=np.sqrt(sigma2); psi_out[chain,save]=psi
                nbar_out[chain,save]=bar; nhat_out[chain,save]=hat
                kappa_out[chain,save]=beta[names.index("kappa_0")]+beta[names.index("delta")]*bar
                save+=1
    diagnostics=_nkpc_diagnostics(names,draws,sigma,psi_out)
    diagnostics["psi_acceptance"]=(psi_accept/(iterations*int(inf["psi_steps_per_sweep"]))).tolist()
    return NKPCFit(cell.role,model,timing,tuple(map(str,cell.periods[idx])),names,draws,sigma,psi_out,
                   nbar_out,nhat_out,kappa_out,prior_mean,prior_sd,diagnostics)


def _summary(values: np.ndarray) -> dict[str, float]:
    flat=np.asarray(values).reshape(-1)
    return {"mean":float(flat.mean()),"sd":float(flat.std(ddof=1)),
            "q2.5":float(np.percentile(flat,2.5)),"q50":float(np.percentile(flat,50)),
            "q97.5":float(np.percentile(flat,97.5)),"p_positive":float(np.mean(flat>0))}


def summarize_state(fit: CompetitionStateFit, omega_prior: tuple[float,float]) -> dict[str, Any]:
    slow_var=fit.omega*fit.tau**2; cycle_var=(1-fit.omega)*fit.tau**2
    out={"law":fit.law,"prior_label":fit.prior_label,"omega_prior":list(omega_prior),
         "omega":_summary(fit.omega),"tau":_summary(fit.tau),
         "slow_innovation_variance":_summary(slow_var),"cycle_innovation_variance":_summary(cycle_var),
         "damping_or_rho":_summary(fit.damping_or_rho),"diagnostics":fit.diagnostics}
    if fit.law=="ar2": out["cycle_period"]=_summary(fit.period)
    return out


def summarize_nkpc(fit: NKPCFit) -> dict[str, Any]:
    flat=fit.draws.reshape(-1,fit.draws.shape[-1]); coefficients={}
    for j,name in enumerate(fit.names):
        summary=_summary(flat[:,j]); summary.update(
            prior_mean=fit.prior_mean[name],prior_sd=fit.prior_sd[name],
            posterior_prior_sd_ratio=summary["sd"]/fit.prior_sd[name],
            rhat=fit.diagnostics["rhat"][name],ess_bulk=fit.diagnostics["ess_bulk"][name],
            ess_tail=fit.diagnostics["ess_tail"][name],
        ); coefficients[name]=summary
    for j in range(3):
        name=f"psi_{j+1}"; summary=_summary(fit.psi[:,:,j]); summary.update(
            prior_mean=0.0,prior_sd=np.nan,posterior_prior_sd_ratio=np.nan,
            rhat=fit.diagnostics["rhat"][name],ess_bulk=fit.diagnostics["ess_bulk"][name],
            ess_tail=fit.diagnostics["ess_tail"][name],
        ); coefficients[name]=summary
    return {"cell":fit.cell,"model":fit.model,"timing":fit.timing,
            "sample":[fit.periods[0],fit.periods[-1],len(fit.periods)],
            "coefficients":coefficients,"diagnostics":fit.diagnostics}


def economic_quantities(fit: NKPCFit, cell: CellData, config: dict) -> list[dict[str, Any]]:
    if fit.model != "slope_only":
        return []
    periods=pd.PeriodIndex(fit.periods,freq="Q"); flat=fit.draws.reshape(-1,fit.draws.shape[-1])
    bar=fit.nbar.reshape(-1,len(periods)); kappa=fit.kappa.reshape(-1,len(periods))
    delta=flat[:,fit.names.index("delta")]
    rows=[]
    for label,(start,end) in config["sample"]["endpoint_windows"].items():
        p0,p1=pd.Period(start,freq="Q"),pd.Period(end,freq="Q")
        if p0 not in periods or p1 not in periods: continue
        i0,i1=periods.get_loc(p0),periods.get_loc(p1)
        change_c=bar[:,i1]-bar[:,i0]; change_k=delta*change_c
        for quantity,values in {
            "competition_change":change_c,"kappa_start":kappa[:,i0],"kappa_end":kappa[:,i1],
            "delta_kappa_comp":change_k,"inflation_effect_at_one_sd_x":change_k*cell.s_x,
        }.items():
            rows.append({"cell":fit.cell,"window":label,"start":str(p0),"end":str(p1),
                         "quantity":quantity,**_summary(values)})
    r0=pd.Period(config["sample"]["counterfactual_reference_start"],freq="Q")
    r1=pd.Period(config["sample"]["counterfactual_reference_end"],freq="Q")
    mask=(periods>=r0)&(periods<=r1); cstar=bar[:,mask].mean(axis=1)
    k0=flat[:,fit.names.index("kappa_0")]; kcf=k0+delta*cstar
    end_gap=kappa[:,-1]-kcf
    for quantity,values in {"counterfactual_C_star":cstar,"counterfactual_kappa":kcf,
                            "end_kappa_minus_counterfactual":end_gap}.items():
        rows.append({"cell":fit.cell,"window":"fixed_competition_counterfactual",
                     "start":str(r0),"end":str(periods[-1]),"quantity":quantity,**_summary(values)})
    return rows


def save_state(path: Path, fit: CompetitionStateFit) -> None:
    path.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(path,law=fit.law,prior_label=fit.prior_label,periods=fit.periods,
                       n_total=fit.n_total,nbar=fit.nbar,nhat=fit.nhat,omega=fit.omega,tau=fit.tau,
                       damping_or_rho=fit.damping_or_rho,period=fit.period,drift=fit.drift)


def load_state(path: Path, diagnostics: dict[str,Any] | None=None) -> CompetitionStateFit:
    z=np.load(path,allow_pickle=False)
    return CompetitionStateFit(str(z["law"]),str(z["prior_label"]),tuple(map(str,z["periods"])),
        z["n_total"],z["nbar"],z["nhat"],z["omega"],z["tau"],z["damping_or_rho"],z["period"],z["drift"],diagnostics or {})


def save_nkpc(path: Path, fit: NKPCFit) -> None:
    path.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(path,cell=fit.cell,model=fit.model,timing=fit.timing,periods=fit.periods,
                       names=fit.names,draws=fit.draws,sigma_u=fit.sigma_u,psi=fit.psi,nbar=fit.nbar,
                       nhat_used=fit.nhat_used,kappa=fit.kappa,
                       prior_mean=np.array([fit.prior_mean[n] for n in fit.names]),
                       prior_sd=np.array([fit.prior_sd[n] for n in fit.names]))


def load_nkpc(path: Path, diagnostics: dict[str,Any] | None=None) -> NKPCFit:
    z=np.load(path,allow_pickle=False); names=tuple(map(str,z["names"])); means=z["prior_mean"]; sds=z["prior_sd"]
    return NKPCFit(str(z["cell"]),str(z["model"]),str(z["timing"]),tuple(map(str,z["periods"])),names,
                   z["draws"],z["sigma_u"],z["psi"],z["nbar"],z["nhat_used"],z["kappa"],
                   dict(zip(names,map(float,means))),dict(zip(names,map(float,sds))),diagnostics or {})
