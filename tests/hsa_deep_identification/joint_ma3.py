"""Exact-N joint state sampler with an invertible MA(3) NKPC disturbance.

This module is deliberately local to the test bundle.  It combines the exact
identity ``q = bar + hat`` with either a zero-mean quarterly slow innovation
(S0) or the external average annual-allocation increment (S1).  The latter is
a transition mean, not an observed slow state and not a zero-variance filter.
"""
from __future__ import annotations

from dataclasses import dataclass

import arviz as az
import numpy as np
from scipy.special import expit, logit
from scipy.stats import beta as beta_dist, norm

from nkpc_hsa.error_robustness.joint_ffbs_ma3 import force_pd
from nkpc_hsa.error_robustness.ma_error import (
    AdaptiveRandomWalk,
    MAWeighting,
    PsiPrior,
    sample_psi,
    state_augmentation,
)
from nkpc_hsa.phillips.state import _draw_ig
from tests.hsa_nested_validation.functions import (
    BASE_NAMES,
    CellData,
    ExperimentData,
    ModelFit,
    ModelSpec,
    _cycle_coefficients,
    _cycle_sumsq,
    _cycle_unit_cov,
    _design,
    _mu,
    _prior_maps,
)


@dataclass(frozen=True)
class JointMA3Fit:
    model_fit: ModelFit
    psi: np.ndarray
    slow_architecture: str
    slow_drift: np.ndarray


def average_allocation_drift(
    experiment: ExperimentData, cell: CellData, architecture: str
) -> np.ndarray:
    """External transition mean for the slow state on the cell's dates."""
    drift = np.zeros(cell.n_periods, dtype=float)
    if architecture == "quarterly_local_level_ar2":
        return drift
    if architecture != "annual_allocation_ar2":
        raise ValueError(f"Unsupported slow architecture: {architecture}")
    annual = experiment.allocation.annual
    weights = np.asarray(experiment.allocation.average_weights, dtype=float)
    for t in range(1, cell.n_periods):
        period = cell.periods[t]
        year = int(period.year)
        if year in annual.index and year - 1 in annual.index:
            drift[t] = weights[period.quarter - 1] * float(annual.loc[year] - annual.loc[year - 1])
    return drift


def _state_filter(
    q: np.ndarray,
    drift: np.ndarray,
    y: np.ndarray,
    constant: np.ndarray,
    h_loading: np.ndarray,
    psi: np.ndarray,
    sigma_v2: float,
    damping: float,
    period: float,
    tau2: float,
    omega: float,
):
    """Kalman filter for the reduced exact-N state representation.

    The slow-state law becomes the noisy observation

        hat_t - hat_{t-1} = Delta q_t - drift_t + noise_b,t,

    while the AR(2) stochastic cycle is the state transition.  This is exactly
    equivalent to applying both Gaussian state laws to ``bar=q-hat`` and
    ``hat``; no measurement error is introduced in total q.
    """
    T = len(q)
    vb = max(1e-10, omega * tau2)
    vh = max(1e-10, (1.0 - omega) * tau2)
    Fv, Qv, hv, P0v = state_augmentation(psi, sigma_v2)
    dimv = len(hv)
    dim = 2 + dimv
    phi1, phi2 = _cycle_coefficients(damping, period)

    F = np.zeros((dim, dim), dtype=float)
    F[:2, :2] = [[phi1, phi2], [1.0, 0.0]]
    F[2:, 2:] = Fv
    Q = np.zeros((dim, dim), dtype=float)
    Q[0, 0] = vh
    Q[2:, 2:] = Qv
    P0 = np.zeros((dim, dim), dtype=float)
    P0[:2, :2] = vh * _cycle_unit_cov(damping, period)
    P0[2:, 2:] = P0v

    mp = np.zeros((T, dim)); Pp = np.zeros((T, dim, dim))
    mf = np.zeros((T, dim)); Pf = np.zeros((T, dim, dim))
    eye = np.eye(dim); loglik = 0.0
    for t in range(T):
        if t == 0:
            mp[t] = 0.0; Pp[t] = force_pd(P0)
        else:
            mp[t] = F @ mf[t - 1]
            Pp[t] = force_pd(F @ Pf[t - 1] @ F.T + Q)

        pi_row = np.zeros(dim, dtype=float)
        pi_row[0] = h_loading[t]
        pi_row[2:] = hv
        if t == 0:
            obs = np.array([y[t] - constant[t]])
            H = pi_row[None, :]
            R = np.zeros((1, 1))
        else:
            slow_row = np.zeros(dim, dtype=float)
            slow_row[0], slow_row[1] = 1.0, -1.0
            obs = np.array([q[t] - q[t - 1] - drift[t], y[t] - constant[t]])
            H = np.vstack((slow_row, pi_row))
            R = np.diag((vb, 0.0))
        S = force_pd(H @ Pp[t] @ H.T + R)
        Sinv = np.linalg.inv(S); innovation = obs - H @ mp[t]
        sign, logdet = np.linalg.slogdet(S)
        if sign <= 0:
            return None
        loglik += -0.5 * (len(obs)*np.log(2*np.pi) + logdet + innovation @ Sinv @ innovation)
        K = Pp[t] @ H.T @ Sinv
        mf[t] = mp[t] + K @ innovation
        KH = K @ H
        Pf[t] = force_pd((eye - KH) @ Pp[t] @ (eye - KH).T + K @ R @ K.T)
    return F, mp, Pp, mf, Pf, float(loglik)


def _state_loglik(q, drift, y, constant, h_loading, psi, sigma_v2,
                  damping, period, tau2, omega):
    try:
        filtered = _state_filter(q, drift, y, constant, h_loading, psi, sigma_v2,
                                 damping, period, tau2, omega)
    except (ValueError, np.linalg.LinAlgError, FloatingPointError):
        return -np.inf
    return -np.inf if filtered is None else filtered[-1]


def _state_draw(
    rng: np.random.Generator,
    q: np.ndarray,
    drift: np.ndarray,
    y: np.ndarray,
    constant: np.ndarray,
    h_loading: np.ndarray,
    psi: np.ndarray,
    sigma_v2: float,
    damping: float,
    period: float,
    tau2: float,
    omega: float,
) -> np.ndarray:
    """FFBS draw for ``hat q`` with the exact identity imposed algebraically.

    The slow-state law becomes the noisy observation

        hat_t - hat_{t-1} = Delta q_t - drift_t + noise_b,t,

    while the AR(2) stochastic cycle is the state transition.  This is exactly
    equivalent to applying both Gaussian state laws to ``bar=q-hat`` and
    ``hat``; no measurement error is introduced in total q.
    """
    filtered = _state_filter(q, drift, y, constant, h_loading, psi, sigma_v2,
                             damping, period, tau2, omega)
    if filtered is None:
        raise np.linalg.LinAlgError("state filter covariance is not positive definite")
    F, mp, Pp, mf, Pf, _ = filtered
    T, dim = mf.shape

    state = np.zeros((T, dim), dtype=float)
    state[-1] = rng.multivariate_normal(mf[-1], force_pd(Pf[-1]))
    for t in range(T - 2, -1, -1):
        pred = force_pd(Pp[t + 1])
        gain = Pf[t] @ F.T @ np.linalg.inv(pred)
        mean = mf[t] + gain @ (state[t + 1] - F @ mf[t])
        covariance = force_pd(Pf[t] - gain @ pred @ gain.T)
        state[t] = rng.multivariate_normal(mean, covariance)
    return state[:, 0]


def _draw_beta_gls(rng, y, X, weighting, sigma2, prior_mean, prior_sd):
    prior_precision = np.diag(1.0 / np.square(prior_sd))
    WX = weighting.solve(X); Wy = weighting.solve(y)
    precision = prior_precision + X.T @ WX / sigma2
    covariance = np.linalg.inv(precision)
    mean = covariance @ (prior_precision @ prior_mean + X.T @ Wy / sigma2)
    return rng.multivariate_normal(mean, force_pd(covariance))


def _draw_lambda_gls(rng, cell, q, h, beta, weighting, sigma2, mean, sd):
    values = dict(zip(BASE_NAMES + ("theta",), beta))
    common = (
        values["intercept"] + values["alpha_b"] * cell.pi_lag
        + values["alpha_f"] * cell.epi + values["kappa_0"] * cell.x
        - values["theta"] * h
    )
    loading = values["theta"] * (q - h) * cell.x
    Wg = weighting.solve(loading)
    precision = 1.0 / sd**2 + float(loading @ Wg) / sigma2
    variance = 1.0 / precision
    post_mean = variance * (
        mean / sd**2 + float((cell.pi - common) @ Wg) / sigma2
    )
    return float(rng.normal(post_mean, np.sqrt(variance)))


def _bounded_value(z, lower, upper):
    return float(lower + (upper - lower) * expit(z))


def _bounded_z(value, lower, upper):
    return float(logit((value - lower) / (upper - lower)))


def _slice_scalar(rng, current, logtarget, width=1.0, max_steps=50):
    """Univariate stepping-out slice update on an unconstrained coordinate."""
    height = logtarget(current) + np.log(rng.uniform())
    left = current - width * rng.uniform(); right = left + width
    jl = int(rng.integers(max_steps + 1)); jr = max_steps - jl
    while jl > 0 and logtarget(left) > height:
        left -= width; jl -= 1
    while jr > 0 and logtarget(right) > height:
        right += width; jr -= 1
    for _ in range(500):
        proposal = rng.uniform(left, right)
        if logtarget(proposal) >= height:
            return float(proposal)
        if proposal < current: left = proposal
        else: right = proposal
    raise RuntimeError("slice sampler failed to find a point on the slice")


def _cycle_logtarget(zd, zp, h, variance, cfg):
    lo_d, hi_d = map(float, cfg["cycle_damping_bounds"])
    lo_p, hi_p = map(float, cfg["cycle_period_bounds_quarters"])
    damping = _bounded_value(zd, lo_d, hi_d); period = _bounded_value(zp, lo_p, hi_p)
    ss, logdet = _cycle_sumsq(h, damping, period)
    ud, up = expit(zd), expit(zp)
    a, b = map(float, cfg["cycle_damping_prior"])
    value = -0.5 * logdet - 0.5 * ss / variance + beta_dist.logpdf(ud, a, b)
    value += norm.logpdf(period, float(cfg["cycle_period_prior_mean"]), float(cfg["cycle_period_prior_sd"]))
    value += np.log(ud) + np.log1p(-ud) + np.log(up) + np.log1p(-up)
    return float(value)


def _update_state_parameters(rng, q, h, drift, damping, period, omega, tau2, s_q, cfg):
    lo_d, hi_d = map(float, cfg["cycle_damping_bounds"])
    lo_p, hi_p = map(float, cfg["cycle_period_bounds_quarters"])
    zd, zp = _bounded_z(damping, lo_d, hi_d), _bounded_z(period, lo_p, hi_p)
    accept_d = accept_p = False
    candidate = zd + rng.normal(0.0, float(cfg["cycle_damping_mh_sd"]))
    if np.log(rng.uniform()) < _cycle_logtarget(candidate, zp, h, (1-omega)*tau2, cfg) - _cycle_logtarget(zd, zp, h, (1-omega)*tau2, cfg):
        zd = candidate; accept_d = True
    candidate = zp + rng.normal(0.0, float(cfg["cycle_period_mh_sd"]))
    if np.log(rng.uniform()) < _cycle_logtarget(zd, candidate, h, (1-omega)*tau2, cfg) - _cycle_logtarget(zd, zp, h, (1-omega)*tau2, cfg):
        zp = candidate; accept_p = True
    damping, period = _bounded_value(zd, lo_d, hi_d), _bounded_value(zp, lo_p, hi_p)

    bar = q - h
    rb = np.diff(bar) - drift[1:]
    cycle_ss, _ = _cycle_sumsq(h, damping, period)
    a, b = map(float, cfg["omega_prior"])

    def omega_target(z):
        value = float(expit(z)); T = len(q)
        out = -(T-1)/2*np.log(value) - float(rb @ rb)/(2*value*tau2)
        out += -T/2*np.log1p(-value) - cycle_ss/(2*(1-value)*tau2)
        out += beta_dist.logpdf(value, a, b) + np.log(value) + np.log1p(-value)
        return float(out)

    old_omega = omega
    z_omega = _slice_scalar(
        rng, logit(omega), omega_target,
        width=float(cfg.get("omega_slice_width", cfg["omega_mh_sd"])),
    )
    omega = float(expit(z_omega)); accept_o = bool(omega != old_omega)
    scale = 2.0 * (float(cfg["tau2_scale_fraction"]) * float(s_q)) ** 2
    scaled = float(rb @ rb) / omega + cycle_ss / (1.0 - omega)
    tau2 = _draw_ig(rng, float(cfg["tau2_prior_shape"]) + (2*len(q)-1)/2, scale + 0.5*scaled)
    return bar, damping, period, omega, tau2, accept_d, accept_p, accept_o


def _collapsed_state_parameters(
    rng, q, drift, y, constant, h_loading, psi, sigma2,
    damping, period, omega, tau2, s_q, cfg,
):
    """Update state-law parameters with the latent state integrated out."""
    lo_d, hi_d = map(float, cfg["cycle_damping_bounds"])
    lo_p, hi_p = map(float, cfg["cycle_period_bounds_quarters"])
    a_o, b_o = map(float, cfg["omega_prior"])
    shape = float(cfg["tau2_prior_shape"])
    scale = 2.0 * (float(cfg["tau2_scale_fraction"]) * float(s_q)) ** 2

    def ll(d, p, o, t2):
        return _state_loglik(q, drift, y, constant, h_loading, psi, sigma2, d, p, t2, o)

    def omega_target(z):
        o = float(expit(z)); value = ll(damping, period, o, tau2)
        value += beta_dist.logpdf(o, a_o, b_o) + np.log(o) + np.log1p(-o)
        return float(value)

    omega = float(expit(_slice_scalar(
        rng, logit(omega), omega_target,
        width=float(cfg.get("omega_slice_width", cfg["omega_mh_sd"])),
    )))

    def tau_target(z):
        t2 = float(np.exp(np.clip(z, -30.0, 30.0)))
        # IG(shape, scale) density plus ds/dlog(s)=s Jacobian.
        return float(ll(damping, period, omega, t2) - shape*z - scale/t2)

    tau2 = float(np.exp(_slice_scalar(
        rng, np.log(tau2), tau_target,
        width=float(cfg.get("tau2_slice_width", 0.5)),
    )))

    zd, zp = _bounded_z(damping, lo_d, hi_d), _bounded_z(period, lo_p, hi_p)

    def damping_target(z):
        d = _bounded_value(z, lo_d, hi_d); u = expit(z)
        aa, bb = map(float, cfg["cycle_damping_prior"])
        return float(ll(d, period, omega, tau2) + beta_dist.logpdf(u, aa, bb)
                     + np.log(u) + np.log1p(-u))

    def period_target(z):
        p = _bounded_value(z, lo_p, hi_p); u = expit(z)
        return float(ll(damping, p, omega, tau2)
                     + norm.logpdf(p, float(cfg["cycle_period_prior_mean"]), float(cfg["cycle_period_prior_sd"]))
                     + np.log(u) + np.log1p(-u))

    damping = _bounded_value(_slice_scalar(rng, zd, damping_target,
                              width=float(cfg.get("cycle_damping_slice_width", .6))), lo_d, hi_d)
    period = _bounded_value(_slice_scalar(rng, zp, period_target,
                            width=float(cfg.get("cycle_period_slice_width", .8))), lo_p, hi_p)
    return damping, period, omega, tau2


def fit_joint_ma3(
    experiment: ExperimentData,
    cell: CellData,
    spec: ModelSpec,
    cfg: dict,
    sampling: dict,
    architecture: str,
    seed: int,
) -> JointMA3Fit:
    priors, means, sds = _prior_maps(spec, cell, {"priors": cfg["priors"]})
    names = spec.coefficient_names + (("lambda",) if spec.free_lambda else ())
    pmean = np.array([means[n] for n in spec.coefficient_names], dtype=float)
    psd = np.array([sds[n] for n in spec.coefficient_names], dtype=float)
    C = int(sampling["chains"]); iterations = int(sampling["iterations"])
    warmup = int(sampling["warmup"]); thin = int(sampling["thin"])
    D = (iterations - warmup + thin - 1)//thin; T = cell.n_periods
    state_cfg = cfg["competition"]; inf_cfg = cfg["inflation"]
    ma_order = int(inf_cfg["ma_order"])
    draws = np.zeros((C,D,len(names))); sigma = np.zeros((C,D)); psi_out = np.zeros((C,D,ma_order))
    nt = np.zeros((C,D,T)); nb = np.zeros_like(nt); nh = np.zeros_like(nt)
    om = np.zeros((C,D)); tau = np.zeros((C,D)); damp = np.zeros((C,D)); per = np.zeros((C,D))
    psi_acc = np.zeros(C); dacc=np.zeros(C); pacc=np.zeros(C); oacc=np.zeros(C)
    drift = average_allocation_drift(experiment, cell, architecture)

    for chain in range(C):
        rng = np.random.default_rng(seed + 7919*chain)
        beta = pmean.copy(); lam = float(cfg["priors"]["lambda_mean"])
        sigma2 = priors.sigma_pi_b/(priors.ig_shape-1.0)
        qref = experiment.mean_q(cell); s_q = cell.s_q
        scale = 2.0*(float(state_cfg["tau2_scale_fraction"])*s_q)**2
        tau2 = scale/(float(state_cfg["tau2_prior_shape"])-1.0)
        a,b = map(float,state_cfg["omega_prior"]); omega=a/(a+b)
        damping=float(state_cfg["cycle_damping_initial"]); period=float(state_cfg["cycle_period_initial"])
        h=np.zeros(T); psi=np.zeros(ma_order); weighting=MAWeighting(psi,T)
        psi_prior=(PsiPrior(np.full(ma_order,float(inf_cfg["psi_prior_mean"])),
                            np.full(ma_order,float(inf_cfg["psi_prior_sd"]))) if ma_order else None)
        proposal=(AdaptiveRandomWalk(ma_order,init_scale=float(inf_cfg["psi_initial_scale"])) if ma_order else None)
        save=0
        for it in range(iterations):
            q=experiment.draw_q(rng,cell); bar=q-h
            X=_design(spec,cell,q,h,lam)
            beta=_draw_beta_gls(rng,cell.pi,X,weighting,sigma2,pmean,psd)
            if spec.free_lambda:
                lam=_draw_lambda_gls(rng,cell,q,h,beta,weighting,sigma2,float(cfg["priors"]["lambda_mean"]),float(cfg["priors"]["lambda_sd"]))
                X=_design(spec,cell,q,h,lam)
            resid=cell.pi-X@beta
            sigma2=_draw_ig(rng,priors.ig_shape+T/2,priors.sigma_pi_b+0.5*weighting.quadratic_form(resid))

            values=dict(zip(spec.coefficient_names,beta))
            theta=float(values.get("theta",0.0)); delta=float(values.get("delta_s",0.0))
            if spec.lambda_fixed is not None: delta=float(spec.lambda_fixed)*theta
            if spec.free_lambda: delta=float(lam)*theta
            base=(values.get("intercept",0.0)+values.get("alpha_b",0.0)*cell.pi_lag
                  +values.get("alpha_f",0.0)*cell.epi+values.get("kappa_0",0.0)*cell.x
                  +delta*q*cell.x)
            hloading=-(theta+delta*cell.x)
            collapsed_stride = int(state_cfg.get("collapsed_stride", 10))
            if it % collapsed_stride == 0:
                damping,period,omega,tau2=_collapsed_state_parameters(
                    rng,q,drift,cell.pi,base,hloading,psi,sigma2,
                    damping,period,omega,tau2,s_q,state_cfg)
            h=_state_draw(rng,q,drift,cell.pi,base,hloading,psi,sigma2,damping,period,tau2,omega)
            if it % collapsed_stride == 0:
                bar=q-h; ad=ap=ao=True
            else:
                bar,damping,period,omega,tau2,ad,ap,ao=_update_state_parameters(
                    rng,q,h,drift,damping,period,omega,tau2,s_q,state_cfg)
            dacc[chain]+=ad; pacc[chain]+=ap; oacc[chain]+=ao

            X=_design(spec,cell,q,h,lam); resid=cell.pi-X@beta
            if ma_order:
                old_accept=proposal.n_accept
                psi,weighting=sample_psi(psi,resid,sigma2,prior=psi_prior,proposal=proposal,rng=rng,
                                         n_steps=int(inf_cfg["psi_steps_per_sweep"]),weighting=weighting)
                psi_acc[chain]+=proposal.n_accept-old_accept
                if it==warmup-1: proposal.freeze()
            if it>=warmup and (it-warmup)%thin==0:
                draws[chain,save]=list(beta)+([lam] if spec.free_lambda else [])
                sigma[chain,save]=np.sqrt(sigma2); psi_out[chain,save]=psi
                nt[chain,save]=q; nh[chain,save]=h; nb[chain,save]=q-h
                om[chain,save]=omega; tau[chain,save]=np.sqrt(tau2)
                damp[chain,save]=damping; per[chain,save]=period; save+=1

    rhat={name:float(az.rhat(draws[:,:,i],method="rank")) for i,name in enumerate(names)}
    for j in range(ma_order): rhat[f"psi_{j+1}"]=float(az.rhat(psi_out[:,:,j],method="rank"))
    rhat.update(sigma_pi=float(az.rhat(sigma,method="rank")),omega=float(az.rhat(om,method="rank")),
                tau=float(az.rhat(tau,method="rank")),cycle_damping=float(az.rhat(damp,method="rank")),
                cycle_period=float(az.rhat(per,method="rank")))
    ess_bulk={name:float(az.ess(draws[:,:,i],method="bulk")) for i,name in enumerate(names)}
    ess_tail={name:float(az.ess(draws[:,:,i],method="tail",prob=(.05,.95))) for i,name in enumerate(names)}
    extra_series = {"sigma_pi": sigma, "omega": om, "tau": tau,
                    "cycle_damping": damp, "cycle_period": per}
    extra_series.update({f"psi_{j+1}": psi_out[:,:,j] for j in range(ma_order)})
    for name, values in extra_series.items():
        ess_bulk[name] = float(az.ess(values, method="bulk"))
        ess_tail[name] = float(az.ess(values, method="tail", prob=(.05,.95)))
    diagnostics={"rhat":rhat,"max_rhat":max(rhat.values()),"ess_bulk":ess_bulk,"ess_tail":ess_tail,
                 "exact_identity_error":float(np.max(np.abs(nt-nb-nh))),
                 "psi_acceptance":((psi_acc/(iterations*int(inf_cfg["psi_steps_per_sweep"]))).tolist()
                                    if ma_order else []),
                 "cycle_damping_acceptance":(dacc/iterations).tolist(),"cycle_period_acceptance":(pacc/iterations).tolist(),
                 "omega_acceptance":(oacc/iterations).tolist(),"slow_architecture":architecture}
    prior_mean={n:(float(cfg["priors"]["lambda_mean"]) if n=="lambda" else float(means[n])) for n in names}
    prior_sd={n:(float(cfg["priors"]["lambda_sd"]) if n=="lambda" else float(sds[n])) for n in names}
    fit=ModelFit(spec,tuple(map(str,cell.periods)),names,draws,sigma,np.zeros_like(sigma),nt,nb,nh,om,tau,damp,per,prior_mean,prior_sd,diagnostics)
    return JointMA3Fit(fit,psi_out,architecture,drift)
