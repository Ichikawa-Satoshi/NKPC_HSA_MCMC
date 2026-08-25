"""Controlled exact-N bridge sampler for theta identification.

The allocation posterior is common across cells.  In cut cells, saved Nbar/Nhat
draws are sampled without inflation.  In joint-split cells, an elliptical-slice
step conditions the exact decomposition on the NKPC while preserving
N = Nbar + Nhat at every draw.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import arviz as az
import numpy as np

from nkpc_hsa.phillips.state import _draw_ig
from nkpc_hsa.report_models.engine import build_priors
from tests.hsa_exact_n_decomposition.functions import (
    ExactData,
    StateFit,
    _draw_rho,
    _omega_logtarget,
)
from tests.hsa_lambda_dynamic.functions import (
    _draw_coefficients,
    _draw_lambda,
    _draw_phi,
    _elliptical_slice,
    _inflation_loglik,
    _prior_maps,
    _sample_zero_precision,
    _solve_precision,
    _tridiagonal_cholesky,
)
from scipy.special import expit, logit


BASE_NAMES = ("intercept", "alpha_b", "alpha_f", "kappa_0", "theta_0")
CELL_SPECS = {
    "cut_fixed6": ("cut", "fixed"),
    "cut_free": ("cut", "free"),
    "joint_fixed6": ("joint", "fixed"),
    "joint_free": ("joint", "free"),
}


@dataclass
class BridgeFit:
    cell: str
    coupling: str
    lambda_mode: str
    names: tuple[str, ...]
    draws: np.ndarray
    sigma_pi: np.ndarray
    phi: np.ndarray
    n_total: np.ndarray
    nbar: np.ndarray
    nhat: np.ndarray
    omega: np.ndarray
    tau: np.ndarray
    rho: np.ndarray
    diagnostics: dict
    prior_mean: dict[str, float]
    prior_sd: dict[str, float]


def _design(data, nbar, nhat, lam):
    return np.column_stack([
        np.ones(data.n_periods),
        data.pi_lag,
        data.epi,
        data.x,
        lam * data.x * nbar - nhat,
    ])


def _mu(data, beta_vec, nbar, nhat, lam):
    return _design(data, nbar, nhat, lam) @ beta_vec


def _conditional_exact_h(total, rho, tau2, omega, rng):
    """Gaussian p(Nhat | exact total N, rho, omega, tau2)."""
    T = len(total)
    vb = max(1e-10, omega * tau2)
    vh = max(1e-10, (1.0 - omega) * tau2)
    diag = np.full(T, 2.0 / vb + (1.0 + rho**2) / vh)
    diag[0] = 1.0 / vb + 1.0 / vh
    diag[-1] = 1.0 / vb + 1.0 / vh
    off = np.full(T - 1, -1.0 / vb - rho / vh)
    dn = np.diff(total)
    rhs = np.zeros(T)
    rhs[0] = -dn[0] / vb
    rhs[-1] = dn[-1] / vb
    if T > 2:
        rhs[1:-1] = (dn[:-1] - dn[1:]) / vb
    ld, ls = _tridiagonal_cholesky(diag, off)
    mean = _solve_precision(ld, ls, rhs)
    return mean, lambda: _sample_zero_precision(ld, ls, rng)


def _coefficient_setup(data, cfg, lambda_mode):
    cp = cfg["coefficient_priors"]
    priors = build_priors(data, coef_scale=float(cp["coefficient_scale"]), hybrid=True)
    _, means, sds = _prior_maps(
        "hsa_static", priors, float(cp["lambda_mean"]), float(cp["lambda_sd"])
    )
    pmean = np.array([means[n] for n in BASE_NAMES])
    psd = np.array([sds[n] for n in BASE_NAMES])
    names = BASE_NAMES + (("lambda",) if lambda_mode == "free" else ())
    return priors, means, sds, pmean, psd, names


def _fit_cut(exact: ExactData, states: StateFit, cell: str, cfg: dict, sampling: dict, seed: int):
    data = exact.case
    coupling, lambda_mode = CELL_SPECS[cell]
    priors, means, sds, pmean, psd, names = _coefficient_setup(data, cfg, lambda_mode)
    cp = cfg["coefficient_priors"]
    iterations = int(sampling["model_iterations"])
    warmup = int(sampling["model_warmup"])
    thin = int(sampling["model_thin"])
    chains = int(sampling["chains"])
    nsave = (iterations - warmup + thin - 1) // thin
    shape = (chains, nsave)
    draws = np.zeros(shape + (len(names),))
    sig = np.zeros(shape); phis = np.zeros(shape)
    nt = np.zeros(shape + (data.n_periods,)); nb = np.zeros_like(nt); nh = np.zeros_like(nt)
    om = np.zeros(shape); tau = np.zeros(shape); rho = np.zeros(shape)
    accept = np.zeros(chains)
    for ch in range(chains):
        rng = np.random.default_rng(seed + 7919 * ch)
        beta_vec = pmean.copy()
        lam = float(cp["lambda_fixed"] if lambda_mode == "fixed" else cp["lambda_mean"])
        phi = float(cp["phi_mean"])
        sigma2 = priors.sigma_pi_b / (priors.ig_shape - 1.0)
        save = 0
        for it in range(iterations):
            j = int(rng.integers(states.nbar.shape[1]))
            nbar = states.nbar[ch, j]
            nhat = states.nhat[ch, j]
            total = states.n_total[ch, j]
            X = _design(data, nbar, nhat, lam)
            beta_vec, sigma2 = _draw_coefficients(
                rng, data.pi, X, phi, pmean, psd, sigma2,
                priors.ig_shape, priors.sigma_pi_b,
            )
            if lambda_mode == "free":
                beta = dict(zip(BASE_NAMES, beta_vec))
                lam = _draw_lambda(
                    rng, "hsa_static", data, beta, nbar, nhat, phi, sigma2,
                    float(cp["lambda_mean"]), float(cp["lambda_sd"]),
                )
            mu = _mu(data, beta_vec, nbar, nhat, lam)
            phi, ok = _draw_phi(
                rng, data.pi, mu, sigma2, phi,
                float(cp["phi_mean"]), float(cp["phi_sd"]),
            )
            accept[ch] += ok
            if it >= warmup and (it - warmup) % thin == 0:
                draws[ch, save] = list(beta_vec) + ([lam] if lambda_mode == "free" else [])
                sig[ch, save] = np.sqrt(sigma2); phis[ch, save] = phi
                nt[ch, save] = total; nb[ch, save] = nbar; nh[ch, save] = nhat
                om[ch, save] = states.omega[ch, j]; tau[ch, save] = states.tau[ch, j]
                rho[ch, save] = states.rho[ch, j]
                save += 1
    diagnostics = _diagnostics(names, draws, sig, phis, nt, nb, nh)
    diagnostics["phi_acceptance"] = (accept / iterations).tolist()
    return BridgeFit(cell, coupling, lambda_mode, names, draws, sig, phis, nt, nb, nh, om, tau, rho,
                     diagnostics, {n: means[n] for n in names}, {n: sds[n] for n in names})


def _fit_joint(exact: ExactData, cell: str, cfg: dict, sampling: dict, seed: int):
    data = exact.case
    coupling, lambda_mode = CELL_SPECS[cell]
    priors, means, sds, pmean, psd, names = _coefficient_setup(data, cfg, lambda_mode)
    cp = cfg["coefficient_priors"]; sp = cfg["state_priors"]
    iterations = int(sampling["model_iterations"])
    warmup = int(sampling["model_warmup"])
    thin = int(sampling["model_thin"])
    chains = int(sampling["chains"])
    nsave = (iterations - warmup + thin - 1) // thin
    shape = (chains, nsave)
    draws = np.zeros(shape + (len(names),))
    sig = np.zeros(shape); phis = np.zeros(shape)
    nt = np.zeros(shape + (data.n_periods,)); nb = np.zeros_like(nt); nh = np.zeros_like(nt)
    om = np.zeros(shape); taus = np.zeros(shape); rhos = np.zeros(shape)
    phi_accept = np.zeros(chains); omega_accept = np.zeros(chains); ess_eval = np.zeros(chains)
    tau_scale = 2.0 * (float(sp["tau_scale_fraction"]) * data.s_N) ** 2
    for ch in range(chains):
        rng = np.random.default_rng(seed + 7919 * ch)
        beta_vec = pmean.copy()
        lam = float(cp["lambda_fixed"] if lambda_mode == "fixed" else cp["lambda_mean"])
        phi = float(cp["phi_mean"])
        sigma2 = priors.sigma_pi_b / (priors.ig_shape - 1.0)
        omega = float(sp["omega_a"]) / (float(sp["omega_a"]) + float(sp["omega_b"]))
        tau2 = tau_scale / (float(sp["tau_shape"]) - 1.0)
        rho = float(sp["rho_mean"])
        total = exact.allocation.draw_path(rng)[exact.allocation_positions]
        total -= total.mean()
        mean_h, sample_h = _conditional_exact_h(total, rho, tau2, omega, rng)
        h = mean_h + sample_h(); save = 0
        for it in range(iterations):
            # The allocation-posterior mean path is identical and external in all cells.
            total = exact.allocation.draw_path(rng)[exact.allocation_positions]
            total -= total.mean()
            mean_h, sample_h = _conditional_exact_h(total, rho, tau2, omega, rng)

            def state_loglik(candidate):
                bar_candidate = total - candidate
                return _inflation_loglik(
                    data.pi, _mu(data, beta_vec, bar_candidate, candidate, lam),
                    sigma2, phi, include_constants=False,
                )

            h, evaluations = _elliptical_slice(h, mean_h, sample_h, state_loglik, rng)
            ess_eval[ch] += evaluations
            bar = total - h
            X = _design(data, bar, h, lam)
            beta_vec, sigma2 = _draw_coefficients(
                rng, data.pi, X, phi, pmean, psd, sigma2,
                priors.ig_shape, priors.sigma_pi_b,
            )
            if lambda_mode == "free":
                beta = dict(zip(BASE_NAMES, beta_vec))
                lam = _draw_lambda(
                    rng, "hsa_static", data, beta, bar, h, phi, sigma2,
                    float(cp["lambda_mean"]), float(cp["lambda_sd"]),
                )
            mu = _mu(data, beta_vec, bar, h, lam)
            phi, ok = _draw_phi(
                rng, data.pi, mu, sigma2, phi,
                float(cp["phi_mean"]), float(cp["phi_sd"]),
            )
            phi_accept[ch] += ok
            rho = _draw_rho(
                rng, h, (1.0 - omega) * tau2,
                float(sp["rho_mean"]), float(sp["rho_sd"]),
                float(sp["rho_lower"]), float(sp["rho_upper"]),
            )
            rb = np.diff(bar); rh = h[1:] - rho * h[:-1]
            proposal = logit(omega) + rng.normal(0.0, 0.16)
            old_target = _omega_logtarget(
                logit(omega), rb, rh, h[0], rho, tau2,
                float(sp["omega_a"]), float(sp["omega_b"]),
            )
            new_target = _omega_logtarget(
                proposal, rb, rh, h[0], rho, tau2,
                float(sp["omega_a"]), float(sp["omega_b"]),
            )
            if np.log(rng.uniform()) < new_target - old_target:
                omega = float(expit(proposal)); omega_accept[ch] += 1
            ssh = float(rh @ rh) + max(1e-8, 1.0 - rho**2) * h[0] ** 2
            scaled = float(rb @ rb) / omega + ssh / (1.0 - omega)
            tau2 = _draw_ig(
                rng, float(sp["tau_shape"]) + (2 * data.n_periods - 1) / 2,
                tau_scale + 0.5 * scaled,
            )
            if it >= warmup and (it - warmup) % thin == 0:
                draws[ch, save] = list(beta_vec) + ([lam] if lambda_mode == "free" else [])
                sig[ch, save] = np.sqrt(sigma2); phis[ch, save] = phi
                nt[ch, save] = total; nb[ch, save] = bar; nh[ch, save] = h
                om[ch, save] = omega; taus[ch, save] = np.sqrt(tau2); rhos[ch, save] = rho
                save += 1
    diagnostics = _diagnostics(names, draws, sig, phis, nt, nb, nh)
    state_rhat = {
        "omega": float(az.rhat(om, method="rank")),
        "tau": float(az.rhat(taus, method="rank")),
        "rho": float(az.rhat(rhos, method="rank")),
    }
    diagnostics["state_rhat"] = state_rhat
    diagnostics["max_rhat"] = max(diagnostics["max_rhat"], *state_rhat.values())
    diagnostics.update(
        phi_acceptance=(phi_accept / iterations).tolist(),
        omega_acceptance=(omega_accept / iterations).tolist(),
        mean_state_ess_evaluations=(ess_eval / iterations).tolist(),
    )
    return BridgeFit(cell, coupling, lambda_mode, names, draws, sig, phis, nt, nb, nh, om, taus, rhos,
                     diagnostics, {n: means[n] for n in names}, {n: sds[n] for n in names})


def _diagnostics(names, draws, sig, phis, nt, nb, nh):
    rhat = {n: float(az.rhat(draws[:, :, i], method="rank")) for i, n in enumerate(names)}
    rhat["sigma_pi"] = float(az.rhat(sig, method="rank"))
    rhat["phi"] = float(az.rhat(phis, method="rank"))
    delta = draws[:, :, names.index("theta_0")] * (
        draws[:, :, names.index("lambda")] if "lambda" in names else 6.0
    )
    rhat["delta"] = float(az.rhat(delta, method="rank"))
    return {
        "rhat": rhat,
        "max_rhat": max(rhat.values()),
        "exact_identity_error": float(np.max(np.abs(nt - nb - nh))),
    }


def fit_bridge(exact, states, cell, cfg, sampling, seed):
    coupling, _ = CELL_SPECS[cell]
    if coupling == "cut":
        return _fit_cut(exact, states, cell, cfg, sampling, seed)
    return _fit_joint(exact, cell, cfg, sampling, seed)


def save_fit(path: Path, fit: BridgeFit):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path, cell=fit.cell, coupling=fit.coupling, lambda_mode=fit.lambda_mode,
        names=fit.names, draws=fit.draws, sigma_pi=fit.sigma_pi, phi=fit.phi,
        n_total=fit.n_total, nbar=fit.nbar, nhat=fit.nhat,
        omega=fit.omega, tau=fit.tau, rho=fit.rho,
    )


def summarize_fit(fit: BridgeFit):
    flat = fit.draws.reshape(-1, fit.draws.shape[-1])
    theta = flat[:, fit.names.index("theta_0")]
    if "lambda" in fit.names:
        lam = flat[:, fit.names.index("lambda")]
    else:
        lam = np.full_like(theta, 6.0)
    delta = theta * lam

    def summary(v):
        return {
            "mean": float(np.mean(v)), "sd": float(np.std(v, ddof=1)),
            "q2.5": float(np.percentile(v, 2.5)), "q97.5": float(np.percentile(v, 97.5)),
            "p_positive": float(np.mean(v > 0)),
        }

    nb_path = fit.nbar.reshape(-1, fit.nbar.shape[-1]).mean(0)
    nh_path = fit.nhat.reshape(-1, fit.nhat.shape[-1]).mean(0)
    nt_path = fit.n_total.reshape(-1, fit.n_total.shape[-1]).mean(0)
    return {
        "theta": summary(theta), "lambda": summary(lam), "delta": summary(delta),
        "state": {
            "nbar_path_sd": float(np.std(nb_path)),
            "nhat_path_sd": float(np.std(nh_path)),
            "corr_nhat_total": float(np.corrcoef(nh_path, nt_path)[0, 1]),
            "omega_mean": float(np.mean(fit.omega)),
            "omega_interval": np.percentile(fit.omega, [2.5, 97.5]).tolist(),
            "rho_mean": float(np.mean(fit.rho)),
        },
        "diagnostics": fit.diagnostics,
    }
