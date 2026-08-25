"""Joint state-space Gibbs/FFBS engine for the report's four-case, five-model design.

This module implements exactly the specification in
``docs/report/estimation/edstimation.tex``:

Latent competition state ``s_t = [Ntilde_t, Nhat_t]`` with

    Ntilde_t = Ntilde_{t-1} + eta_bar_t                       (random walk)
    Nhat_t   = rho * Nhat_{t-1} + lambda_E * gE_t + eta_hat_t  (AR(1), gE only in Case 4)

Inflation measurement (forward-looking NKPC, no lagged inflation):

    pi_t = alpha E_t pi_{t+1} + kappa_0 x_t
           + delta x_t Ntilde_t - theta_0 Nhat_t + gamma Ntilde_t Nhat_t + eps_t

Sign convention: the stationary competition state enters with a MINUS sign, so
the HSA-consistent direction is theta_0 > 0 (more competition -> lower
inflation).  The full theory-consistent region is delta>0, theta_0>0, gamma>0.

Competition measurement: n_t^obs = Ntilde_t + Nhat_t + nu_t (Cases 1-3), or an
exact Q4 anchor (Case 4). Missing competition rows are omitted from the Kalman
update (mixed frequency).

Five nested models select which HSA coefficients are free:

    Model 0 (CES):            delta = theta_0 = gamma = 0
    Model 1 (slope):          delta free
    Model 2 (direct):         theta_0 free
    Model 3 (dynamic):        theta_0, gamma free
    Model 4 (joint):          delta, theta_0, gamma free

Models 0-2 are conditionally linear-Gaussian and use a single 2-D FFBS draw.
Models 3-4 contain the bilinear term gamma*Ntilde*Nhat and use blocked
conditional FFBS (alternating 1-D draws of Ntilde|Nhat and Nhat|Ntilde).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from nkpc_hsa.gibbs.common.joint_ffbs import force_pd
from nkpc_hsa.phillips.state import _draw_ig, _normal_regression_draw, _truncated_regression_draw

# Column order of the full inflation design. A model uses a subset of these.
FULL_COEFFS = ("intercept", "alpha", "alpha_b", "kappa_0", "delta", "theta_0", "gamma", "c_ctrl")

# Baseline (always-present) coefficients: report-faithful (forward-looking only)
# vs. hybrid (adds an intercept and a lagged-inflation term pi_{t-1}).
BASE_FAITHFUL = ("alpha", "kappa_0")
BASE_HYBRID = ("intercept", "alpha", "alpha_b", "kappa_0")
HSA_FREE: dict[int, tuple[str, ...]] = {
    0: (), 1: ("delta",), 2: ("theta_0",), 3: ("theta_0", "gamma"),
    4: ("delta", "theta_0", "gamma"),
}


def coeff_names(model: int, hybrid: bool = False, control: bool = False) -> tuple[str, ...]:
    base = (BASE_HYBRID if hybrid else BASE_FAITHFUL) + HSA_FREE[model]
    return base + (("c_ctrl",) if control else ())


# Report-faithful free-coefficient sets (kept for backward compatibility).
MODEL_FREE: dict[int, tuple[str, ...]] = {m: coeff_names(m, False) for m in range(5)}

# A Q4 anchor variance for Case 4 (observed value reproduced almost exactly).
ANCHOR_REL_VAR = 1.0e-6

# A Q4 anchor variance for Case 4 (observed value reproduced almost exactly).
ANCHOR_REL_VAR = 1.0e-6


@dataclass(frozen=True)
class CaseData:
    """One estimable (case, inflation, forcing, competition-variant) sample."""

    case: int
    label: str
    periods: object  # pd.PeriodIndex
    pi: np.ndarray
    epi: np.ndarray
    x: np.ndarray
    n_obs: np.ndarray          # centered competition observation, NaN where missing
    exact_anchor: bool         # Case 4: Q4 values are exact anchors
    gE: np.ndarray | None      # establishment growth (demeaned), Case 4 only
    s_x: float
    s_N: float
    s_pi: float
    s_E: float | None
    pi_lag: np.ndarray | None = None   # pi_{t-1}, used only by the hybrid NKPC
    control: np.ndarray | None = None  # exogenous inflation control (e.g. oil), demeaned
    control_name: str | None = None

    @property
    def n_periods(self) -> int:
        return self.pi.size


@dataclass(frozen=True)
class Priors:
    """Prior specification derived from robust scales (edstimation.tex, Sec. Priors)."""

    alpha_mean: float
    alpha_sd: float
    kappa0_sd: float
    delta_sd: float
    theta0_sd: float
    gamma_sd: float
    rho_mean: float
    rho_sd: float
    lambda_sd: float
    sigma_pi_b: float          # IG scale numerator target: 2*(0.75 s_pi)^2
    sigma_bar_b: float
    sigma_hat_b: float
    sigma_nu_b: float
    ntilde0_sd: float
    ig_shape: float = 3.0
    intercept_sd: float = 5.0
    alpha_b_mean: float = 0.5
    alpha_b_sd: float = 0.5
    control_sd: float = 10.0

    def coeff_prior(self, names: tuple[str, ...]) -> tuple[np.ndarray, np.ndarray]:
        mean = {"intercept": 0.0, "alpha": self.alpha_mean, "alpha_b": self.alpha_b_mean,
                "kappa_0": 0.0, "delta": 0.0, "theta_0": 0.0, "gamma": 0.0, "c_ctrl": 0.0}
        sd = {
            "intercept": self.intercept_sd,
            "alpha": self.alpha_sd,
            "alpha_b": self.alpha_b_sd,
            "kappa_0": self.kappa0_sd,
            "delta": self.delta_sd,
            "theta_0": self.theta0_sd,
            "gamma": self.gamma_sd,
            "c_ctrl": self.control_sd,
        }
        return (
            np.array([mean[n] for n in names], float),
            np.array([sd[n] for n in names], float),
        )


def build_priors(data: CaseData, *, coef_scale: float = 0.2, var_scale: float = 1.0,
                 hybrid: bool = False) -> Priors:
    s_x, s_N, s_pi = data.s_x, data.s_N, data.s_pi
    s_E = data.s_E if data.s_E is not None else s_N
    return Priors(
        # Hybrid adds pi_{t-1}: relax the forward-looking prior so alpha and
        # alpha_b share the (strongly identified) inflation persistence.
        alpha_mean=0.5 if hybrid else 0.9,
        alpha_sd=0.5 if hybrid else 0.2,
        kappa0_sd=0.5 / s_x,
        intercept_sd=2.0 * s_pi,
        alpha_b_mean=0.5,
        alpha_b_sd=0.5,
        delta_sd=coef_scale / (s_x * s_N),
        theta0_sd=coef_scale / s_N,
        gamma_sd=coef_scale / (s_N ** 2),
        rho_mean=0.5,
        rho_sd=0.25,
        lambda_sd=coef_scale * s_N / s_E,
        sigma_pi_b=2.0 * (0.75 * var_scale * s_pi) ** 2,
        sigma_bar_b=2.0 * (0.05 * var_scale * s_N) ** 2,
        sigma_hat_b=2.0 * (0.20 * var_scale * s_N) ** 2,
        sigma_nu_b=2.0 * (0.10 * var_scale * s_N) ** 2,
        ntilde0_sd=2.0 * s_N,
        control_sd=(5.0 / _robust_scale(data.control) if data.control is not None else 10.0),
    )


def _robust_scale(values: np.ndarray) -> float:
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    s = float(np.subtract(*np.quantile(v, [0.75, 0.25])) / 1.349)
    return s if np.isfinite(s) and s > 0 else 1.0


# --------------------------------------------------------------------------- #
# AR(2) helpers for the stationary competition cycle Nhat.
# Nhat is represented in companion form with lag state Nhat_{t-1}, so an AR(2)
# process is embedded in the same linear-Gaussian FFBS as AR(1).
# --------------------------------------------------------------------------- #
def is_stationary_ar2(rho1: float, rho2: float) -> bool:
    """Standard AR(2) stationarity triangle."""
    return (rho2 > -1.0) and (rho1 + rho2 < 1.0) and (rho2 - rho1 < 1.0)


def ar2_stationary_cov(rho1: float, rho2: float, var: float) -> np.ndarray:
    """Stationary covariance of the companion state [Nhat_t, Nhat_{t-1}].

    Uses the closed-form AR(2) autocovariances; falls back to a diffuse diagonal
    if the parameters are (numerically) non-stationary.
    """
    if not is_stationary_ar2(rho1, rho2):
        return np.diag([var * 100.0, var * 100.0])
    denom = (1.0 + rho2) * ((1.0 - rho2) ** 2 - rho1 ** 2)
    if abs(denom) < 1e-12:
        return np.diag([var * 100.0, var * 100.0])
    g0 = var * (1.0 - rho2) / denom
    g1 = rho1 * g0 / (1.0 - rho2)
    if not (np.isfinite(g0) and np.isfinite(g1)) or g0 <= 0:
        return np.diag([var * 100.0, var * 100.0])
    return np.array([[g0, g1], [g1, g0]])


def _draw_ar2(
    rng: np.random.Generator,
    y: np.ndarray,
    x1: np.ndarray,
    x2: np.ndarray,
    variance: float,
    prior_mean: np.ndarray,
    prior_sd: np.ndarray,
    current: tuple[float, float],
    max_tries: int = 50,
) -> tuple[float, float]:
    """Conjugate bivariate normal draw of (rho1, rho2) with stationarity rejection."""
    X = np.column_stack([x1, x2])
    prior_prec = np.diag(1.0 / np.asarray(prior_sd, float) ** 2)
    v_inv = prior_prec + X.T @ X / variance
    v = np.linalg.inv(force_pd(v_inv))
    b = v @ (prior_prec @ np.asarray(prior_mean, float) + X.T @ y / variance)
    for _ in range(max_tries):
        cand = rng.multivariate_normal(b, force_pd(v))
        if is_stationary_ar2(float(cand[0]), float(cand[1])):
            return float(cand[0]), float(cand[1])
    return current


def _ffbs(
    rng: np.random.Generator,
    *,
    F: np.ndarray,
    c: np.ndarray,       # (T, dim) time-varying transition intercept
    Q: np.ndarray,
    m0: np.ndarray,
    P0: np.ndarray,
    observations: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
) -> np.ndarray:
    """Missing-row Kalman filter + backward simulation with time-varying intercept."""
    F, Q = np.asarray(F, float), np.asarray(Q, float)
    c = np.asarray(c, float)
    m0, P0 = np.asarray(m0, float), np.asarray(P0, float)
    T, dim = len(observations), F.shape[0]
    m_pred = np.zeros((T, dim))
    P_pred = np.zeros((T, dim, dim))
    m_filt = np.zeros((T, dim))
    P_filt = np.zeros((T, dim, dim))
    eye = np.eye(dim)
    for t, (values, H, R) in enumerate(observations):
        if t == 0:
            m_pred[t], P_pred[t] = m0, force_pd(P0)
        else:
            m_pred[t] = c[t] + F @ m_filt[t - 1]
            P_pred[t] = force_pd(F @ P_filt[t - 1] @ F.T + Q)
        values, H, R = np.asarray(values, float), np.asarray(H, float), np.asarray(R, float)
        if values.size == 0:
            m_filt[t], P_filt[t] = m_pred[t], P_pred[t]
            continue
        S = force_pd(H @ P_pred[t] @ H.T + R)
        K = np.linalg.solve(S, H @ P_pred[t]).T
        m_filt[t] = m_pred[t] + K @ (values - H @ m_pred[t])
        P_filt[t] = force_pd((eye - K @ H) @ P_pred[t] @ (eye - K @ H).T + K @ R @ K.T)
    states = np.zeros((T, dim))
    states[-1] = rng.multivariate_normal(m_filt[-1], force_pd(P_filt[-1]))
    for t in range(T - 2, -1, -1):
        Pnext = force_pd(P_pred[t + 1])
        J = np.linalg.solve(Pnext, F @ P_filt[t]).T
        mean = m_filt[t] + J @ (states[t + 1] - c[t + 1] - F @ m_filt[t])
        cov = force_pd(P_filt[t] - J @ Pnext @ J.T)
        states[t] = rng.multivariate_normal(mean, cov)
    return states


def _draw_coeffs(
    rng: np.random.Generator,
    y: np.ndarray,
    X: np.ndarray,
    prior_mean: np.ndarray,
    prior_sd: np.ndarray,
    sigma2: float,
    ig_shape: float,
    ig_b: float,
) -> tuple[np.ndarray, float]:
    """Conjugate Gaussian coefficient draw + inverse-gamma variance draw."""
    prior_prec = np.diag(1.0 / prior_sd ** 2)
    xtx = X.T @ X
    v_inv = prior_prec + xtx / sigma2
    v = np.linalg.inv(force_pd(v_inv))
    b = v @ (prior_prec @ prior_mean + X.T @ y / sigma2)
    beta = rng.multivariate_normal(b, force_pd(v))
    resid = y - X @ beta
    shape = ig_shape + y.size / 2.0
    scale = ig_b + 0.5 * float(resid @ resid)
    sigma2_new = _draw_ig(rng, shape, scale)
    return beta, sigma2_new


def _design(names: tuple[str, ...], epi, x, ntilde, nhat, pi_lag, ctrl) -> np.ndarray:
    ones = np.ones_like(epi)
    cols = {
        "intercept": ones,
        "alpha": epi,
        "alpha_b": pi_lag,
        "kappa_0": x,
        "delta": x * ntilde,
        "theta_0": -nhat,          # inflation loads -theta_0 * Nhat (HSA-consistent theta_0>0)
        "gamma": ntilde * nhat,
        "c_ctrl": ctrl,
    }
    return np.column_stack([cols[n] for n in names])


@dataclass
class GibbsResult:
    case: int
    model: int
    label: str
    coeff_names: tuple[str, ...]
    coeffs: np.ndarray                 # (chains, draws, k)
    sigma_pi: np.ndarray               # (chains, draws)
    rho: np.ndarray                    # AR(1) coeff, or rho_1 when ar_order=2
    rho2: np.ndarray | None            # AR(2) second coeff, None when ar_order=1
    lambda_E: np.ndarray | None
    sigma_bar: np.ndarray
    sigma_hat: np.ndarray
    sigma_nu: np.ndarray | None
    ntilde: np.ndarray                 # (chains, draws, T)
    nhat: np.ndarray
    periods: tuple[str, ...]
    extra: dict = field(default_factory=dict)


def run_gibbs(
    data: CaseData,
    model: int,
    *,
    iterations: int,
    warmup: int,
    thin: int,
    chains: int,
    seed: int,
    priors: Priors | None = None,
    hybrid: bool = False,
    ar_order: int = 1,
    exact_n: bool = False,
    progress_tick: Callable[[], None] | None = None,
) -> GibbsResult:
    if model not in HSA_FREE:
        raise ValueError(f"model must be 0..4, got {model}")
    if ar_order not in (1, 2):
        raise ValueError(f"ar_order must be 1 or 2, got {ar_order}")
    ar2 = ar_order == 2
    has_control = data.control is not None
    priors = priors or build_priors(data, hybrid=hybrid)
    names = coeff_names(model, hybrid, control=has_control)
    bilinear = model in (3, 4)
    case4 = data.case == 4

    pi, epi, x = data.pi, data.epi, data.x
    pi_lag = data.pi_lag if hybrid else np.zeros(data.n_periods)
    ctrl = data.control if has_control else np.zeros(data.n_periods)
    if hybrid and pi_lag is None:
        raise ValueError("hybrid=True requires data.pi_lag")
    n_obs = data.n_obs
    observed = np.isfinite(n_obs)
    gE = data.gE if case4 else np.zeros(data.n_periods)
    T = data.n_periods
    prior_mean, prior_sd = priors.coeff_prior(names)

    n_save = (iterations - warmup + thin - 1) // thin
    coeffs = np.zeros((chains, n_save, len(names)))
    sig_pi = np.zeros((chains, n_save))
    rho_out = np.zeros((chains, n_save))
    rho2_out = np.zeros((chains, n_save)) if ar2 else None
    lam_out = np.zeros((chains, n_save)) if case4 else None
    sbar_out = np.zeros((chains, n_save))
    shat_out = np.zeros((chains, n_save))
    snu_out = None if case4 else np.zeros((chains, n_save))
    ntilde_out = np.zeros((chains, n_save, T))
    nhat_out = np.zeros((chains, n_save, T))

    anchor_var = ANCHOR_REL_VAR * data.s_N ** 2

    for chain in range(chains):
        rng = np.random.default_rng(seed + chain * 7919 + model * 104729 + data.case * 1299827)
        # --- initialise ---
        beta = {n: 0.0 for n in FULL_COEFFS}
        beta["alpha"] = priors.alpha_mean
        sigma2 = priors.sigma_pi_b / (priors.ig_shape - 1)
        rho = 0.5
        rho2 = 0.0
        lam = 0.0
        var_bar = priors.sigma_bar_b / (priors.ig_shape - 1)
        var_hat = priors.sigma_hat_b / (priors.ig_shape - 1)
        var_nu = anchor_var if (case4 or exact_n) else priors.sigma_nu_b / (priors.ig_shape - 1)
        ntilde = np.where(observed, np.nan_to_num(n_obs), 0.0).astype(float)
        # simple init: put observed on slow state, zero fast
        ntilde = np.zeros(T)
        nhat = np.zeros(T)
        save = 0
        for it in range(iterations):
            # ---- (1) state draw ----
            base = (pi - beta["alpha"] * epi - beta["kappa_0"] * x
                    - beta["intercept"] - beta["alpha_b"] * pi_lag
                    - beta["c_ctrl"] * ctrl)
            if not bilinear:
                # Linear FFBS over [Ntilde, Nhat] (AR1) or [Ntilde, Nhat, Nhat_lag] (AR2).
                lag_pad = [0.0] if ar2 else []
                dim = 3 if ar2 else 2
                obs = []
                for t in range(T):
                    vals, rows, rvar = [], [], []
                    # inflation row (loads on Ntilde, Nhat; never on the lag copy)
                    h_pi = [0.0, 0.0]
                    if model == 1:
                        h_pi = [beta["delta"] * x[t], 0.0]
                    elif model == 2:
                        h_pi = [0.0, -beta["theta_0"]]   # inflation loads -theta_0 * Nhat
                    vals.append(base[t]); rows.append(h_pi + lag_pad); rvar.append(sigma2)
                    if observed[t]:
                        vals.append(n_obs[t]); rows.append([1.0, 1.0] + lag_pad); rvar.append(var_nu)
                    obs.append((np.asarray(vals), np.asarray(rows), np.diag(rvar)))
                c = np.zeros((T, dim))
                c[:, 1] = lam * gE
                if ar2:
                    F = np.array([[1.0, 0.0, 0.0], [0.0, rho, rho2], [0.0, 1.0, 0.0]])
                    Q = np.diag([var_bar, var_hat, 0.0])
                    P0 = np.zeros((3, 3))
                    P0[0, 0] = priors.ntilde0_sd ** 2
                    P0[1:, 1:] = ar2_stationary_cov(rho, rho2, var_hat)
                else:
                    F = np.array([[1.0, 0.0], [0.0, rho]])
                    Q = np.diag([var_bar, var_hat])
                    P0 = np.diag([priors.ntilde0_sd ** 2, var_hat / max(1e-6, 1 - rho ** 2)])
                states = _ffbs(rng, F=F, c=c, Q=Q, m0=np.zeros(dim), P0=P0, observations=obs)
                ntilde, nhat = states[:, 0], states[:, 1]
            else:
                # ---- blocked conditional FFBS ----
                # Ntilde | Nhat  (1-D random walk)
                obs = []
                for t in range(T):
                    if model == 3:
                        h = beta["gamma"] * nhat[t]
                        v = base[t] + beta["theta_0"] * nhat[t]
                    else:  # model 4
                        h = beta["delta"] * x[t] + beta["gamma"] * nhat[t]
                        v = base[t] + beta["theta_0"] * nhat[t]
                    vals, rows, rvar = [v], [[h]], [sigma2]
                    if observed[t]:
                        vals.append(n_obs[t] - nhat[t]); rows.append([1.0]); rvar.append(var_nu)
                    obs.append((np.asarray(vals), np.asarray(rows), np.diag(rvar)))
                ntilde = _ffbs(
                    rng, F=np.array([[1.0]]), c=np.zeros((T, 1)), Q=np.array([[var_bar]]),
                    m0=np.zeros(1), P0=np.array([[priors.ntilde0_sd ** 2]]), observations=obs,
                )[:, 0]
                # Nhat | Ntilde  (AR(1), or AR(2) in companion form; gE input in Case 4)
                lag_pad = [0.0] if ar2 else []
                dim = 2 if ar2 else 1
                obs = []
                for t in range(T):
                    h = -beta["theta_0"] + beta["gamma"] * ntilde[t]
                    v = base[t] - (beta["delta"] * x[t] * ntilde[t] if model == 4 else 0.0)
                    vals, rows, rvar = [v], [[h] + lag_pad], [sigma2]
                    if observed[t]:
                        vals.append(n_obs[t] - ntilde[t]); rows.append([1.0] + lag_pad); rvar.append(var_nu)
                    obs.append((np.asarray(vals), np.asarray(rows), np.diag(rvar)))
                c = np.zeros((T, dim)); c[:, 0] = lam * gE
                if ar2:
                    F = np.array([[rho, rho2], [1.0, 0.0]])
                    Q = np.diag([var_hat, 0.0])
                    P0 = ar2_stationary_cov(rho, rho2, var_hat)
                else:
                    F = np.array([[rho]])
                    Q = np.array([[var_hat]])
                    P0 = np.array([[var_hat / max(1e-6, 1 - rho ** 2)]])
                nhat = _ffbs(rng, F=F, c=c, Q=Q, m0=np.zeros(dim), P0=P0, observations=obs)[:, 0]

            # ---- (2)(3) coefficients + sigma_pi ----
            X = _design(names, epi, x, ntilde, nhat, pi_lag, ctrl)
            beta_arr, sigma2 = _draw_coeffs(
                rng, pi, X, prior_mean, prior_sd, sigma2, priors.ig_shape, priors.sigma_pi_b
            )
            beta = {n: 0.0 for n in FULL_COEFFS}
            beta.update(dict(zip(names, beta_arr)))

            # ---- (4) transition parameters ----
            if ar2:
                prior_mean_ar = np.array([priors.rho_mean, 0.0])
                prior_sd_ar = np.array([priors.rho_sd, priors.rho_sd])
                if case4:
                    y_rho = nhat[2:] - lam * gE[2:]
                    rho, rho2 = _draw_ar2(rng, y_rho, nhat[1:-1], nhat[:-2], var_hat,
                                          prior_mean_ar, prior_sd_ar, (rho, rho2))
                    y_lam = nhat[2:] - rho * nhat[1:-1] - rho2 * nhat[:-2]
                    lam = _normal_regression_draw(rng, y_lam, gE[2:], var_hat, 0.0, priors.lambda_sd)
                else:
                    rho, rho2 = _draw_ar2(rng, nhat[2:], nhat[1:-1], nhat[:-2], var_hat,
                                          prior_mean_ar, prior_sd_ar, (rho, rho2))
            elif case4:
                # rho | lambda (truncated), lambda | rho (normal)
                y_rho = nhat[1:] - lam * gE[1:]
                rho = _truncated_regression_draw(rng, y_rho, nhat[:-1], var_hat, priors.rho_mean, priors.rho_sd)
                y_lam = nhat[1:] - rho * nhat[:-1]
                lam = _normal_regression_draw(rng, y_lam, gE[1:], var_hat, 0.0, priors.lambda_sd)
            else:
                rho = _truncated_regression_draw(rng, nhat[1:], nhat[:-1], var_hat, priors.rho_mean, priors.rho_sd)

            # ---- (5) variances ----
            res_bar = np.diff(ntilde)
            var_bar = _draw_ig(rng, priors.ig_shape + res_bar.size / 2, priors.sigma_bar_b + 0.5 * float(res_bar @ res_bar))
            if ar2:
                res_hat = nhat[2:] - rho * nhat[1:-1] - rho2 * nhat[:-2] - (lam * gE[2:] if case4 else 0.0)
            else:
                res_hat = nhat[1:] - rho * nhat[:-1] - (lam * gE[1:] if case4 else 0.0)
            var_hat = _draw_ig(rng, priors.ig_shape + res_hat.size / 2, priors.sigma_hat_b + 0.5 * float(res_hat @ res_hat))
            if not case4 and not exact_n:
                res_nu = n_obs[observed] - (ntilde[observed] + nhat[observed])
                var_nu = _draw_ig(rng, priors.ig_shape + res_nu.size / 2, priors.sigma_nu_b + 0.5 * float(res_nu @ res_nu))

            # ---- (6) store ----
            if it >= warmup and (it - warmup) % thin == 0:
                coeffs[chain, save] = beta_arr
                sig_pi[chain, save] = np.sqrt(sigma2)
                rho_out[chain, save] = rho
                if ar2:
                    rho2_out[chain, save] = rho2
                if case4:
                    lam_out[chain, save] = lam
                sbar_out[chain, save] = np.sqrt(var_bar)
                shat_out[chain, save] = np.sqrt(var_hat)
                if not case4:
                    snu_out[chain, save] = np.sqrt(var_nu)
                ntilde_out[chain, save] = ntilde
                nhat_out[chain, save] = nhat
                save += 1
            if progress_tick is not None:
                progress_tick()

    return GibbsResult(
        case=data.case, model=model, label=data.label, coeff_names=names,
        coeffs=coeffs, sigma_pi=sig_pi, rho=rho_out, rho2=rho2_out, lambda_E=lam_out,
        sigma_bar=sbar_out, sigma_hat=shat_out, sigma_nu=snu_out,
        ntilde=ntilde_out, nhat=nhat_out, periods=tuple(map(str, data.periods)),
        extra={"ar_order": ar_order},
    )
