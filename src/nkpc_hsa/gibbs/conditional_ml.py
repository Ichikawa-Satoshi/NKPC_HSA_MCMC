"""Conditional marginal likelihood  p(pi | x, N_obs, M)  for CES and HSA steady.

What this computes and why
--------------------------
The model comparison of interest is about the *inflation* mechanism, not about
which model fits the firm count better. The right estimand is therefore

    p(pi | x, N_obs, M)

with the static parameters and the latent states integrated out. For the
linear-Gaussian HSA models the latent states can be integrated out exactly by
the Kalman prediction-error decomposition, and the conditioning on N_obs is
obtained from the identity

    log p(pi | x, N_obs, theta) = log p(pi, N_obs | x, theta) - log p(N_obs | theta_N)

where both terms are exact Kalman likelihoods over the same state-space model
(the second one simply drops the inflation observation row). ``tests/`` checks
this identity numerically.

CES carries no firm-count state and no N_obs term, so
p(pi | x, N_obs, CES) = p(pi | x, CES); its conditional likelihood is the plain
Gaussian inflation-equation likelihood. Both models are therefore densities of
the same data vector ``pi`` given the same conditioning set, which is what makes
the comparison a legitimate Bayes factor.

Chib's identity, block by block
-------------------------------
For each model, at a high-density point theta*,

    log m(pi | x, N_obs) = log p(pi | x, N_obs, theta*)      [likelihood]
                         + log p(theta*)                      [prior]
                         - log p(theta* | pi, x, N_obs)       [posterior ordinate]

and the ordinate is factorised in the sampler's own block order

    p(theta*|y) = p(B1*|y) * p(B2*|B1*,y) * ... * p(BG*|B1*..B(G-1)*,y)

Every factor is the *exact* Gibbs full conditional used by the production
sampler, Rao-Blackwellised over draws from a run in which blocks 1..g-1 are
pinned at their starred values. Those reduced runs are produced by the
production samplers themselves via ``opts["fixed"]``, so no conditional is
re-implemented here.

Blocks (in sampler order)
    CES         beta=(alpha,kappa) | lambda_ez | phi_1 | sigma_zeta2 | sigma_eta2
    HSA steady  beta=(alpha,kappa_0,delta) | lambda_ez | phi_1 | sigma_zeta2 |
                sigma_eta2 | rho=(rho_1,rho_2) | sigma_u2 | n | sigma_eps2 | sigma_N2

Truncation
----------
The (rho_1, rho_2) prior and its full conditional are both truncated to the AR(2)
stationary triangle. The truncation constants do NOT cancel between the prior and
the ordinate (they are different distributions), so both are normalised by Monte
Carlo with a fixed seed.

Not supported
-------------
``hsa_full`` has a bilinear latent term and is not linear-Gaussian in the joint
state, so neither the exact conditional likelihood nor a Gibbs ordinate over
these blocks is available. It raises rather than returning a posterior-mean
plug-in dressed up as a marginal likelihood.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

from nkpc_hsa.gibbs.common.joint_ffbs import force_pd

Family = Literal["ces", "steady"]

KAPPA_SCALE = 100.0
_STATIONARITY_MC_DRAWS = 200_000
_STATIONARITY_SEED = 20260812


# ---------------------------------------------------------------------------
# Densities
# ---------------------------------------------------------------------------
def _log_norm_pdf(x, mu, sd):
    return -0.5 * np.log(2.0 * np.pi * sd**2) - 0.5 * ((np.asarray(x, float) - mu) / sd) ** 2


def _log_mvn_pdf(x: np.ndarray, mean: np.ndarray, cov: np.ndarray) -> float:
    """Gaussian log density via Cholesky.

    Cholesky is used rather than ``slogdet`` because these covariances can be
    large and ill-conditioned (the dense validation builds a 2T+1 square matrix),
    where the eigen-repair path both loses precision and trips overflow warnings.
    Eigen-repair is kept only as a fallback for genuinely indefinite input.
    """
    x = np.asarray(x, float).reshape(-1)
    mean = np.asarray(mean, float).reshape(-1)
    cov = np.asarray(cov, float)
    cov = 0.5 * (cov + cov.T)
    try:
        chol = np.linalg.cholesky(cov)
    except np.linalg.LinAlgError:
        chol = np.linalg.cholesky(force_pd(cov))
    from scipy.linalg import solve_triangular

    resid = solve_triangular(chol, x - mean, lower=True)
    logdet = 2.0 * float(np.sum(np.log(np.diag(chol))))
    quad = float(resid @ resid)
    return float(-0.5 * (x.size * np.log(2.0 * np.pi) + logdet + quad))


def _log_ig_pdf_var(x: float, a: float, b: float) -> float:
    """log density of an inverse-gamma(a, b) evaluated at a *variance*."""
    from scipy.special import gammaln

    x = float(x)
    if x <= 0:
        return -np.inf
    return float(a * np.log(b) - gammaln(a) - (a + 1.0) * np.log(x) - b / x)


def _logmeanexp(values: np.ndarray) -> float:
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return -np.inf
    m = float(np.max(values))
    return float(m + np.log(np.mean(np.exp(values - m))))


def _is_stationary_ar2(r1, r2):
    return (np.abs(r2) < 1.0) & ((r1 + r2) < 1.0) & ((r2 - r1) < 1.0)


def log_stationary_mass(mean: np.ndarray, cov: np.ndarray, *, seed: int = _STATIONARITY_SEED) -> float:
    """log Pr[(rho_1, rho_2) in the AR(2) stationary triangle] under N(mean, cov).

    This is the normalising constant of a truncated bivariate normal over the
    triangle {|rho_2|<1, rho_1+rho_2<1, rho_2-rho_1<1}. It has no closed form, so
    it is estimated by Monte Carlo with a fixed seed (reproducible, and the same
    estimator is used for the prior and for every ordinate factor so the errors
    are of the same order and largely common).
    """
    rng = np.random.default_rng(seed)
    draws = rng.multivariate_normal(np.asarray(mean, float), force_pd(cov), size=_STATIONARITY_MC_DRAWS)
    mass = float(np.mean(_is_stationary_ar2(draws[:, 0], draws[:, 1])))
    return float(np.log(max(mass, 1.0 / _STATIONARITY_MC_DRAWS)))


def log_truncated_mvn_pdf(x, mean, cov, *, seed: int = _STATIONARITY_SEED) -> float:
    """Normalised density of a bivariate normal truncated to the AR(2) triangle."""
    if not bool(_is_stationary_ar2(float(x[0]), float(x[1]))):
        return -np.inf
    return _log_mvn_pdf(x, mean, cov) - log_stationary_mass(mean, cov, seed=seed)


# ---------------------------------------------------------------------------
# Exact Kalman likelihoods (states integrated out)
# ---------------------------------------------------------------------------
def kalman_loglik(
    *,
    N_obs: np.ndarray,
    y_tilde: np.ndarray | None,
    h_nhat: np.ndarray | None,
    h_nbar: np.ndarray | None,
    n_drift: float,
    rho1: float,
    rho2: float,
    sigma_eta2: float,
    sigma_u2: float,
    sigma_eps2: float,
    sigma_N2: float,
    m0: np.ndarray,
    P0: np.ndarray,
    include_inflation_row: bool = True,
) -> float:
    """Prediction-error log-likelihood for the firm-count state-space model.

    ``include_inflation_row=False`` gives log p(N_obs | theta_N) -- the same
    filter with the inflation row removed. Missing firm-count observations drop
    the N row for that quarter only; prediction still runs every quarter.

    Initial state, transition and covariance match the estimating model exactly:
    ``s_0 ~ N(m0, P0)`` with the sampler's ``m0 = (0,0,0)`` and ``P0 = 10 I``.
    ``N_obs[0]`` is NOT substituted into ``m0`` -- doing so would both evaluate a
    different model from the one estimated and use the first observation twice.
    """
    N_obs = np.asarray(N_obs, float).reshape(-1)
    T = N_obs.size
    F = np.array([[rho1, rho2, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    c = np.array([0.0, 0.0, n_drift])
    Q = np.diag([sigma_u2, 0.0, sigma_eps2])

    m = np.asarray(m0, float).reshape(-1).copy()
    P = force_pd(np.asarray(P0, float))
    loglik = 0.0

    for t in range(T):
        if t > 0:
            m = c + F @ m
            P = force_pd(F @ P @ F.T + Q)

        rows_H, rows_z, rows_R = [], [], []
        if np.isfinite(N_obs[t]):
            rows_H.append([1.0, 0.0, 1.0])
            rows_z.append(float(N_obs[t]))
            rows_R.append(sigma_N2)
        if include_inflation_row:
            rows_H.append([float(h_nhat[t]), 0.0, float(h_nbar[t])])
            rows_z.append(float(y_tilde[t]))
            rows_R.append(sigma_eta2)
        if not rows_H:
            continue

        H = np.asarray(rows_H, float)
        z = np.asarray(rows_z, float)
        R = np.diag(rows_R)
        S = force_pd(H @ P @ H.T + R)
        v = z - H @ m
        loglik += _log_mvn_pdf(v, np.zeros(v.size), S)
        K = P @ H.T @ np.linalg.inv(S)
        m = m + K @ v
        KH = K @ H
        P = force_pd((np.eye(3) - KH) @ P @ (np.eye(3) - KH).T + K @ R @ K.T)

    return float(loglik)


def steady_conditional_loglik(star: dict, data: dict, *, m0, P0) -> float:
    """log p(pi | x, N_obs, theta) for HSA steady, states integrated out."""
    y = data["pi"] - data["pi_expect"]
    a_t = data["pi_prev"] - data["pi_expect"]
    zeta = data["x"] - star["phi_1"] * data["x_prev"]
    y_tilde = y - star["alpha"] * a_t - star["kappa_0"] * data["x"] - star["lambda_ez"] * zeta
    T = y.size
    shared = dict(
        N_obs=data["N"],
        n_drift=star["n"],
        rho1=star["rho_1"],
        rho2=star["rho_2"],
        sigma_eta2=star["sigma_eta2"],
        sigma_u2=star["sigma_u2"],
        sigma_eps2=star["sigma_eps2"],
        sigma_N2=star["sigma_N2"],
        m0=m0,
        P0=P0,
    )
    joint = kalman_loglik(
        y_tilde=y_tilde,
        h_nhat=np.zeros(T),
        h_nbar=star["delta"] * data["x"],
        include_inflation_row=True,
        **shared,
    )
    n_only = kalman_loglik(
        y_tilde=None, h_nhat=None, h_nbar=None, include_inflation_row=False, **shared
    )
    return float(joint - n_only)


def ces_conditional_loglik(star: dict, data: dict) -> float:
    """log p(pi | x, theta) for CES (no firm-count state to condition on)."""
    y = data["pi"] - data["pi_expect"]
    a_t = data["pi_prev"] - data["pi_expect"]
    zeta = data["x"] - star["phi_1"] * data["x_prev"]
    eta = y - star["alpha"] * a_t - star["kappa"] * data["x"] - star["lambda_ez"] * zeta
    T = eta.size
    return float(-0.5 * T * np.log(2.0 * np.pi * star["sigma_eta2"]) - 0.5 * float(np.sum(eta**2)) / star["sigma_eta2"])


# ---------------------------------------------------------------------------
# Priors
# ---------------------------------------------------------------------------
def _prior_pair(pri: dict, key: str) -> tuple[float, float]:
    value = pri[key]
    return float(value[0]), float(value[1])


def log_prior(star: dict, pri: dict, *, family: Family) -> float:
    out = _log_norm_pdf(star["alpha"], *_prior_pair(pri, "alpha"))
    if family == "ces":
        out += _log_norm_pdf(star["kappa"], *_prior_pair(pri, "kappa"))
    else:
        out += _log_norm_pdf(star["kappa_0"], *_prior_pair(pri, "kappa_0"))
        out += _log_norm_pdf(star["delta"], *_prior_pair(pri, "delta"))
    out += _log_norm_pdf(star["lambda_ez"], *_prior_pair(pri, "lambda_ez"))
    out += _log_norm_pdf(star["phi_1"], *_prior_pair(pri, "phi_1"))
    out += _log_ig_pdf_var(star["sigma_zeta2"], pri["a_z"], pri["b_z"])
    out += _log_ig_pdf_var(star["sigma_eta2"], pri["a_e"], pri["b_e"])
    if family == "steady":
        mu_rho = np.array([_prior_pair(pri, "rho_1")[0], _prior_pair(pri, "rho_2")[0]])
        cov_rho = np.diag([_prior_pair(pri, "rho_1")[1] ** 2, _prior_pair(pri, "rho_2")[1] ** 2])
        # Truncated to the stationary triangle, so subtract the truncation mass.
        out += log_truncated_mvn_pdf([star["rho_1"], star["rho_2"]], mu_rho, cov_rho)
        out += _log_ig_pdf_var(star["sigma_u2"], pri["a_u"], pri["b_u"])
        out += _log_norm_pdf(star["n"], *_prior_pair(pri, "n"))
        out += _log_ig_pdf_var(star["sigma_eps2"], pri["a_eps"], pri["b_eps"])
        out += _log_ig_pdf_var(star["sigma_N2"], pri["a_N"], pri["b_N"])
    return float(out)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------
@dataclass
class ConditionalMLResult:
    log_conditional_marginal_likelihood: float
    log_likelihood: float
    log_prior: float
    log_posterior_ordinate: float
    family: str
    ordinate_terms: dict[str, float] = field(default_factory=dict)
    notes: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "log_conditional_marginal_likelihood": self.log_conditional_marginal_likelihood,
            "log_likelihood": self.log_likelihood,
            "log_prior": self.log_prior,
            "log_posterior_ordinate": self.log_posterior_ordinate,
            "family": self.family,
            "ordinate_terms": self.ordinate_terms,
            "notes": self.notes,
        }


def full_conditional_marginal_likelihood(*args: Any, **kwargs: Any):
    raise NotImplementedError(
        "hsa_full has a bilinear latent term (-gamma*Nbar_t*Nhat_t), so the joint "
        "state is not linear-Gaussian: neither the exact conditional likelihood "
        "p(pi | x, N_obs, theta) nor a Gibbs posterior ordinate over these blocks "
        "is available in closed form. The previous implementation conditioned on "
        "posterior-MEAN latent states and reported the result as a marginal "
        "likelihood; that is a plug-in, it does not integrate the states out, and "
        "it systematically favours the state-richest model. Use a method designed "
        "for nonlinear state spaces (e.g. an SMC-based marginal likelihood) if this "
        "number is needed."
    )


# ---------------------------------------------------------------------------
# Posterior ordinate via Chib's sequential decomposition with reduced runs
# ---------------------------------------------------------------------------
def _beta_cond_moments(y, X, sigma2, prior_mean, prior_sd):
    prior_prec = np.diag(1.0 / np.asarray(prior_sd, float) ** 2)
    cov = np.linalg.inv(X.T @ X / sigma2 + prior_prec)
    mean = cov @ (X.T @ y / sigma2 + prior_prec @ np.asarray(prior_mean, float))
    return mean, cov


def _lambda_cond_moments(e_base, zeta, sigma_eta2, pri):
    mu0, sd0 = _prior_pair(pri, "lambda_ez")
    prec = 1.0 / sd0**2 + float(np.sum(zeta**2)) / sigma_eta2
    var = 1.0 / prec
    mean = var * (mu0 / sd0**2 + float(np.dot(zeta, e_base)) / sigma_eta2)
    return mean, np.sqrt(var)


def _phi_cond_moments(x, x_prev, y_tilde, lambda_ez, sigma_zeta2, sigma_eta2, pri):
    mu0, sd0 = _prior_pair(pri, "phi_1")
    prec = (
        1.0 / sd0**2
        + float(np.sum(x_prev**2)) / sigma_zeta2
        + (lambda_ez**2) * float(np.sum(x_prev**2)) / sigma_eta2
    )
    mean = (
        mu0 / sd0**2
        + float(np.dot(x_prev, x)) / sigma_zeta2
        - lambda_ez * float(np.dot(x_prev, y_tilde - lambda_ez * x)) / sigma_eta2
    ) / prec
    return mean, np.sqrt(1.0 / prec)


def _n_cond_moments(Nbar, sigma_eps2, pri):
    mu0, sd0 = _prior_pair(pri, "n")
    d = Nbar[1:] - Nbar[:-1]
    var = 1.0 / (1.0 / sd0**2 + d.size / sigma_eps2)
    mean = var * (mu0 / sd0**2 + float(np.sum(d)) / sigma_eps2)
    return mean, np.sqrt(var)


def _rho_cond_moments(Nhat, Nhat_lag, sigma_u2, pri):
    """Exactly the sampler's AR(2) conditional, including the initial-lag row.

    The production sampler regresses ``Nhat[1:]`` on ``(Nhat[:-1], [Nhat_lag,
    Nhat[:-2]])`` -- it uses the *sampled* ``Nhat_{-1}`` so the first AR(2)
    likelihood is not dropped. Reproducing that here is what makes the ordinate
    the sampler's own conditional rather than a look-alike.
    """
    y = Nhat[1:]
    second_lag = np.concatenate([[float(Nhat_lag)], Nhat[:-2]])
    X = np.column_stack([Nhat[:-1], second_lag])
    mu = np.array([_prior_pair(pri, "rho_1")[0], _prior_pair(pri, "rho_2")[0]])
    prior_prec = np.diag(
        [1.0 / _prior_pair(pri, "rho_1")[1] ** 2, 1.0 / _prior_pair(pri, "rho_2")[1] ** 2]
    )
    cov = np.linalg.inv(X.T @ X / sigma_u2 + prior_prec)
    mean = cov @ (X.T @ y / sigma_u2 + prior_prec @ mu)
    return mean, cov


def _draws_from(result: dict, key: str) -> np.ndarray:
    return np.asarray(result[key]["draws"], dtype=float)


def _states_from(result: dict, key: str) -> np.ndarray:
    return np.asarray(result["state_draws"][key], dtype=float)


def _run_gibbs(sampler, data, priors_internal, *, family, fixed, n_burn, n_keep, seed, thin=1):
    """One (possibly reduced) run of the production sampler."""
    kwargs = dict(
        pi_data=data["pi"],
        pi_prev_data=data["pi_prev"],
        Epi_data=data["pi_expect"],
        x_data=data["x"],
        x_prev_data=data["x_prev"],
        n_burn=n_burn,
        n_keep=n_keep,
        priors=priors_internal,
        opts={
            "seed": int(seed),
            "store_every": thin,
            "verbose": False,
            "fixed": dict(fixed),
            # Chib's AR(2) ordinate must be the sampler's own initial-lag
            # conditional, so the run has to hand back the sampled Nhat_{-1}.
            "return_state_lag": True,
        },
        orth=False,
    )
    if family == "steady":
        kwargs["N_data"] = data["N"]
    return sampler(**kwargs)


def _star_from_posterior(result: dict, *, family: Family) -> dict[str, float]:
    """theta* = posterior means (any high-density point is admissible for Chib)."""
    # CES stores sigma_zeta2 / sigma_e2 (variances); the HSA samplers store
    # sigma_zeta / sigma_eta (standard deviations).
    star: dict[str, float] = {
        "alpha": float(np.mean(_draws_from(result, "alpha"))),
        "phi_1": float(np.mean(_draws_from(result, "phi_1"))),
        "lambda_ez": float(np.mean(_draws_from(result, "lambda_ez"))),
    }
    if family == "ces":
        star["sigma_zeta2"] = float(np.mean(_draws_from(result, "sigma_zeta2")))
        star["kappa"] = float(np.mean(_draws_from(result, "kappa")))
        sigma_e2 = float(np.mean(_draws_from(result, "sigma_e2")))
        star["sigma_eta2"] = sigma_e2 - star["lambda_ez"] ** 2 * star["sigma_zeta2"]
    else:
        star["sigma_zeta2"] = float(np.mean(_draws_from(result, "sigma_zeta")) ** 2)
        star["kappa_0"] = float(np.mean(_draws_from(result, "kappa_0")))
        star["delta"] = float(np.mean(_draws_from(result, "delta")))
        star["sigma_eta2"] = float(np.mean(_draws_from(result, "sigma_eta")) ** 2)
        star["rho_1"] = float(np.mean(_draws_from(result, "rho1")))
        star["rho_2"] = float(np.mean(_draws_from(result, "rho2")))
        star["sigma_u2"] = float(np.mean(_draws_from(result, "sigma_u")) ** 2)
        star["n"] = float(np.mean(_draws_from(result, "n")))
        star["sigma_eps2"] = float(np.mean(_draws_from(result, "sigma_eps")) ** 2)
        star["sigma_N2"] = float(np.mean(_draws_from(result, "sigma_N")) ** 2)
    star["sigma_eta2"] = max(star["sigma_eta2"], 1e-12)
    return star


def conditional_marginal_likelihood(
    data: dict[str, np.ndarray],
    priors_internal: dict,
    pri: dict,
    *,
    family: Family,
    n_burn: int = 1500,
    n_keep: int = 3000,
    seed: int = 90210,
    m0: np.ndarray | None = None,
    P0: np.ndarray | None = None,
) -> ConditionalMLResult:
    """Chib (1995) conditional marginal likelihood with explicit reduced runs.

    ``priors_internal`` is the sampler-facing prior dict (KAPPA_SCALE applied);
    ``pri`` is the same prior set in physical units, used for the prior term and
    the ordinate conditionals, which are all evaluated in physical units.
    """
    from nkpc_hsa.gibbs.ces.model import func_nkpc_ces
    from nkpc_hsa.gibbs.hsa_steady.model import func_nkpc_hsa_decomp_tv_kappa_kalman

    if family == "ces":
        sampler = func_nkpc_ces
        blocks = ["beta", "lambda_ez", "phi_1", "sigma_zeta2", "sigma_eta2"]
    elif family == "steady":
        sampler = func_nkpc_hsa_decomp_tv_kappa_kalman
        blocks = [
            "beta", "lambda_ez", "phi_1", "sigma_zeta2", "sigma_eta2",
            "rho", "sigma_u2", "n", "sigma_eps2", "sigma_N2",
        ]
    else:
        raise ValueError(f"Unsupported family for the conditional ML: {family!r}")

    m0 = np.zeros(3) if m0 is None else np.asarray(m0, float)
    P0 = np.eye(3) * 10.0 if P0 is None else np.asarray(P0, float)

    y = data["pi"] - data["pi_expect"]
    a_t = data["pi_prev"] - data["pi_expect"]

    # --- full run -> theta* and the first ordinate factor ---
    full = _run_gibbs(sampler, data, priors_internal, family=family, fixed={},
                      n_burn=n_burn, n_keep=n_keep, seed=seed)
    star = _star_from_posterior(full, family=family)

    star_values = {
        "beta": (
            (star["alpha"], star["kappa"]) if family == "ces"
            else (star["alpha"], star["kappa_0"], star["delta"])
        ),
        "lambda_ez": star["lambda_ez"],
        "phi_1": star["phi_1"],
        "sigma_zeta2": star["sigma_zeta2"],
        "sigma_eta2": star["sigma_eta2"],
    }
    if family == "steady":
        star_values.update({
            "rho": (star["rho_1"], star["rho_2"]),
            "sigma_u2": star["sigma_u2"],
            "n": star["n"],
            "sigma_eps2": star["sigma_eps2"],
            "sigma_N2": star["sigma_N2"],
        })
    # `fixed` is in sampler-internal units for the kappa-like beta entries.
    fixed_internal = dict(star_values)
    fixed_internal["beta"] = tuple(
        v * (KAPPA_SCALE if i > 0 else 1.0) for i, v in enumerate(star_values["beta"])
    )

    ordinate_terms: dict[str, float] = {}
    run = full
    for g, block in enumerate(blocks):
        terms = _ordinate_factor(block, run, star, data, pri, family=family, y=y, a_t=a_t)
        ordinate_terms[block] = terms
        if g == len(blocks) - 1:
            break
        # Reduced run for the next factor: pin blocks 0..g at their star values.
        run = _run_gibbs(
            sampler, data, priors_internal, family=family,
            fixed={b: fixed_internal[b] for b in blocks[: g + 1]},
            n_burn=n_burn, n_keep=n_keep, seed=seed + 101 * (g + 1),
        )

    log_ord = float(sum(ordinate_terms.values()))
    if family == "ces":
        log_lik = ces_conditional_loglik(star, data)
    else:
        log_lik = steady_conditional_loglik(star, data, m0=m0, P0=P0)
    log_pri = log_prior(star, pri, family=family)

    return ConditionalMLResult(
        log_conditional_marginal_likelihood=float(log_lik + log_pri - log_ord),
        log_likelihood=float(log_lik),
        log_prior=float(log_pri),
        log_posterior_ordinate=log_ord,
        family=family,
        ordinate_terms=ordinate_terms,
        notes=(
            "Chib (1995) with explicit reduced Gibbs runs; states integrated out "
            "by exact Kalman filter; (rho_1,rho_2) prior and conditional normalised "
            "over the AR(2) stationary triangle."
        ),
    )


def _ordinate_factor(block, run, star, data, pri, *, family, y, a_t) -> float:
    """log of one Chib ordinate factor, Rao-Blackwellised over the run's draws."""
    x, x_prev = data["x"], data["x_prev"]
    T = y.size
    if family == "ces":
        sigma_eta2_draws = np.maximum(
            _draws_from(run, "sigma_e2")
            - _draws_from(run, "lambda_ez") ** 2 * _draws_from(run, "sigma_zeta2"),
            1e-12,
        )
        sigma_zeta2_draws = _draws_from(run, "sigma_zeta2")
    else:
        sigma_eta2_draws = _draws_from(run, "sigma_eta") ** 2
        sigma_zeta2_draws = _draws_from(run, "sigma_zeta") ** 2
    lambda_draws = _draws_from(run, "lambda_ez")
    phi_draws = _draws_from(run, "phi_1")

    if block == "beta":
        if family == "ces":
            beta_star = np.array([star["alpha"], star["kappa"]])
            pm, ps = _beta_prior_arrays(pri, ["alpha", "kappa"])
            terms = []
            for lmb, phi, se in zip(lambda_draws, phi_draws, sigma_eta2_draws):
                X = np.column_stack([a_t, x])
                mean, cov = _beta_cond_moments(y - lmb * (x - phi * x_prev), X, se, pm, ps)
                terms.append(_log_mvn_pdf(beta_star, mean, cov))
        else:
            beta_star = np.array([star["alpha"], star["kappa_0"], star["delta"]])
            pm, ps = _beta_prior_arrays(pri, ["alpha", "kappa_0", "delta"])
            Nbar_draws = _states_from(run, "Nbar")
            terms = []
            for Nbar, lmb, phi, se in zip(Nbar_draws, lambda_draws, phi_draws, sigma_eta2_draws):
                X = np.column_stack([a_t, x, x * Nbar])
                mean, cov = _beta_cond_moments(y - lmb * (x - phi * x_prev), X, se, pm, ps)
                terms.append(_log_mvn_pdf(beta_star, mean, cov))
        return _logmeanexp(np.array(terms))

    kappa_term = (
        star["kappa"] * x if family == "ces"
        else (star["kappa_0"] + star["delta"] * _states_from(run, "Nbar")) * x
    )

    if block == "lambda_ez":
        terms = []
        for i, (phi, se) in enumerate(zip(phi_draws, sigma_eta2_draws)):
            zeta = x - phi * x_prev
            kt = kappa_term if family == "ces" else kappa_term[i]
            mean, sd = _lambda_cond_moments(y - star["alpha"] * a_t - kt, zeta, se, pri)
            terms.append(float(_log_norm_pdf(star["lambda_ez"], mean, sd)))
        return _logmeanexp(np.array(terms))

    if block == "phi_1":
        terms = []
        for i, (sz, se) in enumerate(zip(sigma_zeta2_draws, sigma_eta2_draws)):
            kt = kappa_term if family == "ces" else kappa_term[i]
            mean, sd = _phi_cond_moments(
                x, x_prev, y - star["alpha"] * a_t - kt, star["lambda_ez"], sz, se, pri
            )
            terms.append(float(_log_norm_pdf(star["phi_1"], mean, sd)))
        return _logmeanexp(np.array(terms))

    zeta_star = x - star["phi_1"] * x_prev

    if block == "sigma_zeta2":
        return _log_ig_pdf_var(
            star["sigma_zeta2"], pri["a_z"] + 0.5 * T, pri["b_z"] + 0.5 * float(np.sum(zeta_star**2))
        )

    if block == "sigma_eta2":
        terms = []
        n_draws = 1 if family == "ces" else len(_states_from(run, "Nbar"))
        for i in range(n_draws):
            kt = kappa_term if family == "ces" else kappa_term[i]
            eta = y - star["alpha"] * a_t - kt - star["lambda_ez"] * zeta_star
            terms.append(
                _log_ig_pdf_var(
                    star["sigma_eta2"], pri["a_e"] + 0.5 * T, pri["b_e"] + 0.5 * float(np.sum(eta**2))
                )
            )
        return _logmeanexp(np.array(terms))

    # --- HSA-steady state blocks ---
    Nhat_draws = _states_from(run, "Nhat")
    Nbar_draws = _states_from(run, "Nbar")
    sigma_u2_draws = _draws_from(run, "sigma_u") ** 2
    sigma_eps2_draws = _draws_from(run, "sigma_eps") ** 2

    # Nhat_{-1} as actually sampled. Substituting the prior mean here is NOT a
    # negligible approximation: sigma_u is about 0.045 and rho_2 about -0.8, so a
    # 3-unit error in this single regression row moves the log conditional by
    # order 10^2-10^3 and destroys the ordinate.
    nhat_lag_draws = _states_from(run, "Nhat_lag")

    if block == "rho":
        terms = [
            log_truncated_mvn_pdf(
                [star["rho_1"], star["rho_2"]], *_rho_cond_moments(Nhat, lag, su, pri)
            )
            for Nhat, lag, su in zip(Nhat_draws, nhat_lag_draws, sigma_u2_draws)
        ]
        return _logmeanexp(np.array(terms))

    if block == "sigma_u2":
        terms = []
        for Nhat, lag in zip(Nhat_draws, nhat_lag_draws):
            second_lag = np.concatenate([[float(lag)], Nhat[:-2]])
            resid = Nhat[1:] - star["rho_1"] * Nhat[:-1] - star["rho_2"] * second_lag
            terms.append(
                _log_ig_pdf_var(
                    star["sigma_u2"], pri["a_u"] + 0.5 * resid.size,
                    pri["b_u"] + 0.5 * float(np.sum(resid**2)),
                )
            )
        return _logmeanexp(np.array(terms))

    if block == "n":
        terms = [
            float(_log_norm_pdf(star["n"], *_n_cond_moments(Nbar, sp, pri)))
            for Nbar, sp in zip(Nbar_draws, sigma_eps2_draws)
        ]
        return _logmeanexp(np.array(terms))

    if block == "sigma_eps2":
        terms = []
        for Nbar in Nbar_draws:
            resid = Nbar[1:] - star["n"] - Nbar[:-1]
            terms.append(
                _log_ig_pdf_var(
                    star["sigma_eps2"], pri["a_eps"] + 0.5 * resid.size,
                    pri["b_eps"] + 0.5 * float(np.sum(resid**2)),
                )
            )
        return _logmeanexp(np.array(terms))

    if block == "sigma_N2":
        terms = []
        finite = np.isfinite(data["N"])
        for Nhat, Nbar in zip(Nhat_draws, Nbar_draws):
            resid = data["N"][finite] - Nhat[finite] - Nbar[finite]
            terms.append(
                _log_ig_pdf_var(
                    star["sigma_N2"], pri["a_N"] + 0.5 * resid.size,
                    pri["b_N"] + 0.5 * float(np.sum(resid**2)),
                )
            )
        return _logmeanexp(np.array(terms))

    raise ValueError(f"Unknown ordinate block: {block!r}")


def _beta_prior_arrays(pri, names):
    means = np.array([_prior_pair(pri, name)[0] for name in names], float)
    sds = np.array([_prior_pair(pri, name)[1] for name in names], float)
    return means, sds
