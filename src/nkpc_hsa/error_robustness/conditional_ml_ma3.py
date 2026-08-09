"""Chib (1995) marginal likelihood for the MA(3) error-structure models.

The additive counterpart of
``nkpc_hsa.gibbs.conditional_ml.conditional_marginal_likelihood``, which is left
untouched. Two things change relative to production.

**Every conditional is GLS-weighted.** Production's blocks divide by a scalar
``sigma_eta2``; here the disturbance covariance is ``sigma_v2 * Omega_0(psi)``,
so ``X'X / sigma_eta2`` becomes ``X' Omega_0(psi)^{-1} X / sigma_v2``. Because
``psi`` varies across the reduced run, ``Omega_0`` is rebuilt per draw -- cheap,
since the factorisation is banded.

**There is one Metropolis block, and it is last.** ``psi`` has no conjugate
conditional (see ``ma_error``), so it is drawn by random-walk Metropolis. That
would normally force a Chib-Jeliazkov correction inside every reduced run. It
does not here, because Chib's *final* ordinate factor is the only one evaluated
with every other block already at its starred value, and it needs no averaging:

    p(theta* | y) = p(block_1* | y) * ... * p(psi* | y, everything else starred)

so the last factor is a single conditional density that
``ma_error.log_conditional_psi_ordinate`` computes by normalising numerically
over the invertible region -- validated against a converged 80^3 product grid to
0.001 log units, with a seed-to-seed spread of 0.0014. This is exactly the
device ``conditional_ml.log_stationary_mass`` already uses for the AR(2) block's
truncation, applied to a three-dimensional region instead of a triangle.

Ordering ``psi`` last is therefore not a stylistic choice; it is what keeps the
marginal likelihood computable with the machinery already in the repo.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
import os
from typing import Any, Literal

import numpy as np

from nkpc_hsa.error_robustness.ces_ma3 import func_nkpc_ces_ma3
from nkpc_hsa.error_robustness.hsa_steady_ma3 import func_nkpc_hsa_steady_ma3
from nkpc_hsa.error_robustness.joint_ffbs_ma3 import joint_loglik_ma
from nkpc_hsa.error_robustness.ma_error import (
    MA_ORDER,
    MAWeighting,
    PsiPrior,
    log_conditional_psi_ordinate,
    log_invertible_mass,
)
from nkpc_hsa.gibbs.conditional_ml import (
    KAPPA_SCALE,
    _checked_logmeanexp,
    _draws_from,
    _log_ig_pdf_var,
    _log_mvn_pdf,
    _log_norm_pdf,
    _prior_pair,
    _rho_cond_moments,
    _states_from,
    activity_loglik,
    log_truncated_mvn_pdf,
)

__all__ = [
    "ConditionalComparisonMA",
    "ConditionalMLMAResult",
    "conditional_comparison_ma3",
    "conditional_marginal_likelihood_ma3",
]

Family = Literal["ces", "steady"]


# ---------------------------------------------------------------------------
# GLS-weighted conditional moments
# ---------------------------------------------------------------------------

def _beta_cond_moments_gls(y, X, sigma_v2, weighting, prior_mean, prior_sd):
    prior_prec = np.diag(1.0 / np.asarray(prior_sd, float) ** 2)
    XtWX, XtWy = weighting.gls_moments(y, X)
    cov = np.linalg.inv(XtWX / sigma_v2 + prior_prec)
    mean = cov @ (XtWy / sigma_v2 + prior_prec @ np.asarray(prior_mean, float))
    return mean, cov


def _lambda_cond_moments_gls(e_base, zeta, sigma_v2, weighting, pri):
    mu0, sd0 = _prior_pair(pri, "lambda_ez")
    Wz = weighting.solve(zeta)
    prec = 1.0 / sd0**2 + float(zeta @ Wz) / sigma_v2
    var = 1.0 / prec
    return var * (mu0 / sd0**2 + float(e_base @ Wz) / sigma_v2), np.sqrt(var)


def _phi_cond_moments_gls(x, x_prev, y_tilde, lambda_ez, sigma_zeta2, sigma_v2, weighting, pri):
    mu0, sd0 = _prior_pair(pri, "phi_1")
    Wxp = weighting.solve(x_prev)
    prec = (
        1.0 / sd0**2
        + float(np.sum(x_prev**2)) / sigma_zeta2
        + (lambda_ez**2) * float(x_prev @ Wxp) / sigma_v2
    )
    mean = (
        mu0 / sd0**2
        + float(np.dot(x_prev, x)) / sigma_zeta2
        - lambda_ez * float((y_tilde - lambda_ez * x) @ Wxp) / sigma_v2
    ) / prec
    return mean, np.sqrt(1.0 / prec)


# ---------------------------------------------------------------------------
# Likelihoods with the MA disturbance
# ---------------------------------------------------------------------------

def ces_ma3_loglik(star: dict, data: dict) -> float:
    """``log p(pi | x, theta)`` for CES with an MA(q) disturbance."""
    y = data["pi"] - data["pi_expect"]
    a_t = data["pi_prev"] - data["pi_expect"]
    zeta = data["x"] - star["phi_1"] * data["x_prev"]
    xi = y - star["alpha"] * a_t - star["kappa"] * data["x"] - star["lambda_ez"] * zeta
    return MAWeighting(star["psi"], y.size).log_likelihood(xi, star["sigma_v2"])


def steady_ma3_joint_loglik(star: dict, data: dict, *, m0, P0) -> float:
    """``log p(pi, N_obs | x, theta)`` for HSA steady, states integrated out."""
    y = data["pi"] - data["pi_expect"]
    a_t = data["pi_prev"] - data["pi_expect"]
    zeta = data["x"] - star["phi_1"] * data["x_prev"]
    y_tilde = y - star["alpha"] * a_t - star["kappa_0"] * data["x"] - star["lambda_ez"] * zeta
    return joint_loglik_ma(
        N_obs=data["N"],
        y_tilde=y_tilde,
        h_nhat=np.zeros(y.size),
        h_nbar=star["delta"] * data["x"],
        n_drift=star["n"],
        rho1=star["rho_1"],
        rho2=star["rho_2"],
        psi=star["psi"],
        sigma_v2=star["sigma_v2"],
        sigma_u2=star["sigma_u2"],
        sigma_eps2=star["sigma_eps2"],
        sigma_N2=star["sigma_N2"],
        m0=m0,
        P0=P0,
    )


def log_prior_ma3(star: dict, pri: dict, psi_prior: PsiPrior, *, family: Family) -> float:
    """Production prior plus the psi term, normalised over the invertible region."""
    out = _log_norm_pdf(star["alpha"], *_prior_pair(pri, "alpha"))
    if family == "ces":
        out += _log_norm_pdf(star["kappa"], *_prior_pair(pri, "kappa"))
    else:
        out += _log_norm_pdf(star["kappa_0"], *_prior_pair(pri, "kappa_0"))
        out += _log_norm_pdf(star["delta"], *_prior_pair(pri, "delta"))
    out += _log_norm_pdf(star["lambda_ez"], *_prior_pair(pri, "lambda_ez"))
    out += _log_norm_pdf(star["phi_1"], *_prior_pair(pri, "phi_1"))
    out += _log_ig_pdf_var(star["sigma_zeta2"], pri["a_z"], pri["b_z"])
    out += _log_ig_pdf_var(star["sigma_v2"], pri["a_e"], pri["b_e"])
    if family == "steady":
        mu_rho = np.array([_prior_pair(pri, "rho_1")[0], _prior_pair(pri, "rho_2")[0]])
        cov_rho = np.diag(
            [_prior_pair(pri, "rho_1")[1] ** 2, _prior_pair(pri, "rho_2")[1] ** 2]
        )
        out += log_truncated_mvn_pdf([star["rho_1"], star["rho_2"]], mu_rho, cov_rho)
        out += _log_ig_pdf_var(star["sigma_u2"], pri["a_u"], pri["b_u"])
        out += _log_norm_pdf(star["n"], *_prior_pair(pri, "n"))
        out += _log_ig_pdf_var(star["sigma_eps2"], pri["a_eps"], pri["b_eps"])
        out += _log_ig_pdf_var(star["sigma_N2"], pri["a_N"], pri["b_N"])

    # psi: independent Gaussians renormalised over the invertible region. The
    # untruncated log density minus the log mass the truncation removes.
    if star["psi"].size:
        out += psi_prior.log_pdf(star["psi"])
        out -= log_invertible_mass(psi_prior.mean, np.diag(psi_prior.sd**2))
    return float(out)


# ---------------------------------------------------------------------------
# Reduced runs and star
# ---------------------------------------------------------------------------

def _run_gibbs_ma3(sampler, data, priors_internal, *, family, fixed, ma_order,
                   n_burn, n_keep, seed, thin=1):
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
            "ma_order": int(ma_order),
            "return_state_lag": True,
        },
        orth=False,
    )
    if family == "steady":
        kwargs["N_data"] = data["N"]
    return sampler(**kwargs)


def _star_from_posterior_ma3(result: dict, *, family: Family) -> dict[str, Any]:
    star: dict[str, Any] = {
        "alpha": float(np.mean(_draws_from(result, "alpha"))),
        "phi_1": float(np.mean(_draws_from(result, "phi_1"))),
        "lambda_ez": float(np.mean(_draws_from(result, "lambda_ez"))),
        "psi": np.asarray(np.mean(_draws_from(result, "psi"), axis=0), dtype=float)
        if "psi" in result
        else np.zeros(0),
    }
    if family == "ces":
        star["sigma_zeta2"] = float(np.mean(_draws_from(result, "sigma_zeta2")))
        star["kappa"] = float(np.mean(_draws_from(result, "kappa")))
        star["sigma_v2"] = float(np.mean(_draws_from(result, "sigma_v2")))
    else:
        star["sigma_zeta2"] = float(np.mean(_draws_from(result, "sigma_zeta")) ** 2)
        star["kappa_0"] = float(np.mean(_draws_from(result, "kappa_0")))
        star["delta"] = float(np.mean(_draws_from(result, "delta")))
        star["sigma_v2"] = float(np.mean(_draws_from(result, "sigma_eta")) ** 2)
        star["rho_1"] = float(np.mean(_draws_from(result, "rho1")))
        star["rho_2"] = float(np.mean(_draws_from(result, "rho2")))
        star["sigma_u2"] = float(np.mean(_draws_from(result, "sigma_u")) ** 2)
        star["n"] = float(np.mean(_draws_from(result, "n")))
        star["sigma_eps2"] = float(np.mean(_draws_from(result, "sigma_eps")) ** 2)
        star["sigma_N2"] = float(np.mean(_draws_from(result, "sigma_N")) ** 2)
    return star


def _psi_draws(run: dict, ma_order: int, n: int) -> np.ndarray:
    if ma_order == 0:
        return np.zeros((n, 0))
    return _draws_from(run, "psi").reshape(n, ma_order)


def _ordinate_factor_ma3(block, run, star, data, pri, psi_prior, *, family, y, a_t, ma_order):
    """log of one Chib ordinate factor, Rao-Blackwellised over the run's draws."""
    x, x_prev = data["x"], data["x_prev"]
    T = y.size

    if family == "ces":
        sigma_v2_draws = _draws_from(run, "sigma_v2")
        sigma_zeta2_draws = _draws_from(run, "sigma_zeta2")
    else:
        sigma_v2_draws = _draws_from(run, "sigma_eta") ** 2
        sigma_zeta2_draws = _draws_from(run, "sigma_zeta") ** 2
    lambda_draws = _draws_from(run, "lambda_ez")
    phi_draws = _draws_from(run, "phi_1")
    psis = _psi_draws(run, ma_order, lambda_draws.size)
    weightings = [MAWeighting(p, T) for p in psis]

    if block == "beta":
        if family == "ces":
            beta_star = np.array([star["alpha"], star["kappa"]])
            pm = np.array([_prior_pair(pri, "alpha")[0], _prior_pair(pri, "kappa")[0]])
            ps = np.array([_prior_pair(pri, "alpha")[1], _prior_pair(pri, "kappa")[1]])
            terms = []
            for lmb, phi, sv, W in zip(lambda_draws, phi_draws, sigma_v2_draws, weightings):
                X = np.column_stack([a_t, x])
                mean, cov = _beta_cond_moments_gls(
                    y - lmb * (x - phi * x_prev), X, sv, W, pm, ps
                )
                terms.append(_log_mvn_pdf(beta_star, mean, cov))
        else:
            beta_star = np.array([star["alpha"], star["kappa_0"], star["delta"]])
            names = ["alpha", "kappa_0", "delta"]
            pm = np.array([_prior_pair(pri, nm)[0] for nm in names])
            ps = np.array([_prior_pair(pri, nm)[1] for nm in names])
            Nbar_draws = _states_from(run, "Nbar")
            terms = []
            for Nbar, lmb, phi, sv, W in zip(
                Nbar_draws, lambda_draws, phi_draws, sigma_v2_draws, weightings
            ):
                X = np.column_stack([a_t, x, x * Nbar])
                mean, cov = _beta_cond_moments_gls(
                    y - lmb * (x - phi * x_prev), X, sv, W, pm, ps
                )
                terms.append(_log_mvn_pdf(beta_star, mean, cov))
        return _checked_logmeanexp(np.array(terms), block=block)

    kappa_term = (
        star["kappa"] * x
        if family == "ces"
        else (star["kappa_0"] + star["delta"] * _states_from(run, "Nbar")) * x
    )

    if block == "lambda_ez":
        terms = []
        for i, (phi, sv, W) in enumerate(zip(phi_draws, sigma_v2_draws, weightings)):
            zeta = x - phi * x_prev
            kt = kappa_term if family == "ces" else kappa_term[i]
            mean, sd = _lambda_cond_moments_gls(
                y - star["alpha"] * a_t - kt, zeta, sv, W, pri
            )
            terms.append(float(_log_norm_pdf(star["lambda_ez"], mean, sd)))
        return _checked_logmeanexp(np.array(terms), block=block)

    if block == "phi_1":
        terms = []
        for i, (sz, sv, W) in enumerate(zip(sigma_zeta2_draws, sigma_v2_draws, weightings)):
            kt = kappa_term if family == "ces" else kappa_term[i]
            mean, sd = _phi_cond_moments_gls(
                x, x_prev, y - star["alpha"] * a_t - kt, star["lambda_ez"], sz, sv, W, pri
            )
            terms.append(float(_log_norm_pdf(star["phi_1"], mean, sd)))
        return _checked_logmeanexp(np.array(terms), block=block)

    zeta_star = x - star["phi_1"] * x_prev

    if block == "sigma_zeta2":
        return _log_ig_pdf_var(
            star["sigma_zeta2"],
            pri["a_z"] + 0.5 * T,
            pri["b_z"] + 0.5 * float(np.sum(zeta_star**2)),
        )

    if block == "sigma_v2":
        terms = []
        n_draws = len(weightings)
        for i in range(n_draws):
            kt = kappa_term if family == "ces" else kappa_term[i]
            xi = y - star["alpha"] * a_t - kt - star["lambda_ez"] * zeta_star
            terms.append(
                _log_ig_pdf_var(
                    star["sigma_v2"],
                    pri["a_e"] + 0.5 * T,
                    pri["b_e"] + 0.5 * weightings[i].quadratic_form(xi),
                )
            )
        return _checked_logmeanexp(np.array(terms), block=block)

    if block == "psi":
        # The final factor: everything else is already starred, so no averaging.
        kt = star["kappa"] * x if family == "ces" else None
        if family == "steady":
            Nbar_star = np.mean(_states_from(run, "Nbar"), axis=0)
            kt = (star["kappa_0"] + star["delta"] * Nbar_star) * x
        xi = y - star["alpha"] * a_t - kt - star["lambda_ez"] * zeta_star
        return log_conditional_psi_ordinate(
            star["psi"], xi, star["sigma_v2"], prior=psi_prior
        )

    # --- HSA-steady state blocks: unchanged from production ---
    Nhat_draws = _states_from(run, "Nhat")
    Nbar_draws = _states_from(run, "Nbar")
    sigma_u2_draws = _draws_from(run, "sigma_u") ** 2
    sigma_eps2_draws = _draws_from(run, "sigma_eps") ** 2
    nhat_lag_draws = _states_from(run, "Nhat_lag")

    if block == "rho":
        terms = [
            log_truncated_mvn_pdf(
                [star["rho_1"], star["rho_2"]], *_rho_cond_moments(Nhat, lag, su, pri)
            )
            for Nhat, lag, su in zip(Nhat_draws, nhat_lag_draws, sigma_u2_draws)
        ]
        return _checked_logmeanexp(np.array(terms), block=block)

    if block == "sigma_u2":
        terms = []
        for Nhat, lag in zip(Nhat_draws, nhat_lag_draws):
            second_lag = np.concatenate([[float(lag)], Nhat[:-2]])
            resid = Nhat[1:] - star["rho_1"] * Nhat[:-1] - star["rho_2"] * second_lag
            terms.append(
                _log_ig_pdf_var(
                    star["sigma_u2"],
                    pri["a_u"] + 0.5 * resid.size,
                    pri["b_u"] + 0.5 * float(np.sum(resid**2)),
                )
            )
        return _checked_logmeanexp(np.array(terms), block=block)

    if block == "n":
        from nkpc_hsa.gibbs.conditional_ml import _n_cond_moments

        terms = [
            float(_log_norm_pdf(star["n"], *_n_cond_moments(Nbar, se, pri)))
            for Nbar, se in zip(Nbar_draws, sigma_eps2_draws)
        ]
        return _checked_logmeanexp(np.array(terms), block=block)

    if block == "sigma_eps2":
        terms = []
        for Nbar in Nbar_draws:
            resid = Nbar[1:] - star["n"] - Nbar[:-1]
            terms.append(
                _log_ig_pdf_var(
                    star["sigma_eps2"],
                    pri["a_eps"] + 0.5 * resid.size,
                    pri["b_eps"] + 0.5 * float(np.sum(resid**2)),
                )
            )
        return _checked_logmeanexp(np.array(terms), block=block)

    if block == "sigma_N2":
        from nkpc_hsa.gibbs.common.competition import finite_N_residuals

        terms = []
        for Nhat, Nbar in zip(Nhat_draws, Nbar_draws):
            resid = finite_N_residuals(data["N"], Nhat, Nbar)
            terms.append(
                _log_ig_pdf_var(
                    star["sigma_N2"],
                    pri["a_N"] + 0.5 * resid.size,
                    pri["b_N"] + 0.5 * float(np.sum(resid**2)),
                )
            )
        return _checked_logmeanexp(np.array(terms), block=block)

    raise ValueError(f"Unknown block {block!r}")


@dataclass
class ConditionalMLMAResult:
    log_marginal_likelihood: float
    log_likelihood: float
    log_prior: float
    log_posterior_ordinate: float
    family: str
    ma_order: int
    ordinate_terms: dict[str, float] = field(default_factory=dict)
    star: dict = field(default_factory=dict)
    notes: str = ""

    def as_dict(self) -> dict[str, Any]:
        out = {
            "log_marginal_likelihood": self.log_marginal_likelihood,
            "log_likelihood": self.log_likelihood,
            "log_prior": self.log_prior,
            "log_posterior_ordinate": self.log_posterior_ordinate,
            "family": self.family,
            "ma_order": self.ma_order,
            "ordinate_terms": self.ordinate_terms,
            "notes": self.notes,
        }
        return out


def conditional_marginal_likelihood_ma3(
    data: dict[str, np.ndarray],
    priors_internal: dict,
    pri: dict,
    *,
    family: Family,
    ma_order: int = MA_ORDER,
    n_burn: int = 1500,
    n_keep: int = 3000,
    seed: int = 90210,
    m0: np.ndarray | None = None,
    P0: np.ndarray | None = None,
    max_workers: int | None = None,
) -> ConditionalMLMAResult:
    """Chib (1995) marginal likelihood for the MA(q) specification.

    Returns ``log m(pi, x)`` for CES and ``log m(pi, N_obs, x)`` for HSA steady,
    i.e. the joint target the samplers' posteriors are actually built from.
    """
    if family == "ces":
        sampler = func_nkpc_ces_ma3
        blocks = ["beta", "lambda_ez", "phi_1", "sigma_zeta2", "sigma_v2"]
    elif family == "steady":
        sampler = func_nkpc_hsa_steady_ma3
        blocks = [
            "beta", "lambda_ez", "phi_1", "sigma_zeta2", "sigma_v2",
            "rho", "sigma_u2", "n", "sigma_eps2", "sigma_N2",
        ]
    else:
        raise ValueError(f"Unsupported family: {family!r}")
    if ma_order:
        blocks.append("psi")  # last, for the reason in the module docstring

    # Built at the real order when there is an MA block, and at a harmless
    # order-1 default otherwise so the signature of the helpers stays uniform;
    # with ma_order == 0 nothing ever reads it.
    psi_prior = PsiPrior.from_config(pri, order=ma_order if ma_order else 1)

    m0 = np.zeros(3) if m0 is None else np.asarray(m0, float)
    P0 = np.eye(3) * 10.0 if P0 is None else np.asarray(P0, float)

    y = data["pi"] - data["pi_expect"]
    a_t = data["pi_prev"] - data["pi_expect"]

    full = _run_gibbs_ma3(
        sampler, data, priors_internal, family=family, fixed={}, ma_order=ma_order,
        n_burn=n_burn, n_keep=n_keep, seed=seed,
    )
    star = _star_from_posterior_ma3(full, family=family)

    star_values: dict[str, Any] = {
        "beta": (
            (star["alpha"], star["kappa"]) if family == "ces"
            else (star["alpha"], star["kappa_0"], star["delta"])
        ),
        "lambda_ez": star["lambda_ez"],
        "phi_1": star["phi_1"],
        "sigma_zeta2": star["sigma_zeta2"],
        "sigma_v2": star["sigma_v2"],
    }
    if family == "steady":
        star_values.update({
            "rho": (star["rho_1"], star["rho_2"]),
            "sigma_u2": star["sigma_u2"],
            "n": star["n"],
            "sigma_eps2": star["sigma_eps2"],
            "sigma_N2": star["sigma_N2"],
        })
    if ma_order:
        star_values["psi"] = star["psi"]

    fixed_internal = dict(star_values)
    fixed_internal["beta"] = tuple(
        v * (KAPPA_SCALE if i > 0 else 1.0) for i, v in enumerate(star_values["beta"])
    )

    jobs = [
        (g, block, {b: fixed_internal[b] for b in blocks[:g]}, seed + 101 * g)
        for g, block in enumerate(blocks)
    ]
    runs_by_block: dict[str, Any] = {blocks[0]: full}
    todo = [job for job in jobs if job[0] > 0]
    if todo:
        workers = max_workers or min(len(todo), max(1, (os.cpu_count() or 2) - 1))
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(
                    _run_gibbs_ma3, sampler, data, priors_internal, family=family,
                    fixed=fixed, ma_order=ma_order, n_burn=n_burn, n_keep=n_keep,
                    seed=run_seed,
                ): block
                for _, block, fixed, run_seed in todo
            }
            for future in as_completed(futures):
                runs_by_block[futures[future]] = future.result()

    ordinate_terms = {
        block: _ordinate_factor_ma3(
            block, runs_by_block[block], star, data, pri, psi_prior,
            family=family, y=y, a_t=a_t, ma_order=ma_order,
        )
        for block in blocks
    }
    log_ord = float(sum(ordinate_terms.values()))

    if family == "ces":
        log_lik = ces_ma3_loglik(star, data) + activity_loglik(star, data)
    else:
        log_lik = steady_ma3_joint_loglik(star, data, m0=m0, P0=P0) + activity_loglik(star, data)
    log_pri = log_prior_ma3(star, pri, psi_prior, family=family)

    return ConditionalMLMAResult(
        log_marginal_likelihood=float(log_lik + log_pri - log_ord),
        log_likelihood=float(log_lik),
        log_prior=float(log_pri),
        log_posterior_ordinate=log_ord,
        family=family,
        ma_order=ma_order,
        ordinate_terms=ordinate_terms,
        star=star,
        notes=(
            "Chib (1995) with explicit reduced Gibbs runs; states integrated out by "
            "the augmented Kalman filter; every coefficient conditional GLS-weighted "
            "by Omega_0(psi); psi placed last so its ordinate is a single numerical "
            "normalisation over the invertible region (no Chib-Jeliazkov)."
        ),
    )


# ---------------------------------------------------------------------------
# Conditional comparison
# ---------------------------------------------------------------------------

@dataclass
class ConditionalComparisonMA:
    """Every component of ``log p(pi | x, N_obs, M)``, kept separately.

    The components are reported rather than only their combination because the
    point of the correction is which Occam factors cancel: ``log_m_N`` and
    ``log_m_x`` are charged in the joint term and refunded in the conditioning
    term.
    """

    log_m_joint: float
    log_m_firm_count: float
    log_m_activity: float
    log_m_conditional: float
    family: str
    ma_order: int
    star: dict = field(default_factory=dict)
    ordinate_terms: dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "log_m_joint": self.log_m_joint,
            "log_m_firm_count": self.log_m_firm_count,
            "log_m_activity": self.log_m_activity,
            "log_m_conditional": self.log_m_conditional,
            "family": self.family,
            "ma_order": self.ma_order,
            "ordinate_terms": self.ordinate_terms,
        }


def conditional_comparison_ma3(
    data: dict[str, np.ndarray],
    priors_internal: dict,
    pri: dict,
    *,
    family: Family,
    ma_order: int = MA_ORDER,
    n_burn: int = 1500,
    n_keep: int = 3000,
    seed: int = 90210,
    m0: np.ndarray | None = None,
    P0: np.ndarray | None = None,
    log_m_activity: float | None = None,
    log_m_firm_count: float | None = None,
) -> ConditionalComparisonMA:
    """``log p(pi | x, N_obs, M)`` as a ratio of full marginal likelihoods.

        HSA steady   log m(pi, N, x) - log m(N) - log m(x)
        CES          log m(pi, x)              - log m(x)

    Only the joint term changes with the error structure. ``m(N)`` is the
    firm-count block alone -- ``N_obs = Nhat + Nbar + nu`` with its AR(2) and
    random-walk dynamics -- and ``m(x)`` is the activity equation alone. Neither
    contains the inflation disturbance, so neither contains ``psi``, and both are
    computed by the *production* routines rather than reimplemented. That is not
    just economy: it guarantees the iid and MA(3) conditional likelihoods are
    debited numerically identical Occam factors, which is what makes them
    comparable to each other as well as across models.

    Pass ``log_m_activity`` / ``log_m_firm_count`` to share those terms across
    calls, so CES and HSA -- and iid and MA(3) -- are debited the same number
    rather than two Monte Carlo estimates of it.
    """
    from nkpc_hsa.gibbs.conditional_ml import (
        activity_marginal_likelihood,
        firm_count_marginal_likelihood,
    )

    if family not in ("ces", "steady"):
        raise ValueError(
            f"conditional_comparison_ma3 supports 'ces' and 'steady', not {family!r}. "
            "This mirrors production: hsa_dynamic's shocks are jointly distributed "
            "across a 4x4 Sigma and hsa_full's state is not linear-Gaussian, so for "
            "neither model does m(N) factor out of the joint the way this identity "
            "needs. Their MA(3) posteriors are available; their marginal "
            "likelihoods are not."
        )

    m0 = np.zeros(3) if m0 is None else np.asarray(m0, float)
    P0 = np.eye(3) * 10.0 if P0 is None else np.asarray(P0, float)

    joint = conditional_marginal_likelihood_ma3(
        data, priors_internal, pri, family=family, ma_order=ma_order,
        n_burn=n_burn, n_keep=n_keep, seed=seed, m0=m0, P0=P0,
    )

    if log_m_activity is None:
        log_m_activity, _ = activity_marginal_likelihood(data, pri, seed=seed)
    log_m_firm = 0.0
    if family == "steady":
        if log_m_firm_count is None:
            log_m_firm, _, _ = firm_count_marginal_likelihood(
                data, pri, m0=m0, P0=P0, n_burn=n_burn, n_keep=n_keep, seed=seed
            )
        else:
            log_m_firm = float(log_m_firm_count)

    return ConditionalComparisonMA(
        log_m_joint=joint.log_marginal_likelihood,
        log_m_firm_count=float(log_m_firm),
        log_m_activity=float(log_m_activity),
        log_m_conditional=float(
            joint.log_marginal_likelihood - log_m_firm - float(log_m_activity)
        ),
        family=family,
        ma_order=ma_order,
        star=joint.star,
        ordinate_terms=joint.ordinate_terms,
    )
