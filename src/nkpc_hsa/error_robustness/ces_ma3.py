"""CES NKPC with an MA(3) inflation disturbance.

An additive counterpart to ``nkpc_hsa.gibbs.ces.model.func_nkpc_ces`` -- the
sampler production's ``run_model`` and Chib routines actually dispatch to, *not*
the deprecated ``gibbs.gibbs_ces``. The distinction matters: only the production
one applies ``KAPPA_SCALE``, so a copy of the legacy module silently disagrees
with every caller that passes or reads a kappa. That module is untouched; this
one carries its own copy of every block, generalised from i.i.d. to

    pi_t - E_t pi_{t+1} = alpha * (pi_{t-1} - E_t pi_{t+1}) + kappa * x_t + e_t
    e_t                 = lambda_ez * zeta_t + xi_t
    xi_t                = psi(L) v_t,            v_t ~ iid N(0, sigma_v^2)
    x_t                 = phi_1 * x_{t-1} + zeta_t,  zeta_t ~ N(0, sigma_zeta^2)

Setting ``psi = 0`` collapses every block below to the production sampler's
algebra, which ``tests/test_error_robustness_ma3.py`` checks numerically.

Block order
-----------
``(alpha, kappa)`` -> ``lambda_ez`` -> ``phi_1`` -> ``sigma_zeta^2`` ->
``sigma_v^2`` -> ``psi``.

``psi`` is deliberately **last**. It is the only Metropolis block, and Chib's
final ordinate factor is the one evaluated with every other block at its
starred value -- so putting it last means the marginal likelihood needs a single
numerical normalisation of ``p(psi* | .)`` rather than a Chib-Jeliazkov
correction inside every reduced run. See ``ma_error.log_conditional_psi_ordinate``.

Why psi cannot be a Gibbs block
-------------------------------
Conditional on the coefficients, ``xi`` is data; conditional on ``psi``, the
innovation path ``v`` follows by deterministic inverse recursion. A Gibbs pair
over ``(psi, v)`` is therefore reducible and would never move. Metropolis on the
exact banded likelihood is the fix.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from nkpc_hsa.error_robustness.ma_error import (
    MA_ORDER,
    AdaptiveRandomWalk,
    MAWeighting,
    PsiPrior,
    autocovariance,
    sample_psi,
)
from nkpc_hsa.gibbs.common.constraints import constraint_stats_summary, draw_with_constraints
from nkpc_hsa.gibbs.ces.model import KAPPA_SCALE
from nkpc_hsa.gibbs.gibbs_utils import assert_all_pos, getd, mvnrnd, sample_invgamma

__all__ = ["func_nkpc_ces_ma3"]


def _summary(draws: np.ndarray) -> dict[str, Any]:
    """Posterior draws plus the summaries the rest of the pipeline expects."""
    arr = np.asarray(draws, dtype=float)
    qs = np.quantile(arr, [0.025, 0.05, 0.25, 0.5, 0.75, 0.95, 0.975], axis=0)
    return {
        "draws": arr,
        "mean": np.mean(arr, axis=0),
        "std": np.std(arr, axis=0, ddof=1)
        if arr.shape[0] > 1
        else np.zeros_like(np.mean(arr, axis=0)),
        "quantiles": qs,
    }


def func_nkpc_ces_ma3(
    pi_data,
    pi_prev_data,
    Epi_data,
    x_data,
    x_prev_data,
    n_burn: int,
    n_keep: int,
    priors: Optional[dict[str, Any]] = None,
    opts: Optional[dict[str, Any]] = None,
    *,
    orth: bool = False,
) -> dict[str, Any]:
    """Gibbs/Metropolis sampler for the CES NKPC with an MA(q) disturbance.

    ``opts`` additions over the production sampler
    ---------------------------------------------
    ``ma_order``        MA order, default 3 (0 reproduces the i.i.d. baseline)
    ``psi0``            starting value, default zeros
    ``n_psi_steps``     Metropolis steps per sweep, default 2
    ``psi_init_scale``  initial random-walk scale, default 0.08
    """
    pi_t = np.asarray(pi_data, dtype=float).reshape(-1)
    pi_tm1 = np.asarray(pi_prev_data, dtype=float).reshape(-1)
    pi_expect = np.asarray(Epi_data, dtype=float).reshape(-1)
    x_t = np.asarray(x_data, dtype=float).reshape(-1)
    x_tm1 = np.asarray(x_prev_data, dtype=float).reshape(-1)

    T = pi_t.size
    if not (pi_tm1.size == pi_expect.size == x_t.size == x_tm1.size == T):
        raise ValueError("All input series must have the same length.")

    priors = priors or {}
    mu_alpha = getd(priors, "mu_alpha", 0.5)
    sigma_alpha = getd(priors, "sigma_alpha", 0.2)
    mu_kappa = getd(priors, "mu_kappa", 0.1)
    sigma_kappa = getd(priors, "sigma_kappa", 0.2)
    mu_phi = getd(priors, "mu_phi_1", 0.7)
    sigma_phi = getd(priors, "sigma_phi_1", 0.2)
    mu_lambda = getd(priors, "mu_lambda", 0.0)
    sigma_lambda = getd(priors, "sigma_lambda", 0.5)

    a_e = getd(priors, "a_e", getd(priors, "a_v", 2.0))
    b_e = getd(priors, "b_e", getd(priors, "b_v", 2.0))
    a_z = getd(priors, "a_z", 0.001)
    b_z = getd(priors, "b_z", 0.001)

    assert_all_pos(
        [sigma_alpha, sigma_kappa, sigma_phi, sigma_lambda, a_e, b_e, a_z, b_z],
        "CES prior scales must be positive.",
    )

    opts = opts or {}
    ma_order = int(getd(opts, "ma_order", MA_ORDER))
    if ma_order < 0:
        raise ValueError("ma_order must be nonnegative.")
    psi_prior = PsiPrior.from_config(priors, order=ma_order) if ma_order else None

    alpha = float(getd(opts, "alpha0", mu_alpha))
    kappa = float(getd(opts, "kappa0", mu_kappa))
    phi_1 = float(getd(opts, "phi10", mu_phi))
    lambda_ez = 0.0 if orth else float(getd(opts, "lambda0", 0.0))
    sigma_v2 = float(getd(opts, "sigma_v20", getd(opts, "sigma_e20", 1.0)))
    sigma_zeta2 = float(getd(opts, "sigma_zeta20", 1.0))
    psi = np.asarray(getd(opts, "psi0", np.zeros(ma_order)), dtype=float).reshape(-1)
    if psi.size != ma_order:
        raise ValueError(f"psi0 must have length ma_order={ma_order}.")

    n_psi_steps = int(getd(opts, "n_psi_steps", 2))
    coefficient_constraints = getd(opts, "coefficient_constraints", {})
    constraint_stats: dict[str, int] = {}

    # Reduced-run support for Chib's marginal likelihood: ``opts["fixed"]`` pins
    # named blocks at supplied values and skips their draw, so the reduced runs
    # reuse these exact conditionals rather than a reimplementation. Mirrors the
    # mechanism hsa_steady already has.
    fixed = dict(getd(opts, "fixed", {}) or {})
    unknown_fixed = set(fixed) - {"beta", "lambda_ez", "phi_1", "sigma_zeta2", "sigma_v2", "sigma_eta2", "psi"}
    if unknown_fixed:
        raise ValueError(f"Unknown fixed block(s): {sorted(unknown_fixed)}")
    if "beta" in fixed:
        alpha, kappa = (float(v) for v in fixed["beta"])
    if "lambda_ez" in fixed:
        lambda_ez = float(fixed["lambda_ez"])
    if "phi_1" in fixed:
        phi_1 = float(fixed["phi_1"])
    if "sigma_zeta2" in fixed:
        sigma_zeta2 = float(fixed["sigma_zeta2"])
    # "sigma_eta2" is accepted as an alias so Chib code written against the
    # production block names keeps working.
    if "sigma_v2" in fixed or "sigma_eta2" in fixed:
        sigma_v2 = float(fixed.get("sigma_v2", fixed.get("sigma_eta2")))
    if "psi" in fixed:
        psi = np.asarray(fixed["psi"], dtype=float).reshape(-1)
        if psi.size != ma_order:
            raise ValueError(f"fixed['psi'] must have length ma_order={ma_order}.")

    seed = getd(opts, "seed", None)
    store_every = int(max(1, getd(opts, "store_every", 1)))
    verbose = bool(getd(opts, "verbose", False))
    # Display-only hook installed by the run driver; it never touches the draws.
    progress_callback = getd(opts, "progress_callback", None)
    rng = np.random.default_rng(seed)

    proposal = (
        AdaptiveRandomWalk(ma_order, init_scale=float(getd(opts, "psi_init_scale", 0.08)))
        if ma_order
        else None
    )
    weighting = MAWeighting(psi, T)

    n_store = int(n_keep // store_every)
    alpha_draws = np.zeros(n_store)
    kappa_draws = np.zeros(n_store)
    phi_draws = np.zeros(n_store)
    lambda_draws = np.zeros(n_store)
    sigma_v2_draws = np.zeros(n_store)
    sigma_e2_draws = np.zeros(n_store)
    sigma_zeta2_draws = np.zeros(n_store)
    rho_draws = np.zeros(n_store)
    psi_draws = np.zeros((n_store, ma_order))
    # Implied autocorrelation of the *reduced-form* disturbance e_t at lags 1..q,
    # which is what the residual diagnostic in the report is plotted against.
    e_acf_draws = np.zeros((n_store, ma_order))

    total_iter = n_burn + n_keep
    store_idx = 0

    a_t = pi_tm1 - pi_expect
    y = pi_t - pi_expect
    prior_mean = np.array([mu_alpha, mu_kappa], dtype=float)
    prior_prec = np.diag([1.0 / sigma_alpha**2, 1.0 / sigma_kappa**2])
    phi_prec0 = 1.0 / sigma_phi**2
    lambda_prec0 = 0.0 if orth else 1.0 / sigma_lambda**2
    # kappa is sampled on the KAPPA_SCALE-multiplied scale, matching production:
    # the regression column is x_t / KAPPA_SCALE and stored draws are physical.
    X = np.column_stack([a_t, x_t / KAPPA_SCALE])

    for it in range(1, total_iter + 1):
        zeta = x_t - phi_1 * x_tm1

        # ---- 1. (alpha, kappa) by GLS under Omega_0(psi) ----
        y_adj = y - lambda_ez * zeta
        XtWX, XtWy = weighting.gls_moments(y_adj, X)
        post_cov = np.linalg.inv(XtWX / sigma_v2 + prior_prec)
        post_mean = post_cov @ (XtWy / sigma_v2 + prior_prec @ prior_mean)
        if "beta" not in fixed:
            beta = draw_with_constraints(
                lambda: mvnrnd(post_mean, post_cov, rng),
                ("alpha", "kappa"),
                coefficient_constraints,
                stats=constraint_stats,
            )
            alpha = float(beta[0])
            kappa = float(beta[1])
        kappa_eff = kappa / KAPPA_SCALE

        # ---- 2. cross-equation loading ----
        e_base = y - alpha * a_t - kappa_eff * x_t
        if not orth and "lambda_ez" not in fixed:
            Wzeta = weighting.solve(zeta)
            post_var_lambda = 1.0 / (lambda_prec0 + float(zeta @ Wzeta) / sigma_v2)
            post_mean_lambda = post_var_lambda * (
                mu_lambda * lambda_prec0 + float(e_base @ Wzeta) / sigma_v2
            )
            lambda_ez = float(
                post_mean_lambda + np.sqrt(post_var_lambda) * rng.standard_normal()
            )
        else:
            lambda_ez = 0.0

        # ---- 3. phi_1: enters the x equation and, through lambda*zeta, the
        # inflation equation. Same algebra as production, GLS-weighted.
        g = e_base - lambda_ez * x_t          # xi = g + lambda * phi * x_{t-1}
        Wx_prev = weighting.solve(x_tm1)
        prec_phi = (
            phi_prec0
            + float(np.sum(x_tm1**2)) / sigma_zeta2
            + (lambda_ez**2) * float(x_tm1 @ Wx_prev) / sigma_v2
        )
        mean_num_phi = (
            mu_phi * phi_prec0
            + float(np.dot(x_tm1, x_t)) / sigma_zeta2
            - lambda_ez * float(g @ Wx_prev) / sigma_v2
        )
        if "phi_1" not in fixed:
            phi_1 = float(mean_num_phi / prec_phi + rng.standard_normal() / np.sqrt(prec_phi))

        # ---- 4-5. variances, on the updated innovations ----
        zeta = x_t - phi_1 * x_tm1
        xi = y - alpha * a_t - kappa_eff * x_t - lambda_ez * zeta

        if "sigma_zeta2" not in fixed:
            sigma_zeta2 = sample_invgamma(
                a_z + 0.5 * T,
                b_z + 0.5 * float(np.sum(zeta**2)),
                rng,
            )
        if not ("sigma_v2" in fixed or "sigma_eta2" in fixed):
            sigma_v2 = sample_invgamma(
                a_e + 0.5 * T,
                b_e + 0.5 * weighting.quadratic_form(xi),
                rng,
            )

        # ---- 6. psi, last: random-walk Metropolis on the banded likelihood ----
        if ma_order and "psi" not in fixed:
            psi, weighting = sample_psi(
                psi,
                xi,
                sigma_v2,
                prior=psi_prior,
                proposal=proposal,
                rng=rng,
                n_steps=n_psi_steps,
                weighting=weighting,
            )
            if it == n_burn:
                proposal.freeze()

        # Reduced-form disturbance moments. lambda*zeta is serially independent,
        # so it shifts only the lag-0 variance; every autocovariance at lag >= 1
        # comes from the MA block.
        gamma = autocovariance(psi, sigma_v2)
        sigma_e2 = lambda_ez**2 * sigma_zeta2 + gamma[0]
        rho_corr = 0.0 if orth else float(
            (lambda_ez * np.sqrt(sigma_zeta2)) / max(np.sqrt(sigma_e2), 1e-12)
        )

        if it > n_burn and (it - n_burn) % store_every == 0:
            alpha_draws[store_idx] = alpha
            kappa_draws[store_idx] = kappa_eff
            phi_draws[store_idx] = phi_1
            lambda_draws[store_idx] = lambda_ez
            sigma_v2_draws[store_idx] = sigma_v2
            sigma_e2_draws[store_idx] = sigma_e2
            sigma_zeta2_draws[store_idx] = sigma_zeta2
            rho_draws[store_idx] = rho_corr
            if ma_order:
                psi_draws[store_idx] = psi
                e_acf_draws[store_idx] = gamma[1:] / sigma_e2
            store_idx += 1

        if progress_callback is not None:
            progress_callback(it, total_iter)

        if verbose and it % 5000 == 0:
            acc = proposal.acceptance_rate if proposal is not None else float("nan")
            print(
                f"Iter {it}/{total_iter}: alpha={alpha:.3f}, kappa={kappa:.3f}, "
                f"psi={np.round(psi, 3)}, accept={acc:.2f}"
            )

    out: dict[str, Any] = {
        "alpha": _summary(alpha_draws),
        "kappa": _summary(kappa_draws),
        "phi_1": _summary(phi_draws),
        "lambda_ez": _summary(lambda_draws),
        "sigma_v2": _summary(sigma_v2_draws),
        "sigma_e2": _summary(sigma_e2_draws),
        "sigma_zeta2": _summary(sigma_zeta2_draws),
        "rho": _summary(rho_draws),
        "priors": priors,
        "opts": opts,
        "model": {
            "kappa_scale": KAPPA_SCALE,
            "kappa_internal": "stored kappa * KAPPA_SCALE",
            "stored_units": "physical",
            "coefficient_constraints": coefficient_constraints,
            "coefficient_constraint_stats": constraint_stats_summary(constraint_stats),
        },
        "error_structure": {
            "family": "ma" if ma_order else "iid",
            "order": ma_order,
            "block_order": [
                "alpha_kappa", "lambda_ez", "phi_1", "sigma_zeta2", "sigma_v2", "psi",
            ],
        },
    }
    if ma_order:
        out["psi"] = _summary(psi_draws)
        out["e_acf"] = _summary(e_acf_draws)
        out["error_structure"]["psi_acceptance_rate"] = proposal.acceptance_rate
        out["error_structure"]["psi_proposal_scale"] = float(np.exp(proposal.log_scale))
    return out
