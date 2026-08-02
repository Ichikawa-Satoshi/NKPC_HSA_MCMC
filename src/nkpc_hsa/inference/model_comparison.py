from __future__ import annotations

from collections.abc import Mapping
import json
from pathlib import Path

import numpy as np
import pandas as pd

from nkpc_hsa.inference.wrappers import ESTIMATION_REVISION


def sddr_bf01_normal(draws: np.ndarray, *, point: float = 0.0, prior_mean: float = 0.0, prior_sd: float = 0.2) -> float | None:
    from scipy.stats import gaussian_kde, norm

    values = np.asarray(draws, dtype=float).reshape(-1)
    values = values[np.isfinite(values)]
    if values.size < 20 or np.std(values, ddof=1) <= 0.0:
        return None
    kde = gaussian_kde(values)
    posterior_at_point = float(kde([point])[0])
    prior_at_point = float(norm.pdf(point, loc=prior_mean, scale=prior_sd))
    return posterior_at_point / max(prior_at_point, 1e-300)


def _prior_mean_sd(priors: Mapping[str, object] | None, name: str, default: tuple[float, float]) -> tuple[float, float]:
    if priors and name in priors:
        prior = priors[name]
        if isinstance(prior, Mapping):
            return float(prior["mean"]), float(prior["sd"])
        if isinstance(prior, (list, tuple)) and len(prior) >= 2:
            return float(prior[0]), float(prior[1])
    return default


def _draw_scalar(posterior, name: str) -> np.ndarray | None:
    if name not in posterior:
        return None
    values = np.asarray(posterior[name], dtype=float).reshape(-1)
    return values if values.size and np.all(np.isfinite(values)) else None


def _draw_state(posterior, name: str) -> np.ndarray | None:
    if name not in posterior:
        return None
    values = np.asarray(posterior[name], dtype=float)
    if values.ndim < 3:
        return None
    return values.reshape(-1, values.shape[-1])


def _shock_loading_and_sd(posterior, draws: int) -> tuple[np.ndarray, np.ndarray] | None:
    """Return lambda in e=lambda*zeta+eta and the conditional sd of eta."""
    sigma_e = _draw_scalar(posterior, "sigma_e")
    sigma_zeta = _draw_scalar(posterior, "sigma_zeta")
    lambda_ez = _draw_scalar(posterior, "lambda_ez")

    if lambda_ez is None and "Sigma" in posterior:
        sigma = np.asarray(posterior["Sigma"], dtype=float)
        if sigma.ndim >= 4 and sigma.shape[-2:] == (4, 4):
            sigma = sigma.reshape(-1, 4, 4)
            lambda_ez = sigma[:, 0, 1] / sigma[:, 1, 1]
            sigma_eta = np.sqrt(np.maximum(sigma[:, 0, 0] - sigma[:, 0, 1] ** 2 / sigma[:, 1, 1], 1e-12))
            return lambda_ez[:draws], sigma_eta[:draws]

    if lambda_ez is None:
        rho = _draw_scalar(posterior, "corr_e_zeta")
        if rho is None:
            rho = _draw_scalar(posterior, "rho")
        if rho is not None and sigma_e is not None and sigma_zeta is not None:
            lambda_ez = rho * sigma_e / np.maximum(sigma_zeta, 1e-12)
    if lambda_ez is None:
        lambda_ez = np.zeros(draws, dtype=float)

    sigma_eta = _draw_scalar(posterior, "sigma_eta")
    if sigma_eta is None:
        if sigma_e is None or sigma_zeta is None:
            return None
        sigma_eta = np.sqrt(np.maximum(sigma_e**2 - lambda_ez**2 * sigma_zeta**2, 1e-12))
    return lambda_ez[:draws], sigma_eta[:draws]


def in_sample_conditional_lppd(
    idata,
    data: Mapping[str, np.ndarray],
    model_name: str,
) -> tuple[float, float] | tuple[None, None]:
    """In-sample lppd conditional on each draw's smoothed latent states.

    This is deliberately not called an out-of-sample predictive score.  It
    integrates the observation density over posterior draws and includes the
    cross-equation shock term lambda*zeta.
    """
    posterior = getattr(idata, "posterior", None)
    if posterior is None:
        return None, None
    required = ["pi", "pi_prev", "pi_expect", "x", "x_prev"]
    if any(k not in data for k in required):
        return None, None
    pi = np.asarray(data["pi"], dtype=float)
    pi_prev = np.asarray(data["pi_prev"], dtype=float)
    pi_expect = np.asarray(data["pi_expect"], dtype=float)
    x = np.asarray(data["x"], dtype=float)
    x_prev = np.asarray(data["x_prev"], dtype=float)
    n = min(pi.size, pi_prev.size, pi_expect.size, x.size, x_prev.size)
    pi, pi_prev, pi_expect, x, x_prev = pi[:n], pi_prev[:n], pi_expect[:n], x[:n], x_prev[:n]
    alpha = _draw_scalar(posterior, "alpha")
    if alpha is None:
        return None, None
    draws = alpha.size
    pred = alpha[:, None] * pi_prev[None, :] + (1.0 - alpha[:, None]) * pi_expect[None, :]
    family = model_name.lower()
    if "hsa_full" in family or "full" in family or "const_theta" in family:
        kappa_t = _draw_state(posterior, "kappa_t")
        theta_t = _draw_state(posterior, "theta_t")
        Nhat = _draw_state(posterior, "Nhat")
        Nbar = _draw_state(posterior, "Nbar")
        if kappa_t is None and Nbar is not None:
            kappa_0 = _draw_scalar(posterior, "kappa_0")
            delta = _draw_scalar(posterior, "delta")
            if kappa_0 is not None and delta is not None:
                kappa_t = kappa_0[:, None] + delta[:, None] * Nbar
        if theta_t is None and Nbar is not None:
            theta_0 = _draw_scalar(posterior, "theta_0")
            if theta_0 is None:
                theta_0 = _draw_scalar(posterior, "theta")
            gamma = _draw_scalar(posterior, "gamma")
            if theta_0 is not None:
                theta_t = theta_0[:, None] + (0.0 if gamma is None else gamma[:, None] * Nbar)
                if theta_t.shape[1] == 1:
                    theta_t = np.broadcast_to(theta_t, Nbar.shape)
        if kappa_t is None or theta_t is None or Nhat is None:
            return None, None
        n = min(n, kappa_t.shape[1], theta_t.shape[1], Nhat.shape[1])
        pred = pred[:, :n] + kappa_t[:draws, :n] * x[None, :n] - theta_t[:draws, :n] * Nhat[:draws, :n]
    elif "hsa_steady" in family or "steady" in family:
        kappa_t = _draw_state(posterior, "kappa_t")
        if kappa_t is None:
            kappa_0 = _draw_scalar(posterior, "kappa_0")
            delta = _draw_scalar(posterior, "delta")
            Nbar = _draw_state(posterior, "Nbar")
            if kappa_0 is None or delta is None or Nbar is None:
                return None, None
            kappa_t = kappa_0[:, None] + delta[:, None] * Nbar
        n = min(n, kappa_t.shape[1])
        pred = pred[:, :n] + kappa_t[:draws, :n] * x[None, :n]
    elif "hsa_dynamic" in family or "dynamic" in family:
        kappa = _draw_scalar(posterior, "kappa")
        theta = _draw_scalar(posterior, "theta")
        Nhat = _draw_state(posterior, "Nhat")
        if kappa is None or theta is None or Nhat is None:
            return None, None
        n = min(n, Nhat.shape[1])
        pred = pred[:, :n] + kappa[:, None] * x[None, :n] - theta[:, None] * Nhat[:draws, :n]
    else:
        kappa = _draw_scalar(posterior, "kappa")
        if kappa is None:
            return None, None
        pred = pred + kappa[:, None] * x[None, :]

    phi = _draw_scalar(posterior, "phi_1")
    shock = _shock_loading_and_sd(posterior, draws)
    if phi is None or shock is None:
        return None, None
    lambda_ez, sigma_eta = shock
    draws = min(draws, phi.size, lambda_ez.size, sigma_eta.size, pred.shape[0])
    pred = pred[:draws, :n]
    zeta = x[None, :n] - phi[:draws, None] * x_prev[None, :n]
    pred = pred + lambda_ez[:draws, None] * zeta
    resid = pi[None, :n] - pred
    sigma2 = np.maximum(sigma_eta[:draws, None] ** 2, 1e-12)
    log_density = -0.5 * (np.log(2.0 * np.pi * sigma2) + resid**2 / sigma2)
    from scipy.special import logsumexp

    lppd = float(np.sum(logsumexp(log_density, axis=0) - np.log(draws)))
    rmse = float(np.sqrt(np.mean((pi[:n] - np.mean(pred, axis=0)) ** 2)))
    return lppd, rmse


def posterior_predictive_score(idata, data: Mapping[str, np.ndarray], model_name: str) -> tuple[float, float] | tuple[None, None]:
    """Backward-compatible alias for :func:`in_sample_conditional_lppd`."""
    return in_sample_conditional_lppd(idata, data, model_name)


def model_comparison_table(results: Mapping[str, object], *, data_by_model: Mapping[str, dict] | None = None) -> pd.DataFrame:
    rows = []
    data_by_model = data_by_model or {}
    for name, idata in results.items():
        row = {
            "run": name,
            "model": getattr(idata, "attrs", {}).get("model", name),
            "data_spec": getattr(idata, "attrs", {}).get("data_spec", ""),
            "prior_spec": getattr(idata, "attrs", {}).get("prior_spec", ""),
            "constraint_spec": getattr(idata, "attrs", {}).get("constraint_spec", "unrestricted"),
            "n_transform": getattr(idata, "attrs", {}).get("n_transform", ""),
            "estimation_revision": getattr(idata, "attrs", {}).get("estimation_revision", ""),
            "log_marginal_likelihood": np.nan,
            "bayes_factor_vs_baseline": np.nan,
            "sddr_delta_bf01": np.nan,
            "sddr_theta_bf01": np.nan,
            "sddr_theta0_bf01": np.nan,
            "sddr_gamma_bf01": np.nan,
            "in_sample_conditional_lppd": np.nan,
            "in_sample_posterior_mean_rmse": np.nan,
            "notes": (
                "SDDR uses run-specific saved priors and physical-unit posterior draws. "
                "The lppd is in-sample and conditional on smoothed latent-state draws; it is not an out-of-sample score."
            ),
        }
        posterior = getattr(idata, "posterior", None)
        attrs = getattr(idata, "attrs", {})
        current_revision = str(row["estimation_revision"]) == ESTIMATION_REVISION
        constrained = str(row["constraint_spec"]) not in {"", "unrestricted", "None"}
        if isinstance(attrs.get("run_priors"), Mapping):
            run_priors = attrs["run_priors"]
        else:
            try:
                parsed_priors = json.loads(str(attrs.get("run_priors_json", "{}")))
                run_priors = parsed_priors if isinstance(parsed_priors, Mapping) else {}
            except (TypeError, ValueError, json.JSONDecodeError):
                run_priors = {}
        sddr_specs = {
            "delta": ("sddr_delta_bf01", (0.0, 0.02)),
            "theta": ("sddr_theta_bf01", (0.1, 0.2)),
            "theta_0": ("sddr_theta0_bf01", (0.1, 0.2)),
            "gamma": ("sddr_gamma_bf01", (0.0, 0.02)),
        }
        if posterior is not None and not constrained and current_revision:
            for var, (column, default_prior) in sddr_specs.items():
                if var not in posterior:
                    continue
                mu, sd = _prior_mean_sd(run_priors, var, default_prior)
                bf01 = sddr_bf01_normal(posterior[var].values, point=0.0, prior_mean=mu, prior_sd=sd)
                row[column] = np.nan if bf01 is None else bf01
        elif constrained and current_revision:
            row["notes"] += (
                " SDDR is withheld for constrained runs because the untruncated normal prior density at zero is not valid."
            )
        if not current_revision:
            row["notes"] += (
                f" Stale estimation revision: expected {ESTIMATION_REVISION}; scores and evidence are withheld until re-estimation."
            )
        data = data_by_model.get(name, data_by_model.get("__default__"))
        if data is not None and current_revision:
            score, rmse = in_sample_conditional_lppd(idata, data, str(row["model"]))
            row["in_sample_conditional_lppd"] = np.nan if score is None else score
            row["in_sample_posterior_mean_rmse"] = np.nan if rmse is None else rmse
        row["notes"] += (
            " Automated Chib marginal likelihood is not reported: the existing implementation does not "
            "integrate latent states and does not normalize stationarity/coefficient truncations consistently."
        )
        rows.append(row)
    table = pd.DataFrame(rows)
    if "log_marginal_likelihood" in table and table["log_marginal_likelihood"].notna().any():
        baseline = float(table["log_marginal_likelihood"].dropna().iloc[0])
        table["bayes_factor_vs_baseline"] = np.exp(table["log_marginal_likelihood"] - baseline)
    return table


def save_model_comparison(table: pd.DataFrame, out_dir: str | Path) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if table.empty:
        table = pd.DataFrame({"note": ["No model-comparison runs available."]})
    table.to_csv(out / "model_comparison.csv", index=False)
    display_columns = [
        "model",
        "data_spec",
        "prior_spec",
        "in_sample_conditional_lppd",
        "in_sample_posterior_mean_rmse",
        "log_likelihood",
        "log_marginal_likelihood",
        "bayes_factor_vs_baseline",
        "method",
        "sddr_delta_bf01",
        "sddr_theta_bf01",
        "sddr_gamma_bf01",
    ]
    display = table[[col for col in display_columns if col in table.columns]]
    if display.empty:
        display = table
    display.to_latex(out / "model_comparison.tex", index=False, float_format="%.3f", escape=True)
