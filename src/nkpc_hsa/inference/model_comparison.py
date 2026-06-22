from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd


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


def _flat_mean(posterior, name: str) -> float | None:
    if posterior is None or name not in posterior:
        return None
    values = np.asarray(posterior[name], dtype=float).reshape(-1)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return None
    return float(np.mean(values))


def _mean_state(posterior, name: str) -> np.ndarray | None:
    if posterior is None or name not in posterior:
        return None
    arr = np.asarray(posterior[name], dtype=float)
    if arr.ndim < 3:
        return None
    return np.nanmean(arr.reshape(-1, arr.shape[-1]), axis=0)


def posterior_predictive_score(idata, data: Mapping[str, np.ndarray], model_name: str) -> tuple[float, float] | tuple[None, None]:
    posterior = getattr(idata, "posterior", None)
    if posterior is None:
        return None, None
    required = ["pi", "pi_prev", "pi_expect", "x"]
    if any(k not in data for k in required):
        return None, None
    pi = np.asarray(data["pi"], dtype=float)
    pi_prev = np.asarray(data["pi_prev"], dtype=float)
    pi_expect = np.asarray(data["pi_expect"], dtype=float)
    x = np.asarray(data["x"], dtype=float)
    n = min(pi.size, pi_prev.size, pi_expect.size, x.size)
    pi, pi_prev, pi_expect, x = pi[:n], pi_prev[:n], pi_expect[:n], x[:n]
    alpha = _flat_mean(posterior, "alpha")
    sigma_e = _flat_mean(posterior, "sigma_e")
    if alpha is None:
        return None, None
    pred = alpha * pi_prev + (1.0 - alpha) * pi_expect
    if "hsa_full" in model_name or "full" in model_name:
        kappa_t = _mean_state(posterior, "kappa_t")
        theta_t = _mean_state(posterior, "theta_t")
        Nhat = _mean_state(posterior, "Nhat")
        if kappa_t is None or theta_t is None or Nhat is None:
            return None, None
        n = min(n, kappa_t.size, theta_t.size, Nhat.size)
        pi, pred, x = pi[:n], pred[:n], x[:n]
        pred = pred + kappa_t[:n] * x - theta_t[:n] * Nhat[:n]
    elif "hsa_steady" in model_name or "steady" in model_name:
        kappa_t = _mean_state(posterior, "kappa_t")
        if kappa_t is None:
            kappa_0 = _flat_mean(posterior, "kappa_0")
            delta = _flat_mean(posterior, "delta")
            Nbar = _mean_state(posterior, "Nbar")
            if kappa_0 is None or delta is None or Nbar is None:
                return None, None
            kappa_t = kappa_0 + delta * Nbar
        n = min(n, kappa_t.size)
        pi, pred, x = pi[:n], pred[:n], x[:n]
        pred = pred + kappa_t[:n] * x
    elif "hsa_dynamic" in model_name or "dynamic" in model_name:
        kappa = _flat_mean(posterior, "kappa")
        theta = _flat_mean(posterior, "theta")
        Nhat = _mean_state(posterior, "Nhat")
        if kappa is None or theta is None or Nhat is None:
            return None, None
        n = min(n, Nhat.size)
        pi, pred, x = pi[:n], pred[:n], x[:n]
        pred = pred + kappa * x - theta * Nhat[:n]
    else:
        kappa = _flat_mean(posterior, "kappa")
        if kappa is None:
            return None, None
        pred = pred + kappa * x
    resid = pi - pred[: pi.size]
    if sigma_e is None or not np.isfinite(sigma_e) or sigma_e <= 0.0:
        sigma_e = float(np.std(resid, ddof=1)) if resid.size > 1 else 1.0
    sigma2 = max(float(sigma_e) ** 2, 1e-10)
    log_score = float(-0.5 * resid.size * np.log(2.0 * np.pi * sigma2) - 0.5 * np.sum(resid**2) / sigma2)
    rmse = float(np.sqrt(np.mean(resid**2)))
    return log_score, rmse


def model_comparison_table(results: Mapping[str, object], *, data_by_model: Mapping[str, dict] | None = None) -> pd.DataFrame:
    rows = []
    data_by_model = data_by_model or {}
    for name, idata in results.items():
        row = {
            "model": name,
            "data_spec": getattr(idata, "attrs", {}).get("data_spec", ""),
            "prior_spec": getattr(idata, "attrs", {}).get("prior_spec", ""),
            "constraint_spec": getattr(idata, "attrs", {}).get("constraint_spec", "unrestricted"),
            "log_marginal_likelihood": np.nan,
            "bayes_factor_vs_baseline": np.nan,
            "sddr_delta_bf01": np.nan,
            "sddr_theta_bf01": np.nan,
            "sddr_theta0_bf01": np.nan,
            "sddr_gamma_bf01": np.nan,
            "predictive_score": np.nan,
            "posterior_predictive_rmse": np.nan,
            "notes": "Chib uses physical-unit posterior draws and physical-unit priors.",
        }
        posterior = getattr(idata, "posterior", None)
        if posterior is not None and "delta" in posterior:
            bf01 = sddr_bf01_normal(posterior["delta"].values, point=0.0, prior_mean=0.1, prior_sd=0.2)
            row["sddr_delta_bf01"] = np.nan if bf01 is None else bf01
        if posterior is not None and "theta" in posterior:
            bf01 = sddr_bf01_normal(posterior["theta"].values, point=0.0, prior_mean=0.1, prior_sd=0.2)
            row["sddr_theta_bf01"] = np.nan if bf01 is None else bf01
        if posterior is not None and "theta_0" in posterior:
            bf01 = sddr_bf01_normal(posterior["theta_0"].values, point=0.0, prior_mean=0.1, prior_sd=0.2)
            row["sddr_theta0_bf01"] = np.nan if bf01 is None else bf01
        if posterior is not None and "gamma" in posterior:
            bf01 = sddr_bf01_normal(posterior["gamma"].values, point=0.0, prior_mean=0.1, prior_sd=0.2)
            row["sddr_gamma_bf01"] = np.nan if bf01 is None else bf01
        data = data_by_model.get(name, data_by_model.get("__default__"))
        if data is not None:
            score, rmse = posterior_predictive_score(idata, data, name)
            row["predictive_score"] = np.nan if score is None else score
            row["posterior_predictive_rmse"] = np.nan if rmse is None else rmse
        try:
            family = "ces" if "ces" in name else "steady" if "steady" in name else "dynamic" if "dynamic" in name else None
            if family is None:
                row["notes"] += " Chib unavailable for HSA full in the current legacy implementation."
            elif data is not None:
                from analysis.gibbs.func_gibbs.gibbs_marginal_likelihood import chib_marginal_likelihood

                result = chib_marginal_likelihood(idata.posterior, data, family=family, orth=False)
                row.update(asdict(result))
                row["notes"] += " Chib uses baseline hard-coded prior ordinates in the legacy implementation."
            else:
                row["notes"] += " Chib not computed because comparison data were not supplied."
        except Exception as exc:
            row["notes"] += f" Chib unavailable: {exc}"
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
    table.to_latex(out / "model_comparison.tex", index=False, float_format="%.3f", escape=True)
