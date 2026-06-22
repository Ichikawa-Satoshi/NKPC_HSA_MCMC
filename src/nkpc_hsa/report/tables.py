from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


def posterior_summary_table(idata, *, var_names: Iterable[str] | None = None) -> pd.DataFrame:
    posterior = getattr(idata, "posterior", None)
    if posterior is None:
        return pd.DataFrame()
    names = list(var_names or posterior.data_vars)
    rows = []
    for name in names:
        if name not in posterior:
            continue
        values = np.asarray(posterior[name]).reshape(-1)
        values = values[np.isfinite(values)]
        if values.size == 0:
            continue
        rows.append(
            {
                "parameter": name,
                "mean": float(np.mean(values)),
                "sd": float(np.std(values, ddof=1)) if values.size > 1 else 0.0,
                "ci_2.5": float(np.quantile(values, 0.025)),
                "ci_97.5": float(np.quantile(values, 0.975)),
            }
        )
    return pd.DataFrame(rows)


COEFFICIENT_PARAMETERS = (
    "alpha",
    "kappa",
    "kappa_0",
    "delta",
    "theta",
    "theta_0",
    "gamma",
    "phi_1",
    "rho_1",
    "rho_2",
    "n",
    "lambda_ez",
)


def coefficient_means_table(idata_by_run: dict[str, object]) -> pd.DataFrame:
    rows = []
    for run, idata in idata_by_run.items():
        posterior = getattr(idata, "posterior", None)
        if posterior is None:
            continue
        attrs = getattr(idata, "attrs", {})
        model = attrs.get("model", "")
        constraint_spec = attrs.get("constraint_spec", "unrestricted")
        for name in COEFFICIENT_PARAMETERS:
            if name not in posterior:
                continue
            values = np.asarray(posterior[name], dtype=float).reshape(-1)
            values = values[np.isfinite(values)]
            if values.size == 0:
                continue
            rows.append(
                {
                    "run": run,
                    "model": model,
                    "constraint_spec": constraint_spec,
                    "parameter": name,
                    "posterior_mean": float(np.mean(values)),
                    "posterior_sd": float(np.std(values, ddof=1)) if values.size > 1 else 0.0,
                    "ci_2.5": float(np.quantile(values, 0.025)),
                    "ci_97.5": float(np.quantile(values, 0.975)),
                }
            )
    return pd.DataFrame(rows)


def kappa_comparison_table(idata_by_run: dict[str, object]) -> pd.DataFrame:
    rows = []
    for run, idata in idata_by_run.items():
        posterior = getattr(idata, "posterior", None)
        if posterior is None:
            continue
        attrs = getattr(idata, "attrs", {})
        model = attrs.get("model", "")
        row = {"run": run, "model": model, "constraint_spec": attrs.get("constraint_spec", "unrestricted")}
        for name in ["kappa", "kappa_0", "delta"]:
            if name in posterior:
                values = np.asarray(posterior[name], dtype=float).reshape(-1)
                values = values[np.isfinite(values)]
                if values.size:
                    row[f"{name}_mean"] = float(np.mean(values))
                    row[f"{name}_ci_2.5"] = float(np.quantile(values, 0.025))
                    row[f"{name}_ci_97.5"] = float(np.quantile(values, 0.975))
        if "kappa_t" in posterior:
            arr = np.asarray(posterior["kappa_t"], dtype=float)
            flat = arr.reshape(-1)
            flat = flat[np.isfinite(flat)]
            if flat.size:
                row["kappa_t_overall_mean"] = float(np.mean(flat))
                if arr.ndim >= 3:
                    path = arr.reshape(-1, arr.shape[-1])
                    row["kappa_t_start_mean"] = float(np.nanmean(path[:, 0]))
                    row["kappa_t_end_mean"] = float(np.nanmean(path[:, -1]))
        if len(row) > 3:
            rows.append(row)
    return pd.DataFrame(rows)


def time_varying_coefficients_table(idata_by_run: dict[str, object], *, max_rows: int = 24) -> pd.DataFrame:
    rows = []
    for run, idata in idata_by_run.items():
        posterior = getattr(idata, "posterior", None)
        if posterior is None:
            continue
        attrs = getattr(idata, "attrs", {})
        model = attrs.get("model", "")
        constraint_spec = attrs.get("constraint_spec", "unrestricted")
        for name in ["kappa_t", "theta_t"]:
            if name not in posterior:
                continue
            arr = np.asarray(posterior[name], dtype=float)
            if arr.ndim < 3:
                continue
            path = arr.reshape(-1, arr.shape[-1])
            idx = np.linspace(0, path.shape[1] - 1, min(max_rows, path.shape[1])).round().astype(int)
            for t in idx:
                vals = path[:, t]
                vals = vals[np.isfinite(vals)]
                if vals.size == 0:
                    continue
                rows.append(
                    {
                        "run": run,
                        "model": model,
                        "constraint_spec": constraint_spec,
                        "coefficient": name,
                        "time_index": int(t),
                        "posterior_mean": float(np.mean(vals)),
                        "ci_2.5": float(np.quantile(vals, 0.025)),
                        "ci_97.5": float(np.quantile(vals, 0.975)),
                    }
                )
    return pd.DataFrame(rows)


def sddr_summary_table(idata_by_run: dict[str, object], priors: dict[str, object]) -> pd.DataFrame:
    from nkpc_hsa.inference.model_comparison import sddr_bf01_normal

    rows = []
    prior_names = {
        "kappa": "kappa",
        "kappa_0": "kappa_0",
        "delta": "delta",
        "theta": "theta",
        "theta_0": "theta_0",
        "gamma": "gamma",
    }
    for run, idata in idata_by_run.items():
        posterior = getattr(idata, "posterior", None)
        if posterior is None:
            continue
        attrs = getattr(idata, "attrs", {})
        model = attrs.get("model", "")
        constraint_spec = attrs.get("constraint_spec", "unrestricted")
        for var, prior_key in prior_names.items():
            if var not in posterior or prior_key not in priors:
                continue
            prior = priors[prior_key]
            if isinstance(prior, dict):
                mu, sd = float(prior["mean"]), float(prior["sd"])
            else:
                mu, sd = float(prior[0]), float(prior[1])
            values = np.asarray(posterior[var], dtype=float).reshape(-1)
            bf01 = sddr_bf01_normal(values, point=0.0, prior_mean=mu, prior_sd=sd)
            rows.append(
                {
                    "run": run,
                    "model": model,
                    "constraint_spec": constraint_spec,
                    "restriction": f"{var}=0",
                    "prior_mean": mu,
                    "prior_sd": sd,
                    "sddr_bf01": np.nan if bf01 is None else float(bf01),
                }
            )
    return pd.DataFrame(rows)


def write_latex_fragment(df: pd.DataFrame, path: str | Path, *, index: bool = False) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(df.to_latex(index=index, float_format="%.4f", escape=True), encoding="utf-8")
