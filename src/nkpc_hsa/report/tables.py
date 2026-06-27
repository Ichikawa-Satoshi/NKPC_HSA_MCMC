from __future__ import annotations

from pathlib import Path
from typing import Iterable
from collections.abc import Mapping

import numpy as np
import pandas as pd


PARAMETER_UNITS = {
    "alpha": "share",
    "kappa": "inflation points per x point",
    "kappa_0": "inflation points per x point at average Nbar",
    "kappa_t": "inflation points per x point",
    "delta": "change in kappa_t per +10 log-point Nbar deviation",
    "theta": "inflation effect per +10 log-point Nhat deviation",
    "theta_0": "inflation effect per +10 log-point Nhat deviation at average Nbar",
    "gamma": "change in theta_t per +10 log-point Nbar deviation",
    "rho_1": "AR(2) coefficient",
    "rho_2": "AR(2) coefficient",
    "phi_1": "AR(1) coefficient",
    "n": "Nbar drift in ten-log-point units",
    "lambda_ez": "shock loading",
}


def parameter_unit(name: str) -> str:
    return PARAMETER_UNITS.get(name, "reported physical units")


def _summary_row(values: np.ndarray, *, parameter: str, extra: dict[str, object]) -> dict[str, object]:
    values = np.asarray(values, dtype=float).reshape(-1)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {}
    return {
        **extra,
        "parameter": parameter,
        "unit": parameter_unit(parameter),
        "posterior_mean": float(np.mean(values)),
        "posterior_sd": float(np.std(values, ddof=1)) if values.size > 1 else 0.0,
        "ci_2.5": float(np.quantile(values, 0.025)),
        "ci_97.5": float(np.quantile(values, 0.975)),
        "p_gt_0": float(np.mean(values > 0.0)),
        "p_lt_0": float(np.mean(values < 0.0)),
    }


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
                "unit": parameter_unit(name),
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
        data_spec = attrs.get("data_spec", "")
        prior_spec = attrs.get("prior_spec", "")
        n_transform = attrs.get("n_transform", "")
        sample_start = attrs.get("sample_start", "")
        sample_end = attrs.get("sample_end", "")
        for name in COEFFICIENT_PARAMETERS:
            if name not in posterior:
                continue
            values = np.asarray(posterior[name], dtype=float).reshape(-1)
            values = values[np.isfinite(values)]
            if values.size == 0:
                continue
            base = {
                "run": run,
                "model": model,
                "data_spec": data_spec,
                "prior_spec": prior_spec,
                "constraint_spec": constraint_spec,
                "n_transform": n_transform,
                "sample_start": sample_start,
                "sample_end": sample_end,
            }
            rows.append(
                _summary_row(values, parameter=name, extra=base)
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
        row = {
            "run": run,
            "model": model,
            "data_spec": attrs.get("data_spec", ""),
            "prior_spec": attrs.get("prior_spec", ""),
            "constraint_spec": attrs.get("constraint_spec", "unrestricted"),
            "n_transform": attrs.get("n_transform", ""),
        }
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
        if len(row) > 6:
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
        data_spec = attrs.get("data_spec", "")
        prior_spec = attrs.get("prior_spec", "")
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
                        "data_spec": data_spec,
                        "prior_spec": prior_spec,
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
        data_spec = attrs.get("data_spec", "")
        prior_spec = attrs.get("prior_spec", "")
        run_priors = attrs.get("run_priors") if isinstance(attrs.get("run_priors"), Mapping) else priors
        for var, prior_key in prior_names.items():
            if var not in posterior or prior_key not in run_priors:
                continue
            prior = run_priors[prior_key]
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
                    "data_spec": data_spec,
                    "prior_spec": prior_spec,
                    "constraint_spec": constraint_spec,
                    "restriction": f"{var}=0",
                    "unit": parameter_unit(var),
                    "prior_mean": mu,
                    "prior_sd": sd,
                    "sddr_bf01": np.nan if bf01 is None else float(bf01),
                }
            )
    return pd.DataFrame(rows)


_MODEL_ORDER = ["ces", "hsa_steady", "hsa_dynamic", "hsa_full"]
_PARAM_ORDER = ["alpha", "kappa", "kappa_0", "delta", "theta", "theta_0", "gamma", "rho_1", "rho_2", "phi_1", "lambda_ez", "n"]


def coefficient_means_pivot_table(idata_by_run: dict[str, object]) -> pd.DataFrame:
    """Models as columns, parameters as rows; cell = 'mean [ci_2.5, ci_97.5]'."""
    long = coefficient_means_table(idata_by_run)
    if long.empty or "parameter" not in long.columns or "model" not in long.columns:
        return pd.DataFrame()

    def _fmt(row: pd.Series) -> str:
        return f"{row['posterior_mean']:.3f} [{row['ci_2.5']:.3f}, {row['ci_97.5']:.3f}]"

    long = long.copy()
    long["cell"] = long.apply(_fmt, axis=1)
    pivot = long.pivot_table(index="parameter", columns="model", values="cell", aggfunc="first")
    models = [m for m in _MODEL_ORDER if m in pivot.columns]
    params = [p for p in _PARAM_ORDER if p in pivot.index]
    pivot = pivot.reindex(index=params, columns=models)
    pivot.index.name = "parameter"
    pivot.columns.name = None
    return pivot


def write_latex_fragment(df: pd.DataFrame, path: str | Path, *, index: bool = False) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(df.to_latex(index=index, float_format="%.4f", escape=True), encoding="utf-8")
