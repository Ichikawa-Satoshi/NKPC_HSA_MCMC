from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import arviz as az
import numpy as np
import pandas as pd

from nkpc_hsa.config import configured_data_specs
from nkpc_hsa.data.transforms import DEFAULT_N_TRANSFORM, transform_competition_series
from nkpc_hsa.inference.period_robustness import apply_period


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    x = np.asarray(a, dtype=float).reshape(-1)
    y = np.asarray(b, dtype=float).reshape(-1)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return float("nan")
    x = x[mask]
    y = y[mask]
    if np.nanstd(x) <= 0.0 or np.nanstd(y) <= 0.0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _flat_posterior(idata: Any, name: str) -> np.ndarray | None:
    if not hasattr(idata, "posterior") or name not in idata.posterior:
        return None
    return np.asarray(idata.posterior[name], dtype=float).reshape(-1)


def _state_draws(idata: Any, name: str) -> np.ndarray | None:
    if not hasattr(idata, "posterior") or name not in idata.posterior:
        return None
    arr = np.asarray(idata.posterior[name], dtype=float)
    if arr.ndim < 3:
        return None
    return arr.reshape((-1, arr.shape[-1]))


def _state_mean(idata: Any, name: str) -> np.ndarray | None:
    arr = _state_draws(idata, name)
    if arr is None:
        return None
    return np.nanmean(arr, axis=0)


def _latest_run_key(idata: Any, run_name: str) -> tuple[str, str]:
    attrs = getattr(idata, "attrs", {})
    return str(attrs.get("run_id", "")), run_name


def load_posterior_runs(
    runs_dir: str | Path,
    *,
    data_specs: Sequence[str] | None = None,
    models: Sequence[str] | None = None,
    prior_spec: str | None = "baseline",
    constraint_spec: str | None = "unrestricted",
    n_transform: str | None = DEFAULT_N_TRANSFORM,
    latest_only: bool = True,
) -> dict[str, Any]:
    """Load posterior runs, optionally keeping the latest run per specification.

    The grouping key includes model, data specification, prior, constraint, N
    transform, and sample period so baseline and robustness runs are not mixed.
    """
    data_set = set(data_specs or [])
    model_set = set(models or [])
    selected: dict[tuple[str, ...], tuple[str, Any]] = {}
    loaded: dict[str, Any] = {}

    for posterior_path in sorted(Path(runs_dir).glob("*/posterior.nc")):
        idata = az.from_netcdf(posterior_path)
        attrs = getattr(idata, "attrs", {})
        run_name = posterior_path.parent.name
        if data_set and str(attrs.get("data_spec", "")) not in data_set:
            continue
        if model_set and str(attrs.get("model", "")) not in model_set:
            continue
        if prior_spec is not None and str(attrs.get("prior_spec", "")) != prior_spec:
            continue
        if constraint_spec is not None and str(attrs.get("constraint_spec", "unrestricted")) != constraint_spec:
            continue
        if n_transform is not None and str(attrs.get("n_transform", "")) != n_transform:
            continue

        priors_path = posterior_path.parent / "priors.json"
        if priors_path.exists():
            idata.attrs["run_priors"] = json.loads(priors_path.read_text(encoding="utf-8"))

        if not latest_only:
            loaded[run_name] = idata
            continue

        key = (
            str(attrs.get("model", "")),
            str(attrs.get("data_spec", "")),
            str(attrs.get("prior_spec", "")),
            str(attrs.get("constraint_spec", "unrestricted")),
            str(attrs.get("n_transform", "")),
            str(attrs.get("period", "full")),
            str(attrs.get("sample_start", "")),
            str(attrs.get("sample_end", "")),
        )
        current = selected.get(key)
        if current is None or _latest_run_key(idata, run_name) >= _latest_run_key(current[1], current[0]):
            selected[key] = (run_name, idata)

    if latest_only:
        loaded = {run: idata for run, idata in sorted(selected.values())}
    return loaded


def model_sample(
    data: pd.DataFrame,
    data_spec: Mapping[str, Any],
    *,
    n_transform: str = DEFAULT_N_TRANSFORM,
) -> pd.DataFrame:
    """Return the model-ready sample and canonical variable names."""
    spec = dict(data_spec)
    df = data.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        if "DATE" in df.columns:
            df["DATE"] = pd.to_datetime(df["DATE"])
            df = df.set_index("DATE")
        else:
            raise ValueError("data must have a DatetimeIndex or DATE column.")

    cols = {
        "pi": spec.get("pi_col", "pi"),
        "pi_prev": spec.get("pi_prev_col", "pi_prev"),
        "pi_expect": spec.get("pi_expect_col", "pi_expect"),
        "x": spec.get("x_col", "x"),
        "x_prev": spec.get("x_prev_col", "x_prev"),
        "N_raw": spec.get("n_col", spec.get("N_col", "N")),
    }
    required = list(cols.values())
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise KeyError(f"Missing model-ready columns for {spec.get('name', '')}: {missing}")

    sample = df[required].dropna().rename(columns={v: k for k, v in cols.items()})
    sample["y"] = sample["pi"] - sample["pi_expect"]
    sample["a"] = sample["pi_prev"] - sample["pi_expect"]
    sample["N_model"] = transform_competition_series(sample["N_raw"].to_numpy(dtype=float), transform=n_transform)
    return sample


def data_scale_table(
    data: pd.DataFrame,
    data_specs: Mapping[str, Mapping[str, Any]],
    *,
    n_transform: str = DEFAULT_N_TRANSFORM,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for name, spec in data_specs.items():
        sample = model_sample(data, spec, n_transform=n_transform)
        rows.append(
            {
                "data_spec": name,
                "label": spec.get("label", name),
                "n_obs": int(len(sample)),
                "start": sample.index.min().date().isoformat() if len(sample) else "",
                "end": sample.index.max().date().isoformat() if len(sample) else "",
                "x_mean": float(sample["x"].mean()),
                "x_sd": float(sample["x"].std(ddof=1)),
                "x_min": float(sample["x"].min()),
                "x_max": float(sample["x"].max()),
                "corr_x_y": _safe_corr(sample["x"].to_numpy(), sample["y"].to_numpy()),
                "corr_x_a": _safe_corr(sample["x"].to_numpy(), sample["a"].to_numpy()),
                "corr_x_N_model": _safe_corr(sample["x"].to_numpy(), sample["N_model"].to_numpy()),
                "N_model_sd": float(sample["N_model"].std(ddof=1)),
                "N_model_start": float(sample["N_model"].iloc[0]),
                "N_model_end": float(sample["N_model"].iloc[-1]),
            }
        )
    return pd.DataFrame(rows)


def period_ols_table(
    data: pd.DataFrame,
    data_specs: Mapping[str, Mapping[str, Any]],
    periods: Mapping[str, Mapping[str, Any]],
    *,
    n_transform: str = DEFAULT_N_TRANSFORM,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for name, spec in data_specs.items():
        for period_name, period_spec in periods.items():
            subset = apply_period(data, period_spec)
            sample = model_sample(subset, spec, n_transform=n_transform)
            if len(sample) < 5:
                rows.append({"data_spec": name, "period": period_name, "n_obs": int(len(sample)), "status": "too_few_obs"})
                continue
            X = np.column_stack([sample["a"].to_numpy(dtype=float), sample["x"].to_numpy(dtype=float)])
            beta, *_ = np.linalg.lstsq(X, sample["y"].to_numpy(dtype=float), rcond=None)
            resid = sample["y"].to_numpy(dtype=float) - X @ beta
            rows.append(
                {
                    "data_spec": name,
                    "period": period_name,
                    "n_obs": int(len(sample)),
                    "status": "ok",
                    "alpha_ols": float(beta[0]),
                    "kappa_ols": float(beta[1]),
                    "corr_x_y": _safe_corr(sample["x"].to_numpy(), sample["y"].to_numpy()),
                    "corr_x_N_model": _safe_corr(sample["x"].to_numpy(), sample["N_model"].to_numpy()),
                    "x_mean": float(sample["x"].mean()),
                    "x_sd": float(sample["x"].std(ddof=1)),
                    "rmse": float(np.sqrt(np.mean(resid**2))),
                }
            )
    return pd.DataFrame(rows)


def _design_stats(sample: pd.DataFrame, nbar: np.ndarray, nhat: np.ndarray | None, model: str) -> dict[str, float]:
    n = min(len(sample), len(nbar))
    y = sample["y"].to_numpy(dtype=float)[:n]
    a = sample["a"].to_numpy(dtype=float)[:n]
    x = sample["x"].to_numpy(dtype=float)[:n]
    nbar = np.asarray(nbar, dtype=float)[:n]

    cols = [a, x, x * nbar]
    names = ["alpha_ols_latent", "kappa0_ols_latent", "delta_ols_latent"]
    if model == "hsa_full" and nhat is not None:
        nhat = np.asarray(nhat, dtype=float)[:n]
        cols.extend([-nhat, -(nhat * nbar)])
        names.extend(["theta0_ols_latent", "gamma_ols_latent"])

    X = np.column_stack(cols)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    standardized = X.copy()
    std = np.nanstd(standardized, axis=0)
    keep = std > 0.0
    standardized[:, keep] = (standardized[:, keep] - np.nanmean(standardized[:, keep], axis=0)) / std[keep]
    cond = float(np.linalg.cond(standardized[:, keep])) if keep.sum() > 0 else float("nan")
    out = {
        "corr_x_xNbar": _safe_corr(x, x * nbar),
        "sd_xNbar": float(np.nanstd(x * nbar, ddof=1)),
        "design_condition_number": cond,
        "latent_ols_rmse": float(np.sqrt(np.mean(resid**2))),
    }
    out.update({name: float(value) for name, value in zip(names, beta)})
    return out


def identification_table(
    runs: Mapping[str, Any],
    data_by_spec: Mapping[str, pd.DataFrame],
) -> pd.DataFrame:
    """Summarize whether the HSA data design identifies kappa/delta paths."""
    rows: list[dict[str, Any]] = []
    for run_name, idata in runs.items():
        attrs = getattr(idata, "attrs", {})
        model = str(attrs.get("model", ""))
        data_spec = str(attrs.get("data_spec", ""))
        if model not in {"hsa_steady", "hsa_full"} or data_spec not in data_by_spec:
            continue

        row: dict[str, Any] = {
            "run": run_name,
            "model": model,
            "data_spec": data_spec,
            "prior_spec": str(attrs.get("prior_spec", "")),
            "constraint_spec": str(attrs.get("constraint_spec", "unrestricted")),
            "sample_start": str(attrs.get("sample_start", "")),
            "sample_end": str(attrs.get("sample_end", "")),
        }

        for name, prefix in [("kappa_0", "kappa0"), ("delta", "delta"), ("theta_0", "theta0"), ("gamma", "gamma")]:
            arr = _flat_posterior(idata, name)
            if arr is not None:
                row[f"{prefix}_mean"] = float(np.nanmean(arr))
                row[f"p_{prefix}_gt0"] = float(np.nanmean(arr > 0.0))

        kappa0 = _flat_posterior(idata, "kappa_0")
        delta = _flat_posterior(idata, "delta")
        if kappa0 is not None and delta is not None:
            row["corr_kappa0_delta"] = _safe_corr(kappa0, delta)

        for state in ["kappa_t", "theta_t", "Nbar", "Nhat"]:
            draws = _state_draws(idata, state)
            if draws is None:
                continue
            row[f"{state}_overall_mean"] = float(np.nanmean(draws))
            row[f"{state}_start_mean"] = float(np.nanmean(draws[:, 0]))
            row[f"{state}_end_mean"] = float(np.nanmean(draws[:, -1]))
            row[f"{state}_sd_over_time"] = float(np.nanstd(np.nanmean(draws, axis=0), ddof=1))
            if state == "kappa_t":
                row["p_all_kappa_t_gt0"] = float(np.nanmean(np.all(draws > 0.0, axis=1)))

        nbar_mean = _state_mean(idata, "Nbar")
        nhat_mean = _state_mean(idata, "Nhat")
        sample = data_by_spec[data_spec]
        if nbar_mean is not None:
            row.update(_design_stats(sample, nbar_mean, nhat_mean, model))
            if nhat_mean is not None:
                n = min(len(sample), len(nbar_mean), len(nhat_mean))
                reconstructed = nbar_mean[:n] + nhat_mean[:n]
                row["N_decomposition_rmse"] = float(
                    np.sqrt(np.mean((sample["N_model"].to_numpy(dtype=float)[:n] - reconstructed) ** 2))
                )
        rows.append(row)
    return pd.DataFrame(rows)


def key_diagnostics_table(
    runs: Mapping[str, Any],
    *,
    parameters: Sequence[str] = ("alpha", "kappa", "kappa_0", "delta", "theta", "theta_0", "gamma", "rho_1", "rho_2"),
) -> pd.DataFrame:
    from nkpc_hsa.inference.diagnostics import compute_diagnostics

    rows: list[dict[str, Any]] = []
    for run_name, idata in runs.items():
        attrs = getattr(idata, "attrs", {})
        table = compute_diagnostics(idata, var_names=parameters)
        for _, row in table.iterrows():
            rows.append(
                {
                    "run": run_name,
                    "model": str(attrs.get("model", "")),
                    "data_spec": str(attrs.get("data_spec", "")),
                    "prior_spec": str(attrs.get("prior_spec", "")),
                    "constraint_spec": str(attrs.get("constraint_spec", "unrestricted")),
                    "parameter": row.get("parameter", ""),
                    "mean": row.get("mean", np.nan),
                    "r_hat": row.get("r_hat", np.nan),
                    "ess_bulk": row.get("ess_bulk", np.nan),
                    "ess_tail": row.get("ess_tail", np.nan),
                    "mcse_mean": row.get("mcse_mean", np.nan),
                    "warning": row.get("warning", ""),
                }
            )
    return pd.DataFrame(rows)


def write_identification_outputs(
    *,
    out_dir: str | Path,
    data: pd.DataFrame,
    config: Mapping[str, Any],
    runs: Mapping[str, Any],
    periods: Mapping[str, Mapping[str, Any]] | None = None,
    requested_data_specs: Sequence[str] | None = None,
) -> dict[str, pd.DataFrame]:
    defaults = dict(config.get("defaults", {}) or {})
    n_transform = str(defaults.get("n_transform", DEFAULT_N_TRANSFORM))
    data_specs = configured_data_specs(config, list(requested_data_specs) if requested_data_specs else None)
    data_by_spec = {name: model_sample(data, spec, n_transform=n_transform) for name, spec in data_specs.items()}
    outputs = {
        "data_scale": data_scale_table(data, data_specs, n_transform=n_transform),
        "period_ols": period_ols_table(data, data_specs, periods or {"full": {}}, n_transform=n_transform),
        "identification": identification_table(runs, data_by_spec),
        "key_diagnostics": key_diagnostics_table(runs),
    }

    target = Path(out_dir)
    target.mkdir(parents=True, exist_ok=True)
    for name, table in outputs.items():
        table.to_csv(target / f"{name}.csv", index=False)
    return outputs
