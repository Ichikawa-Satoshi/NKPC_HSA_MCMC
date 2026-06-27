from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

KEY_PARAMETERS = (
    "alpha",
    "kappa",
    "kappa_0",
    "delta",
    "theta",
    "theta_0",
    "gamma",
    "kappa_t",
    "theta_t",
    "phi_1",
    "rho_1",
    "rho_2",
    "n",
    "lambda_ez",
    "rho",
    "sigma_eta",
    "sigma_e",
    "sigma_zeta",
    "sigma_u",
    "sigma_eps",
    "sigma_N",
)


def available_key_parameters(idata, params: Iterable[str] = KEY_PARAMETERS) -> list[str]:
    posterior = getattr(idata, "posterior", None)
    if posterior is None:
        return []
    return [p for p in params if p in posterior]


def _scalarized_diagnostic_idata(idata, params: Iterable[str]):
    """Return an InferenceData with scalar variables suitable for diagnostics plots.

    ArviZ expands every extra dimension in trace/autocorrelation plots. State paths
    such as kappa_t and theta_t can therefore request hundreds of subplots. For
    diagnostics we keep scalar parameters unchanged and summarize each path by its
    mean, start, middle, and end draws.
    """
    import arviz as az

    posterior = getattr(idata, "posterior", None)
    if posterior is None:
        return idata, []
    scalar_draws: dict[str, np.ndarray] = {}
    for name in params:
        if name not in posterior:
            continue
        values = np.asarray(posterior[name], dtype=float)
        if values.ndim == 2:
            scalar_draws[name] = values
            continue
        if values.ndim < 2:
            continue
        path = values.reshape(values.shape[0], values.shape[1], -1)
        if path.shape[-1] == 0:
            continue
        mid = path.shape[-1] // 2
        scalar_draws[f"{name}_mean"] = np.nanmean(path, axis=2)
        scalar_draws[f"{name}_start"] = path[:, :, 0]
        scalar_draws[f"{name}_mid"] = path[:, :, mid]
        scalar_draws[f"{name}_end"] = path[:, :, -1]
    if not scalar_draws:
        return idata, []
    scalar_idata = az.from_dict({"posterior": scalar_draws})
    scalar_idata.attrs.update(getattr(idata, "attrs", {}))
    return scalar_idata, list(scalar_draws)


def compute_diagnostics(idata, *, var_names: Iterable[str] | None = None) -> pd.DataFrame:
    import arviz as az

    names = list(var_names or available_key_parameters(idata))
    idata_for_summary, names = _scalarized_diagnostic_idata(idata, names)
    if not names:
        return pd.DataFrame()
    try:
        summary = az.summary(
            idata_for_summary,
            var_names=names,
            stat_focus="mean",
            kind="all",
            hdi_prob=0.95,
        )
    except TypeError:
        try:
            summary = az.summary(idata_for_summary, var_names=names, kind="all", ci_prob=0.95)
        except Exception:
            summary = _manual_summary(idata_for_summary, names)
    except Exception:
        summary = _manual_summary(idata_for_summary, names)
    if "parameter" not in summary.columns:
        summary = summary.reset_index().rename(columns={"index": "parameter"})
    manual = _basic_chain_diagnostics(idata_for_summary, names)
    if not manual.empty:
        summary = summary.merge(manual, on="parameter", how="left", suffixes=("", "_manual"))
        for column in ["r_hat", "ess_bulk", "ess_tail", "mcse_mean"]:
            manual_col = f"{column}_manual"
            if manual_col not in summary:
                continue
            if column not in summary:
                summary[column] = summary[manual_col]
            else:
                summary[column] = summary[column].where(np.isfinite(pd.to_numeric(summary[column], errors="coerce")), summary[manual_col])
            summary = summary.drop(columns=[manual_col])
    warnings = []
    for _, row in summary.iterrows():
        notes: list[str] = []
        rhat = row.get("r_hat", np.nan)
        ess_bulk = row.get("ess_bulk", np.nan)
        ess_tail = row.get("ess_tail", np.nan)
        if np.isfinite(rhat) and rhat > 1.01:
            notes.append("R-hat > 1.01")
        if np.isfinite(ess_bulk) and ess_bulk < 400:
            notes.append("bulk ESS < 400")
        if np.isfinite(ess_tail) and ess_tail < 400:
            notes.append("tail ESS < 400")
        warnings.append("; ".join(notes))
    summary["warning"] = warnings
    return summary


def _autocorr_ess(values: np.ndarray) -> float:
    draws = np.asarray(values, dtype=float)
    if draws.ndim != 2:
        return float("nan")
    chains, n_draws = draws.shape
    if chains < 1 or n_draws < 3:
        return float("nan")
    centered = draws - np.nanmean(draws, axis=1, keepdims=True)
    var0 = np.nanmean(centered**2)
    if not np.isfinite(var0) or var0 <= 0.0:
        return float(chains * n_draws)

    rhos: list[float] = []
    max_lag = min(n_draws - 1, 1000)
    for lag in range(1, max_lag + 1):
        acov = np.nanmean(centered[:, :-lag] * centered[:, lag:])
        rho = float(acov / var0)
        if not np.isfinite(rho):
            break
        rhos.append(rho)
        if lag % 2 == 0 and len(rhos) >= 2 and (rhos[-1] + rhos[-2]) < 0.0:
            rhos = rhos[:-2]
            break
    tau = 1.0 + 2.0 * float(np.sum(rhos))
    if not np.isfinite(tau) or tau <= 0.0:
        return float(chains * n_draws)
    return float(min(chains * n_draws, max(1.0, chains * n_draws / tau)))


def _tail_ess(values: np.ndarray) -> float:
    flat = np.asarray(values, dtype=float).reshape(-1)
    flat = flat[np.isfinite(flat)]
    if flat.size < 4:
        return float("nan")
    lo, hi = np.nanquantile(flat, [0.05, 0.95])
    draws = np.asarray(values, dtype=float)
    ess_lo = _autocorr_ess((draws <= lo).astype(float))
    ess_hi = _autocorr_ess((draws >= hi).astype(float))
    return float(np.nanmin([ess_lo, ess_hi]))


def _basic_chain_diagnostics(idata, names: list[str]) -> pd.DataFrame:
    posterior = getattr(idata, "posterior", None)
    rows: list[dict[str, float | str]] = []
    if posterior is None:
        return pd.DataFrame()
    for name in names:
        if name not in posterior:
            continue
        values = np.asarray(posterior[name], dtype=float)
        if values.ndim != 2:
            continue
        values = values[np.all(np.isfinite(values), axis=1)]
        if values.ndim != 2 or values.shape[0] < 1 or values.shape[1] < 2:
            continue
        chains, draws = values.shape
        chain_vars = np.var(values, axis=1, ddof=1)
        W = float(np.mean(chain_vars))
        if chains > 1 and W > 0.0:
            B = draws * float(np.var(np.mean(values, axis=1), ddof=1))
            var_hat = ((draws - 1.0) / draws) * W + B / draws
            rhat = float(np.sqrt(max(var_hat / W, 0.0)))
        elif W == 0.0:
            rhat = 1.0
        else:
            rhat = float("nan")
        ess_bulk = _autocorr_ess(values)
        ess_tail = _tail_ess(values)
        sd = float(np.std(values.reshape(-1), ddof=1)) if values.size > 1 else 0.0
        mcse = float(sd / np.sqrt(ess_bulk)) if np.isfinite(ess_bulk) and ess_bulk > 0.0 else float("nan")
        rows.append(
            {
                "parameter": name,
                "r_hat": rhat,
                "ess_bulk": ess_bulk,
                "ess_tail": ess_tail,
                "mcse_mean": mcse,
            }
        )
    return pd.DataFrame(rows)


def _manual_summary(idata, names: list[str]) -> pd.DataFrame:
    posterior = getattr(idata, "posterior", None)
    rows = []
    for name in names:
        if posterior is None or name not in posterior:
            continue
        values = np.asarray(posterior[name], dtype=float)
        flat = values.reshape(-1)
        flat = flat[np.isfinite(flat)]
        if flat.size == 0:
            continue
        rows.append(
            {
                "parameter": name,
                "mean": float(np.mean(flat)),
                "sd": float(np.std(flat, ddof=1)) if flat.size > 1 else 0.0,
                "hdi_2.5%": float(np.quantile(flat, 0.025)),
                "hdi_97.5%": float(np.quantile(flat, 0.975)),
                "mcse_mean": np.nan,
                "ess_bulk": np.nan,
                "ess_tail": np.nan,
                "r_hat": np.nan,
            }
        )
    return pd.DataFrame(rows)


def check_finite_posterior(idata) -> list[str]:
    warnings: list[str] = []
    posterior = getattr(idata, "posterior", None)
    if posterior is None:
        return ["Missing posterior group"]
    for name, var in posterior.data_vars.items():
        values = np.asarray(var)
        if not np.all(np.isfinite(values)):
            warnings.append(f"{name} contains NaN or infinite draws")
    return warnings


def ar2_nonstationary_share(idata) -> float | None:
    posterior = getattr(idata, "posterior", None)
    if posterior is None or "rho_1" not in posterior or "rho_2" not in posterior:
        return None
    r1 = np.asarray(posterior["rho_1"], dtype=float).reshape(-1)
    r2 = np.asarray(posterior["rho_2"], dtype=float).reshape(-1)
    mask = np.isfinite(r1) & np.isfinite(r2)
    if not np.any(mask):
        return None
    stationary = (np.abs(r2[mask]) < 1.0) & ((r1[mask] + r2[mask]) < 1.0) & ((r2[mask] - r1[mask]) < 1.0)
    return float(1.0 - np.mean(stationary))


def check_estimation_success(idata, *, min_ess: float = 400.0, max_rhat: float = 1.01) -> pd.DataFrame:
    """Return one row per warning; an empty table means no automatic flags."""
    rows: list[dict[str, object]] = []
    posterior = getattr(idata, "posterior", None)
    if posterior is None:
        return pd.DataFrame([{"severity": "error", "parameter": "", "warning": "Missing posterior group"}])

    for warning in check_finite_posterior(idata):
        rows.append({"severity": "error", "parameter": "", "warning": warning})

    summary = compute_diagnostics(idata)
    for _, row in summary.iterrows():
        param = str(row.get("parameter", ""))
        rhat = row.get("r_hat", np.nan)
        ess_bulk = row.get("ess_bulk", np.nan)
        ess_tail = row.get("ess_tail", np.nan)
        if np.isfinite(rhat) and float(rhat) > max_rhat:
            rows.append({"severity": "warning", "parameter": param, "warning": f"R-hat {rhat:.3f} > {max_rhat:.2f}"})
        if np.isfinite(ess_bulk) and float(ess_bulk) < min_ess:
            rows.append({"severity": "warning", "parameter": param, "warning": f"bulk ESS {ess_bulk:.1f} < {min_ess:.0f}"})
        if np.isfinite(ess_tail) and float(ess_tail) < min_ess:
            rows.append({"severity": "warning", "parameter": param, "warning": f"tail ESS {ess_tail:.1f} < {min_ess:.0f}"})

    for name, var in posterior.data_vars.items():
        if name.startswith("sigma"):
            values = np.asarray(var, dtype=float).reshape(-1)
            values = values[np.isfinite(values)]
            if values.size == 0:
                continue
            if float(np.nanquantile(values, 0.01)) <= 1e-8:
                rows.append({"severity": "warning", "parameter": name, "warning": "variance/std draws are near zero"})
            if float(np.nanquantile(values, 0.99)) > 1e4:
                rows.append({"severity": "warning", "parameter": name, "warning": "variance/std draws are extremely large"})

    share = ar2_nonstationary_share(idata)
    if share is not None and share > 0.0:
        rows.append({"severity": "error", "parameter": "rho_1,rho_2", "warning": f"AR(2) nonstationary draw share is {share:.3f}"})

    if {"Nhat", "Nbar"} <= set(posterior.data_vars):
        n_sum = np.asarray(posterior["Nhat"], dtype=float) + np.asarray(posterior["Nbar"], dtype=float)
        if not np.all(np.isfinite(n_sum)):
            rows.append({"severity": "error", "parameter": "Nhat,Nbar", "warning": "latent N decomposition contains non-finite values"})

    return pd.DataFrame(rows, columns=["severity", "parameter", "warning"])


def save_diagnostics(idata, out_dir: str | Path, *, var_names: Iterable[str] | None = None) -> pd.DataFrame:
    import arviz as az
    import matplotlib.pyplot as plt

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    params = list(var_names or available_key_parameters(idata))
    summary = compute_diagnostics(idata, var_names=params)
    summary.to_csv(out / "mcmc_diagnostics.csv", index=False)
    warnings = check_estimation_success(idata)
    warnings.to_csv(out / "estimation_warnings.csv", index=False)
    if not warnings.empty:
        (out / "warnings.txt").write_text("\n".join(warnings["warning"].astype(str)), encoding="utf-8")
    if params:
        plot_idata, plot_params = _scalarized_diagnostic_idata(idata, params)
        if not plot_params:
            return summary
        try:
            current_max = int(az.rcParams.get("plot.max_subplots", 40))
            az.rcParams["plot.max_subplots"] = max(current_max, len(plot_params) * 4 + 4)
        except Exception:
            pass
        try:
            az.plot_trace(plot_idata, var_names=plot_params, compact=True)
        except Exception:
            az.plot_trace(plot_idata, var_names=plot_params)
        plt.tight_layout()
        plt.savefig(out / "trace_key_parameters.png", dpi=200, bbox_inches="tight")
        plt.close("all")
        try:
            az.plot_autocorr(plot_idata, var_names=plot_params)
            plt.tight_layout()
            plt.savefig(out / "autocorr_key_parameters.png", dpi=200, bbox_inches="tight")
            plt.close("all")
        except Exception as exc:
            (out / "autocorr_warning.txt").write_text(str(exc), encoding="utf-8")
            plt.close("all")
        if hasattr(az, "plot_posterior"):
            az.plot_posterior(plot_idata, var_names=plot_params, hdi_prob=0.95)
            plt.tight_layout()
            plt.savefig(out / "density_key_parameters.png", dpi=200, bbox_inches="tight")
            plt.close("all")
    return summary
