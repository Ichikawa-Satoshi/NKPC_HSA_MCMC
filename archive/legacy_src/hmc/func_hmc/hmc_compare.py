from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import arviz as az
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import logsumexp


FAMILY_LABELS = {
    "hmc_ces": "CES",
    "hmc_ces_xAR1": "CES xAR1",
    "hsa_dynamic": "HSA dynamic",
    "hsa_steady": "HSA steady",
    "hsa_full": "HSA full",
}

FAMILY_KINDS = {
    "hmc_ces": "ces",
    "hmc_ces_xAR1": "ces",
    "hsa_dynamic": "hsa_dynamic",
    "hsa_steady": "hsa_steady",
    "hsa_full": "hsa_full",
}

KNOWN_PREFIXES = sorted(
    {
        "CES",
        "hsa_dynamic",
        "hsa_steady",
        "hsa_full",
    },
    key=len,
    reverse=True,
)

DEFAULT_COMPARE_PARAMS = (
    "alpha",
    "kappa",
    "theta",
    "delta",
    "gamma",
    "kappa_0",
    "theta_0",
    "phi_1",
    "n",
    "rho_1",
    "rho_2",
    "sigma_e",
    "sigma_zeta",
    "sigma_u",
    "sigma_eps",
)


@dataclass(frozen=True)
class IdataRecord:
    label: str
    family: str
    kind: str
    x_name: str
    path: Path
    idata: object


def _flatten_draws(values: object) -> np.ndarray:
    arr = np.asarray(values)
    if arr.ndim == 0:
        return arr.reshape(1)
    if arr.ndim == 1:
        return arr.reshape(-1)
    return arr.reshape(-1, *arr.shape[2:])


def _stack_time_draws(values: object) -> np.ndarray:
    arr = np.asarray(values)
    if arr.ndim == 0:
        return arr.reshape(1, 1)
    if arr.ndim == 1:
        return arr.reshape(-1, 1)
    if arr.ndim == 2:
        return arr.reshape(-1, 1)
    return arr.reshape(-1, arr.shape[-1])


def _family_label(family: str) -> str:
    return FAMILY_LABELS.get(family, family)


def _family_kind(family: str) -> str:
    return FAMILY_KINDS.get(family, family)


def _parse_run_path(path: Path) -> tuple[str, str, str]:
    family = path.parent.name
    stem = path.stem
    x_name = stem
    for prefix in KNOWN_PREFIXES:
        if stem.startswith(f"{prefix}_"):
            x_name = stem[len(prefix) + 1 :]
            break
    label = f"{_family_label(family)} - {x_name}"
    kind = _family_kind(family)
    return label, kind, x_name


def load_idata_records(idata_root: str | Path) -> list[IdataRecord]:
    root = Path(idata_root)
    records: list[IdataRecord] = []
    for path in sorted(root.rglob("*.nc")):
        label, kind, x_name = _parse_run_path(path)
        records.append(
            IdataRecord(
                label=label,
                family=path.parent.name,
                kind=kind,
                x_name=x_name,
                path=path,
                idata=az.from_netcdf(path),
            )
        )
    return records


def records_to_map(records: Sequence[IdataRecord]) -> dict[str, object]:
    return {record.label: record.idata for record in records}


def group_records(records: Sequence[IdataRecord], key: str = "x_name") -> dict[str, list[IdataRecord]]:
    grouped: dict[str, list[IdataRecord]] = {}
    for record in records:
        value = getattr(record, key)
        grouped.setdefault(value, []).append(record)
    return grouped


def _posterior_var(idata: object, name: str) -> np.ndarray | None:
    posterior = getattr(idata, "posterior", None)
    if posterior is None or name not in posterior:
        return None
    return _flatten_draws(posterior[name])


def _trend_draws(idata: object) -> np.ndarray | None:
    posterior = getattr(idata, "posterior", None)
    if posterior is None:
        return None

    nbar = _posterior_var(idata, "Nbar")
    bar_n0 = _posterior_var(idata, "bar_N_0")
    nbar_vars = []
    for name in posterior.data_vars:
        if name.startswith("Nbar_"):
            try:
                nbar_vars.append((int(name.split("_", 1)[1]), name))
            except Exception:
                continue

    if nbar is not None:
        nbar = np.asarray(nbar)
        if nbar.ndim == 1:
            nbar = nbar.reshape(-1, 1)
        elif nbar.ndim >= 2:
            nbar = nbar.reshape(-1, nbar.shape[-1])

        if bar_n0 is not None:
            bar_n0 = np.asarray(bar_n0).reshape(-1, 1)
            if bar_n0.shape[0] == nbar.shape[0]:
                return np.concatenate([bar_n0, nbar], axis=1)
        return nbar

    series = []
    if bar_n0 is not None:
        series.append(np.asarray(bar_n0).reshape(-1, 1))
    for _, name in sorted(nbar_vars, key=lambda item: item[0]):
        series.append(np.asarray(_posterior_var(idata, name)).reshape(-1, 1))

    if not series:
        return None
    return np.concatenate(series, axis=1)


def _summarize_draw_matrix(draws: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    draws = np.asarray(draws)
    mean = np.nanmean(draws, axis=0)
    lo = np.nanquantile(draws, 0.025, axis=0)
    hi = np.nanquantile(draws, 0.975, axis=0)
    return mean, lo, hi


def build_posterior_means_table(
    records: Sequence[IdataRecord],
    params: Sequence[str] = DEFAULT_COMPARE_PARAMS,
    *,
    hdi_prob: float = 0.95,
) -> pd.DataFrame:
    rows = []
    for record in records:
        posterior = getattr(record.idata, "posterior", None)
        if posterior is None:
            continue
        selected = [param for param in params if param in posterior.data_vars]
        if not selected:
            continue
        summary = az.summary(record.idata, var_names=selected, hdi_prob=hdi_prob).reset_index(names="param")
        summary.insert(0, "model", record.label)
        summary.insert(1, "family", _family_label(record.family))
        summary.insert(2, "x_name", record.x_name)
        rows.append(summary[["family", "x_name", "model", "param", "mean", "hdi_2.5%", "hdi_97.5%"]])
    if not rows:
        return pd.DataFrame(columns=["family", "x_name", "model", "param", "mean", "hdi_2.5%", "hdi_97.5%"])
    return pd.concat(rows, ignore_index=True)


def _combined_log_likelihood(idata: object) -> np.ndarray | None:
    log_lik = getattr(idata, "log_likelihood", None)
    if log_lik is None:
        return None

    parts = []
    for name in log_lik.data_vars:
        arr = np.asarray(log_lik[name])
        if arr.ndim < 2:
            continue
        if arr.ndim == 2:
            arr = arr[..., None]
        else:
            arr = arr.reshape(arr.shape[0], arr.shape[1], -1)
        parts.append(arr)

    if not parts:
        return None
    return np.concatenate(parts, axis=2)


def build_waic_loo_table(records: Sequence[IdataRecord]) -> pd.DataFrame:
    rows = []
    for record in records:
        combined = _combined_log_likelihood(record.idata)
        if combined is None:
            continue
        ll_idata = az.from_dict(log_likelihood={"obs": combined})
        waic = az.waic(ll_idata, var_name="obs")
        loo = az.loo(ll_idata, var_name="obs", reff=1.0)
        rows.append(
            {
                "family": _family_label(record.family),
                "x_name": record.x_name,
                "model": record.label,
                "elpd_waic": float(waic.elpd_waic),
                "p_waic": float(waic.p_waic),
                "waic": float(-2.0 * waic.elpd_waic),
                "elpd_loo": float(loo.elpd_loo),
                "p_loo": float(loo.p_loo),
                "loo": float(-2.0 * loo.elpd_loo),
                "loo_warning": bool(getattr(loo, "warning", False)),
                "waic_warning": bool(getattr(waic, "warning", False)),
            }
        )
    return pd.DataFrame(rows)


def build_ranking_table(ic_table: pd.DataFrame) -> pd.DataFrame:
    if ic_table.empty:
        return ic_table.copy()

    out = ic_table.copy()
    out["rank_waic"] = out.groupby("x_name")["waic"].rank(method="min", ascending=True)
    out["rank_loo"] = out.groupby("x_name")["loo"].rank(method="min", ascending=True)
    return out.sort_values(["x_name", "waic", "loo", "model"]).reset_index(drop=True)


def _prediction_draws(record: IdataRecord, data: pd.DataFrame) -> np.ndarray | None:
    y = np.asarray(data["pi_ppi"], dtype=float)
    y_prev = np.asarray(data["pi_ppi_prev"], dtype=float)
    y_expect = np.asarray(data["Epi_spf_gdp"], dtype=float)
    x = np.asarray(data[record.x_name], dtype=float)
    n = len(y)

    alpha = _posterior_var(record.idata, "alpha")
    sigma_e = _posterior_var(record.idata, "sigma_e")
    if alpha is None or sigma_e is None:
        return None

    alpha = np.asarray(alpha).reshape(-1)
    sigma_e = np.asarray(sigma_e).reshape(-1)

    if record.kind == "ces":
        kappa = _posterior_var(record.idata, "kappa")
        if kappa is None:
            return None
        kappa = np.asarray(kappa).reshape(-1)
        pred = alpha[:, None] * y_prev[None, :] + (1.0 - alpha)[:, None] * y_expect[None, :] + kappa[:, None] * x[None, :]
        return pred

    trend = _trend_draws(record.idata)
    if trend is None or trend.shape[1] <= 1:
        return None

    core = slice(1, n)
    pred = np.full((alpha.shape[0], n), np.nan, dtype=float)

    if record.kind == "hsa_dynamic":
        kappa = _posterior_var(record.idata, "kappa")
        theta = _posterior_var(record.idata, "theta")
        if kappa is None or theta is None:
            return None
        kappa = np.asarray(kappa).reshape(-1)
        theta = np.asarray(theta).reshape(-1)
        n_hat = y[None, :] - trend
        pred[:, core] = (
            alpha[:, None] * y_prev[None, core]
            + (1.0 - alpha)[:, None] * y_expect[None, core]
            + kappa[:, None] * x[None, core]
            - theta[:, None] * n_hat[:, core]
        )
        return pred

    if record.kind == "hsa_steady":
        kappa0 = _posterior_var(record.idata, "kappa_0")
        delta = _posterior_var(record.idata, "delta")
        if kappa0 is None or delta is None:
            return None
        kappa0 = np.asarray(kappa0).reshape(-1)
        delta = np.asarray(delta).reshape(-1)
        kappa_t = kappa0[:, None] + delta[:, None] * trend[:, 1:]
        pred[:, core] = (
            alpha[:, None] * y_prev[None, core]
            + (1.0 - alpha)[:, None] * y_expect[None, core]
            + kappa_t * x[None, core]
        )
        return pred

    if record.kind == "hsa_full":
        kappa0 = _posterior_var(record.idata, "kappa_0")
        delta = _posterior_var(record.idata, "delta")
        theta0 = _posterior_var(record.idata, "theta_0")
        gamma = _posterior_var(record.idata, "gamma")
        if kappa0 is None or delta is None or theta0 is None or gamma is None:
            return None
        kappa0 = np.asarray(kappa0).reshape(-1)
        delta = np.asarray(delta).reshape(-1)
        theta0 = np.asarray(theta0).reshape(-1)
        gamma = np.asarray(gamma).reshape(-1)
        kappa_t = kappa0[:, None] + delta[:, None] * trend[:, 1:]
        theta_t = theta0[:, None] + gamma[:, None] * trend[:, 1:]
        n_hat = y[None, :] - trend
        pred[:, core] = (
            alpha[:, None] * y_prev[None, core]
            + (1.0 - alpha)[:, None] * y_expect[None, core]
            + kappa_t * x[None, core]
            - theta_t * n_hat[:, core]
        )
        return pred

    return None


def build_prediction_metrics_table(records: Sequence[IdataRecord], data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    y_true = np.asarray(data["pi_ppi"], dtype=float)

    for record in records:
        pred = _prediction_draws(record, data)
        if pred is None:
            continue

        mean_pred = np.nanmean(pred, axis=0)
        mask = np.isfinite(mean_pred) & np.isfinite(y_true)
        if not np.any(mask):
            continue

        err = mean_pred[mask] - y_true[mask]
        rmse = float(np.sqrt(np.mean(err**2)))
        mae = float(np.mean(np.abs(err)))
        corr = float(np.corrcoef(y_true[mask], mean_pred[mask])[0, 1]) if mask.sum() > 1 else np.nan
        ss_res = float(np.sum(err**2))
        ss_tot = float(np.sum((y_true[mask] - np.mean(y_true[mask])) ** 2))
        r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else np.nan

        sigma_e = _posterior_var(record.idata, "sigma_e")
        if sigma_e is not None:
            sigma_e = np.asarray(sigma_e).reshape(-1)
            obs = y_true[None, :]
            sigma = sigma_e[:, None]
            finite = np.isfinite(pred) & np.isfinite(obs)
            logpdf = -0.5 * np.log(2.0 * np.pi * sigma**2) - 0.5 * ((obs - pred) / sigma) ** 2
            valid_cols = np.any(finite, axis=0)
            if np.any(valid_cols):
                safe_logpdf = np.where(finite[:, valid_cols], logpdf[:, valid_cols], -np.inf)
                counts = np.sum(finite[:, valid_cols], axis=0)
                col_scores = logsumexp(safe_logpdf, axis=0) - np.log(counts)
                elpd_pred = float(np.sum(col_scores))
            else:
                elpd_pred = np.nan
        else:
            elpd_pred = np.nan

        rows.append(
            {
                "family": _family_label(record.family),
                "x_name": record.x_name,
                "model": record.label,
                "rmse": rmse,
                "mae": mae,
                "corr": corr,
                "r2": r2,
                "elpd_pred": elpd_pred,
            }
        )

    return pd.DataFrame(rows)


def build_prediction_overlay_figure(
    records: Sequence[IdataRecord],
    data: pd.DataFrame,
    *,
    x_name: str,
    title: str | None = None,
    figsize: tuple[float, float] = (13.0, 6.5),
) -> plt.Figure | None:
    subset = [record for record in records if record.x_name == x_name]
    if not subset:
        return None

    y = np.asarray(data["pi_ppi"], dtype=float)
    time_index = list(data.index)
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    ax.plot(time_index, y, color="black", lw=2.0, label="Actual inflation")

    colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    colors = colors if colors else [f"C{i}" for i in range(len(subset))]

    score_rows = []
    for i, record in enumerate(subset):
        pred = _prediction_draws(record, data)
        if pred is None:
            continue
        mean_pred = np.nanmean(pred, axis=0)
        lo, hi = np.nanquantile(pred, 0.025, axis=0), np.nanquantile(pred, 0.975, axis=0)
        mask = np.isfinite(mean_pred) & np.isfinite(y)
        if not np.any(mask):
            continue

        err = mean_pred[mask] - y[mask]
        rmse = float(np.sqrt(np.mean(err**2)))
        mae = float(np.mean(np.abs(err)))
        color = colors[i % len(colors)]
        ax.plot(time_index, mean_pred, color=color, lw=1.8, label=f"{record.label} (RMSE {rmse:.3f})")
        ax.fill_between(time_index, lo, hi, color=color, alpha=0.12)
        score_rows.append((record.label, rmse, mae))

    ax.set_title(title or f"Predicted vs actual inflation: {x_name}")
    ax.set_ylabel("Inflation")
    ax.set_xlabel("Time")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    return fig


def build_comparison_figure_bundle(records: Sequence[IdataRecord], data: pd.DataFrame) -> dict[str, plt.Figure]:
    figures: dict[str, plt.Figure] = {}
    for x_name in sorted({record.x_name for record in records}):
        fig = build_prediction_overlay_figure(records, data, x_name=x_name)
        if fig is not None:
            figures[x_name] = fig
    return figures
