from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import yaml

from nkpc_hsa.data.transforms import DEFAULT_N_TRANSFORM
from nkpc_hsa.inference.diagnostics import compute_diagnostics
from nkpc_hsa.inference.wrappers import model_sample_index, run_model
from nkpc_hsa.paths import project_path
from nkpc_hsa.report.tables import posterior_summary_table


def load_periods(path: str | Path | None = None) -> dict[str, dict[str, Any]]:
    target = Path(path) if path is not None else project_path("configs", "periods.yaml")
    config = yaml.safe_load(target.read_text(encoding="utf-8")) or {}
    return dict(config.get("periods", {}))


def apply_period(data: pd.DataFrame, period: Mapping[str, Any]) -> pd.DataFrame:
    """Filter quarterly data using inclusive quarter boundaries."""
    out = data.copy()
    if not isinstance(out.index, (pd.DatetimeIndex, pd.PeriodIndex)):
        if "DATE" not in out.columns:
            raise ValueError("Period filtering requires a DatetimeIndex or DATE column.")
        out = out.copy()
        out["DATE"] = pd.to_datetime(out["DATE"])
        out = out.set_index("DATE")
    quarters = out.index.asfreq("Q") if isinstance(out.index, pd.PeriodIndex) else out.index.to_period("Q")
    start = period.get("start")
    end = period.get("end")
    if start:
        keep = quarters >= pd.Timestamp(start).to_period("Q")
        out = out.loc[keep]
        quarters = quarters[keep]
    if end:
        keep = quarters <= pd.Timestamp(end).to_period("Q")
        out = out.loc[keep]
        quarters = quarters[keep]
    for exclusion in period.get("exclude", []) or []:
        if not exclusion or len(exclusion) != 2:
            continue
        lo = pd.Timestamp(exclusion[0]).to_period("Q")
        hi = pd.Timestamp(exclusion[1]).to_period("Q")
        keep = ~((quarters >= lo) & (quarters <= hi))
        out = out.loc[keep]
        quarters = quarters[keep]
    return out


def _has_quarterly_gap(index: pd.Index) -> bool:
    if len(index) < 2:
        return False
    if not isinstance(index, (pd.DatetimeIndex, pd.PeriodIndex)):
        return False
    quarters = index.asfreq("Q") if isinstance(index, pd.PeriodIndex) else index.to_period("Q")
    ordinals = np.sort(np.unique(quarters.asi8))
    return bool(ordinals.size > 1 and np.any(np.diff(ordinals) != 1))


def _iso_date(value: Any) -> str:
    if isinstance(value, pd.Period):
        value = value.to_timestamp(how="end")
    return pd.Timestamp(value).date().isoformat()


def run_period_robustness(
    model: str,
    *,
    data: pd.DataFrame,
    periods: Mapping[str, Mapping[str, Any]] | None = None,
    data_spec: Mapping[str, Any] | None = None,
    prior_specs: str | Path | Mapping[str, Any] | None = None,
    prior_name: str = "baseline",
    n_iter: int = 12000,
    burn: int = 4000,
    thin: int = 5,
    chains: int = 2,
    seed: int = 12345,
    min_obs: int = 40,
    n_transform: str = DEFAULT_N_TRANSFORM,
    covariance_structure: str = "e_zeta_only",
    coefficient_constraints: Mapping[str, Any] | None = None,
    enforce_stationary: bool = True,
    ar2_max_tries: int = 2000,
) -> tuple[dict[str, object], pd.DataFrame]:
    periods = dict(periods or load_periods())
    outputs: dict[str, object] = {}
    rows: list[dict[str, Any]] = []
    for i, (period_name, period_spec) in enumerate(periods.items()):
        subset = apply_period(data, period_spec)
        sample_index = model_sample_index(subset, dict(data_spec or {}))
        if sample_index is None:
            sample_index = subset.index
        n_obs = int(len(sample_index))
        has_gap = _has_quarterly_gap(sample_index)
        can_estimate = n_obs >= min_obs and not has_gap
        if n_obs < min_obs:
            warning = f"Too few observations: {n_obs} < {min_obs}"
        elif has_gap:
            warning = (
                "Non-contiguous quarterly sample; skipped because the current state equations "
                "assume one-quarter transitions. Estimate contiguous pre/post subsamples separately."
            )
        else:
            warning = ""
        row = {
            "model": model,
            "period": period_name,
            "start": "" if n_obs == 0 else _iso_date(sample_index.min()),
            "end": "" if n_obs == 0 else _iso_date(sample_index.max()),
            "n_obs": n_obs,
            "status": "estimated" if can_estimate else "skipped",
            "warning": warning,
            "n_transform": n_transform,
        }
        if can_estimate:
            base_spec = dict(data_spec or {})
            base_name = str(base_spec.get("name", "default"))
            period_data_spec = {**base_spec, "name": f"{base_name}_{period_name}"}
            idata = run_model(
                model,
                data=subset,
                data_spec=period_data_spec,
                prior_specs=prior_specs,
                prior_name=prior_name,
                n_iter=n_iter,
                burn=burn,
                thin=thin,
                chains=chains,
                seed=seed + i,
                n_transform=n_transform,
                period_name=period_name,
                covariance_structure=covariance_structure,
                coefficient_constraints=coefficient_constraints,
                enforce_stationary=enforce_stationary,
                ar2_max_tries=ar2_max_tries,
            )
            outputs[period_name] = idata
            diag = compute_diagnostics(idata)
            if not diag.empty and "warning" in diag:
                warnings = "; ".join(sorted(set(w for w in diag["warning"].astype(str) if w)))
                row["warning"] = warnings
            summary = posterior_summary_table(idata, var_names=["alpha", "kappa", "kappa_0", "delta", "theta", "theta_0", "gamma"])
            for _, srow in summary.iterrows():
                rows.append({**row, "parameter": srow["parameter"], "mean": srow["mean"], "ci_2.5": srow["ci_2.5"], "ci_97.5": srow["ci_97.5"]})
        else:
            rows.append({**row, "parameter": "", "mean": float("nan"), "ci_2.5": float("nan"), "ci_97.5": float("nan")})
    return outputs, pd.DataFrame(rows)


def save_period_robustness_table(table: pd.DataFrame, out_dir: str | Path) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "period_robustness.csv", index=False)
    table.to_latex(out / "period_robustness.tex", index=False, float_format="%.4f", escape=True)
