from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import pandas as pd
import yaml

from nkpc_hsa.inference.diagnostics import compute_diagnostics
from nkpc_hsa.inference.wrappers import run_model
from nkpc_hsa.paths import project_path
from nkpc_hsa.report.tables import posterior_summary_table


def load_periods(path: str | Path | None = None) -> dict[str, dict[str, Any]]:
    target = Path(path) if path is not None else project_path("configs", "periods.yaml")
    config = yaml.safe_load(target.read_text(encoding="utf-8")) or {}
    return dict(config.get("periods", {}))


def apply_period(data: pd.DataFrame, period: Mapping[str, Any]) -> pd.DataFrame:
    out = data.copy()
    if not isinstance(out.index, pd.DatetimeIndex):
        if "DATE" not in out.columns:
            raise ValueError("Period filtering requires a DatetimeIndex or DATE column.")
        out = out.copy()
        out["DATE"] = pd.to_datetime(out["DATE"])
        out = out.set_index("DATE")
    start = period.get("start")
    end = period.get("end")
    if start:
        out = out.loc[out.index >= pd.Timestamp(start)]
    if end:
        out = out.loc[out.index <= pd.Timestamp(end)]
    for exclusion in period.get("exclude", []) or []:
        if not exclusion or len(exclusion) != 2:
            continue
        lo, hi = pd.Timestamp(exclusion[0]), pd.Timestamp(exclusion[1])
        out = out.loc[~((out.index >= lo) & (out.index <= hi))]
    return out


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
    n_transform: str = "log100",
    covariance_structure: str = "e_zeta_only",
    coefficient_constraints: Mapping[str, Any] | None = None,
) -> tuple[dict[str, object], pd.DataFrame]:
    periods = dict(periods or load_periods())
    outputs: dict[str, object] = {}
    rows: list[dict[str, Any]] = []
    for i, (period_name, period_spec) in enumerate(periods.items()):
        subset = apply_period(data, period_spec)
        n_obs = int(len(subset))
        row = {
            "model": model,
            "period": period_name,
            "start": "" if subset.empty else subset.index.min().date().isoformat(),
            "end": "" if subset.empty else subset.index.max().date().isoformat(),
            "n_obs": n_obs,
            "status": "skipped" if n_obs < min_obs else "estimated",
            "warning": "" if n_obs >= min_obs else f"Too few observations: {n_obs} < {min_obs}",
        }
        if n_obs >= min_obs:
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
                covariance_structure=covariance_structure,
                coefficient_constraints=coefficient_constraints,
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
