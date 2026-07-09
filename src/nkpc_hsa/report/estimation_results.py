from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


DEFAULT_PARAMETER_NAMES = [
    "alpha",
    "kappa",
    "kappa_0",
    "delta",
    "theta",
    "theta_0",
    "gamma",
    "rho_1",
    "rho_2",
    "phi_1",
    "lambda_ez",
    "n",
    "sigma_N",
    "sigma_e",
    "sigma_eta",
    "sigma_zeta",
    "sigma_u",
    "sigma_eps",
]


def _posterior_array(idata: Any, name: str) -> np.ndarray | None:
    posterior = getattr(idata, "posterior", None)
    if posterior is None or name not in posterior:
        return None
    return np.asarray(posterior[name], dtype=float)


def _summarize_values(values: np.ndarray, *, parameter: str) -> dict[str, Any]:
    x = np.asarray(values, dtype=float).reshape(-1)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {}
    return {
        "parameter": parameter,
        "mean": float(np.mean(x)),
        "sd": float(np.std(x, ddof=1)) if x.size > 1 else 0.0,
        "q05": float(np.quantile(x, 0.05)),
        "q50": float(np.quantile(x, 0.50)),
        "q95": float(np.quantile(x, 0.95)),
        "n_draws": int(x.size),
    }


def posterior_parameter_summary(
    idata: Any,
    *,
    parameter_names: Sequence[str] = DEFAULT_PARAMETER_NAMES,
) -> pd.DataFrame:
    rows = []
    for name in parameter_names:
        arr = _posterior_array(idata, name)
        if arr is None:
            continue
        row = _summarize_values(arr, parameter=name)
        if row:
            rows.append(row)
    return pd.DataFrame(rows)


def competition_decomposition_summary(
    idata: Any,
    quarterly_index: Any,
    *,
    max_rows: int | None = None,
) -> pd.DataFrame:
    nbar = _posterior_array(idata, "Nbar")
    nhat = _posterior_array(idata, "Nhat")
    if nbar is None or nhat is None:
        return pd.DataFrame()
    if nbar.shape != nhat.shape or nbar.ndim < 3:
        return pd.DataFrame()

    nbar_draws = nbar.reshape(-1, nbar.shape[-1])
    nhat_draws = nhat.reshape(-1, nhat.shape[-1])
    total_draws = nbar_draws + nhat_draws
    T = total_draws.shape[-1]

    if isinstance(quarterly_index, pd.PeriodIndex):
        periods = quarterly_index.asfreq("Q")
    elif quarterly_index is not None:
        idx = pd.Index(quarterly_index)
        if np.issubdtype(idx.dtype, np.datetime64):
            periods = pd.DatetimeIndex(idx).to_period("Q")
        else:
            periods = pd.period_range("2000Q1", periods=T, freq="Q")
    else:
        periods = pd.period_range("2000Q1", periods=T, freq="Q")
    if len(periods) != T:
        periods = pd.period_range("2000Q1", periods=T, freq="Q")

    def _path_stats(draws: np.ndarray, prefix: str) -> dict[str, np.ndarray]:
        return {
            f"{prefix}_mean": np.nanmean(draws, axis=0),
            f"{prefix}_q05": np.nanquantile(draws, 0.05, axis=0),
            f"{prefix}_q50": np.nanquantile(draws, 0.50, axis=0),
            f"{prefix}_q95": np.nanquantile(draws, 0.95, axis=0),
        }

    cols: dict[str, Any] = {
        "quarter": [str(p) for p in periods],
    }
    cols.update(_path_stats(total_draws, "N_total"))
    cols.update(_path_stats(nbar_draws, "Nbar"))
    cols.update(_path_stats(nhat_draws, "Nhat"))
    out = pd.DataFrame(cols)
    if max_rows is not None and len(out) > max_rows:
        idx = np.linspace(0, len(out) - 1, max_rows).round().astype(int)
        out = out.iloc[idx].reset_index(drop=True)
    return out


def _markdown_table(df: pd.DataFrame, columns: Sequence[str], *, max_rows: int = 24) -> list[str]:
    if df.empty:
        return ["not available"]
    view = df[[col for col in columns if col in df.columns]].head(max_rows).copy()
    for col in view.columns:
        if pd.api.types.is_numeric_dtype(view[col]):
            view[col] = view[col].map(lambda x: f"{float(x):.6g}")
    lines = ["| " + " | ".join(view.columns) + " |", "| " + " | ".join(["---"] * len(view.columns)) + " |"]
    for _, row in view.iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in view.columns) + " |")
    if len(df) > max_rows:
        lines.append(f"\nShowing {max_rows} of {len(df)} rows. See the CSV artifact for the full path.")
    return lines


def write_estimation_results_report(
    output_dir: str | Path,
    idata: Any,
    *,
    metadata: Mapping[str, Any] | None = None,
    quarterly_index: Any = None,
) -> Path:
    """Write mechanical posterior summaries, including N decomposition paths."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    report_dir = out / "report"
    tables_dir = out / "tables"
    report_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    meta = dict(metadata or {})

    param_summary = posterior_parameter_summary(idata)
    decomp = competition_decomposition_summary(idata, quarterly_index)
    if not param_summary.empty:
        param_summary.to_csv(tables_dir / "posterior_summary.csv", index=False)
        param_summary.to_csv(out / "posterior_summary.csv", index=False)
    if not decomp.empty:
        decomp.to_csv(tables_dir / "competition_decomposition_summary.csv", index=False)
        decomp.to_csv(out / "competition_decomposition_summary.csv", index=False)

    target = report_dir / "estimation_results_report.md"
    created = datetime.now(timezone.utc).isoformat(timespec="seconds")
    lines = [
        "# Estimation Results Report",
        "",
        "This report gives mechanical posterior summaries for the saved run. It does not provide interpretation.",
        "",
        "## Run Metadata",
        "",
        f"- **created_at:** {created}",
        f"- **run_name:** {out.name}",
        f"- **model:** {meta.get('model', 'not available')}",
        f"- **data_spec:** {meta.get('data_spec', 'not available')}",
        f"- **competition_measurement:** {dict(meta.get('competition_measurement', {}) or {}).get('frequency', 'not available')}",
        f"- **posterior_file:** {out / 'posterior.nc'}",
        "",
        "## Parameter Summary",
        "",
    ]
    lines.extend(_markdown_table(param_summary, ["parameter", "mean", "sd", "q05", "q50", "q95", "n_draws"]))
    lines.extend(
        [
            "",
            "## Competition Decomposition",
            "",
            "The decomposition uses posterior draws of `Nbar`, `Nhat`, and `N_total = Nbar + Nhat`.",
            "",
        ]
    )
    lines.extend(
        _markdown_table(
            decomp,
            [
                "quarter",
                "N_total_mean",
                "N_total_q05",
                "N_total_q50",
                "N_total_q95",
                "Nbar_mean",
                "Nhat_mean",
            ],
            max_rows=24,
        )
    )
    lines.extend(
        [
            "",
            "## Output Files",
            "",
            f"- {tables_dir / 'posterior_summary.csv'}" if not param_summary.empty else "- posterior_summary.csv not generated",
            f"- {tables_dir / 'competition_decomposition_summary.csv'}" if not decomp.empty else "- competition_decomposition_summary.csv not generated",
        ]
    )
    target.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    (out / "estimation_results_report.md").write_text(target.read_text(encoding="utf-8"), encoding="utf-8")
    return target
