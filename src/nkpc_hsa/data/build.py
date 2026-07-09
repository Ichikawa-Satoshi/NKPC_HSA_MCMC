from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

from nkpc_hsa.paths import project_path


def _hp_trend_array(values: np.ndarray, lamb: float) -> np.ndarray:
    """Return the Hodrick-Prescott trend for one contiguous finite array."""
    y = np.asarray(values, dtype=float)
    n_obs = y.size
    if n_obs < 4:
        return y.copy()

    main = np.ones(n_obs) + lamb * np.r_[1.0, 5.0, np.repeat(6.0, n_obs - 4), 5.0, 1.0]
    off1 = lamb * np.r_[-2.0, np.repeat(-4.0, n_obs - 3), -2.0]
    off2 = lamb * np.ones(n_obs - 2)
    system = diags([off2, off1, main, off1, off2], [-2, -1, 0, 1, 2], format="csc")
    return np.asarray(spsolve(system, y), dtype=float)


def hp_filter_series(series: pd.Series, lamb: float = 1600.0) -> tuple[pd.Series, pd.Series]:
    """
    Split a quarterly series into HP trend and gap components.

    Missing observations are preserved. The filter is applied independently to
    each contiguous block of finite observations, so raw data gaps do not get
    interpolated silently.
    """
    s = pd.Series(series, copy=True).astype(float)
    trend = pd.Series(np.nan, index=s.index, dtype=float)
    valid = s.notna()
    if not valid.any():
        return trend, s - trend

    groups = valid.ne(valid.shift(fill_value=False)).cumsum()
    for _, block in s[valid].groupby(groups[valid]):
        trend.loc[block.index] = _hp_trend_array(block.to_numpy(dtype=float), lamb)

    return trend, s - trend


def add_hp_output_gap(data: pd.DataFrame, lamb: float = 1600.0) -> pd.DataFrame:
    """
    Add an HP-filtered real-output gap in the same 100-log-point units as BN.

    The legacy builder stores ``output`` as log real GDP minus a constant. The
    BN cycle column is in 100-log-point units, so the HP filter is applied to
    ``100 * output`` before forming the gap.
    """
    if "output" not in data.columns:
        return data

    out = data.copy()
    output_100log = 100.0 * out["output"]
    trend, gap = hp_filter_series(output_100log, lamb=lamb)
    out["output_trend_HP"] = trend
    out["output_gap_HP"] = gap
    out["output_gap_HP_prev"] = out["output_gap_HP"].shift(1)
    return out


def _quarter_end_index(dates: pd.Series | pd.Index) -> pd.DatetimeIndex:
    return pd.to_datetime(dates).to_period("Q").to_timestamp(how="end")


def load_labor_share_gap(raw_dir: str | Path, lamb: float = 1600.0) -> pd.DataFrame:
    """
    Load the quarterly labor-share index and construct an HP-filtered gap.

    The raw FRED series is a positive index. To keep units comparable to the
    output-gap specifications, the cycle is computed from ``100 * log(index)``.
    """
    path = Path(raw_dir) / "laborshare" / "PRS85006173.csv"
    if not path.exists():
        return pd.DataFrame()

    raw = pd.read_csv(path)
    date_col = "DATE" if "DATE" in raw.columns else "observation_date"
    value_cols = [c for c in raw.columns if c != date_col]
    if not value_cols:
        raise ValueError(f"No labor-share value column found in {path}.")
    value_col = value_cols[0]

    labor_share = raw[[date_col, value_col]].copy()
    labor_share[date_col] = pd.to_datetime(labor_share[date_col], errors="coerce")
    labor_share[value_col] = pd.to_numeric(labor_share[value_col], errors="coerce")
    labor_share = labor_share.dropna(subset=[date_col, value_col])
    labor_share = labor_share[labor_share[value_col] > 0.0]
    labor_share = labor_share.set_index(date_col).sort_index()
    labor_share.index = _quarter_end_index(labor_share.index)
    labor_share = labor_share.groupby(level=0).mean()

    out = pd.DataFrame(index=labor_share.index)
    out["labor_share"] = labor_share[value_col]
    out["labor_share_100log"] = 100.0 * np.log(out["labor_share"])
    trend, gap = hp_filter_series(out["labor_share_100log"], lamb=lamb)
    out["labor_share_trend_HP"] = trend
    out["labor_share_gap_HP"] = gap
    return out


def add_labor_share_gap(data: pd.DataFrame, raw_dir: str | Path, lamb: float = 1600.0) -> pd.DataFrame:
    """Merge the HP-filtered labor-share gap into the processed dataset."""
    labor_share = load_labor_share_gap(raw_dir, lamb=lamb)
    if labor_share.empty:
        return data

    out = data.copy()
    for col in labor_share.columns:
        out[col] = labor_share[col].reindex(out.index)
    out["labor_share_gap_HP_prev"] = out["labor_share_gap_HP"].shift(1)
    return out


def build_processed_dataset(raw_dir: str | Path | None = None, out_path: str | Path | None = None) -> pd.DataFrame:
    """Build the processed model-ready dataset without overwriting raw data."""
    try:
        from nkpc_hsa.data.func_data_build import build_dataset
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Legacy data builder is unavailable.") from exc

    raw = Path(raw_dir) if raw_dir is not None else project_path("data", "raw")
    if not (raw / "inflation").exists() and raw == project_path("data", "raw"):
        legacy_raw = project_path("data")
        if (legacy_raw / "inflation").exists():
            raw = legacy_raw
    out = Path(out_path) if out_path is not None else project_path("data", "processed", "model_ready.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    data = build_dataset(raw)
    data = add_hp_output_gap(data)
    data = add_labor_share_gap(data, raw)
    data.to_csv(out, index=False)
    return data
