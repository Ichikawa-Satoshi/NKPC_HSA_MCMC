from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator

from nkpc_hsa.paths import project_path

DEFAULT_COMPETITION_MEASUREMENT: dict[str, str] = {
    "frequency": "quarterly_interpolated",
    "annual_timing": "q4",
}
VALID_COMPETITION_FREQUENCIES = {"quarterly_interpolated", "annual_q4"}


@dataclass(frozen=True)
class CompetitionObservation:
    N_obs: np.ndarray
    quarterly_index: pd.PeriodIndex
    frequency: str
    annual_timing: str
    finite_count: int
    missing_count: int
    first_finite: str
    last_finite: str
    observed_quarters: list[str]
    interpolation_method: str


def normalize_competition_measurement(spec: Mapping[str, Any] | None = None) -> dict[str, str]:
    out = dict(DEFAULT_COMPETITION_MEASUREMENT)
    out.update({k: str(v) for k, v in dict(spec or {}).items() if v is not None})
    frequency = out["frequency"]
    if frequency not in VALID_COMPETITION_FREQUENCIES:
        raise ValueError(
            "competition_measurement.frequency must be one of "
            f"{sorted(VALID_COMPETITION_FREQUENCIES)}, got {frequency!r}."
        )
    annual_timing = out.get("annual_timing", "q4").lower()
    if annual_timing != "q4":
        raise ValueError("Only annual_timing='q4' is currently supported.")
    out["annual_timing"] = annual_timing
    return out


def to_quarter_period_index(index: pd.Index | np.ndarray | list[Any]) -> pd.PeriodIndex:
    if isinstance(index, pd.PeriodIndex):
        return index.asfreq("Q")
    if isinstance(index, pd.DatetimeIndex):
        return index.to_period("Q")
    idx = pd.Index(index)
    if isinstance(idx, pd.PeriodIndex):
        return idx.asfreq("Q")
    if np.issubdtype(idx.dtype, np.datetime64):
        return pd.DatetimeIndex(idx).to_period("Q")
    try:
        return pd.to_datetime(idx).to_period("Q")
    except Exception as exc:
        raise ValueError("quarterly_index must be convertible to quarterly periods.") from exc


def _annual_series_to_year_index(series: pd.Series) -> pd.Series:
    s = pd.Series(series, copy=True).astype(float).dropna()
    if s.empty:
        raise ValueError("annual_N_transformed must contain at least one finite value.")
    idx = s.index
    if isinstance(idx, pd.PeriodIndex):
        years = idx.asfreq("A").year
    elif isinstance(idx, pd.DatetimeIndex):
        years = idx.year
    else:
        raw = pd.Index(idx)
        if np.issubdtype(raw.dtype, np.integer):
            years = raw.astype(int).to_numpy()
        else:
            try:
                years = raw.astype(int).to_numpy()
            except (TypeError, ValueError):
                years = pd.to_datetime(raw).year
    out = pd.Series(s.to_numpy(dtype=float), index=pd.Index(years, name="year"), name=s.name)
    return out.groupby(level=0).last().sort_index()


def _metadata(N_obs: np.ndarray, q_index: pd.PeriodIndex, frequency: str, annual_timing: str) -> CompetitionObservation:
    mask = np.isfinite(N_obs)
    observed = [str(q) for q in q_index[mask]]
    return CompetitionObservation(
        N_obs=np.asarray(N_obs, dtype=float),
        quarterly_index=q_index,
        frequency=frequency,
        annual_timing=annual_timing,
        finite_count=int(mask.sum()),
        missing_count=int(mask.size - mask.sum()),
        first_finite=observed[0] if observed else "",
        last_finite=observed[-1] if observed else "",
        observed_quarters=observed,
        interpolation_method="PCHIP" if frequency == "quarterly_interpolated" else "none",
    )


def build_competition_observation(
    annual_N_transformed: pd.Series | Mapping[Any, float] | np.ndarray,
    quarterly_index: pd.Index | np.ndarray | list[Any],
    *,
    frequency: str = "quarterly_interpolated",
    annual_timing: str = "q4",
    interpolated_N: np.ndarray | pd.Series | None = None,
) -> CompetitionObservation:
    """Build the competition observation vector aligned to a quarterly sample."""
    spec = normalize_competition_measurement({"frequency": frequency, "annual_timing": annual_timing})
    frequency = spec["frequency"]
    annual_timing = spec["annual_timing"]
    q_index = to_quarter_period_index(quarterly_index)

    if frequency == "quarterly_interpolated" and interpolated_N is not None:
        arr = np.asarray(interpolated_N, dtype=float).reshape(-1)
        if arr.size != len(q_index):
            raise ValueError("interpolated_N must have the same length as quarterly_index.")
        if np.any(~np.isfinite(arr)):
            raise ValueError("quarterly_interpolated N_obs must be finite for every quarter.")
        return _metadata(arr, q_index, frequency, annual_timing)

    if not isinstance(annual_N_transformed, pd.Series):
        annual_N_transformed = pd.Series(annual_N_transformed)
    annual = _annual_series_to_year_index(annual_N_transformed)

    if frequency == "annual_q4":
        out = np.full(len(q_index), np.nan, dtype=float)
        years = annual.index.to_numpy(dtype=int)
        values = annual.to_numpy(dtype=float)
        by_year = dict(zip(years, values))
        for i, period in enumerate(q_index):
            if int(period.quarter) == 4 and int(period.year) in by_year:
                out[i] = float(by_year[int(period.year)])
        if not np.isfinite(out).any():
            raise ValueError("annual_q4 produced no finite Q4 competition observations.")
        return _metadata(out, q_index, frequency, annual_timing)

    # Standalone PCHIP path for tests and report metadata. The production
    # wrapper passes the legacy quarterly-interpolated series through
    # interpolated_N to preserve existing behavior exactly.
    x = annual.index.to_numpy(dtype=float)
    y = annual.to_numpy(dtype=float)
    target = np.asarray(q_index.year, dtype=float) + (np.asarray(q_index.quarter, dtype=float) - 1.0) / 4.0
    if y.size >= 2:
        interp = PchipInterpolator(x, y, extrapolate=False)
        out = np.asarray(interp(target), dtype=float)
    else:
        out = np.full(len(q_index), y[0], dtype=float)
    if np.any(~np.isfinite(out)):
        valid = np.isfinite(out)
        if not valid.any():
            raise ValueError("quarterly_interpolated PCHIP produced no finite values.")
        out = np.interp(np.arange(out.size), np.flatnonzero(valid), out[valid])
    return _metadata(out, q_index, frequency, annual_timing)


def pchip_interpolate_annual_q4(
    annual_N_transformed: pd.Series | Mapping[Any, float] | np.ndarray,
    quarterly_index: pd.Index | np.ndarray | list[Any],
    *,
    annual_timing: str = "q4",
) -> np.ndarray:
    """PCHIP comparison line anchored so annual observations occur in Q4."""
    spec = normalize_competition_measurement({"frequency": "annual_q4", "annual_timing": annual_timing})
    q_index = to_quarter_period_index(quarterly_index)
    if not isinstance(annual_N_transformed, pd.Series):
        annual_N_transformed = pd.Series(annual_N_transformed)
    annual = _annual_series_to_year_index(annual_N_transformed)
    x = annual.index.to_numpy(dtype=float) + 0.75
    y = annual.to_numpy(dtype=float)
    target = np.asarray(q_index.year, dtype=float) + (np.asarray(q_index.quarter, dtype=float) - 1.0) / 4.0
    if y.size >= 2:
        interp = PchipInterpolator(x, y, extrapolate=True)
        out = np.asarray(interp(target), dtype=float)
    else:
        out = np.full(len(q_index), y[0], dtype=float)
    if np.any(~np.isfinite(out)):
        valid = np.isfinite(out)
        if not valid.any():
            raise ValueError("Q4-anchored PCHIP comparison produced no finite values.")
        out = np.interp(np.arange(out.size), np.flatnonzero(valid), out[valid])
    # Keep the linter aware that annual_timing validation is intentional.
    _ = spec
    return out


def competition_observation_from_array(
    N_obs: np.ndarray | pd.Series,
    quarterly_index: pd.Index | np.ndarray | list[Any],
    *,
    frequency: str,
    annual_timing: str = "q4",
) -> CompetitionObservation:
    spec = normalize_competition_measurement({"frequency": frequency, "annual_timing": annual_timing})
    q_index = to_quarter_period_index(quarterly_index)
    arr = np.asarray(N_obs, dtype=float).reshape(-1)
    if arr.size != len(q_index):
        raise ValueError("N_obs must have the same length as quarterly_index.")
    if spec["frequency"] == "quarterly_interpolated" and np.any(~np.isfinite(arr)):
        raise ValueError("quarterly_interpolated N_obs must be finite for every quarter.")
    return _metadata(arr, q_index, spec["frequency"], spec["annual_timing"])


def load_raw_annual_competition_series(
    n_col: str = "N_Gustavo",
    *,
    raw_dir: str | Path | None = None,
) -> pd.Series:
    """Load raw annual competition levels before quarterly interpolation."""
    base = Path(raw_dir) if raw_dir is not None else project_path("data", "raw")
    files = {
        "N_Gustavo": "BN_N_Gustavo_26.csv",
        "N_TNIC": "BN_N_TNIC_26.csv",
    }
    if n_col not in files:
        raise ValueError(f"Raw annual competition loader does not know series {n_col!r}.")
    path = base / "competition" / files[n_col]
    raw = pd.read_csv(path)
    if "date" not in raw.columns or "original_series" not in raw.columns:
        raise ValueError(f"{path} must contain 'date' and 'original_series'.")
    years = pd.to_datetime(raw["date"], errors="coerce").dt.year
    values = pd.to_numeric(raw["original_series"], errors="coerce")
    out = pd.Series(values.to_numpy(dtype=float), index=pd.Index(years, name="year"), name=n_col)
    return out.dropna().groupby(level=0).last().sort_index()
