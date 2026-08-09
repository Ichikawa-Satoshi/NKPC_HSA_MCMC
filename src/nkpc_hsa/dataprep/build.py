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


def _numeric_count(series: pd.Series) -> pd.Series:
    """Parse count columns that use thousands separators and Unicode minus."""
    return pd.to_numeric(
        series.astype(str).str.replace(",", "", regex=False).str.replace("−", "-", regex=False),
        errors="coerce",
    )


def load_quarterly_establishment_stock(
    raw_dir: str | Path,
    *,
    anchor_year: int = 1993,
) -> pd.DataFrame:
    """Reconstruct a quarterly establishment stock from BED births and deaths.

    BED reports seasonally adjusted quarterly establishment births and deaths in
    thousands, not the total establishment stock.  The annual BDS ``ESTAB`` count
    supplies the initial level.  Treating that annual count as the Q1 benchmark,
    the end-of-quarter stock evolves as

    ``E_t = E_{t-1} + 1000 * (births_t - deaths_t)``.

    The first quarter for which both flows are finite is 1993Q2.  The resulting
    level is an experimental quarterly proxy: BDS and BED are separate programs,
    so later BDS annual levels need not equal the cumulated BED flow exactly.  We
    keep births, deaths, and net entry alongside the stock so that discrepancy is
    visible rather than silently benchmarked away.
    """
    competition_dir = Path(raw_dir) / "competition"
    bds_path = competition_dir / "bds" / "BDSTIMESERIES_BDSGEO.csv"
    bed_dir = competition_dir / "bed"
    births_path = bed_dir / "BLS-bd-BDS0000000000000000120007LQ5.csv"
    deaths_path = bed_dir / "BLS-bd-BDS0000000000000000120008LQ5.csv"
    for path in (bds_path, births_path, deaths_path):
        if not path.exists():
            raise FileNotFoundError(f"Missing establishment source: {path}")

    annual = pd.read_csv(bds_path)
    year_col = "Year (time)"
    establishment_col = "Number of establishments (ESTAB)"
    if year_col not in annual or establishment_col not in annual:
        raise KeyError(f"BDS source must contain {year_col!r} and {establishment_col!r}.")
    annual[year_col] = pd.to_numeric(annual[year_col], errors="coerce")
    annual[establishment_col] = _numeric_count(annual[establishment_col])
    anchor_values = annual.loc[annual[year_col].eq(anchor_year), establishment_col].dropna()
    if len(anchor_values) != 1:
        raise ValueError(f"Expected one finite BDS ESTAB anchor for {anchor_year}, got {len(anchor_values)}.")
    anchor = float(anchor_values.iloc[0])

    def _load_bed(path: Path, name: str) -> pd.DataFrame:
        raw = pd.read_csv(path)
        if raw.shape[1] != 2:
            raise ValueError(f"Expected period plus one value column in {path}.")
        out = raw.copy()
        out.columns = ["period", name]
        out["period"] = pd.PeriodIndex(out["period"].astype(str), freq="Q")
        out[name] = pd.to_numeric(out[name], errors="coerce") * 1000.0
        return out.set_index("period")

    flows = _load_bed(births_path, "establishment_births").join(
        _load_bed(deaths_path, "establishment_deaths"), how="inner"
    )
    flows = flows.dropna().sort_index()
    if flows.empty:
        raise ValueError("BED births and deaths have no overlapping finite quarters.")
    first_expected = pd.Period(f"{anchor_year}Q2", freq="Q")
    if flows.index[0] != first_expected:
        raise ValueError(f"Expected the first complete BED flow at {first_expected}, got {flows.index[0]}.")

    flows["establishment_net_entry"] = (
        flows["establishment_births"] - flows["establishment_deaths"]
    )
    flows["establishment_stock"] = anchor + flows["establishment_net_entry"].cumsum()
    flows.index = flows.index.to_timestamp(how="end")
    flows.index.name = "DATE"
    return flows


def add_quarterly_establishment_stock(data: pd.DataFrame, raw_dir: str | Path) -> pd.DataFrame:
    """Merge the reconstructed BED establishment stock and its component flows."""
    establishment = load_quarterly_establishment_stock(raw_dir)
    out = data.copy()
    for col in establishment.columns:
        out[col] = establishment[col].reindex(out.index)
    return out


def build_processed_dataset(raw_dir: str | Path | None = None, out_path: str | Path | None = None) -> pd.DataFrame:
    """Build the processed model-ready dataset without overwriting raw data."""
    try:
        from nkpc_hsa.dataprep.func_data_build import build_dataset
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
    data = add_quarterly_establishment_stock(data, raw)
    data.to_csv(out, index=False)
    return data
