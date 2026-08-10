"""Official QCEW quarterly establishment-count ingestion."""

from __future__ import annotations

from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd


def _clean_columns(columns) -> list[str]:
    return [str(value).strip().strip('"').lower() for value in columns]


def _total_industry_mask(frame: pd.DataFrame) -> pd.Series:
    code = frame["industry_code"].astype(str).str.strip().str.strip('"')
    compact = code.str.replace("-", "", regex=False).str.lstrip("0")
    mask = compact.isin({"", "10"}) | code.isin({"10", "10----", "000000"})
    if "industry_title" in frame:
        title = frame["industry_title"].astype(str).str.lower()
        mask |= title.str.contains("total, all industries", regex=False)
        mask |= title.str.fullmatch(r"\s*total\s*", na=False)
    return mask


def load_qcew_national_private_establishments(
    source_dir: str | Path,
    *,
    start_year: int = 1982,
    end_year: int = 2012,
    chunksize: int = 200_000,
) -> pd.DataFrame:
    """Read national, private, all-industry quarterly establishment totals.

    The loader accepts the official BLS yearly ZIP/CSV layouts and selects the
    published ``qtrly_estabs`` field. It never reconstructs a level from BED
    births/deaths and it never treats establishments as firms.
    """
    base = Path(source_dir)
    sources = sorted([*base.glob("*.zip"), *base.glob("*.csv")])
    if not sources:
        raise FileNotFoundError(f"No QCEW ZIP or CSV files found in {base}")
    parts: list[pd.DataFrame] = []

    def consume(reader, source: Path) -> None:
        for chunk in reader:
            chunk.columns = _clean_columns(chunk.columns)
            if "qtrly_estabs" not in chunk and "qtrly_estabs_count" in chunk:
                chunk = chunk.rename(columns={"qtrly_estabs_count": "qtrly_estabs"})
            required = {"area_fips", "own_code", "industry_code", "year", "qtr", "qtrly_estabs"}
            if not required.issubset(chunk.columns):
                continue
            area = chunk["area_fips"].astype(str).str.strip().str.strip('"').str.upper()
            own = pd.to_numeric(chunk["own_code"], errors="coerce")
            year = pd.to_numeric(chunk["year"], errors="coerce")
            quarter = pd.to_numeric(chunk["qtr"], errors="coerce")
            mask = (
                area.eq("US000")
                & own.eq(5)
                & year.between(start_year, end_year)
                & quarter.between(1, 4)
                & _total_industry_mask(chunk)
            )
            if mask.any():
                selected = chunk.loc[mask, ["year", "qtr", "qtrly_estabs"]].copy()
                selected["source_file"] = source.name
                parts.append(selected)

    for source in sources:
        if source.suffix.lower() == ".csv":
            consume(pd.read_csv(source, dtype=str, chunksize=chunksize, low_memory=False), source)
            continue
        with ZipFile(source) as archive:
            for member in archive.namelist():
                if member.lower().endswith(".csv") and "title" not in member.lower():
                    with archive.open(member) as handle:
                        consume(pd.read_csv(handle, dtype=str, chunksize=chunksize, low_memory=False), source)

    if not parts:
        raise ValueError(
            "QCEW sources contained no national/private/all-industry qtrly_estabs records; "
            "verify that the official quarterly totals/by-industry files were downloaded."
        )
    out = pd.concat(parts, ignore_index=True)
    out["year"] = pd.to_numeric(out["year"], errors="raise").astype(int)
    out["qtr"] = pd.to_numeric(out["qtr"], errors="raise").astype(int)
    out["qcew_establishments"] = pd.to_numeric(
        out["qtrly_estabs"].astype(str).str.replace(",", "", regex=False), errors="raise"
    )
    if (out["qcew_establishments"] <= 0).any() or (~np.isfinite(out["qcew_establishments"])).any():
        raise ValueError("QCEW establishment counts must be finite and positive.")
    out["quarter"] = pd.PeriodIndex(
        out["year"].astype(str) + "Q" + out["qtr"].astype(str), freq="Q"
    )
    duplicate = out.duplicated("quarter", keep=False)
    if duplicate.any():
        # Some archives contain equivalent duplicate records. They are harmless
        # only when their published counts agree exactly.
        spread = out.loc[duplicate].groupby("quarter")["qcew_establishments"].nunique()
        if (spread > 1).any():
            raise ValueError(f"Conflicting QCEW totals for quarters: {spread[spread > 1].index.tolist()}")
        out = out.drop_duplicates("quarter", keep="last")
    out = out.sort_values("quarter").reset_index(drop=True)
    expected = pd.period_range(f"{start_year}Q1", f"{end_year}Q4", freq="Q")
    missing = expected.difference(pd.PeriodIndex(out["quarter"], freq="Q"))
    if len(missing):
        raise ValueError(f"QCEW series is incomplete; missing {len(missing)} quarters, beginning {missing[:5].tolist()}")
    return out[["quarter", "qcew_establishments", "source_file"]]


def merge_qcew_establishments(model_data: pd.DataFrame, qcew: pd.DataFrame) -> pd.DataFrame:
    """Merge the quarterly QCEW level without modifying baseline columns."""
    if not {"quarter", "qcew_establishments"}.issubset(qcew.columns):
        raise ValueError("qcew must contain quarter and qcew_establishments.")
    out = model_data.copy()
    if not isinstance(out.index, pd.DatetimeIndex):
        if "DATE" not in out:
            raise ValueError("model_data needs a DatetimeIndex or DATE column.")
        out = out.set_index(pd.to_datetime(out.pop("DATE")))
    series = pd.Series(
        qcew["qcew_establishments"].to_numpy(dtype=float),
        index=pd.PeriodIndex(qcew["quarter"], freq="Q"),
    )
    out["qcew_establishments"] = series.reindex(out.index.to_period("Q")).to_numpy(dtype=float)
    return out
