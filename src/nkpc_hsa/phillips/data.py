from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from nkpc_hsa.config import load_yaml
from nkpc_hsa.dataprep.competition import load_raw_annual_competition_series
from nkpc_hsa.dataprep.func_data_build import (
    load_spf_quarter_ahead_expectations,
    resample_quarterly_mean,
)
from nkpc_hsa.paths import data_root, project_path


PRICE_SOURCES = {
    "ppi": ("PPIACO.csv", "PPIACO"),
    "cpi": ("CPIAUCSL.csv", "CPIAUCSL"),
    "core_cpi": ("CPILFESL.csv", "CPILFESL"),
    "pce": ("PCEPI.csv", "PCEPI"),
    "core_pce": ("PCEPILFE.csv", "PCEPILFE"),
}


def _cells() -> tuple[dict[str, str], ...]:
    prices = ("ppi", "cpi", "core_cpi")
    activities = ("inverse_markup", "bn_output_gap", "negative_unemployment_gap")
    return tuple(
        {
            "cell": str(i + 1),
            "inflation": price,
            "activity": activity,
            "tier": "theory-near" if i == 0 else ("mechanism" if i in {1, 2, 3} else "transportability"),
        }
        for i, (price, activity) in enumerate((p, a) for p in prices for a in activities)
    )


CELL_SPECS = _cells()


@dataclass(frozen=True)
class DesignData:
    quarterly: pd.DataFrame
    annual_observation: np.ndarray
    quarterly_indicator: np.ndarray
    annual_values: pd.Series
    q_scale: float
    e_scale: float
    config: Mapping[str, Any]
    source_paths: tuple[str, ...]

    @property
    def periods(self) -> pd.PeriodIndex:
        return self.quarterly.index


def robust_scale(values: np.ndarray | pd.Series) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        raise ValueError("Cannot scale an empty series.")
    scale = float(np.subtract(*np.quantile(finite, [0.75, 0.25])) / 1.349)
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("The pre-estimation IQR variation gate failed.")
    return scale


def transform_coordinate(values: pd.Series, transform: str) -> pd.Series:
    raw = pd.to_numeric(values, errors="coerce")
    if (raw.dropna() <= 0.0).any():
        raise ValueError("Business-demography levels must be positive before the log transform.")
    logged = 100.0 * np.log(raw)
    if transform == "log100_centered10":
        return (logged - logged.mean()) / 10.0
    if transform == "log100_centered":
        return logged - logged.mean()
    raise ValueError(f"Unsupported competition transform: {transform}")


def _quarterly_price_inflation(raw_dir: Path, price: str) -> pd.Series:
    filename, raw_column = PRICE_SOURCES[price]
    path = raw_dir / "inflation" / filename
    raw = pd.read_csv(path)
    date_col = "DATE" if "DATE" in raw.columns else "observation_date"
    quarterly = resample_quarterly_mean(raw, date_col, [raw_column])[raw_column]
    # The design fixes 400 times the quarterly log difference for every index.
    inflation = 400.0 * np.log(quarterly).diff()
    inflation.index = inflation.index.to_period("Q")
    return inflation.rename(f"pi_{price}_annualized_log")


def _load_qcew(processed_dir: Path) -> tuple[pd.Series, Path]:
    path = processed_dir / "qcew_establishments.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run production/main_scripts/15_build_extension_data.py before the nine-cell design."
        )
    frame = pd.read_csv(path)
    if not {"quarter", "qcew_establishments"}.issubset(frame):
        raise ValueError(f"{path} lacks the required QCEW columns.")
    index = pd.PeriodIndex(frame["quarter"].astype(str), freq="Q")
    return pd.Series(pd.to_numeric(frame["qcew_establishments"], errors="coerce").to_numpy(), index=index), path


def _merge_processed(processed_path: Path) -> pd.DataFrame:
    frame = pd.read_csv(processed_path, parse_dates=["DATE"])
    frame.index = pd.PeriodIndex(frame["DATE"], freq="Q")
    return frame.drop(columns=["DATE"]).sort_index()


def load_design_data(
    config_path: str | Path | None = None,
    *,
    processed_path: str | Path | None = None,
    include_qcew: bool = True,
    sample_start: str | None = None,
    sample_end: str | None = None,
) -> DesignData:
    """Load and freeze the common sample for all nine pre-declared cells."""
    config_file = Path(config_path or project_path("configs", "nine_cell_design.yaml"))
    cfg = load_yaml(config_file)
    data_dir = data_root()
    raw_dir = data_dir / "raw"
    processed_dir = data_dir / "processed"
    model_ready = Path(processed_path or processed_dir / "model_ready.csv")
    processed = _merge_processed(model_ready)

    transform = str(cfg["measurement"]["transform"])
    if include_qcew:
        qcew, qcew_path = _load_qcew(processed_dir)
        e = transform_coordinate(qcew, transform).rename("e_qcew")
    else:
        qcew_path = None
        e = pd.Series(dtype=float, name="e_qcew")

    annual_raw = load_raw_annual_competition_series(
        str(cfg["measurement"]["annual_column"]), raw_dir=raw_dir
    )
    annual = transform_coordinate(annual_raw, transform)

    spf = load_spf_quarter_ahead_expectations(raw_dir)
    spf.index = spf.index.to_period("Q")

    q = processed.join(spf, how="left").join(e, how="left")
    for price in PRICE_SOURCES:
        q[f"pi_{price}"] = _quarterly_price_inflation(raw_dir, price).reindex(q.index)
        q[f"pi_{price}_lag1"] = q[f"pi_{price}"].shift(1)

    mu_ref = float(cfg["benchmark"]["mu_reference"])
    markup = pd.to_numeric(q["markup"], errors="coerce")
    q["x_inverse_markup"] = np.where(markup > 0.0, np.log(mu_ref / markup), np.nan)
    q["x_bn_output_gap"] = pd.to_numeric(q["output_gap_BN"], errors="coerce")
    q["x_hp_output_gap"] = pd.to_numeric(q["output_gap_HP"], errors="coerce")
    q["x_negative_unemployment_gap"] = pd.to_numeric(q["unemp_gap"], errors="coerce")
    q["expectation"] = pd.to_numeric(q["Epi_spf_gdp_1q_ahead_ann_log"], errors="coerce")

    start = pd.Period(str(sample_start or cfg["sample"]["start"]), freq="Q")
    end = pd.Period(str(sample_end or cfg["sample"]["end"]), freq="Q")
    q = q.loc[(q.index >= start) & (q.index <= end)].copy()
    required = [
        *(f"pi_{price}" for price in PRICE_SOURCES),
        *(f"pi_{price}_lag1" for price in PRICE_SOURCES),
        "expectation",
        "x_inverse_markup",
        "x_bn_output_gap",
        "x_hp_output_gap",
        "x_negative_unemployment_gap",
    ]
    if include_qcew:
        required.append("e_qcew")
    missing_counts = q[required].isna().sum()
    if int(missing_counts.sum()) != 0:
        rendered = ", ".join(f"{k}={int(v)}" for k, v in missing_counts.items() if v)
        raise ValueError(
            "The frozen common nine-cell sample is incomplete; no cell-specific dropna is allowed: " + rendered
        )
    expected = pd.period_range(start, end, freq="Q")
    if not q.index.equals(expected):
        raise ValueError("The frozen sample must be a contiguous quarterly index.")

    annual_obs = np.full(len(q), np.nan, dtype=float)
    annual_by_year = annual.groupby(annual.index.astype(int)).last()
    for i, period in enumerate(q.index):
        if period.quarter == 4 and period.year in annual_by_year.index:
            annual_obs[i] = float(annual_by_year.loc[period.year])
    if int(np.isfinite(annual_obs).sum()) != len(set(q.index.year)):
        raise ValueError("The annual-Q4 measurement rule did not produce one observation per sample year.")

    if not include_qcew:
        q["e_qcew"] = np.nan
    keep_columns = required + ["e_qcew", "markup"]
    source_paths = [config_file, model_ready, raw_dir]
    if qcew_path is not None:
        source_paths.insert(2, qcew_path)
    return DesignData(
        quarterly=q[keep_columns],
        annual_observation=annual_obs,
        quarterly_indicator=q["e_qcew"].to_numpy(dtype=float),
        annual_values=annual.loc[(annual.index >= start.year) & (annual.index <= end.year)],
        q_scale=robust_scale(annual.loc[(annual.index >= start.year) & (annual.index <= end.year)]),
        e_scale=robust_scale(q["e_qcew"]) if include_qcew else robust_scale(annual),
        config=cfg,
        source_paths=tuple(map(str, source_paths)),
    )
