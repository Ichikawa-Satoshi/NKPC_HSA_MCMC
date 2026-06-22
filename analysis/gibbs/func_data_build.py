# data_builders.py
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

# --- optional scipy ---
_HAS_SCIPY = False
try:
    from scipy.interpolate import PchipInterpolator
    _HAS_SCIPY = True
except Exception:
    PchipInterpolator = None  # type: ignore


# ---- helper functions ----
def to_datetime(s, fmt: str | None = None):
    return pd.to_datetime(s, format=fmt) if fmt else pd.to_datetime(s)


def yoy_pct(series_q: pd.Series) -> pd.Series:
    # Exact YoY percent change: 100*(x/lag4 - 1)
    return 100 * (series_q / series_q.shift(4) - 1)


def log_yoy(series_q: pd.Series) -> pd.Series:
    # Log-difference YoY ×100
    return 100 * (np.log(series_q) - np.log(series_q).shift(4))


def resample_quarterly_mean(df: pd.DataFrame, date_col: str, value_cols: list[str]) -> pd.DataFrame:
    x = df.copy()
    x[date_col] = to_datetime(x[date_col])
    x = x.set_index(date_col).sort_index()
    q = x.resample("QE").mean()  # quarterly mean
    q.index.freq = "QE"
    return q[value_cols]


def annual_to_quarterly_pchip(df: pd.DataFrame, date_col: str, value_col: str) -> pd.DataFrame:
    """
    Convert annual data to quarterly using PCHIP interpolation.
    Falls back to time/polynomial interpolation if scipy unavailable.
    """
    x = df[[date_col, value_col]].copy()
    x[date_col] = to_datetime(x[date_col])
    x = x.set_index(date_col).sort_index()

    # annual start frequency
    a = x.asfreq("YS")

    # target quarterly index (quarter-end)
    q_index = pd.date_range(a.index.min(), a.index.max(), freq="QE")
    merged = a.reindex(a.index.union(q_index)).sort_index()

    if _HAS_SCIPY:
        xi = merged.index.view("i8")
        mask = merged[value_col].notna().values
        xx = xi[mask].astype(float)
        yy = merged.loc[mask, value_col].astype(float).values
        if len(xx) >= 2:
            f = PchipInterpolator(xx, yy)  # type: ignore[misc]
            merged[value_col] = pd.Series(f(xi.astype(float)), index=merged.index)
        else:
            merged[value_col] = merged[value_col].interpolate(method="time")
    else:
        try:
            merged[value_col] = merged[value_col].interpolate(method="time")
            merged[value_col] = merged[value_col].interpolate(method="polynomial", order=3)
        except Exception:
            merged[value_col] = merged[value_col].interpolate()

    out = merged.reindex(q_index)
    out.index.freq = "QE"
    return out


def annual_firm_average_to_quarterly(
    df: pd.DataFrame,
    year_col: str,
    value_col: str,
    output_col: str,
) -> pd.DataFrame:
    """
    Aggregate firm-year observations to a simple annual country-level average,
    then interpolate to quarter-end frequency.
    """
    annual = (
        df[[year_col, value_col]]
        .dropna()
        .groupby(year_col, as_index=False)[value_col]
        .mean()
        .rename(columns={value_col: output_col})
    )
    annual["DATE"] = pd.to_datetime(annual[year_col].astype(int).astype(str) + "-01-01")
    return annual_to_quarterly_pchip(annual[["DATE", output_col]], "DATE", output_col)


def _detect_date_col(df: pd.DataFrame) -> str:
    if "DATE" in df.columns:
        return "DATE"
    if "observation_date" in df.columns:
        return "observation_date"
    raise ValueError("CSV must have DATE or observation_date column.")


def as_qe_midnight(df: pd.DataFrame) -> pd.DataFrame:
    """Force index to quarter-end at 00:00:00 (no nanoseconds)."""
    out = df.copy()
    out.index = pd.to_datetime(out.index)
    out.index = pd.PeriodIndex(out.index, freq="Q").to_timestamp(how="end").normalize()
    out.index.name = "DATE"
    out.index.freq = None
    return out


def _set_quarter_end_index(df: pd.DataFrame, date_col: str = "DATE") -> pd.DataFrame:
    out = df.set_index(date_col).sort_index()
    out.index = out.index.to_period("Q").to_timestamp(how="end")
    out.index.freq = "QE"
    return out


def load_spf_expectations(base: Path) -> pd.DataFrame:
    spf = pd.read_excel(base / "inflation" / "SPF_Inflation_Expectation.xlsx")
    q_month = (spf["QUARTER"] * 3)
    spf["DATE"] = pd.to_datetime(spf["YEAR"].astype(str) + "-" + q_month.astype(str) + "-01") + pd.offsets.MonthEnd(0)
    spf["Epi_spf_gdp"] = spf["INFPGDP1YR"]
    spf["Epi_spf_cpi"] = spf["INFCPI1YR"]
    return _set_quarter_end_index(spf[["DATE", "Epi_spf_gdp", "Epi_spf_cpi"]])


def load_monthly_inflation_series(
    base: Path,
    filename: str,
    raw_col: str,
    output_col: str,
    transform: str,
) -> pd.DataFrame:
    df = pd.read_csv(base / "inflation" / filename)
    date_col = _detect_date_col(df)
    quarterly = resample_quarterly_mean(df, date_col, [raw_col])

    if transform == "pct_yoy":
        quarterly[output_col] = yoy_pct(quarterly[raw_col])
    elif transform == "log_yoy":
        quarterly[output_col] = log_yoy(quarterly[raw_col])
    else:
        raise ValueError(f"Unknown transform: {transform}")

    return quarterly[[output_col]]


def load_cleveland_fed_expectations(base: Path) -> pd.DataFrame:
    epi = pd.read_csv(base / "inflation" / "Clev_Fed_Inflation_Expectation.csv")
    epi["DATE"] = to_datetime(epi["Date"], fmt="%Y-%m-%d")
    epi["Epi"] = epi[" Epi"] * 100
    tt_epi_m = pd.DataFrame({"DATE": epi["DATE"], "Epi": epi["Epi"]}).set_index("DATE").sort_index()
    return tt_epi_m.resample("QE").mean()


def load_competition_series(base: Path) -> list[pd.DataFrame]:
    bn_n = pd.read_csv(base / "competition" / "BN_N_Gustavo_26.csv")
    level = bn_n.copy()
    level["N_Gustavo"] = level["original_series"]
    level["DATE"] = pd.to_datetime(pd.to_datetime(level["date"]).dt.year.astype(str) + "-01-01")
    tt_n_gustavo_q = annual_to_quarterly_pchip(level[["DATE", "N_Gustavo"]], "DATE", "N_Gustavo")

    cycle_trend = bn_n.copy()
    cycle_trend["N_Gustavo_BN_cycle"] = cycle_trend["cycle"]
    cycle_trend["N_Gustavo_BN_trend"] = cycle_trend["trend"]
    cycle_trend["DATE"] = pd.to_datetime(pd.to_datetime(cycle_trend["date"]).dt.year.astype(str) + "-01-01")
    cycle_trend = cycle_trend[["DATE", "N_Gustavo_BN_cycle", "N_Gustavo_BN_trend"]].dropna()
    tt_n_gustavo_cycle_q = annual_to_quarterly_pchip(cycle_trend, "DATE", "N_Gustavo_BN_cycle")
    tt_n_gustavo_trend_q = annual_to_quarterly_pchip(cycle_trend, "DATE", "N_Gustavo_BN_trend")

    tnic = pd.read_csv(base / "competition" / "BN_N_TNIC_26.csv")
    level = tnic.copy()
    level["N_TNIC"] = level["original_series"]
    level["DATE"] = pd.to_datetime(pd.to_datetime(level["date"]).dt.year.astype(str) + "-01-01")
    tt_n_tnic_q = annual_to_quarterly_pchip(level[["DATE", "N_TNIC"]], "DATE", "N_TNIC")

    cycle_trend = tnic.copy()
    cycle_trend["N_TNIC_BN_cycle"] = cycle_trend["cycle"]
    cycle_trend["N_TNIC_BN_trend"] = cycle_trend["trend"]
    cycle_trend["DATE"] = pd.to_datetime(pd.to_datetime(cycle_trend["date"]).dt.year.astype(str) + "-01-01")
    cycle_trend = cycle_trend[["DATE", "N_TNIC_BN_cycle", "N_TNIC_BN_trend"]].dropna()
    tt_n_tnic_cycle_q = annual_to_quarterly_pchip(cycle_trend, "DATE", "N_TNIC_BN_cycle")
    tt_n_tnic_trend_q = annual_to_quarterly_pchip(cycle_trend, "DATE", "N_TNIC_BN_trend")
    return [tt_n_gustavo_q, tt_n_gustavo_cycle_q, tt_n_gustavo_trend_q, tt_n_tnic_q, tt_n_tnic_cycle_q, tt_n_tnic_trend_q]


def load_markup_series(base: Path) -> list[pd.DataFrame]:
    mk = pd.read_excel(base / "markup" / "nekarda_ramey_markups.xlsx")
    mk["DATE"] = to_datetime(mk["qdate"])
    mk["markup"] = mk["mu_bus"]
    tt_mk = _set_quarter_end_index(mk[["DATE", "markup"]].dropna())

    mk_bn = pd.read_csv(base / "markup" / "BN_markup_inv.csv")
    mk_bn["markup_BN_inv"] = mk_bn["cycle"]
    mk_bn["markup_inv"] = mk_bn["original_series"]
    mk_bn["DATE"] = to_datetime(mk_bn["date"])
    tt_mk_bn = _set_quarter_end_index(mk_bn[["DATE", "markup_BN_inv", "markup_inv"]].dropna())

    return [tt_mk, tt_mk_bn]


def load_labor_market_series(base: Path) -> pd.DataFrame:
    nairu = pd.read_csv(base / "unemp_gap" / "NROU.csv")
    unemp = pd.read_csv(base / "unemp_gap" / "UNRATENSA.csv")
    dc1 = _detect_date_col(nairu)
    dc2 = _detect_date_col(unemp)

    n = nairu[[dc1, "NROU"]].rename(columns={dc1: "DATE"})
    u = unemp[[dc2, "UNRATENSA"]].rename(columns={dc2: "DATE"})
    n["DATE"] = pd.to_datetime(n["DATE"], errors="coerce")
    u["DATE"] = pd.to_datetime(u["DATE"], errors="coerce")

    tt_gap = n.merge(u, on="DATE", how="outer").set_index("DATE").sort_index().resample("QE").mean()
    tt_gap.index.freq = "QE"
    tt_gap["unemp_gap"] = tt_gap["NROU"] - tt_gap["UNRATENSA"]
    return tt_gap[["unemp_gap"]].dropna()


def load_output_gap_series(base: Path) -> pd.DataFrame:
    out = pd.read_csv(base / "output_gap" / "BN_filter_GDPC1_quaterly.csv")
    out["output_BN"] = out["GDPC1_transformed_series"]
    out["output_gap_BN"] = out["cycle"]
    out["output"] = np.log(out["GDPC1_original_series"] * 0.01)
    out["DATE"] = to_datetime(out["date"])
    out["output_trend_BN"] = out["output_BN"] - out["output_gap_BN"]
    return _set_quarter_end_index(out[["DATE", "output_BN", "output_gap_BN", "output_trend_BN", "output"]].dropna())


def load_oil_series(base: Path) -> pd.DataFrame:
    oil = pd.read_csv(base / "others" / "WTISPLC_CPIAUCSL.csv")
    dc = _detect_date_col(oil)
    tt_oil = oil[[dc, "WTISPLC_CPIAUCSL"]].copy()
    tt_oil[dc] = to_datetime(tt_oil[dc])
    tt_oil = tt_oil.set_index(dc).sort_index().resample("QE").mean()
    tt_oil.index.freq = "QE"
    tt_oil["log_oil"] = np.log(tt_oil["WTISPLC_CPIAUCSL"])
    tt_oil["oil"] = tt_oil["log_oil"] - tt_oil["log_oil"].shift(4)
    return tt_oil[["oil"]]


def add_lagged_columns(data: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = data.copy()
    for col in columns:
        if col in out.columns:
            out[f"{col}_prev"] = out[col].shift(1)
    return out


def load_all_series(base: Path) -> list[pd.DataFrame]:
    inflation_series = [
        load_monthly_inflation_series(base, "CPIAUCSL.csv", "CPIAUCSL", "pi_cpi", "pct_yoy"),
        load_cleveland_fed_expectations(base),
        load_spf_expectations(base),
        load_monthly_inflation_series(base, "CPILFESL.csv", "CPILFESL", "pi_cpi_core", "log_yoy"),
        load_monthly_inflation_series(base, "PCEPILFE.csv", "PCEPILFE", "pi_pce_core", "log_yoy"),
        load_monthly_inflation_series(base, "PCEPI.csv", "PCEPI", "pi_pce", "log_yoy"),
        load_monthly_inflation_series(base, "PPIACO.csv", "PPIACO", "pi_ppi", "pct_yoy"),
    ]
    competition_series = load_competition_series(base)
    real_activity_series = [load_output_gap_series(base), load_labor_market_series(base), load_oil_series(base)]
    markup_series = load_markup_series(base)
    return inflation_series + competition_series + real_activity_series + markup_series


# ---- main builder ----
def build_dataset(base_dir: str | Path = "../data") -> pd.DataFrame:
    """
    Build merged quarterly dataset.
    Parameters
    ----------
    base_dir : str | Path
        Path to the data root directory (the one containing inflation/, competition/, markup/, ...)

    Returns
    -------
    pd.DataFrame
        Quarterly merged dataset with DATE column and lagged inflation variables.
    """
    base = Path(base_dir)

    series_list = [as_qe_midnight(df) for df in load_all_series(base)]
    data = pd.concat(series_list, axis=1).sort_index()
    data.index = data.index.to_period("Q").to_timestamp(how="end")
    data.index.freq = "QE"

    data = add_lagged_columns(
        data,
        [
            "pi_cpi", "pi_ppi", "pi_cpi_core", "pi_pce", "pi_pce_core", "unemp_gap",
            "markup_BN_inv", "markup_inv", "output_gap_BN", "HHI_TNIC", "HHI_TNIC_inv"
        ],
    )

    data["DATE"] = pd.to_datetime(data.index)
    return data
