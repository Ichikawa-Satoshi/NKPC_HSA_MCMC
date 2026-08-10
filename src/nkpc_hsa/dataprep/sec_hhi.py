from __future__ import annotations

import re
from collections.abc import Mapping
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd


# Prefer broad, consolidated revenue concepts.  The last concepts are retained as
# fallbacks for older filers, before RevenueFromContractWithCustomer became common.
REVENUE_TAG_PRIORITY: tuple[str, ...] = (
    "RevenueFromContractWithCustomerExcludingAssessedTax",
    "RevenueFromContractWithCustomerIncludingAssessedTax",
    "Revenues",
    "SalesRevenueNet",
    "RegulatedAndUnregulatedOperatingRevenue",
    "OperatingRevenues",
    "UtilityOperatingRevenue",
    "HealthCareOrganizationRevenue",
    "RealEstateRevenueNet",
    "FinancialServicesRevenue",
    "InsuranceServicesRevenue",
    "InsuranceRevenue",
    "SalesRevenueGoodsNet",
    "SalesRevenueServicesNet",
)

# SIC division H (6000-6999): finance, insurance, real estate.  "Revenue" for
# banks and insurers is not the same economic quantity as it is elsewhere, and
# those markets are large, so revenue weighting is far more exposed to them than
# firm-count weighting is.  Every aggregate below is therefore also produced on
# the non-financial subset.
FINANCIAL_SIC_MAJOR_GROUPS: tuple[str, ...] = ("6",)

# The market-level HHI is always a revenue-share HHI.  These aggregates differ
# only in how the ~250 SIC3 markets are collapsed into one number per period:
#
#   inv_hhi_firmw    1 / firm-count-weighted mean HHI   (historical headline;
#                    matches attaching a market HHI to every firm, and so is
#                    comparable to the TNIC-3 firm-level mean)
#   inv_hhi_revw     1 / revenue-weighted mean HHI
#   inv_hhi_logrevw  exp(revenue-weighted mean of log(1 / HHI))
#
# In terms of the market-level effective firm counts 1/HHI_m these are a
# harmonic, a harmonic, and a geometric mean respectively, so Jensen orders
# inv_hhi_revw <= inv_hhi_logrevw <= (revenue-weighted arithmetic mean).
#
# ``inv_hhi_logrevw`` is the aggregate the estimated equation implies.  The NKPC
# is linear in N = (100 log N_raw - c)/10, so aggregating a market-level
# kappa_m = kappa_0 + delta * N_m with expenditure weights w_m yields the
# revenue-weighted mean of log N_m, i.e. the geometric mean of 1/HHI_m.  The
# other two are retained because the firm-weighted one is what the August 2026
# extension grid already ran on and the plain revenue-weighted one is the
# obvious alternative.
SEC_INVERSE_HHI_COLUMNS: tuple[str, ...] = (
    "effective_firms",
    "inv_hhi_revw",
    "inv_hhi_logrevw",
    "inv_hhi_firmw_exfin",
    "inv_hhi_revw_exfin",
    "inv_hhi_logrevw_exfin",
)

# Source column in the HHI CSV -> column name in the model-ready frame.
DEFAULT_SEC_INVERSE_COLUMNS: dict[str, str] = {
    "effective_firms": "N_SEC_inverse_HHI",
    "inv_hhi_revw": "N_SEC_inverse_HHI_revw",
    "inv_hhi_logrevw": "N_SEC_inverse_HHI_logrevw",
    "inv_hhi_logrevw_exfin": "N_SEC_inverse_HHI_logrevw_exfin",
}

_ARCHIVE_RE = re.compile(r"^(?P<year>\d{4})q(?P<quarter>[1-4])\.zip$", re.IGNORECASE)
_SUB_COLUMNS = (
    "adsh",
    "cik",
    "name",
    "sic",
    "form",
    "period",
    "fy",
    "fp",
    "filed",
    "accepted",
)
_NUM_COLUMNS = (
    "adsh",
    "tag",
    "version",
    "ddate",
    "qtrs",
    "uom",
    "segments",
    "coreg",
    "value",
)


def discover_sec_archives(sec_dir: str | Path) -> list[Path]:
    """Return SEC quarterly ZIP files in chronological order."""
    base = Path(sec_dir)
    archives: list[tuple[int, int, Path]] = []
    for path in base.glob("*.zip"):
        match = _ARCHIVE_RE.match(path.name)
        if match:
            archives.append((int(match["year"]), int(match["quarter"]), path))
    if not archives:
        raise FileNotFoundError(f"No quarterly SEC archives found in {base}")
    return [item[2] for item in sorted(archives)]


def latest_complete_fiscal_year(sec_dir: str | Path) -> int:
    """Infer the latest fiscal year covered by the available filing archives.

    A fiscal year's December filers report in Q1 of the next calendar year, so
    the most recent archive calendar year itself is not yet a complete fiscal
    year cross-section.
    """
    latest = discover_sec_archives(sec_dir)[-1]
    match = _ARCHIVE_RE.match(latest.name)
    if match is None:  # guarded by discover_sec_archives
        raise ValueError(f"Unexpected SEC archive name: {latest.name}")
    return int(match["year"]) - 1


def _read_submissions(zf: ZipFile, archive_name: str) -> pd.DataFrame:
    with zf.open("sub.txt") as handle:
        sub = pd.read_csv(
            handle,
            sep="\t",
            usecols=list(_SUB_COLUMNS),
            dtype=str,
            low_memory=False,
        )

    sub = sub.loc[
        sub["form"].isin(("10-K", "10-K/A"))
        & sub["fp"].eq("FY")
        & sub["adsh"].notna()
        & sub["cik"].notna()
        & sub["sic"].notna()
        & sub["period"].notna()
        & sub["fy"].notna()
    ].copy()
    sub["fy"] = pd.to_numeric(sub["fy"], errors="coerce")
    sub["sic"] = pd.to_numeric(sub["sic"], errors="coerce")
    sub = sub.dropna(subset=["fy", "sic"])
    sub["fy"] = sub["fy"].astype(int)
    sub["sic"] = sub["sic"].astype(int)
    sub["period"] = sub["period"].str.replace(r"\.0$", "", regex=True)
    sub["archive"] = archive_name
    return sub


def _read_quarterly_submissions(zf: ZipFile, archive_name: str) -> pd.DataFrame:
    """Read 10-Q Q1--Q3 submissions plus 10-K fiscal-year submissions."""
    with zf.open("sub.txt") as handle:
        sub = pd.read_csv(
            handle,
            sep="\t",
            usecols=list(_SUB_COLUMNS),
            dtype=str,
            low_memory=False,
        )
    eligible = (
        (sub["form"].isin(("10-Q", "10-Q/A")) & sub["fp"].isin(("Q1", "Q2", "Q3")))
        | (sub["form"].isin(("10-K", "10-K/A")) & sub["fp"].eq("FY"))
    )
    sub = sub.loc[
        eligible
        & sub["adsh"].notna()
        & sub["cik"].notna()
        & sub["sic"].notna()
        & sub["period"].notna()
        & sub["fy"].notna()
    ].copy()
    sub["fy"] = pd.to_numeric(sub["fy"], errors="coerce")
    sub["sic"] = pd.to_numeric(sub["sic"], errors="coerce")
    sub = sub.dropna(subset=["fy", "sic"])
    sub["fy"] = sub["fy"].astype(int)
    sub["sic"] = sub["sic"].astype(int)
    sub["period"] = sub["period"].str.replace(r"\.0$", "", regex=True)
    sub["archive"] = archive_name
    return sub


def extract_revenue_facts_from_archive(
    archive: str | Path,
    *,
    chunksize: int = 500_000,
) -> pd.DataFrame:
    """Extract one preferred current-year revenue fact per annual filing."""
    archive = Path(archive)
    tag_rank = {tag: rank for rank, tag in enumerate(REVENUE_TAG_PRIORITY)}
    fact_parts: list[pd.DataFrame] = []

    with ZipFile(archive) as zf:
        sub = _read_submissions(zf, archive.name)
        if sub.empty:
            return pd.DataFrame()
        annual_adsh = set(sub["adsh"])

        with zf.open("num.txt") as handle:
            reader = pd.read_csv(
                handle,
                sep="\t",
                usecols=list(_NUM_COLUMNS),
                dtype={
                    "adsh": str,
                    "tag": str,
                    "version": str,
                    "ddate": str,
                    "qtrs": "Int64",
                    "uom": str,
                    "segments": str,
                    "coreg": str,
                    "value": float,
                },
                chunksize=chunksize,
                low_memory=False,
            )
            for chunk in reader:
                keep = chunk.loc[
                    chunk["tag"].isin(tag_rank)
                    & chunk["adsh"].isin(annual_adsh)
                    & chunk["version"].str.startswith("us-gaap/", na=False)
                    & chunk["qtrs"].eq(4)
                    & chunk["uom"].eq("USD")
                    & chunk["segments"].isna()
                    & chunk["coreg"].isna()
                    & chunk["value"].gt(0)
                ].copy()
                if not keep.empty:
                    keep["ddate"] = keep["ddate"].str.replace(r"\.0$", "", regex=True)
                    fact_parts.append(keep)

    if not fact_parts:
        return pd.DataFrame()

    facts = pd.concat(fact_parts, ignore_index=True)
    facts = facts.merge(sub, on="adsh", how="inner", validate="many_to_one")
    # A 10-K repeats prior-year facts.  Only the fact ending at the filing's
    # current fiscal period belongs to this company-year observation.
    facts = facts.loc[facts["ddate"].eq(facts["period"])].copy()
    if facts.empty:
        return facts
    facts["tag_rank"] = facts["tag"].map(tag_rank)
    facts = facts.sort_values(["adsh", "tag_rank", "tag", "value"], ascending=[True, True, True, False])
    return facts.drop_duplicates("adsh", keep="first").reset_index(drop=True)


def extract_quarterly_revenue_facts_from_archive(
    archive: str | Path,
    *,
    chunksize: int = 500_000,
) -> pd.DataFrame:
    """Extract current-quarter 10-Q facts and current-year 10-K facts."""
    archive = Path(archive)
    tag_rank = {tag: rank for rank, tag in enumerate(REVENUE_TAG_PRIORITY)}
    fact_parts: list[pd.DataFrame] = []

    with ZipFile(archive) as zf:
        sub = _read_quarterly_submissions(zf, archive.name)
        if sub.empty:
            return pd.DataFrame()
        eligible_adsh = set(sub["adsh"])
        with zf.open("num.txt") as handle:
            reader = pd.read_csv(
                handle,
                sep="\t",
                usecols=list(_NUM_COLUMNS),
                dtype={
                    "adsh": str,
                    "tag": str,
                    "version": str,
                    "ddate": str,
                    "qtrs": "Int64",
                    "uom": str,
                    "segments": str,
                    "coreg": str,
                    "value": float,
                },
                chunksize=chunksize,
                low_memory=False,
            )
            for chunk in reader:
                keep = chunk.loc[
                    chunk["tag"].isin(tag_rank)
                    & chunk["adsh"].isin(eligible_adsh)
                    & chunk["version"].str.startswith("us-gaap/", na=False)
                    & chunk["qtrs"].isin((1, 3, 4))
                    & chunk["uom"].eq("USD")
                    & chunk["segments"].isna()
                    & chunk["coreg"].isna()
                    & chunk["value"].notna()
                ].copy()
                if not keep.empty:
                    keep["ddate"] = keep["ddate"].str.replace(r"\.0$", "", regex=True)
                    fact_parts.append(keep)

    if not fact_parts:
        return pd.DataFrame()
    facts = pd.concat(fact_parts, ignore_index=True)
    facts = facts.merge(sub, on="adsh", how="inner", validate="many_to_one")
    eligible_duration = (
        (facts["fp"].isin(("Q1", "Q2")) & facts["qtrs"].eq(1))
        | (facts["fp"].eq("Q3") & facts["qtrs"].isin((1, 3)))
        | (facts["fp"].eq("FY") & facts["qtrs"].eq(4))
    )
    facts = facts.loc[facts["ddate"].eq(facts["period"]) & eligible_duration].copy()
    if facts.empty:
        return facts
    facts["tag_rank"] = facts["tag"].map(tag_rank)
    facts = facts.sort_values(["adsh", "qtrs", "tag_rank", "tag", "value"], ascending=[True, True, True, True, False])
    # Keep one fact per concept and duration.  Q4 is annual minus nine-month
    # revenue, so those two facts must later be matched on the *same* concept.
    # Collapsing across tags here allowed, for example, annual Revenues to be
    # subtracted from nine-month SalesRevenueNet.
    return facts.drop_duplicates(["adsh", "qtrs", "tag"], keep="first").reset_index(drop=True)


def extract_sec_company_revenues(
    sec_dir: str | Path,
    *,
    start_year: int | None = None,
    end_year: int | None = None,
    chunksize: int = 500_000,
    progress: bool = True,
) -> pd.DataFrame:
    """Build a deduplicated company-fiscal-year revenue panel from SEC ZIPs."""
    parts: list[pd.DataFrame] = []
    archives = discover_sec_archives(sec_dir)
    for index, archive in enumerate(archives, start=1):
        if progress:
            print(f"[{index:02d}/{len(archives):02d}] {archive.name}", flush=True)
        part = extract_revenue_facts_from_archive(archive, chunksize=chunksize)
        if not part.empty:
            parts.append(part)
    if not parts:
        raise ValueError("No eligible annual revenue facts were found.")

    companies = pd.concat(parts, ignore_index=True)
    if start_year is not None:
        companies = companies.loc[companies["fy"].ge(start_year)]
    if end_year is not None:
        companies = companies.loc[companies["fy"].le(end_year)]

    companies["filed_sort"] = pd.to_numeric(companies["filed"], errors="coerce").fillna(-1)
    companies["accepted_sort"] = pd.to_datetime(companies["accepted"], errors="coerce")
    companies = companies.sort_values(
        ["cik", "fy", "filed_sort", "accepted_sort", "tag_rank"],
        ascending=[True, True, True, True, False],
        na_position="first",
    )
    # Latest filing wins (including a later 10-K/A); within it the lower tag rank
    # was already selected by extract_revenue_facts_from_archive.
    companies = companies.drop_duplicates(["cik", "fy"], keep="last").copy()
    companies["sic3"] = companies["sic"].map(lambda value: f"{int(value):04d}"[:3])
    companies = companies.rename(columns={"fy": "year", "value": "revenue_usd", "tag": "revenue_tag"})
    return companies.reset_index(drop=True)


def extract_sec_quarterly_company_revenues(
    sec_dir: str | Path,
    *,
    start_quarter: str | None = None,
    end_quarter: str | None = None,
    chunksize: int = 500_000,
    progress: bool = True,
) -> pd.DataFrame:
    """Build actual company-quarter revenues from 10-Q and 10-K filings.

    Q1--Q3 use the one-quarter duration fact in each 10-Q.  Because a 10-K
    normally reports only the full fiscal year, fiscal Q4 is derived as annual
    revenue minus the three preceding one-quarter 10-Q revenues.
    """
    parts: list[pd.DataFrame] = []
    archives = discover_sec_archives(sec_dir)
    for index, archive in enumerate(archives, start=1):
        if progress:
            print(f"[{index:02d}/{len(archives):02d}] {archive.name}", flush=True)
        part = extract_quarterly_revenue_facts_from_archive(archive, chunksize=chunksize)
        if not part.empty:
            parts.append(part)
    if not parts:
        raise ValueError("No eligible quarterly revenue facts were found.")

    facts = pd.concat(parts, ignore_index=True)
    facts["filed_sort"] = pd.to_numeric(facts["filed"], errors="coerce").fillna(-1)
    facts["accepted_sort"] = pd.to_datetime(facts["accepted"], errors="coerce")
    # Select the latest filing/amendment for each fiscal period first, while
    # retaining all eligible revenue concepts within that filing.
    filing_order = (
        facts[["cik", "fy", "fp", "adsh", "filed_sort", "accepted_sort"]]
        .drop_duplicates("adsh")
        .sort_values(
            ["cik", "fy", "fp", "filed_sort", "accepted_sort"],
            ascending=[True, True, True, True, True],
            na_position="first",
        )
        .drop_duplicates(["cik", "fy", "fp"], keep="last")
    )
    facts = facts.loc[facts["adsh"].isin(set(filing_order["adsh"]))].copy()

    direct = facts.loc[facts["fp"].isin(("Q1", "Q2", "Q3")) & facts["qtrs"].eq(1)].copy()
    direct = (
        direct.sort_values(["cik", "fy", "fp", "tag_rank", "tag", "value"], ascending=[True, True, True, True, True, False])
        .drop_duplicates(["cik", "fy", "fp"], keep="first")
    )
    direct["quarter_revenue_usd"] = direct["value"]
    annual = facts.loc[facts["fp"].eq("FY") & facts["qtrs"].eq(4)].copy()
    nine_month = facts.loc[
        facts["fp"].eq("Q3") & facts["qtrs"].eq(3), ["cik", "fy", "tag", "value"]
    ].copy()
    nine_month = nine_month.rename(columns={"value": "nine_month_revenue_usd"})
    q4 = annual.merge(nine_month, on=["cik", "fy", "tag"], how="inner", validate="one_to_one")
    q4 = (
        q4.sort_values(["cik", "fy", "tag_rank", "tag", "value"], ascending=[True, True, True, True, False])
        .drop_duplicates(["cik", "fy"], keep="first")
    )
    q4["quarter_revenue_usd"] = q4["value"] - q4["nine_month_revenue_usd"]
    q4["fp"] = "Q4"

    quarters = pd.concat([direct, q4[direct.columns]], ignore_index=True)
    quarters = quarters.loc[np.isfinite(quarters["quarter_revenue_usd"]) & quarters["quarter_revenue_usd"].gt(0)].copy()
    period_dates = pd.to_datetime(quarters["period"], format="%Y%m%d", errors="coerce")
    quarters["calendar_quarter"] = period_dates.dt.to_period("Q")
    quarters = quarters.dropna(subset=["calendar_quarter"])
    if start_quarter is not None:
        quarters = quarters.loc[quarters["calendar_quarter"].ge(pd.Period(start_quarter, freq="Q"))]
    if end_quarter is not None:
        quarters = quarters.loc[quarters["calendar_quarter"].le(pd.Period(end_quarter, freq="Q"))]
    quarters = quarters.sort_values(["cik", "calendar_quarter", "filed_sort", "accepted_sort"])
    quarters = quarters.drop_duplicates(["cik", "calendar_quarter"], keep="last").copy()
    quarters["sic3"] = quarters["sic"].map(lambda value: f"{int(value):04d}"[:3])
    quarters = quarters.rename(columns={"fy": "fiscal_year", "tag": "revenue_tag", "fp": "fiscal_period"})
    return quarters.reset_index(drop=True)


def _flag_financial(sic3: pd.Series) -> pd.Series:
    return sic3.astype(str).str[0].isin(FINANCIAL_SIC_MAJOR_GROUPS)


def _market_table(firms: pd.DataFrame, market_keys: list[str], revenue_col: str) -> pd.DataFrame:
    """Build the SIC3 market panel: revenue-share HHI, firm count, market revenue."""
    out = firms.copy()
    out["market_revenue_usd"] = out.groupby(market_keys)[revenue_col].transform("sum")
    out["market_firms"] = out.groupby(market_keys)["cik"].transform("nunique")
    out["market_share_sq"] = (out[revenue_col] / out["market_revenue_usd"]) ** 2
    markets = out.groupby(market_keys, as_index=False).agg(
        market_hhi=("market_share_sq", "sum"),
        market_firms=("market_firms", "first"),
        market_revenue_usd=("market_revenue_usd", "first"),
    )
    markets["is_financial"] = _flag_financial(markets["sic3"])
    return markets


def _collapse_markets(markets: pd.DataFrame) -> dict[str, float]:
    """Collapse one period's SIC3 market HHIs into the competition aggregates.

    See ``SEC_INVERSE_HHI_COLUMNS`` for why three aggregators are kept and which
    one the estimated equation implies.
    """
    hhi_m = markets["market_hhi"].to_numpy(dtype=float)
    firm_w = markets["market_firms"].to_numpy(dtype=float)
    revenue_w = markets["market_revenue_usd"].to_numpy(dtype=float)
    firm_weighted = float(np.average(hhi_m, weights=firm_w))
    revenue_weighted = float(np.average(hhi_m, weights=revenue_w))
    # Geometric mean of 1/HHI_m: the aggregate that is linear in log N.
    log_revenue_weighted = float(np.average(np.log(1.0 / hhi_m), weights=revenue_w))
    return {
        "hhi_firm_weighted": firm_weighted,
        "hhi_revenue_weighted": revenue_weighted,
        "hhi_market_mean": float(np.mean(hhi_m)),
        "inv_hhi_firmw": 1.0 / firm_weighted,
        "inv_hhi_revw": 1.0 / revenue_weighted,
        "inv_hhi_logrevw": float(np.exp(log_revenue_weighted)),
    }


def _period_row(markets: pd.DataFrame, firms: pd.DataFrame, revenue_col: str) -> dict[str, float | int]:
    """Aggregate one period, on all markets and on the non-financial subset."""
    agg = _collapse_markets(markets)
    row: dict[str, float | int] = {
        "hhi": agg["hhi_firm_weighted"],
        "hhi_10000": 10_000.0 * agg["hhi_firm_weighted"],
        "effective_firms": agg["inv_hhi_firmw"],
        "hhi_market_mean": agg["hhi_market_mean"],
        "hhi_revenue_weighted": agg["hhi_revenue_weighted"],
        "inv_hhi_revw": agg["inv_hhi_revw"],
        "inv_hhi_logrevw": agg["inv_hhi_logrevw"],
        "n_firms": int(firms["cik"].nunique()),
        "n_markets": int(len(markets)),
        "total_revenue_usd": float(firms[revenue_col].sum()),
    }
    nonfinancial = markets.loc[~markets["is_financial"]]
    if nonfinancial.empty:
        row.update(
            inv_hhi_firmw_exfin=np.nan,
            inv_hhi_revw_exfin=np.nan,
            inv_hhi_logrevw_exfin=np.nan,
            n_firms_exfin=0,
            n_markets_exfin=0,
        )
        return row
    ex = _collapse_markets(nonfinancial)
    row.update(
        inv_hhi_firmw_exfin=ex["inv_hhi_firmw"],
        inv_hhi_revw_exfin=ex["inv_hhi_revw"],
        inv_hhi_logrevw_exfin=ex["inv_hhi_logrevw"],
        n_firms_exfin=int(firms.loc[~_flag_financial(firms["sic3"]), "cik"].nunique()),
        n_markets_exfin=int(len(nonfinancial)),
    )
    return row


def calculate_annual_hhi(company_revenues: pd.DataFrame) -> pd.DataFrame:
    """Calculate SIC3 market HHIs and collapse them to one observation per year.

    ``hhi`` is the firm-weighted mean market HHI and ``effective_firms`` its
    inverse; the revenue-weighted and revenue-weighted-log alternatives, and the
    non-financial versions of all three, are emitted alongside it.
    """
    required = {"year", "cik", "sic3", "revenue_usd"}
    missing = required.difference(company_revenues.columns)
    if missing:
        raise ValueError(f"company_revenues is missing columns: {sorted(missing)}")
    firms = company_revenues.loc[
        np.isfinite(company_revenues["revenue_usd"])
        & company_revenues["revenue_usd"].gt(0)
        & company_revenues["sic3"].notna()
    ].copy()
    if firms.empty:
        raise ValueError("No positive company revenues are available for HHI calculation.")

    markets = _market_table(firms, ["year", "sic3"], "revenue_usd")
    rows: list[dict[str, float | int]] = []
    for year, group in markets.groupby("year", sort=True):
        year_firms = firms.loc[firms["year"].eq(year)]
        rows.append({"year": int(year), **_period_row(group, year_firms, "revenue_usd")})
    return pd.DataFrame(rows).sort_values("year").reset_index(drop=True)


def calculate_quarterly_hhi(company_revenues: pd.DataFrame) -> pd.DataFrame:
    """Calculate actual company-quarter SIC3 HHIs and quarterly aggregates."""
    required = {"calendar_quarter", "cik", "sic3", "quarter_revenue_usd"}
    missing = required.difference(company_revenues.columns)
    if missing:
        raise ValueError(f"company_revenues is missing columns: {sorted(missing)}")
    firms = company_revenues.loc[
        np.isfinite(company_revenues["quarter_revenue_usd"])
        & company_revenues["quarter_revenue_usd"].gt(0)
        & company_revenues["sic3"].notna()
    ].copy()
    if firms.empty:
        raise ValueError("No positive company-quarter revenues are available for HHI calculation.")

    markets = _market_table(firms, ["calendar_quarter", "sic3"], "quarter_revenue_usd")
    rows: list[dict[str, float | int | str]] = []
    for quarter, group in markets.groupby("calendar_quarter", sort=True):
        period_firms = firms.loc[firms["calendar_quarter"].eq(quarter)]
        rows.append({"quarter": str(quarter), **_period_row(group, period_firms, "quarter_revenue_usd")})
    return pd.DataFrame(rows)


def validate_hhi_fraction(hhi_data: pd.DataFrame) -> pd.DataFrame:
    """Validate the repository's SEC HHI normalization and effective number.

    ``calculate_*_hhi`` constructs shares as revenue divided by market revenue,
    so ``hhi`` is on the unit interval and ``effective_firms = 1 / hhi``.  This
    validator makes that scale assumption executable instead of silently relying
    on a column name.
    """
    required = {"hhi", "hhi_10000"}
    missing = required.difference(hhi_data.columns)
    if missing:
        raise ValueError(f"SEC HHI data are missing columns: {sorted(missing)}")
    out = hhi_data.copy()
    hhi = pd.to_numeric(out["hhi"], errors="coerce")
    if hhi.isna().any():
        raise ValueError("SEC HHI contains missing or nonnumeric values.")
    if (hhi <= 0.0).any() or (hhi > 1.0).any():
        bad = hhi[(hhi <= 0.0) | (hhi > 1.0)]
        raise ValueError(f"SEC HHI must be in (0, 1]; invalid values: {bad.tolist()[:5]}")
    conventional = pd.to_numeric(out["hhi_10000"], errors="coerce")
    if not np.allclose(conventional.to_numpy(), 10_000.0 * hhi.to_numpy(), rtol=1e-8, atol=1e-6):
        raise ValueError("hhi_10000 is inconsistent with the unit-fraction hhi column.")
    effective = 1.0 / hhi
    if "effective_firms" in out and not np.allclose(
        pd.to_numeric(out["effective_firms"], errors="coerce").to_numpy(),
        effective.to_numpy(), rtol=1e-8, atol=1e-8,
    ):
        raise ValueError("effective_firms is inconsistent with 1 / hhi.")
    out["effective_firms"] = effective

    if "hhi_revenue_weighted" in out and "inv_hhi_revw" in out:
        revenue_weighted = pd.to_numeric(out["hhi_revenue_weighted"], errors="coerce")
        if not np.allclose(
            pd.to_numeric(out["inv_hhi_revw"], errors="coerce").to_numpy(),
            (1.0 / revenue_weighted).to_numpy(), rtol=1e-8, atol=1e-8,
        ):
            raise ValueError("inv_hhi_revw is inconsistent with 1 / hhi_revenue_weighted.")
    # Every alternative aggregate is an effective firm count, so it is bounded
    # below by one (a single-firm market has HHI = 1).  This catches a raw HHI
    # or a 0-10,000 value being wired in as if it were an inverse.
    for name in SEC_INVERSE_HHI_COLUMNS:
        if name not in out:
            continue
        values = pd.to_numeric(out[name], errors="coerce")
        finite = values[np.isfinite(values)]
        if finite.empty:
            raise ValueError(f"SEC column {name} has no finite values.")
        if (finite < 1.0 - 1e-9).any():
            raise ValueError(f"SEC column {name} must be an inverse HHI (>= 1); min is {finite.min()}.")
    return out


def merge_sec_inverse_hhi(
    model_data: pd.DataFrame,
    hhi_data: pd.DataFrame,
    *,
    column: str | None = None,
    columns: Mapping[str, str] | None = None,
) -> pd.DataFrame:
    """Merge validated quarterly inverse-HHI aggregates into a model-ready frame.

    ``column`` merges only the firm-weighted headline under the given name.
    ``columns`` maps HHI-CSV column -> model-ready column; it defaults to
    ``DEFAULT_SEC_INVERSE_COLUMNS``, and entries whose source column is absent
    are skipped so that an HHI file written before an aggregate existed still
    merges.  Callers that require a specific aggregate must check for it; the
    build script does.
    """
    if column is not None and columns is not None:
        raise ValueError("Pass either column or columns, not both.")
    if column is not None:
        mapping: Mapping[str, str] = {"effective_firms": column}
    else:
        mapping = DEFAULT_SEC_INVERSE_COLUMNS if columns is None else columns

    hhi = validate_hhi_fraction(hhi_data)
    if "quarter" not in hhi:
        raise ValueError("Quarterly SEC HHI data must contain a quarter column.")
    quarter = pd.PeriodIndex(hhi["quarter"].astype(str), freq="Q")
    if quarter.duplicated().any():
        raise ValueError("Quarterly SEC HHI contains duplicate quarters.")
    available = [source for source in mapping if source in hhi]
    if not available:
        raise ValueError(f"SEC HHI data contain none of the requested columns: {sorted(mapping)}")

    out = model_data.copy()
    if not isinstance(out.index, pd.DatetimeIndex):
        if "DATE" not in out:
            raise ValueError("model_data needs a DatetimeIndex or DATE column.")
        out = out.set_index(pd.to_datetime(out.pop("DATE")))
    target = out.index.to_period("Q")
    for source in available:
        series = pd.Series(hhi[source].to_numpy(dtype=float), index=quarter)
        out[mapping[source]] = series.reindex(target).to_numpy(dtype=float)
    return out


def build_sec_hhi_csv(
    sec_dir: str | Path,
    output_path: str | Path,
    *,
    start_year: int | None = 2011,
    end_year: int | None = None,
    chunksize: int = 500_000,
    progress: bool = True,
) -> pd.DataFrame:
    if end_year is None:
        end_year = latest_complete_fiscal_year(sec_dir)
    companies = extract_sec_company_revenues(
        sec_dir,
        start_year=start_year,
        end_year=end_year,
        chunksize=chunksize,
        progress=progress,
    )
    annual = calculate_annual_hhi(companies)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    annual.to_csv(output_path, index=False, float_format="%.10g")
    return annual


def build_sec_quarterly_hhi_csv(
    sec_dir: str | Path,
    output_path: str | Path,
    *,
    start_quarter: str | None = "2012Q1",
    end_quarter: str | None = None,
    chunksize: int = 500_000,
    progress: bool = True,
    company_panel_path: str | Path | None = None,
) -> pd.DataFrame:
    """Extract company-quarter revenues from the SEC archives and aggregate them.

    ``company_panel_path`` caches the extracted company-quarter panel.  Parsing
    the archives dominates the runtime while the aggregation step is seconds, so
    keeping the panel lets a new aggregator be added without re-reading 5+ GB of
    ZIPs (see ``recalculate_quarterly_hhi_from_panel``).
    """
    if end_quarter is None:
        end_quarter = f"{latest_complete_fiscal_year(sec_dir)}Q4"
    companies = extract_sec_quarterly_company_revenues(
        sec_dir,
        start_quarter=start_quarter,
        end_quarter=end_quarter,
        chunksize=chunksize,
        progress=progress,
    )
    if company_panel_path is not None:
        panel_path = Path(company_panel_path)
        panel_path.parent.mkdir(parents=True, exist_ok=True)
        companies.to_csv(panel_path, index=False)
    quarterly = calculate_quarterly_hhi(companies)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    quarterly.to_csv(output_path, index=False, float_format="%.10g")
    return quarterly


def recalculate_quarterly_hhi_from_panel(
    company_panel_path: str | Path,
    output_path: str | Path,
) -> pd.DataFrame:
    """Re-aggregate a cached company-quarter panel without re-reading the archives."""
    companies = pd.read_csv(company_panel_path)
    if "calendar_quarter" not in companies:
        raise ValueError(f"{company_panel_path} is not a company-quarter panel.")
    companies["calendar_quarter"] = pd.PeriodIndex(companies["calendar_quarter"].astype(str), freq="Q")
    companies["sic3"] = companies["sic3"].map(lambda value: f"{int(value):03d}")
    quarterly = calculate_quarterly_hhi(companies)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    quarterly.to_csv(output_path, index=False, float_format="%.10g")
    return quarterly
