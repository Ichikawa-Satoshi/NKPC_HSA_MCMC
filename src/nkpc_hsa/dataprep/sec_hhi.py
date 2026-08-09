from __future__ import annotations

import re
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
    facts = facts.sort_values(["adsh", "tag_rank", "tag", "value"], ascending=[True, True, True, False])
    return facts.drop_duplicates(["adsh", "qtrs"], keep="first").reset_index(drop=True)


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
    facts = facts.sort_values(
        ["cik", "fy", "fp", "filed_sort", "accepted_sort", "tag_rank"],
        ascending=[True, True, True, True, True, False],
        na_position="first",
    ).drop_duplicates(["cik", "fy", "fp", "qtrs"], keep="last")

    direct = facts.loc[facts["fp"].isin(("Q1", "Q2", "Q3")) & facts["qtrs"].eq(1)].copy()
    direct["quarter_revenue_usd"] = direct["value"]
    annual = facts.loc[facts["fp"].eq("FY") & facts["qtrs"].eq(4)].copy()
    nine_month = facts.loc[facts["fp"].eq("Q3") & facts["qtrs"].eq(3), ["cik", "fy", "value"]].copy()
    nine_month = nine_month.rename(columns={"value": "nine_month_revenue_usd"})
    q4 = annual.merge(nine_month, on=["cik", "fy"], how="inner", validate="one_to_one")
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


def calculate_annual_hhi(company_revenues: pd.DataFrame) -> pd.DataFrame:
    """Calculate SIC3 market HHIs and collapse them to one observation per year.

    ``hhi`` is the firm-weighted mean market HHI.  This matches averaging a
    market HHI attached to every firm and is comparable to the existing TNIC-3
    firm-level mean.  Unweighted-market and revenue-weighted alternatives are
    included as diagnostics in the same output CSV.
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

    market_keys = ["year", "sic3"]
    firms["market_revenue_usd"] = firms.groupby(market_keys)["revenue_usd"].transform("sum")
    firms["market_firms"] = firms.groupby(market_keys)["cik"].transform("nunique")
    firms["market_share_sq"] = (firms["revenue_usd"] / firms["market_revenue_usd"]) ** 2
    markets = (
        firms.groupby(market_keys, as_index=False)
        .agg(
            market_hhi=("market_share_sq", "sum"),
            market_firms=("market_firms", "first"),
            market_revenue_usd=("market_revenue_usd", "first"),
        )
    )
    markets["firm_weight"] = markets["market_firms"]
    markets["revenue_weight"] = markets["market_revenue_usd"]

    rows: list[dict[str, float | int]] = []
    for year, group in markets.groupby("year", sort=True):
        hhi = float(np.average(group["market_hhi"], weights=group["firm_weight"]))
        hhi_market_mean = float(group["market_hhi"].mean())
        hhi_revenue_weighted = float(np.average(group["market_hhi"], weights=group["revenue_weight"]))
        year_firms = firms.loc[firms["year"].eq(year)]
        rows.append(
            {
                "year": int(year),
                "hhi": hhi,
                "hhi_10000": 10_000.0 * hhi,
                "effective_firms": 1.0 / hhi,
                "hhi_market_mean": hhi_market_mean,
                "hhi_revenue_weighted": hhi_revenue_weighted,
                "n_firms": int(year_firms["cik"].nunique()),
                "n_markets": int(len(group)),
                "total_revenue_usd": float(year_firms["revenue_usd"].sum()),
            }
        )
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

    market_keys = ["calendar_quarter", "sic3"]
    firms["market_revenue_usd"] = firms.groupby(market_keys)["quarter_revenue_usd"].transform("sum")
    firms["market_firms"] = firms.groupby(market_keys)["cik"].transform("nunique")
    firms["market_share_sq"] = (firms["quarter_revenue_usd"] / firms["market_revenue_usd"]) ** 2
    markets = firms.groupby(market_keys, as_index=False).agg(
        market_hhi=("market_share_sq", "sum"),
        market_firms=("market_firms", "first"),
        market_revenue_usd=("market_revenue_usd", "first"),
    )

    rows: list[dict[str, float | int | str]] = []
    for quarter, group in markets.groupby("calendar_quarter", sort=True):
        hhi = float(np.average(group["market_hhi"], weights=group["market_firms"]))
        hhi_revenue_weighted = float(np.average(group["market_hhi"], weights=group["market_revenue_usd"]))
        period_firms = firms.loc[firms["calendar_quarter"].eq(quarter)]
        rows.append(
            {
                "quarter": str(quarter),
                "hhi": hhi,
                "hhi_10000": 10_000.0 * hhi,
                "effective_firms": 1.0 / hhi,
                "hhi_market_mean": float(group["market_hhi"].mean()),
                "hhi_revenue_weighted": hhi_revenue_weighted,
                "n_firms": int(period_firms["cik"].nunique()),
                "n_markets": int(len(group)),
                "total_revenue_usd": float(period_firms["quarter_revenue_usd"].sum()),
            }
        )
    return pd.DataFrame(rows)


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
) -> pd.DataFrame:
    if end_quarter is None:
        end_quarter = f"{latest_complete_fiscal_year(sec_dir)}Q4"
    companies = extract_sec_quarterly_company_revenues(
        sec_dir,
        start_quarter=start_quarter,
        end_quarter=end_quarter,
        chunksize=chunksize,
        progress=progress,
    )
    quarterly = calculate_quarterly_hhi(companies)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    quarterly.to_csv(output_path, index=False, float_format="%.10g")
    return quarterly
