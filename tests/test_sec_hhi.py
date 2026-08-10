from __future__ import annotations

from io import StringIO
from zipfile import ZipFile

import numpy as np
import pandas as pd

from nkpc_hsa.dataprep.sec_hhi import (
    calculate_annual_hhi,
    calculate_quarterly_hhi,
    extract_sec_company_revenues,
    extract_sec_quarterly_company_revenues,
    latest_complete_fiscal_year,
    merge_sec_inverse_hhi,
    validate_hhi_fraction,
)


def _write_archive(path, submissions: list[dict], facts: list[dict]) -> None:
    sub_columns = ["adsh", "cik", "name", "sic", "form", "period", "fy", "fp", "filed", "accepted"]
    num_columns = ["adsh", "tag", "version", "ddate", "qtrs", "uom", "segments", "coreg", "value"]
    sub = pd.DataFrame(submissions, columns=sub_columns)
    num = pd.DataFrame(facts, columns=num_columns)
    with ZipFile(path, "w") as zf:
        sub_buffer = StringIO()
        num_buffer = StringIO()
        sub.to_csv(sub_buffer, sep="\t", index=False)
        num.to_csv(num_buffer, sep="\t", index=False)
        zf.writestr("sub.txt", sub_buffer.getvalue())
        zf.writestr("num.txt", num_buffer.getvalue())


def test_extraction_uses_current_period_priority_and_latest_amendment(tmp_path) -> None:
    common = {"cik": "1", "name": "A", "sic": "3571", "form": "10-K", "period": "20201231", "fy": "2020", "fp": "FY"}
    _write_archive(
        tmp_path / "2021q1.zip",
        [{**common, "adsh": "old", "filed": "20210201", "accepted": "2021-02-01 10:00:00"}],
        [
            {"adsh": "old", "tag": "SalesRevenueNet", "version": "us-gaap/2020", "ddate": "20201231", "qtrs": 4, "uom": "USD", "segments": None, "coreg": None, "value": 100.0},
            {"adsh": "old", "tag": "Revenues", "version": "us-gaap/2020", "ddate": "20201231", "qtrs": 4, "uom": "USD", "segments": None, "coreg": None, "value": 110.0},
            {"adsh": "old", "tag": "Revenues", "version": "us-gaap/2020", "ddate": "20191231", "qtrs": 4, "uom": "USD", "segments": None, "coreg": None, "value": 90.0},
        ],
    )
    _write_archive(
        tmp_path / "2021q2.zip",
        [{**common, "adsh": "new", "form": "10-K/A", "filed": "20210401", "accepted": "2021-04-01 10:00:00"}],
        [{"adsh": "new", "tag": "Revenues", "version": "us-gaap/2020", "ddate": "20201231", "qtrs": 4, "uom": "USD", "segments": None, "coreg": None, "value": 120.0}],
    )

    result = extract_sec_company_revenues(tmp_path, progress=False)

    assert len(result) == 1
    assert result.loc[0, "revenue_usd"] == 120.0
    assert result.loc[0, "revenue_tag"] == "Revenues"
    assert result.loc[0, "sic3"] == "357"


def test_calculate_annual_hhi_uses_firm_weighted_market_mean() -> None:
    firms = pd.DataFrame(
        {
            "year": [2020, 2020, 2020],
            "cik": [1, 2, 3],
            "sic3": ["100", "100", "200"],
            "revenue_usd": [50.0, 50.0, 100.0],
        }
    )

    result = calculate_annual_hhi(firms).iloc[0]

    # Market HHIs are .5 (two equal firms) and 1 (singleton). Firm weighting
    # therefore gives (2*.5 + 1*1)/3 = 2/3.
    assert np.isclose(result["hhi"], 2.0 / 3.0)
    assert np.isclose(result["effective_firms"], 1.5)
    assert np.isclose(result["hhi_market_mean"], 0.75)
    assert result["n_firms"] == 3
    assert result["n_markets"] == 2


def test_latest_complete_fiscal_year_is_prior_archive_year(tmp_path) -> None:
    (tmp_path / "2025q4.zip").touch()
    (tmp_path / "2026q1.zip").touch()

    assert latest_complete_fiscal_year(tmp_path) == 2025


def test_quarterly_extraction_uses_10q_and_derives_fiscal_q4(tmp_path) -> None:
    submissions = []
    facts = []
    specs = [
        ("q1", "10-Q", "Q1", "20200331", 10.0, 1),
        ("q2", "10-Q", "Q2", "20200630", 20.0, 1),
        ("q3", "10-Q", "Q3", "20200930", 30.0, 1),
        ("fy", "10-K", "FY", "20201231", 100.0, 4),
    ]
    for adsh, form, fp, period, value, qtrs in specs:
        submissions.append(
            {
                "adsh": adsh,
                "cik": "1",
                "name": "A",
                "sic": "3571",
                "form": form,
                "period": period,
                "fy": "2020",
                "fp": fp,
                "filed": period,
                "accepted": f"{period[:4]}-{period[4:6]}-{period[6:]} 10:00:00",
            }
        )
        facts.append(
            {
                "adsh": adsh,
                "tag": "Revenues",
                "version": "us-gaap/2020",
                "ddate": period,
                "qtrs": qtrs,
                "uom": "USD",
                "segments": None,
                "coreg": None,
                "value": value,
            }
        )
        if fp == "Q3":
            facts.append(
                {
                    "adsh": adsh,
                    "tag": "Revenues",
                    "version": "us-gaap/2020",
                    "ddate": period,
                    "qtrs": 3,
                    "uom": "USD",
                    "segments": None,
                    "coreg": None,
                    "value": 60.0,
                }
            )
    _write_archive(tmp_path / "2021q1.zip", submissions, facts)

    companies = extract_sec_quarterly_company_revenues(tmp_path, progress=False)

    values = companies.set_index("fiscal_period")["quarter_revenue_usd"].to_dict()
    assert values == {"Q1": 10.0, "Q2": 20.0, "Q3": 30.0, "Q4": 40.0}
    result = calculate_quarterly_hhi(companies)
    assert result["quarter"].tolist() == ["2020Q1", "2020Q2", "2020Q3", "2020Q4"]
    assert np.allclose(result["hhi"], 1.0)


def test_fiscal_q4_subtracts_the_same_revenue_concept(tmp_path) -> None:
    submissions = [
        {
            "adsh": "q3", "cik": "1", "name": "A", "sic": "3571", "form": "10-Q",
            "period": "20200930", "fy": "2020", "fp": "Q3", "filed": "20201030",
            "accepted": "2020-10-30 10:00:00",
        },
        {
            "adsh": "fy", "cik": "1", "name": "A", "sic": "3571", "form": "10-K",
            "period": "20201231", "fy": "2020", "fp": "FY", "filed": "20210220",
            "accepted": "2021-02-20 10:00:00",
        },
    ]
    common = {
        "version": "us-gaap/2020", "uom": "USD", "segments": None, "coreg": None,
    }
    facts = [
        {**common, "adsh": "q3", "tag": "SalesRevenueNet", "ddate": "20200930", "qtrs": 1, "value": 30.0},
        {**common, "adsh": "q3", "tag": "SalesRevenueNet", "ddate": "20200930", "qtrs": 3, "value": 60.0},
        # Revenues has higher global priority, but it has no matching nine-month
        # fact.  The Q4 subtraction must therefore use SalesRevenueNet on both sides.
        {**common, "adsh": "fy", "tag": "Revenues", "ddate": "20201231", "qtrs": 4, "value": 1_000.0},
        {**common, "adsh": "fy", "tag": "SalesRevenueNet", "ddate": "20201231", "qtrs": 4, "value": 100.0},
    ]
    _write_archive(tmp_path / "2021q1.zip", submissions, facts)

    companies = extract_sec_quarterly_company_revenues(tmp_path, progress=False)
    q4 = companies.loc[companies["fiscal_period"].eq("Q4")].iloc[0]

    assert q4["revenue_tag"] == "SalesRevenueNet"
    assert q4["quarter_revenue_usd"] == 40.0


def test_sec_hhi_fraction_validation_and_inverse_merge() -> None:
    hhi = pd.DataFrame(
        {
            "quarter": ["2020Q1", "2020Q2"],
            "hhi": [0.25, 0.20],
            "hhi_10000": [2500.0, 2000.0],
            "effective_firms": [4.0, 5.0],
        }
    )
    checked = validate_hhi_fraction(hhi)
    np.testing.assert_allclose(checked["effective_firms"], 1.0 / checked["hhi"])
    data = pd.DataFrame({"DATE": ["2020-03-31", "2020-06-30"], "x": [1.0, 2.0]})
    merged = merge_sec_inverse_hhi(data, hhi)
    np.testing.assert_allclose(merged["N_SEC_inverse_HHI"], [4.0, 5.0])


def test_sec_hhi_validator_rejects_conventional_scale_in_fraction_column() -> None:
    bad = pd.DataFrame({"hhi": [2500.0], "hhi_10000": [2500.0]})
    with np.testing.assert_raises_regex(ValueError, "must be in"):
        validate_hhi_fraction(bad)


def _two_market_quarter() -> pd.DataFrame:
    """Small market: 2 equal firms, HHI .5.  Large market: 1 firm, HHI 1."""
    return pd.DataFrame(
        {
            "calendar_quarter": [pd.Period("2020Q1", freq="Q")] * 3,
            "cik": [1, 2, 3],
            "sic3": ["100", "100", "200"],
            "quarter_revenue_usd": [50.0, 50.0, 900.0],
        }
    )


def test_market_aggregates_use_the_intended_weights() -> None:
    result = calculate_quarterly_hhi(_two_market_quarter()).iloc[0]

    # Firm weighting: (2*.5 + 1*1)/3.  Revenue weighting: (100*.5 + 900*1)/1000.
    assert np.isclose(result["hhi"], 2.0 / 3.0)
    assert np.isclose(result["effective_firms"], 1.5)
    assert np.isclose(result["hhi_revenue_weighted"], 0.95)
    assert np.isclose(result["inv_hhi_revw"], 1.0 / 0.95)
    # Geometric mean of 1/HHI: exp((100*log 2 + 900*log 1)/1000).
    assert np.isclose(result["inv_hhi_logrevw"], np.exp(100.0 * np.log(2.0) / 1000.0))


def test_inverse_aggregates_obey_the_jensen_ordering() -> None:
    result = calculate_quarterly_hhi(_two_market_quarter()).iloc[0]

    # 1/E[HHI] is a harmonic mean of the market effective firm counts and the
    # log version a geometric one, so the harmonic can never exceed it.
    assert result["inv_hhi_revw"] <= result["inv_hhi_logrevw"] + 1e-12


def test_financial_markets_are_excluded_from_the_exfin_aggregates() -> None:
    firms = pd.concat(
        [
            _two_market_quarter(),
            pd.DataFrame(
                {
                    "calendar_quarter": [pd.Period("2020Q1", freq="Q")] * 2,
                    "cik": [4, 5],
                    "sic3": ["602", "602"],
                    "quarter_revenue_usd": [5_000.0, 5_000.0],
                }
            ),
        ],
        ignore_index=True,
    )

    result = calculate_quarterly_hhi(firms).iloc[0]

    # The SIC 6 market is huge, so it dominates the revenue-weighted aggregate
    # but must leave the ex-financial one identical to the two-market case.
    assert result["n_markets"] == 3
    assert result["n_markets_exfin"] == 2
    assert result["n_firms_exfin"] == 3
    assert np.isclose(result["inv_hhi_revw_exfin"], 1.0 / 0.95)
    assert not np.isclose(result["inv_hhi_revw"], result["inv_hhi_revw_exfin"])


def test_merge_emits_every_available_aggregate_and_skips_absent_ones() -> None:
    hhi = calculate_quarterly_hhi(_two_market_quarter())
    data = pd.DataFrame({"DATE": ["2020-03-31"], "x": [1.0]})

    merged = merge_sec_inverse_hhi(data, hhi)

    assert np.isclose(merged["N_SEC_inverse_HHI"].iloc[0], 1.5)
    assert np.isclose(merged["N_SEC_inverse_HHI_revw"].iloc[0], 1.0 / 0.95)
    assert "N_SEC_inverse_HHI_logrevw" in merged
    # An HHI file written before the new aggregates existed still merges.
    legacy = hhi[["quarter", "hhi", "hhi_10000", "effective_firms"]]
    legacy_merged = merge_sec_inverse_hhi(data, legacy)
    assert "N_SEC_inverse_HHI" in legacy_merged
    assert "N_SEC_inverse_HHI_revw" not in legacy_merged


def test_validator_rejects_an_inverse_column_holding_a_raw_hhi() -> None:
    bad = pd.DataFrame({"hhi": [0.25], "hhi_10000": [2500.0], "inv_hhi_logrevw": [0.25]})
    with np.testing.assert_raises_regex(ValueError, "inverse HHI"):
        validate_hhi_fraction(bad)


def test_validator_rejects_inconsistent_revenue_weighted_inverse() -> None:
    bad = pd.DataFrame(
        {"hhi": [0.25], "hhi_10000": [2500.0], "hhi_revenue_weighted": [0.5], "inv_hhi_revw": [4.0]}
    )
    with np.testing.assert_raises_regex(ValueError, "inv_hhi_revw is inconsistent"):
        validate_hhi_fraction(bad)
