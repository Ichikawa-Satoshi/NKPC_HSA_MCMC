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
