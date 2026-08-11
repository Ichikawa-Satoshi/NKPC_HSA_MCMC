from __future__ import annotations

from pathlib import Path
import re
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import pandas as pd
import pytest

from nkpc_hsa.dataprep.func_data_build import (
    load_spf_expectations,
    load_spf_quarter_ahead_expectations,
    load_spf_yoy_expectations,
)


def _inflation_dir(tmp_path: Path) -> Path:
    path = tmp_path / "inflation"
    path.mkdir()
    return path


def test_spf_quarter_ahead_loader_uses_dpgdp3_and_preserves_units(tmp_path: Path) -> None:
    inflation = _inflation_dir(tmp_path)
    pd.DataFrame(
        {
            "YEAR": [1982, 1982],
            "QUARTER": [1, 2],
            "DPGDP2": [8.0, 7.0],
            "DPGDP3": [4.0, 8.0],
        }
    ).to_excel(inflation / "Median_PGDP_Growth.xlsx", index=False)

    out = load_spf_quarter_ahead_expectations(tmp_path)

    assert out.index.to_period("Q").astype(str).tolist() == ["1982Q1", "1982Q2"]
    assert out["Epi_spf_gdp_1q_ahead_ann_pct"].tolist() == [4.0, 8.0]
    np.testing.assert_allclose(
        out["Epi_spf_gdp_1q_ahead_ann_log"],
        100.0 * np.log1p(np.array([4.0, 8.0]) / 100.0),
    )
    np.testing.assert_allclose(
        out["Epi_spf_gdp_1q_ahead_qoq_pct"],
        100.0 * np.expm1(np.log1p(np.array([4.0, 8.0]) / 100.0) / 4.0),
    )
    assert not np.allclose(out["Epi_spf_gdp_1q_ahead_qoq_pct"], np.array([4.0, 8.0]) / 4.0)


def test_spf_yoy_loader_has_explicit_horizon_and_legacy_aliases(tmp_path: Path) -> None:
    inflation = _inflation_dir(tmp_path)
    pd.DataFrame(
        {
            "YEAR": [2000, 2000],
            "QUARTER": [3, 4],
            "INFPGDP1YR": [2.5, 3.0],
            "INFCPI1YR": [2.75, 3.25],
        }
    ).to_excel(inflation / "SPF_Inflation_Expectation.xlsx", index=False)

    out = load_spf_yoy_expectations(tmp_path)

    assert out.index.to_period("Q").astype(str).tolist() == ["2000Q3", "2000Q4"]
    assert out["Epi_spf_gdp_yoy_1y_ahead"].tolist() == [2.5, 3.0]
    assert out["Epi_spf_cpi_yoy_1y_ahead"].tolist() == [2.75, 3.25]
    pd.testing.assert_series_equal(
        out["Epi_spf_gdp"], out["Epi_spf_gdp_yoy_1y_ahead"], check_names=False
    )
    pd.testing.assert_series_equal(
        out["Epi_spf_cpi"], out["Epi_spf_cpi_yoy_1y_ahead"], check_names=False
    )
    pd.testing.assert_frame_equal(load_spf_expectations(tmp_path), out)


def test_spf_quarter_ahead_loader_rejects_missing_horizon_column(tmp_path: Path) -> None:
    inflation = _inflation_dir(tmp_path)
    pd.DataFrame({"YEAR": [1982], "QUARTER": [1], "DPGDP2": [4.0]}).to_excel(
        inflation / "Median_PGDP_Growth.xlsx", index=False
    )

    with pytest.raises(KeyError, match="DPGDP3"):
        load_spf_quarter_ahead_expectations(tmp_path)


def test_spf_loader_handles_official_malformed_xlsx_metadata(tmp_path: Path) -> None:
    inflation = _inflation_dir(tmp_path)
    path = inflation / "Median_PGDP_Growth.xlsx"
    pd.DataFrame(
        {"YEAR": [2026], "QUARTER": [2], "DPGDP3": [3.25]}
    ).to_excel(path, index=False)

    # Reproduce the malformed W3CDTF metadata currently found in the official
    # Philadelphia Fed workbooks without altering any worksheet content.
    rewritten = path.with_suffix(".rewritten.xlsx")
    with ZipFile(path) as source, ZipFile(rewritten, "w", ZIP_DEFLATED) as target:
        for member in source.infolist():
            payload = source.read(member.filename)
            if member.filename == "docProps/core.xml":
                payload = re.sub(
                    rb"(>)(\d{4}-\d{2}-\d{2})T[^<]+(</dcterms:(?:created|modified)>)",
                    rb"\1\2T 2:29:10-04:00\3",
                    payload,
                )
            target.writestr(member, payload)
    path.unlink()
    rewritten.rename(path)

    out = load_spf_quarter_ahead_expectations(tmp_path)
    assert out.loc[out.index[0], "Epi_spf_gdp_1q_ahead_ann_pct"] == 3.25
