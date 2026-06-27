from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nkpc_hsa.data.build import add_hp_output_gap, add_labor_share_gap, hp_filter_series, load_labor_share_gap
from nkpc_hsa.data.transforms import transform_competition_series


def test_n_transform_default_centers_and_scales_log100() -> None:
    values = np.exp(np.array([1.0, 1.1, 0.9]))
    out = transform_competition_series(values)
    assert np.allclose(out, [0.0, 1.0, -1.0])


def test_n_transform_log100() -> None:
    values = np.array([1.0, np.e])
    out = transform_competition_series(values, transform="log100")
    assert np.allclose(out, [0.0, 100.0])


def test_n_transform_rejects_nonpositive_levels() -> None:
    with pytest.raises(ValueError):
        transform_competition_series(np.array([1.0, 0.0]), transform="log100")


def test_hp_filter_linear_output_has_near_zero_gap() -> None:
    output = pd.Series(np.linspace(1.0, 2.0, 24))
    _, gap = hp_filter_series(100.0 * output)
    assert np.nanmax(np.abs(gap.to_numpy())) < 1e-8


def test_add_hp_output_gap_uses_100_log_units_and_lag() -> None:
    data = pd.DataFrame({"output": np.linspace(1.0, 2.0, 24)})
    out = add_hp_output_gap(data)
    assert {"output_trend_HP", "output_gap_HP", "output_gap_HP_prev"}.issubset(out.columns)
    assert np.nanmax(np.abs(out["output_gap_HP"].to_numpy())) < 1e-8
    assert np.isnan(out.loc[0, "output_gap_HP_prev"])
    assert out.loc[1, "output_gap_HP_prev"] == pytest.approx(out.loc[0, "output_gap_HP"])


def test_labor_share_gap_is_loaded_in_100_log_units(tmp_path) -> None:
    raw_dir = tmp_path / "raw"
    laborshare = raw_dir / "laborshare"
    laborshare.mkdir(parents=True)
    dates = pd.date_range("2000-01-01", periods=24, freq="QS")
    values = np.exp(np.linspace(4.5, 4.8, 24))
    pd.DataFrame({"observation_date": dates, "PRS85006173": values}).to_csv(
        laborshare / "PRS85006173.csv",
        index=False,
    )

    loaded = load_labor_share_gap(raw_dir)
    assert {"labor_share", "labor_share_100log", "labor_share_trend_HP", "labor_share_gap_HP"}.issubset(loaded.columns)
    assert loaded["labor_share_100log"].iloc[0] == pytest.approx(450.0)
    assert np.nanmax(np.abs(loaded["labor_share_gap_HP"].to_numpy())) < 1e-8

    model_index = dates.to_period("Q").to_timestamp(how="end")
    data = pd.DataFrame({"output": np.arange(24.0)}, index=model_index)
    merged = add_labor_share_gap(data, raw_dir)
    assert "labor_share_gap_HP_prev" in merged.columns
    assert np.isnan(merged["labor_share_gap_HP_prev"].iloc[0])
