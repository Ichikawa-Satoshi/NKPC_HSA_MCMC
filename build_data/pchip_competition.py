"""Minimal Python helper: PCHIP annual->quarterly interpolation of the BN_N series.

This is the ONE piece of the data build that is not done in STATA: STATA has no
native PCHIP (monotone cubic) interpolator, and the legacy competition series
(``N_Gustavo`` / ``N_TNIC`` and their Beveridge-Nelson cycle/trend) are defined
by exactly the SciPy ``PchipInterpolator`` used in the original pipeline.  To
keep the numbers identical we reuse SciPy here and hand STATA a tidy quarterly
CSV keyed by (year, quarter); STATA does everything else and the merge.

Output: data/processed/interim/competition_bn_quarterly.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator


def _dropbox_data() -> Path:
    home = Path.home()
    for base in (
        home / "Library" / "CloudStorage" / "Dropbox" / "NKPC_HSA_MCMC",
        home / "Dropbox" / "NKPC_HSA_MCMC",
    ):
        if (base / "data" / "raw").is_dir():
            return base / "data"
    raise SystemExit("Could not find Dropbox/NKPC_HSA_MCMC/data; set the path.")


def annual_to_quarterly_pchip(annual: pd.DataFrame, value_col: str) -> pd.Series:
    """Interpolate year-stamped annual knots to a quarter-end grid via PCHIP.

    Mirrors the original ``func_data_build.annual_to_quarterly_pchip``: annual
    knots (stamped 1 Jan) denote the whole year, so they are moved to year-end
    before interpolating onto the quarter-end grid -- otherwise Q4 would pick up
    the following year's value.
    """
    x = annual.copy()
    x["DATE"] = pd.to_datetime(x["DATE"])
    x = x.set_index("DATE").sort_index()
    a = x.asfreq("YS")
    a.index = a.index + pd.offsets.YearEnd(0)

    q_index = pd.date_range(a.index.min(), a.index.max(), freq="QE")
    merged = a.reindex(a.index.union(q_index)).sort_index()

    xi = merged.index.view("i8")
    mask = merged[value_col].notna().values
    f = PchipInterpolator(xi[mask].astype(float), merged.loc[mask, value_col].astype(float).values)
    out = pd.Series(f(xi.astype(float)), index=merged.index).reindex(q_index)
    return out


def _year_stamped(raw: pd.DataFrame, source_col: str, out_col: str) -> pd.DataFrame:
    df = raw.copy()
    df[out_col] = df[source_col]
    df["DATE"] = pd.to_datetime(pd.to_datetime(df["date"]).dt.year.astype(str) + "-01-01")
    return df[["DATE", out_col]].dropna()


def annual_observed_q4(annual: pd.DataFrame, value_col: str, q_index: pd.DatetimeIndex) -> pd.Series:
    """Place each annual value at that year's Q4 (year-end) quarter, else missing.

    This is the non-interpolated companion to ``annual_to_quarterly_pchip``: the
    knots are stamped at the same year-end position, so the returned series equals
    the PCHIP series exactly at Q4 and is NaN in Q1-Q3 -- the raw annual signal
    with its native quarterly timing preserved rather than filled in.
    """
    a = annual.copy()
    a["DATE"] = pd.to_datetime(a["DATE"])
    a = a.set_index("DATE").sort_index().asfreq("YS")
    a.index = a.index + pd.offsets.YearEnd(0)
    return a[value_col].reindex(q_index)


def build(base: Path) -> pd.DataFrame:
    comp = base / "raw" / "competition"
    frames: dict[str, pd.Series] = {}
    for stub, source in (("N_Gustavo", "BN_N_Gustavo_26.csv"), ("N_TNIC", "BN_N_TNIC_26.csv")):
        raw = pd.read_csv(comp / source)
        for suffix, col in (("", "original_series"), ("_BN_cycle", "cycle"), ("_BN_trend", "trend")):
            name = f"{stub}{suffix}"
            annual = _year_stamped(raw, col, name)
            interp = annual_to_quarterly_pchip(annual, name)
            frames[name] = interp                                        # PCHIP quarterly
            frames[f"{name}_annual_q4"] = annual_observed_q4(annual, name, interp.index)

    out = pd.concat(frames, axis=1)
    out.index = pd.to_datetime(out.index).to_period("Q")
    out = out.reset_index(names="quarter_period")
    out["year"] = out["quarter_period"].dt.year
    out["quarter"] = out["quarter_period"].dt.quarter
    cols = ["year", "quarter"] + list(frames.keys())
    return out[cols]


def main() -> None:
    base = _dropbox_data()
    out = build(base)
    dest = base / "processed" / "interim"
    dest.mkdir(parents=True, exist_ok=True)
    path = dest / "competition_bn_quarterly.csv"
    out.to_csv(path, index=False, float_format="%.12g")
    print(f"wrote {path} ({len(out)} quarters, {out.shape[1] - 2} series)")


if __name__ == "__main__":
    sys.exit(main())
