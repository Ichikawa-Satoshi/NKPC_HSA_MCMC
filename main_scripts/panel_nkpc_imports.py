"""Panel NKPC with EFFECTIVE competition = domestic HHI adjusted for import
penetration (China import shock, Autor-Dorn-Hanson), interacted with marginal cost.

Domestic Census HHI overstates concentration for import-exposed manufacturing.
Effective HHI ~= HHI_domestic * (1 - m)^2, where m is import penetration, so
effective competition N_eff = -log(HHI) - 2*log(1-m).  We test whether import
competition itself dampens cost pass-through and whether it changes the
domestic-concentration interaction.

  pi_{it} = lambda_t + alpha_i + kappa*mc + delta_dom*(N_dom x mc)
            + delta_imp*(m x mc)   [+ delta_eff*(N_eff x mc)]  + eps
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

import sys as _sys, pathlib as _pathlib  # noqa: E402
_ROOT = next(_p for _p in _pathlib.Path(__file__).resolve().parents if (_p / "pyproject.toml").exists())
_sys.path[:0] = [str(_ROOT), str(_ROOT / "src")]
from nkpc_hsa.paths import data_root  # noqa: E402
from main_scripts.panel_nkpc_census import load_census_hhi, load_nber  # reuse loaders

SAMPLE = (1997, 2014)  # Dorn China exposure ends 2014
IMP = data_root() / "raw" / "import_exposure"


def china_penetration_by_naics() -> pd.DataFrame:
    d = pd.read_stata(IMP / "sic87dd_exposure_9114.dta")
    d["sic87"] = d["sic87dd"].astype(int).astype(str).str.zfill(4)
    imp_cols = [c for c in d.columns if c.startswith("l_import_usch_")]
    long = d.melt(id_vars=["sic87", "market1997"], value_vars=imp_cols,
                  var_name="yc", value_name="china_imp")
    long["year"] = long["yc"].str[-4:].astype(int)
    long["m"] = long["china_imp"] / long["market1997"].replace(0, np.nan)   # China import penetration
    long = long[["sic87", "year", "m"]].dropna()

    cw = pd.read_csv(IMP / "conc_sic87_naics97.csv", dtype={"sic87": str, "naics97": str})
    cw["w"] = pd.to_numeric(cw["ship8797"], errors="coerce").fillna(1.0)
    cw = cw[["sic87", "naics97", "w"]]
    j = long.merge(cw, on="sic87", how="inner")
    # weighted average penetration per (naics97, year)
    j["mw"] = j["m"] * j["w"]
    g = j.groupby(["naics97", "year"]).agg(mw=("mw", "sum"), w=("w", "sum")).reset_index()
    g["m"] = g["mw"] / g["w"]
    return g.rename(columns={"naics97": "naics"})[["naics", "year", "m"]]


def main() -> None:
    nber = load_nber()
    hhi = load_census_hhi()
    pen = china_penetration_by_naics()
    df = nber.merge(hhi, on="naics", how="inner").merge(pen, on=["naics", "year"], how="inner")
    df = df[(df["year"] >= SAMPLE[0]) & (df["year"] <= SAMPLE[1])].dropna(subset=["pi", "mc_mat", "N", "m"])
    df["m"] = df["m"].clip(0, 0.9)
    df["N_eff"] = df["N"] - 2 * np.log(1 - df["m"])          # effective competition (imports raise it)
    cnt = df.groupby("naics")["year"].transform("count")
    df = df[cnt >= 6].copy()
    df["naics"] = df["naics"].astype("category")
    df["year"] = df["year"].astype("category")

    print(f"IMPORT-adjusted panel: {df['naics'].nunique()} industries x {df['year'].nunique()} years, "
          f"{len(df)} obs, {SAMPLE[0]}-{SAMPLE[1]}")
    print(f"  China import penetration m: mean {df['m'].mean():.3f}, p90 {df['m'].quantile(.9):.3f}, max {df['m'].max():.3f}")
    print(f"  N_dom range [{df['N'].min():.2f},{df['N'].max():.2f}]  N_eff range [{df['N_eff'].min():.2f},{df['N_eff'].max():.2f}]\n")

    df["Nmc"] = df["N"] * df["mc_mat"]
    df["mmc"] = df["m"] * df["mc_mat"]
    df["Neffmc"] = df["N_eff"] * df["mc_mat"]
    specs = {
        "(1) domestic HHI only":       "pi ~ mc_mat + Nmc + C(naics) + C(year)",
        "(2) + import interaction":    "pi ~ mc_mat + Nmc + mmc + C(naics) + C(year)",
        "(3) EFFECTIVE competition":   "pi ~ mc_mat + Neffmc + C(naics) + C(year)",
    }
    print("coef [cluster SE by industry], p:")
    for name, f in specs.items():
        m = smf.ols(f, data=df).fit(cov_type="cluster", cov_kwds={"groups": df["naics"]})
        parts = [f"{v}={m.params[v]:+.3f}(p{m.pvalues[v]:.2f})"
                 for v in ("mc_mat", "Nmc", "mmc", "Neffmc") if v in m.params.index]
        print(f"  {name:26s}: " + "  ".join(parts))
    print("\n  Nmc>0 / Neffmc>0 = competition steepens pass-through (HSA);")
    print("  mmc<0 = import competition dampens pass-through (real rigidity from foreign competition)")


if __name__ == "__main__":
    main()
