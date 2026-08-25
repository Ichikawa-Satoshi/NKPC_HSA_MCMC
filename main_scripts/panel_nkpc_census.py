"""Manufacturing panel NKPC with CENSUS concentration (all firms) x marginal cost.

Fixes the underpowered Capital-IQ panel: Census Economic-Census HHI covers ALL
firms (public + private) for ~360 six-digit manufacturing industries, versus
~145 for Capital IQ public firms.  Concentration enters as a fixed industry
attribute interacted with the time-varying, theory-based marginal cost:

    pi_{it} = lambda_t + alpha_i + kappa * mc_{it} + delta * (N_i x mc_{it}) + eps_{it}

  pi : industry output price inflation = 100*dlog(piship)      [NBER-CES, annual]
  mc : marginal cost = log(materials/Y) or log(payroll/Y)      [NBER-CES]
  N  : competition = -log(HHI)  (higher = more competition)    [Census, all firms]

  N_i alone is absorbed by the industry FE; the interaction N_i x mc_{it} varies
  over time (via mc) and is identified from cross-industry differences in cost
  pass-through by concentration.  delta>0 = competition steepens pass-through (HSA).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

import sys as _sys, pathlib as _pathlib  # noqa: E402
_ROOT = next(_p for _p in _pathlib.Path(__file__).resolve().parents if (_p / "pyproject.toml").exists())
_sys.path[:0] = [str(_ROOT), str(_ROOT / "src")]
from nkpc_hsa.paths import data_root  # noqa: E402

SAMPLE = (1997, 2018)
CENSUS = data_root() / "raw" / "census_concentration"


def load_census_hhi() -> pd.DataFrame:
    frames = []
    for f, ncol in [("conc_2012/EC1231SR2.dat", "NAICS2012"),
                    ("conc_2007/EC0731SR12.dat", "NAICS2007")]:
        d = pd.read_csv(CENSUS / f, sep="|", dtype=str)
        d["hhi"] = pd.to_numeric(d["VSHERFI"], errors="coerce")
        d = d.dropna(subset=["hhi"])
        d = d[d[ncol].str.len() == 6]                     # 6-digit industries
        g = d.groupby(ncol)["hhi"].max().reset_index().rename(columns={ncol: "naics"})
        frames.append(g)
    # average the census years for a stable industry concentration
    hhi = pd.concat(frames).groupby("naics")["hhi"].mean().reset_index()
    hhi = hhi[hhi["hhi"] > 0].copy()
    hhi["N"] = -np.log(hhi["hhi"])                        # higher = more competition
    return hhi[["naics", "hhi", "N"]]


def load_nber() -> pd.DataFrame:
    d = pd.read_csv(data_root() / "raw" / "nber_ces" / "nberces5818v1_n1997.csv",
                    usecols=["naics", "year", "vship", "matcost", "pay", "piship"])
    d = d[d["naics"].astype(str).str[:2].isin(["31", "32", "33"])].copy()
    d = d.sort_values(["naics", "year"])
    d["naics"] = d["naics"].astype(str)
    d["pi"] = 100 * d.groupby("naics")["piship"].transform(lambda s: np.log(s).diff())
    d["mc_mat"] = np.log(d["matcost"] / d["vship"])
    d["mc_lab"] = np.log(d["pay"] / d["vship"])
    return d[["naics", "year", "pi", "mc_mat", "mc_lab"]]


def main() -> None:
    nber = load_nber()
    hhi = load_census_hhi()
    df = nber.merge(hhi, on="naics", how="inner")
    df = df[(df["year"] >= SAMPLE[0]) & (df["year"] <= SAMPLE[1])].dropna(subset=["pi", "mc_mat", "N"])
    cnt = df.groupby("naics")["year"].transform("count")
    df = df[cnt >= 8].copy()
    df["naics"] = df["naics"].astype("category")
    df["year"] = df["year"].astype("category")

    print(f"CENSUS-HHI panel: {df['naics'].nunique()} industries x {df['year'].nunique()} years, "
          f"{len(df)} obs, sample {SAMPLE[0]}-{SAMPLE[1]}")
    print(f"  competition N=-log(HHI) range [{df['N'].min():.2f},{df['N'].max():.2f}]  "
          f"(HHI [{df['hhi'].min():.0f},{df['hhi'].max():.0f}])\n")

    print(f"  {'marginal cost':14s} | {'kappa (mc, p)':20s} {'delta (N x mc, p)':22s}  [clusters]")
    for mc, label in [("mc_mat", "materials/Y"), ("mc_lab", "labor/Y")]:
        d2 = df.copy()
        d2["Nxmc"] = d2["N"] * d2[mc]
        m = smf.ols(f"pi ~ {mc} + Nxmc + C(naics) + C(year)", data=d2).fit(
            cov_type="cluster", cov_kwds={"groups": d2["naics"]})
        print(f"  {label:14s} | {m.params[mc]:+.3f} (p{m.pvalues[mc]:.3f})     "
              f"{m.params['Nxmc']:+.3f} (p{m.pvalues['Nxmc']:.3f})     [{df['naics'].nunique()}]")
    print("\n(delta>0 => more competition steepens cost pass-through = HSA-consistent)")
    print("Census HHI covers ALL firms; cluster count is now large enough for valid inference.")


if __name__ == "__main__":
    main()
