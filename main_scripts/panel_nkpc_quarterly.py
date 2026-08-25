"""Quarterly industry-panel NKPC (slack-based, two-way FE).

Marginal-cost data (NBER-CES) is annual, so the quarterly panel uses an
activity/slack forcing instead (as in Hazell et al. 2022 and Luengo-Prado et
al. 2017): industry real activity = nominal Capital IQ industry revenue growth
minus the industry's own PPI inflation.

    pi_{it} = lambda_t + alpha_i + kappa * x_{it} + delta * (N_{it} x x_{it})
              + theta * N_{it} + eps_{it}

  pi : quarterly industry PPI inflation = 100 * dlog(PCU index)   [BLS API]
  x  : real activity = dlog(revenue) - pi                          [Capital IQ - PPI]
  N  : competition = -log(HHI)                                     [Capital IQ]

BLS unregistered API limits (25 series/request, <=10yr/request) => a
manufacturing subset and decade-chunked requests.
"""

from __future__ import annotations

import json
import subprocess
import time

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

import sys as _sys, pathlib as _pathlib  # noqa: E402
_ROOT = next(_p for _p in _pathlib.Path(__file__).resolve().parents if (_p / "pyproject.toml").exists())
_sys.path[:0] = [str(_ROOT), str(_ROOT / "src")]
from nkpc_hsa.paths import data_root  # noqa: E402

MIN_FIRMS = 5
N_INDUSTRIES = 24
CHUNKS = [(1997, 2006), (2007, 2016), (2017, 2018)]
CACHE = data_root() / "raw" / "bls_ppi_quarterly.csv"
EMP_CACHE = data_root() / "raw" / "bls_emp_quarterly.csv"
# CES supersector: nondurable=32, durable=31 (by 3-digit NAICS)
NONDUR = {"311", "312", "313", "314", "315", "316", "322", "323", "324", "325", "326"}


def _hp_gap(level: pd.Series, lam: float = 1600.0) -> pd.Series:
    import scipy.sparse as sp
    from scipy.sparse.linalg import spsolve
    y = np.log(pd.to_numeric(level, errors="coerce").to_numpy(float))
    ok = np.isfinite(y)
    out = np.full(y.shape, np.nan)
    yy = y[ok]
    n = yy.size
    if n < 8:
        return pd.Series(out, index=level.index)
    D2 = sp.diags([1, -2, 1], [0, 1, 2], shape=(n - 2, n)).tocsc()
    trend = spsolve((sp.eye(n) + lam * (D2.T @ D2)).tocsc(), yy)
    out[ok] = (yy - trend) * 100
    return pd.Series(out, index=level.index)


def ces_id(naics6: str) -> str:
    ss = "32" if naics6[:3] in NONDUR else "31"
    return "CES" + ss + naics6 + "01"          # all employees, SA, thousands


def fetch_series(ids: list[str], cache) -> pd.DataFrame:
    if cache.exists():
        return pd.read_csv(cache, dtype={"sid": str})
    rows = []
    for lo, hi in CHUNKS:
        payload = json.dumps({"seriesid": ids, "startyear": str(lo), "endyear": str(hi)})
        out = subprocess.run(["curl", "-sL", "--max-time", "60", "-H", "Content-Type: application/json",
                              "-X", "POST", "https://api.bls.gov/publicAPI/v2/timeseries/data/", "-d", payload],
                             capture_output=True, text=True).stdout
        for s in json.loads(out)["Results"]["series"]:
            for r in s["data"]:
                if r["period"].startswith("M"):
                    rows.append({"sid": s["seriesID"], "year": int(r["year"]),
                                 "month": int(r["period"][1:]), "value": float(r["value"])})
        time.sleep(1)
    df = pd.DataFrame(rows)
    df.to_csv(cache, index=False)
    return df


def top_mfg_naics() -> list[str]:
    ciq = pd.read_csv(data_root() / "processed" / "capital_iq_company_quarter_revenues.csv",
                      usecols=["naics", "tq", "total_revenue_usd"])
    ciq = ciq[ciq["total_revenue_usd"] > 0]
    n = ciq["naics"].astype("Int64").astype(str)
    mfg = n[(n.str[:2].isin(["31", "32", "33"])) & (n.str.len() == 6)]
    return mfg.value_counts().head(N_INDUSTRIES).index.tolist()


def fetch_ppi(naics6: list[str]) -> pd.DataFrame:
    if CACHE.exists():
        return pd.read_csv(CACHE, dtype={"naics": str})
    ids = ["PCU" + c + c for c in naics6]
    rows = []
    for lo, hi in CHUNKS:
        payload = json.dumps({"seriesid": ids, "startyear": str(lo), "endyear": str(hi)})
        out = subprocess.run(["curl", "-sL", "--max-time", "60", "-H", "Content-Type: application/json",
                              "-X", "POST", "https://api.bls.gov/publicAPI/v2/timeseries/data/", "-d", payload],
                             capture_output=True, text=True).stdout
        d = json.loads(out)
        for s in d["Results"]["series"]:
            naics = s["seriesID"][3:9]
            for r in s["data"]:
                if r["period"].startswith("M"):
                    rows.append({"naics": naics, "year": int(r["year"]),
                                 "month": int(r["period"][1:]), "ppi": float(r["value"])})
        time.sleep(1)
    df = pd.DataFrame(rows)
    df.to_csv(CACHE, index=False)
    return df


def build() -> pd.DataFrame:
    naics6 = top_mfg_naics()
    ppi_m = fetch_ppi(naics6)
    ppi_m["q"] = (ppi_m["month"] - 1) // 3 + 1
    ppiq = ppi_m.groupby(["naics", "year", "q"])["ppi"].mean().reset_index()
    ppiq["t"] = ppiq["year"] * 4 + ppiq["q"]
    ppiq = ppiq.sort_values(["naics", "t"])
    ppiq["pi"] = 100 * ppiq.groupby("naics")["ppi"].transform(lambda s: np.log(s).diff())

    ciq = pd.read_csv(data_root() / "processed" / "capital_iq_company_quarter_revenues.csv",
                      usecols=["entity_id", "naics", "tq", "total_revenue_usd"])
    ciq = ciq[ciq["total_revenue_usd"] > 0].copy()
    ciq["naics"] = ciq["naics"].astype("Int64").astype(str)
    ciq = ciq[ciq["naics"].isin(naics6)]
    per = pd.PeriodIndex(ciq["tq"].astype(str).str.upper(), freq="Q")
    ciq["year"] = per.year; ciq["q"] = per.quarter; ciq["t"] = ciq["year"] * 4 + ciq["q"]
    firm = ciq.groupby(["naics", "t", "entity_id"])["total_revenue_usd"].sum().reset_index()
    tot = firm.groupby(["naics", "t"])["total_revenue_usd"].transform("sum")
    firm["sh2"] = (firm["total_revenue_usd"] / tot) ** 2
    ind = firm.groupby(["naics", "t"]).agg(hhi=("sh2", "sum"), n_firms=("entity_id", "count"),
                                           rev=("total_revenue_usd", "sum")).reset_index()
    ind = ind[ind["n_firms"] >= MIN_FIRMS].copy()
    ind = ind.sort_values(["naics", "t"])
    ind["rev_g"] = 100 * ind.groupby("naics")["rev"].transform(lambda s: np.log(s).diff())
    ind["N"] = -np.log(ind["hhi"])

    # --- theory-based marginal cost (NBER-CES annual, quarterized) as forcing ---
    nber = pd.read_csv(data_root() / "raw" / "nber_ces" / "nberces5818v1_n1997.csv",
                       usecols=["naics", "year", "vship", "matcost", "pay"])
    nber["naics"] = nber["naics"].astype(str)
    nber = nber[nber["naics"].isin(naics6)].copy()
    nber["mc_mat"] = np.log(nber["matcost"] / nber["vship"])   # Leith-Malley materials cost
    nber["mc_lab"] = np.log(nber["pay"] / nber["vship"])       # Gali-Gertler labor share
    # quarterize: linear interpolation of the annual cost within each industry
    q_index = pd.DataFrame([(n, y * 4 + q) for n in naics6 for y in range(1997, 2019) for q in (1, 2, 3, 4)],
                           columns=["naics", "t"])
    nber["t"] = nber["year"] * 4 + 2  # place annual value at Q2 (mid-ish)
    mc = q_index.merge(nber[["naics", "t", "mc_mat", "mc_lab"]], on=["naics", "t"], how="left")
    mc = mc.sort_values(["naics", "t"])
    for c in ("mc_mat", "mc_lab"):
        mc[c] = mc.groupby("naics")[c].transform(lambda s: s.interpolate(limit_direction="both"))

    df = ppiq.merge(ind, on=["naics", "t"], how="inner")
    df = df.merge(mc, on=["naics", "t"], how="inner").dropna(subset=["pi", "mc_mat", "N"])
    df["x"] = df["mc_mat"]                                  # primary forcing = materials cost
    df["Nxx"] = df["N"] * df["x"]
    df["Nxlab"] = df["N"] * df["mc_lab"]
    df = df[(df["year"] >= 1997) & (df["year"] <= 2018)]
    cnt = df.groupby("naics")["t"].transform("count")
    df = df[cnt >= 8].copy()
    df["naics"] = df["naics"].astype("category")
    df["t"] = df["t"].astype("category")
    return df


def main() -> None:
    df = build()
    print(f"quarterly panel: {df['naics'].nunique()} industries x {df['t'].nunique()} quarters, "
          f"{len(df)} obs, min_firms>={MIN_FIRMS}")
    print(f"  pi sd {df['pi'].std():.2f}  x sd {df['x'].std():.2f}  N range [{df['N'].min():.2f},{df['N'].max():.2f}]")
    specs = {
        "(1) materials baseline":       "pi ~ mc_mat + C(naics) + C(t)",
        "(2) materials + N":            "pi ~ mc_mat + N + C(naics) + C(t)",
        "(3) materials + interaction":  "pi ~ mc_mat + N + Nxx + C(naics) + C(t)",
        "(4) LABOR share + interaction":"pi ~ mc_lab + N + Nxlab + C(naics) + C(t)",
    }
    print("\ncoef [cluster SE by industry], p:")
    for name, f in specs.items():
        m = smf.ols(f, data=df).fit(cov_type="cluster", cov_kwds={"groups": df["naics"]})
        parts = [f"{v}={m.params[v]:+.4f}(p{m.pvalues[v]:.2f})"
                 for v in ("mc_mat", "mc_lab", "N", "Nxx", "Nxlab") if v in m.params.index]
        print(f"  {name:30s}: " + "  ".join(parts) + f"   [n={int(m.nobs)}]")
    for lab, iv in (("materials", "Nxx"), ("labor", "Nxlab")):
        f = f"pi ~ {'mc_mat' if lab=='materials' else 'mc_lab'} + N + {iv} + C(naics) + C(t)"
        m = smf.ols(f, data=df).fit(cov_type="cluster", cov_kwds={"groups": df["naics"]})
        print(f"\nHSA delta ({lab} cost) = {m.params[iv]:+.4f} (SE {m.bse[iv]:.4f}, p {m.pvalues[iv]:.3f})")


if __name__ == "__main__":
    main()
