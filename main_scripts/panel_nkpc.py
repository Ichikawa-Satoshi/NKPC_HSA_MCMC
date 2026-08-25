"""Annual industry-panel NKPC with two-way (industry + year) fixed effects.

Design follows the added literature:
  * Hazell-Herreno-Nakamura-Steinsson (2022 QJE): year FE difference out expected
    inflation, the monetary regime, oil, and all aggregate shocks; cross-industry
    variation solves the simultaneity problem.
  * Leith-Malley (RESTAT): manufacturing sectoral NKPC with the *intermediate
    input cost share* (materials / gross output) as the marginal-cost forcing,
    and HHI (market power) entering price setting.
  * Luengo-Prado-Rao-Sheremirov (2017): sectoral Phillips curves.

    pi_{it} = lambda_t + alpha_i + kappa * mc_{it} + delta * (N_{it} x mc_{it})
              + theta * N_{it} + eps_{it}

  pi   : industry output price inflation  = 100 * dlog(piship)   [NBER-CES]
  mc   : marginal cost = log(materials cost / value of shipments) [NBER-CES]
  N    : competition = -log(HHI) (higher = more competition)      [Capital IQ]
  delta: does competition change cost pass-through (the HSA slope channel)?
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

import sys as _sys, pathlib as _pathlib  # noqa: E402
_ROOT = next(_p for _p in _pathlib.Path(__file__).resolve().parents if (_p / "pyproject.toml").exists())
_sys.path[:0] = [str(_ROOT), str(_ROOT / "src")]
from nkpc_hsa.paths import data_root  # noqa: E402

MIN_FIRMS = 5
SAMPLE = (1997, 2018)


NDIG = 4  # NAICS granularity for matching (4-digit industry group)


def load_nber() -> pd.DataFrame:
    d = pd.read_csv(data_root() / "raw" / "nber_ces" / "nberces5818v1_n1997.csv",
                    usecols=["naics", "year", "vship", "matcost", "pay", "vadd", "piship"])
    d = d[d["naics"].astype(str).str[:2].isin(["31", "32", "33"])].copy()  # manufacturing
    d = d.sort_values(["naics", "year"])
    d["pi6"] = 100 * d.groupby("naics")["piship"].transform(lambda s: np.log(s).diff())
    d["ind"] = d["naics"].astype(str).str[:NDIG]
    # aggregate 6-digit -> NDIG-digit: vship-weighted inflation, summed cost/output.
    # Theory-based marginal-cost measures (all per unit of gross output):
    #   mc_mat = materials cost / shipments        (Leith-Malley intermediate cost)
    #   mc_lab = payroll / shipments               (labor cost, Gali-Gertler share)
    #   mc_tot = (materials + payroll) / shipments  (total variable cost)
    g = d.dropna(subset=["pi6"]).groupby(["ind", "year"])
    agg = g.apply(lambda x: pd.Series({
        "pi": np.average(x["pi6"], weights=x["vship"]),
        "mc_mat": np.log(x["matcost"].sum() / x["vship"].sum()),
        "mc_lab": np.log(x["pay"].sum() / x["vship"].sum()),
        "mc_tot": np.log((x["matcost"].sum() + x["pay"].sum()) / x["vship"].sum()),
    })).reset_index()
    agg = agg.sort_values(["ind", "year"])
    agg["pi_lag"] = agg.groupby("ind")["pi"].shift(1)
    return agg


def build_ciq_hhi() -> pd.DataFrame:
    ciq = pd.read_csv(data_root() / "processed" / "capital_iq_company_quarter_revenues.csv",
                      usecols=["entity_id", "naics", "tq", "total_revenue_usd"])
    ciq = ciq[ciq["total_revenue_usd"] > 0].copy()
    ciq["naics"] = ciq["naics"].astype("Int64")
    ciq = ciq.dropna(subset=["naics"])
    ciq["ind"] = ciq["naics"].astype(int).astype(str).str[:NDIG]
    ciq["year"] = pd.PeriodIndex(ciq["tq"].astype(str).str.upper(), freq="Q").year
    firm = ciq.groupby(["ind", "year", "entity_id"])["total_revenue_usd"].sum().reset_index()
    tot = firm.groupby(["ind", "year"])["total_revenue_usd"].transform("sum")
    firm["sh2"] = (firm["total_revenue_usd"] / tot) ** 2
    hhi = firm.groupby(["ind", "year"]).agg(hhi=("sh2", "sum"),
                                            n_firms=("entity_id", "count")).reset_index()
    hhi = hhi[hhi["n_firms"] >= MIN_FIRMS].copy()
    hhi["N"] = -np.log(hhi["hhi"])                       # higher = more competition (effective firms, logs)
    return hhi[["ind", "year", "N", "hhi", "n_firms"]]


def main() -> None:
    nber = load_nber()
    hhi = build_ciq_hhi()
    df = nber.merge(hhi, on=["ind", "year"], how="inner")
    df = df[(df["year"] >= SAMPLE[0]) & (df["year"] <= SAMPLE[1])].dropna(subset=["pi", "mc_mat", "N"])
    cnt = df.groupby("ind")["year"].transform("count")
    df = df[cnt >= 5].copy()
    df["ind"] = df["ind"].astype("category")
    df["year"] = df["year"].astype("category")

    print(f"ANNUAL panel: {df['ind'].nunique()} industries x {df['year'].nunique()} years, "
          f"{len(df)} obs, sample {SAMPLE[0]}-{SAMPLE[1]}, min_firms>={MIN_FIRMS}")
    print("Theory-based marginal-cost forcings (per unit gross output). "
          "delta = N x mc interaction (HSA: delta>0):\n")
    print(f"  {'marginal cost':16s} | {'kappa (mc, p)':22s} {'delta (Nxmc, p)':22s} {'N (p)':16s}")
    for mc, label in [("mc_mat", "materials/Y"), ("mc_lab", "labor/Y"), ("mc_tot", "total var/Y")]:
        d2 = df.copy()
        d2["Nxmc"] = d2["N"] * d2[mc]
        formula = f"pi ~ {mc} + N + Nxmc + C(ind) + C(year)"
        m = smf.ols(formula, data=d2).fit(cov_type="cluster", cov_kwds={"groups": d2["ind"]})
        print(f"  {label:16s} | {m.params[mc]:+.3f} (p{m.pvalues[mc]:.2f})        "
              f"{m.params['Nxmc']:+.3f} (p{m.pvalues['Nxmc']:.2f})        "
              f"{m.params['N']:+.3f} (p{m.pvalues['N']:.2f})")
    print("\n(delta>0 => more competition steepens cost pass-through = HSA-consistent)")


if __name__ == "__main__":
    main()
