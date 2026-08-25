"""Two HSA-NKPC estimators on the indicator-allocated Gustavo x Capital IQ N (v2):

  Model 1 (fixed decomposition):  decompose N into Nbar/Nhat with a one-sided EWMA
                                  OUTSIDE the model, then estimate the HSA NKPC.
  Model 2 (joint decomposition):  estimate Nbar/Nhat as latent states TOGETHER with
                                  the HSA NKPC (state-space, FFBS).

Cells: activity in {inverse markup, negative unemployment gap} x inflation in
{PPI, core CPI}; expectations = SPF GDP-deflator forecast. HSA-restricted throughout.

    python tests/hsa_ppi_identification/two_models.py [--quick]
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd

import sys as _sys, pathlib as _pathlib  # noqa: E402
_ROOT = next(_p for _p in _pathlib.Path(__file__).resolve().parents if (_p / "pyproject.toml").exists())
_sys.path[:0] = [str(_ROOT), str(_ROOT / "src"), str(_ROOT / "tests")]
from tests import _bootstrap  # noqa: F401,E402
from nkpc_hsa.config import load_yaml  # noqa: E402
from tests.observed_hhi.functions import (  # noqa: E402
    ObservedHHISample, fit_observed_hhi_model, summarize_observed_fit, transform_inverse_hhi)
from tests.hsa_ppi_identification.functions import _load_frame, gustavo_capiq_quarterly_v2  # noqa: E402
from tests.hsa_ppi_identification.verify import joint_fit  # noqa: E402
from nkpc_hsa.report_models.cases import CaseData  # noqa: E402
from nkpc_hsa.report_models.engine import run_gibbs, build_priors  # noqa: E402


def _rs(v):
    v = np.asarray(v, float); v = v[np.isfinite(v)]
    s = float(np.subtract(*np.quantile(v, [0.75, 0.25])) / 1.349)
    return s if s > 0 else 1.0


def _band(M):
    return [np.round(M.mean(0), 4).tolist(), np.round(np.percentile(M, 2.5, 0), 4).tolist(),
            np.round(np.percentile(M, 97.5, 0), 4).tolist()]


def tvp_fixed(smp, sampling):
    """Time-varying kappa_t, theta_t from the FIXED (EWMA) decomposition, varying_theta."""
    fit = fit_observed_hhi_model(smp, cell=1, fast_definition="ewma_hl8", timing="lag1",
        model_variant="varying_theta", error_model="persistent_ar1", include_level=False,
        zeta_reference=ZETA, b_x=BX, iterations=sampling["iterations"], warmup=sampling["warmup"],
        thin=sampling["thin"], chains=sampling["chains"], seed=305)
    nm = list(fit.names); C = lambda n: fit.coefficients[:, :, nm.index(n)].reshape(-1)
    z = smp.q - np.nanmean(smp.q)
    KT = C("kappa_0")[:, None] + C("kappa_1")[:, None] * z[None, :]
    TT = C("theta_0")[:, None] + C("gamma")[:, None] * z[None, :]
    # downsampled draws + priors for prior-vs-posterior (varying_theta coeffs)
    def ds(v, k=1500):
        v = np.asarray(v); idx = np.linspace(0, len(v) - 1, min(k, len(v))).astype(int)
        return np.round(v[idx], 5).tolist()
    ppd = {p: ds(C(p)) for p in ["kappa_0", "kappa_1", "theta_0", "gamma"]}
    priors = {p: round(float(fit.prior_sds[p]), 4) for p in ["kappa_0", "kappa_1", "theta_0", "gamma"]}
    return {"kappa": _band(KT), "theta": _band(TT), "ppd": ppd, "prior_sds": priors}


def tvp_joint(d, sampling):
    """Time-varying kappa_t, theta_t from the JOINT (state-space, Model 4) decomposition."""
    q = (d["Gq"] - d["Gq"].mean()).to_numpy()
    data = CaseData(case=1, label="gc", periods=d.index, pi=d["y"].to_numpy(), epi=d["e"].to_numpy(),
        x=d["x"].to_numpy(), n_obs=q, exact_anchor=False, gE=None, s_x=_rs(d["x"]), s_N=_rs(q),
        s_pi=_rs(d["y"]), s_E=None, pi_lag=d["lag"].to_numpy())
    pr = build_priors(data, hybrid=True)
    res = run_gibbs(data, 4, iterations=sampling["iterations"], warmup=sampling["warmup"],
                    thin=sampling["thin"], chains=sampling["chains"], seed=405, priors=pr, hybrid=True)
    nm = list(res.coeff_names); C = lambda n: (res.coeffs[:, :, nm.index(n)].reshape(-1) if n in nm
                                               else np.zeros(res.coeffs.shape[0] * res.coeffs.shape[1]))
    nt = res.ntilde.reshape(-1, res.ntilde.shape[-1])
    KT = C("kappa_0")[:, None] + C("delta")[:, None] * nt
    TT = C("theta_0")[:, None] + C("gamma")[:, None] * nt
    return {"kappa": _band(KT), "theta": _band(TT)}

BUNDLE = Path(__file__).resolve().parent
BX, ZETA = 1.0, 6.0
INFL = {"ppi": ("pi_ppi", "pi_ppi_prev"), "core_cpi": ("pi_cpi_core", "pi_cpi_core_prev")}
ACT = {"inverse_markup": "markup_BN_inv", "neg_unemp_gap": "unemp_gap"}


def _pp(mean, sign):
    return sign if mean > 0 else 1 - sign


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--quick", action="store_true"); a = ap.parse_args()
    cfg = load_yaml(BUNDLE / "config.yaml")
    fixed = dict(cfg["sampling"]); jnt = dict(iterations=6000, warmup=2000, thin=3, chains=3)
    if a.quick:
        fixed.update(iterations=2000, warmup=800, thin=2, chains=2); jnt.update(iterations=1500, warmup=500, thin=2, chains=2)
    start = cfg["gustavo_capiq"]["samples"]["long"]
    out = BUNDLE / "results" / "two_models"; out.mkdir(parents=True, exist_ok=True)
    frame = _load_frame(); num = lambda c: pd.to_numeric(frame[c], errors="coerce")
    Gq, info = gustavo_capiq_quarterly_v2(frame, cfg["cell"]["competition"])
    R = {"wbar": info["wbar"], "cells": {}}
    for infl in ("ppi", "core_cpi"):
        y_col, lag_col = INFL[infl]
        for act in ("inverse_markup", "neg_unemp_gap"):
            key = f"{infl}|{act}"
            d = pd.concat({"y": num(y_col), "lag": num(lag_col), "e": num("Epi_spf_gdp"),
                           "x": num(ACT[act]), "Gq": Gq}, axis=1).dropna()
            d = d[d.index >= pd.Period(start, freq="Q")]
            # Model 1: fixed EWMA decomposition (observed), HSA-restricted, lag1, AR(1)
            smp = ObservedHHISample(periods=d.index, y=d["y"].to_numpy(float), pi_lag=d["lag"].to_numpy(float),
                expectation=d["e"].to_numpy(float), activity=d["x"].to_numpy(float),
                q=transform_inverse_hhi(np.exp(d["Gq"].to_numpy(float) / 10.0)),
                inflation=infl, activity_name=act, hhi_variant="gustavo_capiq_v2")
            fit = fit_observed_hhi_model(smp, cell=1, fast_definition="ewma_hl8", timing="lag1",
                model_variant="hsa_restricted", error_model="persistent_ar1", include_level=False,
                zeta_reference=ZETA, b_x=BX, iterations=fixed["iterations"], warmup=fixed["warmup"],
                thin=fixed["thin"], chains=fixed["chains"], seed=201)
            s = summarize_observed_fit(fit); r = s[s.parameter == "theta_hsa"].iloc[0]
            k = s[s.parameter == "kappa_0"].iloc[0]
            m1 = {"theta": [round(float(r["mean"]), 3), round(float(r["ci_2.5"]), 3), round(float(r["ci_97.5"]), 3),
                            round(_pp(r["mean"], r["sign_probability"]), 2), round(float(r["rhat"]), 3)],
                  "kappa_0": [round(float(k["mean"]), 3), round(_pp(k["mean"], k["sign_probability"]), 2)]}
            # Model 2: joint decomposition (state-space), HSA-restricted, hybrid
            j = joint_fit(d["Gq"].to_numpy(float), d["y"].to_numpy(float), d["e"].to_numpy(float),
                          d["x"].to_numpy(float), d["lag"].to_numpy(float), **jnt, seed=7)
            th = j["theta"]
            m2 = {"theta": [round(th[0], 3), round(th[0] - 1.96 * th[1], 3), round(th[0] + 1.96 * th[1], 3),
                            round(th[2], 2)], "kappa_0": [round(j["kappa_0"][0], 3), round(j["kappa_0"][1], 2)],
                  "delta_implied": round(j["delta_implied"], 3), "nbar": j["nbar"], "nhat": j["nhat"],
                  "periods": [str(p) for p in d.index]}
            # time-varying kappa_t, theta_t (fixed varying_theta and joint Model 4)
            tvf = tvp_fixed(smp, fixed)
            tvj = tvp_joint(d, jnt)
            R["cells"][key] = {"n": len(d), "periods": [str(p) for p in d.index],
                               "model1_fixed": m1, "model2_joint": m2,
                               "tvp_fixed": {"kappa": tvf["kappa"], "theta": tvf["theta"]},
                               "tvp_joint": tvj, "ppd": tvf["ppd"], "prior_sds": tvf["prior_sds"]}
            print(f"[{key}] n={len(d)}  fixed theta={m1['theta'][0]:+.3f}(P{m1['theta'][3]:.2f})  "
                  f"joint theta={m2['theta'][0]:+.3f}(P{m2['theta'][3]:.2f})", flush=True)
    (out / "two_models.json").write_text(json.dumps(R, indent=2), encoding="utf-8")
    print(f"\nwrote {out}/two_models.json", flush=True)


if __name__ == "__main__":
    main()
