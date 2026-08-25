"""Nested-model comparison on the v2 Gustavo x Capital IQ competition series.

Estimates the five nested HSA models jointly (latent state-space, report_models):
  0 CES      delta=theta=gamma=0
  1 Slope    delta free              (kappa_t = kappa_0 + delta*Nbar_t)
  2 Direct   theta_0 free            (-theta_0*Nhat_t)
  3 Dynamic  theta_0, gamma free
  4 Joint    delta, theta_0, gamma free
for each of the four cells, and compares them by WAIC and Laplace-Metropolis log
marginal likelihood (exact Kalman for 0-2, particle filter for 3-4).

    python tests/hsa_ppi_identification/model_comparison.py [--quick]
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd, arviz as az

import sys as _sys, pathlib as _pathlib  # noqa: E402
_ROOT = next(_p for _p in _pathlib.Path(__file__).resolve().parents if (_p / "pyproject.toml").exists())
_sys.path[:0] = [str(_ROOT), str(_ROOT / "src"), str(_ROOT / "tests")]
from tests import _bootstrap  # noqa: F401,E402
from nkpc_hsa.config import load_yaml  # noqa: E402
from nkpc_hsa.report_models.cases import CaseData  # noqa: E402
from nkpc_hsa.report_models.engine import run_gibbs, build_priors  # noqa: E402
from nkpc_hsa.report_models.marginal_likelihood import laplace_metropolis_logml  # noqa: E402
from tests.hsa_ppi_identification.functions import _load_frame, gustavo_capiq_quarterly_v2  # noqa: E402

BUNDLE = Path(__file__).resolve().parent
INFL = {"ppi": ("pi_ppi", "pi_ppi_prev"), "core_cpi": ("pi_cpi_core", "pi_cpi_core_prev")}
ACT = {"inverse_markup": "markup_BN_inv", "neg_unemp_gap": "unemp_gap"}
MLAB = {0: "CES", 1: "Slope", 2: "Direct", 3: "Dynamic", 4: "Joint"}


def _rs(v):
    v = np.asarray(v, float); v = v[np.isfinite(v)]
    s = float(np.subtract(*np.quantile(v, [0.75, 0.25])) / 1.349); return s if s > 0 else 1.0


def _waic(res, data):
    pi, epi, x = data.pi, data.epi, data.x
    C, D = res.coeffs.shape[0], res.coeffs.shape[1]; names = res.coeff_names
    lp = np.zeros((C * D, pi.size)); idx = 0
    for c in range(C):
        for d in range(D):
            b = dict(zip(names, res.coeffs[c, d])); nt, nh = res.ntilde[c, d], res.nhat[c, d]
            mu = (b.get("alpha", 0) * epi + b.get("kappa_0", 0) * x + b.get("delta", 0) * x * nt
                  - b.get("theta_0", 0) * nh + b.get("gamma", 0) * nt * nh)
            if "intercept" in b:
                mu = mu + b["intercept"] + b.get("alpha_b", 0) * data.pi_lag
            s = res.sigma_pi[c, d]; lp[idx] = -0.5 * np.log(2 * np.pi * s ** 2) - 0.5 * ((pi - mu) / s) ** 2; idx += 1
    lppd = np.sum(np.log(np.mean(np.exp(lp - lp.max(0)), 0)) + lp.max(0)); pw = np.sum(np.var(lp, 0, ddof=1))
    return float(-2 * (lppd - pw))


def _draws(res):
    return {"coeff_names": np.asarray(res.coeff_names), "coeffs": res.coeffs, "rho": res.rho,
            "rho2": res.rho2 if res.rho2 is not None else np.zeros(0), "sigma_pi": res.sigma_pi,
            "sigma_bar": res.sigma_bar, "sigma_hat": res.sigma_hat,
            "sigma_nu": res.sigma_nu if res.sigma_nu is not None else np.zeros(0),
            "lambda_E": res.lambda_E if res.lambda_E is not None else np.zeros(0)}


TVP_COEFS = ["kappa_0", "delta", "theta_0", "gamma"]


def _band(M):
    return [np.round(M.mean(0), 4).tolist(), np.round(np.percentile(M, 2.5, 0), 4).tolist(),
            np.round(np.percentile(M, 97.5, 0), 4).tolist()]


def _ds(v, k=1500):
    v = np.asarray(v); idx = np.linspace(0, len(v) - 1, min(k, len(v))).astype(int)
    return np.round(v[idx], 5).tolist()


def run_cell(args):
    """Estimate Models 0-4 for one cell. Top-level for ProcessPool picklability."""
    key, cols, S = args
    data = CaseData(case=1, label=key, periods=cols["periods"], pi=np.asarray(cols["pi"]), epi=np.asarray(cols["e"]),
        x=np.asarray(cols["x"]), n_obs=np.asarray(cols["q"]), exact_anchor=False, gE=None,
        s_x=_rs(cols["x"]), s_N=_rs(cols["q"]), s_pi=_rs(cols["pi"]), s_E=None, pi_lag=np.asarray(cols["lag"]))
    pr = build_priors(data, hybrid=True)
    res_cell = {"n": len(cols["pi"]), "periods": [str(p) for p in cols["periods"]],
                "models": {}, "coeff_table": {}}
    for m in (0, 1, 2, 3, 4):
        res = run_gibbs(data, m, iterations=S["iterations"], warmup=S["warmup"], thin=S["thin"],
                        chains=S["chains"], seed=511, priors=pr, hybrid=True)
        nm = list(res.coeff_names)
        def cf(nme):
            if nme not in nm:
                return None
            f = res.coeffs[:, :, nm.index(nme)].reshape(-1)
            return [round(float(f.mean()), 3), round(float(np.mean(f > 0)), 2),
                    round(float(np.asarray(az.rhat(res.coeffs[:, :, nm.index(nme)], method="rank"))), 3)]
        def full(nme):  # mean, ci2.5, ci97.5, P>0, rhat
            f = res.coeffs[:, :, nm.index(nme)].reshape(-1)
            return [round(float(f.mean()), 3), round(float(np.percentile(f, 2.5)), 3),
                    round(float(np.percentile(f, 97.5)), 3), round(float(np.mean(f > 0)), 2),
                    round(float(np.asarray(az.rhat(res.coeffs[:, :, nm.index(nme)], method="rank"))), 3)]
        waic = _waic(res, data)
        try:
            logml = laplace_metropolis_logml(_draws(res), data, m, priors=pr, seed=511)
        except Exception:
            logml = float("nan")
        res_cell["models"][m] = {"label": MLAB[m], "waic": round(waic, 1), "log_ml": round(logml, 1),
                                 "delta": cf("delta"), "theta_0": cf("theta_0"), "gamma": cf("gamma"),
                                 "kappa_0": cf("kappa_0")}
        res_cell["coeff_table"][m] = {n: full(n) for n in nm}
        print(f"[{key}] M{m} {MLAB[m]:8} WAIC={waic:.1f} logML={logml:.1f} "
              f"delta={cf('delta')} theta_0={cf('theta_0')} gamma={cf('gamma')}", flush=True)
        if m == 4:  # Joint model: decomposition paths, time-varying coeffs, prior-vs-posterior
            C = lambda n: res.coeffs[:, :, nm.index(n)].reshape(-1)
            nt = res.ntilde.reshape(-1, res.ntilde.shape[-1])
            nh = res.nhat.reshape(-1, res.nhat.shape[-1])
            KT = C("kappa_0")[:, None] + C("delta")[:, None] * nt
            TT = C("theta_0")[:, None] + C("gamma")[:, None] * nt
            pmean, psd = pr.coeff_prior(tuple(TVP_COEFS))
            res_cell["joint"] = {
                "nbar": _band(nt), "nhat": _band(nh), "kappa": _band(KT), "theta": _band(TT),
                "ppd": {n: _ds(C(n)) for n in TVP_COEFS},
                "prior": {n: [round(float(pmean[i]), 4), round(float(psd[i]), 4)]
                          for i, n in enumerate(TVP_COEFS)}}
    return key, res_cell


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--quick", action="store_true")
    ap.add_argument("--workers", type=int, default=4); a = ap.parse_args()
    cfg = load_yaml(BUNDLE / "config.yaml")
    S = dict(iterations=6000, warmup=2000, thin=3, chains=4)
    if a.quick:
        S = dict(iterations=1500, warmup=500, thin=2, chains=2)
    out = BUNDLE / "results" / "model_comparison"; out.mkdir(parents=True, exist_ok=True)
    frame = _load_frame(); num = lambda c: pd.to_numeric(frame[c], errors="coerce")
    Gq, _ = gustavo_capiq_quarterly_v2(frame, cfg["cell"]["competition"])
    tasks = []
    for infl in ("ppi", "core_cpi"):
        y, lag = INFL[infl]
        for act in ("inverse_markup", "neg_unemp_gap"):
            key = f"{infl}|{act}"
            d = pd.concat({"pi": num(y), "lag": num(lag), "e": num("Epi_spf_gdp"),
                           "x": num(ACT[act]), "Gq": Gq}, axis=1).dropna()
            d = d[d.index >= pd.Period(cfg["gustavo_capiq"]["samples"]["long"], freq="Q")]
            q = (d["Gq"] - d["Gq"].mean()).to_numpy()
            cols = {"pi": d["pi"].to_numpy(), "lag": d["lag"].to_numpy(), "e": d["e"].to_numpy(),
                    "x": d["x"].to_numpy(), "q": q, "periods": d.index}
            tasks.append((key, cols, S))
    R = {}
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=min(a.workers, len(tasks))) as ex:
        for key, rc in ex.map(run_cell, tasks):
            R[key] = rc
    (out / "model_comparison.json").write_text(json.dumps(R, indent=2), encoding="utf-8")
    print(f"\nwrote {out}/model_comparison.json", flush=True)


if __name__ == "__main__":
    main()
