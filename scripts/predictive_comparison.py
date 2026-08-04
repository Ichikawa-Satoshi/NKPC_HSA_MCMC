"""Predictive model comparison: one-step-ahead (prequential) log predictive density.

Complements the Chib marginal likelihoods. For every model we score the SAME target
series -- inflation -- by its one-step-ahead predictive density,

    LPD = sum_t log p(pi_t | pi_{1:t-1}, x, [N_{1:t-1}]),

integrating over posterior draws (log-mean-exp across draws). States are integrated out
by the Kalman filter, so this is a genuine predictive score, not the circular plug-in
score (which reused posterior-mean states smoothed with the same pi).

The HSA models may use the firm count N as extra information; CES cannot. That
asymmetry is the economic question itself ("does the firm count help explain
inflation?"), and unlike the joint marginal likelihood it does not mix different
target variables: every model is scored on p(pi).

Also reports WAIC and PSIS-LOO on the pointwise inflation log-likelihood.

    python scripts/predictive_comparison.py [--draws 300]
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logsumexp

import _bootstrap  # noqa: F401
from _bootstrap import ROOT

from nkpc_hsa.config import configured_data_specs, load_model_config
from nkpc_hsa.inference.wrappers import ESTIMATION_REVISION, _coerce_model_data
from nkpc_hsa.data import transform_competition_series
from nkpc_hsa.data.transforms import DEFAULT_N_TRANSFORM

TAB = ROOT / "results" / "appendix_particle_gibbs" / "tables"
TAB.mkdir(parents=True, exist_ok=True)

MODELS = ["ces", "hsa_steady", "hsa_dynamic", "hsa_full"]
UNEMP = {"Headline CPI": "unemployment_gap",
         "Core CPI": "unemployment_gap_core",
         "PPI": "unemployment_gap_ppi"}


def find_run(model, spec, freq="quarterly_interpolated"):
    best = None
    for d in sorted(glob.glob(str(ROOT / "results" / "runs" / f"{model}_{spec}_baseline_*"))):
        mp = Path(d) / "metadata.json"
        if not mp.exists():
            continue
        try:
            m = json.load(open(mp))
        except Exception:
            continue
        if (m.get("model") == model and m.get("data_spec") == spec
                and m.get("competition_measurement_frequency") == freq
                and m.get("period", "full") == "full"
                and m.get("constraint_spec", "unrestricted") == "unrestricted"
                and str(m.get("estimation_revision", "")) == ESTIMATION_REVISION):
            best = Path(d)
    return best


def _flat(post, name):
    if name not in post:
        return None
    return np.asarray(post[name], dtype=float).reshape(-1)


def _log_norm(x, mu, var):
    return -0.5 * (np.log(2.0 * np.pi * var) + (x - mu) ** 2 / var)


def prequential_ces(post, d, idx):
    """CES: inflation is a static regression given params (no latent state)."""
    a = _flat(post, "alpha")[idx]
    k = _flat(post, "kappa")[idx]
    lam = _flat(post, "lambda_ez")
    lam = np.zeros_like(a) if lam is None else lam[idx]
    se = _flat(post, "sigma_e")[idx]
    sz = _flat(post, "sigma_zeta")[idx]
    eta2 = np.maximum(se ** 2 - (lam ** 2) * (sz ** 2), 1e-10)
    zeta = d["x"][None, :] - _flat(post, "phi_1")[idx][:, None] * d["x_prev"][None, :]
    mu = (a[:, None] * d["pi_prev"][None, :] + (1 - a[:, None]) * d["pi_expect"][None, :]
          + k[:, None] * d["x"][None, :] + lam[:, None] * zeta)
    return _log_norm(d["pi"][None, :], mu, eta2[:, None])   # (S,T)


def prequential_hsa(post, d, idx, kind):
    """HSA: Kalman filter over s=(Nhat, Nhat_lag, Nbar); score the inflation row's
    one-step-ahead predictive density, then update with (N, pi) both observed."""
    S, T = idx.size, d["pi"].size
    a = _flat(post, "alpha")[idx]
    lam = _flat(post, "lambda_ez"); lam = np.zeros(S) if lam is None else lam[idx]
    ph = _flat(post, "phi_1")[idx]
    se, sz = _flat(post, "sigma_e")[idx], _flat(post, "sigma_zeta")[idx]
    eta2 = np.maximum(se ** 2 - (lam ** 2) * (sz ** 2), 1e-10)
    r1, r2 = _flat(post, "rho_1")[idx], _flat(post, "rho_2")[idx]
    nd = _flat(post, "n")[idx]
    su2, sp2 = _flat(post, "sigma_u")[idx] ** 2, _flat(post, "sigma_eps")[idx] ** 2
    sN = _flat(post, "sigma_N"); sN2 = (np.full(S, 1e-6) if sN is None else sN[idx] ** 2)
    if kind == "steady":
        k0, dl = _flat(post, "kappa_0")[idx], _flat(post, "delta")[idx]
        th0 = np.zeros(S); gm = np.zeros(S)
    elif kind == "dynamic":
        k0 = _flat(post, "kappa")[idx]; dl = np.zeros(S)
        th0 = _flat(post, "theta")[idx]; gm = np.zeros(S)
    else:  # full / const_theta
        k0, dl = _flat(post, "kappa_0")[idx], _flat(post, "delta")[idx]
        t0 = _flat(post, "theta_0"); th0 = (_flat(post, "theta") if t0 is None else t0)[idx]
        g = _flat(post, "gamma"); gm = np.zeros(S) if g is None else g[idx]

    out = np.empty((S, T))
    for s in range(S):
        F = np.array([[r1[s], r2[s], 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        c = np.array([0.0, 0.0, nd[s]])
        Q = np.diag([su2[s], 1e-10, sp2[s]])
        m = np.array([0.0, 0.0, d["N"][0]]); P = np.eye(3) * 10.0
        zeta = d["x"] - ph[s] * d["x_prev"]
        for t in range(T):
            if t > 0:
                m = F @ m + c
                P = F @ P @ F.T + Q
            det = (a[s] * d["pi_prev"][t] + (1 - a[s]) * d["pi_expect"][t]
                   + k0[s] * d["x"][t] + lam[s] * zeta[t])
            # inflation row loading on the state (linearised for gamma at current mean)
            h = np.array([-(th0[s] + gm[s] * m[2]), 0.0, dl[s] * d["x"][t] - gm[s] * m[0]])
            mu = det + h @ m
            var = float(h @ P @ h + eta2[s])
            out[s, t] = _log_norm(d["pi"][t], mu, max(var, 1e-12))
            # update with BOTH observations available at t
            H = np.vstack([np.array([1.0, 0.0, 1.0]), h])
            y = np.array([d["N"][t], d["pi"][t] - det])
            R = np.diag([sN2[s], eta2[s]])
            Sm = H @ P @ H.T + R
            K = P @ H.T @ np.linalg.inv(Sm)
            m = m + K @ (y - H @ m)
            P = (np.eye(3) - K @ H) @ P
            P = (P + P.T) / 2.0
    return out


def scores(ll):
    """ll: (S,T) pointwise log-lik. Returns LPD, WAIC, PSIS-LOO, max Pareto k."""
    import arviz as az
    S, T = ll.shape
    lpd_t = logsumexp(ll, axis=0) - np.log(S)
    lpd = float(lpd_t.sum())
    p_waic = float(np.var(ll, axis=0, ddof=1).sum())
    waic = float(lpd - p_waic)
    try:
        dt = az.from_dict({"posterior": {"_d": np.zeros((1, S))},
                           "log_likelihood": {"pi": ll[None, :, :]}})
        r = az.loo(dt, var_name="pi")
        loo = float(np.asarray(r.elpd if hasattr(r, "elpd") else r.elpd_loo))
        kmax = float(np.nanmax(np.asarray(r.pareto_k))) if hasattr(r, "pareto_k") else float("nan")
    except Exception:
        loo, kmax = float("nan"), float("nan")
    return lpd, waic, loo, kmax


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--draws", type=int, default=300)
    args = ap.parse_args()
    specs = configured_data_specs(load_model_config())
    df = pd.read_csv(ROOT / "data" / "processed" / "model_ready.csv")
    import arviz as az

    rows = []
    for price, spec in UNEMP.items():
        md = _coerce_model_data(df, data_spec=specs[spec])
        d = {k: np.asarray(md[k], float) for k in ["pi", "pi_prev", "pi_expect", "x", "x_prev"]}
        d["N"] = np.asarray(transform_competition_series(md["N"], transform=DEFAULT_N_TRANSFORM), float)
        base = None
        for model in MODELS:
            run = find_run(model, spec)
            if run is None:
                continue
            post = az.from_netcdf(run / "posterior.nc").posterior
            n_all = _flat(post, "alpha").size
            idx = np.linspace(0, n_all - 1, min(args.draws, n_all)).astype(int)
            if model == "ces":
                ll = prequential_ces(post, d, idx)
            else:
                kind = {"hsa_steady": "steady", "hsa_dynamic": "dynamic"}.get(model, "full")
                ll = prequential_hsa(post, d, idx, kind)
            lpd, waic, loo, kmax = scores(ll)
            if model == "ces":
                base = lpd
            rows.append({"price": price, "model": model,
                         "LPD_1step": round(lpd, 2), "WAIC": round(waic, 2),
                         "LOO": round(loo, 2), "max_pareto_k": round(kmax, 2),
                         "dLPD_vs_ces": None if base is None else round(lpd - base, 2)})
            print(f"{price:13s} {model:12s} LPD={lpd:9.2f}  WAIC={waic:9.2f}  LOO={loo:9.2f}  k={kmax:4.2f}"
                  + ("" if base is None else f"  dLPD vs CES={lpd-base:+7.2f}"), flush=True)
    tab = pd.DataFrame(rows)
    tab.to_csv(TAB / "predictive_comparison.csv", index=False)
    json.dump(rows, open(TAB / "predictive_comparison.json", "w"), indent=2)
    print(f"\nsaved -> {TAB/'predictive_comparison.csv'}")


if __name__ == "__main__":
    main()
