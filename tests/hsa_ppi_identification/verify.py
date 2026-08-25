"""Verification battery for the Gustavo x Capital IQ HSA cell.

A1 joint decomposition (latent state-space) on the combined quarterly N, compared
with the fixed EWMA decomposition; A2 fixed-decomposition robustness (fast defs);
B2 Capital IQ firm vs revenue within-year profile; C3 timing x error. Saves results
and the joint Nbar/Nhat paths (D2) under results/verification/.

    python tests/hsa_ppi_identification/verify.py [--quick]
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
from nkpc_hsa.report_models.engine import _ffbs, _draw_coeffs, build_priors  # noqa: E402
from nkpc_hsa.report_models.cases import load_case  # noqa: E402
from nkpc_hsa.phillips.state import _draw_ig, _truncated_regression_draw  # noqa: E402
from tests.observed_hhi.functions import (  # noqa: E402
    ObservedHHISample, fit_observed_hhi_model, summarize_observed_fit, transform_inverse_hhi)
from tests.hsa_ppi_identification.functions import _load_frame, gustavo_capiq_quarterly  # noqa: E402

BUNDLE = Path(__file__).resolve().parent
BX, ZETA = 1.0, 6.0


def _pp(mean, sign):  # P(>0) from mean sign + sign_probability
    return sign if mean > 0 else 1 - sign


def joint_fit(Gq, pi, epi, x, pil, iterations, warmup, thin, chains, seed):
    """Latent [Nbar(RW), Nhat(AR1)] jointly with a hybrid, HSA-restricted NKPC.
    q_t = Nbar_t + Nhat_t + nu_t ; pi loads theta on (BX*ZETA*x*Nbar - Nhat)."""
    T = len(pi)
    q = Gq - np.nanmean(Gq)
    s_N = float(np.subtract(*np.quantile(q, [0.75, 0.25])) / 1.349) or 1.0
    s_x = float(np.subtract(*np.quantile(x, [0.75, 0.25])) / 1.349) or 1.0
    s_pi = float(np.subtract(*np.quantile(pi, [0.75, 0.25])) / 1.349) or 1.0
    # priors (weak)
    a_sd, ab_sd, k0_sd, th_sd, ic_sd = 0.5, 0.5, 0.5 / s_x, 0.2 / s_N, 2.0 * s_pi
    sig_pi_b = 2 * (0.75 * s_pi) ** 2; sig_bar_b = 2 * (0.05 * s_N) ** 2
    sig_hat_b = 2 * (0.20 * s_N) ** 2; sig_nu_b = 2 * (0.10 * s_N) ** 2
    ig = 3.0
    names = ["intercept", "alpha_b", "alpha", "kappa_0", "theta"]
    pm = np.array([0.0, 0.5, 0.9, 0.0, 0.0]); psd = np.array([ic_sd, ab_sd, a_sd, k0_sd, th_sd])
    nsave = (iterations - warmup + thin - 1) // thin
    keep = np.zeros((chains, nsave, 5)); ntacc = np.zeros(T); nhacc = np.zeros(T); nacc = 0
    for ch in range(chains):
        rng = np.random.default_rng(seed + ch * 7919)
        b = pm.copy(); sig2 = sig_pi_b / (ig - 1); rho = 0.5
        vbar = sig_bar_b / (ig - 1); vhat = sig_hat_b / (ig - 1); vnu = sig_nu_b / (ig - 1)
        nt = np.zeros(T); nh = np.zeros(T); save = 0
        for it in range(iterations):
            ic, ab, al, k0, th = b
            base = pi - ic - ab * pil - al * epi - k0 * x
            obsl = []
            for t in range(T):
                # inflation loads theta*(BX*ZETA*x*Nbar - Nhat): H=[th*BX*ZETA*x, -th]
                vals = [base[t]]; rows = [[th * BX * ZETA * x[t], -th]]; rv = [sig2]
                vals.append(q[t]); rows.append([1.0, 1.0]); rv.append(vnu)   # competition obs
                obsl.append((np.asarray(vals), np.asarray(rows), np.diag(rv)))
            P0 = np.diag([(2 * s_N) ** 2, vhat / max(1e-6, 1 - rho ** 2)])
            st = _ffbs(rng, F=np.array([[1., 0.], [0., rho]]), c=np.zeros((T, 2)),
                       Q=np.diag([vbar, vhat]), m0=np.zeros(2), P0=P0, observations=obsl)
            nt, nh = st[:, 0], st[:, 1]
            R = BX * ZETA * x * nt - nh
            X = np.column_stack([np.ones(T), pil, epi, x, R])
            b, sig2 = _draw_coeffs(rng, pi, X, pm, psd, sig2, ig, sig_pi_b)
            rho = _truncated_regression_draw(rng, nh[1:], nh[:-1], vhat, 0.5, 0.25)
            rb = np.diff(nt); vbar = _draw_ig(rng, ig + rb.size / 2, sig_bar_b + 0.5 * float(rb @ rb))
            rh = nh[1:] - rho * nh[:-1]; vhat = _draw_ig(rng, ig + rh.size / 2, sig_hat_b + 0.5 * float(rh @ rh))
            res = q - (nt + nh); vnu = _draw_ig(rng, ig + res.size / 2, sig_nu_b + 0.5 * float(res @ res))
            if it >= warmup and (it - warmup) % thin == 0:
                keep[ch, save] = b; save += 1
                ntacc += nt; nhacc += nh; nacc += 1
    flat = keep.reshape(-1, 5)
    th = flat[:, 4]; k0 = flat[:, 3]
    return {"theta": (float(th.mean()), float(th.std(ddof=1)), float(np.mean(th > 0)), th_sd),
            "delta_implied": float((th * BX * ZETA).mean()),
            "kappa_0": (float(k0.mean()), float(np.mean(k0 > 0))),
            "nbar": (ntacc / nacc).tolist(), "nhat": (nhacc / nacc).tolist()}


def obs_fit(frame, cfg, activity, profile_col, fast, timing, err, sampling):
    """Fixed-decomposition observed fit on the Gq built with a given Capital IQ profile."""
    Gq, _ = gustavo_capiq_quarterly(frame, profile_col)
    cell = cfg["cell"]; y_col, lag_col = cell["inflation"]; act_col = cfg["activities"][activity]["column"]
    num = lambda c: pd.to_numeric(frame[c], errors="coerce")
    d = pd.concat({"y": num(y_col), "lag": num(lag_col), "e": num(cell["expectation"]),
                   "x": num(act_col), "Gq": Gq}, axis=1).dropna()
    d = d[d.index >= pd.Period(cfg["gustavo_capiq"]["samples"]["long"], freq="Q")]
    smp = ObservedHHISample(periods=d.index, y=d["y"].to_numpy(float), pi_lag=d["lag"].to_numpy(float),
        expectation=d["e"].to_numpy(float), activity=d["x"].to_numpy(float),
        q=transform_inverse_hhi(np.exp(d["Gq"].to_numpy(float) / 10.0)), inflation="ppi",
        activity_name=activity, hhi_variant="gustavo_capiq")
    out = {}
    for var in ["constant_theta", "hsa_restricted"]:
        fit = fit_observed_hhi_model(smp, cell=1, fast_definition=fast, timing=timing, model_variant=var,
            error_model=err, include_level=False, zeta_reference=ZETA, b_x=BX,
            iterations=sampling["iterations"], warmup=sampling["warmup"], thin=sampling["thin"],
            chains=sampling["chains"], seed=101)
        s = summarize_observed_fit(fit)
        for p in (["kappa_0", "kappa_1", "theta_0"] if var == "constant_theta" else ["theta_hsa"]):
            r = s[s.parameter == p]
            if len(r):
                r = r.iloc[0]; out[p] = (round(float(r["mean"]), 3), round(float(r["ci_2.5"]), 3),
                                         round(float(r["ci_97.5"]), 3), round(_pp(r["mean"], r["sign_probability"]), 2))
    out["n"] = len(d); out["periods"] = (str(d.index.min()), str(d.index.max()))
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--quick", action="store_true"); a = ap.parse_args()
    cfg = load_yaml(BUNDLE / "config.yaml"); design = cfg["design"]
    samp = dict(cfg["sampling"])
    jsamp = dict(iterations=6000, warmup=2000, thin=3, chains=3)
    if a.quick:
        samp.update(iterations=2000, warmup=800, thin=2, chains=2); jsamp.update(iterations=1500, warmup=500, thin=2, chains=2)
    out = BUNDLE / "results" / "verification"; out.mkdir(parents=True, exist_ok=True)
    frame = _load_frame()
    Gq, info = gustavo_capiq_quarterly(frame, cfg["cell"]["competition"])
    num = lambda c: pd.to_numeric(frame[c], errors="coerce")
    R = {"shares": info["s"], "A1_joint": {}, "A2_fastdefs": {}, "B2_profile": {}, "C3_timing_error": {}}

    # ---- A1: joint decomposition, both activities ----
    for act in cfg["activities"]:
        act_col = cfg["activities"][act]["column"]; y_col, lag_col = cfg["cell"]["inflation"]
        d = pd.concat({"y": num(y_col), "lag": num(lag_col), "e": num(cfg["cell"]["expectation"]),
                       "x": num(act_col), "Gq": Gq}, axis=1).dropna()
        d = d[d.index >= pd.Period(cfg["gustavo_capiq"]["samples"]["long"], freq="Q")]
        r = joint_fit(d["Gq"].to_numpy(float), d["y"].to_numpy(float), d["e"].to_numpy(float),
                      d["x"].to_numpy(float), d["lag"].to_numpy(float), **jsamp, seed=7)
        r["periods"] = [str(p) for p in d.index]; r["n"] = len(d)
        R["A1_joint"][act] = r
        print(f"[A1 joint] {act}: theta={r['theta'][0]:+.3f} (P{r['theta'][2]:.2f}) delta_impl={r['delta_implied']:+.3f}", flush=True)

    # ---- A2: fixed-decomposition robustness (fast defs), inverse-markup + unemp ----
    for act in cfg["activities"]:
        R["A2_fastdefs"][act] = {}
        for fast in ["ewma_hl8", "ar1_innovation", "first_difference"]:
            R["A2_fastdefs"][act][fast] = obs_fit(frame, cfg, act, cfg["cell"]["competition"], fast, "lag1", "persistent_ar1", samp)
            print(f"[A2] {act}/{fast}: theta_hsa={R['A2_fastdefs'][act][fast].get('theta_hsa')}", flush=True)

    # ---- B2: Capital IQ revenue-weighted profile ----
    for act in cfg["activities"]:
        R["B2_profile"][act] = {"firm": R["A2_fastdefs"][act]["ewma_hl8"],
                                "revenue": obs_fit(frame, cfg, act, "N_capitaliq_revw", "ewma_hl8", "lag1", "persistent_ar1", samp)}
        print(f"[B2] {act} revenue: theta_hsa={R['B2_profile'][act]['revenue'].get('theta_hsa')}", flush=True)

    # ---- C3: timing x error ----
    for act in cfg["activities"]:
        R["C3_timing_error"][act] = {}
        for timing in ["current", "lag1"]:
            for err in ["iid", "persistent_ar1"]:
                R["C3_timing_error"][act][f"{timing}|{err}"] = obs_fit(frame, cfg, act, cfg["cell"]["competition"], "ewma_hl8", timing, err, samp)
        print(f"[C3] {act} done", flush=True)

    (out / "verification.json").write_text(json.dumps(R, indent=2), encoding="utf-8")
    print(f"\nwrote {out}/verification.json", flush=True)


if __name__ == "__main__":
    main()
