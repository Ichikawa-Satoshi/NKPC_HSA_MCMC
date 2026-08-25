"""Compare an AR(1) vs an AR(2) competition cycle Nhat by marginal likelihood.

For the primary specification (PPI / inverse markup) of each of the four cases,
this estimates every model 0-4 with the fast state Nhat as an AR(1) process and,
separately, as an AR(2) process (companion form), then reports the Laplace-
Metropolis log marginal likelihood, WAIC, and convergence diagnostics side by
side. The marginal likelihood is exact (Kalman) for the linear-Gaussian models
0-2 and particle-filter based (noisier) for the bilinear models 3-4, so the
headline AR(1)-vs-AR(2) evidence is read off models 0-2.

Usage:
    python main_scripts/report_ar_order_comparison.py --quick
    python main_scripts/report_ar_order_comparison.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd

import sys as _sys, pathlib as _pathlib  # noqa: E402
_ROOT = next(_p for _p in _pathlib.Path(__file__).resolve().parents if (_p / "pyproject.toml").exists())
_sys.path[:0] = [str(_ROOT), str(_ROOT / "src")]

from nkpc_hsa.paths import results_root  # noqa: E402
from nkpc_hsa.report_models import available_specs, build_priors, load_case, run_gibbs  # noqa: E402
from nkpc_hsa.report_models.cases import _load_frame  # noqa: E402
from nkpc_hsa.report_models.marginal_likelihood import laplace_metropolis_logml  # noqa: E402

MODELS = (0, 1, 2, 3, 4)
MODEL_LABELS = {0: "CES", 1: "Slope", 2: "Direct", 3: "Dynamic", 4: "Joint"}
EXACT_ML = {0, 1, 2}  # Kalman-exact marginal likelihood; 3-4 use a particle filter
PRIMARY = {"inflation": "ppi", "forcing": "inverse_markup"}


def _primary_variant(case: int) -> str:
    specs = [s for s in available_specs(case)
             if s["inflation"] == PRIMARY["inflation"] and s["forcing"] == PRIMARY["forcing"]]
    # Case 1 offers several competition variants; the firm-weighted Capital IQ
    # aggregate is the headline. Cases 2-4 have the single Gustavo variant.
    for preferred in ("firm_weighted", "gustavo"):
        for s in specs:
            if s["variant"] == preferred:
                return preferred
    return specs[0]["variant"]


def _diag(values: np.ndarray) -> tuple[float, float]:
    r = float(np.asarray(az.rhat(values, method="rank")))
    b = float(np.asarray(az.ess(values, method="bulk")))
    return r, b


def _draws_dict(res) -> dict:
    return {
        "coeff_names": np.asarray(res.coeff_names), "coeffs": res.coeffs,
        "rho": res.rho, "rho2": res.rho2 if res.rho2 is not None else np.zeros(0),
        "sigma_pi": res.sigma_pi, "sigma_bar": res.sigma_bar, "sigma_hat": res.sigma_hat,
        "sigma_nu": res.sigma_nu if res.sigma_nu is not None else np.zeros(0),
        "lambda_E": res.lambda_E if res.lambda_E is not None else np.zeros(0),
    }


def _waic(res, data) -> float:
    pi, epi, x = data.pi, data.epi, data.x
    C, D = res.coeffs.shape[0], res.coeffs.shape[1]
    names = res.coeff_names
    lp = np.zeros((C * D, pi.size))
    idx = 0
    for c in range(C):
        for d in range(D):
            beta = dict(zip(names, res.coeffs[c, d]))
            nt, nh = res.ntilde[c, d], res.nhat[c, d]
            mu = beta.get("alpha", 0) * epi + beta.get("kappa_0", 0) * x \
                + beta.get("delta", 0) * x * nt - beta.get("theta_0", 0) * nh \
                + beta.get("gamma", 0) * nt * nh
            if "intercept" in beta:
                mu = mu + beta["intercept"] + beta.get("alpha_b", 0) * data.pi_lag
            s = res.sigma_pi[c, d]
            lp[idx] = -0.5 * np.log(2 * np.pi * s ** 2) - 0.5 * ((pi - mu) / s) ** 2
            idx += 1
    lppd = np.sum(np.log(np.mean(np.exp(lp - lp.max(axis=0)), axis=0)) + lp.max(axis=0))
    p_waic = np.sum(np.var(lp, axis=0, ddof=1))
    return float(-2 * (lppd - p_waic))


_WORKER_FRAME = None


def _init_worker() -> None:
    global _WORKER_FRAME
    _WORKER_FRAME = _load_frame()


def _stable_seed(base: int, case: int, model: int, ar_order: int) -> int:
    key = f"{case}|{model}|ar{ar_order}"
    return base + int(hashlib.sha256(key.encode()).hexdigest()[:8], 16) % 1_000_000_000


def _run_one(payload: dict) -> dict:
    case, model, ar_order = payload["case"], payload["model"], payload["ar_order"]
    sampling, out, base_seed = payload["sampling"], Path(payload["out"]), payload["seed"]
    frame = _WORKER_FRAME if _WORKER_FRAME is not None else _load_frame()
    variant = payload["variant"]
    t0 = time.perf_counter()
    data = load_case(case, PRIMARY["inflation"], PRIMARY["forcing"], variant, frame=frame)
    priors = build_priors(data)
    seed = _stable_seed(base_seed, case, model, ar_order)
    res = run_gibbs(data, model, iterations=sampling["iterations"], warmup=sampling["warmup"],
                    thin=sampling["thin"], chains=sampling["chains"], seed=seed, priors=priors,
                    ar_order=ar_order)
    spec_dir = out / f"case{case}" / f"ar{ar_order}"
    spec_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(spec_dir / f"model{model}.npz", **_draws_dict(res),
                        ntilde=res.ntilde, nhat=res.nhat, periods=np.asarray(res.periods))
    # diagnostics over the free coefficients + rho (+ rho2)
    max_rhat, min_ess = 1.0, np.inf
    for j in range(res.coeffs.shape[2]):
        r, b = _diag(res.coeffs[:, :, j]); max_rhat = max(max_rhat, r); min_ess = min(min_ess, b)
    r, b = _diag(res.rho); max_rhat = max(max_rhat, r); min_ess = min(min_ess, b)
    rho2_mean = rho2_lo = rho2_hi = float("nan")
    if res.rho2 is not None:
        r2, b2 = _diag(res.rho2); max_rhat = max(max_rhat, r2); min_ess = min(min_ess, b2)
        flat = res.rho2.reshape(-1)
        rho2_mean = float(flat.mean()); rho2_lo = float(np.quantile(flat, 0.025)); rho2_hi = float(np.quantile(flat, 0.975))
    waic = _waic(res, data)
    try:
        log_ml = laplace_metropolis_logml(_draws_dict(res), data, model, priors=priors, seed=seed)
    except Exception as exc:  # noqa: BLE001
        log_ml = float("nan")
    return {
        "case": case, "model": model, "model_label": MODEL_LABELS[model], "ar_order": ar_order,
        "variant": variant, "exact_ml": model in EXACT_ML,
        "rho1_mean": float(res.rho.reshape(-1).mean()),
        "rho2_mean": rho2_mean, "rho2_ci_2.5": rho2_lo, "rho2_ci_97.5": rho2_hi,
        "waic": waic, "log_ml": log_ml, "n_periods": data.n_periods,
        "max_rhat": max_rhat, "min_bulk_ess": float(min_ess),
        "elapsed_s": time.perf_counter() - t0,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cases", type=int, nargs="+", default=[1, 2, 3, 4])
    ap.add_argument("--models", type=int, nargs="+", default=list(MODELS))
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--iterations", type=int, default=6000)
    ap.add_argument("--warmup", type=int, default=2000)
    ap.add_argument("--thin", type=int, default=3)
    ap.add_argument("--chains", type=int, default=4)
    ap.add_argument("--seed", type=int, default=20260819)
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 4) - 1))
    ap.add_argument("--output-dir", type=Path, default=None)
    args = ap.parse_args()

    if args.quick:
        args.iterations, args.warmup, args.thin, args.chains = 1500, 500, 2, 2

    out = args.output_dir or (results_root() / "report_estimation" / "ar_order_comparison")
    out.mkdir(parents=True, exist_ok=True)
    sampling = {"iterations": args.iterations, "warmup": args.warmup,
                "thin": args.thin, "chains": args.chains}

    variants = {case: _primary_variant(case) for case in args.cases}
    tasks = [{"case": case, "model": model, "ar_order": ao, "variant": variants[case],
              "sampling": sampling, "out": str(out), "seed": args.seed}
             for case in args.cases for model in args.models for ao in (1, 2)]

    print(f"ar_order_comparison: {len(tasks)} fits, jobs={args.jobs}, sampling={sampling}", flush=True)
    rows = []
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.jobs, initializer=_init_worker) as ex:
        futures = [ex.submit(_run_one, t) for t in tasks]
        for n, fut in enumerate(as_completed(futures), 1):
            r = fut.result()
            rows.append(r)
            print(f"[{n}/{len(tasks)}] case{r['case']} M{r['model']} AR{r['ar_order']} "
                  f"logML={r['log_ml']:.2f} WAIC={r['waic']:.1f} rho2={r['rho2_mean']:.3f} "
                  f"maxRhat={r['max_rhat']:.3f} [{r['elapsed_s']:.0f}s]", flush=True)

    df = pd.DataFrame(rows).sort_values(["case", "model", "ar_order"])
    df.to_csv(out / "model_comparison.csv", index=False)

    # Pivot to AR(1) vs AR(2) per case x model.
    pv = []
    for (case, model), g in df.groupby(["case", "model"]):
        a1 = g[g.ar_order == 1].iloc[0]; a2 = g[g.ar_order == 2].iloc[0]
        pv.append({
            "case": case, "model": model, "model_label": MODEL_LABELS[model],
            "exact_ml": bool(a1.exact_ml),
            "log_ml_ar1": a1.log_ml, "log_ml_ar2": a2.log_ml,
            "delta_log_ml": a2.log_ml - a1.log_ml,
            "waic_ar1": a1.waic, "waic_ar2": a2.waic, "delta_waic": a2.waic - a1.waic,
            "rho2_mean": a2.rho2_mean, "rho2_ci_2.5": a2["rho2_ci_2.5"], "rho2_ci_97.5": a2["rho2_ci_97.5"],
            "max_rhat": max(a1.max_rhat, a2.max_rhat),
        })
    pvt = pd.DataFrame(pv).sort_values(["case", "model"])
    pvt.to_csv(out / "ar_order_comparison.csv", index=False)

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "mode": "quick" if args.quick else "production",
        "cases": args.cases, "models": args.models, "variants": variants,
        "sampling": {**sampling, "seed": args.seed}, "fits": len(rows),
        "elapsed_seconds": time.perf_counter() - started, "output_dir": str(out),
        "note": "log_ml is Kalman-exact for models 0-2, particle-filter (noisier) for 3-4.",
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("\n=== AR(2) - AR(1) log marginal likelihood (positive favours AR(2)) ===", flush=True)
    print(pvt[["case", "model_label", "exact_ml", "log_ml_ar1", "log_ml_ar2",
               "delta_log_ml", "rho2_mean", "max_rhat"]].to_string(index=False), flush=True)
    print(f"\nwrote {out}/ar_order_comparison.csv and model_comparison.csv "
          f"({len(rows)} fits in {manifest['elapsed_seconds']/60:.1f} min)", flush=True)


if __name__ == "__main__":
    main()
