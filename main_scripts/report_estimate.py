"""Estimate the four-case, five-model HSA design of docs/report/estimation/edstimation.tex.

Runs Gibbs/FFBS for each (case, model, empirical specification), saves posterior
draws and coefficient/decomposition summaries under the Dropbox results tree, and
writes a manifest.

Usage:
    python main_scripts/report_estimate.py --pilot            # fast validation subset
    python main_scripts/report_estimate.py                    # full 4x5x12 grid (production)
    python main_scripts/report_estimate.py --cases 3 --iterations 2000
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
PRIMARY_SPEC = {"inflation": "ppi", "forcing": "inverse_markup"}


def _diag(values: np.ndarray) -> tuple[float, float]:
    r = float(np.asarray(az.rhat(values, method="rank")))
    b = float(np.asarray(az.ess(values, method="bulk")))
    return r, b


def _coeff_summary(res, case: int, spec: dict) -> list[dict]:
    rows = []
    flat_by_name = {}
    for j, name in enumerate(res.coeff_names):
        vals = res.coeffs[:, :, j]
        flat = vals.reshape(-1)
        flat_by_name[name] = flat
        r, b = _diag(vals)
        # HSA-consistent region (theta positive convention): delta>0, theta_0>0, gamma>0.
        sign_prob = (float(np.mean(flat > 0))
                     if name in ("delta", "gamma", "kappa_0", "theta_0") else float("nan"))
        rows.append({
            "case": case, "model": res.model, "model_label": MODEL_LABELS[res.model],
            "inflation": spec["inflation"], "forcing": spec["forcing"], "variant": spec["variant"],
            "parameter": name, "mean": float(flat.mean()), "sd": float(flat.std(ddof=1)),
            "ci_2.5": float(np.quantile(flat, 0.025)), "median": float(np.quantile(flat, 0.5)),
            "ci_97.5": float(np.quantile(flat, 0.975)), "sign_probability": sign_prob,
            "rhat": r, "bulk_ess": b,
        })
    # Scalar auxiliaries.
    for name, vals in (("rho_N", res.rho), ("sigma_pi", res.sigma_pi),
                       ("sigma_bar", res.sigma_bar), ("sigma_hat", res.sigma_hat)):
        flat = vals.reshape(-1)
        r, b = _diag(vals)
        rows.append({
            "case": case, "model": res.model, "model_label": MODEL_LABELS[res.model],
            "inflation": spec["inflation"], "forcing": spec["forcing"], "variant": spec["variant"],
            "parameter": name, "mean": float(flat.mean()), "sd": float(flat.std(ddof=1)),
            "ci_2.5": float(np.quantile(flat, 0.025)), "median": float(np.quantile(flat, 0.5)),
            "ci_97.5": float(np.quantile(flat, 0.975)), "sign_probability": float("nan"),
            "rhat": r, "bulk_ess": b,
        })
    if res.lambda_E is not None:
        flat = res.lambda_E.reshape(-1)
        r, b = _diag(res.lambda_E)
        rows.append({
            "case": case, "model": res.model, "model_label": MODEL_LABELS[res.model],
            "inflation": spec["inflation"], "forcing": spec["forcing"], "variant": spec["variant"],
            "parameter": "lambda_E", "mean": float(flat.mean()), "sd": float(flat.std(ddof=1)),
            "ci_2.5": float(np.quantile(flat, 0.025)), "median": float(np.quantile(flat, 0.5)),
            "ci_97.5": float(np.quantile(flat, 0.975)),
            "sign_probability": float(np.mean(flat > 0)),
            "rhat": r, "bulk_ess": b,
        })
    return rows, flat_by_name


def _joint_probability(flat_by_name: dict) -> dict:
    have = lambda n: n in flat_by_name  # noqa: E731
    out = {}
    if have("delta"):
        out["p_delta_pos"] = float(np.mean(flat_by_name["delta"] > 0))
    if have("theta_0"):
        out["p_theta0_pos"] = float(np.mean(flat_by_name["theta_0"] > 0))
    if have("gamma"):
        out["p_gamma_pos"] = float(np.mean(flat_by_name["gamma"] > 0))
    if have("delta") and have("theta_0") and have("gamma"):
        joint = (flat_by_name["delta"] > 0) & (flat_by_name["theta_0"] > 0) & (flat_by_name["gamma"] > 0)
        out["p_joint_hsa"] = float(np.mean(joint))
    return out


def _draws_dict(res) -> dict:
    return {
        "coeff_names": np.asarray(res.coeff_names), "coeffs": res.coeffs,
        "rho": res.rho, "sigma_pi": res.sigma_pi,
        "sigma_bar": res.sigma_bar, "sigma_hat": res.sigma_hat,
        "sigma_nu": res.sigma_nu if res.sigma_nu is not None else np.zeros(0),
        "lambda_E": res.lambda_E if res.lambda_E is not None else np.zeros(0),
    }


def _waic(res, data) -> float:
    """WAIC on the inflation equation (lower is better)."""
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


# --------------------------------------------------------------------------- #
# parallel worker
# --------------------------------------------------------------------------- #
_WORKER_FRAME = None


def _init_worker() -> None:
    """Load the model-ready frame once per worker process (spawn-safe)."""
    global _WORKER_FRAME
    _WORKER_FRAME = _load_frame()


def _stable_seed(base: int, case: int, spec: dict, model: int) -> int:
    key = f"{case}|{spec['inflation']}|{spec['forcing']}|{spec['variant']}|{model}"
    return base + int(hashlib.sha256(key.encode()).hexdigest()[:8], 16) % 1_000_000_000


def _run_one(payload: dict) -> dict:
    case, spec, model = payload["case"], payload["spec"], payload["model"]
    sampling, out, base_seed = payload["sampling"], Path(payload["out"]), payload["seed"]
    frame = _WORKER_FRAME if _WORKER_FRAME is not None else _load_frame()
    hybrid = bool(payload.get("hybrid", False))
    t0 = time.perf_counter()
    try:
        data = load_case(case, spec["inflation"], spec["forcing"], spec["variant"], frame=frame)
    except Exception as exc:  # noqa: BLE001
        return {"skip": f"case{case} {spec['inflation']}/{spec['forcing']}/{spec['variant']}: {exc}"}
    priors = build_priors(data, hybrid=hybrid)
    seed = _stable_seed(base_seed, case, spec, model)
    res = run_gibbs(data, model, iterations=sampling["iterations"], warmup=sampling["warmup"],
                    thin=sampling["thin"], chains=sampling["chains"], seed=seed, priors=priors, hybrid=hybrid)
    spec_dir = out / f"case{case}" / f"{spec['inflation']}__{spec['forcing']}__{spec['variant']}"
    spec_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        spec_dir / f"model{model}.npz",
        coeffs=res.coeffs, coeff_names=np.asarray(res.coeff_names),
        sigma_pi=res.sigma_pi, rho=res.rho,
        lambda_E=res.lambda_E if res.lambda_E is not None else np.zeros(0),
        sigma_bar=res.sigma_bar, sigma_hat=res.sigma_hat,
        sigma_nu=res.sigma_nu if res.sigma_nu is not None else np.zeros(0),
        ntilde=res.ntilde, nhat=res.nhat, periods=np.asarray(res.periods),
        pi=data.pi, epi=data.epi, x=data.x, n_obs=data.n_obs,
    )
    rows, flat_by_name = _coeff_summary(res, case, spec)
    waic = _waic(res, data)
    try:
        log_ml = laplace_metropolis_logml(_draws_dict(res), data, model, priors=priors, seed=seed)
    except Exception:  # noqa: BLE001
        log_ml = float("nan")
    cmp_row = {
        "case": case, "model": model, "model_label": MODEL_LABELS[model],
        "inflation": spec["inflation"], "forcing": spec["forcing"], "variant": spec["variant"],
        "waic": waic, "log_ml": log_ml, "n_periods": data.n_periods,
        "elapsed_s": time.perf_counter() - t0,
        **_joint_probability(flat_by_name),
    }
    return {"rows": rows, "cmp": cmp_row}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cases", type=int, nargs="+", default=[1, 2, 3, 4])
    ap.add_argument("--models", type=int, nargs="+", default=list(MODELS))
    ap.add_argument("--pilot", action="store_true", help="primary spec only, reduced iterations")
    ap.add_argument("--primary-spec-only", action="store_true")
    ap.add_argument("--iterations", type=int, default=12000)
    ap.add_argument("--warmup", type=int, default=4000)
    ap.add_argument("--thin", type=int, default=4)
    ap.add_argument("--chains", type=int, default=4)
    ap.add_argument("--seed", type=int, default=20260819)
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 4) - 1))
    ap.add_argument("--hybrid", action="store_true",
                    help="hybrid NKPC: add an intercept and lagged inflation pi_{t-1}")
    ap.add_argument("--output-dir", type=Path, default=None)
    args = ap.parse_args()

    if args.pilot:
        args.iterations, args.warmup, args.thin, args.chains = 2000, 800, 2, 2
        args.primary_spec_only = True

    mode = ("pilot" if args.pilot else "production") + ("_hybrid" if args.hybrid else "")
    out = args.output_dir or (results_root() / "report_estimation" / mode)
    out.mkdir(parents=True, exist_ok=True)

    sampling = {"iterations": args.iterations, "warmup": args.warmup,
                "thin": args.thin, "chains": args.chains}
    tasks: list[dict] = []
    for case in args.cases:
        specs = available_specs(case)
        if args.pilot or args.primary_spec_only:
            specs = [s for s in specs if s["inflation"] == PRIMARY_SPEC["inflation"]
                     and s["forcing"] == PRIMARY_SPEC["forcing"]]
        for spec in specs:
            for model in args.models:
                tasks.append({"case": case, "spec": spec, "model": model, "hybrid": args.hybrid,
                              "sampling": sampling, "out": str(out), "seed": args.seed})

    print(f"report_estimate: {len(tasks)} fits, jobs={args.jobs}, sampling={sampling}", flush=True)
    all_summary: list[dict] = []
    model_cmp: list[dict] = []
    started = time.perf_counter()
    n_done = 0
    with ProcessPoolExecutor(max_workers=args.jobs, initializer=_init_worker) as ex:
        futures = [ex.submit(_run_one, t) for t in tasks]
        for fut in as_completed(futures):
            result = fut.result()
            if "skip" in result:
                print(f"[skip] {result['skip']}", flush=True)
                continue
            all_summary.extend(result["rows"])
            model_cmp.append(result["cmp"])
            n_done += 1
            c = result["cmp"]
            elapsed = time.perf_counter() - started
            print(f"[{n_done}/{len(tasks)}] case{c['case']} {c['inflation']}/{c['forcing']}/{c['variant']} "
                  f"M{c['model']} WAIC={c['waic']:.1f} logML={c['log_ml']:.1f} "
                  f"[{c['elapsed_s']:.0f}s | wall {elapsed/60:.1f}m]", flush=True)

    summary_df = pd.DataFrame(all_summary).sort_values(["case", "variant", "inflation", "forcing", "model", "parameter"])
    cmp_df = pd.DataFrame(model_cmp).sort_values(["case", "variant", "inflation", "forcing", "model"])
    summary_df.to_csv(out / "coefficient_summaries.csv", index=False)
    cmp_df.to_csv(out / "model_comparison.csv", index=False)
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "mode": "pilot" if args.pilot else "production",
        "cases": args.cases, "models": args.models, "jobs": args.jobs,
        "sampling": {**sampling, "seed": args.seed},
        "fits": n_done, "elapsed_seconds": time.perf_counter() - started,
        "output_dir": str(out),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\nwrote {out}/coefficient_summaries.csv ({len(summary_df)} rows), "
          f"model_comparison.csv ({len(cmp_df)} rows); {n_done} fits in "
          f"{manifest['elapsed_seconds']/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
