"""Estimate the theory-near HSA cell and write a self-contained result bundle.

Cell: Capital IQ firm-weighted competition x PPI x inverse-markup x SPF
expectations. Competition-level term (psi) EXCLUDED (theory-faithful). Three
samples (full / primary=drop DB-coverage ramp / conservative) x three model
variants (constant_theta, varying_theta, hsa_restricted).

    python tests/hsa_ppi_identification/run.py --quick   # smoke -> results/smoke/
    python tests/hsa_ppi_identification/run.py            # full  -> results/
"""
from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import sys as _sys, pathlib as _pathlib  # noqa: E402
_ROOT = next(_p for _p in _pathlib.Path(__file__).resolve().parents if (_p / "pyproject.toml").exists())
_sys.path[:0] = [str(_ROOT), str(_ROOT / "src"), str(_ROOT / "tests")]
from tests import _bootstrap  # noqa: F401,E402

from nkpc_hsa.config import load_yaml  # noqa: E402
from tests.hsa_ppi_identification.functions import (  # noqa: E402
    _load_frame, build_sample, fit_cell, effective_slope, HSA_PARAMS,
)

BUNDLE = Path(__file__).resolve().parent
BLUE, ORANGE, GREEN = "#0072B2", "#D55E00", "#009E73"


def _run_task(payload: dict) -> dict:
    config = load_yaml(payload["config_path"])
    frame = _load_frame()
    sample = build_sample(frame, config, payload["activity"], payload["sample_start"])
    s = fit_cell(sample, payload["variant"], config["design"], payload["sampling"])
    s["sample"] = payload["sample_name"]
    s["activity"] = payload["activity"]
    return {"summary": s.to_dict(orient="records"), "n": int(len(sample.y)),
            "first": str(sample.periods.min()), "last": str(sample.periods.max()),
            "activity": payload["activity"], "sample_name": payload["sample_name"]}


def _plots(df: pd.DataFrame, frame, config, figures: Path):
    figures.mkdir(parents=True, exist_ok=True)
    pa = config.get("primary_activity", "inverse_markup")
    df = df[df["activity"] == pa]
    # (1) delta (kappa_1) and theta_hsa across samples, primary emphasis
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for ax, (param, variant, title) in zip(axes, [
        ("kappa_1", "constant_theta", r"$\delta=\kappa_1$ (competition $\times$ slope)"),
        ("theta_hsa", "hsa_restricted", r"$\theta$ (HSA-restricted)")]):
        sub = df[(df.parameter == param) & (df.variant == variant)].sort_values("sample")
        ys = np.arange(len(sub))
        ax.errorbar(sub["mean"], ys, xerr=[sub["mean"] - sub["ci_2.5"], sub["ci_97.5"] - sub["mean"]],
                    fmt="o", color=BLUE, capsize=3)
        ax.axvline(0, color="black", ls="--", lw=0.8)
        ax.set_yticks(ys, sub["sample"])
        ax.set_title(title)
        for y, (_, r) in zip(ys, sub.iterrows()):
            ax.annotate(f"P(>0)={r['P_positive']:.2f}", (r["mean"], y + 0.12), fontsize=8, ha="center")
    fig.suptitle("HSA competition channels (PPI / inverse-markup / Capital IQ firm; psi excluded)")
    fig.tight_layout()
    fig.savefig(figures / "hsa_channels.png", dpi=200)
    plt.close(fig)


def _report(out: Path, config: dict, df: pd.DataFrame, meta: dict, gate: dict, quick: bool):
    lines = []
    A = lines.append
    A(f"# HSA PPI Identification — {'SMOKE' if quick else 'results'}")
    A("")
    A(f"Revision `{config['revision']}`. Cell: **Capital IQ firm-weighted N x PPI x "
      f"inverse-markup x SPF expectations**, competition-level term (psi) **excluded** "
      f"(theory-faithful). Fast competition = {config['design']['fast_definition']}, "
      f"timing = {config['design']['timing']}, error = {config['design']['error_model']}. "
      f"Sampling: {config['sampling']['iterations']} iters x {config['sampling']['chains']} chains.")
    A("")
    A(f"Convergence gate: {'PASS' if gate['passed'] else 'FAIL'} "
      f"(max R-hat {gate['max_rhat']:.3f} <= {gate['required']:.2f}).")
    A("")
    for activity in config["activities"]:
        A(f"# Activity cell: {config['activities'][activity]['label']}")
        A("")
        for sample_name in ["full", "primary", "conservative"]:
            m = meta.get((activity, sample_name), {})
            A(f"## Sample `{sample_name}`  ({m.get('first','?')}–{m.get('last','?')}, n={m.get('n','?')})")
            A("")
            A("| variant | param | mean | 95% CI | P(>0) | R-hat |")
            A("|---|---|---:|---|---:|---:|")
            sub = df[(df["sample"] == sample_name) & (df["activity"] == activity)]
            for variant in config["design"]["model_variants"]:
                for p in HSA_PARAMS + ["beta_b", "beta_f"]:
                    r = sub[(sub.variant == variant) & (sub.parameter == p)]
                    if len(r):
                        r = r.iloc[0]
                        A(f"| {variant} | {p} | {r['mean']:+.3f} | "
                          f"[{r['ci_2.5']:+.3f}, {r['ci_97.5']:+.3f}] | {r['P_positive']:.2f} | {r['rhat']:.3f} |")
            A("")
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=BUNDLE / "config.yaml")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--jobs", type=int, default=4)
    args = ap.parse_args()

    config = load_yaml(args.config)
    out = BUNDLE / "results" / ("smoke" if args.quick else "")
    tables, figures = out / "tables", out / "figures"
    for d in (tables, figures):
        d.mkdir(parents=True, exist_ok=True)

    sampling = dict(config["sampling"])
    if args.quick:
        sampling.update(iterations=2000, warmup=800, thin=2, chains=2)

    tasks = [
        {"config_path": str(args.config), "sample_name": sn, "sample_start": config["samples"][sn],
         "activity": act, "variant": v, "sampling": sampling}
        for act in config["activities"]
        for sn in ["full", "primary", "conservative"]
        for v in config["design"]["model_variants"]
    ]
    print(f"hsa_ppi_identification: {len(tasks)} fits, jobs={args.jobs}", flush=True)
    started = time.perf_counter()
    rows, meta = [], {}
    with ProcessPoolExecutor(max_workers=max(1, args.jobs)) as ex:
        futs = {ex.submit(_run_task, t): t for t in tasks}
        for i, f in enumerate(as_completed(futs), 1):
            r = f.result(); t = futs[f]
            rows.extend(r["summary"])
            meta[(r["activity"], r["sample_name"])] = {"n": r["n"], "first": r["first"], "last": r["last"]}
            print(f"[{i}/{len(tasks)}] {t['activity']}/{t['sample_name']}/{t['variant']} done", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(tables / "coefficient_summaries.csv", index=False)
    # headline: HSA channels
    head = df[df.parameter.isin(["kappa_0", "kappa_1", "theta_0", "theta_hsa", "gamma"])]
    head.to_csv(tables / "hsa_channels.csv", index=False)

    max_rhat = float(df["rhat"].max())
    gate = {"passed": max_rhat <= float(config["gates"]["max_rhat"]),
            "max_rhat": max_rhat, "required": float(config["gates"]["max_rhat"])}
    _plots(df, _load_frame(), config, figures)
    _report(out, config, df, meta, gate, args.quick)

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "revision": config["revision"], "mode": "smoke" if args.quick else "full",
        "samples": {f"{a}|{s}": v for (a, s), v in meta.items()},
        "sampling": sampling, "gate": gate,
        "elapsed_seconds": time.perf_counter() - started,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\nwrote {out}/report.md  (max R-hat {max_rhat:.3f}, gate={'PASS' if gate['passed'] else 'FAIL'})", flush=True)
    if not gate["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
