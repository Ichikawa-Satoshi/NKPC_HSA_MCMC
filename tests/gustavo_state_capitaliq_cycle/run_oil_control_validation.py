"""Estimate the prespecified QoQ oil-control extension for PPI and Core CPI."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import arviz as az
import numpy as np
import pandas as pd
from scipy.special import logsumexp

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists())
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT / "tests")]
from tests import _bootstrap  # noqa: F401,E402
from nkpc_hsa.config import load_yaml  # noqa: E402
from nkpc_hsa.paths import data_root  # noqa: E402
from tests.active_firm_stock_bds_bed.functions import ThetaCell, robust_scale  # noqa: E402
from tests.gustavo_state_capitaliq_cycle.dynamic_functions import (  # noqa: E402
    dynamic_loglik,
    dynamic_mu,
    dynamic_summary,
    fit_dynamic,
    simulate_varying_theta,
)
from tests.gustavo_state_capitaliq_cycle.functions import (  # noqa: E402
    CycleFit,
    build_qoq_design,
    fit_qoq_theta,
    load_cycle,
    load_nkpc_cells,
    load_oil_controls,
    load_qoq,
    qoq_pointwise_loglik,
    save_qoq,
    simulate_qoq_combined,
    summarize_qoq,
)

BUNDLE = Path(__file__).resolve().parent
STATE_BASE = BUNDLE / "results" / "mock_qoq"
MODELS = ("direct_only", "free_combined", "varying_theta", "free_dynamic", "hsa_restricted_dynamic")
STATIC = {"direct_only", "free_combined"}


def _json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git(args: list[str]) -> str:
    return subprocess.run(
        ["git", *args], cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL
    ).stdout.strip()


def _configs(price: str) -> tuple[dict, dict, dict]:
    base = load_yaml(BUNDLE / "config.yaml")
    oil = load_yaml(BUNDLE / "oil_control_config.yaml")
    if price == "core_cpi":
        core = load_yaml(BUNDLE / "core_cpi_config.yaml")
        spec = core["price"]
        base["data"]["prices"] = {
            spec["name"]: {k: spec[k] for k in ("inflation", "inflation_lag", "expectation")}
        }
    elif price != "ppi":
        raise ValueError(price)
    control_scale = float(oil["oil"]["standardized_effect_prior_sd"])
    base["nkpc"]["control_coefficient_scale"] = control_scale
    dynamic = load_yaml(BUNDLE / "dynamic_config.yaml")
    dynamic["priors"]["control_coefficient_scale"] = control_scale
    return base, dynamic, oil


def _state(label: str) -> CycleFit:
    path = STATE_BASE / "draws" / "cycle" / f"{label}.npz"
    return load_cycle(path, json.loads(path.with_suffix(".json").read_text()))


def _subset(cell: ThetaCell, mask: np.ndarray) -> ThetaCell:
    return ThetaCell(
        cell.name,
        cell.periods[mask],
        cell.pi[mask],
        cell.pi_lag[mask],
        cell.epi[mask],
        cell.x[mask],
        robust_scale(cell.pi[mask]),
        robust_scale(cell.x[mask]),
    )


def _sampling(oil: dict, model: str, mode: str) -> dict:
    if mode == "full":
        return dict(oil["sampling"]["static" if model in STATIC else "dynamic"])
    return {
        "iterations": 500 if model in STATIC else 700,
        "warmup": 150 if model in STATIC else 200,
        "thin": 2,
        "chains": 2,
    }


def _fit_job(args: tuple) -> dict:
    (
        out_s,
        price,
        label,
        error_model,
        cell_name,
        model,
        sample,
        seed,
        sampling,
        holdout,
        reuse,
        mode,
    ) = args
    out = Path(out_s)
    path = out / "draws" / price / sample / model / label / error_model / f"{cell_name}.npz"
    if reuse and path.exists() and path.with_suffix(".json").exists():
        return json.loads(path.with_suffix(".json").read_text())
    base, dynamic, oil = _configs(price)
    cell = load_nkpc_cells(base)[cell_name]
    controls, _ = load_oil_controls(cell.periods)
    if sample == "train":
        mask = np.asarray(cell.periods < pd.Period(holdout, freq="Q"))
        cell = _subset(cell, mask)
        controls = controls[mask]
    state = _state(label)
    if model in STATIC:
        fit = fit_qoq_theta(
            cell,
            state,
            base,
            seed,
            error_model=error_model,
            include_delta=model == "free_combined",
            sampling_override=sampling,
            controls=controls,
        )
        summary = summarize_qoq(fit)
    else:
        fit = fit_dynamic(
            cell,
            state,
            base,
            dynamic,
            seed,
            model=model,
            error_model=error_model,
            sampling_override=sampling,
            controls=controls,
        )
        summary = dynamic_summary(fit, model)
    summary.update(price=price, fit_sample=sample, oil_control=True, retried=False)
    gates = oil["gates"]
    if mode == "full" and (
        summary["diagnostics"]["max_rhat"] > float(gates["observed_max_rhat"])
        or summary["diagnostics"]["min_bulk_ess"] < float(gates["observed_min_bulk_ess"])
    ):
        longer = {
            "iterations": 7000 if model in STATIC else 12000,
            "warmup": 2100 if model in STATIC else 3500,
            "thin": 3,
            "chains": 4,
        }
        if model in STATIC:
            fit = fit_qoq_theta(
                cell,
                state,
                base,
                seed + 33000001,
                error_model=error_model,
                include_delta=model == "free_combined",
                sampling_override=longer,
                controls=controls,
            )
            summary = summarize_qoq(fit)
        else:
            fit = fit_dynamic(
                cell,
                state,
                base,
                dynamic,
                seed + 33000001,
                model=model,
                error_model=error_model,
                sampling_override=longer,
                controls=controls,
            )
            summary = dynamic_summary(fit, model)
        summary.update(price=price, fit_sample=sample, oil_control=True, retried=True)
    save_qoq(path, fit)
    _json(path.with_suffix(".json"), summary)
    return summary


def _metrics(ll: np.ndarray) -> dict:
    idata = az.from_dict(
        {"posterior": {"dummy": np.zeros((*ll.shape[:2], 1))}, "log_likelihood": {"inflation": ll}}
    )
    loo = az.loo(idata, pointwise=True)
    pareto = np.asarray(loo.pareto_k)
    flat = ll.reshape(-1, ll.shape[-1])
    lppd = logsumexp(flat, axis=0) - np.log(len(flat))
    p_i = np.var(flat, axis=0, ddof=1)
    w_i = lppd - p_i
    return {
        "elpd_loo": float(loo.elpd),
        "se_loo": float(loo.se),
        "p_loo": float(loo.p),
        "elpd_waic": float(w_i.sum()),
        "se_waic": float(np.sqrt(len(w_i) * np.var(w_i, ddof=1))),
        "p_waic": float(p_i.sum()),
        "max_pareto_k": float(pareto.max()),
        "pareto_k_over_0.7": int(np.sum(pareto > 0.7)),
    }


def _holdout(
    train: ThetaCell,
    test: ThetaCell,
    fit,
    state: CycleFit,
    model: str,
    train_controls: np.ndarray,
    test_controls: np.ndarray,
) -> dict:
    sp = pd.PeriodIndex(state.periods, freq="Q")
    tr = sp.get_indexer(train.periods)
    te = sp.get_indexer(test.periods)
    mus, ll = [], []
    for c in range(fit.draws.shape[0]):
        for d in range(fit.draws.shape[1]):
            cs, ds = int(fit.state_chain[c, d]), int(fit.state_draw[c, d])
            hat = state.nhat[cs, ds, te]
            raw = state.nbar_used[cs, ds]
            if model == "direct_only":
                X, _ = build_qoq_design(test, hat, controls=test_controls)
                mu = X @ fit.draws[c, d]
            elif model == "free_combined":
                bar = raw[te] - float(np.mean(raw[tr]))
                X = np.column_stack(
                    [
                        np.ones(len(test.periods)),
                        test.pi_lag,
                        test.epi,
                        test.x,
                        test_controls,
                        bar * test.x,
                        -hat,
                    ]
                )
                mu = X @ fit.draws[c, d]
            else:
                center = (float(np.mean(raw[tr])), float(np.mean((raw[tr] - np.mean(raw[tr])) ** 2)))
                mu = dynamic_mu(
                    test,
                    fit,
                    c,
                    d,
                    bar=raw[te],
                    hat=hat,
                    center=center,
                    controls=test_controls,
                )
            mus.append(mu)
            sig = float(fit.sigma_u[c, d])
            ll.append(-0.5 * np.log(2 * np.pi * sig**2) - 0.5 * (test.pi - mu) ** 2 / sig**2)
    mus, ll = np.asarray(mus), np.asarray(ll)
    return {
        "holdout_elpd": float(np.sum(logsumexp(ll, axis=0) - np.log(len(ll)))),
        "holdout_rmse": float(np.sqrt(np.mean((test.pi - mus.mean(0)) ** 2))),
        "holdout_n": len(test.pi),
    }


def _oracle(periods, hat: np.ndarray, bar: np.ndarray) -> CycleFit:
    return CycleFit(
        "oracle", tuple(map(str, periods)), tuple(), np.zeros((1, 1, 0)), bar[None, None, :],
        hat[None, None, :], {}, hat
    )


def _recovery_group(args: tuple) -> list[dict]:
    price, kind, mode, run_mode, reps, scenarios, seed, sampling, generator_path = args
    base, dynamic, oil = _configs(price)
    state = _state("firm_weighted")
    cell_name = f"{price}_negative_unemployment_gap"
    cell = load_nkpc_cells(base)[cell_name]
    controls, _ = load_oil_controls(cell.periods)
    generator_path = Path(generator_path)
    meta = json.loads(generator_path.with_suffix(".json").read_text())
    observed = load_qoq(generator_path, meta["diagnostics"])
    rows = []
    gates = oil["gates"]
    for j, (scenario, truths) in enumerate(scenarios.items()):
        for rep in range(reps):
            rseed = seed + 1000003 * j + 1009 * rep
            rng = np.random.default_rng(rseed)
            if kind == "static":
                synthetic, hat, bar, true = simulate_qoq_combined(
                    rng, cell, observed, state, float(truths[0]), float(truths[1]), controls
                )
                use = state if mode == "propagated_state" else _oracle(synthetic.periods, hat, bar)
                fit = fit_qoq_theta(
                    synthetic, use, base, rseed + 7000001, error_model="iid", include_delta=True,
                    recovery=True, sampling_override=sampling, controls=controls
                )
                summary = summarize_qoq(fit)
                pairs = (("delta", float(truths[0])), ("theta_CIQ", float(truths[1])))
            else:
                synthetic, hat, bar, true = simulate_varying_theta(
                    rng, cell, observed, state, float(truths[0]), float(truths[1]), controls
                )
                use = state if mode == "propagated_state" else _oracle(synthetic.periods, hat, bar)
                fit = fit_dynamic(
                    synthetic, use, base, dynamic, rseed + 7000001, model="varying_theta",
                    error_model="iid", sampling_override=sampling, controls=controls
                )
                summary = dynamic_summary(fit, "varying_theta")
                pairs = (("theta_0", float(truths[0])), ("gamma", float(truths[1])))
            retried = False
            if summary["diagnostics"]["max_rhat"] > float(gates["recovery_max_rhat"]) or summary["diagnostics"]["min_bulk_ess"] < float(gates["recovery_min_bulk_ess"]):
                longer = {"iterations": 2200, "warmup": 700, "thin": 2, "chains": 4}
                if kind == "static":
                    fit = fit_qoq_theta(
                        synthetic, use, base, rseed + 17000001, error_model="iid", include_delta=True,
                        recovery=True, sampling_override=longer, controls=controls
                    )
                    summary = summarize_qoq(fit)
                else:
                    fit = fit_dynamic(
                        synthetic, use, base, dynamic, rseed + 17000001, model="varying_theta",
                        error_model="iid", sampling_override=longer, controls=controls
                    )
                    summary = dynamic_summary(fit, "varying_theta")
                retried = True
            for parameter, standardized_true in pairs:
                z = summary["coefficients"][parameter]
                raw = true[parameter]
                positive = raw > 0
                learned = z["posterior_prior_sd_ratio"] <= float(gates["posterior_prior_sd_ratio"])
                rows.append(
                    {
                        "price": price,
                        "kind": kind,
                        "mode": mode,
                        "scenario": scenario,
                        "replicate": rep,
                        "parameter": parameter,
                        "standardized_true": standardized_true,
                        "raw_true": raw,
                        "mean": z["mean"],
                        "q2.5": z["q2.5"],
                        "q97.5": z["q97.5"],
                        "p_positive": z["p_positive"],
                        "sd_ratio": z["posterior_prior_sd_ratio"],
                        "coverage": bool(z["q2.5"] <= raw <= z["q97.5"]),
                        "suggestive_detected": bool(positive and z["p_positive"] >= float(gates["suggestive_sign_probability"]) and learned),
                        "strong_detected": bool(positive and z["p_positive"] >= float(gates["strong_sign_probability"]) and z["q2.5"] > 0 and learned),
                        "false_positive": bool(not positive and ((z["p_positive"] >= float(gates["strong_sign_probability"]) and z["q2.5"] > 0) or (z["p_positive"] <= 1 - float(gates["strong_sign_probability"]) and z["q97.5"] < 0))),
                        "max_rhat": summary["diagnostics"]["max_rhat"],
                        "min_bulk_ess": summary["diagnostics"]["min_bulk_ess"],
                        "retried": retried,
                    }
                )
    return rows


def _baseline_table() -> pd.DataFrame:
    result = BUNDLE / "results"
    ppi_static = pd.read_csv(result / "staged_validation" / "tables" / "model_comparison.csv")
    ppi_dynamic = pd.read_csv(result / "dynamic_validation" / "tables" / "model_comparison.csv")
    ppi_dynamic["model"] = ppi_dynamic["model"].replace({"constant_theta": "direct_only"})
    ppi_dynamic = ppi_dynamic[ppi_dynamic.model.isin(MODELS[2:])]
    ppi = pd.concat([ppi_static[ppi_static.model.isin(MODELS[:2])], ppi_dynamic], ignore_index=True)
    ppi["price"] = "ppi"
    core = pd.read_csv(result / "core_cpi_full" / "tables" / "model_comparison.csv")
    core["price"] = "core_cpi"
    return pd.concat([ppi, core], ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "full"), default="full")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--refit", action="store_true")
    parser.add_argument("--skip-recovery", action="store_true")
    args = parser.parse_args()
    started = time.time()
    _, _, oil = _configs("ppi")
    out = BUNDLE / "results" / f"oil_control_{args.mode}"
    (out / "tables").mkdir(parents=True, exist_ok=True)
    hashes = {
        "base": _sha(BUNDLE / "config.yaml"),
        "core": _sha(BUNDLE / "core_cpi_config.yaml"),
        "dynamic": _sha(BUNDLE / "dynamic_config.yaml"),
        "oil": _sha(BUNDLE / "oil_control_config.yaml"),
    }
    old = json.loads((out / "manifest.json").read_text()) if (out / "manifest.json").exists() else {}
    reuse = bool(not args.refit and old.get("config_hashes") == hashes)
    seed = int(load_yaml(BUNDLE / "config.yaml")["sampling"]["seed"]) + int(oil["sampling"]["seed_offset"])
    holdout = oil["comparison"]["holdout_start"]
    labels = list(load_yaml(BUNDLE / "config.yaml")["data"]["capital_iq"])
    jobs, k = [], 0
    samples = {}
    for price in oil["prices"]:
        base, _, _ = _configs(price)
        cells = load_nkpc_cells(base)
        samples.update({name: [str(cell.periods[0]), str(cell.periods[-1]), len(cell.periods)] for name, cell in cells.items()})
        for label in labels:
            for cell_name in cells:
                for model in MODELS:
                    for sample in ("full", "train"):
                        k += 1
                        jobs.append((str(out), price, label, "iid", cell_name, model, sample, seed + 1000003 * k, _sampling(oil, model, args.mode), holdout, reuse, args.mode))
        primary = f"{price}_negative_unemployment_gap"
        for model in MODELS:
            k += 1
            jobs.append((str(out), price, "firm_weighted", "persistent_ar1", primary, model, "full", seed + 1000003 * k, _sampling(oil, model, args.mode), holdout, reuse, args.mode))
    summaries = []
    with ProcessPoolExecutor(max_workers=min(args.workers, len(jobs))) as pool:
        for future in as_completed([pool.submit(_fit_job, job) for job in jobs]):
            z = future.result()
            summaries.append(z)
            print(f"OIL {z['price']} {z['fit_sample']} {z['model']} {z['cycle']} {z['error_model']} {z['cell']} Rhat={z['diagnostics']['max_rhat']:.4f}", flush=True)

    coeff = []
    for z in summaries:
        if z["fit_sample"] != "full":
            continue
        common = {
            "price": z["price"], "cycle": z["cycle"], "error_model": z["error_model"],
            "cell": z["cell"], "model": z["model"], "sample_start": z["sample"][0],
            "sample_end": z["sample"][1], "fit_max_rhat": z["diagnostics"]["max_rhat"],
            "fit_min_bulk_ess": z["diagnostics"]["min_bulk_ess"], "retried": z.get("retried", False),
        }
        for name, values in z["coefficients"].items():
            coeff.append({**common, "parameter": name, **values})
        for name, values in z.get("derived", {}).items():
            coeff.append({**common, "parameter": name + "_derived", **values})
    pd.DataFrame(coeff).to_csv(out / "tables" / "coefficients.csv", index=False)

    comparisons = []
    for price in oil["prices"]:
        base, _, _ = _configs(price)
        for label in labels:
            state = _state(label)
            for cell_name, cell in load_nkpc_cells(base).items():
                controls, _ = load_oil_controls(cell.periods)
                mask = np.asarray(cell.periods < pd.Period(holdout, freq="Q"))
                train, test = _subset(cell, mask), _subset(cell, ~mask)
                for model in MODELS:
                    fp = out / "draws" / price / "full" / model / label / "iid" / f"{cell_name}.npz"
                    tp = out / "draws" / price / "train" / model / label / "iid" / f"{cell_name}.npz"
                    fm, tm = json.loads(fp.with_suffix(".json").read_text()), json.loads(tp.with_suffix(".json").read_text())
                    full, trained = load_qoq(fp, fm["diagnostics"]), load_qoq(tp, tm["diagnostics"])
                    ll = qoq_pointwise_loglik(cell, full, controls) if model in STATIC else dynamic_loglik(cell, full, controls)
                    comparisons.append(
                        {
                            "price": price, "cycle": label, "cell": cell_name, "model": model,
                            **_metrics(ll),
                            **_holdout(train, test, trained, state, model, controls[mask], controls[~mask]),
                            "max_rhat": fm["diagnostics"]["max_rhat"],
                            "min_bulk_ess": fm["diagnostics"]["min_bulk_ess"],
                        }
                    )
    comp = pd.DataFrame(comparisons)
    comp.to_csv(out / "tables" / "model_comparison.csv", index=False)
    baseline = _baseline_table()
    keys = ["price", "cycle", "cell", "model"]
    merged = comp.merge(baseline, on=keys, suffixes=("_oil", "_baseline"), validate="one_to_one")
    for metric in ("elpd_loo", "elpd_waic", "holdout_elpd", "holdout_rmse"):
        merged[f"delta_{metric}_oil_minus_baseline"] = merged[f"{metric}_oil"] - merged[f"{metric}_baseline"]
    merged.to_csv(out / "tables" / "oil_vs_no_oil.csv", index=False)

    rec = pd.DataFrame()
    if not args.skip_recovery:
        groups = []
        for price_index, price in enumerate(oil["prices"]):
            cell_name = f"{price}_negative_unemployment_gap"
            for kind, block in (("static", oil["static_recovery"]), ("dynamic", oil["dynamic_recovery"])):
                reps = int(block["replicates"] if args.mode == "full" else 2)
                model = "free_combined" if kind == "static" else "varying_theta"
                sampling = dict(oil["sampling"][f"{kind}_recovery"])
                generator = out / "draws" / price / "full" / model / "firm_weighted" / "iid" / f"{cell_name}.npz"
                for mode_index, state_mode in enumerate(block["modes"]):
                    groups.append((price, kind, state_mode, args.mode, reps, block["scenarios"], seed + 51000001 + 10000019 * price_index + 20000033 * mode_index + (40000063 if kind == "dynamic" else 0), sampling, str(generator)))
        rows = []
        with ProcessPoolExecutor(max_workers=min(args.workers, len(groups))) as pool:
            for future in as_completed([pool.submit(_recovery_group, group) for group in groups]):
                rows.extend(future.result())
                print(f"OIL RECOVERY {len(rows)} coefficient rows", flush=True)
        rec = pd.DataFrame(rows)
        rec.to_csv(out / "tables" / "recovery_replications.csv", index=False)
        power = rec.groupby(["price", "kind", "mode", "scenario", "parameter"]).agg(
            replicates=("replicate", "size"), standardized_true=("standardized_true", "first"),
            suggestive_rate=("suggestive_detected", "mean"), strong_rate=("strong_detected", "mean"),
            false_positive_rate=("false_positive", "mean"), coverage=("coverage", "mean"),
            mean_estimate=("mean", "mean"), mean_p_positive=("p_positive", "mean"),
            mean_sd_ratio=("sd_ratio", "mean"), max_rhat=("max_rhat", "max"),
            min_bulk_ess=("min_bulk_ess", "min"), retry_rate=("retried", "mean"),
        ).reset_index()
        power.to_csv(out / "tables" / "recovery_power.csv", index=False)

    observed = [z for z in summaries if z["fit_sample"] == "full"]
    gate = {
        "observed_max_rhat": max(z["diagnostics"]["max_rhat"] for z in observed),
        "observed_min_bulk_ess": min(z["diagnostics"]["min_bulk_ess"] for z in observed),
    }
    gate["observed_computational_pass"] = bool(
        gate["observed_max_rhat"] <= float(oil["gates"]["observed_max_rhat"])
        and gate["observed_min_bulk_ess"] >= float(oil["gates"]["observed_min_bulk_ess"])
    )
    if len(rec):
        gate.update(recovery_max_rhat=float(rec.max_rhat.max()), recovery_min_bulk_ess=float(rec.min_bulk_ess.min()))
    ppi_cell = load_nkpc_cells(_configs("ppi")[0])["ppi_negative_unemployment_gap"]
    oil_controls, oil_meta = load_oil_controls(ppi_cell.periods)
    oil_path = Path(oil_meta["source"])
    manifest = {
        "revision": oil["revision"], "profile": args.mode, "not_for_inference": args.mode != "full",
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "elapsed_seconds": time.time() - started, "seed": seed, "config_hashes": hashes,
        "data_sha256": _sha(data_root() / "processed" / "model_ready.csv"), "oil_sha256": _sha(oil_path),
        "measurement_hashes": {label: _sha(STATE_BASE / "draws" / "cycle" / f"{label}.npz") for label in labels},
        "observed_fit_count": len(jobs), "recovery_fit_count": int(len(rec) / 2), "sample": samples,
        "oil": oil_meta, "git_commit": _git(["rev-parse", "HEAD"]), "git_dirty": bool(_git(["status", "--porcelain"])),
        "gate": gate,
        "interpretation": "Prespecified current and one-quarter-lagged real-oil-price QoQ controls; identical specification for PPI and Core CPI; competition states remain cut from inflation.",
    }
    _json(out / "manifest.json", manifest)
    print(json.dumps(gate, indent=2), flush=True)


if __name__ == "__main__":
    main()
