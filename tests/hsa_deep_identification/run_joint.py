"""Run and save the exact-N MA(3) joint identification experiments."""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys

import numpy as np

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists())
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT / "tests")]
from tests import _bootstrap  # noqa: F401,E402
from nkpc_hsa.config import load_yaml  # noqa:E402
from tests.hsa_deep_identification.joint_ma3 import fit_joint_ma3  # noqa:E402
from tests.hsa_nested_validation.functions import (  # noqa:E402
    BASE_NAMES, ModelSpec, load_experiment,
)

BUNDLE = Path(__file__).resolve().parent
NESTED = ROOT / "tests" / "hsa_nested_validation"


def specs() -> dict[str, ModelSpec]:
    out = {
        "ces": ModelSpec("ces", "diagnostic", BASE_NAMES),
        "slope": ModelSpec("slow_slope", "diagnostic", BASE_NAMES + ("delta_s",)),
        "direct": ModelSpec("direct", "diagnostic", BASE_NAMES + ("theta",)),
        "free": ModelSpec("free_static_combined", "diagnostic", BASE_NAMES + ("delta_s", "theta")),
        "free_no_intercept": ModelSpec(
            "free_static_no_intercept", "diagnostic",
            ("alpha_b", "alpha_f", "kappa_0", "delta_s", "theta"),
        ),
        "free_sum1": ModelSpec(
            "free_static_alpha_sum_one", "diagnostic",
            ("intercept", "alpha_b", "kappa_0", "delta_s", "theta"),
        ),
        "free_lambda": ModelSpec("free_lambda_diagnostic", "diagnostic", BASE_NAMES + ("theta",), None, True),
    }
    for value in (3.0, 6.0, 9.0):
        out[f"hsa{int(value)}"] = ModelSpec(f"hsa_fixed_lambda_{value:g}", "diagnostic", BASE_NAMES + ("theta",), value)
    return out


def summary(result) -> dict:
    fit = result.model_fit; flat = fit.draws.reshape(-1, fit.draws.shape[-1])
    coefficients = {}
    for j, name in enumerate(fit.names):
        values = flat[:, j]
        coefficients[name] = {
            "mean": float(values.mean()), "sd": float(values.std(ddof=1)),
            "q2.5": float(np.percentile(values, 2.5)), "q97.5": float(np.percentile(values, 97.5)),
            "p_positive": float(np.mean(values > 0)),
            "posterior_prior_sd_ratio": float(values.std(ddof=1) / fit.prior_sd[name]),
            "rhat": fit.diagnostics["rhat"][name],
            "ess_bulk": fit.diagnostics["ess_bulk"][name],
            "ess_tail": fit.diagnostics["ess_tail"][name],
        }
    k0 = flat[:, fit.names.index("kappa_0")]
    bar = fit.nbar.reshape(-1, fit.nbar.shape[-1])
    if "delta_s" in fit.names:
        delta = flat[:, fit.names.index("delta_s")]
    elif fit.spec.lambda_fixed is not None:
        delta = fit.spec.lambda_fixed * flat[:, fit.names.index("theta")]
    elif fit.spec.free_lambda:
        delta = flat[:, fit.names.index("lambda")] * flat[:, fit.names.index("theta")]
    else:
        delta = np.zeros_like(k0)
    kappa = k0[:, None] + delta[:, None] * bar
    positive_date_share = np.mean(kappa > 0, axis=1)
    psi = (result.psi.reshape(-1, result.psi.shape[-1])
           if result.psi.shape[-1] else np.empty((result.psi.shape[0]*result.psi.shape[1], 0)))
    return {
        "model": fit.spec.model_id, "architecture": result.slow_architecture,
        "coefficients": coefficients,
        "state": {
            "omega_mean": float(fit.omega.mean()),
            "omega_q2.5": float(np.percentile(fit.omega, 2.5)),
            "omega_q97.5": float(np.percentile(fit.omega, 97.5)),
            "slow_innovation_variance_mean": float(np.mean(fit.omega * fit.tau**2)),
            "cycle_innovation_variance_mean": float(np.mean((1-fit.omega) * fit.tau**2)),
            "cycle_damping_mean": float(fit.cycle_damping.mean()),
            "cycle_period_mean": float(fit.cycle_period.mean()),
        },
        "psi": [{"mean": float(psi[:,j].mean()), "q2.5": float(np.percentile(psi[:,j],2.5)),
                 "q97.5": float(np.percentile(psi[:,j],97.5))} for j in range(psi.shape[1])],
        "kappa": {
            "posterior_probability_positive_at_95pct_dates": float(np.mean(positive_date_share >= .95)),
            "mean_positive_date_share": float(positive_date_share.mean()),
            "minimum_pointwise_probability_positive": float(np.min(np.mean(kappa > 0, axis=0))),
        },
        "diagnostics": fit.diagnostics,
    }


def save(path: Path, result) -> None:
    fit = result.model_fit; path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path, model_id=fit.spec.model_id, lambda_fixed=np.nan if fit.spec.lambda_fixed is None else fit.spec.lambda_fixed,
        free_lambda=fit.spec.free_lambda, names=fit.names, periods=fit.periods, draws=fit.draws,
        sigma_pi=fit.sigma_pi, psi=result.psi, n_total=fit.n_total, nbar=fit.nbar, nhat=fit.nhat,
        omega=fit.omega, tau=fit.tau, cycle_damping=fit.cycle_damping, cycle_period=fit.cycle_period,
        slow_drift=result.slow_drift, architecture=result.slow_architecture,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=("mock", "quick", "full"), default="mock")
    parser.add_argument("--architectures", nargs="+", default=["quarterly_local_level_ar2", "annual_allocation_ar2"])
    parser.add_argument("--cells", nargs="+", default=["ppi_negative_unemployment_gap", "ppi_inverse_markup"])
    parser.add_argument("--models", nargs="+", default=["ces", "free", "free_lambda", "hsa3", "hsa6", "hsa9"])
    args = parser.parse_args()
    cfg = load_yaml(BUNDLE / "config.yaml"); nested_cfg = load_yaml(NESTED / "config.yaml")
    experiment = load_experiment(nested_cfg); available = specs(); sampling = cfg["sampling"][args.profile]
    root = BUNDLE / "results" / args.profile / "joint_ma3"
    manifest = []
    for architecture in args.architectures:
        for cell_name in args.cells:
            cell = experiment.cells[cell_name]
            for model_name in args.models:
                spec = available[model_name]
                use_cell = cell
                if model_name == "free_sum1":
                    use_cell = replace(
                        cell, pi=cell.pi-cell.epi, pi_lag=cell.pi_lag-cell.epi,
                        epi=np.zeros(cell.n_periods),
                    )
                seed = int(cfg["sampling"]["seed"]) + 1009*len(manifest)
                print(f"RUN {architecture} {cell_name} {model_name}", flush=True)
                result = fit_joint_ma3(experiment, use_cell, spec, cfg, sampling, architecture, seed)
                folder = root / architecture / cell_name
                save(folder / f"{model_name}.npz", result)
                report = summary(result)
                (folder / f"{model_name}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
                manifest.append({"architecture": architecture, "cell": cell_name, "model": model_name,
                                 "max_rhat": report["diagnostics"]["max_rhat"],
                                 "identity_error": report["diagnostics"]["exact_identity_error"],
                                 "theta": report["coefficients"].get("theta"),
                                 "kappa": report["kappa"]})
                (root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
                print(json.dumps(manifest[-1], indent=2), flush=True)


if __name__ == "__main__":
    main()
