"""Evaluate the cut-to-full-joint inflation-feedback path by importance sampling."""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import sys as _sys, pathlib as _pathlib  # noqa: E402  (bootstrap: importable at any depth)
_ROOT = next(_p for _p in _pathlib.Path(__file__).resolve().parents if (_p / "pyproject.toml").exists())
_sys.path[:0] = [str(_ROOT), str(_ROOT / "src"), str(_ROOT / "tests")]
from experiments import _bootstrap  # noqa: F401,E402
from experiments._bootstrap import RESULTS_DIR, ROOT, data_root


BUNDLE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BUNDLE_DIR / "results"
from nkpc_hsa.config import load_yaml
from nkpc_hsa.dataprep.func_data_build import load_spf_cpi_quarter_ahead_expectations
from nkpc_hsa.phillips.data import load_design_data
from experiments.markup_feedback.functions import (
    feedback_weights,
    log_marginal_regression,
    pareto_k_diagnostic,
    weighted_quantile,
)
from nkpc_hsa.phillips.inflation import (
    _prior_sds,
    _quarterly_design,
    fit_cut_model,
    reference_draws,
)
from nkpc_hsa.phillips.state import MeasurementPosterior
from nkpc_hsa.provenance import stamp_artifact_metadata


def _load_measurement(path: Path) -> MeasurementPosterior:
    payload = np.load(path, allow_pickle=True)
    augmented = {key[2:]: payload[key] for key in payload.files if key.startswith("C_")}
    annual = {key[2:]: payload[key] for key in payload.files if key.startswith("N_")}
    return MeasurementPosterior(
        draws=augmented,
        annual_only_draws=annual,
        information_ratio=float(payload["information_ratio"]),
        periods=tuple(payload["periods"].astype(str)),
    )


def _weighted_summary(values: np.ndarray, weights: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, float).reshape(-1)
    mean = float(weights @ values)
    variance = float(weights @ (values - mean) ** 2)
    lo, median, hi = weighted_quantile(values, weights, np.array([0.025, 0.5, 0.975]))
    return {
        "mean": mean,
        "median": float(median),
        "sd": float(np.sqrt(variance)),
        "ci_2.5": float(lo),
        "ci_97.5": float(hi),
        "p_positive": float(weights @ (values > 0.0)),
    }


def _weighted_path_summary(values: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    flat = np.asarray(values, float).reshape(-1, values.shape[-1])
    mean = weights @ flat
    lo = np.empty(flat.shape[1])
    hi = np.empty(flat.shape[1])
    for t in range(flat.shape[1]):
        lo[t], hi[t] = weighted_quantile(flat[:, t], weights, np.array([0.025, 0.975]))
    return mean, lo, hi


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=BUNDLE_DIR / "config.yaml",
    )
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    out = Path(
        args.output_dir
        or OUTPUT_DIR
    )
    tables, figures, posterior = out / "tables", out / "figures", out / "posterior"
    for directory in (tables, figures, posterior):
        directory.mkdir(parents=True, exist_ok=True)

    data = load_design_data(
        include_qcew=False,
        sample_start=str(cfg["sample"]["start"]),
        sample_end=str(cfg["sample"]["end"]),
    )
    cpi_expectation = load_spf_cpi_quarter_ahead_expectations(data_root() / "raw")
    quarterly = data.quarterly.copy()
    quarterly["expectation"] = cpi_expectation[
        str(cfg["inflation"]["expectation_column"])
    ].reindex(quarterly.index.to_timestamp(how="end")).to_numpy(float)
    if quarterly["expectation"].isna().any():
        raise ValueError("CPI expectation does not cover the requested sample.")
    data = replace(data, quarterly=quarterly)

    measurement_path = ROOT / str(cfg["measurement"]["posterior"])
    measurement = _load_measurement(measurement_path)
    _, q0, _ = reference_draws(data, measurement)
    cut = fit_cut_model(
        data,
        measurement,
        cell=9,
        model="E2",
        transformation="qoq",
        q0=q0,
        seed=int(cfg["seed"]),
        price_override="core_cpi",
        activity_override="negative_unemployment_gap",
    )

    y = data.quarterly["pi_core_cpi"].to_numpy(float)
    lag = data.quarterly["pi_core_cpi_lag1"].to_numpy(float)
    expectation = data.quarterly["expectation"].to_numpy(float)
    x = data.quarterly["x_negative_unemployment_gap"].to_numpy(float)
    names = cut.coefficient_names
    prior_sds = np.asarray([cut.prior_sds[name] for name in names])
    qbar = measurement.draws["qbar"]
    qhat = measurement.draws["qhat"]
    chains, draws, periods = qbar.shape
    checkpoint = posterior / "log_marginal_checkpoint.npz"
    if checkpoint.exists():
        log_marginal = np.load(checkpoint)["log_marginal"]
        if log_marginal.shape != (chains, draws):
            raise ValueError("Saved log-marginal checkpoint has the wrong shape.")
    else:
        log_marginal = np.empty((chains, draws))
        for chain in range(chains):
            for draw in range(draws):
                X, built_names = _quarterly_design(
                    pi_lag=lag,
                    expectation=expectation,
                    x=x,
                    qbar=qbar[chain, draw],
                    qhat=qhat[chain, draw],
                    q0=q0,
                    model="E2",
                )
                if built_names != names:
                    raise RuntimeError("Feedback design order changed.")
                log_marginal[chain, draw] = log_marginal_regression(
                    y,
                    X,
                    prior_sds,
                    variance_shape=float(cfg["inflation"]["variance_shape"]),
                    variance_scale=float(cfg["inflation"]["variance_scale"]),
                    quadrature_nodes=int(cfg["inflation"]["quadrature_nodes"]),
                )
        np.savez_compressed(checkpoint, log_marginal=log_marginal)

    flat_log_marginal = log_marginal.reshape(-1)
    coefficient_draws = cut.coefficients.reshape(-1, len(names))
    qbar_flat = qbar.reshape(-1, periods)
    qhat_flat = qhat.reshape(-1, periods)
    feedback_grid = np.asarray(cfg["feedback_grid"], float)
    diagnostic_rows: list[dict[str, float]] = []
    coefficient_rows: list[dict[str, float | str]] = []
    state_rows: list[dict[str, float | str]] = []
    state_shift_rows: list[dict[str, float]] = []
    saved_weights = np.empty((feedback_grid.size, flat_log_marginal.size))

    baseline_means: dict[str, np.ndarray] = {}
    baseline_sds: dict[str, float] = {}
    for grid_index, feedback in enumerate(feedback_grid):
        result = feedback_weights(flat_log_marginal, float(feedback))
        weights = result.weights
        saved_weights[grid_index] = weights
        pareto_k = pareto_k_diagnostic(float(feedback) * flat_log_marginal)
        diagnostic_rows.append(
            {
                "feedback": float(feedback),
                "raw_ess": result.raw_ess,
                "raw_ess_fraction": result.raw_ess / weights.size,
                "entropy_ess": result.entropy_ess,
                "max_weight": result.max_weight,
                "pareto_k": pareto_k,
            }
        )
        for parameter_index, parameter in enumerate(names):
            row: dict[str, float | str] = {
                "feedback": float(feedback),
                "parameter": parameter,
            }
            row.update(_weighted_summary(coefficient_draws[:, parameter_index], weights))
            coefficient_rows.append(row)

        path_values = {
            "qtotal": qbar_flat + qhat_flat,
            "qbar": qbar_flat,
            "qhat": qhat_flat,
        }
        shifts: dict[str, float] = {"feedback": float(feedback)}
        for state_name, values in path_values.items():
            mean, lo, hi = _weighted_path_summary(values, weights)
            if grid_index == 0:
                baseline_means[state_name] = mean
                baseline_sds[state_name] = float(np.sqrt(np.mean(np.var(values, axis=0))))
            shifts[f"{state_name}_mean_path_rmse_from_cut"] = float(
                np.sqrt(np.mean((mean - baseline_means[state_name]) ** 2))
            )
            shifts[f"{state_name}_rmse_in_cut_posterior_sd"] = (
                shifts[f"{state_name}_mean_path_rmse_from_cut"] / baseline_sds[state_name]
            )
            for t, period in enumerate(data.periods):
                state_rows.append(
                    {
                        "feedback": float(feedback),
                        "state": state_name,
                        "period": str(period),
                        "mean": float(mean[t]),
                        "ci_2.5": float(lo[t]),
                        "ci_97.5": float(hi[t]),
                    }
                )
        state_shift_rows.append(shifts)

    diagnostics = pd.DataFrame(diagnostic_rows)
    coefficients = pd.DataFrame(coefficient_rows)
    states = pd.DataFrame(state_rows)
    shifts = pd.DataFrame(state_shift_rows)
    diagnostics.to_csv(tables / "feedback_diagnostics.csv", index=False)
    coefficients.to_csv(tables / "coefficient_path.csv", index=False)
    states.to_csv(tables / "state_path.csv", index=False)
    shifts.to_csv(tables / "state_shift_summary.csv", index=False)
    np.savez_compressed(
        posterior / "feedback_importance_weights.npz",
        feedback_grid=feedback_grid,
        log_marginal=log_marginal,
        weights=saved_weights,
        coefficient_names=np.asarray(names),
    )

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    axes[0, 0].plot(diagnostics["feedback"], diagnostics["raw_ess_fraction"], marker="o")
    axes[0, 0].set_ylabel("raw ESS / 4000")
    axes[0, 0].set_xlabel("inflation feedback lambda")
    axes[0, 1].plot(diagnostics["feedback"], diagnostics["pareto_k"], marker="o", color="C3")
    axes[0, 1].axhline(0.7, color="black", ls="--", lw=0.8)
    axes[0, 1].set_ylabel("Pareto k")
    axes[0, 1].set_xlabel("inflation feedback lambda")
    for parameter, color in (("kappa_1", "C0"), ("theta_0", "C1"), ("gamma", "C2")):
        subset = coefficients[coefficients["parameter"] == parameter]
        axes[1, 0].plot(subset["feedback"], subset["mean"], marker="o", label=parameter, color=color)
        axes[1, 0].fill_between(
            subset["feedback"].to_numpy(float),
            subset["ci_2.5"].to_numpy(float),
            subset["ci_97.5"].to_numpy(float),
            color=color,
            alpha=0.12,
        )
    axes[1, 0].axhline(0.0, color="black", lw=0.7)
    axes[1, 0].set_xlabel("inflation feedback lambda")
    axes[1, 0].set_ylabel("coefficient posterior")
    axes[1, 0].legend(frameon=False)
    for state_name, color in (("qtotal", "black"), ("qbar", "C0"), ("qhat", "C3")):
        axes[1, 1].plot(
            shifts["feedback"],
            shifts[f"{state_name}_rmse_in_cut_posterior_sd"],
            marker="o",
            label=state_name,
            color=color,
        )
    axes[1, 1].set_xlabel("inflation feedback lambda")
    axes[1, 1].set_ylabel("mean-path shift / cut posterior SD")
    axes[1, 1].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(figures / "feedback_path.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    manifest = stamp_artifact_metadata(
        {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "revision": str(cfg["revision"]),
            "sample": [str(data.periods[0]), str(data.periods[-1])],
            "n_measurement_draws": int(flat_log_marginal.size),
            "target": "p_lambda(q|D) proportional to p(q|annual N,markup) m_pi(pi|q)^lambda",
            "lambda_zero": "modular cut state posterior",
            "lambda_one": "full-joint state marginal",
            "inflation": "core CPI",
            "expectation": "SPF headline CPI CPI3 one-quarter-ahead mean",
            "activity": "NROU - UNRATE",
            "measurement_posterior": str(measurement_path),
            "quadrature": dict(cfg["inflation"]),
            "limitations": [
                "Importance results require adequate overlap, assessed by raw ESS and Pareto k.",
                "Headline CPI expectations proxy for unavailable pre-2007 core CPI expectations.",
                "Coefficient summaries reweight paired conditional cut draws by the state marginal likelihood.",
            ],
        }
    )
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
