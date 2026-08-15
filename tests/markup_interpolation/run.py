"""Run zero-sum inverse-markup timing sensitivities between exact Q4 N anchors."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import sys as _sys, pathlib as _pathlib  # noqa: E402  (bootstrap: importable at any depth)
_ROOT = next(_p for _p in _pathlib.Path(__file__).resolve().parents if (_p / "pyproject.toml").exists())
_sys.path[:0] = [str(_ROOT), str(_ROOT / "src"), str(_ROOT / "tests")]
from experiments import _bootstrap  # noqa: F401,E402
from experiments._bootstrap import RESULTS_DIR, ROOT


BUNDLE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BUNDLE_DIR / "results"
from nkpc_hsa.config import load_yaml
from nkpc_hsa.phillips.data import load_design_data, robust_scale
from nkpc_hsa.phillips.estimation import _save_fit, summarize_fit
from nkpc_hsa.phillips.inflation import fit_cut_model, reference_draws
from experiments.markup_interpolation.functions import (
    as_measurement_posterior,
    build_q4_anchored_path,
    sample_hard_anchor_draws,
)
from nkpc_hsa.progress import ProgressReporter, STYLES as PROGRESS_STYLES
from nkpc_hsa.provenance import stamp_artifact_metadata


CELL_GRID = (
    (1, "ppi", "inverse_markup"),
    (3, "ppi", "negative_unemployment_gap"),
    (7, "core_cpi", "inverse_markup"),
    (9, "core_cpi", "negative_unemployment_gap"),
)


def _label(value: float) -> str:
    return f"lambda_{value:g}".replace(".", "p")


def _diagnostic(values: np.ndarray) -> tuple[float, float]:
    idata = az.from_dict({"posterior": {"value": np.asarray(values, float)}})
    rhat = np.asarray(az.rhat(idata, var_names=["value"], method="rank")["value"], float)
    ess = np.asarray(az.ess(idata, var_names=["value"], method="bulk")["value"], float)
    return float(np.nanmax(rhat)), float(np.nanmin(ess))


def _measurement_summary(label: str, lambda_weight: float, posterior) -> pd.DataFrame:
    draws = dict(posterior.draws)
    draws["qtotal"] = draws["qbar"] + draws["qhat"]
    rows = []
    for parameter, values in draws.items():
        rhat, ess = _diagnostic(values)
        flat = np.asarray(values, float).reshape(-1)
        rows.append(
            {
                "interpolation": label,
                "lambda_weight": lambda_weight,
                "parameter": parameter,
                "kind": "path" if np.asarray(values).ndim == 3 else "scalar",
                "mean": float(np.mean(flat)),
                "sd": float(np.std(flat, ddof=1)),
                "ci_2.5": float(np.quantile(flat, 0.025)),
                "ci_97.5": float(np.quantile(flat, 0.975)),
                "max_rhat": rhat,
                "min_bulk_ess": ess,
                "converged_1.01_400": bool(rhat <= 1.01 and ess >= 400),
                "information_ratio_qhat": posterior.information_ratio,
            }
        )
    return pd.DataFrame(rows)


def _state_summary(data, target: np.ndarray, label: str, lambda_weight: float, posterior) -> pd.DataFrame:
    def stats(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        flat = values.reshape(-1, values.shape[-1])
        return np.mean(flat, axis=0), np.quantile(flat, 0.025, axis=0), np.quantile(flat, 0.975, axis=0)

    qbar = stats(posterior.draws["qbar"])
    qhat = stats(posterior.draws["qhat"])
    qtotal = stats(posterior.draws["qbar"] + posterior.draws["qhat"])
    return pd.DataFrame(
        {
            "interpolation": label,
            "lambda_weight": lambda_weight,
            "period": data.periods.astype(str),
            "annual_q4": data.annual_observation,
            "target_q": target,
            "qtotal_mean": qtotal[0],
            "qtotal_ci_2.5": qtotal[1],
            "qtotal_ci_97.5": qtotal[2],
            "qbar_mean": qbar[0],
            "qbar_ci_2.5": qbar[1],
            "qbar_ci_97.5": qbar[2],
            "qhat_mean": qhat[0],
            "qhat_ci_2.5": qhat[1],
            "qhat_ci_97.5": qhat[2],
        }
    )


def _plot_state(frame: pd.DataFrame, out: Path) -> None:
    x = np.arange(len(frame))
    labels = frame["period"].astype(str)
    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    axes[0].fill_between(x, frame["qtotal_ci_2.5"], frame["qtotal_ci_97.5"], alpha=0.22)
    axes[0].plot(x, frame["qtotal_mean"], lw=1.4, label="posterior total q")
    axes[0].plot(x, frame["target_q"], lw=0.8, alpha=0.7, label="interpolation target")
    mask = np.isfinite(frame["annual_q4"])
    axes[0].scatter(x[mask], frame.loc[mask, "annual_q4"], s=16, color="black", label="N_Gustavo Q4")
    axes[0].legend(frameon=False, ncol=3, fontsize=8)
    axes[0].set_ylabel("total q")
    for ax, prefix, title in ((axes[1], "qbar", "slow qbar"), (axes[2], "qhat", "fast qhat")):
        ax.fill_between(x, frame[f"{prefix}_ci_2.5"], frame[f"{prefix}_ci_97.5"], alpha=0.22)
        ax.plot(x, frame[f"{prefix}_mean"], lw=1.3)
        ax.axhline(0.0, color="black", lw=0.5, alpha=0.5)
        ax.set_ylabel(title)
    ticks = np.arange(0, len(frame), 12)
    axes[-1].set_xticks(ticks, labels.iloc[ticks], rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_targets(states: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 4.5))
    for label, frame in states.groupby("interpolation", sort=False):
        ax.plot(np.arange(len(frame)), frame["target_q"], lw=1.1, label=label)
    first = states[states["interpolation"].eq(states["interpolation"].iloc[0])]
    mask = np.isfinite(first["annual_q4"])
    ax.scatter(np.arange(len(first))[mask], first.loc[mask, "annual_q4"], s=18, color="black", label="N_Gustavo Q4")
    ticks = np.arange(0, len(first), 12)
    ax.set_xticks(ticks, first["period"].iloc[ticks], rotation=45, ha="right")
    ax.set_ylabel("interpolated total q")
    ax.legend(frameon=False, ncol=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=BUNDLE_DIR / "config.yaml",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--progress", choices=PROGRESS_STYLES, default="auto")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    mode = cfg["medium"]
    out = Path(args.output_dir or OUTPUT_DIR)
    posterior_dir, tables, figures = out / "posterior", out / "tables", out / "figures"
    for directory in (posterior_dir, tables, figures):
        directory.mkdir(parents=True, exist_ok=True)

    data = load_design_data(
        include_qcew=False,
        sample_start=str(cfg["sample"]["start"]),
        sample_end=str(cfg["sample"]["end"]),
    )
    proxy = data.quarterly["x_inverse_markup"].to_numpy(float)
    proxy_scale = robust_scale(proxy)
    iterations, warmup, thin, chains = (
        int(mode["iterations"]),
        int(mode["warmup"]),
        int(mode["thin"]),
        int(mode["chains"]),
    )
    seed = int(mode["seed"])
    grid = [float(value) for value in cfg["interpolation"]["lambda_grid"]]

    with ProgressReporter(
        chains * iterations,
        label="annual Q4-only decomposition baseline",
        key="interpolation-baseline",
        style=args.progress,
    ) as progress:
        done = [0]

        def baseline_tick() -> None:
            done[0] += 1
            if done[0] < chains * iterations:
                progress.update(done[0])

        baseline_draws = sample_hard_anchor_draws(
            data.annual_observation,
            q_scale=data.q_scale,
            proxy_scale=proxy_scale,
            iterations=iterations,
            warmup=warmup,
            thin=thin,
            chains=chains,
            seed=seed,
            progress_tick=baseline_tick,
        )
    np.savez_compressed(posterior_dir / "annual_q4_only_baseline.npz", **baseline_draws)

    measurement_tables, state_tables, coefficient_tables = [], [], []
    metadata_by_spec: dict[str, object] = {}
    for spec_index, lambda_weight in enumerate(grid):
        label = _label(lambda_weight)
        target, interpolation_metadata = build_q4_anchored_path(
            data.annual_observation,
            proxy,
            lambda_weight=lambda_weight,
        )
        with ProgressReporter(
            chains * iterations,
            label=f"interpolation decomposition [{label}]",
            key=f"interpolation-{label}",
            style=args.progress,
        ) as progress:
            done = [0]

            def tick() -> None:
                done[0] += 1
                if done[0] < chains * iterations:
                    progress.update(done[0])

            draws = sample_hard_anchor_draws(
                target,
                q_scale=data.q_scale,
                proxy_scale=proxy_scale,
                iterations=iterations,
                warmup=warmup,
                thin=thin,
                chains=chains,
                seed=seed + (spec_index + 1) * 1_000_003,
                progress_tick=tick,
            )
        posterior = as_measurement_posterior(
            draws,
            baseline_draws,
            periods=tuple(map(str, data.periods)),
        )
        np.savez_compressed(
            posterior_dir / f"measurement_{label}.npz",
            periods=np.asarray(posterior.periods),
            target_q=target,
            lambda_weight=lambda_weight,
            **{f"C_{key}": value for key, value in posterior.draws.items()},
            **{f"N_{key}": value for key, value in posterior.annual_only_draws.items()},
        )
        measurement_tables.append(_measurement_summary(label, lambda_weight, posterior))
        state_frame = _state_summary(data, target, label, lambda_weight, posterior)
        state_tables.append(state_frame)
        _, q0, _ = reference_draws(data, posterior)

        for cell, price, activity in CELL_GRID:
            fit = fit_cut_model(
                data,
                posterior,
                cell=cell,
                model=str(cfg["inflation"]["model"]),
                transformation=str(cfg["inflation"]["transformation"]),
                q0=q0,
                seed=seed + (spec_index + 1) * 1_000_003 + 500_009,
                price_override=price,
                activity_override=activity,
            )
            _save_fit(posterior_dir / f"{label}_{price}_{activity}_E2_qoq.npz", fit)
            summary = summarize_fit(fit, test_run=False)
            summary["interpolation"] = label
            summary["lambda_weight"] = lambda_weight
            summary["self_referential_markup_cell"] = activity == "inverse_markup"
            summary["exploratory_medium_run"] = True
            coefficient_tables.append(summary)

        _plot_state(state_frame, figures / f"state_paths_{label}.png")
        metadata_by_spec[label] = {
            **interpolation_metadata,
            "information_ratio_qhat": posterior.information_ratio,
            "q0": q0,
            "retained_draws_per_chain": int(draws["qbar"].shape[1]),
        }

    measurement_all = pd.concat(measurement_tables, ignore_index=True)
    states_all = pd.concat(state_tables, ignore_index=True)
    coefficients_all = pd.concat(coefficient_tables, ignore_index=True)
    measurement_all.to_csv(tables / "measurement_diagnostics.csv", index=False)
    states_all.to_csv(tables / "state_paths.csv", index=False)
    coefficients_all.to_csv(tables / "coefficient_summaries.csv", index=False)
    focus = coefficients_all[coefficients_all["parameter"].isin(["kappa_0", "kappa_1", "theta_0", "gamma"])]
    focus.to_csv(tables / "focus_coefficients.csv", index=False)
    _plot_targets(states_all, figures / "interpolation_targets.png")

    manifest = stamp_artifact_metadata(
        {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "revision": str(cfg["revision"]),
            "status": "exploratory interpolation sensitivity; lambda is not estimated",
            "sample": [str(data.periods[0]), str(data.periods[-1])],
            "n_quarters": len(data.periods),
            "annual_measurement": "N_Gustavo at Q4, exact in every interpolated path",
            "interpolation_equation": str(cfg["interpolation"]["equation"]),
            "lambda_grid": grid,
            "measurement_uses_inflation": False,
            "sampling": dict(mode),
            "specifications": metadata_by_spec,
            "limitations": [
                "Lambda is a sensitivity weight and is not identified by annual N.",
                "The first three quarters are a zero-net-change backcast from the first Q4 anchor.",
                "Slow/fast decomposition is identified only by the declared state laws.",
                "Inverse-markup activity cells reuse the timing signal and are sensitivity cells.",
            ],
        }
    )
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
