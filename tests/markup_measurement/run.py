"""Q4-anchored inverse-markup change bridge for the N_Gustavo state.

The experiment is measurement-first and modular: inflation never updates the
competition state.  It fits both an i.i.d. markup measurement error and a
conservative markup-specific AR(1) state, then estimates four QoQ E2 cells:
PPI/core CPI crossed with inverse markup/negative unemployment gap.
"""

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
from tests import _bootstrap  # noqa: F401,E402
from tests._bootstrap import RESULTS_DIR, ROOT


BUNDLE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BUNDLE_DIR / "results"
from nkpc_hsa.config import load_yaml
from nkpc_hsa.phillips.data import load_design_data, robust_scale
from nkpc_hsa.phillips.estimation import _save_fit, summarize_fit
from nkpc_hsa.phillips.inflation import fit_cut_model, reference_draws
from nkpc_hsa.phillips.markup_measurement import sample_markup_measurement_posterior
from nkpc_hsa.progress import ProgressReporter, STYLES as PROGRESS_STYLES
from nkpc_hsa.provenance import stamp_artifact_metadata


CELL_GRID = (
    (1, "ppi", "inverse_markup"),
    (3, "ppi", "negative_unemployment_gap"),
    (7, "core_cpi", "inverse_markup"),
    (9, "core_cpi", "negative_unemployment_gap"),
)


def _diagnostic(values: np.ndarray) -> tuple[float, float]:
    idata = az.from_dict({"posterior": {"value": np.asarray(values, float)}})
    rhat = np.asarray(az.rhat(idata, var_names=["value"], method="rank")["value"], float)
    ess = np.asarray(az.ess(idata, var_names=["value"], method="bulk")["value"], float)
    return float(np.nanmax(rhat)), float(np.nanmin(ess))


def _measurement_summary(name: str, posterior) -> pd.DataFrame:
    draws = dict(posterior.draws)
    draws["qtotal"] = draws["qbar"] + draws["qhat"]
    rows: list[dict[str, object]] = []
    for parameter, values in draws.items():
        rhat, ess = _diagnostic(values)
        flat = np.asarray(values, float).reshape(-1)
        rows.append(
            {
                "measurement_error": name,
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


def _state_summary(data, proxy: np.ndarray, posterior, name: str) -> pd.DataFrame:
    def stats(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        flat = values.reshape(-1, values.shape[-1])
        return (
            np.mean(flat, axis=0),
            np.quantile(flat, 0.025, axis=0),
            np.quantile(flat, 0.975, axis=0),
        )

    qbar = stats(posterior.draws["qbar"])
    qhat = stats(posterior.draws["qhat"])
    qtotal_draws = posterior.draws["qbar"] + posterior.draws["qhat"]
    qtotal = stats(qtotal_draws)
    delta_q = stats(np.diff(qtotal_draws, axis=2))
    return pd.DataFrame(
        {
            "measurement_error": name,
            "period": data.periods.astype(str),
            "annual_q4": data.annual_observation,
            "centered_log_inverse_markup": proxy,
            "delta_log_inverse_markup": np.r_[np.nan, np.diff(proxy)],
            "qtotal_mean": qtotal[0],
            "qtotal_ci_2.5": qtotal[1],
            "qtotal_ci_97.5": qtotal[2],
            "qbar_mean": qbar[0],
            "qbar_ci_2.5": qbar[1],
            "qbar_ci_97.5": qbar[2],
            "qhat_mean": qhat[0],
            "qhat_ci_2.5": qhat[1],
            "qhat_ci_97.5": qhat[2],
            "delta_q_mean": np.r_[np.nan, delta_q[0]],
            "delta_q_ci_2.5": np.r_[np.nan, delta_q[1]],
            "delta_q_ci_97.5": np.r_[np.nan, delta_q[2]],
        }
    )


def _plot_state(frame: pd.DataFrame, alpha_markup: float, out: Path) -> None:
    x = np.arange(len(frame))
    labels = frame["period"].astype(str)
    fig, axes = plt.subplots(4, 1, figsize=(11, 10), sharex=True)
    axes[0].fill_between(x, frame["qtotal_ci_2.5"], frame["qtotal_ci_97.5"], alpha=0.22)
    axes[0].plot(x, frame["qtotal_mean"], lw=1.5, label="posterior total q")
    mask = np.isfinite(frame["annual_q4"])
    axes[0].scatter(x[mask], frame.loc[mask, "annual_q4"], s=18, color="black", label="N_Gustavo Q4")
    axes[0].legend(frameon=False, ncol=2, fontsize=8)
    axes[0].set_ylabel("total q")
    axes[1].fill_between(x, frame["delta_q_ci_2.5"], frame["delta_q_ci_97.5"], alpha=0.22)
    axes[1].plot(x, frame["delta_q_mean"], lw=1.3, label="posterior delta q")
    axes[1].plot(
        x,
        alpha_markup * frame["delta_log_inverse_markup"],
        lw=0.9,
        alpha=0.75,
        label="mean alpha × delta inverse markup",
    )
    axes[1].axhline(0.0, color="black", lw=0.5, alpha=0.5)
    axes[1].legend(frameon=False, ncol=2, fontsize=8)
    axes[1].set_ylabel("quarterly change")
    for ax, prefix, label in (
        (axes[2], "qbar", "slow qbar"),
        (axes[3], "qhat", "fast qhat"),
    ):
        ax.fill_between(x, frame[f"{prefix}_ci_2.5"], frame[f"{prefix}_ci_97.5"], alpha=0.22)
        ax.plot(x, frame[f"{prefix}_mean"], lw=1.4)
        ax.axhline(0.0, color="black", lw=0.5, alpha=0.5)
        ax.set_ylabel(label)
    ticks = np.arange(0, len(frame), 12)
    axes[-1].set_xticks(ticks, labels.iloc[ticks], rotation=45, ha="right")
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
    parser.add_argument("--quick", action="store_true", help="Light smoke run (tiny chains).")
    parser.add_argument(
        "--progress", choices=PROGRESS_STYLES, default="auto"
    )
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    mode = cfg["medium"]
    if args.quick:
        mode = {**mode, "iterations": 300, "warmup": 100, "thin": 1, "chains": 2}
    out = Path(args.output_dir or OUTPUT_DIR)
    posterior_dir = out / "posterior"
    tables = out / "tables"
    figures = out / "figures"
    for directory in (posterior_dir, tables, figures):
        directory.mkdir(parents=True, exist_ok=True)

    data = load_design_data(
        include_qcew=False,
        sample_start=str(cfg["sample"]["start"]),
        sample_end=str(cfg["sample"]["end"]),
    )
    inverse_markup = data.quarterly["x_inverse_markup"].to_numpy(float)
    proxy_center = float(np.mean(inverse_markup))
    proxy = inverse_markup - proxy_center
    proxy_scale = robust_scale(proxy)
    iterations = int(mode["iterations"])
    warmup = int(mode["warmup"])
    thin = int(mode["thin"])
    chains = int(mode["chains"])
    seed = int(mode["seed"])

    measurement_tables = []
    state_tables = []
    coefficient_tables = []
    manifest_specs: dict[str, object] = {}
    for spec_index, error_model in enumerate(cfg["measurement"]["error_models"]):
        total_steps = 2 * chains * iterations
        with ProgressReporter(
            total_steps,
            label=f"markup measurement [{error_model}]",
            key=f"markup-measurement-{error_model}",
            style=args.progress,
        ) as progress:
            completed = [0]

            def tick() -> None:
                completed[0] += 1
                if completed[0] < total_steps:
                    progress.update(completed[0])

            posterior = sample_markup_measurement_posterior(
                data.annual_observation,
                proxy,
                q_scale=data.q_scale,
                proxy_scale=proxy_scale,
                periods=tuple(map(str, data.periods)),
                markup_error=str(error_model),
                iterations=iterations,
                warmup=warmup,
                thin=thin,
                chains=chains,
                seed=seed + spec_index * 1_000_003,
                progress_tick=tick,
            )
        np.savez_compressed(
            posterior_dir / f"measurement_{error_model}.npz",
            periods=np.asarray(posterior.periods),
            proxy_center=proxy_center,
            proxy_scale=proxy_scale,
            information_ratio=posterior.information_ratio,
            **{f"C_{key}": value for key, value in posterior.draws.items()},
            **{f"N_{key}": value for key, value in posterior.annual_only_draws.items()},
        )
        measurement_frame = _measurement_summary(str(error_model), posterior)
        measurement_tables.append(measurement_frame)
        state_frame = _state_summary(data, proxy, posterior, str(error_model))
        state_tables.append(state_frame)
        _, q0, _ = reference_draws(data, posterior)

        fits = []
        for cell, price, activity in CELL_GRID:
            fit = fit_cut_model(
                data,
                posterior,
                cell=cell,
                model=str(cfg["inflation"]["model"]),
                transformation=str(cfg["inflation"]["transformation"]),
                q0=q0,
                seed=seed + spec_index * 1_000_003 + 500_009,
                price_override=price,
                activity_override=activity,
            )
            _save_fit(
                posterior_dir / f"{error_model}_{price}_{activity}_E2_qoq.npz",
                fit,
            )
            summary = summarize_fit(fit, test_run=False)
            summary["measurement_error"] = error_model
            summary["exploratory_medium_run"] = True
            summary["self_referential_markup_cell"] = activity == "inverse_markup"
            coefficient_tables.append(summary)
            fits.append(fit)

        alpha_markup = float(np.mean(posterior.draws["alpha_markup"]))
        _plot_state(state_frame, alpha_markup, figures / f"state_paths_{error_model}.png")
        manifest_specs[str(error_model)] = {
            "information_ratio_qhat": posterior.information_ratio,
            "q0": q0,
            "alpha_markup_mean": alpha_markup,
            "retained_draws_per_chain": int(posterior.draws["qbar"].shape[1]),
        }

    measurement_all = pd.concat(measurement_tables, ignore_index=True)
    states_all = pd.concat(state_tables, ignore_index=True)
    coefficients_all = pd.concat(coefficient_tables, ignore_index=True)
    measurement_all.to_csv(tables / "measurement_diagnostics.csv", index=False)
    states_all.to_csv(tables / "state_paths.csv", index=False)
    coefficients_all.to_csv(tables / "coefficient_summaries.csv", index=False)
    focus = coefficients_all[coefficients_all["parameter"].isin(["kappa_1", "theta_0", "gamma"])].copy()
    focus.to_csv(tables / "focus_coefficients.csv", index=False)

    manifest = stamp_artifact_metadata(
        {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "revision": str(cfg["revision"]),
            "status": "exploratory medium run; not a production estimate",
            "sample": [str(data.periods[0]), str(data.periods[-1])],
            "n_quarters": len(data.periods),
            "annual_measurement": "N_Gustavo at Q4, hard anchor (numerical sd = 1e-6 * q_scale)",
            "quarterly_measurement": str(cfg["measurement"]["equation"]),
            "proxy_center": proxy_center,
            "proxy_scale": proxy_scale,
            "state_law": "qbar random walk with drift; qhat stationary AR(2)",
            "measurement_uses_inflation": False,
            "inflation_cells": [
                {"price": price, "activity": activity} for _, price, activity in CELL_GRID
            ],
            "self_referential_cells": [
                {"price": price, "activity": activity}
                for _, price, activity in CELL_GRID
                if activity == "inverse_markup"
            ],
            "sampling": dict(mode),
            "specifications": manifest_specs,
            "limitations": [
                "The same inverse-markup series measures q and is the activity regressor in two sensitivity cells.",
                "The AR(1) markup-error specification is weakly identified with one quarterly proxy.",
                "This run is intentionally shorter than the frozen production design.",
            ],
        }
    )
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
