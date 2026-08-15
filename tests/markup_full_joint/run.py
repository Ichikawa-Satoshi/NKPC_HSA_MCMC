"""Run the full-joint core-CPI/unemployment E2 model with CPI expectations."""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import json
from pathlib import Path

import arviz as az
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
from nkpc_hsa.phillips.estimation import _save_fit, summarize_fit
from nkpc_hsa.phillips.inflation import fit_cut_model, reference_draws
from experiments.markup_full_joint.functions import fit_markup_full_joint_qoq_e2
from nkpc_hsa.phillips.state import MeasurementPosterior
from nkpc_hsa.progress import ProgressReporter, STYLES as PROGRESS_STYLES
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


def _diagnostic(values: np.ndarray) -> tuple[float, float]:
    idata = az.from_dict({"posterior": {"value": np.asarray(values, float)}})
    rhat = np.asarray(az.rhat(idata, var_names=["value"], method="rank")["value"], float)
    ess = np.asarray(az.ess(idata, var_names=["value"], method="bulk")["value"], float)
    return float(np.nanmax(rhat)), float(np.nanmin(ess))


def _draw_summary(name: str, values: np.ndarray) -> dict[str, object]:
    rhat, ess = _diagnostic(values)
    flat = np.asarray(values, float).reshape(-1)
    return {
        "parameter": name,
        "mean": float(np.mean(flat)),
        "sd": float(np.std(flat, ddof=1)),
        "ci_2.5": float(np.quantile(flat, 0.025)),
        "ci_97.5": float(np.quantile(flat, 0.975)),
        "max_rhat": rhat,
        "min_bulk_ess": ess,
        "converged_1.01_400": bool(rhat <= 1.01 and ess >= 400),
    }


def _plot_states(periods, annual, cut, joint, out: Path) -> None:
    def stats(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        flat = values.reshape(-1, values.shape[-1])
        return np.mean(flat, axis=0), np.quantile(flat, 0.025, axis=0), np.quantile(flat, 0.975, axis=0)

    x = np.arange(len(periods))
    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    for label, draws, color in (
        ("cut", cut["qbar"] + cut["qhat"], "C0"),
        ("full joint", joint["qbar"] + joint["qhat"], "C3"),
    ):
        mean, lo, hi = stats(draws)
        axes[0].plot(x, mean, color=color, lw=1.2, label=label)
        axes[0].fill_between(x, lo, hi, color=color, alpha=0.12)
    mask = np.isfinite(annual)
    axes[0].scatter(x[mask], annual[mask], color="black", s=15, label="N_Gustavo Q4")
    axes[0].set_ylabel("total q")
    axes[0].legend(frameon=False, ncol=3, fontsize=8)
    for ax, key, title in ((axes[1], "qbar", "slow qbar"), (axes[2], "qhat", "fast qhat")):
        for label, draws, color in (("cut", cut[key], "C0"), ("full joint", joint[key], "C3")):
            mean, lo, hi = stats(draws)
            ax.plot(x, mean, color=color, lw=1.2, label=label)
            ax.fill_between(x, lo, hi, color=color, alpha=0.12)
        ax.axhline(0.0, color="black", lw=0.5, alpha=0.5)
        ax.set_ylabel(title)
    ticks = np.arange(0, len(periods), 12)
    axes[-1].set_xticks(ticks, np.asarray(periods)[ticks], rotation=45, ha="right")
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
    out = Path(
        args.output_dir
        or OUTPUT_DIR
    )
    posterior_dir, tables, figures = out / "posterior", out / "tables", out / "figures"
    for directory in (posterior_dir, tables, figures):
        directory.mkdir(parents=True, exist_ok=True)

    data = load_design_data(
        include_qcew=False,
        sample_start=str(cfg["sample"]["start"]),
        sample_end=str(cfg["sample"]["end"]),
    )
    cpi_expectation = load_spf_cpi_quarter_ahead_expectations(data_root() / "raw")
    expectation_column = str(cfg["inflation"]["expectation_column"])
    quarterly = data.quarterly.copy()
    quarterly["expectation"] = cpi_expectation[expectation_column].reindex(
        quarterly.index.to_timestamp(how="end")
    ).to_numpy(float)
    if quarterly["expectation"].isna().any():
        missing = quarterly.index[quarterly["expectation"].isna()].astype(str).tolist()
        raise ValueError(f"CPI expectation is missing at: {missing}")
    data = replace(data, quarterly=quarterly)

    init_path = ROOT / str(cfg["measurement"]["initialization_posterior"])
    measurement = _load_measurement(init_path)
    _, q0, _ = reference_draws(data, measurement)
    seed = int(mode["seed"])

    cut = fit_cut_model(
        data,
        measurement,
        cell=9,
        model="E2",
        transformation="qoq",
        q0=q0,
        seed=seed + 500_009,
        price_override="core_cpi",
        activity_override="negative_unemployment_gap",
    )
    _save_fit(posterior_dir / "cut_core_cpi_negative_unemployment_gap.npz", cut)

    total = int(mode["chains"]) * int(mode["iterations"])
    with ProgressReporter(
        total,
        label="full joint core CPI x negative unemployment gap",
        key="markup-full-joint-core-unemployment",
        style=args.progress,
    ) as progress:
        completed = [0]

        def tick() -> None:
            completed[0] += 1
            if completed[0] < total:
                progress.update(completed[0])

        joint = fit_markup_full_joint_qoq_e2(
            data,
            measurement,
            q0=q0,
            iterations=int(mode["iterations"]),
            warmup=int(mode["warmup"]),
            thin=int(mode["thin"]),
            chains=int(mode["chains"]),
            seed=seed + 1_000_003,
            progress_tick=tick,
        )
    _save_fit(
        posterior_dir / "full_joint_core_cpi_negative_unemployment_gap.npz",
        joint.fit,
        extra_arrays={key: value for key, value in joint.draws.items() if key not in {"coefficients", "sigma_pi"}},
    )

    cut_summary = summarize_fit(cut, test_run=False)
    joint_summary = summarize_fit(joint.fit, test_run=False)
    cut_summary["posterior_type"] = "cut"
    joint_summary["posterior_type"] = "full_joint"
    coefficients = pd.concat([cut_summary, joint_summary], ignore_index=True)
    coefficients.to_csv(tables / "coefficient_summaries.csv", index=False)

    comparison = []
    cut_index = {name: i for i, name in enumerate(cut.coefficient_names)}
    joint_index = {name: i for i, name in enumerate(joint.fit.coefficient_names)}
    for parameter in cut.coefficient_names:
        c = cut.coefficients[:, :, cut_index[parameter]].reshape(-1)
        j = joint.fit.coefficients[:, :, joint_index[parameter]].reshape(-1)
        comparison.append(
            {
                "parameter": parameter,
                "cut_mean": float(np.mean(c)),
                "joint_mean": float(np.mean(j)),
                "joint_minus_cut_in_cut_sd": float((np.mean(j) - np.mean(c)) / np.std(c, ddof=1)),
                "cut_p_positive": float(np.mean(c > 0.0)),
                "joint_p_positive": float(np.mean(j > 0.0)),
            }
        )
    pd.DataFrame(comparison).to_csv(tables / "cut_joint_comparison.csv", index=False)

    diagnostic_rows = []
    for parameter in (
        "d_q", "rho_1", "rho_2", "sigma_qbar", "sigma_qhat",
        "alpha_markup", "sigma_markup", "rho_markup", "sigma_r_markup",
        "max_anchor_error", "qbar", "qhat",
        "level_shift_acceptance",
    ):
        diagnostic_rows.append(_draw_summary(parameter, joint.draws[parameter]))
    pd.DataFrame(diagnostic_rows).to_csv(tables / "state_diagnostics.csv", index=False)
    _plot_states(
        tuple(map(str, data.periods)),
        data.annual_observation,
        measurement.draws,
        joint.draws,
        figures / "cut_vs_full_joint_states.png",
    )

    manifest = stamp_artifact_metadata(
        {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "revision": str(cfg["revision"]),
            "status": "exploratory medium full-joint run; not a production estimate",
            "sample": [str(data.periods[0]), str(data.periods[-1])],
            "n_quarters": len(data.periods),
            "annual_measurement": "N_Gustavo at Q4, hard anchor",
            "markup_proxy": "-log(mu_bus); no reference markup is needed in first differences",
            "expectation": "SPF mean headline CPI CPI3, one-quarter-ahead, annualized log",
            "inflation": "400 * quarterly log change in CPILFESL",
            "activity": "NROU - UNRATE",
            "posterior": "annual N + markup bridge + inflation jointly update qbar and qhat",
            "q0": q0,
            "sampling": dict(mode),
            "initialization_posterior": str(init_path),
            "limitations": [
                "Headline CPI expectation is used as a proxy for core CPI expectation.",
                "The positive markup loading is imposed.",
                "The inflation disturbance is iid in this exploratory run.",
                "Full-joint inflation feedback can make the competition state outcome-dependent.",
            ],
        }
    )
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
