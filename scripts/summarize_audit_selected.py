"""Write reproducible diagnostics and figures for the independent audit subset."""
from __future__ import annotations

import json
from pathlib import Path

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2

import _bootstrap  # noqa: F401
from _bootstrap import DATA_DIR, RESULTS_DIR
from nkpc_hsa.config import configured_data_specs, load_model_config
from nkpc_hsa.inference.wrappers import ESTIMATION_REVISION, _coerce_model_data

OUT = RESULTS_DIR / "audit" / ESTIMATION_REVISION
RUNS = OUT / "selected_runs"


def _diagnostic(idata, name: str) -> tuple[float, float]:
    rhat = np.asarray(az.rhat(idata, var_names=[name], method="rank")[name], float)
    ess = np.asarray(az.ess(idata, var_names=[name], method="bulk")[name], float)
    return float(np.nanmax(rhat)), float(np.nanmin(ess))


def _acf(values: np.ndarray, nlags: int) -> np.ndarray:
    x = np.asarray(values, float) - np.mean(values)
    denominator = float(x @ x)
    return np.array([float(x[lag:] @ x[:-lag]) / denominator for lag in range(1, nlags + 1)])


def _ljung_box(values: np.ndarray, nlags: int) -> tuple[float, float]:
    correlations = _acf(values, nlags)
    n = len(values)
    statistic = n * (n + 2) * sum(
        correlations[lag - 1] ** 2 / (n - lag) for lag in range(1, nlags + 1)
    )
    return float(statistic), float(chi2.sf(statistic, nlags))


def write_tables() -> None:
    scalar_rows: list[dict] = []
    path_rows: list[dict] = []
    loaded: dict[str, object] = {}
    for run in sorted(path for path in RUNS.iterdir() if (path / "posterior.nc").exists()):
        idata = az.from_netcdf(run / "posterior.nc")
        loaded[run.name] = idata
        metadata = json.loads((run / "metadata.json").read_text(encoding="utf-8"))
        for name, variable in idata.posterior.data_vars.items():
            values = np.asarray(variable, float)
            rhat, ess = _diagnostic(idata, name)
            if values.ndim == 2:
                flat = values.reshape(-1)
                scalar_rows.append(
                    {
                        "run": run.name,
                        "sample_start": metadata["sample_start"],
                        "sample_end": metadata["sample_end"],
                        "n_obs": metadata["n_obs"],
                        "parameter": name,
                        "mean": flat.mean(),
                        "ci_2.5": np.quantile(flat, 0.025),
                        "ci_97.5": np.quantile(flat, 0.975),
                        "rhat": rhat,
                        "bulk_ess": ess,
                    }
                )
            else:
                point_rhat = np.asarray(az.rhat(idata, var_names=[name], method="rank")[name], float)
                point_ess = np.asarray(az.ess(idata, var_names=[name], method="bulk")[name], float)
                path_rows.append(
                    {
                        "run": run.name,
                        "path": name,
                        "max_rhat": rhat,
                        "min_bulk_ess": ess,
                        "n_rhat_gt_1.01": int(np.sum(point_rhat > 1.01)),
                        "n_ess_lt_400": int(np.sum(point_ess < 400)),
                    }
                )
    pd.DataFrame(scalar_rows).to_csv(OUT / "selected_scalar_diagnostics.csv", index=False)
    pd.DataFrame(path_rows).to_csv(OUT / "selected_path_diagnostics.csv", index=False)

    frame = pd.read_csv(DATA_DIR / "processed" / "model_ready.csv", parse_dates=["DATE"]).set_index("DATE")
    spec = configured_data_specs(load_model_config())["unemployment_gap_core"]
    data = _coerce_model_data(frame, data_spec=spec)
    residual_rows: list[dict] = []
    for run_name in ("ces_core", "hsa_steady_core_annual_q4"):
        posterior = loaded[run_name].posterior
        mean = lambda name: float(np.asarray(posterior[name], float).mean())
        zeta = data["x"] - mean("phi_1") * data["x_prev"]
        fitted = (
            mean("alpha") * data["pi_prev"]
            + (1 - mean("alpha")) * data["pi_expect"]
            + mean("lambda_ez") * zeta
        )
        if run_name == "ces_core":
            fitted += mean("kappa") * data["x"]
        else:
            fitted += np.asarray(posterior["kappa_t"], float).mean(axis=(0, 1)) * data["x"]
        residual = data["pi"] - fitted
        acfs = _acf(residual, 12)
        for lag in (4, 8, 12):
            statistic, p_value = _ljung_box(residual, lag)
            residual_rows.append(
                {
                    "run": run_name,
                    "lag": lag,
                    "acf_1": acfs[0],
                    "ljung_box": statistic,
                    "p_value": p_value,
                }
            )
    pd.DataFrame(residual_rows).to_csv(OUT / "inflation_residual_diagnostics.csv", index=False)

    pg_rows: list[dict] = []
    for run in sorted((OUT / "pg_pilot").glob("*/posterior.nc")):
        posterior = az.from_netcdf(run).posterior
        for name in ("pg_ess_mean", "pg_ess_min", "pg_moved_frac"):
            values = np.asarray(posterior[name], float).reshape(-1)
            pg_rows.append(
                {
                    "run": run.parent.name,
                    "diagnostic": name,
                    "mean": values.mean(),
                    "q05": np.quantile(values, 0.05),
                    "median": np.median(values),
                    "minimum": values.min(),
                }
            )
    pd.DataFrame(pg_rows).to_csv(OUT / "particle_gibbs_diagnostics.csv", index=False)


def write_figures() -> None:
    figure_dir = OUT / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(DATA_DIR / "processed" / "model_ready.csv", parse_dates=["DATE"]).set_index("DATE")
    sample = data.loc["1982":"2012"]
    sec = pd.read_csv(DATA_DIR / "processed" / "sec_hhi_quarterly.csv")
    sec.index = pd.PeriodIndex(sec["quarter"], freq="Q").to_timestamp(how="end")

    fig, axes = plt.subplots(3, 1, figsize=(11, 9), constrained_layout=True)
    sample[["pi_cpi", "pi_cpi_core", "pi_ppi", "Epi"]].plot(ax=axes[0], lw=1.2)
    axes[0].set_title("Inflation measures and Cleveland Fed one-year expectation")
    sample[["unemp_gap", "output_gap_BN", "output_gap_HP"]].plot(ax=axes[1], lw=1.1)
    axes[1].axhline(0, color="black", lw=0.6)
    axes[1].set_title("Activity-gap sign and scale")
    sample["N_Gustavo"].plot(ax=axes[2], color="tab:blue", label="PCHIP quarterly series")
    q4 = sample.index.to_period("Q").quarter == 4
    axes[2].scatter(sample.index[q4], sample.loc[q4, "N_Gustavo"], s=18, color="black", label="annual Q4 observations")
    axes[2].set_title("Main competition proxy: interpolation versus actual annual timing")
    axes[2].legend()
    fig.savefig(figure_dir / "audit_data_series.png", dpi=170)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 4.5), constrained_layout=True)
    for name in ("effective_firms", "inv_hhi_revw", "inv_hhi_logrevw", "inv_hhi_logrevw_exfin"):
        ax.plot(sec.index, sec[name], label=name)
    ax.set_title("Corrected SEC inverse-HHI aggregates (different economic aggregators)")
    ax.legend(ncol=2, fontsize=8)
    fig.savefig(figure_dir / "audit_sec_aggregates.png", dpi=170)
    plt.close(fig)

    idata = az.from_netcdf(RUNS / "hsa_steady_core_annual_q4" / "posterior.nc")
    posterior = idata.posterior
    fig, axes = plt.subplots(3, 2, figsize=(12, 9), constrained_layout=True)
    for row, name in enumerate(("rho_1", "rho_2", "delta")):
        for chain in range(posterior.sizes["chain"]):
            axes[row, 0].plot(np.asarray(posterior[name][chain]), lw=0.6, label=f"chain {chain + 1}")
        axes[row, 0].set_title(f"{name} trace")
        axes[row, 0].legend(fontsize=8)
    dates = sample.index
    for chain in range(posterior.sizes["chain"]):
        axes[0, 1].plot(dates, np.asarray(posterior["Nbar"][chain]).mean(0), label=f"chain {chain + 1}")
        axes[1, 1].plot(dates, np.asarray(posterior["Nhat"][chain]).mean(0), label=f"chain {chain + 1}")
        axes[2, 1].plot(dates, np.asarray(posterior["kappa_t"][chain]).mean(0), label=f"chain {chain + 1}")
    for ax, title in zip(axes[:, 1], ("Nbar chain means", "Nhat chain means", "kappa_t chain means")):
        ax.set_title(title); ax.legend(fontsize=8)
    fig.savefig(figure_dir / "audit_hsa_trace_geometry.png", dpi=170)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    write_tables()
    write_figures()
    print(f"wrote audit diagnostics to {OUT}")


if __name__ == "__main__":
    main()
