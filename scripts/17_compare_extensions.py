from __future__ import annotations

import json
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd

from _bootstrap import RESULTS_DIR


PRIMARY_RUNS = {
    "A_annual_firms": RESULTS_DIR / "runs" / "hsa_steady_unemployment_gap_core_baseline_annual_q4",
    "B_qcew_joint": RESULTS_DIR / "extensions" / "qcew_joint" / "runs"
    / "hsa_steady_unemployment_gap_core__qcew_joint_baseline_annual_q4",
    "C_sec_inverse_hhi": RESULTS_DIR / "extensions" / "sec_inverse_hhi" / "runs"
    / "hsa_steady_unemployment_gap_core__sec_inverse_hhi_baseline_quarterly_observed",
}

PARAMETER_ALIASES = {
    "alpha": ("alpha",),
    "kappa0": ("kappa_0", "kappa0"),
    "delta": ("delta",),
    "n_N": ("n_N", "n"),
    "rho_N1": ("rho_N1", "rho_1"),
    "rho_N2": ("rho_N2", "rho_2"),
    "sigma_uN": ("sigma_uN", "sigma_u"),
    "sigma_epsN": ("sigma_epsN", "sigma_eps"),
    "sigma_N": ("sigma_N",),
    "n_E": ("n_E",),
    "rho_E1": ("rho_E1",),
    "rho_E2": ("rho_E2",),
    "sigma_uE": ("sigma_uE",),
    "sigma_epsE": ("sigma_epsE",),
    "sigma_E": ("sigma_E",),
    "corr_uN_uE": ("rho_NE",),
}

GRID_SCALARS = tuple(
    dict.fromkeys(alias for aliases in PARAMETER_ALIASES.values() for alias in aliases)
)
GRID_PATHS = ("Nbar", "Nhat", "Ebar", "Ehat", "kappa_t")


def _source_name(posterior, aliases: tuple[str, ...]) -> str | None:
    return next((name for name in aliases if name in posterior), None)


def _diagnostics(idata, name: str) -> tuple[float, float]:
    rhat = np.asarray(az.rhat(idata, var_names=[name], method="rank")[name], dtype=float)
    ess = np.asarray(az.ess(idata, var_names=[name], method="bulk")[name], dtype=float)
    return float(np.nanmax(rhat)), float(np.nanmin(ess))


def _primary_summaries() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    scalar_rows: list[dict[str, object]] = []
    path_rows: list[dict[str, object]] = []
    metadata_rows: list[dict[str, object]] = []
    for specification, run_dir in PRIMARY_RUNS.items():
        metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
        idata = az.from_netcdf(run_dir / "posterior.nc")
        posterior = idata.posterior
        metadata_rows.append(
            {
                "specification": specification,
                "run_dir": str(run_dir),
                "sample_start": metadata.get("sample_start"),
                "sample_end": metadata.get("sample_end"),
                "n_iter": metadata.get("n_iter"),
                "burn": metadata.get("burn"),
                "thin": metadata.get("thin"),
                "chains": metadata.get("chains"),
                "seed": metadata.get("seed"),
                "revision": metadata.get("estimation_revision"),
            }
        )
        for parameter, aliases in PARAMETER_ALIASES.items():
            source = _source_name(posterior, aliases)
            if source is None:
                continue
            values = np.asarray(posterior[source], dtype=float)
            if values.ndim != 2:
                continue
            finite = values[np.isfinite(values)]
            rhat, ess = _diagnostics(idata, source)
            scalar_rows.append(
                {
                    "specification": specification,
                    "parameter": parameter,
                    "source_variable": source,
                    "mean": float(np.mean(finite)),
                    "median": float(np.median(finite)),
                    "ci_2.5": float(np.quantile(finite, 0.025)),
                    "ci_97.5": float(np.quantile(finite, 0.975)),
                    "rhat": rhat,
                    "bulk_ess": ess,
                }
            )
        for path in GRID_PATHS:
            if path not in posterior:
                continue
            values = np.asarray(posterior[path], dtype=float)
            rhat, ess = _diagnostics(idata, path)
            per_time_rhat = np.asarray(az.rhat(idata, var_names=[path], method="rank")[path], dtype=float)
            per_time_ess = np.asarray(az.ess(idata, var_names=[path], method="bulk")[path], dtype=float)
            path_average = np.nanmean(values, axis=-1)
            posterior_mean_path = np.nanmean(values, axis=(0, 1))
            path_rows.append(
                {
                    "specification": specification,
                    "path": path,
                    "max_rhat": rhat,
                    "min_bulk_ess": ess,
                    "median_bulk_ess": float(np.nanmedian(per_time_ess)),
                    "mean_posterior_sd": float(np.nanmean(np.nanstd(values, axis=(0, 1), ddof=1))),
                    "average_path_mean": float(np.nanmean(path_average)),
                    "average_path_median": float(np.nanmedian(path_average)),
                    "average_path_ci_2.5": float(np.nanquantile(path_average, 0.025)),
                    "average_path_ci_97.5": float(np.nanquantile(path_average, 0.975)),
                    "posterior_mean_path_min": float(np.nanmin(posterior_mean_path)),
                    "posterior_mean_path_max": float(np.nanmax(posterior_mean_path)),
                    "n_time_rhat_gt_1.01": int(np.sum(per_time_rhat > 1.01)),
                    "n_time_bulk_ess_lt_400": int(np.sum(per_time_ess < 400)),
                }
            )
    return pd.DataFrame(scalar_rows), pd.DataFrame(path_rows), pd.DataFrame(metadata_rows)


def _grid_convergence() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    extension_roots = {
        "qcew_joint": RESULTS_DIR / "extensions" / "qcew_joint" / "runs",
        "sec_inverse_hhi": RESULTS_DIR / "extensions" / "sec_inverse_hhi" / "runs",
    }
    for extension, root in extension_roots.items():
        for run_dir in sorted(path.parent for path in root.glob("*/posterior.nc")):
            idata = az.from_netcdf(run_dir / "posterior.nc")
            posterior = idata.posterior
            names = [name for name in GRID_SCALARS + GRID_PATHS if name in posterior]
            worst_rhat = -np.inf
            min_ess = np.inf
            worst_rhat_variable = ""
            min_ess_variable = ""
            for name in names:
                rhat, ess = _diagnostics(idata, name)
                if rhat > worst_rhat:
                    worst_rhat, worst_rhat_variable = rhat, name
                if ess < min_ess:
                    min_ess, min_ess_variable = ess, name
            rows.append(
                {
                    "extension": extension,
                    "run": run_dir.name,
                    "max_rhat": worst_rhat,
                    "max_rhat_variable": worst_rhat_variable,
                    "min_bulk_ess": min_ess,
                    "min_bulk_ess_variable": min_ess_variable,
                    "converged_rhat_1.01_ess_400": bool(worst_rhat <= 1.01 and min_ess >= 400),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    output_dir = RESULTS_DIR / "extensions" / "comparison"
    output_dir.mkdir(parents=True, exist_ok=True)
    scalar, paths, metadata = _primary_summaries()
    grid = _grid_convergence()
    scalar.to_csv(output_dir / "primary_scalar_comparison.csv", index=False)
    paths.to_csv(output_dir / "primary_path_diagnostics.csv", index=False)
    metadata.to_csv(output_dir / "primary_run_metadata.csv", index=False)
    grid.to_csv(output_dir / "extension_grid_convergence.csv", index=False)
    counts = grid.groupby("extension")["converged_rhat_1.01_ess_400"].agg(["count", "sum"])
    print(f"Wrote comparison artifacts to {output_dir}")
    print(counts.to_string())


if __name__ == "__main__":
    main()
