from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy.stats import gaussian_kde, norm

from _bootstrap import ROOT
from nkpc_hsa.config import configured_data_specs, load_model_config
from nkpc_hsa.inference.wrappers import ESTIMATION_REVISION
from nkpc_hsa.report.cpi_ppi_spec import (
    annual_q4_run_keys,
    INFLATION_SPECS,
    MODEL_LABELS,
    MODEL_ORDER,
    PRIMARY_SPECS,
    PRIOR_ORDER,
    report_run_keys,
)


OUT_TABLES = ROOT / "results" / "tables" / "cpi_ppi_report"
OUT_FIGURES = ROOT / "results" / "figures" / "cpi_ppi_report"
RHAT_LIMIT = 1.01
ESS_LIMIT = 400.0

def _run_key(path: Path, idata) -> tuple[str, str]:
    attrs = getattr(idata, "attrs", {})
    return str(attrs.get("run_id", "")), path.parent.name


def _load_runs(
    runs_dir: Path,
    *,
    min_iter: int,
    competition_frequency: str,
) -> dict[tuple[str, str, str], tuple[Path, object]]:
    selected: dict[tuple[str, str, str], tuple[Path, object]] = {}
    for posterior in sorted(runs_dir.glob("*/posterior.nc")):
        idata = az.from_netcdf(posterior)
        attrs = getattr(idata, "attrs", {})
        if str(attrs.get("estimation_revision", "")) != ESTIMATION_REVISION:
            continue
        if int(attrs.get("n_iter", 0) or 0) < min_iter:
            continue
        if str(attrs.get("period", "full") or "full") != "full":
            continue
        if str(attrs.get("constraint_spec", "unrestricted") or "unrestricted") != "unrestricted":
            continue
        if str(attrs.get("competition_measurement_frequency", "quarterly_interpolated")) != competition_frequency:
            continue
        if str(attrs.get("n_transform", "")) != "log100_centered10":
            continue
        model = str(attrs.get("model", ""))
        data_spec = str(attrs.get("data_spec", ""))
        prior = str(attrs.get("prior_spec", "baseline") or "baseline")
        if model not in MODEL_ORDER:
            continue
        priors_path = posterior.parent / "priors.json"
        if priors_path.exists():
            idata.attrs["run_priors"] = json.loads(priors_path.read_text(encoding="utf-8"))
        key = (model, data_spec, prior)
        current = selected.get(key)
        if current is None or _run_key(posterior, idata) >= _run_key(current[0], current[1]):
            selected[key] = (posterior, idata)
    return selected


def _draws(idata, parameter: str) -> np.ndarray | None:
    if parameter not in idata.posterior:
        return None
    values = np.asarray(idata.posterior[parameter], dtype=float).reshape(-1)
    values = values[np.isfinite(values)]
    return values if values.size else None


def _summary(idata, parameter: str) -> dict[str, float] | None:
    values = _draws(idata, parameter)
    if values is None:
        return None
    return {
        "mean": float(np.mean(values)),
        "lo": float(np.quantile(values, 0.025)),
        "hi": float(np.quantile(values, 0.975)),
        "p_gt_0": float(np.mean(values > 0.0)),
    }


def _path_summary(idata, parameter: str) -> dict[str, float] | None:
    if parameter not in idata.posterior:
        return None
    values = np.asarray(idata.posterior[parameter], dtype=float)
    if values.ndim < 3:
        return None
    paths = values.reshape(-1, values.shape[-1])
    return {
        "start": float(np.nanmean(paths[:, 0])),
        "end": float(np.nanmean(paths[:, -1])),
    }


def _bf10(idata, parameter: str) -> float | None:
    values = _draws(idata, parameter)
    if values is None or values.size < 20 or np.std(values, ddof=1) <= 0:
        return None
    priors = idata.attrs.get("run_priors", {})
    prior = priors.get(parameter)
    if prior is None:
        return None
    if isinstance(prior, dict):
        prior_mean, prior_sd = float(prior["mean"]), float(prior["sd"])
    else:
        prior_mean, prior_sd = float(prior[0]), float(prior[1])
    posterior_at_zero = float(gaussian_kde(values)([0.0])[0])
    prior_at_zero = float(norm.pdf(0.0, loc=prior_mean, scale=prior_sd))
    bf01 = posterior_at_zero / max(prior_at_zero, 1e-300)
    return float(1.0 / max(bf01, 1e-300))


def _fmt(summary: dict[str, float] | None) -> str:
    if summary is None:
        return "--"
    return f"{summary['mean']:+.3f} [{summary['lo']:+.3f}, {summary['hi']:+.3f}]"


def _fmt_num(value: float | None, digits: int = 1) -> str:
    if value is None or not np.isfinite(value):
        return "--"
    if value >= 1000:
        return ">999"
    return f"{value:.{digits}f}"


def _diagnostics(idata) -> dict[str, float | bool]:
    scalar_names = [
        name
        for name in ["alpha", "kappa", "kappa_0", "delta", "theta", "theta_0", "gamma", "rho_1", "rho_2"]
        if name in idata.posterior
    ]
    rhats = [float(np.asarray(az.rhat(np.asarray(idata.posterior[name], dtype=float)))) for name in scalar_names]
    esses = [
        float(np.asarray(az.ess(np.asarray(idata.posterior[name], dtype=float), method="bulk")))
        for name in scalar_names
    ]
    max_rhat = max(rhats) if rhats else np.nan
    min_ess = min(esses) if esses else np.nan
    converged = bool(np.isfinite(max_rhat) and np.isfinite(min_ess) and max_rhat <= RHAT_LIMIT and min_ess >= ESS_LIMIT)
    return {"max_rhat": max_rhat, "min_ess": min_ess, "converged": converged}


def _marked(value: str, converged: bool) -> str:
    if value == "--":
        return value
    return value if converged else value + r"\textsuperscript{$\dagger$}"


def _write_latex(df: pd.DataFrame, name: str, columns: list[str]) -> None:
    OUT_TABLES.mkdir(parents=True, exist_ok=True)
    display = df.reindex(columns=columns).copy()
    display.to_latex(
        OUT_TABLES / f"{name}.tex",
        index=False,
        escape=False,
        na_rep="--",
        column_format="l" * len(columns),
    )
    df.to_csv(OUT_TABLES / f"{name}.csv", index=False)


def build_hsa_steady_activity_table(runs) -> pd.DataFrame:
    rows = []
    for inflation, activity_specs in INFLATION_SPECS.items():
        for activity, data_spec in activity_specs.items():
            item = runs.get(("hsa_steady", data_spec, "baseline"))
            if item is None:
                continue
            _, idata = item
            diagnostics = _diagnostics(idata)
            delta = _summary(idata, "delta")
            path = _path_summary(idata, "kappa_t") or {}
            rows.append(
                {
                    "inflation": inflation,
                    "activity": activity,
                    "delta": _marked(_fmt(delta), bool(diagnostics["converged"])),
                    "BF10": _fmt_num(_bf10(idata, "delta")),
                    "kappa_start": path.get("start", np.nan),
                    "kappa_end": path.get("end", np.nan),
                    "delta_mean": np.nan if delta is None else delta["mean"],
                    "delta_lo": np.nan if delta is None else delta["lo"],
                    "delta_hi": np.nan if delta is None else delta["hi"],
                    "converged": bool(diagnostics["converged"]),
                }
            )
    table = pd.DataFrame(rows)
    if not table.empty:
        table["kappa path"] = table.apply(
            lambda r: _marked(
                f"{r['kappa_start']:+.3f} $\\rightarrow$ {r['kappa_end']:+.3f}",
                bool(r["converged"]),
            ),
            axis=1,
        )
    _write_latex(table, "hsa_steady_by_activity", ["inflation", "activity", "delta", "BF10", "kappa path"])
    return table


def build_model_table(runs) -> pd.DataFrame:
    rows = []
    for model in MODEL_ORDER:
        for inflation, data_spec in PRIMARY_SPECS.items():
            item = runs.get((model, data_spec, "baseline"))
            if item is None:
                continue
            _, idata = item
            diagnostics = _diagnostics(idata)
            slope_name = "kappa" if model in {"ces", "hsa_dynamic"} else "kappa_0"
            entry_name = "theta" if model in {"hsa_dynamic", "hsa_const_theta"} else "theta_0"
            delta = _summary(idata, "delta")
            gamma = _summary(idata, "gamma")
            entry = _summary(idata, entry_name)
            path = _path_summary(idata, "kappa_t")
            rows.append(
                {
                    "model": MODEL_LABELS[model],
                    "inflation": inflation,
                    "slope": _fmt(_summary(idata, slope_name)),
                    "delta": _fmt(delta),
                    "BF10(delta)": _fmt_num(_bf10(idata, "delta")),
                    "entry": _fmt(entry),
                    "gamma": _fmt(gamma),
                    "kappa path": "--" if path is None else f"{path['start']:+.3f} $\\rightarrow$ {path['end']:+.3f}",
                    "diagnostics": "OK" if diagnostics["converged"] else r"要注意$^{\dagger}$",
                }
            )
    table = pd.DataFrame(rows)
    _write_latex(
        table,
        "unemployment_by_model",
        ["model", "inflation", "slope", "delta", "BF10(delta)", "entry", "gamma", "kappa path", "diagnostics"],
    )
    return table


def build_output_gap_model_tables(runs) -> pd.DataFrame:
    """Write one complete model-comparison table for each output-gap definition."""
    rows: list[dict[str, object]] = []
    for activity in ["HP output gap", "BN output gap"]:
        for model in MODEL_ORDER:
            for inflation, activity_specs in INFLATION_SPECS.items():
                data_spec = activity_specs[activity]
                item = runs.get((model, data_spec, "baseline"))
                if item is None:
                    continue
                _, idata = item
                diagnostics = _diagnostics(idata)
                slope_name = "kappa" if model in {"ces", "hsa_dynamic"} else "kappa_0"
                entry_name = "theta" if model in {"hsa_dynamic", "hsa_const_theta"} else "theta_0"
                path = _path_summary(idata, "kappa_t")
                rows.append(
                    {
                        "activity": activity,
                        "model": MODEL_LABELS[model],
                        "inflation": inflation,
                        "slope": _marked(_fmt(_summary(idata, slope_name)), bool(diagnostics["converged"])),
                        "delta": _marked(_fmt(_summary(idata, "delta")), bool(diagnostics["converged"])),
                        "BF10(delta)": _fmt_num(_bf10(idata, "delta")),
                        "entry": _marked(_fmt(_summary(idata, entry_name)), bool(diagnostics["converged"])),
                        "gamma": _marked(_fmt(_summary(idata, "gamma")), bool(diagnostics["converged"])),
                        "kappa path": "--" if path is None else f"{path['start']:+.3f} $\\rightarrow$ {path['end']:+.3f}",
                        "diagnostics": "OK" if diagnostics["converged"] else r"要注意$^{\dagger}$",
                    }
                )
    table = pd.DataFrame(rows)
    columns = ["model", "inflation", "slope", "delta", "BF10(delta)", "entry", "gamma", "kappa path", "diagnostics"]
    for activity, filename in [("HP output gap", "output_gap_hp_by_model"), ("BN output gap", "output_gap_bn_by_model")]:
        _write_latex(table.loc[table["activity"] == activity], filename, columns)
    return table


def _paired_difference(lhs: np.ndarray, rhs: np.ndarray, *, seed: int = 81731) -> np.ndarray:
    """Monte Carlo draws for two independently estimated posterior quantities."""
    n = min(lhs.size, rhs.size)
    rng = np.random.default_rng(seed)
    return lhs[:n] - rhs[rng.permutation(rhs.size)[:n]]


def _summarize_draws(values: np.ndarray) -> dict[str, float]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    return {
        "mean": float(np.mean(finite)),
        "lo": float(np.quantile(finite, 0.025)),
        "hi": float(np.quantile(finite, 0.975)),
        "p_lt_0": float(np.mean(finite < 0.0)),
    }


def _hsa_implied_ovb(idata, data_spec: dict[str, object], data: pd.DataFrame) -> np.ndarray:
    """Slope bias from omitting -theta*Nhat, conditional on a_t and zeta_t."""
    cols = {
        "pi": str(data_spec["pi_col"]),
        "pi_prev": str(data_spec["pi_prev_col"]),
        "pi_expect": str(data_spec["pi_expect_col"]),
        "x": str(data_spec["x_col"]),
        "x_prev": str(data_spec["x_prev_col"]),
        "N": str(data_spec["n_col"]),
    }
    sample = data[list(cols.values())].dropna()
    x = sample[cols["x"]].to_numpy(dtype=float)
    x_prev = sample[cols["x_prev"]].to_numpy(dtype=float)
    a = (sample[cols["pi_prev"]] - sample[cols["pi_expect"]]).to_numpy(dtype=float)
    nhat = np.asarray(idata.posterior["Nhat"], dtype=float).reshape(-1, len(x))
    theta = _draws(idata, "theta")
    phi = _draws(idata, "phi_1")
    if theta is None or phi is None or nhat.shape[0] != theta.size or theta.size != phi.size:
        raise ValueError("HSA dynamic posterior arrays are not aligned for the OVB diagnostic.")
    out = np.empty(theta.size)
    for s in range(theta.size):
        zeta = x - phi[s] * x_prev
        controls = np.column_stack([a, zeta])
        x_resid = x - controls @ np.linalg.lstsq(controls, x, rcond=None)[0]
        omitted = -theta[s] * nhat[s]
        omitted_resid = omitted - controls @ np.linalg.lstsq(controls, omitted, rcond=None)[0]
        denom = float(x_resid @ x_resid)
        out[s] = float(x_resid @ omitted_resid / denom) if denom > 0 else np.nan
    return out


def build_ces_hsa_bias_table(runs, *, command_prefix: str = "") -> pd.DataFrame:
    """Compare naive CES and HSA-dynamic slopes and the HSA-implied OVB."""
    config = load_model_config(ROOT / "configs" / "models.yaml")
    specs = configured_data_specs(config, list(config.get("data_specs", {})))
    data = pd.read_csv(ROOT / "data" / "processed" / "model_ready.csv", parse_dates=["DATE"]).set_index("DATE")
    comparisons = [
        ("逆マークアップ（理論対応）", "inv_markup"),
        ("Headline CPI / 失業ギャップ", PRIMARY_SPECS["Headline CPI"]),
        ("Core CPI / 失業ギャップ", PRIMARY_SPECS["Core CPI"]),
        ("PPI / 失業ギャップ", PRIMARY_SPECS["PPI"]),
    ]
    rows: list[dict[str, object]] = []
    direct_macros: list[str] = ["% Generated by scripts/12_build_cpi_ppi_report.py; do not edit by hand."]
    for label, data_spec_name in comparisons:
        ces_item = runs.get(("ces", data_spec_name, "baseline"))
        hsa_item = runs.get(("hsa_dynamic", data_spec_name, "baseline"))
        if ces_item is None or hsa_item is None:
            continue
        ces_idata, hsa_idata = ces_item[1], hsa_item[1]
        ces_draws = _draws(ces_idata, "kappa")
        hsa_draws = _draws(hsa_idata, "kappa")
        if ces_draws is None or hsa_draws is None:
            continue
        difference = _summarize_draws(_paired_difference(ces_draws, hsa_draws))
        ovb = _summarize_draws(_hsa_implied_ovb(hsa_idata, specs[data_spec_name], data))
        converged = bool(_diagnostics(ces_idata)["converged"] and _diagnostics(hsa_idata)["converged"])
        rows.append(
            {
                "specification": label,
                "CES kappa": _marked(_fmt(_summary(ces_idata, "kappa")), converged),
                "HSA kappa": _marked(_fmt(_summary(hsa_idata, "kappa")), converged),
                "CES-HSA": _marked(_fmt(difference), converged),
                "Pr(CES-HSA<0)": f"{difference['p_lt_0']:.3f}",
                "HSA-implied OVB": _marked(_fmt(ovb), converged),
                "Pr(OVB<0)": f"{ovb['p_lt_0']:.3f}",
                "diagnostics": "OK" if converged else r"要注意$^{\dagger}$",
            }
        )
        if data_spec_name == "inv_markup":
            direct_macros.extend(
                [
                    rf"\providecommand{{\{command_prefix}BiasDirectCES}}{{{_fmt(_summary(ces_idata, 'kappa'))}}}",
                    rf"\providecommand{{\{command_prefix}BiasDirectHSA}}{{{_fmt(_summary(hsa_idata, 'kappa'))}}}",
                    rf"\providecommand{{\{command_prefix}BiasDirectDifference}}{{{_fmt(difference)}}}",
                    rf"\providecommand{{\{command_prefix}BiasDirectProbability}}{{{difference['p_lt_0']:.3f}}}",
                    rf"\providecommand{{\{command_prefix}BiasDirectOVB}}{{{_fmt(ovb)}}}",
                    rf"\providecommand{{\{command_prefix}BiasDirectOVBProbability}}{{{ovb['p_lt_0']:.3f}}}",
                ]
            )
    table = pd.DataFrame(rows)
    _write_latex(
        table,
        "ces_hsa_kappa_bias",
        ["specification", "CES kappa", "HSA kappa", "CES-HSA", "Pr(CES-HSA<0)", "HSA-implied OVB", "Pr(OVB<0)", "diagnostics"],
    )
    (OUT_TABLES / "bias_macros.tex").write_text("\n".join(direct_macros) + "\n", encoding="utf-8")
    return table


def build_prior_table(runs) -> pd.DataFrame:
    rows = []
    for model in MODEL_ORDER:
        for inflation, data_spec in PRIMARY_SPECS.items():
            for prior in PRIOR_ORDER:
                item = runs.get((model, data_spec, prior))
                if item is None:
                    continue
                _, idata = item
                diagnostics = _diagnostics(idata)
                parameter = "kappa" if model in {"ces", "hsa_dynamic"} else "delta"
                rows.append(
                    {
                        "model": MODEL_LABELS[model],
                        "inflation": inflation,
                        "prior": prior,
                        "parameter": f"${parameter}$",
                        "posterior": _marked(_fmt(_summary(idata, parameter)), bool(diagnostics["converged"])),
                        "BF10": _fmt_num(_bf10(idata, parameter)),
                        "converged": bool(diagnostics["converged"]),
                    }
                )
    table = pd.DataFrame(rows)
    _write_latex(table, "prior_sensitivity_by_model", ["model", "inflation", "prior", "parameter", "posterior", "BF10"])
    if not table.empty:
        compact = table.pivot(index=["model", "inflation", "parameter"], columns="prior", values="posterior").reset_index()
        for prior in PRIOR_ORDER:
            if prior not in compact:
                compact[prior] = "--"
        _write_latex(
            compact,
            "prior_sensitivity_compact",
            ["model", "inflation", "parameter", "baseline", "weak", "tight"],
        )
        steady = compact[compact["model"] == MODEL_LABELS["hsa_steady"]].copy()
        _write_latex(
            steady,
            "prior_sensitivity_hsa_steady",
            ["inflation", "parameter", "baseline", "weak", "tight"],
        )
    return table


def build_run_manifest(runs) -> pd.DataFrame:
    rows = []
    for (model, data_spec, prior), (path, idata) in sorted(runs.items()):
        attrs = getattr(idata, "attrs", {})
        diagnostics = _diagnostics(idata)
        rows.append(
            {
                "model": MODEL_LABELS[model],
                "data spec": r"\texttt{" + data_spec.replace("_", r"\_") + "}",
                "prior": prior,
                "N frequency": str(
                    attrs.get("competition_measurement_frequency", "quarterly_interpolated")
                ).replace("quarterly_interpolated", "PCHIP quarterly").replace("annual_q4", "annual Q4"),
                "T": int(attrs.get("n_obs", 0) or 0),
                "run id": r"\texttt{" + str(attrs.get("run_id", path.parent.name)).replace("_", r"\_") + "}",
                "max Rhat": diagnostics["max_rhat"],
                "min bulk ESS": diagnostics["min_ess"],
                "converged": bool(diagnostics["converged"]),
            }
        )
    table = pd.DataFrame(rows)
    _write_latex(
        table,
        "report_run_manifest",
        ["model", "data spec", "prior", "N frequency", "T", "run id", "max Rhat", "min bulk ESS"],
    )
    if not table.empty:
        summary = (
            table.groupby("model", sort=False)
            .agg(
                runs=("model", "size"),
                warnings=("converged", lambda x: int((~x).sum())),
                **{"max Rhat": ("max Rhat", "max"), "min bulk ESS": ("min bulk ESS", "min")},
            )
            .reset_index()
        )
        summary["max Rhat"] = summary["max Rhat"].map(lambda x: f"{x:.3f}")
        summary["min bulk ESS"] = summary["min bulk ESS"].map(lambda x: f"{x:.0f}")
        _write_latex(summary, "convergence_summary", ["model", "runs", "warnings", "max Rhat", "min bulk ESS"])

        warnings = table.loc[~table["converged"]].copy()
        warnings = warnings.sort_values(["max Rhat", "min bulk ESS"], ascending=[False, True])
        warning_view = warnings.head(15).copy()
        warning_view["max Rhat"] = warning_view["max Rhat"].map(lambda x: f"{x:.3f}")
        warning_view["min bulk ESS"] = warning_view["min bulk ESS"].map(lambda x: f"{x:.0f}")
        _write_latex(
            warning_view,
            "convergence_warnings",
            ["model", "data spec", "prior", "max Rhat", "min bulk ESS"],
        )
    return table


def build_frequency_outputs(
    runs,
    *,
    tables_dir: Path,
    figures_dir: Path,
    command_prefix: str,
) -> None:
    """Build the same report artifacts for one competition-frequency design."""
    global OUT_TABLES, OUT_FIGURES
    OUT_TABLES = tables_dir
    OUT_FIGURES = figures_dir
    OUT_TABLES.mkdir(parents=True, exist_ok=True)
    OUT_FIGURES.mkdir(parents=True, exist_ok=True)

    hsa_table = build_hsa_steady_activity_table(runs)
    build_model_table(runs)
    build_output_gap_model_tables(runs)
    build_ces_hsa_bias_table(runs, command_prefix=command_prefix)
    build_prior_table(runs)
    build_diagnostics_table(runs)
    build_primary_parameter_state_diagnostics(runs)
    build_run_manifest(runs)
    write_result_macros(runs, command_prefix=command_prefix)
    save_delta_forest(hsa_table)
    save_output_gap_delta_forest(hsa_table)
    save_additional_parameter_forests(runs)
    save_kappa_path_comparison(runs)
    save_main_competition_decomposition(runs)


def write_result_macros(
    runs,
    *,
    command_prefix: str = "",
    filename: str = "result_macros.tex",
) -> Path:
    OUT_TABLES.mkdir(parents=True, exist_ok=True)
    diagnostics = [_diagnostics(idata) for _, idata in runs.values()]
    warning_count = sum(not bool(item["converged"]) for item in diagnostics)
    lines = [
        "% Generated by scripts/12_build_cpi_ppi_report.py; do not edit by hand.",
        rf"\providecommand{{\{command_prefix}ReportEstimationRevision}}{{\texttt{{{ESTIMATION_REVISION}}}}}",
        rf"\providecommand{{\{command_prefix}ReportRunCount}}{{{len(runs)}}}",
        rf"\providecommand{{\{command_prefix}ReportWarningCount}}{{{warning_count}}}",
    ]
    prefixes = {"Headline CPI": "HeadlineUnemp", "Core CPI": "CoreUnemp", "PPI": "PPIUnemp"}
    for inflation, data_spec in PRIMARY_SPECS.items():
        prefix = prefixes[inflation]
        item = runs.get(("hsa_steady", data_spec, "baseline"))
        if item is None:
            continue
        _, idata = item
        delta = _summary(idata, "delta")
        bf10 = _bf10(idata, "delta")
        path = _path_summary(idata, "kappa_t")
        lines.append(rf"\providecommand{{\{command_prefix}{prefix}Delta}}{{{_fmt(delta)}}}")
        lines.append(rf"\providecommand{{\{command_prefix}{prefix}DeltaBF}}{{{_fmt_num(bf10)}}}")
        if path is not None:
            lines.append(rf"\providecommand{{\{command_prefix}{prefix}KappaStart}}{{{path['start']:+.3f}}}")
            lines.append(rf"\providecommand{{\{command_prefix}{prefix}KappaEnd}}{{{path['end']:+.3f}}}")
    output_prefixes = {
        ("Headline CPI", "HP output gap"): "HeadlineHP",
        ("Core CPI", "HP output gap"): "CoreHP",
        ("PPI", "HP output gap"): "PPIHP",
        ("Headline CPI", "BN output gap"): "HeadlineBN",
        ("Core CPI", "BN output gap"): "CoreBN",
        ("PPI", "BN output gap"): "PPIBN",
    }
    for (inflation, activity), prefix in output_prefixes.items():
        data_spec = INFLATION_SPECS[inflation][activity]
        item = runs.get(("hsa_steady", data_spec, "baseline"))
        if item is None:
            continue
        _, idata = item
        lines.append(rf"\providecommand{{\{command_prefix}{prefix}Delta}}{{{_fmt(_summary(idata, 'delta'))}}}")
        lines.append(rf"\providecommand{{\{command_prefix}{prefix}DeltaBF}}{{{_fmt_num(_bf10(idata, 'delta'))}}}")
    target = OUT_TABLES / filename
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return target


def build_diagnostics_table(runs) -> pd.DataFrame:
    rows = []
    for model in MODEL_ORDER:
        data_spec = PRIMARY_SPECS["PPI"]
        item = runs.get((model, data_spec, "baseline"))
        if item is None:
            continue
        _, idata = item
        diagnostics = _diagnostics(idata)
        rows.append(
            {
                "model": MODEL_LABELS[model],
                "max Rhat": diagnostics["max_rhat"],
                "min bulk ESS": diagnostics["min_ess"],
                "status": "OK" if diagnostics["converged"] else "要注意",
            }
        )
    table = pd.DataFrame(rows)
    if not table.empty:
        table["max Rhat"] = table["max Rhat"].map(lambda x: f"{x:.3f}")
        table["min bulk ESS"] = table["min bulk ESS"].map(lambda x: f"{x:.0f}")
    _write_latex(table, "ppi_diagnostics", ["model", "max Rhat", "min bulk ESS", "status"])
    return table


def _array_diagnostics(idata, parameter: str) -> tuple[float, float]:
    values = idata.posterior[parameter]
    rhat = np.asarray(az.rhat(values), dtype=float)
    ess = np.asarray(az.ess(values, method="bulk"), dtype=float)
    return float(np.nanmax(rhat)), float(np.nanmin(ess))


def build_primary_parameter_state_diagnostics(runs) -> pd.DataFrame:
    """Separate mixing of reported coefficients from the latent state paths."""
    rows: list[dict[str, object]] = []
    for inflation, data_spec in PRIMARY_SPECS.items():
        item = runs.get(("hsa_steady", data_spec, "baseline"))
        if item is None:
            continue
        idata = item[1]
        row: dict[str, object] = {"inflation": inflation}
        for parameter, label in [
            ("delta", "delta"),
            ("kappa_0", "kappa0"),
            ("kappa_t", "kappa path"),
            ("Nbar", "Nbar path"),
            ("Nhat", "Nhat path"),
        ]:
            max_rhat, min_ess = _array_diagnostics(idata, parameter)
            row[f"{label} Rhat"] = f"{max_rhat:.3f}"
            row[f"{label} ESS"] = f"{min_ess:.0f}"
        rows.append(row)
    table = pd.DataFrame(rows)
    _write_latex(
        table,
        "primary_parameter_state_diagnostics",
        [
            "inflation",
            "delta Rhat", "delta ESS",
            "kappa0 Rhat", "kappa0 ESS",
            "kappa path Rhat", "kappa path ESS",
            "Nbar path Rhat", "Nbar path ESS",
            "Nhat path Rhat", "Nhat path ESS",
        ],
    )
    return table


def save_delta_forest(table: pd.DataFrame) -> None:
    if table.empty:
        return
    OUT_FIGURES.mkdir(parents=True, exist_ok=True)
    plot = table.dropna(subset=["delta_mean", "delta_lo", "delta_hi"]).reset_index(drop=True)
    y = np.arange(len(plot))[::-1]
    colors = {"Headline CPI": "#4477AA", "Core CPI": "#228833", "PPI": "#CC6677"}
    fig, ax = plt.subplots(figsize=(7.4, 5.4))
    for i, row in plot.iterrows():
        ax.errorbar(
            row["delta_mean"], y[i],
            xerr=[[row["delta_mean"] - row["delta_lo"]], [row["delta_hi"] - row["delta_mean"]]],
            fmt="o", color=colors[row["inflation"]], capsize=3,
        )
    ax.axvline(0, color="black", lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{r.inflation} / {r.activity}" for r in plot.itertuples()])
    ax.set_xlabel(r"Competition dependence $\delta$ (posterior mean and 95% interval)")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT_FIGURES / "delta_by_inflation_activity.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_output_gap_delta_forest(table: pd.DataFrame) -> None:
    if table.empty:
        return
    plot = table.loc[table["activity"].isin(["HP output gap", "BN output gap"])].copy()
    plot = plot.dropna(subset=["delta_mean", "delta_lo", "delta_hi"]).reset_index(drop=True)
    y = np.arange(len(plot))[::-1]
    colors = {"Headline CPI": "#4477AA", "Core CPI": "#228833", "PPI": "#CC6677"}
    fig, ax = plt.subplots(figsize=(7.4, 4.3))
    for i, row in plot.iterrows():
        ax.errorbar(
            row["delta_mean"], y[i],
            xerr=[[row["delta_mean"] - row["delta_lo"]], [row["delta_hi"] - row["delta_mean"]]],
            fmt="o", color=colors[row["inflation"]], capsize=3,
        )
    ax.axvline(0, color="black", lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{r.inflation} / {r.activity}" for r in plot.itertuples()])
    ax.set_xlabel(r"Competition dependence $\delta$ (posterior mean and 95% interval)")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT_FIGURES / "delta_output_gaps.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def _save_forest(
    table: pd.DataFrame,
    *,
    filename: str,
    xlabel: str,
    figsize: tuple[float, float],
    show_status_legend: bool = True,
) -> None:
    if table.empty:
        return
    OUT_FIGURES.mkdir(parents=True, exist_ok=True)
    plot = table.dropna(subset=["mean", "lo", "hi"]).reset_index(drop=True)
    y = np.arange(len(plot))[::-1]
    colors = {"Headline CPI": "#4477AA", "Core CPI": "#228833", "PPI": "#CC6677"}
    fig, ax = plt.subplots(figsize=figsize)
    for i, row in plot.iterrows():
        converged = bool(row["converged"])
        color = colors[row["inflation"]] if converged else "#888888"
        ax.errorbar(
            row["mean"],
            y[i],
            xerr=[[row["mean"] - row["lo"]], [row["hi"] - row["mean"]]],
            fmt="o" if converged else "x",
            color=color,
            ecolor=color,
            alpha=1.0 if converged else 0.65,
            capsize=3,
        )
    ax.axvline(0, color="black", lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(plot["label"])
    ax.set_xlabel(xlabel)
    ax.grid(axis="x", alpha=0.25)
    if show_status_legend and (~plot["converged"]).any():
        ax.legend(
            handles=[
                Line2D([0], [0], marker="o", color="black", lw=0, label="Convergence criteria met"),
                Line2D([0], [0], marker="x", color="#888888", lw=0, label="Diagnostic warning"),
            ],
            loc="best",
            frameon=False,
            fontsize=8,
        )
    fig.tight_layout()
    fig.savefig(OUT_FIGURES / filename, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _parameter_row(idata, *, parameter: str, label: str, inflation: str) -> dict[str, object] | None:
    summary = _summary(idata, parameter)
    if summary is None:
        return None
    diagnostics = _diagnostics(idata)
    return {
        "label": label,
        "inflation": inflation,
        "parameter": parameter,
        "mean": summary["mean"],
        "lo": summary["lo"],
        "hi": summary["hi"],
        "converged": bool(diagnostics["converged"]),
        "max_rhat": diagnostics["max_rhat"],
        "min_bulk_ess": diagnostics["min_ess"],
    }


def save_additional_parameter_forests(runs) -> None:
    steady_rows: list[dict[str, object]] = []
    for inflation, activity_specs in INFLATION_SPECS.items():
        for activity, data_spec in activity_specs.items():
            item = runs.get(("hsa_steady", data_spec, "baseline"))
            if item is None:
                continue
            row = _parameter_row(
                item[1], parameter="kappa_0", label=f"{inflation} / {activity}", inflation=inflation
            )
            if row is not None:
                steady_rows.append(row)
    steady_table = pd.DataFrame(steady_rows)
    steady_table.to_csv(OUT_TABLES / "figure_kappa0_by_inflation_activity.csv", index=False)
    _save_forest(
        steady_table,
        filename="kappa0_by_inflation_activity.png",
        xlabel=r"Baseline slope at mean competition $\kappa_0$",
        figsize=(7.4, 5.4),
        show_status_legend=False,
    )

    slope_rows: list[dict[str, object]] = []
    alpha_rows: list[dict[str, object]] = []
    entry_rows: list[dict[str, object]] = []
    for model in MODEL_ORDER:
        for inflation, data_spec in PRIMARY_SPECS.items():
            item = runs.get((model, data_spec, "baseline"))
            if item is None:
                continue
            idata = item[1]
            label = f"{MODEL_LABELS[model]} / {inflation}"
            slope_parameter = "kappa" if model in {"ces", "hsa_dynamic"} else "kappa_0"
            slope_row = _parameter_row(
                idata, parameter=slope_parameter, label=label, inflation=inflation
            )
            alpha_row = _parameter_row(idata, parameter="alpha", label=label, inflation=inflation)
            if slope_row is not None:
                slope_rows.append(slope_row)
            if alpha_row is not None:
                alpha_rows.append(alpha_row)
            if model in {"hsa_dynamic", "hsa_const_theta", "hsa_full"}:
                entry_parameter = "theta" if model in {"hsa_dynamic", "hsa_const_theta"} else "theta_0"
                entry_row = _parameter_row(
                    idata, parameter=entry_parameter, label=label, inflation=inflation
                )
                if entry_row is not None:
                    entry_rows.append(entry_row)

    slope_table = pd.DataFrame(slope_rows)
    alpha_table = pd.DataFrame(alpha_rows)
    entry_table = pd.DataFrame(entry_rows)
    slope_table.to_csv(OUT_TABLES / "figure_slope_by_model_inflation.csv", index=False)
    alpha_table.to_csv(OUT_TABLES / "figure_alpha_by_model_inflation.csv", index=False)
    entry_table.to_csv(OUT_TABLES / "figure_entry_by_model_inflation.csv", index=False)
    _save_forest(
        slope_table,
        filename="slope_by_model_inflation.png",
        xlabel=r"Slope at mean competition $\kappa$ or $\kappa_0$",
        figsize=(7.8, 7.0),
    )

    if not alpha_table.empty and not entry_table.empty:
        fig, axes = plt.subplots(1, 2, figsize=(12.0, 7.0))
        panels = [
            (axes[0], alpha_table, r"Inflation persistence $\alpha$"),
            (axes[1], entry_table, r"Entry effect $\theta$ or $\theta_0$"),
        ]
        colors = {"Headline CPI": "#4477AA", "Core CPI": "#228833", "PPI": "#CC6677"}
        for ax, table, xlabel in panels:
            plot = table.dropna(subset=["mean", "lo", "hi"]).reset_index(drop=True)
            y = np.arange(len(plot))[::-1]
            for i, row in plot.iterrows():
                converged = bool(row["converged"])
                color = colors[row["inflation"]] if converged else "#888888"
                ax.errorbar(
                    row["mean"], y[i],
                    xerr=[[row["mean"] - row["lo"]], [row["hi"] - row["mean"]]],
                    fmt="o" if converged else "x", color=color, ecolor=color,
                    alpha=1.0 if converged else 0.65, capsize=3,
                )
            ax.axvline(0, color="black", lw=0.8)
            ax.set_yticks(y)
            ax.set_yticklabels(plot["label"], fontsize=8)
            ax.set_xlabel(xlabel)
            ax.grid(axis="x", alpha=0.25)
        axes[1].legend(
            handles=[
                Line2D([0], [0], marker="o", color="black", lw=0, label="Convergence criteria met"),
                Line2D([0], [0], marker="x", color="#888888", lw=0, label="Diagnostic warning"),
            ],
            loc="best", frameon=False, fontsize=8,
        )
        fig.tight_layout()
        fig.savefig(OUT_FIGURES / "alpha_entry_by_model_inflation.png", dpi=220, bbox_inches="tight")
        plt.close(fig)


def save_kappa_path_comparison(runs) -> None:
    colors = {"Headline CPI": "#4477AA", "Core CPI": "#228833", "PPI": "#CC6677"}
    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    plotted = False
    for inflation, data_spec in PRIMARY_SPECS.items():
        item = runs.get(("hsa_steady", data_spec, "baseline"))
        if item is None or "kappa_t" not in item[1].posterior:
            continue
        idata = item[1]
        values = np.asarray(idata.posterior["kappa_t"], dtype=float)
        paths = values.reshape(-1, values.shape[-1])
        mean = np.nanmean(paths, axis=0)
        lo = np.nanquantile(paths, 0.025, axis=0)
        hi = np.nanquantile(paths, 0.975, axis=0)
        attrs = getattr(idata, "attrs", {})
        start = pd.Timestamp(str(attrs.get("sample_start", "1982-03-31"))).to_period("Q")
        dates = pd.period_range(start, periods=len(mean), freq="Q").to_timestamp(how="end")
        color = colors[inflation]
        ax.plot(dates, mean, color=color, lw=2.0, label=inflation)
        ax.fill_between(dates, lo, hi, color=color, alpha=0.13)
        plotted = True
    if not plotted:
        plt.close(fig)
        return
    ax.axhline(0.0, color="black", lw=0.8)
    ax.set_ylabel(r"Implied NKPC slope $\kappa_t$")
    ax.set_xlabel("Quarter")
    ax.grid(alpha=0.22)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT_FIGURES / "kappa_t_unemployment_by_inflation.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_main_competition_decomposition(runs) -> None:
    """Save a stable main-model state decomposition instead of a shared overwritten figure."""
    item = runs.get(("hsa_steady", PRIMARY_SPECS["Core CPI"], "baseline"))
    if item is None:
        return
    idata = item[1]
    if "Nbar" not in idata.posterior or "Nhat" not in idata.posterior:
        return
    nbar_values = np.asarray(idata.posterior["Nbar"], dtype=float)
    nhat_values = np.asarray(idata.posterior["Nhat"], dtype=float)
    nbar = nbar_values.reshape(-1, nbar_values.shape[-1])
    nhat = nhat_values.reshape(-1, nhat_values.shape[-1])
    total = nbar + nhat
    attrs = getattr(idata, "attrs", {})
    start = pd.Timestamp(str(attrs.get("sample_start", "1982-03-31"))).to_period("Q")
    dates = pd.period_range(start, periods=nbar.shape[-1], freq="Q").to_timestamp(how="end")

    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    series = [
        (total, "Latent total $N_t$", "#4477AA"),
        (nbar, r"Trend $\bar N_t$", "#228833"),
        (nhat, r"Cycle $\hat N_t$", "#EE7733"),
    ]
    for paths, label, color in series:
        mean = np.nanmean(paths, axis=0)
        lo = np.nanquantile(paths, 0.025, axis=0)
        hi = np.nanquantile(paths, 0.975, axis=0)
        ax.plot(dates, mean, color=color, lw=1.8, label=label)
        ax.fill_between(dates, lo, hi, color=color, alpha=0.12)
    ax.axhline(0.0, color="black", lw=0.8)
    ax.set_title("Competition-state decomposition: HSA steady / Core CPI / unemployment gap")
    ax.set_ylabel("Ten-log-point units")
    ax.set_xlabel("Quarter")
    ax.grid(alpha=0.22)
    ax.legend(frameon=False, ncol=3, loc="upper right")
    fig.tight_layout()
    fig.savefig(
        OUT_FIGURES / "competition_decomposition_hsa_steady_core_unemployment.png",
        dpi=220,
        bbox_inches="tight",
    )
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", type=Path, default=ROOT / "results" / "runs")
    parser.add_argument("--min-iter", type=int, default=1000)
    parser.add_argument("--allow-incomplete", action="store_true", help="Generate partial tables instead of failing on missing report cells.")
    parser.add_argument("--compile", action="store_true", help="Compile the Japanese CPI/PPI report with XeLaTeX.")
    args = parser.parse_args()

    quarterly_runs = _load_runs(
        args.runs_dir,
        min_iter=args.min_iter,
        competition_frequency="quarterly_interpolated",
    )
    annual_hsa_runs = _load_runs(
        args.runs_dir,
        min_iter=args.min_iter,
        competition_frequency="annual_q4",
    )
    missing_quarterly = sorted(set(report_run_keys()) - set(quarterly_runs))
    missing_annual = sorted(set(annual_q4_run_keys()) - set(annual_hsa_runs))
    missing = [("PCHIP", *key) for key in missing_quarterly] + [("annual-Q4", *key) for key in missing_annual]
    if missing and not args.allow_incomplete:
        preview = ", ".join("/".join(key) for key in missing[:8])
        raise SystemExit(f"Missing {len(missing)} required current-revision runs: {preview}")

    quarterly_required = set(report_run_keys())
    quarterly_display = {key: value for key, value in quarterly_runs.items() if key in quarterly_required}
    annual_required = set(annual_q4_run_keys())
    annual_display = {
        key: value
        for key, value in quarterly_display.items()
        if key[0] == "ces"
    }
    annual_display.update({key: value for key, value in annual_hsa_runs.items() if key in annual_required})

    base_tables = ROOT / "results" / "tables" / "cpi_ppi_report"
    base_figures = ROOT / "results" / "figures" / "cpi_ppi_report"
    build_frequency_outputs(
        quarterly_display,
        tables_dir=base_tables,
        figures_dir=base_figures,
        command_prefix="",
    )
    build_frequency_outputs(
        annual_display,
        tables_dir=base_tables / "annual_q4",
        figures_dir=base_figures / "annual_q4",
        command_prefix="Annual",
    )
    subprocess.run(["python", str(ROOT / "scripts" / "11_additional_report_evidence.py")], cwd=ROOT, check=True)
    print(f"Saved PCHIP and annual-Q4 report inputs under {base_tables} and {base_figures}")
    if args.compile:
        tex = ROOT / "paper" / "nkpc_hsa_report_ja.tex"
        build_dir = ROOT / "results" / "report" / "build" / "nkpc_hsa_report_ja"
        build_dir.mkdir(parents=True, exist_ok=True)
        command = [
            "xelatex", "-interaction=nonstopmode", "-halt-on-error",
            f"-output-directory={build_dir}", str(tex),
        ]
        subprocess.run(command, cwd=ROOT, check=True)
        subprocess.run(command, cwd=ROOT, check=True)
        built_pdf = build_dir / f"{tex.stem}.pdf"
        final_pdf = ROOT / "results" / "report" / f"{tex.stem}.pdf"
        shutil.copy2(built_pdf, final_pdf)
        print(f"Saved {final_pdf}")


if __name__ == "__main__":
    main()
