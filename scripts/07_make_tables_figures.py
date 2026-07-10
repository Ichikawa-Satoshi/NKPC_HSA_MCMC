from __future__ import annotations

from pathlib import Path

import argparse
import arviz as az
import json
import numpy as np
import pandas as pd
import yaml

from _bootstrap import ROOT
from nkpc_hsa.config import configured_data_specs, load_model_config
from nkpc_hsa.data.transforms import DEFAULT_N_TRANSFORM, transform_competition_series
from nkpc_hsa.inference.period_robustness import apply_period, load_periods
from nkpc_hsa.report.figures import (
    save_coefficient_interval_plot,
    save_kappa_model_comparison,
    save_placeholder_figure,
    save_posterior_density,
    save_posterior_predictive_placeholder,
    save_prior_posterior_overlay,
    save_prior_posterior_per_model,
    save_time_varying_path,
)
from nkpc_hsa.report.estimation_results import competition_decomposition_summary
from nkpc_hsa.inference.model_comparison import model_comparison_table, save_model_comparison
from nkpc_hsa.report.tables import (
    coefficient_means_pivot_table,
    coefficient_means_table,
    kappa_comparison_table,
    posterior_summary_table,
    sddr_summary_table,
    time_varying_coefficients_table,
    write_latex_fragment,
)


KEY_COEFFICIENTS = ["alpha", "kappa", "kappa_0", "delta", "theta", "theta_0", "gamma", "rho_1", "rho_2"]
PRIOR_POSTERIOR_PARAMETERS = ["kappa", "kappa_0", "delta", "theta", "theta_0", "gamma"]


def _latex_escape(value: object) -> str:
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "_": r"\_",
        "%": r"\%",
        "&": r"\&",
        "#": r"\#",
        "$": r"\$",
        "{": r"\{",
        "}": r"\}",
    }
    for src, dst in replacements.items():
        text = text.replace(src, dst)
    return text


def _safe_id(*parts: object) -> str:
    raw = "__".join(str(part) for part in parts if str(part))
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in raw)


def _constraint_display(constraint: str) -> str:
    labels = {
        "unrestricted": "unrestricted",
        "restricted_kappa": "kappa >= 0",
    }
    return labels.get(constraint, constraint.replace("restricted_", "").replace("_", " ") + " restricted")


def _data_spec_display(data_spec: str | None, labels: dict[str, str] | None) -> str:
    if data_spec is None:
        return "All configured specifications"
    return (labels or {}).get(data_spec, data_spec.replace("_", " "))


def _frequency_display(frequency: str) -> str:
    labels = {
        "quarterly_interpolated": "quarterly interpolated",
        "annual_q4": "annual Q4",
    }
    return labels.get(frequency, frequency.replace("_", " "))


def _display(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    if df.empty:
        return df
    keep = [col for col in columns if col in df.columns]
    return df[keep].copy() if keep else df


def _combined_columns(data_spec: str | None, columns: list[str]) -> list[str]:
    out = list(columns)
    if "competition_measurement_frequency" not in out:
        insert_at = 0
        if out and out[0] == "data_spec":
            insert_at = 1
        out.insert(insert_at, "competition_measurement_frequency")
    if data_spec is None and "data_spec" not in columns:
        return ["data_spec", *out]
    return out


def _load_prior(path):
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def _load_idata_by_run(runs_dir: str | Path) -> dict[str, object]:
    idata_by_run = {}
    for posterior in sorted(Path(runs_dir).glob("*/posterior.nc")):
        idata = az.from_netcdf(posterior)
        priors_path = posterior.parent / "priors.json"
        if priors_path.exists():
            idata.attrs["run_priors"] = json.loads(priors_path.read_text(encoding="utf-8"))
        idata_by_run[posterior.parent.name] = idata
    return idata_by_run


def _base_period_from_data_spec(raw_data_spec: str, base_names: list[str]) -> tuple[str, str]:
    for base in sorted(base_names, key=len, reverse=True):
        if raw_data_spec == base:
            return base, "full"
        prefix = f"{base}_"
        if raw_data_spec.startswith(prefix):
            return base, raw_data_spec[len(prefix) :]
    return raw_data_spec, "full"


def _run_base_period(idata, base_names: list[str]) -> tuple[str, str]:
    attrs = getattr(idata, "attrs", {})
    raw_data_spec = str(attrs.get("data_spec", ""))
    base, parsed_period = _base_period_from_data_spec(raw_data_spec, base_names)
    period = str(attrs.get("period", "") or parsed_period or "full")
    return base, period


def _filter_by_base_data_spec(idata_by_run: dict[str, object], data_spec: str | None, base_names: list[str]) -> dict[str, object]:
    if data_spec is None:
        return {
            run: idata
            for run, idata in idata_by_run.items()
            if _run_base_period(idata, base_names)[0] in base_names
        }
    return {
        run: idata
        for run, idata in idata_by_run.items()
        if _run_base_period(idata, base_names)[0] == data_spec
    }


def _filter_by_data_spec(idata_by_run: dict[str, object], data_spec: str | None) -> dict[str, object]:
    if data_spec is None:
        return idata_by_run
    return {
        run: idata
        for run, idata in idata_by_run.items()
        if str(getattr(idata, "attrs", {}).get("data_spec", "")) == data_spec
    }


def _filter_by_transform(idata_by_run: dict[str, object], n_transform: str | None) -> dict[str, object]:
    if not n_transform:
        return idata_by_run
    return {
        run: idata
        for run, idata in idata_by_run.items()
        if str(getattr(idata, "attrs", {}).get("n_transform", "")) == n_transform
    }


def _filter_by_competition_frequency(idata_by_run: dict[str, object], frequency: str | None) -> dict[str, object]:
    if not frequency:
        return idata_by_run
    return {
        run: idata
        for run, idata in idata_by_run.items()
        if str(getattr(idata, "attrs", {}).get("competition_measurement_frequency", "quarterly_interpolated")) == frequency
    }


def _latest_key(idata) -> str:
    attrs = getattr(idata, "attrs", {})
    return str(attrs.get("run_id") or "")


def _latest_by_fields(idata_by_run: dict[str, object], fields: tuple[str, ...]) -> dict[str, object]:
    selected: dict[tuple[str, ...], tuple[str, object]] = {}
    for run, idata in idata_by_run.items():
        attrs = getattr(idata, "attrs", {})
        key = tuple(str(attrs.get(field, "")) for field in fields)
        current = selected.get(key)
        if current is None:
            selected[key] = (run, idata)
            continue
        current_run, current_idata = current
        if (_latest_key(idata), run) >= (_latest_key(current_idata), current_run):
            selected[key] = (run, idata)
    return {run: idata for run, idata in sorted(selected.values())}


def _filter_by_prior(idata_by_run: dict[str, object], prior_spec: str | None) -> dict[str, object]:
    if prior_spec is None:
        return idata_by_run
    return {
        run: idata
        for run, idata in idata_by_run.items()
        if str(getattr(idata, "attrs", {}).get("prior_spec", "")) == prior_spec
    }


def _filter_by_constraint(idata_by_run: dict[str, object], constraint_spec: str) -> dict[str, object]:
    return {
        run: idata
        for run, idata in idata_by_run.items()
        if str(getattr(idata, "attrs", {}).get("constraint_spec", "unrestricted") or "unrestricted") == constraint_spec
    }


def _key_coefficients(table: pd.DataFrame) -> pd.DataFrame:
    if table.empty or "parameter" not in table.columns:
        return table
    return table[table["parameter"].astype(str).isin(KEY_COEFFICIENTS)].copy()


def _prior_for_run(idata, fallback_priors: dict, var: str) -> tuple[float, float] | None:
    attrs = getattr(idata, "attrs", {})
    priors = attrs.get("run_priors") if isinstance(attrs.get("run_priors"), dict) else fallback_priors
    if var not in priors:
        return None
    prior = priors[var]
    if isinstance(prior, dict):
        return float(prior["mean"]), float(prior["sd"])
    return float(prior[0]), float(prior[1])


def _overlay_options(var: str) -> dict:
    labels = {
        "delta": "change in kappa_t per +10 log-point Nbar deviation",
        "theta": "inflation effect per +10 log-point Nhat deviation",
        "theta_0": "inflation effect per +10 log-point Nhat deviation at average Nbar",
        "gamma": "change in theta_t per +10 log-point Nbar deviation",
    }
    if var not in labels:
        return {}
    return {"xlabel": labels[var], "zoom_to_posterior": True}


def _write_existing_table(
    csv_path: Path,
    tex_path: Path,
    *,
    data_spec: str | None,
    n_transform: str | None,
    note: str,
    display_columns: list[str] | None = None,
) -> None:
    if not csv_path.exists():
        write_latex_fragment(pd.DataFrame({"note": [note]}), tex_path)
        return
    table = pd.read_csv(csv_path)
    if data_spec is not None and "data_spec" in table.columns:
        table = table[table["data_spec"].astype(str) == data_spec]
    if n_transform is not None:
        if "n_transform" in table.columns:
            table = table[table["n_transform"].astype(str) == n_transform]
        else:
            table = pd.DataFrame()
    if table.empty:
        table = pd.DataFrame({"note": [note]})
    write_latex_fragment(_display(table, display_columns or list(table.columns)), tex_path)


def _write_existing_table_from_candidates(
    csv_paths: list[Path],
    tex_path: Path,
    *,
    data_spec: str | None,
    n_transform: str | None,
    note: str,
    display_columns: list[str] | None = None,
) -> pd.DataFrame:
    for csv_path in csv_paths:
        if csv_path.exists():
            table = pd.read_csv(csv_path)
            if data_spec is not None and "data_spec" in table.columns:
                table = table[table["data_spec"].astype(str) == data_spec]
            if n_transform is not None:
                if "n_transform" in table.columns:
                    table = table[table["n_transform"].astype(str) == n_transform]
                else:
                    table = pd.DataFrame()
            if table.empty:
                table = pd.DataFrame({"note": [note]})
            write_latex_fragment(_display(table, display_columns or list(table.columns)), tex_path)
            return table
    table = pd.DataFrame({"note": [note]})
    write_latex_fragment(table, tex_path)
    return table


def _tex_result_path(path: Path) -> str:
    return "../" + path.resolve().relative_to((ROOT / "results").resolve()).as_posix()


def _read_table_from_candidates(csv_paths: list[Path]) -> pd.DataFrame:
    for csv_path in csv_paths:
        if csv_path.exists():
            return pd.read_csv(csv_path)
    return pd.DataFrame()


def _comparison_data_from_frame(df: pd.DataFrame, spec: dict, n_transform: str) -> dict[str, object] | None:
    cols = {
        "pi": spec.get("pi_col", "pi"),
        "pi_prev": spec.get("pi_prev_col", "pi_prev"),
        "pi_expect": spec.get("pi_expect_col", "pi_expect"),
        "x": spec.get("x_col", "x"),
        "x_prev": spec.get("x_prev_col", "x_prev"),
        "N": spec.get("n_col", spec.get("N_col", "N")),
    }
    required = list(cols.values())
    if any(col not in df.columns for col in required):
        return None
    sample = df[required].dropna()
    if sample.empty:
        return None
    return {
        "pi": sample[cols["pi"]].to_numpy(dtype=float),
        "pi_prev": sample[cols["pi_prev"]].to_numpy(dtype=float),
        "pi_expect": sample[cols["pi_expect"]].to_numpy(dtype=float),
        "x": sample[cols["x"]].to_numpy(dtype=float),
        "x_prev": sample[cols["x_prev"]].to_numpy(dtype=float),
        "N": transform_competition_series(sample[cols["N"]].to_numpy(dtype=float), transform=n_transform),
    }


def _comparison_data_by_condition(
    data: pd.DataFrame | None,
    data_specs: dict[str, dict],
    periods: dict[str, dict],
    n_transform: str,
) -> dict[tuple[str, str], dict[str, object]]:
    if data is None:
        return {}
    out: dict[tuple[str, str], dict[str, object]] = {}
    for data_spec_name, spec in data_specs.items():
        full_data = _comparison_data_from_frame(data, spec, n_transform)
        if full_data is not None:
            out[(data_spec_name, "full")] = full_data
        for period_name, period_spec in periods.items():
            period_data = _comparison_data_from_frame(apply_period(data, period_spec), spec, n_transform)
            if period_data is not None:
                out[(data_spec_name, period_name)] = period_data
    return out


def _filter_period_table(
    table: pd.DataFrame,
    *,
    data_spec: str | None,
    n_transform: str | None,
) -> pd.DataFrame:
    if table.empty:
        return table
    out = table.copy()
    if data_spec is not None and "data_spec" in out.columns:
        out = out[out["data_spec"].astype(str) == data_spec]
    if n_transform is not None:
        if "n_transform" in out.columns:
            out = out[out["n_transform"].astype(str) == n_transform]
        else:
            out = pd.DataFrame()
    return out


def _block_sort_key(item: tuple[tuple[str, str, str, str, str], dict[str, object]]) -> tuple:
    base, prior, period, constraint, frequency = item[0]
    prior_order = {"baseline": 0, "weak": 1, "tight": 2}
    period_order = {"full": 0, "pre_2008": 1, "post_2008": 2, "exclude_covid": 3, "start_1988": 4, "end_2019": 5}
    constraint_order = {"unrestricted": 0}
    frequency_order = {"quarterly_interpolated": 0, "annual_q4": 1}
    return (
        base,
        prior_order.get(prior, 99),
        prior,
        period_order.get(period, 99),
        period,
        constraint_order.get(constraint, 1),
        constraint,
        frequency_order.get(frequency, 99),
        frequency,
    )


def _write_block_table(df: pd.DataFrame, csv_path: Path, tex_path: Path, display_columns: list[str]) -> pd.DataFrame:
    out = df if not df.empty else pd.DataFrame({"note": ["No output available for this block."]})
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(csv_path, index=False)
    write_latex_fragment(_display(out, display_columns), tex_path)
    return out


def _quarterly_index_from_attrs(idata) -> pd.PeriodIndex | None:
    attrs = getattr(idata, "attrs", {})
    start = str(attrs.get("sample_start", "") or "")
    n_obs_raw = attrs.get("n_obs", "")
    try:
        n_obs = int(n_obs_raw)
    except (TypeError, ValueError):
        return None
    if not start or n_obs <= 0:
        return None
    try:
        return pd.period_range(pd.Timestamp(start).to_period("Q"), periods=n_obs, freq="Q")
    except Exception:
        try:
            return pd.period_range(start, periods=n_obs, freq="Q")
        except Exception:
            return None


def _write_competition_decomposition_outputs(
    *,
    runs: dict[str, object],
    tables_dir: Path,
    figures_dir: Path,
    data_spec: str | None,
) -> None:
    candidates: dict[str, object] = {}
    for run, idata in runs.items():
        attrs = getattr(idata, "attrs", {})
        model = str(attrs.get("model", ""))
        if not model.startswith("hsa_"):
            continue
        frequency = str(attrs.get("competition_measurement_frequency", "quarterly_interpolated"))
        if frequency != "annual_q4":
            continue
        if data_spec is not None and str(attrs.get("data_spec", "")) != data_spec:
            continue
        candidates[run] = idata

    if not candidates:
        out = pd.DataFrame({"note": ["No annual_q4 HSA decomposition output available."]})
        out.to_csv(tables_dir / "competition_decomposition.csv", index=False)
        write_latex_fragment(out, tables_dir / "competition_decomposition.tex")
        save_placeholder_figure(figures_dir / "competition_decomposition_path.png", "No annual_q4 HSA decomposition output available.")
        return

    latest = _latest_by_fields(candidates, ("model", "data_spec", "prior_spec", "period", "constraint_spec"))
    selected_run, selected_idata = max(
        latest.items(),
        key=lambda item: (str(getattr(item[1], "attrs", {}).get("run_id", "")), item[0]),
    )
    attrs = getattr(selected_idata, "attrs", {})
    q_index = _quarterly_index_from_attrs(selected_idata)
    decomp = competition_decomposition_summary(selected_idata, q_index)
    if decomp.empty:
        out = pd.DataFrame({"note": ["No Nbar/Nhat draws available for decomposition."]})
        out.to_csv(tables_dir / "competition_decomposition.csv", index=False)
        write_latex_fragment(out, tables_dir / "competition_decomposition.tex")
        return

    decomp.insert(0, "run", selected_run)
    decomp.insert(1, "model", str(attrs.get("model", "")))
    decomp.insert(2, "data_spec", str(attrs.get("data_spec", "")))
    decomp.to_csv(tables_dir / "competition_decomposition.csv", index=False)

    display = decomp[
        [
            "quarter",
            "model",
            "N_total_mean",
            "N_total_q05",
            "N_total_q50",
            "N_total_q95",
            "Nbar_mean",
            "Nhat_mean",
        ]
    ].copy()
    if len(display) > 16:
        idx = np.linspace(0, len(display) - 1, 16).round().astype(int)
        display = display.iloc[idx].reset_index(drop=True)
    write_latex_fragment(display, tables_dir / "competition_decomposition.tex")

    import matplotlib.pyplot as plt

    x = pd.PeriodIndex(decomp["quarter"], freq="Q").to_timestamp(how="end")
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.plot(x, decomp["N_total_mean"].to_numpy(dtype=float), label=r"$N_t=\bar N_t+\hat N_t$", color="C0")
    ax.plot(x, decomp["Nbar_mean"].to_numpy(dtype=float), label=r"$\bar N_t$", color="C2")
    ax.plot(x, decomp["Nhat_mean"].to_numpy(dtype=float), label=r"$\hat N_t$", color="C1")
    ax.fill_between(
        x,
        decomp["N_total_q05"].to_numpy(dtype=float),
        decomp["N_total_q95"].to_numpy(dtype=float),
        color="C0",
        alpha=0.14,
    )
    ax.axhline(0.0, color="black", lw=0.8, alpha=0.5)
    ax.set_title(f"Competition decomposition: {attrs.get('model', '')} / {attrs.get('data_spec', '')}")
    ax.set_xlabel("quarter")
    ax.set_ylabel("N units")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    figures_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figures_dir / "competition_decomposition_path.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def _write_result_blocks(
    *,
    latest_runs: dict[str, object],
    priors: dict,
    tables_dir: Path,
    figures_dir: Path,
    data_spec: str | None,
    base_names: list[str],
    data_spec_labels: dict[str, str] | None,
    period_table: pd.DataFrame,
    comparison_data: dict[tuple[str, str], dict[str, object]] | None = None,
    all_baseline_runs: dict[str, object] | None = None,
) -> None:
    # all_baseline_runs: latest run per (model, base_data_spec, constraint_spec) across ALL n_transforms.
    # Used for per-model prior/posterior so all 4 models appear even if n_transform differs.
    blocks: dict[tuple[str, str, str, str, str], dict[str, object]] = {}
    for run, idata in latest_runs.items():
        base, period = _run_base_period(idata, base_names)
        if data_spec is not None and base != data_spec:
            continue
        attrs = getattr(idata, "attrs", {})
        prior = str(attrs.get("prior_spec", "baseline") or "baseline")
        constraint = str(attrs.get("constraint_spec", "unrestricted") or "unrestricted")
        frequency = str(attrs.get("competition_measurement_frequency", "quarterly_interpolated") or "quarterly_interpolated")
        blocks.setdefault((base, prior, period, constraint, frequency), {})[run] = idata

    if not period_table.empty:
        period_rows = period_table.copy()
        if data_spec is not None and "data_spec" in period_rows.columns:
            period_rows = period_rows[period_rows["data_spec"].astype(str) == data_spec]
        for _, row in period_rows.iterrows():
            base = str(row.get("data_spec", data_spec or ""))
            period = str(row.get("period", "full") or "full")
            constraint = str(row.get("constraint_spec", "unrestricted") or "unrestricted")
            if base:
                known_frequencies = {
                    str(getattr(idata, "attrs", {}).get("competition_measurement_frequency", "quarterly_interpolated") or "quarterly_interpolated")
                    for idata in latest_runs.values()
                    if _run_base_period(idata, base_names) == (base, period)
                } or {"quarterly_interpolated"}
                for frequency in sorted(known_frequencies):
                    blocks.setdefault((base, "baseline", period, constraint, frequency), {})

    lines: list[str] = []
    if not blocks:
        write_latex_fragment(pd.DataFrame({"note": ["No result blocks available."]}), tables_dir / "result_blocks.tex")
        return

    for (base, prior, period, constraint, frequency), block_runs in sorted(blocks.items(), key=_block_sort_key):
        block_id = _safe_id(base, prior, period, constraint, frequency)
        block_table_dir = tables_dir / "blocks" / block_id
        block_figure_dir = figures_dir / "blocks" / block_id
        block_table_dir.mkdir(parents=True, exist_ok=True)
        block_figure_dir.mkdir(parents=True, exist_ok=True)
        data_label = _data_spec_display(base, data_spec_labels)
        constraint_label = _constraint_display(constraint)
        figure_subtitle = (
            f"competition={_frequency_display(frequency)}; prior={prior}; "
            f"period={period}; constraint={constraint_label}"
        )

        # Cross-transform runs for this block: all models matching this constraint, regardless of n_transform.
        # Used for coefficient pivot and prior/posterior figures.
        if all_baseline_runs is not None:
            pp_runs = {
                run: idata for run, idata in all_baseline_runs.items()
                if _run_base_period(idata, base_names) == (base, period)
                and str(getattr(idata, "attrs", {}).get("constraint_spec", "unrestricted") or "unrestricted") == constraint
                and str(getattr(idata, "attrs", {}).get("competition_measurement_frequency", "quarterly_interpolated") or "quarterly_interpolated") == frequency
            } or block_runs
        else:
            pp_runs = block_runs

        # Coefficient table: pivot format (parameters × models).
        pivot = coefficient_means_pivot_table(pp_runs)
        if pivot.empty and not period_table.empty:
            # Fall back: build long-format from period_table and re-pivot.
            coeff = period_table.copy()
            for col, val in [("data_spec", base), ("period", period)]:
                if col in coeff.columns:
                    coeff = coeff[coeff[col].astype(str) == val]
            if "parameter" in coeff.columns:
                coeff = coeff[coeff["parameter"].astype(str).isin(KEY_COEFFICIENTS)]
            coeff = coeff.rename(columns={"mean": "posterior_mean"})
            pivot = coefficient_means_pivot_table({"_fallback": None})  # stays empty
            # Save long format as fallback
            coeff_out = coeff if not coeff.empty else pd.DataFrame({"note": ["No coefficient output for this block."]})
            coeff_out.to_csv(block_table_dir / "coefficients.csv", index=False)
            write_latex_fragment(
                _display(coeff_out, ["model", "parameter", "posterior_mean", "ci_2.5", "ci_97.5", "p_gt_0"]),
                block_table_dir / "coefficients.tex",
            )
        else:
            long_coeff = _key_coefficients(coefficient_means_table(pp_runs))
            long_coeff.to_csv(block_table_dir / "coefficients.csv", index=False)
            pivot_out = pivot if not pivot.empty else pd.DataFrame({"note": ["No coefficient output for this block."]})
            write_latex_fragment(pivot_out, block_table_dir / "coefficients.tex", index=not pivot.empty)

        sddr = sddr_summary_table(block_runs, priors)
        _write_block_table(
            sddr,
            block_table_dir / "sddr.csv",
            block_table_dir / "sddr.tex",
            ["model", "restriction", "prior_mean", "prior_sd", "sddr_bf01"],
        )

        block_data = (comparison_data or {}).get((base, period))
        comparison = model_comparison_table(
            block_runs,
            data_by_model={run: block_data for run in block_runs} if block_data is not None else None,
        )
        _write_block_table(
            comparison,
            block_table_dir / "model_comparison.csv",
            block_table_dir / "model_comparison.tex",
            ["model", "predictive_score", "posterior_predictive_rmse", "log_marginal_likelihood", "bayes_factor_vs_baseline", "sddr_delta_bf01", "sddr_theta_bf01", "sddr_gamma_bf01"],
        )
        prior_opts = {v: _overlay_options(v) for v in PRIOR_POSTERIOR_PARAMETERS}
        per_model_figs = save_prior_posterior_per_model(
            pp_runs,
            PRIOR_POSTERIOR_PARAMETERS,
            lambda idata, var: _prior_for_run(idata, priors, var),
            block_figure_dir,
            overlay_options=prior_opts,
        )

        kappa_path = block_figure_dir / "kappa_t_path.png"
        if not save_time_varying_path(
            block_runs,
            "kappa_t",
            kappa_path,
            title=f"{data_label}: time-varying kappa_t path",
            subtitle=figure_subtitle,
        ):
            save_placeholder_figure(kappa_path, "No kappa_t path for this block.")
        theta_path = block_figure_dir / "theta_t_path.png"
        if not save_time_varying_path(
            block_runs,
            "theta_t",
            theta_path,
            title=f"{data_label}: time-varying theta_t path",
            subtitle=figure_subtitle,
        ):
            save_placeholder_figure(theta_path, "No theta_t path for this block.")

        constraint_suffix = f", constraint={constraint_label}" if constraint != "unrestricted" else ""
        title = f"{base}: {frequency}, prior={prior}, period={period}{constraint_suffix}"
        lines.extend(
            [
                r"\clearpage",
                rf"\section{{{_latex_escape(title)}}}",
                rf"\noindent\textbf{{Condition.}} data=\texttt{{{_latex_escape(base)}}}, competition=\texttt{{{_latex_escape(frequency)}}}, prior=\texttt{{{_latex_escape(prior)}}}, period=\texttt{{{_latex_escape(period)}}}, constraint=\texttt{{{_latex_escape(constraint)}}} ({_latex_escape(constraint_label)}).",
                "",
                r"\subsection{Coefficient Table}",
                r"\begin{center}\scriptsize",
                rf"\resizebox{{\textwidth}}{{!}}{{\input{{{_tex_result_path(block_table_dir / 'coefficients.tex')}}}}}",
                r"\end{center}",
                "",
                r"\subsection{Prior vs Posterior by Model}",
            ]
        )
        for model_name, fig_path in per_model_figs.items():
            lines.extend([
                rf"\noindent\textbf{{{_latex_escape(model_name)}}}\\[2pt]",
                r"\begin{center}",
                rf"\includegraphics[width=\textwidth]{{{_tex_result_path(fig_path)}}}",
                r"\end{center}",
                "",
            ])
        if not per_model_figs:
            lines.append(r"\noindent No prior/posterior figures available for this block.")
        lines.extend(
            [
                "",
                r"\subsection{Savage-Dickey Density Ratio}",
                r"\begin{center}\scriptsize",
                rf"\resizebox{{\textwidth}}{{!}}{{\input{{{_tex_result_path(block_table_dir / 'sddr.tex')}}}}}",
                r"\end{center}",
                "",
                r"\subsection{Time-Varying Kappa Path}",
                r"\begin{center}",
                rf"\includegraphics[width=0.85\textwidth]{{{_tex_result_path(kappa_path)}}}",
                rf"\includegraphics[width=0.85\textwidth]{{{_tex_result_path(theta_path)}}}",
                r"\end{center}",
                "",
                r"\subsection{Model Comparison}",
                r"\begin{center}\scriptsize",
                rf"\resizebox{{\textwidth}}{{!}}{{\input{{{_tex_result_path(block_table_dir / 'model_comparison.tex')}}}}}",
                r"\end{center}",
                "",
            ]
        )

    (tables_dir / "result_blocks.tex").write_text("\n".join(lines), encoding="utf-8")


def _make_outputs(
    idata_by_run: dict[str, object],
    *,
    priors: dict,
    tables_dir: Path,
    figures_dir: Path,
    data_spec: str | None = None,
    n_transform: str | None = None,
    competition_frequency: str | None = None,
    base_data_specs: list[str] | None = None,
    data_spec_labels: dict[str, str] | None = None,
    comparison_data: dict[tuple[str, str], dict[str, object]] | None = None,
) -> None:
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    first_idata = None
    base_names = base_data_specs or []
    scoped_runs = _filter_by_base_data_spec(idata_by_run, data_spec, base_names) if base_names else _filter_by_data_spec(idata_by_run, data_spec)
    scoped_runs = _filter_by_transform(scoped_runs, n_transform)
    scoped_runs = _filter_by_competition_frequency(scoped_runs, competition_frequency)
    latest_runs = _latest_by_fields(
        scoped_runs,
        ("model", "data_spec", "prior_spec", "period", "constraint_spec", "n_transform", "competition_measurement_frequency", "sample_start", "sample_end"),
    )
    baseline_runs = _filter_by_prior(latest_runs, "baseline")
    if not baseline_runs:
        baseline_runs = latest_runs
    for run_name, idata in sorted(baseline_runs.items()):
        if first_idata is None:
            first_idata = idata
        table = posterior_summary_table(idata)
        if not table.empty:
            table.insert(0, "run", run_name)
            table.insert(1, "model", getattr(idata, "attrs", {}).get("model", ""))
            table.insert(2, "data_spec", getattr(idata, "attrs", {}).get("data_spec", ""))
            table.insert(3, "prior_spec", getattr(idata, "attrs", {}).get("prior_spec", ""))
            table.insert(4, "constraint_spec", getattr(idata, "attrs", {}).get("constraint_spec", "unrestricted"))
            rows.append(table)
    summary = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame({"note": ["No posterior summaries available."]})
    summary.to_csv(tables_dir / "posterior_summary.csv", index=False)
    write_latex_fragment(
        _display(summary, _combined_columns(data_spec, ["model", "parameter", "unit", "mean", "sd", "ci_2.5", "ci_97.5"])),
        tables_dir / "posterior_summary.tex",
    )

    coeff = _key_coefficients(coefficient_means_table(baseline_runs))
    coeff_out = coeff if not coeff.empty else pd.DataFrame({"note": ["No baseline coefficient posterior means available for the current N transform. Rerun estimation."]})
    coeff_out.to_csv(tables_dir / "coefficient_means.csv", index=False)
    write_latex_fragment(
        _display(
            coeff_out,
            _combined_columns(data_spec, ["model", "parameter", "unit", "posterior_mean", "ci_2.5", "ci_97.5", "p_gt_0"]),
        ),
        tables_dir / "coefficient_means.tex",
    )

    # Pivot: unrestricted runs per (model, data_spec) regardless of n_transform so all 4 models appear.
    all_scoped = _filter_by_base_data_spec(idata_by_run, data_spec, base_names) if base_names else _filter_by_data_spec(idata_by_run, data_spec)
    all_scoped = _filter_by_competition_frequency(all_scoped, competition_frequency)
    unrestricted_scoped = _filter_by_constraint(all_scoped, "unrestricted") or all_scoped
    latest_all_transforms = _latest_by_fields(
        _filter_by_prior(unrestricted_scoped, "baseline") or unrestricted_scoped,
        ("model", "data_spec"),
    )
    pivot = coefficient_means_pivot_table(latest_all_transforms)
    pivot_out = pivot if not pivot.empty else pd.DataFrame({"note": ["No coefficient pivot available."]})
    pivot_out.to_csv(tables_dir / "coefficient_means_pivot.csv")
    write_latex_fragment(pivot_out, tables_dir / "coefficient_means_pivot.tex", index=not pivot.empty)

    kappa = kappa_comparison_table(baseline_runs)
    kappa_out = kappa if not kappa.empty else pd.DataFrame({"note": ["No kappa draws available for model comparison."]})
    kappa_out.to_csv(tables_dir / "kappa_model_comparison.csv", index=False)
    write_latex_fragment(
        _display(
            kappa_out,
            _combined_columns(
                data_spec,
                [
                    "model",
                    "kappa_mean",
                    "kappa_0_mean",
                    "delta_mean",
                    "kappa_t_overall_mean",
                    "kappa_t_start_mean",
                    "kappa_t_end_mean",
                ],
            ),
        ),
        tables_dir / "kappa_model_comparison.tex",
    )
    if not save_kappa_model_comparison(kappa, figures_dir / "kappa_model_comparison.png"):
        save_placeholder_figure(figures_dir / "kappa_model_comparison.png", "No kappa draws available yet.")

    tv = time_varying_coefficients_table(baseline_runs)
    tv_out = tv if not tv.empty else pd.DataFrame({"note": ["No time-varying kappa_t/theta_t draws available."]})
    tv_out.to_csv(tables_dir / "time_varying_coefficients.csv", index=False)
    write_latex_fragment(
        _display(tv_out, _combined_columns(data_spec, ["model", "coefficient", "time_index", "posterior_mean", "ci_2.5", "ci_97.5"])),
        tables_dir / "time_varying_coefficients.tex",
    )
    data_label = _data_spec_display(data_spec, data_spec_labels)
    if not save_time_varying_path(
        baseline_runs,
        "kappa_t",
        figures_dir / "kappa_t_path.png",
        title=f"{data_label}: time-varying kappa_t path",
        subtitle="baseline runs",
    ):
        save_placeholder_figure(figures_dir / "kappa_t_path.png", "No kappa_t path available. Run HSA steady or HSA full.")
    if not save_time_varying_path(
        baseline_runs,
        "theta_t",
        figures_dir / "theta_t_path.png",
        title=f"{data_label}: time-varying theta_t path",
        subtitle="baseline runs",
    ):
        save_placeholder_figure(figures_dir / "theta_t_path.png", "No theta_t path available. Run HSA full.")

    _write_competition_decomposition_outputs(
        runs=latest_runs,
        tables_dir=tables_dir,
        figures_dir=figures_dir,
        data_spec=data_spec,
    )

    sddr = sddr_summary_table(baseline_runs, priors)
    sddr_out = sddr if not sddr.empty else pd.DataFrame({"note": ["No SDDR restrictions available for current runs."]})
    sddr_out.to_csv(tables_dir / "sddr.csv", index=False)
    write_latex_fragment(
        _display(sddr_out, _combined_columns(data_spec, ["model", "restriction", "prior_mean", "prior_sd", "sddr_bf01"])),
        tables_dir / "sddr.tex",
    )

    prior_coeff = _key_coefficients(coefficient_means_table(latest_runs))
    if not prior_coeff.empty and "prior_spec" in prior_coeff.columns:
        prior_coeff = prior_coeff.sort_values(["data_spec", "prior_spec", "model", "parameter"])
    prior_coeff_out = (
        prior_coeff
        if not prior_coeff.empty
        else pd.DataFrame({"note": ["No prior-set coefficient summaries available for the current N transform. Rerun baseline, weak, and tight prior jobs."]})
    )
    prior_coeff_out.to_csv(tables_dir / "prior_set_coefficients.csv", index=False)
    write_latex_fragment(
        _display(
            prior_coeff_out,
            _combined_columns(data_spec, ["prior_spec", "model", "parameter", "posterior_mean", "ci_2.5", "ci_97.5", "p_gt_0"]),
        ),
        tables_dir / "prior_set_coefficients.tex",
    )
    if not save_coefficient_interval_plot(
        prior_coeff,
        figures_dir / "prior_set_key_coefficients.png",
        title="Key coefficients by prior set",
        label_columns=("prior_spec", "model", "parameter"),
    ):
        save_placeholder_figure(figures_dir / "prior_set_key_coefficients.png", "No prior-set coefficient summaries available.")

    generated_prior_figures: set[str] = set()
    if first_idata is not None:
        for var in ["alpha", "kappa", "kappa_0", "delta", "theta", "theta_0", "gamma"]:
            if var in first_idata.posterior:
                save_posterior_density(first_idata, var, figures_dir / f"posterior_density_{var}.png")
        for var in ["kappa", "kappa_0", "delta", "theta", "theta_0", "gamma"]:
            for idata in baseline_runs.values():
                if var not in getattr(idata, "posterior", {}):
                    continue
                prior_tuple = _prior_for_run(idata, priors, var)
                if prior_tuple is None:
                    continue
                saved = save_prior_posterior_overlay(
                    idata,
                    var,
                    prior_tuple,
                    figures_dir / f"prior_posterior_{var}.png",
                    **_overlay_options(var),
                )
                if saved:
                    generated_prior_figures.add(var)
                break
    for var in ["kappa", "kappa_0", "delta", "theta", "theta_0", "gamma"]:
        if var not in generated_prior_figures:
            save_placeholder_figure(figures_dir / f"prior_posterior_{var}.png", f"No posterior draws for {var} available yet.")
    save_posterior_predictive_placeholder(figures_dir / "posterior_predictive_placeholder.png")

    _write_existing_table(
        ROOT / "results" / "tables" / "mcmc_diagnostics.csv",
        tables_dir / "mcmc_diagnostics.tex",
        data_spec=data_spec,
        n_transform=n_transform,
        note="Run scripts/03_run_diagnostics.py first.",
        display_columns=_combined_columns(data_spec, ["model", "parameter", "mean", "r_hat", "ess_bulk", "ess_tail", "warning"]),
    )
    _write_existing_table(
        ROOT / "results" / "tables" / "prior_robustness.csv",
        tables_dir / "prior_robustness.tex",
        data_spec=data_spec,
        n_transform=n_transform,
        note="Run scripts/04_prior_robustness.py first.",
        display_columns=_combined_columns(data_spec, ["prior", "model", "parameter", "mean", "median", "ci_2.5", "ci_97.5", "p_gt_0", "conclusion"]),
    )
    period_table = _write_existing_table_from_candidates(
        [
            ROOT / "results" / "tables" / "period_robustness.csv",
            ROOT / "results" / "period_robustness" / "period_robustness.csv",
        ],
        tables_dir / "period_robustness.tex",
        data_spec=data_spec,
        n_transform=n_transform,
        note="Run scripts/05_period_robustness.py first.",
        display_columns=_combined_columns(data_spec, ["period", "model", "parameter", "mean", "ci_2.5", "ci_97.5", "status", "warning"]),
    )
    if {"parameter", "mean", "ci_2.5", "ci_97.5"} <= set(period_table.columns):
        plot_table = period_table.rename(columns={"mean": "posterior_mean"})
        plot_table = plot_table[plot_table["parameter"].astype(str).isin(KEY_COEFFICIENTS)]
        if not save_coefficient_interval_plot(
            plot_table,
            figures_dir / "period_set_key_coefficients.png",
            title="Key coefficients by sample period",
            label_columns=("period", "model", "parameter"),
        ):
            save_placeholder_figure(figures_dir / "period_set_key_coefficients.png", "No period coefficient summaries available.")
    else:
        save_placeholder_figure(figures_dir / "period_set_key_coefficients.png", "No period coefficient summaries available.")

    period_source = _filter_period_table(
        _read_table_from_candidates(
            [
                ROOT / "results" / "tables" / "period_robustness.csv",
                ROOT / "results" / "period_robustness" / "period_robustness.csv",
            ]
        ),
        data_spec=data_spec,
        n_transform=n_transform,
    )
    # Build cross-transform baseline runs keyed by (model, data_spec, constraint_spec).
    # Keeps both unrestricted and restricted variants so each block gets its own set.
    all_scoped = _filter_by_base_data_spec(idata_by_run, data_spec, base_names) if base_names else _filter_by_data_spec(idata_by_run, data_spec)
    all_scoped = _filter_by_competition_frequency(all_scoped, competition_frequency)
    all_baseline = _filter_by_prior(all_scoped, "baseline") or all_scoped
    all_baseline_runs = _latest_by_fields(all_baseline, ("model", "data_spec", "constraint_spec", "competition_measurement_frequency"))

    _write_result_blocks(
        latest_runs=latest_runs,
        priors=priors,
        tables_dir=tables_dir,
        figures_dir=figures_dir,
        data_spec=data_spec,
        base_names=base_names,
        data_spec_labels=data_spec_labels,
        period_table=period_source,
        comparison_data=comparison_data,
        all_baseline_runs=all_baseline_runs,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", default=str(ROOT / "results" / "runs"))
    parser.add_argument("--priors", default=str(ROOT / "configs" / "priors_baseline.yaml"))
    parser.add_argument("--config", default=str(ROOT / "configs" / "models.yaml"))
    parser.add_argument("--data", default=str(ROOT / "data" / "processed" / "model_ready.csv"))
    parser.add_argument("--periods", default=str(ROOT / "configs" / "periods.yaml"))
    parser.add_argument(
        "--data-spec",
        action="append",
        dest="data_specs",
        help="Data spec name to render. Repeat for multiple. Defaults to config run_data_specs.",
    )
    parser.add_argument(
        "--competition-frequency",
        choices=["quarterly_interpolated", "annual_q4"],
        help="Restrict table and figure generation to runs with this competition measurement frequency.",
    )
    parser.add_argument("--combined-only", action="store_true")
    args = parser.parse_args()

    config = load_model_config(args.config)
    data_specs = configured_data_specs(config, args.data_specs)
    data_spec_labels = {name: str(spec.get("label") or name) for name, spec in data_specs.items()}
    n_transform = str(config.get("defaults", {}).get("n_transform", DEFAULT_N_TRANSFORM))
    idata_by_run = _load_idata_by_run(args.runs_dir)
    priors = _load_prior(args.priors)
    data_path = Path(args.data)
    data = pd.read_csv(data_path, parse_dates=["DATE"]).set_index("DATE") if data_path.exists() else None
    comparison_data = _comparison_data_by_condition(
        data,
        {name: dict(spec) for name, spec in data_specs.items()},
        load_periods(args.periods),
        n_transform,
    )

    _make_outputs(
        idata_by_run,
        priors=priors,
        tables_dir=ROOT / "results" / "tables",
        figures_dir=ROOT / "results" / "figures",
        data_spec=None,
        n_transform=n_transform,
        competition_frequency=args.competition_frequency,
        base_data_specs=list(data_specs),
        data_spec_labels=data_spec_labels,
        comparison_data=comparison_data,
    )
    if not args.combined_only:
        for data_spec_name in data_specs:
            _make_outputs(
                idata_by_run,
                priors=priors,
                tables_dir=ROOT / "results" / "tables" / data_spec_name,
                figures_dir=ROOT / "results" / "figures" / data_spec_name,
                data_spec=data_spec_name,
                n_transform=n_transform,
                competition_frequency=args.competition_frequency,
                base_data_specs=list(data_specs),
                data_spec_labels=data_spec_labels,
                comparison_data=comparison_data,
            )


if __name__ == "__main__":
    main()
