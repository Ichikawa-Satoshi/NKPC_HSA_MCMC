from __future__ import annotations

import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


def _fmt(value: Any) -> str:
    if value is None:
        return "not available"
    if isinstance(value, float):
        return f"{value:.6g}"
    if isinstance(value, (list, tuple)):
        return ", ".join(str(v) for v in value) if value else "none"
    return str(value)


def _git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return "not available"
    return result.stdout.strip()


def _table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> list[str]:
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        out.append("| " + " | ".join(_fmt(value) for value in row) + " |")
    return out


def _model_variant_row(model: Mapping[str, Any]) -> list[Any]:
    name = str(model.get("model") or model.get("name") or "")
    uses_states = name.startswith("hsa_")
    return [
        name,
        "yes" if uses_states else "no",
        "yes" if uses_states else "no",
        "yes" if name in {"hsa_steady", "hsa_full"} else "no",
        "yes" if name in {"hsa_dynamic", "hsa_full"} else "no",
        "yes" if name == "hsa_dynamic" else "no",
        "yes" if uses_states else "no",
        "yes" if uses_states and model.get("competition_frequency") == "annual_q4" else "no",
    ]


def _available_outputs(output_dir: Path) -> list[str]:
    candidates = [
        "posterior.nc",
        "metadata.json",
        "priors.json",
        "data_spec.json",
        "diagnostics.csv",
        "mcmc_diagnostics.csv",
        "summary.csv",
        "posterior_summary.csv",
        "competition_decomposition_summary.csv",
        "estimation_results_report.md",
    ]
    paths = [str(output_dir / name) for name in candidates if (output_dir / name).exists()]
    figure_paths = sorted((output_dir / "figures").glob("*.png")) if (output_dir / "figures").exists() else []
    paths.extend(str(path) for path in figure_paths)
    return paths


def write_data_model_report(
    output_dir: str | Path,
    run_or_batch_metadata: Mapping[str, Any],
    sample_metadata: Mapping[str, Any] | None = None,
    data_metadata: Mapping[str, Any] | None = None,
    competition_metadata: Mapping[str, Any] | None = None,
    model_variant_metadata: Sequence[Mapping[str, Any]] | None = None,
    priors_metadata: Mapping[str, Any] | None = None,
    scaling_metadata: Mapping[str, Any] | None = None,
    constraint_metadata: Mapping[str, Any] | None = None,
    output_metadata: Mapping[str, Any] | None = None,
) -> Path:
    """Write one consolidated Markdown data/model specification report."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    target = out / "data_model_report.md"

    run = dict(run_or_batch_metadata)
    sample = dict(sample_metadata or {})
    data = dict(data_metadata or {})
    comp = dict(competition_metadata or {})
    priors = dict(priors_metadata or {})
    scaling = dict(scaling_metadata or {})
    constraints = dict(constraint_metadata or {})
    outputs = dict(output_metadata or {})
    variants = list(model_variant_metadata or [{"model": run.get("model", ""), "competition_frequency": comp.get("frequency", "")}])

    frequency = str(comp.get("frequency", run.get("competition_measurement_frequency", "quarterly_interpolated")))
    annual_timing = str(comp.get("annual_timing", run.get("competition_measurement_annual_timing", "q4")))
    created = datetime.now(timezone.utc).isoformat(timespec="seconds")

    lines: list[str] = [
        "# Data and Model Specification Report",
        "",
        "This report documents the data construction and empirical model setup used for this run or batch.",
        "",
        "## Run or Batch Metadata",
        "",
    ]
    metadata_rows = [
        ("report_creation_timestamp", created),
        ("run_or_batch_name", run.get("run_name") or run.get("run_id") or run.get("name")),
        ("repository_commit_hash", run.get("commit_hash") or _git_commit()),
        ("command_or_script", run.get("command") or "not available"),
        ("config_paths", run.get("config_paths") or run.get("config_path") or "not available"),
        ("prior_config_paths", run.get("prior_config_paths") or run.get("prior_config_path") or "not available"),
        ("number_of_models", run.get("number_of_models") or len(variants)),
        ("model_variants", [v.get("model") or v.get("name") for v in variants]),
        ("activity_proxies", run.get("activity_proxies") or data.get("activity_proxy") or data.get("activity_proxies")),
        ("seed", run.get("seed")),
        ("chains", run.get("chains")),
        ("burn_in", run.get("burn")),
        ("kept_draws", run.get("kept_draws") or run.get("n_iter")),
        ("thinning_or_store_every", run.get("thin") or run.get("store_every")),
        ("mcmc_date_time", run.get("mcmc_date_time") or run.get("run_id")),
    ]
    lines.extend(f"- **{key}:** {_fmt(value)}" for key, value in metadata_rows)

    lines.extend(
        [
            "",
            "## Sample and Frequency",
            "",
            f"- **quarterly_estimation_sample_start:** {_fmt(sample.get('sample_start', run.get('sample_start')))}",
            f"- **quarterly_estimation_sample_end:** {_fmt(sample.get('sample_end', run.get('sample_end')))}",
            f"- **quarterly_observations_T:** {_fmt(sample.get('T', run.get('n_obs')))}",
            "- **Inflation:** quarterly",
            "- **Expectations:** quarterly",
            "- **Activity proxies:** quarterly",
            "- **Raw competition source N_Gustavo:** annual",
            f"- **Competition measurement in state-space model:** {frequency}",
            "",
            "## Data Sources and Transformations",
            "",
            f"- **inflation_series_name:** {_fmt(data.get('pi_col'))}",
            f"- **expectation_series_name:** {_fmt(data.get('pi_expect_col'))}",
            f"- **lagged_inflation_term:** {_fmt(data.get('pi_prev_col'))}",
            f"- **activity_variables:** {_fmt(data.get('x_col'))}",
            "- **activity_sign_conventions:** as supplied in the configured activity column",
            f"- **competition_series_name:** {_fmt(data.get('n_col', 'N_Gustavo'))}",
            f"- **competition_units:** {_fmt(scaling.get('N_units', 'centered ten-log-point units under the default transform'))}",
            f"- **N_Gustavo_transformation:** {_fmt(data.get('n_transform_note') or run.get('n_transform_note'))}",
            f"- **interpolation_method:** {_fmt(comp.get('interpolation_method'))}",
            f"- **annual_timing:** {annual_timing}",
        ]
    )
    if frequency == "annual_q4":
        lines.append(
            "- Annual N_Gustavo is transformed into the same centered ten-log-point units as the baseline model. "
            "It is not interpolated. The transformed annual observation is loaded only in Q4 of each year. "
            "Q1-Q3 observations are treated as missing in the N measurement equation."
        )
    else:
        lines.append(
            "- Annual N_Gustavo is transformed into centered ten-log-point units and PCHIP-interpolated to the "
            "quarterly model sample. The interpolated quarterly series is used as N_obs_t in every quarter."
        )

    lines.extend(
        [
            "",
            "## Observed Competition Information",
            "",
            f"- **finite_N_obs_count:** {_fmt(comp.get('finite_N_obs_count'))}",
            f"- **missing_N_obs_count:** {_fmt(comp.get('missing_N_obs_count'))}",
            f"- **first_finite_N_obs_quarter:** {_fmt(comp.get('first_finite_N_obs'))}",
            f"- **last_finite_N_obs_quarter:** {_fmt(comp.get('last_finite_N_obs'))}",
            f"- **observed_annual_Q4_quarters:** {_fmt(comp.get('observed_quarters'))}",
            f"- **finite_N_obs_min:** {_fmt(comp.get('finite_N_obs_min'))}",
            f"- **finite_N_obs_max:** {_fmt(comp.get('finite_N_obs_max'))}",
            f"- **finite_N_obs_mean:** {_fmt(comp.get('finite_N_obs_mean'))}",
            f"- **finite_N_obs_std:** {_fmt(comp.get('finite_N_obs_std'))}",
            "",
            "## Model Variants",
            "",
        ]
    )
    lines.extend(
        _table(
            [
                "model_variant",
                "uses_N_states",
                "uses_N_measurement",
                "kappa_interaction",
                "dynamic_entry",
                "correlated_noise",
                "sigma_N2_sampled",
                "annual_q4_missing_logic",
            ],
            [_model_variant_row({**variant, "competition_frequency": frequency}) for variant in variants],
        )
    )

    lines.extend(
        [
            "",
            "## Model Equations",
            "",
            "Shared HSA state block:",
            "",
            "```text",
            "N_t = Nbar_t + Nhat_t",
            "Nhat_t = rho_1 * Nhat_{t-1} + rho_2 * Nhat_{t-2} + u_t",
            "Nbar_t = Nbar_{t-1} + n + eps_t",
            "N_obs_t = Nhat_t + Nbar_t + nu_t, only when N_obs_t is observed",
            "```",
            "",
            "hsa_steady:",
            "",
            "```text",
            "pi_t - Epi_t = alpha * (pi_{t-1} - Epi_t) + (kappa_0 + delta * Nbar_t) * x_t + e_t",
            "```",
            "",
            "hsa_full:",
            "",
            "```text",
            "pi_t - Epi_t = alpha * (pi_{t-1} - Epi_t) + (kappa_0 + delta * Nbar_t) * x_t",
            "              - (theta_0 + gamma * Nbar_t) * Nhat_t + e_t",
            "```",
            "",
            "hsa_dynamic:",
            "",
            "```text",
            "pi_t - Epi_t = alpha * (pi_{t-1} - Epi_t) + kappa * x_t - theta * Nhat_t + e_t",
            "x_t = phi_1 * x_{t-1} + zeta_t",
            "[e_t, zeta_t, u_t, eps_t]' follows the configured covariance structure.",
            "```",
            "",
            "CES:",
            "",
            "```text",
            "pi_t - Epi_t = alpha * (pi_{t-1} - Epi_t) + kappa * x_t + e_t",
            "```",
            "",
            "## Mixed-Frequency Measurement",
            "",
        ]
    )
    if frequency == "annual_q4":
        lines.append(
            "This run estimates a mixed-frequency state-space model. Inflation and activity variables are observed "
            "quarterly. Competition N is treated as a quarterly latent state, but the raw competition data are annual. "
            "The annual competition observation is loaded only in Q4 through the N measurement equation. In Q1-Q3, "
            "the N measurement equation is omitted, and quarterly latent N is inferred from the state transition "
            "equations and the quarterly inflation/activity equations."
        )
    else:
        lines.append(
            "This run uses the legacy quarterly-interpolated competition measurement. Annual competition data are "
            "converted to quarterly frequency before estimation, and the resulting N_obs_t is included in the N "
            "measurement equation in every quarter."
        )

    lines.extend(["", "## Priors", ""])
    if priors:
        lines.extend(_table(["prior", "value"], sorted(priors.items())))
    else:
        lines.append("No prior metadata was supplied.")

    lines.extend(
        [
            "",
            "## Scaling and Units",
            "",
            f"- **N_units:** {_fmt(scaling.get('N_units', 'centered ten-log-point units under the default transform'))}",
            f"- **kappa_parameter_scaling:** {_fmt(scaling.get('kappa_parameter_scaling', run.get('kappa_unit_note')))}",
            f"- **x_t_divided_by_100_inside_sampler:** {_fmt(scaling.get('x_divided_by_100', 'yes for kappa-like HSA regressors'))}",
            f"- **posterior_draw_units:** {_fmt(scaling.get('posterior_draw_units', 'physical units'))}",
            f"- **table_and_plot_units:** {_fmt(scaling.get('table_and_plot_units', 'physical units'))}",
            "",
            "## Constraints",
            "",
            f"- **constraint_spec:** {_fmt(constraints.get('constraint_spec', run.get('constraint_spec')))}",
            f"- **kappa_t_path_constraint_active:** {_fmt(constraints.get('kappa_t_path_constraint_active'))}",
            f"- **bounds:** {_fmt(constraints.get('bounds'))}",
            f"- **bounds_units:** {_fmt(constraints.get('bounds_units', 'physical units'))}",
            f"- **rejected_proposals:** {_fmt(constraints.get('rejected_proposals'))}",
            f"- **rejection_rate:** {_fmt(constraints.get('rejection_rate'))}",
            "",
            "## Output Artifacts",
            "",
        ]
    )
    available = list(outputs.get("available_outputs") or _available_outputs(out))
    if available:
        lines.extend(f"- {path}" for path in available)
    else:
        lines.append("- not generated")

    target.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return target
