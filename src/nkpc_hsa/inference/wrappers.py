from __future__ import annotations

import inspect
import json
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
import pandas as pd
import yaml

from nkpc_hsa.dataprep.competition import (
    CompetitionObservation,
    build_competition_observation,
    competition_observation_from_array,
    load_raw_annual_competition_series,
    normalize_competition_measurement,
    pchip_interpolate_annual_q4,
    to_quarter_period_index,
)
from nkpc_hsa.dataprep.load import load_processed_dataset
from nkpc_hsa.dataprep.build import hp_filter_series
from nkpc_hsa.dataprep.transforms import DEFAULT_N_TRANSFORM, competition_transform_note, transform_competition_series
from nkpc_hsa.models.common import (
    KAPPA_SCALE,
    KAPPA_UNIT_NOTE,
    coefficient_constraints_to_internal,
    prior_specs_to_internal,
)
from nkpc_hsa.paths import project_path
from nkpc_hsa.provenance import stamp_artifact_metadata
from nkpc_hsa.theory_models import (
    RESTRICTION_TAXONOMY,
    STRUCTURAL_FREQUENCY,
    THEORY_ESTIMATION_REVISION,
    THEORY_MODELS,
    is_theory_model,
    mu_from_zeta,
    theory_model_definition,
    validate_theory_run_design,
)


# Bumped when the estimation inputs or the sampler change in a way that makes older
# runs incomparable. Every run stamps this into its metadata, and both the report
# builder and script 13 select on it, so runs from earlier revisions stay on disk and
# stay out of the report automatically -- that is how a superseded vintage is kept for
# comparison rather than deleted.
#
# 2026-08-unrate-sa-v1  unemp_gap switched from UNRATENSA to UNRATE (seasonally
#                       adjusted).
# 2026-08-theta-centred  theta and theta_0 prior means moved from 0.1 to 0 in all
#                       three prior sets. The sweep is meant to vary dispersion
#                       only; leaving theta centred at the theory-preferred sign
#                       in baseline/tight but at 0 in weak mixed a location
#                       change into it, so the weak-prior entry-coefficient
#                       results were not comparable with the others. 0 also
#                       matches delta and gamma, which were already centred there.
# 2026-08-sample-window-v1  the production report driver now resolves data specs
#                       through configured_data_specs, so the declared
#                       1982Q1--2012Q4 window is applied after the PCHIP date fix.
#                       This deliberately invalidates runs whose revision label
#                       predates the corrected input alignment / window wiring.
# 2026-08-independent-audit-v1  SEC Q4 revenue is subtracted only within the
#                       same XBRL concept; full-Sigma FFBS uses the correct
#                       correlated-noise backward conditional; Particle-Gibbs
#                       path-mixing diagnostics are persisted. Predictive/report
#                       artifacts also enforce the estimator's sample and run
#                       provenance. Earlier posterior files are not silently
#                       promoted across these changes.
ESTIMATION_REVISION = "2026-08-independent-audit-v1"


@dataclass(frozen=True)
class RunMetadata:
    model: str
    data_spec: str
    prior_spec: str
    run_id: str
    n_iter: int
    burn: int
    thin: int
    chains: int
    seed: int
    n_transform: str
    competition_measurement_frequency: str = "annual_q4"
    competition_measurement_annual_timing: str = "q4"
    period: str = "full"
    covariance_structure: str = "e_zeta_only"
    constraint_spec: str = "unrestricted"
    coefficient_constraints: Mapping[str, Any] | None = None
    kappa_scale: float = KAPPA_SCALE
    kappa_units: str = "physical"
    estimation_revision: str = ESTIMATION_REVISION


def _load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _code_revision() -> tuple[str, bool]:
    """Return the checked-out revision and whether tracked/untracked files differ."""
    root = project_path()
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, check=True,
            capture_output=True, text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"], cwd=root, check=True,
                capture_output=True, text=True,
            ).stdout.strip()
        )
        return revision, dirty
    except (OSError, subprocess.SubprocessError):
        return "unknown", True


def _constraint_label(spec: Mapping[str, Any] | None) -> str:
    raw = dict(spec or {})
    enabled = bool(raw.get("enabled") or raw.get("positive") or raw.get("nonnegative") or raw.get("bounds"))
    if not enabled:
        return "unrestricted"
    names = set(str(name) for name in raw.get("positive", []) or [])
    names.update(str(name) for name in raw.get("nonnegative", []) or [])
    names.update(str(name) for name in dict(raw.get("bounds", {}) or {}))
    suffix = "_".join(sorted(name.replace(" ", "_") for name in names))
    return "restricted" if not suffix else f"restricted_{suffix}"


def _default_run_dir(
    model: str,
    data_spec: str,
    prior_spec: str,
    constraint_spec: str,
    run_id: str,
    *,
    competition_frequency: str = "quarterly_interpolated",
) -> Path:
    parts = [model, data_spec, prior_spec]
    if constraint_spec != "unrestricted":
        parts.append(constraint_spec)
    # Always name the observation design. Earlier revisions omitted it for the
    # interpolated case because that was the default; now that mixed frequency is the
    # default, a conditional suffix would silently invert the convention.
    parts.append(competition_frequency)
    parts.append(run_id)
    safe = "_".join(part.replace("/", "-") for part in parts)
    if is_theory_model(model):
        return project_path("results", "theory_runs", THEORY_ESTIMATION_REVISION, safe)
    return project_path("results", "runs", safe)


def _apply_sample_window(frame: pd.DataFrame, spec: Mapping[str, Any]) -> pd.DataFrame:
    """Restrict to the configured estimation window before complete-case selection.

    With no explicit window the sample is whatever ``dropna`` happens to leave,
    which makes it an accident of how far each input series runs. That is not
    hypothetical: correcting the year-start/quarter-end mismatch in
    ``dataprep.func_data_build.annual_to_quarterly_pchip`` extended the
    interpolated firm count by four quarters and silently moved the estimation
    sample from 1982Q1-2012Q4 to 1982Q1-2013Q4. The window is compared on
    quarterly periods because the processed index carries end-of-day timestamps
    (``2012-12-31 23:59:59.999999``), against which a plain ``<= 2012-12-31``
    would drop the final quarter.
    """
    start = spec.get("sample_start")
    end = spec.get("sample_end")
    if (start is None and end is None) or not isinstance(frame.index, pd.DatetimeIndex):
        return frame
    periods = frame.index.to_period("Q")
    mask = np.ones(len(frame), dtype=bool)
    if start is not None:
        mask &= periods >= pd.Period(str(start), freq="Q")
    if end is not None:
        mask &= periods <= pd.Period(str(end), freq="Q")
    return frame.loc[mask]


def _coerce_model_data(
    data: pd.DataFrame | Mapping[str, Any] | None,
    *,
    data_spec: Mapping[str, Any] | None = None,
) -> dict[str, np.ndarray]:
    if data is None:
        data = load_processed_dataset()
    if isinstance(data, pd.DataFrame):
        spec = dict(data_spec or {})
        cols = {
            "pi": spec.get("pi_col", "pi"),
            "pi_prev": spec.get("pi_prev_col", "pi_prev"),
            "pi_expect": spec.get("pi_expect_col", "pi_expect"),
            "x": spec.get("x_col", "x"),
            "x_prev": spec.get("x_prev_col", "x_prev"),
            "N": spec.get("n_col", spec.get("N_col", "N")),
        }
        if spec.get("e_hat_col") is not None:
            cols["Ehat"] = spec["e_hat_col"]
        elif spec.get("e_col") is not None:
            cols["E"] = spec["e_col"]
        required = ["pi", "pi_prev", "pi_expect", "x", "x_prev"]
        if cols["N"] in data.columns:
            required.append("N")
        if "Ehat" in cols:
            required.append("Ehat")
        elif "E" in cols:
            required.append("E")
        missing = [cols[k] for k in required if cols[k] not in data.columns]
        if missing:
            raise KeyError(f"Missing model-ready columns: {missing}")
        sample = _apply_sample_window(data[[cols[k] for k in required]], spec).dropna()
        out = {k: sample[cols[k]].to_numpy(dtype=float) for k in required}
        if "E" in out:
            e_transform = str(spec.get("e_transform", "log100_centered10"))
            e_model = transform_competition_series(out["E"], transform=e_transform)  # type: ignore[arg-type]
            out["E_model"] = np.asarray(e_model, dtype=float)
            establishment_model = str(spec.get("establishment_model", "legacy_hp_loading")).lower()
            if establishment_model == "joint_state":
                out["E_obs"] = np.asarray(e_model, dtype=float)
            elif establishment_model == "legacy_hp_loading":
                e_filter = str(spec.get("e_cycle_filter", "hp")).lower()
                if e_filter != "hp":
                    raise ValueError(f"Unsupported establishment cycle filter: {e_filter!r}")
                e_lambda = float(spec.get("e_hp_lambda", 1600.0))
                ebar, ehat = hp_filter_series(pd.Series(e_model), lamb=e_lambda)
                out["Ebar"] = ebar.to_numpy(dtype=float)
                out["Ehat"] = ehat.to_numpy(dtype=float)
            else:
                raise ValueError(f"Unsupported establishment_model: {establishment_model!r}")
        return out
    return {k: np.asarray(v, dtype=float).reshape(-1) for k, v in data.items()}


def model_sample_index(data: pd.DataFrame | Mapping[str, Any] | None, data_spec: Mapping[str, Any]) -> pd.Index | None:
    """Return the complete-case index used by the sampler for a data specification."""
    if not isinstance(data, pd.DataFrame):
        return None
    cols = {
        "pi": data_spec.get("pi_col", "pi"),
        "pi_prev": data_spec.get("pi_prev_col", "pi_prev"),
        "pi_expect": data_spec.get("pi_expect_col", "pi_expect"),
        "x": data_spec.get("x_col", "x"),
        "x_prev": data_spec.get("x_prev_col", "x_prev"),
        "N": data_spec.get("n_col", data_spec.get("N_col", "N")),
    }
    if data_spec.get("e_hat_col") is not None:
        cols["Ehat"] = data_spec["e_hat_col"]
    elif data_spec.get("e_col") is not None:
        cols["E"] = data_spec["e_col"]
    required = ["pi", "pi_prev", "pi_expect", "x", "x_prev"]
    if cols["N"] in data.columns:
        required.append("N")
    if "Ehat" in cols:
        required.append("Ehat")
    elif "E" in cols:
        required.append("E")
    missing = [cols[k] for k in required if cols[k] not in data.columns]
    if missing:
        return None
    return _apply_sample_window(data[[cols[k] for k in required]], data_spec).dropna().index


def _fallback_quarterly_index(T: int) -> pd.PeriodIndex:
    return pd.period_range("2000Q1", periods=T, freq="Q")


def _transform_competition_observation(arr: np.ndarray, n_transform: str) -> np.ndarray:
    values = np.asarray(arr, dtype=float).reshape(-1)
    if np.all(np.isfinite(values)):
        return transform_competition_series(values, transform=n_transform)  # type: ignore[arg-type]
    if n_transform == "identity":
        if not np.isfinite(values).any():
            raise ValueError("N contains no finite observations.")
        return values.copy()
    raise ValueError(
        "Sparse competition observations require n_transform='identity' unless raw annual data "
        "are available through the standard DataFrame pipeline."
    )


def _transform_annual_competition_like_quarterly(
    annual_raw: pd.Series,
    quarterly_raw_reference: np.ndarray,
    n_transform: str,
) -> np.ndarray:
    annual_values = annual_raw.to_numpy(dtype=float)
    if n_transform == "log100_centered10":
        reference = np.asarray(quarterly_raw_reference, dtype=float).reshape(-1)
        if np.any(annual_values <= 0.0) or np.any(reference <= 0.0):
            raise ValueError("N must be strictly positive before log transformation.")
        center = float(np.mean(100.0 * np.log(reference)))
        return (100.0 * np.log(annual_values) - center) / 10.0
    return transform_competition_series(annual_values, transform=n_transform)  # type: ignore[arg-type]


def _competition_observation_metadata(obs: CompetitionObservation, values: np.ndarray) -> dict[str, Any]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    return {
        "frequency": obs.frequency,
        "annual_timing": obs.annual_timing,
        "finite_N_obs_count": obs.finite_count,
        "missing_N_obs_count": obs.missing_count,
        "first_finite_N_obs": obs.first_finite,
        "last_finite_N_obs": obs.last_finite,
        "observed_quarters": obs.observed_quarters,
        "interpolation_method": obs.interpolation_method,
        "finite_N_obs_min": float(np.min(finite)) if finite.size else None,
        "finite_N_obs_max": float(np.max(finite)) if finite.size else None,
        "finite_N_obs_mean": float(np.mean(finite)) if finite.size else None,
        "finite_N_obs_std": float(np.std(finite, ddof=1)) if finite.size > 1 else 0.0 if finite.size else None,
    }


def _prepare_competition_measurement(
    *,
    model: str,
    data: pd.DataFrame | Mapping[str, Any] | None,
    data_spec: Mapping[str, Any],
    model_data: dict[str, np.ndarray],
    sample_index: pd.Index | None,
    n_transform: str,
    competition_measurement: Mapping[str, Any] | None,
) -> dict[str, Any]:
    spec = normalize_competition_measurement(competition_measurement)
    T = int(len(model_data["pi"]))
    q_index = to_quarter_period_index(sample_index) if sample_index is not None else _fallback_quarterly_index(T)

    context: dict[str, Any] = {
        "competition_measurement": spec,
        "quarterly_index": q_index,
        "N_obs_used": None,
        "N_interpolated_comparison": None,
        "N_annual_q4": None,
        "metadata": {},
    }
    if model == "ces":
        return context
    if "N" not in model_data:
        raise KeyError(f"{model} requires an N series.")

    N_quarterly_raw = np.asarray(model_data["N"], dtype=float).reshape(-1)
    N_interpolated = None
    if np.all(np.isfinite(N_quarterly_raw)):
        N_interpolated = transform_competition_series(N_quarterly_raw, transform=n_transform)  # type: ignore[arg-type]
        context["N_interpolated_comparison"] = N_interpolated
    annual_transformed: pd.Series | None = None
    if isinstance(data, pd.DataFrame) and sample_index is not None:
        n_col = str(data_spec.get("n_col", data_spec.get("N_col", "N_Gustavo")))
        try:
            annual_raw = load_raw_annual_competition_series(n_col)
            years = sorted(set(int(p.year) for p in q_index))
            annual_raw = annual_raw.loc[annual_raw.index.isin(years)]
            annual_transformed = pd.Series(
                _transform_annual_competition_like_quarterly(annual_raw, N_quarterly_raw, n_transform),
                index=annual_raw.index,
                name=n_col,
            )
            annual_obs = build_competition_observation(
                annual_transformed,
                q_index,
                frequency="annual_q4",
                annual_timing=spec["annual_timing"],
            )
            context["N_annual_q4"] = annual_obs.N_obs
            context["N_interpolated_q4_comparison"] = pchip_interpolate_annual_q4(
                annual_transformed,
                q_index,
                annual_timing=spec["annual_timing"],
            )
        except Exception:
            annual_transformed = None

    frequency = spec["frequency"]
    if frequency in {"quarterly_interpolated", "quarterly_observed"}:
        if N_interpolated is None:
            N_interpolated = _transform_competition_observation(N_quarterly_raw, n_transform)
            context["N_interpolated_comparison"] = N_interpolated
        obs = competition_observation_from_array(
            N_interpolated,
            q_index,
            frequency=frequency,
            annual_timing=spec["annual_timing"],
        )
        context["N_obs_used"] = obs.N_obs
        context["metadata"] = _competition_observation_metadata(obs, obs.N_obs)
        return context

    if annual_transformed is not None:
        obs = build_competition_observation(
            annual_transformed,
            q_index,
            frequency="annual_q4",
            annual_timing=spec["annual_timing"],
        )
        context["N_obs_used"] = obs.N_obs
        context["N_annual_q4"] = obs.N_obs
        context["N_interpolated_comparison"] = context.get("N_interpolated_q4_comparison")
        context["metadata"] = _competition_observation_metadata(obs, obs.N_obs)
        return context
    if isinstance(data, pd.DataFrame) and sample_index is not None:
        raise ValueError("annual_q4 competition measurement requires the raw annual competition source file.")

    obs = competition_observation_from_array(
        _transform_competition_observation(N_quarterly_raw, n_transform),
        q_index,
        frequency="annual_q4",
        annual_timing=spec["annual_timing"],
    )
    context["N_obs_used"] = obs.N_obs
    context["N_annual_q4"] = obs.N_obs
    context["metadata"] = _competition_observation_metadata(obs, obs.N_obs)
    return context


def _extract_draws_from_result(result: Mapping[str, Any]) -> dict[str, np.ndarray]:
    draws: dict[str, np.ndarray] = {}
    for key, value in result.items():
        if key in {"priors", "opts", "model", "deprecated"}:
            continue
        if key == "state_draws":
            for state_key, state_value in value.items():
                draws[state_key] = np.asarray(state_value, dtype=float)
            continue
        if key == "pg_diagnostics":
            # The Particle-Gibbs sampler returns these as a plain mapping rather
            # than the usual {"draws": ...} summary.  Persist the iteration-level
            # diagnostics in posterior.nc; otherwise the only evidence about path
            # degeneracy disappears at the production wrapper boundary.
            for diagnostic_key, diagnostic_value in value.items():
                if diagnostic_key == "n_particles":
                    continue
                diagnostic_array = np.asarray(diagnostic_value, dtype=float)
                if diagnostic_array.ndim > 0:
                    draws[f"pg_{diagnostic_key}"] = diagnostic_array
            continue
        if isinstance(value, Mapping) and "draws" in value:
            draws[key] = np.asarray(value["draws"], dtype=float)
    if "sigma_e2" in draws:
        draws["sigma_e"] = np.sqrt(np.maximum(draws.pop("sigma_e2"), 0.0))
    if "sigma_zeta2" in draws:
        draws["sigma_zeta"] = np.sqrt(np.maximum(draws.pop("sigma_zeta2"), 0.0))
    if "rho1" in draws:
        draws["rho_1"] = draws.pop("rho1")
    if "rho2" in draws:
        draws["rho_2"] = draws.pop("rho2")
    if "rho_N1" in draws:
        draws.setdefault("rho_1", draws["rho_N1"])
    if "rho_N2" in draws:
        draws.setdefault("rho_2", draws["rho_N2"])
    if "n_N" in draws:
        draws.setdefault("n", draws["n_N"])
    if "sigma_uN" in draws:
        draws.setdefault("sigma_u", draws["sigma_uN"])
    if "sigma_epsN" in draws:
        draws.setdefault("sigma_eps", draws["sigma_epsN"])
    return draws


def _stack_chains(chain_draws: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    keys = sorted(set().union(*(draws.keys() for draws in chain_draws)))
    stacked: dict[str, np.ndarray] = {}
    for key in keys:
        arrays = [draws[key] for draws in chain_draws if key in draws]
        if len(arrays) != len(chain_draws):
            continue
        stacked[key] = np.stack(arrays, axis=0)
    return stacked


def _to_idata(posterior: Mapping[str, np.ndarray], metadata: Mapping[str, Any]):
    import arviz as az

    dims: dict[str, list[str]] = {}
    coords: dict[str, np.ndarray] = {}
    posterior_dict: dict[str, np.ndarray] = {}
    for key, arr in posterior.items():
        value = np.asarray(arr, dtype=float)
        if value.ndim < 2:
            raise ValueError(f"Posterior variable {key} must include chain and draw axes.")
        posterior_dict[key] = value
        if value.ndim > 2:
            extra_dims = [f"{key}_dim_{i}" for i in range(value.ndim - 2)]
            dims[key] = extra_dims
            for axis, dim_name in enumerate(extra_dims, start=2):
                coords.setdefault(dim_name, np.arange(value.shape[axis]))
    idata = az.from_dict({"posterior": posterior_dict}, coords=coords, dims=dims)
    idata.attrs.update({k: str(v) for k, v in metadata.items()})
    return idata


def _save_run(
    *,
    idata,
    run_dir: Path,
    metadata: Mapping[str, Any],
    prior_specs: Mapping[str, Any],
    data_spec: Mapping[str, Any] | None,
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    (run_dir / "priors.json").write_text(json.dumps(prior_specs, indent=2, sort_keys=True), encoding="utf-8")
    (run_dir / "data_spec.json").write_text(json.dumps(dict(data_spec or {}), indent=2, sort_keys=True), encoding="utf-8")
    idata.to_netcdf(run_dir / "posterior.nc")


def _write_run_data_model_artifacts(
    *,
    idata,
    run_dir: Path,
    metadata: Mapping[str, Any],
    prior_specs: Mapping[str, Any],
    data_spec: Mapping[str, Any] | None,
    competition_context: Mapping[str, Any],
) -> None:
    from nkpc_hsa.reporting.data_model_report import write_data_model_report
    from nkpc_hsa.reporting.estimation_results import write_estimation_results_report
    from nkpc_hsa.reporting.figures import plot_competition_path_comparison

    model = str(metadata.get("model", ""))
    data_spec_dict = dict(data_spec or {})
    comp = dict(metadata.get("competition_measurement", {}) or {})
    posterior = getattr(idata, "posterior", None)
    if (
        model.startswith("hsa_")
        and posterior is not None
        and "Nbar" in posterior
        and "Nhat" in posterior
        and competition_context.get("N_obs_used") is not None
    ):
        plot_competition_path_comparison(
            quarterly_index=competition_context.get("quarterly_index"),
            Nbar_draws=np.asarray(posterior["Nbar"], dtype=float),
            Nhat_draws=np.asarray(posterior["Nhat"], dtype=float),
            N_obs_used=np.asarray(competition_context["N_obs_used"], dtype=float),
            N_annual_q4=competition_context.get("N_annual_q4"),
            N_interpolated_comparison=competition_context.get("N_interpolated_comparison"),
            model_name=model,
            activity_name=str(metadata.get("data_spec", "")),
            frequency=str(comp.get("frequency", "quarterly_interpolated")),
            output_dir=run_dir / "figures",
        )

    write_estimation_results_report(
        run_dir,
        idata,
        metadata=metadata,
        quarterly_index=competition_context.get("quarterly_index"),
    )

    constraints = dict(metadata.get("coefficient_constraints", {}) or {})
    bounds = constraints.get("bounds", {}) if isinstance(constraints.get("bounds", {}), Mapping) else {}
    data_model_report = write_data_model_report(
        run_dir,
        run_or_batch_metadata={
            **dict(metadata),
            "run_name": run_dir.name,
            "kept_draws": int(metadata.get("n_iter", 0)) - int(metadata.get("burn", 0)),
            "store_every": metadata.get("thin"),
        },
        sample_metadata={
            "sample_start": metadata.get("sample_start"),
            "sample_end": metadata.get("sample_end"),
            "T": metadata.get("n_obs"),
        },
        data_metadata={
            **data_spec_dict,
            "n_transform_note": metadata.get("n_transform_note"),
        },
        competition_metadata=comp,
        model_variant_metadata=[
            {
                "model": model,
                "competition_frequency": comp.get("frequency"),
            }
        ],
        priors_metadata=prior_specs,
        scaling_metadata={
            "N_units": metadata.get("n_transform_note"),
            "kappa_parameter_scaling": metadata.get("kappa_unit_note"),
            "x_divided_by_100": "yes for kappa-like sampler regressors",
            "posterior_draw_units": "physical units",
            "table_and_plot_units": "physical units",
        },
        constraint_metadata={
            "constraint_spec": metadata.get("constraint_spec"),
            "kappa_t_path_constraint_active": "kappa_t" in bounds or "kappa" in bounds,
            "bounds": bounds,
            "bounds_units": "physical units",
        },
    )
    report_dir = run_dir / "report"
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "data_model_report.md").write_text(data_model_report.read_text(encoding="utf-8"), encoding="utf-8")


def _call_sampler(func: Callable[..., Mapping[str, Any]], kwargs: dict[str, Any], *, orth: bool) -> Mapping[str, Any]:
    params = inspect.signature(func).parameters
    if "orth" in params:
        kwargs["orth"] = orth
    return func(**kwargs)


def _run_sampler(
    *,
    model: str,
    model_data: Mapping[str, np.ndarray],
    prior_specs: Mapping[str, Any] | None,
    n_iter: int,
    burn: int,
    thin: int,
    chains: int,
    seed: int,
    orth: bool,
    n_transform: str,
    covariance_structure: str,
    coefficient_constraints: Mapping[str, Any] | None,
    enforce_stationary: bool = True,
    ar2_max_tries: int = 2000,
    n_particles: int = 512,
    no_inertia: bool = False,
    theory_options: Mapping[str, Any] | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    from nkpc_hsa.models.ces import func_nkpc_ces
    from nkpc_hsa.models.hsa_const_theta import func_nkpc_hsa_const_theta
    from nkpc_hsa.models.hsa_dynamic import func_nkpc_hsa_decomp
    from nkpc_hsa.models.hsa_full import func_nkpc_hsa_full  # Particle Gibbs
    from nkpc_hsa.models.hsa_steady import func_nkpc_hsa_decomp_tv_kappa_noerror
    from nkpc_hsa.models.hsa_theory import func_hsa_f0, func_hsa_restricted, func_hsa_u

    funcs: dict[str, Callable[..., Mapping[str, Any]]] = {
        "ces": func_nkpc_ces,
        "hsa_steady": func_nkpc_hsa_decomp_tv_kappa_noerror,
        "hsa_dynamic": func_nkpc_hsa_decomp,
        "hsa_full": func_nkpc_hsa_full,
        # gamma = 0 makes the joint state linear-Gaussian, so const-theta uses the
        # exact joint FFBS rather than hsa_full's alternating conditional blocks.
        "hsa_const_theta": func_nkpc_hsa_const_theta,
        "hsa_f0": func_hsa_f0,
        "hsa_u": func_hsa_u,
        "hsa_r1": func_hsa_restricted,
        "hsa_r2": func_hsa_restricted,
        "hsa_r3": func_hsa_restricted,
    }
    if model not in funcs:
        raise ValueError(f"Unknown model: {model}")

    priors_internal = prior_specs_to_internal(prior_specs)
    constraints_internal = coefficient_constraints_to_internal(coefficient_constraints)
    seed_seq = np.random.SeedSequence(seed)
    child_seeds = seed_seq.spawn(chains)
    chain_draws: list[dict[str, np.ndarray]] = []
    chain_metadata: list[dict[str, Any]] = []

    for chain, child in enumerate(child_seeds):
        chain_seed = int(child.generate_state(1)[0])
        kwargs: dict[str, Any] = {
            "pi_data": model_data["pi"],
            "pi_prev_data": model_data["pi_prev"],
            "Epi_data": model_data["pi_expect"],
            "x_data": model_data["x"],
            "x_prev_data": model_data["x_prev"],
            "n_burn": burn,
            "n_keep": n_iter - burn,
            "priors": priors_internal,
            "opts": {
                "seed": chain_seed,
                "store_every": thin,
                "verbose": False,
                "coefficient_constraints": constraints_internal,
                "enforce_stationary": enforce_stationary,
                "ar2_max_tries": ar2_max_tries,
            },
        }
        if model == "hsa_dynamic":
            kwargs["opts"]["covariance_structure"] = covariance_structure
        if no_inertia:
            if model != "hsa_steady":
                raise ValueError("no_inertia is only implemented for hsa_steady.")
            kwargs["opts"]["no_inertia"] = True
        if model == "hsa_full":
            # Particle count for the conditional-SMC state update. Passed
            # explicitly so the operating point is recorded in run metadata
            # rather than living as a module constant.
            kwargs["opts"]["n_particles"] = int(n_particles)
        if is_theory_model(model):
            kwargs["opts"].update(dict(theory_options or {}))
            kwargs["opts"]["restriction_level"] = theory_model_definition(model).hierarchy
            if model in {"hsa_u", "hsa_r1", "hsa_r2"}:
                kwargs["opts"]["n_particles"] = int(n_particles)
        if model != "ces":
            if "N_obs" in model_data:
                kwargs["N_data"] = np.asarray(model_data["N_obs"], dtype=float)
            elif "N" not in model_data:
                raise KeyError(f"{model} requires an N series.")
            else:
                kwargs["N_data"] = transform_competition_series(model_data["N"], transform=n_transform)  # type: ignore[arg-type]
        if "Ehat" in model_data:
            if model not in {"hsa_steady", "hsa_const_theta"}:
                raise ValueError(
                    "The establishment-cycle observation is currently implemented only for "
                    "hsa_steady and hsa_const_theta."
                )
            kwargs["Ehat_data"] = np.asarray(model_data["Ehat"], dtype=float)
        if "E_obs" in model_data:
            if model not in {"hsa_steady", "hsa_const_theta"}:
                raise ValueError(
                    "The six-state joint N--E system is defined for the linear-Gaussian "
                    "hsa_steady and hsa_const_theta specifications."
                )
            kwargs["E_data"] = np.asarray(model_data["E_obs"], dtype=float)
        result = _call_sampler(funcs[model], kwargs, orth=orth)
        chain_draws.append(_extract_draws_from_result(result))
        chain_metadata.append({"chain": chain, "seed": chain_seed, "model_metadata": result.get("model", {})})
    return _stack_chains(chain_draws), {
        "chains": chain_metadata,
        "priors_internal": priors_internal,
        "coefficient_constraints_internal": constraints_internal,
    }


def run_model(
    model: str,
    *,
    data: pd.DataFrame | Mapping[str, Any] | None = None,
    data_spec: Mapping[str, Any] | str | Path | None = None,
    prior_specs: Mapping[str, Any] | str | Path | None = None,
    prior_name: str = "baseline",
    n_iter: int = 12000,
    burn: int = 4000,
    thin: int = 5,
    chains: int = 2,
    seed: int = 12345,
    orth: bool = False,
    n_transform: str = DEFAULT_N_TRANSFORM,
    period_name: str = "full",
    covariance_structure: str = "e_zeta_only",
    coefficient_constraints: Mapping[str, Any] | None = None,
    competition_measurement: Mapping[str, Any] | None = None,
    enforce_stationary: bool = True,
    ar2_max_tries: int = 2000,
    n_particles: int = 512,
    no_inertia: bool = False,
    run_id: str | None = None,
    run_dir: str | Path | None = None,
    save: bool = True,
    inflation_observation: str | None = None,
    zeta0: float = 6.0,
    zeta0_treatment: str = "fixed",
    require_path_positivity: bool = True,
):
    if isinstance(data_spec, (str, Path)):
        data_spec_dict = _load_yaml(data_spec)
        data_spec_name = Path(data_spec).stem
    else:
        data_spec_dict = dict(data_spec or {})
        data_spec_name = str(data_spec_dict.get("name", "default"))
    if isinstance(prior_specs, (str, Path)):
        prior_dict = _load_yaml(prior_specs)
        prior_name = Path(prior_specs).stem.replace("priors_", "")
    else:
        prior_dict = dict(prior_specs or {})

    competition_spec = normalize_competition_measurement(competition_measurement)
    theory_definition = theory_model_definition(model) if is_theory_model(model) else None
    inflation_observation = str(
        inflation_observation
        or data_spec_dict.get("inflation_observation")
        or ("yoy_4q" if not is_theory_model(model) else "")
    )
    if is_theory_model(model) and not inflation_observation:
        raise ValueError(
            "Theory-model runs require an explicit inflation_observation ('qoq' or 'yoy_4q') "
            "in the data spec or run_model call."
        )
    marginal_cost_loading = 1.0
    if theory_definition is not None:
        if zeta0_treatment != "fixed":
            raise ValueError("This revision implements zeta0_treatment='fixed' only; sampled zeta0 needs a separate model.")
        marginal_cost_loading = validate_theory_run_design(
            model,
            data_spec_dict,
            zeta0=zeta0,
            inflation_observation=inflation_observation,
        )
    model_data = _coerce_model_data(data, data_spec=data_spec_dict)
    sample_start = ""
    sample_end = ""
    sample_index = model_sample_index(data, data_spec_dict)
    if isinstance(sample_index, pd.DatetimeIndex) and len(sample_index):
        sample_start = sample_index.min().date().isoformat()
        sample_end = sample_index.max().date().isoformat()
    elif isinstance(sample_index, pd.PeriodIndex) and len(sample_index):
        sample_start = str(sample_index.min())
        sample_end = str(sample_index.max())
    competition_context = _prepare_competition_measurement(
        model=model,
        data=data,
        data_spec=data_spec_dict,
        model_data=model_data,
        sample_index=sample_index,
        n_transform=n_transform,
        competition_measurement=competition_spec,
    )
    model_data_for_sampler = dict(model_data)
    if competition_context.get("N_obs_used") is not None:
        model_data_for_sampler["N_obs"] = np.asarray(competition_context["N_obs_used"], dtype=float)
    run_id = run_id or _timestamp()
    constraint_spec = _constraint_label(coefficient_constraints)
    if no_inertia:
        # A different estimating equation, not a variant of the same one. Marking it as a
        # restricted spec keeps it out of the report's main run-set, which selects on
        # constraint_spec == "unrestricted".
        constraint_spec = "alpha_zero" if constraint_spec == "unrestricted" else f"{constraint_spec}_alpha_zero"
    metadata = RunMetadata(
        model=model,
        data_spec=data_spec_name,
        prior_spec=prior_name,
        run_id=run_id,
        n_iter=n_iter,
        burn=burn,
        thin=thin,
        chains=chains,
        seed=seed,
        n_transform=n_transform,
        competition_measurement_frequency=competition_spec["frequency"],
        competition_measurement_annual_timing=competition_spec["annual_timing"],
        period=period_name,
        covariance_structure=covariance_structure,
        constraint_spec=constraint_spec,
        coefficient_constraints=dict(coefficient_constraints or {}),
    )
    posterior, extra_meta = _run_sampler(
        model=model,
        model_data=model_data_for_sampler,
        prior_specs=prior_dict,
        n_iter=n_iter,
        burn=burn,
        thin=thin,
        chains=chains,
        seed=seed,
        orth=orth,
        n_transform=n_transform,
        covariance_structure=covariance_structure,
        coefficient_constraints=coefficient_constraints,
        enforce_stationary=enforce_stationary,
        ar2_max_tries=ar2_max_tries,
        n_particles=n_particles,
        no_inertia=no_inertia,
        theory_options={
            "zeta0": float(zeta0),
            "marginal_cost_loading": float(marginal_cost_loading),
            "require_path_positivity": bool(require_path_positivity),
        } if theory_definition is not None else None,
    )
    code_revision, code_dirty = _code_revision()
    N_raw = np.asarray(model_data.get("N", []), dtype=float)
    N0_anchor = None
    if N_raw.size and np.all(np.isfinite(N_raw)) and np.all(N_raw > 0.0):
        N0_anchor = float(np.exp(np.mean(np.log(N_raw))))
    meta = {
        **metadata.__dict__,
        "code_revision": code_revision,
        "code_dirty": code_dirty,
        "orth": orth,
        "n_obs": int(len(model_data["pi"])),
        "sample_start": sample_start,
        "sample_end": sample_end,
        "competition_measurement": {
            **competition_spec,
            **dict(competition_context.get("metadata", {}) or {}),
        },
        "kappa_unit_note": KAPPA_UNIT_NOTE,
        "n_transform_note": competition_transform_note(n_transform),
        "coefficient_constraints": dict(coefficient_constraints or {}),
        "run_priors_json": json.dumps(prior_dict, sort_keys=True),
        "enforce_stationary": enforce_stationary,
        "ar2_max_tries": ar2_max_tries,
        "n_particles": n_particles if model == "hsa_full" else None,
        "no_inertia": no_inertia,
        "establishment_measurement": {
            "enabled": bool("Ehat" in model_data_for_sampler or "E_obs" in model_data_for_sampler),
            "model": data_spec_dict.get("establishment_model"),
            "source_column": data_spec_dict.get("e_col", data_spec_dict.get("e_hat_col")),
            "cycle_filter": data_spec_dict.get("e_cycle_filter") if "Ehat" in model_data_for_sampler else None,
            "hp_lambda": data_spec_dict.get("e_hp_lambda") if "Ehat" in model_data_for_sampler else None,
            "transform": data_spec_dict.get("e_transform") if ("Ehat" in model_data_for_sampler or "E_obs" in model_data_for_sampler) else None,
            "finite_Ehat_count": int(np.isfinite(model_data_for_sampler.get("Ehat", [])).sum()),
            "finite_E_obs_count": int(np.isfinite(model_data_for_sampler.get("E_obs", [])).sum()),
        },
        "extra": extra_meta,
    }
    if theory_definition is not None:
        chain_theory = [
            dict((chain.get("model_metadata", {}) or {}).get("theory_restriction", {}) or {})
            for chain in extra_meta.get("chains", [])
        ]
        state_rejections = sum(int(item.get("state_admissibility_rejections", 0) or 0) for item in chain_theory)
        coefficient_rejections = sum(
            int(item.get("coefficient_admissibility_rejections", 0) or 0)
            for item in chain_theory
        )
        state_proposals = max(1, chains * n_iter)
        meta.update(
            {
                "estimation_revision": THEORY_ESTIMATION_REVISION,
                "model_hierarchy": theory_definition.hierarchy,
                "model_definition": theory_definition.metadata(),
                "restriction_taxonomy": RESTRICTION_TAXONOMY,
                "exact_restrictions": list(theory_definition.exact_restrictions),
                "maintained_laws": {
                    "second_law": theory_definition.second_law,
                    "third_law": theory_definition.third_law,
                },
                "domain_admissibility_conditions": [
                    "zeta0 > 1",
                    "mu0 > 1",
                    "kappa0 > 0",
                    *(
                        ["kappa_t > 0 for every draw/time", "theta_t > 0 for every draw/time"]
                        if require_path_positivity else []
                    ),
                ],
                "moving_reference_approximation": theory_definition.moving_reference,
                "structural_frequency": STRUCTURAL_FREQUENCY,
                "inflation_observation": inflation_observation,
                "inflation_transformation": data_spec_dict.get("inflation_transformation"),
                "expectation_series": data_spec_dict.get("pi_expect_col"),
                "expectation_horizon": data_spec_dict.get("expectation_horizon", "unverified"),
                "expectation_information_date": data_spec_dict.get("expectation_information_date", "unverified"),
                "activity_proxy": data_spec_dict.get("x_col"),
                "activity_mapping": data_spec_dict.get("activity_mapping", "gap_proxy"),
                "marginal_cost_loading": marginal_cost_loading,
                "competition_proxy": data_spec_dict.get("n_col", data_spec_dict.get("N_col")),
                "N0_anchor": N0_anchor,
                "zeta0": float(zeta0),
                "mu0": float(mu_from_zeta(zeta0)),
                "zeta0_treatment": zeta0_treatment,
                "observation_frequency": "quarterly",
                "sampler_type": theory_definition.state_sampler,
                "parameter_sampler": (
                    "positive_gaussian_rejection"
                    if model == "hsa_f0"
                    else "linear_truncated_gaussian_coordinate_gibbs"
                ),
                "chain": list(range(chains)),
                "convergence_status": "pending_diagnostics",
                "data_transformation": {
                    "n_transform": n_transform,
                    "inflation": data_spec_dict.get("inflation_transformation"),
                    "expectations": data_spec_dict.get("expectation_transformation", "as_configured"),
                    "activity": data_spec_dict.get("activity_transformation", "as_processed"),
                },
                "historical_compatibility": False,
                "admissibility_sampling": {
                    "state_path_rejections": state_rejections,
                    "state_path_proposals": state_proposals,
                    "state_path_rejection_share": state_rejections / state_proposals,
                    "coefficient_rejections": coefficient_rejections,
                    "interpretation": (
                        "Frequent positivity rejection can indicate that the local linear approximation spans too wide an N range."
                    ),
                },
            }
        )
        meta = stamp_artifact_metadata(meta)
    idata = _to_idata(posterior, meta)
    if save:
        target = (
            Path(run_dir)
            if run_dir is not None
            else _default_run_dir(
                model,
                data_spec_name,
                prior_name,
                constraint_spec,
                run_id,
                competition_frequency=competition_spec["frequency"],
            )
        )
        data_spec_saved = {
            **data_spec_dict,
            "competition_measurement": competition_spec,
        }
        _save_run(idata=idata, run_dir=target, metadata=meta, prior_specs=prior_dict, data_spec=data_spec_saved)
        _write_run_data_model_artifacts(
            idata=idata,
            run_dir=target,
            metadata=meta,
            prior_specs=prior_dict,
            data_spec=data_spec_saved,
            competition_context=competition_context,
        )
    return idata


def run_ces(**kwargs):
    return run_model("ces", **kwargs)


def run_hsa_steady(**kwargs):
    return run_model("hsa_steady", **kwargs)


def run_hsa_dynamic(**kwargs):
    return run_model("hsa_dynamic", **kwargs)


def run_hsa_full(**kwargs):
    return run_model("hsa_full", **kwargs)


def run_hsa_const_theta(**kwargs):
    return run_model("hsa_const_theta", **kwargs)
