"""Run-model entry point for the MA(3) error-structure specifications.

``nkpc_hsa.inference.wrappers.run_model`` resolves its sampler from a dict that
is local to ``_run_sampler``, so an MA(3) model cannot be registered from
outside without editing that function. This module therefore carries its own
orchestration -- but only the orchestration. Every piece of real work (data
coercion, competition-measurement handling, prior translation, chain stacking,
InferenceData assembly, the on-disk run layout) is imported from the production
wrapper and reused unchanged, so an error_robustness run directory has exactly
the same shape and semantics as a production one.

Outputs go under ``results/error_robustness/runs/`` with the *same* directory
names production would use, so the two trees can be diffed cell by cell. Model
names stay ``ces`` / ``hsa_steady`` / ``hsa_dynamic`` / ``hsa_full`` for the same
reason: report tooling that keys on the model name keeps working. The error
structure is recorded in ``metadata.json`` under ``error_structure``, not in the
name.

``hsa_full`` dispatches to the Particle Gibbs sampler, matching production's own
routing of that name to ``hsa_full_pg`` rather than the alternating-block
sampler in ``gibbs.hsa_full``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
import pandas as pd

from nkpc_hsa.dataprep.competition import normalize_competition_measurement
from nkpc_hsa.dataprep.transforms import (
    DEFAULT_N_TRANSFORM,
    competition_transform_note,
    transform_competition_series,
)
from nkpc_hsa.error_robustness.ces_ma3 import func_nkpc_ces_ma3
from nkpc_hsa.error_robustness.hsa_dynamic_ma3 import func_nkpc_hsa_dynamic_ma3
from nkpc_hsa.error_robustness.hsa_full_ma3 import func_nkpc_hsa_full_ma3
from nkpc_hsa.error_robustness.hsa_steady_ma3 import func_nkpc_hsa_steady_ma3
from nkpc_hsa.error_robustness.ma_error import MA_ORDER
from nkpc_hsa.inference.wrappers import (
    KAPPA_UNIT_NOTE,
    RunMetadata,
    _call_sampler,
    _coerce_model_data,
    _constraint_label,
    _extract_draws_from_result,
    _load_yaml,
    _prepare_competition_measurement,
    _save_run,
    _stack_chains,
    _timestamp,
    _to_idata,
    _write_run_data_model_artifacts,
    coefficient_constraints_to_internal,
    model_sample_index,
    prior_specs_to_internal,
)
from nkpc_hsa.paths import results_root

__all__ = ["ERROR_ROBUSTNESS_RUNS", "MA3_MODELS", "run_model_ma3"]

ERROR_ROBUSTNESS_RUNS = results_root() / "error_robustness" / "runs"

MA3_MODELS: dict[str, Callable[..., Mapping[str, Any]]] = {
    "ces": func_nkpc_ces_ma3,
    "hsa_steady": func_nkpc_hsa_steady_ma3,
    "hsa_dynamic": func_nkpc_hsa_dynamic_ma3,
    # hsa_full uses the Particle Gibbs state update, matching production's
    # dispatch of "hsa_full" to hsa_full_pg rather than the alternating-block
    # sampler in gibbs.hsa_full.
    "hsa_full": func_nkpc_hsa_full_ma3,
}


def _run_dir_name(
    model: str,
    data_spec: str,
    prior_spec: str,
    constraint_spec: str,
    *,
    competition_frequency: str,
) -> str:
    """Run directory name: one directory per cell, re-estimated in place.

    Mirrors ``wrappers._default_run_dir`` except that the run id is left out, so
    a re-estimated cell overwrites rather than accumulating. That is the
    convention CLAUDE.md documents and the one ``results/runs/`` actually
    follows on disk; note that ``_default_run_dir`` itself still appends a
    timestamp, so production and its documented convention currently disagree.
    The estimation time is preserved in ``metadata.json`` as ``run_id``.
    """
    parts = [model, data_spec, prior_spec]
    if constraint_spec != "unrestricted":
        parts.append(constraint_spec)
    parts.append(competition_frequency)
    return "_".join(part.replace("/", "-") for part in parts)


def _run_sampler_ma3(
    *,
    model: str,
    model_data: Mapping[str, Any],
    prior_specs: Mapping[str, Any],
    n_iter: int,
    burn: int,
    thin: int,
    chains: int,
    seed: int,
    orth: bool,
    n_transform: str,
    coefficient_constraints: Mapping[str, Any] | None,
    enforce_stationary: bool,
    ar2_max_tries: int,
    no_inertia: bool,
    covariance_structure: str,
    n_particles: int,
    ma_order: int,
    n_psi_steps: int,
    psi_init_scale: float,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Mirror of ``wrappers._run_sampler`` with the MA(3) dispatch and options."""
    if model not in MA3_MODELS:
        raise ValueError(f"Unknown MA(3) model: {model}. Available: {sorted(MA3_MODELS)}.")

    priors_internal = prior_specs_to_internal(prior_specs)
    constraints_internal = coefficient_constraints_to_internal(coefficient_constraints)
    child_seeds = np.random.SeedSequence(seed).spawn(chains)
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
                "ma_order": int(ma_order),
                "n_psi_steps": int(n_psi_steps),
                "psi_init_scale": float(psi_init_scale),
            },
        }
        if model == "hsa_dynamic":
            kwargs["opts"]["covariance_structure"] = covariance_structure
        if model == "hsa_full":
            kwargs["opts"]["n_particles"] = int(n_particles)
        if no_inertia:
            if model != "hsa_steady":
                raise ValueError("no_inertia is only implemented for hsa_steady.")
            kwargs["opts"]["no_inertia"] = True
        if model != "ces":
            if "N_obs" in model_data:
                kwargs["N_data"] = np.asarray(model_data["N_obs"], dtype=float)
            elif "N" not in model_data:
                raise KeyError(f"{model} requires an N series.")
            else:
                kwargs["N_data"] = transform_competition_series(
                    model_data["N"], transform=n_transform
                )
        result = _call_sampler(MA3_MODELS[model], kwargs, orth=orth)
        chain_draws.append(_extract_draws_from_result(result))
        chain_metadata.append(
            {
                "chain": chain,
                "seed": chain_seed,
                "model_metadata": result.get("model", {}),
                "error_structure": result.get("error_structure", {}),
            }
        )

    return _stack_chains(chain_draws), {
        "chains": chain_metadata,
        "priors_internal": priors_internal,
        "coefficient_constraints_internal": constraints_internal,
    }


def run_model_ma3(
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
    coefficient_constraints: Mapping[str, Any] | None = None,
    competition_measurement: Mapping[str, Any] | None = None,
    enforce_stationary: bool = True,
    ar2_max_tries: int = 2000,
    no_inertia: bool = False,
    covariance_structure: str = "e_zeta_only",
    n_particles: int = 512,
    ma_order: int = MA_ORDER,
    n_psi_steps: int = 2,
    psi_init_scale: float = 0.08,
    run_id: str | None = None,
    run_dir: str | Path | None = None,
    runs_root: str | Path | None = None,
    save: bool = True,
):
    """Estimate an MA(q) error-structure specification and write a run directory.

    Signature-compatible with ``wrappers.run_model`` apart from the added
    ``ma_order`` / ``n_psi_steps`` / ``psi_init_scale`` options.
    """
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
        model_data_for_sampler["N_obs"] = np.asarray(
            competition_context["N_obs_used"], dtype=float
        )

    run_id = run_id or _timestamp()
    constraint_spec = _constraint_label(coefficient_constraints)
    if no_inertia:
        constraint_spec = (
            "alpha_zero" if constraint_spec == "unrestricted" else f"{constraint_spec}_alpha_zero"
        )

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
        # Only hsa_dynamic has a shock covariance to restrict; the others carry
        # a scalar disturbance and record "n/a" for schema compatibility.
        covariance_structure=(
            covariance_structure if model == "hsa_dynamic" else "n/a"
        ),
        constraint_spec=constraint_spec,
        coefficient_constraints=dict(coefficient_constraints or {}),
    )

    posterior, extra_meta = _run_sampler_ma3(
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
        coefficient_constraints=coefficient_constraints,
        enforce_stationary=enforce_stationary,
        ar2_max_tries=ar2_max_tries,
        no_inertia=no_inertia,
        covariance_structure=covariance_structure,
        n_particles=n_particles,
        ma_order=ma_order,
        n_psi_steps=n_psi_steps,
        psi_init_scale=psi_init_scale,
    )

    meta = {
        **metadata.__dict__,
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
        "no_inertia": no_inertia,
        "n_particles": n_particles if model == "hsa_full" else None,
        "error_structure": {
            "family": "ma" if ma_order else "iid",
            "order": int(ma_order),
            "n_psi_steps": int(n_psi_steps),
            "psi_init_scale": float(psi_init_scale),
            "note": (
                "Inflation disturbance e_t = lambda_ez*zeta_t + psi(L)v_t. "
                "psi is drawn by random-walk Metropolis and is the last block, "
                "so Chib's final ordinate factor needs only a numerical "
                "normalisation over the invertible region."
            ),
        },
        "extra": extra_meta,
    }

    idata = _to_idata(posterior, meta)
    if save:
        if run_dir is not None:
            target = Path(run_dir)
        else:
            target = Path(runs_root or ERROR_ROBUSTNESS_RUNS) / _run_dir_name(
                model,
                data_spec_name,
                prior_name,
                constraint_spec,
                competition_frequency=competition_spec["frequency"],
            )

        data_spec_saved = {**data_spec_dict, "competition_measurement": competition_spec}
        _save_run(
            idata=idata,
            run_dir=target,
            metadata=meta,
            prior_specs=prior_dict,
            data_spec=data_spec_saved,
        )
        _write_run_data_model_artifacts(
            idata=idata,
            run_dir=target,
            metadata=meta,
            prior_specs=prior_dict,
            data_spec=data_spec_saved,
            competition_context=competition_context,
        )
    return idata
