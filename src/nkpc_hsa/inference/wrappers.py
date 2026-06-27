from __future__ import annotations

import inspect
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
import pandas as pd
import yaml

from nkpc_hsa.data.load import load_processed_dataset
from nkpc_hsa.data.transforms import DEFAULT_N_TRANSFORM, competition_transform_note, transform_competition_series
from nkpc_hsa.models.common import (
    KAPPA_SCALE,
    KAPPA_UNIT_NOTE,
    coefficient_constraints_to_internal,
    prior_specs_to_internal,
)
from nkpc_hsa.paths import project_path


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
    period: str = "full"
    covariance_structure: str = "e_zeta_only"
    constraint_spec: str = "unrestricted"
    coefficient_constraints: Mapping[str, Any] | None = None
    kappa_scale: float = KAPPA_SCALE
    kappa_units: str = "physical"


def _load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


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


def _default_run_dir(model: str, data_spec: str, prior_spec: str, constraint_spec: str, run_id: str) -> Path:
    parts = [model, data_spec, prior_spec]
    if constraint_spec != "unrestricted":
        parts.append(constraint_spec)
    parts.append(run_id)
    safe = "_".join(part.replace("/", "-") for part in parts)
    return project_path("results", "runs", safe)


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
        required = ["pi", "pi_prev", "pi_expect", "x", "x_prev"]
        if cols["N"] in data.columns:
            required.append("N")
        missing = [cols[k] for k in required if cols[k] not in data.columns]
        if missing:
            raise KeyError(f"Missing model-ready columns: {missing}")
        sample = data[[cols[k] for k in required]].dropna()
        return {k: sample[cols[k]].to_numpy(dtype=float) for k in required}
    return {k: np.asarray(v, dtype=float).reshape(-1) for k, v in data.items()}


def _model_sample_index(data: pd.DataFrame | Mapping[str, Any] | None, data_spec: Mapping[str, Any]) -> pd.Index | None:
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
    required = ["pi", "pi_prev", "pi_expect", "x", "x_prev"]
    if cols["N"] in data.columns:
        required.append("N")
    missing = [cols[k] for k in required if cols[k] not in data.columns]
    if missing:
        return None
    return data[[cols[k] for k in required]].dropna().index


def _extract_draws_from_result(result: Mapping[str, Any]) -> dict[str, np.ndarray]:
    draws: dict[str, np.ndarray] = {}
    for key, value in result.items():
        if key in {"priors", "opts", "model", "deprecated"}:
            continue
        if key == "state_draws":
            for state_key, state_value in value.items():
                draws[state_key] = np.asarray(state_value, dtype=float)
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
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    from nkpc_hsa.models.ces import func_nkpc_ces
    from nkpc_hsa.models.hsa_dynamic import func_nkpc_hsa_decomp
    from nkpc_hsa.models.hsa_full import func_nkpc_hsa_full
    from nkpc_hsa.models.hsa_steady import func_nkpc_hsa_decomp_tv_kappa_noerror

    funcs: dict[str, Callable[..., Mapping[str, Any]]] = {
        "ces": func_nkpc_ces,
        "hsa_steady": func_nkpc_hsa_decomp_tv_kappa_noerror,
        "hsa_dynamic": func_nkpc_hsa_decomp,
        "hsa_full": func_nkpc_hsa_full,
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
        if model != "ces":
            if "N" not in model_data:
                raise KeyError(f"{model} requires an N series.")
            kwargs["N_data"] = transform_competition_series(model_data["N"], transform=n_transform)  # type: ignore[arg-type]
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
    enforce_stationary: bool = True,
    ar2_max_tries: int = 2000,
    run_id: str | None = None,
    run_dir: str | Path | None = None,
    save: bool = True,
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

    model_data = _coerce_model_data(data, data_spec=data_spec_dict)
    sample_start = ""
    sample_end = ""
    sample_index = _model_sample_index(data, data_spec_dict)
    if isinstance(sample_index, pd.DatetimeIndex) and len(sample_index):
        sample_start = sample_index.min().date().isoformat()
        sample_end = sample_index.max().date().isoformat()
    run_id = run_id or _timestamp()
    constraint_spec = _constraint_label(coefficient_constraints)
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
        period=period_name,
        covariance_structure=covariance_structure,
        constraint_spec=constraint_spec,
        coefficient_constraints=dict(coefficient_constraints or {}),
    )
    posterior, extra_meta = _run_sampler(
        model=model,
        model_data=model_data,
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
    )
    meta = {
        **metadata.__dict__,
        "orth": orth,
        "n_obs": int(len(model_data["pi"])),
        "sample_start": sample_start,
        "sample_end": sample_end,
        "kappa_unit_note": KAPPA_UNIT_NOTE,
        "n_transform_note": competition_transform_note(n_transform),
        "coefficient_constraints": dict(coefficient_constraints or {}),
        "enforce_stationary": enforce_stationary,
        "ar2_max_tries": ar2_max_tries,
        "extra": extra_meta,
    }
    idata = _to_idata(posterior, meta)
    if save:
        target = (
            Path(run_dir)
            if run_dir is not None
            else _default_run_dir(model, data_spec_name, prior_name, constraint_spec, run_id)
        )
        _save_run(idata=idata, run_dir=target, metadata=meta, prior_specs=prior_dict, data_spec=data_spec_dict)
    return idata


def run_ces(**kwargs):
    return run_model("ces", **kwargs)


def run_hsa_steady(**kwargs):
    return run_model("hsa_steady", **kwargs)


def run_hsa_dynamic(**kwargs):
    return run_model("hsa_dynamic", **kwargs)


def run_hsa_full(**kwargs):
    return run_model("hsa_full", **kwargs)
