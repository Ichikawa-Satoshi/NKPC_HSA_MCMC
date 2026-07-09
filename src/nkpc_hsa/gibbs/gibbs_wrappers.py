from __future__ import annotations

import inspect
from typing import Any, Mapping, Optional

import numpy as np

KAPPA_SCALE = 100.0

try:
    from .ces import func_nkpc_ces
    from .hsa_dynamic import func_nkpc_hsa_decomp
    from .hsa_steady import func_nkpc_hsa_decomp_tv_kappa_noerror, func_nkpc_hsa_decomp_tv_theta_kappa
    from .hsa_full import func_nkpc_hsa_full
except ImportError:  # pragma: no cover
    from nkpc_hsa.gibbs.ces import func_nkpc_ces
    from nkpc_hsa.gibbs.hsa_dynamic import func_nkpc_hsa_decomp
    from nkpc_hsa.gibbs.hsa_steady import func_nkpc_hsa_decomp_tv_kappa_noerror, func_nkpc_hsa_decomp_tv_theta_kappa
    from nkpc_hsa.gibbs.hsa_full import func_nkpc_hsa_full

def _prior_specs_to_dict(prior_specs: Optional[Mapping[str, tuple[float, float]]]) -> dict[str, float]:
    """Convert physical-unit kappa priors to sampler-internal units."""
    specs = dict(prior_specs or {})
    out: dict[str, float] = {}
    if "alpha" in specs:
        out["mu_alpha"], out["sigma_alpha"] = specs["alpha"]
    if "kappa" in specs:
        mu, sigma = specs["kappa"]
        out["mu_kappa"], out["sigma_kappa"] = KAPPA_SCALE * mu, KAPPA_SCALE * sigma
    if "kappa_0" in specs:
        mu, sigma = specs["kappa_0"]
        out["mu_kappa_0"], out["sigma_kappa_0"] = KAPPA_SCALE * mu, KAPPA_SCALE * sigma
    if "delta" in specs:
        mu, sigma = specs["delta"]
        out["mu_delta"], out["sigma_delta"] = KAPPA_SCALE * mu, KAPPA_SCALE * sigma
    if "theta" in specs:
        out["mu_theta"], out["sigma_theta"] = specs["theta"]
    if "phi_1" in specs:
        out["mu_phi_1"], out["sigma_phi_1"] = specs["phi_1"]
    if "rho_1" in specs:
        out["mu_rho1"], out["sigma_rho1"] = specs["rho_1"]
    if "rho_2" in specs:
        out["mu_rho2"], out["sigma_rho2"] = specs["rho_2"]
    if "rho" in specs:
        out["mu_lambda"], out["sigma_lambda"] = specs["rho"]
    return out


def _transform_N_series(N: np.ndarray) -> np.ndarray:
    arr = np.asarray(N, dtype=float)
    if np.any(~np.isfinite(arr)) or np.any(arr <= 0.0):
        raise ValueError("N must be finite and strictly positive before applying log(N) * 100.")
    return 100.0 * np.log(arr)


def _extract_draws_from_result(result: Mapping[str, Any]) -> dict[str, np.ndarray]:
    draws: dict[str, np.ndarray] = {}
    for key, value in result.items():
        if key in {"priors", "opts"}:
            continue
        if key == "state_draws":
            for state_key, state_value in value.items():
                draws[state_key] = np.asarray(state_value, dtype=float)
            continue
        if isinstance(value, dict) and "draws" in value:
            draws[key] = np.asarray(value["draws"], dtype=float)
    return draws


def run_ces(
    *,
    pi: np.ndarray,
    pi_prev: np.ndarray,
    pi_expect: np.ndarray,
    x: np.ndarray,
    x_prev: np.ndarray,
    prior_specs: Optional[Mapping[str, tuple[float, float]]] = None,
    n_iter: int = 12000,
    burn: int = 4000,
    thin: int = 5,
    rng: np.random.Generator | None = None,
    orth: bool = False,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(0) if rng is None else rng
    result = func_nkpc_ces(
        pi_data=pi,
        pi_prev_data=pi_prev,
        Epi_data=pi_expect,
        x_data=x,
        x_prev_data=x_prev,
        n_burn=burn,
        n_keep=n_iter - burn,
        priors=_prior_specs_to_dict(prior_specs),
        opts={"seed": int(rng.integers(0, 2**31 - 1)), "store_every": thin, "verbose": False},
        orth=orth,
    )
    draws = _extract_draws_from_result(result)
    if "sigma_e2" in draws:
        draws["sigma_e"] = np.sqrt(draws.pop("sigma_e2"))
    if "sigma_zeta2" in draws:
        draws["sigma_zeta"] = np.sqrt(draws.pop("sigma_zeta2"))
    return draws


def run_hsa_dynamic(
    *,
    pi: np.ndarray,
    pi_prev: np.ndarray,
    pi_expect: np.ndarray,
    x: np.ndarray,
    x_prev: np.ndarray,
    N: np.ndarray,
    prior_specs: Optional[Mapping[str, tuple[float, float]]] = None,
    n_iter: int = 12000,
    burn: int = 4000,
    thin: int = 5,
    rng: np.random.Generator | None = None,
    orth: bool = False,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(0) if rng is None else rng
    kwargs = {
        "pi_data": pi,
        "pi_prev_data": pi_prev,
        "Epi_data": pi_expect,
        "x_data": x,
        "x_prev_data": x_prev,
        "N_data": _transform_N_series(N),
        "n_burn": burn,
        "n_keep": n_iter - burn,
        "priors": _prior_specs_to_dict(prior_specs),
        "opts": {"seed": int(rng.integers(0, 2**31 - 1)), "store_every": thin, "verbose": False},
    }
    if "orth" in inspect.signature(func_nkpc_hsa_decomp).parameters:
        kwargs["orth"] = orth
    result = func_nkpc_hsa_decomp(**kwargs)
    draws = _extract_draws_from_result(result)
    draws["rho_1"] = draws.pop("rho1")
    draws["rho_2"] = draws.pop("rho2")
    return draws


def run_hsa_steady(
    *,
    pi: np.ndarray,
    pi_prev: np.ndarray,
    pi_expect: np.ndarray,
    x: np.ndarray,
    x_prev: np.ndarray,
    N: np.ndarray,
    prior_specs: Optional[Mapping[str, tuple[float, float]]] = None,
    n_iter: int = 12000,
    burn: int = 4000,
    thin: int = 5,
    rng: np.random.Generator | None = None,
    orth: bool = False,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(0) if rng is None else rng
    result = func_nkpc_hsa_decomp_tv_kappa_noerror(
        pi_data=pi,
        pi_prev_data=pi_prev,
        Epi_data=pi_expect,
        x_data=x,
        x_prev_data=x_prev,
        N_data=_transform_N_series(N),
        n_burn=burn,
        n_keep=n_iter - burn,
        priors=_prior_specs_to_dict(prior_specs),
        opts={"seed": int(rng.integers(0, 2**31 - 1)), "store_every": thin, "verbose": False},
        orth=orth,
    )
    draws = _extract_draws_from_result(result)
    draws["rho_1"] = draws.pop("rho1")
    draws["rho_2"] = draws.pop("rho2")
    return draws


def run_hsa_steady_tv(
    *,
    pi: np.ndarray,
    pi_prev: np.ndarray,
    pi_expect: np.ndarray,
    x: np.ndarray,
    x_prev: np.ndarray,
    N: np.ndarray,
    prior_specs: Optional[Mapping[str, tuple[float, float]]] = None,
    n_iter: int = 12000,
    burn: int = 4000,
    thin: int = 5,
    rng: np.random.Generator | None = None,
    orth: bool = False,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(0) if rng is None else rng
    result = func_nkpc_hsa_decomp_tv_theta_kappa(
        pi_data=pi,
        pi_prev_data=pi_prev,
        Epi_data=pi_expect,
        x_data=x,
        x_prev_data=x_prev,
        N_data=_transform_N_series(N),
        n_burn=burn,
        n_keep=n_iter - burn,
        priors=_prior_specs_to_dict(prior_specs),
        opts={"seed": int(rng.integers(0, 2**31 - 1)), "store_every": thin, "verbose": False},
        orth=orth,
    )
    draws = _extract_draws_from_result(result)
    draws["rho_1"] = draws.pop("rho1")
    draws["rho_2"] = draws.pop("rho2")
    return draws


def run_hsa_full(
    *,
    pi: np.ndarray,
    pi_prev: np.ndarray,
    pi_expect: np.ndarray,
    x: np.ndarray,
    x_prev: np.ndarray,
    N: np.ndarray,
    prior_specs: Optional[Mapping[str, tuple[float, float]]] = None,
    n_iter: int = 12000,
    burn: int = 4000,
    thin: int = 5,
    rng: np.random.Generator | None = None,
    orth: bool = False,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(0) if rng is None else rng
    result = func_nkpc_hsa_full(
        pi_data=pi,
        pi_prev_data=pi_prev,
        Epi_data=pi_expect,
        x_data=x,
        x_prev_data=x_prev,
        N_data=_transform_N_series(N),
        n_burn=burn,
        n_keep=n_iter - burn,
        priors=_prior_specs_to_dict(prior_specs),
        opts={"seed": int(rng.integers(0, 2**31 - 1)), "store_every": thin, "verbose": False},
        orth=orth,
    )
    draws = _extract_draws_from_result(result)
    draws["rho_1"] = draws.pop("rho1")
    draws["rho_2"] = draws.pop("rho2")
    return draws


def draws_to_idata(draws: Mapping[str, np.ndarray]):
    from types import SimpleNamespace

    import xarray as xr

    posterior = {}
    for key, value in draws.items():
        arr = np.asarray(value)
        if arr.ndim == 1:
            dims = ("chain", "draw")
            data = arr[None, :]
        else:
            dims = ("chain", "draw") + tuple(f"{key}_dim_{i}" for i in range(arr.ndim - 1))
            data = arr[None, ...]
        posterior[key] = (dims, data)
    posterior_ds = xr.Dataset(posterior)
    return SimpleNamespace(posterior=posterior_ds)
