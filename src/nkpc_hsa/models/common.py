from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

KAPPA_SCALE = 100.0
KAPPA_LIKE_PARAMETERS = {"kappa", "kappa_0", "kappa_t", "delta"}
CONSTRAINT_ALIASES = {
    "kappa0": "kappa_0",
    "kappat": "kappa_t",
    "theta0": "theta_0",
    "rho1": "rho_1",
    "rho2": "rho_2",
}
KAPPA_UNIT_NOTE = (
    "Priors are specified in physical units. Samplers may use internal kappa "
    "parameters scaled by KAPPA_SCALE when the regressor is divided by KAPPA_SCALE. "
    "Stored posterior draws and report outputs are in physical units."
)


@dataclass(frozen=True)
class NormalPrior:
    mean: float
    sd: float
    units: str = "physical"


def getd(d: Mapping[str, Any] | None, key: str, default: Any) -> Any:
    if isinstance(d, Mapping) and key in d and d[key] is not None:
        return d[key]
    return default


def as_1d(x: Any) -> np.ndarray:
    return np.asarray(x, dtype=float).reshape(-1)


def assert_all_pos(arr: Any, msg: str) -> None:
    values = np.asarray(arr, dtype=float)
    if np.any(~np.isfinite(values)) or np.any(values <= 0.0):
        raise ValueError(msg)


def force_symmetric_positive_definite(S: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    """Symmetrize and repair a covariance matrix using eigh, not eig."""
    S = np.asarray(S, dtype=float)
    S = (S + S.T) / 2.0
    vals, vecs = np.linalg.eigh(S)
    vals = np.maximum(vals, eps)
    repaired = (vecs * vals) @ vecs.T
    return (repaired + repaired.T) / 2.0


def safe_multivariate_normal(
    mean: np.ndarray,
    cov: np.ndarray,
    rng: np.random.Generator,
    *,
    eps: float = 1e-10,
) -> np.ndarray:
    cov_pd = force_symmetric_positive_definite(cov, eps=eps)
    return rng.multivariate_normal(np.asarray(mean, dtype=float), cov_pd)


def sample_invgamma(a_post: float, b_post: float, rng: np.random.Generator) -> float:
    if a_post <= 0.0 or b_post <= 0.0:
        raise ValueError("Inverse-gamma posterior parameters must be positive.")
    return 1.0 / rng.gamma(shape=float(a_post), scale=1.0 / float(b_post))


def is_stationary_ar2(r1: float, r2: float) -> bool:
    return (abs(r2) < 1.0) and ((r1 + r2) < 1.0) and ((r2 - r1) < 1.0)


def kappa_physical_to_internal(value: float | np.ndarray) -> float | np.ndarray:
    return np.asarray(value) * KAPPA_SCALE


def kappa_internal_to_physical(value: float | np.ndarray) -> float | np.ndarray:
    return np.asarray(value) / KAPPA_SCALE


def kappa_prior_physical_to_internal(mean: float, sd: float) -> tuple[float, float]:
    return float(mean * KAPPA_SCALE), float(sd * KAPPA_SCALE)


def _canonical_constraint_name(name: str) -> str:
    key = str(name).strip()
    return CONSTRAINT_ALIASES.get(key, key)


def _coerce_bound_pair(value: Any) -> tuple[float | None, float | None]:
    if isinstance(value, Mapping):
        lo = value.get("lower", value.get("min"))
        hi = value.get("upper", value.get("max"))
    else:
        if len(value) != 2:
            raise ValueError("Coefficient constraint bounds must contain lower and upper values.")
        lo, hi = value
    lower = None if lo is None else float(lo)
    upper = None if hi is None else float(hi)
    if lower is not None and upper is not None and lower > upper:
        raise ValueError("Coefficient constraint lower bound cannot exceed upper bound.")
    return lower, upper


def coefficient_constraints_to_internal(spec: Mapping[str, Any] | None) -> dict[str, Any]:
    """Convert coefficient hard constraints from physical to sampler units.

    Supported user-facing YAML shape:

    ``enabled: true``
    ``positive: [kappa, theta]``
    ``bounds: {alpha: [0, 1], kappa_0: [0, null]}``

    Kappa-like bounds are specified in physical units and converted to the
    internal KAPPA_SCALE-multiplied units used by the Gibbs regression blocks.
    ``kappa_t`` is a path constraint for HSA steady/full models.
    Other coefficients are left unchanged.
    """
    raw = dict(spec or {})
    enabled = bool(raw.get("enabled", bool(raw.get("positive") or raw.get("nonnegative") or raw.get("bounds"))))
    out: dict[str, Any] = {
        "enabled": enabled,
        "max_tries": int(raw.get("max_tries", 1000)),
        "bounds": {},
    }
    if out["max_tries"] <= 0:
        raise ValueError("coefficient_constraints.max_tries must be positive.")
    if not enabled:
        return out

    bounds: dict[str, tuple[float | None, float | None]] = {}
    for key in raw.get("positive", []) or []:
        bounds[_canonical_constraint_name(key)] = (0.0, None)
    for key in raw.get("nonnegative", []) or []:
        bounds[_canonical_constraint_name(key)] = (0.0, None)
    for key, value in dict(raw.get("bounds", {}) or {}).items():
        bounds[_canonical_constraint_name(key)] = _coerce_bound_pair(value)

    converted: dict[str, list[float | None]] = {}
    for key, (lower, upper) in bounds.items():
        if key in KAPPA_LIKE_PARAMETERS:
            lower = None if lower is None else float(kappa_physical_to_internal(lower))
            upper = None if upper is None else float(kappa_physical_to_internal(upper))
        converted[key] = [lower, upper]
    out["bounds"] = converted
    return out


def prior_specs_to_internal(prior_specs: Mapping[str, Any] | None) -> dict[str, Any]:
    """Convert user-facing physical-unit priors to sampler hyperparameters."""
    specs = dict(prior_specs or {})
    out: dict[str, Any] = {}

    def pair(name: str) -> tuple[float, float] | None:
        if name not in specs:
            return None
        value = specs[name]
        if isinstance(value, Mapping):
            return float(value["mean"]), float(value["sd"])
        return float(value[0]), float(value[1])

    for key, target in [
        ("alpha", ("mu_alpha", "sigma_alpha")),
        ("theta", ("mu_theta", "sigma_theta")),
        ("theta_0", ("mu_theta", "sigma_theta")),
        ("gamma", ("mu_gamma", "sigma_gamma")),
        ("phi_1", ("mu_phi_1", "sigma_phi_1")),
        ("rho_1", ("mu_rho1", "sigma_rho1")),
        ("rho_2", ("mu_rho2", "sigma_rho2")),
        ("lambda_ez", ("mu_lambda", "sigma_lambda")),
        ("rho", ("mu_lambda", "sigma_lambda")),
        ("n", ("mu_n", "sigma_n")),
    ]:
        p = pair(key)
        if p is not None:
            out[target[0]], out[target[1]] = p

    for key, target in [
        ("kappa", ("mu_kappa", "sigma_kappa")),
        ("kappa_0", ("mu_kappa_0", "sigma_kappa_0")),
        ("delta", ("mu_delta", "sigma_delta")),
    ]:
        p = pair(key)
        if p is not None:
            out[target[0]], out[target[1]] = kappa_prior_physical_to_internal(*p)

    for key in ["a_e", "b_e", "a_z", "b_z", "a_u", "b_u", "a_eps", "b_eps", "a_N", "b_N", "nu_Sigma"]:
        if key in specs:
            out[key] = float(specs[key])
    if "S_Sigma" in specs:
        out["S_Sigma"] = specs["S_Sigma"]
    return out


def sample_beta_gaussian(
    y: np.ndarray,
    X: np.ndarray,
    sigma2: float,
    prior_mean: np.ndarray,
    prior_var: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    y = as_1d(y)
    X = np.asarray(X, dtype=float)
    prior_mean = as_1d(prior_mean)
    prior_var = as_1d(prior_var)
    assert_all_pos(prior_var, "Prior variances must be positive.")
    assert_all_pos([sigma2], "sigma2 must be positive.")
    V0_inv = np.diag(1.0 / prior_var)
    precision = X.T @ X / sigma2 + V0_inv
    rhs = X.T @ y / sigma2 + V0_inv @ prior_mean
    cov = np.linalg.inv(precision)
    mean = np.linalg.solve(precision, rhs)
    return safe_multivariate_normal(mean, cov, rng)
