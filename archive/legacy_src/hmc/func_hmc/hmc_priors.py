from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpyro.distributions as dist


@dataclass(frozen=True)
class PriorFamily:
    specs: dict[str, tuple[float, float]]
    distributions: dict[str, Any]


_PRIOR_FAMILIES: dict[str, PriorFamily] = {}


def register_prior_family(name: str, *, specs: dict[str, tuple[float, float]], distributions: dict[str, Any]) -> None:
    _PRIOR_FAMILIES[name] = PriorFamily(specs=dict(specs), distributions=dict(distributions))


def get_prior_family(name: str) -> PriorFamily:
    try:
        return _PRIOR_FAMILIES[name]
    except KeyError as exc:
        raise KeyError(f"Unknown prior family: {name}") from exc


def get_prior_specs(name: str) -> dict[str, tuple[float, float]]:
    return get_prior_family(name).specs


def get_prior_distributions(name: str) -> dict[str, Any]:
    return get_prior_family(name).distributions


register_prior_family(
    "ces",
    specs={
        "alpha": (0.5, 0.2),
        "kappa": (0.1, 0.2),
        "phi_1": (0.7, 0.2),
    },
    distributions={
        "alpha": dist.Normal(0.5, 0.2),
        "kappa": dist.Normal(0.1, 0.2),
        "kappa0": dist.Normal(0.1, 0.2),
        "theta": dist.Normal(0.1, 0.2),
        "delta": dist.Normal(0.1, 0.2),
        "gamma": dist.Normal(0.0, 0.2),
        "beta": dist.Normal(0.1, 0.2),
        "phi_1": dist.Normal(0.7, 0.2),
        "phi_2": dist.Normal(0.2, 0.2),
        "phi_3": dist.Normal(0.2, 0.2),
        "n": dist.Normal(0.0, 1.0),
        "sigma_u": dist.InverseGamma(0.001, 0.001),
        "sigma_eps": dist.InverseGamma(0.001, 0.001),
        "sigma_v": dist.InverseGamma(0.001, 0.001),
        "sigma_mu": dist.InverseGamma(0.001, 0.001),
        "sigma_e": dist.InverseGamma(0.001, 0.001),
        "sigma_eta": dist.InverseGamma(0.001, 0.001),
        "sigma_zeta": dist.InverseGamma(0.001, 0.001),
    },
)

_HSA_BASE_DISTRIBUTIONS = {
    "alpha": dist.Normal(0.5, 0.2),
    "kappa": dist.Normal(0.1, 0.2),
    "kappa0": dist.Normal(0.1, 0.2),
    "kappa_0": dist.Normal(0.1, 0.2),
    "theta": dist.Normal(0.1, 0.2),
    "theta_0": dist.Normal(0.1, 0.2),
    "delta": dist.Normal(0.1, 0.2),
    "gamma": dist.Normal(0.0, 0.2),
    "beta": dist.Normal(0.1, 0.2),
    "phi_1": dist.Normal(0.7, 0.2),
    "phi_2": dist.Normal(0.2, 0.2),
    "phi_3": dist.Normal(0.2, 0.2),
    "n": dist.Normal(0.0, 1.0),
    "sigma_u": dist.InverseGamma(0.001, 0.001),
    "sigma_eps": dist.InverseGamma(0.001, 0.001),
    "sigma_v": dist.InverseGamma(0.001, 0.001),
    "sigma_mu": dist.InverseGamma(0.001, 0.001),
    "sigma_e": dist.InverseGamma(0.001, 0.001),
    "sigma_eta": dist.InverseGamma(0.001, 0.001),
    "sigma_zeta": dist.InverseGamma(0.001, 0.001),
}

register_prior_family(
    "hsa_dynamic",
    specs={
        "alpha": (0.5, 0.2),
        "kappa": (0.1, 0.2),
        "theta": (0.1, 0.2),
        "phi_1": (0.7, 0.2),
    },
    distributions=dict(_HSA_BASE_DISTRIBUTIONS),
)

register_prior_family(
    "hsa_steady",
    specs={
        "alpha": (0.5, 0.2),
        "kappa": (0.1, 0.2),
        "kappa_0": (0.1, 0.2),
        "delta": (0.1, 0.2),
        "phi_1": (0.7, 0.2),
    },
    distributions=dict(_HSA_BASE_DISTRIBUTIONS),
)

register_prior_family(
    "hsa_full",
    specs={
        "alpha": (0.5, 0.2),
        "kappa_0": (0.1, 0.2),
        "theta_0": (0.1, 0.2),
        "delta": (0.1, 0.2),
        "gamma": (0.0, 0.2),
        "phi_1": (0.7, 0.2),
    },
    distributions=dict(_HSA_BASE_DISTRIBUTIONS),
)
