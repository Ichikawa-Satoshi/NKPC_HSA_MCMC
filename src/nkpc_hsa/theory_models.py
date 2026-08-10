"""Definitions and unit mappings for the post-migration theory-model namespace.

This module is deliberately declarative.  Estimation, diagnostics, and reporting
all read the same definitions so a slug cannot acquire a different meaning at a
later pipeline stage.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

import numpy as np


THEORY_ESTIMATION_REVISION = "2026-08-moving-reference-hsa-v1"
STRUCTURAL_FREQUENCY = "quarterly"
VALID_INFLATION_OBSERVATIONS = {"qoq", "yoy_4q"}


@dataclass(frozen=True)
class TheoryModelDefinition:
    slug: str
    hierarchy: str
    reference_design: str
    cross_equation_restriction: bool
    second_law: bool
    third_law: str
    gamma_restriction: str
    moving_reference: bool
    state_sampler: str
    exact_restrictions: tuple[str, ...]

    def metadata(self) -> dict[str, Any]:
        out = asdict(self)
        # JSON is the canonical run-metadata representation. Normalize tuples
        # before signing so an in-memory definition and its reloaded JSON are
        # byte-for-byte comparable.
        out["exact_restrictions"] = list(self.exact_restrictions)
        return out


THEORY_MODELS: dict[str, TheoryModelDefinition] = {
    "hsa_f0": TheoryModelDefinition(
        slug="hsa_f0",
        hierarchy="F0",
        reference_design="fixed_reference",
        cross_equation_restriction=False,
        second_law=True,
        third_law="neither",
        gamma_restriction="not_applicable",
        moving_reference=False,
        state_sampler="fixed_reference_ffbs",
        exact_restrictions=("N_reference_t = N0", "kappa_t = kappa0"),
    ),
    "hsa_u": TheoryModelDefinition(
        slug="hsa_u",
        hierarchy="U",
        reference_design="moving_reference",
        cross_equation_restriction=False,
        second_law=False,
        third_law="neither",
        gamma_restriction="unrestricted",
        moving_reference=True,
        state_sampler="particle_gibbs",
        exact_restrictions=(
            "kappa_t = kappa0 + kappa_N_empirical * Nbar_t",
            "theta_t = theta0 + gamma * Nbar_t",
            "kappa_N_empirical, theta0, gamma sampled independently",
        ),
    ),
    "hsa_r1": TheoryModelDefinition(
        slug="hsa_r1",
        hierarchy="R1",
        reference_design="moving_reference",
        cross_equation_restriction=True,
        second_law=True,
        third_law="neither",
        gamma_restriction="unrestricted",
        moving_reference=True,
        state_sampler="particle_gibbs",
        exact_restrictions=(
            "100 * kappa_N_empirical = b_x * zeta0 * theta0",
            "kappa_N_empirical derived, not sampled",
            "theta0 > 0",
        ),
    ),
    "hsa_r2": TheoryModelDefinition(
        slug="hsa_r2",
        hierarchy="R2",
        reference_design="moving_reference",
        cross_equation_restriction=True,
        second_law=True,
        third_law="weak",
        gamma_restriction="gamma <= 0",
        moving_reference=True,
        state_sampler="particle_gibbs",
        exact_restrictions=(
            "100 * kappa_N_empirical = b_x * zeta0 * theta0",
            "kappa_N_empirical derived, not sampled",
            "theta0 > 0",
            "gamma <= 0",
        ),
    ),
    "hsa_r3": TheoryModelDefinition(
        slug="hsa_r3",
        hierarchy="R3",
        reference_design="moving_reference",
        cross_equation_restriction=True,
        second_law=True,
        third_law="neither",
        gamma_restriction="gamma = 0",
        moving_reference=True,
        state_sampler="joint_ffbs",
        exact_restrictions=(
            "100 * kappa_N_empirical = b_x * zeta0 * theta0",
            "kappa_N_empirical derived, not sampled",
            "theta0 > 0",
            "gamma = 0",
        ),
    ),
}


HISTORICAL_MODELS: dict[str, str] = {
    "ces": "historical reduced-form CES benchmark",
    "hsa_steady": "historical reduced-form varying-slope ablation",
    "hsa_dynamic": "historical reduced-form entry-channel ablation",
    "hsa_const_theta": "historical reduced-form constant-theta ablation",
    "hsa_full": "historical unrestricted reduced-form full ablation",
}


RESTRICTION_TAXONOMY: dict[str, tuple[str, ...]] = {
    "theory_identities": (
        "kappa0_structural = (zeta0 - 1) / chi",
        "Theta0 = (1 / chi) * (1 - rho0) / rho0",
        "mu0 = zeta0 / (zeta0 - 1)",
        "d kappa / d log N = zeta * Theta",
    ),
    "maintained_hsa_laws": (
        "Second law: theta0 > 0 and kappa_N_empirical > 0",
        "Weak Third law: gamma <= 0 (R2 only)",
    ),
    "domain_admissibility": (
        "chi > 0",
        "zeta0 > 1",
        "mu0 > 1",
        "kappa0 > 0",
        "optional whole-path kappa_t > 0 and theta_t > 0",
    ),
    "empirical_approximations": (
        "moving Nstar_t is a slow-moving local reference",
        "kappa_t = kappa0 + kappa_N_empirical * Nbar_t",
        "theta_t = theta0 + gamma * Nbar_t",
    ),
    "deliberately_not_imposed": (
        "kappa0 = (zeta0 - 1) / chi",
        "theta0 = (10 / chi) * (1 - rho0) / rho0",
    ),
}


def is_theory_model(slug: str) -> bool:
    return slug in THEORY_MODELS


def theory_model_definition(slug: str) -> TheoryModelDefinition:
    try:
        return THEORY_MODELS[slug]
    except KeyError as exc:
        raise ValueError(f"Unknown theory model {slug!r}; expected one of {sorted(THEORY_MODELS)}") from exc


def zeta_from_mu(mu0: float | np.ndarray) -> np.ndarray:
    mu = np.asarray(mu0, dtype=float)
    if np.any(~np.isfinite(mu)) or np.any(mu <= 1.0):
        raise ValueError("mu0 must be finite and greater than one.")
    return mu / (mu - 1.0)


def mu_from_zeta(zeta0: float | np.ndarray) -> np.ndarray:
    zeta = np.asarray(zeta0, dtype=float)
    if np.any(~np.isfinite(zeta)) or np.any(zeta <= 1.0):
        raise ValueError("zeta0 must be finite and greater than one.")
    return zeta / (zeta - 1.0)


def restricted_kappa_mapping(
    theta0: float | np.ndarray,
    zeta0: float | np.ndarray,
    *,
    marginal_cost_loading: float = 1.0,
) -> dict[str, np.ndarray]:
    """Map the quarterly structural identity into empirical ten-log-point units.

    ``Nbar`` is measured as ``10 * (log N - log N0)`` and inflation is in
    percentage points.  Therefore the stored physical coefficient is

        kappa_N_empirical = b_x * zeta0 * theta0 / 100,

    while ``d_kappa_d_logN = 10 * kappa_N_empirical``.  The sampler's internal
    coefficient is another factor of ``KAPPA_SCALE=100`` larger and therefore
    equals ``b_x * zeta0 * theta0``.
    """
    theta = np.asarray(theta0, dtype=float)
    zeta = np.asarray(zeta0, dtype=float)
    b_x = float(marginal_cost_loading)
    if not np.isfinite(b_x):
        raise ValueError("marginal_cost_loading must be finite.")
    empirical = b_x * zeta * theta / 100.0
    return {
        "kappa_N_empirical": empirical,
        "d_kappa_d_logN": 10.0 * empirical,
        "kappa_N_internal": 100.0 * empirical,
    }


def validate_theory_run_design(
    slug: str,
    data_spec: Mapping[str, Any],
    *,
    zeta0: float,
    inflation_observation: str,
) -> float:
    """Validate restrictions whose meaning depends on data/scaling choices.

    Returns the explicitly configured marginal-cost loading ``b_x``.  Restricted
    models may use an activity gap only when that loading is supplied; this
    prevents the code from silently setting ``b_x=1`` for unemployment/output
    gaps.
    """
    definition = theory_model_definition(slug)
    if inflation_observation not in VALID_INFLATION_OBSERVATIONS:
        raise ValueError(
            f"inflation_observation must be one of {sorted(VALID_INFLATION_OBSERVATIONS)}, "
            f"got {inflation_observation!r}."
        )
    if not np.isfinite(zeta0) or zeta0 <= 1.0:
        raise ValueError("zeta0 must be finite and greater than one (domain restriction).")
    b_x_raw = data_spec.get("marginal_cost_loading")
    mapping = str(data_spec.get("activity_mapping", "gap_proxy"))
    if definition.cross_equation_restriction:
        if inflation_observation == "yoy_4q":
            raise ValueError(
                f"{slug} cannot impose a quarterly structural cross-equation identity on the "
                "current direct 4Q-YoY regression. Use the qoq observation design. A future "
                "YoY likelihood must aggregate four quarterly structural equations without "
                "changing coefficient meaning."
            )
        if b_x_raw is None:
            raise ValueError(
                f"{slug} requires data_spec.marginal_cost_loading. "
                f"The activity mapping is {mapping!r}; b_x=1 is never assumed for a gap proxy."
            )
    return float(b_x_raw if b_x_raw is not None else 1.0)
