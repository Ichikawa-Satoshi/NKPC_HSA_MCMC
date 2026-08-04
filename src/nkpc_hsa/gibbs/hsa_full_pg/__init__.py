"""Appendix-only Particle Gibbs variant of the PCHIP ``hsa_full`` estimator.

This package is a *separate* appendix implementation. It does not modify or
replace the production ``nkpc_hsa.gibbs.hsa_full`` estimator or its results.
The only methodological change is that the two alternating exact-FFBS state
updates (``Nhat | Nbar`` then ``Nbar | Nhat``) are replaced by a single JOINT
Particle Gibbs (conditional bootstrap SMC) update of the full state path.
Every other Gibbs block (coefficients, variances, phi_1, rho1/rho2, n, priors,
scaling, initialization) is reused unchanged from ``hsa_full``.
"""

from nkpc_hsa.gibbs.hsa_full_pg.model import (
    func_nkpc_hsa_full_pg,
    sample_states_joint_ffbs_gamma0,
    sample_states_particle_gibbs,
)

__all__ = [
    "func_nkpc_hsa_full_pg",
    "sample_states_particle_gibbs",
    "sample_states_joint_ffbs_gamma0",
]
