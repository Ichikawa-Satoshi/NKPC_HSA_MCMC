"""Production Particle Gibbs estimator for ``hsa_full``.

The bilinear term ``-gamma * Nbar_t * Nhat_t`` makes the joint firm-count state
non-linear-Gaussian, so the observation loading on ``Nhat_t`` contains a state
and no exact Kalman/FFBS draw of the whole path exists. The two alternating
exact-FFBS blocks (``Nhat | Nbar`` then ``Nbar | Nhat``) are each valid
conditionals but mix badly, because ``N_obs = Nbar + Nhat + nu`` pins the sum
almost exactly (posterior corr(Nbar_0, Nhat_0) is about -0.999).

This package replaces them with a single JOINT Particle Gibbs (conditional
bootstrap SMC) update of the full state path, which leaves the exact conditional
posterior invariant for any particle count. Every other Gibbs block
(coefficients, variances, phi_1, rho1/rho2, n, priors, scaling, initialization)
is imported unchanged from ``hsa_full``, so priors and units match exactly.

``run_model("hsa_full")`` dispatches here. The superseded alternating-FFBS
sampler remains importable as
``nkpc_hsa.models.hsa_full.func_nkpc_hsa_full_alternating_ffbs`` for validation.
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
