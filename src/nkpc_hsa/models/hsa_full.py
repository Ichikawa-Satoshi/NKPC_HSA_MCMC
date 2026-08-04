from __future__ import annotations

"""Public facade for the HSA full model.

The production state sampler is Particle Gibbs (conditional SMC): the bilinear
term ``-gamma * Nbar_t * Nhat_t`` makes the joint firm-count state
non-linear-Gaussian, so no exact Kalman/FFBS draw of the whole path exists.

``func_nkpc_hsa_full`` therefore resolves to the Particle-Gibbs implementation.
The superseded alternating-FFBS sampler is still importable as
``func_nkpc_hsa_full_alternating_ffbs`` for validation and for the const-theta
old-vs-new pilot, but it is no longer reachable from ``run_model``.
"""

from nkpc_hsa.gibbs.hsa_full import func_nkpc_hsa_full as func_nkpc_hsa_full_alternating_ffbs
from nkpc_hsa.gibbs.hsa_full_pg import func_nkpc_hsa_full_pg as func_nkpc_hsa_full

__all__ = ["func_nkpc_hsa_full", "func_nkpc_hsa_full_alternating_ffbs"]
