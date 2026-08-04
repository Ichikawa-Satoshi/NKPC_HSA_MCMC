"""Public facade for the HSA const-theta sampler (gamma = 0, exact joint FFBS).

``func_nkpc_hsa_full_static_theta`` is kept as a deprecated alias: it is the old
alternating-FFBS implementation of the same model and is retained only so that
archived scripts keep running. New work should use
``func_nkpc_hsa_const_theta``.
"""
from __future__ import annotations

from nkpc_hsa.gibbs.hsa_const_theta import func_nkpc_hsa_const_theta
from nkpc_hsa.gibbs.hsa_full import func_nkpc_hsa_full_static_theta

__all__ = ["func_nkpc_hsa_const_theta", "func_nkpc_hsa_full_static_theta"]
