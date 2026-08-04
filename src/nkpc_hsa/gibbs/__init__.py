"""Gibbs/FFBS sampler engine.

The deprecated ``gibbs_wrappers`` helpers are intentionally NOT re-exported
here: they carried a second, lossy prior mapper and use a different N transform
from the production pipeline. Import them explicitly from
``nkpc_hsa.gibbs.gibbs_wrappers`` if an archived script still needs them.
Production estimation goes through ``nkpc_hsa.inference.wrappers.run_model``.
"""

from . import ces, common, hsa_const_theta, hsa_dynamic, hsa_full, hsa_full_pg, hsa_steady

__all__ = [
    "ces",
    "common",
    "hsa_const_theta",
    "hsa_dynamic",
    "hsa_full",
    "hsa_full_pg",
    "hsa_steady",
]
