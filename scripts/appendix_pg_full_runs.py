"""RETIRED: Particle Gibbs is now the production ``hsa_full`` sampler.

This script used to monkeypatch ``nkpc_hsa.models.hsa_full.func_nkpc_hsa_full``
to the Particle-Gibbs implementation for the duration of one process, and write
the resulting PCHIP runs to ``results/appendix_particle_gibbs/runs/`` for the
report builder to merge in.

That indirection is gone. ``run_model("hsa_full")`` now dispatches to Particle
Gibbs directly for both observation designs, so ordinary production runs are
already Particle Gibbs and no merge is needed.

To (re-)estimate the hsa_full cells:

    python scripts/rerun_hsa_full_particle_gibbs.py
"""
import sys

print(__doc__)
sys.exit(
    "appendix_pg_full_runs.py is retired: "
    "run scripts/rerun_hsa_full_particle_gibbs.py instead."
)
