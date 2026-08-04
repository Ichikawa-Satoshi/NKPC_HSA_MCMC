"""RETIRED -- the Particle-Gibbs table override is now part of the main pipeline.

This script used to regenerate a subset of the English report tables from a
merged run-set and overwrite them in place, after ``scripts/12_build_cpi_ppi_report.py``
had already written the alternating-FFBS versions. Because it patched only six
of the report's table files, the ``hsa_full`` cells appeared under Particle Gibbs
in some tables and under alternating FFBS in others -- most visibly
``model_comparison_unemp.tex`` (Table 3) versus ``unemployment_by_model.tex``
(Table 15), which disagreed on both the coefficients and the convergence flag
for the same cells. ``result_macros.tex`` was likewise left stale, so the
reported warning count contradicted the convergence table.

The merge now happens once, where the report's run-set is assembled:

    scripts/12_build_cpi_ppi_report.py :: load_report_runs()

which every report-table script calls. ``hsa_full`` PCHIP cells are routed to
``results/appendix_particle_gibbs/runs`` there, annual-Q4 is deliberately left on
alternating FFBS (no annual-Q4 Particle-Gibbs runs exist), and
``assert_single_sampler_per_cell`` fails the build if any cell would be reported
under two samplers.

To rebuild the report tables:

    python scripts/appendix_pg_full_runs.py       # only if the PG runs are missing
    python scripts/12_build_cpi_ppi_report.py     # merges PG hsa_full automatically
    python scripts/make_headline_results_table.py
    python scripts/build_english_tables.py

Pass ``--no-pg`` to step 2 to report alternating-FFBS hsa_full everywhere instead.
"""
from __future__ import annotations

import sys


def main() -> int:
    print(__doc__)
    print("This script is retired and performs no work. Exiting without changes.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
