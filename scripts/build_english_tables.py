"""RETIRED: the report tables are written in English at source.

This script used to mirror ``results/tables/cpi_ppi_report`` into a parallel
``..._en`` tree, replacing a fixed list of Japanese label literals, because the
same tables were shared with a Japanese edition of the report.

The Japanese edition has been removed and the table builders now emit English
directly; ``scripts/12_build_cpi_ppi_report.py`` additionally refuses to write a
table containing CJK text. The report reads ``results/tables/cpi_ppi_report``
directly, so no mirror and no translation pass exist any more.

    python scripts/12_build_cpi_ppi_report.py      # writes the tables
    python scripts/make_headline_results_table.py  # writes the headline tables
"""
import sys

print(__doc__)
sys.exit("build_english_tables.py is retired: the tables are already English.")
