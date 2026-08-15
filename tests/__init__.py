"""Self-contained experiment ("test") bundles.

Each subpackage holds one experiment as ``functions.py`` (its estimation
functions), ``run.py`` (its run code), ``config.yaml``, ``README.md`` (its
description), and ``results/`` (its estimation results and figures, written next
to the code and git-ignored because they are reproducible).  The heavy shared
engine (samplers, dataprep, the shared Phillips-curve toolkit) lives in
``nkpc_hsa`` and is imported, never copied.
"""
