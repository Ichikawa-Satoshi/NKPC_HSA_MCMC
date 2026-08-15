"""Path bootstrap shared by every ``experiments/<name>/run.py`` bundle.

Mirrors ``production/main_scripts/_bootstrap.py`` but resolves the project root two levels up
(``experiments/<name>/run.py``) so an experiment can be launched directly from
its own directory.  Import it as ``from experiments._bootstrap import ...``.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_DIR_ENV = "NKPC_HSA_PROJECT_DIR"
_configured = os.environ.get(PROJECT_DIR_ENV, "").strip()
ROOT = (
    Path(_configured).expanduser().resolve()
    if _configured
    else next(a for a in Path(__file__).resolve().parents if (a / "pyproject.toml").exists())
)
SRC = ROOT / "src"
for _path in (str(ROOT), str(SRC)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from nkpc_hsa.paths import data_root, results_root  # noqa: E402

DATA_DIR = data_root(ROOT)
RESULTS_DIR = results_root(ROOT)
