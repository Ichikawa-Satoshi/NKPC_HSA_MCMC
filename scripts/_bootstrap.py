from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_DIR_ENV = "NKPC_HSA_PROJECT_DIR"
configured_project = os.environ.get(PROJECT_DIR_ENV, "").strip()
ROOT = (
    Path(configured_project).expanduser().resolve()
    if configured_project
    else Path(__file__).resolve().parents[1]
)
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from nkpc_hsa.paths import data_root, results_root  # noqa: E402

DATA_DIR = data_root(ROOT)
RESULTS_DIR = results_root(ROOT)
