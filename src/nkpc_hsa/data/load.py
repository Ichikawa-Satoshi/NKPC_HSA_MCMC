from __future__ import annotations

from pathlib import Path

import pandas as pd

from nkpc_hsa.paths import project_path


def load_processed_dataset(path: str | Path | None = None) -> pd.DataFrame:
    target = Path(path) if path is not None else project_path("data", "processed", "model_ready.csv")
    return pd.read_csv(target, parse_dates=["DATE"]).set_index("DATE")
