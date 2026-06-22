from __future__ import annotations

from pathlib import Path

import pandas as pd

from nkpc_hsa.paths import project_path


def build_processed_dataset(raw_dir: str | Path | None = None, out_path: str | Path | None = None) -> pd.DataFrame:
    """Build the processed model-ready dataset without overwriting raw data."""
    try:
        from analysis.gibbs.func_data_build import build_dataset
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Legacy data builder is unavailable.") from exc

    raw = Path(raw_dir) if raw_dir is not None else project_path("data", "raw")
    if not (raw / "inflation").exists() and raw == project_path("data", "raw"):
        legacy_raw = project_path("data")
        if (legacy_raw / "inflation").exists():
            raw = legacy_raw
    out = Path(out_path) if out_path is not None else project_path("data", "processed", "model_ready.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    data = build_dataset(raw)
    data.to_csv(out, index=False)
    return data
