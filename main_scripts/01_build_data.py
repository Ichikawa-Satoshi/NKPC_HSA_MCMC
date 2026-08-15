from __future__ import annotations

import argparse

from _bootstrap import DATA_DIR, ROOT
from nkpc_hsa.dataprep.build import build_processed_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", default=str(DATA_DIR / "raw"))
    parser.add_argument("--out", default=str(DATA_DIR / "processed" / "model_ready.csv"))
    args = parser.parse_args()
    data = build_processed_dataset(args.raw_dir, args.out)
    print(f"Saved {args.out} with {len(data)} rows and {len(data.columns)} columns.")


if __name__ == "__main__":
    main()
