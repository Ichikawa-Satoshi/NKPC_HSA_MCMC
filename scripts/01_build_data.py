from __future__ import annotations

import argparse

from _bootstrap import ROOT
from nkpc_hsa.data.build import build_processed_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", default=str(ROOT / "data" / "raw"))
    parser.add_argument("--out", default=str(ROOT / "data" / "processed" / "model_ready.csv"))
    args = parser.parse_args()
    data = build_processed_dataset(args.raw_dir, args.out)
    print(f"Saved {args.out} with {len(data)} rows and {len(data.columns)} columns.")


if __name__ == "__main__":
    main()
