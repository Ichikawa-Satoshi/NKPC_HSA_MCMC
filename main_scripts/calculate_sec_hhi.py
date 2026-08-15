from __future__ import annotations

import argparse

from _bootstrap import DATA_DIR, ROOT
from nkpc_hsa.dataprep.sec_hhi import build_sec_hhi_csv


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calculate annual three-digit-SIC sales HHIs from SEC Financial Statement Data Sets."
    )
    parser.add_argument("--sec-dir", default=str(DATA_DIR / "raw" / "competition" / "sec"))
    parser.add_argument("--out", default=str(DATA_DIR / "processed" / "sec_hhi.csv"))
    parser.add_argument(
        "--start-year",
        type=int,
        default=2011,
        help="First fiscal year (default: 2011, when SEC XBRL coverage became universal).",
    )
    parser.add_argument(
        "--end-year",
        type=int,
        help="Last fiscal year (default: latest archive year minus one).",
    )
    parser.add_argument("--chunksize", type=int, default=500_000)
    args = parser.parse_args()

    annual = build_sec_hhi_csv(
        args.sec_dir,
        args.out,
        start_year=args.start_year,
        end_year=args.end_year,
        chunksize=args.chunksize,
    )
    print(f"Wrote {args.out}: {len(annual)} annual observations")
    print(annual.to_string(index=False))


if __name__ == "__main__":
    main()
