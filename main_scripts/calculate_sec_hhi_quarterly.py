from __future__ import annotations

import argparse

from _bootstrap import DATA_DIR, ROOT
from nkpc_hsa.dataprep.sec_hhi import build_sec_quarterly_hhi_csv, recalculate_quarterly_hhi_from_panel


def main() -> None:
    parser = argparse.ArgumentParser(description="Calculate actual quarterly SIC3 sales HHIs from SEC 10-Q/10-K data.")
    parser.add_argument("--sec-dir", default=str(DATA_DIR / "raw" / "competition" / "sec"))
    parser.add_argument("--out", default=str(DATA_DIR / "processed" / "sec_hhi_quarterly.csv"))
    parser.add_argument(
        "--company-panel",
        default=str(DATA_DIR / "processed" / "sec_company_quarter_revenues.csv"),
        help="Where to cache the extracted company-quarter revenue panel.",
    )
    parser.add_argument(
        "--from-panel",
        action="store_true",
        help="Re-aggregate the cached company panel instead of re-reading the SEC archives.",
    )
    parser.add_argument("--start-quarter", default="2012Q1")
    parser.add_argument("--end-quarter")
    parser.add_argument("--chunksize", type=int, default=500_000)
    args = parser.parse_args()
    if args.from_panel:
        result = recalculate_quarterly_hhi_from_panel(args.company_panel, args.out)
    else:
        result = build_sec_quarterly_hhi_csv(
            args.sec_dir,
            args.out,
            start_quarter=args.start_quarter,
            end_quarter=args.end_quarter,
            chunksize=args.chunksize,
            company_panel_path=args.company_panel,
        )
    print(f"Wrote {args.out}: {len(result)} quarterly observations")
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
