from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from _bootstrap import DATA_DIR
from nkpc_hsa.dataprep.qcew import load_qcew_national_private_establishments, merge_qcew_establishments
from nkpc_hsa.dataprep.sec_hhi import (
    DEFAULT_SEC_INVERSE_COLUMNS,
    merge_sec_inverse_hhi,
    validate_hhi_fraction,
)

# The aggregate the estimated equation implies; see sec_hhi.SEC_INVERSE_HHI_COLUMNS.
REQUIRED_SEC_COLUMN = "N_SEC_inverse_HHI_logrevw"
REBUILD_HINT = "PYTHONPATH=src python scripts/calculate_sec_hhi_quarterly.py [--from-panel]"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build distinct model-ready data for the QCEW and SEC extensions.")
    parser.add_argument("--baseline", type=Path, default=DATA_DIR / "processed" / "model_ready.csv")
    parser.add_argument("--sec-hhi", type=Path, default=DATA_DIR / "processed" / "sec_hhi_quarterly.csv")
    parser.add_argument("--qcew-dir", type=Path, default=DATA_DIR / "raw" / "competition" / "qcew")
    parser.add_argument("--processed-dir", type=Path, default=DATA_DIR / "processed")
    parser.add_argument("--skip-qcew", action="store_true")
    parser.add_argument(
        "--allow-missing-variants",
        action="store_true",
        help="Proceed even if the HHI file predates the revenue-weighted aggregates.",
    )
    args = parser.parse_args()

    baseline = pd.read_csv(args.baseline, parse_dates=["DATE"]).set_index("DATE")
    hhi_raw = pd.read_csv(args.sec_hhi)
    hhi = validate_hhi_fraction(hhi_raw)
    sec = merge_sec_inverse_hhi(baseline, hhi)
    merged = [name for name in DEFAULT_SEC_INVERSE_COLUMNS.values() if name in sec]
    if REQUIRED_SEC_COLUMN not in merged and not args.allow_missing_variants:
        raise SystemExit(
            f"{args.sec_hhi} predates the revenue-weighted aggregates: {REQUIRED_SEC_COLUMN} is missing.\n"
            f"Rebuild it with: {REBUILD_HINT}\n"
            "Pass --allow-missing-variants to build the firm-weighted series only."
        )
    sec_out = args.processed_dir / "model_ready_sec_inverse_hhi.csv"
    sec_out.parent.mkdir(parents=True, exist_ok=True)
    sec.reset_index(names="DATE").to_csv(sec_out, index=False)
    counts = ", ".join(f"{name} finite={int(sec[name].notna().sum())}" for name in merged)
    print(f"Wrote {sec_out}: hhi normalization=fraction, {counts}")

    if args.skip_qcew:
        return
    qcew = load_qcew_national_private_establishments(args.qcew_dir, start_year=1982, end_year=2012)
    qcew_out = args.processed_dir / "qcew_establishments.csv"
    qcew.to_csv(qcew_out, index=False)
    joint = merge_qcew_establishments(baseline, qcew)
    joint_out = args.processed_dir / "model_ready_qcew_joint.csv"
    joint.reset_index(names="DATE").to_csv(joint_out, index=False)
    print(f"Wrote {qcew_out} and {joint_out}: finite={joint['qcew_establishments'].notna().sum()}")


if __name__ == "__main__":
    main()
