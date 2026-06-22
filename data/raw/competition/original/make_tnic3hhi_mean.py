from __future__ import annotations

from pathlib import Path

import pandas as pd

ROUND_DIGITS = 4


def build_yearly_mean(input_path: Path, output_path: Path) -> pd.DataFrame:
    df = pd.read_csv(input_path, sep="\t")
    mean_df = (
        df.groupby("year", as_index=False)
        .agg(
            tnic3hhi_mean=("tnic3hhi", "mean"),
        )
        .sort_values("year")
    )
    mean_df["tnic3hhi_inv_mean"] = 1.0 / mean_df["tnic3hhi_mean"]
    mean_df["tnic3hhi_inv_mean"] = mean_df["tnic3hhi_inv_mean"].round(ROUND_DIGITS)
    mean_df = mean_df[["year", "tnic3hhi_inv_mean"]]
    mean_df.to_csv(output_path, index=False, float_format=f"%.{ROUND_DIGITS}f")
    return mean_df


def main() -> None:
    here = Path(__file__).resolve().parent
    input_path = here / "TNIC3HHIdata.txt"
    output_path = here / "TNIC3HHIdata_mean.csv"
    mean_df = build_yearly_mean(input_path, output_path)
    print(f"Wrote {output_path}")
    print(mean_df.head())
    print(mean_df.tail())


if __name__ == "__main__":
    main()
