#!/usr/bin/env python3
"""
Convert wide-format front timeseries to long format for tripwire detection.

Tripwire expects: (front, quarter, count) in long format
Aggregate outputs: quarters × fronts in wide format

This script converts between the two formats.

Usage:
    python scripts/build_long_timeseries.py

Outputs:
    - data/out/04_front_aggregation/front_timeseries_cumulative_long.csv
    - data/out/04_front_aggregation/front_timeseries_delta_long.csv
"""

import pandas as pd
from pathlib import Path


def melt_wide_to_long(path_in: Path, path_out: Path, value_name: str) -> None:
    """Convert wide-format timeseries to long format."""
    df = pd.read_csv(path_in)

    if 'quarter' not in df.columns:
        raise ValueError(f"'quarter' column not found in {path_in}")

    # Melt wide → long
    long = df.melt(id_vars='quarter', var_name='front', value_name=value_name)

    # Ensure output directory exists
    path_out.parent.mkdir(parents=True, exist_ok=True)

    # Save
    long.to_csv(path_out, index=False)
    print(f"  Wrote {len(long)} rows to {path_out}")


def main():
    print("=" * 70)
    print("CONVERTING WIDE FORMAT TO LONG FORMAT")
    print("=" * 70)

    print("\n[1/2] Converting cumulative timeseries...")
    melt_wide_to_long(
        Path("data/out/04_front_aggregation/front_timeseries_cumulative.csv"),
        Path("data/out/04_front_aggregation/front_timeseries_cumulative_long.csv"),
        value_name="count",
    )

    print("\n[2/2] Converting delta timeseries...")
    melt_wide_to_long(
        Path("data/out/04_front_aggregation/front_timeseries_delta.csv"),
        Path("data/out/04_front_aggregation/front_timeseries_delta_long.csv"),
        value_name="count",
    )

    print("\n" + "=" * 70)
    print("CONVERSION COMPLETE")
    print("=" * 70)
    print("\nReady for tripwire_nb_fdr.py with long-format data!")
    print("\nExample command:")
    print("  python tripwire_nb_fdr.py \\")
    print("    --timeseries data/out/04_front_aggregation/front_timeseries_cumulative_long.csv \\")
    print("    --front-col front \\")
    print("    --date-col quarter \\")
    print("    --count-col count \\")
    print("    --out data/out/05_tripwire_detection/alerts_tripwire.csv \\")
    print("    --lookback 8 --min-history 6 --min-count 2")
    print()


if __name__ == "__main__":
    main()
