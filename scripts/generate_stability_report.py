#!/usr/bin/env python3
"""Generate a community stability report from lineage timeseries data.

Computes lifespan distributions, partition instability (VI), paper identity
alignment (PIA), and activity metrics.  Outputs both JSON and human-readable
text formats.

Usage:
    python scripts/generate_stability_report.py
    python scripts/generate_stability_report.py --timeseries path/to/ts.csv --out-dir results/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from _path_bootstrap import ensure_repo_imports

REPO_ROOT = ensure_repo_imports()

from src.stability_report import compute_stability_report, format_report_text  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate a community stability report from lineage timeseries.",
    )
    parser.add_argument(
        "--timeseries",
        default="data/out/02_lineage_tracking/lineage_timeseries.csv",
        help="Path to lineage timeseries CSV.",
    )
    parser.add_argument(
        "--vi-column",
        default="VI_vs_prev_quarter",
        help="Column name for VI values (default: VI_vs_prev_quarter).",
    )
    parser.add_argument(
        "--pia-column",
        default="pia_rate",
        help="Column name for PIA rate (default: pia_rate).",
    )
    parser.add_argument(
        "--activity-column",
        default="new_works",
        help="Column name for new works count (default: new_works).",
    )
    parser.add_argument(
        "--out-dir",
        default="data/out/02_lineage_tracking",
        help="Output directory for report files.",
    )
    parser.add_argument(
        "--suffix",
        default="",
        help="Suffix appended to output filenames.",
    )
    return parser.parse_args()


def main() -> None:
    """Generate stability report and write outputs."""
    args = parse_args()

    timeseries_path = Path(args.timeseries)
    if not timeseries_path.exists():
        print(f"ERROR: Timeseries file not found: {timeseries_path}")
        sys.exit(1)

    print(f"Loading timeseries from {timeseries_path}")
    ts_df = pd.read_csv(timeseries_path)
    ts_df["quarter"] = ts_df["quarter"].astype(str)
    print(f"  {len(ts_df):,} rows, {ts_df['lineage_id'].nunique():,} lineages")

    report = compute_stability_report(
        ts_df,
        vi_column=args.vi_column,
        pia_column=args.pia_column,
        activity_column=args.activity_column,
    )

    # Print human-readable report
    text = format_report_text(report)
    print()
    print(text)

    # Write outputs
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = args.suffix

    json_out = out_dir / f"stability_report{suffix}.json"
    with json_out.open("w") as fh:
        json.dump(report, fh, indent=2)
    print(f"\nWrote JSON report: {json_out}")

    text_out = out_dir / f"stability_report{suffix}.txt"
    text_out.write_text(text, encoding="utf-8")
    print(f"Wrote text report: {text_out}")


if __name__ == "__main__":
    main()
