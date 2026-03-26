#!/usr/bin/env python3
"""Filter lineage timeseries and features to a stable-lineage cohort.

Applies a minimum-lifespan filter and writes the filtered subsets alongside
summary statistics.  The output is a direct comparison point against the
unrestricted lineage set.

Usage:
    python scripts/filter_stable_lineages.py --min-quarters 8
    python scripts/filter_stable_lineages.py --min-quarters 4 --suffix _4q
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.domain_registry import add_domain_args, resolve_script_paths  # noqa: E402
from src.stable_lineage_filter import (  # noqa: E402
    filter_stable_lineages,
    summarize_filter,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Filter lineages to a stable cohort by minimum lifespan.",
    )
    parser.add_argument(
        "--timeseries",
        default=None,
        help="Path to lineage timeseries CSV.",
    )
    parser.add_argument(
        "--features",
        default=None,
        help="Path to lineage multisignal features CSV. If provided, also writes filtered features.",
    )
    parser.add_argument(
        "--min-quarters",
        type=int,
        default=8,
        help="Minimum lifespan in quarters (default: 8).",
    )
    parser.add_argument(
        "--max-quarters",
        type=int,
        default=None,
        help="Maximum lifespan in quarters (optional upper bound).",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Output directory for filtered files.",
    )
    parser.add_argument(
        "--suffix",
        default="_stable",
        help="Suffix appended to output filenames (default: _stable).",
    )
    add_domain_args(parser)
    return parser.parse_args()


def main() -> None:
    """Filter lineages and write outputs."""
    args = parse_args()

    paths = resolve_script_paths(args, REPO_ROOT)
    if args.timeseries is None:
        args.timeseries = str(paths.lineage_tracking / "lineage_timeseries.csv") if paths else "data/out/02_lineage_tracking/lineage_timeseries.csv"
    if args.out_dir is None:
        args.out_dir = str(paths.lineage_tracking) if paths else "data/out/02_lineage_tracking"

    timeseries_path = Path(args.timeseries)
    if not timeseries_path.exists():
        print(f"ERROR: Timeseries file not found: {timeseries_path}")
        sys.exit(1)

    print(f"Loading timeseries from {timeseries_path}")
    ts_df = pd.read_csv(timeseries_path)
    ts_df["quarter"] = ts_df["quarter"].astype(str)

    # Summarize before filtering
    summary = summarize_filter(
        ts_df,
        min_quarters=args.min_quarters,
        max_quarters=args.max_quarters,
    )
    print(f"\nFilter summary (min_quarters={args.min_quarters}):")
    print(f"  Total lineages:  {summary['total_lineages']:,}")
    print(f"  Stable lineages: {summary['stable_lineages']:,} ({summary['stable_pct']}%%)")
    print(f"  Total records:   {summary['total_records']:,}")
    print(f"  Stable records:  {summary['stable_records']:,} ({summary['stable_records_pct']}%%)")

    # Apply filter
    stable_ids = filter_stable_lineages(
        ts_df,
        min_quarters=args.min_quarters,
        max_quarters=args.max_quarters,
    )
    filtered_ts = ts_df[ts_df["lineage_id"].isin(stable_ids)].copy()

    # Write filtered timeseries
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = args.suffix

    ts_out = out_dir / f"lineage_timeseries{suffix}.csv"
    filtered_ts.to_csv(ts_out, index=False)
    print(f"\nWrote filtered timeseries: {ts_out} ({len(filtered_ts):,} rows)")

    # Write stable lineage ID list
    ids_out = out_dir / f"stable_lineage_ids{suffix}.json"
    with ids_out.open("w") as fh:
        json.dump(sorted(stable_ids), fh)
    print(f"Wrote stable lineage IDs: {ids_out} ({len(stable_ids)} lineages)")

    # Filter features if provided
    if args.features:
        feat_path = Path(args.features)
        if feat_path.exists():
            print(f"\nFiltering features from {feat_path}")
            feat_df = pd.read_csv(feat_path)
            filtered_feat = feat_df[feat_df["lineage_id"].isin(stable_ids)].copy()
            feat_out = out_dir / f"lineage_multisignal_features{suffix}.csv"
            filtered_feat.to_csv(feat_out, index=False)
            print(f"Wrote filtered features: {feat_out} ({len(filtered_feat):,} rows)")
        else:
            print(f"WARNING: Features file not found: {feat_path}")

    # Write summary JSON
    summary_out = out_dir / f"stable_filter_summary{suffix}.json"
    with summary_out.open("w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"Wrote filter summary: {summary_out}")


if __name__ == "__main__":
    main()
