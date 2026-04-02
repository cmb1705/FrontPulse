"""
Aggregate lineage-level time series into research-front-level time series.

This script consumes Phase 5 lineage-to-front mappings and produces
front-level delta and cumulative time series for downstream alerting.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from _path_bootstrap import ensure_repo_imports

repo_root = ensure_repo_imports()

from src.domain_registry import (  # noqa: E402
    add_domain_args,
    apply_domain_path_defaults,
    resolve_script_paths,
)


def _select_mappings_path(mappings_path_selected: Path, mappings_path_full: Path) -> Path:
    """Prefer curated mappings when available, otherwise fall back to full mappings."""
    if mappings_path_selected.exists():
        return mappings_path_selected
    return mappings_path_full


def run_aggregation(
    mappings_path_selected: Path,
    mappings_path_full: Path,
    timeseries_path: Path,
    output_dir: Path,
) -> dict[str, Path]:
    """Aggregate mapped lineage counts into front-level time series."""
    print("=" * 70)
    print("AGGREGATING LINEAGES TO RESEARCH FRONTS")
    print("=" * 70)

    mappings_path = _select_mappings_path(mappings_path_selected, mappings_path_full)
    if not mappings_path.exists():
        raise FileNotFoundError(f"Mappings file not found: {mappings_path}")

    if mappings_path == mappings_path_selected:
        print(f"\n[1/4] Using curated mappings: {mappings_path}")
    else:
        print(f"\n[1/4] Using full mappings (no curation yet): {mappings_path}")
        print(f"  NOTE: Consider creating {mappings_path_selected.name} for better quality")

    mappings = pd.read_csv(mappings_path)
    print(f"  Loaded {len(mappings)} lineage->front mappings")
    print(f"  Confidence distribution: {mappings['confidence'].value_counts().to_dict()}")

    if not timeseries_path.exists():
        raise FileNotFoundError(f"Lineage timeseries not found: {timeseries_path}")

    timeseries = pd.read_csv(timeseries_path)
    print(f"\n[2/4] Loaded lineage timeseries: {len(timeseries)} records")
    print(f"  Quarters: {timeseries['quarter'].min()} to {timeseries['quarter'].max()}")
    print(f"  Unique lineages: {timeseries['lineage_id'].nunique()}")

    lineage_to_front = dict(zip(mappings["lineage_id"], mappings["primary_front"]))
    mapped_timeseries = timeseries[timeseries["lineage_id"].isin(lineage_to_front)].copy()
    print(f"\n[3/4] Filtered to {len(mapped_timeseries)} records from mapped lineages")

    mapped_timeseries["front"] = mapped_timeseries["lineage_id"].map(lineage_to_front)
    front_counts = mapped_timeseries.groupby(["quarter", "front"])["new_works"].sum().reset_index()
    front_counts.rename(columns={"new_works": "count"}, inplace=True)

    front_counts_cumulative = front_counts.copy()
    front_counts_cumulative["count"] = front_counts_cumulative.groupby("front")["count"].cumsum()

    front_timeseries = (
        front_counts
        .pivot(index="quarter", columns="front", values="count")
        .fillna(0)
        .astype(int)
    )
    front_cumulative = (
        front_counts_cumulative
        .pivot(index="quarter", columns="front", values="count")
        .sort_index()
        .ffill()
        .fillna(0)
        .astype(int)
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    print("\n[4/4] Saving outputs (both wide and long formats)...")

    delta_path = output_dir / "front_timeseries_delta.csv"
    cumulative_path = output_dir / "front_timeseries_cumulative.csv"
    delta_long_path = output_dir / "front_timeseries_delta_long.csv"
    cumulative_long_path = output_dir / "front_timeseries_cumulative_long.csv"

    front_timeseries.to_csv(delta_path)
    front_cumulative.to_csv(cumulative_path)
    front_counts.to_csv(delta_long_path, index=False)
    front_counts_cumulative.to_csv(cumulative_long_path, index=False)

    print(f"  Wide delta: {delta_path}")
    print(f"  Wide cumulative: {cumulative_path}")
    print(f"  Long delta: {delta_long_path}")
    print(f"  Long cumulative: {cumulative_long_path}")
    print(
        f"  Shape: {len(front_counts)} records, "
        f"{front_counts['front'].nunique()} fronts, "
        f"{front_counts['quarter'].nunique()} quarters"
    )

    print(f"\n{'Front':<30} {'Total Works':<15} {'Quarters Active'}")
    print("=" * 70)
    for front in sorted(front_cumulative.columns):
        total = int(front_cumulative[front].iloc[-1])
        quarters_active = int((front_timeseries[front] > 0).sum())
        print(f"{front:<30} {total:<15} {quarters_active}")

    print(f"\n{'=' * 70}")
    print("AGGREGATION COMPLETE")
    print(f"{'=' * 70}")
    print("\nOutputs:")
    print("  Wide format (visualization):")
    print(f"    - {delta_path}")
    print(f"    - {cumulative_path}")
    print("  Long format (tripwire input):")
    print(f"    - {delta_long_path}")
    print(f"    - {cumulative_long_path}")
    print("\nReady for tripwire_nb_fdr.py (use *_long.csv files)!")

    return {
        "delta": delta_path,
        "cumulative": cumulative_path,
        "delta_long": delta_long_path,
        "cumulative_long": cumulative_long_path,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate lineage-level time series into research-front time series",
    )
    parser.add_argument(
        "--timeseries",
        type=Path,
        default=None,
        help="Path to lineage_timeseries.csv",
    )
    parser.add_argument(
        "--mappings-selected",
        type=Path,
        default=None,
        help="Path to curated lineage_front_mappings_selected.csv",
    )
    parser.add_argument(
        "--mappings-full",
        type=Path,
        default=None,
        help="Path to lineage_front_mappings.csv fallback",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for front aggregation outputs",
    )
    add_domain_args(parser)

    args = parser.parse_args()

    paths = resolve_script_paths(args, repo_root)
    apply_domain_path_defaults(args, paths, {
        "timeseries": (
            "lineage_tracking",
            "lineage_timeseries.csv",
            "data/out/02_lineage_tracking/lineage_timeseries.csv",
        ),
        "mappings_selected": (
            "out",
            "03_milestone_mapping/lineage_front_mappings_selected.csv",
            "data/out/03_milestone_mapping/lineage_front_mappings_selected.csv",
        ),
        "mappings_full": (
            "out",
            "03_milestone_mapping/lineage_front_mappings.csv",
            "data/out/03_milestone_mapping/lineage_front_mappings.csv",
        ),
        "output_dir": (
            "front_aggregation",
            "",
            "data/out/04_front_aggregation",
        ),
    })

    run_aggregation(
        mappings_path_selected=Path(args.mappings_selected),
        mappings_path_full=Path(args.mappings_full),
        timeseries_path=Path(args.timeseries),
        output_dir=Path(args.output_dir),
    )


if __name__ == "__main__":
    main()
