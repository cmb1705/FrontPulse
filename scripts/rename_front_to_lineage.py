"""
Rename 'front' terminology to 'lineage' throughout pipeline outputs.

This script renames:
1. front_id -> lineage_id in CSV files
2. front_timeseries_cumulative.csv -> lineage_timeseries.csv
3. front_metrics_cumulative.csv -> lineage_metrics.csv
4. front_id_registry_cumulative.json -> lineage_registry.json
"""

import shutil
from pathlib import Path

import pandas as pd


def rename_pipeline_outputs():
    """Rename front terminology to lineage in all output files."""

    base_dir = Path(__file__).parent.parent
    out_dir = base_dir / "data" / "out"

    print("=" * 70)
    print("RENAMING FRONT -> LINEAGE IN PIPELINE OUTPUTS")
    print("=" * 70)

    # 1. Rename and update front_timeseries_cumulative.csv
    print("\n[1/3] Processing front_timeseries_cumulative.csv...")
    timeseries_old = out_dir / "front_timeseries_cumulative.csv"
    timeseries_new = out_dir / "lineage_timeseries.csv"

    if timeseries_old.exists():
        df = pd.read_csv(timeseries_old)
        print(f"   Loaded {len(df):,} rows")

        if 'front_id' in df.columns:
            df = df.rename(columns={'front_id': 'lineage_id'})
            print("   Renamed front_id -> lineage_id")

        df.to_csv(timeseries_new, index=False)
        print(f"   Saved to {timeseries_new.name}")
        print(f"   Columns: {', '.join(df.columns)}")
    else:
        print(f"   WARNING: {timeseries_old} not found")

    # 2. Rename and update front_metrics_cumulative.csv
    print("\n[2/3] Processing front_metrics_cumulative.csv...")
    metrics_old = out_dir / "front_metrics_cumulative.csv"
    metrics_new = out_dir / "lineage_metrics.csv"

    if metrics_old.exists():
        df = pd.read_csv(metrics_old)
        print(f"   Loaded {len(df):,} rows")

        if 'front_id' in df.columns:
            df = df.rename(columns={'front_id': 'lineage_id'})
            print("   Renamed front_id -> lineage_id")

        df.to_csv(metrics_new, index=False)
        print(f"   Saved to {metrics_new.name}")
        print(f"   Columns: {', '.join(df.columns)}")
    else:
        print(f"   WARNING: {metrics_old} not found")

    # 3. Rename front_id_registry_cumulative.json
    print("\n[3/3] Processing front_id_registry_cumulative.json...")
    registry_old = out_dir / "front_id_registry_cumulative.json"
    registry_new = out_dir / "lineage_registry.json"

    if registry_old.exists():
        shutil.copy2(registry_old, registry_new)
        print(f"   Copied to {registry_new.name}")
        print("   Note: JSON structure unchanged (maps community_id -> lineage_id)")
    else:
        print(f"   WARNING: {registry_old} not found")

    print("\n" + "=" * 70)
    print("RENAMING COMPLETE")
    print("=" * 70)
    print("\nNew files created:")
    print(f"  - {timeseries_new}")
    print(f"  - {metrics_new}")
    print(f"  - {registry_new}")
    print("\nOld files remain for backup. Delete manually if no longer needed.")

if __name__ == "__main__":
    rename_pipeline_outputs()
