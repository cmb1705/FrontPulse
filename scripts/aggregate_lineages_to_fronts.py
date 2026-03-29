"""
Aggregate lineage-level time series into research-front-level time series
using Phase 5 lineage->front mappings.

This creates the input needed for tripwire_nb_fdr.py.
"""

from pathlib import Path

import pandas as pd

print("=" * 70)
print("AGGREGATING LINEAGES TO RESEARCH FRONTS")
print("=" * 70)

# Load lineage->front mappings (curated medium+ confidence)
# Try _selected first, fall back to full mappings if not curated yet
mappings_path_selected = Path("data/out/03_milestone_mapping/lineage_front_mappings_selected.csv")
mappings_path_full = Path("data/out/03_milestone_mapping/lineage_front_mappings.csv")

if mappings_path_selected.exists():
    mappings_path = mappings_path_selected
    print(f"\n[1/4] Using curated mappings: {mappings_path}")
else:
    mappings_path = mappings_path_full
    print(f"\n[1/4] Using full mappings (no curation yet): {mappings_path}")
    print("  NOTE: Consider filtering to medium/high confidence and creating")
    print(f"        {mappings_path_selected.name} for better quality")

mappings = pd.read_csv(mappings_path)
print(f"  Loaded {len(mappings)} lineage->front mappings")
print("  Confidence distribution:")
print(f"    {mappings['confidence'].value_counts().to_dict()}")

# Load lineage timeseries
timeseries_path = Path("data/out/02_lineage_tracking/lineage_timeseries.csv")
timeseries = pd.read_csv(timeseries_path)
print(f"\n[2/4] Loaded lineage timeseries: {len(timeseries)} records")
print(f"  Quarters: {timeseries['quarter'].min()} to {timeseries['quarter'].max()}")
print(f"  Unique lineages: {timeseries['lineage_id'].nunique()}")

# Create lineage_id -> primary_front lookup
lineage_to_front = dict(zip(mappings['lineage_id'], mappings['primary_front']))

# Filter timeseries to only mapped lineages
mapped_timeseries = timeseries[timeseries['lineage_id'].isin(lineage_to_front.keys())].copy()
print(f"\n[3/4] Filtered to {len(mapped_timeseries)} records from mapped lineages")

# Add front column
mapped_timeseries['front'] = mapped_timeseries['lineage_id'].map(lineage_to_front)

# Aggregate by front and quarter
# Count new works per front per quarter
front_counts = mapped_timeseries.groupby(['quarter', 'front'])['new_works'].sum().reset_index()
front_counts.rename(columns={'new_works': 'count'}, inplace=True)

# Create cumulative version (long format)
front_counts_cumulative = front_counts.copy()
front_counts_cumulative['count'] = front_counts_cumulative.groupby('front')['count'].cumsum()

# Pivot to wide format (quarters x fronts) for compatibility
front_timeseries = (
    front_counts
    .pivot(index='quarter', columns='front', values='count')
    .fillna(0)
    .astype(int)
)
front_cumulative = (
    front_counts_cumulative
    .pivot(index='quarter', columns='front', values='count')
    .sort_index()
    .ffill()
    .fillna(0)
    .astype(int)
)

# Save outputs
output_dir = Path("data/out/04_front_aggregation")
output_dir.mkdir(parents=True, exist_ok=True)

print("\n[4/4] Saving outputs (both wide and long formats)...")

# Wide format (quarters x fronts) - for visualization
delta_path = output_dir / "front_timeseries_delta.csv"
front_timeseries.to_csv(delta_path)
print(f"  Wide delta: {delta_path}")

cumulative_path = output_dir / "front_timeseries_cumulative.csv"
front_cumulative.to_csv(cumulative_path)
print(f"  Wide cumulative: {cumulative_path}")

# Long format (front, quarter, count) - for tripwire detection
delta_long_path = output_dir / "front_timeseries_delta_long.csv"
front_counts.to_csv(delta_long_path, index=False)
print(f"  Long delta: {delta_long_path}")

cumulative_long_path = output_dir / "front_timeseries_cumulative_long.csv"
front_counts_cumulative.to_csv(cumulative_long_path, index=False)
print(f"  Long cumulative: {cumulative_long_path}")
print(f"  Shape: {len(front_counts)} records, {front_counts['front'].nunique()} fronts, {front_counts['quarter'].nunique()} quarters")

# Summary statistics per front
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
