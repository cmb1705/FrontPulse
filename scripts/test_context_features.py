#!/usr/bin/env python3
"""
Test script to demonstrate context feature computation.
"""
from pathlib import Path
import pandas as pd
import numpy as np
from collections import defaultdict

# Import the functions we added
import sys
sys.path.insert(0, str(Path(__file__).parent))
from compute_lineage_multisignal_features import (
    load_global_metrics,
    compute_context_features,
)
from utils.quarter_utils import quarter_key

def main():
    print("=" * 60)
    print("Testing Context Feature Computation")
    print("=" * 60)

    # Load global metrics
    metrics_dir = Path("data/out/metrics")
    print(f"\n1. Loading global metrics from {metrics_dir}...")
    global_metrics = load_global_metrics(metrics_dir)

    print(f"   Loaded metrics for {len(global_metrics)} quarters:")
    for quarter in sorted(global_metrics.keys(), key=quarter_key):
        print(f"   - {quarter}: {list(global_metrics[quarter].keys())}")

    # Create a synthetic timeseries that overlaps with the metrics
    print("\n2. Creating synthetic timeseries with overlapping quarters...")
    synthetic_quarters = sorted(global_metrics.keys(), key=quarter_key)
    if not synthetic_quarters:
        print("   ERROR: No quarters in global metrics!")
        return

    timeseries_data = []
    for lineage_id in [1, 2]:
        for quarter in synthetic_quarters:
            timeseries_data.append({
                'lineage_id': lineage_id,
                'quarter': quarter,
                'new_works': 10 + lineage_id * 5,
            })

    timeseries_df = pd.DataFrame(timeseries_data)
    print(f"   Created timeseries with {len(timeseries_df)} rows")
    print(f"   Lineages: {sorted(timeseries_df['lineage_id'].unique())}")
    print(f"   Quarters: {sorted(timeseries_df['quarter'].unique())}")

    # Compute context features
    print("\n3. Computing context features...")
    context_features = compute_context_features(timeseries_df, global_metrics)

    print(f"   Computed features for {len(context_features)} lineage-quarter pairs")

    if context_features:
        # Show sample features
        sample_key = list(context_features.keys())[0]
        sample_features = context_features[sample_key]
        print(f"\n4. Sample features for {sample_key}:")
        print(f"   Number of features: {len(sample_features)}")
        print(f"\n   Feature names (showing first 15):")
        for i, (name, value) in enumerate(list(sample_features.items())[:15]):
            print(f"   - {name:40s} = {value:10.4f}")

        # Verify feature structure
        print(f"\n5. Feature structure verification:")
        metric_names = ['author_influx', 'citation_velocity', 'reference_vitality',
                       'topic_diversity', 'cross_cluster_bridging']
        feature_types = ['zscore', 'qoq_delta', 'rolling_1q', 'rolling_2q',
                        'rolling_4q', 'max_dev_4q', 'min_dev_4q']

        expected_features = []
        for metric in metric_names:
            for ftype in feature_types:
                expected_features.append(f"{metric}_{ftype}")

        print(f"   Expected {len(expected_features)} features per lineage-quarter")
        print(f"   Actual: {len(sample_features)} features")

        missing = set(expected_features) - set(sample_features.keys())
        if missing:
            print(f"   WARNING: Missing features: {missing}")
        else:
            print(f"   SUCCESS: All expected features present!")

        # Check value ranges
        print(f"\n6. Value range sanity checks:")
        all_zscores = []
        all_deltas = []
        for features in context_features.values():
            for name, value in features.items():
                if '_zscore' in name:
                    all_zscores.append(value)
                elif '_qoq_delta' in name:
                    all_deltas.append(value)

        if all_zscores:
            print(f"   Z-scores: min={min(all_zscores):.2f}, max={max(all_zscores):.2f}, "
                  f"mean={np.mean(all_zscores):.2f}")
        if all_deltas:
            print(f"   QoQ deltas: min={min(all_deltas):.2f}, max={max(all_deltas):.2f}, "
                  f"mean={np.mean(all_deltas):.2f}")
    else:
        print("   WARNING: No context features computed (quarters may not overlap)")

    print("\n" + "=" * 60)
    print("Test complete!")
    print("=" * 60)

if __name__ == "__main__":
    main()
