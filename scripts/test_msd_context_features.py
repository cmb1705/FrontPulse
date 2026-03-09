#!/usr/bin/env python3
"""
Smoke test for MSD context feature integration (Task 3.1).

Tests that multi_signal_detector.py correctly detects and uses context features.
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from pathlib import Path

# Create synthetic test data
def create_test_features(include_context: bool = False) -> pd.DataFrame:
    """Create synthetic multisignal features for testing."""
    np.random.seed(42)
    n_samples = 100

    # Core features (always present)
    data = {
        'lineage_id': np.arange(1, n_samples + 1),
        'quarter': ['2020Q1'] * n_samples,
        'new_works': np.random.randint(1, 20, n_samples),
        'novel_terms': np.random.randint(0, 10, n_samples),
        'novelty_rate': np.random.rand(n_samples),
        'cross_domain_share': np.random.rand(n_samples),
        'cross_domain_refs': np.random.randint(0, 50, n_samples),
        'within_lineage_refs': np.random.randint(0, 100, n_samples),
        'cd_index': np.random.rand(n_samples) * 2 - 1,  # -1 to 1
        'cd_min': np.random.rand(n_samples) * 2 - 1,
        'cd_max': np.random.rand(n_samples) * 2 - 1,
        'n_papers_cd': np.random.randint(1, 20, n_samples),
        'dormancy_length': np.random.randint(0, 5, n_samples),
        'awakening_intensity': np.random.rand(n_samples),
        'n_new_papers': np.random.randint(1, 20, n_samples),
        # Additional columns needed by engineer_features()
        'semantic_velocity': np.random.rand(n_samples),
        'total_works': np.random.randint(1, 100, n_samples),
        'growth_rate': np.random.rand(n_samples) * 0.2 - 0.1,
    }

    # Context features (optional)
    if include_context:
        for metric in ['author_influx', 'citation_velocity', 'reference_vitality',
                      'topic_diversity', 'cross_cluster_bridging']:
            data[f'{metric}_z'] = np.random.randn(n_samples)
            data[f'{metric}_qoq_delta'] = np.random.randn(n_samples) * 0.1
            data[f'{metric}_roll_1q'] = np.random.rand(n_samples)
            data[f'{metric}_roll_2q'] = np.random.rand(n_samples)
            data[f'{metric}_roll_4q'] = np.random.rand(n_samples)
            data[f'{metric}_max_dev_4q'] = np.random.rand(n_samples) * 2
            data[f'{metric}_min_dev_4q'] = -np.random.rand(n_samples) * 2

    return pd.DataFrame(data)


def test_feature_selection():
    """Test that select_features() detects context features correctly."""
    print("=" * 70)
    print("SMOKE TEST: MSD Context Feature Integration")
    print("=" * 70)

    # Import the select_features function from MSD
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from multi_signal_detector import select_features, engineer_features

    print("\n[TEST 1] Baseline mode (no context features)")
    print("-" * 70)
    df_baseline = create_test_features(include_context=False)
    df_baseline = engineer_features(df_baseline)
    X_baseline, features_baseline = select_features(df_baseline)

    print(f"\nResult: {len(features_baseline)} features selected")
    assert len(features_baseline) >= 20, "Should have at least 20 core features"
    assert 'author_influx_z' not in features_baseline, "Should not have context features"
    print("[PASS] Baseline mode works correctly\n")

    print("\n[TEST 2] Enriched mode (with context features)")
    print("-" * 70)
    df_enriched = create_test_features(include_context=True)
    df_enriched = engineer_features(df_enriched)
    X_enriched, features_enriched = select_features(df_enriched)

    print(f"\nResult: {len(features_enriched)} features selected")
    assert len(features_enriched) > len(features_baseline), "Should have more features with context"
    assert 'author_influx_z' in features_enriched, "Should include context features"

    # Check that all 5 metrics are represented
    context_count = sum(1 for f in features_enriched if any(
        f.startswith(m) for m in ['author_influx', 'citation_velocity',
                                   'reference_vitality', 'topic_diversity',
                                   'cross_cluster_bridging']
    ))
    print(f"Context features found: {context_count}")
    assert context_count == 35, f"Expected 35 context features, got {context_count}"
    print("[PASS] Enriched mode works correctly\n")

    print("\n[TEST 3] Feature counts")
    print("-" * 70)
    baseline_core = len(features_baseline)
    enriched_core = len(features_baseline)
    enriched_context = len(features_enriched) - len(features_baseline)

    print(f"Baseline features:  {baseline_core}")
    print(f"Enriched total:     {len(features_enriched)}")
    print(f"   Core:            {enriched_core}")
    print(f"   Context:         {enriched_context}")
    print(f"Expected context:   35")

    assert enriched_context == 35, f"Expected 35 context features, got {enriched_context}"
    print("[PASS] Feature counts correct\n")

    print("=" * 70)
    print("[SUCCESS] All tests passed!")
    print("=" * 70)
    print("\nNext steps:")
    print("1. Regenerate lineage_multisignal_features.csv with --enable-context-features")
    print("2. Retrain MSD model with enriched features")
    print("3. Compare performance (baseline vs enriched)")


if __name__ == '__main__':
    test_feature_selection()
