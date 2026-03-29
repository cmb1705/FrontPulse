#!/usr/bin/env python3
"""Analyze Phase 2 and Phase 3 thresholds for optimal coverage."""

import numpy as np
import pandas as pd

# Phase 2 Analysis
print("="*70)
print("PHASE 2 (SciBERT) THRESHOLD ANALYSIS")
print("="*70)
print("Goal: ~50 matches per front (50% coverage)\n")

df2 = pd.read_csv('data/out/03_milestone_mapping/lineage_front_similarity.csv')
similarities = df2.iloc[:, 1:].values.flatten()

print("Current Similarity Distribution:")
print(f"  Min: {similarities.min():.3f}")
print(f"  25th percentile: {np.percentile(similarities, 25):.3f}")
print(f"  Median: {np.median(similarities):.3f}")
print(f"  75th percentile: {np.percentile(similarities, 75):.3f}")
print(f"  90th percentile: {np.percentile(similarities, 90):.3f}")
print(f"  95th percentile: {np.percentile(similarities, 95):.3f}")
print(f"  Max: {similarities.max():.3f}")

# Test different thresholds
print("\nTesting thresholds for ~50 matches per front:")
for percentile in [45, 50, 55, 60, 65]:
    threshold = np.percentile(similarities, percentile)
    total_matches = 0
    for col in df2.columns[1:]:
        matches = (df2[col] > threshold).sum()
        total_matches += matches
    avg_matches = total_matches / 16
    print(f"  Threshold {threshold:.3f} ({percentile}th percentile): {avg_matches:.1f} matches/front")

# Recommend threshold
recommended = np.percentile(similarities, 50)
print(f"\nRecommended threshold: {recommended:.3f}")
print("Matches per front at recommended threshold:")
for col in df2.columns[1:]:
    matches = (df2[col] > recommended).sum()
    print(f"  {col:30s}: {matches:2d} matches")

# Phase 3 Analysis
print("\n" + "="*70)
print("PHASE 3 (c-TF-IDF) THRESHOLD ANALYSIS")
print("="*70)
print("Goal: 20-40% coverage (20-40 lineages matched per front)\n")

df3 = pd.read_csv('data/out/03_milestone_mapping/lineage_front_term_similarity.csv')
similarities3 = df3.iloc[:, 1:].values.flatten()
nonzero = similarities3[similarities3 > 0]

print("Current Similarity Distribution (non-zero only):")
print(f"  Min (>0): {nonzero.min():.4f}")
print(f"  25th percentile: {np.percentile(nonzero, 25):.4f}")
print(f"  Median: {np.median(nonzero):.4f}")
print(f"  75th percentile: {np.percentile(nonzero, 75):.4f}")
print(f"  Max: {nonzero.max():.4f}")
print(f"  Total non-zero: {len(nonzero)} / {len(similarities3)} ({len(nonzero)/len(similarities3)*100:.1f}%)")

# Test different thresholds
print("\nTesting thresholds for 20-40 matches per front:")
for threshold in [0.001, 0.002, 0.003, 0.005, 0.007, 0.010]:
    total_matches = 0
    for col in df3.columns[1:]:
        matches = (df3[col] > threshold).sum()
        total_matches += matches
    avg_matches = total_matches / 16
    coverage_pct = (total_matches / (99 * 16)) * 100
    print(f"  Threshold {threshold:.4f}: {avg_matches:.1f} matches/front ({coverage_pct:.1f}% coverage)")

# Recommend threshold
recommended3 = 0.003
print(f"\nRecommended threshold: {recommended3:.4f}")
print("Matches per front at recommended threshold:")
for col in df3.columns[1:]:
    matches = (df3[col] > recommended3).sum()
    print(f"  {col:30s}: {matches:2d} matches")
