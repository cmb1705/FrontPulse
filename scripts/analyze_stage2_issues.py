#!/usr/bin/env python3
"""Quick analysis of Phase 2 similarity issues."""

import pandas as pd
import yaml

# 1. Check anchor DOI distribution
print("=" * 60)
print("ANCHOR DOI DISTRIBUTION")
print("=" * 60)
with open('config/front_aliases.yaml') as f:
    config = yaml.safe_load(f)

fronts_with_anchors = [
    (f, len(info.get('anchor_dois', [])))
    for f, info in config.items()
    if isinstance(info, dict) and 'anchor_dois' in info
]
fronts_with_anchors.sort(key=lambda x: x[1], reverse=True)

total_dois = sum(count for _, count in fronts_with_anchors)
print(f"\nTotal anchor DOIs across all fronts: {total_dois}")
print(f"\nDOIs per front:")
for front, count in fronts_with_anchors:
    print(f"  {front:35s}: {count:3d} DOIs")

# 2. Analyze top-2 margins
print("\n" + "=" * 60)
print("TOP-2 MARGIN ANALYSIS")
print("=" * 60)
sim = pd.read_csv('data/out/03_milestone_mapping/lineage_front_similarity.csv', index_col='lineage_id')

margins = []
for idx in sim.index:
    row = sim.loc[idx]
    top2 = row.nlargest(2)
    margin = top2.iloc[0] - top2.iloc[1]
    margins.append(margin)

print(f"\nAverage margin between top-2 fronts: {sum(margins)/len(margins):.6f}")
print(f"Max margin: {max(margins):.6f}")
print(f"Min margin: {min(margins):.6f}")
print(f"Lineages with margin < 0.01: {sum(1 for m in margins if m < 0.01)}/99")
print(f"Lineages with margin < 0.02: {sum(1 for m in margins if m < 0.02)}/99")

# 3. Check front centroid distances
print("\n" + "=" * 60)
print("FRONT CENTROID SIMILARITY (how similar are fronts to each other?)")
print("=" * 60)

# Load embeddings
import numpy as np
data = np.load('data/out/02_lineage_tracking/lineage_embeddings.npz', allow_pickle=True)
front_centroids = data['front_centroids'].item()

# Compute pairwise cosine similarities between front centroids
from numpy.linalg import norm

front_names = list(front_centroids.keys())
centroid_sims = {}

for i, front1 in enumerate(front_names):
    for front2 in front_names[i+1:]:
        emb1 = front_centroids[front1]
        emb2 = front_centroids[front2]

        # Cosine similarity
        sim = np.dot(emb1, emb2) / (norm(emb1) * norm(emb2))
        centroid_sims[(front1, front2)] = sim

# Sort by similarity (most similar pairs first)
sorted_pairs = sorted(centroid_sims.items(), key=lambda x: x[1], reverse=True)

print(f"\nMost similar front pairs (top 10):")
for (f1, f2), sim in sorted_pairs[:10]:
    print(f"  {f1:30s} <-> {f2:30s}: {sim:.4f}")

print(f"\nLeast similar front pairs (bottom 10):")
for (f1, f2), sim in sorted_pairs[-10:]:
    print(f"  {f1:30s} <-> {f2:30s}: {sim:.4f}")

print(f"\nAverage inter-front similarity: {sum(centroid_sims.values())/len(centroid_sims):.4f}")
print(f"Min inter-front similarity: {min(centroid_sims.values()):.4f}")
print(f"Max inter-front similarity: {max(centroid_sims.values()):.4f}")
