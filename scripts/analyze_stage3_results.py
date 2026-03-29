#!/usr/bin/env python3
"""Analyze Phase 3 c-TF-IDF results."""

from collections import Counter

import numpy as np
import pandas as pd

# Load data
terms_df = pd.read_csv('data/out/02_lineage_tracking/lineage_ctfidf_terms.csv')
sim_df = pd.read_csv('data/out/03_milestone_mapping/lineage_front_term_similarity.csv')

# Calculate statistics
front_cols = [col for col in sim_df.columns if col != 'lineage_id']
results = []

for _idx, row in sim_df.iterrows():
    lineage_id = int(row['lineage_id'])
    similarities = row[front_cols].values
    max_sim = float(similarities.max())
    best_front = front_cols[similarities.argmax()]

    results.append({
        'lineage_id': lineage_id,
        'max_similarity': max_sim,
        'best_front': best_front
    })

results_sorted = sorted(results, key=lambda x: x['max_similarity'], reverse=True)

# Write to file to avoid encoding issues
with open('data/out/phase3_analysis.txt', 'w', encoding='utf-8') as f:
    # Bottom 10
    f.write('=' * 100 + '\n')
    f.write('BOTTOM 10 LINEAGES (LIKELY OFF-TOPIC)\n')
    f.write('=' * 100 + '\n\n')

    for i, result in enumerate(results_sorted[-10:], 1):
        lin_id = result['lineage_id']
        max_sim = result['max_similarity']
        best = result['best_front']

        # Get top 5 terms
        lin_terms = terms_df[terms_df['lineage_id'] == lin_id].head(5)
        terms_list = ', '.join(lin_terms['term'].tolist())

        f.write(f'{i:2d}. Lineage {lin_id:3d} - Max similarity: {max_sim:.4f} ({best})\n')
        f.write(f'    Top terms: {terms_list}\n\n')

    # Distribution of best fronts
    f.write('=' * 100 + '\n')
    f.write('DISTRIBUTION OF BEST FRONT MATCHES (ALL 99 LINEAGES)\n')
    f.write('=' * 100 + '\n\n')

    front_counts = Counter([r['best_front'] for r in results])

    for front, count in sorted(front_counts.items(), key=lambda x: x[1], reverse=True):
        pct = 100 * count / len(results)
        bar = '#' * int(pct / 2)
        f.write(f'{front:35s}: {count:3d} ({pct:5.1f}%) {bar}\n')

    f.write(f'\nTotal lineages: {len(results)}\n\n')

    # Summary statistics
    f.write('=' * 100 + '\n')
    f.write('SUMMARY STATISTICS\n')
    f.write('=' * 100 + '\n\n')

    sims = [r['max_similarity'] for r in results]
    f.write('Max similarity across all lineages:\n')
    f.write(f'  - Mean:   {np.mean(sims):.4f}\n')
    f.write(f'  - Median: {np.median(sims):.4f}\n')
    f.write(f'  - Min:    {np.min(sims):.4f}\n')
    f.write(f'  - Max:    {np.max(sims):.4f}\n')
    f.write(f'  - Std:    {np.std(sims):.4f}\n\n')

    # Similarity thresholds
    f.write('Lineages by similarity threshold:\n')
    f.write(f'  - > 0.0100 (strong match):   {sum(1 for s in sims if s > 0.01):3d} ({100*sum(1 for s in sims if s > 0.01)/len(sims):5.1f}%)\n')
    f.write(f'  - > 0.0050 (moderate match): {sum(1 for s in sims if s > 0.005):3d} ({100*sum(1 for s in sims if s > 0.005)/len(sims):5.1f}%)\n')
    f.write(f'  - > 0.0025 (weak match):     {sum(1 for s in sims if s > 0.0025):3d} ({100*sum(1 for s in sims if s > 0.0025)/len(sims):5.1f}%)\n')
    f.write(f'  - < 0.0025 (likely off-topic): {sum(1 for s in sims if s <= 0.0025):3d} ({100*sum(1 for s in sims if s <= 0.0025)/len(sims):5.1f}%)\n')

print('Analysis written to data/out/phase3_analysis.txt')
