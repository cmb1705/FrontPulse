#!/usr/bin/env python3
"""Analyze lineage-to-front assignments across all phases."""

import pandas as pd
from pathlib import Path

# Define thresholds for each phase
THRESHOLDS = { 
    'Phase 2 (SciBERT)': 0.747,     # Optimized for ~50% coverage with all fronts
    'Phase 3 (c-TF-IDF)': 0.002,    # Optimized for 20-40% coverage
    'Phase 4 (NPMI)': 0.0           # Use all non-zero values
}

# Load similarity matrices (Phase 2 may not exist yet)
phases = {}

phase2_path = Path('data/out/03_milestone_mapping/lineage_front_similarity.csv')
if phase2_path.exists():
    phases['Phase 2 (SciBERT)'] = pd.read_csv(phase2_path)
else:
    print("NOTE: Phase 2 similarity matrix not found. Run compute_lineage_embeddings.py to generate.")

phase3_path = Path('data/out/03_milestone_mapping/lineage_front_term_similarity.csv')
if phase3_path.exists():
    phases['Phase 3 (c-TF-IDF)'] = pd.read_csv(phase3_path)

phase4_path = Path('data/out/03_milestone_mapping/lineage_front_npmi_similarity.csv')
if phase4_path.exists():
    phases['Phase 4 (NPMI)'] = pd.read_csv(phase4_path)

if not phases:
    print("ERROR: No similarity matrices found!")
    exit(1)

# Get front columns from first available phase
first_df = list(phases.values())[0]
front_cols = [col for col in first_df.columns if col != 'lineage_id']

print('=' * 80)
print('PRELIMINARY ASSIGNMENTS (Before Phase 5 Final Scoring)')
print('=' * 80)

# For each phase, show lineages per front
for phase_name, df in phases.items():
    print(f'\n{phase_name}:')
    print('-' * 80)

    threshold = THRESHOLDS.get(phase_name, 0.0)

    for front in front_cols:
        # Get lineages above threshold, sorted
        matches = df[df[front] > threshold][['lineage_id', front]].sort_values(front, ascending=False)

        if len(matches) > 0:
            top_5 = matches.head(5)
            lineage_list = ', '.join([f'L{int(row["lineage_id"])}({row[front]:.3f})'
                                      for _, row in top_5.iterrows()])
            print(f'{front:30s}: {len(matches):3d} matches | Top 5: {lineage_list}')
        else:
            print(f'{front:30s}:   0 matches')

# Summary across all phases
print('\n' + '=' * 80)
print('CROSS-PHASE SUMMARY')
print('=' * 80)

for front in front_cols:
    summary_parts = []

    if 'Phase 2 (SciBERT)' in phases:
        threshold2 = THRESHOLDS['Phase 2 (SciBERT)']
        matches2 = len(phases['Phase 2 (SciBERT)'][phases['Phase 2 (SciBERT)'][front] > threshold2])
        summary_parts.append(f'P2={matches2:2d}')

    if 'Phase 3 (c-TF-IDF)' in phases:
        threshold3 = THRESHOLDS['Phase 3 (c-TF-IDF)']
        matches3 = len(phases['Phase 3 (c-TF-IDF)'][phases['Phase 3 (c-TF-IDF)'][front] > threshold3])
        summary_parts.append(f'P3={matches3:2d}')

    if 'Phase 4 (NPMI)' in phases:
        threshold4 = THRESHOLDS['Phase 4 (NPMI)']
        matches4 = len(phases['Phase 4 (NPMI)'][phases['Phase 4 (NPMI)'][front] > threshold4])
        summary_parts.append(f'P4={matches4:2d}')

    summary_str = ', '.join(summary_parts)
    print(f'{front:30s}: {summary_str}')
