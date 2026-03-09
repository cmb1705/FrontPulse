"""
Analyze lineage stratification by lifetime and temporal evolution.

Examines:
- Total lineage count
- Distribution by lifetime quartiles
- Temporal context (pre-foundational, critical expansion, mature)
- Filtering thresholds and their impact
"""

import pandas as pd
from pathlib import Path
from collections import defaultdict
import json

print("=" * 80)
print("LINEAGE STRATIFICATION ANALYSIS")
print("=" * 80)

# Load lineage timeseries
df = pd.read_csv('data/out/02_lineage_tracking/lineage_timeseries.csv')

# Compute lifetime for each lineage (number of quarters it appears)
lineage_lifetime = df.groupby('lineage_id').size().reset_index(name='lifetime_quarters')

# Get first and last quarter for each lineage
lineage_first = df.groupby('lineage_id')['quarter'].min().reset_index(name='first_quarter')
lineage_last = df.groupby('lineage_id')['quarter'].max().reset_index(name='last_quarter')

# Merge
lineages = lineage_lifetime.merge(lineage_first, on='lineage_id').merge(lineage_last, on='lineage_id')

# Extract year from first quarter
def parse_year(q_str):
    """Extract year from quarter string like '2010Q1'."""
    try:
        return int(q_str[:4])
    except:
        return None

lineages['first_year'] = lineages['first_quarter'].apply(parse_year)

print(f"\nTotal lineages tracked: {len(lineages)}")
print(f"Quarters covered: {df['quarter'].min()} to {df['quarter'].max()}")

# Lifetime distribution
print("\n" + "-" * 80)
print("LIFETIME DISTRIBUTION")
print("-" * 80)

# Define brackets
brackets = [
    (0, 4, "Very short (0-4 quarters)"),
    (5, 9, "Short (5-9 quarters)"),
    (10, 19, "Medium (10-19 quarters)"),
    (20, 39, "Long (20-39 quarters)"),
    (40, 79, "Very long (40-79 quarters)"),
    (80, 999, "Persistent (80+ quarters)")
]

print(f"\n{'Lifetime Range':<30} {'Count':<10} {'Percentage':<12} {'Cumulative'}")
print("-" * 80)

cumulative = 0
for min_q, max_q, label in brackets:
    count = len(lineages[(lineages['lifetime_quarters'] >= min_q) & (lineages['lifetime_quarters'] <= max_q)])
    pct = 100 * count / len(lineages) if len(lineages) > 0 else 0
    cumulative += count
    cum_pct = 100 * cumulative / len(lineages) if len(lineages) > 0 else 0
    print(f"{label:<30} {count:<10} {pct:>6.1f}%      {cumulative:>5} ({cum_pct:>5.1f}%)")

# Filtering analysis
print("\n" + "-" * 80)
print("FILTERING THRESHOLDS")
print("-" * 80)

thresholds = [1, 5, 10, 15, 20, 30, 40, 50]
print(f"\n{'Min Quarters':<15} {'Remaining':<12} {'Filtered Out':<15} {'% Remaining'}")
print("-" * 80)

for threshold in thresholds:
    remaining = len(lineages[lineages['lifetime_quarters'] >= threshold])
    filtered = len(lineages) - remaining
    pct_remaining = 100 * remaining / len(lineages) if len(lineages) > 0 else 0
    print(f"{threshold:>3}+ quarters     {remaining:<12} {filtered:<15} {pct_remaining:>6.1f}%")

# Temporal evolution analysis
print("\n" + "-" * 80)
print("TEMPORAL EVOLUTION")
print("-" * 80)

# Define periods based on perovskite PSC history
periods = [
    (2000, 2008, "Pre-foundational (<2009)"),
    (2009, 2012, "Early period (2009-2012)"),
    (2013, 2016, "Critical expansion (2013-2016)"),
    (2017, 2020, "Mature period (2017-2020)"),
    (2021, 2025, "Recent period (2021+)")
]

print(f"\n{'Period':<35} {'Total':<8} {'5Q+':<8} {'20Q+':<8} {'40Q+'}")
print("-" * 80)

for min_year, max_year, label in periods:
    period_df = lineages[(lineages['first_year'] >= min_year) & (lineages['first_year'] <= max_year)]
    total = len(period_df)
    q5 = len(period_df[period_df['lifetime_quarters'] >= 5])
    q20 = len(period_df[period_df['lifetime_quarters'] >= 20])
    q40 = len(period_df[period_df['lifetime_quarters'] >= 40])
    print(f"{label:<35} {total:<8} {q5:<8} {q20:<8} {q40}")

# Critical expansion period analysis
print("\n" + "-" * 80)
print("DETAILED: CRITICAL EXPANSION PERIOD (2013-2016)")
print("-" * 80)

expansion_lineages = lineages[(lineages['first_year'] >= 2013) & (lineages['first_year'] <= 2016)]
print(f"\nTotal lineages started in 2013-2016: {len(expansion_lineages)}")

for threshold in [5, 10, 20, 30]:
    filtered = expansion_lineages[expansion_lineages['lifetime_quarters'] >= threshold]
    print(f"  - {threshold}+ quarters: {len(filtered)} lineages ({100*len(filtered)/len(expansion_lineages):.1f}%)")

# Key recommendations
print("\n" + "-" * 80)
print("RECOMMENDATIONS")
print("-" * 80)

# After filtering 0-4 quarters
filtered_5plus = lineages[lineages['lifetime_quarters'] >= 5]
print(f"\nFiltering out 0-4 quarter lineages (short-lived):")
print(f"  - Removes: {len(lineages) - len(filtered_5plus)} lineages ({100*(len(lineages) - len(filtered_5plus))/len(lineages):.1f}%)")
print(f"  - Keeps: {len(filtered_5plus)} lineages ({100*len(filtered_5plus)/len(lineages):.1f}%)")

# Critical expansion period (2013-2016) with 5+ quarters
expansion_5q = expansion_lineages[expansion_lineages['lifetime_quarters'] >= 5]
print(f"\nCritical expansion period (2013-2016) with 5+ quarters:")
print(f"  - Count: {len(expansion_5q)} lineages")
print(f"  - Mean lifetime: {expansion_5q['lifetime_quarters'].mean():.1f} quarters")
print(f"  - Median lifetime: {expansion_5q['lifetime_quarters'].median():.1f} quarters")
print(f"  - Max lifetime: {expansion_5q['lifetime_quarters'].max():.0f} quarters")

# Current analysis (20+ quarters)
current = lineages[lineages['lifetime_quarters'] >= 20]
print(f"\nCurrent analysis threshold (20+ quarters):")
print(f"  - Count: {len(current)} lineages")
print(f"  - This represents {100*len(current)/len(lineages):.1f}% of all lineages")
print(f"  - This represents {100*len(current)/len(filtered_5plus):.1f}% of 5+ quarter lineages")

# Distribution of current 20+ set by first year
print(f"\nTemporal distribution of 20+ quarter lineages:")
for min_year, max_year, label in periods:
    period_count = len(current[(current['first_year'] >= min_year) & (current['first_year'] <= max_year)])
    print(f"  - {label}: {period_count} lineages")

print("\n" + "=" * 80)

# Export summary
summary = {
    'total_lineages': int(len(lineages)),
    'quarters_covered': {
        'first': str(df['quarter'].min()),
        'last': str(df['quarter'].max())
    },
    'by_lifetime': {label: int(len(lineages[(lineages['lifetime_quarters'] >= min_q) & (lineages['lifetime_quarters'] <= max_q)]))
                    for min_q, max_q, label in brackets},
    'by_period': {label: {
        'total': int(len(lineages[(lineages['first_year'] >= min_year) & (lineages['first_year'] <= max_year)])),
        '5q+': int(len(lineages[(lineages['first_year'] >= min_year) & (lineages['first_year'] <= max_year) & (lineages['lifetime_quarters'] >= 5)])),
        '20q+': int(len(lineages[(lineages['first_year'] >= min_year) & (lineages['first_year'] <= max_year) & (lineages['lifetime_quarters'] >= 20)]))
    } for min_year, max_year, label in periods},
    'thresholds': {
        f'{t}q+': int(len(lineages[lineages['lifetime_quarters'] >= t]))
        for t in [1, 5, 10, 15, 20, 30, 40, 50]
    }
}

output_path = Path("data/out/02_lineage_tracking/lineage_stratification_summary.json")
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(summary, f, indent=2)

print(f"\nSaved summary to {output_path}")
