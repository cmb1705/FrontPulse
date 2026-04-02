"""Check year distribution in historical coupling data."""
from pathlib import Path

import pandas as pd

print("=" * 80)
print("HISTORICAL COUPLING YEAR ANALYSIS")
print("=" * 80)

archive_path = Path("data/archive_before_reorganization_20251101_172112/out_original/cache_coupling")
edges_file = archive_path / "coupling_edges.parquet"

print(f"\nLoading {edges_file.name}...")
df = pd.read_parquet(edges_file)

print(f"Total edges: {len(df):,}")

# Year distribution
print("\nYear range in edges:")
print(f"  year_a: {df['year_a'].min()} - {df['year_a'].max()}")
print(f"  year_b: {df['year_b'].min()} - {df['year_b'].max()}")

# Count edges by max year (cumulative perspective)
df['max_year'] = df[['year_a', 'year_b']].max(axis=1)

print("\nEdge count by maximum year (cumulative view):")
year_counts = df['max_year'].value_counts().sort_index()
print(year_counts.tail(20))

# How many edges involve papers from 2018 or earlier?
edges_up_to_2018 = df[(df['year_a'] <= 2018) & (df['year_b'] <= 2018)]
print("\n" + "=" * 80)
print("Edges where BOTH papers are from 2018 or earlier:")
print(f"  Count: {len(edges_up_to_2018):,}")
print(f"  Percentage: {len(edges_up_to_2018) / len(df) * 100:.1f}%")

# Node count up to 2018
nodes_file = archive_path / "coupling_nodes.json"
if nodes_file.exists():
    import json
    with open(nodes_file) as f:
        nodes = json.load(f)

    # Nodes are stored as list, but we need to check years from edges
    nodes_in_2018_edges = set(edges_up_to_2018['node_a']).union(set(edges_up_to_2018['node_b']))
    print(f"\nNodes involved in pre-2019 edges: {len(nodes_in_2018_edges):,}")

print("\n" + "=" * 80)
print("INTERPRETATION")
print("=" * 80)

print("\nIf historical cache was for full dataset (up to 2023):")
print(f"  - Then only {len(edges_up_to_2018):,} coupling edges up to 2018Q4")
print("  - Current 2018Q1 has 16.2M coupling edges")
print(f"  - That's {16242733 / len(edges_up_to_2018):.1f}x MORE edges in current build!")
print("\n  VERDICT: Current build has MASSIVE edge bloat compared to historical")

print("\n" + "=" * 80)
