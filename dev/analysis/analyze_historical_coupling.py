"""Analyze historical coupling edge data from archive."""
import json
from pathlib import Path

import pandas as pd

print("=" * 80)
print("HISTORICAL COUPLING EDGE ANALYSIS")
print("=" * 80)

# Load historical coupling cache
archive_path = Path("data/archive_before_reorganization_20251101_172112/out_original/cache_coupling")

# Check config
config_file = archive_path / "coupling_config.json"
if config_file.exists():
    print("\nHistorical coupling configuration:")
    with open(config_file) as f:
        config = json.load(f)
    print(f"  min_shared_refs: {config.get('min_shared_refs', 'N/A')}")
    print(f"  min_coupling_score: {config.get('min_coupling_score', 'N/A')}")
    print(f"  alpha: {config.get('alpha', 'N/A')}")
    print(f"  beta: {config.get('beta', 'N/A')}")

# Load coupling edges
edges_file = archive_path / "coupling_edges.parquet"
if edges_file.exists():
    file_size_mb = edges_file.stat().st_size / (1024 ** 2)
    print("\nHistorical coupling_edges.parquet:")
    print(f"  File size: {file_size_mb:.2f} MB")

    print("\n  Loading parquet file...")
    df = pd.read_parquet(edges_file)

    print(f"  Total coupling edges: {len(df):,}")
    print(f"\n  Columns: {list(df.columns)}")

    if len(df) > 0:
        print("\n  Sample edges (first 5):")
        print(df.head().to_string())

        # Statistics
        if 'score' in df.columns:
            print("\n  Score statistics:")
            print(f"    Mean: {df['score'].mean():.4f}")
            print(f"    Median: {df['score'].median():.4f}")
            print(f"    Min: {df['score'].min():.4f}")
            print(f"    Max: {df['score'].max():.4f}")

        if 'shared_refs' in df.columns:
            print("\n  Shared refs statistics:")
            print(f"    Mean: {df['shared_refs'].mean():.1f}")
            print(f"    Median: {df['shared_refs'].median():.1f}")
            print(f"    Min: {df['shared_refs'].min()}")
            print(f"    Max: {df['shared_refs'].max()}")

# Load nodes
nodes_file = archive_path / "coupling_nodes.json"
if nodes_file.exists():
    file_size_mb = nodes_file.stat().st_size / (1024 ** 2)
    print("\nHistorical coupling_nodes.json:")
    print(f"  File size: {file_size_mb:.2f} MB")

    with open(nodes_file) as f:
        nodes = json.load(f)

    print(f"  Total nodes: {len(nodes):,}")

    if len(nodes) > 0 and isinstance(nodes, list):
        print(f"\n  Nodes stored as list (first 3): {nodes[:3]}")
    elif len(nodes) > 0 and isinstance(nodes, dict):
        sample_node = list(nodes.keys())[0]
        print(f"\n  Sample node ({sample_node}):")
        print(f"    {nodes[sample_node]}")

print("\n" + "=" * 80)
print("COMPARISON WITH CURRENT BUILD")
print("=" * 80)

# Compare with current 2018Q1
current_graph = Path("data/current_graphs/citation_graph_cumulative_2018Q1.pkl")
if current_graph.exists():
    import pickle
    with open(current_graph, 'rb') as f:
        G = pickle.load(f)

    current_edges = G.number_of_edges()
    current_nodes = G.number_of_nodes()

    print("\nCurrent 2018Q1 graph:")
    print(f"  Nodes: {current_nodes:,}")
    print(f"  Total edges: {current_edges:,}")
    print(f"  Avg edges/node: {current_edges/current_nodes:.1f}")

    # Extract coupling vs citation breakdown
    if "coupling_stats" in G.graph:
        stats = G.graph["coupling_stats"]
        coupling_edges_count = stats.get('coupling_edges', 0)
        citation_edges_count = stats.get('citation_edges', 0)
        hybrid_edges_count = stats.get('hybrid_edges', 0)

        print("\n  Edge breakdown:")
        print(f"    Coupling edges: {coupling_edges_count:,}")
        print(f"    Citation edges: {citation_edges_count:,}")
        print(f"    Hybrid edges: {hybrid_edges_count:,}")

        if edges_file.exists():
            historical_coupling = len(df)
            print("\n  Historical vs Current coupling edges:")
            print(f"    Historical: {historical_coupling:,}")
            print(f"    Current: {coupling_edges_count:,}")
            if historical_coupling > 0:
                ratio = coupling_edges_count / historical_coupling
                print(f"    Ratio: {ratio:.2f}x {'(INCREASE)' if ratio > 1 else '(DECREASE)'}")

print("\n" + "=" * 80)
