"""Compare edge/node counts across quarters to detect bloat."""
import pickle
from pathlib import Path

import pandas as pd

graph_dir = Path("data/current_graphs")

# Sample quarters to check
sample_quarters = [
    "2010Q1", "2012Q1", "2014Q1", "2016Q1", "2017Q1", "2018Q1"
]

print("=" * 80)
print("GRAPH SIZE COMPARISON - Edge/Node Analysis")
print("=" * 80)

results = []

for qtr in sample_quarters:
    graph_file = graph_dir / f"citation_graph_cumulative_{qtr}.pkl"

    if not graph_file.exists():
        print(f"\n{qtr}: FILE NOT FOUND")
        continue

    file_size_mb = graph_file.stat().st_size / (1024 ** 2)

    print(f"\nLoading {qtr}...")
    with open(graph_file, 'rb') as f:
        G = pickle.load(f)

    nodes = G.number_of_nodes()
    edges = G.number_of_edges()
    avg_edges_per_node = edges / nodes if nodes > 0 else 0

    # Check attributes
    sample_node = list(G.nodes())[0] if nodes > 0 else None
    has_referenced_works = False
    if sample_node:
        has_referenced_works = 'referenced_works' in G.nodes[sample_node]

    results.append({
        'quarter': qtr,
        'nodes': nodes,
        'edges': edges,
        'avg_edges_per_node': avg_edges_per_node,
        'file_size_mb': file_size_mb,
        'mb_per_1k_edges': (file_size_mb / (edges / 1000)) if edges > 0 else 0,
        'has_referenced_works': has_referenced_works
    })

    print(f"  Nodes: {nodes:,}")
    print(f"  Edges: {edges:,}")
    print(f"  Avg edges/node: {avg_edges_per_node:.1f}")
    print(f"  File size: {file_size_mb:.2f} MB")
    print(f"  MB per 1K edges: {(file_size_mb / (edges / 1000)):.3f}")
    print(f"  Has referenced_works: {has_referenced_works}")

# Summary table
print("\n" + "=" * 80)
print("SUMMARY TABLE")
print("=" * 80)

df = pd.DataFrame(results)
print(df.to_string(index=False))

# Detect anomalies
print("\n" + "=" * 80)
print("ANOMALY DETECTION")
print("=" * 80)

# Check if avg edges/node is increasing over time
if len(results) >= 2:
    first_avg = results[0]['avg_edges_per_node']
    last_avg = results[-1]['avg_edges_per_node']
    growth_factor = last_avg / first_avg if first_avg > 0 else 0

    print("\nEdges per node growth:")
    print(f"  {results[0]['quarter']}: {first_avg:.1f} edges/node")
    print(f"  {results[-1]['quarter']}: {last_avg:.1f} edges/node")
    print(f"  Growth factor: {growth_factor:.2f}x")

    if growth_factor > 3.0:
        print(f"  [WARNING] Edges/node grew by {growth_factor:.1f}x - possible edge bloat!")
    elif growth_factor > 2.0:
        print(f"  [CAUTION] Edges/node grew by {growth_factor:.1f}x - monitor closely")
    else:
        print("  [OK] Growth is reasonable for cumulative graphs")

# Check MB per 1K edges consistency
mb_per_1k = [r['mb_per_1k_edges'] for r in results]
avg_mb_per_1k = sum(mb_per_1k) / len(mb_per_1k)
print("\nFile size efficiency (MB per 1K edges):")
print(f"  Average: {avg_mb_per_1k:.3f} MB/1K edges")
print(f"  Range: {min(mb_per_1k):.3f} - {max(mb_per_1k):.3f}")

# Check for referenced_works
if any(r['has_referenced_works'] for r in results):
    print("\n[CRITICAL] Some graphs still have referenced_works - memory bloat!")
else:
    print("\n[OK] All graphs have referenced_works stripped")

print("\n" + "=" * 80)
