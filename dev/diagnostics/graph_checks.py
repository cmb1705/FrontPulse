# graph_checks.py
import pickle
from pathlib import Path

import networkx as nx
import pandas as pd


def check_graph(path):
    with open(path, "rb") as f:
        G = pickle.load(f)
    n = G.number_of_nodes()
    m = G.number_of_edges()
    # Basic
    self_loops = nx.number_of_selfloops(G)
    # Duplicate edges: DiGraph forbids duplicates; MultiDiGraph would allow. If Multi, collapse:
    is_multi = isinstance(G, nx.MultiDiGraph)
    dup_edges = 0
    if is_multi:
        dup_edges = sum(1 for u,v in G.edges() if G.number_of_edges(u,v) > 1)
    # Orphan edges (shouldn’t exist in DiGraph)
    missing = 0
    for u,v in G.edges():
        if not (G.has_node(u) and G.has_node(v)):
            missing += 1
            break
    return {"nodes": n, "edges": m, "self_loops": self_loops, "is_multi": is_multi, "dup_edges": dup_edges, "missing_endpoints": missing}

# Annual cumulative growth
annual = []
for p in sorted((Path("data/out/graphs")).glob("citation_graph_annual_*.pkl")):
    y = int(p.stem.split("_")[-1])
    r = check_graph(p)
    r["year"] = y
    annual.append(r)
annual_df = pd.DataFrame(annual).sort_values("year")
print("\nAnnual graphs:")
print(annual_df[["year","nodes","edges","self_loops","dup_edges","missing_endpoints"]])

# Quarterly deltas: ensure “new-only” is respected and endpoints exist
deltas = []
for p in sorted((Path("data/out/graphs")).glob("citation_graph_delta_*.pkl")):
    q = p.stem.split("_")[-1]
    r = check_graph(p)
    r["quarter"] = q
    deltas.append(r)
deltas_df = pd.DataFrame(deltas).sort_values("quarter")
print("\nDelta graphs:")
print(deltas_df[["quarter","nodes","edges","self_loops","dup_edges","missing_endpoints"]])

annual_df.to_csv("data/out/graph_check_annual.csv", index=False)
deltas_df.to_csv("data/out/graph_check_deltas.csv", index=False)
