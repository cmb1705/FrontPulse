import pathlib
import pickle

import networkx as nx

gdir = pathlib.Path("data/out/graphs")
for p in sorted(gdir.glob("*.pkl")):
    with p.open("rb") as handle:
        G = pickle.load(handle)
    n = len(list(nx.selfloop_edges(G)))
    if n:
        G.remove_edges_from(nx.selfloop_edges(G))
        with p.open("wb") as handle:
            pickle.dump(G, handle)
        print(f"{p.name}: removed {n} self-loops")
    # re-export GraphML using the cleaned pickle
    try:
        # mirror your sanitize logic lightly for re-export
        H = G.copy()
        S = nx.DiGraph()
        S.add_nodes_from(H.nodes())
        S.add_edges_from(H.edges())
        nx.write_graphml(S, str(p.with_suffix("")) + ".graphml")
    except Exception as e:
        print(f"{p.name}: graphml export failed: {e}")
