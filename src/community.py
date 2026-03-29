from __future__ import annotations

import math
from collections import defaultdict
from typing import Any

import networkx as nx


def _require_leiden():
    try:
        import igraph as ig  # noqa
        import leidenalg as la  # noqa
    except Exception as e:
        raise RuntimeError(
            "Leiden requires 'python-igraph' and 'leidenalg'. Install: pip install python-igraph leidenalg"
        ) from e

def _nx_to_igraph(G: nx.Graph, weight_attr: str = "weight_total"):
    import igraph as ig
    nid_map = {n: i for i, n in enumerate(G.nodes())}
    g = ig.Graph(n=len(nid_map), directed=G.is_directed())
    g.vs["name"] = list(nid_map.keys())
    if len(nid_map):
        edges = [(nid_map[u], nid_map[v]) for u, v in G.edges()]
        if edges:
            g.add_edges(edges)
            weights: list[float] = []
            for u, v in G.edges():
                data = G.edges[u, v]
                weight = data.get(weight_attr, data.get("weight", 1.0))
                try:
                    weights.append(float(weight))
                except Exception:
                    weights.append(1.0)
            if weights:
                g.es["weight"] = weights
    return g

# Load community defaults from config
def _get_community_defaults():
    """Get community defaults from config/defaults.yaml or use fallbacks."""
    try:
        from .config import get_community_defaults
        return get_community_defaults()
    except Exception:
        # Fallback if config not available
        return {
            "default_resolution": 1.0,
            "min_size": 50,
            "max_size": 5000,
        }

_COMMUNITY_DEFAULTS = _get_community_defaults()

def run_leiden(
    G: nx.Graph,
    resolution: float = _COMMUNITY_DEFAULTS.get("default_resolution", 1.0),
    min_size: int = _COMMUNITY_DEFAULTS.get("min_size", 50),
    max_size: int = _COMMUNITY_DEFAULTS.get("max_size", 5000),
    *,
    use_rbconfig: bool = False,
) -> dict[str, Any]:
    _require_leiden()
    import leidenalg as la
    g = _nx_to_igraph(G, weight_attr="weight_total")
    partition_cls = la.RBConfigurationVertexPartition if use_rbconfig else la.CPMVertexPartition
    weights = g.es["weight"] if "weight" in g.es.attributes() else None
    part = la.find_partition(
        g,
        partition_cls,
        resolution_parameter=resolution,
        weights=weights,
    )
    labels = part.membership
    nodes = g.vs["name"]
    pairs = list(zip(nodes, labels))

    from collections import Counter, defaultdict
    cnt = Counter(labels)
    keep = {cid for cid, sz in cnt.items() if min_size <= sz <= max_size}
    kept_pairs = [(n, c) for (n, c) in pairs if c in keep]

    comm_nodes = defaultdict(list)
    for n, c in kept_pairs:
        comm_nodes[c].append(n)
    communities = [{"id": int(cid), "size": len(v), "nodes": v} for cid, v in comm_nodes.items()]

    return {
        "partition": kept_pairs,
        "communities": communities,
        "raw_n_communities": int(len(cnt)),
        "modularity": float(part.modularity),
    }

def _require_ecg():
    """Check that partition-igraph is installed for ECG support."""
    try:
        import partition_igraph  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "ECG requires 'partition-igraph'. Install: pip install partition-igraph>=0.0.7"
        ) from exc


def run_ecg(
    G: nx.Graph,
    resolution: float = _COMMUNITY_DEFAULTS.get("default_resolution", 1.0),
    min_size: int = _COMMUNITY_DEFAULTS.get("min_size", 50),
    max_size: int = _COMMUNITY_DEFAULTS.get("max_size", 5000),
    *,
    ens_size: int = 16,
    min_weight: float = 0.05,
    final: str = "leiden",
) -> dict[str, Any]:
    """Run ECG ensemble clustering on a NetworkX graph.

    ECG (Ensemble Clustering for Graphs) runs an ensemble of randomized
    single-level Louvain partitions, aggregates co-membership votes into
    edge weights, then runs a final Leiden/Louvain on the re-weighted graph.

    This produces more stable partitions than single-run Leiden because
    the ensemble smooths out stochastic variation.

    Args:
        G: NetworkX citation graph with ``weight_total`` edge attribute.
        resolution: Resolution parameter for the final partition.
        min_size: Minimum community size to retain.
        max_size: Maximum community size to retain.
        ens_size: Number of ensemble members (default 16).
        min_weight: ECG weight for edges with zero ensemble votes.
        final: Algorithm for the final partition ('leiden' or 'louvain').

    Returns:
        Dict with same structure as ``run_leiden`` plus ECG-specific fields:
        partition, communities, raw_n_communities, modularity,
        original_modularity, community_strength_index.
    """
    _require_leiden()
    _require_ecg()
    import partition_igraph  # noqa: F401

    g = _nx_to_igraph(G, weight_attr="weight_total")
    weights = g.es["weight"] if "weight" in g.es.attributes() else None

    part = g.community_ecg(
        weights=weights,
        ens_size=ens_size,
        min_weight=min_weight,
        final=final,
        resolution=resolution,
    )

    labels = part.membership
    nodes = g.vs["name"]
    pairs = list(zip(nodes, labels))

    from collections import Counter

    cnt = Counter(labels)
    keep = {cid for cid, sz in cnt.items() if min_size <= sz <= max_size}
    kept_pairs = [(n, c) for (n, c) in pairs if c in keep]

    comm_nodes: dict[int, list[str]] = defaultdict(list)
    for n, c in kept_pairs:
        comm_nodes[c].append(n)
    communities = [
        {"id": int(cid), "size": len(v), "nodes": v}
        for cid, v in comm_nodes.items()
    ]

    result: dict[str, Any] = {
        "partition": kept_pairs,
        "communities": communities,
        "raw_n_communities": int(len(cnt)),
        "modularity": float(part.modularity),
    }

    # ECG-specific outputs
    if hasattr(part, "original_modularity"):
        result["original_modularity"] = float(part.original_modularity)
    if hasattr(part, "CSI"):
        result["community_strength_index"] = [float(x) for x in part.CSI]

    return result


def adaptive_cluster_bounds(
    n_nodes: int,
    n_edges: int,
    *,
    min_size: int,
    max_size: int | None,
    adaptive_enabled: bool,
    max_fraction: float,
    max_floor: int,
    max_ceiling: int,
    min_fraction: float | None = None,
    max_floor_fraction: float | None = None,
    max_ceiling_fraction: float | None = None,
    min_absolute_floor: int = 10,
    max_absolute_ceiling: int = 15000,
    avg_degree_threshold: float = 15.0,
) -> tuple[int, int | None]:
    """
    Determine effective min/max cluster bounds with density-aware regime switching.

    Sparse graphs (average degree below ``avg_degree_threshold``) keep the permissive fractional cap.
    Dense graphs enforce adaptive floor/ceiling constraints to maintain interpretable cluster sizes.
    """
    if n_nodes <= 0:
        avg_degree = 0.0
    else:
        max_possible_edges = n_nodes * (n_nodes - 1)
        (n_edges / max_possible_edges) if max_possible_edges > 0 else 0.0
        avg_degree = n_edges / n_nodes

    min_eff = max(int(min_size), 1)
    if adaptive_enabled and n_nodes > 0 and min_fraction and min_fraction > 0:
        adaptive_min = int(math.ceil(n_nodes * min_fraction))
        if adaptive_min > 0:
            min_eff = max(min_eff, adaptive_min)

    max_eff: int | None = None if max_size is None or max_size <= 0 else int(max_size)

    if adaptive_enabled and n_nodes > 0:
        frac_cap = int(math.ceil(n_nodes * max(0.0, max_fraction)))

        dense_regime = avg_degree >= max(0.0, float(avg_degree_threshold))

        if not dense_regime:
            if frac_cap > 0:
                max_eff = frac_cap if max_eff is None else min(max_eff, frac_cap)
        else:
            if max_floor_fraction and max_floor_fraction > 0:
                adaptive_floor = max(
                    int(min_absolute_floor),
                    int(math.ceil(n_nodes * max_floor_fraction)),
                )
            else:
                adaptive_floor = int(max_floor) if max_floor > 0 else 0

            if max_ceiling_fraction and max_ceiling_fraction > 0:
                adaptive_ceiling = min(
                    int(max_absolute_ceiling),
                    int(math.ceil(n_nodes * max_ceiling_fraction)),
                )
            else:
                adaptive_ceiling = int(max_ceiling) if max_ceiling > 0 else 0

            if adaptive_floor > 0:
                frac_cap = max(frac_cap, adaptive_floor)
            if adaptive_ceiling > 0:
                frac_cap = min(frac_cap, adaptive_ceiling)

            if frac_cap > 0:
                max_eff = frac_cap if max_eff is None else min(max_eff, frac_cap)

    if max_eff is not None and max_eff < min_eff:
        max_eff = min_eff
    return min_eff, max_eff

def _compute_silhouette_width(
    node: str,
    cluster_id: int,
    partition_map: dict[str, int],
    G: nx.Graph,
    cluster_nodes: dict[int, list[str]]
) -> float:
    """
    Compute silhouette width for a node using citation-based dissimilarity.

    Dissimilarity definition:
    - 1 if no citation relation between nodes (rij = 0)
    - 0 if citation relation exists (rij = 1)

    Args:
        node: Node ID
        cluster_id: Cluster containing the node
        partition_map: Mapping from node to cluster ID
        G: Citation graph
        cluster_nodes: Pre-computed mapping from cluster_id to list of nodes

    Returns:
        Silhouette width in [-1, 1]. Negative values indicate probable misassignment.
    """
    # Get neighbors (citations) for this node
    if hasattr(G, "successors") and G.is_directed():
        neighbors = set(G.successors(node))
    else:
        neighbors = set(G.neighbors(node))

    # a(i): Average dissimilarity to other nodes in same cluster
    same_cluster = [n for n in cluster_nodes[cluster_id] if n != node]
    if not same_cluster:
        return 0.0  # Single-node cluster

    a_i = sum(1 if n not in neighbors else 0 for n in same_cluster) / len(same_cluster)

    # b(i): Minimum average dissimilarity to nodes in other clusters
    b_i = float('inf')
    for other_cid, other_nodes in cluster_nodes.items():
        if other_cid == cluster_id:
            continue
        if not other_nodes:
            continue

        avg_dissim = sum(1 if n not in neighbors else 0 for n in other_nodes) / len(other_nodes)
        b_i = min(b_i, avg_dissim)

    if b_i == float('inf'):
        return 0.0  # Only one cluster exists

    # s(i) = (b(i) - a(i)) / max(a(i), b(i))
    max_val = max(a_i, b_i)
    if max_val == 0:
        return 0.0

    return (b_i - a_i) / max_val


def compute_pia_flags(
    G: nx.DiGraph,
    partition_map: dict[str, int],
    *,
    min_links: int = 20,
    within_ratio: float = 0.10,
) -> dict[str, Any]:
    """
    Estimate "PIA" (Probably Inaccurate Assignments) statistics for each community.

    A node is flagged as PIA if it satisfies ALL three conditions:
    (a) Has at least ``min_links`` citation relations (default: 20)
    (b) Has ≤ ``within_ratio`` of citations within its cluster (default: 10%)
    (c) Has negative silhouette width (more similar to another cluster)

    Lower PIA rates indicate better community quality.

    Reference: Waltman & van Eck (2012) "A new methodology for constructing
    a publication-level classification system of science"
    """
    cluster_nodes: dict[int, list[str]] = defaultdict(list)
    for node, cid in partition_map.items():
        cluster_nodes[int(cid)].append(str(node))

    cluster_stats: dict[int, dict[str, float | None]] = {}
    total_eligible = 0
    total_pia = 0

    for cid, nodes in cluster_nodes.items():
        eligible = 0
        pia = 0
        for node in nodes:
            if not G.has_node(node):
                continue
            if hasattr(G, "successors") and G.is_directed():
                neighbors_iter = G.successors(node)
            else:
                neighbors_iter = G.neighbors(node)
            neighbors = [nbr for nbr in neighbors_iter if nbr in partition_map]
            total_links = len(neighbors)
            if total_links < min_links:
                continue
            eligible += 1
            within_links = sum(1 for nbr in neighbors if partition_map.get(str(nbr)) == cid)
            share_within = (within_links / total_links) if total_links else 0.0

            # Check all three PIA conditions (a, b, c)
            if share_within <= within_ratio:  # Condition (b)
                # Condition (c): Negative silhouette width
                silhouette = _compute_silhouette_width(node, cid, partition_map, G, cluster_nodes)
                if silhouette < 0:
                    pia += 1
        rate = (pia / eligible) if eligible else None
        cluster_stats[int(cid)] = {
            "eligible": eligible,
            "pia": pia,
            "pia_rate": rate,
        }
        total_eligible += eligible
        total_pia += pia

    totals = {
        "eligible": total_eligible,
        "pia": total_pia,
        "pia_rate": (total_pia / total_eligible) if total_eligible else None,
    }
    return {"cluster_stats": cluster_stats, "totals": totals}
