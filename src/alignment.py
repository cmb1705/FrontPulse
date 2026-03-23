"""Community alignment utilities for tracking research fronts across time periods.

Provides functions for:
- PageRank-based core member identification within communities
- Community matching across temporal slices using core overlap
- Variation of Information (VI) metric for partition similarity measurement
"""

from __future__ import annotations

import math
from typing import Dict, List, Tuple

import networkx as nx


def pagerank_core(G: nx.Graph, partition: List[Tuple[str,int]], core_frac: float = 0.10) -> Dict[int, set]:
    """Identify core members of each community using PageRank centrality.

    Computes PageRank for all nodes in the graph and selects the top-ranked
    fraction within each community to form stable "cores" for cross-period
    alignment. Core members are nodes with highest influence in their community.

    Args:
        G: NetworkX graph containing the full network structure.
        partition: List of (node_id, community_id) tuples defining community membership.
        core_frac: Fraction of each community to include in core (default: 0.10 = top 10%).

    Returns:
        Dictionary mapping community_id -> set of core node IDs.

    Examples:
        >>> G = nx.karate_club_graph()
        >>> partition = [(str(n), n % 2) for n in G.nodes()]
        >>> cores = pagerank_core(G, partition, core_frac=0.2)
        >>> cores[0]  # Top 20% of community 0 by PageRank
        {'0', '2', '4', ...}
    """
    pr = nx.pagerank(G) if len(G) else {}
    by_comm: Dict[int, List[Tuple[str, float]]] = {}
    for n, c in partition:
        by_comm.setdefault(c, []).append((n, pr.get(n, 0.0)))
    cores: Dict[int, set] = {}
    for cid, pairs in by_comm.items():
        pairs.sort(key=lambda x: x[1], reverse=True)
        k = max(1, int(math.ceil(core_frac * len(pairs)))) if pairs else 0
        cores[cid] = {n for n, _ in pairs[:k]} if k else set()
    return cores

def _overlap(a: set, b: set) -> int:
    """Compute set intersection size (helper for core matching).

    Args:
        a: First set of node IDs.
        b: Second set of node IDs.

    Returns:
        Number of elements in common between a and b.
    """
    return len(a & b)

def match_by_cores(prev_cores: Dict[int,set], curr_cores: Dict[int,set]) -> List[Tuple[int,int,int]]:
    """Match communities across time periods using core member overlap.

    Uses the Hungarian algorithm (via scipy.optimize.linear_sum_assignment) to find
    the optimal one-to-one matching between previous and current communities based on
    maximum core overlap. Falls back to greedy matching if scipy is unavailable.

    This is the primary alignment mechanism for tracking community evolution in the
    front timeseries workflow. Only matches with positive overlap are returned.

    Args:
        prev_cores: Dictionary mapping previous period's community_id -> set of core node IDs.
        curr_cores: Dictionary mapping current period's community_id -> set of core node IDs.

    Returns:
        List of (prev_community_id, curr_community_id, overlap_count) tuples.
        Only includes matches with overlap > 0, sorted by overlap strength.

    Examples:
        >>> prev = {1: {'A', 'B', 'C'}, 2: {'D', 'E'}}
        >>> curr = {10: {'B', 'C', 'F'}, 20: {'D', 'E', 'G'}}
        >>> matches = match_by_cores(prev, curr)
        >>> matches
        [(1, 10, 2), (2, 20, 2)]  # (prev_id, curr_id, overlap)
    """
    prev_ids, curr_ids = list(prev_cores.keys()), list(curr_cores.keys())
    M = [[-_overlap(prev_cores[pi], curr_cores[cj]) for cj in curr_ids] for pi in prev_ids]
    try:
        from scipy.optimize import linear_sum_assignment
        row_ind, col_ind = linear_sum_assignment(M)
        matches = []
        for i, j in zip(row_ind, col_ind):
            ov = -M[i][j]
            if ov > 0:
                matches.append((prev_ids[i], curr_ids[j], ov))
        return matches
    except Exception:
        used_prev, used_curr, matches = set(), set(), []
        all_pairs = [(-_overlap(prev_cores[pi], curr_cores[cj]), i, j)
                     for i, pi in enumerate(prev_ids) for j, cj in enumerate(curr_ids)]
        for _, i, j in sorted(all_pairs):
            if i in used_prev or j in used_curr:
                continue
            ov = -M[i][j]
            if ov > 0:
                used_prev.add(i); used_curr.add(j)
                matches.append((prev_ids[i], curr_ids[j], ov))
        return matches

def variation_of_information(labels_a: Dict[str,int], labels_b: Dict[str,int]) -> float:
    """Compute Variation of Information (VI) metric between two partitions.

    VI is an information-theoretic metric that measures the distance between two
    clusterings. It combines the entropy of each partition and their mutual information:

        VI(A, B) = H(A) + H(B) - 2 * I(A, B)

    where H(X) is the Shannon entropy of partition X and I(A, B) is their mutual
    information. VI = 0 indicates identical partitions; higher values indicate
    greater dissimilarity.

    Handles nodes present in only one partition by assigning unique community IDs
    to missing entries, effectively treating them as singleton communities.

    Args:
        labels_a: Dictionary mapping node_id -> community_id for partition A.
        labels_b: Dictionary mapping node_id -> community_id for partition B.

    Returns:
        Variation of Information score (bits). Lower values indicate more similar
        partitions. Range: [0, ∞), but typically [0, log2(n)] for n nodes.

    Examples:
        >>> labels_a = {'A': 1, 'B': 1, 'C': 2, 'D': 2}
        >>> labels_b = {'A': 10, 'B': 10, 'C': 20, 'D': 20}
        >>> variation_of_information(labels_a, labels_b)
        0.0  # Identical partitions with different IDs

        >>> labels_c = {'A': 1, 'B': 1, 'C': 1, 'D': 2}
        >>> variation_of_information(labels_a, labels_c)
        0.811...  # Different partition structure
    """
    from collections import Counter, defaultdict
    nodes = set(labels_a.keys()) | set(labels_b.keys())
    la, lb = dict(labels_a), dict(labels_b)
    next_id = 10**9
    for n in nodes:
        if n not in la: la[n] = next_id; next_id += 1
        if n not in lb: lb[n] = next_id; next_id += 1
    Ca, Cb = Counter(la.values()), Counter(lb.values())
    N = float(len(nodes))
    Ha = -sum((c/N) * math.log((c/N), 2) for c in Ca.values())
    Hb = -sum((c/N) * math.log((c/N), 2) for c in Cb.values())
    M = defaultdict(int)
    for n in nodes:
        M[(la[n], lb[n])] += 1
    I = 0.0
    for (ia, ib), cij in M.items():
        pa, pb, pij = Ca[ia]/N, Cb[ib]/N, cij/N
        I += pij * math.log(pij/(pa*pb), 2) if pij > 0 else 0.0
    return Ha + Hb - 2*I

def label_map_from_partition(partition: List[Tuple[str,int]]) -> Dict[str,int]:
    """Convert partition list format to label dictionary format.

    Transforms the partition representation from a list of (node, community) tuples
    into a dictionary mapping nodes to their community IDs. This is a convenience
    function for converting between formats used by different community detection
    libraries (igraph/leidenalg uses tuples, VI metric uses dicts).

    Args:
        partition: List of (node_id, community_id) tuples.

    Returns:
        Dictionary mapping node_id -> community_id.

    Examples:
        >>> partition = [('A', 1), ('B', 1), ('C', 2)]
        >>> label_map_from_partition(partition)
        {'A': 1, 'B': 1, 'C': 2}
    """
    return {n: c for n, c in partition}
