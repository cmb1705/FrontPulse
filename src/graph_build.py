"""Citation graph construction with optional bibliographic coupling.

This module provides the core graph-building functionality for the 2YP pipeline:

1. **Direct Citation Graphs**: Build NetworkX DiGraphs from work DataFrames,
   capturing citation edges between papers and rich node metadata (titles,
   publication dates, authors, etc.).

2. **Bibliographic Coupling**: Optionally augment graphs with weighted coupling
   edges between papers citing common references. Coupling scores use a decay
   function based on publication date differences and reference overlap.

3. **Caching**: Smart caching of coupling calculations to accelerate incremental
   builds (cumulative mode). Cache invalidation on config changes.

4. **Export Utilities**: Convenience functions for annual/quarterly graph exports
   with automatic file naming conventions.

Key Functions:
- build_direct_citation_graph(): Main graph builder
- save_graph(): Export to pickle and/or GraphML
- export_annual_full(): Build cumulative graph up to a year
- export_quarter_delta(): Build quarterly slice graph

Performance:
- Parallel coupling calculation (configurable workers)
- Progress bars for long-running operations (>1000 nodes)
- Memory monitoring integration (via memory_utils)
"""

from __future__ import annotations

import json
import logging
import math
import pathlib
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
import pandas as pd
from tqdm import tqdm

# PERF-3: Import memory monitoring utilities
try:
    from src.memory_utils import (
        check_memory_availability,
        get_memory_info,
        log_memory_usage,
        suggest_worker_count_for_memory,
    )
    MEMORY_UTILS_AVAILABLE = True
except ImportError:
    MEMORY_UTILS_AVAILABLE = False

from src.trusted_io import save_trusted_pickle


# Global default for parallel workers (matches run.py default)
DEFAULT_PARALLEL_WORKERS = 12

# Load coupling defaults from config
def _get_coupling_defaults():
    """Get coupling defaults from config/defaults.yaml or use fallbacks."""
    try:
        from .config import get_coupling_defaults
        return get_coupling_defaults()
    except Exception:
        # Fallback if config not available
        return {
            "alpha": 1.0,
            "beta": 0.3,
            "lambda_decay": 0.15,
            "min_shared_refs": 5,
            "min_coupling_score": 0.25,
            "max_year_diff": 5,
        }

_COUPLING_DEFAULTS = _get_coupling_defaults()

@dataclass
class CouplingConfig:
    enabled: bool = False
    alpha: float = _COUPLING_DEFAULTS.get("alpha", 1.0)
    beta: float = _COUPLING_DEFAULTS.get("beta", 0.3)
    lambda_decay: float = _COUPLING_DEFAULTS.get("lambda_decay", 0.15)
    min_shared_refs: int = _COUPLING_DEFAULTS.get("min_shared_refs", 5)
    min_coupling_score: float = _COUPLING_DEFAULTS.get("min_coupling_score", 0.25)
    max_year_diff: Optional[int] = _COUPLING_DEFAULTS.get("max_year_diff", 5)
    cache_dir: Optional[pathlib.Path] = None
    workers: int = DEFAULT_PARALLEL_WORKERS


def _shared_counts_worker(args: Tuple[List[List[str]], Tuple[str, ...], bool]) -> Dict[Tuple[str, str], int]:
    """Worker function for parallel bibliographic coupling calculation.

    Processes a batch of reference lists to count shared citations between pairs
    of works. This function is designed to be called in parallel via multiprocessing.

    Args:
        args: Tuple of (ref_lists, new_nodes_serialized, restrict_to_new) where:
            - ref_lists: List of reference lists, one per work
            - new_nodes_serialized: Tuple of node IDs to consider "new" (for caching)
            - restrict_to_new: If True, only count pairs where at least one node is new

    Returns:
        Dictionary mapping (node_a, node_b) -> shared_reference_count.
        Node pairs are canonically ordered (smaller ID first).
    """
    ref_lists, new_nodes_serialized, restrict_to_new = args
    new_nodes: set[str] = set(new_nodes_serialized)
    counts: Dict[Tuple[str, str], int] = defaultdict(int)
    for nodes in ref_lists:
        if len(nodes) < 2:
            continue
        nodes_sorted = sorted(set(nodes))
        for idx, node_a in enumerate(nodes_sorted):
            for node_b in nodes_sorted[idx + 1 :]:
                if restrict_to_new and (node_a not in new_nodes and node_b not in new_nodes):
                    continue
                key = (node_a, node_b) if node_a < node_b else (node_b, node_a)
                counts[key] += 1
    return counts

COUPLING_EDGE_COLUMNS = [
    "node_a",
    "node_b",
    "shared_refs",
    "ref_count_a",
    "ref_count_b",
    "coupling_score",
    "year_a",
    "year_b",
]

NODES_CACHE_FILENAME = "coupling_nodes.json"
EDGES_CACHE_FILENAME = "coupling_edges.parquet"
EDGE_CACHE_MAX_BYTES = 1_000_000_000  # ~1 GB guardrail for cached edges
CONFIG_CACHE_FILENAME = "coupling_config.json"


def _normalize_refs(value: Any) -> List[Any]:
    """Normalize reference values from various formats into a consistent list.

    Handles multiple input types from DataFrame columns: None, NaN, lists, tuples,
    sets, pandas Series, numpy arrays. Always returns a clean list.

    Args:
        value: Reference value in any supported format (list, tuple, set, Series, ndarray, etc.).

    Returns:
        List of reference values with None/NaN entries removed.
    """
    if value is None:
        return []
    if isinstance(value, float) and pd.isna(value):
        return []
    if isinstance(value, (list, tuple, set)):
        return list(value)
    if isinstance(value, pd.Series):
        return value.dropna().tolist()
    if isinstance(value, np.ndarray):
        return [v for v in value.tolist() if v is not None]
    return [value]


def _load_edge_cache(
    cache_dir: pathlib.Path,
    *,
    max_bytes: Optional[int] = None,
) -> pd.DataFrame:
    path = cache_dir / EDGES_CACHE_FILENAME
    if not path.exists():
        return pd.DataFrame(columns=COUPLING_EDGE_COLUMNS)
    if max_bytes is not None:
        try:
            if path.stat().st_size > max_bytes:
                return pd.DataFrame(columns=COUPLING_EDGE_COLUMNS)
        except OSError:
            pass
    try:
        df = pd.read_parquet(path)
        if not df.empty:
            missing = [col for col in COUPLING_EDGE_COLUMNS if col not in df.columns]
            for col in missing:
                df[col] = np.nan
            df = df[COUPLING_EDGE_COLUMNS]
        return df
    except MemoryError:
        raise
    except Exception:
        return pd.DataFrame(columns=COUPLING_EDGE_COLUMNS)


def _save_edge_cache(cache_dir: pathlib.Path, df: pd.DataFrame) -> None:
    path = cache_dir / EDGES_CACHE_FILENAME
    if df.empty:
        if path.exists():
            path.unlink()
        return
    df.to_parquet(path, index=False)


def _load_seen_nodes(cache_dir: pathlib.Path) -> set[str]:
    path = cache_dir / NODES_CACHE_FILENAME
    if path.exists():
        try:
            return set(json.loads(path.read_text()))
        except Exception:
            return set()
    return set()


def _save_seen_nodes(cache_dir: pathlib.Path, nodes: Iterable[str]) -> None:
    path = cache_dir / NODES_CACHE_FILENAME
    path.write_text(json.dumps(sorted(set(nodes))))


def _config_signature(config: CouplingConfig) -> Dict[str, float]:
    return {
        "alpha": float(config.alpha),
        "beta": float(config.beta),
        "lambda_decay": float(config.lambda_decay),
        "min_shared_refs": float(config.min_shared_refs),
        "min_coupling_score": float(config.min_coupling_score),
        "max_year_diff": config.max_year_diff if config.max_year_diff is not None else -1,
    }


def _load_cached_config(cache_dir: pathlib.Path) -> Optional[Dict[str, float]]:
    path = cache_dir / CONFIG_CACHE_FILENAME
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
        return {k: float(v) for k, v in data.items()}
    except Exception:
        return None


def _save_cached_config(cache_dir: pathlib.Path, signature: Dict[str, float]) -> None:
    path = cache_dir / CONFIG_CACHE_FILENAME
    safe = {k: float(v) for k, v in signature.items()}
    path.write_text(json.dumps(safe, sort_keys=True))

def _initialize_direct_edge_weights(G: nx.DiGraph, citation_weight: float) -> Counter:
    counts = Counter()
    for u, v in G.edges():
        edge = G.edges[u, v]
        edge["weight_citation"] = citation_weight
        edge["weight_coupling"] = 0.0
        edge["weight_total"] = citation_weight
        edge["edge_type"] = "citation"
        counts["citation_edges"] += 1
    return counts


def strip_referenced_works(G: nx.DiGraph) -> None:
    """
    Remove referenced_works from all nodes while preserving edges and weights.

    This function strips the referenced_works list from node attributes after
    bibliographic coupling calculation completes. This achieves ~91% memory
    reduction (377MB -> 33MB per graph) while preserving all graph structure,
    edges, and weights.

    The referenced_works metadata is only needed during coupling calculation.
    After coupling edges are added with proper weights, the referenced_works
    lists can be safely removed. Rich metadata remains available via lookup
    in ingest.parquet.

    Args:
        G: NetworkX DiGraph with referenced_works in node attributes

    Modifies:
        G in-place, removing 'referenced_works' key from all node attribute dicts

    Example:
        >>> G = build_direct_citation_graph(df, coupling=cfg)
        >>> G.nodes['W123']['referenced_works']  # exists
        ['W456', 'W789']
        >>> strip_referenced_works(G)
        >>> 'referenced_works' in G.nodes['W123']  # removed
        False
        >>> G.number_of_edges()  # edges preserved
        12345
    """
    nodes_with_refs = 0
    for node in G.nodes():
        if 'referenced_works' in G.nodes[node]:
            del G.nodes[node]['referenced_works']
            nodes_with_refs += 1

    logger = logging.getLogger("2yp.graph_build")
    if nodes_with_refs > 0:
        logger.info(f"Stripped referenced_works from {nodes_with_refs:,} nodes (memory optimization)")


def build_direct_citation_graph(
    df: pd.DataFrame,
    *,
    coupling: Optional[CouplingConfig] = None,
) -> nx.DiGraph:
    """Build a directed citation network with optional bibliographic coupling.

    Constructs a NetworkX DiGraph from a DataFrame of works, adding citation edges
    where referenced works are present in the corpus. Optionally augments with
    weighted coupling edges based on shared references.

    MINIMAL GRAPH ARCHITECTURE (default):
    To optimize memory usage (~91% reduction), only essential temporal attributes
    are embedded in the graph. Rich metadata (title, citations, authors, etc.)
    remains available via lookup in ingest.parquet by joining on work_id.

    Node attributes embedded (minimal mode):
    - publication_date: ISO8601 timestamp (for windowing)
    - pub_year: Publication year (for windowing)
    - pub_qtr: Publication quarter (CRITICAL: required for new-work counting)

    referenced_works is temporarily stored during coupling calculation but is
    automatically stripped before graph is returned (preserving all edges/weights)

    Edge attributes:
    - weight_total: Combined citation + coupling weight (default: 1.0 for direct citations)
    - edge_type: 'citation', 'coupling', or 'hybrid'

    Args:
        df: DataFrame with required column 'work_id' and optional columns for
            node/edge attributes (publication_date, title, referenced_works, etc.).
        coupling: Optional CouplingConfig to enable bibliographic coupling.
            When enabled, adds weighted edges between papers citing common references.

    Returns:
        NetworkX DiGraph with nodes (works), citation edges, and optional coupling edges.
        Self-loops are automatically removed. Graph metadata includes coupling_config
        and coupling_stats when coupling is enabled.

    Examples:
        >>> df = pd.DataFrame({
        ...     'work_id': ['W1', 'W2', 'W3'],
        ...     'publication_date': ['2020-01-01', '2020-06-01', '2021-01-01'],
        ...     'referenced_works': [[], ['W1'], ['W1', 'W2']]
        ... })
        >>> G = build_direct_citation_graph(df)
        >>> G.number_of_nodes()
        3
        >>> list(G.edges())
        [('W2', 'W1'), ('W3', 'W1'), ('W3', 'W2')]

        >>> # With coupling enabled
        >>> cfg = CouplingConfig(enabled=True, min_shared_refs=1)
        >>> G_coupled = build_direct_citation_graph(df, coupling=cfg)
        >>> G_coupled.graph['coupling_stats']['coupling_edges_added']
        1  # W2 and W3 both cite W1
    """
    logger = logging.getLogger("2yp.graph_build")

    # Log memory usage at start of graph building
    if MEMORY_UTILS_AVAILABLE:
        log_memory_usage(logger, f"Starting graph build with {len(df)} works")

    G = nx.DiGraph()
    if df.empty:
        return G
    if coupling and coupling.enabled:
        cfg_dict = asdict(coupling)
        if cfg_dict.get("cache_dir") is not None:
            cfg_dict["cache_dir"] = str(cfg_dict["cache_dir"])
        G.graph["coupling_config"] = cfg_dict
    node_refs_map: Dict[str, set[str]] = {}
    node_year_map: Dict[str, Optional[int]] = {}
    # Minimal attributes for memory efficiency (91% reduction vs full metadata)
    # Rich metadata available via lookup in ingest.parquet
    attrs = [
        "publication_date",  # Required for windowing in communities.py
        "pub_year",          # Required for windowing
        "pub_qtr",           # CRITICAL: Required for new-work counting in communities.py:832
    ]
    # PERF-2: Use itertuples() instead of iterrows() for better performance
    # LP-1: Add progress bar for graph construction
    rows = list(df.itertuples(index=False))
    for row in tqdm(rows, desc="Building citation graph", unit="nodes", disable=len(rows) < 1000):
        wid = getattr(row, "work_id", None)
        if pd.isna(wid):
            continue
        wid = str(wid)
        G.add_node(wid)
        pub_year_val = getattr(row, "pub_year", None) if "pub_year" in df.columns else None
        year_int: Optional[int] = None
        if pub_year_val is not None and not (isinstance(pub_year_val, float) and pd.isna(pub_year_val)):
            try:
                year_int = int(pub_year_val)
            except Exception:
                year_int = None
        for a in attrs:
            if a in df.columns:
                value = getattr(row, a, None)
                if pd.isna(value):
                    continue
                if a == "publication_date" and hasattr(value, "isoformat"):
                    value = value.isoformat()
                G.nodes[wid][a] = value
        if year_int is not None:
            node_year_map[wid] = year_int
        elif "publication_date" in G.nodes[wid]:
            try:
                node_year_map[wid] = pd.to_datetime(G.nodes[wid]["publication_date"]).year
            except Exception:
                node_year_map[wid] = None
        refs_set: set[str] = set()
        if "referenced_works" in df.columns:
            node_refs = _normalize_refs(getattr(row, "referenced_works", None))
            if node_refs:
                G.nodes[wid]["referenced_works"] = node_refs
                refs_set = {str(ref).split("/")[-1] if isinstance(ref, str) else str(ref) for ref in node_refs}
        node_refs_map[wid] = refs_set
    present = set(df["work_id"].dropna().astype(str))
    if "referenced_works" in df.columns:
        # PERF-2: Use itertuples() instead of iterrows() for better performance
        for row in df.itertuples(index=False):
            wid = getattr(row, "work_id", None)
            if pd.isna(wid):
                continue
            wid = str(wid)
            refs = _normalize_refs(getattr(row, "referenced_works", None))
            for ref in refs:
                ref_id = str(ref).split("/")[-1] if isinstance(ref, str) else str(ref)
                if ref_id in present:
                    G.add_edge(wid, ref_id)
    # remove self-loops
    G.remove_edges_from(nx.selfloop_edges(G))

    # Log graph statistics before coupling
    logger.info(f"Built citation graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    if coupling and coupling.enabled:
        direct_counts = _initialize_direct_edge_weights(G, coupling.alpha)
        coupling_stats = _augment_with_coupling(G, node_refs_map, node_year_map, coupling, direct_counts)
        if coupling_stats:
            G.graph["coupling_stats"] = coupling_stats
        logger.info(f"After coupling: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        # Strip referenced_works after coupling completes (91% memory reduction)
        strip_referenced_works(G)
    else:
        _initialize_direct_edge_weights(G, 1.0)
        # Even without coupling, strip referenced_works to maintain consistency
        strip_referenced_works(G)

    # Log memory usage at end of graph building
    if MEMORY_UTILS_AVAILABLE:
        log_memory_usage(logger, "Completed graph build")

    return G


def _augment_with_coupling(
    G: nx.DiGraph,
    node_refs_map: Dict[str, set[str]],
    node_year_map: Dict[str, Optional[int]],
    config: CouplingConfig,
    direct_counts: Counter,
) -> Dict[str, Any]:
    logger = logging.getLogger("2yp.graph_build")

    # Log configuration for coupling
    logger.info(f"Coupling calculation starting with {config.workers} workers configured")

    # PERF-3: Check memory before expensive coupling calculation
    if MEMORY_UTILS_AVAILABLE:
        log_memory_usage(logger, "Before coupling calculation")
        check_memory_availability(logger)

    stats = Counter(direct_counts)
    current_nodes = set(node_refs_map.keys())
    if not current_nodes:
        return dict(stats)

    config_sig = _config_signature(config)
    use_cache = bool(config.cache_dir)
    cache_dir = config.cache_dir if config.cache_dir else None

    cached_edges = pd.DataFrame(columns=COUPLING_EDGE_COLUMNS)
    cached_nodes: set[str] = set()
    cache_reset_reason: Optional[str] = None
    if use_cache and cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / EDGES_CACHE_FILENAME
        oversize = False
        if cache_path.exists():
            try:
                oversize = cache_path.stat().st_size > EDGE_CACHE_MAX_BYTES
            except OSError:
                oversize = False
        if oversize:
            cached_edges = pd.DataFrame(columns=COUPLING_EDGE_COLUMNS)
            cache_reset_reason = "edge_cache_oversize"
        else:
            try:
                cached_edges = _load_edge_cache(cache_dir, max_bytes=EDGE_CACHE_MAX_BYTES)
            except MemoryError:
                cached_edges = pd.DataFrame(columns=COUPLING_EDGE_COLUMNS)
                cache_reset_reason = "edge_cache_memory"
            except Exception:
                cached_edges = pd.DataFrame(columns=COUPLING_EDGE_COLUMNS)
                cache_reset_reason = "edge_cache_error"
        cached_config = _load_cached_config(cache_dir)
        config_changed = cached_config is not None and cached_config != config_sig
        load_nodes = True
        if config_changed:
            cached_edges = pd.DataFrame(columns=COUPLING_EDGE_COLUMNS)
            cached_nodes = set()
            if cache_reset_reason is None:
                cache_reset_reason = "config_changed"
            load_nodes = False
        elif not cached_edges.empty:
            cached_edges = cached_edges[
                cached_edges["node_a"].isin(current_nodes)
                & cached_edges["node_b"].isin(current_nodes)
            ].copy()
        if load_nodes and not cached_nodes:
            cached_nodes = _load_seen_nodes(cache_dir)
    else:
        use_cache = False

    new_nodes = current_nodes - cached_nodes if use_cache else current_nodes

    ref_to_nodes: Dict[str, List[str]] = defaultdict(list)
    for node, refs in node_refs_map.items():
        if not refs:
            continue
        for ref in refs:
            ref_to_nodes[ref].append(node)

    ref_lists = [nodes for nodes in ref_to_nodes.values() if len(set(nodes)) >= 2]
    restrict_to_new = use_cache
    new_nodes_serialized = tuple(sorted(new_nodes))

    # Adaptive worker scaling based on memory pressure
    effective_workers = config.workers
    if MEMORY_UTILS_AVAILABLE and config.workers > 1:
        mem_info = get_memory_info()
        available_gb = mem_info.get("available", 10.0)
        total_refs = sum(len(refs) for refs in node_refs_map.values())

        adjusted_workers, reason = suggest_worker_count_for_memory(
            available_gb=available_gb,
            node_count=len(current_nodes),
            total_refs=total_refs,
            max_workers=config.workers,
            logger=logger
        )

        if adjusted_workers != config.workers:
            logger.warning(f"Adaptive worker scaling: {reason}")
            effective_workers = adjusted_workers
        else:
            logger.info(f"Worker count: {config.workers} ({reason})")

    pair_shared: Dict[Tuple[str, str], int]
    workers_used = 1
    if effective_workers > 1 and ref_lists:
        try:
            from concurrent.futures import ProcessPoolExecutor

            worker_count = min(effective_workers, len(ref_lists))
            chunk_size = max(1, int(math.ceil(len(ref_lists) / (worker_count * 2))))
            chunks: List[List[List[str]]] = []
            for idx in range(0, len(ref_lists), chunk_size):
                chunks.append([list(nodes) for nodes in ref_lists[idx : idx + chunk_size]])

            logger.info(f"Starting parallel coupling calculation with {worker_count} workers across {len(chunks)} chunks")

            pair_shared = defaultdict(int)
            with ProcessPoolExecutor(max_workers=worker_count) as executor:
                futures = (
                    executor.submit(
                        _shared_counts_worker,
                        (chunk, new_nodes_serialized, restrict_to_new),
                    )
                    for chunk in chunks
                )
                for fut in futures:
                    try:
                        counts = fut.result()
                    except Exception as e:
                        logger.warning(f"Worker failed during coupling calculation: {e}")
                        counts = {}
                    for key, value in counts.items():
                        pair_shared[key] += int(value)
            workers_used = worker_count
            logger.info(f"Parallel coupling calculation completed using {workers_used} workers")
        except Exception as e:
            logger.warning(f"Parallel coupling failed ({e}), falling back to single-threaded")
            pair_shared = _shared_counts_worker((ref_lists, new_nodes_serialized, restrict_to_new))
    else:
        if effective_workers == 1:
            logger.info("Running coupling calculation in single-worker mode (as configured or adjusted)")
        else:
            logger.info("Running coupling calculation in single-worker mode (insufficient work for parallel processing)")
        pair_shared = _shared_counts_worker((ref_lists, new_nodes_serialized, restrict_to_new))

    rows: List[Dict[str, Any]] = []
    # LP-1: Add progress bar for coupling calculation
    pair_items = pair_shared.items()
    desc = f"Computing coupling scores ({len(pair_shared):,} pairs)"
    for (node_a, node_b), shared in tqdm(pair_items, desc=desc, unit="pairs", disable=len(pair_shared) < 1000):
        refs_a = len(node_refs_map.get(node_a, set()))
        refs_b = len(node_refs_map.get(node_b, set()))
        if refs_a == 0 or refs_b == 0:
            continue
        coupling_score = shared / math.sqrt(refs_a * refs_b)
        rows.append(
            {
                "node_a": node_a,
                "node_b": node_b,
                "shared_refs": int(shared),
                "ref_count_a": int(refs_a),
                "ref_count_b": int(refs_b),
                "coupling_score": float(coupling_score),
                "year_a": node_year_map.get(node_a),
                "year_b": node_year_map.get(node_b),
            }
        )

    new_pairs_df = (
        pd.DataFrame(rows, columns=COUPLING_EDGE_COLUMNS)
        if rows
        else pd.DataFrame(columns=COUPLING_EDGE_COLUMNS)
    )

    def _prepare(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df.copy()
        df = df.copy()
        for col in COUPLING_EDGE_COLUMNS:
            if col not in df.columns:
                df[col] = np.nan
        df = df[COUPLING_EDGE_COLUMNS]
        df["shared_refs"] = df["shared_refs"].fillna(0).astype(float)
        df["ref_count_a"] = df["ref_count_a"].fillna(0).astype(float).clip(lower=1)
        df["ref_count_b"] = df["ref_count_b"].fillna(0).astype(float).clip(lower=1)
        if "coupling_score" not in df or df["coupling_score"].isna().any():
            df["coupling_score"] = df["shared_refs"] / np.sqrt(df["ref_count_a"] * df["ref_count_b"])
        return df

    cached_prepared = _prepare(cached_edges)
    new_prepared = _prepare(new_pairs_df)

    cached_filtered = cached_prepared[
        (cached_prepared["shared_refs"] >= config.min_shared_refs)
        & (cached_prepared["coupling_score"] >= config.min_coupling_score)
    ].copy()
    new_filtered = new_prepared[
        (new_prepared["shared_refs"] >= config.min_shared_refs)
        & (new_prepared["coupling_score"] >= config.min_coupling_score)
    ].copy()

    if not cached_filtered.empty:
        cached_filtered = cached_filtered.drop_duplicates(subset=["node_a", "node_b"])
    if not new_filtered.empty:
        new_filtered = new_filtered.drop_duplicates(subset=["node_a", "node_b"])

    if cached_filtered.empty and new_filtered.empty:
        combined_retained = pd.DataFrame(columns=COUPLING_EDGE_COLUMNS)
    elif cached_filtered.empty:
        combined_retained = new_filtered
    elif new_filtered.empty:
        combined_retained = cached_filtered
    else:
        combined_retained = (
            pd.concat([cached_filtered, new_filtered], ignore_index=True)
            .drop_duplicates(subset=["node_a", "node_b"], keep="last")
        )

    stats.update(
        {
            "coupling_pairs_considered": int(len(new_pairs_df) + len(cached_filtered)),
            "new_coupling_pairs": int(len(new_filtered)),
            "new_nodes_considered": int(len(new_nodes)),
            "cached_pairs": int(len(cached_filtered)),
            "coupling_workers_used": int(workers_used),
        }
    )
    if cache_reset_reason:
        stats["cache_reset_reason"] = cache_reset_reason

    if use_cache and cache_dir is not None:
        merged_nodes = cached_nodes | current_nodes
        _save_seen_nodes(cache_dir, merged_nodes)
        _save_cached_config(cache_dir, config_sig)
        if combined_retained.empty:
            _save_edge_cache(cache_dir, pd.DataFrame(columns=COUPLING_EDGE_COLUMNS))
        else:
            store_df = combined_retained[COUPLING_EDGE_COLUMNS].sort_values(["node_a", "node_b"]).reset_index(drop=True)
            _save_edge_cache(cache_dir, store_df)

    if combined_retained.empty:
        stats.update(
            {
                "coupling_pairs_retained": 0,
                "coupling_edges": 0,
                "hybrid_edges": stats.get("hybrid_edges", 0),
                "coupling_weight_sum_pairs": 0.0,
                "coupling_weight_sum_edges": 0.0,
            }
        )
        return dict(stats)

    stats["coupling_pairs_retained"] = int(len(combined_retained))

    pair_weights: List[float] = []

    # LP-1: Add progress bar for adding coupling edges
    rows = list(combined_retained.itertuples(index=False))
    for row in tqdm(rows, desc="Adding coupling edges", unit="pairs", disable=len(rows) < 1000):
        node_a = row.node_a
        node_b = row.node_b
        if node_a not in G or node_b not in G:
            continue

        year_a = row.year_a if row.year_a is not None else node_year_map.get(node_a)
        year_b = row.year_b if row.year_b is not None else node_year_map.get(node_b)
        try:
            year_diff = abs(int(year_a) - int(year_b)) if year_a is not None and year_b is not None else 0
        except Exception:
            year_diff = 0

        # Apply temporal filter (literature-standard: 5-year window)
        if config.max_year_diff is not None and year_diff > config.max_year_diff:
            continue

        coupling_weight = config.beta * float(row.coupling_score) * math.exp(-config.lambda_decay * year_diff)
        if coupling_weight <= 0:
            continue

        pair_weights.append(coupling_weight)

        for u, v in ((node_a, node_b), (node_b, node_a)):
            if not G.has_node(u) or not G.has_node(v):
                continue
            if G.has_edge(u, v):
                edge = G.edges[u, v]
                citation_weight = float(edge.get("weight_citation", 0.0))
                edge["weight_coupling"] = float(coupling_weight)
                edge["weight_total"] = citation_weight + coupling_weight
                if edge.get("edge_type") == "citation":
                    edge["edge_type"] = "hybrid"
            else:
                G.add_edge(
                    u,
                    v,
                    weight_citation=0.0,
                    weight_coupling=float(coupling_weight),
                    weight_total=float(coupling_weight),
                    edge_type="coupling",
                )
                edge = G.edges[u, v]
            edge["coupling_shared_refs"] = int(row.shared_refs)
            edge["coupling_score"] = float(row.coupling_score)
            edge["coupling_year_diff"] = int(year_diff)

    if pair_weights:
        sum_pairs = float(sum(pair_weights))
        stats["coupling_weight_sum_pairs"] = sum_pairs
        stats["coupling_weight_mean_pair"] = sum_pairs / len(pair_weights)
        stats["coupling_weight_max_pair"] = float(max(pair_weights))
        stats["coupling_weight_min_pair"] = float(min(pair_weights))
    else:
        stats["coupling_weight_sum_pairs"] = 0.0
        stats["coupling_weight_mean_pair"] = 0.0
        stats["coupling_weight_max_pair"] = 0.0
        stats["coupling_weight_min_pair"] = 0.0

    edge_type_counts = Counter(edge_data.get("edge_type", "citation") for _, _, edge_data in G.edges(data=True))
    stats["citation_edges"] = int(edge_type_counts.get("citation", 0))
    stats["coupling_edges"] = int(edge_type_counts.get("coupling", 0))
    stats["hybrid_edges"] = int(edge_type_counts.get("hybrid", 0))
    stats["total_edges"] = int(sum(edge_type_counts.values()))

    edge_coupling_weights = [
        float(edge_data.get("weight_coupling", 0.0))
        for _, _, edge_data in G.edges(data=True)
        if edge_data.get("weight_coupling", 0.0) > 0
    ]
    if edge_coupling_weights:
        sum_edges = float(sum(edge_coupling_weights))
        stats["coupling_weight_sum_edges"] = sum_edges
        stats["coupling_weight_mean_edge"] = sum_edges / len(edge_coupling_weights)
        stats["coupling_weight_max_edge"] = float(max(edge_coupling_weights))
        stats["coupling_weight_min_edge"] = float(min(edge_coupling_weights))
        stats["coupling_edge_count"] = len(edge_coupling_weights)
    else:
        stats["coupling_weight_sum_edges"] = 0.0
        stats["coupling_weight_mean_edge"] = 0.0
        stats["coupling_weight_max_edge"] = 0.0
        stats["coupling_weight_min_edge"] = 0.0
        stats["coupling_edge_count"] = 0

    # PERF-3: Log memory after coupling calculation
    if MEMORY_UTILS_AVAILABLE:
        log_memory_usage(logger, "After coupling calculation")

    return dict(stats)

def save_graph(
    G: nx.DiGraph,
    basepath: pathlib.Path,
    *,
    write_pickle: bool = True,
    write_graphml: bool = True,
    graphml_compression: Optional[str] = None,
) -> None:
    """Save NetworkX graph to disk in pickle and/or GraphML formats.

    Exports the graph with proper attribute sanitization for GraphML compatibility.
    Self-loops are removed as a safety measure. Attributes are type-cast to ensure
    GraphML XML compatibility (timestamps -> strings, numpy types -> Python types, etc.).

    The function creates two files (when both formats enabled):
    - {basepath}.pkl: Binary NetworkX pickle (fast, Python-only)
    - {basepath}.graphml[.gz]: XML format (slower, interoperable with Gephi/Cytoscape)

    Args:
        G: NetworkX DiGraph to export.
        basepath: Path without extension (e.g., Path("graphs/citation_graph_2020")).
            Extensions .pkl and .graphml are appended automatically.
        write_pickle: Whether to write .pkl file (default: True).
        write_graphml: Whether to write .graphml file (default: True).
        graphml_compression: Optional compression for GraphML ('gzip' or None).
            When 'gzip', writes .graphml.gz instead of .graphml.

    Returns:
        None. Files are written to {basepath}.pkl and/or {basepath}.graphml[.gz].

    Examples:
        >>> G = nx.DiGraph()
        >>> G.add_node('W1', publication_date='2020-01-01', title='Paper A')
        >>> G.add_edge('W2', 'W1', weight_total=1.0)
        >>> save_graph(G, Path('data/graphs/test'), write_graphml=False)
        # Creates: data/graphs/test.pkl

        >>> save_graph(G, Path('data/graphs/test_full'), graphml_compression='gzip')
        # Creates: data/graphs/test_full.pkl and data/graphs/test_full.graphml.gz
    """
    import datetime as dt
    import math

    import numpy as np
    import pandas as pd

    # 1) copy and remove self-loops (safety even if builder already removed)
    H = G.copy()
    H.remove_edges_from(nx.selfloop_edges(H))

    def _cast(v):
        try:
            if v is None or (isinstance(v, float) and math.isnan(v)) or pd.isna(v):
                return "__DROP__"
        except Exception:
            pass
        if isinstance(v, (pd.Timestamp, dt.datetime, dt.date)): return str(v)
        if isinstance(v, (np.integer,)):  return int(v)
        if isinstance(v, (np.floating,)): return float(v)
        if isinstance(v, (np.bool_,)):    return bool(v)
        if isinstance(v, (list, tuple, set)): return ",".join(map(str, v))
        if isinstance(v, (str, int, float, bool)): return v
        return str(v)

    # 2) sanitize attributes for GraphML
    S = nx.DiGraph()
    S.add_nodes_from(H.nodes())
    S.add_edges_from(H.edges())
    for n, data in H.nodes(data=True):
        clean = {}
        for k, v in data.items():
            key = "doc_type" if k == "type" else k
            cv = _cast(v)
            if cv != "__DROP__": clean[key] = cv
        S.nodes[n].update(clean)
    for u, v, data in H.edges(data=True):
        clean = {}
        for k, vv in data.items():
            cv = _cast(vv)
            if cv != "__DROP__": clean[k] = cv
        S.edges[u, v].update(clean)

    logger = logging.getLogger("2yp.graph_build")

    basepath.parent.mkdir(parents=True, exist_ok=True)
    if write_pickle:
        pkl_path = str(basepath) + ".pkl"
        try:
            save_trusted_pickle(
                H, pkl_path, description="citation graph pickle"
            )
        except Exception as e:
            logger.error(f"FAILED to save graph pickle to {pkl_path}: {e}")
            raise
    if write_graphml:
        graphml_path = str(basepath) + ".graphml"
        try:
            if graphml_compression == "gzip":
                import gzip

                with gzip.open(graphml_path + ".gz", "wb") as fh:
                    nx.write_graphml(S, fh)
                logger.info(f"Successfully saved compressed GraphML to {graphml_path}.gz")
            else:
                nx.write_graphml(S, graphml_path)
                logger.info(f"Successfully saved GraphML to {graphml_path}")
        except Exception as e:
            logger.error(f"FAILED to save GraphML to {graphml_path}: {e}")
            raise

def export_annual_full(
    df: pd.DataFrame,
    *,
    year: int,
    outdir: pathlib.Path,
    coupling: Optional[CouplingConfig] = None,
) -> pathlib.Path:
    """Build and save an annual cumulative citation graph up to a given year.

    Creates a citation network containing all works published up to and including
    December 31 of the specified year. This is useful for analyzing the full state
    of the research landscape at yearly checkpoints.

    Args:
        df: DataFrame of works with 'publication_date' column.
        year: Year for cutoff (inclusive). All works published <= {year}-12-31 are included.
        outdir: Directory where graph files will be saved.
        coupling: Optional CouplingConfig for bibliographic coupling.

    Returns:
        Path to the saved graph (without extension). Files created:
        - {outdir}/citation_graph_annual_{year}.pkl
        - {outdir}/citation_graph_annual_{year}.graphml

    Examples:
        >>> df = pd.DataFrame({
        ...     'work_id': ['W1', 'W2', 'W3'],
        ...     'publication_date': pd.to_datetime(['2019-06-01', '2020-03-01', '2021-01-01'])
        ... })
        >>> path = export_annual_full(df, year=2020, outdir=Path('graphs'))
        >>> path
        PosixPath('graphs/citation_graph_annual_2020')
        # Creates graphs/citation_graph_annual_2020.pkl with W1 and W2 only
    """
    cutoff = pd.Timestamp(f"{year}-12-31")
    sub = df[df["publication_date"] <= cutoff]
    G = build_direct_citation_graph(sub, coupling=coupling)
    base = outdir / f"citation_graph_annual_{year}"
    save_graph(G, base)
    return base

def export_quarter_delta(
    df: pd.DataFrame,
    *,
    year: int,
    quarter: int,
    outdir: pathlib.Path,
    coupling: Optional[CouplingConfig] = None,
) -> pathlib.Path:
    """Build and save a quarterly delta citation graph for a specific period.

    Creates a citation network containing only works published during the specified
    quarter. This "delta" graph shows new publications and their citations within
    that time slice, useful for tracking research front emergence and evolution.

    Args:
        df: DataFrame of works with 'pub_qtr' column (format: "YYYYQN", e.g., "2020Q3").
        year: Year of the quarter (e.g., 2020).
        quarter: Quarter number 1-4 (1=Jan-Mar, 2=Apr-Jun, 3=Jul-Sep, 4=Oct-Dec).
        outdir: Directory where graph files will be saved.
        coupling: Optional CouplingConfig for bibliographic coupling.

    Returns:
        Path to the saved graph (without extension). Files created:
        - {outdir}/citation_graph_delta_{year}Q{quarter}.pkl
        - {outdir}/citation_graph_delta_{year}Q{quarter}.graphml

    Examples:
        >>> df = pd.DataFrame({
        ...     'work_id': ['W1', 'W2', 'W3'],
        ...     'pub_qtr': ['2020Q1', '2020Q1', '2020Q2']
        ... })
        >>> path = export_quarter_delta(df, year=2020, quarter=1, outdir=Path('graphs'))
        >>> path
        PosixPath('graphs/citation_graph_delta_2020Q1')
        # Creates graphs/citation_graph_delta_2020Q1.pkl with W1 and W2 only
    """
    qstr = f"{year}Q{quarter}"
    sub = df[df["pub_qtr"] == qstr]
    G = build_direct_citation_graph(sub, coupling=coupling)
    base = outdir / f"citation_graph_delta_{qstr}"
    save_graph(G, base)
    return base
