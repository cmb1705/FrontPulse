"""
Shared lineage text index for Phase 2-5 pipeline.

This module provides a unified interface to the lineage registry, abstract extraction,
and related resources. Loading these once and sharing across phases eliminates redundant
I/O and speeds up the pipeline significantly.

Usage:
    # Standalone script (loads fresh)
    from src.lineage_text_store import LineageTextStore
    store = LineageTextStore(registry_path="...", raw_dir="...")

    # Pipeline mode (reuses shared store)
    store = get_shared_store()

    # Both modes work identically
    papers = store.get_lineage_papers(lineage_id)
    texts = store.extractor.get_texts_batch(work_ids)

The store reuses the serialized abstract index produced by ``AbstractExtractor``.
When ``abstract_cache_path`` is provided (or left as ``None`` to use the default),
Stage 4 worker processes can mount the cache instead of rebuilding the global index.
"""

from __future__ import annotations
import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Set
import time

# Lazy import for abstract extractor
_AbstractExtractor = None


def get_abstract_extractor():
    """Lazy import of AbstractExtractor to avoid circular dependencies."""
    global _AbstractExtractor
    if _AbstractExtractor is None:
        from scripts.extract_abstracts import AbstractExtractor
        _AbstractExtractor = AbstractExtractor
    return _AbstractExtractor


class LineageTextStore:
    """
    Centralized store for lineage registry, paper texts, and related resources.

    This class loads expensive resources once and provides fast access for all
    Phase 2-5 operations. Key resources:

    - lineage_registry.json: Maps quarters -> lineages -> work IDs
    - partition JSONs: Community assignments for graph loading
    - raw JSONL files: For abstract/title extraction
    - graphs directory: Citation graph PKL files

    Attributes:
        registry: Dict mapping {quarter: {lineage_id: community_data}}
        registry_by_lineage: Inverted index {lineage_id: {quarter: data}}
        extractor: AbstractExtractor instance for text retrieval
        graphs_dir: Path to citation graph PKL files
        partitions_dir: Path to partition JSON files
    """

    def __init__(
        self,
        registry_path: Path,
        raw_dir: Path,
        graphs_dir: Optional[Path] = None,
        partitions_dir: Optional[Path] = None,
        abstract_cache_path: Optional[Path] = None,
        verbose: bool = True
    ):
        """
        Load shared resources for lineage analysis.

        Args:
            registry_path: Path to lineage_registry.json
            raw_dir: Path to raw JSONL files for abstract extraction
            graphs_dir: Path to citation graph PKL files (optional)
            partitions_dir: Path to partition JSON files (optional)
            abstract_cache_path: Optional path for serialized abstract index cache
            verbose: Print loading progress
        """
        self.verbose = verbose
        self.graphs_dir = graphs_dir
        self.partitions_dir = partitions_dir
        self.abstract_cache_path = Path(abstract_cache_path) if abstract_cache_path else None

        # Load lineage registry
        if verbose:
            print(f"[LineageTextStore] Loading registry from {registry_path}...")
            t0 = time.time()

        with open(registry_path, 'r') as f:
            self.registry = json.load(f)

        if verbose:
            print(f"[LineageTextStore] Registry loaded in {time.time()-t0:.2f}s")

        # Build inverted index: {lineage_id: {quarter: data}}
        if verbose:
            print(f"[LineageTextStore] Building lineage index...")
            t0 = time.time()

        self.registry_by_lineage = self._build_lineage_index()

        if verbose:
            n_lineages = len(self.registry_by_lineage)
            n_quarters = len(self.registry)
            print(f"[LineageTextStore] Index built in {time.time()-t0:.2f}s")
            print(f"[LineageTextStore]   {n_lineages} lineages across {n_quarters} quarters")

        # Initialize abstract extractor
        if verbose:
            print(f"[LineageTextStore] Initializing abstract extractor...")
            t0 = time.time()

        AbstractExtractor = get_abstract_extractor()
        self.extractor = AbstractExtractor(raw_dir, cache_path=self.abstract_cache_path)
        self.abstract_cache_path = Path(self.extractor.cache_path)

        if verbose:
            print(f"[LineageTextStore] Abstract extractor ready in {time.time()-t0:.2f}s")

    def _build_lineage_index(self) -> Dict[int, Dict[str, Dict[str, int]]]:
        """
        Invert registry from {quarter: {community_id: lineage_id}} to
        {lineage_id: {quarter: {community_id: lineage_id}}}.

        The input registry format is {quarter: {community_id: lineage_id}}.
        Phase 3/4 need {lineage_id: {quarter: {community_id: lineage_id}}}.

        Returns:
            Lineage-keyed index matching Phase 3/4 expectations
        """
        lineage_index = {}

        for quarter, community_map in self.registry.items():
            # community_map is {community_id: lineage_id}
            for comm_id, lin_id in community_map.items():
                lin_id_int = int(lin_id)  # Ensure lineage_id is int

                if lin_id_int not in lineage_index:
                    lineage_index[lin_id_int] = {}
                if quarter not in lineage_index[lin_id_int]:
                    lineage_index[lin_id_int][quarter] = {}

                # Store community mapping: {community_id: lineage_id}
                lineage_index[lin_id_int][quarter][comm_id] = lin_id_int

        return lineage_index

    def get_lineage_papers(
        self,
        lineage_id,
        quarter: Optional[str] = None
    ) -> List[str]:
        """
        Get all paper IDs for a lineage using partition JSONs.

        NOTE: This method is deprecated since the store now only contains the
        registry index, not paper-level data. Use get_lineage_papers_from_partitions()
        or Phase 2's load_lineage_papers_fast() instead.

        Args:
            lineage_id: Lineage identifier (int or str)
            quarter: Optional quarter filter (e.g., "2020Q1")

        Returns:
            Empty list (method stub for backward compatibility)
        """
        # Convert to int if needed
        lin_id_int = int(lineage_id) if isinstance(lineage_id, str) else lineage_id

        if lin_id_int not in self.registry_by_lineage:
            return []

        # This method can't return papers since registry_by_lineage only has
        # {quarter: {community_id: lineage_id}}, not actual work IDs
        # Callers should use load_lineage_papers_fast() from Phase 2
        return []

    def get_lineage_papers_from_graph(
        self,
        lineage_id,
        mode: str = "cumulative"
    ) -> List[str]:
        """
        Load papers for a lineage from citation graph PKL files.

        This matches the behavior of load_lineage_papers() in compute_lineage_embeddings.py.

        Args:
            lineage_id: Lineage identifier (int or str)
            mode: Graph mode ("cumulative" or "delta"), default "cumulative"

        Returns:
            List of work IDs from graph files
        """
        if not self.graphs_dir:
            raise ValueError("graphs_dir not provided to LineageTextStore")

        # Convert to int if needed
        lin_id_int = int(lineage_id) if isinstance(lineage_id, str) else lineage_id

        papers = []

        if lin_id_int not in self.registry_by_lineage:
            return papers

        for quarter, community_map in self.registry_by_lineage[lin_id_int].items():
            # community_map is {community_id: lineage_id}
            # Get all community_ids that belong to this lineage
            community_ids = list(community_map.keys())

            if not community_ids:
                continue

            # Correct graph filename pattern: citation_graph_{mode}_{quarter}.pkl
            graph_path = self.graphs_dir / f"citation_graph_{mode}_{quarter}.pkl"

            if not graph_path.exists():
                continue

            # Load graph and extract papers
            with open(graph_path, 'rb') as f:
                G = pickle.load(f)

            # Get works from all communities belonging to this lineage
            for work_id in G.nodes():
                node_community = G.nodes[work_id].get('community')
                # Convert to string for comparison (community_ids in map are strings)
                if str(node_community) in community_ids:
                    papers.append(work_id)

        return papers

    def get_lineage_quarters(self, lineage_id) -> List[str]:
        """
        Get all quarters where a lineage exists.

        Args:
            lineage_id: Lineage identifier (int or str)

        Returns:
            Sorted list of quarters
        """
        lin_id_int = int(lineage_id) if isinstance(lineage_id, str) else lineage_id
        if lin_id_int not in self.registry_by_lineage:
            return []
        return sorted(self.registry_by_lineage[lin_id_int].keys())

    def get_all_lineages(self) -> Set[int]:
        """Get set of all lineage IDs in the registry (as ints)."""
        return set(self.registry_by_lineage.keys())

    def get_persistent_lineages(self, min_quarters: int = 12) -> List[int]:
        """
        Get lineages that persist for at least min_quarters.

        Args:
            min_quarters: Minimum number of quarters required

        Returns:
            List of persistent lineage IDs (as ints)
        """
        persistent = []
        for lineage_id, quarters in self.registry_by_lineage.items():
            if len(quarters) >= min_quarters:
                persistent.append(lineage_id)
        return persistent


# Global shared store (set by pipeline driver)
_SHARED_STORE: Optional[LineageTextStore] = None


def set_shared_store(store: LineageTextStore) -> None:
    """
    Set the global shared store for pipeline mode.

    Called by the pipeline driver before running phase scripts.
    """
    global _SHARED_STORE
    _SHARED_STORE = store


def get_shared_store() -> Optional[LineageTextStore]:
    """
    Get the global shared store if available.

    Returns None if running in standalone mode (not from pipeline).
    """
    return _SHARED_STORE


def load_or_get_store(
    registry_path: Optional[Path] = None,
    raw_dir: Optional[Path] = None,
    graphs_dir: Optional[Path] = None,
    partitions_dir: Optional[Path] = None,
    verbose: bool = True
) -> LineageTextStore:
    """
    Load a fresh store or return the shared one if available.

    This is the recommended way for phase scripts to get a store:
    - In pipeline mode: Returns the pre-loaded shared store
    - In standalone mode: Loads a fresh store from provided paths

    Args:
        registry_path: Path to registry (required if not in pipeline mode)
        raw_dir: Path to raw JSONL (required if not in pipeline mode)
        graphs_dir: Path to graph PKLs (optional)
        partitions_dir: Path to partitions (optional)
        verbose: Print loading messages

    Returns:
        LineageTextStore instance (shared or fresh)
    """
    shared = get_shared_store()

    if shared is not None:
        if verbose:
            print("[LineageTextStore] Using shared store (pipeline mode)")
        return shared

    # Standalone mode: require paths
    if registry_path is None or raw_dir is None:
        raise ValueError(
            "registry_path and raw_dir required when not in pipeline mode"
        )

    if verbose:
        print("[LineageTextStore] Loading fresh store (standalone mode)")

    return LineageTextStore(
        registry_path=registry_path,
        raw_dir=raw_dir,
        graphs_dir=graphs_dir,
        partitions_dir=partitions_dir,
        verbose=verbose
    )
