#!/usr/bin/env python3
"""Compute cross-front convergence features for lineage-level MSD integration.

Orchestrates four convergence channels per quarter (semantic similarity,
author migration, citation bridging, terminological overlap), computes
rolling window features, and outputs CSV keyed by (lineage_id, quarter)
with all ``conv_*`` columns.

Channels degrade gracefully when underlying data is unavailable:

- Semantic channel requires quarterly embeddings NPZ (optional).
- Author/citation/terminology channels use parquet slices and partitions.

Usage::

    python scripts/compute_convergence_features.py --verbose
    python scripts/compute_convergence_features.py --top-k 10 --out my_output.csv
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from _path_bootstrap import ensure_repo_imports

REPO_ROOT = ensure_repo_imports()

from src.convergence import (  # noqa: E402
    CONVERGENCE_FEATURE_DEFAULTS,
    aggregate_convergence_features,
    compute_author_overlap,
    compute_citation_bridges,
    compute_pairwise_semantic_similarity,
    compute_rolling_convergence_features,
    compute_terminology_jaccard,
)
from src.domain_registry import (  # noqa: E402
    add_domain_args,
    apply_domain_path_defaults,
    resolve_script_paths,
)

LOG = logging.getLogger("convergence_features")

# Simple stopwords for title tokenization (common English academic words).
_STOPWORDS = frozenset({
    "the", "and", "for", "with", "from", "that", "this", "was", "are",
    "were", "been", "being", "have", "has", "had", "not", "but", "its",
    "can", "may", "will", "all", "one", "two", "each", "into", "than",
    "also", "use", "used", "using", "based", "new", "study", "via",
    "after", "before", "between", "through", "during", "about", "their",
    "these", "those", "which", "other", "more", "most", "such", "both",
    "only", "some", "our", "when", "how", "what", "over", "very",
})


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def configure_logging(verbose: bool) -> None:
    """Set up logging with appropriate verbosity level."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def load_registry(path: Path) -> dict[str, dict[str, int]]:
    """Load lineage registry JSON.

    Returns:
        Mapping of quarter -> {local_community_id_str -> global_lineage_id}.
    """
    return json.loads(path.read_text(encoding="utf-8"))


def load_partition(partition_path: Path) -> dict[int, list[str]]:
    """Load partition JSON and invert to {community_id -> [work_ids]}."""
    data = json.loads(partition_path.read_text(encoding="utf-8"))
    inverted: dict[int, list[str]] = defaultdict(list)
    for work_id, comm_id in data.get("labels", {}).items():
        inverted[int(comm_id)].append(work_id)
    return inverted


def tokenize_title(title: str) -> set[str]:
    """Extract term set from a paper title for Jaccard computation.

    Uses lowercase alphanumeric tokens of length >= 3, minus stopwords.
    """
    if not title:
        return set()
    tokens = re.findall(r"\b[a-z][a-z0-9\-]{2,}\b", title.lower())
    return {t for t in tokens if t not in _STOPWORDS}


def normalize_author_ids(raw_value: object) -> list[str]:
    """Parse comma-separated OpenAlex author URLs into short author IDs."""
    if raw_value is None:
        return []
    try:
        if pd.isna(raw_value):
            return []
    except (TypeError, ValueError):
        pass
    if isinstance(raw_value, str):
        return [
            v.strip().rsplit("/", 1)[-1]
            for v in raw_value.split(",")
            if v.strip()
        ]
    return []


def extract_references(raw_value: object) -> list[str]:
    """Extract reference work IDs from parquet ``referenced_works`` column."""
    if raw_value is None:
        return []
    if isinstance(raw_value, np.ndarray):
        return [str(r) for r in raw_value if r]
    if isinstance(raw_value, (list, tuple)):
        return [str(r) for r in raw_value if r]
    return []


def load_quarterly_embeddings(
    npz_path: Path,
) -> dict[str, dict[int, np.ndarray]]:
    """Load quarterly embeddings NPZ into per-quarter dicts.

    Expected NPZ arrays: ``lineage_ids``, ``quarters``, ``embeddings``
    (shape ``[N, 768]``).

    Returns:
        Mapping of quarter -> {lineage_id -> L2-normalized 768-d vector}.
    """
    data = np.load(npz_path)
    lineage_ids = data["lineage_ids"]
    quarters = data["quarters"]
    embeddings = data["embeddings"]

    result: dict[str, dict[int, np.ndarray]] = defaultdict(dict)
    for i in range(len(lineage_ids)):
        q = str(quarters[i])
        lid = int(lineage_ids[i])
        vec = embeddings[i].astype(np.float32)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        result[q][lid] = vec

    return dict(result)


# ---------------------------------------------------------------------------
# Per-quarter data extraction
# ---------------------------------------------------------------------------


def build_quarter_data(
    quarter: str,
    registry: dict[str, dict[str, int]],
    partitions_dir: Path,
    slices_dir: Path,
    global_paper_lineage: dict[str, int],
) -> tuple[
    dict[int, set[str]],
    dict[int, set[str]],
    dict[int, set[str]],
    dict[str, list[str]],
]:
    """Extract per-lineage data structures for one quarter.

    Args:
        quarter: Quarter label (e.g. ``"2020Q1"``).
        registry: Full registry (quarter -> {comm_id -> lineage_id}).
        partitions_dir: Directory with ``part_YYYYQN.json`` files.
        slices_dir: Directory with ``by_quarter__YYYYQN.parquet`` files.
        global_paper_lineage: Accumulated paper -> lineage mapping
            (mutated in place across quarters).

    Returns:
        Tuple of ``(lineage_papers, lineage_authors, lineage_terms,
        paper_references)``.
    """
    lineage_papers: dict[int, set[str]] = defaultdict(set)
    lineage_authors: dict[int, set[str]] = defaultdict(set)
    lineage_terms: dict[int, set[str]] = defaultdict(set)
    paper_references: dict[str, list[str]] = {}

    # Map from community to lineage for this quarter
    comm_to_lineage = registry.get(quarter, {})
    if not comm_to_lineage:
        return lineage_papers, lineage_authors, lineage_terms, paper_references

    # Load partition to get paper -> community mapping
    partition_path = partitions_dir / f"part_{quarter}.json"
    if not partition_path.exists():
        LOG.warning("Partition file missing for %s, skipping", quarter)
        return lineage_papers, lineage_authors, lineage_terms, paper_references

    community_papers = load_partition(partition_path)

    # Build paper -> lineage mapping for this quarter
    paper_to_lineage: dict[str, int] = {}
    for comm_id_str, lineage_id in comm_to_lineage.items():
        comm_id = int(comm_id_str)
        for paper_id in community_papers.get(comm_id, []):
            paper_to_lineage[paper_id] = lineage_id
            lineage_papers[lineage_id].add(paper_id)
            global_paper_lineage.setdefault(paper_id, lineage_id)

    # Load parquet slice for author, reference, and term extraction
    slice_path = slices_dir / f"by_quarter__{quarter}.parquet"
    if not slice_path.exists():
        LOG.warning(
            "Parquet slice missing for %s, skipping author/ref/term extraction",
            quarter,
        )
        return lineage_papers, lineage_authors, lineage_terms, paper_references

    try:
        df = pd.read_parquet(
            slice_path,
            columns=["work_id", "author_ids", "referenced_works", "title"],
        )
    except Exception:
        LOG.warning("Failed to read parquet slice for %s", quarter, exc_info=True)
        return lineage_papers, lineage_authors, lineage_terms, paper_references

    # Process each paper that belongs to a known lineage
    for row in df.itertuples(index=False):
        work_id = str(row.work_id)
        lineage_id = paper_to_lineage.get(work_id)
        if lineage_id is None:
            continue

        # Authors
        authors = normalize_author_ids(row.author_ids)
        lineage_authors[lineage_id].update(authors)

        # References
        refs = extract_references(row.referenced_works)
        paper_references[work_id] = refs

        # Terms from title
        try:
            title = str(row.title) if row.title is not None and not pd.isna(row.title) else ""
        except (TypeError, ValueError):
            title = ""
        terms = tokenize_title(title)
        lineage_terms[lineage_id].update(terms)

    return lineage_papers, lineage_authors, lineage_terms, paper_references


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    ap = argparse.ArgumentParser(
        description="Compute cross-front convergence features per (lineage, quarter).",
    )
    ap.add_argument(
        "--registry",
        default=None,
        help="Lineage registry JSON.",
    )
    ap.add_argument(
        "--timeseries",
        default=None,
        help="Lineage timeseries CSV for quarter list.",
    )
    ap.add_argument(
        "--quarterly-embeddings",
        default=None,
        help="Quarterly SciBERT embeddings NPZ. "
             "If missing, the semantic channel is skipped.",
    )
    ap.add_argument(
        "--partitions-dir",
        default=None,
        help="Directory with partition JSON files.",
    )
    ap.add_argument(
        "--slices-dir",
        default=None,
        help="Directory with quarterly parquet slices.",
    )
    ap.add_argument(
        "--out",
        default=None,
        help="Output CSV path.",
    )
    ap.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="Nearest semantic neighbors per lineage (default: %%(default)s).",
    )
    ap.add_argument(
        "--min-neighbors",
        type=int,
        default=0,
        help="Minimum active semantic neighbors for a lineage-quarter to "
             "receive convergence features. Lineages with fewer neighbors "
             "get NaN instead of 0, letting gradient boosting models treat "
             "isolated lineages differently from zero-convergence ones. "
             "Set to 0 (default) to disable (backward compatible).",
    )
    ap.add_argument("--verbose", action="store_true")
    add_domain_args(ap)
    return ap.parse_args()


def main() -> None:
    """Orchestrate convergence feature computation across all quarters."""
    args = parse_args()
    configure_logging(args.verbose)

    paths = resolve_script_paths(args, REPO_ROOT)
    apply_domain_path_defaults(args, paths, {
        "registry": ("lineage_tracking", "lineage_registry.json", "data/out/02_lineage_tracking/lineage_registry.json"),
        "timeseries": ("lineage_tracking", "lineage_timeseries.csv", "data/out/02_lineage_tracking/lineage_timeseries.csv"),
        "quarterly_embeddings": ("experiments", "stage1_quarterly_embeddings/quarterly_embeddings.npz", "data/out/experiments/stage1_quarterly_embeddings/quarterly_embeddings.npz"),
        "partitions_dir": ("cache_cum", "partitions_cum", "data/out/cache_cum/partitions_cum"),
        "slices_dir": ("slices", "", "data/current_ingest/slices"),
        "out": ("lineage_tracking", "convergence_features.csv", "data/out/02_lineage_tracking/convergence_features.csv"),
    })

    start_total = time.perf_counter()

    registry_path = Path(args.registry)
    timeseries_path = Path(args.timeseries)
    partitions_dir = Path(args.partitions_dir)
    slices_dir = Path(args.slices_dir)
    out_path = Path(args.out)

    LOG.info("Loading lineage registry from %s", registry_path)
    registry = load_registry(registry_path)

    LOG.info("Loading timeseries for quarter list from %s", timeseries_path)
    ts_df = pd.read_csv(timeseries_path)
    ts_df["quarter"] = ts_df["quarter"].astype(str)
    quarters_sorted = sorted(ts_df["quarter"].unique())
    LOG.info(
        "Quarters: %d (%s to %s)",
        len(quarters_sorted), quarters_sorted[0], quarters_sorted[-1],
    )

    # Load quarterly embeddings if available
    embeddings_path = Path(args.quarterly_embeddings)
    quarterly_embeddings: dict[str, dict[int, np.ndarray]] | None = None
    if embeddings_path.exists():
        LOG.info("Loading quarterly embeddings from %s", embeddings_path)
        quarterly_embeddings = load_quarterly_embeddings(embeddings_path)
        LOG.info("Embeddings loaded for %d quarters", len(quarterly_embeddings))
    else:
        LOG.warning(
            "Quarterly embeddings not found at %s; semantic channel will be skipped.",
            embeddings_path,
        )

    # Iterate quarters
    global_paper_lineage: dict[str, int] = {}
    quarterly_features: dict[str, dict[int, dict[str, float]]] = {}
    all_lineage_ids: set[int] = set()
    prev_semantic_sim: dict[int, list[tuple[int, float]]] | None = None

    for qi, quarter in enumerate(quarters_sorted):
        t0 = time.perf_counter()

        lineage_papers, lineage_authors, lineage_terms, paper_references = (
            build_quarter_data(
                quarter, registry, partitions_dir, slices_dir, global_paper_lineage,
            )
        )

        active_lineages = (
            set(lineage_papers.keys())
            | set(lineage_authors.keys())
            | set(lineage_terms.keys())
        )
        if not active_lineages:
            LOG.debug("No active lineages for %s, skipping", quarter)
            continue

        all_lineage_ids.update(active_lineages)

        # Channel 1: Semantic similarity
        semantic_sim: dict[int, list[tuple[int, float]]] = {}
        if quarterly_embeddings and quarter in quarterly_embeddings:
            q_emb = quarterly_embeddings[quarter]
            active_emb = {
                lid: q_emb[lid] for lid in active_lineages if lid in q_emb
            }
            if len(active_emb) >= 2:
                semantic_sim = compute_pairwise_semantic_similarity(
                    active_emb, top_k=args.top_k,
                )

        # Channel 2: Author overlap
        author_overlap = (
            compute_author_overlap(lineage_authors) if lineage_authors else {}
        )

        # Channel 3: Citation bridges
        citation_bridges: dict[int, dict[str, float]] = {}
        if lineage_papers and paper_references:
            citation_bridges = compute_citation_bridges(
                lineage_papers, paper_references, global_paper_lineage,
            )

        # Channel 4: Terminology Jaccard (requires semantic neighbors)
        terminology_jaccard: dict[int, dict[str, float]] = {}
        if lineage_terms and semantic_sim:
            terminology_jaccard = compute_terminology_jaccard(
                lineage_terms, semantic_sim, top_n=5,
            )

        # Aggregate all channels
        agg = aggregate_convergence_features(
            semantic_sim, prev_semantic_sim,
            author_overlap, citation_bridges, terminology_jaccard,
        )

        # Fill defaults for active lineages missing from aggregation
        for lid in active_lineages:
            if lid not in agg:
                agg[lid] = {
                    k: v for k, v in CONVERGENCE_FEATURE_DEFAULTS.items()
                    if not k.endswith(("_roll_2q", "_roll_4q", "_max_dev_4q"))
                }

        # Apply min-neighbors threshold: replace features with NaN for
        # lineages that have fewer than min_neighbors semantic neighbors.
        # This lets gradient boosting models distinguish "isolated" from
        # "measured zero convergence".
        min_neighbors = getattr(args, "min_neighbors", 0)
        if min_neighbors > 0:
            for lid in list(agg.keys()):
                n_sem_neighbors = len(semantic_sim.get(lid, []))
                if n_sem_neighbors < min_neighbors:
                    agg[lid] = {
                        k: float("nan") for k in agg[lid]
                    }

        quarterly_features[quarter] = agg
        prev_semantic_sim = semantic_sim if semantic_sim else prev_semantic_sim

        elapsed = time.perf_counter() - t0
        LOG.info(
            "[%d/%d] %s: %d lineages, sem=%d auth=%d cit=%d term=%d (%.1fs)",
            qi + 1, len(quarters_sorted), quarter,
            len(active_lineages),
            len(semantic_sim), len(author_overlap),
            len(citation_bridges), len(terminology_jaccard),
            elapsed,
        )

    # Compute rolling window features
    LOG.info(
        "Computing rolling window features across %d quarters...",
        len(quarterly_features),
    )
    rolling = compute_rolling_convergence_features(
        quarterly_features, quarters_sorted, all_lineage_ids,
    )

    # Build output DataFrame
    rows: list[dict[str, Any]] = []
    for (lid, quarter), features in sorted(rolling.items()):
        row: dict[str, Any] = {"lineage_id": lid, "quarter": quarter}
        for col, default_val in CONVERGENCE_FEATURE_DEFAULTS.items():
            row[col] = features.get(col, default_val)
        rows.append(row)

    if not rows:
        LOG.warning("No convergence features computed. Check input data.")
        return

    df_out = pd.DataFrame(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(out_path, index=False)

    elapsed_total = time.perf_counter() - start_total
    LOG.info(
        "Done. Wrote %d rows (%d lineages, %d quarters) to %s in %.1fs",
        len(df_out), df_out["lineage_id"].nunique(),
        df_out["quarter"].nunique(), out_path, elapsed_total,
    )


if __name__ == "__main__":
    main()
