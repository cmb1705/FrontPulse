"""Cross-front convergence detection utilities.

Computes pairwise interaction signals across four channels (semantic
drift, author migration, citation bridging, terminological overlap)
and aggregates them to per-lineage features for MSD integration.

All functions are **pure** -- no file I/O, no side effects -- so they
can be unit-tested with small synthetic inputs.
"""

from __future__ import annotations

import logging
from collections import defaultdict

import numpy as np

LOG = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Channel 1: Semantic similarity
# ---------------------------------------------------------------------------


def compute_pairwise_semantic_similarity(
    embeddings: dict[int, np.ndarray],
    top_k: int = 20,
) -> dict[int, list[tuple[int, float]]]:
    """Top-k most similar lineages per lineage via cosine similarity.

    Uses brute-force pairwise cosine distance (tractable for N <= ~500
    lineages active per quarter).

    Args:
        embeddings: lineage_id -> L2-normalized 768-d embedding vector.
        top_k: Number of nearest neighbors to return per lineage.

    Returns:
        lineage_id -> [(neighbor_id, cosine_similarity), ...] sorted
        descending by similarity.
    """
    ids = sorted(embeddings.keys())
    n = len(ids)
    if n < 2:
        return {lid: [] for lid in ids}

    k = min(top_k, n - 1)
    matrix = np.stack([embeddings[lid] for lid in ids])

    # Cosine similarity via dot product (assumes L2-normalized vectors).
    # Fall back to explicit normalization if norms aren't unit length.
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    matrix = matrix / norms
    sim_matrix = matrix @ matrix.T

    result: dict[int, list[tuple[int, float]]] = {}
    for i, lid in enumerate(ids):
        row = sim_matrix[i].copy()
        row[i] = -1.0  # exclude self
        top_idx = np.argpartition(row, -k)[-k:]
        top_idx = top_idx[np.argsort(row[top_idx])[::-1]]
        result[lid] = [(ids[j], float(row[j])) for j in top_idx]

    return result


# ---------------------------------------------------------------------------
# Channel 2: Author migration
# ---------------------------------------------------------------------------


def compute_author_overlap(
    lineage_authors: dict[int, set[str]],
) -> dict[int, dict[str, float]]:
    """Author migration features per lineage.

    For each lineage, count how many of its authors also publish in at
    least one other lineage during the same quarter.

    Args:
        lineage_authors: lineage_id -> set of author IDs active this quarter.

    Returns:
        lineage_id -> {conv_author_migration_count, conv_author_migration_rate}.
    """
    # Build inverted index: author -> set of lineages
    author_lineages: dict[str, set[int]] = defaultdict(set)
    for lid, authors in lineage_authors.items():
        for aid in authors:
            author_lineages[aid].add(lid)

    result: dict[int, dict[str, float]] = {}
    for lid, authors in lineage_authors.items():
        total = len(authors)
        if total == 0:
            result[lid] = {
                "conv_author_migration_count": 0.0,
                "conv_author_migration_rate": 0.0,
            }
            continue

        migrating = sum(
            1 for aid in authors if len(author_lineages[aid]) > 1
        )
        result[lid] = {
            "conv_author_migration_count": float(migrating),
            "conv_author_migration_rate": float(migrating) / float(total),
        }

    return result


# ---------------------------------------------------------------------------
# Channel 3: Citation bridging
# ---------------------------------------------------------------------------


def compute_citation_bridges(
    lineage_papers: dict[int, set[str]],
    paper_references: dict[str, list[str]],
    paper_lineage: dict[str, int],
) -> dict[int, dict[str, float]]:
    """Count papers that cite works from 2+ distinct lineages.

    A paper is a "bridge" if its outbound references span at least two
    distinct lineages (including or excluding its own).

    Args:
        lineage_papers: lineage_id -> set of paper IDs in this quarter.
        paper_references: paper_id -> list of referenced paper IDs.
        paper_lineage: paper_id -> lineage_id for all known papers
            (across all quarters).

    Returns:
        lineage_id -> {conv_citation_bridge_count, conv_citation_bridge_rate}.
    """
    result: dict[int, dict[str, float]] = {}

    for lid, papers in lineage_papers.items():
        total = len(papers)
        bridges = 0

        for pid in papers:
            refs = paper_references.get(pid, [])
            if not refs:
                continue
            ref_lineages = {
                paper_lineage[r] for r in refs if r in paper_lineage
            }
            # Bridge = references span 2+ distinct lineages
            if len(ref_lineages) >= 2:
                bridges += 1

        result[lid] = {
            "conv_citation_bridge_count": float(bridges),
            "conv_citation_bridge_rate": (
                float(bridges) / float(total) if total > 0 else 0.0
            ),
        }

    return result


# ---------------------------------------------------------------------------
# Channel 4: Terminological overlap
# ---------------------------------------------------------------------------


def compute_terminology_jaccard(
    lineage_terms: dict[int, set[str]],
    nearest_neighbors: dict[int, list[tuple[int, float]]],
    top_n: int = 5,
) -> dict[int, dict[str, float]]:
    """Jaccard similarity of term sets with nearest semantic neighbors.

    For each lineage, compute the mean Jaccard index of its term set
    against the term sets of its top-n nearest semantic neighbors.

    Args:
        lineage_terms: lineage_id -> set of terms used this quarter.
        nearest_neighbors: lineage_id -> [(neighbor_id, sim), ...]
            from :func:`compute_pairwise_semantic_similarity`.
        top_n: Number of nearest neighbors to include in the mean.

    Returns:
        lineage_id -> {conv_terminology_overlap}.
    """
    result: dict[int, dict[str, float]] = {}

    for lid, terms_a in lineage_terms.items():
        neighbors = nearest_neighbors.get(lid, [])[:top_n]
        if not neighbors or not terms_a:
            result[lid] = {"conv_terminology_overlap": 0.0}
            continue

        jaccards: list[float] = []
        for nid, _ in neighbors:
            terms_b = lineage_terms.get(nid, set())
            if not terms_b:
                jaccards.append(0.0)
                continue
            intersection = len(terms_a & terms_b)
            union = len(terms_a | terms_b)
            jaccards.append(float(intersection) / float(union) if union > 0 else 0.0)

        result[lid] = {
            "conv_terminology_overlap": float(np.mean(jaccards)),
        }

    return result


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def aggregate_convergence_features(
    semantic_sim: dict[int, list[tuple[int, float]]],
    prev_semantic_sim: dict[int, list[tuple[int, float]]] | None,
    author_overlap: dict[int, dict[str, float]],
    citation_bridges: dict[int, dict[str, float]],
    terminology_jaccard: dict[int, dict[str, float]],
) -> dict[int, dict[str, float]]:
    """Combine all four channels into per-lineage feature dict.

    Args:
        semantic_sim: Current quarter's top-k neighbors per lineage.
        prev_semantic_sim: Previous quarter's top-k neighbors (for velocity).
        author_overlap: Author migration features per lineage.
        citation_bridges: Citation bridge features per lineage.
        terminology_jaccard: Terminology overlap features per lineage.

    Returns:
        lineage_id -> dict with all ``conv_*`` base features.
    """
    all_ids = set(semantic_sim.keys())
    all_ids.update(author_overlap.keys())
    all_ids.update(citation_bridges.keys())
    all_ids.update(terminology_jaccard.keys())

    # Build lookup for previous quarter max similarity
    prev_max: dict[int, float] = {}
    if prev_semantic_sim:
        for lid, neighbors in prev_semantic_sim.items():
            prev_max[lid] = neighbors[0][1] if neighbors else 0.0

    result: dict[int, dict[str, float]] = {}
    for lid in all_ids:
        # Semantic channel
        neighbors = semantic_sim.get(lid, [])
        max_sim = neighbors[0][1] if neighbors else 0.0
        top5 = neighbors[:5]
        mean_top5 = float(np.mean([s for _, s in top5])) if top5 else 0.0

        prev_val = prev_max.get(lid)
        velocity = (max_sim - prev_val) if prev_val is not None else 0.0

        # Author channel
        auth = author_overlap.get(lid, {})
        mig_count = auth.get("conv_author_migration_count", 0.0)
        mig_rate = auth.get("conv_author_migration_rate", 0.0)

        # Citation channel
        cit = citation_bridges.get(lid, {})
        bridge_count = cit.get("conv_citation_bridge_count", 0.0)
        bridge_rate = cit.get("conv_citation_bridge_rate", 0.0)

        # Terminology channel
        term = terminology_jaccard.get(lid, {})
        term_overlap = term.get("conv_terminology_overlap", 0.0)

        # Composite score: equal-weighted mean of bounded [0,1] signals.
        # Semantic: max_sim already [0,1]; author: mig_rate [0,1];
        # citation: bridge_rate [0,1]; terminology: term_overlap [0,1].
        composite = float(np.mean([max_sim, mig_rate, bridge_rate, term_overlap]))

        result[lid] = {
            "conv_max_semantic_sim": max_sim,
            "conv_mean_top5_sim": mean_top5,
            "conv_semantic_velocity": velocity,
            "conv_author_migration_count": mig_count,
            "conv_author_migration_rate": mig_rate,
            "conv_citation_bridge_count": bridge_count,
            "conv_citation_bridge_rate": bridge_rate,
            "conv_terminology_overlap": term_overlap,
            "conv_composite_score": composite,
        }

    return result


# ---------------------------------------------------------------------------
# Rolling window features
# ---------------------------------------------------------------------------

# Features that get rolling window derivatives
_ROLLING_TARGETS = ("conv_composite_score", "conv_max_semantic_sim")


def compute_rolling_convergence_features(
    quarterly_features: dict[str, dict[int, dict[str, float]]],
    quarters_sorted: list[str],
    lineage_ids: set[int],
) -> dict[tuple[int, str], dict[str, float]]:
    """Compute trailing rolling-window derivatives.

    For each target feature, compute 2-quarter and 4-quarter trailing
    means, plus max deviation from the 4-quarter mean.

    Args:
        quarterly_features: quarter -> lineage_id -> base feature dict.
        quarters_sorted: Chronologically sorted quarter labels.
        lineage_ids: Set of all lineage IDs across all quarters.

    Returns:
        (lineage_id, quarter) -> dict with ``_roll_2q``, ``_roll_4q``,
        ``_max_dev_4q`` suffixed features, plus all base features.
    """
    result: dict[tuple[int, str], dict[str, float]] = {}

    for lid in lineage_ids:
        # Collect per-quarter values for this lineage
        history: dict[str, dict[str, float]] = {}
        for q in quarters_sorted:
            qf = quarterly_features.get(q, {})
            if lid in qf:
                history[q] = qf[lid]

        active_quarters = [q for q in quarters_sorted if q in history]

        for qi, q in enumerate(active_quarters):
            base = dict(history[q])  # copy base features

            for target in _ROLLING_TARGETS:
                base.get(target, 0.0)

                # 2-quarter trailing mean
                window_2 = [
                    history[active_quarters[j]].get(target, 0.0)
                    for j in range(max(0, qi - 1), qi + 1)
                ]
                roll_2q = float(np.mean(window_2))

                # 4-quarter trailing mean
                window_4 = [
                    history[active_quarters[j]].get(target, 0.0)
                    for j in range(max(0, qi - 3), qi + 1)
                ]
                roll_4q = float(np.mean(window_4))

                # Max deviation from 4q mean
                max_dev = max(abs(w - roll_4q) for w in window_4)

                prefix = target
                base[f"{prefix}_roll_2q"] = roll_2q
                base[f"{prefix}_roll_4q"] = roll_4q
                base[f"{prefix}_max_dev_4q"] = max_dev

            result[(lid, q)] = base

    return result


# ---------------------------------------------------------------------------
# Default feature dict (for missing data)
# ---------------------------------------------------------------------------

#: All convergence feature names with their zero-defaults.
CONVERGENCE_FEATURE_DEFAULTS: dict[str, float] = {
    "conv_max_semantic_sim": 0.0,
    "conv_mean_top5_sim": 0.0,
    "conv_semantic_velocity": 0.0,
    "conv_author_migration_count": 0.0,
    "conv_author_migration_rate": 0.0,
    "conv_citation_bridge_count": 0.0,
    "conv_citation_bridge_rate": 0.0,
    "conv_terminology_overlap": 0.0,
    "conv_composite_score": 0.0,
    "conv_composite_score_roll_2q": 0.0,
    "conv_composite_score_roll_4q": 0.0,
    "conv_composite_score_max_dev_4q": 0.0,
    "conv_max_semantic_sim_roll_2q": 0.0,
    "conv_max_semantic_sim_roll_4q": 0.0,
}
