"""Tests for src.convergence -- cross-front convergence detection utilities.

Uses synthetic inputs to exercise pairwise similarity, author overlap,
citation bridging, terminology overlap, aggregation, and rolling features.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.convergence import (
    CONVERGENCE_FEATURE_DEFAULTS,
    aggregate_convergence_features,
    compute_author_overlap,
    compute_citation_bridges,
    compute_pairwise_semantic_similarity,
    compute_rolling_convergence_features,
    compute_terminology_jaccard,
)

pytestmark = pytest.mark.unit


# -- Helpers -----------------------------------------------------------------


def _random_unit_vectors(n: int, dim: int = 768, seed: int = 42) -> dict[int, np.ndarray]:
    """Generate *n* L2-normalized random vectors keyed by lineage ID."""
    rng = np.random.RandomState(seed)
    result = {}
    for i in range(1, n + 1):
        vec = rng.randn(dim).astype(np.float32)
        vec /= np.linalg.norm(vec)
        result[i] = vec
    return result


# -- Semantic similarity -----------------------------------------------------


class TestPairwiseSemanticSimilarity:

    def test_returns_top_k(self):
        """Each lineage should have exactly top_k neighbors."""
        embeddings = _random_unit_vectors(10)
        result = compute_pairwise_semantic_similarity(embeddings, top_k=5)

        for _lid, neighbors in result.items():
            assert len(neighbors) == 5
            # Sorted descending by similarity
            sims = [s for _, s in neighbors]
            assert sims == sorted(sims, reverse=True)

    def test_fewer_lineages_than_k(self):
        """When N < k, return min(k, N-1) neighbors."""
        embeddings = _random_unit_vectors(3)
        result = compute_pairwise_semantic_similarity(embeddings, top_k=20)

        for _lid, neighbors in result.items():
            assert len(neighbors) == 2  # 3 lineages, exclude self = 2

    def test_single_lineage_returns_empty(self):
        """With only one lineage, no neighbors are possible."""
        embeddings = _random_unit_vectors(1)
        result = compute_pairwise_semantic_similarity(embeddings, top_k=5)

        assert result[1] == []

    def test_identical_vectors_have_high_similarity(self):
        """Two identical vectors should have similarity close to 1.0."""
        vec = np.ones(768, dtype=np.float32)
        vec /= np.linalg.norm(vec)
        embeddings = {1: vec.copy(), 2: vec.copy(), 3: _random_unit_vectors(1)[1]}
        result = compute_pairwise_semantic_similarity(embeddings, top_k=2)

        # Lineage 1's top neighbor should be lineage 2 with sim near 1
        top_neighbor = result[1][0]
        assert top_neighbor[0] == 2
        assert top_neighbor[1] > 0.99


# -- Author overlap ----------------------------------------------------------


class TestAuthorOverlap:

    def test_no_shared_authors(self):
        """Disjoint author sets return zeros."""
        lineage_authors = {
            1: {"a1", "a2"},
            2: {"a3", "a4"},
        }
        result = compute_author_overlap(lineage_authors)

        assert result[1]["conv_author_migration_count"] == 0.0
        assert result[1]["conv_author_migration_rate"] == 0.0
        assert result[2]["conv_author_migration_count"] == 0.0

    def test_full_overlap(self):
        """Identical author sets return rate=1.0."""
        shared = {"a1", "a2", "a3"}
        lineage_authors = {1: shared, 2: shared.copy()}
        result = compute_author_overlap(lineage_authors)

        assert result[1]["conv_author_migration_rate"] == 1.0
        assert result[2]["conv_author_migration_rate"] == 1.0

    def test_empty_authors(self):
        """Lineage with no authors returns zeros."""
        lineage_authors = {1: set(), 2: {"a1"}}
        result = compute_author_overlap(lineage_authors)

        assert result[1]["conv_author_migration_count"] == 0.0
        assert result[1]["conv_author_migration_rate"] == 0.0


# -- Citation bridges --------------------------------------------------------


class TestCitationBridges:

    def test_bridge_detection(self):
        """Paper citing works from 2 distinct lineages counts as a bridge."""
        lineage_papers = {1: {"p1", "p2"}, 2: {"p3"}}
        # p1 references p3 (in lineage 2) and p_ext (in lineage 1)
        paper_references = {
            "p1": ["p3", "p_ext"],
            "p2": ["p_ext"],
            "p3": [],
        }
        paper_lineage = {"p1": 1, "p2": 1, "p3": 2, "p_ext": 1}
        result = compute_citation_bridges(lineage_papers, paper_references, paper_lineage)

        # p1 references lineage 2 (p3) and lineage 1 (p_ext) = bridge
        assert result[1]["conv_citation_bridge_count"] == 1.0
        assert result[1]["conv_citation_bridge_rate"] == 0.5  # 1 of 2 papers

    def test_no_bridges(self):
        """Papers referencing only their own lineage produce no bridges."""
        lineage_papers = {1: {"p1"}, 2: {"p2"}}
        paper_references = {"p1": ["p_int1"], "p2": ["p_int2"]}
        paper_lineage = {"p1": 1, "p_int1": 1, "p2": 2, "p_int2": 2}
        result = compute_citation_bridges(lineage_papers, paper_references, paper_lineage)

        assert result[1]["conv_citation_bridge_count"] == 0.0
        assert result[2]["conv_citation_bridge_count"] == 0.0


# -- Terminology overlap -----------------------------------------------------


class TestTerminologyJaccard:

    def test_identical_sets(self):
        """Identical term sets with self-neighbors returns Jaccard=1.0."""
        terms = {1: {"gene", "editing", "crispr"}, 2: {"gene", "editing", "crispr"}}
        neighbors = {1: [(2, 0.9)], 2: [(1, 0.9)]}
        result = compute_terminology_jaccard(terms, neighbors, top_n=1)

        assert result[1]["conv_terminology_overlap"] == pytest.approx(1.0)

    def test_disjoint_sets(self):
        """Disjoint term sets return Jaccard=0.0."""
        terms = {1: {"alpha", "beta"}, 2: {"gamma", "delta"}}
        neighbors = {1: [(2, 0.5)], 2: [(1, 0.5)]}
        result = compute_terminology_jaccard(terms, neighbors, top_n=1)

        assert result[1]["conv_terminology_overlap"] == pytest.approx(0.0)


# -- Aggregation -------------------------------------------------------------


class TestAggregation:

    def test_composite_score_near_expected_range(self):
        """Composite score should be near [0, 1] for typical inputs.

        Random unit vectors can produce slightly negative cosine
        similarity, so the composite may be fractionally below zero.
        """
        embeddings = _random_unit_vectors(5)
        sem_sim = compute_pairwise_semantic_similarity(embeddings, top_k=3)
        authors = {i: {f"a{i}"} for i in range(1, 6)}
        auth = compute_author_overlap(authors)
        papers = {i: {f"p{i}"} for i in range(1, 6)}
        refs = {f"p{i}": [] for i in range(1, 6)}
        plin = {f"p{i}": i for i in range(1, 6)}
        cit = compute_citation_bridges(papers, refs, plin)
        terms = {i: {f"t{i}"} for i in range(1, 6)}
        tj = compute_terminology_jaccard(terms, sem_sim, top_n=3)

        agg = aggregate_convergence_features(sem_sim, None, auth, cit, tj)
        for _lid, features in agg.items():
            score = features["conv_composite_score"]
            assert -0.01 <= score <= 1.01

    def test_semantic_velocity_positive_on_converging(self):
        """Converging pair should have positive semantic velocity."""
        # Previous quarter: low similarity
        prev_sim = {1: [(2, 0.3)]}
        # Current quarter: high similarity
        cur_sim = {1: [(2, 0.9)]}
        auth = {1: {"a1"}}
        auth_result = compute_author_overlap(auth)
        cit = {1: {"conv_citation_bridge_count": 0.0, "conv_citation_bridge_rate": 0.0}}
        tj = {1: {"conv_terminology_overlap": 0.0}}

        agg = aggregate_convergence_features(cur_sim, prev_sim, auth_result, cit, tj)
        assert agg[1]["conv_semantic_velocity"] > 0


# -- Rolling features --------------------------------------------------------


class TestRollingFeatures:

    def test_short_history(self):
        """Lineage with < 4 quarters still gets rolling features."""
        quarterly = {
            "2020Q1": {1: {"conv_composite_score": 0.5, "conv_max_semantic_sim": 0.3}},
            "2020Q2": {1: {"conv_composite_score": 0.6, "conv_max_semantic_sim": 0.4}},
        }
        result = compute_rolling_convergence_features(
            quarterly, ["2020Q1", "2020Q2"], {1},
        )
        assert (1, "2020Q2") in result
        features = result[(1, "2020Q2")]
        # 2q mean of [0.5, 0.6] = 0.55
        assert features["conv_composite_score_roll_2q"] == pytest.approx(0.55)
        # Only 2 quarters, so 4q mean uses available data
        assert "conv_composite_score_roll_4q" in features
        assert "conv_composite_score_max_dev_4q" in features


# -- Defaults dict -----------------------------------------------------------


class TestDefaults:

    def test_defaults_cover_all_features(self):
        """CONVERGENCE_FEATURE_DEFAULTS should have all 14 base+rolling features."""
        assert len(CONVERGENCE_FEATURE_DEFAULTS) >= 14
        assert all(
            k.startswith("conv_") for k in CONVERGENCE_FEATURE_DEFAULTS
        )
