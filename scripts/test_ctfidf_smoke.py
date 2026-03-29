#!/usr/bin/env python3
"""
Smoke test for c-TF-IDF extraction - tests on single lineage to verify logic.
"""

import sys
import time
from pathlib import Path

from _path_bootstrap import ensure_repo_imports

ensure_repo_imports()

from scripts.compute_lineage_ctfidf import LineageTermExtractor  # noqa: E402


def main():
    print("=" * 70)
    print("SMOKE TEST: c-TF-IDF Extraction (Single Lineage)")
    print("=" * 70)

    # Initialize with minimal lineages (just test we can instantiate)
    extractor = LineageTermExtractor(
        registry_path=Path('data/out/02_lineage_tracking/lineage_registry.json'),
        partitions_dir=Path('data/out/cache_cum/partitions_cum'),
        raw_dir=Path('data/current_ingest/raw'),
        front_config_path=Path('config/front_aliases.yaml'),
        min_quarters=12
    )

    # Pick first lineage for smoke test
    if not extractor.persistent_lineages:
        print("ERROR: No persistent lineages found!")
        return 1

    test_lineage_id = extractor.persistent_lineages[0]
    print(f"\n[Test] Extracting terms from lineage {test_lineage_id}...")

    # Test: Load papers
    start = time.time()
    papers = extractor.load_lineage_papers_fast(test_lineage_id)
    print(f"  - Loaded {len(papers)} papers in {time.time()-start:.3f}s")

    if len(papers) == 0:
        print("ERROR: No papers found for lineage!")
        return 1

    # Test: Extract terms
    start = time.time()
    term_counts = extractor.extract_lineage_terms(test_lineage_id)
    print(f"  - Extracted {len(term_counts)} unique terms in {time.time()-start:.3f}s")

    # Show top 10 terms by frequency
    sorted_terms = sorted(term_counts.items(), key=lambda x: x[1], reverse=True)
    print("\n  Top 10 terms by frequency:")
    for i, (term, count) in enumerate(sorted_terms[:10], start=1):
        print(f"    {i:2d}. {term:20s} (count={count})")

    # Test: Get front terms
    print("\n[Test] Extracting terms from research fronts...")
    front_names = sorted(extractor.fronts_config.keys())
    print(f"  - Found {len(front_names)} fronts")

    # Test one front
    test_front = front_names[0]
    front_terms = extractor.get_front_terms(test_front)
    print(f"  - Front '{test_front}' has {len(front_terms)} terms")
    print(f"    Example terms: {list(front_terms)[:10]}")

    # Test: Compute document frequency for smoke test
    print("\n[Test] Computing DF for terms...")
    extractor.term_document_frequency.update(term_counts.keys())
    print(f"  - DF computed for {len(extractor.term_document_frequency)} terms")

    # Test: Compute c-TF-IDF
    print("\n[Test] Computing c-TF-IDF scores...")
    ctfidf_scores = extractor.compute_ctfidf_for_lineage(test_lineage_id, term_counts)
    sorted_ctfidf = sorted(ctfidf_scores.items(), key=lambda x: x[1], reverse=True)
    print(f"  - Computed {len(ctfidf_scores)} c-TF-IDF scores")
    print("\n  Top 10 terms by c-TF-IDF:")
    for i, (term, score) in enumerate(sorted_ctfidf[:10], start=1):
        print(f"    {i:2d}. {term:20s} (score={score:.2f})")

    # Test: Compute term similarity
    print("\n[Test] Computing term similarity to fronts...")
    for front_name in front_names[:3]:  # Test first 3 fronts
        front_terms = extractor.get_front_terms(front_name)
        similarity = extractor.compute_term_similarity(ctfidf_scores, front_terms)
        print(f"  - {front_name:30s}: {similarity:.4f}")

    print("\n" + "=" * 70)
    print("SMOKE TEST PASSED")
    print("=" * 70)
    return 0


if __name__ == '__main__':
    sys.exit(main())
