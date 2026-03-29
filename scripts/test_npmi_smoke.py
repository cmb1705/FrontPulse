#!/usr/bin/env python3
"""
Smoke test for NPMI co-term discovery - tests on single lineage to verify logic.
"""

import sys
import time
from pathlib import Path

from _path_bootstrap import ensure_repo_imports

ensure_repo_imports()

from scripts.compute_lineage_ctfidf import LineageTermExtractor  # noqa: E402
from scripts.compute_lineage_npmi import LineageNPMIAnalyzer  # noqa: E402


def main():
    print("=" * 70)
    print("SMOKE TEST: NPMI Co-term Discovery (Single Lineage)")
    print("=" * 70)

    # Initialize term extractor
    print("\n[1/4] Initializing LineageTermExtractor...")
    start = time.time()
    term_extractor = LineageTermExtractor(
        registry_path=Path('data/out/02_lineage_tracking/lineage_registry.json'),
        partitions_dir=Path('data/out/cache_cum/partitions_cum'),
        raw_dir=Path('data/current_ingest/raw'),
        front_config_path=Path('config/front_aliases.yaml'),
        min_quarters=12
    )
    print(f"      OK in {time.time()-start:.3f}s")
    print(f"      Found {len(term_extractor.persistent_lineages)} persistent lineages")

    # Pick first 10 lineages for smoke test
    if not term_extractor.persistent_lineages:
        print("ERROR: No persistent lineages found!")
        return 1

    test_lineages = term_extractor.persistent_lineages[:10]
    print(f"\n[2/4] Testing NPMI extraction on {len(test_lineages)} lineages...")

    # Initialize NPMI analyzer
    npmi_analyzer = LineageNPMIAnalyzer(
        term_extractor=term_extractor,
        text_extractor=term_extractor.text_extractor,
        min_npmi=0.1,
        min_pair_count=3
    )

    # Test: Extract term pairs for all 10 lineages
    overall_start = time.time()
    all_pairs = {}

    for lineage_id in test_lineages:
        start = time.time()
        pairs = npmi_analyzer.extract_lineage_term_pairs(lineage_id)
        elapsed = time.time() - start
        all_pairs[lineage_id] = pairs
        print(f"  Lineage {lineage_id:3d}: {len(pairs):5d} pairs in {elapsed:.3f}s")

    total_elapsed = time.time() - overall_start
    total_pairs = sum(len(p) for p in all_pairs.values())
    print(f"\n  Total: {total_pairs} pairs from {len(test_lineages)} lineages in {total_elapsed:.3f}s")
    print(f"  Average: {total_elapsed/len(test_lineages):.3f}s per lineage")

    if total_pairs == 0:
        print("ERROR: No NPMI pairs found for any lineage!")
        return 1

    # Show top 10 pairs from EACH lineage
    print("\n  Top 10 co-occurring pairs per lineage:")
    for lineage_id in test_lineages:
        pairs = all_pairs[lineage_id]
        sorted_pairs = sorted(pairs.items(), key=lambda x: x[1], reverse=True)
        print(f"\n  === Lineage {lineage_id} ===")
        for i, ((term1, term2), npmi) in enumerate(sorted_pairs[:10], start=1):
            print(f"    {i:2d}. ({term1:20s}, {term2:20s}) NPMI={npmi:.3f}")

    # Test: Get front terms
    print("\n[3/4] Testing front term matching...")
    front_names = sorted(term_extractor.fronts_config.keys())
    print(f"  - Found {len(front_names)} fronts")

    # Test similarity computation for first lineage and first 3 fronts
    print("\n[4/4] Computing NPMI similarity to fronts (sample)...")
    test_lineage_id = test_lineages[0]
    npmi_analyzer.lineage_npmi_pairs[test_lineage_id] = all_pairs[test_lineage_id]

    for front_name in front_names[:3]:
        front_terms = term_extractor.get_front_terms(front_name)
        similarity = npmi_analyzer.compute_front_npmi_similarity(test_lineage_id, front_terms)
        print(f"  - {front_name:30s}: {similarity:.4f}")

    print("\n" + "=" * 70)
    print("SMOKE TEST PASSED")
    print("=" * 70)
    avg_time = total_elapsed / len(test_lineages)
    avg_pairs = total_pairs / len(test_lineages)
    print("\nPerformance:")
    print(f"  - {len(test_lineages)} lineages processed in {total_elapsed:.3f}s")
    print(f"  - Average: {avg_time:.3f}s per lineage, {avg_pairs:.0f} pairs per lineage")
    print(f"  - Estimated time for 99 lineages: ~{avg_time * 99 / 60:.1f} minutes")
    return 0


if __name__ == '__main__':
    sys.exit(main())
