"""
Profile a single lineage embedding computation with detailed timing.
"""
import time
import sys
from pathlib import Path
import json
import pickle

# Add repo to path
repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from scripts.extract_abstracts import AbstractExtractor
from scripts.compute_lineage_embeddings import LineageEmbedder, load_lineage_papers_fast


def profile_single_lineage(lineage_id: int = 2):
    """Profile embedding computation for a single lineage."""

    print(f"\n{'='*70}")
    print(f"PROFILING LINEAGE {lineage_id}")
    print(f"{'='*70}\n")

    # Paths
    raw_dir = Path("data/current_ingest/raw")
    registry_path = Path("data/out/02_lineage_tracking/lineage_registry.json")

    # Step 1: Load lineage registry
    print("[1/6] Loading lineage registry...")
    t0 = time.time()
    with open(registry_path, 'r') as f:
        lineage_registry = json.load(f)
    t1 = time.time()
    print(f"      OK - Loaded in {t1-t0:.3f}s\n")

    # Step 2: Load papers for this lineage (using cached JSON partitions)
    print(f"[2/6] Loading papers for lineage {lineage_id} from cached partitions...")
    t2 = time.time()
    papers = load_lineage_papers_fast(lineage_id, lineage_registry)
    t3 = time.time()
    print(f"      OK - Found {len(papers)} papers in {t3-t2:.3f}s\n")

    if len(papers) == 0:
        print(f"ERROR: No papers found for lineage {lineage_id}. Cannot continue.\n")
        return None

    # Step 3: Initialize embedder (loads SciBERT + abstract extractor)
    print("[3/6] Initializing SciBERT and abstract extractor...")
    t4 = time.time()
    embedder = LineageEmbedder(raw_dir, device='cuda')
    t5 = time.time()
    print(f"      OK - Initialized in {t5-t4:.3f}s\n")

    # Step 4: Extract texts from raw JSONL (with detailed breakdown)
    print(f"[4/6] Extracting {len(papers)} texts from raw JSONL...")
    t6 = time.time()
    texts = embedder.extractor.get_texts_batch(papers, include_title=True)
    t7 = time.time()
    print(f"      OK - Extracted {len(texts)} texts in {t7-t6:.3f}s")
    print(f"        ({len(texts)/len(papers)*100:.1f}% coverage)\n")

    # Step 5: Stopword filtering
    print(f"[5/6] Filtering stopwords from {len(texts)} texts...")
    t8 = time.time()
    text_list = list(texts.values())
    filtered_texts = [embedder.filter_stopwords(t) for t in text_list]
    t9 = time.time()
    print(f"      OK - Filtered in {t9-t8:.3f}s\n")

    # Step 6: SciBERT embedding computation (GPU)
    print(f"[6/6] Computing SciBERT embeddings (batch_size=32, GPU)...")
    t10 = time.time()
    embeddings = embedder.embed_texts_batch(text_list, batch_size=32)
    t11 = time.time()
    print(f"      OK - Computed embeddings in {t11-t10:.3f}s")
    print(f"        ({len(text_list)/(t11-t10):.1f} texts/second)\n")

    # Summary
    total_time = t11 - t0
    print(f"{'='*70}")
    print("TIMING BREAKDOWN")
    print(f"{'='*70}")
    print(f"1. Load registry:        {t1-t0:7.3f}s  ({(t1-t0)/total_time*100:5.1f}%)")
    print(f"2. Load papers (PKL):    {t3-t2:7.3f}s  ({(t3-t2)/total_time*100:5.1f}%)")
    print(f"3. Init embedder:        {t5-t4:7.3f}s  ({(t5-t4)/total_time*100:5.1f}%)")
    print(f"4. Extract texts (I/O):  {t7-t6:7.3f}s  ({(t7-t6)/total_time*100:5.1f}%)")
    print(f"5. Filter stopwords:     {t9-t8:7.3f}s  ({(t9-t8)/total_time*100:5.1f}%)")
    print(f"6. SciBERT (GPU):        {t11-t10:7.3f}s  ({(t11-t10)/total_time*100:5.1f}%)")
    print(f"{'-'*70}")
    print(f"TOTAL:                   {total_time:7.3f}s")
    print(f"{'='*70}\n")

    # Identify bottleneck
    timings = {
        'Load registry': t1-t0,
        'Load papers (PKL)': t3-t2,
        'Init embedder': t5-t4,
        'Extract texts (I/O)': t7-t6,
        'Filter stopwords': t9-t8,
        'SciBERT (GPU)': t11-t10
    }

    bottleneck = max(timings.items(), key=lambda x: x[1])
    print(f"🔴 PRIMARY BOTTLENECK: {bottleneck[0]} ({bottleneck[1]:.3f}s, {bottleneck[1]/total_time*100:.1f}%)")

    return timings


if __name__ == "__main__":
    profile_single_lineage()
