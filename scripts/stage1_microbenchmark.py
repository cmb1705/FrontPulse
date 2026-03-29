"""
Stage 1 MICRO-BENCHMARK: 100 Lineage-Quarters

Goals:
- Measure actual runtime
- Profile GPU utilization
- Identify bottlenecks
- Determine optimal batch size

Outputs timing breakdown for optimization.
"""

import json
import time
from pathlib import Path

import pandas as pd
import torch
from _path_bootstrap import ensure_repo_imports
from transformers import AutoModel, AutoTokenizer

# Import AbstractExtractor
repo_root = ensure_repo_imports()
from scripts.extract_abstracts import AbstractExtractor  # noqa: E402


def load_sample_lineage_quarters(
    registry_path: Path,
    partitions_dir: Path,
    n_samples: int = 100
) -> pd.DataFrame:
    """Load sample of lineage-quarter assignments for benchmarking."""
    print(f"[MICRO-BENCHMARK] Loading ~{n_samples} lineage-quarters...")
    t0 = time.time()

    with open(registry_path) as f:
        registry = json.load(f)

    assignments = []

    # Take from middle quarters (2015-2020)
    all_quarters = sorted(registry.keys())
    start_idx = len(all_quarters) // 2
    sample_quarters = all_quarters[start_idx:start_idx + 20]  # 20 quarters for diversity

    for quarter in sample_quarters:
        community_map = registry[quarter]
        partition_path = partitions_dir / f"part_{quarter}.json"

        if not partition_path.exists():
            continue

        with open(partition_path) as f:
            partition_data = json.load(f)
            partition = partition_data.get('labels', {})

        if not partition:
            continue

        # Take first 100 papers per quarter
        for paper_id, community_id in list(partition.items())[:100]:
            lineage_id = community_map.get(str(community_id))
            if lineage_id is not None:
                assignments.append({
                    'paper_id': paper_id,
                    'lineage_id': lineage_id,
                    'quarter': quarter
                })

        if len(assignments) >= n_samples * 20:  # Enough papers
            break

    df = pd.DataFrame(assignments)
    t1 = time.time()

    print(f"   Loaded {len(df)} paper assignments in {t1-t0:.2f}s")
    print(f"   Lineages: {df['lineage_id'].nunique()}")
    print(f"   Quarters: {df['quarter'].nunique()}")

    return df, t1-t0


def aggregate_sample_texts(
    assignments_df: pd.DataFrame,
    extractor: AbstractExtractor,
    n_samples: int = 100
) -> tuple[list, float]:
    """Aggregate texts for sample lineage-quarters."""
    print("[MICRO-BENCHMARK] Aggregating texts...")
    t0 = time.time()

    # Group by lineage-quarter
    lineage_quarter_data = []

    for (lineage_id, quarter), group in assignments_df.groupby(['lineage_id', 'quarter']):
        papers = group['paper_id'].tolist()[:10]  # Max 10 papers per LQ

        # Get texts using batch method
        texts_dict = extractor.get_texts_batch(papers)
        texts = [texts_dict.get(pid, '') for pid in papers if pid in texts_dict]

        if not texts:
            continue

        # Concatenate
        combined_text = ' '.join(texts)[:3000]  # Limit to 3000 chars

        lineage_quarter_data.append((lineage_id, quarter, combined_text, len(texts)))

        if len(lineage_quarter_data) >= n_samples:
            break

    t1 = time.time()
    print(f"   Aggregated {len(lineage_quarter_data)} lineage-quarters in {t1-t0:.2f}s")
    return lineage_quarter_data, t1-t0


def benchmark_embedding(
    lineage_quarter_data: list,
    batch_size: int = 32
) -> tuple[dict, float, float]:
    """Benchmark embedding computation."""
    print(f"[MICRO-BENCHMARK] Testing embedding (batch_size={batch_size})...")

    # Load model
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')
    model = AutoModel.from_pretrained('allenai/scibert_scivocab_uncased')
    model.eval()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    t_model_load = time.time() - t0
    print(f"   Model loaded in {t_model_load:.2f}s, device: {device}")

    embeddings = {}

    # Extract data
    lineage_ids = [item[0] for item in lineage_quarter_data]
    quarters = [item[1] for item in lineage_quarter_data]
    texts = [item[2] for item in lineage_quarter_data]

    # Process in batches
    t_embed_start = time.time()
    n_batches = (len(texts) + batch_size - 1) // batch_size

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        batch_lineages = lineage_ids[i:i+batch_size]
        batch_quarters = quarters[i:i+batch_size]

        # Tokenize
        inputs = tokenizer(batch_texts, return_tensors='pt', max_length=512, truncation=True, padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Forward pass
        with torch.inference_mode():
            outputs = model(**inputs)

            # Mean pooling
            attention_mask = inputs['attention_mask']
            token_embeddings = outputs.last_hidden_state
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            batch_embeddings = (sum_embeddings / sum_mask).cpu().numpy()

        for j in range(len(batch_lineages)):
            key = (int(batch_lineages[j]), batch_quarters[j])
            embeddings[key] = batch_embeddings[j]

    t_embed = time.time() - t_embed_start

    print(f"   Computed {len(embeddings)} embeddings in {t_embed:.2f}s")
    print(f"   Throughput: {len(embeddings)/t_embed:.1f} embeddings/sec")
    print(f"   Time per batch ({batch_size}): {t_embed/n_batches:.3f}s")

    return embeddings, t_model_load, t_embed


def benchmark_velocity(embeddings: dict) -> tuple[pd.DataFrame, float]:
    """Benchmark velocity computation."""
    print("[MICRO-BENCHMARK] Testing velocity...")
    t0 = time.time()

    from collections import defaultdict

    from scipy.spatial.distance import cosine

    results = []

    # Group by lineage
    lineage_quarters = defaultdict(list)
    for (lineage_id, quarter), emb in embeddings.items():
        lineage_quarters[lineage_id].append((quarter, emb))

    for lineage_id in lineage_quarters:
        lineage_quarters[lineage_id].sort(key=lambda x: x[0])

    for lineage_id, quarter_embs in lineage_quarters.items():
        for i, (quarter, emb) in enumerate(quarter_embs):
            if i == 0:
                velocity = 0.0
            else:
                prev_quarter, prev_emb = quarter_embs[i-1]
                velocity = cosine(emb, prev_emb)

            results.append({
                'lineage_id': lineage_id,
                'quarter': quarter,
                'semantic_velocity': velocity
            })

    df = pd.DataFrame(results)
    t1 = time.time()

    print(f"   Computed velocity for {len(df)} lineage-quarters in {t1-t0:.2f}s")
    if len(df[df['semantic_velocity'] > 0]) > 0:
        print(f"   Mean velocity (non-zero): {df[df['semantic_velocity'] > 0]['semantic_velocity'].mean():.4f}")
        print(f"   Max velocity: {df['semantic_velocity'].max():.4f}")

    return df, t1-t0


def main():
    print("="*70)
    print("PHASE 1 MICRO-BENCHMARK: 100 Lineage-Quarters")
    print("="*70)
    print()

    # Paths
    registry_path = Path('data/out/02_lineage_tracking/lineage_registry.json')
    partitions_dir = Path('data/out/cache_cum/partitions_cum')
    raw_dir = Path('data/current_ingest/raw')

    timings = {}

    # Step 1: Load sample
    assignments_df, t_load = load_sample_lineage_quarters(registry_path, partitions_dir, n_samples=100)
    timings['data_loading'] = t_load

    if len(assignments_df) == 0:
        print("\n[BENCHMARK FAILED] No data loaded!")
        return

    # Step 2: Initialize extractor
    print("[MICRO-BENCHMARK] Initializing AbstractExtractor...")
    t0 = time.time()
    extractor = AbstractExtractor(raw_dir)
    t_extractor = time.time() - t0
    timings['extractor_init'] = t_extractor
    print(f"   Initialized in {t_extractor:.2f}s, indexed {len(extractor._work_to_store)} works")

    # Step 3: Aggregate texts
    lineage_quarter_data, t_aggregate = aggregate_sample_texts(assignments_df, extractor, n_samples=100)
    timings['text_aggregation'] = t_aggregate

    if len(lineage_quarter_data) == 0:
        print("\n[BENCHMARK FAILED] No lineage-quarter data aggregated!")
        return

    # Step 4: Benchmark embedding
    embeddings, t_model_load, t_embed = benchmark_embedding(lineage_quarter_data, batch_size=32)
    timings['model_loading'] = t_model_load
    timings['embedding'] = t_embed

    # Step 5: Benchmark velocity
    velocity_df, t_velocity = benchmark_velocity(embeddings)
    timings['velocity'] = t_velocity

    # Summary
    print("\n" + "="*70)
    print("TIMING BREAKDOWN")
    print("="*70)
    total_time = sum(timings.values())
    for step, t in timings.items():
        pct = (t / total_time) * 100
        print(f"   {step:20s}: {t:6.2f}s ({pct:5.1f}%)")
    print(f"   {'TOTAL':20s}: {total_time:6.2f}s")

    # Extrapolation to full dataset
    print("\n" + "="*70)
    print("EXTRAPOLATION TO FULL DATASET")
    print("="*70)
    n_full = 30000  # Estimated full size
    n_current = len(embeddings)
    scale_factor = n_full / n_current

    # Extractor init is one-time (already done)
    # Model loading is one-time (already done)
    # Data loading scales linearly
    # Text aggregation scales linearly
    # Embedding scales linearly
    # Velocity scales linearly

    t_full_estimated = (
        timings['extractor_init'] +  # One-time
        timings['model_loading'] +   # One-time
        timings['data_loading'] * scale_factor +
        timings['text_aggregation'] * scale_factor +
        timings['embedding'] * scale_factor +
        timings['velocity'] * scale_factor
    )

    print(f"   Current: {n_current} lineage-quarters in {total_time:.1f}s")
    print(f"   Estimated full: {n_full} lineage-quarters in {t_full_estimated/60:.1f} minutes")

    print("\n" + "="*70)
    print("MICRO-BENCHMARK COMPLETE")
    print("="*70)
    print("\nNext: Optimize based on bottlenecks, then scale test with 1000 samples")


if __name__ == '__main__':
    main()
