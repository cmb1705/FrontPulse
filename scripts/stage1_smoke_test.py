"""
Stage 1 SMOKE TEST: Quarterly Embeddings with Limited Data

Test with 10 lineage-quarters to verify:
1. Data loading works
2. AbstractExtractor works
3. Embedding computation works
4. Velocity computation works

Then hand off to user for full run.
"""

from pathlib import Path
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch
import json
from typing import Dict, Tuple
import sys

# Import AbstractExtractor
repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))
from scripts.extract_abstracts import AbstractExtractor


def load_sample_lineage_quarters(
    registry_path: Path,
    partitions_dir: Path,
    n_samples: int = 10
) -> pd.DataFrame:
    """Load SMALL sample of lineage-quarter assignments for smoke test."""
    print(f"[SMOKE TEST] Loading {n_samples} sample lineage-quarters...")

    with open(registry_path) as f:
        registry = json.load(f)

    assignments = []

    # Take quarters from middle of range (2015-2020 likely has data)
    all_quarters = sorted(registry.keys())
    start_idx = len(all_quarters) // 2  # Start from middle
    sample_quarters = all_quarters[start_idx:start_idx + 10]  # Take 10 quarters

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

        # Take only first 50 papers per quarter
        for paper_id, community_id in list(partition.items())[:50]:
            lineage_id = community_map.get(str(community_id))
            if lineage_id is not None:
                assignments.append({
                    'paper_id': paper_id,
                    'lineage_id': lineage_id,
                    'quarter': quarter
                })

        if len(assignments) >= n_samples * 10:  # Enough papers for N lineage-quarters
            break

    df = pd.DataFrame(assignments)
    print(f"   Loaded {len(df)} paper assignments")

    if len(df) == 0:
        print("   ERROR: No assignments loaded!")
        return df

    print(f"   Lineages: {df['lineage_id'].nunique()}")
    print(f"   Quarters: {df['quarter'].nunique()}")

    return df


def aggregate_sample_texts(
    assignments_df: pd.DataFrame,
    extractor: AbstractExtractor,
    n_samples: int = 10
) -> list:
    """Aggregate texts for sample lineage-quarters."""
    print(f"[SMOKE TEST] Aggregating texts...")

    # Group by lineage-quarter
    lineage_quarter_data = []

    for (lineage_id, quarter), group in assignments_df.groupby(['lineage_id', 'quarter']):
        papers = group['paper_id'].tolist()[:5]  # Max 5 papers per LQ

        # Get texts using batch method
        texts_dict = extractor.get_texts_batch(papers)
        texts = [texts_dict.get(pid, '') for pid in papers if pid in texts_dict]

        if not texts:
            continue

        # Concatenate
        combined_text = ' '.join(texts)[:2000]  # Short for smoke test

        lineage_quarter_data.append((lineage_id, quarter, combined_text, len(texts)))

        if len(lineage_quarter_data) >= n_samples:
            break

    print(f"   Aggregated {len(lineage_quarter_data)} lineage-quarters")
    return lineage_quarter_data


def test_embedding(lineage_quarter_data: list) -> Dict[Tuple[int, str], np.ndarray]:
    """Test embedding computation on small sample."""
    print(f"[SMOKE TEST] Testing embedding computation...")

    # Load model
    print(f"   Loading SciBERT...")
    tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')
    model = AutoModel.from_pretrained('allenai/scibert_scivocab_uncased')
    model.eval()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    print(f"   Device: {device}")

    embeddings = {}

    # Process in single batch
    lineage_ids = [item[0] for item in lineage_quarter_data]
    quarters = [item[1] for item in lineage_quarter_data]
    texts = [item[2] for item in lineage_quarter_data]

    print(f"   Tokenizing {len(texts)} texts...")
    inputs = tokenizer(texts, return_tensors='pt', max_length=512, truncation=True, padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    print(f"   Computing embeddings...")
    with torch.inference_mode():
        outputs = model(**inputs)

        # Mean pooling
        attention_mask = inputs['attention_mask']
        token_embeddings = outputs.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        batch_embeddings = (sum_embeddings / sum_mask).cpu().numpy()

    for i in range(len(lineage_ids)):
        key = (int(lineage_ids[i]), quarters[i])
        embeddings[key] = batch_embeddings[i]

    print(f"   Computed {len(embeddings)} embeddings")
    print(f"   Embedding shape: {batch_embeddings[0].shape}")

    return embeddings


def test_velocity(embeddings: Dict) -> pd.DataFrame:
    """Test velocity computation."""
    print(f"[SMOKE TEST] Testing velocity computation...")

    from scipy.spatial.distance import cosine

    results = []

    # Group by lineage
    from collections import defaultdict
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
    print(f"   Computed velocity for {len(df)} lineage-quarters")
    if len(df) > 0:
        print(f"   Mean velocity: {df['semantic_velocity'].mean():.4f}")
        print(f"   Max velocity: {df['semantic_velocity'].max():.4f}")

    return df


def main():
    print("="*70)
    print("PHASE 1 SMOKE TEST: 10 Lineage-Quarters")
    print("="*70)
    print()

    # Paths
    registry_path = Path('data/out/02_lineage_tracking/lineage_registry.json')
    partitions_dir = Path('data/out/cache_cum/partitions_cum')
    raw_dir = Path('data/current_ingest/raw')

    # Step 1: Load small sample
    assignments_df = load_sample_lineage_quarters(registry_path, partitions_dir, n_samples=10)

    if len(assignments_df) == 0:
        print("\n[SMOKE TEST FAILED] No data loaded!")
        return

    # Step 2: Initialize extractor
    print("[SMOKE TEST] Initializing AbstractExtractor...")
    extractor = AbstractExtractor(raw_dir)
    print(f"   Indexed {len(extractor._work_to_store)} works")

    # Step 3: Aggregate texts
    lineage_quarter_data = aggregate_sample_texts(assignments_df, extractor, n_samples=10)

    if len(lineage_quarter_data) == 0:
        print("ERROR: No lineage-quarter data aggregated!")
        return

    # Step 4: Test embedding
    embeddings = test_embedding(lineage_quarter_data)

    # Step 5: Test velocity
    velocity_df = test_velocity(embeddings)

    # Verify outputs
    print("\n[SMOKE TEST] Verification:")
    print(f"   Embeddings: {len(embeddings)} computed")
    print(f"   Velocities: {len(velocity_df)} computed")
    print(f"   Embedding dtype: {list(embeddings.values())[0].dtype}")
    print(f"   Velocity range: [{velocity_df['semantic_velocity'].min():.4f}, {velocity_df['semantic_velocity'].max():.4f}]")

    print("\n" + "="*70)
    print("SMOKE TEST PASSED")
    print("="*70)
    print("\nNext: Micro-benchmark with 100 lineage-quarters to measure performance")


if __name__ == '__main__':
    main()
