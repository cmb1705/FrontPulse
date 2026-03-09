"""
Stage 1: Quarterly SciBERT Embeddings for True Semantic Velocity

Computes quarterly embeddings for each lineage (not aggregate) to enable
measurement of semantic velocity over time.

This addresses the Stage 0 limitation where we used growth volatility as a proxy.
"""

from pathlib import Path
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch
from collections import defaultdict
import json
import shutil
from typing import Dict, List

STAGE1_OUTPUT_DIR = Path('data/out/experiments/stage1_quarterly_embeddings')
LEGACY_PHASE1_OUTPUT_DIR = Path('data/out/experiments/phase1_quarterly_embeddings')
PREFERRED_PAPERS_PATH = Path('data/out/01_community_detection/leiden_papers.parquet')
FALLBACK_PAPERS_PATH = Path('data/current_ingest/ingest.parquet')
STAGE0_TIGHT_MAPPING = Path('data/out/experiments/stage0_tight_mapping/milestone_lineage_mapping_tight.csv')
LEGACY_PHASE0_TIGHT_MAPPING = Path('data/out/experiments/phase0_tight_mapping/milestone_lineage_mapping_tight.csv')

def load_lineage_data(
    registry_path: Path,
    partitions_dir: Path,
    papers_path: Path
) -> pd.DataFrame:
    """Load lineage assignments from registry + partitions."""
    print("[1/6] Loading lineage data...")

    # Load lineage registry: {quarter: {community_id: lineage_id}}
    with open(registry_path) as f:
        registry = json.load(f)

    print(f"   Loaded lineage registry: {len(registry)} quarters")

    # Extract paper-lineage-quarter assignments from partitions
    assignments = []

    for quarter, community_map in registry.items():
        # Load partition for this quarter
        partition_path = partitions_dir / f"part_{quarter}.json"

        if not partition_path.exists():
            continue

        with open(partition_path) as f:
            partition_data = json.load(f)
            partition = partition_data.get('labels', {})  # {paper_id: community_id}

        # Map papers to lineages
        for paper_id, community_id in partition.items():
            lineage_id = community_map.get(str(community_id))

            if lineage_id is not None:
                assignments.append({
                    'paper_id': paper_id,
                    'lineage_id': lineage_id,
                    'quarter': quarter
                })

    assignments_df = pd.DataFrame(assignments)
    print(f"   Extracted {len(assignments_df)} paper-lineage-quarter assignments")
    print(f"   Lineages: {assignments_df['lineage_id'].nunique()}")
    print(f"   Quarters: {assignments_df['quarter'].nunique()}")

    # Load paper metadata
    papers_df = pd.read_parquet(papers_path)
    print(f"   Loaded {len(papers_df)} papers")

    # Merge to get titles and abstracts
    merged = assignments_df.merge(
        papers_df[['paper_id', 'title', 'abstract']],
        left_on='paper_id',
        right_on='paper_id',
        how='left'
    )

    print(f"   Merged dataset: {len(merged)} papers with metadata")

    return merged


def aggregate_texts_by_lineage_quarter(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate paper texts by (lineage_id, quarter).

    For each lineage-quarter, concatenate titles and abstracts.
    """
    print("[2/6] Aggregating texts by lineage-quarter...")

    # Combine title and abstract
    df['text'] = df['title'].fillna('') + '. ' + df['abstract'].fillna('')

    # Group by lineage_id and quarter
    aggregated = []
    for (lineage_id, quarter), group in df.groupby(['lineage_id', 'quarter']):
        # Concatenate all texts in this lineage-quarter (limit to avoid overflow)
        texts = group['text'].tolist()[:50]  # Max 50 papers per quarter
        combined_text = ' '.join(texts)

        # Truncate to max length for SciBERT (512 tokens ~ 2000 chars)
        combined_text = combined_text[:4000]  # Conservative estimate

        aggregated.append({
            'lineage_id': lineage_id,
            'quarter': quarter,
            'text': combined_text,
            'n_papers': len(group)
        })

    agg_df = pd.DataFrame(aggregated)
    print(f"   Aggregated to {len(agg_df)} lineage-quarters")
    print(f"   Lineages: {agg_df['lineage_id'].nunique()}")
    print(f"   Quarters: {agg_df['quarter'].nunique()}")

    return agg_df


def compute_quarterly_embeddings(
    agg_df: pd.DataFrame,
    model_name: str = 'allenai/scibert_scivocab_uncased',
    batch_size: int = 32
) -> Dict:
    """
    Compute SciBERT embeddings for each lineage-quarter.

    Returns dict: {(lineage_id, quarter): embedding}
    """
    print("[3/6] Computing quarterly embeddings...")

    # Load SciBERT
    print(f"   Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    print(f"   Using device: {device}")

    embeddings = {}

    # Process in batches
    n_batches = (len(agg_df) + batch_size - 1) // batch_size

    for i in range(0, len(agg_df), batch_size):
        batch = agg_df.iloc[i:i+batch_size]

        # Tokenize batch
        texts = batch['text'].tolist()
        inputs = tokenizer(
            texts,
            return_tensors='pt',
            truncation=True,
            max_length=512,
            padding=True
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Embed
        with torch.no_grad():
            outputs = model(**inputs)
            # Use CLS token
            batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()

        # Store
        for j, row in enumerate(batch.itertuples()):
            key = (row.lineage_id, row.quarter)
            embeddings[key] = batch_embeddings[j]

        if (i // batch_size + 1) % 10 == 0:
            print(f"   Processed {i // batch_size + 1}/{n_batches} batches")

    print(f"   Computed {len(embeddings)} quarterly embeddings")
    return embeddings


def compute_semantic_velocity(
    embeddings: Dict,
    agg_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Compute semantic velocity: cosine distance between consecutive quarters.

    Returns DataFrame with columns: lineage_id, quarter, velocity
    """
    print("[4/6] Computing semantic velocity...")

    from scipy.spatial.distance import cosine

    results = []

    for lineage_id in agg_df['lineage_id'].unique():
        # Get quarters for this lineage (sorted)
        lineage_quarters = agg_df[agg_df['lineage_id'] == lineage_id]['quarter'].unique()
        lineage_quarters = sorted(lineage_quarters)

        # Compute velocity between consecutive quarters
        for i in range(len(lineage_quarters)):
            quarter = lineage_quarters[i]

            if i == 0:
                # First quarter: no velocity
                velocity = 0.0
            else:
                prev_quarter = lineage_quarters[i-1]

                # Get embeddings
                emb_current = embeddings.get((lineage_id, quarter))
                emb_prev = embeddings.get((lineage_id, prev_quarter))

                if emb_current is not None and emb_prev is not None:
                    # Cosine distance (1 - similarity)
                    velocity = cosine(emb_current, emb_prev)
                else:
                    velocity = 0.0

            results.append({
                'lineage_id': lineage_id,
                'quarter': quarter,
                'semantic_velocity': velocity
            })

    velocity_df = pd.DataFrame(results)
    print(f"   Computed velocity for {len(velocity_df)} lineage-quarters")
    print(f"   Mean velocity: {velocity_df['semantic_velocity'].mean():.4f}")
    print(f"   Max velocity: {velocity_df['semantic_velocity'].max():.4f}")

    return velocity_df


def save_embeddings_and_velocity(
    embeddings: Dict,
    velocity_df: pd.DataFrame,
    output_dir: Path
):
    """Save quarterly embeddings and velocity data."""
    print("[5/6] Saving embeddings and velocity...")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Convert embeddings dict to arrays
    keys = sorted(embeddings.keys())
    lineage_ids = [k[0] for k in keys]
    quarters = [k[1] for k in keys]
    embedding_array = np.array([embeddings[k] for k in keys])

    # Save as npz
    np.savez_compressed(
        output_dir / 'quarterly_embeddings.npz',
        lineage_ids=lineage_ids,
        quarters=quarters,
        embeddings=embedding_array
    )

    # Save velocity as CSV
    velocity_df.to_csv(output_dir / 'semantic_velocity.csv', index=False)

    # Summary statistics
    summary = {
        'n_embeddings': len(embeddings),
        'n_lineages': len(set(k[0] for k in keys)),
        'n_quarters': len(set(k[1] for k in keys)),
        'velocity_mean': float(velocity_df['semantic_velocity'].mean()),
        'velocity_median': float(velocity_df['semantic_velocity'].median()),
        'velocity_std': float(velocity_df['semantic_velocity'].std()),
        'velocity_min': float(velocity_df['semantic_velocity'].min()),
        'velocity_max': float(velocity_df['semantic_velocity'].max())
    }

    with open(output_dir / 'quarterly_embeddings_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"   Saved:")
    print(f"      - quarterly_embeddings.npz ({len(embeddings)} embeddings)")
    print(f"      - semantic_velocity.csv ({len(velocity_df)} rows)")
    print(f"      - quarterly_embeddings_summary.json")

    print(f"\n   Summary:")
    for key, value in summary.items():
        print(f"      {key}: {value}")

    if LEGACY_PHASE1_OUTPUT_DIR:
        LEGACY_PHASE1_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        for filename in ['quarterly_embeddings.npz', 'semantic_velocity.csv', 'quarterly_embeddings_summary.json']:
            src = output_dir / filename
            if src.exists():
                shutil.copy2(src, LEGACY_PHASE1_OUTPUT_DIR / filename)


def resolve_papers_path() -> Path:
    if PREFERRED_PAPERS_PATH.exists():
        return PREFERRED_PAPERS_PATH
    if FALLBACK_PAPERS_PATH.exists():
        print(f"[WARN] {PREFERRED_PAPERS_PATH} not found. Falling back to {FALLBACK_PAPERS_PATH}.")
        return FALLBACK_PAPERS_PATH
    raise FileNotFoundError(
        f"Neither {PREFERRED_PAPERS_PATH} nor {FALLBACK_PAPERS_PATH} exists. "
        "Run community detection or provide ingest.parquet."
    )


def resolve_tight_mapping_path() -> Path:
    if STAGE0_TIGHT_MAPPING.exists():
        return STAGE0_TIGHT_MAPPING
    if LEGACY_PHASE0_TIGHT_MAPPING.exists():
        return LEGACY_PHASE0_TIGHT_MAPPING
    raise FileNotFoundError(
        "Tight mapping file not found in either stage0 or phase0 directories. "
        "Run stage0_semantic_milestone_mapping.py first."
    )


def main():
    print("="*70)
    print("STAGE 1: QUARTERLY SCIBERT EMBEDDINGS")
    print("="*70)
    print()

    # Paths
    registry_path = Path('data/out/02_lineage_tracking/lineage_registry.json')
    partitions_dir = Path('data/out/cache_cum/partitions_cum')
    papers_path = resolve_papers_path()
    output_dir = STAGE1_OUTPUT_DIR

    # Step 1: Load lineage data with paper metadata
    df = load_lineage_data(registry_path, partitions_dir, papers_path)

    # Step 2: Aggregate texts by lineage-quarter
    agg_df = aggregate_texts_by_lineage_quarter(df)

    # Step 3: Compute quarterly embeddings
    embeddings = compute_quarterly_embeddings(agg_df)

    # Step 4: Compute semantic velocity
    velocity_df = compute_semantic_velocity(embeddings, agg_df)

    # Step 5: Save
    save_embeddings_and_velocity(embeddings, velocity_df, output_dir)

    print("\n[6/6] Testing on Stage 0 tight mapping...")

    tight_mapping = pd.read_csv(resolve_tight_mapping_path())
    milestone_lineages = set(tight_mapping['lineage_id'].unique())

    # Check coverage
    embedded_lineages = set(k[0] for k in embeddings.keys())
    coverage = len(milestone_lineages & embedded_lineages)

    print(f"   Milestone lineages: {len(milestone_lineages)}")
    print(f"   Embedded lineages: {len(embedded_lineages)}")
    print(f"   Coverage: {coverage}/{len(milestone_lineages)} ({coverage/len(milestone_lineages)*100:.1f}%)")

    print("\n" + "="*70)
    print("STAGE 1 COMPLETE")
    print("="*70)


if __name__ == '__main__':
    main()
