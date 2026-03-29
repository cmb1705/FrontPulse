"""
Stage 0: Semantic Milestone-to-Lineage Mapping

Uses SciBERT embeddings to map each milestone to the top-K most semantically similar
lineages, ensuring each lineage is assigned to at most one milestone.

This fixes the evaluation framework by eliminating the many-to-many mapping artifact.
"""

import argparse
import json
import shutil
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from scipy.spatial.distance import cosine
from transformers import AutoModel, AutoTokenizer

STAGE0_DIR = Path('data/out/experiments/stage0_tight_mapping')
LEGACY_PHASE0_DIR = Path('data/out/experiments/phase0_tight_mapping')


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Stage 0 semantic milestone-to-lineage mapping.")
    parser.add_argument("--milestones", type=Path, required=True, help="Path to the milestone catalog CSV.")
    parser.add_argument(
        "--embeddings",
        type=Path,
        default=Path('data/out/02_lineage_tracking/lineage_embeddings.npz'),
        help="Path to lineage embeddings NPZ (default: %(default)s).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=STAGE0_DIR / 'milestone_lineage_mapping_tight.csv',
        help="Output CSV path for the tight mapping (default: %(default)s).",
    )
    return parser.parse_args()

def load_lineage_embeddings(embedding_path: Path) -> dict[int, np.ndarray]:
    """Load Stage 2 lineage embeddings."""
    print("[1/6] Loading lineage embeddings...")
    data = np.load(embedding_path)
    lineage_ids = data['lineage_ids']
    embeddings = data['embeddings']

    embedding_dict = {int(lid): emb for lid, emb in zip(lineage_ids, embeddings)}
    print(f"   Loaded {len(embedding_dict)} lineage embeddings")
    return embedding_dict


def embed_milestone_descriptions(
    milestones_df: pd.DataFrame,
    model_name: str = 'allenai/scibert_scivocab_uncased'
) -> dict[str, np.ndarray]:
    """
    Embed milestone descriptions using SciBERT.

    Returns dict: {event_id: embedding}
    """
    print("[2/6] Embedding milestone descriptions...")

    # Load SciBERT
    print("   Loading SciBERT model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()

    # Check for GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    print(f"   Using device: {device}")

    milestone_embeddings = {}

    for idx, row in milestones_df.iterrows():
        event_id = row['event_id']
        description = row['description']

        # Tokenize and embed
        inputs = tokenizer(description, return_tensors='pt', truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            # Use CLS token embedding
            embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy().flatten()

        milestone_embeddings[event_id] = embedding

        if (idx + 1) % 10 == 0:
            print(f"   Embedded {idx + 1}/{len(milestones_df)} milestones")

    print(f"   Embedded {len(milestone_embeddings)} milestone descriptions")
    return milestone_embeddings


def compute_similarity_matrix(
    milestone_embeddings: dict[str, np.ndarray],
    lineage_embeddings: dict[int, np.ndarray]
) -> pd.DataFrame:
    """
    Compute cosine similarity between all milestones and lineages.

    Returns DataFrame with milestones as rows, lineages as columns.
    """
    print("[3/6] Computing similarity matrix...")

    milestone_ids = list(milestone_embeddings.keys())
    lineage_ids = list(lineage_embeddings.keys())

    # Initialize similarity matrix
    similarity_matrix = np.zeros((len(milestone_ids), len(lineage_ids)))

    for i, m_id in enumerate(milestone_ids):
        m_emb = milestone_embeddings[m_id]
        for j, l_id in enumerate(lineage_ids):
            l_emb = lineage_embeddings[l_id]
            # Cosine similarity = 1 - cosine distance
            similarity_matrix[i, j] = 1 - cosine(m_emb, l_emb)

    df = pd.DataFrame(similarity_matrix, index=milestone_ids, columns=lineage_ids)
    print(f"   Similarity matrix: {df.shape[0]} milestones × {df.shape[1]} lineages")
    print(f"   Mean similarity: {df.values.mean():.3f}")
    print(f"   Max similarity: {df.values.max():.3f}")
    return df


def assign_unique_lineages(
    similarity_df: pd.DataFrame,
    k: int = 3
) -> pd.DataFrame:
    """
    Assign each milestone to top-K most similar lineages,
    ensuring each lineage is assigned to at most one milestone.

    Strategy: Greedy assignment in order of decreasing similarity.

    Returns DataFrame with columns: event_id, lineage_id, similarity, rank
    """
    print(f"[4/6] Assigning unique lineages (K={k})...")

    # Flatten similarity matrix to (milestone, lineage, similarity) tuples
    candidates = []
    for milestone in similarity_df.index:
        for lineage in similarity_df.columns:
            similarity = similarity_df.loc[milestone, lineage]
            candidates.append({
                'event_id': milestone,
                'lineage_id': int(lineage),
                'similarity': similarity
            })

    candidates_df = pd.DataFrame(candidates)

    # Sort by similarity (descending)
    candidates_df = candidates_df.sort_values('similarity', ascending=False)

    # Greedy assignment
    assigned_lineages = set()
    milestone_assignments = {m: [] for m in similarity_df.index}

    for _, row in candidates_df.iterrows():
        event_id = row['event_id']
        lineage_id = row['lineage_id']
        similarity = row['similarity']

        # Skip if lineage already assigned
        if lineage_id in assigned_lineages:
            continue

        # Skip if milestone already has K lineages
        if len(milestone_assignments[event_id]) >= k:
            continue

        # Assign
        milestone_assignments[event_id].append({
            'lineage_id': lineage_id,
            'similarity': similarity
        })
        assigned_lineages.add(lineage_id)

    # Convert to DataFrame
    assignments = []
    for event_id, lineage_list in milestone_assignments.items():
        for rank, assignment in enumerate(lineage_list, 1):
            assignments.append({
                'event_id': event_id,
                'lineage_id': assignment['lineage_id'],
                'similarity': assignment['similarity'],
                'rank': rank
            })

    assignments_df = pd.DataFrame(assignments)

    print(f"   Assigned {len(assignments_df)} milestone-lineage pairs")
    print(f"   Milestones with >=1 lineage: {assignments_df['event_id'].nunique()}/{len(similarity_df)}")
    print(f"   Avg lineages per milestone: {len(assignments_df) / assignments_df['event_id'].nunique():.2f}")
    print(f"   Unique lineages used: {assignments_df['lineage_id'].nunique()}/{len(similarity_df.columns)}")

    return assignments_df


def merge_with_milestone_metadata(
    assignments_df: pd.DataFrame,
    milestones_df: pd.DataFrame
) -> pd.DataFrame:
    """Merge assignments with milestone metadata."""
    print("[5/6] Merging with milestone metadata...")

    result = assignments_df.merge(
        milestones_df[['event_id', 'event_quarter', 'detection_window_start',
                       'detection_window_end', 'mapped_fronts', 'category']],
        on='event_id',
        how='left'
    )

    print(f"   Final dataset: {len(result)} milestone-lineage pairs")
    return result


def save_tight_mapping(
    mapping_df: pd.DataFrame,
    output_path: Path,
    legacy_dirs: Optional[list[Path]] = None,
):
    """Save tight milestone-lineage mapping."""
    print(f"[6/6] Saving tight mapping to {output_path}...")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    mapping_df.to_csv(output_path, index=False)

    # Also save summary statistics
    summary = {
        'total_pairs': len(mapping_df),
        'unique_milestones': int(mapping_df['event_id'].nunique()),
        'unique_lineages': int(mapping_df['lineage_id'].nunique()),
        'avg_lineages_per_milestone': float(len(mapping_df) / mapping_df['event_id'].nunique()),
        'avg_similarity': float(mapping_df['similarity'].mean()),
        'min_similarity': float(mapping_df['similarity'].min()),
        'max_similarity': float(mapping_df['similarity'].max())
    }

    summary_path = output_path.parent / 'tight_mapping_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print("   Summary:")
    for key, value in summary.items():
        print(f"      {key}: {value}")

    print("\n   Saved:")
    print(f"      - {output_path}")
    print(f"      - {summary_path}")

    if legacy_dirs:
        for legacy_dir in legacy_dirs:
            legacy_dir.mkdir(parents=True, exist_ok=True)
            legacy_file = legacy_dir / output_path.name
            legacy_summary = legacy_dir / summary_path.name
            shutil.copy2(output_path, legacy_file)
            shutil.copy2(summary_path, legacy_summary)
            print(f"      - {legacy_file} (legacy)")
            print(f"      - {legacy_summary} (legacy)")


def main():
    args = parse_args()

    print("="*70)
    print("STAGE 0: SEMANTIC MILESTONE-TO-LINEAGE MAPPING")
    print("="*70)
    print()

    # Paths
    milestones_path = args.milestones
    embeddings_path = args.embeddings
    output_path = args.output

    # Load data
    milestones_df = pd.read_csv(milestones_path)
    milestones_df = milestones_df[milestones_df['detectable']].copy()
    print(f"Loaded {len(milestones_df)} detectable milestones\n")

    # Load lineage embeddings
    lineage_embeddings = load_lineage_embeddings(embeddings_path)

    # Embed milestone descriptions
    milestone_embeddings = embed_milestone_descriptions(milestones_df)

    # Compute similarity matrix
    similarity_df = compute_similarity_matrix(milestone_embeddings, lineage_embeddings)

    # Assign unique lineages (K=3)
    assignments_df = assign_unique_lineages(similarity_df, k=3)

    # Merge with metadata
    final_df = merge_with_milestone_metadata(assignments_df, milestones_df)

    # Save
    save_tight_mapping(final_df, output_path, legacy_dirs=[LEGACY_PHASE0_DIR])

    print("\n" + "="*70)
    print("STAGE 0 COMPLETE")
    print("="*70)


if __name__ == '__main__':
    main()
