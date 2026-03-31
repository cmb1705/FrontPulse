"""
Stage 1: Optimized Quarterly SciBERT Embeddings for True Semantic Velocity

Optimizations:
- FP16 mixed precision (2x speedup)
- DataLoader with multiprocessing (parallel preprocessing)
- torch.compile (20-30% speedup)
- Reuses AbstractExtractor from compute_lineage_embeddings.py
- Memory-efficient streaming

Estimated runtime: 20-30 minutes on GPU
"""

import json
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from _path_bootstrap import ensure_repo_imports
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

# Import AbstractExtractor and domain helpers
repo_root = ensure_repo_imports()
from scripts.extract_abstracts import AbstractExtractor  # noqa: E402
from src.domain_registry import (  # noqa: E402
    add_domain_args,
    apply_domain_path_defaults,
    resolve_script_paths,
)

STAGE1_OUTPUT_DIR = Path('data/out/experiments/stage1_quarterly_embeddings')
LEGACY_PHASE1_OUTPUT_DIR = Path('data/out/experiments/phase1_quarterly_embeddings')
STAGE0_TIGHT_MAPPING = Path('data/out/experiments/stage0_tight_mapping/milestone_lineage_mapping_tight.csv')
LEGACY_PHASE0_TIGHT_MAPPING = Path('data/out/experiments/phase0_tight_mapping/milestone_lineage_mapping_tight.csv')


class LineageQuarterDataset(Dataset):
    """
    Dataset for lineage-quarter text aggregates.

    Handles text preprocessing in parallel via DataLoader workers.
    """

    def __init__(self, lineage_quarter_data: list[tuple[int, str, str, int]]):
        """
        Args:
            lineage_quarter_data: List of (lineage_id, quarter, text, n_papers)
        """
        self.data = lineage_quarter_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        lineage_id, quarter, text, n_papers = self.data[idx]
        return {
            'lineage_id': lineage_id,
            'quarter': quarter,
            'text': text,
            'n_papers': n_papers
        }


def load_lineage_quarter_assignments(
    registry_path: Path,
    partitions_dir: Path
) -> pd.DataFrame:
    """
    Load paper-lineage-quarter assignments from registry + partitions.

    Returns DataFrame with: paper_id, lineage_id, quarter
    """
    print("[1/6] Loading lineage-quarter assignments...")

    with open(registry_path) as f:
        registry = json.load(f)

    print(f"   Registry: {len(registry)} quarters")

    assignments = []

    for quarter, community_map in tqdm(registry.items(), desc="   Loading partitions"):
        partition_path = partitions_dir / f"part_{quarter}.json"

        if not partition_path.exists():
            continue

        with open(partition_path) as f:
            partition_data = json.load(f)
            partition = partition_data.get('labels', {})

        for paper_id, community_id in partition.items():
            lineage_id = community_map.get(str(community_id))

            if lineage_id is not None:
                assignments.append({
                    'paper_id': paper_id,
                    'lineage_id': lineage_id,
                    'quarter': quarter
                })

    df = pd.DataFrame(assignments)
    print(f"   Loaded {len(df)} assignments")
    print(f"   Lineages: {df['lineage_id'].nunique()}")
    print(f"   Quarters: {df['quarter'].nunique()}")

    return df


def aggregate_texts_by_lineage_quarter(
    assignments_df: pd.DataFrame,
    extractor: AbstractExtractor,
    max_papers_per_lq: int = 50,
    n_samples: int = None
) -> list[tuple[int, str, str, int]]:
    """
    Aggregate paper texts by (lineage_id, quarter).

    Args:
        assignments_df: Paper-lineage-quarter assignments
        extractor: AbstractExtractor instance
        max_papers_per_lq: Max papers per lineage-quarter
        n_samples: Limit to N lineage-quarters (for testing)

    Returns list of (lineage_id, quarter, combined_text, n_papers)
    """
    print("[2/6] Aggregating texts by lineage-quarter...")

    # OPTIMIZATION: If n_samples is set, first select which lineage-quarters to process
    # Then only load abstracts for papers in those lineage-quarters
    grouped = list(assignments_df.groupby(['lineage_id', 'quarter']))

    if n_samples and n_samples < len(grouped):
        print(f"   Limiting to first {n_samples} lineage-quarters (out of {len(grouped)})")
        grouped = grouped[:n_samples]

    # Extract paper IDs only from selected lineage-quarters
    selected_papers = set()
    for (_lineage_id, _quarter), group in grouped:
        papers = group['paper_id'].tolist()[:max_papers_per_lq]
        selected_papers.update(papers)

    print(f"   Loading abstracts for {len(selected_papers)} unique papers...")

    # Load abstracts in bulk using correct API
    paper_texts = extractor.get_texts_batch(list(selected_papers), verbose=True)

    print(f"   Loaded {len(paper_texts)} papers with text")

    # Aggregate texts
    print("   Aggregating texts...")
    lineage_quarter_data = []

    for (lineage_id, quarter), group in tqdm(grouped, desc="   Aggregating"):
        # Get texts for papers in this lineage-quarter
        papers = group['paper_id'].tolist()[:max_papers_per_lq]
        texts = [paper_texts.get(pid, '') for pid in papers if pid in paper_texts]

        if not texts:
            continue

        # Concatenate and truncate
        combined_text = ' '.join(texts)[:4000]  # ~512 tokens

        lineage_quarter_data.append((
            lineage_id,
            quarter,
            combined_text,
            len(texts)
        ))

    print(f"   Aggregated {len(lineage_quarter_data)} lineage-quarters")

    return lineage_quarter_data


def compute_quarterly_embeddings_optimized(
    lineage_quarter_data: list[tuple[int, str, str, int]],
    model_name: str = 'allenai/scibert_scivocab_uncased',
    batch_size: int = 32,
    num_workers: int = 4,
    use_fp16: bool = True,
    use_compile: bool = True
) -> dict[tuple[int, str], np.ndarray]:
    """
    Compute SciBERT embeddings with optimizations.

    Optimizations:
    - FP16 mixed precision (if use_fp16=True)
    - DataLoader with multiprocessing (num_workers)
    - torch.compile (if use_compile=True, PyTorch 2.0+)

    Args:
        lineage_quarter_data: List of (lineage_id, quarter, text, n_papers)
        model_name: HuggingFace model identifier
        batch_size: Batch size for inference
        num_workers: Number of DataLoader workers
        use_fp16: Enable FP16 mixed precision
        use_compile: Enable torch.compile

    Returns:
        Dict mapping (lineage_id, quarter) -> embedding
    """
    print("[3/6] Computing quarterly embeddings (OPTIMIZED)...")

    # Initialize model
    print(f"   Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    print(f"   Device: {device}")

    # Optimization 1: torch.compile (PyTorch 2.0+)
    if use_compile and hasattr(torch, 'compile'):
        print("   Applying torch.compile...")
        try:
            model = torch.compile(model, mode="reduce-overhead")
            print("   [ENABLED] torch.compile")
        except Exception as e:
            print(f"   [DISABLED] torch.compile failed: {e}")
            use_compile = False
    else:
        print("   [DISABLED] torch.compile (requires PyTorch 2.0+)")
        use_compile = False

    # Optimization 2: Set matmul precision
    if device.type == 'cuda':
        torch.set_float32_matmul_precision("high")
        print("   [ENABLED] High matmul precision")

    # Optimization 3: FP16 check
    if use_fp16:
        if device.type == 'cuda':
            print("   [ENABLED] FP16 mixed precision")
        else:
            print("   [DISABLED] FP16 (requires CUDA)")
            use_fp16 = False

    # Create dataset and dataloader
    dataset = LineageQuarterDataset(lineage_quarter_data)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == 'cuda'),
        persistent_workers=(num_workers > 0)
    )

    print(f"   DataLoader: batch_size={batch_size}, num_workers={num_workers}")
    print(f"   Total batches: {len(dataloader)}")

    # Compute embeddings
    embeddings = {}

    with torch.inference_mode():
        for batch in tqdm(dataloader, desc="   Embedding batches"):
            texts = batch['text']
            lineage_ids = batch['lineage_id'].numpy()
            quarters = batch['quarter']

            # Tokenize
            inputs = tokenizer(
                texts,
                return_tensors='pt',
                max_length=512,
                truncation=True,
                padding=True
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}

            # Forward pass with optional FP16
            if use_fp16:
                with autocast():
                    outputs = model(**inputs)
            else:
                outputs = model(**inputs)

            # Mean pooling
            attention_mask = inputs['attention_mask']
            token_embeddings = outputs.last_hidden_state

            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            batch_embeddings = (sum_embeddings / sum_mask).cpu().numpy()

            # Store embeddings
            for i in range(len(lineage_ids)):
                key = (int(lineage_ids[i]), quarters[i])
                embeddings[key] = batch_embeddings[i]

    print(f"   Computed {len(embeddings)} embeddings")

    return embeddings


def compute_semantic_velocity(
    embeddings: dict[tuple[int, str], np.ndarray]
) -> pd.DataFrame:
    """
    Compute semantic velocity: cosine distance between consecutive quarters.

    Returns DataFrame with columns: lineage_id, quarter, semantic_velocity
    """
    print("[4/6] Computing semantic velocity...")

    from scipy.spatial.distance import cosine

    # Group embeddings by lineage
    lineage_quarters = defaultdict(list)
    for (lineage_id, quarter), emb in embeddings.items():
        lineage_quarters[lineage_id].append((quarter, emb))

    # Sort quarters for each lineage
    for lineage_id in lineage_quarters:
        lineage_quarters[lineage_id].sort(key=lambda x: x[0])

    results = []

    for lineage_id, quarter_embs in tqdm(lineage_quarters.items(), desc="   Computing velocity"):
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
    print(f"   Mean velocity: {df['semantic_velocity'].mean():.4f}")
    print(f"   Max velocity: {df['semantic_velocity'].max():.4f}")

    return df


def resolve_tight_mapping_path() -> Path:
    if STAGE0_TIGHT_MAPPING.exists():
        return STAGE0_TIGHT_MAPPING
    if LEGACY_PHASE0_TIGHT_MAPPING.exists():
        return LEGACY_PHASE0_TIGHT_MAPPING
    raise FileNotFoundError(
        "Tight mapping file not found in either stage0 or phase0 directories. "
        "Run stage0_semantic_milestone_mapping.py first."
    )


def save_embeddings_and_velocity(
    embeddings: dict[tuple[int, str], np.ndarray],
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
        'n_lineages': len({k[0] for k in keys}),
        'n_quarters': len({k[1] for k in keys}),
        'velocity_mean': float(velocity_df['semantic_velocity'].mean()),
        'velocity_median': float(velocity_df['semantic_velocity'].median()),
        'velocity_std': float(velocity_df['semantic_velocity'].std()),
        'velocity_min': float(velocity_df['semantic_velocity'].min()),
        'velocity_max': float(velocity_df['semantic_velocity'].max())
    }

    with open(output_dir / 'quarterly_embeddings_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print("   Saved:")
    print(f"      - quarterly_embeddings.npz ({len(embeddings)} embeddings, {embedding_array.nbytes / 1e9:.2f} GB)")
    print(f"      - semantic_velocity.csv ({len(velocity_df)} rows)")
    print("      - quarterly_embeddings_summary.json")

    print("\n   Summary:")
    for key, value in summary.items():
        print(f"      {key}: {value}")

    LEGACY_PHASE1_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for filename in ['quarterly_embeddings.npz', 'semantic_velocity.csv', 'quarterly_embeddings_summary.json']:
        src = output_dir / filename
        if src.exists():
            shutil.copy2(src, LEGACY_PHASE1_OUTPUT_DIR / filename)


def test_coverage(embeddings: dict, tight_mapping_path: Path):
    """Test coverage of milestone lineages."""
    print("[6/6] Testing coverage on Stage 0 tight mapping...")

    # Load tight mapping
    tight_mapping = pd.read_csv(tight_mapping_path)
    milestone_lineages = set(tight_mapping['lineage_id'].unique())

    # Check coverage
    embedded_lineages = {k[0] for k in embeddings}
    coverage = len(milestone_lineages & embedded_lineages)

    print(f"   Milestone lineages: {len(milestone_lineages)}")
    print(f"   Embedded lineages: {len(embedded_lineages)}")
    print(f"   Coverage: {coverage}/{len(milestone_lineages)} ({coverage/len(milestone_lineages)*100:.1f}%)")

    # Show missing lineages
    missing = milestone_lineages - embedded_lineages
    if missing:
        print(f"   Missing {len(missing)} milestone lineages: {sorted(missing)[:10]}...")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Stage 1: Quarterly SciBERT Embeddings")
    parser.add_argument('--n-samples', type=int, default=None,
                       help='Limit to N lineage-quarters for testing (default: all)')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--no-fp16', action='store_true',
                       help='Disable FP16 mixed precision')
    parser.add_argument('--no-compile', action='store_true',
                       help='Disable torch.compile')
    parser.add_argument('--model', default='allenai/scibert_scivocab_uncased',
                       help='HuggingFace model ID (default: SciBERT). '
                            'Use allenai/specter2_base for SPECTER2.')
    add_domain_args(parser)
    args = parser.parse_args()

    # Resolve domain paths
    paths = resolve_script_paths(args, repo_root)
    apply_domain_path_defaults(args, paths, {
        "registry": (
            "lineage_tracking", "lineage_registry.json",
            "data/out/02_lineage_tracking/lineage_registry.json",
        ),
        "partitions_dir": (
            "cache_cum", "partitions_cum",
            "data/out/cache_cum/partitions_cum",
        ),
        "raw_dir": (
            "raw", "",
            "data/current_ingest/raw",
        ),
        "output_dir": (
            "experiments", "stage1_quarterly_embeddings",
            "data/out/experiments/stage1_quarterly_embeddings",
        ),
    })

    print("="*70)
    if args.n_samples:
        print(f"STAGE 1: QUARTERLY EMBEDDINGS [TEST MODE: {args.n_samples} samples]")
    else:
        print("STAGE 1: OPTIMIZED QUARTERLY SCIBERT EMBEDDINGS [PRODUCTION]")
    print("="*70)
    print()

    # Paths (from domain-aware defaults or argparse)
    registry_path = Path(getattr(args, "registry", "") or "data/out/02_lineage_tracking/lineage_registry.json")
    partitions_dir = Path(getattr(args, "partitions_dir", "") or "data/out/cache_cum/partitions_cum")
    raw_dir = Path(getattr(args, "raw_dir", "") or "data/current_ingest/raw")
    output_dir = Path(getattr(args, "output_dir", "") or str(STAGE1_OUTPUT_DIR))
    tight_mapping_path = resolve_tight_mapping_path()

    # Configuration
    BATCH_SIZE = args.batch_size
    NUM_WORKERS = args.num_workers
    USE_FP16 = not args.no_fp16
    USE_COMPILE = not args.no_compile
    N_SAMPLES = args.n_samples

    print("Configuration:")
    print(f"   Sample limit: {N_SAMPLES if N_SAMPLES else 'None (full dataset)'}")
    print(f"   Batch size: {BATCH_SIZE}")
    print(f"   DataLoader workers: {NUM_WORKERS}")
    print(f"   FP16: {USE_FP16}")
    print(f"   torch.compile: {USE_COMPILE}")
    print()

    # Step 1: Load assignments
    assignments_df = load_lineage_quarter_assignments(registry_path, partitions_dir)

    # Step 2: Initialize abstract extractor
    print("[2/6] Initializing AbstractExtractor...")
    extractor = AbstractExtractor(raw_dir)
    print(f"   Loaded extractor with {len(extractor._work_to_store)} works indexed")

    # Step 3: Aggregate texts
    lineage_quarter_data = aggregate_texts_by_lineage_quarter(
        assignments_df,
        extractor,
        n_samples=N_SAMPLES
    )

    # Step 4: Compute embeddings
    embeddings = compute_quarterly_embeddings_optimized(
        lineage_quarter_data,
        model_name=args.model,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        use_fp16=USE_FP16,
        use_compile=USE_COMPILE,
    )

    # Step 5: Compute velocity
    velocity_df = compute_semantic_velocity(embeddings)

    # Step 6: Save
    save_embeddings_and_velocity(embeddings, velocity_df, output_dir)

    # Step 7: Test coverage
    test_coverage(embeddings, tight_mapping_path)

    print("\n" + "="*70)
    print("STAGE 1 COMPLETE")
    print("="*70)
    print("\nNext step: Run Stage 1b to test multi-signal detection with real semantic velocity")


if __name__ == '__main__':
    main()
