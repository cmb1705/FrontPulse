"""
Compute SciBERT embeddings for community lineages (Stage 2).

This module creates lineage-level semantic embeddings by:
1. Extracting titles+abstracts for all papers in each lineage
2. Filtering stopwords and common terms
3. Computing SciBERT embeddings with recency weighting
4. Aggregating to lineage-level representations

Usage:
    python scripts/compute_lineage_embeddings.py \
        --lineage-metrics data/out/02_lineage_tracking/lineage_metrics.csv \
        --lineage-registry data/out/02_lineage_tracking/lineage_registry.json \
        --graphs-dir data/current_graphs \
        --raw-dir data/current_ingest/raw \
        --output data/out/02_lineage_tracking/lineage_embeddings.npz \
        --min-quarters 6  # Only persistent lineages

Output:
    - lineage_embeddings.npz: Compressed array of embeddings [n_lineages x 768]
    - lineage_embedding_metadata.json: Lineage IDs, paper counts, coverage stats
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.preprocessing import normalize
from tqdm import tqdm

# NLP and transformer imports
from transformers import AutoModel, AutoTokenizer

# Custom imports
repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from scripts.extract_abstracts import AbstractExtractor  # noqa: E402
from src.domain_registry import add_domain_args, resolve_script_paths  # noqa: E402

# English stopwords - common words to filter out
STOPWORDS = {
    'a', 'about', 'above', 'after', 'again', 'against', 'all', 'am', 'an', 'and',
    'any', 'are', 'as', 'at', 'be', 'because', 'been', 'before', 'being', 'below',
    'between', 'both', 'but', 'by', 'can', 'cannot', 'could', 'did', 'do', 'does',
    'doing', 'down', 'during', 'each', 'few', 'for', 'from', 'further', 'had',
    'has', 'have', 'having', 'he', 'her', 'here', 'hers', 'herself', 'him',
    'himself', 'his', 'how', 'i', 'if', 'in', 'into', 'is', 'it', 'its', 'itself',
    'just', 'me', 'might', 'more', 'most', 'must', 'my', 'myself', 'no', 'nor',
    'not', 'now', 'of', 'off', 'on', 'once', 'only', 'or', 'other', 'our', 'ours',
    'ourselves', 'out', 'over', 'own', 'same', 'she', 'should', 'so', 'some',
    'such', 'than', 'that', 'the', 'their', 'theirs', 'them', 'themselves', 'then',
    'there', 'these', 'they', 'this', 'those', 'through', 'to', 'too', 'under',
    'until', 'up', 'very', 'was', 'we', 'were', 'what', 'when', 'where', 'which',
    'while', 'who', 'whom', 'why', 'will', 'with', 'would', 'you', 'your', 'yours',
    'yourself', 'yourselves'
}


class LineageEmbedder:
    """
    Compute SciBERT embeddings for community lineages.

    Uses allenai/scibert_scivocab_uncased for scientific text understanding.
    Aggregates paper-level embeddings to lineage-level with recency weighting.
    """

    def __init__(
        self,
        raw_dir: Path = None,
        model_name: str = "allenai/scibert_scivocab_uncased",
        device: str = None,
        extractor=None
    ):
        """
        Initialize embedder with SciBERT model and abstract extractor.

        Args:
            raw_dir: Directory containing raw JSONL files (not needed if extractor provided)
            model_name: HuggingFace model identifier
            device: 'cuda' or 'cpu' (auto-detects if None)
            extractor: Optional AbstractExtractor instance (for pipeline mode)
        """
        print(f"[EMBEDDER] Initializing SciBERT model: {model_name}")

        # Auto-detect device
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        print(f"[EMBEDDER] Using device: {self.device}")

        # Load SciBERT tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()  # Inference mode

        # Use provided extractor or create new one
        if extractor is not None:
            print("[EMBEDDER] Using shared abstract extractor")
            self.extractor = extractor
        else:
            print(f"[EMBEDDER] Loading abstract extractor from {raw_dir}")
            self.extractor = AbstractExtractor(raw_dir)

        print("[EMBEDDER] Ready")

    def filter_stopwords(self, text: str) -> str:
        """
        Remove common stopwords from text while preserving technical terms.

        Args:
            text: Input text

        Returns:
            Filtered text with stopwords removed
        """
        if not text:
            return ""

        # Simple word-level filtering
        words = text.lower().split()
        filtered = [w for w in words if w not in STOPWORDS and len(w) > 2]
        return ' '.join(filtered)

    def embed_texts_batch(
        self,
        texts: list[str],
        batch_size: int = 32,
        max_length: int = 512
    ) -> np.ndarray:
        """
        Generate SciBERT embeddings for multiple texts (batched for speed).

        Args:
            texts: List of input texts (title + abstract)
            batch_size: Number of texts to process at once
            max_length: Maximum token length (SciBERT max is 512)

        Returns:
            Array of embeddings [n_texts x 768]
        """
        if not texts:
            return np.array([])

        # Filter stopwords for all texts
        filtered_texts = [self.filter_stopwords(t) for t in texts]

        # Remove empty texts but track indices
        valid_indices = [i for i, t in enumerate(filtered_texts) if t]
        valid_texts = [filtered_texts[i] for i in valid_indices]

        if not valid_texts:
            return np.zeros((len(texts), 768))

        # Process in batches
        all_embeddings = []

        # Set float32 matmul precision for speed (torch 2.6+)
        torch.set_float32_matmul_precision("high")

        with torch.inference_mode():  # Faster than no_grad() in torch 2.6+
            for i in range(0, len(valid_texts), batch_size):
                batch = valid_texts[i:i + batch_size]

                # Tokenize batch
                inputs = self.tokenizer(
                    batch,
                    return_tensors='pt',
                    max_length=max_length,
                    truncation=True,
                    padding=True
                ).to(self.device)

                # Generate embeddings with mean pooling
                outputs = self.model(**inputs)
                attention_mask = inputs['attention_mask']
                token_embeddings = outputs.last_hidden_state

                # Mean pooling
                input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
                sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
                sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                batch_embeddings = (sum_embeddings / sum_mask).cpu().numpy()

                all_embeddings.append(batch_embeddings)

        # Concatenate all batches
        embeddings_array = np.vstack(all_embeddings)

        # Reconstruct full array with zeros for filtered texts
        result = np.zeros((len(texts), 768))
        result[valid_indices] = embeddings_array

        return result

    def embed_text(self, text: str, max_length: int = 512) -> np.ndarray:
        """
        Generate SciBERT embedding for a single text (use embed_texts_batch for multiple).

        Args:
            text: Input text (title + abstract)
            max_length: Maximum token length (SciBERT max is 512)

        Returns:
            768-dimensional embedding vector
        """
        return self.embed_texts_batch([text], batch_size=1, max_length=max_length)[0]

    def compute_lineage_embedding(
        self,
        work_ids: list[str],
        quarters: list[str] = None,
        recency_weight: bool = True,
        batch_size: int = 32,
        profile: bool = False
    ) -> tuple[np.ndarray, dict]:
        """
        Compute aggregated embedding for a lineage from its papers (batched for speed).

        Args:
            work_ids: List of OpenAlex work IDs in this lineage
            quarters: Corresponding quarters for recency weighting (optional)
            recency_weight: Whether to weight recent papers more heavily
            batch_size: Batch size for SciBERT inference (default: 32)
            profile: Enable timing profiling (default: False)

        Returns:
            (embedding_vector, metadata_dict)
        """
        # Extract texts in batch
        t0 = time.time() if profile else None
        texts = self.extractor.get_texts_batch(work_ids, include_title=True)
        t1 = time.time() if profile else None
        if profile:
            print(f"      Text extraction: {t1-t0:.2f}s")

        if not texts:
            return np.zeros(768), {
                'n_papers': len(work_ids),
                'n_with_text': 0,
                'coverage': 0.0
            }

        # Build ordered list of texts and weights
        text_list = []
        weights = []

        for i, work_id in enumerate(work_ids):
            text = texts.get(work_id)
            if not text:
                continue

            text_list.append(text)

            # Recency weighting: more recent papers weighted higher
            if recency_weight and quarters:
                # Simple linear weighting: older = lower weight
                # Normalize to [0.5, 1.0] range so old papers still contribute
                position = i / len(work_ids)
                weight = 0.5 + 0.5 * position
                weights.append(weight)
            else:
                weights.append(1.0)

        if not text_list:
            return np.zeros(768), {
                'n_papers': len(work_ids),
                'n_with_text': 0,
                'coverage': 0.0
            }

        # Compute embeddings in batches (much faster!)
        t2 = time.time() if profile else None
        embeddings = self.embed_texts_batch(text_list, batch_size=batch_size)
        t3 = time.time() if profile else None
        if profile:
            print(f"      SciBERT embedding: {t3-t2:.2f}s")

        # Aggregate with weights
        weights = np.array(weights).reshape(-1, 1)

        # Weighted mean
        lineage_embedding = (embeddings * weights).sum(axis=0) / weights.sum()

        # L2 normalize
        lineage_embedding = normalize(lineage_embedding.reshape(1, -1))[0]

        metadata = {
            'n_papers': len(work_ids),
            'n_with_text': len(texts),
            'coverage': len(texts) / len(work_ids)
        }

        return lineage_embedding, metadata


# Global cache for partition data
_PARTITION_CACHE = {}

def load_lineage_papers_fast(
    lineage_id: int,
    lineage_registry: dict[str, dict[int, int]],
    partitions_dir: Path = Path("data/out/cache_cum/partitions_cum")
) -> list[str]:
    """
    Extract all paper IDs belonging to a specific lineage using cached JSON partitions.

    This is ~100x faster than loading PKL files because:
    - JSON is much faster to parse than pickle
    - Partition JSONs are small ({work_id: community_id} only)
    - We cache the inverted index ({community_id: [work_ids]}) per quarter

    Args:
        lineage_id: Lineage ID to extract
        lineage_registry: Mapping of quarter -> {community_id: lineage_id}
        partitions_dir: Directory containing partition JSON files

    Returns:
        List of OpenAlex work IDs in this lineage
    """
    all_papers = []

    # Find all quarters where this lineage appears
    for quarter, community_map in lineage_registry.items():
        # Find community_id(s) that map to this lineage_id
        # NOTE: community_ids in registry are strings, need to convert to int
        community_ids = [
            int(comm_id) for comm_id, lin_id in community_map.items()
            if lin_id == lineage_id
        ]

        if not community_ids:
            continue

        # Load partition from cache or disk
        if quarter not in _PARTITION_CACHE:
            partition_path = partitions_dir / f"part_{quarter}.json"

            if not partition_path.exists():
                print(f"  Warning: Partition {quarter} not found")
                continue

            try:
                with open(partition_path) as f:
                    data = json.load(f)

                labels = data.get('labels', {})

                # Invert to {community_id: [work_ids]} for O(1) lookup
                inverted = {}
                for work_id, comm_id in labels.items():
                    if comm_id not in inverted:
                        inverted[comm_id] = []
                    inverted[comm_id].append(work_id)

                _PARTITION_CACHE[quarter] = inverted

            except Exception as e:
                print(f"  Warning: Could not load partition {quarter}: {e}")
                continue

        # Get papers for this community from cache
        partition_inv = _PARTITION_CACHE[quarter]
        for comm_id in community_ids:
            if comm_id in partition_inv:
                all_papers.extend(partition_inv[comm_id])

    # Remove duplicates while preserving order
    seen = set()
    unique_papers = []
    for paper in all_papers:
        if paper not in seen:
            seen.add(paper)
            unique_papers.append(paper)

    return unique_papers


# Keep old function for backward compatibility
def load_lineage_papers(
    lineage_id: int,
    lineage_registry: dict[str, dict[int, int]],
    _graphs_dir: Path
) -> list[str]:
    """
    DEPRECATED: Use load_lineage_papers_fast() instead.

    This function loads heavy PKL files (~9 minutes) instead of lightweight
    JSON partitions (~seconds). Kept for backward compatibility only.
    """
    # Delegate to fast version
    return load_lineage_papers_fast(lineage_id, lineage_registry)


def run_embeddings(
    min_quarters: int = 6,
    device: str = None,
    profile: bool = False,
    output_path: Path = Path("data/out/02_lineage_tracking/lineage_embeddings.npz"),
    lineage_metrics_path: Path = Path("data/out/02_lineage_tracking/lineage_metrics.csv"),
    front_config_path: Path = Path('config/front_aliases.yaml'),
    partitions_dir: Path = Path('data/out/cache_cum/partitions_cum'),
    output_root: Path = Path('data/out'),
    registry_path: Path = None,  # For standalone mode
    raw_dir: Path = None,  # For standalone mode
    _graphs_dir: Path = None,  # For standalone mode (optional)
    store=None,
    validate: bool = True
) -> tuple:
    """
    Compute SciBERT embeddings for persistent lineages.

    Can be called standalone or from pipeline with shared store.

    Args:
        min_quarters: Minimum quarters for persistent lineages
        device: 'cuda', 'cpu', or None for auto-detect
        profile: Enable detailed timing profiling
        output_path: Output path for embeddings NPZ file
        lineage_metrics_path: Path to lineage metrics CSV
        front_config_path: Path to front aliases YAML
        partitions_dir: Path to partition JSON files
        output_root: Base directory for Stage 2 outputs (similarity, validation)
        registry_path: Path to lineage registry JSON (standalone mode)
        raw_dir: Path to raw JSONL files (standalone mode)
        graphs_dir: Path to citation graph PKL files (standalone mode, optional)
        store: Optional LineageTextStore for pipeline mode
        validate: Run validation checks and generate reports (default: True)

    Returns:
        (embeddings_array, lineage_ids, metadata_list, validation_results)
    """

    # Load lineage metrics to identify persistent lineages
    print(f"\n{'='*70}")
    print("STAGE 2: SciBERT EMBEDDING EXPANSION")
    print(f"{'='*70}\n")

    # Get resources from shared store or load fresh
    if store is not None:
        print("[LOAD] Using shared store (pipeline mode)")
        persistent_lineages = store.get_persistent_lineages(min_quarters)
        lineage_registry = store.registry
        embedder = LineageEmbedder(device=device, extractor=store.extractor)
    else:
        print(f"[LOAD] Loading lineage metrics from {lineage_metrics_path}")
        df_metrics = pd.read_csv(lineage_metrics_path)

        # Filter to persistent lineages
        lineage_lifespans = df_metrics.groupby('lineage_id').size()
        persistent_lineages = lineage_lifespans[
            lineage_lifespans >= min_quarters
        ].index.tolist()

        print(f"[LOAD] Found {len(persistent_lineages)} persistent lineages "
              f"(>= {min_quarters} quarters)")

        # Load lineage registry
        lineage_registry_path = registry_path or Path("data/out/02_lineage_tracking/lineage_registry.json")
        print(f"[LOAD] Loading lineage registry from {lineage_registry_path}")
        with open(lineage_registry_path) as f:
            lineage_registry = json.load(f)

        # Initialize embedder
        raw_dir_path = raw_dir or Path("data/current_ingest/raw")
        embedder = LineageEmbedder(raw_dir_path, device=device)

    print(f"[LOAD] Found {len(persistent_lineages)} persistent lineages "
          f"(>= {min_quarters} quarters)")

    # Compute embeddings for each lineage
    print(f"\n[EMBED] Computing embeddings for {len(persistent_lineages)} lineages...")

    embeddings = []
    metadata_list = []
    lineage_ids = []

    # Profiling accumulators
    if profile:
        timings = {
            'load_papers': [],
            'extract_texts': [],
            'embed_compute': []
        }

    for lineage_id in tqdm(persistent_lineages, desc="Lineages"):
        # Extract all papers in this lineage
        t0 = time.time()
        papers = load_lineage_papers_fast(lineage_id, lineage_registry, partitions_dir)
        t1 = time.time()
        if profile:
            timings['load_papers'].append(t1 - t0)

        if not papers:
            print(f"  Warning: No papers found for lineage {lineage_id}")
            continue

        # Compute embedding (includes text extraction + GPU computation)
        t2 = time.time()
        embedding, metadata = embedder.compute_lineage_embedding(
            papers,
            recency_weight=True,
            profile=profile
        )
        t3 = time.time()
        if profile:
            timings['embed_compute'].append(t3 - t2)

        embeddings.append(embedding)
        metadata_list.append(metadata)
        lineage_ids.append(lineage_id)

        if profile:
            print(f"\n  Lineage {lineage_id}: {len(papers)} papers")
            print(f"    Load papers:  {t1-t0:.2f}s")
            print(f"    Embed compute: {t3-t2:.2f}s")
            print(f"    Total:        {t3-t0:.2f}s")

    # Convert to numpy array
    embeddings_array = np.array(embeddings)

    if len(embeddings_array) == 0:
        print("\n[WARNING] No embeddings generated.")
        return None, None, None

    # Save embeddings
    print(f"\n[SAVE] Saving embeddings to {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        output_path,
        embeddings=embeddings_array,
        lineage_ids=np.array(lineage_ids)
    )

    # Save metadata
    metadata_path = output_path.with_suffix('.json')
    metadata_output = {
        'lineages': [
            {
                'lineage_id': int(lid),
                'n_papers': meta['n_papers'],
                'n_with_text': meta['n_with_text'],
                'coverage': meta['coverage']
            }
            for lid, meta in zip(lineage_ids, metadata_list)
        ],
        'summary': {
            'n_lineages': len(lineage_ids),
            'embedding_dim': embeddings_array.shape[1],
            'avg_papers_per_lineage': np.mean([m['n_papers'] for m in metadata_list]),
            'avg_coverage': np.mean([m['coverage'] for m in metadata_list])
        }
    }

    with open(metadata_path, 'w') as f:
        json.dump(metadata_output, f, indent=2)

    print(f"[SAVE] Saved metadata to {metadata_path}")

    # Print summary
    print(f"\n{'='*70}")
    print("EMBEDDING SUMMARY")
    print(f"{'='*70}")
    print(f"Total lineages: {len(lineage_ids)}")
    print(f"Embedding dimensions: {embeddings_array.shape[1]}")
    print(f"Average papers per lineage: {metadata_output['summary']['avg_papers_per_lineage']:.1f}")
    print(f"Average text coverage: {metadata_output['summary']['avg_coverage']*100:.1f}%")
    print("\nOutputs:")
    print(f"  - Embeddings: {output_path}")
    print(f"  - Metadata: {metadata_path}")

    # Compute front similarity matrix
    print(f"\n{'='*70}")
    print("COMPUTING FRONT SIMILARITY")
    print(f"{'='*70}\n")

    # Load front definitions
    print(f"[FRONT] Loading front definitions from {front_config_path}")
    with open(front_config_path) as f:
        fronts_config = yaml.safe_load(f)

    front_names = sorted(fronts_config.keys())
    print(f"[FRONT] Loaded {len(front_names)} research fronts")

    # Compute front centroids from anchor DOI abstracts
    print("[FRONT] Computing front centroids from anchor DOI abstracts...")
    front_centroids = {}

    for front_name in tqdm(front_names, desc="Front centroids"):
        anchor_dois = fronts_config[front_name].get('anchor_dois', [])
        if not anchor_dois:
            print(f"  Warning: No anchor DOIs for {front_name}")
            front_centroids[front_name] = np.zeros(768)
            continue

        # Fetch abstracts for each anchor DOI
        anchor_abstracts = []
        missing_dois = []
        for doi in anchor_dois:
            abstract = embedder.extractor.get_text_by_doi(doi, include_title=True)
            if abstract:
                anchor_abstracts.append(abstract)
            else:
                missing_dois.append(doi)

        if not anchor_abstracts:
            print(f"  Warning: No abstracts found for {front_name} (all {len(anchor_dois)} DOIs missing)")
            front_centroids[front_name] = np.zeros(768)
            continue

        if missing_dois:
            print(f"  Note: {front_name} missing {len(missing_dois)}/{len(anchor_dois)} DOIs: {missing_dois[:3]}...")

        # Embed each anchor abstract
        abstract_embeddings = embedder.embed_texts_batch(anchor_abstracts, batch_size=32)

        # Average to get front centroid
        front_centroid = abstract_embeddings.mean(axis=0)

        # L2 normalize
        front_centroid = normalize(front_centroid.reshape(1, -1))[0]

        front_centroids[front_name] = front_centroid

    # FIXED: Use contrastive similarity to amplify differences in narrow domains
    # Standard cosine similarity saturates at 0.87-0.95 for perovskite documents.
    # Centering embeddings by subtracting the mean amplifies discriminative features.
    print("[SIMILARITY] Computing contrastive similarity matrix...")

    # Compute mean embedding across all lineages and fronts
    all_embeddings = np.vstack([embeddings_array] + [front_centroids[f].reshape(1, -1) for f in front_names])
    mean_embedding = all_embeddings.mean(axis=0)

    print(f"  Mean embedding computed from {len(all_embeddings)} vectors")

    # Center embeddings by subtracting mean
    centered_lineage_embs = embeddings_array - mean_embedding
    centered_front_embs = {f: front_centroids[f] - mean_embedding for f in front_names}

    # Re-normalize centered embeddings
    centered_lineage_embs = normalize(centered_lineage_embs)
    for f in front_names:
        centered_front_embs[f] = normalize(centered_front_embs[f].reshape(1, -1))[0]

    print("  Embeddings centered and re-normalized")

    # Compute cosine similarity on centered embeddings
    similarity_matrix = []

    for i, lineage_id in enumerate(lineage_ids):
        lineage_emb = centered_lineage_embs[i]
        similarities = {}

        for front_name in front_names:
            front_emb = centered_front_embs[front_name]
            # Cosine similarity on centered, normalized vectors
            sim = np.dot(lineage_emb, front_emb)
            similarities[front_name] = float(sim)

        similarity_matrix.append({'lineage_id': lineage_id, **similarities})

    # Save similarity matrix
    similarity_df = pd.DataFrame(similarity_matrix)
    mapping_dir = Path(output_root) / "03_milestone_mapping"
    mapping_dir.mkdir(parents=True, exist_ok=True)
    similarity_path = mapping_dir / "lineage_front_similarity.csv"
    similarity_path.parent.mkdir(parents=True, exist_ok=True)
    similarity_df.to_csv(similarity_path, index=False)
    print(f"[SIMILARITY] Saved {len(lineage_ids)} x {len(front_names)} similarity matrix to {similarity_path}")

    print("\nOutputs:")
    print(f"  - Embeddings: {output_path}")
    print(f"  - Metadata: {metadata_path}")
    print(f"  - Similarity Matrix: {similarity_path}")

    # Close extractor only if we created it (not using shared store)
    if store is None:
        embedder.extractor.close()

    # Run validation if requested
    if validate:
        print(f"\n{'='*70}")
        print("STAGE 2 VALIDATION")
        print(f"{'='*70}\n")

        validation_results = run_phase2_validation(
            embeddings_array=embeddings_array,
            lineage_ids=lineage_ids,
            metadata_list=metadata_list,
            similarity_path=similarity_path,
            output_root=output_root
        )
    else:
        validation_results = None

    return embeddings_array, lineage_ids, metadata_list, validation_results


# ============================================================================
# VALIDATION FUNCTIONS (integrated from validate_stage2.py)
# ============================================================================

def run_phase2_validation(
    embeddings_array: np.ndarray,
    lineage_ids: np.ndarray,
    metadata_list: list[dict],
    similarity_path: Path,
    output_root: Path = Path('data/out')
) -> dict:
    """
    Run Stage 2 validation checks and generate outputs.

    Args:
        embeddings_array: Embedding vectors [n_lineages x 768]
        lineage_ids: Lineage IDs
        metadata_list: Metadata for each lineage
        similarity_path: Path to similarity CSV

    Returns:
        Dictionary with validation results
    """
    # Lazy imports to avoid overhead when validation disabled

    # Create output directory
    output_dir = Path(output_root) / '06_validation' / 'stage2'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load similarity matrix
    print("[1/6] Loading similarity matrix...")
    similarity_df = pd.read_csv(similarity_path)

    # Create metadata DataFrame
    metadata_df = pd.DataFrame(metadata_list)
    metadata_df['lineage_id'] = lineage_ids

    # Run validation checks
    print("[2/6] Running data integrity checks...")
    checks = _validate_stage2_integrity(embeddings_array, metadata_df, similarity_df)

    # Generate visualizations
    print("[3/6] Generating similarity heatmap...")
    _generate_phase2_heatmap(similarity_df, output_dir / 'phase_2_similarity_heatmap.png')

    print("[4/6] Generating t-SNE visualization...")
    _generate_phase2_tsne(embeddings_array, similarity_df, output_dir / 'phase_2_tsne.png')

    print("[5/6] Generating score distributions...")
    _generate_phase2_distributions(embeddings_array, similarity_df, output_dir / 'phase_2_distributions.png')

    # Generate report
    print("[6/6] Generating validation report...")
    _generate_phase2_report(checks, output_dir / 'phase_2_validation_report.md')

    # Save JSON results
    validation_path = output_dir / 'phase_2_validation_results.json'
    with open(validation_path, 'w') as f:
        json.dump(checks, f, indent=2)

    print(f"\n[Validation] Complete! Results saved to {output_dir}/")

    return checks


def _validate_stage2_integrity(
    embeddings_array: np.ndarray,
    metadata: pd.DataFrame,
    similarity_df: pd.DataFrame
) -> dict:
    """Run data integrity checks on Stage 2 outputs."""
    checks = {}

    # Check 1: Data shapes
    n_lineages = len(embeddings_array)
    embedding_dim = embeddings_array.shape[1]
    n_fronts = len(similarity_df.columns) - 1

    checks['n_lineages'] = int(n_lineages)
    checks['embedding_dim'] = int(embedding_dim)
    checks['n_fronts'] = int(n_fronts)

    # Check 2: Embeddings are normalized
    norms = np.linalg.norm(embeddings_array, axis=1)
    checks['embeddings_normalized'] = bool(np.allclose(norms, 1.0, rtol=1e-3))
    checks['norm_mean'] = float(norms.mean())
    checks['norm_std'] = float(norms.std())

    # Check 3: No NaN/Inf
    checks['embeddings_finite'] = bool(np.isfinite(embeddings_array).all())

    # Check 4: No nulls in metadata
    metadata_nulls = metadata.isnull().sum().sum()
    checks['metadata_no_nulls'] = bool(metadata_nulls == 0)

    # Check 5: Similarity scores in [-1, 1]
    similarity_values = similarity_df.iloc[:, 1:].values.flatten()
    checks['similarity_range_ok'] = bool((similarity_values >= -1).all() and (similarity_values <= 1).all())
    checks['similarity_min'] = float(similarity_values.min())
    checks['similarity_max'] = float(similarity_values.max())
    checks['similarity_mean'] = float(similarity_values.mean())

    # Check 6: Coverage
    threshold = 0.747
    has_match = (similarity_df.iloc[:, 1:] > threshold).any(axis=1)
    checks['coverage_pct'] = float(has_match.sum() / len(similarity_df) * 100)
    checks['lineages_with_matches'] = int(has_match.sum())
    checks['coverage_threshold'] = threshold

    # Check 7: Front matches
    front_matches = {}
    for col in similarity_df.columns[1:]:
        n_matches = int((similarity_df[col] > threshold).sum())
        front_matches[col] = n_matches
    checks['front_matches'] = front_matches

    # Check 8: Embedding variance
    embedding_variance = embeddings_array.var(axis=0)
    checks['embedding_variance_mean'] = float(embedding_variance.mean())
    checks['embedding_variance_std'] = float(embedding_variance.std())

    return checks


def _generate_phase2_heatmap(similarity_df: pd.DataFrame, output_path: Path):
    """Generate heatmap of lineage-front cosine similarity."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Get top 30 lineages by max similarity
    max_similarity = similarity_df.iloc[:, 1:].max(axis=1)
    top_indices = max_similarity.nlargest(30).index
    top_lineages = similarity_df.loc[top_indices]

    lineage_ids = top_lineages['lineage_id'].values
    heatmap_data = top_lineages.iloc[:, 1:].values
    front_names = top_lineages.columns[1:]

    fig, ax = plt.subplots(figsize=(14, 10))
    sns.heatmap(
        heatmap_data,
        xticklabels=front_names,
        yticklabels=[f"L{lid}" for lid in lineage_ids],
        cmap='RdYlGn',
        center=0,
        vmin=-0.2,
        vmax=1.0,
        cbar_kws={'label': 'Cosine Similarity'},
        ax=ax
    )

    ax.set_title('Stage 2: Lineage-Front Cosine Similarity (SciBERT, Top 30)',
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Research Front', fontsize=12)
    ax.set_ylabel('Lineage ID', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def _generate_phase2_tsne(embeddings_array: np.ndarray, similarity_df: pd.DataFrame, output_path: Path):
    """Generate t-SNE visualization."""
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE

    # Compute t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings_array)-1))
    embeddings_2d = tsne.fit_transform(embeddings_array)

    # Get best matching front
    similarity_df.columns[1:].tolist()
    best_fronts = similarity_df.iloc[:, 1:].idxmax(axis=1).values

    # Create color map
    unique_fronts = sorted(set(best_fronts))
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_fronts)))
    front_to_color = {front: colors[i] for i, front in enumerate(unique_fronts)}

    fig, ax = plt.subplots(figsize=(14, 10))
    for front in unique_fronts:
        mask = best_fronts == front
        ax.scatter(
            embeddings_2d[mask, 0],
            embeddings_2d[mask, 1],
            c=[front_to_color[front]],
            label=front,
            alpha=0.6,
            s=50
        )

    ax.set_title('Stage 2: Lineage Embedding Space (t-SNE, Colored by Best Front)',
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('t-SNE Dimension 1', fontsize=12)
    ax.set_ylabel('t-SNE Dimension 2', fontsize=12)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def _generate_phase2_distributions(embeddings_array: np.ndarray, similarity_df: pd.DataFrame, output_path: Path):
    """Generate 4-panel distribution plots."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Panel 1: Cosine similarity distribution
    ax = axes[0, 0]
    similarity_values = similarity_df.iloc[:, 1:].values.flatten()
    ax.hist(similarity_values, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
    ax.set_xlabel('Cosine Similarity', fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    ax.set_title('Cosine Similarity Distribution', fontsize=12, fontweight='bold')
    ax.axvline(0.747, color='red', linestyle='--', label='Threshold (0.747)')
    ax.grid(axis='y', alpha=0.3)
    ax.legend()

    # Panel 2: Embedding norm distribution
    ax = axes[0, 1]
    norms = np.linalg.norm(embeddings_array, axis=1)
    try:
        ax.hist(norms, bins=10, color='coral', edgecolor='black', alpha=0.7)
    except ValueError:
        ax.axvline(norms.mean(), color='coral', linewidth=10, alpha=0.7, label=f'All norms = {norms.mean():.4f}')
        ax.legend()
    ax.set_xlabel('L2 Norm', fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    ax.set_title('Embedding Norm Distribution', fontsize=12, fontweight='bold')
    ax.axvline(1.0, color='red', linestyle='--', label='Target (1.0)')
    ax.grid(axis='y', alpha=0.3)
    ax.legend()

    # Panel 3: Matches per front
    ax = axes[1, 0]
    threshold = 0.747
    matches_per_front = (similarity_df.iloc[:, 1:] > threshold).sum(axis=0)
    front_names = matches_per_front.index

    ax.barh(range(len(matches_per_front)), matches_per_front.values,
            color='mediumseagreen', edgecolor='black', alpha=0.7)
    ax.set_yticks(range(len(matches_per_front)))
    ax.set_yticklabels(front_names, fontsize=9)
    ax.set_xlabel(f'Number of Lineages (sim > {threshold})', fontsize=11)
    ax.set_title(f'Matches per Research Front (threshold={threshold})', fontsize=12, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)

    # Panel 4: Best similarity per lineage
    ax = axes[1, 1]
    best_similarities = similarity_df.iloc[:, 1:].max(axis=1)
    ax.hist(best_similarities.values, bins=30, color='orchid', edgecolor='black', alpha=0.7)
    ax.set_xlabel('Best Similarity Score', fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    ax.set_title('Best Front Match per Lineage', fontsize=12, fontweight='bold')
    ax.axvline(0.747, color='red', linestyle='--', label='Threshold (0.747)')
    ax.grid(axis='y', alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def _generate_phase2_report(checks: dict, output_path: Path):
    """Generate markdown validation report."""
    report = []
    report.append("# Stage 2 Validation Report")
    report.append("**SciBERT Embedding Quality**")
    report.append("")
    report.append("## Data Integrity")
    report.append("")
    report.append("| Check | Status | Details |")
    report.append("|-------|--------|---------|")

    report.append(f"| Dataset Shape | [OK] | {checks['n_lineages']} lineages, {checks['embedding_dim']}D embeddings, {checks['n_fronts']} fronts |")

    norm_status = "[OK]" if checks['embeddings_normalized'] else "[WARN]"
    report.append(f"| Embeddings Normalized | {norm_status} | Mean norm: {checks['norm_mean']:.4f}, Std: {checks['norm_std']:.4f} |")

    finite_status = "[OK]" if checks['embeddings_finite'] else "[FAIL]"
    report.append(f"| No NaN/Inf | {finite_status} | All embedding values are finite |")

    meta_status = "[OK]" if checks['metadata_no_nulls'] else "[FAIL]"
    report.append(f"| Metadata Complete | {meta_status} | All lineage metadata present |")

    sim_range_status = "[OK]" if checks['similarity_range_ok'] else "[FAIL]"
    report.append(f"| Similarity Range | {sim_range_status} | All scores in [-1, 1] |")

    coverage_status = "[OK]" if checks['coverage_pct'] > 50 else "[WARN]" if checks['coverage_pct'] > 0 else "[FAIL]"
    report.append(f"| Lineage Coverage | {coverage_status} | {checks['lineages_with_matches']}/{checks['n_lineages']} ({checks['coverage_pct']:.1f}%) above threshold {checks['coverage_threshold']} |")

    var_status = "[OK]" if checks['embedding_variance_mean'] > 0.001 else "[WARN]"
    report.append(f"| Embedding Variance | {var_status} | Mean: {checks['embedding_variance_mean']:.4f}, Std: {checks['embedding_variance_std']:.4f} |")

    report.append("")
    report.append("## Research Front Matches")
    report.append(f"**Threshold: {checks['coverage_threshold']:.3f}**")
    report.append("")
    report.append("| Research Front | Lineages Matched |")
    report.append("|----------------|------------------|")

    for front_name in sorted(checks['front_matches'].keys()):
        n_matches = checks['front_matches'][front_name]
        report.append(f"| {front_name} | {n_matches} |")

    report.append("")
    report.append("## Overall Assessment")
    report.append("")

    all_checks_pass = (
        checks['embeddings_normalized'] and
        checks['embeddings_finite'] and
        checks['metadata_no_nulls'] and
        checks['similarity_range_ok']
    )

    if all_checks_pass:
        report.append("[OK] **ALL VALIDATION CHECKS PASSED**")
    else:
        report.append("[FAIL] **SOME VALIDATION CHECKS FAILED**")

    with open(output_path, 'w') as f:
        f.write('\n'.join(report))


def main():
    parser = argparse.ArgumentParser(
        description="Compute SciBERT embeddings for community lineages"
    )
    parser.add_argument(
        "--lineage-metrics",
        type=Path,
        default=None,
        help="Path to lineage metrics CSV"
    )
    parser.add_argument(
        "--lineage-registry",
        type=Path,
        default=None,
        help="Path to lineage registry JSON"
    )
    parser.add_argument(
        "--graphs-dir",
        type=Path,
        default=None,
        help="Directory containing citation graph PKL files"
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=None,
        help="Directory containing raw JSONL files"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output path for embeddings (compressed numpy)"
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Base directory for Stage 2 artifacts (default: data/out)"
    )
    parser.add_argument(
        "--min-quarters",
        type=int,
        default=6,
        help="Minimum quarters for persistent lineages (default: 6)"
    )
    parser.add_argument(
        "--device",
        choices=["cuda", "cpu", "auto"],
        default="auto",
        help="Device for SciBERT inference"
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Enable detailed timing profiling"
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        default=True,
        help="Run validation checks and generate reports (default: True)"
    )
    parser.add_argument(
        "--no-validate",
        dest="validate",
        action="store_false",
        help="Skip validation checks"
    )
    add_domain_args(parser)

    args = parser.parse_args()

    paths = resolve_script_paths(args, repo_root)
    if args.lineage_metrics is None:
        args.lineage_metrics = paths.lineage_tracking / "lineage_metrics.csv" if paths else Path("data/out/02_lineage_tracking/lineage_metrics.csv")
    if args.lineage_registry is None:
        args.lineage_registry = paths.lineage_tracking / "lineage_registry.json" if paths else Path("data/out/02_lineage_tracking/lineage_registry.json")
    if args.graphs_dir is None:
        args.graphs_dir = paths.graphs if paths else Path("data/current_graphs")
    if args.raw_dir is None:
        args.raw_dir = paths.raw if paths else Path("data/current_ingest/raw")
    if args.output is None:
        args.output = paths.lineage_tracking / "lineage_embeddings.npz" if paths else Path("data/out/02_lineage_tracking/lineage_embeddings.npz")
    if args.output_root is None:
        args.output_root = paths.out if paths else Path("data/out")

    device = None if args.device == "auto" else args.device

    partitions_dir = paths.cache_cum / "partitions_cum" if paths else Path("data/out/cache_cum/partitions_cum")

    # Call run_embeddings() with standalone mode
    run_embeddings(
        min_quarters=args.min_quarters,
        device=device,
        profile=args.profile,
        output_path=args.output,
        lineage_metrics_path=args.lineage_metrics,
        front_config_path=Path('config/front_aliases.yaml'),
        partitions_dir=partitions_dir,
        output_root=args.output_root,
        registry_path=args.lineage_registry,  # Pass CLI argument
        raw_dir=args.raw_dir,  # Pass CLI argument
        graphs_dir=args.graphs_dir,  # Pass CLI argument
        store=None,  # Standalone mode
        validate=args.validate  # Pass validate flag
    )


if __name__ == "__main__":
    main()
