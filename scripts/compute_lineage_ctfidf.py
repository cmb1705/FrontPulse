#!/usr/bin/env python3
"""
Stage 3: c-TF-IDF Distinctive Term Extraction for Lineages

Extracts distinctive terms from each community lineage using class-based TF-IDF
and computes term similarity to research fronts.

Usage:
    python scripts/compute_lineage_ctfidf.py --min-quarters 6

Outputs (relative to --output-root):
    - 02_lineage_tracking/lineage_ctfidf_terms.csv: Top distinctive terms per lineage
    - 03_milestone_mapping/lineage_front_term_similarity.csv: Term similarity matrix (99 x 15)
"""

import json
import math
import re
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from _path_bootstrap import ensure_repo_imports
from tqdm import tqdm

# Add project root to path
REPO_ROOT = ensure_repo_imports()

# Import from Stage 2
from scripts.extract_abstracts import AbstractExtractor  # noqa: E402
from src.domain_registry import add_domain_args, resolve_script_paths  # noqa: E402

# Stopwords (same as Stage 2)
STOPWORDS = {
    'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from', 'has', 'he',
    'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the', 'to', 'was', 'will',
    'with', 'been', 'have', 'had', 'were', 'this', 'these', 'those', 'their',
    'there', 'which', 'can', 'we', 'our', 'but', 'not', 'all', 'they', 'may',
    'such', 'into', 'more', 'also', 'than', 'most', 'other', 'some', 'up', 'out',
    'only', 'so', 'no', 'if', 'about', 'or', 'when', 'after', 'between', 'then',
    'before', 'however', 'through', 'where', 'both', 'each', 'under', 'during',
    'using', 'very', 'any', 'over', 'how', 'same', 'would', 'could', 'should',
}

# Technical bigrams to preserve (will be converted to single tokens with underscore)
TECHNICAL_BIGRAMS = {
    'spin coating', 'thermal annealing', 'charge transport', 'charge carrier',
    'power conversion', 'conversion efficiency', 'open circuit', 'circuit voltage',
    'short circuit', 'fill factor', 'quantum efficiency', 'electron transport',
    'hole transport', 'energy level', 'band gap', 'fermi level', 'valence band',
    'conduction band', 'crystal structure', 'grain boundary', 'grain size',
    'atomic layer', 'layer deposition', 'chemical vapor', 'vapor deposition',
    'scanning electron', 'electron microscopy', 'transmission electron',
    'x ray', 'solar cell', 'solar cells', 'photovoltaic device',
    'device performance', 'device stability', 'thin film', 'perovskite solar',
    'tandem solar', 'charge extraction', 'carrier lifetime', 'recombination rate',
    'interface engineering', 'surface passivation', 'work function',
}

# Chemical formula patterns (map to semantic classes)
FORMULA_PATTERNS = {
    # ABX3 halide perovskites (A = MA/FA/Cs, B = Pb/Sn, X = I/Br/Cl)
    r'\b(ma|fa|cs)pb(i|br|cl)\d*\b': 'apb_halide_perovskite',
    r'\b(ma|fa|cs)sn(i|br|cl)\d*\b': 'asn_halide_perovskite',

    # Mixed cation/anion variants (e.g., cs0.05, facs, mapbi3-xbrx)
    r'\b(ma|fa|cs)\d*\.?\d*(ma|fa|cs)?\d*\.?\d*pb(i|br|cl)\d*\.?\d*\b': 'apb_halide_perovskite',

    # Double perovskites (A2B'B"X6)
    r'\bcs2(ag|cu|au|na)(bi|in|sb)(cl|br|i)\d+\b': 'double_perovskite',

    # Tin-based (lead-free)
    r'\b(ma|fa|cs)\d*sn\d*(i|br|cl)\d+\b': 'tin_perovskite',

    # 2D perovskites with organic spacers
    r'\b(ba|pea|dma|gua|3amp|4amp)\d*pb\d*(i|br|cl)\d+\b': '2d_perovskite',

    # Metal oxides (common transport materials)
    r'\btio2\b': 'titanium_dioxide',
    r'\bsno2\b': 'tin_dioxide',
    r'\bzno\b': 'zinc_oxide',
    r'\bniox?\b': 'nickel_oxide',
    r'\bwo3\b': 'tungsten_oxide',
    r'\bmoo3\b': 'molybdenum_oxide',

    # Fullerenes and derivatives
    r'\bpc\d+bm\b': 'fullerene_derivative',
    r'\bc\d+\b': 'fullerene',
}


class LineageTermExtractor:
    """Extract and score distinctive terms for lineages using c-TF-IDF."""

    def __init__(
        self,
        registry_path: Path = None,
        partitions_dir: Path = None,
        raw_dir: Path = None,
        front_config_path: Path = None,
        min_quarters: int = 12,
        cache_path: Path = None,
        store=None  # Optional LineageTextStore
    ):
        """
        Initialize term extractor.

        Args:
            registry_path: Path to lineage registry (or None if using store)
            partitions_dir: Path to partition JSONs
            raw_dir: Path to raw JSONL (or None if using store)
            front_config_path: Path to front config YAML
            min_quarters: Minimum quarters for persistent lineages
            cache_path: Optional path to serialized abstract index cache
            store: Optional pre-loaded LineageTextStore (pipeline mode)
        """
        self.partitions_dir = partitions_dir
        self.min_quarters = min_quarters
        self.cache_path = cache_path

        if store is not None:
            # Pipeline mode: use shared store
            print("[1/6] Using shared LineageTextStore...")
            start = time.time()

            # Use store's registry
            self.lineage_registry = store.registry_by_lineage

            # Filter persistent lineages
            self.persistent_lineages = store.get_persistent_lineages(min_quarters)

            # Use store's text extractor
            self.text_extractor = store.extractor

            print(f"      Found {len(self.persistent_lineages)} persistent lineages (>={min_quarters}q) in {time.time()-start:.3f}s")
        else:
            # Standalone mode: load fresh
            if registry_path is None or raw_dir is None:
                raise ValueError("registry_path and raw_dir required in standalone mode")

            self.registry_path = registry_path

            # Load lineage registry
            print("[1/6] Loading lineage registry...")
            start = time.time()
            with open(registry_path) as f:
                registry_by_quarter = json.load(f)

            # Invert to get {lineage_id: {quarter: {community_id: lineage_id}}}
            self.lineage_registry = {}
            for quarter, community_map in registry_by_quarter.items():
                for comm_id, lin_id in community_map.items():
                    if lin_id not in self.lineage_registry:
                        self.lineage_registry[lin_id] = {}
                    if quarter not in self.lineage_registry[lin_id]:
                        self.lineage_registry[lin_id][quarter] = {}
                    self.lineage_registry[lin_id][quarter][comm_id] = lin_id

            # Filter persistent lineages
            self.persistent_lineages = [
                lin_id for lin_id, quarters in self.lineage_registry.items()
                if len(quarters) >= min_quarters
            ]
            print(f"      Found {len(self.persistent_lineages)} persistent lineages (>={min_quarters}q) in {time.time()-start:.3f}s")

            # Initialize text extractor
            print("[3/6] Initializing AbstractExtractor...")
            start = time.time()
            self.text_extractor = AbstractExtractor(raw_dir, cache_path=cache_path)
            print(f"      OK in {time.time()-start:.3f}s")

        # Load research fronts (always needed)
        print("[2/6] Loading research front definitions...")
        start = time.time()
        with open(front_config_path) as f:
            self.fronts_config = yaml.safe_load(f)
        print(f"      Loaded {len(self.fronts_config)} research fronts in {time.time()-start:.3f}s")

        # Cache for partition data (reuse from Stage 2 approach)
        self._partition_cache = {}

        # Will store: {lineage_id: {term: count}}
        self.lineage_term_counts = {}

        # Will store: {term: num_lineages_containing_term}
        self.term_document_frequency = Counter()

    def load_lineage_papers_fast(self, lineage_id: int) -> list[str]:
        """
        Load papers for a lineage using cached JSON partitions.
        Reuses optimization from Stage 2.
        """
        quarters_map = self.lineage_registry[lineage_id]
        all_papers = []

        for quarter, community_map in quarters_map.items():
            # Get community IDs for this lineage in this quarter
            community_ids = [
                int(comm_id) for comm_id, lin_id in community_map.items()
                if lin_id == lineage_id
            ]

            # Load partition if not cached
            if quarter not in self._partition_cache:
                partition_path = self.partitions_dir / f"part_{quarter}.json"
                if not partition_path.exists():
                    continue

                with open(partition_path) as f:
                    data = json.load(f)

                # Invert to {community_id: [work_ids]}
                inverted = {}
                for work_id, comm_id in data.get('labels', {}).items():
                    if comm_id not in inverted:
                        inverted[comm_id] = []
                    inverted[comm_id].append(work_id)

                self._partition_cache[quarter] = inverted

            # Get papers from cache
            partition_inv = self._partition_cache[quarter]
            for comm_id in community_ids:
                if comm_id in partition_inv:
                    all_papers.extend(partition_inv[comm_id])

        # Deduplicate
        seen = set()
        unique_papers = []
        for paper in all_papers:
            if paper not in seen:
                seen.add(paper)
                unique_papers.append(paper)

        return unique_papers

    def tokenize_and_filter(self, text: str) -> list[str]:
        """
        Tokenize text and filter stopwords/short tokens.

        IMPROVED: Now includes domain normalization:
        - Normalizes chemical formulas to semantic classes
        - Preserves technical bigrams as single tokens

        Returns list of lowercase tokens.
        """
        # Lowercase
        text = text.lower()

        # DOMAIN NORMALIZATION 1: Preserve technical bigrams
        # Replace bigrams with underscore-joined versions before tokenization
        for bigram in TECHNICAL_BIGRAMS:
            bigram_lower = bigram.lower()
            # Replace with underscore version (e.g., "spin coating" → "spin_coating")
            text = text.replace(bigram_lower, bigram_lower.replace(' ', '_'))

        # DOMAIN NORMALIZATION 2: Normalize chemical formulas
        # Replace specific formulas with semantic class names
        for pattern, normalized_term in FORMULA_PATTERNS.items():
            text = re.sub(pattern, normalized_term, text)

        # Extract alphanumeric tokens with basic pattern
        # Keep hyphenated compounds (e.g., "p-i-n", "2D-3D") and underscores (bigrams)
        tokens = re.findall(r'\b[\w\-]+\b', text)

        # Filter
        filtered = [
            token for token in tokens
            if len(token) > 2 and token not in STOPWORDS
        ]

        return filtered

    def extract_lineage_terms(self, lineage_id: int) -> dict[str, int]:
        """
        Extract term counts for a single lineage.
        Returns: {term: count}
        """
        # Load papers
        paper_ids = self.load_lineage_papers_fast(lineage_id)

        # Extract texts using batch method for efficiency
        text_dict = self.text_extractor.get_texts_batch(paper_ids)
        texts = list(text_dict.values())

        # Tokenize and count
        term_counts = Counter()
        for text in texts:
            tokens = self.tokenize_and_filter(text)
            term_counts.update(tokens)

        return dict(term_counts)

    def compute_ctfidf_for_lineage(self, _lineage_id: int, term_counts: dict[str, int]) -> dict[str, float]:
        """
        Compute c-TF-IDF scores for terms in a lineage.

        c-TF-IDF = TF * log(N / DF)
        where:
            TF = term frequency in lineage
            N = total number of lineages
            DF = number of lineages containing term

        Returns: {term: ctfidf_score}
        """
        N = len(self.persistent_lineages)
        ctfidf_scores = {}

        for term, tf in term_counts.items():
            df = self.term_document_frequency.get(term, 1)  # Avoid division by zero
            idf = math.log(N / df)
            ctfidf_scores[term] = tf * idf

        return ctfidf_scores

    def extract_all_lineage_terms(self):
        """
        Extract term counts for all lineages and compute document frequencies.
        This is a two-pass process:
        1. Extract terms from all lineages
        2. Compute DF for each term across lineages
        """
        print(f"\n[4/6] Extracting terms from {len(self.persistent_lineages)} lineages...")

        # Pass 1: Extract term counts
        for lineage_id in tqdm(self.persistent_lineages, desc="Extracting terms"):
            term_counts = self.extract_lineage_terms(int(lineage_id))
            self.lineage_term_counts[int(lineage_id)] = term_counts

        # Pass 2: Compute document frequency
        print("\n[5/6] Computing term document frequencies...")
        for _lineage_id, term_counts in self.lineage_term_counts.items():
            for term in term_counts:
                self.term_document_frequency[term] += 1

        print(f"      Found {len(self.term_document_frequency)} unique terms across all lineages")

    def compute_ctfidf_scores(self) -> dict[int, dict[str, float]]:
        """
        Compute c-TF-IDF scores for all lineages.
        Returns: {lineage_id: {term: score}}
        """
        print("\n[6/6] Computing c-TF-IDF scores...")
        ctfidf_scores = {}

        for lineage_id in tqdm(self.persistent_lineages, desc="Computing c-TF-IDF"):
            term_counts = self.lineage_term_counts[int(lineage_id)]
            scores = self.compute_ctfidf_for_lineage(int(lineage_id), term_counts)
            ctfidf_scores[int(lineage_id)] = scores

        return ctfidf_scores

    def get_front_terms(self, front_name: str) -> set[str]:
        """
        Extract all terms (canonical + aliases) for a research front.
        Returns set of lowercase terms.
        """
        front_config = self.fronts_config[front_name]
        terms = set()

        # Add canonical terms
        for term in front_config.get('canonical', []):
            # Tokenize multi-word terms
            tokens = self.tokenize_and_filter(term)
            terms.update(tokens)

        # Add aliases
        for term in front_config.get('aliases', []):
            tokens = self.tokenize_and_filter(term)
            terms.update(tokens)

        return terms

    def compute_term_similarity(
        self,
        lineage_terms: dict[str, float],
        front_terms: set[str],
        min_threshold: float = 0.01,
        top_k: int = 50
    ) -> float:
        """
        Compute term overlap similarity between lineage and front.

        Uses weighted overlap: sum of c-TF-IDF scores for matching terms,
        normalized by total lineage term score.

        IMPROVED: Now uses only top-K terms to avoid long-tail dilution.

        Args:
            lineage_terms: {term: ctfidf_score} for lineage
            front_terms: Set of canonical terms for front
            min_threshold: Minimum similarity to return non-zero (default: 0.01 = 1%)
            top_k: Use only top-K highest scoring terms (default: 50)

        Returns: similarity score in [0, 1], or 0 if below threshold
        """
        if not lineage_terms:
            return 0.0

        # TOP-K REWEIGHTING: Use only top-K terms to avoid long-tail dilution
        # Sort terms by score and take top-K
        sorted_terms = sorted(lineage_terms.items(), key=lambda x: x[1], reverse=True)
        top_terms = dict(sorted_terms[:top_k])

        # Sum c-TF-IDF scores for matching terms (only in top-K)
        overlap_score = sum(
            score for term, score in top_terms.items()
            if term in front_terms
        )

        # Normalize by total top-K term score (not all terms!)
        total_score = sum(top_terms.values())

        if total_score == 0:
            return 0.0

        similarity = overlap_score / total_score

        # Apply threshold to filter weak matches
        return similarity if similarity >= min_threshold else 0.0

    def generate_outputs(
        self,
        ctfidf_scores: dict[int, dict[str, float]],
        top_n: int = 100,
        similarity_threshold: float = 0.01,
        output_dir_lineage: Path = Path("data/out/02_lineage_tracking"),
        output_dir_mapping: Path = Path("data/out/03_milestone_mapping")
    ):
        """
        Generate output files:
        1. Top N distinctive terms per lineage
        2. Term similarity matrix (lineage x front)

        Args:
            ctfidf_scores: {lineage_id: {term: score}}
            top_n: Number of top terms to output per lineage
            similarity_threshold: Minimum similarity for non-zero match (default: 0.01 = 1%)
        """
        # Set up output directories
        output_dir_lineage = Path(output_dir_lineage)
        output_dir_mapping = Path(output_dir_mapping)
        output_dir_lineage.mkdir(parents=True, exist_ok=True)
        output_dir_mapping.mkdir(parents=True, exist_ok=True)

        print("\n[Output 1/2] Writing top distinctive terms per lineage...")

        # Output 1: Top terms per lineage
        rows = []
        for lineage_id in sorted(ctfidf_scores.keys()):
            scores = ctfidf_scores[lineage_id]

            # Sort by score descending
            sorted_terms = sorted(scores.items(), key=lambda x: x[1], reverse=True)

            # Take top N
            for rank, (term, score) in enumerate(sorted_terms[:top_n], start=1):
                rows.append({
                    'lineage_id': lineage_id,
                    'rank': rank,
                    'term': term,
                    'ctfidf_score': score
                })

        df_terms = pd.DataFrame(rows)
        output_path = output_dir_lineage / "lineage_ctfidf_terms.csv"
        df_terms.to_csv(output_path, index=False)
        print(f"      Wrote {len(rows)} term records to {output_path}")

        # Output 2: Term similarity matrix
        print("\n[Output 2/2] Computing lineage-front term similarity matrix...")

        similarity_rows = []
        front_names = sorted(self.fronts_config.keys())

        # Pre-extract front terms
        front_term_sets = {
            front_name: self.get_front_terms(front_name)
            for front_name in front_names
        }

        for lineage_id in tqdm(sorted(ctfidf_scores.keys()), desc="Computing similarity"):
            lineage_terms = ctfidf_scores[lineage_id]

            row = {'lineage_id': lineage_id}

            for front_name in front_names:
                front_terms = front_term_sets[front_name]
                similarity = self.compute_term_similarity(
                    lineage_terms, front_terms, min_threshold=similarity_threshold
                )
                row[front_name] = similarity

            similarity_rows.append(row)

        df_similarity = pd.DataFrame(similarity_rows)
        output_path = output_dir_mapping / "lineage_front_term_similarity.csv"
        df_similarity.to_csv(output_path, index=False)
        print(f"      Wrote {len(similarity_rows)} x {len(front_names)} similarity matrix to {output_path}")

        return df_terms, df_similarity


def run_ctfidf(
    min_quarters: int = 6,
    top_n: int = 100,
    similarity_threshold: float = 0.01,
    front_config_path: Path = Path('config/front_aliases.yaml'),
    partitions_dir: Path = Path('data/out/cache_cum/partitions_cum'),
    registry_path: Path = None,  # For standalone mode
    raw_dir: Path = None,  # For standalone mode
    abstract_cache_path: Path = None,
    store=None,  # Optional LineageTextStore for pipeline mode
    validate: bool = True,  # Run validation checks and generate reports
    output_root: Path = Path('data/out')  # Base directory for outputs
) -> tuple:
    """
    Run Stage 3: c-TF-IDF term extraction.

    This function can be called from the pipeline (with pre-loaded store) or
    standalone (loads fresh resources).

    Args:
        min_quarters: Minimum quarters for persistent lineages
        top_n: Number of top terms per lineage
        similarity_threshold: Minimum similarity for non-zero match
        front_config_path: Path to research front YAML
        partitions_dir: Path to partition JSONs
        registry_path: Path to lineage registry JSON (standalone mode)
        raw_dir: Path to raw JSONL files (standalone mode)
        abstract_cache_path: Optional path to serialized abstract index cache
        store: Optional pre-loaded LineageTextStore (pipeline mode)
        validate: Run validation checks and generate reports (default: True)

    Returns:
        (df_terms, df_similarity, validation_results): DataFrames and validation dict
    """
    output_root = Path(output_root)
    output_dir_lineage = output_root / "02_lineage_tracking"
    output_dir_mapping = output_root / "03_milestone_mapping"
    validation_dir = output_root / "06_validation" / "stage3"

    print("=" * 70)
    print("Stage 3: c-TF-IDF Distinctive Term Extraction")
    print("=" * 70)

    # Initialize extractor (with or without shared store)
    extractor = LineageTermExtractor(
        registry_path=None if store else (registry_path or Path('data/out/02_lineage_tracking/lineage_registry.json')),
        partitions_dir=partitions_dir,
        raw_dir=None if store else (raw_dir or Path('data/current_ingest/raw')),
        front_config_path=front_config_path,
        min_quarters=min_quarters,
        cache_path=abstract_cache_path,
        store=store
    )

    # Extract terms
    overall_start = time.time()
    extractor.extract_all_lineage_terms()

    # Compute c-TF-IDF
    ctfidf_scores = extractor.compute_ctfidf_scores()

    # Generate outputs
    df_terms, df_similarity = extractor.generate_outputs(
        ctfidf_scores,
        top_n=top_n,
        similarity_threshold=similarity_threshold,
        output_dir_lineage=output_dir_lineage,
        output_dir_mapping=output_dir_mapping
    )

    overall_time = time.time() - overall_start

    print("\n" + "=" * 70)
    print("STAGE 3 COMPLETE")
    print("=" * 70)
    print(f"Total time: {overall_time:.1f}s ({overall_time/60:.1f} minutes)")
    print(f"Lineages processed: {len(extractor.persistent_lineages)}")
    print(f"Unique terms: {len(extractor.term_document_frequency)}")
    print(f"Average terms per lineage: {sum(len(counts) for counts in extractor.lineage_term_counts.values()) / len(extractor.lineage_term_counts):.1f}")
    print("\nOutputs:")
    print(f"  - {output_dir_lineage / 'lineage_ctfidf_terms.csv'}")
    print(f"  - {output_dir_mapping / 'lineage_front_term_similarity.csv'}")

    # Run validation if requested
    if validate:
        print(f"\n{'='*70}")
        print("STAGE 3 VALIDATION")
        print(f"{'='*70}\n")

        validation_dir.mkdir(parents=True, exist_ok=True)
        validation_results = run_phase3_validation(df_terms, df_similarity, validation_dir=validation_dir)
    else:
        validation_results = None

    return df_terms, df_similarity, validation_results


# ============================================================================
# VALIDATION FUNCTIONS (integrated from validate_stage3.py)
# ============================================================================

def run_phase3_validation(
    terms_df: pd.DataFrame,
    similarity_df: pd.DataFrame,
    validation_dir: Path = Path('data/out/06_validation/stage3')
) -> dict:
    """
    Run Stage 3 validation checks and generate outputs.

    Args:
        terms_df: DataFrame with c-TF-IDF terms
        similarity_df: DataFrame with lineage-front similarity scores

    Returns:
        Dictionary with validation results
    """
    # Lazy imports to avoid overhead when validation disabled

    # Create output directory
    output_dir = Path(validation_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run validation checks
    print("[1/5] Running data integrity checks...")
    checks = _validate_stage3_integrity(terms_df, similarity_df)

    # Generate visualizations
    print("[2/5] Generating similarity heatmap...")
    _generate_phase3_heatmap(similarity_df, output_dir / 'phase_3_similarity_heatmap.png')

    print("[3/5] Generating top terms showcase...")
    _generate_phase3_top_terms_showcase(terms_df, similarity_df, output_dir / 'phase_3_top_terms.png')

    print("[4/5] Generating score distributions...")
    _generate_phase3_distributions(terms_df, similarity_df, output_dir / 'phase_3_distributions.png')

    # Generate report
    print("[5/5] Generating validation report...")
    _generate_phase3_report(checks, output_dir / 'phase_3_validation_report.md')

    # Save JSON results
    validation_path = output_dir / 'phase_3_validation_results.json'
    with open(validation_path, 'w') as f:
        json.dump(checks, f, indent=2)

    print(f"\n[Validation] Complete! Results saved to {output_dir}/")

    return checks


def _validate_stage3_integrity(terms_df: pd.DataFrame, similarity_df: pd.DataFrame) -> dict:
    """Run data integrity checks on Stage 3 outputs."""
    checks = {}

    # Check 1: Data shapes
    n_lineages = len(similarity_df)
    n_fronts = len(similarity_df.columns) - 1  # Exclude lineage_id
    n_term_records = len(terms_df)

    checks['n_lineages'] = int(n_lineages)
    checks['n_fronts'] = int(n_fronts)
    checks['n_term_records'] = int(n_term_records)

    # Check 2: No nulls
    terms_nulls = terms_df.isnull().sum().sum()
    similarity_nulls = similarity_df.isnull().sum().sum()
    checks['terms_no_nulls'] = bool(terms_nulls == 0)
    checks['similarity_no_nulls'] = bool(similarity_nulls == 0)

    # Check 3: c-TF-IDF scores are positive
    ctfidf_scores = terms_df['ctfidf_score'].values
    checks['ctfidf_positive'] = bool((ctfidf_scores >= 0).all())
    checks['ctfidf_min'] = float(ctfidf_scores.min())
    checks['ctfidf_max'] = float(ctfidf_scores.max())
    checks['ctfidf_mean'] = float(ctfidf_scores.mean())

    # Check 4: Similarity scores in [0, 1]
    similarity_values = similarity_df.iloc[:, 1:].values.flatten()
    checks['similarity_range_ok'] = bool((similarity_values >= 0).all() and
                                         (similarity_values <= 1).all())
    checks['similarity_min'] = float(similarity_values.min())
    checks['similarity_max'] = float(similarity_values.max())
    checks['similarity_mean'] = float(similarity_values.mean())

    # Check 5: Coverage (lineages with at least one non-zero match)
    has_match = (similarity_df.iloc[:, 1:] > 0).any(axis=1)
    checks['coverage_pct'] = float(has_match.sum() / len(similarity_df) * 100)
    checks['lineages_with_matches'] = int(has_match.sum())

    # Check 6: Front match counts
    front_matches = {}
    for col in similarity_df.columns[1:]:
        n_matches = int((similarity_df[col] > 0).sum())
        front_matches[col] = n_matches
    checks['front_matches'] = front_matches

    return checks


def _generate_phase3_heatmap(similarity_df: pd.DataFrame, output_path: Path):
    """Generate heatmap of lineage-front term similarity."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Get top 30 lineages by max similarity
    max_similarity = similarity_df.iloc[:, 1:].max(axis=1)
    top_indices = max_similarity.nlargest(30).index
    top_lineages = similarity_df.loc[top_indices]

    # Prepare data for heatmap
    lineage_ids = top_lineages['lineage_id'].values
    heatmap_data = top_lineages.iloc[:, 1:].values
    front_names = top_lineages.columns[1:]

    # Create heatmap
    fig, ax = plt.subplots(figsize=(14, 10))
    sns.heatmap(
        heatmap_data,
        xticklabels=front_names,
        yticklabels=[f"L{lid}" for lid in lineage_ids],
        cmap='YlOrRd',
        vmin=0,
        vmax=heatmap_data.max(),
        cbar_kws={'label': 'Term Similarity'},
        ax=ax
    )

    ax.set_title('Stage 3: Lineage-Front Term Similarity (c-TF-IDF, Top 30)',
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Research Front', fontsize=12)
    ax.set_ylabel('Lineage ID', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def _generate_phase3_top_terms_showcase(terms_df: pd.DataFrame, similarity_df: pd.DataFrame,
                                         output_path: Path, n_examples: int = 15):
    """Generate showcase of top distinctive terms for top-matched lineages."""
    import matplotlib.pyplot as plt

    # Get top N lineages by max similarity
    max_similarity = similarity_df.iloc[:, 1:].max(axis=1)
    top_lineage_ids = similarity_df.loc[max_similarity.nlargest(n_examples).index, 'lineage_id'].values

    # Prepare data
    showcase_data = []
    for lineage_id in top_lineage_ids:
        lineage_terms = terms_df[terms_df['lineage_id'] == lineage_id].nlargest(10, 'ctfidf_score')

        # Get best matching front
        lineage_row = similarity_df[similarity_df['lineage_id'] == lineage_id].iloc[0]
        best_front = lineage_row[1:].idxmax()
        best_score = lineage_row[best_front]

        # Format term list
        term_list = ', '.join(lineage_terms['term'].head(10).tolist())

        showcase_data.append({
            'lineage': f"L{lineage_id}",
            'front': best_front,
            'similarity': best_score,
            'terms': term_list
        })

    # Create text-based visualization
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.axis('off')

    y_pos = 0.95
    for _i, row in enumerate(showcase_data):
        # Lineage header
        header = f"{row['lineage']} -> {row['front']} (sim={row['similarity']:.3f})"
        ax.text(0.05, y_pos, header, fontsize=10, fontweight='bold',
                verticalalignment='top', family='monospace')
        y_pos -= 0.03

        # Terms (wrapped)
        terms_text = row['terms']
        # Wrap at ~100 chars
        if len(terms_text) > 100:
            # Find last comma before 100 chars
            wrap_pos = terms_text[:100].rfind(',')
            if wrap_pos > 0:
                ax.text(0.05, y_pos, terms_text[:wrap_pos+1], fontsize=8,
                        verticalalignment='top', family='monospace', color='#333333')
                y_pos -= 0.025
                ax.text(0.05, y_pos, terms_text[wrap_pos+2:], fontsize=8,
                        verticalalignment='top', family='monospace', color='#333333')
            else:
                ax.text(0.05, y_pos, terms_text, fontsize=8,
                        verticalalignment='top', family='monospace', color='#333333')
        else:
            ax.text(0.05, y_pos, terms_text, fontsize=8,
                    verticalalignment='top', family='monospace', color='#333333')

        y_pos -= 0.04

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title('Stage 3: Top Distinctive Terms for Top-Matched Lineages',
                 fontsize=14, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def _generate_phase3_distributions(terms_df: pd.DataFrame, similarity_df: pd.DataFrame, output_path: Path):
    """Generate 4-panel distribution plots."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Panel 1: c-TF-IDF score distribution
    ax = axes[0, 0]
    ctfidf_scores = terms_df['ctfidf_score'].values
    ax.hist(ctfidf_scores, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
    ax.set_xlabel('c-TF-IDF Score', fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    ax.set_title('c-TF-IDF Score Distribution', fontsize=12, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    # Add stats
    stats_text = f"Mean: {ctfidf_scores.mean():.2f}\nMedian: {np.median(ctfidf_scores):.2f}\nMax: {ctfidf_scores.max():.2f}"
    ax.text(0.98, 0.97, stats_text, transform=ax.transAxes,
            fontsize=9, verticalalignment='top', horizontalalignment='right',
            bbox={'boxstyle': 'round', 'facecolor': 'wheat', 'alpha': 0.5})

    # Panel 2: Similarity score distribution
    ax = axes[0, 1]
    similarity_values = similarity_df.iloc[:, 1:].values.flatten()
    nonzero_similarities = similarity_values[similarity_values > 0]

    if len(nonzero_similarities) > 0:
        ax.hist(nonzero_similarities, bins=50, color='coral', edgecolor='black', alpha=0.7)
        stats_text = f"Mean: {nonzero_similarities.mean():.3f}\nMedian: {np.median(nonzero_similarities):.3f}\nMax: {nonzero_similarities.max():.3f}\nN={len(nonzero_similarities)}"
    else:
        ax.text(0.5, 0.5, 'No non-zero similarities', transform=ax.transAxes,
                ha='center', va='center', fontsize=12)
        stats_text = "No matches"

    ax.set_xlabel('Similarity Score (non-zero)', fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    ax.set_title('Similarity Score Distribution', fontsize=12, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    if len(nonzero_similarities) > 0:
        ax.text(0.98, 0.97, stats_text, transform=ax.transAxes,
                fontsize=9, verticalalignment='top', horizontalalignment='right',
                bbox={'boxstyle': 'round', 'facecolor': 'wheat', 'alpha': 0.5})

    # Panel 3: Terms per lineage distribution
    ax = axes[1, 0]
    terms_per_lineage = terms_df.groupby('lineage_id').size()
    ax.hist(terms_per_lineage.values, bins=30, color='mediumseagreen', edgecolor='black', alpha=0.7)
    ax.set_xlabel('Number of Terms', fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    ax.set_title('Terms per Lineage Distribution', fontsize=12, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    stats_text = f"Mean: {terms_per_lineage.mean():.1f}\nMedian: {terms_per_lineage.median():.1f}\nMax: {terms_per_lineage.max()}"
    ax.text(0.98, 0.97, stats_text, transform=ax.transAxes,
            fontsize=9, verticalalignment='top', horizontalalignment='right',
            bbox={'boxstyle': 'round', 'facecolor': 'wheat', 'alpha': 0.5})

    # Panel 4: Matches per front distribution
    ax = axes[1, 1]
    matches_per_front = (similarity_df.iloc[:, 1:] > 0).sum(axis=0)
    front_names = matches_per_front.index

    ax.barh(range(len(matches_per_front)), matches_per_front.values,
            color='orchid', edgecolor='black', alpha=0.7)
    ax.set_yticks(range(len(matches_per_front)))
    ax.set_yticklabels(front_names, fontsize=9)
    ax.set_xlabel('Number of Lineages Matched', fontsize=11)
    ax.set_title('Matches per Research Front', fontsize=12, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def _generate_phase3_report(checks: dict, output_path: Path):
    """Generate markdown validation report."""
    report = []
    report.append("# Stage 3 Validation Report")
    report.append("**c-TF-IDF Distinctive Term Extraction**")
    report.append("")
    report.append("*Similarity threshold: 0.002 (0.2% term overlap), optimized for 20-40% coverage*")
    report.append("")

    # Data Integrity
    report.append("## Data Integrity")
    report.append("")
    report.append("| Check | Status | Details |")
    report.append("|-------|--------|---------|")

    # Shape checks
    report.append(f"| Dataset Shape | [OK] | {checks['n_lineages']} lineages, {checks['n_fronts']} fronts, {checks['n_term_records']} term records |")

    # Null checks
    terms_status = "[OK]" if checks['terms_no_nulls'] else "[FAIL]"
    sim_status = "[OK]" if checks['similarity_no_nulls'] else "[FAIL]"
    report.append(f"| No Nulls (Terms) | {terms_status} | All term records complete |")
    report.append(f"| No Nulls (Similarity) | {sim_status} | All similarity values present |")

    # c-TF-IDF score checks
    ctfidf_status = "[OK]" if checks['ctfidf_positive'] else "[FAIL]"
    report.append(f"| c-TF-IDF Positive | {ctfidf_status} | Range: [{checks['ctfidf_min']:.2f}, {checks['ctfidf_max']:.2f}], Mean: {checks['ctfidf_mean']:.2f} |")

    # Similarity range checks
    sim_range_status = "[OK]" if checks['similarity_range_ok'] else "[FAIL]"
    report.append(f"| Similarity Range | {sim_range_status} | All scores in [0, 1] |")

    # Coverage check
    coverage_status = "[OK]" if checks['coverage_pct'] > 0 else "[WARN]"
    report.append(f"| Lineage Coverage | {coverage_status} | {checks['lineages_with_matches']}/{checks['n_lineages']} ({checks['coverage_pct']:.1f}%) have matches |")

    report.append("")

    # Front Match Summary
    report.append("## Research Front Matches")
    report.append("")
    report.append("| Research Front | Lineages Matched |")
    report.append("|----------------|------------------|")

    for front_name in sorted(checks['front_matches'].keys()):
        n_matches = checks['front_matches'][front_name]
        report.append(f"| {front_name} | {n_matches} |")

    report.append("")

    # Key Statistics
    report.append("## Key Statistics")
    report.append("")
    report.append(f"- **c-TF-IDF Scores**: Mean={checks['ctfidf_mean']:.2f}, Max={checks['ctfidf_max']:.2f}")
    report.append(f"- **Similarity Scores**: Mean={checks['similarity_mean']:.3f}, Max={checks['similarity_max']:.3f}")
    report.append(f"- **Total Matches**: {sum(checks['front_matches'].values())} lineage-front pairs")
    report.append(f"- **Average Matches/Front**: {sum(checks['front_matches'].values()) / checks['n_fronts']:.1f}")

    report.append("")

    # Overall Assessment
    all_checks_pass = (
        checks['terms_no_nulls'] and
        checks['similarity_no_nulls'] and
        checks['ctfidf_positive'] and
        checks['similarity_range_ok']
    )

    if all_checks_pass:
        report.append("## Overall Assessment")
        report.append("")
        report.append("[OK] **ALL VALIDATION CHECKS PASSED**")
    else:
        report.append("## Overall Assessment")
        report.append("")
        report.append("[FAIL] **SOME VALIDATION CHECKS FAILED** - Review data integrity section")

    # Write report
    with open(output_path, 'w') as f:
        f.write('\n'.join(report))


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Compute c-TF-IDF term extraction for lineages")
    parser.add_argument(
        '--min-quarters',
        type=int,
        default=6,
        help='Minimum quarters for persistent lineages (default: 6)'
    )
    parser.add_argument(
        '--top-n',
        type=int,
        default=100,
        help='Number of top terms to extract per lineage (default: 100)'
    )
    parser.add_argument(
        '--registry',
        type=Path,
        default=None,
        help='Path to lineage registry JSON'
    )
    parser.add_argument(
        '--partitions',
        type=Path,
        default=None,
        help='Path to cached partition JSONs'
    )
    parser.add_argument(
        '--raw',
        type=Path,
        default=None,
        help='Path to raw JSONL data'
    )
    parser.add_argument(
        '--abstract-cache',
        type=Path,
        default=None,
        help='Path to serialized abstract index cache (default auto-generated)'
    )
    parser.add_argument(
        '--fronts',
        type=Path,
        default=Path('config/front_aliases.yaml'),
        help='Path to research front definitions'
    )
    parser.add_argument(
        '--similarity-threshold',
        type=float,
        default=0.01,
        help='Minimum similarity score for non-zero match (default: 0.01 = 1%%)'
    )
    parser.add_argument(
        '--validate',
        action='store_true',
        default=True,
        help='Run validation checks and generate reports (default: True)'
    )
    parser.add_argument(
        '--no-validate',
        dest='validate',
        action='store_false',
        help='Skip validation checks'
    )
    parser.add_argument(
        '--output-root',
        type=Path,
        default=None,
        help='Base directory for outputs (default: data/out)'
    )
    add_domain_args(parser)

    args = parser.parse_args()

    paths = resolve_script_paths(args, REPO_ROOT)
    if args.registry is None:
        args.registry = paths.lineage_tracking / "lineage_registry.json" if paths else Path("data/out/02_lineage_tracking/lineage_registry.json")
    if args.partitions is None:
        args.partitions = paths.cache_cum / "partitions_cum" if paths else Path("data/out/cache_cum/partitions_cum")
    if args.raw is None:
        args.raw = paths.raw if paths else Path("data/current_ingest/raw")
    if args.output_root is None:
        args.output_root = paths.out if paths else Path("data/out")

    # Call the refactored function (standalone mode, store=None)
    run_ctfidf(
        min_quarters=args.min_quarters,
        top_n=args.top_n,
        similarity_threshold=args.similarity_threshold,
        front_config_path=args.fronts,
        partitions_dir=args.partitions,
        registry_path=args.registry,  # Pass CLI argument
        raw_dir=args.raw,  # Pass CLI argument
        abstract_cache_path=args.abstract_cache,
        store=None,  # Standalone mode
        validate=args.validate,  # Pass validate flag
        output_root=args.output_root
    )


if __name__ == '__main__':
    main()
