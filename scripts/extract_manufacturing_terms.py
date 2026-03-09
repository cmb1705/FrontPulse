"""
Extract Manufacturing/Fabrication Terms for Phase 3 Alias Expansion

This script mines manufacturing and fabrication terminology from lineage abstracts
to expand the Phase 3 c-TF-IDF aliases. It uses TF-IDF to identify domain-specific
terms and filters them by manufacturing-relevance.

Usage:
    python scripts/extract_manufacturing_terms.py

Outputs:
    - data/out/manufacturing_terms_candidates.csv - Ranked candidate terms
    - data/out/manufacturing_terms_by_front.yaml - Terms grouped by front
"""

import re
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, List, Set, Tuple
import pandas as pd
import yaml
import json
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

# Import existing utilities
import sys
sys.path.append(str(Path(__file__).parent.parent))
from scripts.extract_abstracts import AbstractExtractor


class ManufacturingTermExtractor:
    """Extract manufacturing/fabrication terms from lineage abstracts."""

    def __init__(self):
        """Initialize extractor."""
        self.abstract_extractor = None
        self.lineage_registry = None
        self.front_config = None

        # Manufacturing-related seed patterns (for filtering)
        self.manufacturing_seeds = {
            # Fabrication methods
            'spin', 'coat', 'coating', 'deposit', 'deposition', 'anneal', 'annealing',
            'print', 'printing', 'roll', 'slot', 'blade', 'spray', 'evaporate',
            'sputtering', 'vacuum', 'thermal', 'chemical', 'bath', 'solution',
            'vapor', 'cvd', 'pvd', 'ald', 'pulsed',

            # Manufacturing processes
            'fabricate', 'fabrication', 'manufacture', 'manufacturing', 'process', 'processing',
            'scale', 'scalable', 'scalability', 'pilot', 'production', 'batch',
            'continuous', 'sequential', 'layer', 'multilayer', 'stack', 'stacking',

            # Equipment/tools
            'chamber', 'substrate', 'precursor', 'nozzle', 'slot-die', 'doctor',
            'meniscus', 'blade', 'bar', 'roller', 'drum', 'press', 'hot-plate',
            'oven', 'furnace', 'glovebox', 'hood',

            # Scale descriptors
            'large-area', 'large', 'area', 'module', 'panel', 'commercial',
            'industrial', 'prototype', 'laboratory', 'bench', 'pilot-scale',
            'upscaling', 'downscaling',

            # Quality/control
            'uniformity', 'uniform', 'reproducible', 'reproducibility', 'yield',
            'throughput', 'quality', 'control', 'optimize', 'optimization',
            'parameter', 'condition', 'temperature', 'speed', 'rate', 'time',
            'thickness', 'morphology', 'crystallization', 'drying',

            # Architecture-related
            'flexible', 'rigid', 'planar', 'mesoscopic', 'inverted', 'normal',
            'architecture', 'structure', 'configuration', 'geometry', 'design'
        }

    def load_data(self):
        """Load lineage registry and research fronts."""
        print("[1/6] Loading lineage registry...")
        registry_path = Path("data/out/02_lineage_tracking/lineage_registry.json")
        with open(registry_path, 'r', encoding='utf-8') as f:
            self.lineage_registry = json.load(f)

        # Filter to persistent lineages (>=20q)
        self.lineage_registry = {
            lid: data for lid, data in self.lineage_registry.items()
            if data.get('lifetime_quarters', 0) >= 20
        }
        print(f"      Found {len(self.lineage_registry)} persistent lineages (>=20q)")

        print("[2/6] Loading research front definitions...")
        front_config_path = Path("config/front_aliases.yaml")
        with open(front_config_path, 'r', encoding='utf-8') as f:
            self.front_config = yaml.safe_load(f)
        print(f"      Loaded {len(self.front_config)} research fronts")

        print("[3/6] Initializing AbstractExtractor...")
        self.abstract_extractor = AbstractExtractor('data/current_ingest/raw')
        print("      OK")

    def extract_lineage_texts(self) -> Dict[str, List[str]]:
        """
        Extract abstract texts for each lineage.

        Returns:
            Dictionary mapping lineage_id -> list of abstracts
        """
        print("[4/6] Extracting abstracts for all lineages...")
        lineage_texts = {}

        for lineage_id, lineage_data in self.lineage_registry.items():
            work_ids = lineage_data.get('work_ids', [])
            if not work_ids:
                continue

            # Get abstracts for all works in lineage
            abstracts = []
            for work_id in work_ids:
                abstract = self.abstract_extractor.get_abstract(work_id)
                if abstract and len(abstract.strip()) > 50:  # Minimum length
                    abstracts.append(abstract)

            if abstracts:
                lineage_texts[lineage_id] = abstracts

        print(f"      Extracted abstracts for {len(lineage_texts)} lineages")
        total_abstracts = sum(len(texts) for texts in lineage_texts.values())
        print(f"      Total abstracts: {total_abstracts}")

        return lineage_texts

    def tokenize_and_filter(self, text: str) -> List[str]:
        """
        Tokenize text and apply manufacturing-focused filtering.

        Args:
            text: Input text

        Returns:
            List of filtered tokens
        """
        # Convert to lowercase
        text = text.lower()

        # Tokenize (keep hyphens and underscores)
        tokens = re.findall(r'\b[\w-]+\b', text)

        filtered = []
        for token in tokens:
            # Skip very short tokens
            if len(token) < 3:
                continue

            # Skip pure numbers
            if token.isdigit():
                continue

            # Skip common stopwords (minimal list)
            stopwords = {'the', 'and', 'for', 'with', 'this', 'that', 'from',
                        'are', 'was', 'were', 'been', 'being', 'have', 'has',
                        'had', 'can', 'will', 'would', 'could', 'should'}
            if token in stopwords:
                continue

            # Skip tokens with too many numbers (e.g., "r2=0.95")
            if sum(c.isdigit() for c in token) > len(token) // 2:
                continue

            filtered.append(token)

        return filtered

    def compute_tfidf_scores(self, lineage_texts: Dict[str, List[str]]) -> pd.DataFrame:
        """
        Compute TF-IDF scores for all terms across lineages.

        Args:
            lineage_texts: Dictionary mapping lineage_id -> list of abstracts

        Returns:
            DataFrame with columns: term, tfidf_score, lineage_count, front_association
        """
        print("[5/6] Computing TF-IDF scores for term extraction...")

        # Flatten: Create one document per lineage (concatenate all abstracts)
        lineage_docs = {}
        for lineage_id, abstracts in lineage_texts.items():
            # Join all abstracts for this lineage
            full_text = ' '.join(abstracts)
            lineage_docs[lineage_id] = full_text

        # Convert to list (preserve lineage_id order)
        lineage_ids = list(lineage_docs.keys())
        documents = [lineage_docs[lid] for lid in lineage_ids]

        # Compute TF-IDF
        vectorizer = TfidfVectorizer(
            tokenizer=self.tokenize_and_filter,
            max_features=5000,  # Keep top 5000 terms
            min_df=2,  # Term must appear in at least 2 lineages
            max_df=0.8,  # Term must appear in at most 80% of lineages
            ngram_range=(1, 3)  # Unigrams, bigrams, trigrams
        )

        tfidf_matrix = vectorizer.fit_transform(documents)
        feature_names = vectorizer.get_feature_names_out()

        # Compute average TF-IDF score per term (across all lineages)
        avg_tfidf = np.asarray(tfidf_matrix.mean(axis=0)).flatten()

        # Count how many lineages each term appears in
        lineage_counts = np.asarray((tfidf_matrix > 0).sum(axis=0)).flatten()

        # Create DataFrame
        term_data = []
        for idx, term in enumerate(feature_names):
            term_data.append({
                'term': term,
                'avg_tfidf': avg_tfidf[idx],
                'lineage_count': lineage_counts[idx],
                'is_manufacturing_related': self._is_manufacturing_related(term)
            })

        df = pd.DataFrame(term_data)
        print(f"      Extracted {len(df)} unique terms")
        print(f"      Manufacturing-related: {df['is_manufacturing_related'].sum()}")

        return df

    def _is_manufacturing_related(self, term: str) -> bool:
        """
        Check if term is manufacturing/fabrication related.

        Args:
            term: Term to check

        Returns:
            True if term contains manufacturing-related keywords
        """
        term_lower = term.lower()

        # Check if any seed pattern appears in term
        for seed in self.manufacturing_seeds:
            if seed in term_lower:
                return True

        return False

    def rank_and_export(self, df: pd.DataFrame):
        """
        Rank terms and export candidates.

        Args:
            df: DataFrame with term statistics
        """
        print("[6/6] Ranking and exporting manufacturing term candidates...")

        # Filter to manufacturing-related terms
        mfg_terms = df[df['is_manufacturing_related']].copy()

        # Rank by TF-IDF score
        mfg_terms = mfg_terms.sort_values('avg_tfidf', ascending=False)

        # Export candidates
        output_path = Path("data/out/manufacturing_terms_candidates.csv")
        mfg_terms.to_csv(output_path, index=False)
        print(f"      Exported {len(mfg_terms)} manufacturing terms to {output_path}")

        # Also export top 300 as YAML for manual review
        top_terms = mfg_terms.head(300)['term'].tolist()

        # Group by front (heuristic: based on seed patterns in term)
        front_terms = self._group_terms_by_front(top_terms)

        yaml_path = Path("data/out/manufacturing_terms_by_front.yaml")
        with open(yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(front_terms, f, default_flow_style=False, sort_keys=False)
        print(f"      Exported top 300 terms grouped by front to {yaml_path}")

        # Print summary statistics
        print("\n=== Manufacturing Term Extraction Summary ===")
        print(f"Total terms extracted: {len(df)}")
        print(f"Manufacturing-related: {len(mfg_terms)} ({100*len(mfg_terms)/len(df):.1f}%)")
        print(f"\nTop 10 manufacturing terms:")
        for idx, row in mfg_terms.head(10).iterrows():
            print(f"  {idx+1}. {row['term']:<30} (TF-IDF: {row['avg_tfidf']:.4f}, appears in {row['lineage_count']} lineages)")

    def _group_terms_by_front(self, terms: List[str]) -> Dict[str, List[str]]:
        """
        Group terms by research front using heuristic matching.

        Args:
            terms: List of terms to group

        Returns:
            Dictionary mapping front name -> list of terms
        """
        # Define heuristic keywords for each front
        front_keywords = {
            'scalable_manufacturing': ['scale', 'scalable', 'manufacture', 'manufacturing', 'production',
                                      'batch', 'continuous', 'industrial', 'commercial', 'pilot',
                                      'throughput', 'yield', 'process', 'processing', 'upscale'],
            'large_area_modules': ['large', 'area', 'module', 'panel', 'blade', 'roll', 'slot',
                                  'doctor', 'coating', 'printing', 'spray', 'bar', 'meniscus'],
            'flexible_devices': ['flexible', 'flex', 'bend', 'roll', 'substrate', 'plastic',
                                'pet', 'pen', 'polymer', 'ito-free'],
            'inverted_architecture': ['inverted', 'architecture', 'structure', 'configuration',
                                     'stack', 'layer', 'sequence', 'normal', 'planar'],
            'mesoscopic_architecture': ['mesoscopic', 'scaffold', 'porous', 'tio2', 'zro2',
                                       'infiltration', 'mp-tio2'],
            'simplified_architectures': ['simplified', 'single', 'layer', 'htl-free', 'etl-free',
                                        'hole-transport-free', 'electron-transport-free'],
            'interface_passivation': ['passivation', 'interface', 'defect', 'trap', 'surface',
                                     'modify', 'modification', 'treatment', 'layer'],
            'stability_engineering': ['stability', 'stable', 'degrade', 'degradation', 'aging',
                                     'encapsulation', 'moisture', 'oxygen', 'uv', 'thermal'],
            'core_psc': ['perovskite', 'solar', 'cell', 'device', 'efficiency', 'pce',
                        'fabrication', 'preparation', 'synthesis']
        }

        # Group terms
        front_terms = defaultdict(list)

        for term in terms:
            term_lower = term.lower()
            matched = False

            # Try to match to a front
            for front_name, keywords in front_keywords.items():
                if any(kw in term_lower for kw in keywords):
                    front_terms[front_name].append(term)
                    matched = True
                    break

            # If no match, add to "general" category
            if not matched:
                front_terms['general_fabrication'].append(term)

        return dict(front_terms)

    def run(self):
        """Run the full extraction pipeline."""
        print("=" * 70)
        print("MANUFACTURING TERM EXTRACTION")
        print("=" * 70)

        self.load_data()
        lineage_texts = self.extract_lineage_texts()
        df_terms = self.compute_tfidf_scores(lineage_texts)
        self.rank_and_export(df_terms)

        print("\n" + "=" * 70)
        print("EXTRACTION COMPLETE")
        print("=" * 70)
        print("\nOutputs:")
        print("  - data/out/manufacturing_terms_candidates.csv")
        print("  - data/out/manufacturing_terms_by_front.yaml")


if __name__ == "__main__":
    extractor = ManufacturingTermExtractor()
    extractor.run()
