#!/usr/bin/env python3
"""
Stage 5: Ensemble Voting for Lineage-to-Front Mapping

Combines Stage 2 (SciBERT), Stage 3 (c-TF-IDF), and Stage 4 (NPMI) signals
to produce final lineage-to-front mappings with confidence scores and evidence.

Usage:
    python scripts/stage5_ensemble_mapping.py

Outputs (relative to --output-root):
    - 03_milestone_mapping/lineage_front_mappings.csv: Final mappings with confidence
    - mapping_evidence/: Evidence bundles per lineage (JSON)
    - phase5_summary.md: Human-readable summary report
"""

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm


class EnsembleFrontMapper:
    """Ensemble mapping of lineages to research fronts using Stage 2-4 signals."""

    def __init__(
        self,
        stage2_similarity_path: Path,
        stage3_similarity_path: Path,
        stage4_similarity_path: Path,
        stage3_terms_path: Path,
        stage4_pairs_path: Path,
        front_config_path: Path,
        top_k: int = 3
    ):
        """
        Initialize the ensemble front mapper.

        Args:
            stage2_similarity_path: Path to Stage 2 similarity matrix CSV
            stage3_similarity_path: Path to Stage 3 similarity matrix CSV
            stage4_similarity_path: Path to Stage 4 similarity matrix CSV
            stage3_terms_path: Path to Stage 3 top terms CSV
            stage4_pairs_path: Path to Stage 4 top pairs CSV
            front_config_path: Path to front configuration YAML
            top_k: Number of top fronts each Stage votes for (default: 3)
        """
        self.top_k = top_k

        # Load similarity matrices
        print("[1/6] Loading Stage 2-4 similarity matrices...")
        self.stage2_sim = pd.read_csv(stage2_similarity_path, index_col='lineage_id')
        self.stage3_sim = pd.read_csv(stage3_similarity_path, index_col='lineage_id')
        self.stage4_sim = pd.read_csv(stage4_similarity_path, index_col='lineage_id')

        print(f"      Stage 2: {len(self.stage2_sim)} lineages x {len(self.stage2_sim.columns)} fronts")
        print(f"      Stage 3: {len(self.stage3_sim)} lineages x {len(self.stage3_sim.columns)} fronts")
        print(f"      Stage 4: {len(self.stage4_sim)} lineages x {len(self.stage4_sim.columns)} fronts")

        # Load supporting data
        print("[2/6] Loading Stage 3 terms and Stage 4 pairs...")
        self.stage3_terms = pd.read_csv(stage3_terms_path)
        self.stage4_pairs = pd.read_csv(stage4_pairs_path)

        print(f"      Stage 3 terms: {len(self.stage3_terms)} records")
        print(f"      Stage 4 pairs: {len(self.stage4_pairs)} records")

        # Load front configuration
        print("[3/6] Loading front configuration...")
        with open(front_config_path) as f:
            self.fronts_config = yaml.safe_load(f)

        self.front_names = sorted(self.fronts_config.keys())
        print(f"      Loaded {len(self.front_names)} research fronts")

        # Validate data consistency
        print("[4/6] Validating data consistency...")
        self._validate_data()

        print(f"[5/6] Ready to map {len(self.stage2_sim)} lineages to {len(self.front_names)} fronts")

    def _validate_data(self):
        """Validate that all Stages have consistent lineages and fronts."""
        lineages_p2 = set(self.stage2_sim.index)
        lineages_p3 = set(self.stage3_sim.index)
        lineages_p4 = set(self.stage4_sim.index)

        # Check lineage consistency
        common_lineages = lineages_p2 & lineages_p3 & lineages_p4

        if len(common_lineages) < len(lineages_p2):
            print(f"  Warning: Using {len(common_lineages)} lineages common to all Stages")

        # Check front consistency
        fronts_p2 = set(self.stage2_sim.columns)
        fronts_p3 = set(self.stage3_sim.columns)
        fronts_p4 = set(self.stage4_sim.columns)

        common_fronts = fronts_p2 & fronts_p3 & fronts_p4

        if len(common_fronts) < len(self.front_names):
            print(f"  Warning: Using {len(common_fronts)} fronts common to all Stages")

        print(f"  OK Validated: {len(common_lineages)} lineages, {len(common_fronts)} fronts")

    def get_top_k_fronts(self, scores: dict[str, float], k: int) -> list[tuple[str, float]]:
        """
        Get top-k fronts by score.

        Args:
            scores: Dictionary of {front_name: score}
            k: Number of top fronts to return

        Returns:
            List of (front_name, score) tuples, sorted by score descending
        """
        sorted_fronts = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_fronts[:k]

    def compute_agreement_level(self, votes: list[list[tuple[str, float]]]) -> str:
        """
        Compute agreement level from Stage votes.

        FIXED: Now uses only the top-1 front from each Stage, not all top-k votes.

        Args:
            votes: List of top-k votes from each Stage [(front, score), ...]

        Returns:
            Agreement level: 'high', 'medium', or 'low'
        """
        # Extract only the TOP-1 front from each Stage
        top1_fronts = [vote_list[0][0] if vote_list else None for vote_list in votes]

        # Remove None values (shouldn't happen but be defensive)
        top1_fronts = [f for f in top1_fronts if f is not None]

        if not top1_fronts:
            return 'none'

        # Count how many Stages agree on each front
        vote_counts = Counter(top1_fronts)
        max_count = vote_counts.most_common(1)[0][1]

        # Agreement thresholds based on TOP-1 consensus
        if max_count == 3:  # All 3 Stages agree on same top-1 front
            return 'high'
        elif max_count == 2:  # 2 out of 3 Stages agree on top-1
            return 'medium'
        else:  # All 3 Stages chose different top-1 fronts
            return 'low'

    def get_evidence(
        self,
        lineage_id: int,
        Stage2_votes: list[tuple[str, float]],
        Stage3_votes: list[tuple[str, float]],
        Stage4_votes: list[tuple[str, float]]
    ) -> dict:
        """
        Generate evidence bundle for a lineage mapping.

        Args:
            lineage_id: Lineage ID
            Stage2_votes: Stage 2 top-k votes
            Stage3_votes: Stage 3 top-k votes
            Stage4_votes: Stage 4 top-k votes

        Returns:
            Dictionary with evidence from all Stages
        """
        evidence = {
            'lineage_id': int(lineage_id),
            'Stage2': {
                'top_front': Stage2_votes[0][0] if Stage2_votes else None,
                'score': float(Stage2_votes[0][1]) if Stage2_votes else 0.0,
                'top_3': [(f, float(s)) for f, s in Stage2_votes],
            },
            'Stage3': {
                'top_front': Stage3_votes[0][0] if Stage3_votes else None,
                'score': float(Stage3_votes[0][1]) if Stage3_votes else 0.0,
                'top_3': [(f, float(s)) for f, s in Stage3_votes],
                'matched_terms': self._get_matched_terms(lineage_id, Stage3_votes[0][0] if Stage3_votes else None)
            },
            'Stage4': {
                'top_front': Stage4_votes[0][0] if Stage4_votes else None,
                'score': float(Stage4_votes[0][1]) if Stage4_votes else 0.0,
                'top_3': [(f, float(s)) for f, s in Stage4_votes],
                'top_pairs': self._get_top_pairs(lineage_id, n=5)
            }
        }

        return evidence

    def _get_matched_terms(self, lineage_id: int, front_name: str, n: int = 10) -> list[tuple[str, float]]:
        """Get top matched terms for a lineage-front pair."""
        if front_name is None:
            return []

        # Get lineage terms
        lineage_terms = self.stage3_terms[self.stage3_terms['lineage_id'] == lineage_id]

        if len(lineage_terms) == 0:
            return []

        # Get front vocabulary
        front_config = self.fronts_config.get(front_name, {})
        front_vocab = set()

        # Add canonical terms
        for term in front_config.get('canonical', []):
            front_vocab.update(term.lower().split())

        # Add aliases
        for term in front_config.get('aliases', []):
            front_vocab.update(term.lower().split())

        # Find matching terms
        matches = []
        for _, row in lineage_terms.iterrows():
            term = row['term']
            score = row['ctfidf_score']
            if term in front_vocab:
                matches.append((term, float(score)))

        # Sort by score and return top n
        matches.sort(key=lambda x: x[1], reverse=True)
        return matches[:n]

    def _get_top_pairs(self, lineage_id: int, n: int = 5) -> list[tuple[str, str, float]]:
        """Get top NPMI pairs for a lineage."""
        lineage_pairs = self.stage4_pairs[self.stage4_pairs['lineage_id'] == lineage_id]

        if len(lineage_pairs) == 0:
            return []

        # Sort by NPMI score
        top_pairs = lineage_pairs.nlargest(n, 'npmi_score')

        return [
            (row['term1'], row['term2'], float(row['npmi_score']))
            for _, row in top_pairs.iterrows()
        ]

    def map_lineage(self, lineage_id: int) -> dict:
        """
        Map a single lineage to fronts using weighted ensemble voting.

        FIXED: Now uses weighted voting based on Stage scores, not simple vote counting.

        Args:
            lineage_id: Lineage ID

        Returns:
            Mapping dictionary with fronts, confidence, and evidence
        """
        # Get scores from each Stage
        Stage2_scores = dict(self.stage2_sim.loc[lineage_id])
        if lineage_id in self.stage3_sim.index:
            Stage3_scores = dict(self.stage3_sim.loc[lineage_id])
        else:
            Stage3_scores = dict.fromkeys(self.front_names, 0.0)
        if lineage_id in self.stage4_sim.index:
            Stage4_scores = dict(self.stage4_sim.loc[lineage_id])
        else:
            Stage4_scores = dict.fromkeys(self.front_names, 0.0)

        # Get top-k votes from each Stage
        Stage2_votes = self.get_top_k_fronts(Stage2_scores, self.top_k)
        Stage3_votes = self.get_top_k_fronts(Stage3_scores, self.top_k)
        Stage4_votes = self.get_top_k_fronts(Stage4_scores, self.top_k)

        # Compute agreement level (based on top-1 consensus)
        agreement = self.compute_agreement_level([Stage2_votes, Stage3_votes, Stage4_votes])

        # Compute weighted scores for all candidate fronts
        # TUNED WEIGHTS: Based on statistical calibration (see docs/Stage_WEIGHT_TUNING_ANALYSIS.md)
        # Stage 2: Good discrimination, provides diversity → 1.1
        # Stage 3: Weak until aliases improved → 0.5
        # Stage 4: Strongest signal, highest scores → 1.4
        Stage_weights = {'Stage2': 1.1, 'Stage3': 0.5, 'Stage4': 1.4}

        weighted_scores = {}
        for front, score in Stage2_votes:
            weighted_scores[front] = weighted_scores.get(front, 0.0) + Stage_weights['Stage2'] * score
        for front, score in Stage3_votes:
            weighted_scores[front] = weighted_scores.get(front, 0.0) + Stage_weights['Stage3'] * score
        for front, score in Stage4_votes:
            weighted_scores[front] = weighted_scores.get(front, 0.0) + Stage_weights['Stage4'] * score

        # Sort fronts by weighted score
        ranked_fronts = sorted(weighted_scores.items(), key=lambda x: x[1], reverse=True)

        # Determine front assignments based on agreement level
        if agreement == 'high':
            # All 3 Stages agree on top-1 front - use unanimous choice
            primary_front = Stage2_votes[0][0]  # They all agree, so pick any top-1

            # FIX: Check if scores are meaningful (avoid zero-score high confidence)
            min_scores = [Stage2_votes[0][1], Stage3_votes[0][1], Stage4_votes[0][1]]
            if any(s < 0.05 for s in min_scores):
                # Downgrade to medium if any Stage has weak score
                agreement = 'medium'
                alternative_fronts = [ranked_fronts[1][0]] if len(ranked_fronts) > 1 else []
                review_needed = True
            else:
                alternative_fronts = []
                review_needed = False

        elif agreement == 'medium':
            # 2/3 Stages agree - use weighted ranking for primary and alternatives
            primary_front = ranked_fronts[0][0]
            alternative_fronts = [ranked_fronts[1][0]] if len(ranked_fronts) > 1 else []
            review_needed = True

        else:
            # No agreement - use weighted ranking, flag for review
            primary_front = ranked_fronts[0][0]
            alternative_fronts = [f for f, _ in ranked_fronts[1:3]] if len(ranked_fronts) > 1 else []
            review_needed = True

        # Generate evidence bundle
        evidence = self.get_evidence(lineage_id, Stage2_votes, Stage3_votes, Stage4_votes)

        return {
            'lineage_id': int(lineage_id),
            'primary_front': primary_front,
            'alternative_fronts': '|'.join(alternative_fronts) if alternative_fronts else None,
            'confidence': agreement,
            'review_needed': review_needed,
            'Stage2_top': Stage2_votes[0][0] if Stage2_votes else None,
            'Stage2_score': float(Stage2_votes[0][1]) if Stage2_votes else 0.0,
            'Stage3_top': Stage3_votes[0][0] if Stage3_votes else None,
            'Stage3_score': float(Stage3_votes[0][1]) if Stage3_votes else 0.0,
            'Stage4_top': Stage4_votes[0][0] if Stage4_votes else None,
            'Stage4_score': float(Stage4_votes[0][1]) if Stage4_votes else 0.0,
            'evidence': evidence
        }

    def map_all_lineages(self) -> tuple[pd.DataFrame, dict[int, dict]]:
        """
        Map all lineages to fronts.

        Returns:
            Tuple of (mappings_df, evidence_dict)
        """
        print("\n[6/6] Mapping lineages to fronts using ensemble voting...")

        mappings = []
        evidence_dict = {}

        for lineage_id in tqdm(self.stage2_sim.index, desc="Mapping lineages"):
            mapping = self.map_lineage(lineage_id)
            evidence_dict[lineage_id] = mapping.pop('evidence')
            mappings.append(mapping)

        df = pd.DataFrame(mappings)
        return df, evidence_dict

    def generate_summary_report(self, mappings_df: pd.DataFrame) -> str:
        """
        Generate human-readable summary report.

        Args:
            mappings_df: Mappings DataFrame

        Returns:
            Markdown-formatted report
        """
        total = len(mappings_df)
        high_conf = len(mappings_df[mappings_df['confidence'] == 'high'])
        medium_conf = len(mappings_df[mappings_df['confidence'] == 'medium'])
        low_conf = len(mappings_df[mappings_df['confidence'] == 'low'])
        review_needed = len(mappings_df[mappings_df['review_needed']])
        multi_label = len(mappings_df[mappings_df['alternative_fronts'].notna()])

        # Front distribution
        front_counts = mappings_df['primary_front'].value_counts()

        # Agreement statistics
        Stage_agreement = []
        for _, row in mappings_df.iterrows():
            Stages = [row['Stage2_top'], row['Stage3_top'], row['Stage4_top']]
            unique = len(set(Stages))
            if unique == 1:
                Stage_agreement.append('unanimous')
            elif unique == 2:
                Stage_agreement.append('2/3 agree')
            else:
                Stage_agreement.append('no agreement')

        agreement_counts = Counter(Stage_agreement)

        report = f"""# Stage 5: Lineage-to-Front Mapping Summary

## Overall Statistics

- **Total lineages mapped**: {total}
- **High confidence** (all 3 Stages agree): {high_conf} ({100*high_conf/total:.1f}%)
- **Medium confidence** (2/3 Stages agree): {medium_conf} ({100*medium_conf/total:.1f}%)
- **Low confidence** (no agreement): {low_conf} ({100*low_conf/total:.1f}%)
- **Multi-label assignments**: {multi_label} ({100*multi_label/total:.1f}%)
- **Flagged for review**: {review_needed} ({100*review_needed/total:.1f}%)

## Stage Agreement Analysis

- **Unanimous** (all 3 Stages choose same top front): {agreement_counts.get('unanimous', 0)} ({100*agreement_counts.get('unanimous', 0)/total:.1f}%)
- **2/3 Agreement**: {agreement_counts.get('2/3 agree', 0)} ({100*agreement_counts.get('2/3 agree', 0)/total:.1f}%)
- **No Agreement**: {agreement_counts.get('no agreement', 0)} ({100*agreement_counts.get('no agreement', 0)/total:.1f}%)

## Front Distribution (Primary Assignments)

| Front | Count | Percentage |
|-------|-------|------------|
"""

        for front, count in front_counts.items():
            report += f"| {front} | {count} | {100*count/total:.1f}% |\n"

        report += f"""
## Score Distributions

- **Stage 2 (SciBERT)**: Mean={mappings_df['Stage2_score'].mean():.3f}, Std={mappings_df['Stage2_score'].std():.3f}
- **Stage 3 (c-TF-IDF)**: Mean={mappings_df['Stage3_score'].mean():.3f}, Std={mappings_df['Stage3_score'].std():.3f}
- **Stage 4 (NPMI)**: Mean={mappings_df['Stage4_score'].mean():.3f}, Std={mappings_df['Stage4_score'].std():.3f}

## Recommendations

### High Confidence Mappings ({high_conf} lineages)
These mappings have unanimous agreement from all 3 Stages. Use directly without review.

### Medium Confidence Mappings ({medium_conf} lineages)
These have 2/3 Stage agreement. Consider multi-label or verify top assignment.

### Low Confidence Mappings ({low_conf} lineages)
No Stage agreement. **Priority for human review.** Check evidence bundles to determine:
- Is this truly multi-label (spans multiple fronts)?
- Is one Stage clearly wrong?
- Is this an edge case or new phenomenon?

## Next Steps

1. **Review flagged lineages** ({review_needed} cases) - see `data/out/mapping_evidence/` for details
2. **Validate on labeled dataset** - measure precision/recall per confidence level
3. **Tune if needed** - if accuracy insufficient, adjust Stage weights
4. **Use high-confidence mappings** - {high_conf} lineages ready for analysis

---

**Generated**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
**Input**: {total} lineages, {len(self.front_names)} research fronts
**Method**: Ensemble voting (top-3 from each Stage, majority vote)
"""

        return report


def run_ensemble(
    stage2_similarity_path: Path = Path('data/out/03_milestone_mapping/lineage_front_similarity.csv'),
    stage3_similarity_path: Path = Path('data/out/03_milestone_mapping/lineage_front_term_similarity.csv'),
    stage4_similarity_path: Path = Path('data/out/03_milestone_mapping/lineage_front_npmi_similarity.csv'),
    stage3_terms_path: Path = Path('data/out/02_lineage_tracking/lineage_ctfidf_terms.csv'),
    stage4_pairs_path: Path = Path('data/out/02_lineage_tracking/lineage_npmi_pairs.csv'),
    front_config_path: Path = Path('config/front_aliases.yaml'),
    top_k: int = 3,
    store=None,  # Optional shared store (unused in Stage 5, but kept for consistency)
    validate: bool = True,  # Run validation checks and generate reports
    output_root: Path = Path('data/out')  # Base directory for outputs
) -> tuple[pd.DataFrame, dict, dict]:
    """
    Run Stage 5 ensemble mapping.

    Stage 5 doesn't use the shared store currently (only loads similarity matrices),
    but accepts it as a parameter for future extensibility.

    Args:
        stage2_similarity_path: Path to Stage 2 similarity matrix
        stage3_similarity_path: Path to Stage 3 similarity matrix
        stage4_similarity_path: Path to Stage 4 similarity matrix
        stage3_terms_path: Path to Stage 3 terms CSV
        stage4_pairs_path: Path to Stage 4 pairs CSV
        front_config_path: Path to front configuration YAML
        top_k: Number of top fronts each Stage votes for
        store: Optional LineageTextStore (unused in Stage 5)
        validate: Run validation checks and generate reports (default: True)

    Returns:
        Tuple of (mappings_df, evidence_dict)
    """

    print("=" * 70)
    print("Stage 5: ENSEMBLE LINEAGE-TO-FRONT MAPPING")
    print("=" * 70)

    # Initialize mapper
    mapper = EnsembleFrontMapper(
        stage2_similarity_path=stage2_similarity_path,
        stage3_similarity_path=stage3_similarity_path,
        stage4_similarity_path=stage4_similarity_path,
        stage3_terms_path=stage3_terms_path,
        stage4_pairs_path=stage4_pairs_path,
        front_config_path=front_config_path,
        top_k=top_k
    )

    # Map all lineages
    mappings_df, evidence_dict = mapper.map_all_lineages()

    # Set up output directories
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    output_dir_mapping = output_root / '03_milestone_mapping'
    output_dir_mapping.mkdir(parents=True, exist_ok=True)

    print("\n[Output 1/3] Saving lineage-front mappings...")
    mappings_path = output_dir_mapping / 'lineage_front_mappings.csv'
    mappings_df.to_csv(mappings_path, index=False)
    print(f"  Wrote {len(mappings_df)} mappings to {mappings_path}")

    # Save evidence bundles
    print("\n[Output 2/3] Saving evidence bundles...")
    evidence_dir = output_root / 'mapping_evidence'
    evidence_dir.mkdir(parents=True, exist_ok=True)

    for lineage_id, evidence in tqdm(evidence_dict.items(), desc="Writing evidence"):
        evidence_path = evidence_dir / f'lineage_{lineage_id}_evidence.json'
        with open(evidence_path, 'w') as f:
            json.dump(evidence, f, indent=2)

    print(f"  Wrote {len(evidence_dict)} evidence bundles to {evidence_dir}/")

    # Generate summary report
    print("\n[Output 3/3] Generating summary report...")
    report = mapper.generate_summary_report(mappings_df)
    report_path = output_root / 'phase5_summary.md'
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"  Wrote summary report to {report_path}")

    # Print summary statistics
    print("\n" + "=" * 70)
    print("Stage 5 COMPLETE")
    print("=" * 70)
    print(f"Total lineages: {len(mappings_df)}")
    print(f"High confidence: {len(mappings_df[mappings_df['confidence']=='high'])} ({100*len(mappings_df[mappings_df['confidence']=='high'])/len(mappings_df):.1f}%)")
    print(f"Medium confidence: {len(mappings_df[mappings_df['confidence']=='medium'])} ({100*len(mappings_df[mappings_df['confidence']=='medium'])/len(mappings_df):.1f}%)")
    print(f"Low confidence: {len(mappings_df[mappings_df['confidence']=='low'])} ({100*len(mappings_df[mappings_df['confidence']=='low'])/len(mappings_df):.1f}%)")
    print(f"Review needed: {len(mappings_df[mappings_df['review_needed']])}")
    print("\nOutputs:")
    print(f"  - {mappings_path}")
    print(f"  - {evidence_dir}/")
    print(f"  - {report_path}")

    # Run validation if requested
    if validate:
        print(f"\n{'='*70}")
        print("Stage 5 VALIDATION")
        print(f"{'='*70}\n")

        validation_results = run_Stage5_validation(mappings_df, evidence_dict)
    else:
        validation_results = None

    return mappings_df, evidence_dict, validation_results


# ============================================================================
# VALIDATION FUNCTIONS (integrated from validate_stage5.py)
# ============================================================================

def run_Stage5_validation(mappings_df: pd.DataFrame, evidence_dict: dict) -> dict:
    """
    Run Stage 5 validation checks and generate outputs.

    Args:
        mappings_df: DataFrame with final lineage-front mappings
        evidence_dict: Dictionary with evidence for each lineage

    Returns:
        Dictionary with validation results
    """
    # Lazy imports to avoid overhead when validation disabled

    # Create output directory
    output_dir = Path('data/out/06_validation/Stage5')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Compute validation statistics
    print("[1/4] Computing validation statistics...")
    checks = _validate_stage5_statistics(mappings_df, evidence_dict)

    # Generate visualizations
    print("[2/4] Generating confidence distribution...")
    _generate_Stage5_confidence_viz(mappings_df, output_dir / 'Stage5_confidence_distribution.png')

    print("[3/4] Generating front distribution...")
    _generate_Stage5_front_distribution(mappings_df, output_dir / 'Stage5_front_distribution.png')

    # Generate report
    print("[4/4] Generating validation report...")
    _generate_Stage5_report(checks, mappings_df, output_dir / 'Stage5_validation_report.md')

    # Save JSON results
    validation_path = output_dir / 'Stage5_validation_results.json'
    with open(validation_path, 'w') as f:
        json.dump(checks, f, indent=2)

    print(f"\n[Validation] Complete! Results saved to {output_dir}/")

    return checks


def _validate_stage5_statistics(mappings_df: pd.DataFrame, evidence_dict: dict) -> dict:
    """Compute validation statistics for Stage 5 outputs."""
    checks = {}

    # Ensure expected secondary label column exists (backward compatibility)
    if "secondary_front" not in mappings_df.columns:
        mappings_df = mappings_df.copy()
        if "alternative_fronts" in mappings_df.columns:
            mappings_df["secondary_front"] = mappings_df["alternative_fronts"].apply(
                lambda val: val.split("|")[0].strip()
                if isinstance(val, str) and val.strip()
                else pd.NA
            )
        else:
            mappings_df["secondary_front"] = pd.NA

    # Basic counts
    checks['total_lineages'] = int(len(mappings_df))
    checks['unique_fronts'] = int(mappings_df['primary_front'].nunique())

    # Confidence distribution
    confidence_counts = mappings_df['confidence'].value_counts().to_dict()
    checks['confidence_distribution'] = {k: int(v) for k, v in confidence_counts.items()}
    checks['high_confidence_pct'] = float(100 * confidence_counts.get('high', 0) / len(mappings_df))
    checks['medium_confidence_pct'] = float(100 * confidence_counts.get('medium', 0) / len(mappings_df))
    checks['low_confidence_pct'] = float(100 * confidence_counts.get('low', 0) / len(mappings_df))

    # Review status
    review_counts = mappings_df['review_needed'].value_counts().to_dict()
    checks['review_needed_count'] = int(review_counts.get(True, 0))
    checks['ready_to_use_count'] = int(review_counts.get(False, 0))
    checks['review_needed_pct'] = float(100 * checks['review_needed_count'] / len(mappings_df))

    # Front distribution
    front_counts = mappings_df['primary_front'].value_counts().to_dict()
    checks['front_distribution'] = {k: int(v) for k, v in front_counts.items()}

    # Multi-label analysis
    multi_label = mappings_df[mappings_df['secondary_front'].notna()]
    checks['multi_label_count'] = int(len(multi_label))
    checks['multi_label_pct'] = float(100 * len(multi_label) / len(mappings_df))

    return checks


def _generate_Stage5_confidence_viz(mappings_df: pd.DataFrame, output_path: Path):
    """Generate confidence distribution visualization."""
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Confidence distribution
    confidence_counts = mappings_df['confidence'].value_counts()
    colors = {'high': '#2ecc71', 'medium': '#f39c12', 'low': '#e74c3c', 'none': '#95a5a6'}

    ax1.bar(confidence_counts.index, confidence_counts.values,
            color=[colors.get(c, '#95a5a6') for c in confidence_counts.index])
    ax1.set_xlabel('Confidence Level', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Number of Lineages', fontsize=12, fontweight='bold')
    ax1.set_title('Stage 5: Confidence Distribution', fontsize=14, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)

    # Add counts on bars
    for i, (_conf, count) in enumerate(confidence_counts.items()):
        ax1.text(i, count + 1, f'{count}\n({100*count/len(mappings_df):.1f}%)',
                ha='center', va='bottom', fontweight='bold')

    # Review needed vs not needed
    review_counts = mappings_df['review_needed'].value_counts()
    colors_review = {True: '#e74c3c', False: '#2ecc71'}
    labels_review = {True: 'Review Needed', False: 'Ready to Use'}

    ax2.bar([labels_review[k] for k in review_counts.index], review_counts.values,
            color=[colors_review[k] for k in review_counts.index])
    ax2.set_xlabel('Review Status', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Number of Lineages', fontsize=12, fontweight='bold')
    ax2.set_title('Stage 5: Review Status', fontsize=14, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)

    # Add counts on bars
    for i, (status, count) in enumerate(review_counts.items()):
        labels_review[status]
        ax2.text(i, count + 1, f'{count}\n({100*count/len(mappings_df):.1f}%)',
                ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def _generate_Stage5_front_distribution(mappings_df: pd.DataFrame, output_path: Path):
    """Generate front assignment distribution."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(14, 8))

    front_counts = mappings_df['primary_front'].value_counts()
    colors = plt.cm.Set3(np.linspace(0, 1, len(front_counts)))

    ax.barh(range(len(front_counts)), front_counts.values, color=colors)
    ax.set_yticks(range(len(front_counts)))
    ax.set_yticklabels(front_counts.index)
    ax.set_xlabel('Number of Lineages', fontsize=12, fontweight='bold')
    ax.set_title('Stage 5: Primary Front Assignments', fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)

    # Add counts and percentages
    for i, (_front, count) in enumerate(front_counts.items()):
        ax.text(count + 0.5, i, f'{count} ({100*count/len(mappings_df):.1f}%)',
                va='center', fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def _generate_Stage5_report(checks: dict, mappings_df: pd.DataFrame, output_path: Path):
    """Generate markdown validation report."""
    report = f"""# Stage 5 Validation Report
**Ensemble Voting & Final Mapping Assignment**

## Summary

- **Total Lineages Mapped**: {checks['total_lineages']}
- **Unique Research Fronts**: {checks['unique_fronts']}
- **Multi-label Assignments**: {checks['multi_label_count']} ({checks['multi_label_pct']:.1f}%)

## Confidence Distribution

| Confidence Level | Count | Percentage |
|-----------------|-------|------------|
| High | {checks['confidence_distribution'].get('high', 0)} | {checks['high_confidence_pct']:.1f}% |
| Medium | {checks['confidence_distribution'].get('medium', 0)} | {checks['medium_confidence_pct']:.1f}% |
| Low | {checks['confidence_distribution'].get('low', 0)} | {checks['low_confidence_pct']:.1f}% |

## Review Status

| Status | Count | Percentage |
|--------|-------|------------|
| Ready to Use | {checks['ready_to_use_count']} | {100 - checks['review_needed_pct']:.1f}% |
| Review Needed | {checks['review_needed_count']} | {checks['review_needed_pct']:.1f}% |

## Front Distribution

| Research Front | Lineages Assigned |
|----------------|-------------------|
"""

    # Add front distribution
    for front, count in sorted(checks['front_distribution'].items(), key=lambda x: x[1], reverse=True):
        pct = 100 * count / checks['total_lineages']
        report += f"| {front} | {count} ({pct:.1f}%) |\n"

    report += f"""

## Overall Assessment

The ensemble voting process successfully mapped **{checks['total_lineages']} lineages** to **{checks['unique_fronts']} research fronts**.

- **{checks['high_confidence_pct']:.1f}%** of mappings have high confidence (all 3 Stages agree)
- **{checks['medium_confidence_pct']:.1f}%** have medium confidence (2/3 Stages agree)
- **{checks['low_confidence_pct']:.1f}%** have low confidence (split votes)
- **{checks['review_needed_pct']:.1f}%** require human review

The distribution suggests {'a well-balanced' if checks['review_needed_pct'] < 20 else 'some challenges in'} mapping quality across the dataset.
"""

    output_path.write_text(report, encoding='utf-8')


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Stage 5: Ensemble lineage-to-front mapping")
    parser.add_argument(
        '--Stage2-similarity',
        type=Path,
        default=Path('data/out/03_milestone_mapping/lineage_front_similarity.csv'),
        help='Path to Stage 2 similarity matrix'
    )
    parser.add_argument(
        '--Stage3-similarity',
        type=Path,
        default=Path('data/out/03_milestone_mapping/lineage_front_term_similarity.csv'),
        help='Path to Stage 3 similarity matrix'
    )
    parser.add_argument(
        '--Stage4-similarity',
        type=Path,
        default=Path('data/out/03_milestone_mapping/lineage_front_npmi_similarity.csv'),
        help='Path to Stage 4 similarity matrix'
    )
    parser.add_argument(
        '--Stage3-terms',
        type=Path,
        default=Path('data/out/02_lineage_tracking/lineage_ctfidf_terms.csv'),
        help='Path to Stage 3 terms'
    )
    parser.add_argument(
        '--Stage4-pairs',
        type=Path,
        default=Path('data/out/02_lineage_tracking/lineage_npmi_pairs.csv'),
        help='Path to Stage 4 pairs'
    )
    parser.add_argument(
        '--fronts',
        type=Path,
        default=Path('config/front_aliases.yaml'),
        help='Path to front configuration'
    )
    parser.add_argument(
        '--top-k',
        type=int,
        default=3,
        help='Number of top fronts each Stage votes for (default: 3)'
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
        default=Path('data/out'),
        help='Base directory for outputs (default: data/out)'
    )

    args = parser.parse_args()

    # Call run_ensemble() with standalone mode
    run_ensemble(
        stage2_similarity_path=args.stage2_similarity,
        stage3_similarity_path=args.stage3_similarity,
        stage4_similarity_path=args.stage4_similarity,
        stage3_terms_path=args.stage3_terms,
        stage4_pairs_path=args.stage4_pairs,
        front_config_path=args.fronts,
        top_k=args.top_k,
        store=None,  # Standalone mode
        validate=args.validate,  # Pass validate flag
        output_root=args.output_root
    )


if __name__ == '__main__':
    main()



