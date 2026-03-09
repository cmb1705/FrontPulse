#!/usr/bin/env python3
"""
Stage Weight Calibration - Statistical Analysis

Analyzes internal consistency metrics to recommend stage weights WITHOUT requiring expert labels.

Key metrics:
1. Discrimination power (top-2 margins, score spread)
2. Agreement patterns (stage concordance)
3. Uncertainty quantification (entropy, flat scores)
4. Calibration quality (score-to-confidence mapping)
5. Stability analysis (bootstrap resampling)

Usage:
    python scripts/calibrate_stage_weights.py
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import entropy
from collections import Counter
import matplotlib.pyplot as plt


class StageWeightCalibrator:
    """Calibrate ensemble stage weights using internal consistency metrics."""

    def __init__(self):
        """Initialize calibrator with Stage 2-4 similarity matrices."""
        self.stage2_sim = pd.read_csv('data/out/03_milestone_mapping/lineage_front_similarity.csv', index_col='lineage_id')
        self.stage3_sim = pd.read_csv('data/out/03_milestone_mapping/lineage_front_term_similarity.csv', index_col='lineage_id')
        self.stage4_sim = pd.read_csv('data/out/03_milestone_mapping/lineage_front_npmi_similarity.csv', index_col='lineage_id')

        self.lineage_ids = self.stage2_sim.index.tolist()
        self.front_names = self.stage2_sim.columns.tolist()

        print(f"Loaded {len(self.lineage_ids)} lineages × {len(self.front_names)} fronts")

    def measure_discrimination_power(self) -> dict:
        """
        Measure how well each stage discriminates between fronts.

        Higher discrimination = clearer winner, better for weighting.
        """
        results = {}

        for stage_name, sim_matrix in [('stage2', self.stage2_sim),
                                        ('stage3', self.stage3_sim),
                                        ('stage4', self.stage4_sim)]:

            margins = []
            max_scores = []
            entropies = []

            for lineage_id in self.lineage_ids:
                scores = sim_matrix.loc[lineage_id].values

                # Top-2 margin
                top2 = np.sort(scores)[-2:]
                margin = top2[1] - top2[0]
                margins.append(margin)

                # Max score (confidence indicator)
                max_scores.append(scores.max())

                # Entropy (lower = more decisive)
                # Normalize scores to [0, 1] for entropy calculation
                scores_norm = (scores - scores.min()) / (scores.max() - scores.min() + 1e-10)
                scores_norm = scores_norm / (scores_norm.sum() + 1e-10)
                entropies.append(entropy(scores_norm))

            results[stage_name] = {
                'avg_margin': np.mean(margins),
                'std_margin': np.std(margins),
                'avg_max_score': np.mean(max_scores),
                'std_max_score': np.std(max_scores),
                'avg_entropy': np.mean(entropies),
                'std_entropy': np.std(entropies),
                'clear_winners': sum(1 for m in margins if m > 0.1) / len(margins)
            }

        return results

    def measure_stage_agreement(self) -> dict:
        """
        Measure how often stages agree on top-1 fronts.

        High pairwise agreement suggests stages are measuring similar things.
        """
        # Get top-1 front for each stage-lineage
        stage2_top1 = {lid: self.stage2_sim.loc[lid].idxmax() for lid in self.lineage_ids}
        stage3_top1 = {lid: self.stage3_sim.loc[lid].idxmax() for lid in self.lineage_ids}
        stage4_top1 = {lid: self.stage4_sim.loc[lid].idxmax() for lid in self.lineage_ids}

        # Compute pairwise agreement rates
        agreement_12 = sum(1 for lid in self.lineage_ids if stage2_top1[lid] == stage3_top1[lid]) / len(self.lineage_ids)
        agreement_13 = sum(1 for lid in self.lineage_ids if stage2_top1[lid] == stage4_top1[lid]) / len(self.lineage_ids)
        agreement_23 = sum(1 for lid in self.lineage_ids if stage3_top1[lid] == stage4_top1[lid]) / len(self.lineage_ids)

        # Count unanimous agreement
        unanimous = sum(1 for lid in self.lineage_ids
                       if stage2_top1[lid] == stage3_top1[lid] == stage4_top1[lid])
        unanimous_rate = unanimous / len(self.lineage_ids)

        # Identify "tie-breaker" stage
        # When two stages agree but third disagrees, which one is the odd one out?
        stage2_odd = sum(1 for lid in self.lineage_ids
                        if stage3_top1[lid] == stage4_top1[lid] != stage2_top1[lid])
        stage3_odd = sum(1 for lid in self.lineage_ids
                        if stage2_top1[lid] == stage4_top1[lid] != stage3_top1[lid])
        stage4_odd = sum(1 for lid in self.lineage_ids
                        if stage2_top1[lid] == stage3_top1[lid] != stage4_top1[lid])

        return {
            'stage2_stage3_agreement': agreement_12,
            'stage2_stage4_agreement': agreement_13,
            'stage3_stage4_agreement': agreement_23,
            'unanimous_agreement_rate': unanimous_rate,
            'stage2_odd_out_count': stage2_odd,
            'stage3_odd_out_count': stage3_odd,
            'stage4_odd_out_count': stage4_odd,
            'top1_distributions': {
                'stage2': Counter(stage2_top1.values()),
                'stage3': Counter(stage3_top1.values()),
                'stage4': Counter(stage4_top1.values())
            }
        }

    def measure_calibration_quality(self) -> dict:
        """
        Measure whether high scores truly indicate high confidence.

        We can't validate against ground truth, but we can check internal consistency:
        - Do high scores from Phase X correlate with high scores from Phase Y?
        - Does a 0.9 score mean the same thing across stages?
        """
        results = {}

        # Get top-1 scores for each stage
        stage2_top_scores = [self.stage2_sim.loc[lid].max() for lid in self.lineage_ids]
        stage3_top_scores = [self.stage3_sim.loc[lid].max() for lid in self.lineage_ids]
        stage4_top_scores = [self.stage4_sim.loc[lid].max() for lid in self.lineage_ids]

        # Compute correlation between stage top scores
        from scipy.stats import pearsonr, spearmanr

        corr_12_pearson, p_12 = pearsonr(stage2_top_scores, stage3_top_scores)
        corr_13_pearson, p_13 = pearsonr(stage2_top_scores, stage4_top_scores)
        corr_23_pearson, p_23 = pearsonr(stage3_top_scores, stage4_top_scores)

        corr_12_spearman, _ = spearmanr(stage2_top_scores, stage3_top_scores)
        corr_13_spearman, _ = spearmanr(stage2_top_scores, stage4_top_scores)
        corr_23_spearman, _ = spearmanr(stage3_top_scores, stage4_top_scores)

        results['score_correlations'] = {
            'stage2_stage3_pearson': corr_12_pearson,
            'stage2_stage4_pearson': corr_13_pearson,
            'stage3_stage4_pearson': corr_23_pearson,
            'stage2_stage3_spearman': corr_12_spearman,
            'stage2_stage4_spearman': corr_13_spearman,
            'stage3_stage4_spearman': corr_23_spearman,
        }

        # Score distribution statistics
        results['score_distributions'] = {
            'stage2': {
                'mean': np.mean(stage2_top_scores),
                'std': np.std(stage2_top_scores),
                'min': np.min(stage2_top_scores),
                'max': np.max(stage2_top_scores),
                'q25': np.percentile(stage2_top_scores, 25),
                'q50': np.percentile(stage2_top_scores, 50),
                'q75': np.percentile(stage2_top_scores, 75)
            },
            'stage3': {
                'mean': np.mean(stage3_top_scores),
                'std': np.std(stage3_top_scores),
                'min': np.min(stage3_top_scores),
                'max': np.max(stage3_top_scores),
                'q25': np.percentile(stage3_top_scores, 25),
                'q50': np.percentile(stage3_top_scores, 50),
                'q75': np.percentile(stage3_top_scores, 75)
            },
            'stage4': {
                'mean': np.mean(stage4_top_scores),
                'std': np.std(stage4_top_scores),
                'min': np.min(stage4_top_scores),
                'max': np.max(stage4_top_scores),
                'q25': np.percentile(stage4_top_scores, 25),
                'q50': np.percentile(stage4_top_scores, 50),
                'q75': np.percentile(stage4_top_scores, 75)
            }
        }

        return results

    def compute_optimal_weights(self, discrimination: dict, agreement: dict, calibration: dict) -> dict:
        """
        Compute recommended stage weights based on statistical analysis.

        Weighting heuristics:
        1. Higher discrimination power → higher weight (clearer signal)
        2. Lower entropy → higher weight (more decisive)
        3. Being "odd one out" often → lower weight (less agreement with others)
        4. Higher score variability → potential for higher weight (more informative)

        Returns:
            dict: Recommended weights with reasoning
        """
        # Normalize discrimination metrics to [0, 1]
        stages = ['stage2', 'stage3', 'stage4']

        # 1. Discrimination scores (avg_margin × clear_winners)
        disc_scores = {p: discrimination[p]['avg_margin'] * discrimination[p]['clear_winners']
                      for p in stages}
        max_disc = max(disc_scores.values())
        disc_scores_norm = {p: disc_scores[p] / max_disc for p in stages}

        # 2. Decisiveness scores (inverse of entropy)
        decisiveness = {p: 1.0 / (discrimination[p]['avg_entropy'] + 0.1) for p in stages}
        max_decisiveness = max(decisiveness.values())
        decisiveness_norm = {p: decisiveness[p] / max_decisiveness for p in stages}

        # 3. Agreement penalty (being odd-out reduces weight)
        total_odd = sum([agreement['stage2_odd_out_count'],
                        agreement['stage3_odd_out_count'],
                        agreement['stage4_odd_out_count']])

        if total_odd > 0:
            agreement_penalty = {
                'stage2': 1.0 - (agreement['stage2_odd_out_count'] / total_odd),
                'stage3': 1.0 - (agreement['stage3_odd_out_count'] / total_odd),
                'stage4': 1.0 - (agreement['stage4_odd_out_count'] / total_odd)
            }
        else:
            agreement_penalty = {'stage2': 1.0, 'stage3': 1.0, 'stage4': 1.0}

        # 4. Score variability (higher std = more informative)
        variability = {p: calibration['score_distributions'][p]['std'] for p in stages}
        max_var = max(variability.values())
        variability_norm = {p: variability[p] / max_var for p in stages}

        # Combined score (equal weighting of factors)
        combined = {
            p: (disc_scores_norm[p] * 0.35 +
                decisiveness_norm[p] * 0.25 +
                agreement_penalty[p] * 0.20 +
                variability_norm[p] * 0.20)
            for p in stages
        }

        # Normalize to sum to 3.0 (for comparison with equal weights of 1.0 each)
        total = sum(combined.values())
        weights = {p: (combined[p] / total) * 3.0 for p in stages}

        return {
            'recommended_weights': weights,
            'components': {
                'discrimination': disc_scores_norm,
                'decisiveness': decisiveness_norm,
                'agreement_penalty': agreement_penalty,
                'variability': variability_norm,
                'combined': combined
            },
            'reasoning': {
                'stage2': self._explain_weight(weights['stage2'], discrimination['stage2'],
                                              agreement['stage2_odd_out_count']),
                'stage3': self._explain_weight(weights['stage3'], discrimination['stage3'],
                                              agreement['stage3_odd_out_count']),
                'stage4': self._explain_weight(weights['stage4'], discrimination['stage4'],
                                              agreement['stage4_odd_out_count'])
            }
        }

    def _explain_weight(self, weight: float, disc: dict, odd_count: int) -> str:
        """Generate human-readable explanation for weight."""
        if weight > 1.1:
            strength = "STRONG"
        elif weight > 0.95:
            strength = "MODERATE"
        else:
            strength = "WEAK"

        reasons = []
        if disc['avg_margin'] > 0.1:
            reasons.append("high discrimination")
        if disc['clear_winners'] > 0.5:
            reasons.append("clear winners")
        if disc['avg_entropy'] < 2.0:
            reasons.append("low entropy")
        if odd_count < 10:
            reasons.append("good agreement")

        return f"{strength} signal ({', '.join(reasons)})"

    def generate_report(self) -> str:
        """Generate comprehensive calibration report."""
        print("\n" + "="*70)
        print("PHASE WEIGHT CALIBRATION ANALYSIS")
        print("="*70)

        # Run analyses
        print("\n[1/3] Measuring discrimination power...")
        discrimination = self.measure_discrimination_power()

        print("[2/3] Measuring stage agreement...")
        agreement = self.measure_stage_agreement()

        print("[3/3] Measuring calibration quality...")
        calibration = self.measure_calibration_quality()

        print("\n" + "="*70)
        print("RESULTS")
        print("="*70)

        # Discrimination results
        print("\n## Discrimination Power")
        print("-" * 70)
        for stage in ['stage2', 'stage3', 'stage4']:
            d = discrimination[stage]
            print(f"\n{stage.upper()}:")
            print(f"  Avg top-2 margin:  {d['avg_margin']:.4f} ± {d['std_margin']:.4f}")
            print(f"  Avg max score:     {d['avg_max_score']:.4f} ± {d['std_max_score']:.4f}")
            print(f"  Avg entropy:       {d['avg_entropy']:.4f} ± {d['std_entropy']:.4f}")
            print(f"  Clear winners:     {d['clear_winners']:.1%} (margin > 0.1)")

        # Agreement results
        print("\n## Phase Agreement")
        print("-" * 70)
        print(f"  Phase2-Phase3 agreement: {agreement['stage2_stage3_agreement']:.1%}")
        print(f"  Phase2-Phase4 agreement: {agreement['stage2_stage4_agreement']:.1%}")
        print(f"  Phase3-Phase4 agreement: {agreement['stage3_stage4_agreement']:.1%}")
        print(f"  Unanimous (all 3):       {agreement['unanimous_agreement_rate']:.1%}")
        print(f"\n  Odd-one-out counts:")
        print(f"    Phase2: {agreement['stage2_odd_out_count']} times")
        print(f"    Phase3: {agreement['stage3_odd_out_count']} times")
        print(f"    Phase4: {agreement['stage4_odd_out_count']} times")

        # Calibration results
        print("\n## Calibration Quality")
        print("-" * 70)
        print("  Score correlations (Pearson):")
        corr = calibration['score_correlations']
        print(f"    Phase2-Phase3: {corr['stage2_stage3_pearson']:.3f}")
        print(f"    Phase2-Phase4: {corr['stage2_stage4_pearson']:.3f}")
        print(f"    Phase3-Phase4: {corr['stage3_stage4_pearson']:.3f}")

        print("\n  Top-1 score distributions:")
        for stage in ['stage2', 'stage3', 'stage4']:
            s = calibration['score_distributions'][stage]
            print(f"    {stage}: mean={s['mean']:.3f}, std={s['std']:.3f}, range=[{s['min']:.3f}, {s['max']:.3f}]")

        # Compute optimal weights
        print("\n" + "="*70)
        print("RECOMMENDED WEIGHTS")
        print("="*70)

        weights_result = self.compute_optimal_weights(discrimination, agreement, calibration)
        weights = weights_result['recommended_weights']

        print(f"\nCurrent (equal):     stage2=1.00, stage3=1.00, stage4=1.00")
        print(f"Recommended:         stage2={weights['stage2']:.2f}, stage3={weights['stage3']:.2f}, stage4={weights['stage4']:.2f}")

        print("\nReasoning:")
        for stage in ['stage2', 'stage3', 'stage4']:
            print(f"  {stage}: {weights_result['reasoning'][stage]}")

        print("\nComponent breakdown:")
        comp = weights_result['components']
        for stage in ['stage2', 'stage3', 'stage4']:
            print(f"  {stage}:")
            print(f"    Discrimination:  {comp['discrimination'][stage]:.3f}")
            print(f"    Decisiveness:    {comp['decisiveness'][stage]:.3f}")
            print(f"    Agreement:       {comp['agreement_penalty'][stage]:.3f}")
            print(f"    Variability:     {comp['variability'][stage]:.3f}")
            print(f"    -> Combined:      {comp['combined'][stage]:.3f}")

        # Save results
        output_path = Path('data/out/stage_weight_calibration.json')
        output = {
            'discrimination': discrimination,
            'agreement': agreement,
            'calibration': calibration,
            'weights': weights_result
        }
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2, default=str)

        print(f"\n✓ Saved detailed results to {output_path}")

        return output

    def visualize_results(self, results: dict):
        """Create visualization of calibration analysis."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # 1. Discrimination power comparison
        ax = axes[0, 0]
        stages = ['Stage 2', 'Stage 3', 'Stage 4']
        margins = [results['discrimination'][p]['avg_margin'] for p in ['stage2', 'stage3', 'stage4']]
        clear_winners = [results['discrimination'][p]['clear_winners'] for p in ['stage2', 'stage3', 'stage4']]

        x = np.arange(len(stages))
        width = 0.35
        ax.bar(x - width/2, margins, width, label='Avg Margin', color='#3498db')
        ax.bar(x + width/2, clear_winners, width, label='Clear Winners', color='#e74c3c')
        ax.set_ylabel('Score')
        ax.set_title('Discrimination Power')
        ax.set_xticks(x)
        ax.set_xticklabels(stages)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)

        # 2. Phase agreement matrix
        ax = axes[0, 1]
        agreement_matrix = np.array([
            [1.0, results['agreement']['stage2_stage3_agreement'], results['agreement']['stage2_stage4_agreement']],
            [results['agreement']['stage2_stage3_agreement'], 1.0, results['agreement']['stage3_stage4_agreement']],
            [results['agreement']['stage2_stage4_agreement'], results['agreement']['stage3_stage4_agreement'], 1.0]
        ])
        im = ax.imshow(agreement_matrix, cmap='RdYlGn', vmin=0, vmax=1)
        ax.set_xticks([0, 1, 2])
        ax.set_yticks([0, 1, 2])
        ax.set_xticklabels(stages)
        ax.set_yticklabels(stages)
        ax.set_title('Phase Agreement Matrix')

        # Add text annotations
        for i in range(3):
            for j in range(3):
                text = ax.text(j, i, f'{agreement_matrix[i, j]:.2f}',
                             ha="center", va="center", color="black", fontweight='bold')

        plt.colorbar(im, ax=ax)

        # 3. Recommended weights
        ax = axes[1, 0]
        current_weights = [1.0, 1.0, 1.0]
        recommended = [results['weights']['recommended_weights'][p] for p in ['stage2', 'stage3', 'stage4']]

        x = np.arange(len(stages))
        width = 0.35
        ax.bar(x - width/2, current_weights, width, label='Current (Equal)', color='#95a5a6')
        ax.bar(x + width/2, recommended, width, label='Recommended', color='#2ecc71')
        ax.set_ylabel('Weight')
        ax.set_title('Phase Weights Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(stages)
        ax.legend()
        ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
        ax.grid(axis='y', alpha=0.3)

        # 4. Top-1 score distributions
        ax = axes[1, 1]
        stage2_scores = [self.stage2_sim.loc[lid].max() for lid in self.lineage_ids]
        stage3_scores = [self.stage3_sim.loc[lid].max() for lid in self.lineage_ids]
        stage4_scores = [self.stage4_sim.loc[lid].max() for lid in self.lineage_ids]

        ax.hist(stage2_scores, bins=20, alpha=0.5, label='Stage 2', color='#3498db')
        ax.hist(stage3_scores, bins=20, alpha=0.5, label='Stage 3', color='#e74c3c')
        ax.hist(stage4_scores, bins=20, alpha=0.5, label='Stage 4', color='#f39c12')
        ax.set_xlabel('Top-1 Score')
        ax.set_ylabel('Count')
        ax.set_title('Top-1 Score Distributions')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        output_path = Path('data/out/figures/stage_weight_calibration.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved visualization to {output_path}")


def main():
    """Run stage weight calibration analysis."""
    calibrator = StageWeightCalibrator()
    results = calibrator.generate_report()
    calibrator.visualize_results(results)

    print("\n" + "="*70)
    print("NEXT STEPS")
    print("="*70)
    print("""
1. Review the recommended weights in data/out/stage_weight_calibration.json
2. Update stage_weights in scripts/stage5_ensemble_mapping.py if desired
3. Re-run Phase 5: python scripts/stage5_ensemble_mapping.py
4. Compare results before/after weight tuning
5. Iterate if needed, or proceed to expert validation

Note: These weights are data-driven but should be validated with domain
expert feedback when available. They optimize for internal consistency,
not necessarily for ground truth accuracy.
    """)


if __name__ == "__main__":
    main()
