#!/usr/bin/env python3
"""
Comprehensive QA Plot Inspection
Automated triage and quality assessment of inflection point labels.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json

def load_labels(labels_path: Path) -> pd.DataFrame:
    """Load inflection labels with validation."""
    df = pd.read_csv(labels_path)
    print(f"Loaded {len(df)} inflection labels")
    return df

def compute_statistics(df: pd.DataFrame):
    """Compute aggregate statistics."""
    print("\n" + "="*80)
    print("INFLECTION LABELS - AGGREGATE STATISTICS")
    print("="*80)

    print(f"\nTotal Labels: {len(df)}")

    # Detection method breakdown
    print(f"\nDetection Method Breakdown:")
    method_counts = df['inflection_type'].value_counts()
    for method, count in method_counts.items():
        pct = 100 * count / len(df)
        print(f"  {method:12s}: {count:4d} ({pct:5.1f}%)")

    # R^2 distribution (for logistic method)
    logistic_df = df[df['inflection_type'] == 'logistic'].copy()
    if len(logistic_df) > 0:
        print(f"\nR^2 Distribution (Logistic Method, n={len(logistic_df)}):")
        print(f"  Mean:   {logistic_df['inflection_score'].mean():.4f}")
        print(f"  Median: {logistic_df['inflection_score'].median():.4f}")
        print(f"  Std:    {logistic_df['inflection_score'].std():.4f}")
        print(f"  Min:    {logistic_df['inflection_score'].min():.4f}")
        print(f"  Max:    {logistic_df['inflection_score'].max():.4f}")

        # Percentiles
        percentiles = [5, 10, 25, 50, 75, 90, 95]
        print(f"\n  Percentiles:")
        for p in percentiles:
            val = np.percentile(logistic_df['inflection_score'], p)
            print(f"    {p:2d}th: {val:.4f}")

    # Derivative method statistics
    derivative_df = df[df['inflection_type'] == 'derivative'].copy()
    if len(derivative_df) > 0:
        print(f"\nDerivative Method (n={len(derivative_df)}):")
        print(f"  Scores: {derivative_df['inflection_score'].value_counts().to_dict()}")

    # Milestone linkage
    print(f"\nMilestone Linkage:")
    milestone_counts = df['lag_bucket'].value_counts()
    for bucket, count in milestone_counts.items():
        pct = 100 * count / len(df)
        print(f"  {bucket:15s}: {count:4d} ({pct:5.1f}%)")

    return {
        'total': len(df),
        'logistic_count': len(logistic_df),
        'derivative_count': len(derivative_df),
        'r2_mean': logistic_df['inflection_score'].mean() if len(logistic_df) > 0 else None,
        'r2_median': logistic_df['inflection_score'].median() if len(logistic_df) > 0 else None
    }

def auto_triage(df: pd.DataFrame):
    """Automatically categorize labels into bins based on quality indicators."""
    print("\n" + "="*80)
    print("AUTOMATED TRIAGE")
    print("="*80)

    bins = {
        'BIN_A_CLEAR_ACCEPT': [],
        'BIN_B_BORDERLINE': [],
        'BIN_C_SUSPECT': []
    }

    for idx, row in df.iterrows():
        lineage_id = row['lineage_id']
        score = row['inflection_score']
        method = row['inflection_type']

        # Bin assignment logic
        if method == 'logistic':
            if score >= 0.90:
                bins['BIN_A_CLEAR_ACCEPT'].append({
                    'lineage_id': lineage_id,
                    'quarter': row['quarter'],
                    'score': score,
                    'method': method,
                    'reason': 'High R² (≥0.90)'
                })
            elif score >= 0.85:
                bins['BIN_B_BORDERLINE'].append({
                    'lineage_id': lineage_id,
                    'quarter': row['quarter'],
                    'score': score,
                    'method': method,
                    'reason': 'Moderate R² (0.85-0.90)'
                })
            elif score >= 0.70:
                bins['BIN_B_BORDERLINE'].append({
                    'lineage_id': lineage_id,
                    'quarter': row['quarter'],
                    'score': score,
                    'method': method,
                    'reason': 'Low R² (0.70-0.85)'
                })
            else:
                bins['BIN_C_SUSPECT'].append({
                    'lineage_id': lineage_id,
                    'quarter': row['quarter'],
                    'score': score,
                    'method': method,
                    'reason': 'Very low R² (<0.70)'
                })

        elif method == 'derivative':
            # Derivative method is heuristic-based, requires visual review
            bins['BIN_B_BORDERLINE'].append({
                'lineage_id': lineage_id,
                'quarter': row['quarter'],
                'score': score,
                'method': method,
                'reason': 'Derivative method (needs visual check)'
            })

    # Print summary
    print(f"\nBIN A (Clear Accept - High Confidence):")
    print(f"  Count: {len(bins['BIN_A_CLEAR_ACCEPT'])}")
    print(f"  % of Total: {100 * len(bins['BIN_A_CLEAR_ACCEPT']) / len(df):.1f}%")
    print(f"  Criteria: Logistic method with R^2 >= 0.90")

    print(f"\nBIN B (Borderline - Needs Review):")
    print(f"  Count: {len(bins['BIN_B_BORDERLINE'])}")
    print(f"  % of Total: {100 * len(bins['BIN_B_BORDERLINE']) / len(df):.1f}%")
    print(f"  Criteria: R^2 0.70-0.90 OR derivative method")

    print(f"\nBIN C (Suspect - Likely Reject):")
    print(f"  Count: {len(bins['BIN_C_SUSPECT'])}")
    print(f"  % of Total: {100 * len(bins['BIN_C_SUSPECT']) / len(df):.1f}%")
    print(f"  Criteria: R^2 < 0.70")

    return bins

def generate_review_manifest(bins: dict, output_path: Path):
    """Generate manifest files for manual review."""

    # BIN B: Priority review list
    bin_b_df = pd.DataFrame(bins['BIN_B_BORDERLINE'])
    if len(bin_b_df) > 0:
        bin_b_df = bin_b_df.sort_values('score', ascending=True)
        bin_b_path = output_path / 'qa_review_bin_b_borderline.csv'
        bin_b_df.to_csv(bin_b_path, index=False)
        print(f"\nGenerated BIN B review manifest: {bin_b_path}")
        print(f"  Review these {len(bin_b_df)} cases in priority order (lowest score first)")

    # BIN C: Rejection candidates
    bin_c_df = pd.DataFrame(bins['BIN_C_SUSPECT'])
    if len(bin_c_df) > 0:
        bin_c_path = output_path / 'qa_review_bin_c_suspect.csv'
        bin_c_df.to_csv(bin_c_path, index=False)
        print(f"\nGenerated BIN C review manifest: {bin_c_path}")
        print(f"  Consider removing these {len(bin_c_df)} cases (very low R²)")

    # Summary report
    summary = {
        'total_labels': sum(len(bins[k]) for k in bins),
        'bin_a_clear_accept': len(bins['BIN_A_CLEAR_ACCEPT']),
        'bin_b_borderline': len(bins['BIN_B_BORDERLINE']),
        'bin_c_suspect': len(bins['BIN_C_SUSPECT']),
        'review_required': len(bins['BIN_B_BORDERLINE']) + len(bins['BIN_C_SUSPECT']),
        'pct_clear_accept': 100 * len(bins['BIN_A_CLEAR_ACCEPT']) / sum(len(bins[k]) for k in bins),
        'pct_review_needed': 100 * (len(bins['BIN_B_BORDERLINE']) + len(bins['BIN_C_SUSPECT'])) / sum(len(bins[k]) for k in bins)
    }

    summary_path = output_path / 'qa_inspection_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nGenerated summary report: {summary_path}")

    return summary

def analyze_low_r2_cases(df: pd.DataFrame, bins: dict):
    """Analyze characteristics of low R^2 cases."""
    print("\n" + "="*80)
    print("LOW R^2 CASES ANALYSIS")
    print("="*80)

    # Get all borderline and suspect cases
    review_cases = bins['BIN_B_BORDERLINE'] + bins['BIN_C_SUSPECT']
    review_df = pd.DataFrame(review_cases)

    if len(review_df) == 0:
        print("No cases flagged for review!")
        return

    print(f"\nTotal cases flagged: {len(review_df)}")

    # Breakdown by reason
    print(f"\nBreakdown by reason:")
    reason_counts = review_df['reason'].value_counts()
    for reason, count in reason_counts.items():
        pct = 100 * count / len(review_df)
        print(f"  {reason:40s}: {count:4d} ({pct:5.1f}%)")

    # R^2 distribution of flagged logistic cases
    logistic_review = review_df[review_df['method'] == 'logistic']
    if len(logistic_review) > 0:
        print(f"\nR^2 Distribution (Flagged Logistic Cases, n={len(logistic_review)}):")
        print(f"  Mean:   {logistic_review['score'].mean():.4f}")
        print(f"  Median: {logistic_review['score'].median():.4f}")
        print(f"  Min:    {logistic_review['score'].min():.4f}")
        print(f"  Max:    {logistic_review['score'].max():.4f}")

    # Derivative cases
    derivative_review = review_df[review_df['method'] == 'derivative']
    if len(derivative_review) > 0:
        print(f"\nDerivative Method Cases: {len(derivative_review)}")
        print(f"  All derivative cases require visual review")
        print(f"  Sample lineage IDs: {derivative_review['lineage_id'].head(5).tolist()}")

def main():
    # Paths
    labels_path = Path('data/out/02_lineage_tracking/inflection_labels.csv')
    output_dir = Path('data/out/qa_inspection')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    df = load_labels(labels_path)

    # Step 1: Aggregate statistics
    stats = compute_statistics(df)

    # Step 2: Automated triage
    bins = auto_triage(df)

    # Step 3: Generate review manifests
    summary = generate_review_manifest(bins, output_dir)

    # Step 4: Analyze low R² cases
    analyze_low_r2_cases(df, bins)

    # Final recommendations
    print("\n" + "="*80)
    print("RECOMMENDATIONS")
    print("="*80)

    print(f"\n1. IMMEDIATE ACTIONS:")
    print(f"   - BIN A ({summary['bin_a_clear_accept']} cases): Auto-accept, no review needed")
    print(f"   - BIN B ({summary['bin_b_borderline']} cases): Manual review recommended")
    print(f"   - BIN C ({summary['bin_c_suspect']} cases): Strong candidates for removal")

    if summary['pct_review_needed'] > 20:
        print(f"\n2. PARAMETER TUNING:")
        print(f"   - {summary['pct_review_needed']:.1f}% of labels need review (>20% threshold)")
        print(f"   - Consider increasing R^2 threshold in labeling script:")
        print(f"     Current: r_squared_threshold = 0.70 (inferred)")
        print(f"     Recommended: r_squared_threshold = 0.85")
        print(f"   - This would reduce false positives in training data")
    else:
        print(f"\n2. PARAMETER TUNING:")
        print(f"   - {summary['pct_review_needed']:.1f}% need review (acceptable)")
        print(f"   - Current thresholds appear well-calibrated")

    print(f"\n3. ESTIMATED REVIEW TIME:")
    review_time_min = summary['review_required'] * 2 / 60  # 2 min per case
    review_time_max = summary['review_required'] * 3 / 60
    print(f"   - {summary['review_required']} cases × 2-3 min/case")
    print(f"   - Total: {review_time_min:.1f}-{review_time_max:.1f} hours")

    print(f"\n4. WORKFLOW:")
    print(f"   a. Review BIN C first (likely rejects): data/out/qa_inspection/qa_review_bin_c_suspect.csv")
    print(f"   b. Review BIN B (borderline): data/out/qa_inspection/qa_review_bin_b_borderline.csv")
    print(f"   c. Use 5-point checklist for each case")
    print(f"   d. Document decisions and update labels.csv")

    print(f"\n" + "="*80)
    print("INSPECTION COMPLETE")
    print("="*80)

if __name__ == '__main__':
    main()
