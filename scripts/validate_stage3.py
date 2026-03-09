#!/usr/bin/env python3
"""
Phase 3 Validation: c-TF-IDF Term Extraction Quality

Validates Phase 3 outputs and generates user-friendly visualizations:
- Data integrity checks (nulls, ranges, coverage)
- Top distinctive terms per front
- Term similarity distribution
- Lineage coverage analysis

Usage:
    python scripts/validate_stage3.py
"""

import json
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml


def validate_data_integrity(terms_df: pd.DataFrame, similarity_df: pd.DataFrame) -> Dict:
    """
    Run data integrity checks on Phase 3 outputs.

    Returns: dict with check results
    """
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


def generate_similarity_heatmap(similarity_df: pd.DataFrame, output_path: Path):
    """
    Generate heatmap of lineage-front term similarity.
    Shows top 30 lineages (by max similarity) x all fronts.
    """
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

    ax.set_title('Phase 3: Lineage-Front Term Similarity (c-TF-IDF, Top 30)',
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Research Front', fontsize=12)
    ax.set_ylabel('Lineage ID', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  [OK] Saved similarity heatmap: {output_path}")


def generate_top_terms_showcase(terms_df: pd.DataFrame, similarity_df: pd.DataFrame,
                                 output_path: Path, n_examples: int = 15):
    """
    Generate showcase of top distinctive terms for top-matched lineages.
    Shows top 10 terms for the 15 lineages with highest similarity scores.
    """
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
    for i, row in enumerate(showcase_data):
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
    ax.set_title('Phase 3: Top Distinctive Terms for Top-Matched Lineages',
                 fontsize=14, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  [OK] Saved top terms showcase: {output_path}")


def generate_score_distributions(terms_df: pd.DataFrame, similarity_df: pd.DataFrame,
                                  output_path: Path):
    """
    Generate 4-panel distribution plots:
    1. c-TF-IDF score distribution
    2. Similarity score distribution
    3. Terms per lineage distribution
    4. Matches per front distribution
    """
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
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

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
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

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
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

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

    print(f"  [OK] Saved score distributions: {output_path}")


def generate_markdown_report(checks: Dict, output_path: Path):
    """
    Generate markdown summary report.
    """
    report = []
    report.append("# Phase 3 Validation Report")
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

    print(f"  [OK] Saved validation report: {output_path}")


def main():
    print("="*70)
    print("PHASE 3 VALIDATION")
    print("="*70)
    print()

    # Load data
    print("[1/5] Loading Phase 3 outputs...")
    terms_path = Path('data/out/02_lineage_tracking/lineage_ctfidf_terms.csv')
    similarity_path = Path('data/out/03_milestone_mapping/lineage_front_term_similarity.csv')

    if not terms_path.exists():
        print(f"  [ERROR] Terms file not found: {terms_path}")
        return
    if not similarity_path.exists():
        print(f"  [ERROR] Similarity file not found: {similarity_path}")
        return

    terms_df = pd.read_csv(terms_path)
    similarity_df = pd.read_csv(similarity_path)
    print(f"  [OK] Loaded {len(terms_df)} term records, {len(similarity_df)} lineages")

    # Validate data integrity
    print("\n[2/5] Running data integrity checks...")
    checks = validate_data_integrity(terms_df, similarity_df)
    print(f"  [OK] Completed {len([k for k in checks.keys() if k.endswith('_ok') or k.endswith('_nulls')])} checks")

    # Generate visualizations
    output_dir = Path('data/out/06_validation/phase3')
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n[3/5] Generating visualizations...")
    generate_similarity_heatmap(similarity_df, output_dir / 'phase_3_similarity_heatmap.png')
    generate_top_terms_showcase(terms_df, similarity_df, output_dir / 'phase_3_top_terms.png')
    generate_score_distributions(terms_df, similarity_df, output_dir / 'phase_3_distributions.png')

    # Generate markdown report
    print("\n[4/5] Generating validation report...")
    generate_markdown_report(checks, output_dir / 'phase_3_validation_report.md')

    # Save JSON validation results
    print("\n[5/5] Saving validation results...")
    validation_path = output_dir / 'phase_3_validation_results.json'
    with open(validation_path, 'w') as f:
        json.dump(checks, f, indent=2)
    print(f"  [OK] Saved validation results: {validation_path}")

    print("\n" + "="*70)
    print("VALIDATION COMPLETE")
    print("="*70)

    # Summary
    all_checks_pass = (
        checks['terms_no_nulls'] and
        checks['similarity_no_nulls'] and
        checks['ctfidf_positive'] and
        checks['similarity_range_ok']
    )

    if all_checks_pass:
        print("\n[OK] ALL VALIDATION CHECKS PASSED")
    else:
        print("\n[FAIL] SOME VALIDATION CHECKS FAILED")

    print(f"\nOutputs:")
    print(f"  - {output_dir / 'phase_3_similarity_heatmap.png'}")
    print(f"  - {output_dir / 'phase_3_top_terms.png'}")
    print(f"  - {output_dir / 'phase_3_distributions.png'}")
    print(f"  - {output_dir / 'phase_3_validation_report.md'}")
    print(f"  - {output_dir / 'phase_3_validation_results.json'}")


if __name__ == '__main__':
    main()
