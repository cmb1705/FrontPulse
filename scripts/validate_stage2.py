#!/usr/bin/env python3
"""
Phase 2 Validation: SciBERT Embedding Quality

Validates Phase 2 outputs and generates user-friendly visualizations:
- Data integrity checks (shape, nulls, embedding quality)
- Embedding similarity heatmap
- Cosine similarity distribution
- Dimensionality reduction visualization (t-SNE)

Usage:
    python scripts/validate_stage2.py
"""

import json
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.preprocessing import normalize


def validate_data_integrity(embeddings_array: np.ndarray, metadata: pd.DataFrame,
                             similarity_df: pd.DataFrame) -> Dict:
    """
    Run data integrity checks on Phase 2 outputs.

    Returns: dict with check results
    """
    checks = {}

    # Check 1: Data shapes
    n_lineages = len(embeddings_array)
    embedding_dim = embeddings_array.shape[1]
    n_fronts = len(similarity_df.columns) - 1  # Exclude lineage_id

    checks['n_lineages'] = int(n_lineages)
    checks['embedding_dim'] = int(embedding_dim)
    checks['n_fronts'] = int(n_fronts)

    # Check 2: Embeddings are normalized
    norms = np.linalg.norm(embeddings_array, axis=1)
    checks['embeddings_normalized'] = bool(np.allclose(norms, 1.0, rtol=1e-3))
    checks['norm_mean'] = float(norms.mean())
    checks['norm_std'] = float(norms.std())

    # Check 3: No NaN/Inf in embeddings
    checks['embeddings_finite'] = bool(np.isfinite(embeddings_array).all())

    # Check 4: No nulls in metadata
    metadata_nulls = metadata.isnull().sum().sum()
    checks['metadata_no_nulls'] = bool(metadata_nulls == 0)

    # Check 5: Similarity scores in [-1, 1] (cosine similarity range)
    similarity_values = similarity_df.iloc[:, 1:].values.flatten()
    checks['similarity_range_ok'] = bool((similarity_values >= -1).all() and
                                         (similarity_values <= 1).all())
    checks['similarity_min'] = float(similarity_values.min())
    checks['similarity_max'] = float(similarity_values.max())
    checks['similarity_mean'] = float(similarity_values.mean())

    # Check 6: Coverage (lineages with at least one match above threshold)
    threshold = 0.747  # Optimized for ~50% coverage with all fronts
    has_match = (similarity_df.iloc[:, 1:] > threshold).any(axis=1)
    checks['coverage_pct'] = float(has_match.sum() / len(similarity_df) * 100)
    checks['lineages_with_matches'] = int(has_match.sum())
    checks['coverage_threshold'] = threshold

    # Check 7: Front match counts (above threshold)
    front_matches = {}
    for col in similarity_df.columns[1:]:
        n_matches = int((similarity_df[col] > threshold).sum())
        front_matches[col] = n_matches
    checks['front_matches'] = front_matches

    # Check 8: Embedding variance (should have meaningful variation)
    embedding_variance = embeddings_array.var(axis=0)
    checks['embedding_variance_mean'] = float(embedding_variance.mean())
    checks['embedding_variance_std'] = float(embedding_variance.std())

    return checks


def generate_similarity_heatmap(similarity_df: pd.DataFrame, output_path: Path):
    """
    Generate heatmap of lineage-front cosine similarity.
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
        cmap='RdYlGn',
        center=0,
        vmin=-0.2,
        vmax=1.0,
        cbar_kws={'label': 'Cosine Similarity'},
        ax=ax
    )

    ax.set_title('Phase 2: Lineage-Front Cosine Similarity (SciBERT, Top 30)',
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Research Front', fontsize=12)
    ax.set_ylabel('Lineage ID', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  [OK] Saved similarity heatmap: {output_path}")


def generate_tsne_visualization(embeddings_array: np.ndarray, metadata: pd.DataFrame,
                                 similarity_df: pd.DataFrame, output_path: Path):
    """
    Generate t-SNE 2D visualization of lineage embeddings.
    Color-code by best matching research front.
    """
    # Compute t-SNE
    print("    Computing t-SNE (this may take a minute)...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings_array)-1))
    embeddings_2d = tsne.fit_transform(embeddings_array)

    # Get best matching front for each lineage
    front_names = similarity_df.columns[1:].tolist()
    best_fronts = similarity_df.iloc[:, 1:].idxmax(axis=1).values
    best_scores = similarity_df.iloc[:, 1:].max(axis=1).values

    # Create color map
    unique_fronts = sorted(set(best_fronts))
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_fronts)))
    front_to_color = {front: colors[i] for i, front in enumerate(unique_fronts)}

    # Plot
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

    ax.set_title('Phase 2: Lineage Embedding Space (t-SNE, Colored by Best Front)',
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('t-SNE Dimension 1', fontsize=12)
    ax.set_ylabel('t-SNE Dimension 2', fontsize=12)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  [OK] Saved t-SNE visualization: {output_path}")


def generate_score_distributions(embeddings_array: np.ndarray, similarity_df: pd.DataFrame,
                                  output_path: Path):
    """
    Generate 4-panel distribution plots:
    1. Cosine similarity distribution
    2. Embedding norm distribution
    3. Matches per front distribution
    4. Best similarity per lineage distribution
    """
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

    # Add stats
    stats_text = f"Mean: {similarity_values.mean():.3f}\nMedian: {np.median(similarity_values):.3f}\nMax: {similarity_values.max():.3f}"
    ax.text(0.98, 0.97, stats_text, transform=ax.transAxes,
            fontsize=9, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Panel 2: Embedding norm distribution
    ax = axes[0, 1]
    norms = np.linalg.norm(embeddings_array, axis=1)
    # For normalized embeddings, use fixed small number of bins
    # All norms should be very close to 1.0
    try:
        ax.hist(norms, bins=10, color='coral', edgecolor='black', alpha=0.7)
    except ValueError:
        # If histogram fails (constant values), show bar instead
        ax.axvline(norms.mean(), color='coral', linewidth=10, alpha=0.7, label=f'All norms = {norms.mean():.4f}')
        ax.legend()
    ax.set_xlabel('L2 Norm', fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    ax.set_title('Embedding Norm Distribution', fontsize=12, fontweight='bold')
    ax.axvline(1.0, color='red', linestyle='--', label='Target (1.0)')
    ax.grid(axis='y', alpha=0.3)
    ax.legend()

    stats_text = f"Mean: {norms.mean():.4f}\nStd: {norms.std():.4f}"
    ax.text(0.98, 0.97, stats_text, transform=ax.transAxes,
            fontsize=9, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Panel 3: Matches per front distribution (threshold = 0.747)
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

    # Panel 4: Best similarity per lineage distribution
    ax = axes[1, 1]
    best_similarities = similarity_df.iloc[:, 1:].max(axis=1)
    ax.hist(best_similarities.values, bins=30, color='orchid', edgecolor='black', alpha=0.7)
    ax.set_xlabel('Best Similarity Score', fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    ax.set_title('Best Front Match per Lineage', fontsize=12, fontweight='bold')
    ax.axvline(0.747, color='red', linestyle='--', label='Threshold (0.747)')
    ax.grid(axis='y', alpha=0.3)
    ax.legend()

    stats_text = f"Mean: {best_similarities.mean():.3f}\nMedian: {best_similarities.median():.3f}\nMax: {best_similarities.max():.3f}"
    ax.text(0.98, 0.97, stats_text, transform=ax.transAxes,
            fontsize=9, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  [OK] Saved score distributions: {output_path}")


def generate_markdown_report(checks: Dict, output_path: Path):
    """
    Generate markdown summary report.
    """
    report = []
    report.append("# Phase 2 Validation Report")
    report.append("**SciBERT Embedding Quality**")
    report.append("")

    # Data Integrity
    report.append("## Data Integrity")
    report.append("")
    report.append("| Check | Status | Details |")
    report.append("|-------|--------|---------|")

    # Shape checks
    report.append(f"| Dataset Shape | [OK] | {checks['n_lineages']} lineages, {checks['embedding_dim']}D embeddings, {checks['n_fronts']} fronts |")

    # Embedding quality checks
    norm_status = "[OK]" if checks['embeddings_normalized'] else "[WARN]"
    report.append(f"| Embeddings Normalized | {norm_status} | Mean norm: {checks['norm_mean']:.4f}, Std: {checks['norm_std']:.4f} |")

    finite_status = "[OK]" if checks['embeddings_finite'] else "[FAIL]"
    report.append(f"| No NaN/Inf | {finite_status} | All embedding values are finite |")

    # Metadata checks
    meta_status = "[OK]" if checks['metadata_no_nulls'] else "[FAIL]"
    report.append(f"| Metadata Complete | {meta_status} | All lineage metadata present |")

    # Similarity range checks
    sim_range_status = "[OK]" if checks['similarity_range_ok'] else "[FAIL]"
    report.append(f"| Similarity Range | {sim_range_status} | All scores in [-1, 1] |")

    # Coverage check
    coverage_status = "[OK]" if checks['coverage_pct'] > 50 else "[WARN]" if checks['coverage_pct'] > 0 else "[FAIL]"
    report.append(f"| Lineage Coverage | {coverage_status} | {checks['lineages_with_matches']}/{checks['n_lineages']} ({checks['coverage_pct']:.1f}%) above threshold {checks['coverage_threshold']} |")

    # Embedding variance
    var_status = "[OK]" if checks['embedding_variance_mean'] > 0.001 else "[WARN]"
    report.append(f"| Embedding Variance | {var_status} | Mean: {checks['embedding_variance_mean']:.4f}, Std: {checks['embedding_variance_std']:.4f} |")

    report.append("")

    # Front Match Summary
    report.append("## Research Front Matches")
    report.append(f"**Threshold: {checks['coverage_threshold']:.3f} (optimized for ~50% coverage with all fronts)**")
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
    report.append(f"- **Embedding Dimension**: {checks['embedding_dim']}D")
    report.append(f"- **Cosine Similarity**: Mean={checks['similarity_mean']:.3f}, Max={checks['similarity_max']:.3f}")
    report.append(f"- **Total Matches** (>{checks['coverage_threshold']}): {sum(checks['front_matches'].values())} lineage-front pairs")
    report.append(f"- **Average Matches/Front**: {sum(checks['front_matches'].values()) / checks['n_fronts']:.1f}")

    report.append("")

    # Overall Assessment
    all_checks_pass = (
        checks['embeddings_normalized'] and
        checks['embeddings_finite'] and
        checks['metadata_no_nulls'] and
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
    print("PHASE 2 VALIDATION")
    print("="*70)
    print()

    # Load data
    print("[1/6] Loading Phase 2 outputs...")
    embeddings_path = Path('data/out/02_lineage_tracking/lineage_embeddings.npz')
    similarity_path = Path('data/out/03_milestone_mapping/lineage_front_similarity.csv')

    if not embeddings_path.exists():
        print(f"  [ERROR] Embeddings file not found: {embeddings_path}")
        return
    if not similarity_path.exists():
        print(f"  [ERROR] Similarity file not found: {similarity_path}")
        return

    # Load embeddings
    data = np.load(embeddings_path)
    embeddings_array = data['embeddings']
    lineage_ids = data['lineage_ids']
    print(f"  [OK] Loaded {len(embeddings_array)} lineage embeddings ({embeddings_array.shape[1]}D)")

    # Create metadata DataFrame
    metadata = pd.DataFrame({'lineage_id': lineage_ids})

    # Load similarity matrix
    similarity_df = pd.read_csv(similarity_path)
    print(f"  [OK] Loaded similarity matrix: {len(similarity_df)} x {len(similarity_df.columns)-1}")

    # Validate data integrity
    print("\n[2/6] Running data integrity checks...")
    checks = validate_data_integrity(embeddings_array, metadata, similarity_df)
    print(f"  [OK] Completed {len([k for k in checks.keys() if k.endswith('_ok') or k.endswith('_normalized') or k.endswith('_finite')])} checks")

    # Generate visualizations
    output_dir = Path('data/out/06_validation/phase2')
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n[3/6] Generating similarity heatmap...")
    generate_similarity_heatmap(similarity_df, output_dir / 'phase_2_similarity_heatmap.png')

    print("\n[4/6] Generating t-SNE visualization...")
    generate_tsne_visualization(embeddings_array, metadata, similarity_df,
                                output_dir / 'phase_2_tsne.png')

    print("\n[5/6] Generating score distributions...")
    generate_score_distributions(embeddings_array, similarity_df,
                                  output_dir / 'phase_2_distributions.png')

    # Generate markdown report
    print("\n[6/6] Generating validation report...")
    generate_markdown_report(checks, output_dir / 'phase_2_validation_report.md')

    # Save JSON validation results
    print("\n[Final] Saving validation results...")
    validation_path = output_dir / 'phase_2_validation_results.json'
    with open(validation_path, 'w') as f:
        json.dump(checks, f, indent=2)
    print(f"  [OK] Saved validation results: {validation_path}")

    print("\n" + "="*70)
    print("VALIDATION COMPLETE")
    print("="*70)

    # Summary
    all_checks_pass = (
        checks['embeddings_normalized'] and
        checks['embeddings_finite'] and
        checks['metadata_no_nulls'] and
        checks['similarity_range_ok']
    )

    if all_checks_pass:
        print("\n[OK] ALL VALIDATION CHECKS PASSED")
    else:
        print("\n[FAIL] SOME VALIDATION CHECKS FAILED")

    print(f"\nOutputs:")
    print(f"  - {output_dir / 'phase_2_similarity_heatmap.png'}")
    print(f"  - {output_dir / 'phase_2_tsne.png'}")
    print(f"  - {output_dir / 'phase_2_distributions.png'}")
    print(f"  - {output_dir / 'phase_2_validation_report.md'}")
    print(f"  - {output_dir / 'phase_2_validation_results.json'}")


if __name__ == '__main__':
    main()
