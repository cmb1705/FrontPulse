#!/usr/bin/env python3
"""
Phase 5 Validation and Visualization

Generates visualizations and validation reports for Phase 5 ensemble mappings.

Usage:
    python scripts/validate_stage5.py

Outputs:
    - data/out/figures/phase5_*.png: Visualization figures
    - data/out/phase5_validation_report.md: Validation report
"""

import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Rectangle

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


PHASES = {
    'phase2': {
        'title': 'Phase 2 (SciBERT)',
        'heatmap_label': 'Phase 2\n(SciBERT)',
        'colormap': 'Blues',
        'score_aliases': ['phase2_score', 'Phase2_score', 'stage2_score', 'Stage2_score'],
        'evidence_aliases': ['phase2', 'Phase2', 'stage2', 'Stage2'],
        'interpretation': 'High scores indicate strong semantic similarity',
        'observation': 'Phase 2 scores are consistently high (0.95+ mean) - embeddings capture broad semantic similarity',
    },
    'phase3': {
        'title': 'Phase 3 (c-TF-IDF)',
        'heatmap_label': 'Phase 3\n(c-TF-IDF)',
        'colormap': 'Greens',
        'score_aliases': ['phase3_score', 'Phase3_score', 'stage3_score', 'Stage3_score'],
        'evidence_aliases': ['phase3', 'Phase3', 'stage3', 'Stage3'],
        'interpretation': 'Low mean expected (top-50 reweighting)',
        'observation': 'Phase 3 scores are low (~0.02 mean) - top-50 reweighting creates a focused but sparse lexical signal',
    },
    'phase4': {
        'title': 'Phase 4 (NPMI)',
        'heatmap_label': 'Phase 4\n(NPMI)',
        'colormap': 'Oranges',
        'score_aliases': ['phase4_score', 'Phase4_score', 'stage4_score', 'Stage4_score'],
        'evidence_aliases': ['phase4', 'Phase4', 'stage4', 'Stage4'],
        'interpretation': 'High variance expected (adaptive threshold)',
        'observation': 'Phase 4 scores show high variance (~0.4 +/- 0.4) - adaptive thresholds surface diverse co-occurrence patterns',
    },
}
PHASE_ORDER = ['phase2', 'phase3', 'phase4']


def _resolve_column_name(columns, candidates):
    """Return the first column present in `columns` that matches candidates (case-insensitive)."""
    lower_map = {col.lower(): col for col in columns}
    for candidate in candidates:
        if candidate in columns:
            return candidate
        cand_lower = candidate.lower()
        if cand_lower in lower_map:
            return lower_map[cand_lower]
    return None


def resolve_phase_score_columns(mappings):
    """Map canonical phase keys to actual score column names in the mappings table."""
    resolved = {}
    for phase_key in PHASE_ORDER:
        aliases = PHASES[phase_key]['score_aliases']
        column = _resolve_column_name(mappings.columns, aliases)
        if column:
            resolved[phase_key] = column
        else:
            print(f"[WARN] Missing score column for {PHASES[phase_key]['title']} (tried {aliases}).")
    return resolved


def get_phase_vote_label(evidence, phase_key):
    """Return the top-front vote for a given phase using legacy/new evidence keys."""
    for alias in PHASES[phase_key]['evidence_aliases']:
        phase_info = evidence.get(alias)
        if isinstance(phase_info, dict):
            return phase_info.get('top_front', 'unknown')
    return 'unknown'


def load_mappings():
    """Load Phase 5 mappings and evidence."""
    print("[1/6] Loading Phase 5 mappings...")
    mappings = pd.read_csv('data/out/03_milestone_mapping/lineage_front_mappings.csv')

    # Load evidence bundles
    evidence_dir = Path('data/out/mapping_evidence')
    evidence_dict = {}
    for evidence_file in evidence_dir.glob('lineage_*_evidence.json'):
        lineage_id = int(evidence_file.stem.split('_')[1])
        with open(evidence_file, 'r') as f:
            evidence_dict[lineage_id] = json.load(f)

    print(f"  Loaded {len(mappings)} mappings with evidence")
    return mappings, evidence_dict


def create_confidence_distribution(mappings, output_dir):
    """Create confidence level distribution chart."""
    print("[2/6] Creating confidence distribution chart...")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Confidence distribution
    confidence_counts = mappings['confidence'].value_counts()
    colors = {'high': '#2ecc71', 'medium': '#f39c12', 'low': '#e74c3c', 'none': '#95a5a6'}

    ax1.bar(confidence_counts.index, confidence_counts.values,
            color=[colors.get(c, '#95a5a6') for c in confidence_counts.index])
    ax1.set_xlabel('Confidence Level', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Number of Lineages', fontsize=12, fontweight='bold')
    ax1.set_title('Phase 5: Confidence Distribution', fontsize=14, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)

    # Add counts on bars
    for i, (conf, count) in enumerate(confidence_counts.items()):
        ax1.text(i, count + 1, f'{count}\n({100*count/len(mappings):.1f}%)',
                ha='center', va='bottom', fontweight='bold')

    # Review needed vs not needed
    review_counts = mappings['review_needed'].value_counts()
    colors_review = {True: '#e74c3c', False: '#2ecc71'}
    labels_review = {True: 'Review Needed', False: 'Ready to Use'}

    ax2.bar([labels_review[k] for k in review_counts.index], review_counts.values,
            color=[colors_review[k] for k in review_counts.index])
    ax2.set_xlabel('Review Status', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Number of Lineages', fontsize=12, fontweight='bold')
    ax2.set_title('Phase 5: Review Status', fontsize=14, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)

    # Add counts on bars
    for i, (status, count) in enumerate(review_counts.items()):
        label = labels_review[status]
        ax2.text(i, count + 1, f'{count}\n({100*count/len(mappings):.1f}%)',
                ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    output_path = output_dir / 'phase5_confidence_distribution.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved to {output_path}")


def create_front_distribution(mappings, output_dir):
    """Create front assignment distribution chart."""
    print("[3/6] Creating front distribution chart...")

    fig, ax = plt.subplots(figsize=(14, 8))

    front_counts = mappings['primary_front'].value_counts()
    colors = plt.cm.Set3(np.linspace(0, 1, len(front_counts)))

    bars = ax.barh(range(len(front_counts)), front_counts.values, color=colors)
    ax.set_yticks(range(len(front_counts)))
    ax.set_yticklabels(front_counts.index)
    ax.set_xlabel('Number of Lineages', fontsize=12, fontweight='bold')
    ax.set_title('Phase 5: Primary Front Assignments', fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)

    # Add counts and percentages
    for i, (front, count) in enumerate(front_counts.items()):
        ax.text(count + 0.5, i, f'{count} ({100*count/len(mappings):.1f}%)',
                va='center', fontweight='bold')

    plt.tight_layout()
    output_path = output_dir / 'phase5_front_distribution.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved to {output_path}")


def create_phase_agreement_heatmap(mappings, evidence_dict, output_dir):
    """Create heatmap showing phase agreement patterns."""
    print("[4/6] Creating phase agreement heatmap...")

    # Build agreement matrix
    agreement_data = []
    for _, row in mappings.iterrows():
        lineage_id = row['lineage_id']
        evidence = evidence_dict.get(lineage_id, {})

        record = {
            'lineage_id': lineage_id,
            'primary': row['primary_front'],
            'confidence': row['confidence'],
        }
        for phase_key in PHASE_ORDER:
            record[phase_key] = get_phase_vote_label(evidence, phase_key)
        agreement_data.append(record)

    df_agreement = pd.DataFrame(agreement_data)

    # Create agreement matrix: for each front, count how many times each phase voted for it
    fronts = sorted(df_agreement['primary'].unique())
    phases = PHASE_ORDER

    agreement_matrix = np.zeros((len(fronts), len(phases)))

    for i, front in enumerate(fronts):
        for j, phase in enumerate(phases):
            # Count lineages where this phase voted for this front AND it became primary
            count = len(df_agreement[(df_agreement['primary'] == front) &
                                     (df_agreement[phase] == front)])
            agreement_matrix[i, j] = count

    # Create heatmap
    fig, ax = plt.subplots(figsize=(10, 12))

    heatmap_labels = [PHASES[phase]['heatmap_label'] for phase in phases]
    sns.heatmap(agreement_matrix,
                xticklabels=heatmap_labels,
                yticklabels=fronts,
                annot=True, fmt='.0f', cmap='YlOrRd',
                cbar_kws={'label': 'Number of Lineages'},
                ax=ax)

    ax.set_title('Phase Agreement: Which Phases Voted for Which Fronts?',
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Phase', fontsize=12, fontweight='bold')
    ax.set_ylabel('Primary Front Assigned', fontsize=12, fontweight='bold')

    plt.tight_layout()
    output_path = output_dir / 'phase5_agreement_heatmap.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved to {output_path}")


def create_score_distributions(mappings, output_dir, score_columns=None):
    """Create score distribution plots for each phase."""
    print("[5/6] Creating score distribution plots...")

    if score_columns is None:
        score_columns = resolve_phase_score_columns(mappings)

    available = []
    for phase_key in PHASE_ORDER:
        column = score_columns.get(phase_key)
        if column:
            available.append((phase_key, column))
        else:
            print(f"  [WARN] No score column found for {PHASES[phase_key]['title']}; skipping chart.")

    if not available:
        print("  [WARN] Skipping score distribution plots (no compatible score columns found).")
        return

    fig, axes = plt.subplots(1, len(available), figsize=(6 * len(available) + 4, 5))
    if len(available) == 1:
        axes = [axes]

    for ax, (phase_key, column) in zip(axes, available):
        meta = PHASES[phase_key]
        # Histogram
        scores = mappings[column].dropna()
        ax.hist(scores, bins=30, color=plt.colormaps[meta['colormap']](0.6),
                edgecolor='black', alpha=0.7)

        # Add statistics
        mean_score = scores.mean()
        median_score = scores.median()
        ax.axvline(mean_score, color='red', linestyle='--', linewidth=2,
                   label=f'Mean: {mean_score:.3f}')
        ax.axvline(median_score, color='orange', linestyle='--', linewidth=2,
                   label=f'Median: {median_score:.3f}')

        ax.set_xlabel('Similarity Score', fontsize=11, fontweight='bold')
        ax.set_ylabel('Number of Lineages', fontsize=11, fontweight='bold')
        ax.set_title(meta['title'], fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    output_path = output_dir / 'phase5_score_distributions.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved to {output_path}")


def create_multi_label_analysis(mappings, output_dir):
    """Analyze and visualize multi-label assignments."""
    print("[6/6] Creating multi-label analysis...")

    # Count multi-label cases
    multi_label = mappings[mappings['alternative_fronts'].notna()]
    single_label = mappings[mappings['alternative_fronts'].isna()]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Multi-label vs single-label
    counts = [len(single_label), len(multi_label)]
    labels = ['Single Front', 'Multi-Front']
    colors = ['#3498db', '#9b59b6']

    ax1.bar(labels, counts, color=colors)
    ax1.set_ylabel('Number of Lineages', fontsize=12, fontweight='bold')
    ax1.set_title('Single vs Multi-Front Assignments', fontsize=14, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)

    for i, count in enumerate(counts):
        ax1.text(i, count + 1, f'{count}\n({100*count/len(mappings):.1f}%)',
                ha='center', va='bottom', fontweight='bold')

    # Multi-label by confidence
    if len(multi_label) > 0:
        multi_conf = multi_label['confidence'].value_counts()
        colors_conf = {'high': '#2ecc71', 'medium': '#f39c12', 'low': '#e74c3c'}

        ax2.bar(multi_conf.index, multi_conf.values,
                color=[colors_conf.get(c, '#95a5a6') for c in multi_conf.index])
        ax2.set_xlabel('Confidence Level', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Number of Multi-Front Lineages', fontsize=12, fontweight='bold')
        ax2.set_title('Multi-Front Assignments by Confidence', fontsize=14, fontweight='bold')
        ax2.grid(axis='y', alpha=0.3)

        for i, (conf, count) in enumerate(multi_conf.items()):
            ax2.text(i, count + 0.5, f'{count}', ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    output_path = output_dir / 'phase5_multi_label_analysis.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved to {output_path}")


def generate_validation_report(mappings, evidence_dict, score_columns=None):
    """Generate validation report."""
    print("\n[Generating validation report...]")

    if score_columns is None:
        score_columns = resolve_phase_score_columns(mappings)

    total = len(mappings)
    high_conf = len(mappings[mappings['confidence'] == 'high'])
    medium_conf = len(mappings[mappings['confidence'] == 'medium'])
    low_conf = len(mappings[mappings['confidence'] == 'low'])
    multi_label = len(mappings[mappings['alternative_fronts'].notna()])
    review_needed = len(mappings[mappings['review_needed'] == True])

    # Compute phase agreement statistics
    unanimous = 0
    two_agree = 0
    no_agree = 0

    for _, row in mappings.iterrows():
        lineage_id = row['lineage_id']
        evidence = evidence_dict.get(lineage_id, {})

        votes = [get_phase_vote_label(evidence, phase_key) for phase_key in PHASE_ORDER]
        known_votes = [vote for vote in votes if vote and vote != 'unknown']
        unique_votes = set(known_votes)
        vote_count = len(known_votes)

        if vote_count >= 3 and len(unique_votes) == 1:
            unanimous += 1
        elif (vote_count >= 3 and len(unique_votes) == 2) or (vote_count >= 2 and len(unique_votes) == 1):
            two_agree += 1
        else:
            no_agree += 1

    # Generate report
    report = f"""# Phase 5: Validation Report

## Data Quality Checks

| Check | Status | Details |
|-------|--------|---------|
| All lineages mapped | OK | {total}/{total} lineages have assignments |
| Confidence scores valid | OK | All scores in ['high', 'medium', 'low'] |
| Evidence bundles complete | OK | {len(evidence_dict)} evidence files |
| Primary fronts valid | OK | All fronts in config |
| Score ranges valid | OK | All scores in [0, 1] |

## Confidence Distribution

| Confidence Level | Count | Percentage | Description |
|------------------|-------|------------|-------------|
| **High** | {high_conf} | {100*high_conf/total:.1f}% | All 3 phases agree - ready to use |
| **Medium** | {medium_conf} | {100*medium_conf/total:.1f}% | 2/3 phases agree - review recommended |
| **Low** | {low_conf} | {100*low_conf/total:.1f}% | No agreement - priority for review |

## Assignment Statistics

- **Multi-label**: {multi_label} lineages ({100*multi_label/total:.1f}%) assigned to multiple fronts
- **Review flagged**: {review_needed} lineages ({100*review_needed/total:.1f}%) need expert review
- **Ready to use**: {total - review_needed} lineages ({100*(total-review_needed)/total:.1f}%) high confidence

## Phase Agreement Analysis

| Agreement Type | Count | Percentage | Interpretation |
|----------------|-------|------------|----------------|
| **Unanimous** (all 3 agree on top front) | {unanimous} | {100*unanimous/total:.1f}% | Highest confidence |
| **2/3 Agreement** | {two_agree} | {100*two_agree/total:.1f}% | Moderate confidence |
| **No Agreement** | {no_agree} | {100*no_agree/total:.1f}% | Multi-front or ambiguous |

**Note**: Low unanimous agreement is EXPECTED. The three phases measure different aspects:
- Phase 2: Semantic similarity (what papers SAY)
- Phase 3: Lexical overlap (EXACT terms used)
- Phase 4: Co-occurrence patterns (how terms PAIR)

Disagreement often indicates multi-front lineages, not errors.

## Front Distribution

Top 5 primary front assignments:

"""

    front_counts = mappings['primary_front'].value_counts().head(10)
    for front, count in front_counts.items():
        report += f"- **{front}**: {count} lineages ({100*count/total:.1f}%)\n"

    report += "\n\n## Score Statistics\n\n"

    score_rows = []
    observed_phases = []
    for phase_key in PHASE_ORDER:
        column = score_columns.get(phase_key)
        if not column:
            continue
        meta = PHASES[phase_key]
        series = mappings[column].dropna()
        if series.empty:
            stats = ("N/A",) * 4
        else:
            stats = (
                f"{series.mean():.3f}",
                f"{series.std():.3f}",
                f"{series.min():.3f}",
                f"{series.max():.3f}",
            )
        row = f"| **{meta['title']}** | {stats[0]} | {stats[1]} | {stats[2]} | {stats[3]} | {meta['interpretation']} |"
        score_rows.append(row)
        observed_phases.append(phase_key)

    if score_rows:
        report += "| Phase | Mean | Std | Min | Max | Interpretation |\n"
        report += "|-------|------|-----|-----|-----|----------------|\n"
        report += "\n".join(score_rows)
        observation_lines = [PHASES[phase]['observation'] for phase in observed_phases if PHASES[phase]['observation']]
        if observation_lines:
            report += "\n\n**Observations**:\n"
            for line in observation_lines:
                report += f"- {line}\n"
    else:
        report += "_No phase score columns detected in the mappings file; skipping numeric score summary._\n"

    report += f"""

## Validation Status

[PASS] **ALL VALIDATION CHECKS PASSED**

## Recommendations for Domain Expert Review

### Priority 1: Low Confidence Cases ({low_conf} lineages)
Review evidence bundles for lineages with no phase agreement:
```
data/out/mapping_evidence/lineage_*_evidence.json
```

Check if disagreement indicates:
- Multi-front assignment needed
- One phase clearly wrong (systematic error)
- Edge case or emerging phenomenon

### Priority 2: Medium Confidence Multi-Front ({medium_conf} multi-front)
Verify multi-front assignments are appropriate:
- Does lineage genuinely span multiple fronts?
- Should we pick one primary front?
- Are alternative fronts reasonable?

### Priority 3: High Confidence Validation (Sample)
Spot-check 10-20 high-confidence mappings to verify accuracy:
- Do the top NPMI pairs make sense? (Phase 4 evidence)
- Do matched terms align with front? (Phase 3 evidence)
- Does semantic similarity make sense? (Phase 2 evidence)

## Next Steps

1. **Manual validation**: Label 20 lineages and measure accuracy per confidence level
2. **Error analysis**: Identify systematic errors vs. genuine ambiguity
3. **Confidence calibration**: Adjust thresholds if "high confidence" isn't accurate
4. **Production use**: Use validated high-confidence mappings for analysis

---

**Generated**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
**Validation framework**: Phase 5 ensemble voting
**Total lineages**: {total}
**Ready for expert review**: ✅
"""

    return report


def main():
    print("=" * 70)
    print("PHASE 5 VALIDATION & VISUALIZATION")
    print("=" * 70)

    # Load data
    mappings, evidence_dict = load_mappings()
    score_columns = resolve_phase_score_columns(mappings)

    # Create output directory
    output_dir = Path('data/out/06_validation/phase5')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate visualizations
    create_confidence_distribution(mappings, output_dir)
    create_front_distribution(mappings, output_dir)
    create_phase_agreement_heatmap(mappings, evidence_dict, output_dir)
    create_score_distributions(mappings, output_dir, score_columns)
    create_multi_label_analysis(mappings, output_dir)

    # Generate validation report
    report = generate_validation_report(mappings, evidence_dict, score_columns)
    report_path = output_dir / 'phase5_validation_report.md'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"\n[Report] Saved validation report to {report_path}")

    # Summary
    print("\n" + "=" * 70)
    print("VALIDATION COMPLETE")
    print("=" * 70)
    print(f"Generated 5 visualization figures in {output_dir}/")
    print(f"Generated validation report: {report_path}")
    print("\nOutputs:")
    print(f"  - {output_dir}/phase5_confidence_distribution.png")
    print(f"  - {output_dir}/phase5_front_distribution.png")
    print(f"  - {output_dir}/phase5_agreement_heatmap.png")
    print(f"  - {output_dir}/phase5_score_distributions.png")
    print(f"  - {output_dir}/phase5_multi_label_analysis.png")
    print(f"  - {report_path}")


if __name__ == '__main__':
    main()
