"""
Generate scorecard.md from scorecard.json results
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for scorecard markdown generation."""
    parser = argparse.ArgumentParser(description="Generate scorecard markdown from scorecard JSON.")
    parser.add_argument(
        "--scorecard",
        type=Path,
        default=REPO_ROOT / "data" / "out" / "scorecard.json",
        help="Path to the scorecard JSON file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output path for the markdown report (defaults to scorecard.md beside the input).",
    )
    return parser.parse_args()


def format_number(value, decimals=4):
    """Format a number for display."""
    if value is None:
        return "N/A"
    if isinstance(value, (int, float)):
        return f"{value:.{decimals}f}"
    return str(value)


def get_metric_value(metrics, key):
    """Safely get a metric value."""
    if metrics is None:
        return None
    return metrics.get(key)


def get_best_metrics(result):
    """Get the best metrics from either inflection or milestone."""
    inf_metrics = result.get('inflection_metrics')
    mil_metrics = result.get('milestone_metrics')

    # Prefer inflection metrics if available, otherwise milestone
    return inf_metrics if inf_metrics else mil_metrics


def main():
    args = parse_args()
    scorecard_path = args.scorecard.resolve()
    md_path = args.output.resolve() if args.output else scorecard_path.with_name("scorecard.md")

    # Load scorecard
    with open(scorecard_path) as f:
        scorecard = json.load(f)

    # Start markdown content
    md_lines = [
        "# Prediction Scorecard",
        "",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d')}  ",
        f"**Last Evaluated:** {scorecard['metadata']['last_evaluated']}  ",
        f"**Total Files:** {scorecard['metadata']['total_evaluated']}  ",
        f"**Successful Evaluations:** {scorecard['metadata']['successful_evaluations']}  ",
        "",
        "---",
        "",
        "Metrics: AP = Average Precision; PR-AUC = area under precision-recall curve; Spec = specificity.",
        "",
    ]

    # Categorize results (original categories)
    baselines = []
    msd_inflection_leakage_free = []
    stage2 = []
    time_forward = []
    msd_training_other = []
    archived = []
    missing_columns = []

    for result in scorecard['results']:
        exp = result['experiment']

        # Check for missing columns
        if result.get('missing_inflection_columns') or result.get('missing_milestone_columns'):
            missing_columns.append(result)
            continue

        if 'archive' in exp:
            archived.append(result)
        elif exp.startswith('baselines'):
            baselines.append(result)
        elif 'msd_inflection' in exp and ('leakage_free' in exp or exp.endswith('msd_inflection')):
            msd_inflection_leakage_free.append(result)
        elif 'stage2' in exp:
            stage2.append(result)
        elif 'time_forward' in exp:
            time_forward.append(result)
        elif 'msd_training' in exp or 'msd_' in exp or 'multi_signal' in exp:
            msd_training_other.append(result)
        else:
            msd_training_other.append(result)

    # Helper function to create table
    def create_table(results, title):
        """Create a markdown table for results."""
        lines = [
            f"## {title}",
            "",
        ]

        if not results:
            lines.append("*No results in this category*")
            lines.append("")
            return lines

        # Sort by PR-AUC descending
        sorted_results = sorted(
            results,
            key=lambda x: get_metric_value(get_best_metrics(x), 'pr_auc') or 0,
            reverse=True
        )

        # Create table header
        lines.extend([
            "| Experiment | Split | AP | PR-AUC | ROC-AUC | P | R | F1 | Spec | N | Pos | Warnings |",
            "|------------|-------|----|--------|---------|---|---|----|------|---|-----|----------|"
        ])

        # Add rows
        for result in sorted_results:
            exp_name = result['experiment'].replace('msd_training/', '').replace('baselines/', '')
            split_type = result.get('split_type', 'retrospective')
            warnings = ','.join(result.get('warnings', [])) if result.get('warnings') else ''

            # Use best available metrics
            m = get_best_metrics(result)
            if not m:
                continue

            row = f"| {exp_name} | {split_type} | "
            row += f"{format_number(m.get('average_precision'))} | "
            row += f"{format_number(m.get('pr_auc'))} | "
            row += f"{format_number(m.get('roc_auc'))} | "
            row += f"{format_number(m.get('precision'))} | "
            row += f"{format_number(m.get('recall'))} | "
            row += f"{format_number(m.get('f1_score'))} | "
            row += f"{format_number(m.get('specificity'))} | "
            row += f"{m.get('n_samples', 'N/A')} | "
            row += f"{m.get('n_positives', 'N/A')} | "
            row += f"{warnings} |"
            lines.append(row)

        lines.append("")
        return lines

    # Generate sections in original order
    if baselines:
        md_lines.extend(create_table(baselines, "Baseline Methods"))

    if msd_inflection_leakage_free:
        md_lines.extend(create_table(
            msd_inflection_leakage_free,
            "MSD Inflection - Leakage-Free Variants"
        ))

    if stage2:
        md_lines.extend(create_table(stage2, "Stage 2 Models"))

    if time_forward:
        md_lines.extend(create_table(time_forward, "Time-Forward Validation"))

    if msd_training_other:
        md_lines.extend(create_table(msd_training_other, "Other MSD Training Experiments"))

    if archived:
        md_lines.extend(create_table(archived, "Archived Experiments"))

    # Add files with missing columns
    if missing_columns:
        md_lines.extend([
            "---",
            "",
            "## Files with Missing Columns",
            "",
            "The following files are missing required prediction columns:",
            ""
        ])
        for result in missing_columns:
            md_lines.append(f"- **{result['experiment']}**")
            if result.get('missing_inflection_columns'):
                md_lines.append(f"  - Missing inflection columns: {', '.join(result['missing_inflection_columns'])}")
            if result.get('missing_milestone_columns'):
                md_lines.append(f"  - Missing milestone columns: {', '.join(result['missing_milestone_columns'])}")
        md_lines.append("")

    # Add performance insights
    md_lines.extend([
        "---",
        "",
        "## Key Insights",
        "",
        "### Top Performers (by PR-AUC)",
        "",
    ])

    # Find top performers
    all_with_metrics = [r for r in scorecard['results']
                       if get_best_metrics(r) is not None]

    if all_with_metrics:
        # Sort by PR-AUC
        top_performers = sorted(
            all_with_metrics,
            key=lambda x: get_metric_value(get_best_metrics(x), 'pr_auc') or 0,
            reverse=True
        )[:10]

        md_lines.append("**Top 10 Experiments:**")
        for i, r in enumerate(top_performers, 1):
            m = get_best_metrics(r)
            pr_auc = m['pr_auc']
            ap = m['average_precision']
            md_lines.append(f"{i}. `{r['experiment']}` - PR-AUC: {pr_auc:.4f}, AP: {ap:.4f}")
        md_lines.append("")

    # Add notes section
    md_lines.extend([
        "---",
        "",
        "## Notes",
        "",
        "- **Inflection** predictions identify when a technology begins rapid growth",
        "- **Milestone** predictions identify breakthrough moments (subset of inflections)",
        "- Perfect scores (1.0000) may indicate data leakage or overfitting on small test sets",
        "- **Split type**: prospective = time-forward/holdout validation, retrospective = in-sample or random CV",
        "- **Warnings**: Models that predict all negatives (precision/recall = 0)",
        "- Archived results are from snapshot: `inflection_s2_complete_20251108_143542`",
        "",
        "---",
        "",
        "*Scorecard data: [scorecard.json](scorecard.json)*  ",
        "*Generated by: `scripts/scorecard_evaluation.py`*  ",
        f"*Duplicates removed: {len(scorecard['metadata'].get('duplicates_removed', []))} experiments*"
    ])

    # Write markdown file
    md_path.parent.mkdir(parents=True, exist_ok=True)
    with open(md_path, 'w') as f:
        f.write('\n'.join(md_lines))

    print(f"Scorecard markdown generated: {md_path}")
    print(f"Total categories: {sum([bool(baselines), bool(msd_inflection_leakage_free), bool(stage2), bool(time_forward), bool(msd_training_other), bool(archived)])}")


if __name__ == '__main__':
    main()
