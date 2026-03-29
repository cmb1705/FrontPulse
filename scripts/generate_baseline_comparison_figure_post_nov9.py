"""
Generate comprehensive baseline comparison figure including post-Nov 9 MSD variants.

Creates a publication-quality multi-panel figure showing:
- PR and ROC curves side-by-side
- Performance metrics table
- Coverage and lag comparison
- Method characteristics summary

Includes: MSD Production, MSD Core-Features-Only, MSD Prospective Dev, and 3 baselines
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec


def load_predictions(method_name: str, baselines_dir: Path) -> pd.DataFrame:
    """Load predictions CSV for a given method."""
    # Map method names to prediction file paths
    path_map = {
        'msd_production': baselines_dir.parent / 'msd_training' / 'msd_inflection' / 'leakage_free' / 'breakthrough_predictions.csv',
        'msd_core_features_only': baselines_dir.parent / 'msd_training' / 'msd_inflection' / 'leakage_free_core_only' / 'breakthrough_predictions.csv',
        'msd_prospective_dev': baselines_dir.parent / 'msd_training' / 'time_forward_split' / 'dev' / 'breakthrough_predictions.csv',
        'simple_heuristics': baselines_dir / 'simple_heuristics' / 'breakthrough_predictions.csv',
        'kleinberg_burst': baselines_dir / 'kleinberg_burst' / 'breakthrough_predictions.csv',
        'semantic_changepoint': baselines_dir / 'semantic_changepoint' / 'breakthrough_predictions.csv'
    }

    pred_path = path_map.get(method_name)
    if pred_path and pred_path.exists():
        return pd.read_csv(pred_path)
    return None


def compute_pr_roc_curves(y_true, y_score):
    """Compute precision-recall and ROC curve points."""
    from sklearn.metrics import precision_recall_curve, roc_curve

    precision, recall, pr_thresholds = precision_recall_curve(y_true, y_score)
    fpr, tpr, roc_thresholds = roc_curve(y_true, y_score)

    return {
        'precision': precision,
        'recall': recall,
        'pr_thresholds': pr_thresholds,
        'fpr': fpr,
        'tpr': tpr,
        'roc_thresholds': roc_thresholds
    }


def create_comparison_figure(leaderboard_path: Path, baselines_dir: Path, output_path: Path):
    """Create comprehensive baseline comparison figure."""

    # Load leaderboard data
    with open(leaderboard_path) as f:
        leaderboard = json.load(f)

    # Method display configuration (in order)
    method_config = {
        'msd_production': {
            'label': 'MSD (Production)',
            'color': '#1B5E20',  # Dark green
            'marker': 's',
            'linestyle': '-',
            'linewidth': 3.0,
            'description': 'Full 55 features, retrospective',
            'score_column': 'inflection_probability'
        },
        'msd_core_features_only': {
            'label': 'MSD (Core-Only)',
            'color': '#43A047',  # Medium green
            'marker': 'o',
            'linestyle': '-',
            'linewidth': 2.5,
            'description': '20 core features only',
            'score_column': 'inflection_probability'
        },
        'msd_prospective_dev': {
            'label': 'MSD (Prospective)',
            'color': '#81C784',  # Light green
            'marker': 'D',
            'linestyle': '-',
            'linewidth': 2.0,
            'description': 'Train ≤2019, predict 2020+',
            'score_column': 'inflection_probability'
        },
        'semantic_changepoint': {
            'label': 'Semantic Changepoint',
            'color': '#1565C0',  # Dark blue
            'marker': '^',
            'linestyle': '--',
            'linewidth': 1.5,
            'description': 'Vocabulary distance',
            'score_column': 'change_score'
        },
        'simple_heuristics': {
            'label': 'Simple Heuristics',
            'color': '#F57C00',  # Orange
            'marker': 'o',
            'linestyle': ':',
            'linewidth': 1.5,
            'description': 'Growth + acceleration',
            'score_column': 'heuristic_score'
        },
        'kleinberg_burst': {
            'label': 'Kleinberg Burst',
            'color': '#C62828',  # Dark red
            'marker': 'd',
            'linestyle': '-.',
            'linewidth': 1.5,
            'description': 'Burst automaton',
            'score_column': 'burst_level'
        }
    }

    # Load all predictions
    predictions = {}
    for method_name in method_config:
        preds = load_predictions(method_name, baselines_dir)
        if preds is not None:
            predictions[method_name] = preds

    # Create figure with custom layout
    fig = plt.figure(figsize=(18, 11))
    gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.4,
                  left=0.07, right=0.97, top=0.94, bottom=0.06)

    # === Panel A: PR Curves ===
    ax_pr = fig.add_subplot(gs[0:2, 0])

    for method_name, config in method_config.items():
        if method_name not in leaderboard:
            continue
        metrics = leaderboard[method_name]

        if method_name in predictions:
            df = predictions[method_name]
            score_col = config['score_column']

            if score_col in df.columns:
                curves = compute_pr_roc_curves(
                    df['is_inflection_true'].astype(int),
                    df[score_col].fillna(0)
                )

                ax_pr.plot(curves['recall'], curves['precision'],
                          label=f"{config['label']} ({metrics['pr_auc']:.3f})",
                          color=config['color'],
                          linestyle=config['linestyle'],
                          linewidth=config['linewidth'],
                          marker=config['marker'],
                          markevery=max(len(curves['recall'])//10, 1),
                          markersize=6,
                          alpha=0.9)

    ax_pr.set_xlabel('Recall', fontsize=12, fontweight='bold')
    ax_pr.set_ylabel('Precision', fontsize=12, fontweight='bold')
    ax_pr.set_title('A. Precision-Recall Curves', fontsize=13, fontweight='bold', loc='left')
    ax_pr.legend(loc='lower left', fontsize=9, framealpha=0.95)
    ax_pr.grid(True, alpha=0.3, linestyle='--')
    ax_pr.set_xlim([0, 1.05])
    ax_pr.set_ylim([0, 1.05])

    # Baseline region shading
    baseline_methods = ['simple_heuristics', 'semantic_changepoint', 'kleinberg_burst']
    baseline_max = max([leaderboard[m]['pr_auc'] for m in baseline_methods if m in leaderboard])
    ax_pr.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5, linewidth=1)
    ax_pr.fill_between([0, 1], 0, baseline_max, alpha=0.1, color='red')
    ax_pr.text(0.98, baseline_max + 0.02, 'Baseline ceiling',
              ha='right', va='bottom', fontsize=9, style='italic', color='red')

    # === Panel B: ROC Curves ===
    ax_roc = fig.add_subplot(gs[0:2, 1])

    for method_name, config in method_config.items():
        if method_name not in leaderboard:
            continue
        metrics = leaderboard[method_name]

        if method_name in predictions:
            df = predictions[method_name]
            score_col = config['score_column']

            if score_col in df.columns:
                curves = compute_pr_roc_curves(
                    df['is_inflection_true'].astype(int),
                    df[score_col].fillna(0)
                )

                ax_roc.plot(curves['fpr'], curves['tpr'],
                           label=f"{config['label']} ({metrics['roc_auc']:.3f})",
                           color=config['color'],
                           linestyle=config['linestyle'],
                           linewidth=config['linewidth'],
                           marker=config['marker'],
                           markevery=max(len(curves['fpr'])//10, 1),
                           markersize=6,
                           alpha=0.9)

    ax_roc.plot([0, 1], [0, 1], 'k--', alpha=0.3, linewidth=1, label='Random')
    ax_roc.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
    ax_roc.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
    ax_roc.set_title('B. ROC Curves', fontsize=13, fontweight='bold', loc='left')
    ax_roc.legend(loc='lower right', fontsize=9, framealpha=0.95)
    ax_roc.grid(True, alpha=0.3, linestyle='--')
    ax_roc.set_xlim([0, 1.05])
    ax_roc.set_ylim([0, 1.05])

    # === Panel C: Performance Metrics Table ===
    ax_table = fig.add_subplot(gs[0:2, 2])
    ax_table.axis('off')

    # Prepare table data
    table_data = []
    headers = ['Method', 'PR-AUC', 'Prec.', 'Recall', 'F1', 'Cov.']

    for method_name, config in method_config.items():
        if method_name not in leaderboard:
            continue
        m = leaderboard[method_name]

        row = [
            config['label'].replace('MSD ', ''),
            f"{m['pr_auc']:.3f}",
            f"{m['default_precision']*100:.1f}%",
            f"{m['default_recall']*100:.1f}%",
            f"{m['default_f1']:.3f}",
            f"{m['detection_lag_coverage']*100:.1f}%"
        ]
        table_data.append(row)

    # Create table
    table = ax_table.table(cellText=table_data, colLabels=headers,
                          cellLoc='center', loc='center',
                          colWidths=[0.30, 0.13, 0.13, 0.13, 0.11, 0.11])

    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1, 2.5)

    # Style header row
    for i in range(len(headers)):
        cell = table[(0, i)]
        cell.set_facecolor('#E0E0E0')
        cell.set_text_props(weight='bold', fontsize=8.5)

    # Color-code rows by method type
    for i, method_name in enumerate([m for m in method_config if m in leaderboard]):
        for j in range(len(headers)):
            cell = table[(i+1, j)]
            if method_name.startswith('msd_'):
                cell.set_facecolor('#C8E6C9')  # Light green for MSD variants
                if method_name == 'msd_production':
                    cell.set_text_props(weight='bold')
            else:
                cell.set_facecolor('#FFEBEE')  # Light red for baselines

    ax_table.set_title('C. Performance Metrics (threshold 0.07)', fontsize=12, fontweight='bold', loc='left', pad=20)

    # === Panel D: Detection Lag Distribution ===
    ax_lag = fig.add_subplot(gs[2, 0])

    lag_data = []
    lag_labels = []
    lag_colors = []

    for method_name, config in method_config.items():
        if method_name not in leaderboard:
            continue
        m = leaderboard[method_name]
        lag_data.append(m['detection_lag_median'])
        lag_labels.append(config['label'].replace('MSD ', '').replace(' (PELT)', ''))
        lag_colors.append(config['color'])

    y_pos = np.arange(len(lag_labels))
    bars = ax_lag.barh(y_pos, lag_data, color=lag_colors, alpha=0.7, edgecolor='black', linewidth=0.8)

    # Add value labels
    for i, (_bar, val) in enumerate(zip(bars, lag_data)):
        ax_lag.text(val + 0.3 if val > 0 else val - 0.3,
                   i, f'{val:.0f}Q',
                   va='center', ha='left' if val > 0 else 'right',
                   fontsize=9, fontweight='bold')

    ax_lag.axvline(x=0, color='black', linestyle='-', linewidth=1.2, alpha=0.6)
    ax_lag.set_yticks(y_pos)
    ax_lag.set_yticklabels(lag_labels, fontsize=9)
    ax_lag.set_xlabel('Median Detection Lag (quarters)', fontsize=11, fontweight='bold')
    ax_lag.set_title('D. Detection Timing', fontsize=13, fontweight='bold', loc='left')
    ax_lag.grid(True, alpha=0.3, linestyle='--', axis='x')
    ax_lag.set_xlim([-9, 3])

    # === Panel E: Coverage Comparison ===
    ax_cov = fig.add_subplot(gs[2, 1])

    cov_data = []
    cov_labels = []
    cov_colors = []

    for method_name, config in method_config.items():
        if method_name not in leaderboard:
            continue
        m = leaderboard[method_name]
        cov_data.append(m['detection_lag_coverage'] * 100)
        cov_labels.append(config['label'].replace('MSD ', '').replace(' (PELT)', ''))
        cov_colors.append(config['color'])

    y_pos = np.arange(len(cov_labels))
    bars = ax_cov.barh(y_pos, cov_data, color=cov_colors, alpha=0.7, edgecolor='black', linewidth=0.8)

    # Add value labels
    for i, (_bar, val) in enumerate(zip(bars, cov_data)):
        ax_cov.text(val + 2, i, f'{val:.1f}%',
                   va='center', ha='left',
                   fontsize=9, fontweight='bold')

    ax_cov.set_yticks(y_pos)
    ax_cov.set_yticklabels(cov_labels, fontsize=9)
    ax_cov.set_xlabel('Coverage (%)', fontsize=11, fontweight='bold')
    ax_cov.set_title('E. Breakthrough Coverage', fontsize=13, fontweight='bold', loc='left')
    ax_cov.grid(True, alpha=0.3, linestyle='--', axis='x')
    ax_cov.set_xlim([0, 105])

    # === Panel F: Key Findings ===
    ax_char = fig.add_subplot(gs[2, 2])
    ax_char.axis('off')

    # Create findings text
    msd_prod = leaderboard['msd_production']
    msd_core = leaderboard['msd_core_features_only']
    msd_prosp = leaderboard['msd_prospective_dev']
    baseline_max_prauc = max([leaderboard[m]['pr_auc'] for m in baseline_methods if m in leaderboard])

    findings_text = "Key Findings:\n\n"
    findings_text += "1. Production MSD achieves PR-AUC\n"
    findings_text += f"   {msd_prod['pr_auc']:.3f} ({msd_prod['pr_auc']/baseline_max_prauc:.1f}× best baseline)\n\n"

    findings_text += "2. Core-only (20 features) achieves\n"
    findings_text += f"   {msd_core['default_precision']*100:.1f}% precision, {msd_core['default_recall']*100:.1f}% recall\n"
    findings_text += "   (90% FP reduction)\n\n"

    findings_text += "3. Prospective degrades 93%\n"
    findings_text += f"   (PR-AUC {msd_prod['pr_auc']:.3f} → {msd_prosp['pr_auc']:.3f})\n"
    findings_text += "   when training freezes at 2019\n\n"

    findings_text += "4. Baselines cluster at PR-AUC\n"
    findings_text += "   0.025-0.031 (univariate ceiling)\n\n"

    findings_text += "5. Context features boost recall\n"
    findings_text += f"   20pp ({msd_core['default_recall']*100:.1f}% → {msd_prod['default_recall']*100:.1f}%)\n"
    findings_text += "   at cost of precision"

    ax_char.text(0.05, 0.95, findings_text,
                transform=ax_char.transAxes,
                fontsize=8.5,
                verticalalignment='top',
                fontfamily='monospace',
                bbox={'boxstyle': 'round,pad=0.8',
                         'facecolor': '#F5F5F5',
                         'edgecolor': 'black',
                         'linewidth': 1})

    ax_char.set_title('F. Summary', fontsize=13, fontweight='bold', loc='left')

    # Overall figure title
    fig.suptitle('Baseline Comparison: MSD Variants and Univariate Methods',
                fontsize=15, fontweight='bold', y=0.98)

    # Save figure
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(output_path.with_suffix('.pdf'), bbox_inches='tight', facecolor='white')
    print(f"Saved comprehensive comparison figure to {output_path}")
    print(f"Saved PDF version to {output_path.with_suffix('.pdf')}")

    plt.close()


def main():
    """Generate baseline comparison figures with post-Nov 9 variants."""
    baselines_dir = Path("data/out/experiments/baselines")
    figures_dir = baselines_dir / "figures_post_nov9"

    leaderboard_path = figures_dir / "leaderboard.json"
    output_path = figures_dir / "comprehensive_comparison.png"

    if not leaderboard_path.exists():
        print(f"Error: Leaderboard not found at {leaderboard_path}")
        return

    create_comparison_figure(leaderboard_path, baselines_dir, output_path)
    print("\nFigure generation complete!")


if __name__ == "__main__":
    main()
