"""
Generate comprehensive baseline comparison figure for scientific reporting.

Creates a publication-quality multi-panel figure showing:
- PR and ROC curves side-by-side
- Performance metrics table
- Coverage and lag comparison
- Method characteristics summary
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec


def load_predictions(method_dir: Path) -> pd.DataFrame:
    """Load predictions CSV for a given method."""
    pred_path = method_dir / "breakthrough_predictions.csv"
    if not pred_path.exists():
        return None
    return pd.read_csv(pred_path)


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

    # Method display configuration
    method_config = {
        'msd_lightgbm': {
            'label': 'MSD (LightGBM)',
            'color': '#2E7D32',  # Dark green
            'marker': 's',
            'linestyle': '-',
            'linewidth': 2.5,
            'description': 'Multi-signal detector (52 features)'
        },
        'semantic_changepoint': {
            'label': 'Semantic Changepoint (PELT)',
            'color': '#1565C0',  # Dark blue
            'marker': '^',
            'linestyle': '--',
            'linewidth': 1.5,
            'description': 'PELT changepoint on embeddings'
        },
        'simple_heuristics': {
            'label': 'Simple Heuristics',
            'color': '#F57C00',  # Orange
            'marker': 'o',
            'linestyle': ':',
            'linewidth': 1.5,
            'description': 'Growth accel. + persistence'
        },
        'kleinberg_burst': {
            'label': 'Kleinberg Burst',
            'color': '#C62828',  # Dark red
            'marker': 'd',
            'linestyle': '-.',
            'linewidth': 1.5,
            'description': 'Citation burst detection'
        }
    }

    # Load all predictions
    predictions = {}
    for method_name in leaderboard.keys():
        if method_name == 'msd_lightgbm':
            method_dir = baselines_dir.parent / 'msd_training' / 'msd_inflection' / 'leakage_free'
        else:
            method_dir = baselines_dir / method_name
        preds = load_predictions(method_dir)
        if preds is not None:
            predictions[method_name] = preds

    # Create figure with custom layout
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.4,
                  left=0.08, right=0.96, top=0.94, bottom=0.06)

    # === Panel A: PR Curves ===
    ax_pr = fig.add_subplot(gs[0:2, 0])

    for method_name, config in method_config.items():
        if method_name not in leaderboard:
            continue
        metrics = leaderboard[method_name]

        if method_name in predictions:
            df = predictions[method_name]
            # Use appropriate score column
            if method_name == 'msd_lightgbm':
                score_col = 'probability'
            elif method_name == 'simple_heuristics':
                score_col = 'heuristic_score'
            elif method_name == 'kleinberg_burst':
                score_col = 'burst_level'
            elif method_name == 'semantic_changepoint':
                score_col = 'change_score'

            if score_col in df.columns:
                curves = compute_pr_roc_curves(
                    df['is_inflection_true'].astype(int),
                    df[score_col].fillna(0)
                )

                ax_pr.plot(curves['recall'], curves['precision'],
                          label=f"{config['label']} (AUC={metrics['pr_auc']:.3f})",
                          color=config['color'],
                          linestyle=config['linestyle'],
                          linewidth=config['linewidth'],
                          marker=config['marker'],
                          markevery=max(len(curves['recall'])//10, 1),
                          markersize=6,
                          alpha=0.9)

    ax_pr.set_xlabel('Recall', fontsize=11, fontweight='bold')
    ax_pr.set_ylabel('Precision', fontsize=11, fontweight='bold')
    ax_pr.set_title('A. Precision-Recall Curves', fontsize=12, fontweight='bold', loc='left')
    ax_pr.legend(loc='upper right', fontsize=9, framealpha=0.95)
    ax_pr.grid(True, alpha=0.3, linestyle='--')
    ax_pr.set_xlim([0, 1.05])
    ax_pr.set_ylim([0, 1.05])

    # Baseline region shading
    baseline_max = max([leaderboard[m]['pr_auc'] for m in leaderboard if m != 'msd_lightgbm'])
    ax_pr.axhline(y=baseline_max, color='gray', linestyle=':', alpha=0.5, linewidth=1)
    ax_pr.fill_between([0, 1], 0, baseline_max, alpha=0.1, color='red', label='Baseline ceiling')

    # === Panel B: ROC Curves ===
    ax_roc = fig.add_subplot(gs[0:2, 1])

    for method_name, config in method_config.items():
        if method_name not in leaderboard:
            continue
        metrics = leaderboard[method_name]

        if method_name in predictions:
            df = predictions[method_name]
            if method_name == 'msd_lightgbm':
                score_col = 'probability'
            elif method_name == 'simple_heuristics':
                score_col = 'heuristic_score'
            elif method_name == 'kleinberg_burst':
                score_col = 'burst_level'
            elif method_name == 'semantic_changepoint':
                score_col = 'change_score'

            if score_col in df.columns:
                curves = compute_pr_roc_curves(
                    df['is_inflection_true'].astype(int),
                    df[score_col].fillna(0)
                )

                ax_roc.plot(curves['fpr'], curves['tpr'],
                           label=f"{config['label']} (AUC={metrics['roc_auc']:.3f})",
                           color=config['color'],
                           linestyle=config['linestyle'],
                           linewidth=config['linewidth'],
                           marker=config['marker'],
                           markevery=max(len(curves['fpr'])//10, 1),
                           markersize=6,
                           alpha=0.9)

    ax_roc.plot([0, 1], [0, 1], 'k--', alpha=0.3, linewidth=1, label='Random')
    ax_roc.set_xlabel('False Positive Rate', fontsize=11, fontweight='bold')
    ax_roc.set_ylabel('True Positive Rate', fontsize=11, fontweight='bold')
    ax_roc.set_title('B. ROC Curves', fontsize=12, fontweight='bold', loc='left')
    ax_roc.legend(loc='lower right', fontsize=9, framealpha=0.95)
    ax_roc.grid(True, alpha=0.3, linestyle='--')
    ax_roc.set_xlim([0, 1.05])
    ax_roc.set_ylim([0, 1.05])

    # === Panel C: Performance Metrics Table ===
    ax_table = fig.add_subplot(gs[0:2, 2])
    ax_table.axis('off')

    # Prepare table data
    table_data = []
    headers = ['Method', 'PR-AUC', 'Precision', 'Recall', 'F1', 'Coverage']

    for method_name, config in method_config.items():
        if method_name not in leaderboard:
            continue
        m = leaderboard[method_name]

        # Use default metrics (at threshold 0)
        row = [
            config['label'].replace(' (LightGBM)', ''),
            f"{m['pr_auc']:.3f}",
            f"{m['default_precision']:.3f}",
            f"{m['default_recall']:.3f}",
            f"{m['default_f1']:.3f}",
            f"{m['detection_lag_coverage']:.3f}"
        ]
        table_data.append(row)

    # Create table
    table = ax_table.table(cellText=table_data, colLabels=headers,
                          cellLoc='center', loc='center',
                          colWidths=[0.25, 0.13, 0.13, 0.13, 0.13, 0.13])

    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2.2)

    # Style header row
    for i in range(len(headers)):
        cell = table[(0, i)]
        cell.set_facecolor('#E0E0E0')
        cell.set_text_props(weight='bold', fontsize=9)

    # Color-code rows by method
    for i, method_name in enumerate([m for m in method_config if m in leaderboard]):
        config = method_config[method_name]
        for j in range(len(headers)):
            cell = table[(i+1, j)]
            if method_name == 'msd_lightgbm':
                cell.set_facecolor('#C8E6C9')  # Light green for MSD
                cell.set_text_props(weight='bold')
            else:
                cell.set_facecolor('#FFFFFF')

        # Highlight best baseline metrics in light yellow
        if method_name != 'msd_lightgbm':
            for col_idx, col_name in enumerate(['pr_auc', 'default_precision', 'default_recall', 'default_f1', 'detection_lag_coverage']):
                val = leaderboard[method_name][col_name]
                baseline_vals = [leaderboard[m][col_name] for m in leaderboard if m != 'msd_lightgbm']
                if val == max(baseline_vals):
                    cell = table[(i+1, col_idx+1)]
                    cell.set_facecolor('#FFF9C4')  # Light yellow

    ax_table.set_title('C. Performance Metrics Summary', fontsize=12, fontweight='bold', loc='left', pad=20)

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
        lag_labels.append(config['label'].replace(' (LightGBM)', '').replace(' (PELT)', ''))
        lag_colors.append(config['color'])

    y_pos = np.arange(len(lag_labels))
    bars = ax_lag.barh(y_pos, lag_data, color=lag_colors, alpha=0.7, edgecolor='black', linewidth=0.5)

    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, lag_data)):
        ax_lag.text(val + 0.3 if val > 0 else val - 0.3,
                   i, f'{val:.0f}Q',
                   va='center', ha='left' if val > 0 else 'right',
                   fontsize=9, fontweight='bold')

    ax_lag.axvline(x=0, color='black', linestyle='-', linewidth=1, alpha=0.5)
    ax_lag.set_yticks(y_pos)
    ax_lag.set_yticklabels(lag_labels, fontsize=9)
    ax_lag.set_xlabel('Median Detection Lag (quarters)', fontsize=10, fontweight='bold')
    ax_lag.set_title('D. Detection Timing', fontsize=12, fontweight='bold', loc='left')
    ax_lag.grid(True, alpha=0.3, linestyle='--', axis='x')
    ax_lag.set_xlim([-10, 3])

    # Add early/late labels
    ax_lag.text(-9, len(lag_labels), 'Early\ndetection', ha='center', va='center',
               fontsize=8, style='italic', color='green', weight='bold',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.3))
    ax_lag.text(2, len(lag_labels), 'Late\ndetection', ha='center', va='center',
               fontsize=8, style='italic', color='red', weight='bold',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='lightcoral', alpha=0.3))

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
        cov_labels.append(config['label'].replace(' (LightGBM)', '').replace(' (PELT)', ''))
        cov_colors.append(config['color'])

    y_pos = np.arange(len(cov_labels))
    bars = ax_cov.barh(y_pos, cov_data, color=cov_colors, alpha=0.7, edgecolor='black', linewidth=0.5)

    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, cov_data)):
        ax_cov.text(val + 2, i, f'{val:.1f}%',
                   va='center', ha='left',
                   fontsize=9, fontweight='bold')

    ax_cov.set_yticks(y_pos)
    ax_cov.set_yticklabels(cov_labels, fontsize=9)
    ax_cov.set_xlabel('Coverage (%)', fontsize=10, fontweight='bold')
    ax_cov.set_title('E. Breakthrough Coverage', fontsize=12, fontweight='bold', loc='left')
    ax_cov.grid(True, alpha=0.3, linestyle='--', axis='x')
    ax_cov.set_xlim([0, 105])

    # === Panel F: Method Characteristics ===
    ax_char = fig.add_subplot(gs[2, 2])
    ax_char.axis('off')

    # Create characteristics text box
    char_text = "Method Characteristics:\n\n"

    for method_name, config in method_config.items():
        if method_name not in leaderboard:
            continue
        m = leaderboard[method_name]

        char_text += f"• {config['label']}:\n"
        char_text += f"  {config['description']}\n"
        char_text += f"  Coverage: {m['detection_lag_coverage']*100:.1f}% | "
        char_text += f"Lag: {m['detection_lag_median']:.0f}Q\n\n"

    # Add summary statistics
    baseline_pr_aucs = [leaderboard[m]['pr_auc'] for m in leaderboard if m != 'msd_lightgbm']
    msd_pr_auc = leaderboard['msd_lightgbm']['pr_auc']

    char_text += "\nKey Findings:\n"
    char_text += f"• MSD achieves {msd_pr_auc/max(baseline_pr_aucs):.1f}× higher\n"
    char_text += f"  PR-AUC than best baseline\n"
    char_text += f"• Baselines cluster at PR-AUC\n"
    char_text += f"  {min(baseline_pr_aucs):.3f}-{max(baseline_pr_aucs):.3f}\n"
    char_text += f"• Multi-signal approach essential\n"
    char_text += f"  for high precision/recall"

    ax_char.text(0.05, 0.95, char_text,
                transform=ax_char.transAxes,
                fontsize=8.5,
                verticalalignment='top',
                fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=0.8',
                         facecolor='#F5F5F5',
                         edgecolor='black',
                         linewidth=1))

    ax_char.set_title('F. Summary', fontsize=12, fontweight='bold', loc='left')

    # Overall figure title
    fig.suptitle('Baseline Method Comparison: Scientific Breakthrough Detection',
                fontsize=14, fontweight='bold', y=0.98)

    # Save figure
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(output_path.with_suffix('.pdf'), bbox_inches='tight', facecolor='white')
    print(f"Saved comprehensive comparison figure to {output_path}")
    print(f"Saved PDF version to {output_path.with_suffix('.pdf')}")

    plt.close()


def main():
    """Generate baseline comparison figures."""
    baselines_dir = Path("data/out/experiments/baselines")
    figures_dir = baselines_dir / "figures"

    leaderboard_path = figures_dir / "leaderboard.json"
    output_path = figures_dir / "comprehensive_comparison.png"

    if not leaderboard_path.exists():
        print(f"Error: Leaderboard not found at {leaderboard_path}")
        return

    create_comparison_figure(leaderboard_path, baselines_dir, output_path)
    print("\nFigure generation complete!")


if __name__ == "__main__":
    main()
