#!/usr/bin/env python3
"""
Generate publication-ready figures for research article.

Generates:
- Figure 1: System architecture (placeholder - requires manual design)
- Figure 2: PR/ROC curves (2-panel)
- Figure 3: Detection lag histogram
- Figure 4: Feature importance (top 15)
- S-Figure 1: Example lineage trajectories
- S-Figure 2: Threshold sweep analysis
- S-Figure 3: Confusion matrix
"""

import argparse
import json
from pathlib import Path

from _path_bootstrap import ensure_repo_imports

REPO_ROOT = ensure_repo_imports()

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve  # noqa: E402

from src.trusted_io import load_trusted_pickle  # noqa: E402

# Okabe-Ito color palette (color-blind friendly)
COLORS = {
    'msd': '#0173B2',
    'simple_heuristics': '#DE8F05',
    'kleinberg_burst': '#029E73',
    'semantic_changepoint': '#CC78BC',
    'neutral': '#4682B4',
}

# Publication settings
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'legend.fontsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for publication figure generation."""
    parser = argparse.ArgumentParser(description="Generate publication-ready figures from experiment artifacts.")
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=REPO_ROOT / "data" / "out",
        help="Base directory containing experiment outputs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "_local" / "psc" / "reporting" / "figures",
        help="Directory to write generated figures.",
    )
    parser.add_argument(
        "--allow-external-pickle",
        action="store_true",
        help="Allow loading model pickles from outside the repository root.",
    )
    return parser.parse_args()


def load_data(base: Path, *, allow_external_pickle: bool):
    """Load all required data sources."""
    # Leaderboard with PR-AUC values
    with open(base / 'experiments/baselines/figures_post_nov9/leaderboard.json') as f:
        leaderboard = json.load(f)

    # Main predictions with lag
    predictions = pd.read_csv(
        base / 'experiments/msd_training/msd_inflection/leakage_free/breakthrough_predictions.csv'
    )

    # Baseline predictions for curves
    baseline_preds = {}
    for method in ['simple_heuristics', 'kleinberg_burst', 'semantic_changepoint']:
        path = base / f'experiments/baselines/{method}/breakthrough_predictions.csv'
        if path.exists():
            baseline_preds[method] = pd.read_csv(path)

    # Model for feature importances
    model_path = base / 'experiments/msd_training/msd_inflection/leakage_free/breakthrough_detector_model.pkl'
    model = load_trusted_pickle(
        model_path,
        description="Publication figure model pickle",
        allow_external=allow_external_pickle,
    )

    # Feature names
    with open(base / 'experiments/msd_training/msd_inflection/leakage_free/feature_names.json') as f:
        feature_names = json.load(f)

    # Threshold sweep
    threshold_sweep = pd.read_csv(
        base / 'experiments/msd_training/msd_inflection/leakage_free/threshold_sweep.csv'
    )

    # Lineage timeseries
    timeseries = pd.read_csv(base / 'lineage_timeseries.csv')

    return {
        'leaderboard': leaderboard,
        'predictions': predictions,
        'baseline_preds': baseline_preds,
        'model': model,
        'feature_names': feature_names,
        'threshold_sweep': threshold_sweep,
        'timeseries': timeseries,
    }


def generate_figure2_pr_roc(data, output_dir):
    """Generate Figure 2: 2-panel PR and ROC curves with dual y-axes for scale."""
    print('Generating Figure 2: PR/ROC Curves...')

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Get MSD predictions
    preds = data['predictions']
    y_true = preds['is_inflection_true'].values
    y_score = preds['inflection_probability'].values

    # PR Curve - Panel A with dual y-axes
    ax = axes[0]
    ax2 = ax.twinx()  # Secondary y-axis for baselines

    # MSD on primary axis (left, 0.8-1.0 scale)
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    msd_pr_auc = data['leaderboard']['msd_production']['pr_auc']
    ax.plot(recall, precision, color=COLORS['msd'], linewidth=2,
            label=f'MSD (PR-AUC={msd_pr_auc:.3f})')

    # Baselines on secondary axis (right, 0-0.15 scale)
    baseline_lines = []
    baseline_labels = []
    for method, color_key in [('simple_heuristics', 'simple_heuristics'),
                               ('semantic_changepoint', 'semantic_changepoint')]:
        if method in data['baseline_preds']:
            bdf = data['baseline_preds'][method]
            prob_col = [c for c in bdf.columns if 'prob' in c.lower() or 'score' in c.lower()]
            if prob_col:
                by_true = bdf['is_inflection_true'].values if 'is_inflection_true' in bdf.columns else bdf['inflection_label'].values
                by_score = bdf[prob_col[0]].values
                bp, br, _ = precision_recall_curve(by_true, by_score)
                pr_auc = data['leaderboard'].get(method, {}).get('pr_auc', 0)
                line, = ax2.plot(br, bp, color=COLORS[color_key], linewidth=1.5,
                        linestyle='--')
                baseline_lines.append(line)
                baseline_labels.append(f'{method.replace("_", " ").title()} (PR-AUC={pr_auc:.3f})')

    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision (MSD)', color=COLORS['msd'])
    ax.tick_params(axis='y', labelcolor=COLORS['msd'])
    ax2.set_ylabel('Precision (Baselines)', color=COLORS['simple_heuristics'])
    ax2.tick_params(axis='y', labelcolor=COLORS['simple_heuristics'])
    ax.set_title('(A) Precision-Recall Curves')

    # Combined legend
    lines1, labels1 = ax.get_legend_handles_labels()
    ax.legend(lines1 + baseline_lines, labels1 + baseline_labels,
              loc='center right', fontsize=8)

    ax.set_xlim([0, 1])
    ax.set_ylim([0.7, 1.02])  # MSD scale
    ax2.set_ylim([0, 0.15])   # Baseline scale
    ax.grid(True, alpha=0.3)

    # ROC Curve - Panel B
    ax = axes[1]

    # MSD
    fpr, tpr, _ = roc_curve(y_true, y_score)
    msd_roc_auc = data['leaderboard']['msd_production']['roc_auc']
    ax.plot(fpr, tpr, color=COLORS['msd'], linewidth=2,
            label=f'MSD (ROC-AUC={msd_roc_auc:.3f})')

    # Baselines
    for method, color_key in [('simple_heuristics', 'simple_heuristics'),
                               ('kleinberg_burst', 'kleinberg_burst'),
                               ('semantic_changepoint', 'semantic_changepoint')]:
        if method in data['baseline_preds']:
            bdf = data['baseline_preds'][method]
            prob_col = [c for c in bdf.columns if 'prob' in c.lower() or 'score' in c.lower()]
            if prob_col:
                by_true = bdf['is_inflection_true'].values if 'is_inflection_true' in bdf.columns else bdf['inflection_label'].values
                by_score = bdf[prob_col[0]].values
                bfpr, btpr, _ = roc_curve(by_true, by_score)
                roc_auc = data['leaderboard'].get(method, {}).get('roc_auc', 0)
                ax.plot(bfpr, btpr, color=COLORS[color_key], linewidth=1.5,
                        label=f'{method.replace("_", " ").title()} (ROC-AUC={roc_auc:.3f})')

    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5, label='Random')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('(B) ROC Curves')
    ax.legend(loc='lower right', fontsize=9)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = output_dir / 'fig2_pr_roc_curves.png'
    plt.savefig(output_path)
    plt.close()
    print(f'  Saved: {output_path}')
    return output_path


def generate_figure3_lag(data, output_dir):
    """Generate Figure 3: Detection lag histogram."""
    print('Generating Figure 3: Detection Lag Distribution...')

    preds = data['predictions']

    # Filter to true positives at threshold 0.07
    tp_mask = (preds['is_inflection_true'] == 1) & (preds['inflection_probability'] >= 0.07)
    lags = preds.loc[tp_mask, 'detection_lag_quarters'].dropna()

    fig, ax = plt.subplots(figsize=(6, 4))

    # Histogram
    bins = np.arange(-10.5, 11.5, 1)
    ax.hist(lags, bins=bins, color=COLORS['msd'], edgecolor='black', alpha=0.8)

    # Annotations
    median_lag = lags.median()
    mean_lag = lags.mean()
    ax.axvline(median_lag, color='red', linestyle='--', linewidth=2,
               label=f'Median: {median_lag:.1f}Q')
    ax.axvline(mean_lag, color='darkred', linestyle=':', linewidth=2,
               label=f'Mean: {mean_lag:.2f}Q')

    ax.set_xlabel('Detection Lag (Quarters)')
    ax.set_ylabel('Count')
    ax.set_title('Detection Lag Distribution (True Positives)')
    ax.legend(loc='upper left')
    ax.set_xlim([-10, 10])
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    output_path = output_dir / 'fig3_lag_distribution.png'
    plt.savefig(output_path)
    plt.close()
    print(f'  Saved: {output_path}')
    return output_path


def generate_figure4_importance(data, output_dir):
    """Generate Figure 4: Feature importance bar chart (top 15)."""
    print('Generating Figure 4: Feature Importance...')

    # Extract feature importances from calibrated model
    model = data['model']
    clf = model.calibrated_classifiers_[0].estimator.named_steps['classifier']
    importances = clf.feature_importances_
    feature_names = data['feature_names']

    # Sort and get top 15
    indices = np.argsort(importances)[::-1][:15]
    top_features = [feature_names[i] for i in indices]
    top_importances = importances[indices]

    # Convert to percentages
    total = sum(importances)
    top_pct = 100 * top_importances / total

    fig, ax = plt.subplots(figsize=(8, 6))

    # Horizontal bar chart
    y_pos = np.arange(len(top_features))
    ax.barh(y_pos, top_pct, color=COLORS['neutral'], edgecolor='black', alpha=0.8)

    ax.set_yticks(y_pos)
    ax.set_yticklabels([f.replace('_', ' ').title() for f in top_features])
    ax.invert_yaxis()  # Top feature at top
    ax.set_xlabel('Feature Importance (%)')
    ax.set_title('Top 15 Features by Importance (LightGBM Gain)')

    # Add value labels
    for i, (pct, _feat) in enumerate(zip(top_pct, top_features)):
        ax.text(pct + 0.2, i, f'{pct:.1f}%', va='center', fontsize=9)

    ax.set_xlim([0, max(top_pct) * 1.15])
    ax.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    output_path = output_dir / 'fig4_feature_importance.png'
    plt.savefig(output_path)
    plt.close()
    print(f'  Saved: {output_path}')
    return output_path


def generate_sfig1_trajectories(data, output_dir):
    """Generate S-Figure 1: Example lineage trajectories."""
    print('Generating S-Figure 1: Example Trajectories...')

    preds = data['predictions']
    ts = data['timeseries']

    # Classify predictions at t=0.07
    threshold = 0.07
    preds['pred'] = (preds['inflection_probability'] >= threshold).astype(int)

    # Find examples of each type
    tp = preds[(preds['is_inflection_true'] == 1) & (preds['pred'] == 1)]['lineage_id'].unique()
    fp = preds[(preds['is_inflection_true'] == 0) & (preds['pred'] == 1)]['lineage_id'].unique()
    fn = preds[(preds['is_inflection_true'] == 1) & (preds['pred'] == 0)]['lineage_id'].unique()
    preds[(preds['is_inflection_true'] == 0) & (preds['pred'] == 0)]['lineage_id'].unique()

    # Select representative lineages
    examples = {
        'True Positive 1': tp[0] if len(tp) > 0 else None,
        'True Positive 2': tp[1] if len(tp) > 1 else None,
        'False Positive': fp[0] if len(fp) > 0 else None,
        'False Negative': fn[0] if len(fn) > 0 else None,
    }

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.flatten()

    for ax, (label, lineage_id) in zip(axes, examples.items()):
        if lineage_id is None:
            ax.text(0.5, 0.5, 'No example found', ha='center', va='center')
            ax.set_title(label)
            continue

        # Get timeseries for this lineage
        lineage_ts = ts[ts['lineage_id'] == lineage_id].sort_values('quarter')
        lineage_preds = preds[preds['lineage_id'] == lineage_id]

        # Calculate cumulative works
        lineage_ts['cumulative'] = lineage_ts['new_works'].cumsum()

        ax.plot(range(len(lineage_ts)), lineage_ts['cumulative'].values,
                color=COLORS['msd'], linewidth=2)
        ax.fill_between(range(len(lineage_ts)), 0, lineage_ts['cumulative'].values,
                        alpha=0.3, color=COLORS['msd'])

        # Mark inflection point if exists
        infl = lineage_preds[lineage_preds['is_inflection_true'] == 1]
        if len(infl) > 0:
            infl_q = infl.iloc[0]['quarter']
            q_idx = lineage_ts[lineage_ts['quarter'] == infl_q].index
            if len(q_idx) > 0:
                ax.axvline(lineage_ts.index.get_loc(q_idx[0]), color='red',
                           linestyle='--', linewidth=2, label='True inflection')

        ax.set_xlabel('Quarter')
        ax.set_ylabel('Cumulative Works')
        ax.set_title(f'{label} (Lineage {lineage_id})')
        ax.grid(True, alpha=0.3)
        handles, labels = ax.get_legend_handles_labels()
        if labels:
            ax.legend()

    plt.tight_layout()
    output_path = output_dir / 'sfig1_trajectories.png'
    plt.savefig(output_path)
    plt.close()
    print(f'  Saved: {output_path}')
    return output_path


def generate_sfig2_threshold(data, output_dir):
    """Generate S-Figure 2: Threshold sweep analysis."""
    print('Generating S-Figure 2: Threshold Sweep...')

    sweep = data['threshold_sweep']

    fig, ax = plt.subplots(figsize=(7, 5))

    ax.plot(sweep['threshold'], sweep['precision'], color=COLORS['msd'],
            linewidth=2, label='Precision')
    ax.plot(sweep['threshold'], sweep['recall'], color=COLORS['simple_heuristics'],
            linewidth=2, label='Recall')
    ax.plot(sweep['threshold'], sweep['f1'], color=COLORS['kleinberg_burst'],
            linewidth=2, label='F1 Score')

    # Mark selected threshold
    ax.axvline(0.07, color='red', linestyle='--', linewidth=1.5,
               label='Selected (t=0.07)')

    ax.set_xlabel('Decision Threshold')
    ax.set_ylabel('Score')
    ax.set_title('Precision, Recall, and F1 vs Threshold')
    ax.legend(loc='center right')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = output_dir / 'sfig2_threshold_sweep.png'
    plt.savefig(output_path)
    plt.close()
    print(f'  Saved: {output_path}')
    return output_path


def generate_sfig3_confusion(data, output_dir):
    """Generate S-Figure 3: Confusion matrix."""
    print('Generating S-Figure 3: Confusion Matrix...')

    preds = data['predictions']

    y_true = preds['is_inflection_true'].values
    y_pred = (preds['inflection_probability'] >= 0.07).astype(int).values

    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(6, 5))

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive'],
                annot_kws={'size': 14})

    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title('Confusion Matrix (t=0.07)')

    # Add metrics annotation
    tn, fp, fn, tp = cm.ravel()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    ax.text(1.5, -0.15, f'TP={tp:,}  FP={fp:,}  FN={fn:,}  TN={tn:,}',
            ha='center', va='top', fontsize=10, transform=ax.transAxes)
    ax.text(1.5, -0.25, f'Precision={precision:.1%}  Recall={recall:.1%}',
            ha='center', va='top', fontsize=10, transform=ax.transAxes)

    plt.tight_layout()
    output_path = output_dir / 'sfig3_confusion_matrix.png'
    plt.savefig(output_path)
    plt.close()
    print(f'  Saved: {output_path}')
    return output_path


def main():
    """Generate all publication figures."""
    args = parse_args()
    print('=' * 60)
    print('PUBLICATION FIGURE GENERATION')
    print('=' * 60)

    # Setup output directory
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print('\nLoading data...')
    data = load_data(args.base_dir.resolve(), allow_external_pickle=args.allow_external_pickle)
    print('Data loaded successfully.')

    # Generate figures
    print('\n--- Main Figures ---')
    generate_figure2_pr_roc(data, output_dir)
    generate_figure3_lag(data, output_dir)
    generate_figure4_importance(data, output_dir)

    print('\n--- Supplemental Figures ---')
    generate_sfig1_trajectories(data, output_dir)
    generate_sfig2_threshold(data, output_dir)
    generate_sfig3_confusion(data, output_dir)

    print('\n' + '=' * 60)
    print('Figure generation complete!')
    print(f'Output directory: {output_dir}')
    print('\nNote: Figure 1 (System Architecture) requires manual design')
    print('      using draw.io, Graphviz, or similar tool.')
    print('=' * 60)


if __name__ == '__main__':
    main()
