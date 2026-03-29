#!/usr/bin/env python3
"""
Comprehensive Inflection Detection Evaluation Visualizations

Generates publication-quality visualizations across 4 phases:
- Phase 1: Trust & Calibration
- Phase 2: Visual Examples (Growth Trajectories & Case Studies)
- Phase 3: Performance Context (PR/ROC, Operating Characteristics, Error Analysis)
- Phase 4: Advanced Analysis (Feature Space, Temporal Stability)
"""

import argparse
import json
import logging
import warnings
from pathlib import Path
from typing import Optional

import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from persistence_utils import ensure_persistence_column
from seasonality_utils import add_seasonal_context, attach_temporal_context
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

# Set publication-quality defaults
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['font.size'] = 9
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['grid.linewidth'] = 0.5
plt.rcParams['lines.linewidth'] = 1.5

# Color palette
COLORS = {
    'tp': '#2ecc71',      # Green
    'fp': '#e74c3c',      # Red
    'tn': '#95a5a6',      # Gray
    'fn': '#f39c12',      # Orange
    'milestone': '#f1c40f',  # Gold
    'organic': '#3498db',    # Blue
    'neutral': '#34495e',    # Dark gray
    'accent': '#9b59b6'      # Purple
}
TIER_COLORS = {
    'tier_1_high': '#b03a2e',
    'tier_2_medium': '#d35400',
    'tier_3_watch': '#7f8c8d'
}

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def load_data(predictions_path: Path, features_path: Path, timeseries_path: Path,
              labels_path: Path, threshold_sweep_path: Path, field_metrics_path: Path) -> dict[str, pd.DataFrame]:
    """Load all required datasets."""
    logger.info("Loading datasets...")

    data = {
        'predictions': pd.read_csv(predictions_path),
        'features': pd.read_csv(features_path),
        'timeseries': pd.read_csv(timeseries_path),
        'labels': pd.read_csv(labels_path),
        'threshold_sweep': pd.read_csv(threshold_sweep_path) if threshold_sweep_path.exists() else pd.DataFrame(),
        'field_metrics': load_field_metrics_table(field_metrics_path)
    }

    logger.info(f"  Predictions: {len(data['predictions']):,} rows")
    logger.info(f"  Features: {len(data['features']):,} rows")
    logger.info(f"  Timeseries: {len(data['timeseries']):,} rows")
    logger.info(f"  Labels: {len(data['labels']):,} rows")
    if not data['threshold_sweep'].empty:
        logger.info(f"  Threshold sweep: {len(data['threshold_sweep']):,} rows")
    else:
        logger.warning("  Threshold sweep file missing; will compute dynamically.")

    return data


def compute_threshold_sweep_from_predictions(
    predictions: pd.DataFrame,
    output_path: Optional[Path] = None,
    thresholds: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """Compute a simple threshold sweep when a precomputed file is unavailable."""
    logger.info("Generating threshold sweep from predictions...")
    thresholds = thresholds if thresholds is not None else np.linspace(0.01, 0.5, 50)
    y_true = predictions['is_inflection_true'].fillna(0).astype(int)
    total_actual = int((y_true == 1).sum())
    rows: list[dict[str, float]] = []
    for thresh in thresholds:
        preds_flag = (predictions['inflection_probability'] >= thresh).astype(int)
        tp = int(((preds_flag == 1) & (y_true == 1)).sum())
        fp = int(((preds_flag == 1) & (y_true == 0)).sum())
        fn = int(((preds_flag == 0) & (y_true == 1)).sum())
        tn = int(((preds_flag == 0) & (y_true == 0)).sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        fpr = fp / (fp + tn) if fp + tn else 0.0
        detections = predictions[
            (preds_flag == 1) &
            (predictions['is_inflection_true'] == 1) &
            (predictions['detection_lag_quarters'].notna())
        ]
        if not detections.empty:
            lag_median = float(detections['detection_lag_quarters'].median())
            lag_mean = float(detections['detection_lag_quarters'].mean())
            lag_coverage = len(detections) / total_actual if total_actual else 0.0
        else:
            lag_median = np.nan
            lag_mean = np.nan
            lag_coverage = 0.0
        rows.append({
            'threshold': thresh,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'fpr': fpr,
            'lag_median': lag_median,
            'lag_mean': lag_mean,
            'lag_coverage': lag_coverage,
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'tn': tn,
        })
    sweep_df = pd.DataFrame(rows)
    if output_path:
        sweep_df.to_csv(output_path, index=False)
        logger.info("Saved generated threshold sweep to %s", output_path)
    return sweep_df


def load_field_metrics_table(path: Path) -> pd.DataFrame:
    """Load field metrics table (csv/parquet) and normalize column names."""
    if not path.exists():
        logger.warning("Field metrics file %s not found; skipping field context visuals.", path)
        return pd.DataFrame()
    df = pd.read_parquet(path) if path.suffix == '.parquet' else pd.read_csv(path)
    df = df.copy()
    df['quarter'] = df['quarter'].astype(str)
    rename_map = {col: (col if col == 'quarter' else f"field_{col}") for col in df.columns}
    df.rename(columns=rename_map, inplace=True)
    return df


def compute_probability_xlim(prob_series: pd.Series, margin: float = 0.02) -> float:
    """Return a tighter x-maximum so plots focus on the populated probability range."""
    if prob_series.empty:
        return 1.0
    q_high = float(prob_series.quantile(0.995))
    upper = min(1.0, max(0.15, q_high + margin))
    return upper


def format_yoy_pct(value: float) -> str:
    """Utility for consistent YoY annotations."""
    if pd.isna(value):
        return "YoY N/A"
    sign = "+" if value >= 0 else ""
    return f"YoY {sign}{value:.0f}%"


def prepare_analysis_data(
    data: dict[str, pd.DataFrame],
    threshold: float = 0.07,
    persistence_window: int = 2,
) -> pd.DataFrame:
    """Merge and prepare data for analysis."""
    logger.info(f"Preparing analysis dataset at threshold {threshold}...")

    # Get predictions at threshold
    preds = data['predictions'].copy()
    required_cols = {'impact_score', 'impact_tier'}
    missing_cols = [c for c in required_cols if c not in preds.columns]
    if missing_cols:
        logger.warning("Predictions file missing %s; filling defaults for visualization.", ", ".join(missing_cols))
        if 'impact_score' not in preds.columns:
            preds['impact_score'] = np.nan
        if 'impact_tier' not in preds.columns:
            preds['impact_tier'] = 'tier_3_watch'
    preds['is_inflection_pred'] = (preds['inflection_probability'] >= threshold).astype(int)
    ensure_persistence_column(
        preds,
        threshold=threshold,
        window=persistence_window,
        column_name='is_inflection_pred_persistent'
    )

    # Classify outcomes
    preds['outcome'] = 'TN'
    preds.loc[(preds['is_inflection_true'] == 1) & (preds['is_inflection_pred'] == 1), 'outcome'] = 'TP'
    preds.loc[(preds['is_inflection_true'] == 0) & (preds['is_inflection_pred'] == 1), 'outcome'] = 'FP'
    preds.loc[(preds['is_inflection_true'] == 1) & (preds['is_inflection_pred'] == 0), 'outcome'] = 'FN'

    # Merge with features
    df = preds.merge(
        data['features'],
        on=['lineage_id', 'quarter'],
        how='left'
    )

    drop_cols = [c for c in df.columns if c == 'new_works' or c.startswith('new_works_')]
    if drop_cols:
        df = df.drop(columns=drop_cols)

    # Merge with enriched timeseries context (seasonality aware)
    df = attach_temporal_context(df, data['timeseries'])
    df['yoy_label'] = df.get('new_works_yoy_pct', pd.Series(dtype=float)).apply(format_yoy_pct)

    logger.info(f"  Analysis dataset: {len(df):,} rows")
    logger.info(f"  Outcomes: TP={sum(df.outcome=='TP')}, FP={sum(df.outcome=='FP')}, "
                f"FN={sum(df.outcome=='FN')}, TN={sum(df.outcome=='TN')}")

    return df


# ============================================================================
# PHASE 1: TRUST & CALIBRATION
# ============================================================================

def plot_confidence_distribution_analysis(df: pd.DataFrame, output_path: Path):
    """
    Phase 1.1: Confidence Distribution Analysis
    Shows how model distributes confidence scores across outcomes.

    NOTE: Does NOT evaluate calibration quality (would require held-out data).
    This is a descriptive analysis of the calibrated model's confidence patterns.
    """
    logger.info("Generating confidence distribution analysis...")

    fig = plt.figure(figsize=(14, 6))
    gs = gridspec.GridSpec(2, 2, height_ratios=[3, 1], width_ratios=[1, 1],
                          hspace=0.3, wspace=0.3)
    x_max = compute_probability_xlim(df['inflection_probability'])

    # Left top: Confidence distribution by outcome (histogram)
    ax1 = fig.add_subplot(gs[0, 0])

    for outcome, color in [('TP', COLORS['tp']), ('FP', COLORS['fp']), ('FN', COLORS['fn'])]:
        subset = df[df['outcome'] == outcome]['inflection_probability']
        if len(subset) > 0:
            ax1.hist(subset, bins=50, alpha=0.6, color=color, label=f'{outcome} (n={len(subset)})',
                    density=False, edgecolor='black', linewidth=0.5)

    ax1.axvline(0.07, color='black', linestyle='--', linewidth=2.5, alpha=0.7, label='Threshold (0.07)')
    ax1.set_xlabel('Confidence Score', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Count', fontsize=11, fontweight='bold')
    ax1.set_title('Confidence Distribution by Outcome', fontsize=12, fontweight='bold', pad=10)
    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_xlim(0, x_max)
    ax1.set_yscale('log')

    # Right top: Cumulative distribution
    ax2 = fig.add_subplot(gs[0, 1])

    for outcome, color in [('TP', COLORS['tp']), ('FP', COLORS['fp']), ('FN', COLORS['fn']), ('TN', COLORS['tn'])]:
        subset = df[df['outcome'] == outcome]['inflection_probability'].sort_values()
        if len(subset) > 0:
            cumulative = np.arange(1, len(subset) + 1) / len(subset)
            ax2.plot(subset, cumulative, color=color, linewidth=2, label=outcome, alpha=0.8)

    ax2.axvline(0.07, color='black', linestyle='--', linewidth=2.5, alpha=0.7)
    ax2.set_xlabel('Confidence Score', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Cumulative Fraction', fontsize=11, fontweight='bold')
    ax2.set_title('Cumulative Distribution by Outcome', fontsize=12, fontweight='bold', pad=10)
    ax2.legend(loc='lower right', fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, x_max)
    ax2.set_ylim(0, 1)

    # Bottom: Summary statistics table
    ax3 = fig.add_subplot(gs[1, :])
    ax3.axis('off')

    # Compute summary stats
    stats_data = [['Outcome', 'Count', 'Mean Confidence', 'Median Confidence', '% Above 0.07']]

    for outcome in ['TP', 'FP', 'FN', 'TN']:
        subset = df[df['outcome'] == outcome]['inflection_probability']
        if len(subset) > 0:
            mean_val = f'{subset.mean():.3f}'
            median_val = f'{subset.median():.3f}'
            pct_above = f'{100 * (subset >= 0.07).sum() / len(subset):.1f}%'
        else:
            mean_val = median_val = pct_above = 'N/A'
        stats_data.append([
            outcome,
            f'{len(subset):,}',
            mean_val,
            median_val,
            pct_above,
        ])

    table = ax3.table(cellText=stats_data, cellLoc='center', loc='center',
                     colWidths=[0.15, 0.15, 0.25, 0.25, 0.20])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.6)

    # Style header row
    for i in range(5):
        table[(0, i)].set_facecolor('#9b59b6')
        table[(0, i)].set_text_props(weight='bold', color='white')

    # Add color to outcome cells
    outcome_colors = {'TP': COLORS['tp'], 'FP': COLORS['fp'], 'FN': COLORS['fn'], 'TN': COLORS['tn']}
    for i, outcome in enumerate(['TP', 'FP', 'FN', 'TN'], start=1):
        table[(i, 0)].set_facecolor(outcome_colors[outcome])
        table[(i, 0)].set_text_props(weight='bold', color='white')

    # Impact tier breakdown (alerts only)
    alerts_only = df[df['is_inflection_pred'] == 1]
    if 'impact_tier' in alerts_only.columns and not alerts_only.empty:
        tier_order = ['tier_1_high', 'tier_2_medium', 'tier_3_watch']
        outcome_order = ['TP', 'FP', 'FN']
        summary_lines = ["Impact Tier Distribution (alerts only)"]
        for tier in tier_order:
            tier_df = alerts_only[alerts_only['impact_tier'] == tier]
            if tier_df.empty:
                summary_lines.append(f"{tier.replace('_', ' ').title()}: (no alerts)")
                continue
            counts = [f"{out}:{int((tier_df['outcome'] == out).sum())}" for out in outcome_order]
            summary_lines.append(
                f"{tier.replace('_', ' ').title()}: " + ", ".join(counts) +
                f", Total={len(tier_df)}"
            )
        ax3.text(
            0.02, -0.35, "\n".join(summary_lines),
            transform=ax3.transAxes,
            fontsize=9,
            ha='left',
            va='top',
            bbox={'boxstyle': 'round', 'facecolor': 'white', 'alpha': 0.9, 'edgecolor': COLORS['neutral']}
        )

    # Add warning note
    warning_text = """
⚠️  IMPORTANT: This shows the calibrated model's confidence distribution on the full dataset
(which includes training data). This is NOT a calibration quality assessment.

For true calibration quality, see cross-validated Brier score or calibration curves on held-out data.
This analysis describes HOW the model distributes confidence, not WHETHER those confidences are accurate.
    """

    ax3.text(0.5, -0.45, warning_text.strip(), ha='center', va='top',
             fontsize=8, style='italic', transform=ax3.transAxes,
             bbox={'boxstyle': 'round', 'facecolor': '#fff7c2', 'alpha': 0.5, 'pad': 0.4})

    plt.suptitle('Confidence Score Distribution Analysis\n(Calibrated Model on Full Dataset)',
                fontsize=13, fontweight='bold', y=0.98)

    fig.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()
    logger.info(f"  Saved to {output_path}")


def plot_confidence_vs_outcome(df: pd.DataFrame, output_path: Path):
    """
    Phase 1.2: Confidence vs Outcome Scatter
    Shows how confidence scores relate to prediction outcomes.
    """
    logger.info("Generating confidence vs outcome scatter...")

    fig = plt.figure(figsize=(14, 6))
    gs = gridspec.GridSpec(
        2, 2,
        height_ratios=[3, 1],
        width_ratios=[4, 1],
        hspace=0.08,
        wspace=0.15
    )
    x_max = compute_probability_xlim(df['inflection_probability'])

    # Main scatter plot
    ax_main = fig.add_subplot(gs[0, 0])

    # Add jitter to y for visualization
    np.random.seed(42)
    outcome_map = {'TN': 0, 'FP': 1, 'FN': 2, 'TP': 3}
    df['outcome_y'] = df['outcome'].map(outcome_map) + np.random.normal(0, 0.1, len(df))

    # Sample for visualization (too many points otherwise)
    sample_size = min(5000, len(df))
    df_sample = df.sample(n=sample_size, random_state=42)

    for outcome, _y_val, color in [('TP', 3, COLORS['tp']), ('FN', 2, COLORS['fn']),
                                    ('FP', 1, COLORS['fp']), ('TN', 0, COLORS['tn'])]:
        subset = df_sample[df_sample['outcome'] == outcome]
        ax_main.scatter(subset['inflection_probability'], subset['outcome_y'],
                       alpha=0.3, s=20, color=color, label=outcome, edgecolors='none')

    ax_main.axvline(0.07, color='black', linestyle='--', linewidth=2, alpha=0.7)
    ax_main.set_xlim(0, x_max)
    ax_main.set_ylim(-0.5, 3.5)
    ax_main.set_yticks([0, 1, 2, 3])
    ax_main.set_yticklabels(['TN', 'FP', 'FN', 'TP'], fontsize=10, fontweight='bold')
    ax_main.set_ylabel('Outcome', fontsize=11, fontweight='bold')
    ax_main.set_xlabel('Predicted probability', fontsize=11, fontweight='bold')
    ax_main.legend(loc='upper left', fontsize=9, ncol=2)
    ax_main.grid(True, alpha=0.3, axis='x')
    ax_main.set_title('Confidence Score vs Prediction Outcome', fontsize=12, fontweight='bold', pad=10)

    # Bottom: marginal histogram (confidence distribution)
    ax_bottom = fig.add_subplot(gs[1, 0], sharex=ax_main)
    ax_bottom.hist(df['inflection_probability'], bins=50, color=COLORS['neutral'],
                   alpha=0.6, edgecolor='black', linewidth=0.5)
    ax_bottom.axvline(0.07, color='black', linestyle='--', linewidth=2, alpha=0.7)
    ax_bottom.set_xlabel('Predicted probability', fontsize=11, fontweight='bold')
    ax_bottom.set_ylabel('Count', fontsize=9)
    ax_bottom.set_title('Distribution of predicted probabilities', fontsize=10, fontweight='bold')
    ax_bottom.grid(True, alpha=0.3, axis='y')

    # Right column: summary table
    ax_table = fig.add_subplot(gs[:, 1])
    ax_table.axis('off')
    summary_rows = [['Outcome', 'Count', 'Median Prob', '% ≥ 0.07']]
    for outcome in ['TP', 'FP', 'FN', 'TN']:
        subset = df[df['outcome'] == outcome]['inflection_probability']
        if subset.empty:
            summary_rows.append([outcome, '0', 'N/A', 'N/A'])
        else:
            summary_rows.append([
                outcome,
                f"{len(subset):,}",
                f"{subset.median():.3f}",
                f"{(subset >= 0.07).mean() * 100:.1f}%"
            ])

    table = ax_table.table(
        cellText=summary_rows,
        colWidths=[0.3, 0.2, 0.25, 0.25],
        cellLoc='center',
        loc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)
    for j in range(4):
        table[(0, j)].set_facecolor('#2c3e50')
        table[(0, j)].set_text_props(color='white', weight='bold')
    for row_idx, outcome in enumerate(['TP', 'FP', 'FN', 'TN'], start=1):
        table[(row_idx, 0)].set_facecolor(COLORS[outcome.lower()])
        table[(row_idx, 0)].set_text_props(color='white', weight='bold')

    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()
    logger.info(f"  Saved to {output_path}")


# ============================================================================
# PHASE 2: VISUAL EXAMPLES
# ============================================================================

def plot_growth_trajectory_fingerprints(df: pd.DataFrame, timeseries: pd.DataFrame,
                                       labels: pd.DataFrame, output_path: Path):
    """
    Phase 2.1: Growth Trajectory Fingerprints (4×3 grid)
    Shows actual growth patterns for TPs (organic/milestone), FPs, and FNs.

    Uses stratified selection to show representative patterns, not random samples.
    """
    logger.info("Generating growth trajectory fingerprints...")

    def ensure_cases(cases_df: pd.DataFrame, fallback_df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
        if len(cases_df) >= n:
            return cases_df.head(n)
        needed = n - len(cases_df)
        fallback = fallback_df if len(fallback_df) > 0 else cases_df
        if len(fallback) == 0:
            return cases_df
        extra = fallback.sample(
            n=needed,
            replace=len(fallback) < needed,
            random_state=seed
        )
        return pd.concat([cases_df, extra])

    tp_all = df[df['outcome'] == 'TP']
    fp_all = df[df['outcome'] == 'FP']
    fn_all = df[df['outcome'] == 'FN']

    # Row 1: TP Organic (3 cases with diverse confidence/magnitude)
    tp_organic = df[(df['outcome'] == 'TP') & (df['is_milestone_true'] == 0)]
    if len(tp_organic) >= 3:
        sorted_by_conf = tp_organic.sort_values('inflection_probability', ascending=False)
        tp_organic_cases = pd.concat([
            sorted_by_conf.iloc[[0]],
            sorted_by_conf.iloc[[len(sorted_by_conf)//2]],
            sorted_by_conf.iloc[[-1]]
        ])
    else:
        tp_organic_cases = tp_organic.sample(n=min(3, len(tp_organic)), random_state=42)

    # Row 2: TP Milestone (3 cases with diverse confidence/magnitude)
    tp_milestone = df[(df['outcome'] == 'TP') & (df['is_milestone_true'] == 1)]
    if len(tp_milestone) >= 3:
        sorted_by_conf = tp_milestone.sort_values('inflection_probability', ascending=False)
        tp_milestone_cases = pd.concat([
            sorted_by_conf.iloc[[0]],
            sorted_by_conf.iloc[[len(sorted_by_conf)//2]],
            sorted_by_conf.iloc[[-1]]
        ])
    elif len(tp_milestone) > 0:
        tp_milestone_cases = tp_milestone.copy()
        additional_organic = tp_organic[~tp_organic.index.isin(tp_organic_cases.index)]
        if len(additional_organic) > 0:
            tp_milestone_cases = pd.concat([
                tp_milestone_cases,
                additional_organic.sample(n=min(3 - len(tp_milestone), len(additional_organic)), random_state=43)
            ])
    else:
        additional_organic = tp_organic[~tp_organic.index.isin(tp_organic_cases.index)]
        tp_milestone_cases = additional_organic.sample(n=min(3, len(additional_organic)), random_state=43)

    tp_organic_cases = ensure_cases(tp_organic_cases, tp_all, 3, seed=50)
    tp_milestone_cases = ensure_cases(tp_milestone_cases, tp_all, 3, seed=51)

    # Row 3: False Positives (3 cases showing different failure modes)
    fp_df = df[df['outcome'] == 'FP']
    if len(fp_df) >= 3:
        sorted_by_conf = fp_df.sort_values('inflection_probability', ascending=False)
        fp_cases = pd.concat([
            sorted_by_conf.iloc[[0]],
            sorted_by_conf.iloc[[len(sorted_by_conf)//2]],
            sorted_by_conf.iloc[[-1]]
        ])
    else:
        fp_cases = fp_df.sample(n=min(3, len(fp_df)), random_state=44)
    fp_cases = ensure_cases(fp_cases, fp_all, 3, seed=52)

    # Row 4: False Negatives (3 cases showing what the model misses)
    fn_df = df[df['outcome'] == 'FN']
    if len(fn_df) >= 3:
        sorted_by_conf = fn_df.sort_values('inflection_probability', ascending=False)
        fn_cases = pd.concat([
            sorted_by_conf.iloc[[0]],
            sorted_by_conf.iloc[[len(sorted_by_conf)//2]],
            sorted_by_conf.iloc[[-1]]
        ])
    else:
        fn_cases = fn_df.sample(n=min(3, len(fn_df)), random_state=45)
    fn_cases = ensure_cases(fn_cases, fn_all if len(fn_all) > 0 else df[df['outcome'] == 'TP'], 3, seed=53)

    sections = [
        ('TP (Organic)', tp_organic_cases),
        ('TP (Milestone)', tp_milestone_cases),
        ('False Positives', fp_cases),
        ('False Negatives', fn_cases),
    ]
    sections = [(label, cases_df) for label, cases_df in sections if len(cases_df) > 0]

    if not sections:
        logger.warning("No cases available for growth trajectory fingerprints.")
        fig = plt.figure(figsize=(8, 3))
        fig.text(0.5, 0.5, "No cases available after filtering.", ha='center', va='center', fontsize=12)
        fig.savefig(output_path, bbox_inches='tight', dpi=300)
        plt.close(fig)
        return

    n_rows = len(sections)
    fig_height = max(6, 3 * n_rows + 2)
    fig, axes = plt.subplots(n_rows, 3, figsize=(14, fig_height))
    axes = np.atleast_2d(axes)
    fig.suptitle('Growth Trajectory Fingerprints (Persistent detections)', fontsize=14, fontweight='bold', y=0.995)

    export_rows = []
    legend_handles = []
    bar_color = '#a9c8f4'
    smooth_color = '#1864ab'

    for row_idx, (row_label, section_df) in enumerate(sections):
        row_axes = axes[row_idx]
        section_cases = list(section_df.head(3).iterrows())
        for col in range(3):
            ax = row_axes[col]
            if col >= len(section_cases):
                ax.axis('off')
                continue

            _, case = section_cases[col]

            lineage_history = timeseries[timeseries['lineage_id'] == case['lineage_id']].sort_values('quarter_timestamp')
            if lineage_history.empty:
                ax.axis('off')
                continue

            tier_label = case.get('impact_tier', 'tier_3_watch')
            tier_label_nice = tier_label.replace('_', ' ').title()
            tier_color = TIER_COLORS.get(tier_label, COLORS['neutral'])
            ax.set_facecolor(mcolors.to_rgba(tier_color, 0.08))

            # Plot QoQ bars
            ax.bar(lineage_history['quarter_numeric'], lineage_history['new_works'],
                   width=0.18, color=bar_color, alpha=0.8, label='New works / Q')

            # Rolling 4Q sum line (same axis)
            ax.plot(lineage_history['quarter_numeric'], lineage_history['new_works_rolling_4q_sum'],
                    color=smooth_color, linewidth=1.8, label='Rolling 4Q sum')
            ax.set_ylabel('New works / quarter', fontsize=8)
            ax.set_ylim(bottom=0)

            # Cumulative works on twin axis
            ax2 = ax.twinx()
            ax2.plot(lineage_history['quarter_numeric'], lineage_history['new_works_cumulative'],
                     color='black', linestyle='--', linewidth=1.2, label='Cumulative works')
            ax2.set_ylabel('Cumulative works', fontsize=8, color='black')
            ax2.tick_params(axis='y', labelcolor='black')

            if case['is_inflection_true'] == 1:
                case_quarter_num = lineage_history[lineage_history['quarter'] == case['quarter']]['quarter_numeric']
                case_new_works = lineage_history[lineage_history['quarter'] == case['quarter']]['new_works'].values
                if len(case_quarter_num) > 0 and len(case_new_works) > 0:
                    ax.axvline(case_quarter_num.iloc[0], color=COLORS['accent'], linestyle='--',
                               alpha=0.5, linewidth=1.2)
                    ax.scatter([case_quarter_num.iloc[0]], [case_new_works[0]],
                               color=COLORS['accent'], s=80, zorder=5, marker='D')

            if case['is_inflection_pred'] == 1:
                detection_slice = lineage_history[lineage_history['quarter'] == case['quarter']]
                if not detection_slice.empty:
                    det_q_num = detection_slice['quarter_numeric'].iloc[0]
                    det_new_works = detection_slice['new_works'].iloc[0]
                    outcome_color = COLORS[case['outcome'].lower()]
                    ax.scatter([det_q_num], [det_new_works],
                               color=outcome_color, s=140, zorder=6, marker='*',
                               edgecolors='black', linewidth=1)
                    yoy_value = detection_slice['new_works_yoy_pct'].iloc[0]
                    yoy_label = format_yoy_pct(yoy_value)
                    ax.annotate(
                        yoy_label,
                        xy=(det_q_num, det_new_works),
                        xytext=(det_q_num, det_new_works * 1.25 + 0.5),
                        fontsize=7,
                        ha='center',
                        color=outcome_color,
                        arrowprops={'arrowstyle': '-', 'color': outcome_color, 'alpha': 0.5, 'linewidth': 0.8}
                    )

            ax.set_xlabel('Year', fontsize=8)
            ax.grid(True, alpha=0.2, axis='y')
            ax.set_xlim(lineage_history['quarter_numeric'].min() - 0.1,
                        lineage_history['quarter_numeric'].max() + 0.1)

            impact_val = case.get('impact_score', np.nan)
            impact_str = f"{impact_val:.2f}" if pd.notna(impact_val) else "N/A"
            title = (
                f"{case['outcome']} | {tier_label_nice}\n"
                f"Prob: {case['inflection_probability']:.3f} | Impact: {impact_str}"
            )
            ax.set_title(title, fontsize=9, fontweight='bold', pad=5)

            if col == 0:
                ax.text(-0.25, 0.5, row_label, transform=ax.transAxes,
                        fontsize=10, fontweight='bold', va='center', rotation=90)

            export_slice = lineage_history[[
                'lineage_id', 'quarter', 'new_works', 'new_works_rolling_4q_mean',
                'new_works_rolling_4q_sum', 'new_works_cumulative', 'new_works_yoy_pct'
            ]].copy()
            export_slice['panel_row'] = row_idx + 1
            export_slice['panel_col'] = col + 1
            export_slice['outcome'] = case['outcome']
            export_rows.append(export_slice)

    if not legend_handles:
        legend_handles = [
            Rectangle((0, 0), 1, 1, facecolor=bar_color, alpha=0.8, label='New works / Q'),
            Line2D([0], [0], color=smooth_color, linewidth=1.8, label='Rolling 4Q sum'),
            Line2D([0], [0], color='black', linestyle='--', linewidth=1.2, label='Cumulative works')
        ]
    fig.legend(handles=legend_handles, loc='upper center', ncol=3, bbox_to_anchor=(0.5, 1.02), fontsize=9)

    plt.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()
    logger.info(f"  Saved to {output_path}")

    if export_rows:
        export_df = pd.concat(export_rows, ignore_index=True)
        csv_path = output_path.with_suffix('.csv')
        export_df.to_csv(csv_path, index=False)
        logger.info(f"  Saved panel data to {csv_path}")


def select_representative_cases(df: pd.DataFrame, timeseries: pd.DataFrame) -> list:
    """
    Select genuinely representative cases that tell a story about model behavior.

    Strategy:
    - TPs: Stratify by organic/milestone and confidence
    - FPs: Cluster by failure mode (volatile growth, end-of-series)
    - FNs: Cluster by failure mode (subtle transition, insufficient history)
    """
    import pandas as pd

    cases = []

    # Add temporal period classification
    df = df.copy()
    df['quarter_dt'] = pd.to_datetime(df['quarter'])
    df['year'] = df['quarter_dt'].dt.year
    df['period'] = pd.cut(df['year'], bins=[2000, 2010, 2017, 2024],
                         labels=['Early (2003-2010)', 'Middle (2011-2017)', 'Late (2018-2023)'])
    persistent_df = df[df['is_inflection_pred_persistent'] == 1].copy()

    # ===== TRUE POSITIVES (2 cases) =====
    # Goal: Show diversity of inflections the model catches correctly

    tp_df = persistent_df[persistent_df['outcome'] == 'TP'].copy()

    if len(tp_df) > 0:
        # Case 1: Organic, high confidence, middle period (most common scenario)
        organic_high = tp_df[
            (tp_df['is_milestone_true'] == 0) &
            (tp_df['inflection_probability'] > 0.5) &
            (tp_df['period'] == 'Middle (2011-2017)')
        ]
        if len(organic_high) > 0:
            # Select median confidence to be representative
            case1 = organic_high.iloc[(organic_high['inflection_probability'] -
                                      organic_high['inflection_probability'].median()).abs().argsort()[:1]]
            cases.append(('TP: Organic High-Conf', case1))
        else:
            # Fallback: any high-confidence TP
            high_conf = tp_df[tp_df['inflection_probability'] > 0.5]
            if len(high_conf) > 0:
                case1 = high_conf.sample(n=1, random_state=100)
                cases.append(('TP: High Confidence', case1))

        # Case 2: Borderline confidence (shows operating threshold)
        # Prefer milestone-linked if available to show milestone detection works
        borderline = tp_df[
            (tp_df['inflection_probability'] >= 0.07) &
            (tp_df['inflection_probability'] < 0.15)
        ]
        if len(borderline) > 0:
            # Prefer milestone-linked if exists
            milestone_borderline = borderline[borderline['is_milestone_true'] == 1]
            if len(milestone_borderline) > 0:
                case2 = milestone_borderline.sample(n=1, random_state=101)
                cases.append(('TP: Milestone Borderline', case2))
            else:
                case2 = borderline.sample(n=1, random_state=102)
                cases.append(('TP: Borderline', case2))

    # ===== FALSE POSITIVES (2 cases) =====
    # Goal: Diagnose what confuses the model

    fp_df = persistent_df[persistent_df['outcome'] == 'FP'].copy()

    if len(fp_df) >= 2:
        # Merge with timeseries to compute volatility
        fp_with_ts = []
        for idx, row in fp_df.iterrows():
            lineage_ts = timeseries[timeseries['lineage_id'] == row['lineage_id']].sort_values('quarter')
            if len(lineage_ts) >= 4:
                # Compute coefficient of variation (volatility measure)
                cv = lineage_ts['new_works'].std() / (lineage_ts['new_works'].mean() + 1e-6)
                fp_with_ts.append({
                    'index': idx,
                    'lineage_id': row['lineage_id'],
                    'quarter': row['quarter'],
                    'confidence': row['inflection_probability'],
                    'volatility': cv,
                    'year': row['year']
                })

        if len(fp_with_ts) >= 2:
            fp_analysis = pd.DataFrame(fp_with_ts)

            # Case 1: High volatility FP (model confused spike with inflection)
            high_vol_idx = fp_analysis['volatility'].idxmax()
            high_vol_case = fp_df.loc[fp_analysis.loc[high_vol_idx, 'index']:fp_analysis.loc[high_vol_idx, 'index']]
            cases.append(('FP: Volatile Growth', high_vol_case))

            # Case 2: Low volatility FP (different failure mode - maybe end-of-series or subtle pattern)
            remaining = fp_analysis[fp_analysis.index != high_vol_idx]
            if len(remaining) > 0:
                low_vol_idx = remaining['volatility'].idxmin()
                low_vol_case = fp_df.loc[remaining.loc[low_vol_idx, 'index']:remaining.loc[low_vol_idx, 'index']]
                cases.append(('FP: Non-Volatile', low_vol_case))
        else:
            # Fallback: random sample
            cases.append(('FP: Example 1', fp_df.sample(n=1, random_state=200)))
            if len(fp_df) > 1:
                remaining_fp = fp_df.drop(cases[-1][1].index)
                cases.append(('FP: Example 2', remaining_fp.sample(n=1, random_state=201)))
    elif len(fp_df) == 1:
        cases.append(('FP: Example', fp_df))

    # ===== FALSE NEGATIVES (2 cases) =====
    # Goal: Understand what the model misses

    fn_df = df[df['outcome'] == 'FN'].copy()

    if len(fn_df) >= 2:
        # Merge with timeseries to analyze history length
        fn_with_ts = []
        for idx, row in fn_df.iterrows():
            lineage_ts = timeseries[timeseries['lineage_id'] == row['lineage_id']].sort_values('quarter')
            # Count quarters before the inflection point
            quarters_before = len(lineage_ts[lineage_ts['quarter'] < row['quarter']])
            # Compute growth magnitude at inflection
            inflection_row = lineage_ts[lineage_ts['quarter'] == row['quarter']]
            if len(inflection_row) > 0:
                growth_at_inflection = inflection_row['new_works'].values[0]
            else:
                growth_at_inflection = 0

            fn_with_ts.append({
                'index': idx,
                'lineage_id': row['lineage_id'],
                'quarter': row['quarter'],
                'confidence': row['inflection_probability'],
                'quarters_before': quarters_before,
                'growth_magnitude': growth_at_inflection
            })

        fn_analysis = pd.DataFrame(fn_with_ts)

        # Case 1: Insufficient history (early-career FN)
        early_idx = fn_analysis['quarters_before'].idxmin()
        early_case = fn_df.loc[fn_analysis.loc[early_idx, 'index']:fn_analysis.loc[early_idx, 'index']]
        cases.append(('FN: Insufficient History', early_case))

        # Case 2: Subtle transition (low growth magnitude)
        remaining = fn_analysis[fn_analysis.index != early_idx]
        if len(remaining) > 0:
            subtle_idx = remaining['growth_magnitude'].idxmin()
            subtle_case = fn_df.loc[remaining.loc[subtle_idx, 'index']:remaining.loc[subtle_idx, 'index']]
            cases.append(('FN: Subtle Transition', subtle_case))
    elif len(fn_df) == 1:
        cases.append(('FN: Example', fn_df))

    return cases


def plot_case_study_montage(df: pd.DataFrame, timeseries: pd.DataFrame,
                            features: pd.DataFrame, output_path: Path,
                            persistence_window: int = 2):
    """
    Phase 2.2: Annotated Case Study Montage
    Detailed analysis of 6-8 representative cases with feature annotations.

    Uses stratified selection to show:
    - TPs: Diversity of correctly detected inflections (organic/milestone, high/borderline conf)
    - FPs: Different failure modes (volatile growth, end-of-series artifacts)
    - FNs: Different failure modes (insufficient history, subtle transitions)
    """
    logger.info("Generating case study montage...")

    cases_to_plot = select_representative_cases(df, timeseries)

    selected_cases: list[tuple[str, pd.Series]] = []
    for case_type, case_df in cases_to_plot:
        for _, case in case_df.iterrows():
            if len(selected_cases) >= 8:
                break
            selected_cases.append((case_type, case))
        if len(selected_cases) >= 8:
            break

    if not selected_cases:
        fig = plt.figure(figsize=(10, 4))
        ax = fig.add_subplot(111)
        ax.axis('off')
        ax.text(0.5, 0.5, 'No representative cases available', ha='center', va='center', fontsize=12, fontweight='bold')
        fig.savefig(output_path, bbox_inches='tight', dpi=300)
        plt.close(fig)
        return

    n_panels = len(selected_cases)
    fig_height = max(6, 2.8 * n_panels)
    fig = plt.figure(figsize=(16, fig_height))
    gs = gridspec.GridSpec(n_panels, 2, hspace=0.5, wspace=0.25)

    export_rows = []

    for plot_idx, (case_type, case) in enumerate(selected_cases):
        inner_gs = gridspec.GridSpecFromSubplotSpec(
            2, 1, subplot_spec=gs[plot_idx, :], height_ratios=[3.0, 1], hspace=0.1
        )
        ax_main = fig.add_subplot(inner_gs[0])
        ax_bar = fig.add_subplot(inner_gs[1], sharex=ax_main)

        lineage_history = timeseries[timeseries['lineage_id'] == case['lineage_id']].sort_values('quarter_timestamp')
        if lineage_history.empty:
            ax_main.axis('off')
            ax_bar.axis('off')
            continue

        tier_label = case.get('impact_tier', 'tier_3_watch')
        tier_label_nice = tier_label.replace('_', ' ').title()
        tier_color = TIER_COLORS.get(tier_label, COLORS['neutral'])
        ax_main.set_facecolor(mcolors.to_rgba(tier_color, 0.08))
        ax_bar.set_facecolor(mcolors.to_rgba(tier_color, 0.04))

        ax_main.plot(
            lineage_history['quarter_numeric'],
            lineage_history['new_works_log_cumulative'],
            color='#1f3a93',
            linewidth=2.2
        )
        if plot_idx == 0:
            ax_main.set_ylabel('log(1 + cumulative works)', fontsize=10, fontweight='bold')
        ax_main.grid(True, alpha=0.2)

        detection_slice = lineage_history[lineage_history['quarter'] == case['quarter']]
        outcome_color = COLORS.get(case['outcome'].lower(), COLORS['neutral'])
        if not detection_slice.empty:
            det_x = detection_slice['quarter_numeric'].iloc[0]
            det_y = detection_slice['new_works_log_cumulative'].iloc[0]
            ax_main.axvline(det_x, color=outcome_color, linestyle='--', linewidth=1.3, alpha=0.6)
            ax_main.scatter(det_x, det_y, color=outcome_color, s=120, marker='*',
                            edgecolors='black', linewidth=1.2, zorder=5)
            yoy_label = format_yoy_pct(detection_slice['new_works_yoy_pct'].iloc[0])
            ax_main.text(det_x, det_y + 0.15, yoy_label, fontsize=8, color=outcome_color, ha='center')

        if case['is_inflection_true'] == 1:
            gt_slice = lineage_history[lineage_history['quarter'] == case['quarter']]
            if not gt_slice.empty:
                ax_main.scatter(
                    gt_slice['quarter_numeric'],
                    lineage_history['new_works_log_cumulative'].loc[gt_slice.index],
                    color=COLORS['accent'],
                    s=60,
                    marker='D',
                    zorder=4
                )

        ax_main.set_xticklabels([])

        ax_bar.bar(
            lineage_history['quarter_numeric'],
            lineage_history['new_works'],
            color='#bcd8ff',
            width=0.18,
            alpha=0.9
        )
        ax_bar.plot(
            lineage_history['quarter_numeric'],
            lineage_history['new_works_rolling_4q_mean'],
            color='#1a6fb3',
            linewidth=1.5,
            label='Rolling 4Q mean'
        )
        if plot_idx == n_panels - 1:
            ax_bar.set_ylabel('New works/Q', fontsize=9)
        ax_bar.set_xlabel('Year', fontsize=10, fontweight='bold')
        ax_bar.set_ylim(bottom=0)
        ax_bar.grid(True, axis='y', alpha=0.2)

        impact_val = case.get('impact_score', np.nan)
        impact_str = f"{impact_val:.2f}" if pd.notna(impact_val) else "N/A"

        info_lines = [
            f"Lineage ID: {case['lineage_id']}",
            f"Quarter: {case['quarter']}",
            f"Confidence: {case['inflection_probability']:.3f}",
            f"Impact Score: {impact_str}",
            f"Impact Tier: {tier_label_nice}",
            f"Outcome: {case['outcome']} ({case_type})",
        ]
        if 'detection_lag_quarters' in case and not pd.isna(case['detection_lag_quarters']):
            lag_value = int(case['detection_lag_quarters'])
            info_lines.append(f"Detection Lag: {lag_value}Q")
        else:
            info_lines.append("Detection Lag: N/A")
        classification = "Milestone-linked" if case.get('is_milestone_true', 0) else "Organic"
        info_lines.append(f"Type: {classification}")
        yoy_text = format_yoy_pct(detection_slice['new_works_yoy_pct'].iloc[0]) if not detection_slice.empty else "YoY N/A"
        info_lines.append(f"YoY Trend: {yoy_text}")
        persistence_flag = "Yes" if case.get('is_inflection_pred_persistent', 0) else "No"
        info_lines.append(f"Persistent ≥{persistence_window}Q: {persistence_flag}")
        ax_main.text(
            0.02, 0.98, "\n".join(info_lines), transform=ax_main.transAxes,
            fontsize=9, va='top', ha='left', family='monospace',
            bbox={'boxstyle': 'round', 'facecolor': 'white', 'alpha': 0.96,
                      'edgecolor': tier_color, 'linewidth': 2}
        )

        export_slice = lineage_history[[
            'lineage_id', 'quarter', 'new_works', 'new_works_rolling_4q_mean',
            'new_works_cumulative', 'new_works_log_cumulative', 'new_works_yoy_pct'
        ]].copy()
        export_slice['case_label'] = case_type
        export_slice['panel_index'] = plot_idx + 1
        export_slice['impact_score'] = impact_val
        export_slice['impact_tier'] = tier_label
        export_rows.append(export_slice)

    plt.suptitle('Detailed Case Studies (Cumulative focus)', fontsize=16, fontweight='bold', y=0.995)
    fig.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()
    logger.info(f"  Saved to {output_path}")

    if export_rows:
        export_df = pd.concat(export_rows, ignore_index=True)
        csv_path = output_path.with_suffix('.csv')
        export_df.to_csv(csv_path, index=False)
        logger.info(f"  Saved case-study data to {csv_path}")


def plot_field_vs_lineage_panel(
    df: pd.DataFrame,
    field_metrics: pd.DataFrame,
    output_path: Path,
    threshold: float,
) -> None:
    """Phase 2 addon: contextualize detections against field-wide growth."""
    if field_metrics.empty:
        logger.warning("Field metrics missing; skipping field vs lineage panel.")
        return

    detection_counts = df[
        (df['inflection_probability'] >= threshold) &
        (df['is_inflection_pred'] == 1)
    ].groupby('quarter').size().rename('detections').reset_index()

    timeline = field_metrics.merge(detection_counts, on='quarter', how='left')
    timeline['detections'] = timeline['detections'].fillna(0)
    timeline = timeline.sort_values('quarter')
    timeline['quarter_dt'] = pd.PeriodIndex(timeline['quarter'], freq='Q').to_timestamp()

    try:
        timeline['field_growth_bucket'] = pd.qcut(
            timeline['field_total_new_works'].rank(method='first'),
            q=4,
            labels=['Q1 (low)', 'Q2', 'Q3', 'Q4 (high)'],
        )
    except ValueError:
        timeline['field_growth_bucket'] = 'All'

    bucket_summary = timeline.groupby('field_growth_bucket', observed=False).agg(
        detections=('detections', 'sum'),
        quarters=('quarter', 'count'),
        median_field=('field_total_new_works', 'median'),
    ).reset_index()
    total_detections = bucket_summary['detections'].sum()
    if total_detections == 0:
        bucket_summary['detection_share'] = 0.0
    else:
        bucket_summary['detection_share'] = bucket_summary['detections'] / total_detections

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), height_ratios=[3, 1.4])
    x = np.arange(len(timeline))
    ax1.bar(x, timeline['field_total_new_works'], color='#95a5a6', label='Field Output')
    ax1.set_ylabel('Field New Works', fontsize=11, fontweight='bold')
    ax1.set_title('Field Production vs MSD Detections', fontsize=13, fontweight='bold', pad=12)
    ax1.grid(alpha=0.2, axis='y')
    ax1b = ax1.twinx()
    ax1b.plot(x, timeline['detections'], color='#8e44ad', linewidth=2, label='Detections')
    ax1b.set_ylabel('# Detections', color='#8e44ad', fontsize=11, fontweight='bold')
    ax1b.tick_params(axis='y', labelcolor='#8e44ad')
    tick_idx = np.linspace(0, len(x) - 1, min(12, len(x))).astype(int) if len(x) else []
    ax1.set_xticks(tick_idx)
    ax1.set_xticklabels(timeline['quarter'].iloc[tick_idx] if len(tick_idx) else [], rotation=45, ha='right')
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1b.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc='upper left')

    ax2.bar(bucket_summary['field_growth_bucket'].astype(str), bucket_summary['detection_share'] * 100, color='#2c3e50')
    ax2.set_ylabel('% of Detections', fontsize=11, fontweight='bold')
    ax2.set_title('Detection Share by Field Growth Regime', fontsize=12, fontweight='bold', pad=10)
    ax2.set_ylim(0, max(5, bucket_summary['detection_share'].max() * 120 if not bucket_summary.empty else 5))
    for idx, row in bucket_summary.iterrows():
        ax2.text(
            idx,
            (row['detection_share'] * 100) + 0.5,
            f"{row['detection_share'] * 100:.1f}%\n(n={int(row['detections'])})",
            ha='center',
            va='bottom',
            fontsize=9,
        )
    ax2.grid(alpha=0.2, axis='y')
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    logger.info(f"Saved Phase 2 field context panel to {output_path}")


# ============================================================================
# PHASE 3: PERFORMANCE CONTEXT
# ============================================================================

def plot_cv_performance_summary(eval_metrics_path: Path, threshold_sweep: pd.DataFrame, output_path: Path):
    """
    Phase 3.1: Cross-Validated Performance Summary
    Shows true generalization metrics from CV folds + calibrated model performance.

    IMPORTANT: Does NOT compute PR/ROC on full dataset (which includes training data).
    Instead references CV metrics from evaluation_metrics.json.
    """
    logger.info("Generating CV performance summary...")

    # Load CV metrics
    with open(eval_metrics_path) as f:
        cv_metrics = json.load(f)

    fig = plt.figure(figsize=(14, 10))
    gs = gridspec.GridSpec(3, 2, hspace=0.4, wspace=0.3)

    # Panel 1: CV Performance Bars
    ax1 = fig.add_subplot(gs[0, :])

    metrics = ['PR-AUC', 'ROC-AUC', 'Precision', 'Recall', 'F1']
    cv_values = [
        cv_metrics['cv_pr_auc_mean'],
        cv_metrics['cv_roc_auc_mean'],
        cv_metrics['cv_precision_mean'],
        cv_metrics['cv_recall_mean'],
        cv_metrics['cv_f1_mean']
    ]
    cv_stds = [
        cv_metrics['cv_pr_auc_std'],
        cv_metrics['cv_roc_auc_std'],
        cv_metrics['cv_precision_std'],
        cv_metrics['cv_recall_std'],
        cv_metrics['cv_f1_std']
    ]

    x_pos = np.arange(len(metrics))
    ax1.bar(x_pos, cv_values, yerr=cv_stds, color=COLORS['accent'], alpha=0.7,
           capsize=10, edgecolor='black', linewidth=1.5)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(metrics, fontsize=11, fontweight='bold')
    ax1.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax1.set_title('Cross-Validated Performance (5-Fold, Pre-Calibration)',
                 fontsize=13, fontweight='bold', pad=10)
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_ylim(0, 1)

    # Add value labels on bars
    for i, (v, std) in enumerate(zip(cv_values, cv_stds)):
        ax1.text(i, v + std + 0.02, f'{v:.3f}±{std:.3f}',
                ha='center', fontsize=9, fontweight='bold')

    # Panel 2: Calibrated Model Performance (at threshold 0.07)
    ax2 = fig.add_subplot(gs[1, 0])

    calib_metrics = ['Precision', 'Recall', 'F1', 'FPR']
    calib_values = [
        cv_metrics['precision_threshold'],
        cv_metrics['recall_threshold'],
        cv_metrics['f1_threshold'],
        cv_metrics['fpr_threshold']
    ]

    colors = [COLORS['tp'], COLORS['organic'], COLORS['accent'], COLORS['fp']]
    ax2.barh(calib_metrics, calib_values, color=colors, alpha=0.7,
            edgecolor='black', linewidth=1.5)
    ax2.set_xlabel('Score', fontsize=11, fontweight='bold')
    ax2.set_title('Calibrated Model @ Threshold 0.07\n(Full Dataset, Post-Calibration)',
                 fontsize=11, fontweight='bold', pad=10)
    ax2.grid(True, alpha=0.3, axis='x')
    ax2.set_xlim(0, 1)

    for i, v in enumerate(calib_values):
        ax2.text(v + 0.02, i, f'{v:.3f}', va='center', fontsize=9, fontweight='bold')

    # Panel 3: Comparison table
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.axis('off')

    comparison_data = [
        ['Metric', 'CV (5-fold)', 'Calibrated (0.07)'],
        ['PR-AUC', f'{cv_metrics["cv_pr_auc_mean"]:.3f}', 'N/A*'],
        ['ROC-AUC', f'{cv_metrics["cv_roc_auc_mean"]:.3f}', 'N/A*'],
        ['Precision', f'{cv_metrics["cv_precision_mean"]:.3f}', f'{cv_metrics["precision_threshold"]:.3f}'],
        ['Recall', f'{cv_metrics["cv_recall_mean"]:.3f}', f'{cv_metrics["recall_threshold"]:.3f}'],
        ['F1', f'{cv_metrics["cv_f1_mean"]:.3f}', f'{cv_metrics["f1_threshold"]:.3f}'],
        ['FPR', 'N/A', f'{cv_metrics["fpr_threshold"]:.4f}']
    ]

    table = ax3.table(cellText=comparison_data, cellLoc='center', loc='center',
                     colWidths=[0.3, 0.35, 0.35])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)

    # Style header row
    for i in range(3):
        table[(0, i)].set_facecolor('#9b59b6')
        table[(0, i)].set_text_props(weight='bold', color='white')

    ax3.set_title('Performance Comparison', fontsize=11, fontweight='bold', pad=20)

    # Panel 4: Key notes
    ax4 = fig.add_subplot(gs[2, :])
    ax4.axis('off')

    notes_text = """
KEY POINTS:

1. Cross-Validated Metrics (Left Column):
   - Computed on 5-fold CV with held-out test sets
   - Reflects true generalization performance
   - PR-AUC 0.073 and ROC-AUC 0.798 are reliable estimates
   - Low precision/recall at default 0.5 threshold (class imbalance)

2. Calibrated Model Metrics (Right Column):
   - Computed on full dataset after isotonic calibration
   - Optimized threshold (0.07) dramatically improves precision/recall
   - Cannot compute AUC on full dataset (would include training data)
   - FPR 0.48% shows effective discrimination at operating point

3. Why the Difference?
   - CV uses default threshold (0.5), calibrated model uses optimized threshold (0.07)
   - Isotonic calibration improves probability estimates
   - Severe class imbalance (538 inflections / 21,608 quarters) benefits from threshold tuning

* AUC metrics for calibrated model omitted to avoid evaluating on training data.
  CV metrics provide the true generalization performance.
    """

    ax4.text(0.05, 0.95, notes_text, transform=ax4.transAxes, fontsize=9,
            verticalalignment='top', family='monospace',
            bbox={'boxstyle': 'round', 'facecolor': 'wheat', 'alpha': 0.3})

    plt.suptitle('Model Performance: Cross-Validation vs Calibrated Deployment',
                fontsize=14, fontweight='bold', y=0.98)

    fig.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()
    logger.info(f"  Saved to {output_path}")


def plot_operating_characteristics(threshold_sweep: pd.DataFrame, output_path: Path):
    """
    Phase 3.2: Operating Characteristics Dashboard (2×2)
    Shows operational tradeoffs as threshold varies.
    """
    logger.info("Generating operating characteristics dashboard...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Compute alert volume upfront
    threshold_sweep = threshold_sweep.copy()
    if threshold_sweep.empty:
        logger.warning("Threshold sweep empty; skipping operating characteristics dashboard.")
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.axis('off')
        ax.text(0.5, 0.5, 'Threshold sweep unavailable', ha='center', va='center', fontsize=12, fontweight='bold')
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        return
    threshold_sweep['alert_volume'] = threshold_sweep['tp'] + threshold_sweep['fp']

    # Mark threshold 0.07
    if (threshold_sweep['threshold'] == 0.07).any():
        row_07 = threshold_sweep[threshold_sweep['threshold'] == 0.07].iloc[0]
    else:
        closest_idx = (threshold_sweep['threshold'] - 0.07).abs().idxmin()
        row_07 = threshold_sweep.loc[closest_idx]

    # Panel 1: Precision vs Recall
    ax = axes[0, 0]
    ax.plot(threshold_sweep['recall'], threshold_sweep['precision'],
           color=COLORS['accent'], linewidth=2.5, marker='o', markersize=5)
    ax.scatter([row_07['recall']], [row_07['precision']],
              color='red', s=250, zorder=5, marker='D', edgecolors='black', linewidth=2)
    ax.annotate(f'0.07\n(P={row_07["precision"]:.2f}, R={row_07["recall"]:.2f})',
               xy=(row_07['recall'], row_07['precision']), xytext=(10, -20),
               textcoords='offset points', fontsize=9, fontweight='bold',
               bbox={'boxstyle': 'round', 'facecolor': 'white', 'alpha': 0.9},
               arrowprops={'arrowstyle': '->', 'color': 'red', 'linewidth': 2})

    ax.set_xlabel('Recall', fontsize=11, fontweight='bold')
    ax.set_ylabel('Precision', fontsize=11, fontweight='bold')
    ax.set_title('Precision-Recall Tradeoff', fontsize=12, fontweight='bold', pad=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)

    # Panel 2: Alert Volume vs Precision
    ax = axes[0, 1]
    ax.plot(threshold_sweep['precision'], threshold_sweep['alert_volume'],
           color=COLORS['fp'], linewidth=2.5, marker='o', markersize=5)

    ax.scatter([row_07['precision']], [row_07['alert_volume']],
              color='red', s=250, zorder=5, marker='D', edgecolors='black', linewidth=2)
    ax.annotate(f'0.07\n({row_07["alert_volume"]} alerts)',
               xy=(row_07['precision'], row_07['alert_volume']), xytext=(10, 20),
               textcoords='offset points', fontsize=9, fontweight='bold',
               bbox={'boxstyle': 'round', 'facecolor': 'white', 'alpha': 0.9},
               arrowprops={'arrowstyle': '->', 'color': 'red', 'linewidth': 2})

    ax.set_xlabel('Precision', fontsize=11, fontweight='bold')
    ax.set_ylabel('Total Alerts (TP + FP)', fontsize=11, fontweight='bold')
    ax.set_title('Alert Volume vs Precision', fontsize=12, fontweight='bold', pad=10)
    ax.grid(True, alpha=0.3)

    # Panel 3: F1 Score vs Threshold
    ax = axes[1, 0]
    ax.plot(threshold_sweep['threshold'], threshold_sweep['f1'],
           color=COLORS['tp'], linewidth=2.5, marker='o', markersize=5)

    ax.scatter([0.07], [row_07['f1']],
              color='red', s=250, zorder=5, marker='D', edgecolors='black', linewidth=2)
    ax.axvline(0.07, color='red', linestyle='--', alpha=0.5, linewidth=2)

    ax.set_xlabel('Threshold', fontsize=11, fontweight='bold')
    ax.set_ylabel('F1 Score', fontsize=11, fontweight='bold')
    ax.set_title('F1 Score vs Threshold', fontsize=12, fontweight='bold', pad=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 0.2)

    # Panel 4: FPR vs Threshold
    ax = axes[1, 1]
    ax.plot(threshold_sweep['threshold'], threshold_sweep['fpr'],
           color=COLORS['fp'], linewidth=2.5, marker='o', markersize=5)

    ax.scatter([0.07], [row_07['fpr']],
              color='red', s=250, zorder=5, marker='D', edgecolors='black', linewidth=2)
    ax.axvline(0.07, color='red', linestyle='--', alpha=0.5, linewidth=2)
    ax.axhline(0.05, color='orange', linestyle=':', alpha=0.5, linewidth=2, label='5% FPR')
    ax.axhline(0.01, color='green', linestyle=':', alpha=0.5, linewidth=2, label='1% FPR')

    ax.set_xlabel('Threshold', fontsize=11, fontweight='bold')
    ax.set_ylabel('False Positive Rate', fontsize=11, fontweight='bold')
    ax.set_title('FPR vs Threshold', fontsize=12, fontweight='bold', pad=10)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=9)
    ax.set_xlim(0, 0.2)

    plt.tight_layout()
    fig.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()
    logger.info(f"  Saved to {output_path}")


def plot_error_analysis(df: pd.DataFrame, output_path: Path):
    """
    Phase 3.3: Error Analysis Deep Dive (3×2)
    Shows what distinguishes errors from successes.
    """
    logger.info("Generating error analysis deep dive...")

    # Get top 3 most important features (using correlation with outcome)
    feature_cols = ['growth_acceleration', 'growth_rate_diff', 'new_works_roll_std_4q',
                   'logistic_growth_rate', 'cumulative_works', 'new_works_roll_mean_4q']

    available_features = [f for f in feature_cols if f in df.columns][:3]

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # Row 1: Feature distributions for top 3 features
    legend_handles = [Patch(facecolor=COLORS[o.lower()], label=o) for o in ['TP', 'FP', 'FN']]

    for idx, feature in enumerate(available_features):
        ax = axes[0, idx]

        # Prepare data
        plot_data = []
        for outcome in ['TP', 'FP', 'FN']:
            subset = df[df['outcome'] == outcome][feature].dropna()
            if len(subset) > 0:
                plot_data.append({'outcome': outcome, 'values': subset})

        # Violin plot
        positions = []
        for i, item in enumerate(plot_data):
            pos = [i] * len(item['values'])
            positions.extend(pos)

            parts = ax.violinplot([item['values']], positions=[i],
                                 showmeans=True, showmedians=True)

            # Color by outcome
            color = COLORS[item['outcome'].lower()]
            for pc in parts['bodies']:
                pc.set_facecolor(color)
                pc.set_alpha(0.6)

        ax.set_xticks(range(len(plot_data)))
        ax.set_xticklabels([item['outcome'] for item in plot_data], fontsize=10, fontweight='bold')
        ax.set_ylabel(feature.replace('_', ' ').title(), fontsize=10, fontweight='bold')
        ax.set_title(f'{feature.replace("_", " ").title()} by Outcome',
                    fontsize=11, fontweight='bold', pad=8)
        ax.grid(True, alpha=0.3, axis='y')
        if idx == 0 and legend_handles:
            ax.legend(handles=legend_handles, loc='upper right', fontsize=9)
        if plot_data:
            combined = pd.concat([item['values'] for item in plot_data])
            q_low, q_high = combined.quantile([0.02, 0.98])
            if q_high > q_low:
                margin = max(1e-3, (q_high - q_low) * 0.15)
                ax.set_ylim(q_low - margin, q_high + margin)
            for position, item in enumerate(plot_data):
                ax.text(position, np.median(item['values']), f"{np.median(item['values']):.1f}",
                        ha='center', va='bottom', fontsize=8, color='black', fontweight='bold')

    # Row 2: Growth pattern characteristics
    characteristics = [
        ('growth_acceleration', 'Acceleration'),
        ('new_works_roll_std_4q', 'Volatility (4Q Std)'),
        ('logistic_fit_r2', 'Logistic Fit Quality')
    ]

    available_chars = [(col, name) for col, name in characteristics if col in df.columns][:3]
    empty_axes = []

    for idx, (feature, name) in enumerate(available_chars):
        ax = axes[1, idx]

        # Box plot
        data_to_plot = []
        labels_to_plot = []
        colors_to_plot = []

        for outcome in ['TP', 'FP', 'FN']:
            subset = df[df['outcome'] == outcome][feature].dropna()
            if len(subset) > 0:
                data_to_plot.append(subset)
                labels_to_plot.append(outcome)
                colors_to_plot.append(COLORS[outcome.lower()])

        bp = ax.boxplot(data_to_plot, labels=labels_to_plot, patch_artist=True,
                       showfliers=False, widths=0.6)

        for patch, color in zip(bp['boxes'], colors_to_plot):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)

        ax.set_ylabel(name, fontsize=10, fontweight='bold')
        ax.set_title(f'{name} Distribution', fontsize=11, fontweight='bold', pad=8)
        ax.grid(True, alpha=0.3, axis='y')
        ax.tick_params(axis='x', labelsize=10)

        if data_to_plot:
            combined = pd.concat(data_to_plot)
            q_low, q_high = combined.quantile([0.02, 0.98])
            if q_high > q_low:
                margin = (q_high - q_low) * 0.1
                ax.set_ylim(q_low - margin, q_high + margin)
        if idx == 0 and legend_handles:
            ax.legend(handles=legend_handles, loc='upper right', fontsize=9)
        for pos, median in zip(range(1, len(data_to_plot) + 1), [np.median(values) for values in data_to_plot]):
            ax.text(pos, median, f"{median:.1f}", ha='center', va='bottom', fontsize=8, color='black', fontweight='bold')

    for empty_idx in range(len(available_chars), 3):
        empty_axes.append(axes[1, empty_idx])

    if empty_axes:
        ax_table = empty_axes[0]
        ax_table.axis('off')
        summary = df.groupby('outcome')['inflection_probability'].agg(['count', 'median']).reindex(['TP', 'FP', 'FN', 'TN']).fillna(0)
        summary['count'] = summary['count'].astype(int)
        table = ax_table.table(
            cellText=summary.values,
            rowLabels=summary.index,
            colLabels=['Count', 'Median Prob'],
            loc='center',
            cellLoc='center'
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.4)
        ax_table.set_title('Outcome Summary', fontsize=10, fontweight='bold')
        for remaining_ax in empty_axes[1:]:
            remaining_ax.axis('off')

    plt.suptitle('Error Analysis: Feature Distributions by Outcome',
                fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    fig.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()
    logger.info(f"  Saved to {output_path}")


# ============================================================================
# PHASE 4: ADVANCED ANALYSIS
# ============================================================================

def plot_feature_space_projection(df: pd.DataFrame, output_path: Path):
    """
    Phase 4.1: Feature Space Projection (PCA 2D)
    Shows if inflections are separable in feature space.
    """
    logger.info("Generating feature space projection...")

    # Select numerical features
    feature_cols = ['growth_acceleration', 'growth_rate_diff', 'new_works_roll_std_4q',
                   'logistic_growth_rate', 'cumulative_works', 'new_works_roll_mean_4q',
                   'new_works_roll_mean_2q', 'logistic_carrying_capacity']

    available_features = [f for f in feature_cols if f in df.columns]

    # Prepare data
    df_features = df[available_features + ['outcome', 'inflection_probability']].dropna()

    if len(df_features) < 100:
        logger.warning("  Not enough data for PCA projection, skipping...")
        return

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_features[available_features])

    # PCA
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    fig = plt.figure(figsize=(14, 10))
    gs = gridspec.GridSpec(2, 2, height_ratios=[3, 1], width_ratios=[3, 1],
                          hspace=0.02, wspace=0.02)

    # Main scatter plot
    ax_main = fig.add_subplot(gs[0, 0])

    # Sample for visualization
    sample_size = min(5000, len(df_features))
    sample_indices = np.random.choice(len(df_features), sample_size, replace=False)

    # Plot by outcome
    for outcome, color, marker in [('TN', COLORS['tn'], '.'),
                                    ('FP', COLORS['fp'], 'x'),
                                    ('FN', COLORS['fn'], 's'),
                                    ('TP', COLORS['tp'], 'o')]:
        mask = df_features.iloc[sample_indices]['outcome'] == outcome
        ax_main.scatter(X_pca[sample_indices][mask, 0], X_pca[sample_indices][mask, 1],
                       c=color, label=outcome, alpha=0.4, s=30, marker=marker,
                       edgecolors='none')

    pc1_vals = X_pca[:, 0]
    pc2_vals = X_pca[:, 1]
    pc1_low, pc1_high = np.percentile(pc1_vals, [0.5, 99.5])
    pc2_low, pc2_high = np.percentile(pc2_vals, [0.5, 99.5])
    def _expand(low, high):
        span = high - low if high > low else 1
        margin = span * 0.1
        return low - margin, high + margin
    x_limits = _expand(pc1_low, pc1_high)
    y_limits = _expand(pc2_low, pc2_high)
    ax_main.set_xlim(*x_limits)
    ax_main.set_ylim(*y_limits)

    ax_main.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} var)',
                      fontsize=11, fontweight='bold')
    ax_main.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} var)',
                      fontsize=11, fontweight='bold')
    ax_main.legend(loc='best', fontsize=10, markerscale=2)
    ax_main.grid(True, alpha=0.3)
    ax_main.set_title('Feature Space Projection (PCA)', fontsize=13, fontweight='bold', pad=10)

    # Bottom marginal: PC1 distribution
    ax_bottom = fig.add_subplot(gs[1, 0], sharex=ax_main)
    ax_bottom.hist(X_pca[:, 0], bins=50, color=COLORS['neutral'], alpha=0.6, edgecolor='black')
    ax_bottom.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} var)',
                        fontsize=11, fontweight='bold')
    ax_bottom.set_ylabel('Count', fontsize=9)
    ax_bottom.grid(True, alpha=0.3, axis='y')

    # Right marginal: PC2 distribution
    ax_right = fig.add_subplot(gs[0, 1], sharey=ax_main)
    ax_right.hist(X_pca[:, 1], bins=50, color=COLORS['neutral'], alpha=0.6,
                 orientation='horizontal', edgecolor='black')
    ax_right.set_xlabel('Count', fontsize=9)
    ax_right.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} var)', fontsize=10, fontweight='bold')
    ax_right.set_title('PC2 distribution', fontsize=10, fontweight='bold')
    ax_right.grid(True, alpha=0.3, axis='x')

    # Add feature loadings
    ax_loadings = fig.add_subplot(gs[1, 1])
    loadings = pca.components_.T

    # Show top 5 features
    feature_importance = np.abs(loadings).sum(axis=1)
    top_features_idx = np.argsort(feature_importance)[-5:][::-1]

    indices = np.arange(len(top_features_idx))
    top_loading_subset = loadings[top_features_idx]
    height = 0.35
    ax_loadings.barh(indices + height/2, top_loading_subset[:, 0],
                     height=height, color=COLORS['accent'], alpha=0.7, label='PC1')
    ax_loadings.barh(indices - height/2, top_loading_subset[:, 1],
                     height=height, color=COLORS['organic'], alpha=0.7, label='PC2')
    ax_loadings.axvline(0, color='gray', linewidth=0.8, linestyle='--')

    for i, idx in enumerate(top_features_idx):
        ax_loadings.text(loadings[idx, 0], indices[i] + height/2 + 0.02, f'{loadings[idx, 0]:.2f}',
                         fontsize=7, ha='left' if loadings[idx, 0] >= 0 else 'right',
                         va='center')
        ax_loadings.text(loadings[idx, 1], indices[i] - height/2 + 0.02, f'{loadings[idx, 1]:.2f}',
                         fontsize=7, ha='left' if loadings[idx, 1] >= 0 else 'right',
                         va='center')

    ax_loadings.set_yticks(indices)
    ax_loadings.set_yticklabels([available_features[i][:24] for i in top_features_idx], fontsize=8)
    ax_loadings.set_xlabel('Component Loading', fontsize=9)
    ax_loadings.set_title('Top Feature Contributions', fontsize=10, fontweight='bold')
    ax_loadings.legend(fontsize=8, loc='lower right')
    ax_loadings.grid(True, alpha=0.3, axis='x')

    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()
    logger.info(f"  Saved to {output_path}")


def plot_temporal_stability(df: pd.DataFrame, output_path: Path):
    """
    Phase 4.2: Temporal Stability Analysis
    Shows if model works consistently across time periods.
    """
    logger.info("Generating temporal stability analysis...")

    # Convert quarters to datetime
    df['quarter_dt'] = pd.to_datetime(df['quarter'])
    df['year'] = df['quarter_dt'].dt.year

    # Compute rolling metrics
    yearly_metrics = []

    for year in sorted(df['year'].unique()):
        year_data = df[df['year'] == year]

        if len(year_data) > 0 and sum(year_data['is_inflection_true']) > 0:
            tp = sum((year_data['is_inflection_true'] == 1) & (year_data['is_inflection_pred'] == 1))
            fp = sum((year_data['is_inflection_true'] == 0) & (year_data['is_inflection_pred'] == 1))
            fn = sum((year_data['is_inflection_true'] == 1) & (year_data['is_inflection_pred'] == 0))

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

            yearly_metrics.append({
                'year': year,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'n_inflections': sum(year_data['is_inflection_true']),
                'n_predictions': sum(year_data['is_inflection_pred'])
            })

    metrics_df = pd.DataFrame(yearly_metrics)

    fig = plt.figure(figsize=(14, 10))
    gs = gridspec.GridSpec(3, 1, height_ratios=[2, 1, 1], hspace=0.3)

    # Top: Timeline of inflections and detections
    ax1 = fig.add_subplot(gs[0])

    # Inflections
    inflections = df[df['is_inflection_true'] == 1].groupby('year').size()
    detections = df[df['is_inflection_pred'] == 1].groupby('year').size()

    years = sorted(df['year'].unique())
    ax1.bar([y - 0.2 for y in years], [inflections.get(y, 0) for y in years],
           width=0.4, color=COLORS['tp'], alpha=0.6, label='True Inflections', edgecolor='black')
    ax1.bar([y + 0.2 for y in years], [detections.get(y, 0) for y in years],
           width=0.4, color=COLORS['accent'], alpha=0.6, label='Predictions', edgecolor='black')

    ax1.set_ylabel('Count', fontsize=11, fontweight='bold')
    ax1.set_title('Inflections and Predictions Over Time', fontsize=13, fontweight='bold', pad=10)
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, alpha=0.3, axis='y')

    # Middle: Rolling performance metrics with sample-size context
    ax2 = fig.add_subplot(gs[1])

    # Background shading based on sample size (lighter = fewer samples, darker = more samples)
    # Normalize sample sizes to [0, 1] for alpha values
    max_samples = metrics_df['n_inflections'].max()
    for _i, row in metrics_df.iterrows():
        alpha = 0.1 + 0.15 * (row['n_inflections'] / max_samples)  # Range: 0.1 to 0.25
        ax2.axvspan(row['year'] - 0.5, row['year'] + 0.5,
                   color=COLORS['neutral'], alpha=alpha, zorder=0)

    # Plot performance metrics
    ax2.plot(metrics_df['year'], metrics_df['precision'],
            color=COLORS['tp'], linewidth=2.5, marker='o', markersize=6, label='Precision', zorder=3)
    ax2.plot(metrics_df['year'], metrics_df['recall'],
            color=COLORS['organic'], linewidth=2.5, marker='s', markersize=6, label='Recall', zorder=3)
    ax2.plot(metrics_df['year'], metrics_df['f1'],
            color=COLORS['accent'], linewidth=2.5, marker='^', markersize=6, label='F1', zorder=3)

    ax2.set_ylabel('Score', fontsize=11, fontweight='bold')
    ax2.set_title('Performance Metrics Over Time (Background shading = sample size)',
                 fontsize=13, fontweight='bold', pad=10)
    ax2.legend(loc='best', fontsize=10)
    ax2.grid(True, alpha=0.3, zorder=1)
    ax2.set_ylim(0, 1.05)

    # Add secondary axis for sample size
    ax2_twin = ax2.twinx()
    ax2_twin.bar(metrics_df['year'], metrics_df['n_inflections'],
                width=0.6, color=COLORS['neutral'], alpha=0.2, edgecolor='gray',
                linewidth=0.5, zorder=2, label='Sample Size')
    ax2_twin.set_ylabel('n (True Inflections)', fontsize=10, fontweight='bold', color='gray')
    ax2_twin.tick_params(axis='y', labelcolor='gray')
    ax2_twin.set_ylim(0, max_samples * 1.2)  # Give some headroom

    # Add note explaining the shading
    note_text = "Darker shading indicates more samples (higher statistical power)"
    ax2.text(0.98, 0.02, note_text, transform=ax2.transAxes,
            fontsize=8, ha='right', va='bottom', style='italic',
            bbox={'boxstyle': 'round', 'facecolor': 'white', 'alpha': 0.8, 'edgecolor': 'gray'})

    # Bottom: Sample size per year
    ax3 = fig.add_subplot(gs[2])

    ax3.bar(metrics_df['year'], metrics_df['n_inflections'],
           color=COLORS['neutral'], alpha=0.6, edgecolor='black')
    ax3.set_xlabel('Year', fontsize=11, fontweight='bold')
    ax3.set_ylabel('True Inflections', fontsize=11, fontweight='bold')
    ax3.set_title('Sample Size (True Inflections per Year)', fontsize=13, fontweight='bold', pad=10)
    ax3.grid(True, alpha=0.3, axis='y')

    plt.suptitle('Temporal Stability Analysis', fontsize=14, fontweight='bold', y=0.995)
    fig.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()
    logger.info(f"  Saved to {output_path}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Generate comprehensive evaluation visualizations')
    parser.add_argument('--predictions', type=Path,
                       default=Path('data/out/experiments/msd_full_field/breakthrough_predictions.csv'))
    parser.add_argument('--features', type=Path,
                       default=Path('data/out/02_lineage_tracking/lineage_multisignal_features.csv'))
    parser.add_argument('--timeseries', type=Path,
                       default=Path('data/out/02_lineage_tracking/lineage_timeseries.csv'))
    parser.add_argument('--labels', type=Path,
                       default=Path('data/out/02_lineage_tracking/inflection_labels.csv'))
    parser.add_argument('--threshold-sweep', type=Path,
                       default=None)
    parser.add_argument('--eval-metrics', type=Path,
                       default=Path('data/out/experiments/msd_full_field/evaluation_metrics.json'))
    parser.add_argument('--field-metrics', type=Path,
                       default=Path('data/out/04_front_aggregation/field_metrics.parquet'))
    parser.add_argument('--output-dir', type=Path,
                       default=Path('data/out/figures/comprehensive_evaluation'))
    parser.add_argument('--threshold', type=float, default=0.07)
    parser.add_argument('--persistence-window', type=int, default=2,
                       help='Require detections to stay above threshold for this many quarters (set 1 to disable)')

    args = parser.parse_args()

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {args.output_dir}")

    # Load data
    threshold_sweep_path = args.threshold_sweep or (args.predictions.parent / 'threshold_sweep.csv')
    data = load_data(args.predictions, args.features, args.timeseries,
                    args.labels, threshold_sweep_path, args.field_metrics)
    data['timeseries'] = add_seasonal_context(data['timeseries'])
    if data['threshold_sweep'].empty:
        generated_sweep = args.output_dir / 'threshold_sweep_generated.csv'
        data['threshold_sweep'] = compute_threshold_sweep_from_predictions(
            data['predictions'],
            output_path=generated_sweep,
        )

    # Prepare analysis dataset
    df = prepare_analysis_data(
        data,
        threshold=args.threshold,
        persistence_window=args.persistence_window
    )

    # PHASE 1: Confidence Analysis
    logger.info("\n=== PHASE 1: CONFIDENCE DISTRIBUTION ANALYSIS ===")
    plot_confidence_distribution_analysis(df, args.output_dir / 'phase1_confidence_distribution.png')
    plot_confidence_vs_outcome(df, args.output_dir / 'phase1_confidence_vs_outcome.png')

    # PHASE 2: Visual Examples
    logger.info("\n=== PHASE 2: VISUAL EXAMPLES ===")
    plot_growth_trajectory_fingerprints(df, data['timeseries'], data['labels'],
                                       args.output_dir / 'phase2_growth_trajectories.png')
    plot_case_study_montage(
        df,
        data['timeseries'],
        data['features'],
        args.output_dir / 'phase2_case_studies.png',
        persistence_window=args.persistence_window
    )
    plot_field_vs_lineage_panel(
        df,
        data['field_metrics'],
        args.output_dir / 'phase2_field_vs_lineage.png',
        threshold=args.threshold,
    )

    # PHASE 3: Performance Context
    logger.info("\n=== PHASE 3: PERFORMANCE CONTEXT ===")
    plot_cv_performance_summary(args.eval_metrics, data['threshold_sweep'],
                               args.output_dir / 'phase3_cv_performance.png')
    plot_operating_characteristics(data['threshold_sweep'],
                                   args.output_dir / 'phase3_operating_characteristics.png')
    plot_error_analysis(df, args.output_dir / 'phase3_error_analysis.png')

    # PHASE 4: Advanced Analysis
    logger.info("\n=== PHASE 4: ADVANCED ANALYSIS ===")
    plot_feature_space_projection(df, args.output_dir / 'phase4_feature_space.png')
    plot_temporal_stability(df, args.output_dir / 'phase4_temporal_stability.png')

    logger.info("\n=== ALL PHASES COMPLETE ===")
    logger.info(f"All visualizations saved to: {args.output_dir}")

    # Generate summary report
    summary = {
        'threshold': args.threshold,
        'total_samples': len(df),
        'outcomes': df['outcome'].value_counts().to_dict(),
        'output_directory': str(args.output_dir),
        'visualizations_generated': [
            'phase1_confidence_distribution.png',
        'phase1_confidence_vs_outcome.png',
        'phase2_growth_trajectories.png',
        'phase2_case_studies.png',
        'phase2_field_vs_lineage.png',
        'phase3_cv_performance.png',
        'phase3_operating_characteristics.png',
        'phase3_error_analysis.png',
        'phase4_feature_space.png',
        'phase4_temporal_stability.png'
        ]
    }

    with open(args.output_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Summary saved to: {args.output_dir / 'summary.json'}")


if __name__ == '__main__':
    main()
