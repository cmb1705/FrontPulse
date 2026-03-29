#!/usr/bin/env python3
"""
Inflection Detection Evaluation - S5 Deliverables

Generates lag-focused evaluation dashboards and analyses:
1. Lag distribution dashboard (histogram/CDF with breakdowns)
2. Milestone → Inflection → Detection timeline plots
3. False positive analysis
4. Performance summary table
5. Comparative analysis (milestone coverage vs detection speed)

Usage:
    python scripts/evaluate_inflection_detection.py [--output-dir DIR] [--threshold FLOAT] [--n-timeline-plots INT]
"""

import argparse
import json
import logging
import warnings
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from persistence_utils import ensure_persistence_column
from seasonality_utils import add_seasonal_context, attach_temporal_context

warnings.filterwarnings('ignore', category=FutureWarning)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_data(
    predictions_path: Path,
    inflection_labels_path: Path,
    milestone_analysis_path: Path,
    timeseries_path: Path,
    threshold_sweep_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load all required datasets."""
    logger.info("Loading datasets...")

    predictions = pd.read_csv(predictions_path)
    labels = pd.read_csv(inflection_labels_path)
    milestone_analysis = pd.read_csv(milestone_analysis_path)
    timeseries = pd.read_csv(timeseries_path)
    if threshold_sweep_path.exists():
        threshold_sweep = pd.read_csv(threshold_sweep_path)
        logger.info(f"  Threshold sweep: {len(threshold_sweep):,} thresholds")
    else:
        logger.warning("Threshold sweep file not found at %s; will compute dynamically.", threshold_sweep_path)
        threshold_sweep = pd.DataFrame()

    logger.info(f"  Predictions: {len(predictions):,} rows")
    logger.info(f"  Labels: {len(labels):,} inflection points")
    logger.info(f"  Milestone analysis: {len(milestone_analysis):,} rows")
    logger.info(f"  Timeseries: {len(timeseries):,} rows")

    return predictions, labels, milestone_analysis, timeseries, threshold_sweep


def compute_threshold_sweep_from_predictions(
    predictions: pd.DataFrame,
    thresholds: Optional[np.ndarray] = None,
    output_path: Optional[Path] = None,
) -> pd.DataFrame:
    """Derive a simple threshold sweep in case precomputed metrics are unavailable."""
    logger.info("Computing threshold sweep directly from predictions...")
    thresholds = thresholds if thresholds is not None else np.linspace(0.01, 0.5, 50)
    rows: list[dict[str, float]] = []
    y_true = predictions['is_inflection_true'].fillna(0).astype(int)
    total_actual = int((y_true == 1).sum())
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
    """Read field metrics (csv or parquet) and normalize column names."""
    if not path.exists():
        logger.warning("Field metrics file %s not found; skipping field vs lineage dashboard.", path)
        return pd.DataFrame()
    df = pd.read_parquet(path) if path.suffix == '.parquet' else pd.read_csv(path)
    df = df.copy()
    df['quarter'] = df['quarter'].astype(str)
    rename_map = {col: (col if col == 'quarter' else f"field_{col}") for col in df.columns}
    df.rename(columns=rename_map, inplace=True)
    return df


def create_field_vs_lineage_dashboard(
    predictions: pd.DataFrame,
    timeseries: pd.DataFrame,
    field_metrics: pd.DataFrame,
    threshold: float,
    output_dir: Path,
) -> dict[str, float]:
    """Visualize detections vs. field-wide growth to highlight contextual performance."""
    if field_metrics.empty:
        return {}

    detections = predictions[
        (predictions['inflection_probability'] >= threshold) & (predictions['is_inflection_pred'] == 1)
    ].copy()
    det_counts = detections.groupby('quarter').size().rename('detections').reset_index()

    timeline = field_metrics.merge(det_counts, on='quarter', how='left')
    timeline['detections'] = timeline['detections'].fillna(0)
    lineage_totals = timeseries.groupby('quarter')['new_works'].sum().rename('lineage_total_new_works').reset_index()
    timeline = timeline.merge(lineage_totals, on='quarter', how='left')
    timeline = timeline.sort_values('quarter')
    timeline['quarter_dt'] = pd.PeriodIndex(timeline['quarter'], freq='Q').to_timestamp()

    # Detection share by field growth bucket
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

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), height_ratios=[3, 1.5])
    x = np.arange(len(timeline))
    ax1.bar(x, timeline['field_total_new_works'], color='#bdc3c7', label='Field New Works')
    ax1.set_ylabel('Field New Works', fontsize=11, fontweight='bold')
    ax1.set_title('Field Output vs. Detection Volume', fontsize=12, fontweight='bold', pad=12)
    ax1.grid(alpha=0.2, axis='y')
    ax1b = ax1.twinx()
    ax1b.plot(x, timeline['detections'], color='#9b59b6', linewidth=2, label='Detections')
    ax1b.set_ylabel('# Detections', color='#9b59b6', fontsize=11, fontweight='bold')
    ax1b.tick_params(axis='y', labelcolor='#9b59b6')
    tick_idx = np.linspace(0, len(x) - 1, min(12, len(x))).astype(int) if len(x) > 0 else []
    ax1.set_xticks(tick_idx)
    ax1.set_xticklabels(timeline['quarter'].iloc[tick_idx] if len(tick_idx) else [] , rotation=45, ha='right')
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1b.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc='upper left')

    # Detection share vs field bucket
    ax2.bar(bucket_summary['field_growth_bucket'].astype(str), bucket_summary['detection_share'] * 100, color='#34495e')
    ax2.set_ylabel('% of Detections', fontsize=11, fontweight='bold')
    ax2.set_title('Detections Concentrated in High-Growth Field Quarters', fontsize=12, fontweight='bold', pad=10)
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
    field_fig_path = output_dir / 'field_vs_lineage_dashboard.png'
    fig.savefig(field_fig_path)
    plt.close(fig)
    logger.info("Saved field vs lineage dashboard to %s", field_fig_path)

    summary = {
        'total_detections': int(total_detections),
        'quarters': int(len(timeline)),
        'share_top_quartile': float(bucket_summary.loc[bucket_summary['field_growth_bucket'].astype(str).str.contains('Q4'), 'detection_share'].sum()) if not bucket_summary.empty else 0.0,
        'median_field_new_works': float(timeline['field_total_new_works'].median()) if not timeline.empty else 0.0,
    }

    bucket_summary.to_csv(output_dir / 'field_vs_lineage_summary.csv', index=False)
    return summary


def prepare_predictions_with_context(
    predictions: pd.DataFrame,
    timeseries: pd.DataFrame,
    threshold: float,
    persistence_window: int,
) -> pd.DataFrame:
    """Attach rolling/YoY context + persistence flags to the predictions frame."""
    enriched_preds = predictions.copy()
    ensure_persistence_column(
        enriched_preds,
        threshold=threshold,
        window=persistence_window,
        column_name="is_inflection_pred_persistent",
    )
    enriched_preds = attach_temporal_context(enriched_preds, timeseries)
    return enriched_preds


def format_yoy_pct(value: float) -> str:
    """Format YoY percentage for annotations."""
    if pd.isna(value):
        return "YoY N/A"
    sign = "+" if value >= 0 else ""
    return f"YoY {sign}{value:.0f}%"


def create_lag_distribution_dashboard(
    predictions: pd.DataFrame,
    labels: pd.DataFrame,
    milestone_analysis: pd.DataFrame,
    threshold: float,
    output_dir: Path
) -> dict:
    """
    Generate lag distribution dashboard with breakdowns.

    Returns summary statistics dict.
    """
    logger.info(f"Creating lag distribution dashboard (threshold={threshold})...")

    # Filter predictions at threshold
    preds_at_threshold = predictions[predictions['inflection_probability'] >= threshold].copy()

    # Get detections with lag data
    detections = preds_at_threshold[
        (preds_at_threshold['is_inflection_true'] == 1) &
        (preds_at_threshold['is_inflection_pred'] == 1) &
        (preds_at_threshold['detection_lag_quarters'].notna())
    ].copy()

    logger.info(f"  Detections with lag data: {len(detections):,}")
    if detections.empty:
        logger.warning("No detections available at this threshold; skipping lag plots.")
        fig = plt.figure(figsize=(6, 3))
        ax = fig.add_subplot(111)
        ax.axis('off')
        ax.text(0.5, 0.5, 'No detections meet the threshold', ha='center', va='center', fontsize=12, fontweight='bold')
        output_path = output_dir / 'lag_distribution_dashboard.png'
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        return {
            'median_lag': float('nan'),
            'mean_lag': float('nan'),
            'n_detections': 0,
            'milestone_pct': 0.0,
        }

    # Merge with milestone analysis for breakdown
    detections = detections.merge(
        milestone_analysis[['lineage_id', 'inflection_quarter', 'nearest_milestone_id', 'lag_bucket']].rename(columns={'inflection_quarter': 'quarter'}),
        on=['lineage_id', 'quarter'],
        how='left'
    )

    # Add milestone presence flag
    detections['has_milestone'] = detections['nearest_milestone_id'].notna()

    # Create figure
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)

    # 1. Overall lag distribution (histogram)
    ax1 = fig.add_subplot(gs[0, :2])
    lag_data = detections['detection_lag_quarters'].dropna()
    ax1.hist(lag_data, bins=30, edgecolor='black', alpha=0.7, color='steelblue')
    ax1.axvline(lag_data.median(), color='red', linestyle='--', linewidth=2, label=f'Median: {lag_data.median():.1f}Q')
    ax1.axvline(lag_data.mean(), color='orange', linestyle='--', linewidth=2, label=f'Mean: {lag_data.mean():.2f}Q')
    ax1.set_xlabel('Detection Lag (quarters)', fontsize=11)
    ax1.set_ylabel('Count', fontsize=11)
    ax1.set_title(f'Detection Lag Distribution (n={len(lag_data):,}, threshold={threshold})', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(alpha=0.3)

    # 2. CDF
    ax2 = fig.add_subplot(gs[0, 2])
    sorted_lag = np.sort(lag_data)
    cdf = np.arange(1, len(sorted_lag) + 1) / len(sorted_lag)
    ax2.plot(sorted_lag, cdf, linewidth=2, color='steelblue')
    ax2.axhline(0.5, color='red', linestyle='--', alpha=0.5)
    ax2.axvline(lag_data.median(), color='red', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Detection Lag (quarters)', fontsize=11)
    ax2.set_ylabel('Cumulative Probability', fontsize=11)
    ax2.set_title('CDF', fontsize=12, fontweight='bold')
    ax2.grid(alpha=0.3)

    # 3. Breakdown by milestone presence (median ± IQR)
    ax3 = fig.add_subplot(gs[1, 0])
    group_labels = []
    medians = []
    err_low = []
    err_high = []
    counts = []
    for has_ms, label in [(False, 'No Milestone'), (True, 'Has Milestone')]:
        subset = detections[detections['has_milestone'] == has_ms]['detection_lag_quarters'].dropna()
        if subset.empty:
            continue
        q1, med, q3 = subset.quantile([0.25, 0.5, 0.75])
        group_labels.append(label)
        medians.append(med)
        err_low.append(med - q1)
        err_high.append(q3 - med)
        counts.append(len(subset))
    if group_labels:
        x = np.arange(len(group_labels))
        ax3.bar(x, medians, color='lightsteelblue', edgecolor='black')
        ax3.errorbar(x, medians, yerr=[err_low, err_high], fmt='none', capsize=6, color='black')
        peak = np.max(np.abs(np.array(medians) + np.array(err_high))) if medians else 0
        y_pad = max(0.05, peak + 0.01)
        ax3.set_ylim(-y_pad, y_pad)
        for _idx, (xpos, med, count) in enumerate(zip(x, medians, counts)):
            ax3.text(xpos, y_pad * 0.8, f"n={count}", ha='center', va='bottom', fontsize=9, fontweight='bold')
        ax3.set_xticks(x)
        ax3.set_xticklabels(group_labels)
        ax3.set_ylabel('Detection Lag (quarters)', fontsize=11)
        ax3.set_title('Lag by Milestone Presence (median ± IQR)', fontsize=12, fontweight='bold')
        ax3.grid(alpha=0.3, axis='y')
        if np.allclose(medians, 0, atol=1e-6):
            ax3.text(0.5, 0.65, 'All detections occur at labeled quarter (lag = 0)', transform=ax3.transAxes,
                     ha='center', fontsize=10, fontweight='bold', color='gray')
    else:
        ax3.axis('off')
        ax3.text(0.5, 0.5, 'No milestone-linked detections', ha='center', va='center', fontsize=11)

    # 4. Breakdown by lag bucket (from milestone analysis)
    ax4 = fig.add_subplot(gs[1, 1:])
    bucket_counts = detections['lag_bucket'].value_counts().sort_index()
    if bucket_counts.empty:
        ax4.text(0.5, 0.5, 'No detections by lag bucket', ha='center', va='center', fontsize=11)
        ax4.axis('off')
    else:
        bucket_counts.plot(kind='bar', ax=ax4, color='steelblue', edgecolor='black')
        ax4.set_xlabel('Lag Bucket', fontsize=11)
        ax4.set_ylabel('Count', fontsize=11)
        ax4.set_title('Detection Count by Milestone Lag Bucket', fontsize=12, fontweight='bold')
        ax4.tick_params(axis='x', rotation=45)
        ax4.grid(alpha=0.3, axis='y')

    # 5. Summary statistics table
    ax5 = fig.add_subplot(gs[2, :])
    ax5.axis('off')

    # Compute stats
    stats_overall = {
        'Metric': ['Count', 'Median', 'Mean', 'Std', '% ≤ 0Q', '% ≤ 2Q'],
        'Overall': [
            f"{len(lag_data):,}",
            f"{lag_data.median():.2f}",
            f"{lag_data.mean():.2f}",
            f"{lag_data.std():.2f}",
            f"{(lag_data <= 0).sum() / len(lag_data) * 100:.1f}%",
            f"{(lag_data <= 2).sum() / len(lag_data) * 100:.1f}%"
        ]
    }

    # Add milestone vs no-milestone columns
    for has_ms, label in [(False, 'No Milestone'), (True, 'Has Milestone')]:
        subset = detections[detections['has_milestone'] == has_ms]['detection_lag_quarters'].dropna()
        if len(subset) > 0:
            stats_overall[label] = [
                f"{len(subset):,}",
                f"{subset.median():.2f}",
                f"{subset.mean():.2f}",
                f"{subset.std():.2f}",
                f"{(subset <= 0).sum() / len(subset) * 100:.1f}%",
                f"{(subset <= 2).sum() / len(subset) * 100:.1f}%"
            ]
        else:
            stats_overall[label] = ['0'] + ['N/A'] * 5

    stats_df = pd.DataFrame(stats_overall)
    table = ax5.table(
        cellText=stats_df.values,
        colLabels=stats_df.columns,
        cellLoc='center',
        loc='center',
        bbox=[0, 0, 1, 1]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)

    # Style header
    for i in range(len(stats_df.columns)):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')

    plt.suptitle(f'Inflection Detection Lag Analysis (Threshold={threshold})',
                 fontsize=14, fontweight='bold', y=0.98)

    output_path = output_dir / 'lag_distribution_dashboard.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"  Saved: {output_path}")

    # Return summary dict
    summary = {
        'n_detections': len(lag_data),
        'median_lag': float(lag_data.median()),
        'mean_lag': float(lag_data.mean()),
        'std_lag': float(lag_data.std()),
        'pct_le_0': float((lag_data <= 0).sum() / len(lag_data) * 100),
        'pct_le_2': float((lag_data <= 2).sum() / len(lag_data) * 100),
        'milestone_count': int(detections['has_milestone'].sum()),
        'milestone_pct': float(detections['has_milestone'].sum() / len(detections) * 100)
    }

    return summary


def create_timeline_plots(
    predictions: pd.DataFrame,
    labels: pd.DataFrame,
    milestone_analysis: pd.DataFrame,
    timeseries: pd.DataFrame,
    threshold: float,
    n_plots: int,
    output_dir: Path,
    persistence_col: str = "is_inflection_pred_persistent",
):
    """
    Generate timeline plots showing milestones, inflections, and detections on growth curves.
    Sample diverse lineages (TP, milestone-linked, organic, different fronts).
    """
    logger.info(f"Creating timeline plots (n={n_plots})...")

    # Filter predictions at threshold
    if persistence_col in predictions.columns:
        persistence_mask = predictions[persistence_col] == 1
    else:
        persistence_mask = predictions['is_inflection_pred'] == 1

    preds_at_threshold = predictions[
        (predictions['inflection_probability'] >= threshold) &
        persistence_mask
    ].copy()

    # Get true positives
    tp_lineages = preds_at_threshold[
        (preds_at_threshold['is_inflection_true'] == 1) &
        (preds_at_threshold['is_inflection_pred'] == 1)
    ]['lineage_id'].unique()

    # Merge with milestone data
    tp_with_info = pd.DataFrame({'lineage_id': tp_lineages})
    tp_with_info = tp_with_info.merge(
        milestone_analysis[['lineage_id', 'nearest_milestone_id']].drop_duplicates(),
        on='lineage_id',
        how='left'
    )
    tp_with_info['has_milestone'] = tp_with_info['nearest_milestone_id'].notna()

    # Sample: 50% with milestone, 50% without (if possible)
    with_pool = tp_with_info[tp_with_info['has_milestone']]
    without_pool = tp_with_info[~tp_with_info['has_milestone']]

    target_with = min(n_plots // 2, len(with_pool))
    sample_with = with_pool.sample(n=target_with, random_state=42) if target_with > 0 else with_pool.head(0)

    remaining = n_plots - len(sample_with)
    target_without = min(remaining, len(without_pool))
    sample_without = without_pool.sample(n=target_without, random_state=42) if target_without > 0 else without_pool.head(0)

    combined = pd.concat([sample_with, sample_without])
    if len(combined) < n_plots:
        deficit = n_plots - len(combined)
        leftover = tp_with_info[~tp_with_info.index.isin(combined.index)]
        if deficit > 0 and len(leftover) > 0:
            combined = pd.concat([combined, leftover.sample(n=min(deficit, len(leftover)), random_state=43)])

    selected_lineages = combined['lineage_id'].values
    logger.info(f"  Selected {len(selected_lineages)} lineages ({len(sample_with)} with milestones, {len(sample_without)} without)")

    # Create plots directory
    plots_dir = output_dir / 'timeline_plots'
    plots_dir.mkdir(parents=True, exist_ok=True)

    for i, lineage_id in enumerate(selected_lineages, 1):
        _plot_single_timeline(
            lineage_id,
            timeseries,
            labels,
            preds_at_threshold,
            milestone_analysis,
            threshold,
            plots_dir,
            i,
            persistence_col=persistence_col,
        )

    logger.info(f"  Saved {len(selected_lineages)} timeline plots to {plots_dir}")


def _plot_single_timeline(
    lineage_id: int,
    timeseries: pd.DataFrame,
    labels: pd.DataFrame,
    predictions: pd.DataFrame,
    milestone_analysis: pd.DataFrame,
    threshold: float,
    plots_dir: Path,
    plot_num: int,
    persistence_col: str = "is_inflection_pred_persistent",
):
    """Plot a single lineage timeline."""

    # Get data for this lineage
    ts = timeseries[timeseries['lineage_id'] == lineage_id].sort_values('quarter').copy()
    label_row = labels[labels['lineage_id'] == lineage_id]
    if persistence_col in predictions.columns:
        pred_rows = predictions[
            (predictions['lineage_id'] == lineage_id) &
            (predictions[persistence_col] == 1)
        ]
    else:
        pred_rows = predictions[
            (predictions['lineage_id'] == lineage_id) &
            (predictions['is_inflection_pred'] == 1)
        ]
    milestone_row = milestone_analysis[milestone_analysis['lineage_id'] == lineage_id]

    if len(ts) == 0:
        return

    # Compute cumulative works
    ts['cumulative_works'] = ts['new_works'].cumsum()

    # Convert quarter to numeric for plotting
    ts['quarter_num'] = pd.to_datetime(ts['quarter']).dt.year + (pd.to_datetime(ts['quarter']).dt.quarter - 1) / 4

    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot growth curve
    ax.plot(ts['quarter_num'], ts['cumulative_works'], 'o-', linewidth=2, markersize=4, color='steelblue', label='Cumulative Works')

    # Mark inflection onset
    if len(label_row) > 0:
        infl_quarter = label_row.iloc[0]['quarter']
        infl_quarter_num = pd.to_datetime(infl_quarter).year + (pd.to_datetime(infl_quarter).quarter - 1) / 4
        ax.axvline(infl_quarter_num, color='red', linestyle='--', linewidth=2, alpha=0.7, label='True Inflection')

    # Mark detections
    if len(pred_rows) > 0:
        for _, pred_row in pred_rows.iterrows():
            det_quarter = pred_row['quarter']
            det_quarter_num = pd.to_datetime(det_quarter).year + (pd.to_datetime(det_quarter).quarter - 1) / 4
            ax.axvline(det_quarter_num, color='green', linestyle=':', linewidth=2, alpha=0.7, label='Detection' if _ == pred_rows.index[0] else '')

    # Mark milestone (if exists)
    if len(milestone_row) > 0 and pd.notna(milestone_row.iloc[0]['nearest_milestone_id']):
        # Milestone quarter not in milestone_analysis, but we can infer from label quarter + lag
        ms_id = milestone_row.iloc[0]['nearest_milestone_id']
        ax.scatter([], [], marker='*', s=300, color='gold', edgecolor='black', linewidth=1.5, label=f'Milestone (ID={ms_id})')

    ax.set_xlabel('Year', fontsize=11)
    ax.set_ylabel('Cumulative Works', fontsize=11)
    ax.set_title(f'Lineage {lineage_id} Timeline', fontsize=12, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(alpha=0.3)

    output_path = plots_dir / f'timeline_{plot_num:03d}_lineage_{lineage_id}.png'
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close()


def analyze_false_positives(
    predictions: pd.DataFrame,
    timeseries: pd.DataFrame,
    milestone_analysis: pd.DataFrame,
    threshold: float,
    output_dir: Path,
    persistence_col: str = "is_inflection_pred_persistent",
) -> dict:
    """
    Characterize false positive cases.
    """
    logger.info(f"Analyzing false positives (threshold={threshold})...")

    # Get FPs
    if persistence_col in predictions.columns:
        persistence_mask = predictions[persistence_col] == 1
    else:
        persistence_mask = predictions['is_inflection_pred'] == 1

    fps = predictions[
        (predictions['inflection_probability'] >= threshold) &
        (predictions['is_inflection_true'] == 0) &
        persistence_mask
    ].copy()

    logger.info(f"  False positives: {len(fps):,}")

    if len(fps) == 0:
        logger.warning("  No false positives found!")
        return {}

    # Merge with timeseries to get growth data if not already attached
    if 'new_works' not in fps.columns:
        fps = fps.merge(
            timeseries[['lineage_id', 'quarter', 'new_works']],
            on=['lineage_id', 'quarter'],
            how='left'
        )

    # Merge with milestone analysis
    fps = fps.merge(
        milestone_analysis[['lineage_id', 'inflection_quarter', 'nearest_milestone_id']].rename(columns={'inflection_quarter': 'quarter'}),
        on=['lineage_id', 'quarter'],
        how='left'
    )
    fps['has_milestone'] = fps['nearest_milestone_id'].notna()

    # Summary stats
    summary = {
        'n_fps': len(fps),
        'unique_lineages': fps['lineage_id'].nunique(),
        'fps_with_milestone': int(fps['has_milestone'].sum()),
        'fps_with_milestone_pct': float(fps['has_milestone'].sum() / len(fps) * 100),
        'mean_new_works': float(fps['new_works'].mean()),
        'median_new_works': float(fps['new_works'].median())
    }

    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 1. FP count by lineage
    ax = axes[0]
    fp_by_lineage = fps['lineage_id'].value_counts().head(20)
    fp_by_lineage.plot(kind='bar', ax=ax, color='salmon', edgecolor='black')
    ax.set_xlabel('Lineage ID', fontsize=10)
    ax.set_ylabel('FP Count', fontsize=10)
    ax.set_title('Top 20 Lineages by FP Count', fontsize=11, fontweight='bold')
    ax.tick_params(axis='x', rotation=90, labelsize=8)
    ax.grid(alpha=0.3, axis='y')

    # 2. Milestone presence
    ax = axes[1]
    milestone_counts = fps['has_milestone'].value_counts()
    # Handle case where only one value exists
    milestone_labels = []
    milestone_colors = []
    for val in milestone_counts.index:
        if val:
            milestone_labels.append('Has Milestone')
            milestone_colors.append('gold')
        else:
            milestone_labels.append('No Milestone')
            milestone_colors.append('lightcoral')
    milestone_counts.index = milestone_labels
    milestone_counts.plot(kind='bar', ax=ax, color=milestone_colors, edgecolor='black')
    ax.set_xlabel('Milestone Presence', fontsize=10)
    ax.set_ylabel('Count', fontsize=10)
    ax.set_title('FPs by Milestone Presence', fontsize=11, fontweight='bold')
    ax.tick_params(axis='x', rotation=0)
    ax.grid(alpha=0.3, axis='y')

    # 3. New works distribution
    ax = axes[2]
    ax.hist(fps['new_works'].dropna(), bins=30, color='salmon', edgecolor='black', alpha=0.7)
    ax.axvline(fps['new_works'].median(), color='red', linestyle='--', linewidth=2,
               label=f'Median: {fps["new_works"].median():.1f}')
    ax.set_xlabel('New Works (quarter of FP)', fontsize=10)
    ax.set_ylabel('Count', fontsize=10)
    ax.set_title('New Works Distribution (FPs)', fontsize=11, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)

    plt.suptitle(f'False Positive Analysis (n={len(fps):,}, threshold={threshold})',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()

    output_path = output_dir / 'false_positive_analysis.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"  Saved: {output_path}")

    return summary


def export_expert_review_sample(
    predictions: pd.DataFrame,
    milestone_analysis: pd.DataFrame,
    output_dir: Path,
    threshold: float,
    persistence_col: str = "is_inflection_pred_persistent",
    sample_plan: dict[str, int] = None,
) -> Path:
    """
    Persist a balanced TP/FP/FN sample with seasonal context for expert review.
    """
    if sample_plan is None:
        sample_plan = {"TP": 20, "FP": 10, "FN": 5}

    logger.info("Exporting expert review sample...")

    if persistence_col in predictions.columns:
        detection_mask = predictions[persistence_col] == 1
    else:
        detection_mask = predictions["is_inflection_pred"] == 1

    detections = predictions[
        (predictions["inflection_probability"] >= threshold) &
        detection_mask
    ].copy()

    tp_candidates = detections[detections["is_inflection_true"] == 1]
    fp_candidates = detections[detections["is_inflection_true"] == 0]
    fn_candidates = predictions[
        (predictions["is_inflection_true"] == 1) &
        (predictions["is_inflection_pred"] == 0)
    ]

    rng_seed = 42

    def _draw_sample(df: pd.DataFrame, n: int, label: str) -> pd.DataFrame:
        if n <= 0 or df.empty:
            return pd.DataFrame(columns=df.columns)
        chosen = df.copy() if len(df) <= n else df.sample(n=n, random_state=rng_seed)
        chosen = chosen.copy()
        chosen["review_category"] = label
        return chosen

    samples = [
        _draw_sample(tp_candidates, sample_plan.get("TP", 0), "TP"),
        _draw_sample(fp_candidates, sample_plan.get("FP", 0), "FP"),
        _draw_sample(fn_candidates, sample_plan.get("FN", 0), "FN"),
    ]
    samples = [s for s in samples if not s.empty]

    if not samples:
        logger.warning("  No samples available for expert review export.")
        return output_dir / "expert_review_sample.csv"

    sample_df = pd.concat(samples, ignore_index=True)
    milestone_context = milestone_analysis[['lineage_id', 'inflection_quarter', 'nearest_milestone_id', 'inflection_type']].rename(
        columns={'inflection_quarter': 'quarter'}
    )
    sample_df = sample_df.merge(
        milestone_context,
        on=['lineage_id', 'quarter'],
        how='left'
    )
    sample_df['has_milestone'] = sample_df['nearest_milestone_id'].notna()
    if "new_works_yoy_pct" in sample_df.columns:
        yoy_values = sample_df["new_works_yoy_pct"]
    else:
        yoy_values = pd.Series([np.nan] * len(sample_df))
    sample_df["yoy_label"] = yoy_values.apply(format_yoy_pct)

    # Reorder columns to surface the new context metrics
    context_cols = [
        "new_works",
        "new_works_rolling_4q_mean",
        "new_works_rolling_4q_sum",
        "new_works_cumulative",
        "new_works_log_cumulative",
        "new_works_yoy_delta",
        "new_works_yoy_pct",
        "is_inflection_pred_persistent",
        "yoy_label",
    ]
    ordered_cols = [
        "lineage_id",
        "quarter",
        "is_inflection_true",
        "is_inflection_pred",
        "inflection_probability",
        "is_milestone_true",
        "is_milestone_pred",
        "milestone_probability",
        "detection_lag_quarters",
        "nearest_milestone_id",
        "inflection_type",
        "has_milestone",
        "review_category",
    ]
    for col in context_cols:
        if col in sample_df.columns:
            ordered_cols.append(col)

    missing = [c for c in ordered_cols if c not in sample_df.columns]
    ordered_cols = [c for c in ordered_cols if c in sample_df.columns]

    if missing:
        logger.debug(f"  Columns missing from sample export (skipped): {missing}")

    output_path = output_dir / "expert_review_sample.csv"
    sample_df.to_csv(output_path, index=False, columns=ordered_cols)
    logger.info(f"  Saved expert review sample ({len(sample_df)} rows) -> {output_path}")
    return output_path


def create_performance_table(threshold_sweep: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    """
    Generate performance summary table for key thresholds.
    """
    logger.info("Creating performance summary table...")

    # Select key thresholds
    key_thresholds = [0.05, 0.06, 0.07, 0.08]
    rows = []
    for key in key_thresholds:
        if threshold_sweep.empty:
            break
        idx = (threshold_sweep['threshold'] - key).abs().idxmin()
        row = threshold_sweep.loc[idx].copy()
        row['threshold'] = key  # label with target threshold
        rows.append(row)

    perf_table = pd.DataFrame(rows)
    if perf_table.empty:
        logger.warning("  Threshold sweep is empty - skipping performance table.")
        return pd.DataFrame()

    # Ensure expected columns exist even if not produced by sweep (e.g., no detections)
    for col in ['fpr', 'lag_median', 'lag_mean', 'lag_coverage']:
        if col not in perf_table.columns:
            perf_table[col] = np.nan

    # Select and format columns
    perf_table = perf_table[[
        'threshold', 'precision', 'recall', 'f1', 'fpr',
        'lag_median', 'lag_mean', 'lag_coverage'
    ]].copy()

    perf_table.columns = ['Threshold', 'Precision', 'Recall', 'F1', 'FPR',
                          'Lag Median', 'Lag Mean', 'Lag Coverage']

    # Format numbers
    for col in ['Precision', 'Recall', 'F1', 'FPR', 'Lag Coverage']:
        perf_table[col] = perf_table[col].apply(lambda x: f"{x:.3f}" if pd.notna(x) else "N/A")

    for col in ['Lag Median', 'Lag Mean']:
        perf_table[col] = perf_table[col].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "N/A")

    perf_table['Threshold'] = perf_table['Threshold'].apply(lambda x: f"{x:.2f}")

    # Save as CSV
    csv_path = output_dir / 'performance_summary_table.csv'
    perf_table.to_csv(csv_path, index=False)
    logger.info(f"  Saved: {csv_path}")

    # Create visualization
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.axis('off')

    table = ax.table(
        cellText=perf_table.values,
        colLabels=perf_table.columns,
        cellLoc='center',
        loc='center',
        bbox=[0, 0, 1, 1]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.5)

    # Style header
    for i in range(len(perf_table.columns)):
        table[(0, i)].set_facecolor('#2196F3')
        table[(0, i)].set_text_props(weight='bold', color='white')

    # Highlight threshold 0.07
    for i in range(len(perf_table)):
        if perf_table.iloc[i]['Threshold'] == '0.07':
            for j in range(len(perf_table.columns)):
                table[(i+1, j)].set_facecolor('#FFFFCC')

    plt.title('Performance Summary at Key Thresholds', fontsize=13, fontweight='bold', pad=20)

    img_path = output_dir / 'performance_summary_table.png'
    plt.savefig(img_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"  Saved: {img_path}")

    return perf_table


def comparative_analysis(
    predictions: pd.DataFrame,
    milestone_analysis: pd.DataFrame,
    milestone_summary_path: Path,
    threshold: float,
    output_dir: Path
) -> dict:
    """
    Cross-reference detection performance with milestone coverage and lag buckets.
    """
    logger.info("Running comparative analysis...")

    # Load milestone summary
    with open(milestone_summary_path) as f:
        ms_summary = json.load(f)

    # Filter predictions at threshold
    preds_at_threshold = predictions[predictions['inflection_probability'] >= threshold].copy()

    # Get detections
    detections = preds_at_threshold[
        (preds_at_threshold['is_inflection_true'] == 1) &
        (preds_at_threshold['is_inflection_pred'] == 1)
    ].copy()

    # Merge with milestone analysis
    detections = detections.merge(
        milestone_analysis[['lineage_id', 'inflection_quarter', 'nearest_milestone_id', 'lag_bucket', 'lag_since_milestone']].rename(columns={'inflection_quarter': 'quarter'}),
        on=['lineage_id', 'quarter'],
        how='left'
    )
    detections['has_milestone'] = detections['nearest_milestone_id'].notna()
    if detections.empty:
        logger.warning("No detections for comparative analysis; creating placeholder figure.")
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.axis('off')
        ax.text(0.5, 0.5, 'No detections at current threshold', ha='center', va='center', fontsize=12, fontweight='bold')
        output_path = output_dir / 'comparative_analysis.png'
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        return {
            'milestone_coverage_pct': ms_summary.get('coverage_pct', 0),
            'detection_with_milestone_pct': 0.0,
            'median_lag_with_milestone': None,
            'median_lag_without_milestone': None,
        }

    # Compute detection speed by milestone presence
    speed_by_milestone = detections.groupby('has_milestone')['detection_lag_quarters'].agg(['median', 'mean', 'count'])
    speed_by_milestone.index = ['No Milestone', 'Has Milestone']

    # Compute detection speed by lag bucket
    speed_by_bucket = detections[detections['lag_bucket'] != 'no_milestone'].groupby('lag_bucket')['detection_lag_quarters'].agg(['median', 'mean', 'count'])

    summary = {
        'milestone_coverage_pct': ms_summary.get('coverage_pct', 0),
        'detection_with_milestone_pct': float(detections['has_milestone'].sum() / len(detections) * 100),
        'median_lag_with_milestone': float(speed_by_milestone.loc['Has Milestone', 'median']) if 'Has Milestone' in speed_by_milestone.index else None,
        'median_lag_without_milestone': float(speed_by_milestone.loc['No Milestone', 'median']) if 'No Milestone' in speed_by_milestone.index else None
    }

    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))

    # 1. Detection speed by milestone presence
    ax = axes[0]
    speed_by_milestone = speed_by_milestone.reindex(['No Milestone', 'Has Milestone']).dropna(how='all')
    if len(speed_by_milestone) > 0:
        indices = np.arange(len(speed_by_milestone))
        bar_width = 0.35
        ax.bar(indices - bar_width / 2, speed_by_milestone['median'], bar_width,
               color='steelblue', edgecolor='black', label='Median Lag')
        ax.bar(indices + bar_width / 2, speed_by_milestone['mean'], bar_width,
               color='orange', edgecolor='black', label='Mean Lag')
        vals = speed_by_milestone[['median', 'mean']].values
        span = np.nanmax(np.abs(vals)) if vals.size else 0.0
        if not np.isfinite(span):
            span = 0.0
        span = max(span, 0.05)
        ax.set_ylim(-span * 1.2, span * 1.2)
        for i, (_, row) in enumerate(speed_by_milestone.iterrows()):
            ax.text(indices[i] - bar_width / 2, row['median'] + 0.02, f"{row['median']:.1f}",
                    ha='center', va='bottom', fontsize=8, color='white', fontweight='bold')
            ax.text(indices[i] + bar_width / 2, row['mean'] + 0.02, f"{row['mean']:.1f}",
                    ha='center', va='bottom', fontsize=8, color='black', fontweight='bold')
            ax.text(indices[i], span * 0.8, f"n={int(row['count'])}",
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
        ax.set_xticks(indices)
        ax.set_xticklabels(speed_by_milestone.index, rotation=0)
        ax.set_xlabel('Milestone Presence', fontsize=11)
        ax.set_ylabel('Detection Lag (quarters)', fontsize=11)
        ax.set_title('Detection Speed vs Milestone Presence', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3, axis='y')
    else:
        ax.axis('off')
        ax.text(0.5, 0.5, 'No detections to summarize', ha='center', va='center', fontsize=11)

    # 2. Detection speed by milestone lag bucket
    ax = axes[1]
    if len(speed_by_bucket) > 0:
        speed_by_bucket[['median', 'mean']].plot(kind='bar', ax=ax, color=['steelblue', 'orange'], edgecolor='black')
        ax.set_xlabel('Milestone Lag Bucket', fontsize=11)
        ax.set_ylabel('Detection Lag (quarters)', fontsize=11)
        ax.set_title('Detection Speed by Milestone Lag Bucket', fontsize=12, fontweight='bold')
        ax.legend(['Median Lag', 'Mean Lag'])
        ax.tick_params(axis='x', rotation=45)
        ax.grid(alpha=0.3, axis='y')

        vals = speed_by_bucket[['median', 'mean']].values
        max_abs = np.nanmax(np.abs(vals)) if vals.size else 0.0
        if not np.isfinite(max_abs):
            max_abs = 0.0
        y_pad = max(0.05, max_abs + 0.02)
        ax.set_ylim(-y_pad, y_pad)

        for i, (_idx, row) in enumerate(speed_by_bucket.iterrows()):
            ax.text(i, y_pad * 0.85, f"n={int(row['count'])}",
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
    else:
        ax.text(0.5, 0.5, 'No milestone-linked detections',
                ha='center', va='center', fontsize=12, transform=ax.transAxes)
        ax.set_xticks([])
        ax.set_yticks([])

    plt.suptitle(f'Milestone Coverage vs Detection Speed (threshold={threshold})',
                 fontsize=13, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    output_path = output_dir / 'comparative_analysis.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"  Saved: {output_path}")

    return summary


def main():
    parser = argparse.ArgumentParser(description='Evaluate inflection detection performance (S5)')
    parser.add_argument('--output-dir', type=Path, default=Path('data/out/figures/inflection_evaluation'),
                        help='Output directory for figures and tables')
    parser.add_argument('--threshold', type=float, default=0.07,
                        help='Detection threshold for analysis')
    parser.add_argument('--persistence-window', type=int, default=2,
                        help='Require detections to stay above threshold for this many quarters (set 1 to disable)')
    parser.add_argument('--n-timeline-plots', type=int, default=20,
                        help='Number of timeline plots to generate')
    parser.add_argument('--predictions-path', type=Path,
                        default=Path('data/out/experiments/msd_full_field/breakthrough_predictions.csv'),
                        help='Path to breakthrough_predictions.csv')
    parser.add_argument('--inflection-labels-path', type=Path,
                        default=Path('data/out/02_lineage_tracking/inflection_labels.csv'))
    parser.add_argument('--milestone-analysis-path', type=Path,
                        default=Path('data/out/analysis/milestone_inflection_analysis.csv'))
    parser.add_argument('--timeseries-path', type=Path,
                        default=Path('data/out/02_lineage_tracking/lineage_timeseries.csv'))
    parser.add_argument('--threshold-sweep-path', type=Path, default=None,
                        help='Optional precomputed threshold sweep CSV. If omitted/missing, it will be generated.')
    parser.add_argument('--field-metrics-path', type=Path,
                        default=Path('data/out/04_front_aggregation/field_metrics.parquet'))

    args = parser.parse_args()

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=== Inflection Detection Evaluation (S5) ===")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Threshold: {args.threshold}")
    logger.info(f"Persistence window: {args.persistence_window}Q (>= threshold)")

    # Define data paths
    predictions_path = args.predictions_path
    inflection_labels_path = args.inflection_labels_path
    milestone_analysis_path = args.milestone_analysis_path
    milestone_summary_path = Path('data/out/analysis/milestone_inflection_summary.json')
    timeseries_path = args.timeseries_path
    threshold_sweep_path = args.threshold_sweep_path or (predictions_path.parent / 'threshold_sweep.csv')

    # Load data
    predictions_raw, labels, milestone_analysis, timeseries, threshold_sweep = load_data(
        predictions_path, inflection_labels_path, milestone_analysis_path,
        timeseries_path, threshold_sweep_path
    )
    if threshold_sweep.empty:
        generated_sweep_path = args.output_dir / 'threshold_sweep_generated.csv'
        threshold_sweep = compute_threshold_sweep_from_predictions(
            predictions_raw,
            output_path=generated_sweep_path,
        )

    timeseries = add_seasonal_context(timeseries)
    predictions = prepare_predictions_with_context(
        predictions_raw,
        timeseries,
        threshold=args.threshold,
        persistence_window=args.persistence_window,
    )
    field_metrics_df = load_field_metrics_table(args.field_metrics_path)

    # Execute deliverables
    results = {}

    # 1. Lag distribution dashboard
    lag_summary = create_lag_distribution_dashboard(
        predictions, labels, milestone_analysis, args.threshold, args.output_dir
    )
    results['lag_distribution'] = lag_summary

    # 2. Timeline plots
    create_timeline_plots(
        predictions, labels, milestone_analysis, timeseries,
        args.threshold, args.n_timeline_plots, args.output_dir,
        persistence_col="is_inflection_pred_persistent"
    )

    # 3. False positive analysis
    fp_summary = analyze_false_positives(
        predictions, timeseries, milestone_analysis, args.threshold, args.output_dir,
        persistence_col="is_inflection_pred_persistent"
    )
    results['false_positives'] = fp_summary

    # 4. Performance summary table
    create_performance_table(threshold_sweep, args.output_dir)

    # 5. Comparative analysis
    comp_summary = comparative_analysis(
        predictions, milestone_analysis, milestone_summary_path,
        args.threshold, args.output_dir
    )
    results['comparative'] = comp_summary

    field_context_summary = create_field_vs_lineage_dashboard(
        predictions,
        timeseries,
        field_metrics_df,
        args.threshold,
        args.output_dir,
    )
    if field_context_summary:
        results['field_context'] = field_context_summary

    export_expert_review_sample(
        predictions,
        milestone_analysis,
        args.output_dir,
        args.threshold,
        persistence_col="is_inflection_pred_persistent",
    )

    # Save consolidated summary
    summary_path = args.output_dir / 'evaluation_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved evaluation summary: {summary_path}")

    logger.info("\n=== Evaluation Complete ===")
    logger.info(f"All outputs saved to: {args.output_dir}")
    logger.info("\nKey findings:")
    logger.info(f"  - Median detection lag: {lag_summary.get('median_lag', 'N/A'):.2f}Q")
    logger.info(f"  - Detection coverage: {lag_summary.get('n_detections', 0):,} / {len(labels):,} inflections")
    logger.info(f"  - False positives: {fp_summary.get('n_fps', 0):,}")
    logger.info(f"  - Milestone-linked detections: {lag_summary.get('milestone_pct', 0):.1f}%")


if __name__ == '__main__':
    main()
