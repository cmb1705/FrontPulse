#!/usr/bin/env python3
"""
Comprehensive Tripwire Visualization Suite

Creates informative visualizations for tripwire validation results:
1. Timeline with milestone markers and alert spikes
2. Precision-Recall analysis with false positive breakdown
3. Z-score distribution by front
4. Alert ranking quality (precision@K curve)
5. Detection performance heatmap
6. Lead time vs magnitude scatter
"""
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import seaborn as sns
from pathlib import Path
from datetime import datetime
import warnings

# Suppress matplotlib GUI
plt.switch_backend('Agg')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.size'] = 9

# Suppress pandas datetime parsing warnings for quarter strings
warnings.filterwarnings('ignore', message='Could not infer format')
# Suppress tight_layout warnings for complex multi-axes figures
warnings.filterwarnings('ignore', message='This figure includes Axes that are not compatible with tight_layout')

def parse_quarter_to_datetime(quarter_series):
    """
    Parse quarter strings (e.g., '2010Q1') to datetime objects.
    Converts to first day of the quarter for plotting.
    """
    # Use pandas period conversion which handles quarter format natively
    return pd.PeriodIndex(quarter_series, freq='Q').to_timestamp()

def load_data(outdir: Path):
    """Load validation results and alerts."""
    alerts_path = outdir / "tripwire_alerts.csv"
    validation_path = outdir / "validation_results.csv"

    if not alerts_path.exists():
        raise FileNotFoundError(f"Tripwire alert file not found: {alerts_path}")
    if not validation_path.exists():
        raise FileNotFoundError(f"Tripwire validation file not found: {validation_path}")

    alerts = pd.read_csv(alerts_path)
    validation = pd.read_csv(validation_path)
    return alerts, validation

def compute_precision_recall_metrics(alerts, validation):
    """
    Compute precision and recall with false positive analysis.

    Precision = TP / (TP + FP)
    Recall = TP / (TP + FN)

    Where:
    - TP = milestones detected (significant alert in detection window)
    - FP = significant alerts NOT in any milestone window
    - FN = milestones missed (no significant alert in window)
    """
    # Get significant alerts
    sig_alerts = alerts[alerts['significant'] == True].copy()

    # For each significant alert, check if it matches any milestone
    # Milestone detection window is typically event_quarter ± 1 quarter
    validation['quarter'] = pd.to_datetime(validation['event_quarter'])
    sig_alerts['quarter_dt'] = pd.to_datetime(sig_alerts['quarter'])

    # Mark each significant alert as TP or FP
    sig_alerts['is_true_positive'] = False
    for idx, alert in sig_alerts.iterrows():
        # Check if this alert falls within any milestone's window
        for _, event in validation.iterrows():
            window_start = event['quarter'] - pd.DateOffset(months=3)  # 1 quarter before
            window_end = event['quarter'] + pd.DateOffset(months=3)    # 1 quarter after

            if window_start <= alert['quarter_dt'] <= window_end:
                sig_alerts.at[idx, 'is_true_positive'] = True
                break

    # Compute metrics
    tp = sig_alerts['is_true_positive'].sum()
    fp = (~sig_alerts['is_true_positive']).sum()
    fn = (~validation['detected']).sum()
    tn = len(alerts) - len(sig_alerts) - fn  # Non-significant alerts that are true negatives

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return {
        'tp': int(tp),
        'fp': int(fp),
        'fn': int(fn),
        'tn': int(tn),
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'false_positive_alerts': sig_alerts[~sig_alerts['is_true_positive']]
    }

def plot_timeline_with_milestones(alerts, validation, outdir):
    """
    Timeline showing publication counts by front, with milestone markers and alerts.
    """
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(4, 1, height_ratios=[1, 1, 1, 1], hspace=0.3)

    # Get top 4 most active fronts
    front_activity = alerts.groupby('community_id')['observed'].sum().sort_values(ascending=False)
    top_fronts = front_activity.head(4).index.tolist()

    for i, front in enumerate(top_fronts):
        ax = fig.add_subplot(gs[i])

        # Get data for this front
        front_data = alerts[alerts['community_id'] == front].copy()
        front_data['quarter_dt'] = pd.to_datetime(front_data['quarter'])
        front_data = front_data.sort_values('quarter_dt')

        # Plot observed counts
        ax.plot(front_data['quarter_dt'], front_data['observed'],
                'o-', color='steelblue', linewidth=1.5, markersize=4, label='Observed', alpha=0.7)

        # Plot expected baseline
        ax.plot(front_data['quarter_dt'], front_data['expected'],
                '--', color='gray', linewidth=1, label='Expected (NB model)', alpha=0.6)

        # Highlight significant alerts
        sig = front_data[front_data['significant'] == True]
        if len(sig) > 0:
            ax.scatter(sig['quarter_dt'], sig['observed'],
                      s=100, marker='*', color='red', label='Significant alert',
                      zorder=5, edgecolor='darkred', linewidth=0.5)

        # Mark milestone events for this front
        validation_copy = validation.copy()
        validation_copy['quarter_dt'] = pd.to_datetime(validation_copy['event_quarter'])

        # Simple heuristic: if front name appears in milestone description or mapped_fronts
        front_milestones = validation_copy[
            validation_copy['description'].str.contains(front.replace('_', ' '), case=False, na=False) |
            validation_copy['event_id'].str.contains(front, case=False, na=False)
        ]

        for _, event in front_milestones.iterrows():
            color = 'green' if event['detected'] else 'red'
            marker = 'v' if event['detected'] else 'x'
            ax.axvline(event['quarter_dt'], color=color, linestyle=':', alpha=0.3, linewidth=1.5)
            # Only apply edgecolor to filled markers (not 'x')
            scatter_kwargs = {'marker': marker, 's': 80, 'color': color, 'zorder': 10}
            if marker != 'x':
                scatter_kwargs.update({'edgecolor': 'black', 'linewidth': 0.5})
            ax.scatter(event['quarter_dt'], ax.get_ylim()[1] * 0.95, **scatter_kwargs)

        # Styling
        ax.set_ylabel('Publications', fontweight='bold')
        ax.set_title(f"{front.replace('_', ' ').title()}", fontsize=11, fontweight='bold')
        ax.legend(loc='upper left', fontsize=8, framealpha=0.9)
        ax.grid(True, alpha=0.3)

        if i == len(top_fronts) - 1:
            ax.set_xlabel('Quarter', fontweight='bold')

    plt.suptitle('Tripwire Timeline: Publication Spikes, Alerts, and Milestone Events',
                 fontsize=14, fontweight='bold', y=0.995)

    # Add legend for milestone markers
    green_v = mpatches.Patch(color='green', label='Detected milestone')
    red_x = mpatches.Patch(color='red', label='Missed milestone')
    fig.legend(handles=[green_v, red_x], loc='upper right', fontsize=9,
              bbox_to_anchor=(0.98, 0.98))

    plt.tight_layout(rect=[0, 0, 1, 0.99])  # Leave space for suptitle
    plt.savefig(outdir / "01_timeline_comprehensive.png", dpi=150, bbox_inches='tight')
    plt.close()

def plot_precision_recall_analysis(metrics, outdir):
    """
    Precision-Recall analysis with confusion matrix and false positive breakdown.
    """
    fig = plt.figure(figsize=(14, 6))
    gs = GridSpec(1, 3, width_ratios=[1, 1, 1.2], wspace=0.35)

    # 1. Confusion Matrix
    ax1 = fig.add_subplot(gs[0])
    cm = np.array([[metrics['tn'], metrics['fp']],
                   [metrics['fn'], metrics['tp']]])

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax1,
                linewidths=2, linecolor='black',
                xticklabels=['Predicted Negative', 'Predicted Positive'],
                yticklabels=['Actual Negative', 'Actual Positive'])

    ax1.set_title('Confusion Matrix\n(Alert Level)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Ground Truth', fontweight='bold')
    ax1.set_xlabel('Tripwire Prediction', fontweight='bold')

    # Add text annotations
    total = cm.sum()
    ax1.text(0.5, -0.25, f"Accuracy: {(cm[0,0] + cm[1,1]) / total:.1%}",
             transform=ax1.transAxes, ha='center', fontsize=9, style='italic')

    # 2. Precision-Recall Bar Chart
    ax2 = fig.add_subplot(gs[1])

    metric_names = ['Precision', 'Recall', 'F1-Score']
    metric_values = [metrics['precision'], metrics['recall'], metrics['f1']]
    colors = ['#2ecc71', '#3498db', '#9b59b6']

    bars = ax2.barh(metric_names, metric_values, color=colors, alpha=0.7, edgecolor='black')

    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, metric_values)):
        ax2.text(val + 0.02, i, f'{val:.1%}', va='center', fontweight='bold')

    ax2.set_xlim(0, 1.1)
    ax2.set_xlabel('Score', fontweight='bold')
    ax2.set_title('Detection Performance Metrics', fontsize=11, fontweight='bold')
    ax2.axvline(0.8, color='gray', linestyle='--', alpha=0.5, linewidth=1, label='80% target')
    ax2.legend(fontsize=8)
    ax2.grid(axis='x', alpha=0.3)

    # 3. False Positive Breakdown
    ax3 = fig.add_subplot(gs[2])

    # Group false positives by front
    fp_alerts = metrics['false_positive_alerts']
    if len(fp_alerts) > 0:
        fp_by_front = fp_alerts['community_id'].value_counts().head(8)

        bars = ax3.barh(range(len(fp_by_front)), fp_by_front.values,
                       color='salmon', alpha=0.7, edgecolor='darkred')
        ax3.set_yticks(range(len(fp_by_front)))
        ax3.set_yticklabels([f.replace('_', ' ').title() for f in fp_by_front.index], fontsize=8)
        ax3.set_xlabel('False Positive Alerts', fontweight='bold')
        ax3.set_title(f'False Positive Distribution\n({len(fp_alerts)} total FP alerts)',
                     fontsize=11, fontweight='bold')

        # Add value labels
        for i, (bar, val) in enumerate(zip(bars, fp_by_front.values)):
            ax3.text(val + 0.1, i, f'{val}', va='center', fontweight='bold', fontsize=9)

        ax3.grid(axis='x', alpha=0.3)
    else:
        ax3.text(0.5, 0.5, 'No False Positives!', ha='center', va='center',
                fontsize=14, fontweight='bold', color='green',
                transform=ax3.transAxes)
        ax3.set_xticks([])
        ax3.set_yticks([])

    plt.suptitle('Precision-Recall Analysis: True vs False Positives',
                 fontsize=14, fontweight='bold', y=1.02)

    plt.savefig(outdir / "02_precision_recall_analysis.png", dpi=150, bbox_inches='tight')
    plt.close()

def plot_zscore_distributions(alerts, outdir):
    """
    Z-score distributions showing spike detection characteristics.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Overall z-score distribution
    ax1 = axes[0, 0]

    sig = alerts[alerts['significant'] == True]
    non_sig = alerts[alerts['significant'] == False]

    ax1.hist(non_sig['z_score'], bins=50, alpha=0.5, label='Non-significant',
            color='gray', edgecolor='black')
    ax1.hist(sig['z_score'], bins=30, alpha=0.7, label='Significant (FDR q=0.1)',
            color='red', edgecolor='darkred')

    ax1.axvline(alerts['z_score'].mean(), color='blue', linestyle='--',
               linewidth=2, label=f'Mean z = {alerts["z_score"].mean():.2f}')
    ax1.axvline(0, color='black', linestyle='-', linewidth=1, alpha=0.5)

    ax1.set_xlabel('Z-Score', fontweight='bold')
    ax1.set_ylabel('Frequency', fontweight='bold')
    ax1.set_title('Z-Score Distribution (All Alerts)', fontsize=11, fontweight='bold')
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)

    # 2. Z-score by front (top 8 fronts)
    ax2 = axes[0, 1]

    top_fronts = alerts.groupby('community_id')['observed'].sum().nlargest(8).index
    front_data = []
    front_labels = []

    for front in top_fronts:
        front_z = alerts[alerts['community_id'] == front]['z_score']
        front_data.append(front_z)
        front_labels.append(front.replace('_', '\n').title())

    bp = ax2.boxplot(front_data, tick_labels=front_labels, patch_artist=True,
                    showfliers=False, medianprops={'color': 'red', 'linewidth': 2})

    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')
        patch.set_alpha(0.7)

    ax2.axhline(0, color='black', linestyle='-', linewidth=1, alpha=0.5)
    ax2.set_ylabel('Z-Score', fontweight='bold')
    ax2.set_title('Z-Score by Research Front', fontsize=11, fontweight='bold')
    ax2.tick_params(axis='x', rotation=0, labelsize=7)
    ax2.grid(axis='y', alpha=0.3)

    # 3. Z-score vs observed counts (scatter)
    ax3 = axes[1, 0]

    scatter = ax3.scatter(alerts['observed'], alerts['z_score'],
                         c=alerts['significant'], cmap='RdYlGn',
                         alpha=0.5, s=20, edgecolor='none')

    ax3.set_xlabel('Observed Publications', fontweight='bold')
    ax3.set_ylabel('Z-Score', fontweight='bold')
    ax3.set_title('Z-Score vs Publication Count', fontsize=11, fontweight='bold')
    ax3.axhline(0, color='black', linestyle='-', linewidth=1, alpha=0.5)
    ax3.grid(alpha=0.3)

    cbar = plt.colorbar(scatter, ax=ax3)
    cbar.set_label('Significant', fontweight='bold')

    # 4. Expected vs Observed (residual analysis)
    ax4 = axes[1, 1]

    residuals = alerts['observed'] - alerts['expected']

    ax4.scatter(alerts['expected'], residuals,
               c=alerts['significant'], cmap='RdYlGn',
               alpha=0.5, s=20, edgecolor='none')

    ax4.axhline(0, color='black', linestyle='-', linewidth=2, alpha=0.7)
    ax4.set_xlabel('Expected Publications (NB Model)', fontweight='bold')
    ax4.set_ylabel('Residual (Observed - Expected)', fontweight='bold')
    ax4.set_title('Model Residuals: Spike Detection', fontsize=11, fontweight='bold')
    ax4.grid(alpha=0.3)

    plt.suptitle('Z-Score Analysis: Spike Detection Characteristics',
                 fontsize=14, fontweight='bold', y=1.00)

    plt.tight_layout()
    plt.savefig(outdir / "03_zscore_distributions.png", dpi=150, bbox_inches='tight')
    plt.close()

def plot_detection_performance_heatmap(validation, outdir):
    """
    Heatmap showing detection status by magnitude and event type.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 1. Detection by magnitude and type
    ax1 = axes[0]

    pivot = validation.pivot_table(
        index='magnitude',
        columns='event_type',
        values='detected',
        aggfunc=['sum', 'count']
    )

    # Compute detection rates
    detection_rate = pivot['sum'] / pivot['count']

    sns.heatmap(detection_rate, annot=True, fmt='.1%', cmap='RdYlGn',
               vmin=0, vmax=1, cbar_kws={'label': 'Detection Rate'},
               ax=ax1, linewidths=1, linecolor='black')

    ax1.set_xlabel('Event Type', fontweight='bold')
    ax1.set_ylabel('Magnitude', fontweight='bold')
    ax1.set_title('Detection Rate by Magnitude & Type', fontsize=11, fontweight='bold')

    # 2. Timeline of detections
    ax2 = axes[1]

    validation_copy = validation.copy()
    validation_copy['year'] = pd.to_datetime(validation_copy['event_quarter']).dt.year
    validation_copy['quarter_num'] = pd.to_datetime(validation_copy['event_quarter']).dt.quarter

    yearly_pivot = validation_copy.pivot_table(
        index='magnitude',
        columns='year',
        values='detected',
        aggfunc=['sum', 'count']
    )

    if len(yearly_pivot.columns.levels[1]) > 0:
        detection_by_year = yearly_pivot['sum'] / yearly_pivot['count']
        detection_by_year = detection_by_year.fillna(0)

        sns.heatmap(detection_by_year, annot=True, fmt='.1%', cmap='RdYlGn',
                   vmin=0, vmax=1, cbar_kws={'label': 'Detection Rate'},
                   ax=ax2, linewidths=1, linecolor='black')

        ax2.set_xlabel('Year', fontweight='bold')
        ax2.set_ylabel('Magnitude', fontweight='bold')
        ax2.set_title('Detection Rate Over Time', fontsize=11, fontweight='bold')

    plt.suptitle('Detection Performance Breakdown',
                 fontsize=14, fontweight='bold', y=1.02)

    plt.tight_layout()
    plt.savefig(outdir / "04_detection_performance_heatmap.png", dpi=150, bbox_inches='tight')
    plt.close()

def plot_precision_at_k_curve(alerts, validation, outdir):
    """
    Precision@K curve showing alert ranking quality.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Sort alerts by composite score (descending)
    alerts_sorted = alerts.sort_values('composite_score', ascending=False).copy()

    # For each alert, check if it's within detection window of any milestone
    validation_copy = validation.copy()
    validation_copy['quarter_dt'] = pd.to_datetime(validation_copy['event_quarter'])
    alerts_sorted['quarter_dt'] = pd.to_datetime(alerts_sorted['quarter'])

    alerts_sorted['is_relevant'] = False
    for idx, alert in alerts_sorted.iterrows():
        for _, event in validation_copy.iterrows():
            window_start = event['quarter_dt'] - pd.DateOffset(months=3)
            window_end = event['quarter_dt'] + pd.DateOffset(months=3)

            if window_start <= alert['quarter_dt'] <= window_end:
                alerts_sorted.at[idx, 'is_relevant'] = True
                break

    # Compute precision@K for K = 1 to 100
    k_values = range(1, min(101, len(alerts_sorted) + 1))
    precisions = []

    for k in k_values:
        top_k = alerts_sorted.head(k)
        precision_k = top_k['is_relevant'].sum() / k
        precisions.append(precision_k)

    # 1. Precision@K curve
    ax1 = axes[0]

    ax1.plot(k_values, precisions, 'o-', color='steelblue', linewidth=2, markersize=4)
    ax1.axhline(0.6, color='red', linestyle='--', label='Target: 60%', linewidth=2)
    ax1.axhline(0.5, color='orange', linestyle='--', label='Target: 50%', linewidth=2)

    # Mark specific K values
    for k in [10, 20, 30]:
        if k < len(precisions):
            ax1.scatter([k], [precisions[k-1]], s=100, color='red', zorder=5,
                       edgecolor='black', linewidth=1.5)
            ax1.text(k, precisions[k-1] + 0.05, f'K={k}\n{precisions[k-1]:.1%}',
                    ha='center', fontsize=8, fontweight='bold')

    ax1.set_xlabel('K (Number of Top Alerts)', fontweight='bold')
    ax1.set_ylabel('Precision@K', fontweight='bold')
    ax1.set_title('Alert Ranking Quality: Precision@K', fontsize=11, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.3)
    ax1.set_ylim(0, 1.05)

    # 2. Composite score distribution
    ax2 = axes[1]

    relevant = alerts_sorted[alerts_sorted['is_relevant'] == True]
    irrelevant = alerts_sorted[alerts_sorted['is_relevant'] == False]

    ax2.hist(irrelevant['composite_score'], bins=40, alpha=0.5,
            label='Irrelevant alerts', color='gray', edgecolor='black')
    ax2.hist(relevant['composite_score'], bins=30, alpha=0.7,
            label='Relevant alerts', color='green', edgecolor='darkgreen')

    ax2.set_xlabel('Composite Score', fontweight='bold')
    ax2.set_ylabel('Frequency', fontweight='bold')
    ax2.set_title('Composite Score Distribution', fontsize=11, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)

    plt.suptitle('Alert Ranking Analysis: How Well Does Composite Score Prioritize True Events?',
                 fontsize=13, fontweight='bold', y=1.02)

    plt.tight_layout()
    plt.savefig(outdir / "05_precision_at_k_curve.png", dpi=150, bbox_inches='tight')
    plt.close()

def plot_lead_time_analysis(validation, outdir):
    """
    Lead time analysis with magnitude breakdown.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    detected = validation[validation['detected'] == True].copy()

    # 1. Lead time distribution
    ax1 = axes[0, 0]

    if len(detected) > 0:
        lead_times = detected['lead_time_quarters'].dropna()

        ax1.hist(lead_times, bins=[-0.5, 0.5, 1.5, 2.5, 3.5],
                color='steelblue', alpha=0.7, edgecolor='black')

        ax1.axvline(lead_times.median(), color='red', linestyle='--',
                   linewidth=2, label=f'Median = {lead_times.median():.1f}q')
        ax1.axvline(1, color='green', linestyle=':', linewidth=2,
                   label='Target = 1q', alpha=0.7)

        ax1.set_xlabel('Lead Time (Quarters)', fontweight='bold')
        ax1.set_ylabel('Number of Events', fontweight='bold')
        ax1.set_title('Lead Time Distribution', fontsize=11, fontweight='bold')
        ax1.legend(fontsize=9)
        ax1.grid(alpha=0.3)

    # 2. Lead time vs magnitude
    ax2 = axes[0, 1]

    if len(detected) > 0:
        for mag in sorted(detected['magnitude'].unique()):
            mag_data = detected[detected['magnitude'] == mag]
            lead = mag_data['lead_time_quarters'].dropna()

            if len(lead) > 0:
                ax2.scatter([mag] * len(lead), lead, s=60, alpha=0.6,
                           label=f'Mag {mag} (n={len(lead)})')

        ax2.axhline(1, color='green', linestyle=':', linewidth=2, alpha=0.5)
        ax2.set_xlabel('Event Magnitude', fontweight='bold')
        ax2.set_ylabel('Lead Time (Quarters)', fontweight='bold')
        ax2.set_title('Lead Time by Magnitude', fontsize=11, fontweight='bold')
        ax2.legend(fontsize=8, loc='best')
        ax2.grid(alpha=0.3)

    # 3. Detection window analysis
    ax3 = axes[1, 0]

    if len(detected) > 0:
        lead_counts = detected['lead_time_quarters'].value_counts().sort_index()

        colors = ['green' if x > 0 else 'orange' if x == 0 else 'red'
                 for x in lead_counts.index]

        bars = ax3.bar(lead_counts.index, lead_counts.values,
                      color=colors, alpha=0.7, edgecolor='black')

        # Add value labels
        for bar, val in zip(bars, lead_counts.values):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val}', ha='center', va='bottom', fontweight='bold')

        ax3.set_xlabel('Lead Time (Quarters)', fontweight='bold')
        ax3.set_ylabel('Events Detected', fontweight='bold')
        ax3.set_title('Detection Timing Breakdown', fontsize=11, fontweight='bold')
        ax3.grid(axis='y', alpha=0.3)

    # 4. Lead time statistics table
    ax4 = axes[1, 1]
    ax4.axis('off')

    if len(detected) > 0:
        lead_times = detected['lead_time_quarters'].dropna()

        stats_data = [
            ['Metric', 'Value'],
            ['Median lead time', f'{lead_times.median():.2f} quarters'],
            ['Mean lead time', f'{lead_times.mean():.2f} quarters'],
            ['Std deviation', f'{lead_times.std():.2f} quarters'],
            ['Min lead time', f'{lead_times.min():.0f} quarters'],
            ['Max lead time', f'{lead_times.max():.0f} quarters'],
            ['', ''],
            ['Same-quarter (lead=0)', f'{(lead_times == 0).sum()}/{len(lead_times)} ({(lead_times == 0).mean():.1%})'],
            ['Positive lead (>0q)', f'{(lead_times > 0).sum()}/{len(lead_times)} ({(lead_times > 0).mean():.1%})'],
            ['Negative lead (<0q)', f'{(lead_times < 0).sum()}/{len(lead_times)} ({(lead_times < 0).mean():.1%})'],
        ]

        table = ax4.table(cellText=stats_data, loc='center', cellLoc='left',
                         colWidths=[0.5, 0.5])

        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2)

        # Style header row
        for i in range(2):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')

        # Style data rows
        for i in range(1, len(stats_data)):
            for j in range(2):
                if i == 6:  # Empty row
                    continue
                table[(i, j)].set_facecolor('#f0f0f0' if i % 2 == 0 else 'white')

        ax4.set_title('Lead Time Statistics', fontsize=11, fontweight='bold', pad=20)

    plt.suptitle('Lead Time Analysis: How Early Does Tripwire Detect Events?',
                 fontsize=14, fontweight='bold', y=0.98)

    plt.tight_layout()
    plt.savefig(outdir / "06_lead_time_analysis.png", dpi=150, bbox_inches='tight')
    plt.close()

def create_summary_dashboard(metrics, validation, alerts, outdir):
    """
    One-page summary dashboard with key metrics.
    """
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(3, 3, hspace=0.4, wspace=0.4)

    # Title
    fig.suptitle('Tripwire Validation Dashboard - Summary View',
                 fontsize=16, fontweight='bold', y=0.98)

    # 1. Overall metrics (large display)
    ax1 = fig.add_subplot(gs[0, :])
    ax1.axis('off')

    detected = validation[validation['detected'] == True]
    detection_rate = len(detected) / len(validation)

    metrics_text = f"""
    OVERALL PERFORMANCE

    Detection Rate: {detection_rate:.1%} ({len(detected)}/{len(validation)} events)
    Precision: {metrics['precision']:.1%}    Recall: {metrics['recall']:.1%}    F1-Score: {metrics['f1']:.1%}

    False Positives: {metrics['fp']}    False Negatives: {metrics['fn']}    True Positives: {metrics['tp']}

    Max Z-Score: {alerts['z_score'].max():.1f}    Mean Z-Score: {alerts['z_score'].mean():.2f}    Significant Alerts: {alerts['significant'].sum()}/{ len(alerts)}
    """

    ax1.text(0.5, 0.5, metrics_text, ha='center', va='center',
            fontsize=12, family='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    # 2. Detection by magnitude
    ax2 = fig.add_subplot(gs[1, 0])

    mag_stats = validation.groupby('magnitude').agg({
        'detected': ['sum', 'count']
    })
    mag_stats['rate'] = mag_stats[('detected', 'sum')] / mag_stats[('detected', 'count')]

    bars = ax2.bar(mag_stats.index, mag_stats['rate'],
                   color=['red', 'orange', 'yellow', 'lightgreen'],
                   alpha=0.7, edgecolor='black')

    for bar, rate, count in zip(bars, mag_stats['rate'], mag_stats[('detected', 'count')]):
        ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.05,
                f'{rate:.0%}\n({int(mag_stats.loc[bar.get_x() + bar.get_width()/2., ("detected", "sum")])}/{int(count)})',
                ha='center', fontsize=9, fontweight='bold')

    ax2.set_xlabel('Magnitude', fontweight='bold')
    ax2.set_ylabel('Detection Rate', fontweight='bold')
    ax2.set_title('Detection by Magnitude', fontsize=10, fontweight='bold')
    ax2.set_ylim(0, 1.2)
    ax2.axhline(0.8, color='green', linestyle='--', alpha=0.5)
    ax2.grid(axis='y', alpha=0.3)

    # 3. Lead time pie chart
    ax3 = fig.add_subplot(gs[1, 1])

    lead_cats = detected['lead_time_quarters'].apply(
        lambda x: '2Q ahead' if x >= 2 else '1Q ahead' if x == 1 else 'Same quarter'
    ).value_counts()

    colors_pie = ['#2ecc71', '#f39c12', '#e74c3c']
    ax3.pie(lead_cats.values, labels=lead_cats.index, autopct='%1.1f%%',
           colors=colors_pie, startangle=90)
    ax3.set_title('Lead Time Distribution', fontsize=10, fontweight='bold')

    # 4. Confusion matrix
    ax4 = fig.add_subplot(gs[1, 2])

    cm = np.array([[metrics['tn'], metrics['fp']],
                   [metrics['fn'], metrics['tp']]])

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax4,
                linewidths=2, linecolor='black',
                xticklabels=['Pred-', 'Pred+'],
                yticklabels=['True-', 'True+'])

    ax4.set_title('Confusion Matrix', fontsize=10, fontweight='bold')

    # 5-7. Top performing fronts (timeseries)
    top_fronts = alerts.groupby('community_id')['observed'].sum().nlargest(3).index

    for i, front in enumerate(top_fronts):
        ax = fig.add_subplot(gs[2, i])

        front_data = alerts[alerts['community_id'] == front].copy()
        front_data['quarter_dt'] = pd.to_datetime(front_data['quarter'])
        front_data = front_data.sort_values('quarter_dt')

        ax.plot(front_data['quarter_dt'], front_data['observed'],
               'o-', color='steelblue', linewidth=1, markersize=3)
        ax.fill_between(front_data['quarter_dt'], 0, front_data['observed'],
                       alpha=0.3, color='steelblue')

        sig = front_data[front_data['significant'] == True]
        if len(sig) > 0:
            ax.scatter(sig['quarter_dt'], sig['observed'],
                      marker='*', s=80, color='red', zorder=5)

        ax.set_title(f"{front.replace('_', ' ').title()}",
                    fontsize=9, fontweight='bold')
        ax.tick_params(axis='x', rotation=45, labelsize=7)
        ax.tick_params(axis='y', labelsize=7)
        ax.grid(alpha=0.3)

        if i == 0:
            ax.set_ylabel('Publications', fontsize=8, fontweight='bold')

    plt.savefig(outdir / "00_dashboard_summary.png", dpi=150, bbox_inches='tight')
    plt.close()

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Comprehensive Tripwire visualization suite"
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("data/out/06_validation"),
        help="Directory containing tripwire_alerts.csv and validation_results.csv "
             "(default: data/out/06_validation)",
    )
    return parser.parse_args()

def main():
    """Generate all visualizations."""
    args = parse_args()
    outdir: Path = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    alerts, validation = load_data(outdir)

    print("Computing precision/recall metrics...")
    metrics = compute_precision_recall_metrics(alerts, validation)

    print(f"\n=== PRECISION/RECALL METRICS ===")
    print(f"True Positives (TP): {metrics['tp']}")
    print(f"False Positives (FP): {metrics['fp']}")
    print(f"False Negatives (FN): {metrics['fn']}")
    print(f"True Negatives (TN): {metrics['tn']}")
    print(f"\nPrecision: {metrics['precision']:.1%}")
    print(f"Recall: {metrics['recall']:.1%}")
    print(f"F1-Score: {metrics['f1']:.1%}")
    print(f"\nFalse Positive Alerts: {len(metrics['false_positive_alerts'])}")

    print("\nGenerating visualizations...")
    print("  [1/7] Summary dashboard...")
    create_summary_dashboard(metrics, validation, alerts, outdir)

    print("  [2/7] Precision-recall analysis...")
    plot_precision_recall_analysis(metrics, outdir)

    print("  [3/7] Z-score distributions...")
    plot_zscore_distributions(alerts, outdir)

    print("  [4/7] Detection performance heatmap...")
    plot_detection_performance_heatmap(validation, outdir)

    print("  [5/7] Precision@K curve...")
    plot_precision_at_k_curve(alerts, validation, outdir)

    print("  [6/7] Lead time analysis...")
    plot_lead_time_analysis(validation, outdir)

    print("  [7/7] Timeline with milestones...")
    plot_timeline_with_milestones(alerts, validation, outdir)

    print(f"\n[OK] All visualizations saved to {outdir}/")
    print("  - 00_dashboard_summary.png")
    print("  - 01_timeline_comprehensive.png")
    print("  - 02_precision_recall_analysis.png")
    print("  - 03_zscore_distributions.png")
    print("  - 04_detection_performance_heatmap.png")
    print("  - 05_precision_at_k_curve.png")
    print("  - 06_lead_time_analysis.png")

if __name__ == '__main__':
    main()
