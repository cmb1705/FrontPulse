#!/usr/bin/env python3
"""
Field vs Front Comparative Diagnostics (Task 4.2)

Generates comparative analysis between field-level and front-level (lineage) breakthrough detections.

Usage:
    python scripts/field_front_diagnostics.py \\
        --field-predictions data/out/experiments/msd_field_predictions.csv \\
        --front-predictions data/out/03_ensemble_mapping/front_lineage_predictions.csv \\
        --output-dir data/out/experiments/field_front_diagnostics

Author: Multi-Signal Context Integration (Task 4.2)
Date: 2025-11-06
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.figure import Figure

# Add src to path for imports
repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root / 'src'))


def load_field_predictions(path: Path) -> pd.DataFrame:
    """Load field-level breakthrough predictions."""
    print(f"[1/6] Loading field-level predictions from {path}")
    df = pd.read_csv(path)
    print(f"   Loaded {len(df)} quarterly predictions")
    print(f"   Columns: {list(df.columns)}")
    return df


def load_front_predictions(path: Path) -> pd.DataFrame:
    """Load front-level (lineage) breakthrough predictions."""
    print(f"\n[2/6] Loading front-level predictions from {path}")

    if not path.exists():
        print(f"\nERROR: Front predictions file not found: {path}")
        print(f"\nFront-level predictions are required for comparative diagnostics.")
        print(f"Please run lineage-level MSD first to generate predictions:")
        print(f"  python scripts/multi_signal_detector.py --train --predict ...")
        print(f"\nAlternatively, specify the correct path with --front-predictions")
        sys.exit(1)

    df = pd.read_csv(path)
    print(f"   Loaded {len(df)} lineage-level predictions")
    print(f"   Columns: {list(df.columns)}")
    return df


def aggregate_front_to_quarter(df_front: pd.DataFrame) -> pd.DataFrame:
    """Aggregate front-level predictions to quarter level for comparison."""
    print(f"\n[3/6] Aggregating front-level predictions to quarterly statistics")

    grouped = df_front.groupby('quarter').agg({
        'lineage_id': 'count',  # Number of lineages with predictions
        'probability': ['mean', 'median', 'max', 'std'],
        'prediction': 'sum'  # Number of positive predictions
    }).reset_index()

    # Flatten column names
    grouped.columns = [
        'quarter',
        'num_front_lineages',
        'front_prob_mean',
        'front_prob_median',
        'front_prob_max',
        'front_prob_std',
        'front_positive_count'
    ]

    print(f"   Aggregated to {len(grouped)} quarters")
    print(f"   Average positive predictions per quarter: {grouped['front_positive_count'].mean():.1f}")

    return grouped


def merge_field_front_data(df_field: pd.DataFrame, df_front_agg: pd.DataFrame) -> pd.DataFrame:
    """Merge field and front data for comparison."""
    print(f"\n[4/6] Merging field and front data")

    df_merged = df_field.merge(df_front_agg, on='quarter', how='outer')

    # Sort by quarter
    df_merged = df_merged.sort_values('quarter')

    print(f"   Merged dataset: {len(df_merged)} quarters")
    print(f"   Overlapping quarters: {df_merged[['probability', 'front_prob_mean']].notna().all(axis=1).sum()}")

    return df_merged


def create_trend_plots(df: pd.DataFrame, output_dir: Path) -> None:
    """Create trend comparison plots."""
    print(f"\n[5/6] Creating comparative trend plots")

    # Convert quarter to datetime for plotting (use period instead of manual parsing)
    df['quarter_date'] = pd.PeriodIndex(df['quarter'], freq='Q').to_timestamp()

    # Create figure with multiple subplots
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    fig.suptitle('Field vs Front Breakthrough Detection: Comparative Analysis', fontsize=16, fontweight='bold')

    # Plot 1: Probability comparison
    ax1 = axes[0]
    ax1.plot(df['quarter_date'], df['probability'], 'b-', linewidth=2, label='Field-level probability', marker='o')
    ax1.plot(df['quarter_date'], df['front_prob_mean'], 'r-', linewidth=2, label='Front-level mean probability', marker='s', alpha=0.7)
    ax1.fill_between(df['quarter_date'],
                      df['front_prob_mean'] - df['front_prob_std'],
                      df['front_prob_mean'] + df['front_prob_std'],
                      alpha=0.2, color='red', label='Front ±1 std dev')
    ax1.set_ylabel('Breakthrough Probability', fontsize=12, fontweight='bold')
    ax1.set_title('A. Breakthrough Probability Trends', fontsize=13, fontweight='bold')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax1.xaxis.set_major_locator(mdates.YearLocator())

    # Plot 2: Detection counts
    ax2 = axes[1]
    ax2_twin = ax2.twinx()

    ax2.bar(df['quarter_date'], df['prediction'], alpha=0.5, color='blue', label='Field detections (binary)', width=80)
    ax2_twin.plot(df['quarter_date'], df['front_positive_count'], 'r-', linewidth=2, label='Front detections (count)', marker='o')

    ax2.set_ylabel('Field Detections (binary)', fontsize=12, fontweight='bold', color='blue')
    ax2_twin.set_ylabel('Front Detections (count)', fontsize=12, fontweight='bold', color='red')
    ax2.set_title('B. Breakthrough Detection Counts', fontsize=13, fontweight='bold')
    ax2.legend(loc='upper left')
    ax2_twin.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax2.xaxis.set_major_locator(mdates.YearLocator())

    # Plot 3: Activity metrics
    ax3 = axes[2]
    ax3_twin = ax3.twinx()

    ax3.plot(df['quarter_date'], df['num_lineages'], 'g-', linewidth=2, label='Field: active lineages', marker='o')
    ax3_twin.plot(df['quarter_date'], df['total_works'], 'purple', linewidth=2, label='Field: total publications', marker='s', linestyle='--')

    ax3.set_ylabel('Active Lineages', fontsize=12, fontweight='bold', color='green')
    ax3_twin.set_ylabel('Total Publications', fontsize=12, fontweight='bold', color='purple')
    ax3.set_title('C. Field Activity Metrics', fontsize=13, fontweight='bold')
    ax3.set_xlabel('Quarter', fontsize=12, fontweight='bold')
    ax3.legend(loc='upper left')
    ax3_twin.legend(loc='upper right')
    ax3.grid(True, alpha=0.3)
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax3.xaxis.set_major_locator(mdates.YearLocator())

    plt.tight_layout()

    output_path = output_dir / 'field_front_trends.png'
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"   Saved trend plot: {output_path}")
    plt.close(fig)


def create_diagnostic_tables(df: pd.DataFrame, output_dir: Path) -> None:
    """Create diagnostic summary tables."""
    print(f"\n[6/6] Creating diagnostic tables")

    # Table 1: Top field detection quarters
    top_field = df.nlargest(10, 'probability')[
        ['quarter', 'probability', 'prediction', 'num_lineages', 'total_works',
         'front_prob_mean', 'front_positive_count']
    ].copy()

    top_field.to_csv(output_dir / 'top_field_quarters.csv', index=False)
    print(f"   Saved top field quarters: {output_dir / 'top_field_quarters.csv'}")

    # Table 2: Top front detection quarters
    top_front = df.nlargest(10, 'front_positive_count')[
        ['quarter', 'front_positive_count', 'front_prob_mean', 'front_prob_max',
         'probability', 'prediction', 'num_lineages']
    ].copy()

    top_front.to_csv(output_dir / 'top_front_quarters.csv', index=False)
    print(f"   Saved top front quarters: {output_dir / 'top_front_quarters.csv'}")

    # Table 3: Agreement analysis
    # Define "high signal" quarters for field and front
    field_threshold = df['probability'].quantile(0.75)
    front_threshold = df['front_prob_mean'].quantile(0.75)

    df['field_high'] = df['probability'] >= field_threshold
    df['front_high'] = df['front_prob_mean'] >= front_threshold

    agreement = pd.crosstab(df['field_high'], df['front_high'], margins=True)
    agreement.to_csv(output_dir / 'field_front_agreement.csv')
    print(f"   Saved agreement table: {output_dir / 'field_front_agreement.csv'}")

    # Table 4: Summary statistics
    summary_stats = {
        'metric': [
            'Field: Mean probability',
            'Field: Median probability',
            'Field: Max probability',
            'Field: Total detections',
            'Front: Mean probability',
            'Front: Median probability',
            'Front: Max probability',
            'Front: Total detections',
            'Field: Quarters with detection',
            'Front: Quarters with detection',
            'Correlation (prob)',
        ],
        'value': [
            df['probability'].mean(),
            df['probability'].median(),
            df['probability'].max(),
            df['prediction'].sum(),
            df['front_prob_mean'].mean(),
            df['front_prob_mean'].median(),
            df['front_prob_max'].max(),
            df['front_positive_count'].sum(),
            (df['prediction'] > 0).sum(),
            (df['front_positive_count'] > 0).sum(),
            df[['probability', 'front_prob_mean']].corr().iloc[0, 1],
        ]
    }

    df_summary = pd.DataFrame(summary_stats)
    df_summary.to_csv(output_dir / 'summary_statistics.csv', index=False)
    print(f"   Saved summary statistics: {output_dir / 'summary_statistics.csv'}")

    # Print summary to console
    print("\n" + "=" * 70)
    print("FIELD VS FRONT DIAGNOSTICS SUMMARY")
    print("=" * 70)
    print("\nSummary Statistics:")
    for _, row in df_summary.iterrows():
        print(f"  {row['metric']:<40} {row['value']:>10.3f}" if pd.notna(row['value']) else f"  {row['metric']:<40} {'N/A':>10}")

    print("\n" + "=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Field vs Front comparative diagnostics (Task 4.2)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python scripts/field_front_diagnostics.py \\
      --field-predictions data/out/experiments/msd_field_predictions.csv \\
      --front-predictions data/out/03_ensemble_mapping/front_lineage_predictions.csv

  # With custom output directory
  python scripts/field_front_diagnostics.py \\
      --field-predictions data/out/experiments/msd_field_predictions.csv \\
      --front-predictions data/out/03_ensemble_mapping/front_lineage_predictions.csv \\
      --output-dir data/out/experiments/diagnostics_custom
        """
    )

    parser.add_argument(
        '--field-predictions',
        type=Path,
        required=True,
        help='Path to field-level predictions CSV (from field_level_detector.py)'
    )

    parser.add_argument(
        '--front-predictions',
        type=Path,
        default=Path('data/out/03_ensemble_mapping/front_lineage_predictions.csv'),
        help='Path to front-level (lineage) predictions CSV'
    )

    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('data/out/experiments/field_front_diagnostics'),
        help='Output directory for diagnostic plots and tables'
    )

    args = parser.parse_args()

    # Validate inputs
    if not args.field_predictions.exists():
        print(f"[ERROR] Field predictions file not found: {args.field_predictions}")
        return 1

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("FIELD VS FRONT COMPARATIVE DIAGNOSTICS")
    print("=" * 70)
    print(f"Field predictions: {args.field_predictions}")
    print(f"Front predictions: {args.front_predictions}")
    print(f"Output directory: {args.output_dir}")
    print("=" * 70)

    try:
        # Load data
        df_field = load_field_predictions(args.field_predictions)
        df_front = load_front_predictions(args.front_predictions)

        # Aggregate front to quarter level
        df_front_agg = aggregate_front_to_quarter(df_front)

        # Merge data
        df_merged = merge_field_front_data(df_field, df_front_agg)

        # Create visualizations
        create_trend_plots(df_merged, args.output_dir)

        # Create diagnostic tables
        create_diagnostic_tables(df_merged, args.output_dir)

        print("\n[SUCCESS] Comparative diagnostics completed successfully!")
        print(f"\nOutputs saved to: {args.output_dir}")
        print("  - field_front_trends.png (trend comparison plots)")
        print("  - top_field_quarters.csv (quarters with highest field signals)")
        print("  - top_front_quarters.csv (quarters with highest front signals)")
        print("  - field_front_agreement.csv (detection agreement analysis)")
        print("  - summary_statistics.csv (overall statistics)")
        return 0

    except Exception as e:
        print(f"\n[ERROR] Diagnostic generation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
