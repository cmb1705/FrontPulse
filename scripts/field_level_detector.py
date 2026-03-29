#!/usr/bin/env python3
"""
Field-Level Breakthrough Detection (Task 4.1)

Aggregates lineage-level features to field level and generates breakthrough predictions
for the entire PSC research field per quarter.

Usage:
    python scripts/field_level_detector.py \\
        --lineage-features data/out/02_lineage_tracking/lineage_multisignal_features.csv \\
        --metrics-dir data/out/metrics \\
        --model data/out/experiments/msd_model.pkl \\
        --output data/out/experiments/msd_field_predictions.csv

Author: Multi-Signal Context Integration (Task 4.1)
Date: 2025-11-06
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

from _path_bootstrap import ensure_repo_imports

repo_root = ensure_repo_imports()

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from trusted_io import load_trusted_pickle  # noqa: E402


def load_lineage_features(path: Path) -> pd.DataFrame:
    """Load lineage-level multisignal features."""
    print(f"[1/8] Loading lineage features from {path}")
    df = pd.read_csv(path)
    print(f"   Loaded {len(df)} lineage-quarter records")
    print(f"   Lineages: {df['lineage_id'].nunique()}")
    print(f"   Quarters: {df['quarter'].nunique()}")
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer additional features from raw signals (mirrors MSD feature engineering).

    This creates the same engineered features that the MSD model was trained on:
    - Temporal derivatives (velocity_acceleration, growth_acceleration)
    - Feature interactions (novelty_momentum, citation_balance)
    - Flags and indicators (is_awakening, disruption_intensity)
    """
    print("\n[2/8] Engineering lineage-level features...")

    df = df.sort_values(['lineage_id', 'quarter'])

    # Compute total_works if not present (cumulative sum of new_works per lineage)
    if 'total_works' not in df.columns and 'new_works' in df.columns:
        df['total_works'] = df.groupby('lineage_id')['new_works'].cumsum()

    # Compute growth_rate if not present (quarter-over-quarter change in total_works)
    if 'growth_rate' not in df.columns and 'total_works' in df.columns:
        df['growth_rate'] = df.groupby('lineage_id')['total_works'].pct_change()
        df['growth_rate'] = df['growth_rate'].fillna(0)

    # Semantic velocity (if not present, approximate from novel terms or set to 0)
    if 'semantic_velocity' not in df.columns:
        if 'novel_terms' in df.columns:
            # Approximate: normalize novel_terms by total_works
            df['semantic_velocity'] = df['novel_terms'] / (df.get('total_works', 1) + 1)
        else:
            df['semantic_velocity'] = 0.0

    # Velocity acceleration (change in semantic velocity)
    df['velocity_acceleration'] = df.groupby('lineage_id')['semantic_velocity'].diff()
    df['velocity_acceleration'] = df['velocity_acceleration'].fillna(0)

    # Growth acceleration
    df['growth_acceleration'] = df.groupby('lineage_id')['growth_rate'].diff()
    df['growth_acceleration'] = df['growth_acceleration'].fillna(0)

    # Novelty momentum (product of novelty rate and new papers)
    if 'novelty_rate' in df.columns and 'n_new_papers' in df.columns:
        df['novelty_momentum'] = df['novelty_rate'] * df['n_new_papers']
    elif 'novelty_rate' in df.columns and 'new_works' in df.columns:
        df['novelty_momentum'] = df['novelty_rate'] * df['new_works']
    else:
        df['novelty_momentum'] = 0.0

    # Awakening flag (dormancy followed by activity)
    if 'awakening_intensity' in df.columns:
        df['is_awakening'] = (df['awakening_intensity'] > 0).astype(int)
    else:
        df['is_awakening'] = 0

    # Citation balance (cross-domain vs within-lineage)
    if 'cross_domain_refs' in df.columns and 'within_lineage_refs' in df.columns:
        df['citation_balance'] = df['cross_domain_refs'] / (df['within_lineage_refs'] + 1)
    else:
        df['citation_balance'] = 0.0

    # Disruption intensity (combined CD index extremes)
    if 'cd_max' in df.columns and 'cd_min' in df.columns:
        df['disruption_intensity'] = df['cd_max'] - df['cd_min']
    else:
        df['disruption_intensity'] = 0.0

    print("   Engineered features added")
    print(f"   Total columns: {len(df.columns)}")

    return df


def load_global_metrics(metrics_dir: Path) -> dict[str, pd.DataFrame]:
    """Load all global context metrics."""
    print(f"\n[3/8] Loading global context metrics from {metrics_dir / 'global'}")

    metrics = {}
    metric_names = [
        'author_influx',
        'citation_velocity',
        'reference_vitality',
        'topic_diversity',
        'cross_cluster_bridging'
    ]

    for metric_name in metric_names:
        parquet_path = metrics_dir / 'global' / f'{metric_name}.parquet'
        if not parquet_path.exists():
            print(f"   [WARNING] Missing {metric_name}, skipping")
            continue

        df = pd.read_parquet(parquet_path)
        metrics[metric_name] = df[['quarter', 'value']].rename(columns={'value': metric_name})
        print(f"   Loaded {metric_name}: {len(df)} quarters")

    return metrics


def aggregate_lineage_to_field(df_lineage: pd.DataFrame, method: str = 'weighted_mean') -> pd.DataFrame:
    """
    Aggregate lineage-level features to field level per quarter.

    Args:
        df_lineage: DataFrame with lineage features (already engineered)
        method: Aggregation method ('weighted_mean', 'simple_mean', 'median')

    Returns:
        DataFrame with one row per quarter, field-level aggregated features
    """
    print(f"\n[4/8] Aggregating lineage features to field level (method: {method})")

    # Features to aggregate (exclude identifiers)
    feature_cols = [col for col in df_lineage.columns
                   if col not in ['lineage_id', 'quarter']]

    # Group by quarter
    grouped = df_lineage.groupby('quarter')

    field_data = []

    for quarter, group in grouped:
        record = {'quarter': quarter}

        # Basic counts
        record['num_lineages'] = len(group)

        if method == 'weighted_mean' and 'new_works' in group.columns:
            # Weight by publication count
            weights = group['new_works'].fillna(1).replace(0, 1)

            # For new_works specifically, use sum (total field publications)
            record['new_works'] = group['new_works'].sum()
            # Also store as total_works for compatibility
            record['total_works'] = record['new_works']

            for col in feature_cols:
                if col in group.columns and col not in ['new_works', 'total_works']:
                    values = group[col].fillna(0)
                    # Weighted mean - use base name (no _mean suffix) for model compatibility
                    record[col] = np.average(values, weights=weights)

        elif method == 'simple_mean':
            # For new_works, use sum
            if 'new_works' in group.columns:
                record['new_works'] = group['new_works'].sum()
                record['total_works'] = record['new_works']

            for col in feature_cols:
                if col in group.columns and col not in ['new_works', 'total_works']:
                    values = group[col].fillna(0)
                    # Simple mean - use base name for model compatibility
                    record[col] = values.mean()

        elif method == 'median':
            # For new_works, use sum
            if 'new_works' in group.columns:
                record['new_works'] = group['new_works'].sum()
                record['total_works'] = record['new_works']

            for col in feature_cols:
                if col in group.columns and col not in ['new_works', 'total_works']:
                    values = group[col].fillna(0)
                    # Median - use base name for model compatibility
                    record[col] = values.median()

        field_data.append(record)

    df_field = pd.DataFrame(field_data)
    print(f"   Created field-level dataset: {len(df_field)} quarters")
    print(f"   Features: {len([c for c in df_field.columns if c not in ['quarter', 'num_lineages', 'total_works']])} aggregated features")

    return df_field


def join_global_metrics(df_field: pd.DataFrame, metrics: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Join global context metrics to field-level data."""
    print("\n[5/8] Joining global context metrics")

    df_merged = df_field.copy()

    for metric_name, df_metric in metrics.items():
        df_merged = df_merged.merge(df_metric, on='quarter', how='left')
        print(f"   Joined {metric_name}")

    # Fill missing metric values with 0 or mean
    metric_cols = list(metrics.keys())
    for col in metric_cols:
        if col in df_merged.columns:
            df_merged[col] = df_merged[col].fillna(df_merged[col].mean())

    print(f"   Final dataset: {len(df_merged)} quarters, {len(df_merged.columns)} total columns")

    return df_merged


def prepare_features_for_prediction(
    df: pd.DataFrame,
    model_features: list[str] | None = None
) -> tuple[pd.DataFrame, list[str]]:
    """
    Prepare features for MSD prediction.

    Args:
        df: Field-level dataset with aggregated features
        model_features: List of features the model expects (if None, auto-detect)

    Returns:
        Tuple of (feature_df, feature_names)
    """
    print("\n[6/8] Preparing features for prediction")

    # If model features not provided, use all numeric columns except identifiers
    if model_features is None:
        exclude_cols = ['quarter', 'num_lineages', 'total_works', 'probability', 'prediction', 'threshold']
        feature_names = [col for col in df.columns
                        if col not in exclude_cols and pd.api.types.is_numeric_dtype(df[col])]
    else:
        feature_names = model_features

    # Create feature matrix
    X = df[feature_names].copy()

    # Handle inf/nan
    X = X.replace([np.inf, -np.inf], 0)
    X = X.fillna(0)

    print(f"   Selected {len(feature_names)} features for prediction")
    print(f"   Sample features: {feature_names[:5]}")

    return X, feature_names


def generate_predictions(
    df: pd.DataFrame,
    model_path: Path,
    threshold: float = 0.5,
    allow_external_pickle: bool = False,
) -> pd.DataFrame:
    """
    Generate breakthrough predictions using trained MSD model.

    Args:
        df: Field-level dataset with features
        model_path: Path to trained model pickle
        threshold: Probability threshold for binary prediction

    Returns:
        DataFrame with predictions added
    """
    print(f"\n[7/8] Generating predictions with model: {model_path}")

    # Load model
    model_data = load_trusted_pickle(
        model_path,
        description="Field detector model pickle",
        allow_external=allow_external_pickle,
    )

    # Extract model and metadata
    if isinstance(model_data, dict):
        model = model_data.get('model')
        model_config = model_data.get('config', {})
        print(f"   Model type: {model_config.get('model_type', 'unknown')}")
    else:
        model = model_data
        model_config = {}

    # Extract model's expected features if available
    model_features = None
    if hasattr(model, 'feature_names_in_'):
        model_features = list(model.feature_names_in_)
    elif hasattr(model, '__getitem__') and hasattr(model[0], 'feature_names_in_'):
        # Pipeline - get from first step
        model_features = list(model[0].feature_names_in_)

    # Prepare features
    X, feature_names = prepare_features_for_prediction(df, model_features=model_features)

    # Generate predictions
    if hasattr(model, 'predict_proba'):
        probabilities = model.predict_proba(X)[:, 1]
    elif hasattr(model, 'predict'):
        probabilities = model.predict(X)
    else:
        raise ValueError("Model does not have predict or predict_proba method")

    # Add predictions to dataframe
    df_pred = df.copy()
    df_pred['probability'] = probabilities
    df_pred['threshold'] = threshold
    df_pred['prediction'] = (probabilities >= threshold).astype(int)

    # Add metadata
    df_pred['model_version'] = model_path.stem
    df_pred['model_type'] = model_config.get('model_type', 'unknown')
    df_pred['generated_at'] = datetime.now().isoformat()

    print(f"   Generated {len(df_pred)} predictions")
    print(f"   Positive predictions: {df_pred['prediction'].sum()} ({df_pred['prediction'].mean():.1%})")
    print(f"   Probability range: [{probabilities.min():.3f}, {probabilities.max():.3f}]")

    return df_pred


def save_predictions(df: pd.DataFrame, output_path: Path) -> None:
    """Save field-level predictions to CSV."""
    print(f"\n[8/8] Saving predictions to {output_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Select output columns
    output_cols = [
        'quarter',
        'num_lineages',
        'total_works',
        'probability',
        'threshold',
        'prediction',
        'model_version',
        'model_type',
        'generated_at'
    ]

    # Add global metrics if available
    metric_names = ['author_influx', 'citation_velocity', 'reference_vitality',
                   'topic_diversity', 'cross_cluster_bridging']
    for metric in metric_names:
        if metric in df.columns:
            output_cols.append(metric)

    df_out = df[output_cols].copy()
    df_out.to_csv(output_path, index=False)

    print(f"   Saved {len(df_out)} records")
    print(f"   Output columns: {len(output_cols)}")

    # Print summary statistics
    print("\n" + "=" * 70)
    print("FIELD-LEVEL BREAKTHROUGH DETECTION SUMMARY")
    print("=" * 70)
    print(f"Total quarters analyzed: {len(df_out)}")
    print(f"Breakthrough quarters detected: {df_out['prediction'].sum()}")
    print(f"Detection rate: {df_out['prediction'].mean():.1%}")
    print(f"Average probability: {df_out['probability'].mean():.3f}")
    print(f"Probability std dev: {df_out['probability'].std():.3f}")
    print("\nQuarters with highest breakthrough probability:")
    top_quarters = df_out.nlargest(5, 'probability')[['quarter', 'probability', 'prediction', 'num_lineages']]
    print(top_quarters.to_string(index=False))
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Field-level breakthrough detection (Task 4.1)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with default model
  python scripts/field_level_detector.py \\
      --lineage-features data/out/02_lineage_tracking/lineage_multisignal_features.csv \\
      --metrics-dir data/out/metrics \\
      --model data/out/experiments/msd_model.pkl

  # Use tuned model with custom threshold
  python scripts/field_level_detector.py \\
      --lineage-features data/out/02_lineage_tracking/lineage_multisignal_features.csv \\
      --metrics-dir data/out/metrics \\
      --model data/out/experiments/msd_tuning/best_model.pkl \\
      --threshold 0.3
        """
    )

    parser.add_argument(
        '--lineage-features',
        type=Path,
        default=Path('data/out/02_lineage_tracking/lineage_multisignal_features.csv'),
        help='Path to lineage multisignal features CSV'
    )

    parser.add_argument(
        '--metrics-dir',
        type=Path,
        default=Path('data/out/metrics'),
        help='Directory containing global metrics'
    )

    parser.add_argument(
        '--model',
        type=Path,
        required=True,
        help='Path to trained MSD model pickle file'
    )

    parser.add_argument(
        '--output',
        type=Path,
        default=Path('data/out/experiments/msd_field_predictions.csv'),
        help='Output path for field-level predictions'
    )

    parser.add_argument(
        '--threshold',
        type=float,
        default=0.5,
        help='Probability threshold for binary prediction (default: 0.5)'
    )

    parser.add_argument(
        '--aggregation-method',
        choices=['weighted_mean', 'simple_mean', 'median'],
        default='weighted_mean',
        help='Method for aggregating lineage features (default: weighted_mean)'
    )
    parser.add_argument(
        '--allow-external-pickle',
        action='store_true',
        help='Allow loading model pickles from outside the repository root.'
    )

    args = parser.parse_args()

    # Validate inputs
    if not args.lineage_features.exists():
        print(f"[ERROR] Lineage features file not found: {args.lineage_features}")
        return 1

    if not args.model.exists():
        print(f"[ERROR] Model file not found: {args.model}")
        return 1

    if not args.metrics_dir.exists():
        print(f"[WARNING] Metrics directory not found: {args.metrics_dir}")
        print("   Continuing without global context metrics")

    print("=" * 70)
    print("FIELD-LEVEL BREAKTHROUGH DETECTION")
    print("=" * 70)
    print(f"Lineage features: {args.lineage_features}")
    print(f"Metrics directory: {args.metrics_dir}")
    print(f"Model: {args.model}")
    print(f"Output: {args.output}")
    print(f"Threshold: {args.threshold}")
    print(f"Aggregation: {args.aggregation_method}")
    print("=" * 70)

    try:
        # Load data
        df_lineage = load_lineage_features(args.lineage_features)

        # Engineer features at lineage level (before aggregation)
        df_lineage = engineer_features(df_lineage)

        # Load global metrics
        metrics = load_global_metrics(args.metrics_dir)

        # Aggregate to field level
        df_field = aggregate_lineage_to_field(df_lineage, method=args.aggregation_method)

        # Join global metrics
        df_field = join_global_metrics(df_field, metrics)

        # Generate predictions
        df_pred = generate_predictions(
            df_field,
            args.model,
            threshold=args.threshold,
            allow_external_pickle=args.allow_external_pickle,
        )

        # Save results
        save_predictions(df_pred, args.output)

        print("\n[SUCCESS] Field-level detection completed successfully!")
        return 0

    except Exception as e:
        print(f"\n[ERROR] Field-level detection failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
