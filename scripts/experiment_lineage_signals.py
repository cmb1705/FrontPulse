"""
Experiment: Lineage-Level Multi-Signal Breakthrough Detection

Tests whether combining growth acceleration + embedding velocity signals
can detect research milestones at the lineage level (bypassing front mapping).

Quick experiment to validate the feasibility of the lineage-centric approach.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run the lineage-level multi-signal detection experiment.")
    parser.add_argument(
        "--timeseries",
        type=Path,
        default=Path('data/out/02_lineage_tracking/lineage_timeseries.csv'),
        help="Path to lineage time series CSV (default: %(default)s).",
    )
    parser.add_argument(
        "--embeddings",
        type=Path,
        default=Path('data/out/02_lineage_tracking/lineage_embeddings.npz'),
        help="Path to lineage embeddings NPZ (default: %(default)s).",
    )
    parser.add_argument("--milestones", type=Path, required=True, help="Path to the milestone catalog CSV.")
    parser.add_argument(
        "--mappings",
        type=Path,
        default=Path('data/out/03_milestone_mapping/lineage_front_mappings.csv'),
        help="Path to lineage/front mapping CSV (default: %(default)s).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path('data/out/experiments/lineage_signals'),
        help="Output directory for experiment artifacts (default: %(default)s).",
    )
    return parser.parse_args()

def compute_growth_acceleration(timeseries_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute growth rate and acceleration for each lineage.

    Returns DataFrame with columns:
        - lineage_id, quarter, new_works, growth_rate, acceleration
    """
    print("[1/6] Computing growth acceleration...")

    results = []
    for lineage_id, group in timeseries_df.groupby('lineage_id'):
        group = group.sort_values('quarter')

        # Compute growth rate (% change from previous quarter)
        new_works = group['new_works'].values
        growth_rate = np.zeros(len(new_works))
        growth_rate[1:] = (new_works[1:] - new_works[:-1]) / (new_works[:-1] + 1)  # +1 to avoid div by 0

        # Compute acceleration (change in growth rate)
        acceleration = np.zeros(len(growth_rate))
        acceleration[1:] = growth_rate[1:] - growth_rate[:-1]

        for i, row in enumerate(group.itertuples()):
            results.append({
                'lineage_id': lineage_id,
                'quarter': row.quarter,
                'new_works': row.new_works,
                'growth_rate': growth_rate[i],
                'acceleration': acceleration[i]
            })

    df = pd.DataFrame(results)
    print(f"   Computed for {df['lineage_id'].nunique()} lineages across {df['quarter'].nunique()} quarters")
    return df


def load_embeddings(embedding_path: Path) -> dict:
    """
    Load Phase 2 lineage embeddings.

    Returns dict: {lineage_id: embedding_vector}
    """
    print("[2/6] Loading Phase 2 embeddings...")

    data = np.load(embedding_path)
    lineage_ids = data['lineage_ids']
    embeddings = data['embeddings']

    embedding_dict = {int(lid): emb for lid, emb in zip(lineage_ids, embeddings)}
    print(f"   Loaded embeddings for {len(embedding_dict)} lineages")
    return embedding_dict


def compute_embedding_velocity(
    lineage_embeddings: dict,
    timeseries_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Compute embedding velocity (cosine distance between consecutive quarters).

    NOTE: Phase 2 embeddings are aggregate per lineage, not per quarter.
    This is a limitation - we'll compute a proxy using growth patterns.

    Returns DataFrame with lineage_id, quarter, embedding_velocity
    """
    print("[3/6] Computing embedding velocity (proxy)...")
    print("   NOTE: Phase 2 embeddings are lineage-level aggregates, not quarterly")
    print("   Using growth pattern as proxy for semantic velocity")

    # Simplified proxy: Use growth volatility as a stand-in for semantic shift
    # This is not ideal but allows us to test the framework
    results = []
    for lineage_id, group in timeseries_df.groupby('lineage_id'):
        group = group.sort_values('quarter')

        # Rolling stddev of growth rate as proxy for "semantic turbulence"
        if 'growth_rate' in group.columns:
            velocity = group['growth_rate'].rolling(window=3, min_periods=1).std().fillna(0)
        else:
            velocity = [0] * len(group)

        for i, (_idx, row) in enumerate(group.iterrows()):
            results.append({
                'lineage_id': lineage_id,
                'quarter': row['quarter'],
                'embedding_velocity_proxy': velocity.iloc[i] if isinstance(velocity, pd.Series) else velocity[i]
            })

    df = pd.DataFrame(results)
    print(f"   Computed velocity proxy for {df['lineage_id'].nunique()} lineages")
    return df


def load_milestones(milestone_path: Path) -> pd.DataFrame:
    """Load milestone definitions."""
    print("[4/6] Loading milestones...")

    df = pd.read_csv(milestone_path)
    # Filter to detectable milestones only
    df = df[df['detectable']].copy()

    print(f"   Loaded {len(df)} detectable milestones")
    print(f"   Date range: {df['event_quarter'].min()} to {df['event_quarter'].max()}")
    return df


def map_milestones_to_lineages(
    milestones_df: pd.DataFrame,
    front_mappings_path: Path
) -> pd.DataFrame:
    """
    Map milestones to lineages via research fronts.

    Returns DataFrame with columns:
        - event_id, event_quarter, mapped_fronts, lineage_id, confidence
    """
    print("[5/6] Mapping milestones to lineages...")

    # Load lineage-front mappings
    mappings_df = pd.read_csv(front_mappings_path)

    milestone_lineages = []
    for _, milestone in milestones_df.iterrows():
        fronts = str(milestone['mapped_fronts']).split('|')

        for front in fronts:
            front = front.strip()
            # Find lineages mapped to this front (check primary_front or alternative_fronts)
            matched_primary = mappings_df[mappings_df['primary_front'] == front]
            matched_alt = mappings_df[mappings_df['alternative_fronts'].str.contains(front, na=False)]
            matched = pd.concat([matched_primary, matched_alt]).drop_duplicates()

            for _, mapping in matched.iterrows():
                milestone_lineages.append({
                    'event_id': milestone['event_id'],
                    'event_quarter': milestone['event_quarter'],
                    'detection_window_start': milestone['detection_window_start'],
                    'detection_window_end': milestone['detection_window_end'],
                    'mapped_front': front,
                    'lineage_id': mapping['lineage_id'],
                    'confidence': mapping.get('confidence', 'unknown')
                })

    df = pd.DataFrame(milestone_lineages)
    print(f"   Mapped {df['event_id'].nunique()} milestones to {df['lineage_id'].nunique()} lineages")
    print(f"   Total milestone-lineage pairs: {len(df)}")
    return df


def test_two_signal_detection(
    growth_df: pd.DataFrame,
    velocity_df: pd.DataFrame,
    milestone_lineages_df: pd.DataFrame,
    acceleration_threshold_percentile: float = 90,
    velocity_threshold_percentile: float = 90
) -> dict:
    """
    Test 2-signal detection: flag lineages with high acceleration + high velocity.

    Check if flagged lineages align with milestone timing.
    """
    print("[6/6] Testing 2-signal detection...")

    # Merge growth and velocity data
    signal_df = growth_df.merge(velocity_df, on=['lineage_id', 'quarter'], how='left')

    # Compute thresholds
    accel_threshold = signal_df['acceleration'].quantile(acceleration_threshold_percentile / 100)
    velocity_threshold = signal_df['embedding_velocity_proxy'].quantile(velocity_threshold_percentile / 100)

    print(f"   Acceleration threshold (p{acceleration_threshold_percentile}): {accel_threshold:.4f}")
    print(f"   Velocity threshold (p{velocity_threshold_percentile}): {velocity_threshold:.4f}")

    # Flag lineages with both signals
    signal_df['high_acceleration'] = signal_df['acceleration'] > accel_threshold
    signal_df['high_velocity'] = signal_df['embedding_velocity_proxy'] > velocity_threshold
    signal_df['flagged'] = signal_df['high_acceleration'] & signal_df['high_velocity']

    n_flagged = signal_df['flagged'].sum()
    print(f"   Flagged {n_flagged} lineage-quarters with both signals")

    # Test detection on milestones
    detections = []
    for _, milestone in milestone_lineages_df.iterrows():
        lineage_id = milestone['lineage_id']
        event_quarter = milestone['event_quarter']
        detection_start = milestone['detection_window_start']
        detection_end = milestone['detection_window_end']

        # Check if this lineage was flagged in detection window
        detection_window = signal_df[
            (signal_df['lineage_id'] == lineage_id) &
            (signal_df['quarter'] >= detection_start) &
            (signal_df['quarter'] <= detection_end)
        ]

        was_flagged = detection_window['flagged'].any() if len(detection_window) > 0 else False

        detections.append({
            'event_id': milestone['event_id'],
            'event_quarter': event_quarter,
            'mapped_front': milestone['mapped_front'],
            'lineage_id': lineage_id,
            'was_flagged': was_flagged,
            'detection_window_start': detection_start,
            'detection_window_end': detection_end
        })

    detections_df = pd.DataFrame(detections)

    # Compute detection metrics
    unique_milestones = detections_df.groupby('event_id')['was_flagged'].any()
    n_detected = unique_milestones.sum()
    n_total = len(unique_milestones)

    print("\n   === DETECTION RESULTS ===")
    print(f"   Milestones detected: {n_detected}/{n_total} ({n_detected/n_total*100:.1f}%)")
    print("   (vs current tripwire: 3/56 = 5.4%)")

    # Breakdown by milestone
    print("\n   Detected milestones:")
    for event_id, was_detected in unique_milestones.items():
        if was_detected:
            event_quarter = detections_df[detections_df['event_id'] == event_id]['event_quarter'].iloc[0]
            print(f"      - {event_id} ({event_quarter})")

    return {
        'signal_df': signal_df,
        'detections_df': detections_df,
        'n_detected': int(n_detected),
        'n_total': int(n_total),
        'recall': float(n_detected / n_total),
        'accel_threshold': float(accel_threshold),
        'velocity_threshold': float(velocity_threshold),
        'n_flagged_quarters': int(n_flagged)
    }


def generate_report(results: dict, output_dir: Path):
    """Generate experiment report."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save detection results
    results['detections_df'].to_csv(output_dir / 'milestone_detections.csv', index=False)
    results['signal_df'].to_csv(output_dir / 'lineage_signals.csv', index=False)

    # Save summary
    summary = {
        'n_detected': results['n_detected'],
        'n_total': results['n_total'],
        'recall': results['recall'],
        'baseline_recall': 3/56,  # Current tripwire performance
        'improvement_factor': results['recall'] / (3/56),
        'accel_threshold': results['accel_threshold'],
        'velocity_threshold': results['velocity_threshold'],
        'n_flagged_quarters': results['n_flagged_quarters']
    }

    with open(output_dir / 'experiment_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    # Generate markdown report
    report = []
    report.append("# Lineage-Level Multi-Signal Detection Experiment")
    report.append("")
    report.append("## Objective")
    report.append("Test whether combining growth acceleration + embedding velocity can detect")
    report.append("research breakthroughs at the lineage level (bypassing front mapping bottleneck).")
    report.append("")
    report.append("## Method")
    report.append("1. Computed growth acceleration for all lineages")
    report.append("2. Computed embedding velocity proxy (growth volatility)")
    report.append("3. Flagged lineages with both high acceleration AND high velocity (p90)")
    report.append("4. Tested detection on known milestones (0-4 quarters before event)")
    report.append("")
    report.append("## Results")
    report.append("")
    report.append(f"**Detection Rate: {results['n_detected']}/{results['n_total']} = {results['recall']*100:.1f}%**")
    report.append("")
    report.append("- Baseline (current tripwire): 3/56 = 5.4%")
    report.append(f"- This approach: {results['n_detected']}/{results['n_total']} = {results['recall']*100:.1f}%")
    report.append(f"- **Improvement factor: {summary['improvement_factor']:.2f}x**")
    report.append("")
    report.append("## Signal Thresholds")
    report.append("")
    report.append(f"- Acceleration (p90): {results['accel_threshold']:.4f}")
    report.append(f"- Velocity proxy (p90): {results['velocity_threshold']:.4f}")
    report.append(f"- Flagged quarters: {results['n_flagged_quarters']}")
    report.append("")
    report.append("## Detected Milestones")
    report.append("")

    detected_milestones = results['detections_df'].groupby('event_id')['was_flagged'].any()
    for event_id, was_detected in detected_milestones.items():
        if was_detected:
            event_quarter = results['detections_df'][results['detections_df']['event_id'] == event_id]['event_quarter'].iloc[0]
            report.append(f"- {event_id} ({event_quarter})")

    report.append("")
    report.append("## Interpretation")
    report.append("")

    if results['recall'] > 0.1:
        report.append(f"**Promising results!** This simple 2-signal approach detects {results['recall']*100:.1f}% of milestones,")
        report.append("compared to the current tripwire's 5.4%. This suggests the lineage-level")
        report.append("multi-signal approach has merit and warrants full implementation.")
    else:
        report.append(f"**Mixed results.** Detection rate of {results['recall']*100:.1f}% is only marginally better")
        report.append("than the current tripwire (5.4%). This could be due to:")
        report.append("- Embedding velocity proxy being too crude (need true temporal embeddings)")
        report.append("- Milestone-lineage mapping being incomplete")
        report.append("- Need for additional signals (disruption, novelty, network metrics)")

    report.append("")
    report.append("## Next Steps")
    report.append("")

    if results['recall'] > 0.1:
        report.append("1. Implement true temporal embeddings (quarterly SciBERT embeddings)")
        report.append("2. Add disruption score (CD index)")
        report.append("3. Add novelty score (NPMI new term rate)")
        report.append("4. Test full multi-signal ensemble")
    else:
        report.append("1. Investigate milestone-lineage mapping quality")
        report.append("2. Compute true temporal embeddings (not aggregate)")
        report.append("3. Consider alternative signals (network-based, citation-based)")

    report_path = output_dir / 'experiment_report.md'
    with open(report_path, 'w') as f:
        f.write('\n'.join(report))

    print(f"\n[Report] Saved to {output_dir}/")
    print("   - experiment_summary.json")
    print("   - experiment_report.md")
    print("   - milestone_detections.csv")
    print("   - lineage_signals.csv")


def main():
    args = parse_args()

    print("="*70)
    print("LINEAGE-LEVEL MULTI-SIGNAL DETECTION EXPERIMENT")
    print("="*70)
    print()

    # Paths
    timeseries_path = args.timeseries
    milestones_path = args.milestones
    mappings_path = args.mappings
    output_dir = args.output_dir

    # Load data
    timeseries_df = pd.read_csv(timeseries_path)

    # Step 1: Compute growth acceleration
    growth_df = compute_growth_acceleration(timeseries_df)

    # Step 2: Load embeddings (NOTE: Not using them yet due to temporal resolution issue)
    # embeddings = load_embeddings(embeddings_path)

    # Step 3: Compute embedding velocity proxy
    velocity_df = compute_embedding_velocity({}, timeseries_df.merge(growth_df, on=['lineage_id', 'quarter']))

    # Step 4: Load milestones
    milestones_df = load_milestones(milestones_path)

    # Step 5: Map milestones to lineages
    milestone_lineages_df = map_milestones_to_lineages(milestones_df, mappings_path)

    # Step 6: Test 2-signal detection
    results = test_two_signal_detection(growth_df, velocity_df, milestone_lineages_df)

    # Generate report
    generate_report(results, output_dir)

    print("\n" + "="*70)
    print("EXPERIMENT COMPLETE")
    print("="*70)


if __name__ == '__main__':
    main()
