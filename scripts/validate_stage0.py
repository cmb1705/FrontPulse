"""
Phase 0 Validation: Verify tight mapping properties and re-run detection experiment.

Validates:
1. Each lineage appears at most once (uniqueness constraint)
2. Each milestone has exactly K lineages (K=3)
3. Re-runs 2-signal detection with tight mapping
4. Computes precision, recall, F1 vs random baseline
"""

from pathlib import Path
import pandas as pd
import numpy as np
import json
import shutil

STAGE0_DIR = Path('data/out/experiments/stage0_tight_mapping')
LEGACY_PHASE0_DIR = Path('data/out/experiments/phase0_tight_mapping')
STAGE0_TIGHT_MAPPING = STAGE0_DIR / 'milestone_lineage_mapping_tight.csv'
LEGACY_TIGHT_MAPPING = LEGACY_PHASE0_DIR / 'milestone_lineage_mapping_tight.csv'


def resolve_tight_mapping_path() -> Path:
    if STAGE0_TIGHT_MAPPING.exists():
        return STAGE0_TIGHT_MAPPING
    if LEGACY_TIGHT_MAPPING.exists():
        return LEGACY_TIGHT_MAPPING
    raise FileNotFoundError(
        "Tight mapping not found. Run stage0_semantic_milestone_mapping.py first."
    )
import shutil

STAGE0_DIR = Path('data/out/experiments/stage0_tight_mapping')
LEGACY_PHASE0_DIR = Path('data/out/experiments/phase0_tight_mapping')

def validate_tight_mapping(mapping_path: Path) -> dict:
    """Validate tight mapping properties."""
    print("="*70)
    print("STAGE 0 VALIDATION: TIGHT MAPPING PROPERTIES")
    print("="*70)
    print()

    df = pd.read_csv(mapping_path)

    # Check 1: Each lineage appears at most once
    lineage_counts = df['lineage_id'].value_counts()
    max_lineage_count = lineage_counts.max()
    duplicate_lineages = lineage_counts[lineage_counts > 1]

    print(f"[1/3] Uniqueness constraint:")
    print(f"   Total lineages: {df['lineage_id'].nunique()}")
    print(f"   Max appearances per lineage: {max_lineage_count}")

    if len(duplicate_lineages) > 0:
        print(f"   [FAILED] {len(duplicate_lineages)} lineages appear multiple times")
        for lid, count in duplicate_lineages.items():
            print(f"      - Lineage {lid}: {count} times")
        uniqueness_passed = False
    else:
        print(f"   [PASSED] Each lineage appears exactly once")
        uniqueness_passed = True

    # Check 2: Each milestone has exactly K lineages
    milestone_counts = df['event_id'].value_counts()
    min_lineages = milestone_counts.min()
    max_lineages = milestone_counts.max()

    print(f"\n[2/3] K-per-milestone constraint (K=3):")
    print(f"   Total milestones: {df['event_id'].nunique()}")
    print(f"   Min lineages per milestone: {min_lineages}")
    print(f"   Max lineages per milestone: {max_lineages}")

    if min_lineages == 3 and max_lineages == 3:
        print(f"   [PASSED] All milestones have exactly 3 lineages")
        k_constraint_passed = True
    else:
        print(f"   [FAILED] Not all milestones have exactly 3 lineages")
        k_constraint_passed = False

    # Check 3: Similarity scores
    print(f"\n[3/3] Similarity score distribution:")
    print(f"   Mean: {df['similarity'].mean():.3f}")
    print(f"   Median: {df['similarity'].median():.3f}")
    print(f"   Min: {df['similarity'].min():.3f}")
    print(f"   Max: {df['similarity'].max():.3f}")
    print(f"   Std: {df['similarity'].std():.3f}")

    validation_result = {
        'uniqueness_passed': uniqueness_passed,
        'k_constraint_passed': k_constraint_passed,
        'total_pairs': len(df),
        'unique_milestones': int(df['event_id'].nunique()),
        'unique_lineages': int(df['lineage_id'].nunique()),
        'similarity_mean': float(df['similarity'].mean()),
        'similarity_min': float(df['similarity'].min()),
        'similarity_max': float(df['similarity'].max())
    }

    if uniqueness_passed and k_constraint_passed:
        print(f"\n[VALIDATION PASSED] Tight mapping satisfies all constraints")
    else:
        print(f"\n[VALIDATION FAILED] Tight mapping violates constraints")

    return validation_result


def rerun_detection_experiment(
    tight_mapping_path: Path,
    signals_path: Path
) -> dict:
    """
    Re-run 2-signal detection with tight mapping.

    This time with:
    - Tight 1:1 mapping (3 lineages per milestone)
    - Proper precision/recall/F1 calculation
    - Random baseline comparison
    """
    print("\n")
    print("="*70)
    print("STAGE 0 EXPERIMENT: DETECTION WITH TIGHT MAPPING")
    print("="*70)
    print()

    # Load tight mapping
    mapping_df = pd.read_csv(tight_mapping_path)
    print(f"[1/4] Loaded tight mapping:")
    print(f"   {len(mapping_df)} milestone-lineage pairs")
    print(f"   {mapping_df['event_id'].nunique()} milestones")
    print(f"   {mapping_df['lineage_id'].nunique()} unique lineages")

    # Load signals
    signals_df = pd.read_csv(signals_path)
    print(f"\n[2/4] Loaded lineage signals:")
    print(f"   {len(signals_df)} lineage-quarters")
    print(f"   {signals_df['lineage_id'].nunique()} lineages")
    print(f"   {signals_df['quarter'].nunique()} quarters")

    # Count flagged lineage-quarters
    n_flagged = signals_df['flagged'].sum()
    flagged_rate = n_flagged / len(signals_df)
    print(f"\n[3/4] Signal flagging rate:")
    print(f"   Flagged quarters: {n_flagged}/{len(signals_df)} ({flagged_rate*100:.2f}%)")

    # Test detection on tight mapping
    print(f"\n[4/4] Testing detection...")

    detections = []
    for _, milestone in mapping_df.iterrows():
        lineage_id = milestone['lineage_id']
        event_quarter = milestone['event_quarter']
        detection_start = milestone['detection_window_start']
        detection_end = milestone['detection_window_end']

        # Check if this lineage was flagged in detection window
        detection_window = signals_df[
            (signals_df['lineage_id'] == lineage_id) &
            (signals_df['quarter'] >= detection_start) &
            (signals_df['quarter'] <= detection_end)
        ]

        was_flagged = detection_window['flagged'].any() if len(detection_window) > 0 else False

        detections.append({
            'event_id': milestone['event_id'],
            'event_quarter': event_quarter,
            'lineage_id': lineage_id,
            'rank': milestone['rank'],
            'similarity': milestone['similarity'],
            'was_flagged': was_flagged,
            'detection_window_start': detection_start,
            'detection_window_end': detection_end
        })

    detections_df = pd.DataFrame(detections)

    # Calculate metrics at milestone level (milestone detected if ANY of its 3 lineages flagged)
    milestone_detected = detections_df.groupby('event_id')['was_flagged'].any()

    true_positives = milestone_detected.sum()
    false_negatives = len(milestone_detected) - true_positives
    total_milestones = len(milestone_detected)

    recall = true_positives / total_milestones if total_milestones > 0 else 0

    print(f"\n   === MILESTONE-LEVEL DETECTION ===")
    print(f"   True Positives: {true_positives}")
    print(f"   False Negatives: {false_negatives}")
    print(f"   Recall: {recall*100:.1f}% ({true_positives}/{total_milestones})")

    # Calculate precision (need to check flagged lineages that are NOT in milestone set)
    milestone_lineage_set = set(mapping_df['lineage_id'].unique())

    # Count flagged quarters for milestone lineages vs non-milestone lineages
    milestone_signals = signals_df[signals_df['lineage_id'].isin(milestone_lineage_set)]
    non_milestone_signals = signals_df[~signals_df['lineage_id'].isin(milestone_lineage_set)]

    milestone_flagged = milestone_signals['flagged'].sum()
    non_milestone_flagged = non_milestone_signals['flagged'].sum()

    # Precision at lineage-quarter level
    total_flagged = signals_df['flagged'].sum()
    precision_lq = milestone_flagged / total_flagged if total_flagged > 0 else 0

    print(f"\n   === LINEAGE-QUARTER LEVEL ===")
    print(f"   Milestone lineages flagged: {milestone_flagged}")
    print(f"   Non-milestone lineages flagged: {non_milestone_flagged}")
    print(f"   Total flagged: {total_flagged}")
    print(f"   Precision (lineage-quarter): {precision_lq*100:.1f}%")

    # F1 score
    f1 = 2 * (precision_lq * recall) / (precision_lq + recall) if (precision_lq + recall) > 0 else 0
    print(f"   F1 Score: {f1*100:.1f}%")

    # Random baseline comparison
    print(f"\n   === RANDOM BASELINE ===")

    # Expected recall if we randomly flag at rate p=flagged_rate
    # For each milestone with 3 lineages, each with ~5 quarters in window
    # P(detect) = 1 - (1-p)^(3*5) = 1 - (1-p)^15
    avg_window_quarters = 5  # Typical detection window
    tests_per_milestone = 3 * avg_window_quarters  # 3 lineages × 5 quarters

    random_recall = 1 - (1 - flagged_rate) ** tests_per_milestone

    print(f"   Flagging rate: {flagged_rate*100:.2f}%")
    print(f"   Tests per milestone: {tests_per_milestone} (3 lineages × {avg_window_quarters} quarters)")
    print(f"   Expected random recall: {random_recall*100:.1f}%")
    print(f"   Actual recall: {recall*100:.1f}%")
    print(f"   Improvement over random: {recall/random_recall:.2f}x" if random_recall > 0 else "   N/A")

    # Detected milestones
    print(f"\n   Detected milestones:")
    for event_id, was_detected in milestone_detected.items():
        if was_detected:
            event_quarter = detections_df[detections_df['event_id'] == event_id]['event_quarter'].iloc[0]
            # Get which ranks detected it
            detected_ranks = detections_df[
                (detections_df['event_id'] == event_id) &
                (detections_df['was_flagged'] == True)
            ]['rank'].tolist()
            print(f"      - {event_id} ({event_quarter}) [ranks: {detected_ranks}]")

    return {
        'detections_df': detections_df,
        'milestone_detected': milestone_detected,
        'true_positives': int(true_positives),
        'false_negatives': int(false_negatives),
        'recall': float(recall),
        'precision_lq': float(precision_lq),
        'f1': float(f1),
        'milestone_flagged': int(milestone_flagged),
        'non_milestone_flagged': int(non_milestone_flagged),
        'total_flagged': int(total_flagged),
        'flagged_rate': float(flagged_rate),
        'random_baseline_recall': float(random_recall),
        'improvement_over_random': float(recall / random_recall) if random_recall > 0 else None
    }


def save_results(validation_result: dict, detection_result: dict, output_dir: Path):
    """Save validation and detection results."""
    output_dir.mkdir(parents=True, exist_ok=True)

    detections_path = output_dir / 'tight_mapping_detections.csv'
    detection_result['detections_df'].to_csv(detections_path, index=False)

    summary = {
        'validation': validation_result,
        'detection': {
            'true_positives': detection_result['true_positives'],
            'false_negatives': detection_result['false_negatives'],
            'recall': detection_result['recall'],
            'precision_lq': detection_result['precision_lq'],
            'f1': detection_result['f1'],
            'milestone_flagged': detection_result['milestone_flagged'],
            'non_milestone_flagged': detection_result['non_milestone_flagged'],
            'total_flagged': detection_result['total_flagged'],
            'flagged_rate': detection_result['flagged_rate'],
            'random_baseline_recall': detection_result['random_baseline_recall'],
            'improvement_over_random': detection_result['improvement_over_random']
        }
    }

    summary_file = output_dir / 'stage0_validation_summary.json'
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)

    print("\nSummary:")
    print(f"   - {detections_path.name}")
    print(f"   - {summary_file.name}")
    if LEGACY_PHASE0_DIR:
        LEGACY_PHASE0_DIR.mkdir(parents=True, exist_ok=True)
        shutil.copy2(detections_path, LEGACY_PHASE0_DIR / detections_path.name)
        shutil.copy2(summary_file, LEGACY_PHASE0_DIR / summary_file.name)
        print("   - legacy copies written to data/out/experiments/phase0_tight_mapping")


def main():
    # Paths
    tight_mapping_path = resolve_tight_mapping_path()
    signals_path = Path('data/out/experiments/lineage_signals/lineage_signals.csv')
    output_dir = STAGE0_DIR

    # Step 1: Validate tight mapping properties
    validation_result = validate_tight_mapping(tight_mapping_path)

    # Step 2: Re-run detection experiment with tight mapping
    detection_result = rerun_detection_experiment(tight_mapping_path, signals_path)

    # Step 3: Save results
    save_results(validation_result, detection_result, output_dir)

    print("\n" + "="*70)
    print("STAGE 0 VALIDATION COMPLETE")
    print("="*70)


if __name__ == '__main__':
    main()
