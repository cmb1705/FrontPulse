#!/usr/bin/env python3
"""
Quick evaluation script for front-level milestone validation.
Bypasses the full evaluate_tripwire.py to do a simple comparison.

Usage:
    python scripts/eval_front_milestones.py [--alerts ALERTS_CSV] [--milestones MILESTONES_CSV]
"""

import argparse

import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="Validate tripwire alerts against milestone catalog")
    parser.add_argument(
        '--alerts',
        type=str,
        default='data/out/05_tripwire_detection/alerts_tripwire.csv',
        help='Path to tripwire alerts CSV (default: alerts_tripwire.csv)'
    )
    parser.add_argument(
        '--milestones',
        type=str,
        required=True,
        help='Path to milestone catalog CSV'
    )
    args = parser.parse_args()

    print("="*70)
    print("FRONT-LEVEL MILESTONE VALIDATION")
    print("="*70)

    # Load alerts
    alerts = pd.read_csv(args.alerts)
    alerts_filtered = alerts[alerts['alert']].copy()

    print(f"\n[1/4] Loaded {len(alerts_filtered)} alerts from {len(alerts)} total quarters")
    print(f"  Quarters with alerts: {alerts_filtered['quarter'].nunique()}")
    print(f"  Fronts with alerts: {alerts_filtered['front'].nunique()}")

    # Load milestones (filtered to 2010-2023)
    milestones = pd.read_csv(args.milestones)
    detectable = milestones[milestones['detectable']].copy()

    print(f"\n[2/4] Loaded {len(detectable)} detectable milestones (2010-2023)")

    # Extract fronts from mapped_fronts column (can be pipe-separated)
    milestone_events = []
    for _, row in detectable.iterrows():
        fronts = str(row['mapped_fronts']).split('|')
        for front in fronts:
            milestone_events.append({
                'event_id': row['event_id'],
                'event_quarter': row['event_quarter'],
                'front': front.strip(),
                'detection_window_start': row['detection_window_start'],
                'detection_window_end': row['detection_window_end'],
                'magnitude': row['magnitude'],
                'event_type': row['event_type']
            })

    milestone_df = pd.DataFrame(milestone_events)
    print(f"  Expanded to {len(milestone_df)} front-specific milestone events")

    # Match alerts to milestones
    print("\n[3/4] Matching alerts to milestones...")

    matches = []
    for _, alert in alerts_filtered.iterrows():
        quarter = alert['quarter']
        front = alert['front']

        # Find milestones for this front where quarter is in detection window
        front_milestones = milestone_df[milestone_df['front'] == front]

        for _, ms in front_milestones.iterrows():
            if ms['detection_window_start'] <= quarter <= ms['detection_window_end']:
                matches.append({
                    'alert_quarter': quarter,
                    'front': front,
                    'event_id': ms['event_id'],
                    'event_quarter': ms['event_quarter'],
                    'magnitude': ms['magnitude'],
                    'lead_time_quarters': calculate_lead_quarters(quarter, ms['event_quarter']),
                    'p_value': alert.get('p_value', np.nan),
                    'rate_ratio': alert.get('rr_obs_over_mu', np.nan)
                })

    matches_df = pd.DataFrame(matches)

    # Calculate metrics
    print("\n[4/4] RESULTS:")
    print("="*70)

    if len(matches) > 0:
        print(f"\nTrue Positives: {len(matches)} alerts matched to milestones")
        print(f"  Unique milestone events detected: {matches_df['event_id'].nunique()}")
        print(f"  Detection rate: {matches_df['event_id'].nunique()}/{len(detectable)} = {matches_df['event_id'].nunique()/len(detectable)*100:.1f}%")

        print("\nBy Event Type:")
        for event_type in milestone_df['event_type'].unique():
            type_milestones = milestone_df[milestone_df['event_type'] == event_type]['event_id'].nunique()
            type_detected = matches_df[matches_df['event_id'].isin(
                milestone_df[milestone_df['event_type'] == event_type]['event_id']
            )]['event_id'].nunique()
            print(f"  {event_type}: {type_detected}/{type_milestones} detected")

        print("\nBy Magnitude:")
        for mag in sorted(milestone_df['magnitude'].unique(), reverse=True):
            mag_milestones = milestone_df[milestone_df['magnitude'] == mag]['event_id'].nunique()
            mag_detected = matches_df[matches_df['magnitude'] == mag]['event_id'].nunique()
            print(f"  Magnitude {int(mag)}: {mag_detected}/{mag_milestones} detected")

        print("\nLead Time Statistics:")
        lead_times = matches_df['lead_time_quarters'].dropna()
        if len(lead_times) > 0:
            print(f"  Mean: {lead_times.mean():.2f} quarters")
            print(f"  Median: {lead_times.median():.2f} quarters")
            print(f"  Range: [{lead_times.min():.0f}, {lead_times.max():.0f}]")
    else:
        print("\nNo matches found!")

    # False positives (alerts not matched to any milestone)
    alerted_fronts = set(zip(alerts_filtered['quarter'], alerts_filtered['front']))
    matched_fronts = set(zip(matches_df['alert_quarter'], matches_df['front'])) if len(matches) > 0 else set()
    false_positives = alerted_fronts - matched_fronts

    print(f"\nFalse Positives: {len(false_positives)} alerts not matched to milestones")
    if len(false_positives) > 0 and len(false_positives) < 20:
        print("  (May represent undocumented milestones or noise)")
        for quarter, front in sorted(false_positives)[:10]:
            alert_row = alerts_filtered[(alerts_filtered['quarter'] == quarter) &
                                       (alerts_filtered['front'] == front)].iloc[0]
            print(f"    {quarter} {front}: p={alert_row.get('p_value', 'N/A'):.4f}, RR={alert_row.get('rr_obs_over_mu', 'N/A'):.2f}")

    # False negatives (milestones not matched to any alert)
    detected_events = set(matches_df['event_id']) if len(matches) > 0 else set()
    missed_events = set(detectable['event_id']) - detected_events

    print(f"\nFalse Negatives: {len(missed_events)} milestones not detected")
    if len(missed_events) > 0 and len(missed_events) < 20:
        for event_id in sorted(missed_events):
            event = detectable[detectable['event_id'] == event_id].iloc[0]
            print(f"    {event_id} ({event['event_quarter']}): {event['description'][:60]}...")

    # Save results
    if len(matches) > 0:
        matches_df.to_csv('data/out/06_validation/archived/2025_11_01_intercept_fix/validation_matches.csv', index=False)
        print("\nSaved matches to: data/out/06_validation/archived/2025_11_01_intercept_fix/validation_matches.csv")

    print("="*70)
    print("VALIDATION COMPLETE")
    print("="*70)


def calculate_lead_quarters(alert_quarter: str, event_quarter: str) -> int:
    """Calculate lead time in quarters (negative = lag, positive = lead)"""
    def quarter_to_num(q):
        year, quarter = q.split('Q')
        return int(year) * 4 + int(quarter)

    return quarter_to_num(event_quarter) - quarter_to_num(alert_quarter)


if __name__ == '__main__':
    import os
    os.makedirs('data/out/eval_updated', exist_ok=True)
    main()
