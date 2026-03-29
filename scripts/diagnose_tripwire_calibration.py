#!/usr/bin/env python3
"""
Diagnose why tripwire z-scores are so low.

Check if the Negative Binomial expected activity model is overestimating baseline rates.
"""


import pandas as pd


def main():
    # Load alerts with z-scores
    alerts = pd.read_csv("data/out/06_validation/tripwire_alerts.csv")

    # Load actual counts
    counts = pd.read_csv("data/out/04_front_aggregation/front_timeseries_delta_long.csv")

    # Normalize quarter format in counts
    counts['quarter'] = counts['quarter'].astype(str).str.replace(
        r'(?<=\d)Q', '-Q', regex=True
    )

    print("=" * 70)
    print("TRIPWIRE MODEL CALIBRATION DIAGNOSTIC")
    print("=" * 70)

    # Overall z-score distribution
    print("\n[Z-SCORE DISTRIBUTION]")
    print(f"  Mean:   {alerts['z_score'].mean():.3f}")
    print(f"  Median: {alerts['z_score'].median():.3f}")
    print(f"  Std:    {alerts['z_score'].std():.3f}")
    print(f"  Min:    {alerts['z_score'].min():.3f}")
    print(f"  Max:    {alerts['z_score'].max():.3f}")

    # Alerts by significance level
    print("\n[SIGNIFICANCE BREAKDOWN]")
    print(f"  z > 3:    {(alerts['z_score'] > 3).sum():3d} ({100*(alerts['z_score'] > 3).sum()/len(alerts):5.1f}%)")
    print(f"  z > 2:    {(alerts['z_score'] > 2).sum():3d} ({100*(alerts['z_score'] > 2).sum()/len(alerts):5.1f}%)")
    print(f"  z > 1:    {(alerts['z_score'] > 1).sum():3d} ({100*(alerts['z_score'] > 1).sum()/len(alerts):5.1f}%)")
    print(f"  z < 0:    {(alerts['z_score'] < 0).sum():3d} ({100*(alerts['z_score'] < 0).sum()/len(alerts):5.1f}%)")
    print(f"  FDR sig:  {alerts['significant'].sum():3d} ({100*alerts['significant'].sum()/len(alerts):5.1f}%)")

    # Check a sample front: inverted_architecture (should have strongest signals)
    sample_front = "inverted_architecture"
    front_alerts = alerts[alerts['community_id'] == sample_front].sort_values('quarter')
    counts[counts['front'] == sample_front].sort_values('quarter')

    print(f"\n[SAMPLE FRONT: {sample_front}]")
    print(f"  Total alerts: {len(front_alerts)}")
    print(f"  Significant:  {front_alerts['significant'].sum()}")
    print(f"  Median z:     {front_alerts['z_score'].median():.3f}")

    # Merge to compare observed vs expected
    # Note: alerts don't include observed/expected directly, but we can infer from z-score formula
    # z = (observed - expected) / sqrt(variance)
    # For NB: variance = mu + mu^2/r

    print("\n[TOP 10 ALERTS BY Z-SCORE]")
    top_alerts = alerts.nlargest(10, 'z_score')[
        ['quarter', 'community_id', 'z_score', 'composite_score', 'significant']
    ]
    print(top_alerts.to_string(index=False))

    print("\n[BOTTOM 10 ALERTS BY Z-SCORE]")
    bottom_alerts = alerts.nsmallest(10, 'z_score')[
        ['quarter', 'community_id', 'z_score', 'composite_score', 'significant']
    ]
    print(bottom_alerts.to_string(index=False))

    # Check if there's a time trend (are later periods worse?)
    alerts['year'] = alerts['quarter'].str[:4].astype(int)
    print("\n[Z-SCORE BY TIME PERIOD]")
    time_summary = alerts.groupby('year')['z_score'].agg(['mean', 'median', 'count'])
    print(time_summary.to_string())

    print("\n[RECOMMENDATION]")
    mean_z = alerts['z_score'].mean()
    sig_rate = 100 * alerts['significant'].sum() / len(alerts)

    if mean_z < 0:
        print(f"  WARNING: Negative mean z-score ({mean_z:.3f}) indicates model")
        print("           is systematically overestimating expected activity.")
        print("  LIKELY CAUSE: mu_floor parameter too high, or NB model poorly fit")
    elif mean_z < 0.5:
        print(f"  WARNING: Low mean z-score ({mean_z:.3f}) indicates weak signal detection")
        print("  LIKELY CAUSE: Model variance estimates too high, or data is very noisy")

    if sig_rate < 5:
        print(f"  WARNING: FDR control rejecting {100-sig_rate:.1f}% of alerts")
        print("  LIKELY CAUSE: FDR q=0.1 too strict for this application")
        print("  SUGGESTION: Try q=0.2 or use raw z-score threshold (z > 1.5)")

if __name__ == "__main__":
    main()
