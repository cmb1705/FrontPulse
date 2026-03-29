#!/usr/bin/env python3
"""Analyze tripwire alert distribution."""
import pandas as pd

alerts = pd.read_csv('data/out/06_validation/archived/2025_11_01_floor_10/tripwire_alerts.csv')
print(f'Total alerts: {len(alerts)}')
print(f'Significant alerts: {alerts["significant"].sum()}')
print(f'Max z-score: {alerts["z_score"].max():.1f}')
print(f'Mean z-score: {alerts["z_score"].mean():.2f}')
print('\nTop 10 z-scores:')
print(alerts.nlargest(10, 'z_score')[['quarter', 'community_id', 'z_score', 'expected', 'observed', 'significant']])

sparse = alerts[alerts['z_score'] > 50]
print(f'\n\nAlerts with z > 50: {len(sparse)}')
if len(sparse) > 0:
    print(sparse[['quarter', 'community_id', 'z_score', 'expected', 'observed']])

# Check specific quarters that lost detection
print("\n\nAlerts for 2016-Q2 (missed events - stability breakthroughs):")
q2_2016 = alerts[alerts['quarter'] == '2016-Q2'].sort_values('z_score', ascending=False)
print(q2_2016[['community_id', 'z_score', 'p_adjusted', 'significant', 'expected', 'observed']].to_string())

print("\n\nAlerts for 2022-Q2 (missed events - inverted + tandem breakthroughs):")
q2_2022 = alerts[alerts['quarter'] == '2022-Q2'].sort_values('z_score', ascending=False)
print(q2_2022[['community_id', 'z_score', 'p_adjusted', 'significant', 'expected', 'observed']].to_string())
