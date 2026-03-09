#!/usr/bin/env python3
"""Compare validation results before and after mu_floor."""
import pandas as pd

old = pd.read_csv('data/out/06_validation/archived/2025_11_01_intercept_fix/validation_results.csv')
new = pd.read_csv('data/out/06_validation/archived/2025_11_01_floor_10/validation_results.csv')

print(f"Old detection rate: {old['detected'].sum()}/{len(old)} = {100*old['detected'].mean():.1f}%")
print(f"New detection rate: {new['detected'].sum()}/{len(new)} = {100*new['detected'].mean():.1f}%")

changes = pd.merge(
    old[['event_id', 'detected', 'event_quarter']],
    new[['event_id', 'detected']],
    on='event_id',
    suffixes=('_old', '_new')
)
changes = changes[changes['detected_old'] != changes['detected_new']]

print('\nEvents with changed detection status:')
print(changes.to_string())
print(f'\nTotal changed: {len(changes)}')

# Show which went from True to False
lost = changes[changes['detected_old'] & ~changes['detected_new']]
gained = changes[~changes['detected_old'] & changes['detected_new']]

print(f'\nLost detections (was True, now False): {len(lost)}')
if len(lost) > 0:
    print(lost[['event_id', 'event_quarter']].to_string())

print(f'\nGained detections (was False, now True): {len(gained)}')
if len(gained) > 0:
    print(gained[['event_id', 'event_quarter']].to_string())
