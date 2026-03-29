#!/usr/bin/env python3
"""Compare validation results across different mu_floor values."""
import pandas as pd

original = pd.read_csv('data/out/06_validation/archived/2025_11_01_intercept_fix/validation_results.csv')
floor_10 = pd.read_csv('data/out/06_validation/archived/2025_11_01_floor_10/validation_results.csv')
floor_02 = pd.read_csv('data/out/06_validation/archived/2025_11_01_floor_02/validation_results.csv')

print("Detection rates:")
print(f"  No floor issues (with intercept fix): {original['detected'].sum()}/38 = {100*original['detected'].mean():.1f}%")
print(f"  mu_floor = 1.0: {floor_10['detected'].sum()}/38 = {100*floor_10['detected'].mean():.1f}%")
print(f"  mu_floor = 0.2: {floor_02['detected'].sum()}/38 = {100*floor_02['detected'].mean():.1f}%")

# Compare original to floor_02
comparison = pd.merge(
    original[['event_id', 'event_quarter', 'detected']],
    floor_02[['event_id', 'detected']],
    on='event_id',
    suffixes=('_orig', '_02')
)

still_lost = comparison[(comparison['detected_orig']) & (not comparison['detected_02'])]
print("\nStill missing with mu_floor=0.2 (lost from original):")
print(still_lost[['event_id', 'event_quarter']].to_string(index=False))

recovered = comparison[(not comparison['detected_orig']) & (comparison['detected_02'])]
if len(recovered) > 0:
    print("\nGained with mu_floor=0.2 (vs original):")
    print(recovered[['event_id', 'event_quarter']].to_string(index=False))

# Compare floor_10 to floor_02
comparison2 = pd.merge(
    floor_10[['event_id', 'event_quarter', 'detected']],
    floor_02[['event_id', 'detected']],
    on='event_id',
    suffixes=('_10', '_02')
)

recovered_from_10 = comparison2[(not comparison2['detected_10']) & (comparison2['detected_02'])]
print("\nRecovered when relaxing floor from 1.0 to 0.2:")
print(recovered_from_10[['event_id', 'event_quarter']].to_string(index=False))
