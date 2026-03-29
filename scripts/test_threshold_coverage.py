import numpy as np
import pandas as pd

df = pd.read_csv('data/out/03_milestone_mapping/lineage_front_similarity.csv')

print('Testing thresholds for full front coverage:\n')

for percentile in [35, 40, 42, 45, 48, 50]:
    threshold = np.percentile(df.iloc[:, 1:].values.flatten(), percentile)

    zeros = 0
    total_matches = 0
    for col in df.columns[1:]:
        matches = (df[col] > threshold).sum()
        total_matches += matches
        if matches == 0:
            zeros += 1

    avg_matches = total_matches / 16
    print(f'{percentile}th percentile (threshold={threshold:.3f}):')
    print(f'  Avg: {avg_matches:.1f} matches/front, {zeros} fronts with 0 matches\n')
