"""Debug script to find NaN in Phase 4 output."""
from pathlib import Path

import pandas as pd

# Load the file
pairs_file = Path("data/out/02_lineage_tracking/lineage_npmi_pairs.csv")
df = pd.read_csv(pairs_file)

print(f"Total rows: {len(df)}")
print(f"\nColumns: {df.columns.tolist()}")

# Check for NaN in each column
print("\n=== NaN Check by Column ===")
for col in df.columns:
    nan_count = df[col].isna().sum()
    if nan_count > 0:
        print(f"{col}: {nan_count} NaN values")
        # Show rows with NaN in this column
        nan_rows = df[df[col].isna()]
        print(f"Rows with NaN in {col}:")
        print(nan_rows.head(10))
        print()

# Check for string 'nan' (lowercase)
print("\n=== Checking for string 'nan' ===")
for col in ['term1', 'term2']:
    if col in df.columns:
        string_nan = df[df[col].astype(str).str.lower() == 'nan']
        if len(string_nan) > 0:
            print(f"Found {len(string_nan)} rows with string 'nan' in {col}")
            print(string_nan.head(10))

# Check for empty strings
print("\n=== Checking for empty strings ===")
for col in ['term1', 'term2']:
    if col in df.columns:
        empty = df[df[col].astype(str).str.strip() == '']
        if len(empty) > 0:
            print(f"Found {len(empty)} rows with empty string in {col}")
            print(empty.head(10))
