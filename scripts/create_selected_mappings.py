#!/usr/bin/env python3
"""
Create curated lineage-front mappings by filtering to medium+ confidence.

This automates the most common Step 6 curation: keeping medium and high confidence
mappings while excluding low confidence ones that add noise to downstream analysis.

Usage:
    python scripts/create_selected_mappings.py

Output:
    - data/out/03_milestone_mapping/lineage_front_mappings_selected.csv

You can manually edit the output file to add back any vetted low-confidence
mappings that you determine are legitimate.
"""

from pathlib import Path

import pandas as pd


def main():
    SRC = Path("data/out/03_milestone_mapping/lineage_front_mappings.csv")
    DST = Path("data/out/03_milestone_mapping/lineage_front_mappings_selected.csv")

    print("=" * 70)
    print("CREATING CURATED MAPPINGS (MEDIUM+ CONFIDENCE)")
    print("=" * 70)

    # Load full mappings
    df = pd.read_csv(SRC)
    print(f"\nLoaded {len(df)} total mappings from {SRC.name}")

    # Show distribution
    print("\nConfidence distribution:")
    for conf, count in df['confidence'].value_counts().items():
        pct = 100 * count / len(df)
        print(f"  {conf:10s}: {count:3d} ({pct:5.1f}%)")

    # Filter to medium+ confidence
    selected = df[df["confidence"].isin(["high", "medium"])].copy()

    print(f"\n[FILTER] Keeping medium+ confidence: {len(selected)} lineages")
    print(f"  Dropped {len(df) - len(selected)} low confidence mappings")

    # Save curated subset
    selected.to_csv(DST, index=False)
    print(f"\n[OUTPUT] Saved curated mappings to {DST.name}")

    print(f"\n{'=' * 70}")
    print("CURATION COMPLETE")
    print("=" * 70)
    print("\nNext steps:")
    print(f"1. (Optional) Review {DST.name} and manually add back any")
    print("   low-confidence mappings you've vetted as legitimate")
    print("2. Run: python scripts/aggregate_lineages_to_fronts.py")
    print()

if __name__ == "__main__":
    main()
