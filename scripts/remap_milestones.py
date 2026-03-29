#!/usr/bin/env python3
"""
Apply scientifically-grounded remapping of milestones from application categories
to detected research communities.

Based on milestone_remapping_proposal.md analysis:
- EXCLUDE field-wide events (no community-specific citation spike expected)
- Remap application categories (tandems, core_psc) → technical communities
- Use multi-front assignments where milestones span multiple communities

Usage:
    python scripts/remap_milestones.py --input path/to/validated_milestones.csv
"""

import argparse
from pathlib import Path

import pandas as pd

# Remapping from milestone_remapping_proposal.md
# None = EXCLUDE from validation (field-wide events)
# "front1|front2" = multi-front assignment
REMAPPING = {
    # ============================================================
    # EXCLUDE: Field-wide events (not detectable by community alerts)
    # ============================================================
    'psc_2009_kojima': None,          # Field inception, no communities exist yet
    'psc_2018_dyesol': None,          # Company bankruptcy (external shock)
    'psc_2020_covid': None,           # Pandemic shutdown (external shock)
    'psc_2022_lead_policy': None,     # Regulatory event (external shock)

    # ============================================================
    # CORE_PSC → Existing Communities
    # ============================================================
    # Stability shock event
    'psc_2015_stability': 'stability_engineering',

    # HTL breakthrough
    'psc_2012_kim': 'interface_passivation',

    # Efficiency records (architectural optimization)
    'psc_2013_epfl_141': 'inverted_architecture',
    'psc_2013_krict_179': 'inverted_architecture',
    'psc_2014_krict_201': 'inverted_architecture',
    'psc_2016_krict_221': 'inverted_architecture',
    'psc_2016_stanford_236': 'inverted_architecture',
    'psc_2025_single_junction_27': 'inverted_architecture',

    # ============================================================
    # PEROVSKITE_SILICON_TANDEMS → Split by Technical Content
    # ============================================================
    # Interface/contact engineering
    'psc_2015_tandem_experimental': 'interface_passivation',
    'psc_2020_tandem_29pct': 'interface_passivation',  # SAMs for contacts
    'psc_2022_epfl_313': 'interface_passivation',
    'psc_2022_hzb_325': 'interface_passivation',

    # Commercial/industrial manufacturing
    'psc_2018_oxford_273': 'large_area_modules',  # Oxford PV commercial
    'psc_2020_oxford_295': 'large_area_modules',  # Oxford PV commercial
    'psc_2023_longi_339': 'large_area_modules',   # LONGi industrial
    'psc_2024_longi_348': 'large_area_modules',   # LONGi industrial
    'psc_2025_longi_current': 'large_area_modules',  # LONGi current record

    # Theoretical/architectural work
    'psc_2014_tandem_theory': 'inverted_architecture',  # Architectural design
    'psc_2015_tandem_18pct': 'inverted_architecture',   # Heterojunction architecture

    # Multi-front: Stability + Commercial scale-up
    'psc_2024_longi_t80_stability': 'stability_engineering|large_area_modules',

    # Multi-front: Module scale-up
    'psc_2024_oxford_tandem_module': 'large_area_modules|interface_passivation',

    # ============================================================
    # ALL_PEROVSKITE_TANDEMS → Split by Technical Approach
    # ============================================================
    # Novel perovskite compositions
    'psc_2016_allpero_tandem_emergence': 'double_perovskites',  # Tin-lead composition

    # 2D perovskite structures
    'psc_2024_allpero_tandem_297': '2d_perovskites',  # 2D intermediate passivation

    # Interface/passivation engineering
    'psc_2022_allpero_tandem_26pct': 'interface_passivation',
    'psc_2022_nrel_gas_quenching': 'interface_passivation',  # Processing for interface quality
    'psc_2025_allpero_tandem_3138': 'interface_passivation',  # SolaEon commercial
    'psc_2025_allpero_tandem_306': 'interface_passivation',   # Dipolar passivation at interfaces

    # Flexible/all-perovskite hybrid
    'psc_2020_nrel_apex_flex': 'flexible_devices',  # Flexible substrate focus, drop all_perovskite_tandems

    # ============================================================
    # LEAD_FREE_ALTERNATIVES → Double Perovskites (Tin-based)
    # ============================================================
    'psc_2014_leadfree_emergence': 'double_perovskites',  # Tin perovskites
    'psc_2025_tin_1665': 'double_perovskites',  # Tin-based record

    # ============================================================
    # SIMPLIFIED_ARCHITECTURES → Inverted Architecture
    # ============================================================
    'psc_2023_etl_free_22pct': 'inverted_architecture',  # ETL-free architectural innovation
    'psc_2025_htl_free_26pct': 'inverted_architecture',  # HTL-free architectural innovation
}


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Remap milestones from application categories to detected communities.")
    parser.add_argument("--input", type=Path, required=True, help="Path to the validated milestones CSV.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output CSV path. Defaults to a sibling file named '<input stem>_remapped.csv'.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Load original validated milestones
    input_path = args.input
    output_path = args.output or input_path.with_name(f"{input_path.stem}_remapped.csv")

    print(f"[Remapping] Loading milestones from {input_path}")
    df = pd.read_csv(input_path)

    original_count = len(df)
    print(f"[Remapping] Original milestone count: {original_count}")

    # Apply remapping
    remapped = []
    excluded = []
    unchanged = []

    for _, row in df.iterrows():
        event_id = row['event_id']

        if event_id in REMAPPING:
            new_front = REMAPPING[event_id]

            if new_front is None:
                # EXCLUDE this milestone
                excluded.append(event_id)
                continue  # Don't include in output
            else:
                # Remap to new front(s)
                row = row.copy()
                old_front = row['mapped_fronts']
                row['mapped_fronts'] = new_front
                remapped.append((event_id, old_front, new_front))
        else:
            # Keep original mapping
            unchanged.append(event_id)

        # Include in output (either remapped or unchanged)
        df.loc[df['event_id'] == event_id, 'mapped_fronts'] = row['mapped_fronts']

    # Remove excluded milestones
    df_output = df[~df['event_id'].isin(excluded)].copy()

    # Normalize quarter format to match tripwire (YYYY-QX)
    if 'event_quarter' in df_output.columns:
        df_output['event_quarter'] = df_output['event_quarter'].astype(str).str.replace(
            r'(?<=\d)Q', '-Q', regex=True
        )

    # Save remapped milestones
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_output.to_csv(output_path, index=False)

    # Summary report
    print("\n[Remapping] Summary:")
    print(f"  Excluded (field-wide events): {len(excluded)}")
    for event_id in excluded:
        print(f"    - {event_id}")

    print(f"\n  Remapped: {len(remapped)}")
    for event_id, old_front, new_front in remapped:
        multi = "|" in new_front
        marker = "[MULTI]" if multi else ""
        print(f"    - {event_id}: {old_front} -> {new_front} {marker}")

    print(f"\n  Unchanged: {len(unchanged)}")

    final_count = len(df_output)
    print(f"\n[Remapping] Final milestone count: {final_count} (excluded {original_count - final_count})")
    print(f"[Remapping] Saved to {output_path}")

    # Count multi-front assignments
    multi_front_count = df_output['mapped_fronts'].str.contains(r'\|', na=False).sum()
    print(f"[Remapping] Multi-front assignments: {multi_front_count}")

    # Show front distribution
    print("\n[Remapping] Front distribution:")
    front_counts = {}
    for fronts_str in df_output['mapped_fronts'].dropna():
        for front in str(fronts_str).split('|'):
            front = front.strip()
            front_counts[front] = front_counts.get(front, 0) + 1

    for front in sorted(front_counts.keys()):
        print(f"  {front}: {front_counts[front]} milestones")


if __name__ == "__main__":
    main()
