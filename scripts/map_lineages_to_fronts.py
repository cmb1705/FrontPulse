"""
Map community lineages to research fronts using automated methods.

Phase 1: Anchor-based seeding from milestone DOIs
Phase 2: SciBERT embedding expansion
Phase 3: c-TF-IDF distinctive term extraction
Phase 4: NPMI co-term discovery
Phase 5: Weighted scoring and final assignment

Usage:
    python scripts/map_lineages_to_fronts.py \
        --milestones path/to/milestones.csv \
        --lineage-registry data/out/02_lineage_tracking/lineage_registry.json \
        --graphs-dir data/current_graphs \
        --output data/out/03_milestone_mapping/lineage_to_front_mapping.json \
        --phase anchor  # or 'all' for full pipeline
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path
from collections import defaultdict
from typing import Dict, Set, List, Tuple
import sys

import pandas as pd
import numpy as np

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from src.trusted_io import load_trusted_pickle


def load_milestone_anchors(milestone_csv: Path) -> Dict[str, List[str]]:
    """
    Load milestone DOIs and extract anchor papers for each research front.

    Returns:
        Dict mapping front_label -> list of DOI strings
    """
    print(f"[ANCHOR] Loading milestones from {milestone_csv}")
    df = pd.read_csv(milestone_csv)

    # Filter to detectable events only
    df = df[df['detectable'] == True].copy()
    print(f"[ANCHOR] Found {len(df)} detectable milestones")

    # Some milestones may reference multiple fronts (pipe-separated)
    # Example: mapped_fronts = "stability_engineering|interface_passivation"
    anchor_map = defaultdict(list)

    for _, row in df.iterrows():
        event_id = row['event_id']
        mapped_fronts = row['mapped_fronts']

        if pd.isna(mapped_fronts):
            print(f"[ANCHOR] Warning: {event_id} has no mapped_fronts, skipping")
            continue

        # Split pipe-separated fronts
        fronts = [f.strip() for f in str(mapped_fronts).split('|')]

        # Use event_id as proxy for DOI (milestones reference landmark papers)
        # In full implementation, we would extract actual DOIs from descriptions
        # For now, use event_id to track which events correspond to which fronts
        for front in fronts:
            anchor_map[front].append(event_id)

    print(f"[ANCHOR] Extracted anchors for {len(anchor_map)} research fronts")
    for front, events in sorted(anchor_map.items()):
        print(f"  - {front}: {len(events)} milestone events")

    return dict(anchor_map)


def find_lineage_for_paper(
    paper_doi: str,
    quarter: str,
    lineage_registry: Dict[str, Dict[int, int]],
    graphs_dir: Path,
    allow_external_pickle: bool = False,
) -> int | None:
    """
    Find which lineage contains a specific paper in a given quarter.

    Args:
        paper_doi: DOI to search for (OpenAlex work_id format)
        quarter: Quarter string (e.g., "2016Q2")
        lineage_registry: Mapping of quarter -> {community_id: lineage_id}
        graphs_dir: Directory containing graph PKL files

    Returns:
        lineage_id if found, None otherwise
    """
    # Load the community partition for this quarter
    graph_path = graphs_dir / f"citation_graph_cumulative_{quarter}.pkl"

    if not graph_path.exists():
        print(f"[ANCHOR] Warning: Graph not found for {quarter}")
        return None

    try:
        G = load_trusted_pickle(
            graph_path,
            description="Milestone mapping graph pickle",
            allow_external=allow_external_pickle,
        )

        # Check if paper exists in graph
        if paper_doi not in G.nodes:
            return None

        # Get community assignment
        if 'labels' not in G.graph:
            print(f"[ANCHOR] Warning: No labels found in graph for {quarter}")
            return None

        labels = G.graph['labels']
        community_id = labels.get(paper_doi)

        if community_id is None:
            return None

        # Map community_id to lineage_id
        quarter_registry = lineage_registry.get(quarter, {})
        lineage_id = quarter_registry.get(int(community_id))

        return lineage_id

    except Exception as e:
        print(f"[ANCHOR] Error loading graph for {quarter}: {e}")
        return None


def build_anchor_map_phase1(
    milestone_csv: Path,
    lineage_registry_json: Path,
    graphs_dir: Path,
    verbose: bool = True
) -> Dict[str, Set[int]]:
    """
    Phase 1: Build anchor map from milestone DOIs to lineage IDs.

    Returns:
        Dict mapping front_label -> set of lineage_ids
    """
    print("\n" + "="*70)
    print("PHASE 1: ANCHOR-BASED SEEDING")
    print("="*70)

    # Load milestone anchors
    milestone_anchors = load_milestone_anchors(milestone_csv)

    # Load lineage registry
    print(f"\n[ANCHOR] Loading lineage registry from {lineage_registry_json}")
    with open(lineage_registry_json, 'r') as f:
        lineage_registry = json.load(f)
    print(f"[ANCHOR] Registry covers {len(lineage_registry)} quarters")

    # Build front -> lineage mapping
    front_to_lineages: Dict[str, Set[int]] = defaultdict(set)

    # Load milestone CSV again to get quarters
    df_milestones = pd.read_csv(milestone_csv)
    df_milestones = df_milestones[df_milestones['detectable'] == True]

    print(f"\n[ANCHOR] Searching for milestone papers in citation graphs...")

    for _, row in df_milestones.iterrows():
        event_id = row['event_id']
        event_quarter = row['event_quarter']
        mapped_fronts = row['mapped_fronts']

        if pd.isna(mapped_fronts):
            continue

        fronts = [f.strip() for f in str(mapped_fronts).split('|')]

        # For demonstration, we'll search in the event quarter and the following 2 quarters
        # In a real implementation, we would have actual DOIs to search for
        # For now, we'll create synthetic mappings based on event characteristics

        # This is a placeholder - in full implementation, you would:
        # 1. Extract landmark DOI from milestone description
        # 2. Search for that DOI in the appropriate quarter's graph
        # 3. Find which lineage contains that paper

        if verbose:
            print(f"  [{event_id}] {event_quarter}: {', '.join(fronts)}")

    # Since we don't have actual DOIs in the milestone CSV currently,
    # we'll return the structure with a note that this needs DOI enrichment
    print("\n[ANCHOR] Phase 1 complete")
    print("[ANCHOR] NOTE: Full implementation requires DOI extraction from milestone descriptions")
    print("[ANCHOR] Current version returns front definitions without lineage mappings")
    print(f"[ANCHOR] Identified {len(milestone_anchors)} research fronts")

    # Return empty mapping for now - to be filled when DOIs are available
    return dict(front_to_lineages)


def save_anchor_results(
    anchor_map: Dict[str, Set[int]],
    output_path: Path,
    milestone_csv: Path
):
    """Save Phase 1 anchor mapping results."""

    # Convert sets to lists for JSON serialization
    serializable_map = {
        front: {
            "lineage_ids": sorted(list(lineages)),
            "n_lineages": len(lineages)
        }
        for front, lineages in anchor_map.items()
    }

    # Add metadata
    df = pd.read_csv(milestone_csv)
    df_detectable = df[df['detectable'] == True]

    output_data = {
        "phase": "anchor_seeding",
        "fronts": serializable_map,
        "metadata": {
            "n_fronts": len(anchor_map),
            "n_milestones": len(df_detectable),
            "total_lineage_assignments": sum(len(v) for v in anchor_map.values())
        }
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\n[ANCHOR] Results saved to {output_path}")
    print(f"[ANCHOR] Mapped {len(anchor_map)} fronts to lineages")


def main():
    parser = argparse.ArgumentParser(
        description="Map community lineages to research fronts"
    )
    parser.add_argument(
        "--milestones",
        type=Path,
        required=True,
        help="Path to validated milestones CSV"
    )
    parser.add_argument(
        "--lineage-registry",
        type=Path,
        default=Path("data/out/02_lineage_tracking/lineage_registry.json"),
        help="Path to lineage registry JSON"
    )
    parser.add_argument(
        "--graphs-dir",
        type=Path,
        default=Path("data/current_graphs"),
        help="Directory containing citation graph PKL files"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/out/03_milestone_mapping/lineage_to_front_mapping.json"),
        help="Output path for mapping results"
    )
    parser.add_argument(
        "--phase",
        choices=["anchor", "all"],
        default="anchor",
        help="Which phase(s) to run"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose output"
    )
    parser.add_argument(
        "--allow-external-pickle",
        action="store_true",
        help="Allow loading graph pickles from outside the repository root.",
    )

    args = parser.parse_args()

    # Phase 1: Anchor-based seeding
    anchor_map = build_anchor_map_phase1(
        args.milestones,
        args.lineage_registry,
        args.graphs_dir,
        args.verbose
    )

    # Save results
    save_anchor_results(anchor_map, args.output, args.milestones)

    print("\n" + "="*70)
    print("MAPPING PHASE 1 COMPLETE")
    print("="*70)
    print("\nNext steps:")
    print("1. Enrich milestone CSV with landmark DOIs")
    print("2. Re-run Phase 1 to build lineage anchor mappings")
    print("3. Proceed to Phase 2 (SciBERT embeddings)")


if __name__ == "__main__":
    main()
