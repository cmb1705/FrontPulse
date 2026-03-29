#!/usr/bin/env python3
"""
Extract DOIs for paper milestones only (not certifications/records/events).
"""

import argparse
import re
from pathlib import Path
from typing import Optional

import pandas as pd


def is_paper_milestone(description: str) -> bool:
    """Determine if milestone is a published paper (not certification/record/event)."""
    desc_lower = description.lower()

    # Exclude certifications and records (without author attribution)
    exclude_keywords = [
        'certif', 'nrel chart', 'bankruptcy', 'covid', 'shutdowns',
        'regulatory', 'policy', 'pullback'
    ]

    # Check if description has author attribution
    has_author_pattern = bool(re.search(r'\b\w+\s+et al\b', description))

    # If has author, it's likely a paper
    if has_author_pattern:
        return True

    # Otherwise check for journal indicators
    journal_indicators = [
        'nature', 'science', 'angew', 'energy environ', 'adv mater',
        'joule', 'nano lett', 'chem rev', 'acs', 'commun'
    ]

    has_journal = any(j in desc_lower for j in journal_indicators)

    # Exclude if it's just a record/certification without author
    has_exclude = any(kw in desc_lower for kw in exclude_keywords)

    return has_journal and not has_exclude

def extract_paper_metadata(description: str) -> tuple[Optional[str], list[str]]:
    """Extract author and keywords from description."""

    # Extract author (various patterns)
    author = None

    # Pattern 1: "Author et al."
    match = re.match(r'^(\w+)\s+et al', description)
    if match:
        author = match.group(1)

    # Pattern 2: "Author et al. (Journal"
    if not author:
        match = re.search(r'(\w+)\s+et al\.\s*\(', description)
        if match:
            author = match.group(1)

    # Pattern 3: Multiple authors "Author1, Author2, Author3"
    if not author:
        match = re.search(r'(\w+),\s+(\w+),\s+(\w+)', description)
        if match:
            author = match.group(1)

    # Pattern 4: "with Author et al."
    if not author:
        match = re.search(r'with\s+(\w+)\s+et al', description)
        if match:
            author = match.group(1)

    # Extract keywords for matching
    keywords = []

    # Journal keywords
    journal_map = {
        'nature energy': ['nature', 'energy'],
        'nature communications': ['nature', 'commun'],
        'nature photonics': ['nature', 'photon'],
        'nature materials': ['nature', 'mater'],
        'nature': ['nature'],
        'science': ['science'],
        'angew': ['angew'],
        'energy environ': ['energy', 'environ'],
        'adv mater': ['adv', 'mater'],
        'joule': ['joule'],
        'nano lett': ['nano', 'lett'],
        'chem rev': ['chem', 'rev'],
        'acs energy': ['acs', 'energy'],
    }

    desc_lower = description.lower()
    for key, kw_list in journal_map.items():
        if key in desc_lower:
            keywords.extend(kw_list)

    # Technical keywords
    tech_terms = [
        'perovskite', 'tandem', 'inverted', 'stability', 'degradation',
        '2d', '3d', 'hybrid', 'lead-free', 'tin', 'passivation',
        'flexible', 'blade', 'roll-to-roll', 'spiro', 'moisture', 'oxygen'
    ]

    for term in tech_terms:
        if term in desc_lower:
            keywords.append(term)

    return author, keywords


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Extract likely DOI matches for milestone papers.")
    parser.add_argument("--milestones", type=Path, required=True, help="Path to the milestone catalog CSV.")
    parser.add_argument(
        "--mapping",
        type=Path,
        default=Path('data/out/experiments/stage0_tight_mapping/milestone_lineage_mapping_tight.csv'),
        help="Path to the milestone-lineage mapping CSV (default: %(default)s).",
    )
    parser.add_argument(
        "--ingest",
        type=Path,
        default=Path('data/current_ingest/ingest.parquet'),
        help="Path to the ingest parquet file (default: %(default)s).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path('data/out/milestone_doi_mapping.csv'),
        help="Path to the output CSV (default: %(default)s).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Load data
    milestones_df = pd.read_csv(args.milestones)
    mapping_df = pd.read_csv(args.mapping)
    ingest_df = pd.read_parquet(args.ingest)

    print(f'Loaded {len(milestones_df)} milestone events')
    print(f'Loaded {len(ingest_df):,} works from ingest')

    # Filter to paper milestones only
    paper_milestones = []
    for _idx, milestone in milestones_df.iterrows():
        if is_paper_milestone(milestone['description']):
            paper_milestones.append(milestone)

    print(f'\nFiltered to {len(paper_milestones)} paper milestones')
    print(f'Excluded {len(milestones_df) - len(paper_milestones)} certifications/events')

    results = []
    stats = {'high': 0, 'medium': 0, 'low': 0, 'no_match': 0}

    for milestone in paper_milestones:
        event_id = milestone['event_id']
        quarter = milestone['event_quarter']
        description = milestone['description']

        # Extract metadata
        year = int(quarter[:4])
        author, keywords = extract_paper_metadata(description)

        # Get lineage mapping
        lineage_matches = mapping_df[mapping_df['event_id'] == event_id]
        lineage_id = lineage_matches.iloc[0]['lineage_id'] if len(lineage_matches) > 0 else 'Not mapped'

        # Initialize result
        doi = 'Not found'
        title = 'Not found'
        work_id = 'Not found'
        confidence = 'None'
        match_reason = 'No match'

        if author:
            # Find all candidates by author + year
            candidates = ingest_df[
                (ingest_df['publication_year'] == year) &
                (ingest_df['first_author_name'].str.contains(author, case=False, na=False))
            ].copy()

            if len(candidates) > 0:
                # Rank candidates by keyword match
                best_match_idx = None
                best_score = -1

                for cidx, candidate in candidates.iterrows():
                    candidate_title = (candidate['title'] if pd.notna(candidate['title']) else '').lower()
                    candidate_venue = (candidate['primary_venue_name'] if pd.notna(candidate['primary_venue_name']) else '').lower()
                    combined = candidate_title + ' ' + candidate_venue

                    # Count keyword matches
                    keyword_matches = sum(1 for kw in keywords if kw in combined)

                    if keyword_matches > best_score:
                        best_score = keyword_matches
                        best_match_idx = cidx

                # Use best match
                if best_match_idx is not None and best_score >= 0:
                    paper = candidates.loc[best_match_idx]
                    doi = paper['doi'] if pd.notna(paper['doi']) else 'Not found'
                    title = paper['title'] if pd.notna(paper['title']) else 'Not found'
                    work_id = paper['work_id'] if pd.notna(paper['work_id']) else 'Not found'

                    # Determine confidence
                    if best_score >= 3:
                        confidence = 'HIGH'
                        match_reason = f'{best_score} keywords (author+year+venue/title)'
                        stats['high'] += 1
                    elif best_score >= 1:
                        confidence = 'MEDIUM'
                        match_reason = f'{best_score} keyword(s) (author+year)'
                        stats['medium'] += 1
                    else:
                        confidence = 'LOW'
                        match_reason = f'Author+year only ({len(candidates)} candidates)'
                        stats['low'] += 1
            else:
                stats['no_match'] += 1
                match_reason = f'Author "{author}" not found in {year} papers'
        else:
            stats['no_match'] += 1
            match_reason = 'Author not extractable'

        results.append({
            'Event_ID': event_id,
            'Author': author if author else 'Not extracted',
            'Year': year,
            'Lineage': lineage_id,
            'DOI': doi,
            'Title': title,
            'Work_ID': work_id,
            'Confidence': confidence,
            'Match_Reason': match_reason,
            'Description': description
        })

    # Save results
    result_df = pd.DataFrame(results)
    output_path = args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(output_path, index=False, encoding='utf-8')

    print('\n=== DOI Matching Results (Papers Only) ===')
    print(f'Total paper milestones: {len(result_df)}')
    print(f'DOIs found: {(result_df["DOI"] != "Not found").sum()}')
    print(f'  High confidence: {stats["high"]}')
    print(f'  Medium confidence: {stats["medium"]}')
    print(f'  Low confidence: {stats["low"]}')
    print(f'  No match: {stats["no_match"]}')
    print(f'\nResults saved to: {output_path}')

    # Show sample matches
    print('\n=== Sample High-Confidence Matches ===')
    high_conf = result_df[result_df['Confidence'] == 'HIGH'][['Author', 'Year', 'DOI', 'Title']].head(5)
    for _idx, row in high_conf.iterrows():
        print(f"\n{row['Author']} ({row['Year']})")
        print(f"  DOI: {row['DOI']}")
        print(f"  Title: {row['Title'][:80]}...")

if __name__ == '__main__':
    main()
