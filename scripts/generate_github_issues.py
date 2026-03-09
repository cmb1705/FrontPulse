"""
Generate GitHub issues from a roadmap-style Markdown file.

This script parses a roadmap document and creates a JSON file that can be
used to bulk-create GitHub issues via the GitHub API or CLI.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Any


def parse_roadmap_items(roadmap_path: Path) -> List[Dict[str, Any]]:
    """
    Parse a roadmap Markdown file and extract items marked as 🔴 (Not Started).

    Returns:
        List of dictionaries with issue data
    """
    with open(roadmap_path, 'r', encoding='utf-8') as f:
        content = f.read()

    issues = []

    # Pattern to match roadmap items
    # Matches: ### 🔴 ITEM-ID: Title
    pattern = r'### 🔴 ([A-Z]+-\d+): (.+?)\n\*\*Status\*\*: (.+?)\n\*\*Priority\*\*: (.+?)\n\*\*Effort\*\*: (.+?)\n\n\*\*Current State\*\*: (.+?)\n\n\*\*Proposed Solution\*\*:\n(.*?)(?=\n\*\*Files to|---|\n###)'

    matches = re.finditer(pattern, content, re.DOTALL)

    for match in matches:
        item_id = match.group(1)
        title = match.group(2).strip()
        status = match.group(3).strip()
        priority = match.group(4).strip()
        effort = match.group(5).strip()
        current_state = match.group(6).strip()
        proposed_solution = match.group(7).strip()

        # Skip if not "Not Started"
        if status != "Not Started":
            continue

        # Determine labels based on priority and item ID
        labels = []
        if priority == "High":
            labels.append("priority: high")
        elif priority == "Medium":
            labels.append("priority: medium")
        elif priority == "Low":
            labels.append("priority: low")

        # Add category label based on ID prefix
        if item_id.startswith("HP-"):
            labels.append("type: bug")
            labels.append("category: functionality")
        elif item_id.startswith("MP-"):
            labels.append("type: enhancement")
            labels.append("category: code quality")
        elif item_id.startswith("LP-"):
            labels.append("type: enhancement")
            labels.append("category: user experience")
        elif item_id.startswith("DOC-"):
            labels.append("type: documentation")
        elif item_id.startswith("ARCH-"):
            labels.append("type: enhancement")
            labels.append("category: architecture")
        elif item_id.startswith("PERF-"):
            labels.append("type: enhancement")
            labels.append("category: performance")

        # Add effort label
        if "Small" in effort:
            labels.append("effort: small")
        elif "Medium" in effort:
            labels.append("effort: medium")
        elif "Large" in effort:
            labels.append("effort: large")

        # Build issue body
        body = f"""## Current State
{current_state}

## Proposed Solution
{proposed_solution}

## Effort Estimate
{effort}

## Reference
Item `{item_id}` from the supplied roadmap document
"""

        issues.append({
            "title": f"[{item_id}] {title}",
            "body": body.strip(),
            "labels": labels
        })

    return issues


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Generate GitHub issues JSON from a roadmap Markdown file.")
    parser.add_argument("--roadmap", required=True, help="Path to the roadmap Markdown file.")
    parser.add_argument(
        "--output",
        default="data/out/github_issues.json",
        help="Path to the output JSON file (default: %(default)s).",
    )
    return parser.parse_args()


def main():
    """Generate GitHub issues JSON file from a roadmap document."""
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    roadmap_path = Path(args.roadmap)
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = repo_root / output_path

    if not roadmap_path.exists():
        print(f"[ERROR] Roadmap file not found at {roadmap_path}")
        return

    print(f"Parsing {roadmap_path}...")
    issues = parse_roadmap_items(roadmap_path)

    print(f"Found {len(issues)} items marked as Not Started")

    # Group by category
    by_category = {}
    for issue in issues:
        category = issue["title"].split(":")[0].strip("[]")
        prefix = category.split("-")[0]
        if prefix not in by_category:
            by_category[prefix] = []
        by_category[prefix].append(issue)

    print("\nBreakdown by category:")
    for prefix, items in sorted(by_category.items()):
        print(f"  {prefix}: {len(items)} items")

    # Write to JSON
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(issues, f, indent=2, ensure_ascii=False)

    print(f"\nSaved to {output_path}")
    print("\nTo create GitHub issues:")
    print("   1. Install GitHub CLI: https://cli.github.com/")
    print("   2. Authenticate: gh auth login")
    print("   3. Run the import script: python scripts/import_github_issues.py")


if __name__ == "__main__":
    main()
