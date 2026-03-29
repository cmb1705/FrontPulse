"""
Import GitHub issues from github_issues.json using GitHub CLI.

Prerequisites:
    - GitHub CLI installed: https://cli.github.com/
    - Authenticated: gh auth login

Usage:
    python scripts/import_github_issues.py [--dry-run]

Options:
    --dry-run    Show what would be created without actually creating issues
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any


def check_gh_cli() -> bool:
    """Check if GitHub CLI is installed and authenticated."""
    try:
        result = subprocess.run(
            ["gh", "auth", "status"],
            capture_output=True,
            text=True
        )
        return result.returncode == 0
    except FileNotFoundError:
        return False


def create_issue(issue_data: dict[str, Any], dry_run: bool = False) -> bool:
    """
    Create a GitHub issue using gh CLI.

    Args:
        issue_data: Dictionary with title, body, and labels
        dry_run: If True, only print what would be created

    Returns:
        True if successful, False otherwise
    """
    title = issue_data["title"]
    body = issue_data["body"]
    labels = ",".join(issue_data.get("labels", []))

    if dry_run:
        print(f"\n{'='*80}")
        print(f"Would create issue: {title}")
        print(f"Labels: {labels}")
        print(f"Body preview (first 200 chars):\n{body[:200]}...")
        return True

    try:
        cmd = [
            "gh", "issue", "create",
            "--title", title,
            "--body", body,
        ]

        if labels:
            cmd.extend(["--label", labels])

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )

        print(f"[OK] Created: {title}")
        # Extract issue number from output (format: "https://github.com/user/repo/issues/123")
        output = result.stdout.strip()
        if output:
            print(f"   {output}")

        return True

    except subprocess.CalledProcessError as e:
        print(f"[FAIL] Failed to create: {title}")
        print(f"   Error: {e.stderr}")
        return False


def main():
    """Import GitHub issues from JSON file."""
    dry_run = "--dry-run" in sys.argv

    repo_root = Path(__file__).parent.parent
    issues_file = repo_root / "github_issues.json"

    if not issues_file.exists():
        print(f"[ERROR] {issues_file} not found")
        print("   Run: python scripts/generate_github_issues.py")
        return

    # Check GitHub CLI
    print("Checking GitHub CLI...")
    if not check_gh_cli():
        print("[ERROR] GitHub CLI not found or not authenticated")
        print("\nTo fix:")
        print("   1. Install: https://cli.github.com/")
        print("   2. Authenticate: gh auth login")
        return

    print("[OK] GitHub CLI authenticated")

    # Load issues
    with open(issues_file, encoding='utf-8') as f:
        issues = json.load(f)

    print(f"\nFound {len(issues)} issues to create")

    if dry_run:
        print("\n[DRY RUN] No issues will be created\n")

    # Create issues
    success_count = 0
    fail_count = 0

    for i, issue in enumerate(issues, 1):
        print(f"\n[{i}/{len(issues)}] ", end="")

        if create_issue(issue, dry_run=dry_run):
            success_count += 1
        else:
            fail_count += 1

        # Small delay to avoid rate limiting
        if not dry_run:
            import time
            time.sleep(0.5)

    print(f"\n{'='*80}")
    print(f"[OK] Success: {success_count}")
    print(f"[FAIL] Failed: {fail_count}")

    if dry_run:
        print("\nRun without --dry-run to actually create issues")


if __name__ == "__main__":
    main()
