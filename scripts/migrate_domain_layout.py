#!/usr/bin/env python3
"""One-time migration: move existing data into domain-isolated directories.

Moves legacy shared directories into the canonical data/{domain_id}/ layout
defined by DomainDataPaths.  Supports dry-run rehearsal, conflict detection,
an auditable migration log, and optional backward-compatibility links.

Usage:
    python scripts/migrate_domain_layout.py --dry-run
    python scripts/migrate_domain_layout.py
    python scripts/migrate_domain_layout.py --domain psc
    python scripts/migrate_domain_layout.py --create-links
"""
from __future__ import annotations

import argparse
import contextlib
import datetime
import json
import logging
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from _path_bootstrap import ensure_repo_imports

REPO = ensure_repo_imports()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Migration map: (legacy_relative_path, domain_relative_path)
# Paths are relative to the repo root.
# ---------------------------------------------------------------------------

MOVES: dict[str, list[tuple[str, str]]] = {
    "psc": [
        ("data/current_ingest", "data/psc/ingest"),
        ("data/current_graphs", "data/psc/graphs"),
        ("data/out", "data/psc/out"),
    ],
    "crispr": [
        ("data/out_crispr", "data/crispr/out"),
        # data/crispr/ingest and data/crispr/graphs start empty;
        # CRISPR raw data needs re-ingestion into the isolated tree.
    ],
}

# Links that allow legacy paths to keep working until all scripts are ported.
# Mapping: legacy_relative_path -> target_relative_path (post-migration)
COMPAT_LINKS: dict[str, str] = {
    "data/current_ingest": "data/psc/ingest",
    "data/current_graphs": "data/psc/graphs",
    "data/out": "data/psc/out",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _count_items(directory: Path) -> dict[str, int]:
    """Count files and directories under *directory* (non-recursive)."""
    if not directory.exists():
        return {"files": 0, "dirs": 0}
    files = sum(1 for p in directory.iterdir() if p.is_file())
    dirs = sum(1 for p in directory.iterdir() if p.is_dir())
    return {"files": files, "dirs": dirs}


def _dir_size_bytes(directory: Path) -> int:
    """Approximate total size in bytes (walks recursively)."""
    total = 0
    if not directory.exists():
        return 0
    for p in directory.rglob("*"):
        if p.is_file():
            with contextlib.suppress(OSError):
                total += p.stat().st_size
    return total


def _human_size(nbytes: int) -> str:
    """Format byte count as human-readable string."""
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(nbytes) < 1024.0:
            return f"{nbytes:.1f} {unit}"
        nbytes /= 1024.0  # type: ignore[assignment]
    return f"{nbytes:.1f} PB"


def _is_junction_or_symlink(path: Path) -> bool:
    """Check whether *path* is a symlink or Windows junction."""
    if path.is_symlink():
        return True
    # On Windows, junctions are reparse points but not always flagged as symlinks.
    if platform.system() == "Windows" and path.exists():
        try:
            attrs = path.lstat().st_file_attributes  # type: ignore[attr-defined]
            _IO_REPARSE_TAG = 0x0400
            return bool(attrs & _IO_REPARSE_TAG)
        except (AttributeError, OSError):
            pass
    return False


# ---------------------------------------------------------------------------
# Core migration logic
# ---------------------------------------------------------------------------


def check_conflicts(
    moves: list[tuple[Path, Path]],
) -> list[str]:
    """Return a list of conflict descriptions (empty = safe to proceed)."""
    conflicts: list[str] = []
    for _src, dst in moves:
        if dst.exists() and not _is_junction_or_symlink(dst):
            conflicts.append(
                f"Destination already exists: {dst} "
                f"({_count_items(dst)['files']} files, "
                f"{_count_items(dst)['dirs']} subdirs)"
            )
    return conflicts


def execute_moves(
    moves: list[tuple[Path, Path]],
    dry_run: bool = False,
) -> list[dict[str, Any]]:
    """Move directories from legacy to domain-isolated locations.

    Args:
        moves: List of (source, destination) absolute paths.
        dry_run: If True, only log planned moves without executing.

    Returns:
        List of move records for the migration log.
    """
    records: list[dict[str, Any]] = []
    for src, dst in moves:
        record: dict[str, Any] = {
            "source": str(src),
            "destination": str(dst),
            "status": "skipped",
        }

        if not src.exists():
            record["status"] = "skipped_not_found"
            logger.info("SKIP (not found): %s", src)
            records.append(record)
            continue

        # Remove any existing junction/symlink at the destination
        if _is_junction_or_symlink(dst):
            if not dry_run:
                if platform.system() == "Windows" and dst.is_dir():
                    # Junctions on Windows: rmdir removes the junction, not the target
                    os.rmdir(dst)
                else:
                    dst.unlink()
            logger.info("Removed existing link at destination: %s", dst)

        counts = _count_items(src)
        size = _dir_size_bytes(src)
        record["source_files"] = counts["files"]
        record["source_dirs"] = counts["dirs"]
        record["source_size_bytes"] = size

        if dry_run:
            record["status"] = "dry_run"
            logger.info(
                "DRY RUN: %s -> %s (%d files, %d dirs, %s)",
                src, dst, counts["files"], counts["dirs"], _human_size(size),
            )
        else:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(dst))
            record["status"] = "moved"
            logger.info(
                "MOVED: %s -> %s (%d files, %d dirs, %s)",
                src, dst, counts["files"], counts["dirs"], _human_size(size),
            )

        records.append(record)
    return records


def create_compat_links(
    links: dict[str, str],
    project_root: Path,
    dry_run: bool = False,
) -> list[dict[str, Any]]:
    """Create backward-compatibility links (junctions on Windows, symlinks on Unix).

    Args:
        links: Mapping of legacy relative path -> target relative path.
        project_root: Repository root.
        dry_run: If True, only log planned links.

    Returns:
        List of link records for the migration log.
    """
    records: list[dict[str, Any]] = []
    for legacy_rel, target_rel in links.items():
        legacy = project_root / legacy_rel
        target = project_root / target_rel
        record: dict[str, Any] = {
            "link": str(legacy),
            "target": str(target),
            "status": "skipped",
        }

        if legacy.exists() and not _is_junction_or_symlink(legacy):
            record["status"] = "skipped_exists"
            logger.warning(
                "SKIP link: %s already exists as a real directory", legacy
            )
            records.append(record)
            continue

        if not target.exists():
            record["status"] = "skipped_no_target"
            logger.warning("SKIP link: target %s does not exist", target)
            records.append(record)
            continue

        # Remove any stale link first
        if _is_junction_or_symlink(legacy) and not dry_run:
            if platform.system() == "Windows" and legacy.is_dir():
                os.rmdir(legacy)
            else:
                legacy.unlink()

        if dry_run:
            record["status"] = "dry_run"
            link_type = "junction" if platform.system() == "Windows" else "symlink"
            logger.info("DRY RUN link (%s): %s -> %s", link_type, legacy, target)
        else:
            if platform.system() == "Windows":
                # Use mklink /J for junctions (no admin privileges required)
                subprocess.run(
                    ["cmd", "/c", "mklink", "/J", str(legacy), str(target)],
                    check=True,
                    capture_output=True,
                )
                record["status"] = "junction_created"
                logger.info("JUNCTION: %s -> %s", legacy, target)
            else:
                legacy.symlink_to(target)
                record["status"] = "symlink_created"
                logger.info("SYMLINK: %s -> %s", legacy, target)

        records.append(record)
    return records


def ensure_empty_domain_dirs(
    domains: list[str],
    project_root: Path,
    dry_run: bool = False,
) -> None:
    """Create empty domain directory trees for domains that have no source data.

    For CRISPR, ingest/ and graphs/ start empty since the raw data
    needs to be re-ingested into the isolated tree.
    """
    from src.domain_registry import get_domain

    for domain_id in domains:
        domain = get_domain(domain_id)
        paths = domain.resolve_data_paths(project_root)
        if dry_run:
            logger.info(
                "DRY RUN: would create directory tree for domain '%s' at %s",
                domain_id, paths.base,
            )
        else:
            paths.ensure_dirs()
            logger.info(
                "Created directory tree for domain '%s' at %s",
                domain_id, paths.base,
            )


def write_migration_log(
    log_path: Path,
    move_records: list[dict[str, Any]],
    link_records: list[dict[str, Any]],
    dry_run: bool,
) -> None:
    """Write an auditable migration log."""
    log_data = {
        "timestamp": datetime.datetime.now(tz=datetime.timezone.utc).isoformat(),
        "dry_run": dry_run,
        "moves": move_records,
        "links": link_records,
    }
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(json.dumps(log_data, indent=2))
    logger.info("Migration log written to %s", log_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """Build the migration script argument parser."""
    ap = argparse.ArgumentParser(
        description="Migrate existing data into domain-isolated directory layout.",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned moves without executing",
    )
    ap.add_argument(
        "--domain",
        default=None,
        choices=sorted(MOVES.keys()),
        help="Migrate only this domain (default: all registered domains)",
    )
    ap.add_argument(
        "--create-links",
        action="store_true",
        help="Create backward-compatibility links after migration",
    )
    ap.add_argument(
        "--project-root",
        type=Path,
        default=REPO,
        help="Repository root directory (default: auto-detected)",
    )
    ap.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging",
    )
    return ap


def main() -> None:
    """Run the domain layout migration."""
    args = build_parser().parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    project_root = args.project_root.resolve()
    logger.info("Project root: %s", project_root)
    logger.info("Mode: %s", "DRY RUN" if args.dry_run else "EXECUTE")

    # Determine which domains to migrate
    domains_to_migrate = [args.domain] if args.domain else list(MOVES.keys())

    # Build absolute move list
    all_moves: list[tuple[Path, Path]] = []
    for domain_id in domains_to_migrate:
        for src_rel, dst_rel in MOVES.get(domain_id, []):
            all_moves.append((project_root / src_rel, project_root / dst_rel))

    # Check for conflicts
    conflicts = check_conflicts(all_moves)
    if conflicts:
        logger.error("Destination conflicts detected -- aborting:")
        for c in conflicts:
            logger.error("  %s", c)
        sys.exit(1)

    # Execute moves
    move_records = execute_moves(all_moves, dry_run=args.dry_run)

    # Ensure all domain directory trees exist (even empty ones like CRISPR ingest)
    ensure_empty_domain_dirs(domains_to_migrate, project_root, dry_run=args.dry_run)

    # Create compatibility links if requested
    link_records: list[dict[str, Any]] = []
    if args.create_links:
        link_records = create_compat_links(
            COMPAT_LINKS, project_root, dry_run=args.dry_run,
        )

    # Write migration log
    log_path = project_root / "data" / ".migration_log.json"
    write_migration_log(log_path, move_records, link_records, args.dry_run)

    # Summary
    moved = sum(1 for r in move_records if r["status"] == "moved")
    skipped = sum(1 for r in move_records if r["status"].startswith("skipped"))
    dry = sum(1 for r in move_records if r["status"] == "dry_run")
    links_created = sum(
        1 for r in link_records
        if r["status"] in ("junction_created", "symlink_created")
    )

    logger.info("--- Migration Summary ---")
    if args.dry_run:
        logger.info("DRY RUN: %d moves planned, %d skipped", dry, skipped)
    else:
        logger.info("Executed: %d moved, %d skipped", moved, skipped)
    if link_records:
        logger.info("Links: %d created", links_created)
    logger.info("Log: %s", log_path)

    if not args.dry_run and moved > 0:
        logger.info(
            "Migration complete. Verify with: "
            "ls data/psc/ data/crispr/"
        )
        logger.info(
            "To create backward-compatibility links, rerun with --create-links"
        )


if __name__ == "__main__":
    main()
