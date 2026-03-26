"""Tests for the domain layout migration script.

Validates dry-run, conflict detection, move execution,
compatibility link creation, and migration log output.
"""
from __future__ import annotations

import json
import platform
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Helpers -- build a fake legacy data tree for testing
# ---------------------------------------------------------------------------


def _create_legacy_tree(root: Path) -> None:
    """Create a minimal legacy directory structure under *root*."""
    # PSC legacy locations
    (root / "data" / "current_ingest" / "slices").mkdir(parents=True)
    (root / "data" / "current_ingest" / "ingest.parquet").write_text("fake")
    (root / "data" / "current_ingest" / "slices" / "q1.parquet").write_text("q1")
    (root / "data" / "current_graphs").mkdir(parents=True)
    (root / "data" / "current_graphs" / "graph_2020Q1.bin").write_text("graph")
    (root / "data" / "out" / "02_lineage_tracking").mkdir(parents=True)
    (root / "data" / "out" / "02_lineage_tracking" / "registry.json").write_text("{}")

    # CRISPR legacy location
    (root / "data" / "out_crispr" / "02_lineage_tracking").mkdir(parents=True)
    (root / "data" / "out_crispr" / "02_lineage_tracking" / "registry.json").write_text(
        "{}"
    )


# ---------------------------------------------------------------------------
# Import the migration module (uses sys.path setup)
# ---------------------------------------------------------------------------

import sys  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.migrate_domain_layout import (  # noqa: E402
    COMPAT_LINKS,
    MOVES,
    check_conflicts,
    create_compat_links,
    ensure_empty_domain_dirs,
    execute_moves,
    write_migration_log,
)

# ---------------------------------------------------------------------------
# Tests: check_conflicts
# ---------------------------------------------------------------------------


class TestCheckConflicts:
    """Tests for conflict detection before migration."""

    def test_no_conflicts_when_destinations_absent(self, tmp_path: Path) -> None:
        """No conflicts when destination directories don't exist."""
        moves = [
            (tmp_path / "src1", tmp_path / "dst1"),
            (tmp_path / "src2", tmp_path / "dst2"),
        ]
        assert check_conflicts(moves) == []

    def test_conflict_when_destination_exists(self, tmp_path: Path) -> None:
        """Detects conflict when destination already exists as real directory."""
        dst = tmp_path / "dst"
        dst.mkdir()
        (dst / "file.txt").write_text("data")
        moves = [(tmp_path / "src", dst)]
        conflicts = check_conflicts(moves)
        assert len(conflicts) == 1
        assert "already exists" in conflicts[0]


# ---------------------------------------------------------------------------
# Tests: execute_moves
# ---------------------------------------------------------------------------


class TestExecuteMoves:
    """Tests for directory move execution."""

    def test_dry_run_does_not_move(self, tmp_path: Path) -> None:
        """Dry run logs moves but doesn't execute them."""
        _create_legacy_tree(tmp_path)
        src = tmp_path / "data" / "current_ingest"
        dst = tmp_path / "data" / "psc" / "ingest"
        records = execute_moves([(src, dst)], dry_run=True)
        assert records[0]["status"] == "dry_run"
        assert src.exists()
        assert not dst.exists()

    def test_move_executes(self, tmp_path: Path) -> None:
        """Move physically relocates directory."""
        _create_legacy_tree(tmp_path)
        src = tmp_path / "data" / "current_ingest"
        dst = tmp_path / "data" / "psc" / "ingest"
        records = execute_moves([(src, dst)], dry_run=False)
        assert records[0]["status"] == "moved"
        assert not src.exists()
        assert dst.exists()
        assert (dst / "ingest.parquet").exists()
        assert (dst / "slices" / "q1.parquet").exists()

    def test_skip_when_source_absent(self, tmp_path: Path) -> None:
        """Skips move when source directory doesn't exist."""
        src = tmp_path / "nonexistent"
        dst = tmp_path / "destination"
        records = execute_moves([(src, dst)], dry_run=False)
        assert records[0]["status"] == "skipped_not_found"

    def test_move_records_file_counts(self, tmp_path: Path) -> None:
        """Move records include source file and directory counts."""
        _create_legacy_tree(tmp_path)
        src = tmp_path / "data" / "current_graphs"
        dst = tmp_path / "data" / "psc" / "graphs"
        records = execute_moves([(src, dst)], dry_run=False)
        assert records[0]["source_files"] >= 1
        assert records[0]["source_size_bytes"] > 0


# ---------------------------------------------------------------------------
# Tests: create_compat_links
# ---------------------------------------------------------------------------


class TestCreateCompatLinks:
    """Tests for backward-compatibility link creation."""

    def test_dry_run_does_not_create_link(self, tmp_path: Path) -> None:
        """Dry run logs link creation without acting."""
        target = tmp_path / "data" / "psc" / "ingest"
        target.mkdir(parents=True)
        links = {"data/current_ingest": "data/psc/ingest"}
        records = create_compat_links(links, tmp_path, dry_run=True)
        assert records[0]["status"] == "dry_run"
        assert not (tmp_path / "data" / "current_ingest").exists()

    def test_skip_when_target_missing(self, tmp_path: Path) -> None:
        """Skips link creation when target directory doesn't exist."""
        links = {"data/current_ingest": "data/psc/ingest"}
        records = create_compat_links(links, tmp_path, dry_run=False)
        assert records[0]["status"] == "skipped_no_target"

    def test_skip_when_real_dir_exists(self, tmp_path: Path) -> None:
        """Skips when legacy path is a real directory (not a link)."""
        (tmp_path / "data" / "current_ingest").mkdir(parents=True)
        (tmp_path / "data" / "psc" / "ingest").mkdir(parents=True)
        links = {"data/current_ingest": "data/psc/ingest"}
        records = create_compat_links(links, tmp_path, dry_run=False)
        assert records[0]["status"] == "skipped_exists"

    @pytest.mark.skipif(
        platform.system() == "Windows",
        reason="Symlink creation may require privileges on Windows",
    )
    def test_symlink_created_unix(self, tmp_path: Path) -> None:
        """Creates symlink on Unix platforms."""
        target = tmp_path / "data" / "psc" / "ingest"
        target.mkdir(parents=True)
        links = {"data/current_ingest": "data/psc/ingest"}
        records = create_compat_links(links, tmp_path, dry_run=False)
        link_path = tmp_path / "data" / "current_ingest"
        assert records[0]["status"] == "symlink_created"
        assert link_path.is_symlink()


# ---------------------------------------------------------------------------
# Tests: ensure_empty_domain_dirs
# ---------------------------------------------------------------------------


class TestEnsureEmptyDomainDirs:
    """Tests for domain directory tree creation."""

    def test_creates_domain_tree(self, tmp_path: Path) -> None:
        """Creates the full domain directory tree."""
        ensure_empty_domain_dirs(["psc"], tmp_path, dry_run=False)
        assert (tmp_path / "data" / "psc" / "ingest").exists()
        assert (tmp_path / "data" / "psc" / "graphs").exists()
        assert (tmp_path / "data" / "psc" / "out").exists()
        assert (tmp_path / "data" / "psc" / "archive").exists()

    def test_dry_run_does_not_create(self, tmp_path: Path) -> None:
        """Dry run does not create directories."""
        ensure_empty_domain_dirs(["psc"], tmp_path, dry_run=True)
        assert not (tmp_path / "data" / "psc").exists()


# ---------------------------------------------------------------------------
# Tests: write_migration_log
# ---------------------------------------------------------------------------


class TestWriteMigrationLog:
    """Tests for the migration log output."""

    def test_writes_valid_json(self, tmp_path: Path) -> None:
        """Migration log is valid JSON with expected structure."""
        log_path = tmp_path / "data" / ".migration_log.json"
        write_migration_log(
            log_path,
            move_records=[{"source": "/a", "destination": "/b", "status": "moved"}],
            link_records=[],
            dry_run=False,
        )
        data = json.loads(log_path.read_text())
        assert "timestamp" in data
        assert data["dry_run"] is False
        assert len(data["moves"]) == 1
        assert data["moves"][0]["status"] == "moved"

    def test_dry_run_flag_in_log(self, tmp_path: Path) -> None:
        """Migration log records dry_run flag."""
        log_path = tmp_path / ".migration_log.json"
        write_migration_log(log_path, [], [], dry_run=True)
        data = json.loads(log_path.read_text())
        assert data["dry_run"] is True


# ---------------------------------------------------------------------------
# Tests: full migration flow
# ---------------------------------------------------------------------------


class TestFullMigrationFlow:
    """End-to-end migration tests."""

    def test_psc_migration_moves_all_directories(self, tmp_path: Path) -> None:
        """Full PSC migration moves ingest, graphs, and out directories."""
        _create_legacy_tree(tmp_path)
        moves = [
            (tmp_path / src, tmp_path / dst)
            for src, dst in MOVES["psc"]
        ]
        records = execute_moves(moves, dry_run=False)
        assert all(r["status"] == "moved" for r in records)
        assert (tmp_path / "data" / "psc" / "ingest" / "ingest.parquet").exists()
        assert (tmp_path / "data" / "psc" / "graphs" / "graph_2020Q1.bin").exists()
        assert (
            tmp_path / "data" / "psc" / "out" / "02_lineage_tracking" / "registry.json"
        ).exists()
        # Legacy directories should be gone
        assert not (tmp_path / "data" / "current_ingest").exists()
        assert not (tmp_path / "data" / "current_graphs").exists()
        assert not (tmp_path / "data" / "out").exists()

    def test_crispr_migration_moves_out_only(self, tmp_path: Path) -> None:
        """CRISPR migration moves out_crispr to crispr/out; no ingest/graphs."""
        _create_legacy_tree(tmp_path)
        moves = [
            (tmp_path / src, tmp_path / dst)
            for src, dst in MOVES["crispr"]
        ]
        records = execute_moves(moves, dry_run=False)
        assert records[0]["status"] == "moved"
        assert (
            tmp_path / "data" / "crispr" / "out" / "02_lineage_tracking" / "registry.json"
        ).exists()
        assert not (tmp_path / "data" / "out_crispr").exists()

    def test_full_migration_with_log(self, tmp_path: Path) -> None:
        """Complete migration writes log with all records."""
        _create_legacy_tree(tmp_path)
        all_moves = []
        for domain_id in ("psc", "crispr"):
            for src_rel, dst_rel in MOVES[domain_id]:
                all_moves.append((tmp_path / src_rel, tmp_path / dst_rel))

        move_records = execute_moves(all_moves, dry_run=False)
        log_path = tmp_path / "data" / ".migration_log.json"
        write_migration_log(log_path, move_records, [], dry_run=False)

        data = json.loads(log_path.read_text())
        assert len(data["moves"]) == 4  # 3 PSC + 1 CRISPR
        moved_count = sum(1 for m in data["moves"] if m["status"] == "moved")
        assert moved_count == 4

    def test_rerun_after_migration_skips(self, tmp_path: Path) -> None:
        """Re-running migration after successful run skips all moves."""
        _create_legacy_tree(tmp_path)
        # First run
        moves = [
            (tmp_path / src, tmp_path / dst)
            for src, dst in MOVES["psc"]
        ]
        execute_moves(moves, dry_run=False)
        # Second run -- sources are gone
        records = execute_moves(moves, dry_run=False)
        assert all(r["status"] == "skipped_not_found" for r in records)


# ---------------------------------------------------------------------------
# Tests: MOVES and COMPAT_LINKS constants
# ---------------------------------------------------------------------------


class TestConstants:
    """Tests for the migration configuration constants."""

    def test_all_move_domains_registered(self) -> None:
        """All domains in MOVES are valid domain IDs."""
        from src.domain_registry import DOMAIN_REGISTRY

        for domain_id in MOVES:
            assert domain_id in DOMAIN_REGISTRY

    def test_compat_links_reference_psc_targets(self) -> None:
        """Compatibility links point to PSC domain targets."""
        for target in COMPAT_LINKS.values():
            assert "psc" in target

    def test_moves_cover_psc_and_crispr(self) -> None:
        """MOVES includes both PSC and CRISPR domains."""
        assert "psc" in MOVES
        assert "crispr" in MOVES

    def test_psc_has_three_moves(self) -> None:
        """PSC migration has 3 directory moves."""
        assert len(MOVES["psc"]) == 3

    def test_crispr_has_one_move(self) -> None:
        """CRISPR migration has 1 directory move (out only)."""
        assert len(MOVES["crispr"]) == 1
