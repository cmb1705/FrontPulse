"""Tests for src.run_provenance -- experiment run provenance tracking."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from src.run_provenance import (
    collect_run_provenance,
    compute_file_hash,
    save_run_provenance,
    validate_provenance_exists,
)


@pytest.fixture()
def sample_args() -> argparse.Namespace:
    """Minimal argparse namespace simulating MSD CLI args."""
    return argparse.Namespace(
        labels="data/crispr/out/02_lineage_tracking/onset_labels_msd.csv",
        multisignal="data/crispr/out/02_lineage_tracking/lineage_multisignal_features.csv",
        output_dir="data/crispr/out/experiments/msd_test",
        model="catboost",
        use_cv=True,
        cv_folds=5,
        threshold=0.70,
        persistence_window=2,
        leakage_safe=True,
        train_start=None,
        train_end=None,
        predict_start=None,
        predict_end=None,
        lag_min=2,
        lag_max=8,
        domain="crispr",
    )


@pytest.fixture()
def dummy_input_file(tmp_path: Path) -> Path:
    """Create a small file for hash testing."""
    fp = tmp_path / "dummy.csv"
    fp.write_text("lineage_id,quarter,value\n1,2020Q1,42\n", encoding="utf-8")
    return fp


class TestComputeFileHash:
    """Tests for compute_file_hash."""

    def test_deterministic(self, dummy_input_file: Path) -> None:
        h1 = compute_file_hash(dummy_input_file)
        h2 = compute_file_hash(dummy_input_file)
        assert h1 == h2

    def test_sha256_length(self, dummy_input_file: Path) -> None:
        h = compute_file_hash(dummy_input_file)
        assert len(h) == 64  # SHA-256 hex digest is 64 chars

    def test_different_content_different_hash(self, tmp_path: Path) -> None:
        f1 = tmp_path / "a.txt"
        f2 = tmp_path / "b.txt"
        f1.write_text("alpha", encoding="utf-8")
        f2.write_text("beta", encoding="utf-8")
        assert compute_file_hash(f1) != compute_file_hash(f2)


class TestCollectRunProvenance:
    """Tests for collect_run_provenance."""

    def test_contains_required_fields(
        self, sample_args: argparse.Namespace, tmp_path: Path
    ) -> None:
        prov = collect_run_provenance(
            sample_args,
            {"labels": None, "multisignal": None},
            tmp_path,
        )
        assert "generated_at" in prov
        assert "python_version" in prov
        assert "cli_args" in prov
        assert "input_files" in prov
        assert "output_dir" in prov

    def test_cli_args_captured(
        self, sample_args: argparse.Namespace, tmp_path: Path
    ) -> None:
        prov = collect_run_provenance(
            sample_args,
            {},
            tmp_path,
        )
        assert prov["cli_args"]["model"] == "catboost"
        assert prov["cli_args"]["leakage_safe"] is True
        assert prov["cli_args"]["threshold"] == 0.70

    def test_input_file_hashed(
        self,
        sample_args: argparse.Namespace,
        dummy_input_file: Path,
        tmp_path: Path,
    ) -> None:
        prov = collect_run_provenance(
            sample_args,
            {"labels": dummy_input_file},
            tmp_path,
        )
        entry = prov["input_files"]["labels"]
        assert entry["sha256"] is not None
        assert len(entry["sha256"]) == 64

    def test_missing_input_file(
        self, sample_args: argparse.Namespace, tmp_path: Path
    ) -> None:
        prov = collect_run_provenance(
            sample_args,
            {"labels": None},
            tmp_path,
        )
        assert prov["input_files"]["labels"]["sha256"] is None

    def test_nonexistent_input_file_path(
        self, sample_args: argparse.Namespace, tmp_path: Path
    ) -> None:
        prov = collect_run_provenance(
            sample_args,
            {"labels": tmp_path / "no_such_file.csv"},
            tmp_path,
        )
        assert prov["input_files"]["labels"]["sha256"] is None

    def test_extra_metadata_included(
        self, sample_args: argparse.Namespace, tmp_path: Path
    ) -> None:
        prov = collect_run_provenance(
            sample_args,
            {},
            tmp_path,
            extra={"n_features": 65, "n_train_samples": 21608},
        )
        assert prov["extra"]["n_features"] == 65

    def test_git_sha_present(
        self, sample_args: argparse.Namespace, tmp_path: Path
    ) -> None:
        """Git SHA should be populated when run inside a git repo."""
        prov = collect_run_provenance(
            sample_args,
            {},
            tmp_path,
            repo_root=Path(__file__).resolve().parent.parent,
        )
        # In a git repo, git_sha should be a 40-char hex string
        assert "git_sha" in prov
        assert len(prov["git_sha"]) == 40

    def test_path_args_serialized_as_strings(
        self, tmp_path: Path
    ) -> None:
        args = argparse.Namespace(some_path=Path("/tmp/test"))
        prov = collect_run_provenance(args, {}, tmp_path)
        assert isinstance(prov["cli_args"]["some_path"], str)


class TestSaveRunProvenance:
    """Tests for save_run_provenance."""

    def test_creates_file(self, tmp_path: Path) -> None:
        prov = {"cli_args": {"model": "catboost"}, "generated_at": "2026-03-31T00:00:00Z"}
        out = save_run_provenance(prov, tmp_path)
        assert out.exists()
        assert out.name == "run_provenance.json"

    def test_valid_json(self, tmp_path: Path) -> None:
        prov = {"cli_args": {"threshold": 0.7}, "generated_at": "2026-03-31T00:00:00Z"}
        out = save_run_provenance(prov, tmp_path)
        loaded = json.loads(out.read_text(encoding="utf-8"))
        assert loaded["cli_args"]["threshold"] == 0.7

    def test_creates_parent_dirs(self, tmp_path: Path) -> None:
        nested = tmp_path / "a" / "b" / "c"
        prov = {"cli_args": {}, "generated_at": "now"}
        out = save_run_provenance(prov, nested)
        assert out.exists()

    def test_custom_filename(self, tmp_path: Path) -> None:
        prov = {"cli_args": {}, "generated_at": "now"}
        out = save_run_provenance(prov, tmp_path, filename="custom_prov.json")
        assert out.name == "custom_prov.json"


class TestValidateProvenanceExists:
    """Tests for validate_provenance_exists."""

    def test_valid_provenance(self, tmp_path: Path) -> None:
        prov = {"cli_args": {"model": "catboost"}, "generated_at": "now"}
        save_run_provenance(prov, tmp_path)
        assert validate_provenance_exists(tmp_path) is True

    def test_missing_provenance(self, tmp_path: Path) -> None:
        assert validate_provenance_exists(tmp_path) is False

    def test_invalid_json(self, tmp_path: Path) -> None:
        (tmp_path / "run_provenance.json").write_text("not json", encoding="utf-8")
        assert validate_provenance_exists(tmp_path) is False

    def test_missing_cli_args_key(self, tmp_path: Path) -> None:
        (tmp_path / "run_provenance.json").write_text(
            '{"generated_at": "now"}', encoding="utf-8"
        )
        assert validate_provenance_exists(tmp_path) is False

    def test_custom_filename(self, tmp_path: Path) -> None:
        prov = {"cli_args": {}, "generated_at": "now"}
        save_run_provenance(prov, tmp_path, filename="alt.json")
        assert validate_provenance_exists(tmp_path, filename="alt.json") is True
        assert validate_provenance_exists(tmp_path) is False
