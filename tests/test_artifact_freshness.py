"""Tests for src.artifact_freshness -- stale input detection and manifests."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from src.artifact_freshness import (
    StaleInputError,
    build_input_manifest,
    check_freshness,
    compute_file_hash,
    require_inputs,
    save_input_manifest,
    validate_manifest_exists,
)


@pytest.fixture()
def sample_file(tmp_path: Path) -> Path:
    """Create a small test file."""
    fp = tmp_path / "data.csv"
    fp.write_text("a,b\n1,2\n", encoding="utf-8")
    return fp


class TestRequireInputs:
    """Tests for require_inputs."""

    def test_all_present(self, sample_file: Path) -> None:
        require_inputs({"data": sample_file})

    def test_missing_raises(self, tmp_path: Path) -> None:
        with pytest.raises(StaleInputError, match="Required input files missing"):
            require_inputs({"labels": tmp_path / "no_such.csv"})

    def test_missing_with_context(self, tmp_path: Path) -> None:
        with pytest.raises(StaleInputError, match="compute_features"):
            require_inputs(
                {"labels": tmp_path / "missing.csv"},
                context="compute_features",
            )

    def test_multiple_missing(self, tmp_path: Path) -> None:
        with pytest.raises(StaleInputError) as exc_info:
            require_inputs({
                "a": tmp_path / "a.csv",
                "b": tmp_path / "b.csv",
            })
        assert "a:" in str(exc_info.value)
        assert "b:" in str(exc_info.value)

    def test_partial_missing(self, sample_file: Path, tmp_path: Path) -> None:
        with pytest.raises(StaleInputError, match="missing"):
            require_inputs({
                "present": sample_file,
                "missing": tmp_path / "gone.csv",
            })


class TestCheckFreshness:
    """Tests for check_freshness."""

    def test_fresh_output_passes(self, tmp_path: Path) -> None:
        inp = tmp_path / "input.csv"
        inp.write_text("data", encoding="utf-8")
        time.sleep(0.05)
        out = tmp_path / "output.csv"
        out.write_text("result", encoding="utf-8")
        check_freshness(out, {"input": inp})

    def test_stale_output_raises(self, tmp_path: Path) -> None:
        out = tmp_path / "output.csv"
        out.write_text("old result", encoding="utf-8")
        time.sleep(0.05)
        inp = tmp_path / "input.csv"
        inp.write_text("new data", encoding="utf-8")
        with pytest.raises(StaleInputError, match="stale"):
            check_freshness(out, {"input": inp})

    def test_nonexistent_output_passes(self, tmp_path: Path) -> None:
        inp = tmp_path / "input.csv"
        inp.write_text("data", encoding="utf-8")
        check_freshness(tmp_path / "not_yet.csv", {"input": inp})

    def test_nonexistent_input_ignored(self, tmp_path: Path) -> None:
        out = tmp_path / "output.csv"
        out.write_text("result", encoding="utf-8")
        check_freshness(out, {"missing": tmp_path / "gone.csv"})


class TestBuildInputManifest:
    """Tests for build_input_manifest."""

    def test_contains_generated_at(self, sample_file: Path) -> None:
        manifest = build_input_manifest({"data": sample_file})
        assert "generated_at" in manifest

    def test_input_hashed(self, sample_file: Path) -> None:
        manifest = build_input_manifest({"data": sample_file})
        assert manifest["inputs"]["data"]["sha256"] is not None
        assert len(manifest["inputs"]["data"]["sha256"]) == 64

    def test_none_input_recorded(self) -> None:
        manifest = build_input_manifest({"labels": None})
        assert manifest["inputs"]["labels"]["path"] is None

    def test_missing_file_no_hash(self, tmp_path: Path) -> None:
        manifest = build_input_manifest({"x": tmp_path / "missing.csv"})
        assert "sha256" not in manifest["inputs"]["x"]


class TestSaveInputManifest:
    """Tests for save_input_manifest."""

    def test_creates_file(self, sample_file: Path, tmp_path: Path) -> None:
        out = save_input_manifest({"data": sample_file}, tmp_path)
        assert out.exists()
        assert out.name == "input_manifest.json"

    def test_valid_json(self, sample_file: Path, tmp_path: Path) -> None:
        out = save_input_manifest({"data": sample_file}, tmp_path)
        data = json.loads(out.read_text(encoding="utf-8"))
        assert "inputs" in data
        assert "data" in data["inputs"]

    def test_custom_filename(self, tmp_path: Path) -> None:
        out = save_input_manifest({}, tmp_path, filename="custom.json")
        assert out.name == "custom.json"


class TestValidateManifestExists:
    """Tests for validate_manifest_exists."""

    def test_valid(self, sample_file: Path, tmp_path: Path) -> None:
        save_input_manifest({"data": sample_file}, tmp_path)
        assert validate_manifest_exists(tmp_path) is True

    def test_missing(self, tmp_path: Path) -> None:
        assert validate_manifest_exists(tmp_path) is False

    def test_invalid_json(self, tmp_path: Path) -> None:
        (tmp_path / "input_manifest.json").write_text("not json", encoding="utf-8")
        assert validate_manifest_exists(tmp_path) is False

    def test_missing_inputs_key(self, tmp_path: Path) -> None:
        (tmp_path / "input_manifest.json").write_text('{"x": 1}', encoding="utf-8")
        assert validate_manifest_exists(tmp_path) is False


class TestComputeFileHash:
    """Tests for compute_file_hash (mirrored from run_provenance)."""

    def test_deterministic(self, sample_file: Path) -> None:
        assert compute_file_hash(sample_file) == compute_file_hash(sample_file)

    def test_sha256_length(self, sample_file: Path) -> None:
        assert len(compute_file_hash(sample_file)) == 64
