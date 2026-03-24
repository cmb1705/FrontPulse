"""Tests for model versioning registry."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.model_registry import (
    ModelVersion,
    _next_version_id,
    compare_versions,
    get_latest_version_id,
    list_versions,
    load_model_version,
    save_versioned_model,
)


# ---------------------------------------------------------------------------
# Version ID generation
# ---------------------------------------------------------------------------


class TestNextVersionId:
    """Tests for version ID sequence generation."""

    def test_first_version_of_day(self, tmp_path: Path) -> None:
        vid = _next_version_id(tmp_path, date_str="20260323")
        assert vid == "v_20260323_001"

    def test_increments_within_day(self, tmp_path: Path) -> None:
        (tmp_path / "v_20260323_001").mkdir()
        vid = _next_version_id(tmp_path, date_str="20260323")
        assert vid == "v_20260323_002"

    def test_ignores_other_dates(self, tmp_path: Path) -> None:
        (tmp_path / "v_20260322_005").mkdir()
        vid = _next_version_id(tmp_path, date_str="20260323")
        assert vid == "v_20260323_001"

    def test_nonexistent_directory(self, tmp_path: Path) -> None:
        vid = _next_version_id(tmp_path / "no_such_dir", date_str="20260323")
        assert vid == "v_20260323_001"

    def test_multiple_existing_versions(self, tmp_path: Path) -> None:
        for i in range(1, 6):
            (tmp_path / f"v_20260323_{i:03d}").mkdir()
        vid = _next_version_id(tmp_path, date_str="20260323")
        assert vid == "v_20260323_006"


# ---------------------------------------------------------------------------
# ModelVersion dataclass
# ---------------------------------------------------------------------------


class TestModelVersion:
    """Tests for ModelVersion metadata."""

    def test_summary_metrics_extracts_known_keys(self) -> None:
        v = ModelVersion(
            version_id="v_20260323_001",
            created_at="2026-03-23T00:00:00Z",
            model_type="catboost",
            metrics={
                "cv_pr_auc_mean": 0.17,
                "cv_roc_auc_mean": 0.93,
                "cv_recall_mean": 0.31,
                "unknown_metric": 42,
            },
        )
        summary = v.summary_metrics()
        assert "cv_pr_auc_mean" in summary
        assert "cv_roc_auc_mean" in summary
        assert "unknown_metric" not in summary

    def test_summary_metrics_empty_when_no_known_keys(self) -> None:
        v = ModelVersion(
            version_id="v_test",
            created_at="2026-03-23",
            model_type="catboost",
            metrics={"custom": 1.0},
        )
        assert v.summary_metrics() == {}


# ---------------------------------------------------------------------------
# Save and load round-trip
# ---------------------------------------------------------------------------


class TestSaveLoadRoundTrip:
    """Tests for saving and loading versioned models."""

    def _make_version(self, version_id: str = "v_20260323_001") -> ModelVersion:
        return ModelVersion(
            version_id=version_id,
            created_at="2026-03-23T12:00:00Z",
            model_type="catboost",
            train_quarters="2003Q1-2023Q4",
            n_train_samples=21000,
            n_features=51,
            feature_names=["growth_rate", "new_works", "semantic_velocity"],
            metrics={"cv_pr_auc_mean": 0.17, "cv_roc_auc_mean": 0.93},
            config={"cat_depth": 8, "cat_iterations": 1000},
            retrain_mode="full",
            notes="test run",
        )

    def test_save_creates_expected_files(self, tmp_path: Path) -> None:
        version = self._make_version()
        pipeline = {"mock": "pipeline"}  # Simplified for test

        version_dir = save_versioned_model(pipeline, version, tmp_path, allow_external=True)

        assert (version_dir / "model.pkl").exists()
        assert (version_dir / "feature_names.json").exists()
        assert (version_dir / "manifest.json").exists()
        assert (tmp_path / "registry.json").exists()

    def test_round_trip_preserves_metadata(self, tmp_path: Path) -> None:
        version = self._make_version()
        pipeline = {"key": "value", "numbers": [1, 2, 3]}

        save_versioned_model(pipeline, version, tmp_path, allow_external=True)
        loaded_pipeline, loaded_version = load_model_version(tmp_path, "v_20260323_001", allow_external=True)

        assert loaded_pipeline == pipeline
        assert loaded_version.version_id == version.version_id
        assert loaded_version.model_type == "catboost"
        assert loaded_version.n_train_samples == 21000
        assert loaded_version.feature_names == ["growth_rate", "new_works", "semantic_velocity"]
        assert loaded_version.metrics["cv_pr_auc_mean"] == 0.17

    def test_load_latest_when_no_id_given(self, tmp_path: Path) -> None:
        v1 = self._make_version("v_20260322_001")
        v2 = self._make_version("v_20260323_001")
        save_versioned_model({"v": 1}, v1, tmp_path, allow_external=True)
        save_versioned_model({"v": 2}, v2, tmp_path, allow_external=True)

        pipeline, version = load_model_version(tmp_path, allow_external=True)
        assert version.version_id == "v_20260323_001"
        assert pipeline == {"v": 2}

    def test_load_nonexistent_version_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_model_version(tmp_path, "v_99999999_001", allow_external=True)

    def test_load_empty_registry_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="No model versions"):
            load_model_version(tmp_path, allow_external=True)


# ---------------------------------------------------------------------------
# Registry index
# ---------------------------------------------------------------------------


class TestRegistryIndex:
    """Tests for registry index management."""

    def test_registry_index_tracks_versions(self, tmp_path: Path) -> None:
        v1 = ModelVersion(
            version_id="v_20260322_001",
            created_at="2026-03-22",
            model_type="catboost",
            metrics={"cv_pr_auc_mean": 0.15},
        )
        v2 = ModelVersion(
            version_id="v_20260323_001",
            created_at="2026-03-23",
            model_type="catboost",
            metrics={"cv_pr_auc_mean": 0.17},
        )
        save_versioned_model({"v": 1}, v1, tmp_path, allow_external=True)
        save_versioned_model({"v": 2}, v2, tmp_path, allow_external=True)

        index = json.loads((tmp_path / "registry.json").read_text())
        assert len(index["versions"]) == 2
        assert index["versions"][0]["version_id"] == "v_20260322_001"
        assert index["versions"][1]["version_id"] == "v_20260323_001"

    def test_get_latest_version_id(self, tmp_path: Path) -> None:
        assert get_latest_version_id(tmp_path) is None

        v = ModelVersion(
            version_id="v_20260323_001",
            created_at="2026-03-23",
            model_type="catboost",
        )
        save_versioned_model({}, v, tmp_path, allow_external=True)
        assert get_latest_version_id(tmp_path) == "v_20260323_001"

    def test_list_versions_returns_ordered(self, tmp_path: Path) -> None:
        for i, vid in enumerate(["v_20260321_001", "v_20260322_001", "v_20260323_001"]):
            v = ModelVersion(
                version_id=vid,
                created_at=f"2026-03-2{i + 1}",
                model_type="catboost",
                metrics={"cv_pr_auc_mean": 0.10 + i * 0.02},
            )
            save_versioned_model({}, v, tmp_path, allow_external=True)

        versions = list_versions(tmp_path)
        assert len(versions) == 3
        assert versions[0].version_id == "v_20260321_001"
        assert versions[2].version_id == "v_20260323_001"

    def test_save_updates_existing_entry(self, tmp_path: Path) -> None:
        v = ModelVersion(
            version_id="v_20260323_001",
            created_at="2026-03-23",
            model_type="catboost",
            metrics={"cv_pr_auc_mean": 0.15},
        )
        save_versioned_model({}, v, tmp_path, allow_external=True)

        # Save again with updated metrics
        v.metrics = {"cv_pr_auc_mean": 0.20}
        save_versioned_model({}, v, tmp_path, allow_external=True)

        index = json.loads((tmp_path / "registry.json").read_text())
        assert len(index["versions"]) == 1
        assert index["versions"][0]["summary_metrics"]["cv_pr_auc_mean"] == 0.20


# ---------------------------------------------------------------------------
# Version comparison
# ---------------------------------------------------------------------------


class TestCompareVersions:
    """Tests for metric comparison between versions."""

    def test_improved_pr_auc(self) -> None:
        v_old = ModelVersion(
            version_id="v_old", created_at="2026-03-22", model_type="catboost",
            metrics={"cv_pr_auc_mean": 0.15, "cv_roc_auc_mean": 0.92},
        )
        v_new = ModelVersion(
            version_id="v_new", created_at="2026-03-23", model_type="catboost",
            metrics={"cv_pr_auc_mean": 0.18, "cv_roc_auc_mean": 0.93},
        )
        comp = compare_versions(v_new, v_old)
        assert comp["improved"] is True
        assert comp["primary_metric"] == "cv_pr_auc_mean"
        assert comp["deltas"]["cv_pr_auc_mean"] == pytest.approx(0.03)

    def test_regressed_pr_auc(self) -> None:
        v_old = ModelVersion(
            version_id="v_old", created_at="2026-03-22", model_type="catboost",
            metrics={"cv_pr_auc_mean": 0.20},
        )
        v_new = ModelVersion(
            version_id="v_new", created_at="2026-03-23", model_type="catboost",
            metrics={"cv_pr_auc_mean": 0.15},
        )
        comp = compare_versions(v_new, v_old)
        assert comp["improved"] is False

    def test_no_common_metrics(self) -> None:
        v_old = ModelVersion(
            version_id="v_old", created_at="2026-03-22", model_type="catboost",
            metrics={"custom_a": 1.0},
        )
        v_new = ModelVersion(
            version_id="v_new", created_at="2026-03-23", model_type="catboost",
            metrics={"custom_b": 2.0},
        )
        comp = compare_versions(v_new, v_old)
        assert comp["improved"] is None
        assert comp["deltas"] == {}

    def test_falls_back_to_test_pr_auc(self) -> None:
        v_old = ModelVersion(
            version_id="v_old", created_at="2026-03-22", model_type="catboost",
            metrics={"pr_auc_test": 0.10},
        )
        v_new = ModelVersion(
            version_id="v_new", created_at="2026-03-23", model_type="catboost",
            metrics={"pr_auc_test": 0.12},
        )
        comp = compare_versions(v_new, v_old)
        assert comp["primary_metric"] == "pr_auc_test"
        assert comp["improved"] is True
