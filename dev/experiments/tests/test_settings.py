"""Tests for src/settings.py - Settings management."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.settings import DEFAULTS, load_settings, save_settings


def _isolate_settings_paths(monkeypatch: pytest.MonkeyPatch, legacy_path: Path) -> None:
    """Force tests to ignore any real local YAML settings file."""
    monkeypatch.setattr("src.settings.YAML_SETTINGS_PATH", legacy_path.parent / "settings.yaml")
    monkeypatch.setattr("src.settings.LEGACY_JSON_PATH", legacy_path)
    monkeypatch.setattr("src.settings.SETTINGS_PATH", legacy_path)


@pytest.mark.unit
class TestDefaults:
    """Test suite for DEFAULTS dictionary."""

    def test_defaults_is_dict(self):
        """Test that DEFAULTS is a dictionary."""
        assert isinstance(DEFAULTS, dict)

    def test_contains_required_keys(self):
        """Test that DEFAULTS contain required keys."""
        # Check for expected keys based on the codebase
        assert "topics_id" in DEFAULTS
        assert "from_date" in DEFAULTS
        assert "to_date" in DEFAULTS
        assert "max_records" in DEFAULTS
        assert "per_page" in DEFAULTS
        assert "graph_mode" in DEFAULTS
        assert "mailto" in DEFAULTS

    def test_defaults_are_valid_types(self):
        """Test that default values have appropriate types."""
        for _key, value in DEFAULTS.items():
            # Values should be JSON-serializable
            assert value is None or isinstance(value, (str, int, float, bool, dict, list))


@pytest.mark.unit
class TestLoadSettings:
    """Test suite for load_settings function."""

    def test_loads_existing_settings(self, sample_settings_json, monkeypatch):
        """Test that existing settings file is loaded correctly."""
        settings_path = Path(sample_settings_json)
        _isolate_settings_paths(monkeypatch, settings_path)

        settings = load_settings()

        assert isinstance(settings, dict)
        assert "mailto" in settings
        assert settings["mailto"] == "test@example.com"

    def test_returns_defaults_when_file_missing(self, temp_dir, monkeypatch):
        """Test that defaults are returned when settings file doesn't exist."""
        _isolate_settings_paths(monkeypatch, temp_dir / ".2yp_settings.json")

        settings = load_settings()

        assert isinstance(settings, dict)
        # Should return defaults
        assert set(DEFAULTS.keys()).issubset(set(settings.keys()))

    def test_merges_with_defaults(self, temp_dir, monkeypatch):
        """Test that loaded settings are merged with defaults."""
        # Create partial settings file
        settings_file = temp_dir / ".2yp_settings.json"
        partial_settings = {"mailto": "partial@example.com"}
        with open(settings_file, "w") as f:
            json.dump(partial_settings, f)

        _isolate_settings_paths(monkeypatch, settings_file)

        settings = load_settings()

        # Should have both partial settings and defaults
        assert settings["mailto"] == "partial@example.com"
        # Should also have keys from defaults
        for key in DEFAULTS:
            if key != "mailto":  # We overrode this one
                assert key in settings

    def test_handles_invalid_json(self, temp_dir, monkeypatch):
        """Test that invalid JSON file is handled gracefully."""
        settings_file = temp_dir / ".2yp_settings.json"
        settings_file.write_text("{invalid json content")

        _isolate_settings_paths(monkeypatch, settings_file)

        # Should return defaults instead of crashing
        settings = load_settings()

        assert isinstance(settings, dict)

    def test_handles_empty_file(self, temp_dir, monkeypatch):
        """Test that empty settings file is handled gracefully."""
        settings_file = temp_dir / ".2yp_settings.json"
        settings_file.write_text("")

        _isolate_settings_paths(monkeypatch, settings_file)

        settings = load_settings()

        assert isinstance(settings, dict)


@pytest.mark.unit
class TestSaveSettings:
    """Test suite for save_settings function."""

    def test_saves_settings_to_file(self, temp_dir, monkeypatch):
        """Test that settings are saved to JSON file."""
        settings_file = temp_dir / ".2yp_settings.json"
        _isolate_settings_paths(monkeypatch, settings_file)

        test_settings = {
            "mailto": "save_test@example.com",
            "filters": {"topics.id": "T12345"},
            "slice_mode": "quarterly"
        }

        save_settings(test_settings)

        assert settings_file.exists()

        with open(settings_file) as f:
            loaded = json.load(f)

        assert loaded == test_settings

    def test_overwrites_existing_file(self, temp_dir, monkeypatch):
        """Test that save_settings overwrites existing file."""
        settings_file = temp_dir / ".2yp_settings.json"
        _isolate_settings_paths(monkeypatch, settings_file)

        # Save initial settings
        initial = {"mailto": "initial@example.com"}
        save_settings(initial)

        # Save new settings
        updated = {"mailto": "updated@example.com", "new_key": "value"}
        save_settings(updated)

        with open(settings_file) as f:
            loaded = json.load(f)

        assert loaded == updated
        assert loaded["mailto"] == "updated@example.com"

    def test_creates_valid_json(self, temp_dir, monkeypatch):
        """Test that saved file is valid JSON."""
        settings_file = temp_dir / ".2yp_settings.json"
        _isolate_settings_paths(monkeypatch, settings_file)

        settings = {
            "string": "value",
            "number": 42,
            "boolean": True,
            "null": None,
            "dict": {"nested": "value"},
            "list": [1, 2, 3]
        }

        save_settings(settings)

        # Should not raise exception
        with open(settings_file) as f:
            loaded = json.load(f)

        assert loaded == settings

    def test_handles_empty_settings(self, temp_dir, monkeypatch):
        """Test that empty settings dict can be saved."""
        settings_file = temp_dir / ".2yp_settings.json"
        _isolate_settings_paths(monkeypatch, settings_file)

        save_settings({})

        assert settings_file.exists()

        with open(settings_file) as f:
            loaded = json.load(f)

        assert loaded == {}


@pytest.mark.integration
class TestSettingsWorkflow:
    """Integration tests for complete settings workflow."""

    def test_save_and_load_roundtrip(self, temp_dir, monkeypatch):
        """Test that settings survive save/load cycle."""
        settings_file = temp_dir / ".2yp_settings.json"
        _isolate_settings_paths(monkeypatch, settings_file)

        original = {
            "mailto": "roundtrip@example.com",
            "filters": {"topics.id": "T99999"},
            "slice_mode": "annual",
            "graph_mode": "delta"
        }

        save_settings(original)
        loaded = load_settings()

        # Should have all original keys plus any defaults
        for key, value in original.items():
            assert loaded[key] == value
