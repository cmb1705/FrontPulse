"""Smoke tests for FrontPulse pipeline integrity.

These tests run without data or API access and are designed to catch
configuration drift, import breakage, and CLI wiring issues early.
Run the full suite with ``pytest tests/test_smoke.py -m smoke``.
"""

from __future__ import annotations

import ast
import importlib
import sys
from pathlib import Path

import pytest
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Ensure scripts/ is importable for CLI tests
if str(PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Markers
# ---------------------------------------------------------------------------

pytestmark = pytest.mark.smoke


# ---------------------------------------------------------------------------
# Config file parsing
# ---------------------------------------------------------------------------

_CONFIG_FILES = [
    "config/datasources.yaml",
    "config/schema.yaml",
    "config/slices.yaml",
    "config/defaults.yaml",
    "config/multisignal_config.yaml",
]


@pytest.mark.parametrize("rel_path", _CONFIG_FILES)
def test_config_yaml_parses(rel_path: str) -> None:
    """Every YAML config must parse without error."""
    path = PROJECT_ROOT / rel_path
    if not path.exists():
        pytest.skip(f"{rel_path} not present")
    with path.open() as fh:
        data = yaml.safe_load(fh)
    assert isinstance(data, dict), f"{rel_path} did not parse as a dict"


def test_datasource_topic_id() -> None:
    """PSC baseline requires OpenAlex topic T10247."""
    path = PROJECT_ROOT / "config" / "datasources.yaml"
    if not path.exists():
        pytest.skip("datasources.yaml not present")
    raw = path.read_text()
    assert "T10247" in raw, "PSC topic ID T10247 missing from datasources.yaml"


def test_schema_has_work_id_constraint() -> None:
    """Schema must enforce non-null on work_id."""
    path = PROJECT_ROOT / "config" / "schema.yaml"
    if not path.exists():
        pytest.skip("schema.yaml not present")
    with path.open() as fh:
        schema = yaml.safe_load(fh)
    non_null = schema.get("constraints", {}).get("non_null", [])
    assert "work_id" in non_null, "work_id must be non-null in schema"


# ---------------------------------------------------------------------------
# Core module imports
# ---------------------------------------------------------------------------

_CORE_MODULES = [
    "src.trusted_io",
    "src.logging_config",
    "src.validate",
    "src.transform",
    "src.config",
    "src.slicing",
]


@pytest.mark.parametrize("module_name", _CORE_MODULES)
def test_core_module_imports(module_name: str) -> None:
    """Core src modules must be importable."""
    try:
        importlib.import_module(module_name)
    except ImportError as exc:
        msg = str(exc)
        optional = ("torch", "igraph", "leidenalg", "transformers", "catboost")
        if any(pkg in msg for pkg in optional):
            pytest.skip(f"Optional dependency missing: {msg}")
        raise


# ---------------------------------------------------------------------------
# Trusted IO round-trip
# ---------------------------------------------------------------------------


def test_trusted_io_save_load_roundtrip(tmp_path: Path) -> None:
    """save/load round-trip through trusted_io."""
    from src.trusted_io import save_trusted_pickle, load_trusted_pickle

    obj = {"key": "value", "numbers": [1, 2, 3]}
    artifact = tmp_path / "test.pkl"

    save_trusted_pickle(obj, artifact, description="test", allow_external=True)
    loaded = load_trusted_pickle(artifact, description="test", allow_external=True)
    assert loaded == obj


def test_trusted_io_blocks_external_by_default(tmp_path: Path) -> None:
    """External paths must be rejected without allow_external."""
    from src.trusted_io import save_trusted_pickle

    with pytest.raises(ValueError, match="repository root"):
        save_trusted_pickle(
            {"x": 1},
            tmp_path / "bad.pkl",
            description="external",
        )


# ---------------------------------------------------------------------------
# Script syntax validation
# ---------------------------------------------------------------------------

_KEY_SCRIPTS = [
    "scripts/multi_signal_detector.py",
    "scripts/communities.py",
    "scripts/label_inflection_points.py",
    "scripts/compute_lineage_multisignal_features.py",
    "scripts/stage5_ensemble_mapping.py",
    "scripts/aggregate_lineages_to_fronts.py",
]


@pytest.mark.parametrize("rel_path", _KEY_SCRIPTS)
def test_script_syntax_valid(rel_path: str) -> None:
    """Key pipeline scripts must have valid Python syntax."""
    path = PROJECT_ROOT / rel_path
    if not path.exists():
        pytest.skip(f"{rel_path} not present")
    source = path.read_text(encoding="utf-8")
    ast.parse(source, filename=rel_path)


# ---------------------------------------------------------------------------
# Feature registry consistency
# ---------------------------------------------------------------------------


def test_feature_groups_yaml_parses() -> None:
    """Feature groups config must be valid YAML with a 'groups' key."""
    path = PROJECT_ROOT / "config" / "features" / "feature_groups.yaml"
    if not path.exists():
        pytest.skip("feature_groups.yaml not present")
    with path.open() as fh:
        data = yaml.safe_load(fh)
    assert "groups" in data, "feature_groups.yaml missing 'groups' key"
    assert "core" in data["groups"], "feature_groups.yaml missing 'core' group"


# ---------------------------------------------------------------------------
# Baseline documentation guards
# ---------------------------------------------------------------------------


def test_baseline_freeze_doc_exists() -> None:
    """PSC baseline freeze document must exist for downstream reproducibility."""
    path = PROJECT_ROOT / "docs" / "implementation" / "psc_baseline_freeze.md"
    assert path.exists(), "PSC baseline freeze document missing"


def test_artifact_persistence_policy_exists() -> None:
    """Artifact persistence policy must exist for trusted IO guidance."""
    path = PROJECT_ROOT / "docs" / "implementation" / "artifact_persistence_policy.md"
    assert path.exists(), "Artifact persistence policy missing"
