from pathlib import Path

import pytest
from utils.feature_registry import FeatureRegistry

PROJECT_ROOT = Path(__file__).resolve().parents[1]

FEATURE_CONFIG = PROJECT_ROOT / "config" / "features" / "feature_groups.yaml"


@pytest.fixture(scope="module")
def registry():
    return FeatureRegistry(FEATURE_CONFIG)


def test_core_group_contains_expected_features(registry: FeatureRegistry):
    core = registry.get_group("core")
    assert "semantic_velocity" in core
    assert "disruption_intensity" in core
    assert len(core) >= 20


def test_resolve_with_patterns(registry: FeatureRegistry):
    resolved = registry.resolve_features(
        include_groups=["core"],
        include_patterns=["field_*"],
        exclude_patterns=["field_quarter_*"],
    )
    assert any(col.startswith("field_total_new_works") for col in resolved)
    assert "field_quarter_of_year" not in resolved


def test_feature_metadata_exposed(registry: FeatureRegistry):
    meta = registry.describe_feature("novelty_rate")
    assert isinstance(meta, dict)
    assert meta.get("description")
    assert meta.get("source")
