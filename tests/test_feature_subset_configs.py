from pathlib import Path
from typing import Any

import yaml
from utils.feature_registry import FeatureRegistry

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SUBSET_CONFIG = PROJECT_ROOT / "config" / "features" / "feature_subset_configs.yaml"
FEATURE_CONFIG = PROJECT_ROOT / "config" / "features" / "feature_groups.yaml"


def load_subset_configs():
    with SUBSET_CONFIG.open("r", encoding="utf-8") as fp:
        data = yaml.safe_load(fp) or {}
    return data.get("configs", {})


def subset_kwargs(definition: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "include_groups",
        "include_columns",
        "include_patterns",
        "exclude_groups",
        "exclude_columns",
        "exclude_patterns",
    ]
    return {key: definition.get(key) for key in keys}


def test_baseline_subset_contains_context_features():
    registry = FeatureRegistry(FEATURE_CONFIG)
    configs = load_subset_configs()
    baseline = configs["baseline"]
    features = registry.resolve_features(**subset_kwargs(baseline))
    assert "semantic_velocity" in features
    assert "author_influx_z" in features


def test_core_only_excludes_context_groups():
    registry = FeatureRegistry(FEATURE_CONFIG)
    configs = load_subset_configs()
    core_only = configs["core_only"]
    features = registry.resolve_features(**subset_kwargs(core_only))
    assert "author_influx_z" not in features
    assert "semantic_velocity" in features
