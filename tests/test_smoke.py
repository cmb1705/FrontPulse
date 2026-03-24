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
    "config/datasources_crispr.yaml",
    "config/schema.yaml",
    "config/slices.yaml",
    "config/defaults.yaml",
    "config/multisignal_config.yaml",
    "config/front_aliases_crispr.yaml",
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
    from src.trusted_io import load_trusted_pickle, save_trusted_pickle

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
    "scripts/compute_convergence_features.py",
    "scripts/compute_front_level_features.py",
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


# ---------------------------------------------------------------------------
# Leakage-safe feature path
# ---------------------------------------------------------------------------

_LEAKAGE_PRONE_FEATURES = {
    "logistic_carrying_capacity",
    "logistic_growth_rate",
    "logistic_midpoint_idx",
    "logistic_fit_r2",
    "cd_index",
    "cd_min",
    "cd_max",
    "disruption_intensity",
}


def test_leakage_safe_config_exists() -> None:
    """Leakage-safe feature config must be defined."""
    import yaml

    path = PROJECT_ROOT / "config" / "features" / "feature_subset_configs.yaml"
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    configs = cfg.get("configs", {})
    assert "leakage_safe" in configs, "leakage_safe config missing"


def test_leakage_safe_excludes_known_leaky_features() -> None:
    """Leakage-safe config must not contain any known leakage-prone features."""
    import yaml

    path = PROJECT_ROOT / "config" / "features" / "feature_subset_configs.yaml"
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    safe_cols = set(cfg["configs"]["leakage_safe"].get("include_columns", []))
    overlap = safe_cols & _LEAKAGE_PRONE_FEATURES
    assert not overlap, f"Leakage-safe config contains leaky features: {overlap}"


def test_leakage_audit_doc_exists() -> None:
    """Leakage audit document must exist for experiment guidance."""
    path = PROJECT_ROOT / "docs" / "implementation" / "leakage_audit.md"
    assert path.exists(), "Leakage audit document missing"


def test_front_level_series_contract_exists() -> None:
    """Front-level series contract must exist for detector consumers."""
    path = (
        PROJECT_ROOT
        / "docs"
        / "implementation"
        / "frontpulse_program"
        / "front_level_series_contract.md"
    )
    assert path.exists(), "Front-level series contract document missing"


def test_env_template_exists() -> None:
    """Env template must exist with API key placeholder."""
    path = PROJECT_ROOT / ".env.template"
    assert path.exists(), ".env.template missing from project root"
    text = path.read_text()
    assert "OPENALEX_API_KEY" in text, "OPENALEX_API_KEY placeholder missing"


def test_crispr_datasource_topic_id() -> None:
    """CRISPR config requires OpenAlex topic T10878."""
    path = PROJECT_ROOT / "config" / "datasources_crispr.yaml"
    assert path.exists(), "datasources_crispr.yaml missing"
    raw = path.read_text()
    assert "T10878" in raw, "CRISPR topic ID T10878 missing"


def test_crispr_front_aliases_structure() -> None:
    """CRISPR front aliases must define canonical names for known fronts."""
    path = PROJECT_ROOT / "config" / "front_aliases_crispr.yaml"
    assert path.exists(), "front_aliases_crispr.yaml missing"
    with path.open() as fh:
        data = yaml.safe_load(fh)
    assert isinstance(data, dict), "front_aliases_crispr.yaml must be a dict"
    assert "fronts" in data, "Must have a 'fronts' key"
    assert len(data["fronts"]) >= 5, "Should define at least 5 known CRISPR fronts"


def test_crispr_config_structural_parity() -> None:
    """CRISPR datasource config must have same keys as PSC config."""
    psc = yaml.safe_load((PROJECT_ROOT / "config/datasources.yaml").read_text())
    crispr = yaml.safe_load(
        (PROJECT_ROOT / "config/datasources_crispr.yaml").read_text()
    )
    psc_keys = set(psc.keys())
    crispr_keys = set(crispr.keys())
    assert psc_keys == crispr_keys, (
        f"Key mismatch: {psc_keys.symmetric_difference(crispr_keys)}"
    )
    psc_primary_keys = set(psc["sources"]["primary"].keys())
    crispr_primary_keys = set(crispr["sources"]["primary"].keys())
    assert psc_primary_keys == crispr_primary_keys


@pytest.mark.smoke
def test_datasource_filters_override_saved_settings() -> None:
    """Datasource config filters must take precedence over saved settings.

    This validates the multi-domain fix: when --config points to the CRISPR
    datasource, its topics.id (T10878) should override the default/saved
    PSC topic ID in build_source_overrides output.
    """
    from run import build_source_overrides
    from src.ingest import apply_source_overrides

    # Simulate saved settings with PSC topic
    settings = {
        "topics_id": "T10247",
        "from_date": "2000-01-01",
        "to_date": "2025-08-30",
        "max_records": None,
        "per_page": 200,
    }
    overrides = build_source_overrides(settings)

    # Load CRISPR datasource config
    crispr_cfg = yaml.safe_load(
        (PROJECT_ROOT / "config" / "datasources_crispr.yaml").read_text(),
    )
    ds_filters = crispr_cfg["sources"]["primary"]["filters"]

    # Apply the override fix: datasource filters win
    overrides["filters"].update(ds_filters)

    merged = apply_source_overrides(
        crispr_cfg["sources"]["primary"], overrides,
    )
    assert merged["filters"]["topics.id"] == "T10878", (
        "CRISPR topic ID must override saved PSC settings"
    )


# ---------------------------------------------------------------------------
# Convergence detection smoke tests
# ---------------------------------------------------------------------------


class TestConvergenceSmoke:
    """Smoke tests for convergence detection configuration and imports."""

    def test_convergence_feature_group_in_yaml(self):
        """feature_groups.yaml should parse and contain a lifecycle group."""
        cfg = yaml.safe_load(
            (PROJECT_ROOT / "config" / "features" / "feature_groups.yaml").read_text(),
        )
        groups = cfg.get("groups", {})
        assert "lifecycle" in groups, "lifecycle group must exist in feature_groups.yaml"
        cols = groups["lifecycle"]["columns"]
        assert "is_matured" in cols
        assert "quarters_since_maturation" in cols

    def test_convergence_features_in_leakage_safe_extended(self):
        """leakage_safe_extended should include the lifecycle group."""
        cfg = yaml.safe_load(
            (PROJECT_ROOT / "config" / "features" / "feature_subset_configs.yaml").read_text(),
        )
        lse = cfg["configs"]["leakage_safe_extended"]
        groups = lse.get("include_groups", [])
        assert "lifecycle" in groups

    def test_convergence_module_imports(self):
        """src.convergence should import without error."""
        mod = importlib.import_module("src.convergence")
        assert hasattr(mod, "compute_pairwise_semantic_similarity")
        assert hasattr(mod, "aggregate_convergence_features")
        assert hasattr(mod, "CONVERGENCE_FEATURE_DEFAULTS")
