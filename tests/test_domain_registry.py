"""Tests for research domain registry and selector."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.domain_registry import (
    DOMAIN_REGISTRY,
    DomainConfig,
    DomainDataPaths,
    apply_domain_path_defaults,
    get_domain,
    list_domains,
    resolve_domain_args,
)

# ---------------------------------------------------------------------------
# Domain registry
# ---------------------------------------------------------------------------


class TestDomainRegistry:
    """Tests for the domain registry."""

    def test_psc_domain_exists(self) -> None:
        assert "psc" in DOMAIN_REGISTRY
        assert DOMAIN_REGISTRY["psc"].datasources_config == "config/datasources.yaml"

    def test_crispr_domain_exists(self) -> None:
        assert "crispr" in DOMAIN_REGISTRY
        assert "crispr" in DOMAIN_REGISTRY["crispr"].datasources_config

    def test_get_domain_case_insensitive(self) -> None:
        d1 = get_domain("psc")
        d2 = get_domain("PSC")
        assert d1.domain_id == d2.domain_id

    def test_get_unknown_domain_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown domain"):
            get_domain("nonexistent")

    def test_list_domains_returns_all(self) -> None:
        domains = list_domains()
        ids = {d["domain_id"] for d in domains}
        assert "psc" in ids
        assert "crispr" in ids


# ---------------------------------------------------------------------------
# Domain config
# ---------------------------------------------------------------------------


class TestDomainConfig:
    """Tests for DomainConfig dataclass."""

    def test_resolve_paths_existing(self, tmp_path: Path) -> None:
        # Create mock config files
        (tmp_path / "config").mkdir()
        (tmp_path / "config" / "datasources.yaml").write_text("test: true")
        (tmp_path / "config" / "front_aliases.yaml").write_text("test: true")

        dc = DomainConfig(
            domain_id="test",
            display_name="Test",
            datasources_config="config/datasources.yaml",
            front_aliases_config="config/front_aliases.yaml",
            description="test domain",
        )
        paths = dc.resolve_paths(tmp_path)
        assert paths["datasources"] is not None
        assert paths["front_aliases"] is not None

    def test_resolve_paths_missing_aliases(self, tmp_path: Path) -> None:
        (tmp_path / "config").mkdir()
        (tmp_path / "config" / "datasources.yaml").write_text("test: true")

        dc = DomainConfig(
            domain_id="test",
            display_name="Test",
            datasources_config="config/datasources.yaml",
            front_aliases_config="config/missing.yaml",
            description="test domain",
        )
        paths = dc.resolve_paths(tmp_path)
        assert paths["datasources"] is not None
        assert paths["front_aliases"] is None

    def test_resolve_paths_no_aliases_config(self, tmp_path: Path) -> None:
        (tmp_path / "config").mkdir()
        (tmp_path / "config" / "ds.yaml").write_text("test: true")

        dc = DomainConfig(
            domain_id="test",
            display_name="Test",
            datasources_config="config/ds.yaml",
            front_aliases_config=None,
            description="test domain",
        )
        paths = dc.resolve_paths(tmp_path)
        assert paths["front_aliases"] is None


# ---------------------------------------------------------------------------
# resolve_domain_args
# ---------------------------------------------------------------------------


class TestResolveDomainArgs:
    """Tests for domain argument resolution."""

    def test_domain_overrides_config(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        result = resolve_domain_args("psc", "some/other.yaml", project_root)
        assert result == "config/datasources.yaml"

    def test_falls_back_to_config(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        result = resolve_domain_args(None, "config/custom.yaml", project_root)
        assert result == "config/custom.yaml"

    def test_neither_domain_nor_config_raises(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        with pytest.raises(ValueError, match="must be specified"):
            resolve_domain_args(None, None, project_root)

    def test_invalid_domain_raises(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        with pytest.raises(ValueError, match="Unknown domain"):
            resolve_domain_args("bogus", None, project_root)

    def test_crispr_resolves(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        result = resolve_domain_args("crispr", None, project_root)
        assert "crispr" in result


# ---------------------------------------------------------------------------
# apply_domain_path_defaults
# ---------------------------------------------------------------------------

class TestApplyDomainPathDefaults:
    """Tests for the shared CLI path-defaulting helper."""

    @staticmethod
    def _make_paths(root: Path) -> DomainDataPaths:
        """Build a DomainDataPaths for testing."""
        domain = get_domain("psc")
        return domain.resolve_data_paths(root)

    def test_sets_unset_args_from_domain_paths(self, tmp_path: Path) -> None:
        """Unset args get domain-derived values."""
        import argparse

        args = argparse.Namespace(registry=None, slices_dir=None)
        paths = self._make_paths(tmp_path)
        apply_domain_path_defaults(args, paths, {
            "registry": ("lineage_tracking", "lineage_registry.json",
                         "data/out/02_lineage_tracking/lineage_registry.json"),
            "slices_dir": ("slices", "", "data/current_ingest/slices"),
        })
        assert args.registry == str(paths.lineage_tracking / "lineage_registry.json")
        assert args.slices_dir == str(paths.slices)

    def test_preserves_explicit_args(self, tmp_path: Path) -> None:
        """Explicitly set args are not overwritten."""
        import argparse

        args = argparse.Namespace(registry="/my/custom/path", slices_dir=None)
        paths = self._make_paths(tmp_path)
        apply_domain_path_defaults(args, paths, {
            "registry": ("lineage_tracking", "lineage_registry.json",
                         "data/out/02_lineage_tracking/lineage_registry.json"),
            "slices_dir": ("slices", "", "data/current_ingest/slices"),
        })
        assert args.registry == "/my/custom/path"
        assert args.slices_dir == str(paths.slices)

    def test_uses_fallback_when_no_domain(self) -> None:
        """When paths is None, fallback values are used."""
        import argparse

        args = argparse.Namespace(registry=None, slices_dir=None)
        apply_domain_path_defaults(args, None, {
            "registry": ("lineage_tracking", "lineage_registry.json",
                         "data/out/02_lineage_tracking/lineage_registry.json"),
            "slices_dir": ("slices", "", "data/current_ingest/slices"),
        })
        assert args.registry == "data/out/02_lineage_tracking/lineage_registry.json"
        assert args.slices_dir == "data/current_ingest/slices"

    def test_empty_sub_path(self, tmp_path: Path) -> None:
        """Empty sub_path uses the base attribute directly."""
        import argparse

        args = argparse.Namespace(out_dir=None)
        paths = self._make_paths(tmp_path)
        apply_domain_path_defaults(args, paths, {
            "out_dir": ("out", "", "data/out"),
        })
        assert args.out_dir == str(paths.out)

    def test_missing_arg_attribute_skipped(self, tmp_path: Path) -> None:
        """Args without the named attribute are skipped (no error)."""
        import argparse

        args = argparse.Namespace()  # no attributes at all
        paths = self._make_paths(tmp_path)
        # Should not raise
        apply_domain_path_defaults(args, paths, {
            "nonexistent": ("out", "", "data/out"),
        })

    def test_nested_sub_path(self, tmp_path: Path) -> None:
        """Sub-paths with multiple components resolve correctly."""
        import argparse

        args = argparse.Namespace(features=None)
        paths = self._make_paths(tmp_path)
        apply_domain_path_defaults(args, paths, {
            "features": ("lineage_tracking",
                         "lineage_multisignal_features.csv",
                         "data/out/02_lineage_tracking/lineage_multisignal_features.csv"),
        })
        expected = str(paths.lineage_tracking / "lineage_multisignal_features.csv")
        assert args.features == expected
