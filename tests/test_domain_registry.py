"""Tests for research domain registry and selector."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.domain_registry import (
    DOMAIN_REGISTRY,
    DomainConfig,
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
