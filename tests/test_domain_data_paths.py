"""Tests for domain-isolated data path resolution.

Validates the DomainDataPaths abstraction and all shared path-resolution
helpers introduced for the domain isolation rollout (FP-5uo.1).
"""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import pytest

from src.domain_registry import (
    DOMAIN_REGISTRY,
    DomainDataPaths,
    add_domain_args,
    get_domain,
    resolve_pipeline_paths,
    resolve_script_paths,
)

# ---------------------------------------------------------------------------
# DomainDataPaths dataclass
# ---------------------------------------------------------------------------


class TestDomainDataPaths:
    """Tests for the DomainDataPaths frozen dataclass."""

    def test_resolve_data_paths_convention(self) -> None:
        """Convention derives data/{domain_id}/ subtree."""
        domain = get_domain("psc")
        root = Path("/project")
        paths = domain.resolve_data_paths(root)
        assert paths.base == root / "data" / "psc"
        assert paths.ingest == root / "data" / "psc" / "ingest"
        assert paths.graphs == root / "data" / "psc" / "graphs"
        assert paths.out == root / "data" / "psc" / "out"

    def test_all_paths_under_base(self) -> None:
        """Every resolved path sits under the domain base."""
        domain = get_domain("psc")
        paths = domain.resolve_data_paths(Path("/project"))
        warnings = paths.validate_paths()
        assert len(warnings) == 0

    def test_ensure_dirs_creates_tree(self, tmp_path: Path) -> None:
        """ensure_dirs() creates every directory in the tree."""
        domain = get_domain("crispr")
        paths = domain.resolve_data_paths(tmp_path)
        paths.ensure_dirs()
        assert paths.ingest.exists()
        assert paths.raw.exists()
        assert paths.slices.exists()
        assert paths.graphs.exists()
        assert paths.out.exists()
        assert paths.lineage_tracking.exists()
        assert paths.front_aggregation.exists()
        assert paths.cache_cum.exists()
        assert paths.cache_lineage.exists()
        assert paths.cache_coupling.exists()
        assert paths.assessments.exists()
        assert paths.experiments.exists()
        assert paths.archive.exists()

    def test_frozen_immutable(self) -> None:
        """Frozen dataclass rejects attribute assignment."""
        domain = get_domain("psc")
        paths = domain.resolve_data_paths(Path("/project"))
        with pytest.raises(AttributeError):
            paths.ingest = Path("/other")  # type: ignore[misc]

    def test_to_dict_serialization(self) -> None:
        """to_dict() returns string values, excludes domain_id."""
        domain = get_domain("psc")
        paths = domain.resolve_data_paths(Path("/project"))
        d = paths.to_dict()
        assert isinstance(d, dict)
        assert "domain_id" not in d
        assert "ingest" in d
        assert isinstance(d["ingest"], str)
        assert "base" in d

    def test_crispr_uses_crispr_subdirectory(self) -> None:
        """CRISPR domain uses data/crispr/ subtree."""
        domain = get_domain("crispr")
        root = Path("/project")
        paths = domain.resolve_data_paths(root)
        assert "crispr" in str(paths.base)
        assert paths.lineage_tracking == (
            root / "data" / "crispr" / "out" / "02_lineage_tracking"
        )

    def test_validate_paths_detects_escape(self) -> None:
        """validate_paths() flags paths outside the domain base."""
        paths = DomainDataPaths(
            domain_id="test",
            base=Path("/project/data/test"),
            ingest=Path("/other/ingest"),  # outside base
            raw=Path("/project/data/test/ingest/raw"),
            slices=Path("/project/data/test/ingest/slices"),
            graphs=Path("/project/data/test/graphs"),
            out=Path("/project/data/test/out"),
            lineage_tracking=Path("/project/data/test/out/02_lineage_tracking"),
            front_aggregation=Path("/project/data/test/out/04_front_aggregation"),
            cache_cum=Path("/project/data/test/out/cache_cum"),
            cache_lineage=Path("/project/data/test/out/cache_lineage"),
            cache_coupling=Path("/project/data/test/out/cache_coupling"),
            assessments=Path("/project/data/test/out/assessments"),
            experiments=Path("/project/data/test/out/experiments"),
            archive=Path("/project/data/test/archive"),
        )
        warnings = paths.validate_paths()
        assert len(warnings) == 1
        assert "ingest" in warnings[0]
        assert "outside domain base" in warnings[0]

    def test_all_registered_domains_resolve(self) -> None:
        """Every domain in the registry produces valid paths."""
        root = Path("/project")
        for domain_id, config in DOMAIN_REGISTRY.items():
            paths = config.resolve_data_paths(root)
            assert paths.domain_id == domain_id
            assert len(paths.validate_paths()) == 0

    def test_subdirectory_relationships(self) -> None:
        """Ingest subdirs are under ingest; out subdirs are under out."""
        paths = get_domain("psc").resolve_data_paths(Path("/project"))
        # raw and slices under ingest
        assert str(paths.raw).startswith(str(paths.ingest))
        assert str(paths.slices).startswith(str(paths.ingest))
        # output subdirs under out
        assert str(paths.lineage_tracking).startswith(str(paths.out))
        assert str(paths.cache_cum).startswith(str(paths.out))
        assert str(paths.assessments).startswith(str(paths.out))
        assert str(paths.experiments).startswith(str(paths.out))

    def test_field_count(self) -> None:
        """DomainDataPaths has exactly the expected number of fields."""
        # 1 string (domain_id) + 14 Paths = 15 fields total
        assert len(DomainDataPaths.__dataclass_fields__) == 15


# ---------------------------------------------------------------------------
# resolve_pipeline_paths (run.py helper)
# ---------------------------------------------------------------------------


class TestResolvePipelinePaths:
    """Tests for the run.py path resolution helper."""

    def test_domain_only_derives_all_paths(self) -> None:
        """With domain only, all paths come from convention."""
        root = Path("/project")
        paths = resolve_pipeline_paths("psc", None, None, None, None, root)
        assert paths is not None
        assert paths.ingest == root / "data" / "psc" / "ingest"
        assert paths.graphs == root / "data" / "psc" / "graphs"
        assert paths.out == root / "data" / "psc" / "out"

    def test_cli_ingest_override(self) -> None:
        """Explicit --ingest-dir overrides convention, cascading to raw/slices."""
        root = Path("/project")
        paths = resolve_pipeline_paths(
            "psc", "/custom/ingest", None, None, None, root
        )
        assert paths is not None
        assert paths.ingest == Path("/custom/ingest")
        assert paths.raw == Path("/custom/ingest/raw")
        assert paths.slices == Path("/custom/ingest/slices")
        # Non-overridden paths stay at convention
        assert paths.graphs == root / "data" / "psc" / "graphs"

    def test_cli_graphs_override(self) -> None:
        """Explicit --graphs-dir overrides convention."""
        root = Path("/project")
        paths = resolve_pipeline_paths(
            "psc", None, "/fast/graphs", None, None, root
        )
        assert paths is not None
        assert paths.graphs == Path("/fast/graphs")
        assert paths.ingest == root / "data" / "psc" / "ingest"

    def test_cli_outdir_override(self) -> None:
        """Explicit --outdir overrides convention and cascades sub-outputs."""
        root = Path("/project")
        paths = resolve_pipeline_paths(
            "psc", None, None, "/custom/out", None, root
        )
        assert paths is not None
        assert paths.out == Path("/custom/out")
        assert paths.assessments == Path("/custom/out/assessments")
        assert paths.lineage_tracking == Path("/custom/out/02_lineage_tracking")
        assert paths.cache_coupling == Path("/custom/out/cache_coupling")

    def test_cli_coupling_cache_override(self) -> None:
        """Explicit --coupling-cache-dir overrides even if outdir also set."""
        root = Path("/project")
        paths = resolve_pipeline_paths(
            "psc", None, None, "/custom/out", "/special/coupling", root
        )
        assert paths is not None
        # coupling_cache specifically overridden, not derived from outdir
        assert paths.cache_coupling == Path("/special/coupling")
        # other out subdirs derived from outdir override
        assert paths.assessments == Path("/custom/out/assessments")

    def test_no_domain_returns_none(self) -> None:
        """Without --domain, returns None for explicit-path fallback."""
        paths = resolve_pipeline_paths(
            None, None, None, None, None, Path("/project")
        )
        assert paths is None

    def test_invalid_domain_raises(self) -> None:
        """Unknown domain raises ValueError."""
        with pytest.raises(ValueError, match="Unknown domain"):
            resolve_pipeline_paths(
                "nonexistent", None, None, None, None, Path("/p")
            )


# ---------------------------------------------------------------------------
# resolve_script_paths (per-script helper)
# ---------------------------------------------------------------------------


class TestResolveScriptPaths:
    """Tests for the per-script domain resolution helper."""

    def test_with_domain(self) -> None:
        """Returns DomainDataPaths when --domain is set."""
        args = Namespace(domain="psc")
        root = Path("/project")
        paths = resolve_script_paths(args, root)
        assert paths is not None
        assert paths.domain_id == "psc"
        assert paths.ingest == root / "data" / "psc" / "ingest"

    def test_without_domain(self) -> None:
        """Returns None when --domain is None."""
        args = Namespace(domain=None)
        paths = resolve_script_paths(args, Path("/project"))
        assert paths is None

    def test_missing_domain_attr(self) -> None:
        """Returns None when namespace lacks domain attribute entirely."""
        args = Namespace()
        paths = resolve_script_paths(args, Path("/project"))
        assert paths is None

    def test_crispr_domain(self) -> None:
        """CRISPR domain resolves to crispr subtree."""
        args = Namespace(domain="crispr")
        paths = resolve_script_paths(args, Path("/project"))
        assert paths is not None
        assert paths.domain_id == "crispr"
        assert "crispr" in str(paths.base)


# ---------------------------------------------------------------------------
# add_domain_args
# ---------------------------------------------------------------------------


class TestAddDomainArgs:
    """Tests for the argparse helper."""

    def test_adds_domain_argument(self) -> None:
        """Parser gains --domain with correct choices."""
        import argparse

        parser = argparse.ArgumentParser()
        add_domain_args(parser)
        # Should parse valid domain
        args = parser.parse_args(["--domain", "psc"])
        assert args.domain == "psc"

    def test_domain_default_is_none(self) -> None:
        """--domain defaults to None when omitted."""
        import argparse

        parser = argparse.ArgumentParser()
        add_domain_args(parser)
        args = parser.parse_args([])
        assert args.domain is None

    def test_invalid_domain_rejected(self) -> None:
        """Argparse rejects unregistered domain choices."""
        import argparse

        parser = argparse.ArgumentParser()
        add_domain_args(parser)
        with pytest.raises(SystemExit):
            parser.parse_args(["--domain", "bogus"])


# ---------------------------------------------------------------------------
# outdir_suffix removal verification
# ---------------------------------------------------------------------------


class TestOutdirSuffixRemoved:
    """Verify that outdir_suffix has been removed from DomainConfig."""

    def test_no_outdir_suffix_field(self) -> None:
        """DomainConfig no longer has outdir_suffix."""
        from src.domain_registry import DomainConfig

        assert "outdir_suffix" not in DomainConfig.__dataclass_fields__

    def test_registry_entries_have_no_suffix(self) -> None:
        """No registry entry carries outdir_suffix."""
        for config in DOMAIN_REGISTRY.values():
            assert not hasattr(config, "outdir_suffix")
