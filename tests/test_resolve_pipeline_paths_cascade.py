"""Integration tests for resolve_pipeline_paths() cascade override behavior.

Verifies that CLI overrides (--outdir, --ingest-dir, --graphs-dir,
--coupling-cache-dir) cascade correctly to all derived sub-paths, and
that non-overridden paths remain at their domain convention defaults.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.domain_registry import DomainDataPaths, resolve_pipeline_paths

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.integration
class TestDomainOnlyConvention:
    """Domain-only invocation (no CLI overrides) derives all paths from convention."""

    def test_base_is_data_domain(self) -> None:
        """Base directory follows data/{domain_id}/ convention."""
        root = Path("/project")
        paths = resolve_pipeline_paths("psc", None, None, None, None, root)
        assert paths is not None
        assert paths.base == root / "data" / "psc"

    def test_ingest_subtree(self) -> None:
        """Ingest, raw, and slices derive from convention."""
        root = Path("/project")
        paths = resolve_pipeline_paths("psc", None, None, None, None, root)
        assert paths is not None
        assert paths.ingest == root / "data" / "psc" / "ingest"
        assert paths.raw == root / "data" / "psc" / "ingest" / "raw"
        assert paths.slices == root / "data" / "psc" / "ingest" / "slices"

    def test_graphs_convention(self) -> None:
        """Graphs directory derives from convention."""
        root = Path("/project")
        paths = resolve_pipeline_paths("psc", None, None, None, None, root)
        assert paths is not None
        assert paths.graphs == root / "data" / "psc" / "graphs"

    def test_out_subtree(self) -> None:
        """All output sub-paths derive from data/{domain}/out/."""
        root = Path("/project")
        paths = resolve_pipeline_paths("psc", None, None, None, None, root)
        assert paths is not None
        out = root / "data" / "psc" / "out"
        assert paths.out == out
        assert paths.lineage_tracking == out / "02_lineage_tracking"
        assert paths.front_aggregation == out / "04_front_aggregation"
        assert paths.cache_cum == out / "cache_cum"
        assert paths.cache_lineage == out / "cache_lineage"
        assert paths.cache_coupling == out / "cache_coupling"
        assert paths.assessments == out / "assessments"
        assert paths.experiments == out / "experiments"

    def test_archive_convention(self) -> None:
        """Archive directory derives from convention."""
        root = Path("/project")
        paths = resolve_pipeline_paths("psc", None, None, None, None, root)
        assert paths is not None
        assert paths.archive == root / "data" / "psc" / "archive"

    def test_domain_id_preserved(self) -> None:
        """domain_id on the returned paths matches the input."""
        paths = resolve_pipeline_paths("crispr", None, None, None, None, Path("/r"))
        assert paths is not None
        assert paths.domain_id == "crispr"


@pytest.mark.integration
class TestOutdirCascade:
    """--outdir override cascades to all output sub-directories."""

    def test_out_is_overridden(self) -> None:
        """The out field itself takes the CLI value."""
        root = Path("/project")
        paths = resolve_pipeline_paths("psc", None, None, "/custom/out", None, root)
        assert paths is not None
        assert paths.out == Path("/custom/out")

    def test_lineage_tracking_cascades(self) -> None:
        """lineage_tracking derives from the overridden outdir."""
        paths = resolve_pipeline_paths("psc", None, None, "/custom/out", None, Path("/r"))
        assert paths is not None
        assert paths.lineage_tracking == Path("/custom/out/02_lineage_tracking")

    def test_front_aggregation_cascades(self) -> None:
        """front_aggregation derives from the overridden outdir."""
        paths = resolve_pipeline_paths("psc", None, None, "/custom/out", None, Path("/r"))
        assert paths is not None
        assert paths.front_aggregation == Path("/custom/out/04_front_aggregation")

    def test_cache_cum_cascades(self) -> None:
        """cache_cum derives from the overridden outdir."""
        paths = resolve_pipeline_paths("psc", None, None, "/custom/out", None, Path("/r"))
        assert paths is not None
        assert paths.cache_cum == Path("/custom/out/cache_cum")

    def test_cache_lineage_cascades(self) -> None:
        """cache_lineage derives from the overridden outdir."""
        paths = resolve_pipeline_paths("psc", None, None, "/custom/out", None, Path("/r"))
        assert paths is not None
        assert paths.cache_lineage == Path("/custom/out/cache_lineage")

    def test_cache_coupling_cascades(self) -> None:
        """cache_coupling derives from the overridden outdir."""
        paths = resolve_pipeline_paths("psc", None, None, "/custom/out", None, Path("/r"))
        assert paths is not None
        assert paths.cache_coupling == Path("/custom/out/cache_coupling")

    def test_assessments_cascades(self) -> None:
        """assessments derives from the overridden outdir."""
        paths = resolve_pipeline_paths("psc", None, None, "/custom/out", None, Path("/r"))
        assert paths is not None
        assert paths.assessments == Path("/custom/out/assessments")

    def test_experiments_cascades(self) -> None:
        """experiments derives from the overridden outdir."""
        paths = resolve_pipeline_paths("psc", None, None, "/custom/out", None, Path("/r"))
        assert paths is not None
        assert paths.experiments == Path("/custom/out/experiments")

    def test_non_out_paths_unchanged(self) -> None:
        """Ingest, graphs, base, and archive remain at convention when only outdir is set."""
        root = Path("/project")
        paths = resolve_pipeline_paths("psc", None, None, "/custom/out", None, root)
        assert paths is not None
        assert paths.base == root / "data" / "psc"
        assert paths.ingest == root / "data" / "psc" / "ingest"
        assert paths.raw == root / "data" / "psc" / "ingest" / "raw"
        assert paths.slices == root / "data" / "psc" / "ingest" / "slices"
        assert paths.graphs == root / "data" / "psc" / "graphs"
        assert paths.archive == root / "data" / "psc" / "archive"


@pytest.mark.integration
class TestIngestDirCascade:
    """--ingest-dir override cascades to raw and slices."""

    def test_ingest_is_overridden(self) -> None:
        """The ingest field takes the CLI value."""
        root = Path("/project")
        paths = resolve_pipeline_paths("psc", "/fast/ingest", None, None, None, root)
        assert paths is not None
        assert paths.ingest == Path("/fast/ingest")

    def test_raw_cascades_from_ingest(self) -> None:
        """raw derives from the overridden ingest directory."""
        paths = resolve_pipeline_paths("psc", "/fast/ingest", None, None, None, Path("/r"))
        assert paths is not None
        assert paths.raw == Path("/fast/ingest/raw")

    def test_slices_cascades_from_ingest(self) -> None:
        """slices derives from the overridden ingest directory."""
        paths = resolve_pipeline_paths("psc", "/fast/ingest", None, None, None, Path("/r"))
        assert paths is not None
        assert paths.slices == Path("/fast/ingest/slices")

    def test_non_ingest_paths_unchanged(self) -> None:
        """Output, graphs, base, and archive remain at convention."""
        root = Path("/project")
        paths = resolve_pipeline_paths("psc", "/fast/ingest", None, None, None, root)
        assert paths is not None
        assert paths.graphs == root / "data" / "psc" / "graphs"
        assert paths.out == root / "data" / "psc" / "out"
        assert paths.lineage_tracking == root / "data" / "psc" / "out" / "02_lineage_tracking"
        assert paths.archive == root / "data" / "psc" / "archive"


@pytest.mark.integration
class TestGraphsDirOverride:
    """--graphs-dir only affects the graphs path."""

    def test_graphs_is_overridden(self) -> None:
        """The graphs field takes the CLI value."""
        root = Path("/project")
        paths = resolve_pipeline_paths("psc", None, "/ssd/graphs", None, None, root)
        assert paths is not None
        assert paths.graphs == Path("/ssd/graphs")

    def test_all_other_paths_unchanged(self) -> None:
        """Ingest, output, and archive remain at convention."""
        root = Path("/project")
        paths = resolve_pipeline_paths("psc", None, "/ssd/graphs", None, None, root)
        assert paths is not None
        assert paths.ingest == root / "data" / "psc" / "ingest"
        assert paths.out == root / "data" / "psc" / "out"
        assert paths.lineage_tracking == root / "data" / "psc" / "out" / "02_lineage_tracking"
        assert paths.cache_coupling == root / "data" / "psc" / "out" / "cache_coupling"
        assert paths.archive == root / "data" / "psc" / "archive"


@pytest.mark.integration
class TestCouplingCachePrecedence:
    """--coupling-cache overrides --outdir for cache_coupling specifically."""

    def test_coupling_cache_overrides_outdir(self) -> None:
        """Explicit coupling-cache takes precedence over outdir-derived value."""
        root = Path("/project")
        paths = resolve_pipeline_paths(
            "psc", None, None, "/custom/out", "/special/coupling", root
        )
        assert paths is not None
        assert paths.cache_coupling == Path("/special/coupling")

    def test_other_out_subdirs_still_from_outdir(self) -> None:
        """Non-coupling output sub-dirs still derive from outdir when both set."""
        root = Path("/project")
        paths = resolve_pipeline_paths(
            "psc", None, None, "/custom/out", "/special/coupling", root
        )
        assert paths is not None
        assert paths.out == Path("/custom/out")
        assert paths.lineage_tracking == Path("/custom/out/02_lineage_tracking")
        assert paths.front_aggregation == Path("/custom/out/04_front_aggregation")
        assert paths.cache_cum == Path("/custom/out/cache_cum")
        assert paths.cache_lineage == Path("/custom/out/cache_lineage")
        assert paths.assessments == Path("/custom/out/assessments")
        assert paths.experiments == Path("/custom/out/experiments")

    def test_coupling_cache_alone_without_outdir(self) -> None:
        """coupling-cache override works when outdir is not set."""
        root = Path("/project")
        paths = resolve_pipeline_paths(
            "psc", None, None, None, "/special/coupling", root
        )
        assert paths is not None
        assert paths.cache_coupling == Path("/special/coupling")
        # Other out sub-dirs remain at convention
        assert paths.out == root / "data" / "psc" / "out"
        assert paths.cache_cum == root / "data" / "psc" / "out" / "cache_cum"


@pytest.mark.integration
class TestMultipleSimultaneousOverrides:
    """Multiple CLI overrides applied simultaneously."""

    def test_all_four_overrides(self) -> None:
        """All CLI override arguments set at once."""
        root = Path("/project")
        paths = resolve_pipeline_paths(
            "psc",
            "/nas/ingest",
            "/ssd/graphs",
            "/fast/out",
            "/ramdisk/coupling",
            root,
        )
        assert paths is not None
        # Ingest cascade
        assert paths.ingest == Path("/nas/ingest")
        assert paths.raw == Path("/nas/ingest/raw")
        assert paths.slices == Path("/nas/ingest/slices")
        # Graphs
        assert paths.graphs == Path("/ssd/graphs")
        # Outdir cascade
        assert paths.out == Path("/fast/out")
        assert paths.lineage_tracking == Path("/fast/out/02_lineage_tracking")
        assert paths.front_aggregation == Path("/fast/out/04_front_aggregation")
        assert paths.cache_cum == Path("/fast/out/cache_cum")
        assert paths.cache_lineage == Path("/fast/out/cache_lineage")
        assert paths.assessments == Path("/fast/out/assessments")
        assert paths.experiments == Path("/fast/out/experiments")
        # Coupling-cache precedence over outdir
        assert paths.cache_coupling == Path("/ramdisk/coupling")

    def test_outdir_plus_ingest(self) -> None:
        """outdir and ingest-dir set together, each cascade independently."""
        root = Path("/project")
        paths = resolve_pipeline_paths("psc", "/ext/ingest", None, "/ext/out", None, root)
        assert paths is not None
        # Ingest cascade
        assert paths.ingest == Path("/ext/ingest")
        assert paths.raw == Path("/ext/ingest/raw")
        assert paths.slices == Path("/ext/ingest/slices")
        # Outdir cascade
        assert paths.out == Path("/ext/out")
        assert paths.assessments == Path("/ext/out/assessments")
        assert paths.cache_coupling == Path("/ext/out/cache_coupling")
        # Graphs stays at convention
        assert paths.graphs == root / "data" / "psc" / "graphs"

    def test_outdir_plus_graphs(self) -> None:
        """outdir and graphs-dir set together."""
        root = Path("/project")
        paths = resolve_pipeline_paths("psc", None, "/ssd/graphs", "/fast/out", None, root)
        assert paths is not None
        assert paths.graphs == Path("/ssd/graphs")
        assert paths.out == Path("/fast/out")
        assert paths.experiments == Path("/fast/out/experiments")
        # Ingest stays at convention
        assert paths.ingest == root / "data" / "psc" / "ingest"

    def test_base_and_archive_never_overridden(self) -> None:
        """base and archive are never changed by any CLI override combination."""
        root = Path("/project")
        paths = resolve_pipeline_paths(
            "psc", "/a", "/b", "/c", "/d", root
        )
        assert paths is not None
        assert paths.base == root / "data" / "psc"
        assert paths.archive == root / "data" / "psc" / "archive"
        assert paths.domain_id == "psc"


@pytest.mark.integration
class TestNoneDomainReturnsNone:
    """None domain returns None without errors."""

    def test_none_domain_returns_none(self) -> None:
        """resolve_pipeline_paths returns None when domain_id is None."""
        result = resolve_pipeline_paths(None, None, None, None, None, Path("/project"))
        assert result is None

    def test_none_domain_ignores_cli_overrides(self) -> None:
        """CLI overrides are irrelevant when domain_id is None."""
        result = resolve_pipeline_paths(
            None, "/a", "/b", "/c", "/d", Path("/project")
        )
        assert result is None


@pytest.mark.integration
class TestReturnTypeInvariants:
    """Structural invariants on the returned DomainDataPaths."""

    def test_returns_frozen_dataclass(self) -> None:
        """Result is a frozen DomainDataPaths even after overrides."""
        paths = resolve_pipeline_paths("psc", None, None, "/custom/out", None, Path("/r"))
        assert paths is not None
        assert isinstance(paths, DomainDataPaths)
        with pytest.raises(AttributeError):
            paths.out = Path("/other")  # type: ignore[misc]

    def test_overridden_paths_are_path_objects(self) -> None:
        """All path fields are Path instances after override application."""
        paths = resolve_pipeline_paths(
            "psc", "/a", "/b", "/c", "/d", Path("/r")
        )
        assert paths is not None
        for field_name in DomainDataPaths.__dataclass_fields__:
            val = getattr(paths, field_name)
            if field_name != "domain_id":
                assert isinstance(val, Path), f"{field_name} should be Path, got {type(val)}"

    def test_crispr_domain_cascades_same_as_psc(self) -> None:
        """Override logic is domain-agnostic; CRISPR follows the same cascade."""
        root = Path("/project")
        paths = resolve_pipeline_paths("crispr", None, None, "/custom/out", None, root)
        assert paths is not None
        assert paths.domain_id == "crispr"
        assert paths.base == root / "data" / "crispr"
        assert paths.out == Path("/custom/out")
        assert paths.lineage_tracking == Path("/custom/out/02_lineage_tracking")
        # Ingest stays at crispr convention
        assert paths.ingest == root / "data" / "crispr" / "ingest"
