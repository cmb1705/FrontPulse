"""Research domain registry for multi-domain pipeline configuration.

Maps domain identifiers (e.g., 'psc', 'crispr') to their corresponding
configuration files, data directories, and metadata.  Prevents accidental
cross-domain data mixing by making domain selection explicit at pipeline
startup.

Adding a new domain:
1. Create ``config/datasources_<domain>.yaml`` and optionally
   ``config/front_aliases_<domain>.yaml``.
2. Add an entry to ``DOMAIN_REGISTRY`` below.
3. Run ``python -c "from src.domain_registry import list_domains; print(list_domains())"``
   to verify.

Data directory convention:
    data/{domain_id}/ingest/     -- ingested parquets, raw chunks, slices
    data/{domain_id}/graphs/     -- citation graph artifacts
    data/{domain_id}/out/        -- pipeline outputs, caches, experiments
    data/{domain_id}/archive/    -- timestamped run snapshots
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data path abstraction
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DomainDataPaths:
    """All data directory paths for a single research domain.

    Derived from convention: ``data/{domain_id}/{ingest,graphs,out,archive}/``.
    Frozen to prevent accidental mutation after resolution.

    Attributes:
        domain_id: Identifier for the research domain.
        base: Root of the domain data tree (``data/{domain_id}/``).
        ingest: Flattened ingest parquet and supporting files.
        raw: Raw API response chunks (ndjson.gz).
        slices: Quarterly parquet slices.
        graphs: Citation graph artifacts (NetworkX binary files).
        out: Top-level output directory.
        lineage_tracking: Lineage registry, timeseries, features, labels.
        front_aggregation: Field-level aggregated metrics.
        cache_cum: Cumulative partition and core caches.
        cache_lineage: Lineage-level reference data cache.
        cache_coupling: Bibliographic coupling intermediates.
        assessments: Assessment history, horizon estimates, reports.
        experiments: MSD, Optuna, embedding, and mapping outputs.
        archive: Timestamped run snapshots.
    """

    domain_id: str
    base: Path
    ingest: Path
    raw: Path
    slices: Path
    graphs: Path
    out: Path
    lineage_tracking: Path
    front_aggregation: Path
    cache_cum: Path
    cache_lineage: Path
    cache_coupling: Path
    assessments: Path
    experiments: Path
    archive: Path

    def ensure_dirs(self) -> None:
        """Create all directories in the tree if they don't exist."""
        for field_name in self.__dataclass_fields__:
            val = getattr(self, field_name)
            if isinstance(val, Path):
                val.mkdir(parents=True, exist_ok=True)

    def validate_paths(self) -> list[str]:
        """Check that all paths are under the domain base directory.

        Returns:
            List of warning strings for paths outside the base.
            Empty list means all paths are properly contained.
        """
        warnings: list[str] = []
        for field_name in self.__dataclass_fields__:
            val = getattr(self, field_name)
            if isinstance(val, Path) and field_name != "base":
                try:
                    val.relative_to(self.base)
                except ValueError:
                    warnings.append(
                        f"{field_name}={val} is outside domain base {self.base}"
                    )
        return warnings

    def to_dict(self) -> dict[str, str]:
        """Serialize path fields to string dict for manifest storage.

        Returns:
            Dictionary mapping field names to string paths.
            The ``domain_id`` field is excluded.
        """
        return {
            k: str(v) for k, v in self.__dict__.items()
            if k != "domain_id"
        }


# ---------------------------------------------------------------------------
# Domain configuration
# ---------------------------------------------------------------------------


@dataclass
class DomainConfig:
    """Configuration for a research domain."""

    domain_id: str
    display_name: str
    datasources_config: str
    front_aliases_config: str | None
    description: str

    def resolve_paths(self, project_root: Path) -> dict[str, Path | None]:
        """Resolve configuration file paths relative to project root.

        Args:
            project_root: Repository root directory.

        Returns:
            Dictionary with resolved paths (None if file doesn't exist).
        """
        ds = project_root / self.datasources_config
        fa = None
        if self.front_aliases_config:
            fa_path = project_root / self.front_aliases_config
            fa = fa_path if fa_path.exists() else None

        return {
            "datasources": ds if ds.exists() else None,
            "front_aliases": fa,
        }

    def resolve_data_paths(self, project_root: Path) -> DomainDataPaths:
        """Derive all data directory paths from domain_id convention.

        Uses the canonical layout ``data/{domain_id}/`` to construct
        every directory the pipeline reads from or writes to.

        Args:
            project_root: Repository root directory.

        Returns:
            Frozen DomainDataPaths with all directories resolved.
        """
        base = project_root / "data" / self.domain_id
        return DomainDataPaths(
            domain_id=self.domain_id,
            base=base,
            ingest=base / "ingest",
            raw=base / "ingest" / "raw",
            slices=base / "ingest" / "slices",
            graphs=base / "graphs",
            out=base / "out",
            lineage_tracking=base / "out" / "02_lineage_tracking",
            front_aggregation=base / "out" / "04_front_aggregation",
            cache_cum=base / "out" / "cache_cum",
            cache_lineage=base / "out" / "cache_lineage",
            cache_coupling=base / "out" / "cache_coupling",
            assessments=base / "out" / "assessments",
            experiments=base / "out" / "experiments",
            archive=base / "archive",
        )


# ---------------------------------------------------------------------------
# Domain registry
# ---------------------------------------------------------------------------

DOMAIN_REGISTRY: dict[str, DomainConfig] = {
    "psc": DomainConfig(
        domain_id="psc",
        display_name="Polymer Solar Cells (PSC)",
        datasources_config="config/datasources.yaml",
        front_aliases_config="config/front_aliases.yaml",
        description="Baseline domain: polymer solar cell research fronts (T10247)",
    ),
    "crispr": DomainConfig(
        domain_id="crispr",
        display_name="CRISPR and Genetic Engineering",
        datasources_config="config/datasources_crispr.yaml",
        front_aliases_config="config/front_aliases_crispr.yaml",
        description="Second domain: CRISPR gene editing research fronts (T10878)",
    ),
}


def get_domain(domain_id: str) -> DomainConfig:
    """Look up a domain by its identifier.

    Args:
        domain_id: Case-insensitive domain identifier.

    Returns:
        DomainConfig for the requested domain.

    Raises:
        ValueError: If the domain_id is not found in the registry.
    """
    key = domain_id.lower()
    if key not in DOMAIN_REGISTRY:
        available = ", ".join(sorted(DOMAIN_REGISTRY.keys()))
        raise ValueError(
            f"Unknown domain '{domain_id}'. Available: {available}"
        )
    return DOMAIN_REGISTRY[key]


def list_domains() -> list[dict[str, Any]]:
    """List all registered domains with their metadata.

    Returns:
        List of domain info dictionaries.
    """
    return [
        {
            "domain_id": d.domain_id,
            "display_name": d.display_name,
            "description": d.description,
            "datasources_config": d.datasources_config,
        }
        for d in DOMAIN_REGISTRY.values()
    ]


def resolve_domain_args(
    domain_id: str | None,
    config_path: str | None,
    project_root: Path,
) -> str:
    """Resolve the datasources config path from domain or explicit config.

    If ``domain_id`` is provided, it takes precedence and returns the
    domain's datasources config path.  Otherwise falls back to the
    explicit ``config_path``.

    Args:
        domain_id: Optional domain identifier (e.g., 'psc', 'crispr').
        config_path: Explicit --config path (fallback).
        project_root: Repository root for path resolution.

    Returns:
        Resolved config file path as a string.

    Raises:
        ValueError: If domain_id is invalid or config file doesn't exist.
    """
    if domain_id is not None:
        domain = get_domain(domain_id)
        paths = domain.resolve_paths(project_root)
        ds_path = paths["datasources"]
        if ds_path is None:
            raise ValueError(
                f"Datasources config for domain '{domain_id}' not found:"
                f" {domain.datasources_config}"
            )
        logger.info(
            "Domain '%s' selected: using %s",
            domain.display_name, domain.datasources_config,
        )
        return domain.datasources_config

    if config_path is not None:
        return config_path

    raise ValueError("Either --domain or --config must be specified")


# ---------------------------------------------------------------------------
# Script integration helpers
# ---------------------------------------------------------------------------


def add_domain_args(parser: argparse.ArgumentParser) -> None:
    """Add ``--domain`` argument to any script's argument parser.

    Scripts call this in their argparse setup to gain domain-aware
    path resolution.  After parsing, pass the namespace to
    :func:`resolve_script_paths` to obtain a :class:`DomainDataPaths`
    instance (or ``None`` when ``--domain`` is omitted).

    Args:
        parser: The script's argument parser.
    """
    parser.add_argument(
        "--domain",
        default=None,
        choices=sorted(DOMAIN_REGISTRY),
        help="Research domain (derives all data paths from convention)",
    )


def resolve_script_paths(
    args: argparse.Namespace,
    project_root: Path,
) -> DomainDataPaths | None:
    """Resolve data paths for scripts that accept ``--domain``.

    Returns :class:`DomainDataPaths` if ``--domain`` is set, ``None``
    otherwise.  When ``None`` is returned, scripts fall back to their
    explicit CLI arguments (backward-compatible behavior).

    Args:
        args: Parsed argparse namespace (must have ``domain`` attribute
              if :func:`add_domain_args` was called).
        project_root: Repository root directory.

    Returns:
        DomainDataPaths or None.
    """
    domain_id = getattr(args, "domain", None)
    if domain_id is None:
        return None
    domain = get_domain(domain_id)
    paths = domain.resolve_data_paths(project_root)
    warnings = paths.validate_paths()
    for w in warnings:
        logger.warning("Domain path validation: %s", w)
    return paths


def resolve_pipeline_paths(
    domain_id: str | None,
    cli_ingest_dir: str | None,
    cli_graphs_dir: str | None,
    cli_outdir: str | None,
    cli_coupling_cache: str | None,
    project_root: Path,
) -> DomainDataPaths | None:
    """Resolve data paths for run.py with optional CLI overrides.

    If ``domain_id`` is set, derives all paths from the domain
    convention.  Individual CLI arguments override specific directories
    when provided (not None).

    Returns ``None`` if ``domain_id`` is ``None`` -- the caller must
    use explicit paths in that case.

    Args:
        domain_id: Optional domain identifier.
        cli_ingest_dir: Explicit ``--ingest-dir`` value (or None).
        cli_graphs_dir: Explicit ``--graphs-dir`` value (or None).
        cli_outdir: Explicit ``--outdir`` value (or None).
        cli_coupling_cache: Explicit ``--coupling-cache-dir`` value (or None).
        project_root: Repository root directory.

    Returns:
        DomainDataPaths with any CLI overrides applied, or None.

    Raises:
        ValueError: If ``domain_id`` is provided but not recognized.
    """
    if domain_id is None:
        return None

    domain = get_domain(domain_id)
    paths = domain.resolve_data_paths(project_root)

    # Collect CLI overrides that differ from convention
    overrides: dict[str, Any] = {}
    if cli_ingest_dir is not None:
        ingest = Path(cli_ingest_dir)
        overrides["ingest"] = ingest
        overrides["raw"] = ingest / "raw"
        overrides["slices"] = ingest / "slices"
    if cli_graphs_dir is not None:
        overrides["graphs"] = Path(cli_graphs_dir)
    if cli_outdir is not None:
        out = Path(cli_outdir)
        overrides["out"] = out
        overrides["lineage_tracking"] = out / "02_lineage_tracking"
        overrides["front_aggregation"] = out / "04_front_aggregation"
        overrides["cache_cum"] = out / "cache_cum"
        overrides["cache_lineage"] = out / "cache_lineage"
        overrides["cache_coupling"] = out / "cache_coupling"
        overrides["assessments"] = out / "assessments"
        overrides["experiments"] = out / "experiments"
    if cli_coupling_cache is not None:
        overrides["cache_coupling"] = Path(cli_coupling_cache)

    if overrides:
        # Rebuild frozen dataclass with overrides applied
        field_values = {k: getattr(paths, k) for k in paths.__dataclass_fields__}
        field_values.update(overrides)
        paths = DomainDataPaths(**field_values)

    warnings = paths.validate_paths()
    for w in warnings:
        logger.warning("Domain path validation: %s", w)
    return paths
