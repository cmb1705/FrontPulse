"""Research domain registry for multi-domain pipeline configuration.

Maps domain identifiers (e.g., 'psc', 'crispr') to their corresponding
configuration files, output directories, and metadata.  Prevents accidental
cross-domain data mixing by making domain selection explicit at pipeline
startup.

Adding a new domain:
1. Create ``config/datasources_<domain>.yaml`` and optionally
   ``config/front_aliases_<domain>.yaml``.
2. Add an entry to ``DOMAIN_REGISTRY`` below.
3. Run ``python -c "from src.domain_registry import list_domains; print(list_domains())"``
   to verify.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class DomainConfig:
    """Configuration for a research domain."""

    domain_id: str
    display_name: str
    datasources_config: str
    front_aliases_config: str | None
    description: str
    outdir_suffix: str = ""

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
        outdir_suffix="_crispr",
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
