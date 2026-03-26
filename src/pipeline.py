"""Programmatic pipeline interface for research front monitoring.

This module provides a clean API for running the 2YP pipeline programmatically,
enabling use in Jupyter notebooks, automated scripts, and testing frameworks.

The Pipeline class encapsulates all phases from run.py:
- Ingest (with caching and raw snapshot support)
- Slicing (temporal and categorical)
- Graph building (annual, delta, cumulative with coupling)
- Community detection (Leiden with alignment)
- Manifest and report generation

Example Usage:
    ```python
    from src.pipeline import Pipeline

    # Basic usage
    pipeline = Pipeline(
        config_path="config/datasources.yaml",
        schema_path="config/schema.yaml",
        slices_path="config/slices.yaml",
        outdir="data/psc/out",
        ingest_dir="data/psc/ingest",
        graphs_dir="data/psc/graphs"
    )

    # Run full pipeline (or use --domain psc for automatic path resolution)
    results = pipeline.run()

    # Or run individual phases
    df = pipeline.ingest(skip_cache=False)
    slices = pipeline.slice(df)
    graphs = pipeline.build_graphs(df, mode='cumulative')
    communities = pipeline.detect_communities(mode='cumulative')
    ```
"""

from __future__ import annotations

import pathlib
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from src.graph_build import (
    CouplingConfig,
    build_direct_citation_graph,
    export_annual_full,
    export_quarter_delta,
    save_graph,
)
from src.ingest import ingest
from src.logging_config import get_logger
from src.slicing import apply_slices
from src.transform import add_time_vars, enforce_schema


@dataclass
class PipelineConfig:
    """Configuration for Pipeline execution.

    Attributes:
        config_path: Path to datasources YAML config
        schema_path: Path to schema YAML config
        slices_path: Path to slices YAML config
        outdir: Output directory for reports and manifests
        ingest_dir: Directory for cached ingest data
        graphs_dir: Directory for graph exports
        raw_dir: Directory for raw NDJSON snapshots (optional)
        graph_mode: Graph building mode: 'annual', 'delta', 'cumulative', or 'both'
        coupling_config: Optional coupling configuration
        mailto: Email for OpenAlex API (recommended for rate limits)
        log_level: Logging verbosity (DEBUG, INFO, WARNING, ERROR)
    """
    config_path: pathlib.Path
    schema_path: pathlib.Path
    slices_path: pathlib.Path
    outdir: pathlib.Path
    ingest_dir: pathlib.Path
    graphs_dir: pathlib.Path
    raw_dir: pathlib.Path | None = None
    graph_mode: str = "cumulative"
    coupling_config: CouplingConfig | None = None
    mailto: str | None = None
    log_level: str = "INFO"

    def __post_init__(self):
        """Convert string paths to Path objects."""
        for attr in ['config_path', 'schema_path', 'slices_path', 'outdir', 'ingest_dir', 'graphs_dir']:
            value = getattr(self, attr)
            if isinstance(value, str):
                setattr(self, attr, pathlib.Path(value))
        if self.raw_dir and isinstance(self.raw_dir, str):
            self.raw_dir = pathlib.Path(self.raw_dir)


@dataclass
class PipelineResults:
    """Results from pipeline execution.

    Attributes:
        df: Ingested and transformed DataFrame
        slices: Dictionary of slice_name -> DataFrame
        graphs: Dictionary of graph type -> paths
        communities: Community detection results (if run)
        manifest: Final pipeline manifest
        errors: List of errors encountered (if any)
    """
    df: pd.DataFrame | None = None
    slices: dict[str, pd.DataFrame] = field(default_factory=dict)
    graphs: dict[str, list[pathlib.Path]] = field(default_factory=dict)
    communities: dict[str, Any] | None = None
    manifest: dict[str, Any] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)


class Pipeline:
    """Research front monitoring pipeline with programmatic API.

    This class provides a clean interface to the full 2YP pipeline,
    making it easy to run all phases or individual steps programmatically.

    Attributes:
        config: PipelineConfig with all settings
        logger: Logger instance
        results: PipelineResults accumulating outputs

    Example:
        >>> pipeline = Pipeline(
        ...     config_path="config/datasources.yaml",
        ...     schema_path="config/schema.yaml",
        ...     slices_path="config/slices.yaml",
        ...     outdir="data/psc/out",
        ...     ingest_dir="data/psc/ingest",
        ...     graphs_dir="data/psc/graphs"
        ... )
        >>> results = pipeline.run()
        >>> print(f"Ingested {len(results.df)} records")
    """

    def __init__(
        self,
        config_path: pathlib.Path | str,
        schema_path: pathlib.Path | str,
        slices_path: pathlib.Path | str,
        outdir: pathlib.Path | str,
        ingest_dir: pathlib.Path | str,
        graphs_dir: pathlib.Path | str,
        raw_dir: pathlib.Path | str | None = None,
        graph_mode: str = "cumulative",
        coupling_config: CouplingConfig | None = None,
        mailto: str | None = None,
        log_level: str = "INFO"
    ):
        """Initialize pipeline with configuration.

        Args:
            config_path: Path to datasources YAML config
            schema_path: Path to schema YAML config
            slices_path: Path to slices YAML config
            outdir: Output directory for reports and manifests
            ingest_dir: Directory for cached ingest data
            graphs_dir: Directory for graph exports
            raw_dir: Optional directory for raw NDJSON snapshots
            graph_mode: Graph building mode ('annual', 'delta', 'cumulative', 'both')
            coupling_config: Optional CouplingConfig for bibliographic coupling
            mailto: Email for OpenAlex API (helps with rate limits)
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        """
        self.config = PipelineConfig(
            config_path=pathlib.Path(config_path),
            schema_path=pathlib.Path(schema_path),
            slices_path=pathlib.Path(slices_path),
            outdir=pathlib.Path(outdir),
            ingest_dir=pathlib.Path(ingest_dir),
            graphs_dir=pathlib.Path(graphs_dir),
            raw_dir=pathlib.Path(raw_dir) if raw_dir else None,
            graph_mode=graph_mode,
            coupling_config=coupling_config,
            mailto=mailto,
            log_level=log_level
        )

        self.logger = get_logger(__name__, level=log_level)
        self.results = PipelineResults()

        # Create directories
        self._setup_directories()

    def _setup_directories(self) -> None:
        """Create necessary directories if they don't exist."""
        for dir_path in [
            self.config.ingest_dir,
            self.config.ingest_dir / "slices",
            self.config.graphs_dir,
            self.config.outdir,
        ]:
            dir_path.mkdir(parents=True, exist_ok=True)

        if self.config.raw_dir:
            self.config.raw_dir.mkdir(parents=True, exist_ok=True)

    def ingest(
        self,
        skip_cache: bool = False,
        rebuild_from_raw: bool = False,
        raw_manifest_path: pathlib.Path | None = None
    ) -> pd.DataFrame:
        """Execute ingest phase to fetch or load works data.

        Args:
            skip_cache: If True, use cached ingest.parquet instead of fetching
            rebuild_from_raw: If True, rebuild from raw NDJSON snapshot
            raw_manifest_path: Path to raw manifest (required if rebuild_from_raw=True)

        Returns:
            DataFrame with ingested works data, transformed and deduplicated

        Raises:
            ValueError: If cache missing when skip_cache=True
            FileNotFoundError: If raw manifest missing when rebuild_from_raw=True

        Example:
            >>> df = pipeline.ingest(skip_cache=False)
            >>> print(f"Ingested {len(df)} works")
        """
        cache_path = self.config.ingest_dir / "ingest.parquet"

        # Mode 1: Rebuild from raw snapshot
        if rebuild_from_raw:
            if not raw_manifest_path or not raw_manifest_path.exists():
                raise FileNotFoundError(f"Raw manifest not found: {raw_manifest_path}")

            self.logger.info("Rebuilding from raw snapshot...")
            from src.raw_store import rebuild_ingest_from_raw
            df, raw_manifest = rebuild_ingest_from_raw(
                self.config.raw_dir or self.config.ingest_dir / "raw",
                raw_manifest_path
            )

        # Mode 2: Use cached data
        elif skip_cache:
            if not cache_path.exists():
                raise ValueError(f"Cache missing: {cache_path}. Run without skip_cache first.")

            self.logger.info(f"Loading cached dataset from {cache_path}")
            df = pd.read_parquet(cache_path)

        # Mode 3: Fresh fetch from OpenAlex
        else:
            self.logger.info("Fetching from OpenAlex API...")
            source_overrides = {"mailto": self.config.mailto} if self.config.mailto else None
            df, raw_records = ingest(
                str(self.config.config_path),
                source_overrides=source_overrides,
            )

            # Cache the dataset
            try:
                df.to_parquet(cache_path, index=False)
                self.logger.info(f"Cached dataset to {cache_path}")
            except Exception as e:
                self.logger.warning(f"Failed to cache dataset: {e}")

        # Transform and deduplicate
        df = add_time_vars(df)
        df = enforce_schema(df, str(self.config.schema_path))

        from run import deduplicate_efficiently
        df, dedup_stats = deduplicate_efficiently(df, "work_id", self.logger)

        self.results.df = df
        self.results.manifest['deduplication'] = dedup_stats

        self.logger.info(f"Ingest complete: {len(df)} records")
        return df

    def slice(self, df: pd.DataFrame | None = None) -> dict[str, pd.DataFrame]:
        """Apply temporal/categorical slicing to DataFrame.

        Args:
            df: DataFrame to slice (uses self.results.df if not provided)

        Returns:
            Dictionary mapping slice_name -> DataFrame

        Example:
            >>> slices = pipeline.slice()
            >>> print(f"Created {len(slices)} slices")
        """
        if df is None:
            if self.results.df is None:
                raise ValueError("No DataFrame available. Run ingest() first.")
            df = self.results.df

        self.logger.info(f"Slicing {len(df)} records...")
        slices = apply_slices(df, str(self.config.slices_path))

        self.results.slices = slices
        self.logger.info(f"Created {len(slices)} slices")

        return slices

    def build_graphs(
        self,
        df: pd.DataFrame | None = None,
        mode: str | None = None
    ) -> dict[str, list[pathlib.Path]]:
        """Build citation graphs in specified mode.

        Args:
            df: DataFrame with works (uses self.results.df if not provided)
            mode: Graph mode override ('annual', 'delta', 'cumulative', 'both')

        Returns:
            Dictionary with keys 'annual', 'delta', or 'cumulative' mapping to lists of graph paths

        Example:
            >>> graphs = pipeline.build_graphs(mode='cumulative')
            >>> print(f"Built {len(graphs['cumulative'])} cumulative graphs")
        """
        if df is None:
            if self.results.df is None:
                raise ValueError("No DataFrame available. Run ingest() first.")
            df = self.results.df

        mode = mode or self.config.graph_mode
        graphs: dict[str, list[pathlib.Path]] = {}

        # Extract time periods
        from run import extract_time_periods
        years, quarters = extract_time_periods(df)

        coupling_cfg = self.config.coupling_config

        # Build annual graphs
        if mode in ("annual", "both"):
            self.logger.info(f"Building annual graphs for {len(years)} years...")
            graphs['annual'] = []
            for year in years:
                base = export_annual_full(
                    df,
                    year=int(year),
                    outdir=self.config.graphs_dir,
                    coupling=coupling_cfg
                )
                graphs['annual'].append(base)
                self.logger.info(f"Built annual graph for {year}")

        # Build delta graphs
        if mode in ("delta", "both"):
            self.logger.info(f"Building delta graphs for {len(quarters)} quarters...")
            graphs['delta'] = []
            for qtr in quarters:
                year, q = qtr.split("Q")
                base = export_quarter_delta(
                    df,
                    year=int(year),
                    quarter=int(q),
                    outdir=self.config.graphs_dir,
                    coupling=coupling_cfg
                )
                graphs['delta'].append(base)
                self.logger.info(f"Built delta graph for {qtr}")

        # Build cumulative graphs
        if mode == "cumulative":
            self.logger.info(f"Building cumulative graphs for {len(quarters)} quarters...")
            graphs['cumulative'] = []

            for qtr in quarters:
                # Filter to works up to this quarter
                df_cum = df[df['pub_qtr'] <= qtr].copy()
                G = build_direct_citation_graph(df_cum, coupling=coupling_cfg)

                base = self.config.graphs_dir / f"citation_graph_cumulative_{qtr}"
                save_graph(G, base)
                graphs['cumulative'].append(base)
                self.logger.info(f"Built cumulative graph for {qtr}")

        self.results.graphs = graphs
        return graphs

    def detect_communities(
        self,
        mode: str | None = None,
        resume: bool = False
    ) -> dict[str, Any]:
        """Run community detection on built graphs.

        Args:
            mode: Graph mode to analyze ('cumulative', 'annual', 'delta')
            resume: Whether to resume from cache (cumulative mode only)

        Returns:
            Dictionary with community detection results

        Example:
            >>> communities = pipeline.detect_communities(mode='cumulative')
            >>> print(f"Detected {len(communities['fronts'])} research fronts")
        """
        mode = mode or self.config.graph_mode

        # Import here to avoid circular dependency
        import json
        import subprocess

        # Call scripts/communities.py as subprocess
        cmd = [
            "python",
            "scripts/communities.py",
            "--graphs-dir", str(self.config.graphs_dir),
            "--out-dir", str(self.config.outdir),
            "--mode", mode
        ]

        if resume:
            cmd.append("--resume")

        self.logger.info(f"Running community detection in {mode} mode...")
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            self.logger.error(f"Community detection failed: {result.stderr}")
            self.results.errors.append(f"Community detection failed: {result.stderr}")
            return {}

        # Load results from output files
        registry_path = self.config.outdir / f"front_id_registry_{mode}.json"
        if registry_path.exists():
            with open(registry_path) as f:
                communities = json.load(f)
                self.results.communities = communities
                self.logger.info("Community detection complete")
                return communities

        return {}

    def run(
        self,
        skip_ingest: bool = False,
        run_communities: bool = True
    ) -> PipelineResults:
        """Execute full pipeline from ingest to community detection.

        Args:
            skip_ingest: If True, use cached ingest data
            run_communities: If True, run community detection after graph building

        Returns:
            PipelineResults with all outputs

        Example:
            >>> results = pipeline.run()
            >>> print(f"Pipeline complete: {len(results.graphs)} graph types")
        """
        try:
            # Phase 1: Ingest
            self.logger.info("=" * 60)
            self.logger.info("Starting full pipeline execution")
            self.logger.info("=" * 60)

            df = self.ingest(skip_cache=skip_ingest)

            # Phase 2: Slice
            self.slice(df)

            # Phase 3: Build graphs
            self.build_graphs(df)

            # Phase 4: Community detection (optional)
            if run_communities:
                self.detect_communities()

            self.logger.info("=" * 60)
            self.logger.info("Pipeline execution complete")
            self.logger.info("=" * 60)

            return self.results

        except Exception as e:
            self.logger.error(f"Pipeline failed: {e}", exc_info=True)
            self.results.errors.append(str(e))
            raise
