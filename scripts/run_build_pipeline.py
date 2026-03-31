#!/usr/bin/env python3
"""
Run Stages 2-5 in sequence with shared resources.

This pipeline orchestrates Stages 2-5 by:
1. Loading shared resources once (LineageTextStore)
2. Calling each Stage function directly (no subprocess overhead)
3. Passing the shared store to Stages 2, 3, 4 for fast execution

Performance benefits:
- Stages 2, 3, 4 share a single lineage registry load (~3x faster)
- No subprocess spawning overhead
- All stages retain standalone CLI compatibility

Usage:
    # Run full pipeline with defaults
    python scripts/run_build_pipeline.py

    # Run with domain selection
    python scripts/run_build_pipeline.py --domain crispr

    # Run only specific Stages
    python scripts/run_build_pipeline.py --stages 2,3,4

    # Enable parallel Stage 4 with RAM-aware worker spawning
    python scripts/run_build_pipeline.py --stages 4 --npmi-workers 6 --npmi-worker-mem-gb 3

    # Skip validation to save time
    python scripts/run_build_pipeline.py --no-validate

    # Refresh metrics before pipeline (Task 5.1)
    python scripts/run_build_pipeline.py --refresh-metrics

Options:
    --domain: Research domain (derives all data paths from convention)
    --stages: Comma-separated list of Stages to run (default: 2,3,4,5)
    --min-quarters: Minimum quarters for persistent lineages (default: 6)
    --abstract-cache: Path to serialized abstract index cache (default auto-generated)
    --npmi-workers: Max worker processes for Stage 4 (default: 1 = sequential)
    --npmi-worker-mem-gb: Estimated memory per Stage 4 worker (default: 4.0 GB)
    --npmi-memory-reserve-gb: Memory to keep free before launching workers (default: 4.0 GB)
    --profile: Enable detailed timing profiling
    --validate: Run validation checks (default: True)
    --refresh-metrics: Regenerate global metrics before pipeline (Task 5.1)
    --metrics-dir: Directory for metric outputs (default: data/out/metrics)
    --slices-dir: Directory for quarterly slices (default: data/current_ingest/slices)
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from _path_bootstrap import ensure_repo_imports

repo_root = ensure_repo_imports()

from src.domain_registry import add_domain_args, get_domain, resolve_script_paths  # noqa: E402
from src.lineage_text_store import LineageTextStore  # noqa: E402


def resolve_front_config_path(domain: str | None) -> Path:
    """Resolve the front-alias config for the active domain."""
    if domain is not None:
        front_aliases = get_domain(domain).resolve_paths(repo_root).get("front_aliases")
        if front_aliases is not None:
            return front_aliases
    return repo_root / "config" / "front_aliases.yaml"


class PipelineConfig:
    """Configuration for pipeline execution."""

    def __init__(self, args):
        # Output root (all stage artifacts live under here)
        self.output_root = Path(args.output_dir)
        self.lineage_dir = self.output_root / "02_lineage_tracking"
        self.mapping_dir = self.output_root / "03_milestone_mapping"
        self.validation_root = self.output_root / "06_validation"

        # Input paths (allow overrides, otherwise derive from output root)
        self.registry_path = Path(args.registry) if args.registry else self.lineage_dir / "lineage_registry.json"
        self.lineage_metrics_path = Path(args.lineage_metrics) if args.lineage_metrics else self.lineage_dir / "lineage_metrics.csv"
        self.raw_dir = Path(args.raw)
        self.graphs_dir = Path(args.graphs)
        self.partitions_dir = Path(args.partitions)
        self.abstract_cache_path = Path(args.abstract_cache) if args.abstract_cache else None

        # Output paths (legacy attribute for backward compatibility)
        self.output_dir = self.output_root

        # Parameters
        self.min_quarters = args.min_quarters
        self.stages = [int(p) for p in args.stages.split(',')]
        self.profile = args.profile
        self.validate = args.validate

        # Metric refresh parameters (Task 5.1)
        self.refresh_metrics = args.refresh_metrics
        self.metrics_dir = Path(args.metrics_dir)
        self.slices_dir = Path(args.slices_dir)

        # Domain forwarding
        self.domain = getattr(args, "domain", None)
        self.front_config_path = resolve_front_config_path(self.domain)

        # Stage-specific parameters (can be expanded)
        self.embedding_device = args.device
        self.npmi_min_score = args.npmi_min_score
        self.npmi_min_count = args.npmi_min_count
        self.npmi_workers = args.npmi_workers
        self.npmi_worker_mem_gb = args.npmi_worker_mem_gb
        self.npmi_memory_reserve_gb = args.npmi_memory_reserve_gb


def refresh_metrics_if_requested(config: PipelineConfig) -> None:
    """
    Run metric refresh if requested (Task 5.1).

    Invokes run_metric_refresh.py to regenerate global metrics.
    """
    if not config.refresh_metrics:
        print("[Pipeline] Skipping metric refresh (use --refresh-metrics to enable)")
        return

    print("\n" + "=" * 70)
    print("METRIC REFRESH (Pre-Pipeline Step)")
    print("=" * 70)
    print(f"[Pipeline] Refreshing metrics in {config.metrics_dir}")

    t_start = time.time()

    # Import and run metric refresh
    import subprocess
    cmd = [
        sys.executable,
        str(repo_root / 'scripts' / 'run_metric_refresh.py'),
        '--slices-dir', str(config.slices_dir),
        '--out-dir', str(config.metrics_dir),
    ]

    # Forward --domain to metric refresh
    if config.domain is not None:
        cmd.extend(['--domain', config.domain])

    print(f"[Pipeline] Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, check=False)

    if result.returncode != 0:
        print(f"[WARNING] Metric refresh failed with exit code {result.returncode}")
        print("[WARNING] Continuing with existing metrics")
    else:
        t_refresh = time.time() - t_start
        print(f"\n[Pipeline] Metrics refreshed in {t_refresh:.2f}s")

    print("=" * 70)


def load_shared_resources(config: PipelineConfig) -> LineageTextStore:
    """
    Load shared resources once for all stages.

    The store is shared with Stages 2, 3, 4 (which use it).
    Stage 5 doesn't need the store but also uses direct calls (no subprocess).

    Args:
        config: Pipeline configuration

    Returns:
        Loaded LineageTextStore instance
    """
    print("\n" + "=" * 70)
    print("LOADING SHARED RESOURCES")
    print("=" * 70)

    t_start = time.time()

    store = LineageTextStore(
        registry_path=config.registry_path,
        raw_dir=config.raw_dir,
        graphs_dir=config.graphs_dir,
        partitions_dir=config.partitions_dir,
        abstract_cache_path=config.abstract_cache_path,
        verbose=True
    )

    config.abstract_cache_path = store.abstract_cache_path

    t_load = time.time() - t_start
    print(f"\n[Pipeline] Store loaded in {t_load:.2f}s")
    print("[Pipeline] All stages refactored - No subprocess overhead!")
    print("[Pipeline] Stages 2, 3, 4 use shared store (faster)")
    print("[Pipeline] Stage 5 uses direct call (Stage 5 only needs similarity matrices)")
    print("=" * 70)

    return store


def run_stage2_embeddings(config: PipelineConfig, store: LineageTextStore) -> None:
    """
    Run Stage 2: SciBERT embeddings.

    Uses shared store - NO subprocess, NO redundant loading!

    Args:
        config: Pipeline configuration
        store: Shared text store (ACTUALLY USED!)
    """
    from scripts.compute_lineage_embeddings import run_embeddings

    print("\n" + "=" * 70)
    print("Stage 2: SciBERT Embeddings")
    print("=" * 70)
    print("[Pipeline] Starting Stage 2 with shared store...")

    t_start = time.time()

    # Device handling: convert 'auto' to None
    device = None if config.embedding_device == "auto" else config.embedding_device

    # Call directly with shared store
    config.lineage_dir.mkdir(parents=True, exist_ok=True)

    run_embeddings(
        min_quarters=config.min_quarters,
        device=device,
        profile=config.profile,
        output_path=config.lineage_dir / "lineage_embeddings.npz",
        lineage_metrics_path=config.lineage_metrics_path,
        front_config_path=config.front_config_path,
        partitions_dir=config.partitions_dir,
        output_root=config.output_root,
        store=store,  # Pass shared store!
        validate=config.validate  # Pass validate flag
    )

    t_stage = time.time() - t_start
    print(f"\n[Pipeline] Stage 2 completed in {t_stage:.2f}s (using shared store)")


def run_stage3_ctfidf(config: PipelineConfig, store: LineageTextStore) -> None:
    """
    Run Stage 3: c-TF-IDF term extraction.

    Uses shared store - NO subprocess, NO redundant loading!

    Args:
        config: Pipeline configuration
        store: Shared text store (ACTUALLY USED!)
    """
    from scripts.compute_lineage_ctfidf import run_ctfidf

    print("\n[Pipeline] Starting Stage 3 with shared store...")

    t_start = time.time()

    # Call directly with shared store
    config.mapping_dir.mkdir(parents=True, exist_ok=True)
    (config.validation_root / "stage3").mkdir(parents=True, exist_ok=True)

    run_ctfidf(
        min_quarters=config.min_quarters,
        top_n=50,
        similarity_threshold=0.01,
        front_config_path=config.front_config_path,
        partitions_dir=config.partitions_dir,
        store=store,  # Pass shared store!
        output_root=config.output_root,
        validate=config.validate  # Pass validate flag
    )

    t_stage = time.time() - t_start
    print(f"\n[Pipeline] Stage 3 completed in {t_stage:.2f}s (using shared store)")


def run_stage4_npmi(config: PipelineConfig, store: LineageTextStore) -> None:
    """
    Run Stage 4: NPMI co-term discovery.

    Uses shared store - NO subprocess, NO redundant loading!

    Args:
        config: Pipeline configuration
        store: Shared text store (ACTUALLY USED!)
    """
    from scripts.compute_lineage_npmi import run_npmi

    print("\n" + "=" * 70)
    print("Stage 4: NPMI Co-term Discovery")
    print("=" * 70)
    print("[Pipeline] Starting Stage 4 with shared store...")

    t_start = time.time()

    # Call directly with shared store
    (config.validation_root / "stage4").mkdir(parents=True, exist_ok=True)

    run_npmi(
        min_quarters=config.min_quarters,
        min_npmi=config.npmi_min_score,
        min_pair_count=config.npmi_min_count,
        output_threshold=0.8,
        front_config_path=config.front_config_path,
        partitions_dir=config.partitions_dir,
        ctfidf_vocab_path=config.lineage_dir / "lineage_ctfidf_terms.csv",
        vocab_size=100,
        registry_path=config.registry_path,
        raw_dir=config.raw_dir,
        abstract_cache_path=config.abstract_cache_path,
        store=store,  # Pass shared store!
        output_root=config.output_root,
        validate=config.validate,  # Pass validate flag
        max_workers=config.npmi_workers,
        worker_memory_gb=config.npmi_worker_mem_gb,
        memory_reserve_gb=config.npmi_memory_reserve_gb
    )

    t_stage = time.time() - t_start
    print(f"\n[Pipeline] Stage 4 completed in {t_stage:.2f}s (using shared store)")


def run_stage5_ensemble(config: PipelineConfig, _store: LineageTextStore) -> None:
    """
    Run Stage 5: Ensemble mapping.

    Stage 5 doesn't use shared store (only loads similarity matrices),
    but we call it directly to avoid subprocess overhead.

    Args:
        config: Pipeline configuration
        store: Shared text store (unused in Stage 5)
    """
    from scripts.stage5_ensemble_mapping import run_ensemble

    print("\n" + "=" * 70)
    print("Stage 5: Ensemble Front Mapping")
    print("=" * 70)
    print("[Pipeline] Starting Stage 5 (direct call, no subprocess overhead)...")

    t_start = time.time()

    # Call directly (Stage 5 doesn't use shared store but benefits from no subprocess)
    (config.validation_root / "stage5").mkdir(parents=True, exist_ok=True)

    run_ensemble(
        stage2_similarity_path=config.mapping_dir / "lineage_front_similarity.csv",
        stage3_similarity_path=config.mapping_dir / "lineage_front_term_similarity.csv",
        stage4_similarity_path=config.mapping_dir / "lineage_front_npmi_similarity.csv",
        stage3_terms_path=config.lineage_dir / "lineage_ctfidf_terms.csv",
        stage4_pairs_path=config.lineage_dir / "lineage_npmi_pairs.csv",
        front_config_path=config.front_config_path,
        top_k=3,
        store=None,  # Stage 5 doesn't use the store
        output_root=config.output_root,
        validate=config.validate  # Pass validate flag
    )

    t_stage = time.time() - t_start
    print(f"\n[Pipeline] Stage 5 completed in {t_stage:.2f}s")


def main():
    parser = argparse.ArgumentParser(
        description="Run Stages 2-5 pipeline with shared text index"
    )

    # Input paths (None defaults allow domain resolution)
    parser.add_argument(
        "--registry",
        type=str,
        default=None,
        help="Path to lineage registry JSON (defaults to <output_dir>/02_lineage_tracking/lineage_registry.json)"
    )
    parser.add_argument(
        "--lineage-metrics",
        type=str,
        default=None,
        help="Path to lineage metrics CSV (defaults to <output_dir>/02_lineage_tracking/lineage_metrics.csv)"
    )
    parser.add_argument(
        "--raw",
        type=str,
        default=None,
        help="Path to raw JSONL files"
    )
    parser.add_argument(
        "--graphs",
        type=str,
        default=None,
        help="Path to citation graph files"
    )
    parser.add_argument(
        "--partitions",
        type=str,
        default=None,
        help="Path to partition JSON files"
    )
    parser.add_argument(
        "--abstract-cache",
        type=str,
        default=None,
        help="Path to serialized abstract index cache (default auto-generated)"
    )

    # Output paths
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Base output directory for all stages (default: data/out)"
    )

    # Pipeline control
    parser.add_argument(
        "--stages",
        type=str,
        default="2,3,4,5",
        help="Comma-separated list of Stages to run (default: 2,3,4,5)"
    )
    parser.add_argument(
        "--min-quarters",
        type=int,
        default=6,
        help="Minimum quarters for persistent lineages (default: 6)"
    )

    # Stage 2 parameters
    parser.add_argument(
        "--device",
        choices=["cuda", "cpu", "auto"],
        default="auto",
        help="Device for SciBERT inference (Stage 2)"
    )

    # Stage 4 parameters
    parser.add_argument(
        "--npmi-min-score",
        type=float,
        default=0.2,
        help="Minimum NPMI score threshold (Stage 4, default: 0.2)"
    )
    parser.add_argument(
        "--npmi-min-count",
        type=int,
        default=3,
        help="Minimum co-occurrence count (Stage 4, default: 3)"
    )
    parser.add_argument(
        "--npmi-workers",
        type=int,
        default=1,
        help="Maximum Stage 4 worker processes (default: 1 = sequential)"
    )
    parser.add_argument(
        "--npmi-worker-mem-gb",
        type=float,
        default=4.0,
        help="Estimated memory footprint per Stage 4 worker process in GB (default: 4.0)"
    )
    parser.add_argument(
        "--npmi-memory-reserve-gb",
        type=float,
        default=4.0,
        help="Memory in GB to keep free when launching Stage 4 workers (default: 4.0)"
    )

    # Performance options
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Enable detailed timing profiling"
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        default=True,
        help="Run validation checks and generate reports (default: True)"
    )
    parser.add_argument(
        "--no-validate",
        dest="validate",
        action="store_false",
        help="Skip validation checks"
    )

    # Metric refresh options (Task 5.1)
    parser.add_argument(
        "--refresh-metrics",
        action="store_true",
        help="Run metric refresh before pipeline (regenerate global metrics)"
    )
    parser.add_argument(
        "--metrics-dir",
        type=str,
        default=None,
        help="Directory for metric outputs (default: data/out/metrics)"
    )
    parser.add_argument(
        "--slices-dir",
        type=str,
        default=None,
        help="Directory for quarterly slices (default: data/current_ingest/slices)"
    )

    add_domain_args(parser)
    args = parser.parse_args()

    # Resolve domain paths (returns None when --domain is omitted)
    dpaths = resolve_script_paths(args, repo_root)

    # Apply domain defaults for paths not explicitly provided
    args.raw = args.raw or (str(dpaths.raw) if dpaths else "data/current_ingest/raw")
    args.graphs = args.graphs or (str(dpaths.graphs) if dpaths else "data/current_graphs")
    args.partitions = args.partitions or (
        str(dpaths.cache_cum / "partitions_cum") if dpaths else "data/out/cache_cum/partitions_cum"
    )
    args.output_dir = args.output_dir or (str(dpaths.out) if dpaths else "data/out")
    args.metrics_dir = args.metrics_dir or (
        str(dpaths.out / "metrics") if dpaths else "data/out/metrics"
    )
    args.slices_dir = args.slices_dir or (
        str(dpaths.slices) if dpaths else "data/current_ingest/slices"
    )

    config = PipelineConfig(args)

    # Pipeline execution
    print("\n" + "=" * 70)
    print("Stages 2-5 PIPELINE WITH SHARED INDEX")
    print("=" * 70)
    print(f"\nStages to run: {config.stages}")
    print(f"Min quarters: {config.min_quarters}")
    print(f"Output dir: {config.output_dir}")
    if config.domain:
        print(f"Domain: {config.domain}")
    if config.refresh_metrics:
        print("Metric refresh: ENABLED (will regenerate metrics)")
    else:
        print("Metric refresh: DISABLED (use --refresh-metrics to enable)")

    # Metric refresh (optional pre-step, Task 5.1)
    t_pipeline_start = time.time()
    refresh_metrics_if_requested(config)

    # Load shared resources ONCE
    store = load_shared_resources(config)

    # Run requested Stages
    stage_runners = {
        2: run_stage2_embeddings,
        3: run_stage3_ctfidf,
        4: run_stage4_npmi,
        5: run_stage5_ensemble
    }

    for stage in config.stages:
        if stage not in stage_runners:
            print(f"\n[WARNING] Unknown Stage {stage}, skipping")
            continue

        runner = stage_runners[stage]
        runner(config, store)

    # Final summary
    t_pipeline_total = time.time() - t_pipeline_start

    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)
    print(f"Total time: {t_pipeline_total:.2f}s ({t_pipeline_total/60:.1f} min)")
    print(f"Stages run: {', '.join(str(p) for p in config.stages)}")
    print("\nNext steps:")
    print("  1. Validate Stage 5 mappings: python scripts/validate_stage5.py")
    print("  2. Aggregate to fronts: python scripts/aggregate_lineages_to_fronts.py")
    print("  3. Run tripwire validation: python scripts/evaluate_tripwire.py")
    print("=" * 70)


if __name__ == "__main__":
    main()
