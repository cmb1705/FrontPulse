from __future__ import annotations

import argparse
import json
import logging
import os
import pathlib
import shutil
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
import yaml

try:
    import pyarrow as pa
except Exception:
    pa = None

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed; rely on environment variables

import contextlib

from src.graph_build import (
    CouplingConfig,
    build_direct_citation_graph,
    export_annual_full,
    export_quarter_delta,
    save_graph,
)
from src.ingest import apply_source_overrides, ingest
from src.logging_config import log_section, setup_logging
from src.memory_utils import (
    check_memory_availability,
    log_memory_usage,
    memory_monitor,
    suggest_chunking_strategy,
    suggest_graph_worker_count,
)
from src.openalex import results_to_df
from src.raw_store import write_raw_chunks
from src.settings import load_settings, redact_mailto, save_settings, summary
from src.slicing import apply_slices
from src.transform import add_time_vars
from src.validate import enforce_schema

# ============================================================================
# Global Configuration (Performance)
# ============================================================================

# Load defaults from config/defaults.yaml
try:
    from src.config import get_coupling_defaults, get_graph_defaults
    _COUPLING_CFG = get_coupling_defaults()
    _GRAPH_CFG = get_graph_defaults()
except Exception:
    # Fallback if config not available
    _COUPLING_CFG = {
        "alpha": 1.0,
        "beta": 0.3,
        "lambda_decay": 0.15,
        "min_shared_refs": 5,
        "min_coupling_score": 0.05,
        "default_workers": 12,
    }
    _GRAPH_CFG = {"default_workers": 12}

# Default number of parallel workers for all parallel operations
# Optimized for systems with 16+ cores and 32GB+ RAM
# Adjust this value based on your system:
#   - 4-8 core systems: Set to 4-6
#   - 8-12 core systems: Set to 6-10
#   - 16+ core systems: Set to 10-14
# Or adjust in config/defaults.yaml
DEFAULT_PARALLEL_WORKERS = _GRAPH_CFG.get("default_workers", 12)

# ============================================================================
# Performance Helpers (PERF-2)
# ============================================================================

def deduplicate_efficiently(df: pd.DataFrame, key: str, logger: logging.Logger) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Efficiently deduplicate DataFrame with progress logging and statistics.
    PERF-2: Optimized deduplication with chunking for large DataFrames.
    HP-5: Enhanced with deduplication statistics and example logging.

    Args:
        df: Input DataFrame
        key: Column to deduplicate on (typically "work_id")
        logger: Logger for messages

    Returns:
        Tuple of (deduplicated_df, stats_dict) where stats_dict contains:
        - original_count: Number of rows before deduplication
        - duplicate_count: Number of duplicate rows removed
        - final_count: Number of rows after deduplication
        - duplicate_rate: Percentage of duplicates (0.0-100.0)
        - example_duplicates: List of up to 5 example duplicate keys
    """
    n_rows = len(df)
    dup_mask = df.duplicated(key, keep="first")
    dups = int(dup_mask.sum())

    # Gather example duplicates before removal (up to 5)
    example_duplicates = []
    if dups > 0:
        duplicate_keys = df.loc[dup_mask, key].head(5).tolist()
        example_duplicates = [str(k) for k in duplicate_keys]

    # Build statistics dictionary
    stats = {
        "original_count": n_rows,
        "duplicate_count": dups,
        "final_count": n_rows - dups,
        "duplicate_rate": (100.0 * dups / n_rows) if n_rows > 0 else 0.0,
        "example_duplicates": example_duplicates,
    }

    if dups == 0:
        logger.debug(f"No duplicate {key} rows found")
        return df, stats

    logger.warning(
        f"Found {dups} duplicate {key} rows ({stats['duplicate_rate']:.2f}%)"
    )

    # Log example duplicates
    if example_duplicates:
        logger.info(f"Example duplicate {key}s: {', '.join(example_duplicates[:3])}")

    # Reset index level to avoid ambiguity if key is both index and column
    # Only reset the specific level, preserving other index levels if MultiIndex
    if key in df.index.names:
        df = df.reset_index(level=key, drop=key in df.columns)

    # For large DataFrames (>100k rows), use more memory-efficient approach
    if n_rows > 100_000:
        logger.info(f"Using memory-efficient deduplication for {n_rows:,} rows")
        # Sort by key first for better memory locality
        df_sorted = df.sort_values(key)
        df_deduped = df_sorted.drop_duplicates(key, keep="first")
        logger.info(f"Deduplication complete: {len(df_deduped):,} rows retained")
        return df_deduped, stats
    else:
        df_deduped = df.drop_duplicates(key, keep="first")
        logger.info(f"Deduplication complete: {len(df_deduped):,} rows retained")
        return df_deduped, stats


# --- interactive helpers ---
def ask(prompt: str, default: str | None = None) -> str:
    sfx = f" [{default}]" if default is not None else ""
    val = input(f"{prompt}{sfx}: ").strip()
    return val if val else (default if default is not None else "")

def ask_yes_no(prompt: str, default: str = "N") -> bool:
    d = default.upper()
    val = input(f"{prompt} [Y/N] (default {d}): ").strip().upper()
    if not val:
        val = d
    return val.startswith("Y")

def build_source_overrides(cfg: dict[str, Any]) -> dict[str, Any]:
    """Build runtime datasource overrides without rewriting tracked YAML."""
    overrides: dict[str, Any] = {
        "per_page": int(cfg["per_page"]),
        "max_records": None if cfg["max_records"] in (None, "", "None") else int(cfg["max_records"]),
        "mailto": cfg.get("mailto"),
        "filters": {
            "topics.id": cfg["topics_id"],
            "from_publication_date": cfg["from_date"],
            "to_publication_date": cfg["to_date"],
        },
    }
    api_key = cfg.get("api_key")
    if api_key:
        overrides["api_key"] = api_key
    return overrides

def validate_configs(args, logger: logging.Logger) -> bool:
    """
    Validate YAML configuration files at startup.

    Args:
        args: Parsed command-line arguments
        logger: Logger instance

    Returns:
        True if all configs are valid (or pydantic not installed)

    Raises:
        ValueError: If validation fails
    """
    try:
        from src.config_models import PYDANTIC_AVAILABLE, validate_all_configs

        if not PYDANTIC_AVAILABLE:
            logger.debug("Pydantic not installed; skipping config validation")
            return True

        logger.info("Validating configuration files...")
        validate_all_configs(
            pathlib.Path(args.config),
            pathlib.Path(args.schema),
            pathlib.Path(args.slices)
        )
        logger.info("Configuration validation passed")
        return True

    except Exception as e:
        logger.error(f"Configuration validation failed: {e}")
        raise ValueError(f"Invalid configuration: {e}") from e


def preflight_check(args, settings: dict[str, Any]) -> None:
    """
    1) API field check on 1 record
    2) 20-record mini ingest -> schema coerce
    3) Tiny GraphML smoke test
    Abort unless user confirms.
    """
    import yaml

    from src.openalex import fetch_openalex, results_to_df

    # load config and build minimal request
    y = yaml.safe_load(pathlib.Path(args.config).read_text())
    primary = apply_source_overrides(
        y["sources"]["primary"],
        settings.get("source_overrides") or build_source_overrides(settings),
    )
    primary["per_page"] = 20
    primary["max_records"] = 40

    # 1) API ping
    print("[Preflight] API ping...")
    raw = fetch_openalex(
        entity=primary.get("entity", "works"),
        mailto=primary.get("mailto"),
        api_key=primary.get("api_key"),
        filters=primary.get("filters"),
        search=primary.get("search"),
        select=primary.get("select"),
        sort=primary.get("sort"),
        per_page=int(primary.get("per_page", 20)),
        max_records=int(primary.get("max_records", 40)),
        sleep_s=0.0,
    )
    if not raw:
        print("[Preflight] No records returned. Check filters.")
        sys.exit(1)

    # 2) mini-DF + schema coerce
    print("[Preflight] Flatten + schema check...")
    mini_df = results_to_df(primary.get("entity","works"), raw)
    mini_df = add_time_vars(mini_df)
    try:
        _ = enforce_schema(mini_df, args.schema)
        print("[Preflight] Schema coerce ok.")
    except Exception as e:
        print(f"[Preflight] Schema mismatch: {e}")
        sys.exit(1)

    # 3) GraphML smoke
    print("[Preflight] GraphML smoke test...")
    from src.graph_build import build_direct_citation_graph, save_graph
    G = build_direct_citation_graph(mini_df.head(50))
    tmp = pathlib.Path(args.outdir) / "graphs_preflight"
    tmp.parent.mkdir(parents=True, exist_ok=True)
    try:
        save_graph(G, tmp)
        for ext in (".pkl", ".graphml"):
            p = pathlib.Path(str(tmp) + ext)
            if p.exists():
                p.unlink()
        print("[Preflight] GraphML write ok.")
    except Exception as e:
        print(f"[Preflight] GraphML serialization error: {e}")
        sys.exit(1)

    # config warnings
    mr = settings.get("max_records")
    if mr not in (None, "", "None"):
        try:
            if int(mr) <= 10000:
                print(f"[Preflight] Warning: max_records={mr} will truncate results.")
        except Exception:
            pass
def collect_raw_metadata(
    config_path: pathlib.Path,
    source_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    meta: dict[str, Any] = {"config_path": str(config_path)}
    try:
        data = yaml.safe_load(config_path.read_text())
        primary = apply_source_overrides(
            (data.get("sources") or {}).get("primary") or {},
            source_overrides,
        )
        meta.update({
            "mailto": redact_mailto(primary.get("mailto")),
            "entity": primary.get("entity"),
            "filters": primary.get("filters"),
            "select": primary.get("select"),
            "sort": primary.get("sort"),
        })
    except Exception:
        pass
    return meta

def relativize_raw_manifest(raw_manifest: dict[str, Any], base: pathlib.Path) -> dict[str, Any]:
    def _rel(path_str: str | None) -> str | None:
        if not path_str:
            return path_str
        try:
            return str(pathlib.Path(path_str).relative_to(base))
        except ValueError:
            return path_str

    normalized = dict(raw_manifest)
    normalized["outdir"] = _rel(normalized.get("outdir"))
    normalized["manifest_path"] = _rel(normalized.get("manifest_path"))

    chunks: list[dict[str, Any]] = []
    for chunk in raw_manifest.get("chunks", []):
        chunk_copy = dict(chunk)
        for key in ("basepath", "ndjson_path", "index_path", "compressed_path"):
            chunk_copy[key] = _rel(chunk_copy.get(key))
        chunks.append(chunk_copy)
    normalized["chunks"] = chunks
    normalized["metadata"] = raw_manifest.get("metadata", {})
    return normalized


def load_latest_raw_manifest(raw_dir: pathlib.Path) -> dict[str, Any] | None:
    manifests = sorted(raw_dir.glob("*_manifest.json"))
    if not manifests:
        return None
    latest = manifests[-1]
    try:
        data = json.loads(latest.read_text())
        data.setdefault("manifest_path", str(latest))
        return data
    except Exception:
        return None

def _load_raw_manifest(raw_dir: pathlib.Path, manifest_arg: str | None) -> dict[str, Any]:
    if manifest_arg:
        manifest_path = pathlib.Path(manifest_arg)
        if not manifest_path.exists():
            raise FileNotFoundError(f"[Raw] Manifest not found: {manifest_path}")
        data = json.loads(manifest_path.read_text())
        data.setdefault("manifest_path", str(manifest_path))
        return data
    manifest = load_latest_raw_manifest(raw_dir)
    if manifest is None:
        raise FileNotFoundError(f"[Raw] No manifest found under {raw_dir}")
    return manifest

def rebuild_ingest_from_raw(raw_dir: pathlib.Path, manifest_arg: str | None) -> tuple[pd.DataFrame, dict[str, Any]]:
    manifest = _load_raw_manifest(raw_dir, manifest_arg)
    chunks = manifest.get("chunks") or []
    if not chunks:
        raise ValueError("[Raw] Manifest contains no chunk metadata.")

    records: list[dict[str, Any]] = []
    for chunk in chunks:
        raw_path_str = chunk.get("ndjson_path", "")
        ndjson_path = pathlib.Path(raw_path_str)
        if not ndjson_path.exists():
            candidate = raw_dir / ndjson_path.name
            if candidate.exists():
                ndjson_path = candidate
            else:
                raise FileNotFoundError(f"[Raw] NDJSON chunk missing: {raw_path_str}")
        with ndjson_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                records.append(json.loads(line))
    entity = (manifest.get("metadata") or {}).get("entity", "works")
    df = results_to_df(entity, records)
    return df, manifest

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", default=None, choices=["psc", "crispr"],
                    help="Research domain shortcut (psc or crispr). Overrides --config.")
    ap.add_argument("--config", default=None, help="config/datasources.yaml (or use --domain)")
    ap.add_argument("--schema", required=True, help="config/schema.yaml")
    ap.add_argument("--slices", required=True, help="config/slices.yaml")
    ap.add_argument("--outdir", required=True, help="output directory")
    ap.add_argument("--ingest-dir", default="data/current_ingest", help="Directory for cached ingest data")
    ap.add_argument("--graphs-dir", default="data/current_graphs", help="Directory for generated graph files")
    ap.add_argument("--raw-dir", default=None, help="Directory for raw OpenAlex snapshots (defaults to <ingest-dir>/raw)")
    ap.add_argument("--raw-basename", default="openalex_raw", help="Basename for raw NDJSON chunk files")
    ap.add_argument("--raw-chunk-size", type=int, default=2000, help="Max records per raw NDJSON chunk (<=0 keeps single file)")
    ap.add_argument("--raw-compression", choices=["none", "gzip"], default="gzip",
                    help="Compression for archival copies of raw NDJSON chunks")
    ap.add_argument("--skip-raw", action="store_true", help="Disable raw snapshot capture during ingest")
    ap.add_argument("--rebuild-ingest-from-raw", action="store_true",
                    help="Rebuild ingest.parquet from cached raw NDJSON (no API requests).")
    ap.add_argument("--raw-manifest", default=None,
                    help="Manifest JSON to use when rebuilding ingest (defaults to latest under raw directory).")
    ap.add_argument("--cutoff", default=None, help="YYYY-MM-DD cutoff for time slices")
    ap.add_argument("--mailto", default=None, help="Contact email for OpenAlex requests")
    ap.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                    help="Console logging level (default: INFO)")
    ap.add_argument("--log-file", default=None, type=pathlib.Path,
                    help="Path to log file (enables file logging with rotation)")
    ap.add_argument("--verbose", "-v", action="store_true",
                    help="Enable verbose logging (equivalent to --log-level DEBUG)")
    ap.add_argument("--graph-mode", choices=["none","annual","delta","both","cumulative"],
                    default=None,
                    help="Graph export mode (default: cumulative)")
    ap.add_argument("--resume-graphs", action="store_true",
                    help="Skip rebuilding graph files that already exist on disk (cumulative run)")
    ap.add_argument("--skip-graphml", action="store_true",
                    help="Skip all GraphML exports (write only pickle) - deprecated, use --graphml-for instead")
    ap.add_argument("--graphml-for", type=str, default=None,
                    help="Export GraphML only for specified modes (comma-separated: annual,delta,cumulative). Default: none (pickle only)")
    ap.add_argument("--graphml-compression", choices=["none", "gzip"], default="none",
                    help="Compression mode when writing GraphML (default: none)")
    ap.add_argument("--interactive", action="store_true", help="Prompt for settings before run (deprecated: use --configure)")
    ap.add_argument("--configure", action="store_true", help="Interactive configuration mode - prompt for all settings")
    ap.add_argument("--use-last", action="store_true", help="(Deprecated) Use last saved settings - now the default behavior")
    ap.add_argument("--skip-preflight", action="store_true", help="Skip preflight checks")
    ap.add_argument("--skip-ingest", action="store_true", help="Reuse cached ingest data from a previous run")
    ap.add_argument("--incremental", action="store_true",
                    help="Fetch only new works since last ingestion (uses last_ingested_date watermark)")
    ap.add_argument("--since", default=None,
                    help="Override incremental start date (YYYY-MM-DD). Implies --incremental.")
    ap.add_argument("--archive", action="store_true", help="Archive ingest/graphs/reports after successful run")
    ap.add_argument("--archive-only", action="store_true", help="Archive ingest/graphs/reports and exit")
    ap.add_argument("--communities", default="none",
                choices=["none","annual","delta","both"],
                help="Run Leiden + alignment on annual, delta, or both")
    ap.add_argument("--enable-coupling", action="store_true",
                    help="Enable bibliographic coupling augmentation when building graphs")
    ap.add_argument("--skip-communities", action="store_true",
                    help="Skip running community detection (scripts/communities.py) during this pipeline run")
    ap.add_argument("--coupling-alpha", type=float, default=_COUPLING_CFG.get("alpha", 1.0),
                    help=f"Citation importance weight alpha (default {_COUPLING_CFG.get('alpha', 1.0)})")
    ap.add_argument("--coupling-beta", type=float, default=_COUPLING_CFG.get("beta", 0.3),
                    help=f"Coupling importance weight beta (default {_COUPLING_CFG.get('beta', 0.3)})")
    ap.add_argument("--coupling-decay", type=float, default=_COUPLING_CFG.get("lambda_decay", 0.15),
                    help=f"Temporal decay lambda applied to coupling weights (default {_COUPLING_CFG.get('lambda_decay', 0.15)})")
    ap.add_argument("--coupling-min-shared", type=int, default=_COUPLING_CFG.get("min_shared_refs", 5),
                    help=f"Minimum shared references required to add a coupling edge (default {_COUPLING_CFG.get('min_shared_refs', 5)})")
    ap.add_argument("--coupling-min-score", type=float, default=_COUPLING_CFG.get("min_coupling_score", 0.05),
                    help=f"Minimum normalized coupling score required to add an edge (default {_COUPLING_CFG.get('min_coupling_score', 0.05)})")
    ap.add_argument("--coupling-cache-dir", type=pathlib.Path, default=pathlib.Path("data/out/cache_coupling"),
                    help="Directory for caching bibliographic coupling intermediates")
    ap.add_argument("--clear-coupling-cache", action="store_true",
                    help="Clear coupling cache before building graphs")
    ap.add_argument("--coupling-workers", type=int, default=DEFAULT_PARALLEL_WORKERS,
                    help=f"Worker process count for coupling pair counting (default {DEFAULT_PARALLEL_WORKERS})")
    ap.add_argument("--graph-workers", type=int, default=DEFAULT_PARALLEL_WORKERS,
                    help=f"Parallel workers for graph building (PERF-1, default {DEFAULT_PARALLEL_WORKERS})")
    return ap.parse_args()

def _iter_refs_cell(x):
    if x is None:
        return []
    if pa is not None and isinstance(x, pa.lib.ListScalar):
        x = x.as_py()
    if isinstance(x, (list, tuple, set, np.ndarray)):
        return [str(t).split("/")[-1] for t in list(x)]
    if isinstance(x, str):
        s = x.strip()
        if not s:
            return []
        if s[0] == "[" and s[-1] == "]":
            inner = [t.strip(" '\"") for t in s[1:-1].split(",") if t.strip()]
            return [i.split("/")[-1] for i in inner]
        return [t.strip().split("/")[-1] for t in s.split(",") if t.strip()]
    return []

def _ref_resolution_for_slice(path: pathlib.Path, all_ids: set[str]) -> dict:
    df_quarter = pd.read_parquet(path, engine="pyarrow", columns=["work_id","referenced_works"])
    refs = df_quarter["referenced_works"].apply(_iter_refs_cell)
    tot = int(refs.apply(len).sum())
    in_c = int(refs.apply(lambda L: sum(1 for r in L if r in all_ids)).sum())
    return {"total_refs": tot, "in_corpus_refs": in_c, "in_corpus_share": (in_c / tot) if tot else 0.0}

def archive_current(ingest_dir: pathlib.Path, graphs_dir: pathlib.Path, out_dir: pathlib.Path) -> None:
    """Archive current ingest/graphs/out directories to timestamped snapshot."""
    archive_root = pathlib.Path("data") / "archive"
    archive_root.mkdir(parents=True, exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    dest_root = archive_root / ts
    dest_root.mkdir(exist_ok=True)
    targets = [("ingest", ingest_dir), ("graphs", graphs_dir), ("out", out_dir)]
    for label, src in targets:
        if not src.exists():
            print(f"[Archive] Skipping {label}: {src} missing")
            continue
        dest = dest_root / label
        try:
            shutil.copytree(src, dest, dirs_exist_ok=True)
            print(f"[Archive] {label} -> {dest}")
        except Exception as e:
            print(f"[Archive] Failed to archive {label}: {e}")


# ============================================================================
# PIPELINE PHASE FUNCTIONS
# ============================================================================

def setup_directories(args) -> tuple[pathlib.Path, pathlib.Path, pathlib.Path, pathlib.Path]:
    """
    Create and return pipeline directories.

    Args:
        args: Parsed command-line arguments

    Returns:
        Tuple of (ingest_dir, slices_dir, graphs_dir, raw_dir)
    """
    ingest_dir = pathlib.Path(args.ingest_dir)
    ingest_dir.mkdir(parents=True, exist_ok=True)

    slices_dir = ingest_dir / "slices"
    slices_dir.mkdir(parents=True, exist_ok=True)

    graphs_dir = pathlib.Path(args.graphs_dir)
    graphs_dir.mkdir(parents=True, exist_ok=True)

    raw_dir = pathlib.Path(args.raw_dir) if args.raw_dir else ingest_dir / "raw"

    return ingest_dir, slices_dir, graphs_dir, raw_dir


def get_cache_size(cache_dir: pathlib.Path) -> dict:
    """
    Calculate cache directory size and file counts.

    Args:
        cache_dir: Cache directory path

    Returns:
        Dictionary with size metrics
    """
    if not cache_dir.exists():
        return {"size_bytes": 0, "size_mb": 0.0, "file_count": 0}

    total_size = 0
    file_count = 0
    for file in cache_dir.rglob("*"):
        if file.is_file():
            total_size += file.stat().st_size
            file_count += 1

    return {
        "size_bytes": total_size,
        "size_mb": round(total_size / (1024 * 1024), 2),
        "file_count": file_count
    }


def clear_coupling_cache(cache_dir: pathlib.Path, logger: logging.Logger) -> None:
    """
    Clear all files in the coupling cache directory.

    Args:
        cache_dir: Cache directory to clear
        logger: Logger instance
    """
    if not cache_dir.exists():
        logger.info(f"Cache directory does not exist: {cache_dir}")
        return

    cache_stats = get_cache_size(cache_dir)
    if cache_stats["file_count"] == 0:
        logger.info("Cache is already empty")
        return

    # Remove all files in cache
    import shutil
    for item in cache_dir.iterdir():
        if item.is_file():
            item.unlink()
        elif item.is_dir():
            shutil.rmtree(item)

    logger.info(f"Cleared {cache_stats['file_count']} files ({cache_stats['size_mb']} MB) from coupling cache")


def setup_coupling_config(args, logger: logging.Logger) -> CouplingConfig | None:
    """
    Create coupling configuration if enabled.

    Args:
        args: Parsed command-line arguments
        logger: Logger instance

    Returns:
        CouplingConfig object if enabled, None otherwise
    """
    if not args.enable_coupling:
        return None

    cache_dir = args.coupling_cache_dir
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)

        # Clear cache if requested
        if args.clear_coupling_cache:
            clear_coupling_cache(cache_dir, logger)

        # Report cache size
        cache_stats = get_cache_size(cache_dir)
        if cache_stats["file_count"] > 0:
            logger.info(f"Coupling cache: {cache_stats['file_count']} files, {cache_stats['size_mb']} MB")
        else:
            logger.info("Coupling cache is empty")

    coupling_cfg = CouplingConfig(
        enabled=True,
        alpha=args.coupling_alpha,
        beta=args.coupling_beta,
        lambda_decay=args.coupling_decay,
        min_shared_refs=args.coupling_min_shared,
        min_coupling_score=args.coupling_min_score,
        cache_dir=cache_dir,
        workers=max(0, int(args.coupling_workers)),
    )

    logger.info(
        "Coupling enabled: "
        f"alpha={coupling_cfg.alpha}, beta={coupling_cfg.beta}, "
        f"lambda={coupling_cfg.lambda_decay}, "
        f"min_shared={coupling_cfg.min_shared_refs}, "
        f"min_score={coupling_cfg.min_coupling_score}, "
        f"workers={coupling_cfg.workers}"
    )

    return coupling_cfg


def handle_settings_flow(args, logger: logging.Logger) -> dict[str, Any] | None:
    """
    Handle interactive or automated settings configuration.

    Args:
        args: Parsed command-line arguments
        logger: Logger instance

    Returns:
        Settings dictionary with validated mailto, or None if validation fails
    """
    settings = load_settings()

    # Support --interactive as alias for --configure (backward compatibility)
    configure_mode = args.configure or args.interactive

    if configure_mode:
        # Interactive configuration mode
        logger.info("Interactive configuration mode")
        settings["topics_id"] = ask("OpenAlex topics.id", settings["topics_id"])
        settings["from_date"] = ask("From publication date YYYY-MM-DD", settings["from_date"])
        settings["to_date"] = ask("To publication date YYYY-MM-DD", settings["to_date"])
        settings["max_records"] = ask("Max records (blank for uncapped)",
                                     str(settings["max_records"] if settings["max_records"] is not None else ""))
        settings["per_page"] = int(ask("Per page (1..200)", str(settings["per_page"])))
        gm = ask("Graph mode [none|annual|delta|both|cumulative]", settings["graph_mode"])
        settings["graph_mode"] = gm if gm in ("none", "annual", "delta", "both", "cumulative") else settings["graph_mode"]
        mailto_input = ask("Contact email for OpenAlex (mailto)", settings.get("mailto") or "")
        settings["mailto"] = mailto_input.strip() or None
        save_settings(settings)
        logger.info(f"Settings saved: {summary(settings)}")
    else:
        # Non-interactive mode: use saved settings or CLI args
        logger.info(f"Using saved settings: {summary(settings)}")

    # Validate authentication (mailto or API key)
    mailto_effective = args.mailto.strip() if args.mailto else (settings.get("mailto") or "")
    api_key = os.environ.get("OPENALEX_API_KEY", "").strip()

    if not mailto_effective and not api_key:
        logger.error("Authentication required for OpenAlex API.")
        logger.error("Provide one of:")
        logger.error("  - OPENALEX_API_KEY in .env file or environment")
        logger.error("  - --mailto YOUR_EMAIL")
        logger.error("  - Interactive setup: python run.py --configure")
        return None

    if mailto_effective:
        settings["mailto"] = mailto_effective
        if args.mailto:
            save_settings(settings)

    if api_key:
        settings["api_key"] = api_key

    # Handle incremental ingestion: adjust from_date to watermark
    if getattr(args, "since", None):
        args.incremental = True
    if getattr(args, "incremental", False):
        incremental_from = getattr(args, "since", None)
        if not incremental_from:
            incremental_from = settings.get("last_ingested_date")
        if incremental_from:
            logger.info(
                "Incremental mode: fetching works since %s (was %s)",
                incremental_from,
                settings["from_date"],
            )
            settings["_original_from_date"] = settings["from_date"]
            settings["from_date"] = incremental_from
        else:
            logger.warning(
                "Incremental mode requested but no watermark found. "
                "Running full ingestion."
            )

    settings["source_overrides"] = build_source_overrides(settings)

    # Let datasource config filters take precedence over saved settings.
    # This enables multi-domain support (e.g., CRISPR config overrides PSC topic ID).
    ds_cfg = yaml.safe_load(pathlib.Path(args.config).read_text())
    ds_filters = ds_cfg.get("sources", {}).get("primary", {}).get("filters", {})
    if ds_filters:
        settings["source_overrides"]["filters"].update(ds_filters)

    # Set effective graph mode
    if args.graph_mode is None:
        args.graph_mode = settings.get("graph_mode", "cumulative")
    logger.info(f"Effective graph_mode: {args.graph_mode}")

    return settings


@memory_monitor
def run_ingest_phase(
    args,
    logger: logging.Logger,
    ingest_dir: pathlib.Path,
    raw_dir: pathlib.Path,
    cache_path: pathlib.Path,
    settings: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any] | None, dict[str, Any]] | None:
    """
    Execute ingest phase: rebuild from raw, skip ingest, or fresh fetch.

    Args:
        args: Parsed command-line arguments
        logger: Logger instance
        ingest_dir: Ingest directory path
        raw_dir: Raw data directory path
        cache_path: Path to cached ingest.parquet
        settings: Pipeline settings dictionary (source_overrides, mailto, etc.)

    Returns:
        Tuple of (DataFrame, raw_manifest_rel, dedup_stats) or None if error
        where dedup_stats contains deduplication statistics for the manifest
    """
    raw_manifest_abs: dict[str, Any] | None = None
    raw_manifest_rel: dict[str, Any] | None = None
    dedup_stats: dict[str, Any] = {}

    # Mode 1: Rebuild from raw snapshot
    if args.rebuild_ingest_from_raw:
        try:
            log_section(logger, "Ingest Phase: Rebuilding from Raw Snapshot")
            logger.info("Rebuilding cached dataset from raw snapshot...")
            df_raw, raw_manifest_abs = rebuild_ingest_from_raw(raw_dir, args.raw_manifest)
        except Exception as exc:
            logger.error(f"Failed to rebuild from raw: {exc}", exc_info=True)
            return None

        df = df_raw
        df = add_time_vars(df)
        df = enforce_schema(df, args.schema)
        # HP-5: Deduplicate with statistics tracking
        df, dedup_stats = deduplicate_efficiently(df, "work_id", logger)

        try:
            df.to_parquet(cache_path, index=False)
            logger.info(f"Cached dataset to {cache_path}")
        except Exception as e:
            logger.warning(f"Failed to cache dataset: {e}")

        if raw_manifest_abs:
            raw_manifest_rel = relativize_raw_manifest(raw_manifest_abs, ingest_dir)

    # Mode 2: Skip ingest (use cached)
    elif args.skip_ingest:
        if not cache_path.exists():
            logger.error(f"Cannot skip ingest: missing {cache_path}")
            return None

        log_section(logger, "Ingest Phase: Using Cached Dataset")
        logger.info(f"Using cached dataset: {cache_path}")
        df = pd.read_parquet(cache_path)
        df = add_time_vars(df)
        df = enforce_schema(df, args.schema)
        # HP-5: Deduplicate with statistics tracking
        df, dedup_stats = deduplicate_efficiently(df, "work_id", logger)

    # Mode 3: Fresh fetch from OpenAlex
    else:
        is_incremental = getattr(args, "incremental", False)
        if is_incremental:
            log_section(logger, "Ingest Phase: Incremental Fetch from OpenAlex")
        else:
            log_section(logger, "Ingest Phase: Fetching from OpenAlex")

        df_new, raw_records = ingest(
            args.config,
            source_overrides=settings.get("source_overrides"),
        )

        if not args.skip_raw and raw_records:
            raw_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
            raw_basename = f"{args.raw_basename}_{timestamp}"
            raw_metadata = collect_raw_metadata(
                pathlib.Path(args.config),
                source_overrides=settings.get("source_overrides"),
            )
            raw_manifest_abs = write_raw_chunks(
                raw_records,
                outdir=raw_dir,
                basename=raw_basename,
                chunk_size=args.raw_chunk_size,
                compression=args.raw_compression,
                metadata=raw_metadata,
            )
            logger.info(
                f"Raw snapshot captured {raw_manifest_abs['records']} records "
                f"in {len(raw_manifest_abs['chunks'])} chunk(s) under {raw_dir}"
            )

        raw_records = None
        df_new = add_time_vars(df_new)
        df_new = enforce_schema(df_new, args.schema)

        # Incremental: merge new data with cached data
        if is_incremental and cache_path.exists():
            logger.info("Merging incremental data with cached dataset...")
            df_cached = pd.read_parquet(cache_path)
            n_cached = len(df_cached)
            df = pd.concat([df_cached, df_new], ignore_index=True)
            logger.info(
                "Merged: %d cached + %d new = %d total (before dedup)",
                n_cached, len(df_new), len(df),
            )
        else:
            df = df_new

        # HP-5: Deduplicate with statistics tracking
        df, dedup_stats = deduplicate_efficiently(df, "work_id", logger)

        try:
            df.to_parquet(cache_path, index=False)
            logger.info(f"Cached dataset to {cache_path}")
        except Exception as e:
            logger.warning(f"Failed to cache dataset: {e}")

        # Update watermark for incremental ingestion
        if "publication_date" in df.columns:
            max_date = df["publication_date"].max()
            if pd.notna(max_date):
                watermark = str(max_date)[:10]
                settings["last_ingested_date"] = watermark
                save_settings(settings)
                logger.info("Updated ingestion watermark: %s", watermark)

        if raw_manifest_abs:
            raw_manifest_rel = relativize_raw_manifest(raw_manifest_abs, ingest_dir)

    return df, raw_manifest_rel, dedup_stats


def run_slicing_phase(df: pd.DataFrame, args, logger: logging.Logger) -> dict[str, pd.DataFrame]:
    """
    Apply temporal/categorical slicing to DataFrame.

    Args:
        df: Input DataFrame
        args: Parsed command-line arguments
        logger: Logger instance

    Returns:
        Dictionary mapping slice names to DataFrames
    """
    cutoff = pd.Timestamp(args.cutoff) if args.cutoff else None
    log_section(logger, "Slicing Phase")
    logger.info(f"Applying slices with cutoff: {cutoff}")
    sliced = apply_slices(df, args.slices, cutoff=cutoff)
    logger.info(f"Generated {len(sliced)} slices")
    return sliced


def extract_time_periods(df: pd.DataFrame) -> tuple[list[int], list[str]]:
    """
    Extract sorted lists of years and quarters from DataFrame.

    Args:
        df: DataFrame with pub_year and pub_qtr columns

    Returns:
        Tuple of (years, quarters) as sorted lists
    """
    if "pub_year" in df.columns and "pub_qtr" in df.columns:
        years = sorted([int(y) for y in df["pub_year"].dropna().unique()])
        quarters = sorted(df["pub_qtr"].dropna().unique())
    else:
        years, quarters = [], []
    return years, quarters


# ============================================================================
# Parallel Graph Building Workers (PERF-1)
# ============================================================================

def _build_annual_graph_worker(args_tuple):
    """Worker function for parallel annual graph building."""
    df, year, graphs_dir, coupling_cfg = args_tuple
    try:
        base = export_annual_full(df, year=int(year), outdir=graphs_dir, coupling=coupling_cfg)
        return ("success", year, str(base.name))
    except Exception as e:
        return ("error", year, str(e))


def _build_delta_graph_worker(args_tuple):
    """Worker function for parallel delta graph building."""
    df, quarter, graphs_dir, coupling_cfg = args_tuple
    try:
        y, qq = quarter.split("Q")
        base = export_quarter_delta(
            df,
            year=int(y),
            quarter=int(qq),
            outdir=graphs_dir,
            coupling=coupling_cfg,
        )
        return ("success", quarter, str(base.name))
    except Exception as e:
        return ("error", quarter, str(e))


def _build_cumulative_graph_worker(args_tuple):
    """Worker function for parallel cumulative graph building."""
    df_sorted, quarter, graphs_dir, coupling_cfg, skip_graphml, graphml_compression, resume_graphs = args_tuple
    try:
        graphml_required = not skip_graphml
        compression_value = None if graphml_compression == "none" else graphml_compression

        # Check for resume capability
        existing_pkl = graphs_dir / f"citation_graph_cumulative_{quarter}.pkl"
        existing_graphml = graphs_dir / f"citation_graph_cumulative_{quarter}.graphml"
        graphml_gz = existing_graphml.with_suffix(".graphml.gz")
        has_graphml = existing_graphml.exists() or graphml_gz.exists()
        built_already = existing_pkl.exists() and (has_graphml or not graphml_required)

        if resume_graphs and built_already:
            return ("skipped", quarter, f"citation_graph_cumulative_{quarter}")

        if resume_graphs and existing_pkl.exists() and graphml_required and not has_graphml:
            # Regenerate GraphML
            pass

        y, qq = quarter.split("Q")
        period = pd.Period(quarter, freq="Q")
        cutoff_q = period.to_timestamp(how="end")
        sub = df_sorted[df_sorted["publication_date"] <= cutoff_q]
        Gcum = build_direct_citation_graph(sub, coupling=coupling_cfg)
        base = graphs_dir / f"citation_graph_cumulative_{quarter}"
        save_graph(
            Gcum,
            base,
            write_graphml=not skip_graphml,
            graphml_compression=compression_value,
        )
        return ("success", quarter, str(base.name))
    except Exception as e:
        return ("error", quarter, str(e))


def _should_export_graphml(args, mode: str) -> bool:
    """
    Determine if GraphML should be exported for a given graph mode.

    Args:
        args: Parsed command-line arguments
        mode: Graph mode ("annual", "delta", or "cumulative")

    Returns:
        True if GraphML export is enabled for this mode
    """
    # Legacy --skip-graphml flag (backward compatibility)
    if args.skip_graphml:
        return False

    # New --graphml-for flag takes precedence
    if args.graphml_for is not None:
        enabled_modes = [m.strip().lower() for m in args.graphml_for.split(",")]
        return mode.lower() in enabled_modes

    # Default: no GraphML export (pickle only)
    return False


@memory_monitor
def build_graphs_phase(
    args,
    df: pd.DataFrame,
    years: list[int],
    quarters: list[str],
    graphs_dir: pathlib.Path,
    coupling_cfg: CouplingConfig | None,
    logger: logging.Logger
) -> dict[str, Any]:
    """
    Build annual, delta, and/or cumulative graphs based on graph_mode.
    Uses parallel processing when multiple graphs need to be built (PERF-1).

    Args:
        args: Parsed command-line arguments
        df: DataFrame with publication data
        years: List of years to process
        quarters: List of quarters to process
        graphs_dir: Directory to save graphs
        coupling_cfg: Coupling configuration (if enabled)
        logger: Logger instance

    Returns:
        Manifest dictionary with graph metadata
    """
    # PERF-3: Check memory before building graphs
    if not check_memory_availability(logger):
        logger.warning("Proceeding with graph building despite low memory warning")

    # PERF-3: Suggest chunking strategy if dataset is large
    n_rows = len(df)
    if n_rows > 100_000:
        from src.memory_utils import get_memory_info
        mem_info = get_memory_info()
        strategy = suggest_chunking_strategy(n_rows, mem_info.get('available', 8.0))
        if strategy['should_chunk']:
            logger.info(f"Chunking suggestion: {strategy['reason']}")

    manifest: dict[str, Any] = {}
    max_workers = getattr(args, 'graph_workers', DEFAULT_PARALLEL_WORKERS)

    # Adaptive worker scaling for graph-level parallelization
    if max_workers > 1 and len(quarters) > 1:
        mem_info = get_memory_info()
        available_gb = mem_info.get("available", 10.0)
        coupling_workers_count = coupling_cfg.workers if (coupling_cfg and coupling_cfg.enabled) else 0

        # Use quarters for estimation (most common case: cumulative graphs)
        num_graphs_to_build = len(quarters) if args.graph_mode == "cumulative" else len(years)

        adjusted_workers, reason = suggest_graph_worker_count(
            available_gb=available_gb,
            total_works=len(df),
            num_graphs=num_graphs_to_build,
            max_workers=max_workers,
            coupling_workers=coupling_workers_count,
            logger=logger
        )

        if adjusted_workers != max_workers:
            logger.warning(f"Adaptive graph worker scaling: {reason}")
            max_workers = adjusted_workers
        else:
            logger.info(f"Graph worker count: {max_workers} ({reason})")

    # CRITICAL: Force sequential mode if coupling cache is enabled
    # Parallel graph building with shared coupling cache causes race conditions
    # where multiple workers overwrite each other's cache, corrupting incremental builds
    if coupling_cfg and coupling_cfg.enabled and coupling_cfg.cache_dir is not None and max_workers > 1:
        logger.warning(
            "COUPLING CACHE SAFETY: Forcing graph_workers=1 (sequential mode) "
            "to prevent cache corruption. Parallel graph building with shared coupling cache "
            "causes race conditions where workers overwrite each other's cache. "
            "To use parallel building: disable coupling cache (set cache_dir=null) or "
            "disable coupling entirely. See GRAPH_PARALLELIZATION_ANALYSIS.md for details."
        )
        max_workers = 1

    # Build annual graphs (parallelized)
    if args.graph_mode in ("annual", "both"):
        log_section(logger, "Graph Building Phase: Annual")
        export_graphml_annual = _should_export_graphml(args, "annual")
        graphml_status = "enabled" if export_graphml_annual else "disabled"
        logger.info(f"Building annual graphs for {len(years)} years (parallel workers: {max_workers}, GraphML: {graphml_status})")

        if len(years) == 1 or max_workers == 1:
            # Single graph or sequential mode
            for y in years:
                logger.info(f"Building annual graph for year {y}")
                base = export_annual_full(df, year=int(y), outdir=graphs_dir, coupling=coupling_cfg)
                manifest.setdefault("graphs", {}).setdefault("annual", []).append(str(base.name))
        else:
            # Parallel mode
            tasks = [(df, y, graphs_dir, coupling_cfg) for y in years]
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(_build_annual_graph_worker, task): task[1] for task in tasks}
                for future in as_completed(futures):
                    year = futures[future]
                    try:
                        status, y, result = future.result()
                        if status == "success":
                            logger.info(f"Completed annual graph for year {y}")
                            manifest.setdefault("graphs", {}).setdefault("annual", []).append(result)
                        else:
                            logger.error(f"Failed to build annual graph for year {y}: {result}")
                    except Exception as exc:
                        logger.error(f"Annual graph for year {year} generated an exception: {exc}")

    # Build delta graphs (parallelized)
    if args.graph_mode in ("delta", "both"):
        log_section(logger, "Graph Building Phase: Delta")
        export_graphml_delta = _should_export_graphml(args, "delta")
        graphml_status = "enabled" if export_graphml_delta else "disabled"
        logger.info(f"Building delta graphs for {len(quarters)} quarters (parallel workers: {max_workers}, GraphML: {graphml_status})")

        if len(quarters) == 1 or max_workers == 1:
            # Single graph or sequential mode
            for q in quarters:
                try:
                    y, qq = q.split("Q")
                    base = export_quarter_delta(
                        df,
                        year=int(y),
                        quarter=int(qq),
                        outdir=graphs_dir,
                        coupling=coupling_cfg,
                    )
                    manifest.setdefault("graphs", {}).setdefault("delta", []).append(str(base.name))
                except Exception:
                    pass
        else:
            # Parallel mode
            tasks = [(df, q, graphs_dir, coupling_cfg) for q in quarters]
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(_build_delta_graph_worker, task): task[1] for task in tasks}
                for future in as_completed(futures):
                    quarter = futures[future]
                    try:
                        status, q, result = future.result()
                        if status == "success":
                            logger.info(f"Completed delta graph for quarter {q}")
                            manifest.setdefault("graphs", {}).setdefault("delta", []).append(result)
                        else:
                            logger.warning(f"Failed to build delta graph for quarter {q}: {result}")
                    except Exception as exc:
                        logger.error(f"Delta graph for quarter {quarter} generated an exception: {exc}")

    # Build cumulative graphs (parallelized)
    if args.graph_mode == "cumulative":
        log_section(logger, "Graph Building Phase: Cumulative")
        df_sorted = df.sort_values("publication_date")

        # Check for existing graphs (for resume capability)
        existing_cumulative: set[str] = set()
        if args.resume_graphs:
            existing_cumulative = {
                p.stem.replace("citation_graph_cumulative_", "")
                for p in graphs_dir.glob("citation_graph_cumulative_*.pkl")
                if p.is_file()
            }
            if existing_cumulative:
                logger.info(f"Resume enabled; found {len(existing_cumulative)} existing cumulative graphs")
            else:
                logger.info("Resume enabled but no existing cumulative graphs found")

        export_graphml_cumulative = _should_export_graphml(args, "cumulative")
        graphml_status = "enabled" if export_graphml_cumulative else "disabled"
        logger.info(f"Building cumulative graphs for {len(quarters)} quarters (parallel workers: {max_workers}, GraphML: {graphml_status})")

        if len(quarters) == 1 or max_workers == 1:
            # Single graph or sequential mode
            graphml_required = export_graphml_cumulative
            compression_value = None if args.graphml_compression == "none" else args.graphml_compression

            for q in quarters:
                existing_pkl = graphs_dir / f"citation_graph_cumulative_{q}.pkl"
                existing_graphml = graphs_dir / f"citation_graph_cumulative_{q}.graphml"
                graphml_gz = existing_graphml.with_suffix(".graphml.gz")
                has_graphml = existing_graphml.exists() or graphml_gz.exists()
                built_already = existing_pkl.exists() and (has_graphml or not graphml_required)

                if args.resume_graphs and q in existing_cumulative and built_already:
                    manifest.setdefault("graphs", {}).setdefault("cumulative", []).append(f"citation_graph_cumulative_{q}")
                    logger.debug(f"{q}: skipping (already built)")
                    continue

                if args.resume_graphs and q in existing_cumulative and existing_pkl.exists() and graphml_required and not has_graphml:
                    logger.info(f"{q}: pickle exists but GraphML missing; regenerating")

                y, qq = q.split("Q")
                period = pd.Period(q, freq="Q")
                cutoff_q = period.to_timestamp(how="end")
                sub = df_sorted[df_sorted["publication_date"] <= cutoff_q]

                # Log memory before building graph
                log_memory_usage(logger, f"Before building cumulative graph {q} ({len(sub)} works)")

                logger.info(f"Building cumulative graph for {q} with {len(sub)} works")
                if coupling_cfg and coupling_cfg.enabled:
                    logger.info(f"Coupling enabled with {coupling_cfg.workers} workers")

                Gcum = build_direct_citation_graph(sub, coupling=coupling_cfg)
                base = graphs_dir / f"citation_graph_cumulative_{q}"
                save_graph(
                    Gcum,
                    base,
                    write_graphml=export_graphml_cumulative,
                    graphml_compression=compression_value,
                )

                # Log memory after saving graph
                log_memory_usage(logger, f"After saving cumulative graph {q}")

                manifest.setdefault("graphs", {}).setdefault("cumulative", []).append(str(base.name))
        else:
            # Parallel mode
            skip_graphml_value = not export_graphml_cumulative  # Worker expects skip flag
            tasks = [
                (df_sorted, q, graphs_dir, coupling_cfg, skip_graphml_value, args.graphml_compression, args.resume_graphs)
                for q in quarters
            ]
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(_build_cumulative_graph_worker, task): task[1] for task in tasks}
                for future in as_completed(futures):
                    quarter = futures[future]
                    try:
                        status, q, result = future.result()
                        if status == "success":
                            logger.info(f"Completed cumulative graph for quarter {q}")
                            manifest.setdefault("graphs", {}).setdefault("cumulative", []).append(result)
                        elif status == "skipped":
                            logger.debug(f"{q}: skipping (already built)")
                            manifest.setdefault("graphs", {}).setdefault("cumulative", []).append(result)
                        else:
                            logger.error(f"Failed to build cumulative graph for quarter {q}: {result}")
                    except Exception as exc:
                        logger.error(f"Cumulative graph for quarter {quarter} generated an exception: {exc}")

    return manifest


def save_slices_with_stats(
    sliced: dict[str, pd.DataFrame],
    slices_dir: pathlib.Path,
    ingest_dir: pathlib.Path,
    all_ids: set
) -> dict[str, Any]:
    """
    Save slices to parquet files with reference resolution statistics.

    Args:
        sliced: Dictionary of slice name to DataFrame
        slices_dir: Directory to save slice parquet files
        ingest_dir: Base ingest directory for relativizing paths
        all_ids: Set of all work IDs in corpus for ref resolution

    Returns:
        Manifest dictionary with slice metadata
    """
    manifest: dict[str, Any] = {}

    for name, sdf in sliced.items():
        p = slices_dir / f"{name}.parquet"
        sdf.to_parquet(p, index=False)
        stats = _ref_resolution_for_slice(p, all_ids)
        try:
            rel = p.relative_to(ingest_dir)
            file_ref = str(rel)
        except ValueError:
            file_ref = str(p)
        manifest[name] = {"rows": int(len(sdf)), "file": file_ref, "refs": stats}

    return manifest


def handle_raw_manifest(
    raw_manifest_rel: dict[str, Any] | None,
    prior_raw: dict[str, Any] | None,
    raw_dir: pathlib.Path,
    ingest_dir: pathlib.Path
) -> dict[str, Any] | None:
    """
    Determine which raw manifest to use (current, prior, or fallback).

    Args:
        raw_manifest_rel: Current run's relativized raw manifest (if any)
        prior_raw: Prior raw manifest from existing manifest.json (if any)
        raw_dir: Raw data directory
        ingest_dir: Ingest directory for relativizing paths

    Returns:
        Raw manifest dictionary or None
    """
    if raw_manifest_rel:
        return raw_manifest_rel
    elif prior_raw:
        return prior_raw
    else:
        fallback_raw_abs = load_latest_raw_manifest(raw_dir)
        if fallback_raw_abs:
            return relativize_raw_manifest(fallback_raw_abs, ingest_dir)
    return None


def run_community_detection(
    args,
    graphs_dir: pathlib.Path,
    outdir: pathlib.Path,
    logger: logging.Logger
) -> dict[str, str]:
    """
    Run community detection (cumulative + optional additional modes).

    Args:
        args: Parsed command-line arguments
        graphs_dir: Directory containing graph files
        outdir: Output directory for community results
        logger: Logger instance

    Returns:
        Manifest dictionary with community detection outputs
    """
    communities_script = pathlib.Path(__file__).parent / "scripts" / "communities.py"
    comm_manifest: dict[str, str] = {}

    # Always run cumulative
    cumulative_cmd = [
        sys.executable,
        str(communities_script),
        "--mode",
        "cumulative",
        "--graphs-dir",
        str(graphs_dir),
        "--out-dir",
        str(outdir),
        "--resume",
    ]
    try:
        log_section(logger, "Community Detection Phase: Cumulative")
        subprocess.check_call(cumulative_cmd)
        if (outdir / "communities_cumulative.json").exists():
            comm_manifest["cumulative"] = "communities_cumulative.json"
        if (outdir / "front_id_registry_cumulative.json").exists():
            comm_manifest["front_id_registry_cumulative"] = "front_id_registry_cumulative.json"
        if (outdir / "front_metrics_cumulative.csv").exists():
            comm_manifest["front_metrics_cumulative"] = "front_metrics_cumulative.csv"
        if (outdir / "front_timeseries_cumulative.csv").exists():
            comm_manifest["front_timeseries_cumulative"] = "front_timeseries_cumulative.csv"
        if (outdir / "communities_cumulative_summary.json").exists():
            comm_manifest["cumulative_summary"] = "communities_cumulative_summary.json"
        logger.info("Cumulative community detection completed successfully")
    except subprocess.CalledProcessError as e:
        logger.error(f"communities.py cumulative run failed: {e}", exc_info=True)

    # Optional additional modes
    if args.communities in ("annual", "delta", "both"):
        mode_map = {"annual": "annual", "delta": "delta", "both": "both"}
        call = [
            sys.executable,
            str(communities_script),
            "--mode",
            mode_map[args.communities],
            "--graphs-dir",
            str(graphs_dir),
            "--out-dir",
            str(outdir),
            "--align-deltas",
            "preceding",
        ]
        try:
            log_section(logger, f"Community Detection Phase: {args.communities.title()}")
            subprocess.check_call(call)
            if (outdir / "communities_annual.json").exists():
                comm_manifest["annual"] = "communities_annual.json"
            if (outdir / "communities_delta.json").exists():
                comm_manifest["delta"] = "communities_delta.json"
            logger.info(f"{args.communities} community detection completed successfully")
        except subprocess.CalledProcessError as e:
            logger.error(f"communities.py {args.communities} mode failed: {e}", exc_info=True)

    return comm_manifest


def write_pipeline_manifest(
    manifest_path: pathlib.Path,
    manifest: dict[str, Any],
    logger: logging.Logger
) -> None:
    """
    Write pipeline manifest and log summary.

    Args:
        manifest_path: Path to write manifest.json
        manifest: Manifest dictionary
        logger: Logger instance
    """
    manifest_path.write_text(json.dumps(manifest, indent=2))

    summary_bits = []
    raw_meta = manifest.get("raw")
    if isinstance(raw_meta, dict):
        raw_records = raw_meta.get("records")
        raw_chunks = len(raw_meta.get("chunks", [])) if raw_meta.get("chunks") else None
        if raw_records is not None:
            summary_bits.append(f"raw_records={raw_records}")
        if raw_chunks is not None:
            summary_bits.append(f"raw_chunks={raw_chunks}")

    graphs_meta = manifest.get("graphs", {})
    if isinstance(graphs_meta, dict):
        for key, entries in graphs_meta.items():
            with contextlib.suppress(Exception):
                summary_bits.append(f"{key}={len(entries)}")

    logger.info(f"Wrote manifest: {manifest_path} " + (" ".join(summary_bits) if summary_bits else ""))


# ============================================================================
# MAIN ORCHESTRATION
# ============================================================================

def main():
    """
    Main pipeline orchestration function.

    Coordinates all phases of the 2YP pipeline:
    1. Setup (logging, directories, configuration)
    2. Settings management
    3. Preflight checks (optional)
    4. Ingest (rebuild/skip/fresh fetch)
    5. Slicing
    6. Graph building (annual/delta/cumulative)
    7. Community detection
    8. Manifest writing
    9. Archival (optional)
    """
    # Parse arguments and initialize output directory
    args = parse_args()

    # Resolve --domain to --config if specified
    if args.domain is not None:
        from src.domain_registry import resolve_domain_args
        args.config = resolve_domain_args(
            args.domain, args.config,
            project_root=pathlib.Path(__file__).resolve().parent,
        )
    elif args.config is None:
        print("Error: either --domain or --config must be specified.")
        sys.exit(1)

    outdir = pathlib.Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Setup logging
    log_level = "DEBUG" if args.verbose else args.log_level
    log_file = args.log_file or (outdir / "logs" / f"pipeline_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.log")
    logger = setup_logging(level=log_level, log_file=log_file, console=True)

    logger.info("=" * 70)
    logger.info("  2YP Research-Front Monitoring Pipeline")
    logger.info("=" * 70)
    logger.info(f"Log level: {log_level}")
    logger.info(f"Log file: {log_file}")
    logger.info("")

    # Load prior manifest for raw data continuity
    manifest_path = outdir / "manifest.json"
    prior_raw = None
    if manifest_path.exists():
        try:
            prior_manifest = json.loads(manifest_path.read_text())
            prior_raw = prior_manifest.get("raw")
        except Exception:
            prior_raw = None

    # Setup directories
    ingest_dir, slices_dir, graphs_dir, raw_dir = setup_directories(args)

    # Setup coupling configuration
    coupling_cfg = setup_coupling_config(args, logger)

    # Handle archive-only mode
    if args.archive_only:
        archive_current(ingest_dir, graphs_dir, outdir)
        return

    # Handle settings flow (interactive or use-last)
    settings = handle_settings_flow(args, logger)
    if settings is None:
        return  # Settings validation failed

    # Adjust flags for rebuild mode
    if args.rebuild_ingest_from_raw:
        args.skip_preflight = True
        args.skip_ingest = False

    # Config validation (LP-4)
    try:
        validate_configs(args, logger)
    except ValueError as e:
        logger.error(f"Startup aborted due to invalid configuration: {e}")
        return

    # Preflight checks
    if not args.skip_ingest and not args.skip_preflight:
        preflight_check(args, settings)
        logger.info("Preflight checks complete. Proceeding with ingest...")

    # --- PIPELINE EXECUTION ---

    # Phase 1: Ingest
    cache_path = ingest_dir / "ingest.parquet"
    result = run_ingest_phase(args, logger, ingest_dir, raw_dir, cache_path, settings)
    if result is None:
        return  # Ingest failed
    df, raw_manifest_rel, dedup_stats = result

    # Phase 2: Slicing
    sliced = run_slicing_phase(df, args, logger)

    # Phase 3: Extract time periods for graph building
    years, quarters = extract_time_periods(df)

    # Phase 4: Build graphs
    manifest: dict[str, Any] = {}
    if args.graph_mode in ("annual", "both", "cumulative", "delta"):
        graph_manifest = build_graphs_phase(args, df, years, quarters, graphs_dir, coupling_cfg, logger)
        manifest.update(graph_manifest)

    # Phase 5: Add coupling config to manifest
    if coupling_cfg:
        cache_stats = get_cache_size(coupling_cfg.cache_dir) if coupling_cfg.cache_dir else {"size_mb": 0.0, "file_count": 0}
        manifest["coupling"] = {
            "enabled": True,
            "alpha": coupling_cfg.alpha,
            "beta": coupling_cfg.beta,
            "lambda_decay": coupling_cfg.lambda_decay,
            "min_shared_refs": coupling_cfg.min_shared_refs,
            "min_coupling_score": coupling_cfg.min_coupling_score,
            "cache_dir": str(coupling_cfg.cache_dir) if coupling_cfg.cache_dir else None,
            "cache_size_mb": cache_stats["size_mb"],
            "cache_file_count": cache_stats["file_count"],
            "workers": coupling_cfg.workers,
        }

    # Phase 5b: Add deduplication statistics to manifest (HP-5)
    if dedup_stats:
        manifest["deduplication"] = dedup_stats

    # Phase 6: Save slices with reference resolution stats
    all_ids = set(df["work_id"].dropna().astype(str).tolist())
    slices_manifest = save_slices_with_stats(sliced, slices_dir, ingest_dir, all_ids)
    manifest["slices"] = slices_manifest

    # Phase 7: Handle raw manifest
    raw_manifest_final = handle_raw_manifest(raw_manifest_rel, prior_raw, raw_dir, ingest_dir)
    if raw_manifest_final:
        manifest["raw"] = raw_manifest_final

    # Phase 8: Run community detection
    comm_manifest: dict[str, Any] = {}
    if args.skip_communities:
        logger.info("Skipping community detection (flag --skip-communities)")
    else:
        comm_manifest = run_community_detection(args, graphs_dir, outdir, logger)
        if comm_manifest:
            manifest["communities"] = comm_manifest

    # Phase 9: Write final manifest
    write_pipeline_manifest(manifest_path, manifest, logger)

    # Phase 10: Archive if requested
    if args.archive:
        log_section(logger, "Archive Phase")
        archive_current(ingest_dir, graphs_dir, outdir)

    # Complete
    logger.info("=" * 70)
    logger.info("  Pipeline completed successfully")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
