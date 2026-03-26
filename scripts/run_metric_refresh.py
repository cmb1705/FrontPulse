#!/usr/bin/env python
"""
Orchestrator for refreshing all scientometric metrics.

This script discovers and executes all metric_*.py scripts, validates their
outputs, and maintains the central manifest file.

Usage:
    python scripts/run_metric_refresh.py
    python scripts/run_metric_refresh.py --domain crispr
    python scripts/run_metric_refresh.py --limit-quarters 4
    python scripts/run_metric_refresh.py --metrics author_influx citation_velocity
    python scripts/run_metric_refresh.py --dry-run
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

# Windows-safe symbols
CHECK = "[OK]"
CROSS = "[FAIL]"

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.domain_registry import add_domain_args, resolve_script_paths  # noqa: E402
from src.metrics.common import verify_manifest_entry  # noqa: E402

# Metric scripts in execution order
METRIC_SCRIPTS = [
    "metric_author_influx.py",
    "metric_citation_velocity.py",
    "metric_reference_vitality.py",
    "metric_topic_diversity.py",
    "metric_cross_cluster_bridging.py",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Refresh all scientometric metrics and update manifest."
    )
    parser.add_argument(
        "--slices-dir",
        default=None,
        type=Path,
        help="Directory containing quarterly slice parquet files",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        type=Path,
        help="Output directory for metrics",
    )
    parser.add_argument(
        "--graphs-dir",
        default=None,
        type=Path,
        help="Directory containing graph files",
    )
    parser.add_argument(
        "--registry",
        default=None,
        type=Path,
        help="Path to front_id_registry_cumulative.json",
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        type=Path,
        help="Path to partition cache directory",
    )
    parser.add_argument(
        "--ingest-path",
        default=None,
        type=Path,
        help="Path to ingest.parquet",
    )
    parser.add_argument(
        "--limit-quarters",
        type=int,
        default=None,
        help="Limit processing to first N quarters (for testing)",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=None,
        help="Specific metrics to run (e.g., author_influx citation_velocity)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be executed without running",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only validate existing outputs against manifest",
    )
    add_domain_args(parser)
    return parser.parse_args()


def get_metric_name(script_path: str) -> str:
    """Extract metric name from script filename (e.g., metric_author_influx.py -> author_influx)."""
    stem = Path(script_path).stem
    if stem.startswith("metric_"):
        return stem[7:]  # Remove 'metric_' prefix
    return stem


def run_metric_script(
    script_path: Path,
    slices_dir: Path,
    out_dir: Path,
    limit_quarters: int | None = None,
    dry_run: bool = False,
    domain: str | None = None,
    graphs_dir: Path | None = None,
    registry: Path | None = None,
    cache_dir: Path | None = None,
    ingest_path: Path | None = None,
) -> dict[str, any]:
    """
    Execute a single metric script.

    Args:
        script_path: Path to metric script
        slices_dir: Input slices directory
        out_dir: Output directory
        limit_quarters: Optional limit on quarters processed
        dry_run: If True, don't actually execute
        domain: Optional domain identifier to forward
        graphs_dir: Optional graphs directory for cross_cluster_bridging
        registry: Optional registry path for cross_cluster_bridging
        cache_dir: Optional cache directory for cross_cluster_bridging
        ingest_path: Optional ingest parquet path for cross_cluster_bridging

    Returns:
        Dictionary with execution results
    """
    metric_name = get_metric_name(script_path.name)

    cmd = [
        sys.executable,
        str(script_path),
    ]

    if metric_name == "cross_cluster_bridging":
        # cross_cluster_bridging uses --graphs-dir, --registry, --cache-dir
        if graphs_dir is not None:
            cmd.append(f"--graphs-dir={graphs_dir}")
        if registry is not None:
            cmd.append(f"--registry={registry}")
        if cache_dir is not None:
            cmd.append(f"--cache-dir={cache_dir}")
        if ingest_path is not None:
            cmd.append(f"--ingest-path={ingest_path}")
    else:
        cmd.append(f"--slices-dir={slices_dir}")

    cmd.append(f"--out-dir={out_dir}")

    if limit_quarters is not None:
        cmd.append(f"--limit={limit_quarters}")

    # Forward --domain to child scripts
    if domain is not None:
        cmd.append(f"--domain={domain}")

    result = {
        "metric": metric_name,
        "script": script_path.name,
        "command": " ".join(cmd),
        "success": False,
        "duration_seconds": 0.0,
        "error": None,
    }

    if dry_run:
        print(f"[DRY RUN] Would execute: {result['command']}")
        result["success"] = True
        return result

    print(f"\n{'=' * 70}")
    print(f"Running: {metric_name}")
    print(f"Command: {result['command']}")
    print(f"{'=' * 70}")

    start_time = time.time()

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
            timeout=600,  # 10 minute timeout
        )
        result["success"] = True
        result["duration_seconds"] = time.time() - start_time

        # Print output
        if proc.stdout:
            print(proc.stdout)
        if proc.stderr:
            print(f"STDERR:\n{proc.stderr}", file=sys.stderr)

        print(f"{CHECK} {metric_name} completed in {result['duration_seconds']:.1f}s")

    except subprocess.CalledProcessError as e:
        result["error"] = str(e)
        result["duration_seconds"] = time.time() - start_time

        print(f"{CROSS} {metric_name} FAILED after {result['duration_seconds']:.1f}s")
        print(f"Error: {e}")
        if e.stdout:
            print(f"STDOUT:\n{e.stdout}")
        if e.stderr:
            print(f"STDERR:\n{e.stderr}", file=sys.stderr)

    except subprocess.TimeoutExpired as e:
        result["error"] = f"Timeout after {e.timeout}s"
        result["duration_seconds"] = time.time() - start_time
        print(f"{CROSS} {metric_name} TIMEOUT after {e.timeout}s")

    except Exception as e:
        result["error"] = str(e)
        result["duration_seconds"] = time.time() - start_time
        print(f"{CROSS} {metric_name} FAILED: {e}")

    return result


def validate_outputs(out_dir: Path, metric_names: list[str]) -> dict[str, any]:
    """
    Validate metric outputs against manifest.

    Args:
        out_dir: Output directory containing manifest
        metric_names: List of metric names to validate

    Returns:
        Dictionary with validation results
    """
    manifest_path = out_dir / "manifest.json"

    if not manifest_path.exists():
        return {
            "success": False,
            "error": f"Manifest not found: {manifest_path}",
            "validated": [],
        }

    results = {
        "success": True,
        "error": None,
        "validated": [],
    }

    print(f"\n{'=' * 70}")
    print("Validating outputs against manifest")
    print(f"{'=' * 70}")

    for metric_name in metric_names:
        try:
            verification = verify_manifest_entry(manifest_path, metric_name, "global")

            validated = {
                "metric": metric_name,
                "exists": verification["exists"],
                "parquet_valid": verification["parquet_valid"],
                "metadata_valid": verification["metadata_valid"],
                "errors": verification["errors"],
            }

            results["validated"].append(validated)

            if verification["parquet_valid"] and verification["metadata_valid"]:
                print(f"{CHECK} {metric_name}: valid")
            else:
                print(f"{CROSS} {metric_name}: INVALID")
                for error in verification["errors"]:
                    print(f"  - {error}")
                results["success"] = False

        except KeyError as e:
            print(f"{CROSS} {metric_name}: Not found in manifest")
            results["validated"].append({
                "metric": metric_name,
                "exists": False,
                "parquet_valid": False,
                "metadata_valid": False,
                "errors": [str(e)],
            })
            results["success"] = False

        except Exception as e:
            print(f"{CROSS} {metric_name}: Validation error - {e}")
            results["validated"].append({
                "metric": metric_name,
                "exists": False,
                "parquet_valid": False,
                "metadata_valid": False,
                "errors": [str(e)],
            })
            results["success"] = False

    return results


def main() -> None:
    args = parse_args()

    # Resolve domain paths (returns None when --domain is omitted)
    dpaths = resolve_script_paths(args, REPO_ROOT)

    # Apply domain defaults for paths not explicitly provided
    args.slices_dir = args.slices_dir or (dpaths.slices if dpaths else Path("data/current_ingest/slices"))
    args.out_dir = args.out_dir or (dpaths.out / "metrics" if dpaths else Path("data/out/metrics"))
    args.graphs_dir = args.graphs_dir or (dpaths.graphs if dpaths else Path("data/current_graphs"))
    args.registry = args.registry or (
        dpaths.out / "front_id_registry_cumulative.json"
        if dpaths
        else Path("data/out/front_id_registry_cumulative.json")
    )
    args.cache_dir = args.cache_dir or (
        dpaths.cache_cum / "partitions_cum" if dpaths else Path("data/out/cache_cum/partitions_cum")
    )
    args.ingest_path = args.ingest_path or (
        dpaths.ingest / "ingest.parquet"
        if dpaths
        else Path("data/current_ingest/ingest.parquet")
    )

    # Determine which metrics to run
    scripts_to_run = []
    if args.metrics:
        # Filter to requested metrics
        for metric_name in args.metrics:
            script_name = f"metric_{metric_name}.py"
            if script_name in METRIC_SCRIPTS:
                scripts_to_run.append(script_name)
            else:
                print(f"Warning: Unknown metric '{metric_name}', skipping")
    else:
        scripts_to_run = METRIC_SCRIPTS

    if not scripts_to_run:
        print("No metrics to run!")
        sys.exit(1)

    metric_names = [get_metric_name(s) for s in scripts_to_run]

    # Validation-only mode
    if args.validate_only:
        validation = validate_outputs(args.out_dir, metric_names)
        if validation["success"]:
            print(f"\n{CHECK} All validations passed")
            sys.exit(0)
        else:
            print(f"\n{CROSS} Validation failed")
            sys.exit(1)

    # Execute metrics
    print(f"\n{'=' * 70}")
    print("Metric Refresh Orchestrator")
    print(f"{'=' * 70}")
    print(f"Slices directory: {args.slices_dir}")
    print(f"Output directory: {args.out_dir}")
    if getattr(args, "domain", None):
        print(f"Domain: {args.domain}")
    print(f"Limit quarters: {args.limit_quarters if args.limit_quarters else 'None (all)'}")
    print(f"Metrics to run: {', '.join(metric_names)}")
    print(f"Dry run: {args.dry_run}")

    results = []
    scripts_dir = REPO_ROOT / "scripts"

    for script_name in scripts_to_run:
        script_path = scripts_dir / script_name
        if not script_path.exists():
            print(f"\n{CROSS} Script not found: {script_path}")
            results.append({
                "metric": get_metric_name(script_name),
                "script": script_name,
                "success": False,
                "error": "Script file not found",
            })
            continue

        result = run_metric_script(
            script_path,
            args.slices_dir,
            args.out_dir,
            args.limit_quarters,
            args.dry_run,
            domain=getattr(args, "domain", None),
            graphs_dir=args.graphs_dir,
            registry=args.registry,
            cache_dir=args.cache_dir,
            ingest_path=args.ingest_path,
        )
        results.append(result)

    # Summary
    print(f"\n{'=' * 70}")
    print("Summary")
    print(f"{'=' * 70}")

    successful = [r for r in results if r["success"]]
    failed = [r for r in results if not r["success"]]

    print(f"Total metrics: {len(results)}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")

    if successful:
        total_time = sum(r["duration_seconds"] for r in successful)
        print(f"Total execution time: {total_time:.1f}s")

    if failed:
        print("\nFailed metrics:")
        for r in failed:
            print(f"  {CROSS} {r['metric']}: {r.get('error', 'Unknown error')}")

    # Validate outputs (unless dry run)
    if not args.dry_run and successful:
        validation = validate_outputs(args.out_dir, metric_names)
        if not validation["success"]:
            print(f"\n{CROSS} Output validation failed")
            sys.exit(1)

    # Exit with error if any metric failed
    if failed:
        sys.exit(1)

    print(f"\n{CHECK} Metric refresh completed successfully")


if __name__ == "__main__":
    main()
