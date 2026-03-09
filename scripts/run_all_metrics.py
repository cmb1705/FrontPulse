from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run all metric scripts against current or archived datasets."
    )

    # Archive mode (backward compatibility)
    parser.add_argument(
        "--archive-dir",
        type=Path,
        help="Archive root containing ingest/, graphs/, and out/ directories. "
             "If specified, overrides --ingest-dir, --graphs-dir, and --out-dir.",
    )

    # Individual directory overrides
    parser.add_argument(
        "--ingest-dir",
        type=Path,
        default=Path("data/current_ingest"),
        help="Directory containing slices/ and ingest.parquet (default: data/current_ingest)",
    )
    parser.add_argument(
        "--graphs-dir",
        type=Path,
        default=Path("data/current_graphs"),
        help="Directory containing graph pickle files (default: data/current_graphs)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/out"),
        help="Directory containing front_id_registry and cache (default: data/out)",
    )

    # Metrics output configuration
    parser.add_argument(
        "--metrics-out",
        type=Path,
        help="Output directory for metrics results. If not specified, uses <out-dir>/metrics/",
    )
    parser.add_argument(
        "--output-suffix",
        type=str,
        default="",
        help="Suffix to add to output filenames (e.g., '_min3_res0012')",
    )

    # Python executable
    parser.add_argument(
        "--python",
        type=str,
        default=sys.executable,
        help="Python executable to use when invoking metric scripts.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Determine data paths (archive mode or individual directories)
    if args.archive_dir:
        # Archive mode: construct paths from archive root
        archive_dir = args.archive_dir.resolve()
        if not archive_dir.exists():
            raise FileNotFoundError(f"Archive directory not found: {archive_dir}")

        ingest_dir = archive_dir / "ingest"
        graphs_dir = archive_dir / "graphs"
        out_dir = archive_dir / "out"

        # Default metrics output to archive/out/metrics/
        if args.metrics_out:
            metrics_out = args.metrics_out.resolve()
        else:
            metrics_out = (out_dir / "metrics").resolve()
    else:
        # Individual directory mode
        ingest_dir = args.ingest_dir.resolve()
        graphs_dir = args.graphs_dir.resolve()
        out_dir = args.out_dir.resolve()

        # Default metrics output to out_dir/metrics/
        if args.metrics_out:
            metrics_out = args.metrics_out.resolve()
        else:
            metrics_out = (out_dir / "metrics").resolve()

    # Verify required paths exist
    if not ingest_dir.exists():
        raise FileNotFoundError(f"Ingest directory not found: {ingest_dir}")
    if not graphs_dir.exists():
        raise FileNotFoundError(f"Graphs directory not found: {graphs_dir}")
    if not out_dir.exists():
        raise FileNotFoundError(f"Output directory not found: {out_dir}")

    # Construct specific paths for metrics
    slices_dir = ingest_dir / "slices"
    ingest_parquet = ingest_dir / "ingest.parquet"
    registry_path = out_dir / "front_id_registry_cumulative.json"
    cache_dir = out_dir / "cache_cum" / "partitions_cum"

    # Create metrics output directory
    metrics_out.mkdir(parents=True, exist_ok=True)

    # Determine output filename suffix
    suffix = args.output_suffix

    # Build commands for all 5 metric scripts
    commands = [
        [
            "scripts/metric_author_influx.py",
            "--slices-dir",
            str(slices_dir),
            "--out-dir",
            str(metrics_out),
            "--json-name",
            f"author_influx{suffix}.json",
            "--figure-name",
            f"author_influx{suffix}.png",
        ],
        [
            "scripts/metric_citation_velocity.py",
            "--slices-dir",
            str(slices_dir),
            "--out-dir",
            str(metrics_out),
            "--json-name",
            f"citation_velocity{suffix}.json",
            "--figure-name",
            f"citation_velocity{suffix}.png",
        ],
        [
            "scripts/metric_topic_diversity.py",
            "--slices-dir",
            str(slices_dir),
            "--out-dir",
            str(metrics_out),
            "--json-name",
            f"topic_diversity{suffix}.json",
            "--figure-name",
            f"topic_diversity{suffix}.png",
        ],
        [
            "scripts/metric_cross_cluster_bridging.py",
            "--graphs-dir",
            str(graphs_dir),
            "--out-dir",
            str(metrics_out),
            "--registry",
            str(registry_path),
            "--cache-dir",
            str(cache_dir),
            "--json-name",
            f"cross_cluster_bridging{suffix}.json",
            "--figure-name",
            f"cross_cluster_bridging{suffix}.png",
        ],
        [
            "scripts/metric_reference_vitality.py",
            "--slices-dir",
            str(slices_dir),
            "--ingest-path",
            str(ingest_parquet),
            "--out-dir",
            str(metrics_out),
            "--json-name",
            f"reference_vitality{suffix}.json",
            "--figure-name",
            f"reference_vitality{suffix}.png",
        ],
    ]

    # Print configuration summary
    print(f"[Config] Running metrics with:")
    print(f"  Ingest:  {ingest_dir}")
    print(f"  Graphs:  {graphs_dir}")
    print(f"  Output:  {out_dir}")
    print(f"  Metrics: {metrics_out}")
    print(f"  Suffix:  '{suffix}'" if suffix else "  Suffix:  (none)")
    print()

    # Run each metric script
    for cmd_args in commands:
        script_path = Path(cmd_args[0])
        if not script_path.exists():
            raise FileNotFoundError(f"Metric script missing: {script_path}")

        full_cmd = [args.python, str(script_path)] + cmd_args[1:]
        print(f"[Run] {script_path.name}")
        subprocess.run(full_cmd, check=True)
        print()

    print(f"[Done] All metrics written to {metrics_out}")


if __name__ == "__main__":
    main()
