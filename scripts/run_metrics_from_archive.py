from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run all metric scripts against an archived dataset."
    )
    parser.add_argument(
        "--archive-dir",
        type=Path,
        default=Path("data/archive/20251027_074256"),
        help="Archive root containing ingest/, graphs/, and out/ directories.",
    )
    parser.add_argument(
        "--out-subdir",
        type=str,
        default="metrics_archive",
        help="Subdirectory (or absolute path) for metric outputs. "
             "Relative paths are created under <archive>/out/.",
    )
    parser.add_argument(
        "--python",
        type=str,
        default=sys.executable,
        help="Python executable to use when invoking metric scripts.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    archive_dir: Path = args.archive_dir.resolve()
    if not archive_dir.exists():
        raise FileNotFoundError(f"Archive directory not found: {archive_dir}")

    ingest_dir = archive_dir / "ingest"
    graphs_dir = archive_dir / "graphs"
    out_dir = archive_dir / "out"

    slices_dir = ingest_dir / "slices"
    ingest_parquet = ingest_dir / "ingest.parquet"
    registry_path = out_dir / "front_id_registry_cumulative.json"
    cache_dir = out_dir / "cache_cum" / "partitions_cum"
    out_subdir_path = Path(args.out_subdir)
    if out_subdir_path.is_absolute():
        metrics_out = out_subdir_path
    else:
        metrics_out = (out_dir / out_subdir_path).resolve()
    metrics_out.mkdir(parents=True, exist_ok=True)

    commands = [
        [
            "scripts/metric_author_influx.py",
            "--slices-dir",
            str(slices_dir),
            "--out-dir",
            str(metrics_out),
            "--json-name",
            "author_influx_archive.json",
            "--figure-name",
            "author_influx_archive.png",
        ],
        [
            "scripts/metric_citation_velocity.py",
            "--slices-dir",
            str(slices_dir),
            "--out-dir",
            str(metrics_out),
            "--json-name",
            "citation_velocity_archive.json",
            "--figure-name",
            "citation_velocity_archive.png",
        ],
        [
            "scripts/metric_topic_diversity.py",
            "--slices-dir",
            str(slices_dir),
            "--out-dir",
            str(metrics_out),
            "--json-name",
            "topic_diversity_archive.json",
            "--figure-name",
            "topic_diversity_archive.png",
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
            "cross_cluster_bridging_archive.json",
            "--figure-name",
            "cross_cluster_bridging_archive.png",
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
            "reference_vitality_archive.json",
            "--figure-name",
            "reference_vitality_archive.png",
        ],
    ]

    for cmd_args in commands:
        script_path = Path(cmd_args[0])
        if not script_path.exists():
            raise FileNotFoundError(f"Metric script missing: {script_path}")
        full_cmd = [args.python, str(script_path)] + cmd_args[1:]
        print(f"[Run] {' '.join(full_cmd)}")
        subprocess.run(full_cmd, check=True)

    print(f"[Done] Metrics written to {metrics_out}")


if __name__ == "__main__":
    main()
