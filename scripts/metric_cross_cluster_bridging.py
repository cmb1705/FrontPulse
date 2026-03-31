from __future__ import annotations

import argparse
import gc
import json
from datetime import datetime, timezone
from multiprocessing import Pool, cpu_count
from pathlib import Path

from _path_bootstrap import ensure_repo_imports

REPO_ROOT = ensure_repo_imports()

import matplotlib.pyplot as plt  # noqa: E402
import networkx as nx  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src import trusted_io  # noqa: E402
from src.community import run_leiden  # noqa: E402
from src.domain_registry import (  # noqa: E402
    add_domain_args,
    apply_domain_path_defaults,
    resolve_script_paths,
)
from src.metrics.common import (  # noqa: E402
    create_metric_metadata,
    ensure_dir,
    get_metric_output_paths,
    list_quarter_files,
    update_manifest,
    write_metric_metadata,
    write_metric_parquet,
    write_placeholder_metric,
)

try:
    from src.memory_utils import get_memory_info  # noqa: E402

    MEMORY_UTILS_AVAILABLE = True
except ImportError:
    MEMORY_UTILS_AVAILABLE = False


DEFAULT_MAX_WORKERS = 12
DEFAULT_MEMORY_RESERVE_GB = 8.0
LARGE_GRAPH_BYTES = 100 * 1024 * 1024
VERY_LARGE_GRAPH_BYTES = 175 * 1024 * 1024
ESTIMATED_GRAPH_EXPANSION_MULTIPLIER = 16.0
MIN_ESTIMATED_WORKER_MEMORY_GB = 2.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Quantify cross-cluster bridging nodes over time.")
    parser.add_argument("--graphs-dir", default=None, type=Path)
    parser.add_argument("--out-dir", default=None, type=Path)
    parser.add_argument("--registry", default=None, type=Path)
    parser.add_argument("--cache-dir", default=None, type=Path)
    parser.add_argument("--ingest-path", default=None, type=Path)
    parser.add_argument("--json-name", default="cross_cluster_bridging.json")
    parser.add_argument("--figure-name", default="cross_cluster_bridging.png")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resolution", type=float, default=1.0)
    parser.add_argument("--min-size", type=int, default=50)
    parser.add_argument("--max-size", type=int, default=5000)
    parser.add_argument("--min-degree", type=int, default=3)
    parser.add_argument("--top-nodes", type=int, default=10)
    parser.add_argument("--high-ratio-threshold", type=float, default=0.5)
    parser.add_argument("--max-workers", type=int, default=DEFAULT_MAX_WORKERS)
    parser.add_argument("--memory-reserve-gb", type=float, default=DEFAULT_MEMORY_RESERVE_GB)
    add_domain_args(parser)
    return parser.parse_args()


def load_registry(path: Path) -> dict[str, dict[str, int]]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text())
    registry: dict[str, dict[str, int]] = {}
    for quarter, mapping in data.items():
        registry[quarter] = {str(k): int(v) for k, v in mapping.items()}
    return registry


def load_cached_partition(cache_dir: Path, quarter: str) -> dict[str, int] | None:
    if not cache_dir or not cache_dir.exists():
        return None
    path = cache_dir / f"part_{quarter}.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    raw = data.get("labels", {})
    return {str(node): int(label) for node, label in raw.items() if label is not None}


def compute_partition(
    G: nx.DiGraph,
    quarter: str,
    args: argparse.Namespace,
    cache_dir: Path,
) -> dict[str, int]:
    cached = load_cached_partition(cache_dir, quarter)
    if cached is not None:
        return cached
    max_size = args.max_size if args.max_size > 0 else None
    result = run_leiden(G, resolution=args.resolution, min_size=args.min_size, max_size=max_size)
    return {str(node): int(label) for node, label in result["partition"]}


def build_undirected(G: nx.DiGraph) -> nx.Graph:
    """
    Return an undirected view without duplicating the graph in memory.

    Memory expectation:
        Uses a view over the original directed graph instead of allocating a
        second NetworkX graph, which avoids the extra O(|V| + |E|) copy that
        caused CRISPR-scale runs to exhaust RAM.
    """
    return nx.subgraph_view(
        G.to_undirected(as_view=True),
        filter_edge=lambda u, v: u != v,
    )


def suggest_worker_count(
    graph_paths: list[Path],
    requested_workers: int,
    memory_reserve_gb: float,
) -> tuple[int, str]:
    """
    Right-size quarter-level parallelism for memory-heavy bridging analysis.

    Memory expectation:
        Estimates each worker as one loaded NetworkX graph plus Leiden and
        traversal overhead. The heuristic is intentionally conservative because
        CRISPR cumulative graphs expand far beyond their on-disk pickle size.
    """
    if not graph_paths:
        return 1, "no graph files selected"

    max_workers = max(1, min(requested_workers, len(graph_paths), cpu_count()))
    if not MEMORY_UTILS_AVAILABLE:
        return max_workers, "psutil unavailable; using requested worker count"

    available_gb = get_memory_info().get("available")
    if available_gb is None:
        return max_workers, "memory info unavailable; using requested worker count"

    memory_reserve_gb = max(0.0, float(memory_reserve_gb))
    usable_gb = max(1.0, available_gb - memory_reserve_gb)
    largest_graph_bytes = max(
        (path.stat().st_size for path in graph_paths if path.exists()),
        default=0,
    )
    largest_graph_gb = largest_graph_bytes / (1024 ** 3)
    estimated_worker_memory_gb = max(
        MIN_ESTIMATED_WORKER_MEMORY_GB,
        largest_graph_gb * ESTIMATED_GRAPH_EXPANSION_MULTIPLIER,
    )
    memory_limited_workers = max(1, int(usable_gb // estimated_worker_memory_gb))

    size_cap = max_workers
    if largest_graph_bytes >= VERY_LARGE_GRAPH_BYTES:
        size_cap = min(size_cap, 2)
    elif largest_graph_bytes >= LARGE_GRAPH_BYTES:
        size_cap = min(size_cap, 4)

    adjusted_workers = max(1, min(max_workers, memory_limited_workers, size_cap))
    reason = (
        f"available={available_gb:.1f} GB, reserve={memory_reserve_gb:.1f} GB, "
        f"largest_graph={largest_graph_gb:.2f} GB, "
        f"estimated={estimated_worker_memory_gb:.2f} GB/worker"
    )
    return adjusted_workers, reason


def front_lookup(registry: dict[str, dict[str, int]], quarter: str, community: int) -> int | None:
    mapping = registry.get(quarter)
    if not mapping:
        return None
    return mapping.get(str(community))


def analyze_quarter(
    quarter: str,
    graph_path: Path,
    args: argparse.Namespace,
    registry: dict[str, dict[str, int]],
) -> dict[str, object]:
    Gfull: nx.DiGraph = trusted_io.load_trusted_binary(
        graph_path, description="citation graph",
    )

    partition_map = compute_partition(Gfull, quarter, args, args.cache_dir)
    if not partition_map:
        return {
            "quarter": quarter,
            "n_nodes_with_partition": 0,
            "mean_bridge_ratio": None,
            "median_bridge_ratio": None,
            "p90_bridge_ratio": None,
            "share_high_ratio": None,
            "cross_edge_ratio": None,
            "top_nodes": [],
        }
    G = build_undirected(Gfull)
    bridge_rows: list[dict[str, object]] = []
    considered_edges = 0
    external_edges = 0

    for u, v in G.edges():
        cu = partition_map.get(str(u))
        cv = partition_map.get(str(v))
        if cu is None or cv is None:
            continue
        considered_edges += 1
        if cu != cv:
            external_edges += 1

    for node, community in partition_map.items():
        if node not in G:
            continue
        degree = int(G.degree(node))
        if degree < args.min_degree:
            continue
        external = 0
        external_comms: set[int] = set()
        for nbr in G.neighbors(node):
            nbr_comm = partition_map.get(str(nbr))
            if nbr_comm is None:
                continue
            if nbr_comm != community:
                external += 1
                external_comms.add(int(nbr_comm))
        bridge_ratio = float(external / degree) if degree else 0.0
        node_data = G.nodes[node]
        bridge_rows.append(
            {
                "node_id": node,
                "community_id": int(community),
                "front_id": front_lookup(registry, quarter, community),
                "degree": degree,
                "external_degree": external,
                "external_communities": len(external_comms),
                "bridge_ratio": bridge_ratio,
                "publication_date": node_data.get("publication_date"),
            }
        )

    if not bridge_rows:
        return {
            "quarter": quarter,
            "n_nodes_with_partition": 0,
            "mean_bridge_ratio": None,
            "median_bridge_ratio": None,
            "p90_bridge_ratio": None,
            "share_high_ratio": None,
            "cross_edge_ratio": (external_edges / considered_edges) if considered_edges else None,
            "top_nodes": [],
        }

    df = pd.DataFrame(bridge_rows)
    mean_ratio = float(df["bridge_ratio"].mean())
    median_ratio = float(df["bridge_ratio"].median())
    p90_ratio = float(np.percentile(df["bridge_ratio"], 90))
    share_high = float((df["bridge_ratio"] >= args.high_ratio_threshold).mean())
    top_nodes = (
        df.sort_values(["bridge_ratio", "external_degree", "degree"], ascending=[False, False, False])
        .head(args.top_nodes)
        .to_dict(orient="records")
    )

    return {
        "quarter": quarter,
        "n_nodes_with_partition": int(len(df)),
        "mean_bridge_ratio": mean_ratio,
        "median_bridge_ratio": median_ratio,
        "p90_bridge_ratio": p90_ratio,
        "share_high_ratio": share_high,
        "cross_edge_ratio": (external_edges / considered_edges) if considered_edges else None,
        "top_nodes": top_nodes,
    }


def render_plot(payload: dict[str, object], out_path: Path) -> None:
    quarters = [row["quarter"] for row in payload["quarters"]]
    mean_ratios = [row["mean_bridge_ratio"] or 0 for row in payload["quarters"]]
    p90_ratios = [row["p90_bridge_ratio"] or 0 for row in payload["quarters"]]
    share_high = [row["share_high_ratio"] or 0 for row in payload["quarters"]]
    cross_edge_ratio = [row["cross_edge_ratio"] or 0 for row in payload["quarters"]]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    ax1.plot(quarters, mean_ratios, label="Mean bridge ratio", color="#1f77b4", linewidth=2)
    ax1.plot(quarters, p90_ratios, label="90th percentile", color="#ff7f0e", linewidth=2, linestyle="--")
    ax1.set_ylabel("Bridge ratio")
    ax1.set_title("Cross-Cluster Bridging Intensity")
    ax1.legend(loc="upper right")

    ax2.plot(quarters, cross_edge_ratio, label="Cross-community edge share", color="#9467bd", linewidth=2)
    ax2_2 = ax2.twinx()
    ax2_2.plot(quarters, share_high, label="Share of high-bridging nodes", color="#2ca02c", linewidth=1.8, linestyle="-.")
    ax2.set_ylabel("Edge share")
    ax2_2.set_ylabel("Node share")
    ax2.set_ylim(0, 1)
    ax2_2.set_ylim(0, 1)

    lines = ax2.get_lines() + ax2_2.get_lines()
    labels = [line.get_label() for line in lines]
    ax2.legend(lines, labels, loc="upper right")

    ax2.tick_params(axis="x", rotation=75)
    ax2.set_xlabel("Quarter")
    step = max(1, len(quarters) // 16)
    ax2.set_xticks(range(0, len(quarters), step))
    ax2.set_xticklabels([quarters[i] for i in range(0, len(quarters), step)])

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def analyze_quarter_wrapper(args_tuple):
    """Wrapper for parallel processing."""
    quarter, path, args, registry = args_tuple
    return analyze_quarter(quarter, path, args, registry)


def enrich_top_nodes_with_metadata(
    quarters: list[dict[str, object]],
    ingest_path: Path | None,
) -> None:
    """
    Add lightweight metadata to the final top-node summaries only.

    Memory expectation:
        Loads the selected ingest columns once in the parent process after all
        graph work completes, instead of repeating the same parquet read in
        every worker.
    """
    if not quarters or ingest_path is None or not ingest_path.exists():
        return

    top_node_ids = sorted({
        str(node["node_id"])
        for row in quarters
        for node in row.get("top_nodes", [])
        if node.get("node_id") is not None
    })
    if not top_node_ids:
        return

    metadata_df = pd.read_parquet(
        ingest_path,
        columns=["work_id", "title", "cited_by_count"],
    )
    metadata_df["work_id"] = metadata_df["work_id"].astype(str)
    metadata_lookup = (
        metadata_df.loc[metadata_df["work_id"].isin(top_node_ids)]
        .set_index("work_id")
        .to_dict("index")
    )

    for row in quarters:
        for node in row.get("top_nodes", []):
            node_metadata = metadata_lookup.get(str(node.get("node_id")), {})
            node["title"] = node_metadata.get("title")
            node["cited_by_count"] = node_metadata.get("cited_by_count")

    del metadata_df
    del metadata_lookup
    gc.collect()


def is_memory_pressure_error(exc: BaseException) -> bool:
    message = f"{type(exc).__name__}: {exc}".lower()
    return isinstance(exc, MemoryError) or any(
        marker in message
        for marker in (
            "memoryerror",
            "out of memory",
            "not enough memory",
            "paging file",
        )
    )


def run_quarter_analyses_with_backoff(
    quarter_args: list[tuple[str, Path, argparse.Namespace, dict[str, dict[str, int]]]],
    graph_paths: list[Path],
    requested_workers: int,
    memory_reserve_gb: float,
) -> tuple[list[dict[str, object]], int, str]:
    """
    Run quarter analyses with adaptive worker selection and memory backoff.

    Memory expectation:
        Starts from a conservative worker estimate and halves the pool after a
        memory-pressure failure until the run succeeds or sequential mode fails.
    """
    worker_count, selection_reason = suggest_worker_count(
        graph_paths=graph_paths,
        requested_workers=requested_workers,
        memory_reserve_gb=memory_reserve_gb,
    )
    attempts = [worker_count]

    while True:
        try:
            print(
                f"Processing {len(quarter_args)} quarters using {worker_count} "
                f"worker(s) ({selection_reason})..."
            )
            if worker_count == 1:
                quarters = []
                for quarter_arg in quarter_args:
                    quarters.append(analyze_quarter_wrapper(quarter_arg))
                    gc.collect()
            else:
                with Pool(worker_count) as pool:
                    quarters = pool.map(analyze_quarter_wrapper, quarter_args, chunksize=1)
            return quarters, worker_count, selection_reason
        except Exception as exc:
            if not is_memory_pressure_error(exc) or worker_count == 1:
                raise
            next_worker_count = max(1, worker_count // 2)
            print(
                f"Memory pressure detected with {worker_count} worker(s) "
                f"({type(exc).__name__}). Retrying with {next_worker_count}."
            )
            worker_count = next_worker_count
            attempts.append(worker_count)
            selection_reason = (
                f"{selection_reason}; memory backoff attempted workers={attempts}"
            )
            gc.collect()


def write_standardized_outputs(
    payload: dict[str, object],
    input_files: list[Path],
    args: argparse.Namespace,
    workers_used: int,
    worker_selection_reason: str,
) -> None:
    """
    Write standardized parquet outputs and metadata for cross-cluster bridging metric.
    """
    metric_name = "cross_cluster_bridging"

    # Convert quarters data to DataFrame for global level
    if not payload["quarters"]:
        return

    df_global = pd.DataFrame([
        {
            "quarter": row["quarter"],
            "value": row["median_bridge_ratio"],  # Primary metric: median bridge ratio
            "mean_bridge_ratio": row["mean_bridge_ratio"],
            "p90_bridge_ratio": row["p90_bridge_ratio"],
            "share_high_ratio": row["share_high_ratio"],
            "cross_edge_ratio": row["cross_edge_ratio"],
            "n_nodes_with_partition": row["n_nodes_with_partition"],
        }
        for row in payload["quarters"]
    ])

    # Get output paths
    paths_global = get_metric_output_paths(metric_name, args.out_dir, "global")

    # Write global parquet
    write_metric_parquet(df_global, paths_global["parquet"], "global", metric_name)

    # Create metadata
    metadata = create_metric_metadata(
        metric_name=metric_name,
        description="Quarterly cross-cluster bridging tracking nodes connecting different communities",
        formula="bridge_ratio = external_degree / total_degree; aggregated by mean/median/p90 across nodes",
        units="dimensionless bridge ratio (0-1); dimensionless (rate) for shares",
        parameters={
            "graphs_dir": str(args.graphs_dir),
            "resolution": args.resolution,
            "min_size": args.min_size,
            "max_size": args.max_size,
            "min_degree": args.min_degree,
            "high_ratio_threshold": args.high_ratio_threshold,
            "num_input_files": len(input_files),
            "max_workers_requested": args.max_workers,
            "workers_used": workers_used,
            "memory_reserve_gb": args.memory_reserve_gb,
            "worker_selection_reason": worker_selection_reason,
        },
        input_files=input_files,  # Track all input graph files
        level="global",
        column_descriptions={
            "quarter": "Quarter identifier (YYYYQN format)",
            "value": "Median bridge ratio (share of connections to other communities)",
            "mean_bridge_ratio": "Mean bridge ratio across nodes",
            "p90_bridge_ratio": "90th percentile bridge ratio",
            "share_high_ratio": f"Share of nodes with bridge ratio >= {args.high_ratio_threshold}",
            "cross_edge_ratio": "Share of all edges connecting different communities",
            "n_nodes_with_partition": "Count of nodes analyzed",
        },
    )

    write_metric_metadata(metadata, paths_global["metadata"])

    # Update central manifest (Task 1.2)
    manifest_path = args.out_dir / "manifest.json"
    update_manifest(manifest_path, metric_name, "global", metadata, paths_global)

    placeholder_reason = (
        "Per-front/per-lineage cross-cluster metrics require membership exports; "
        "placeholder emitted."
    )
    write_placeholder_metric(metric_name, args.out_dir, "front", metadata, placeholder_reason)
    write_placeholder_metric(metric_name, args.out_dir, "lineage", metadata, placeholder_reason)

    print(f"Wrote {paths_global['parquet']}")
    print(f"Wrote {paths_global['metadata']}")
    print(f"Updated manifest: {manifest_path}")


def main() -> None:
    args = parse_args()
    paths = resolve_script_paths(args, REPO_ROOT)
    apply_domain_path_defaults(args, paths, {
        "graphs_dir": ("graphs", "", "data/current_graphs"),
        "out_dir": ("out", "metrics", "data/out/metrics"),
        "registry": ("out", "front_id_registry_cumulative.json",
                      "data/out/front_id_registry_cumulative.json"),
        "cache_dir": ("cache_cum", "partitions_cum",
                       "data/out/cache_cum/partitions_cum"),
        "ingest_path": ("ingest", "ingest.parquet",
                         "data/current_ingest/ingest.parquet"),
    })
    args.graphs_dir = Path(args.graphs_dir)
    args.out_dir = Path(args.out_dir)
    args.registry = Path(args.registry)
    args.cache_dir = Path(args.cache_dir)
    args.ingest_path = Path(args.ingest_path)
    ensure_dir(args.out_dir)
    registry = load_registry(args.registry) if args.registry else {}
    graph_pairs = list_quarter_files(args.graphs_dir, "citation_graph_cumulative_*.pkl")
    if args.limit is not None:
        graph_pairs = graph_pairs[: args.limit]

    if not graph_pairs:
        raise FileNotFoundError(f"No cumulative graph files found in {args.graphs_dir}")

    # Track input files for provenance
    input_files = [path for quarter, path in graph_pairs]

    # Prepare arguments for parallel processing
    quarter_args = [(quarter, path, args, registry) for quarter, path in graph_pairs]
    quarters, workers_used, worker_selection_reason = run_quarter_analyses_with_backoff(
        quarter_args=quarter_args,
        graph_paths=input_files,
        requested_workers=args.max_workers,
        memory_reserve_gb=args.memory_reserve_gb,
    )
    print(f"Completed processing {len(quarters)} quarters using {workers_used} worker(s).")

    enrich_top_nodes_with_metadata(quarters, args.ingest_path)

    payload = {
        "metric": "cross_cluster_bridging",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "parameters": {
            "resolution": args.resolution,
            "min_size": args.min_size,
            "max_size": args.max_size,
            "min_degree": args.min_degree,
            "top_nodes": args.top_nodes,
            "high_ratio_threshold": args.high_ratio_threshold,
            "max_workers_requested": args.max_workers,
            "workers_used": workers_used,
            "memory_reserve_gb": args.memory_reserve_gb,
            "worker_selection_reason": worker_selection_reason,
        },
        "quarters": quarters,
    }

    # Legacy JSON output (backward compatibility)
    json_path = args.out_dir / args.json_name
    json_path.write_text(json.dumps(payload, indent=2))
    print(f"Wrote {json_path}")

    # Figure output
    figure_path = args.out_dir / args.figure_name
    if quarters:
        render_plot(payload, figure_path)
        print(f"Wrote {figure_path}")
    else:
        if figure_path.exists():
            figure_path.unlink()

    # Standardized parquet outputs with provenance tracking (Task 1.1 + 1.2)
    write_standardized_outputs(
        payload,
        input_files,
        args,
        workers_used=workers_used,
        worker_selection_reason=worker_selection_reason,
    )


if __name__ == "__main__":
    main()
