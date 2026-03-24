from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

from src.community import run_leiden
from src import trusted_io
from src.metrics.common import (
    ensure_dir,
    list_quarter_files,
    create_metric_metadata,
    write_metric_parquet,
    write_metric_metadata,
    get_metric_output_paths,
    update_manifest,
    write_placeholder_metric,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Quantify cross-cluster bridging nodes over time.")
    parser.add_argument("--graphs-dir", default="data/current_graphs", type=Path)
    parser.add_argument("--out-dir", default="data/out/metrics", type=Path)
    parser.add_argument("--registry", default="data/out/front_id_registry_cumulative.json", type=Path)
    parser.add_argument("--cache-dir", default="data/out/cache_cum/partitions_cum", type=Path)
    parser.add_argument("--json-name", default="cross_cluster_bridging.json")
    parser.add_argument("--figure-name", default="cross_cluster_bridging.png")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resolution", type=float, default=1.0)
    parser.add_argument("--min-size", type=int, default=50)
    parser.add_argument("--max-size", type=int, default=5000)
    parser.add_argument("--min-degree", type=int, default=3)
    parser.add_argument("--top-nodes", type=int, default=10)
    parser.add_argument("--high-ratio-threshold", type=float, default=0.5)
    return parser.parse_args()


def load_registry(path: Path) -> Dict[str, Dict[str, int]]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text())
    registry: Dict[str, Dict[str, int]] = {}
    for quarter, mapping in data.items():
        registry[quarter] = {str(k): int(v) for k, v in mapping.items()}
    return registry


def load_cached_partition(cache_dir: Path, quarter: str) -> Optional[Dict[str, int]]:
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
) -> Dict[str, int]:
    cached = load_cached_partition(cache_dir, quarter)
    if cached is not None:
        return cached
    max_size = args.max_size if args.max_size > 0 else None
    result = run_leiden(G, resolution=args.resolution, min_size=args.min_size, max_size=max_size)
    return {str(node): int(label) for node, label in result["partition"]}


def build_undirected(G: nx.DiGraph) -> nx.Graph:
    H = nx.Graph()
    H.add_nodes_from(G.nodes(data=True))
    for u, v in G.edges():
        if u == v:
            continue
        if not H.has_edge(u, v):
            H.add_edge(u, v)
    return H


def front_lookup(registry: Dict[str, Dict[str, int]], quarter: str, community: int) -> Optional[int]:
    mapping = registry.get(quarter)
    if not mapping:
        return None
    return mapping.get(str(community))


def analyze_quarter(
    quarter: str,
    graph_path: Path,
    args: argparse.Namespace,
    registry: Dict[str, Dict[str, int]],
) -> Dict[str, object]:
    Gfull: nx.DiGraph = trusted_io.load_trusted_binary(
        graph_path, description="citation graph",
    )

    # Load metadata lookup table (graphs now use minimal attributes for memory efficiency)
    ingest_path = REPO_ROOT / "data" / "current_ingest" / "ingest.parquet"
    metadata_df = pd.read_parquet(ingest_path, columns=['work_id', 'title', 'cited_by_count'])
    metadata_lookup = metadata_df.set_index('work_id').to_dict('index')

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
    bridge_rows: List[Dict[str, object]] = []
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
        # Lookup rich metadata from ingest.parquet (not embedded in minimal graphs)
        node_metadata = metadata_lookup.get(str(node), {})
        bridge_rows.append(
            {
                "node_id": node,
                "community_id": int(community),
                "front_id": front_lookup(registry, quarter, community),
                "degree": degree,
                "external_degree": external,
                "external_communities": len(external_comms),
                "bridge_ratio": bridge_ratio,
                "title": node_metadata.get("title"),
                "publication_date": node_data.get("publication_date"),  # Still in minimal graph
                "cited_by_count": node_metadata.get("cited_by_count"),
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


def render_plot(payload: Dict[str, object], out_path: Path) -> None:
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


def write_standardized_outputs(
    payload: Dict[str, object],
    input_files: List[Path],
    args: argparse.Namespace,
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
    ensure_dir(args.out_dir)
    registry = load_registry(args.registry) if args.registry else {}
    graph_pairs = list_quarter_files(args.graphs_dir, "citation_graph_cumulative_*.pkl")
    if args.limit is not None:
        graph_pairs = graph_pairs[: args.limit]

    # Track input files for provenance
    input_files = [path for quarter, path in graph_pairs]

    # Parallelize across quarters using multiprocessing
    from multiprocessing import Pool, cpu_count
    n_workers = min(cpu_count(), len(graph_pairs))
    print(f"Processing {len(graph_pairs)} quarters using {n_workers} parallel workers...")

    # Prepare arguments for parallel processing
    quarter_args = [(quarter, path, args, registry) for quarter, path in graph_pairs]

    with Pool(n_workers) as pool:
        quarters = pool.map(analyze_quarter_wrapper, quarter_args)

    print(f"Completed processing {len(quarters)} quarters.")

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
    write_standardized_outputs(payload, input_files, args)


if __name__ == "__main__":
    main()
