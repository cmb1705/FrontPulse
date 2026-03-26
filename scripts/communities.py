#!/usr/bin/env python3
from __future__ import annotations

"""Unified community pipeline supporting cumulative (default), annual, and quarterly community detection.

Examples:
  # Default cumulative run (20-quarter window, reuse caches)
  python scripts/communities.py \
      --graphs-dir data/current_graphs \
      --out-dir data/out \
      --window-quarters 20 --resume

  # Run annual + quarterly alignment only
  python scripts/communities.py --mode both --graphs-dir data/current_graphs --out-dir data/out

  # Resolution sweep for cumulative graphs
  python scripts/communities.py --mode cumulative \
      --res-sweep 0.0001,0.0005,0.001,0.005,0.01 --res-sweep-only
"""


import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import sys  # noqa: E402
from collections import Counter, defaultdict  # noqa: E402
from collections.abc import Iterable  # noqa: E402
from itertools import count  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

import networkx as nx  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

try:
    import plotly.graph_objects as go  # type: ignore
except Exception:
    go = None

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

try:
    # project imports
    from src.alignment import (
        label_map_from_partition,
        match_by_cores,
        pagerank_core,
        variation_of_information,
    )
    from src.community import adaptive_cluster_bounds, compute_pia_flags, run_ecg, run_leiden
except ModuleNotFoundError as e:
    raise SystemExit("Cannot import src.*. Run from repo root.") from e
from src.graph_lite import LiteGraph  # noqa: E402
from src.trusted_io import load_trusted_pickle  # noqa: E402

# Module-level clustering dispatch. Defaults to Leiden; overridden by
# --use-ecg flag in __main__ to use ECG ensemble clustering.
_cluster_fn = run_leiden


def _canonical_quarter(label: str) -> str:
    import re
    m = re.search(r"(\d{4}Q[1-4])", str(label))
    return m.group(1) if m else str(label)

def _write_resolution_html(entries: list[dict[str, Any]], out_path: Path) -> None:
    if go is None:
        print("[Sweep] Plotly not available; skipping HTML visualization.")
        return
    fig = go.Figure()
    for entry in entries:
        resolution = entry.get("resolution")
        quarters = entry.get("quarters") or []
        if not quarters:
            continue
        stats = entry.get("cluster_stats_by_quarter") or {}
        counts = [(stats.get(q) or {}).get("n_clusters", 0) for q in quarters]
        x = pd.PeriodIndex(quarters, freq="Q").to_timestamp("Q", "end")
        fig.add_trace(
            go.Scatter(
                x=x,
                y=counts,
                mode="lines+markers",
                name=str(resolution),
                customdata=quarters,
                hovertemplate="<b>Clusters</b><br>Quarter: %{customdata}<br>Value: %{y:.0f}<br>Resolution: "
                              f"{resolution}<extra></extra>",
            )
        )
    fig.update_layout(
        title="Cluster counts by resolution",
        xaxis_title="Quarter",
        yaxis_title="Number of communities",
        template="plotly_white",
        height=700,
        width=1100,
    )
    fig.write_html(out_path, include_plotlyjs="cdn")
    print(f"[Sweep] Saved {out_path}")


def _quarter_sort_key(label: str) -> tuple[int, int]:
    canon = _canonical_quarter(label)
    y = int(canon[:4]) if canon[:4].isdigit() else 0
    q = int(canon[-1]) if canon[-1].isdigit() else 0
    return y, q


def _part_label_map(part) -> dict:
    lm = label_map_from_partition(part)
    if not isinstance(lm, dict):
        try:
            if hasattr(lm, "items"):
                lm = dict(lm.items())
            elif hasattr(lm, "__iter__"):
                lm = dict(lm)
        except Exception:
            pass
    return lm


def _unique_cids(part) -> list:
    lm = _part_label_map(part)
    return sorted(set(lm.values()))


def _full_core_overlaps(cores_prev: dict[int, set], cores_curr: dict[int, set]) -> list[list[int]]:
    rows: list[list[int]] = []
    for a, Sa in cores_prev.items():
        la = len(Sa) or 1
        for b, Sb in cores_curr.items():
            lb = len(Sb) or 1
            ov = len(Sa & Sb)
            rows.append([int(a), int(b), int(ov), int(la), int(lb)])
    return rows


def _events_from_overlaps(
    overlaps: list[list[int]], ov_min: int, ov_frac: float
) -> tuple[int, int, list[dict]]:
    prev_to_curr, curr_to_prev = defaultdict(list), defaultdict(list)
    events: list[dict] = []
    for a, b, ov, la, lb in overlaps:
        m = max(1, min(la, lb))
        if ov >= ov_min and (ov / m) >= ov_frac:
            prev_to_curr[a].append((b, ov))
            curr_to_prev[b].append((a, ov))
    splits = sum(1 for vs in prev_to_curr.values() if len({b for b, _ in vs}) > 1)
    merges = sum(1 for vs in curr_to_prev.values() if len({a for a, _ in vs}) > 1)
    for a, lst in prev_to_curr.items():
        if len({b for b, _ in lst}) > 1:
            events.append({
                "type": "split",
                "prev": int(a),
                "to": [{"curr": int(b), "overlap": int(ov)} for b, ov in sorted(lst, key=lambda x: -x[1])],
            })
    for b, lst in curr_to_prev.items():
        if len({a for a, _ in lst}) > 1:
            events.append({
                "type": "merge",
                "curr": int(b),
                "from": [{"prev": int(a), "overlap": int(ov)} for a, ov in sorted(lst, key=lambda x: -x[1])],
            })
    return int(splits), int(merges), events


def _collect_cumulative_graph_paths(graphs_dir: Path) -> dict[str, dict[str, Path]]:
    mapping: dict[str, dict[str, Path]] = {}
    for path in graphs_dir.glob("citation_graph_cumulative_*.pkl"):
        q = _canonical_quarter(path.stem)
        mapping.setdefault(q, {})["pkl"] = path
    for path in graphs_dir.glob("citation_graph_cumulative_*.lite.npz"):
        q = _canonical_quarter(path.stem)
        mapping.setdefault(q, {})["lite"] = path
    return mapping


def _load_cumulative_graph(
    paths: dict[str, Path],
    *,
    include_publication_dates: bool,
    allow_external_pickle: bool = False,
) -> nx.DiGraph:
    if "lite" in paths:
        lite = LiteGraph.load(paths["lite"])
        try:
            graph = lite.to_networkx(include_publication_dates=include_publication_dates)

            # Defensive check: Ensure lite graph has required metadata
            # If pub_qtr is missing, new_works counting will fail silently
            if graph.number_of_nodes() > 0:
                sample_node = next(iter(graph.nodes(data=True)))
                node_data = sample_node[1]
                if "pub_qtr" not in node_data:
                    print(f"[WARN] Lite graph {paths['lite'].name} lacks pub_qtr metadata.")
                    print("[WARN] New-work counting will fail. Regenerate with build_lite_graphs.py")
                    print("[WARN] Falling back to pickle if available...")
                    # Try to fall back to pickle
                    if "pkl" in paths:
                        return load_trusted_pickle(
                            paths["pkl"],
                            description="Cumulative graph pickle",
                            allow_external=allow_external_pickle,
                        )
                    else:
                        raise ValueError(
                            f"Lite graph {paths['lite']} lacks required metadata (pub_qtr, pub_year). "
                            "Regenerate with: python scripts/build_lite_graphs.py --force"
                        )
        finally:
            del lite
        return graph
    if "pkl" not in paths:
        raise FileNotFoundError("No cumulative graph available for requested quarter.")
    return load_trusted_pickle(
        paths["pkl"],
        description="Cumulative graph pickle",
        allow_external=allow_external_pickle,
    )




def _vi_stats(part_curr: dict[str, int], part_prev: dict[str, int], node_subset: Iterable[str] | None = None) -> tuple[float | None, float | None, int]:
    nodes_curr = set(part_curr.keys())
    nodes_prev = set(part_prev.keys())
    if node_subset is None:
        nodes = nodes_curr & nodes_prev
    else:
        nodes = {str(n) for n in node_subset} & nodes_curr & nodes_prev
    N = len(nodes)
    if N <= 1:
        return (None, None, N)
    map_curr = {n: int(part_curr[n]) for n in nodes}
    map_prev = {n: int(part_prev[n]) for n in nodes}
    vi = variation_of_information(map_prev, map_curr)
    nvi = vi / math.log2(N) if N > 1 else None
    return float(vi), (float(nvi) if nvi is not None else None), N


def _nodes_recent(G, q: str, years: int) -> set[str]:
    cutoff = _quarter_end(q) - pd.DateOffset(years=int(years))
    out = set()
    for n, data in G.nodes(data=True):
        dt = _parse_ts(data.get("publication_date"))
        if pd.notna(dt) and dt >= cutoff:
            out.add(str(n))
    return out


def _core_overlap_summary(matches, cores_prev: dict[int, set[str]], cores_curr: dict[int, set[str]], curr_map: dict[int, int]) -> dict[int, float | None]:
    summary: dict[int, tuple[float, float]] = {}
    for prev_cid, curr_cid, ov in matches:
        prev_cid = int(prev_cid)
        curr_cid = int(curr_cid)
        fid = curr_map.get(curr_cid)
        if fid is None:
            continue
        prev_core = cores_prev.get(prev_cid) or set()
        curr_core = cores_curr.get(curr_cid) or set()
        if not prev_core or not curr_core:
            continue
        denom = min(len(prev_core), len(curr_core))
        if denom <= 0:
            continue
        frac = ov / denom
        num, den = summary.get(fid, (0.0, 0.0))
        summary[fid] = (num + frac * denom, den + denom)
    return {int(fid): ((num / den) if den else None) for fid, (num, den) in summary.items()}


def _label_change_rates(front_nodes: dict[int, list[str]], part_prev: dict[str, int], prev_front_map: dict[int, int]) -> tuple[dict[int, float | None], dict[int, int]]:
    rates: dict[int, float | None] = {}
    counts: dict[int, int] = {}
    for fid, nodes in front_nodes.items():
        total = 0
        change = 0
        for n in nodes:
            prev_cid = part_prev.get(n)
            if prev_cid is None:
                continue
            prev_fid = prev_front_map.get(int(prev_cid))
            if prev_fid is None:
                continue
            total += 1
            if prev_fid != fid:
                change += 1
        rates[fid] = (change / total) if total else None
        counts[fid] = total
    return rates, counts


def _cluster_size_summary(sizes: list[int]) -> dict[str, float | None]:
    if not sizes:
        return {
            "min": 0,
            "median": 0,
            "mean": 0.0,
            "90th_percentile": 0.0,
            "max": 0,
            "n_clusters": 0,
            "n_oversized": 0,
        }
    arr = np.asarray(sizes, dtype=float)
    stats = {
        "min": int(np.min(arr)),
        "median": float(np.median(arr)),
        "mean": float(arr.mean()),
        "90th_percentile": float(np.percentile(arr, 90)),
        "max": int(np.max(arr)),
        "n_clusters": int(len(arr)),
        "n_oversized": int(np.sum(arr > 5000)),
    }
    return stats

def _parse_ts(x):
    try:
        return pd.to_datetime(x, errors="coerce")
    except Exception:
        return pd.NaT


def _quarter_end(q: str) -> pd.Timestamp:
    p = pd.Period(q, freq="Q")
    try:
        return p.end_time
    except Exception:
        return p.to_timestamp(how="end")


def _window_and_prune(G, q: str, window_quarters: int | None, prune_age_years: int | None, prune_min_in_deg: int | None):
    H = G
    if window_quarters and window_quarters > 0:
        end = _quarter_end(q)
        start = (end - pd.offsets.QuarterEnd(window_quarters - 1))
        keep = []
        for n, d in H.nodes(data=True):
            dt = _parse_ts(d.get("publication_date"))
            if pd.notna(dt) and dt >= (start - pd.offsets.QuarterEnd(0)):
                keep.append(n)
        H = H.subgraph(keep).copy()
    if prune_age_years and prune_age_years > 0:
        end = _quarter_end(q)
        age_cut = end - pd.DateOffset(years=int(prune_age_years))
        drop = []
        threshold = prune_min_in_deg if (prune_min_in_deg is not None and prune_min_in_deg > 0) else None
        for n, d in H.nodes(data=True):
            dt = _parse_ts(d.get("publication_date"))
            if pd.notna(dt) and dt < age_cut and threshold is not None:
                in_deg = H.in_degree(n) if H.is_directed() else H.degree(n)
                if in_deg <= threshold:
                    drop.append(n)
        if drop:
            H.remove_nodes_from(drop)
    return H



def run_cumulative(
    *,
    graphs_dir: Path,
    out_dir: Path,
    resolution: float,
    min_size: int,
    max_size: int,
    adaptive_max: bool,
    max_fraction: float,
    max_floor: int,
    max_ceiling: int,
    min_fraction: float | None = None,
    max_floor_fraction: float | None = None,
    max_ceiling_fraction: float | None = None,
    min_absolute_floor: int = 10,
    max_absolute_ceiling: int = 15000,
    avg_degree_threshold: float = 15.0,
    core_frac: float,
    overlap_min: int,
    overlap_frac: float,
    window_quarters: int | None,
    prune_age_years: int | None,
    prune_min_in_deg: int | None = 0,
    cache_dir: Path | None,
    resume: bool,
    force: bool,
    limit_quarters: int | None,
    write_outputs: bool = True,
    allow_external_pickle: bool = False,
) -> dict:
    lineage_dir = out_dir / "02_lineage_tracking"

    if write_outputs:
        out_dir.mkdir(parents=True, exist_ok=True)
        lineage_dir.mkdir(parents=True, exist_ok=True)
    cache_parts = (cache_dir / "partitions_cum") if (cache_dir and write_outputs) else None
    cache_cores = (cache_dir / "cores_cum") if (cache_dir and write_outputs) else None
    if cache_parts:
        cache_parts.mkdir(parents=True, exist_ok=True)
    if cache_cores:
        cache_cores.mkdir(parents=True, exist_ok=True)

    graph_entries = _collect_cumulative_graph_paths(graphs_dir)
    quarters = sorted(graph_entries.keys(), key=_quarter_sort_key)
    if limit_quarters:
        quarters = quarters[:limit_quarters]
    if not quarters:
        print("[Cumulative] No citation_graph_cumulative_* graphs found.")
        return {"delta": []}

    results = {"delta": []}
    registry: dict[str, dict[int, int]] = {}
    states: dict[str, dict[str, object]] = {}
    ts_rows: list[dict[str, Any]] = []
    lineage_metrics_rows: list[dict[str, Any]] = []
    next_id = count(start=1)
    summary: dict[str, Any] = {
        "resolution": resolution,
        "quarters": [],
        "cluster_stats_by_quarter": {},
        "pia_by_quarter": {},
        "graph_density_by_quarter": {},
        "n_nodes_by_quarter": {},
        "n_edges_by_quarter": {},
        "avg_degree_by_quarter": {},
        "raw_communities_by_quarter": {},
        "filtered_communities_by_quarter": {},
        "coupling_stats_by_quarter": {},
        "graph_density_values": [],
        "n_nodes_values": [],
        "n_edges_values": [],
        "avg_degree_values": [],
        "raw_communities_values": [],
        "filtered_communities_values": [],
        "coupling_pairs_retained_values": [],
        "coupling_weight_pair_totals": [],
        "edge_type_counts_by_quarter": {},
        "coupling_edge_counts_values": [],
        "hybrid_edge_counts_values": [],
        "citation_edge_counts_values": [],
        "coupling_weight_edge_totals": [],
        "coupling_weight_edge_means": [],
        "vi_qoq_values": [],
        "vi_qoq_by_quarter": {},
        "nvi_qoq_values": [],
        "nvi_qoq_by_quarter": {},
        "vi_qoq_2y_values": [],
        "vi_qoq_2y_by_quarter": {},
        "nvi_qoq_2y_values": [],
        "nvi_qoq_2y_by_quarter": {},
        "vi_yoy_values": [],
        "vi_yoy_by_quarter": {},
        "nvi_yoy_values": [],
        "nvi_yoy_by_quarter": {},
        "vi_qv20_values": [],
        "vi_qv20_by_quarter": {},
        "nvi_qv20_values": [],
        "nvi_qv20_by_quarter": {},
        "modularity_by_quarter": {},
        "modularity_values": [],
        "largest_community_by_quarter": {},
        "isolated_pct_by_quarter": {},
        "alerts_by_quarter": {},
    }

    include_pub_dates = bool(window_quarters) or bool(prune_age_years)

    for q in quarters:
        paths = graph_entries.get(q, {})
        period = pd.Period(q, freq="Q")
        key_qoq = str(period - 1)
        key_yoy = str(period - 4)
        key_q20 = str(period - 20)
        state_qoq = states.get(key_qoq)
        state_yoy = states.get(key_yoy)
        state_q20 = states.get(key_q20)
        prev_front_map_qoq = registry.get(key_qoq, {}) if state_qoq else {}
        prev_front_map_yoy = registry.get(key_yoy, {}) if state_yoy else {}

        Gfull = _load_cumulative_graph(
            paths,
            include_publication_dates=include_pub_dates,
            allow_external_pickle=allow_external_pickle,
        )
        Gq = _window_and_prune(Gfull, q, window_quarters, prune_age_years, prune_min_in_deg)
        summary["quarters"].append(q)

        n_nodes = int(Gq.number_of_nodes())
        n_edges = int(Gq.number_of_edges())
        max_possible_edges = n_nodes * (n_nodes - 1)
        density = (n_edges / max_possible_edges) if max_possible_edges > 0 else 0.0
        avg_degree = (n_edges / n_nodes) if n_nodes else 0.0
        summary["n_nodes_by_quarter"][q] = n_nodes
        summary["n_edges_by_quarter"][q] = n_edges
        summary["graph_density_by_quarter"][q] = density
        summary["avg_degree_by_quarter"][q] = avg_degree
        summary["n_nodes_values"].append(n_nodes)
        summary["n_edges_values"].append(n_edges)
        summary["graph_density_values"].append(density)
        summary["avg_degree_values"].append(avg_degree)

        part_fp = cache_parts / f"part_{q}.json" if cache_parts else None
        cores_fp = cache_cores / f"cores_{q}.json" if cache_cores else None
        part_map = None
        cores_q = None
        raw_n_communities: int | None = None
        modularity_val: float | None = None
        if resume and part_fp and part_fp.exists() and not force:
            data = json.loads(part_fp.read_text())
            part_map = {str(k): int(v) for k, v in (data.get("labels") or {}).items()}
            raw_val = data.get("raw_n_communities")
            if raw_val is not None:
                try:
                    raw_n_communities = int(raw_val)
                except Exception:
                    raw_n_communities = None
            modularity_data = data.get("modularity")
            if modularity_data is not None:
                try:
                    modularity_val = float(modularity_data)
                except Exception:
                    modularity_val = None
        if resume and cores_fp and cores_fp.exists() and not force:
            data = json.loads(cores_fp.read_text())
            cores_q = {int(k): {str(n) for n in (v or [])} for k, v in (data or {}).items()}

        min_size_eff, max_size_eff = adaptive_cluster_bounds(
            Gq.number_of_nodes(),
            Gq.number_of_edges(),
            min_size=min_size,
            max_size=None if max_size <= 0 else max_size,
            adaptive_enabled=adaptive_max,
            max_fraction=max_fraction,
            max_floor=max_floor,
            max_ceiling=max_ceiling,
            min_fraction=min_fraction,
            max_floor_fraction=max_floor_fraction,
            max_ceiling_fraction=max_ceiling_fraction,
        min_absolute_floor=min_absolute_floor,
        max_absolute_ceiling=max_absolute_ceiling,
        avg_degree_threshold=avg_degree_threshold,
    )

        if part_map is None:
            rq = _cluster_fn(Gq, resolution=resolution, min_size=min_size_eff, max_size=max_size_eff)
            part_map = _part_label_map(rq["partition"])
            part_map = {str(n): int(c) for n, c in part_map.items()}
            try:
                raw_n_communities = int(rq.get("raw_n_communities")) if rq.get("raw_n_communities") is not None else None
            except Exception:
                raw_n_communities = None
            try:
                modularity_val = float(rq.get("modularity")) if rq.get("modularity") is not None else None
            except Exception:
                modularity_val = None
        else:
            sizes = Counter(part_map.values())
            rq = {
                "partition": list(part_map.items()),
                "communities": [{"id": int(cid), "size": int(sz)} for cid, sz in sizes.items()],
                "modularity": modularity_val,
            }
            if raw_n_communities is None:
                raw_n_communities = int(len(sizes))
        if raw_n_communities is None:
            raw_n_communities = int(len(rq.get("communities", [])))
        rq["raw_n_communities"] = int(raw_n_communities)
        if modularity_val is None:
            mod_tmp = rq.get("modularity")
            if mod_tmp is not None:
                try:
                    modularity_val = float(mod_tmp)
                except Exception:
                    modularity_val = None

        if cores_q is None:
            cores_tmp = pagerank_core(Gq, rq["partition"], core_frac=core_frac)
            cores_q = {int(cid): {str(n) for n in nodes} for cid, nodes in cores_tmp.items()}

        if modularity_val is None:
            try:
                import networkx as nx  # type: ignore
                communities_sets: dict[int, list[str]] = defaultdict(list)
                for node, cid in part_map.items():
                    communities_sets[int(cid)].append(node)
                community_partition = [set(nodes) for nodes in communities_sets.values() if nodes]
                if community_partition:
                    modularity_val = nx.algorithms.community.quality.modularity(
                        Gq.to_undirected() if Gq.is_directed() else Gq,
                        community_partition,
                    )
            except Exception:
                modularity_val = None

        cluster_sizes = [int(comm.get("size", len(comm.get("nodes", [])))) for comm in rq.get("communities", [])]
        summary["cluster_stats_by_quarter"][q] = _cluster_size_summary(cluster_sizes)
        largest_cluster = int(max(cluster_sizes)) if cluster_sizes else 0
        summary["largest_community_by_quarter"][q] = largest_cluster
        if modularity_val is not None:
            summary["modularity_by_quarter"][q] = float(modularity_val)
            summary["modularity_values"].append(float(modularity_val))
        else:
            summary["modularity_by_quarter"][q] = None
        isolated_count = sum(1 for _, deg in Gq.degree() if deg == 0) if n_nodes else 0
        isolated_pct = (isolated_count / n_nodes) if n_nodes else 0.0
        summary["isolated_pct_by_quarter"][q] = float(isolated_pct)
        summary["raw_communities_by_quarter"][q] = int(raw_n_communities)
        filtered_count = len(rq.get("communities", []))
        summary["filtered_communities_by_quarter"][q] = int(filtered_count)
        summary["raw_communities_values"].append(int(raw_n_communities))
        summary["filtered_communities_values"].append(int(filtered_count))
        lm = _part_label_map(rq["partition"])
        lm = {str(n): int(c) for n, c in lm.items()}

        coupling_stats = Gq.graph.get("coupling_stats") or {}
        summary["coupling_stats_by_quarter"][q] = coupling_stats
        summary["edge_type_counts_by_quarter"][q] = None
        if coupling_stats:
            summary["edge_type_counts_by_quarter"][q] = {
                "citation": int(coupling_stats.get("citation_edges", 0) or 0),
                "coupling": int(coupling_stats.get("coupling_edges", 0) or 0),
                "hybrid": int(coupling_stats.get("hybrid_edges", 0) or 0),
            }
            if coupling_stats.get("coupling_pairs_retained") is not None:
                summary["coupling_pairs_retained_values"].append(
                    float(coupling_stats.get("coupling_pairs_retained", 0))
                )
            if coupling_stats.get("coupling_weight_sum_pairs") is not None:
                summary["coupling_weight_pair_totals"].append(
                    float(coupling_stats.get("coupling_weight_sum_pairs", 0.0))
                )
            if coupling_stats.get("coupling_edge_count") is not None:
                summary["coupling_edge_counts_values"].append(
                    float(coupling_stats.get("coupling_edge_count", 0))
                )
            if coupling_stats.get("hybrid_edges") is not None:
                summary["hybrid_edge_counts_values"].append(float(coupling_stats.get("hybrid_edges", 0)))
            if coupling_stats.get("citation_edges") is not None:
                summary["citation_edge_counts_values"].append(float(coupling_stats.get("citation_edges", 0)))
            if coupling_stats.get("coupling_weight_sum_edges") is not None:
                summary["coupling_weight_edge_totals"].append(
                    float(coupling_stats.get("coupling_weight_sum_edges", 0.0))
                )
            if coupling_stats.get("coupling_weight_mean_edge") is not None:
                summary["coupling_weight_edge_means"].append(
                    float(coupling_stats.get("coupling_weight_mean_edge", 0.0))
                )
            if coupling_stats.get("cache_reset_reason"):
                print(f"[Coupling] cache reset -> {coupling_stats.get('cache_reset_reason')}")
            print(
                "[Coupling] "
                f"{q}: retained={int(coupling_stats.get('coupling_pairs_retained', 0))} "
                f"pairs, weight_sum={float(coupling_stats.get('coupling_weight_sum_pairs', 0.0)):.3f} "
                f"edges={int(coupling_stats.get('coupling_edge_count', 0))} "
                f"hybrid_edges={int(coupling_stats.get('hybrid_edges', 0))} "
                f"workers={int(coupling_stats.get('coupling_workers_used', 1))}"
            )

        if part_fp and (force or not part_fp.exists()):
            cache_payload = {"quarter": q, "labels": part_map}
            if raw_n_communities is not None:
                cache_payload["raw_n_communities"] = int(raw_n_communities)
            if modularity_val is not None:
                cache_payload["modularity"] = float(modularity_val)
            part_fp.write_text(json.dumps(cache_payload, indent=2))
        if cores_fp and (force or not cores_fp.exists()):
            cores_fp.write_text(json.dumps({int(k): sorted(v) for k, v in cores_q.items()}, indent=2))

        curr_map: dict[int, int] = {}
        alignment = {"matches": [], "overlaps": [], "VI": None, "splits": 0, "merges": 0, "events": []}
        matches_qoq = []
        if state_qoq:
            cores_prev = state_qoq["cores"]  # type: ignore[index]
            matches_qoq = match_by_cores(cores_prev, cores_q)
            overlaps = _full_core_overlaps(cores_prev, cores_q)
            splits, merges, ev = _events_from_overlaps(overlaps, ov_min=overlap_min, ov_frac=overlap_frac)
            vi_align = variation_of_information(state_qoq["part_map"], part_map)  # type: ignore[index]
            alignment = {
                "matches": [(int(a), int(b), int(ov)) for a, b, ov in matches_qoq],
                "overlaps": overlaps,
                "VI": float(vi_align),
                "splits": int(splits),
                "merges": int(merges),
                "events": ev,
            }
            prev_front_map = state_qoq["front_map"]  # type: ignore[index]
            for a_prev, b_curr, _ov in matches_qoq:
                a_prev = int(a_prev)
                b_curr = int(b_curr)
                prev_front = prev_front_map.get(a_prev)
                if prev_front is not None and b_curr not in curr_map:
                    curr_map[b_curr] = prev_front
        method_alerts: list[str] = []
        if modularity_val is not None and modularity_val < 0.25:
            method_alerts.append("modularity_below_0.25")
        if avg_degree < 12.0:
            method_alerts.append("avg_degree_below_12")
        if isolated_pct > 0.30:
            method_alerts.append("isolated_pct_above_30")
        summary["alerts_by_quarter"][q] = method_alerts
        if method_alerts:
            print(f"[Alert] {q}: method review flags -> {', '.join(method_alerts)}")

        entry = {
            "quarter": q,
            "n_nodes": n_nodes,
            "n_edges": n_edges,
            "density": density,
            "avg_degree": avg_degree,
            "min_size": int(min_size_eff),
            "max_size": int(max_size_eff) if max_size_eff is not None else None,
            "n_communities": len(rq.get("communities", [])),
            "raw_communities": int(raw_n_communities),
            "filtered_communities": len(rq.get("communities", [])),
            "modularity": modularity_val,
            "largest_community_size": largest_cluster,
            "isolated_node_pct": isolated_pct,
            "method_alerts": method_alerts,
            "coupling_stats": coupling_stats,
            "alignment": alignment,
        }

        for cid in _unique_cids(rq["partition"]):
            cid_i = int(cid) if isinstance(cid, (int, float, str)) else cid
            if cid_i not in curr_map:
                curr_map[cid_i] = next(next_id)

        registry[q] = {int(k): int(v) for k, v in curr_map.items()}

        front_nodes: dict[int, list[str]] = defaultdict(list)
        for node, cid in part_map.items():
            fid = curr_map.get(int(cid))
            if fid is not None:
                front_nodes[int(fid)].append(node)

        vi_qoq = nvi_qoq = None
        N_qoq = 0
        vi_qoq_2y = nvi_qoq_2y = None
        N_qoq_2y = 0
        vi_yoy = nvi_yoy = None
        N_yoy = 0
        vi_q20 = nvi_q20 = None
        N_q20 = 0
        if state_qoq:
            vi_qoq, nvi_qoq, N_qoq = _vi_stats(part_map, state_qoq["part_map"])  # type: ignore[index]
            nodes_2y = _nodes_recent(Gq, q, 2)
            vi_qoq_2y, nvi_qoq_2y, N_qoq_2y = _vi_stats(part_map, state_qoq["part_map"], nodes_2y)  # type: ignore[index]
        if state_yoy:
            vi_yoy, nvi_yoy, N_yoy = _vi_stats(part_map, state_yoy["part_map"])  # type: ignore[index]
        if state_q20:
            vi_q20, nvi_q20, N_q20 = _vi_stats(part_map, state_q20["part_map"])  # type: ignore[index]

        entry.update({
            "VI_QoQ": vi_qoq,
            "nVI_QoQ": nvi_qoq,
            "N_QoQ": N_qoq,
            "nVI_QoQ_2y": nvi_qoq_2y,
            "N_QoQ_2y": N_qoq_2y,
            "VI_YoY": vi_yoy,
            "nVI_YoY": nvi_yoy,
            "N_YoY": N_yoy,
            "VI_QvQ20": vi_q20,
            "nVI_QvQ20": nvi_q20,
            "N_QvQ20": N_q20,
        })

        summary["vi_qoq_by_quarter"][q] = float(vi_qoq) if vi_qoq is not None else None
        summary["nvi_qoq_by_quarter"][q] = float(nvi_qoq) if nvi_qoq is not None else None
        summary["vi_qoq_2y_by_quarter"][q] = float(vi_qoq_2y) if vi_qoq_2y is not None else None
        summary["nvi_qoq_2y_by_quarter"][q] = float(nvi_qoq_2y) if nvi_qoq_2y is not None else None
        summary["vi_yoy_by_quarter"][q] = float(vi_yoy) if vi_yoy is not None else None
        summary["nvi_yoy_by_quarter"][q] = float(nvi_yoy) if nvi_yoy is not None else None
        summary["vi_qv20_by_quarter"][q] = float(vi_q20) if vi_q20 is not None else None
        summary["nvi_qv20_by_quarter"][q] = float(nvi_q20) if nvi_q20 is not None else None

        if vi_qoq is not None:
            summary["vi_qoq_values"].append(float(vi_qoq))
        if nvi_qoq is not None:
            summary["nvi_qoq_values"].append(float(nvi_qoq))
        if vi_qoq_2y is not None:
            summary["vi_qoq_2y_values"].append(float(vi_qoq_2y))
        if nvi_qoq_2y is not None:
            summary["nvi_qoq_2y_values"].append(float(nvi_qoq_2y))
        if vi_yoy is not None:
            summary["vi_yoy_values"].append(float(vi_yoy))
        if nvi_yoy is not None:
            summary["nvi_yoy_values"].append(float(nvi_yoy))
        if vi_q20 is not None:
            summary["vi_qv20_values"].append(float(vi_q20))
        if nvi_q20 is not None:
            summary["nvi_qv20_values"].append(float(nvi_q20))

        overlap_qoq = {}
        overlap_yoy = {}
        label_rate_qoq: dict[int, float | None] = {}
        label_rate_yoy: dict[int, float | None] = {}
        count_qoq: dict[int, int] = {}
        count_yoy: dict[int, int] = {}
        if state_qoq:
            overlap_qoq = _core_overlap_summary(matches_qoq, state_qoq["cores"], cores_q, curr_map)  # type: ignore[index]
            label_rate_qoq, count_qoq = _label_change_rates(front_nodes, state_qoq["part_map"], prev_front_map_qoq)  # type: ignore[index]
        if state_yoy:
            matches_yoy = match_by_cores(state_yoy["cores"], cores_q)  # type: ignore[index]
            overlap_yoy = _core_overlap_summary(matches_yoy, state_yoy["cores"], cores_q, curr_map)  # type: ignore[index]
            label_rate_yoy, count_yoy = _label_change_rates(front_nodes, state_yoy["part_map"], prev_front_map_yoy)  # type: ignore[index]

        pia_result = compute_pia_flags(Gq, lm, min_links=20, within_ratio=0.10)
        pia_cluster_stats = pia_result["cluster_stats"]
        pia_totals = pia_result["totals"]
        entry.update({
            "pia_eligible": int(pia_totals.get("eligible", 0) or 0),
            "pia_count": int(pia_totals.get("pia", 0) or 0),
            "pia_rate": pia_totals.get("pia_rate"),
        })
        summary["pia_by_quarter"][q] = {
            "pia_eligible": entry["pia_eligible"],
            "pia_count": entry["pia_count"],
            "pia_rate": entry["pia_rate"],
        }
        front_pia_raw: dict[int, dict[str, int]] = defaultdict(lambda: {"eligible": 0, "pia": 0})
        for cid, stats in pia_cluster_stats.items():
            fid = curr_map.get(int(cid))
            if fid is None:
                continue
            front_pia_raw[int(fid)]["eligible"] += int(stats.get("eligible", 0) or 0)
            front_pia_raw[int(fid)]["pia"] += int(stats.get("pia", 0) or 0)
        front_pia_metrics: dict[int, dict[str, float | None]] = {}
        for fid, stats in front_pia_raw.items():
            eligible = int(stats.get("eligible", 0) or 0)
            pia = int(stats.get("pia", 0) or 0)
            rate = (pia / eligible) if eligible else None
            front_pia_metrics[int(fid)] = {
                "pia_eligible": eligible,
                "pia_count": pia,
                "pia_rate": rate,
            }

        for fid, _nodes in front_nodes.items():
            lineage_metrics_rows.append({
                "lineage_id": int(fid),
                "quarter": q,
                "core_overlap_qoq": overlap_qoq.get(fid),
                "core_overlap_yoy": overlap_yoy.get(fid),
                "label_change_rate_qoq": label_rate_qoq.get(fid),
                "label_change_rate_yoy": label_rate_yoy.get(fid),
                "n_nodes_qoq": count_qoq.get(fid),
                "n_nodes_yoy": count_yoy.get(fid),
                "pia_eligible": front_pia_metrics.get(fid, {}).get("pia_eligible", 0),
                "pia_count": front_pia_metrics.get(fid, {}).get("pia_count", 0),
                "pia_rate": front_pia_metrics.get(fid, {}).get("pia_rate"),
            })

        vi_val = alignment.get("VI")
        new_counts = Counter()
        for n, cid in lm.items():
            if n not in Gq:
                continue
            data = Gq.nodes[n]
            if str(data.get("pub_qtr", "")) == q:
                new_counts[int(cid)] += 1
        for cid, new_sz in new_counts.items():
            fid = curr_map.get(int(cid))
            if fid is None:
                continue
            pia_front = front_pia_metrics.get(int(fid), {"pia_eligible": 0, "pia_count": 0, "pia_rate": None})
            ts_rows.append({
                "lineage_id": int(fid),
                "period": q,
                "quarter": q,
                "new_works": int(new_sz),
                "n_communities": int(entry["n_communities"]),
                "VI_vs_prev_quarter": float(vi_val) if vi_val is not None else None,
                "pia_eligible": int(pia_front.get("pia_eligible", 0) or 0),
                "pia_count": int(pia_front.get("pia_count", 0) or 0),
                "pia_rate": pia_front.get("pia_rate"),
            })

        results["delta"].append(entry)
        states[q] = {
            "part_map": part_map,
            "cores": cores_q,
            "front_map": curr_map,
        }
        del Gfull
        del Gq

    def _set_summary_stats(key: str, values: list[float]) -> None:
        arr = [float(v) for v in values if v is not None]
        summary[f"{key}_mean"] = float(np.mean(arr)) if arr else None
        summary[f"{key}_median"] = float(np.median(arr)) if arr else None

    _set_summary_stats("vi_qoq", summary["vi_qoq_values"])
    _set_summary_stats("nvi_qoq", summary["nvi_qoq_values"])
    _set_summary_stats("vi_qoq_2y", summary["vi_qoq_2y_values"])
    _set_summary_stats("nvi_qoq_2y", summary["nvi_qoq_2y_values"])
    _set_summary_stats("vi_yoy", summary["vi_yoy_values"])
    _set_summary_stats("nvi_yoy", summary["nvi_yoy_values"])
    _set_summary_stats("vi_qv20", summary["vi_qv20_values"])
    _set_summary_stats("nvi_qv20", summary["nvi_qv20_values"])

    _set_summary_stats("density", summary["graph_density_values"])
    _set_summary_stats("avg_degree", summary["avg_degree_values"])
    _set_summary_stats("raw_communities", summary["raw_communities_values"])
    _set_summary_stats("filtered_communities", summary["filtered_communities_values"])
    _set_summary_stats("n_nodes", summary["n_nodes_values"])
    _set_summary_stats("n_edges", summary["n_edges_values"])
    _set_summary_stats("modularity", summary["modularity_values"])
    _set_summary_stats("coupling_pairs_retained", summary["coupling_pairs_retained_values"])
    _set_summary_stats("coupling_weight_pairs", summary["coupling_weight_pair_totals"])
    _set_summary_stats("coupling_edges", summary["coupling_edge_counts_values"])
    _set_summary_stats("hybrid_edges", summary["hybrid_edge_counts_values"])
    _set_summary_stats("citation_edges", summary["citation_edge_counts_values"])
    _set_summary_stats("coupling_weight_edges", summary["coupling_weight_edge_totals"])
    _set_summary_stats("coupling_weight_edges_mean", summary["coupling_weight_edge_means"])

    if summary["quarters"]:
        final_q = summary["quarters"][-1]
        summary["final_quarter"] = final_q
        summary["cluster_size_stats"] = summary["cluster_stats_by_quarter"].get(
            final_q,
            _cluster_size_summary([]),
        )
        summary["pia_summary"] = summary["pia_by_quarter"].get(
            final_q,
            {"pia_eligible": 0, "pia_count": 0, "pia_rate": None},
        )
        summary["nvi_qv20_final"] = summary["nvi_qv20_by_quarter"].get(final_q)
        summary["vi_qv20_final"] = summary["vi_qv20_by_quarter"].get(final_q)
        summary["vi_qoq_final"] = summary["vi_qoq_by_quarter"].get(final_q)
        summary["nvi_qoq_final"] = summary["nvi_qoq_by_quarter"].get(final_q)
        summary["vi_qoq_2y_final"] = summary["vi_qoq_2y_by_quarter"].get(final_q)
        summary["nvi_qoq_2y_final"] = summary["nvi_qoq_2y_by_quarter"].get(final_q)
        summary["vi_yoy_final"] = summary["vi_yoy_by_quarter"].get(final_q)
        summary["nvi_yoy_final"] = summary["nvi_yoy_by_quarter"].get(final_q)
        summary["density_final"] = summary["graph_density_by_quarter"].get(final_q)
        summary["avg_degree_final"] = summary["avg_degree_by_quarter"].get(final_q)
        summary["raw_communities_final"] = summary["raw_communities_by_quarter"].get(final_q)
        summary["filtered_communities_final"] = summary["filtered_communities_by_quarter"].get(final_q)
        summary["n_nodes_final"] = summary["n_nodes_by_quarter"].get(final_q)
        summary["n_edges_final"] = summary["n_edges_by_quarter"].get(final_q)
        summary["modularity_final"] = summary["modularity_by_quarter"].get(final_q)
        summary["largest_community_final"] = summary["largest_community_by_quarter"].get(final_q)
        summary["isolated_pct_final"] = summary["isolated_pct_by_quarter"].get(final_q)
        summary["coupling_stats_final"] = summary["coupling_stats_by_quarter"].get(final_q)
        summary["alerts_final"] = summary["alerts_by_quarter"].get(final_q)
    else:
        summary["final_quarter"] = None
        summary["cluster_size_stats"] = _cluster_size_summary([])
        summary["pia_summary"] = {"pia_eligible": 0, "pia_count": 0, "pia_rate": None}
        summary["nvi_qv20_final"] = None
        summary["vi_qv20_final"] = None
        summary["vi_qoq_final"] = None
        summary["nvi_qoq_final"] = None
        summary["vi_qoq_2y_final"] = None
        summary["nvi_qoq_2y_final"] = None
        summary["vi_yoy_final"] = None
        summary["nvi_yoy_final"] = None
        summary["density_final"] = None
        summary["avg_degree_final"] = None
        summary["raw_communities_final"] = None
        summary["filtered_communities_final"] = None
        summary["n_nodes_final"] = None
        summary["n_edges_final"] = None
        summary["modularity_final"] = None
        summary["largest_community_final"] = None
        summary["isolated_pct_final"] = None
        summary["coupling_stats_final"] = None
        summary["alerts_final"] = None

    results["summary"] = summary
    if not write_outputs:
        return results

    (out_dir / "communities_cumulative.json").write_text(json.dumps(results, indent=2))
    (lineage_dir / "lineage_registry.json").write_text(json.dumps(registry, indent=2))

    ts = (
        pd.DataFrame(ts_rows).sort_values(["lineage_id", "period"])
        if ts_rows
        else pd.DataFrame(
            columns=[
                "lineage_id",
                "period",
                "quarter",
                "new_works",
                "n_communities",
                "VI_vs_prev_quarter",
                "pia_eligible",
                "pia_count",
                "pia_rate",
            ]
        )
    )
    ts.to_csv(lineage_dir / "lineage_timeseries.csv", index=False)

    lineage_df = pd.DataFrame(lineage_metrics_rows)
    if lineage_df.empty:
        lineage_df = pd.DataFrame(columns=[
            "lineage_id",
            "quarter",
            "core_overlap_qoq",
            "core_overlap_yoy",
            "label_change_rate_qoq",
            "label_change_rate_yoy",
            "n_nodes_qoq",
            "n_nodes_yoy",
            "pia_eligible",
            "pia_count",
            "pia_rate",
        ])
    else:
        lineage_df = lineage_df.sort_values(["lineage_id", "quarter"])
    lineage_df.to_csv(lineage_dir / "lineage_metrics.csv", index=False)
    (out_dir / "communities_cumulative_summary.json").write_text(json.dumps(summary, indent=2))
    print("[Cumulative] Final cluster size stats:", summary.get("cluster_size_stats"))
    print("[Cumulative] Final PIA summary:", summary.get("pia_summary"))

    if write_outputs:
        write_cumulative_report(results, out_dir)

    return results


def _write_report(annual_json: dict, delta_json: dict, out_dir: Path, list_events: bool) -> None:
    """Persist a combined CSV (annual + quarterly) and optional event JSON."""
    rows_a = []
    for e in annual_json.get("annual", []):
        y, n = e.get("year"), e.get("n_communities", 0)
        al = e.get("alignment") or {}
        rows_a.append(
            {
                "year": y,
                "n_communities": n,
                "VI": al.get("VI"),
                "splits": al.get("splits"),
                "merges": al.get("merges"),
            }
        )
    df_a = (
        pd.DataFrame(rows_a).sort_values("year")
        if rows_a
        else pd.DataFrame(columns=["year", "n_communities", "VI", "splits", "merges"])
    )

    rows_d = []
    for e in (delta_json or {}).get("delta", []):
        vi = e.get(
            "VI_vs_prev_quarter",
            e.get("VI_vs_ref_annual", e.get("VI_vs_latest_annual")),
        )
        rows_d.append(
            {
                "quarter": e.get("quarter"),
                "prev_quarter": e.get("prev_quarter"),
                "n_communities": e.get("n_communities", 0),
                "VI_vs_prev_quarter": vi,
            }
        )
    df_d = (
        pd.DataFrame(rows_d).sort_values("quarter")
        if rows_d
        else pd.DataFrame(
            columns=["quarter", "prev_quarter", "n_communities", "VI_vs_prev_quarter"]
        )
    )

    report_csv = out_dir / "community_report.csv"
    with report_csv.open("w", encoding="utf-8") as f:
        f.write("# Annual summary\n")
        df_a.to_csv(f, index=False)
        f.write("\n# Delta summary\n")
        df_d.to_csv(f, index=False)
    print(f"Wrote {report_csv}")

    if list_events:
        ev_path = out_dir / "communities_events.json"
        events_json = []
        for e in annual_json.get("annual", []):
            al = e.get("alignment") or {}
            for ev in al.get("events", []) or []:
                ev2 = {"year": e.get("year"), **ev}
                events_json.append(ev2)
        for e in (delta_json or {}).get("delta", []):
            al = e.get("alignment") or {}
            for ev in al.get("events", []) or []:
                events_json.append({"quarter": e.get("quarter"), **ev})
        if events_json:
            ev_path.write_text(json.dumps(events_json, indent=2))
            print(f"Wrote {ev_path}")


def write_cumulative_report(results: dict, out_dir: Path) -> None:
    """Persist annual and quarterly summaries derived from the cumulative run."""
    delta_entries = results.get("delta") or []
    if not delta_entries:
        return

    def _entry_quarter(entry: dict) -> str:
        return str(entry.get("quarter", ""))

    sorted_entries = sorted(delta_entries, key=lambda e: _quarter_sort_key(_entry_quarter(e)))

    rows_quarter = []
    prev_q = None
    for entry in sorted_entries:
        quarter = _entry_quarter(entry)
        alignment = entry.get("alignment") or {}
        rows_quarter.append(
            {
                "quarter": quarter,
                "prev_quarter": prev_q,
                "n_communities": entry.get("n_communities"),
                "VI_vs_prev_quarter": entry.get("VI_QoQ"),
                "splits": alignment.get("splits"),
                "merges": alignment.get("merges"),
                "pia_rate": entry.get("pia_rate"),
            }
        )
        prev_q = quarter

    rows_annual = []
    for entry in sorted_entries:
        quarter = _entry_quarter(entry)
        if not quarter.endswith("Q4"):
            continue
        try:
            year = int(quarter[:4])
        except Exception:
            continue
        alignment = entry.get("alignment") or {}
        rows_annual.append(
            {
                "year": year,
                "quarter": quarter,
                "n_communities": entry.get("n_communities"),
                "VI_YoY": entry.get("VI_YoY"),
                "nVI_YoY": entry.get("nVI_YoY"),
                "VI_Q4_vs_prev_quarter": entry.get("VI_QoQ"),
                "splits_qoq": alignment.get("splits"),
                "merges_qoq": alignment.get("merges"),
                "pia_rate": entry.get("pia_rate"),
            }
        )

    df_quarter = pd.DataFrame(rows_quarter)
    df_annual = (
        pd.DataFrame(rows_annual).sort_values("year")
        if rows_annual
        else pd.DataFrame(
            columns=[
                "year",
                "quarter",
                "n_communities",
                "VI_YoY",
                "nVI_YoY",
                "VI_Q4_vs_prev_quarter",
                "splits_qoq",
                "merges_qoq",
                "pia_rate",
            ]
        )
    )

    report_csv = out_dir / "community_report.csv"
    with report_csv.open("w", encoding="utf-8") as f:
        f.write("# Annual summary (derived from cumulative run)\n")
        df_annual.to_csv(f, index=False)
        f.write("\n# Quarterly summary (derived from cumulative run)\n")
        df_quarter.to_csv(f, index=False)
    print(f"Wrote {report_csv}")


def run_annual(
    resolution: float,
    annual_min: int,
    annual_max: int,
    core_frac: float,
    overlap_min: int,
    overlap_frac: float,
    graphs_dir: Path,
    out_dir: Path,
    debug_events: bool = False,
    allow_external_pickle: bool = False,
) -> dict:
    """Run Leiden on annual graphs + build a front-id registry."""
    out_dir.mkdir(parents=True, exist_ok=True)
    lineage_dir = out_dir / "02_lineage_tracking"
    lineage_dir.mkdir(parents=True, exist_ok=True)

    results: dict = {"annual": []}
    ann = sorted(graphs_dir.glob("citation_graph_annual_*.pkl"))
    prev = None

    registry: dict[int, dict[int, int]] = {}
    next_lineage_id = count(start=1)
    prev_map: dict[int, int] | None = None

    for p in ann:
        year = int(p.stem.split("_")[-1])
        G = load_trusted_pickle(
            p,
            description="Annual graph pickle",
            allow_external=allow_external_pickle,
        )

        res = _cluster_fn(
            G,
            resolution=resolution,
            min_size=annual_min,
            max_size=annual_max,
        )
        entry = {"year": year, "n_communities": len(res["communities"])}
        curr_part = res["partition"]
        year_map: dict[int, int] = {}

        if prev:
            cores_prev = pagerank_core(prev["G"], prev["part"], core_frac=core_frac)
            cores_curr = pagerank_core(G, curr_part, core_frac=core_frac)
            matches = match_by_cores(cores_prev, cores_curr)
            overlaps = _full_core_overlaps(cores_prev, cores_curr)
            splits, merges, ev = _events_from_overlaps(overlaps, ov_min=overlap_min, ov_frac=overlap_frac)

            if debug_events:
                print(
                    f"[DEBUG] {year}: sample matches (prev->curr, ov) first 10:",
                    [(a, b, ov) for a, b, ov in sorted(matches, key=lambda x: -x[2])[:10]],
                )

            vi = variation_of_information(
                label_map_from_partition(prev["part"]),
                label_map_from_partition(curr_part),
            )

            entry["alignment"] = {
                "matches": [(int(a), int(b), int(ov)) for a, b, ov in matches],
                "overlaps": overlaps,
                "VI": float(vi),
                "splits": int(splits),
                "merges": int(merges),
                "events": ev,
            }

            if prev_map:
                for a_prev, b_curr, _ov in matches:
                    a_prev = int(a_prev)
                    b_curr = int(b_curr)
                    if a_prev in prev_map and b_curr not in year_map:
                        year_map[b_curr] = prev_map[a_prev]

        for cid in _unique_cids(curr_part):
            cid = int(cid) if isinstance(cid, (int, float, str)) else cid
            if cid not in year_map:
                year_map[cid] = next(next_lineage_id)

        registry[year] = year_map
        results["annual"].append(entry)
        prev = {"G": G, "part": curr_part}
        prev_map = year_map

    if results["annual"]:
        (out_dir / "communities_annual.json").write_text(json.dumps(results, indent=2))
        print("Saved", out_dir / "communities_annual.json")
        (lineage_dir / "lineage_registry_annual.json").write_text(json.dumps(registry, indent=2))
        print("Saved", lineage_dir / "lineage_registry_annual.json")
    else:
        print("[Annual] No annual graphs found. Skipping.")
    return results


def run_quarterly(
    resolution: float,
    delta_min: int,
    delta_max: int,
    core_frac: float,
    overlap_min: int,
    overlap_frac: float,
    align_mode: str,
    graphs_dir: Path,
    out_dir: Path,
    source: str = "delta",
    debug_events: bool = False,
    allow_external_pickle: bool = False,
) -> dict:
    """Run Leiden on quarterly graphs (delta or cumulative) and align consecutive quarters."""
    out_dir.mkdir(parents=True, exist_ok=True)
    lineage_dir = out_dir / "02_lineage_tracking"
    lineage_dir.mkdir(parents=True, exist_ok=True)

    raw_delta = sorted(graphs_dir.glob("citation_graph_delta_*.pkl" if source == "delta" else "citation_graph_cumulative_*.pkl"))
    selected: dict[str, Path] = {}
    for p in raw_delta:
        label = p.stem.split("_")[-1]
        canon = _canonical_quarter(label)
        selected.setdefault(canon, p)
    keys = sorted(selected.keys(), key=_quarter_sort_key)
    delta_files = [selected[k] for k in keys]
    if not delta_files:
        print("[Quarterly] No matching graphs found. Skipping.")
        return {"delta": []}

    results: dict = {"delta": []}
    registry: dict[str, dict[int, int]] = {}
    ts_rows: list[dict] = []

    next_lineage_id = count(start=1)
    prev = None
    prev_map: dict[int, int] | None = None
    prev_quarter: str | None = None

    for p in delta_files:
        raw_label = p.stem.split("_")[-1]
        q = _canonical_quarter(raw_label)
        Gq = load_trusted_pickle(
            p,
            description="Quarterly graph pickle",
            allow_external=allow_external_pickle,
        )
        rq = _cluster_fn(
            Gq,
            resolution=resolution,
            min_size=delta_min,
            max_size=delta_max,
        )
        entry: dict = {"quarter": q, "n_communities": len(rq["communities"])}
        curr_part = rq["partition"]
        curr_map: dict[int, int] = {}

        if prev:
            cores_prev = pagerank_core(prev["G"], prev["part"], core_frac=core_frac)
            cores_curr = pagerank_core(Gq, curr_part, core_frac=core_frac)
            matches = match_by_cores(cores_prev, cores_curr)
            overlaps = _full_core_overlaps(cores_prev, cores_curr)
            splits, merges, ev = _events_from_overlaps(overlaps, ov_min=overlap_min, ov_frac=overlap_frac)
            vi = variation_of_information(
                label_map_from_partition(prev["part"]),
                label_map_from_partition(curr_part),
            )
            entry.update(
                {
                    "prev_quarter": prev_quarter,
                    "VI_vs_prev_quarter": float(vi),
                    "alignment": {
                        "matches": [(int(a), int(b), int(ov)) for a, b, ov in matches],
                        "overlaps": overlaps,
                        "VI": float(vi),
                        "splits": int(splits),
                        "merges": int(merges),
                        "events": ev,
                    },
                }
            )
            if debug_events:
                print(
                    f"[DEBUG] {q}: matches prev={prev_quarter} count={len(matches)}",
                    [(a, b, ov) for a, b, ov in sorted(matches, key=lambda x: -x[2])[:10]],
                )
            if prev_map:
                for a_prev, b_curr, _ov in matches:
                    a_prev = int(a_prev)
                    b_curr = int(b_curr)
                    if a_prev in prev_map and b_curr not in curr_map:
                        curr_map[b_curr] = prev_map[a_prev]
        else:
            entry["alignment"] = {"matches": [], "overlaps": [], "VI": None, "splits": 0, "merges": 0, "events": []}

        for cid in _unique_cids(curr_part):
            cid = int(cid) if isinstance(cid, (int, float, str)) else cid
            if cid not in curr_map:
                curr_map[cid] = next(next_lineage_id)

        registry[q] = {int(k): int(v) for k, v in curr_map.items()}

        lm_q = _part_label_map(curr_part)
        q_sizes = Counter(int(v) for v in lm_q.values())
        vi_val = entry.get("VI_vs_prev_quarter")
        for q_cid, size in q_sizes.items():
            fid = curr_map.get(q_cid)
            if not fid:
                continue
            ts_rows.append(
                {
                    "lineage_id": int(fid),
                    "period": q,
                    "quarter": q,
                    "new_works": int(size),
                    "n_communities": int(entry["n_communities"]),
                    "VI_vs_prev_quarter": float(vi_val) if vi_val is not None else None,
                }
            )

        results["delta"].append(entry)
        prev = {"G": Gq, "part": curr_part}
        prev_map = curr_map
        prev_quarter = q

    (out_dir / "communities_delta.json").write_text(json.dumps(results, indent=2))
    print("Saved", out_dir / "communities_delta.json")

    (lineage_dir / "lineage_registry_quarter.json").write_text(
        json.dumps(registry, indent=2)
    )
    print("Saved", lineage_dir / "lineage_registry_quarter.json")

    ts_df = (
        pd.DataFrame(ts_rows).sort_values(["lineage_id", "period"])
        if ts_rows
        else pd.DataFrame(
            columns=["lineage_id", "period", "quarter", "new_works", "n_communities", "VI_vs_prev_quarter"]
        )
    )
    ts_path = lineage_dir / "lineage_timeseries_quarter.csv"
    ts_df.to_csv(ts_path, index=False)
    print(f"Saved {ts_path} rows={len(ts_df)}")

    if align_mode != "preceding":
        print(f"[Quarterly] align_mode '{align_mode}' is accepted but only 'preceding' is supported.")
    return results

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["cumulative", "annual", "delta", "both", "all"], default="cumulative",
                    help="Select which community pipelines to run (default cumulative).")
    from src.domain_registry import add_domain_args, resolve_script_paths
    add_domain_args(ap)
    ap.add_argument("--graphs-dir", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--res", type=float, default=0.001, help="Resolution for cumulative runs")
    ap.add_argument("--annual-res", type=float, default=0.001, help="Resolution for annual graphs")
    ap.add_argument("--delta-res", type=float, default=0.001, help="Resolution for quarterly graphs")
    ap.add_argument("--min", type=int, default=10, help="Minimum community size for cumulative runs")
    ap.add_argument("--max", type=int, default=0, help="Maximum community size for cumulative runs (0=adaptive)")
    ap.add_argument("--adaptive-max", dest="adaptive_max", action="store_true", default=True,
                    help="Enable adaptive maximum cluster size (default)")
    ap.add_argument("--no-adaptive-max", dest="adaptive_max", action="store_false",
                    help="Disable adaptive maximum cluster size")
    ap.add_argument("--max-fraction", type=float, default=0.05,
                    help="Target fraction of corpus for adaptive maximum (default 0.05 = 5%%)")
    ap.add_argument("--max-floor", type=int, default=1000,
                    help="Minimum cap applied when adaptive maximum is enabled")
    ap.add_argument("--max-ceiling", type=int, default=10000,
                    help="Maximum cap applied when adaptive maximum is enabled (0 disables)")
    ap.add_argument("--min-fraction", type=float, default=0.0005,
                    help="Minimum community size as a fraction of corpus when adaptive max is enabled (default 0.0005 = 0.05%%)")
    ap.add_argument("--max-floor-fraction", type=float, default=0.02,
                    help="Adaptive floor fraction for max size (default 0.02 = 2%% of corpus). Overrides --max-floor when > 0.")
    ap.add_argument("--max-ceiling-fraction", type=float, default=0.10,
                    help="Adaptive ceiling fraction for max size (default 0.10 = 10%% of corpus). Overrides --max-ceiling when > 0.")
    ap.add_argument("--min-absolute-floor", type=int, default=10,
                    help="Absolute minimum for adaptive max floor even if fraction is lower (default 10).")
    ap.add_argument("--max-absolute-ceiling", type=int, default=15000,
                    help="Absolute maximum for adaptive ceiling even if fraction is higher (default 15000).")
    ap.add_argument("--avg-degree-threshold", type=float, default=15.0,
                    help="Average degree threshold separating sparse vs. dense regimes (default 15).")
    ap.add_argument("--core-frac", type=float, default=0.10)
    ap.add_argument("--overlap-min", type=int, default=10)
    ap.add_argument("--overlap-frac", type=float, default=0.10)
    ap.add_argument("--window-quarters", type=int, default=None)
    ap.add_argument("--prune-age-years", type=int, default=None)
    ap.add_argument("--prune-min-in-degree", type=int, default=0)
    ap.add_argument("--cache-dir", type=Path, default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--limit-quarters", type=int, default=None)
    ap.add_argument("--res-sweep", type=str, default=None,
                    help="Comma-separated list of resolution parameters to sweep (cumulative only).")
    ap.add_argument("--res-sweep-only", action="store_true",
                    help="Only run the resolution sweep (skip writing the primary cumulative run).")
    ap.add_argument("--annual-min", type=int, default=50, help="Minimum community size for annual graphs")
    ap.add_argument("--annual-max", type=int, default=5000, help="Maximum community size for annual graphs")
    ap.add_argument("--delta-min", type=int, default=40, help="Minimum community size for quarterly graphs")
    ap.add_argument("--delta-max", type=int, default=5000, help="Maximum community size for quarterly graphs")
    ap.add_argument(
        "--align-deltas",
        choices=["preceding", "nearest"],
        default="preceding",
        help="Quarter alignment mode (currently only 'preceding' is supported).",
    )
    ap.add_argument("--source", choices=["delta", "cumulative"], default="cumulative",
                    help="Quarter graph source (citation_graph_delta_* or citation_graph_cumulative_*)")
    ap.add_argument("--use-ecg", action="store_true",
                    help="Use ECG ensemble clustering instead of Leiden (experimental).")
    ap.add_argument("--ecg-ens-size", type=int, default=16,
                    help="ECG ensemble size (default: 16). Only used with --use-ecg.")
    ap.add_argument("--debug-events", action="store_true", help="Print candidate split/merge overlaps")
    ap.add_argument(
        "--allow-external-pickle",
        action="store_true",
        help="Allow loading pickle graphs from outside the repository root.",
    )
    args = ap.parse_args()

    # Resolve domain-derived paths with legacy fallbacks
    paths = resolve_script_paths(args, REPO)
    if args.graphs_dir is None:
        args.graphs_dir = paths.graphs if paths else Path("data/current_graphs")
    if args.out_dir is None:
        args.out_dir = paths.out if paths else Path("data/out")
    if args.cache_dir is None:
        args.cache_dir = paths.cache_cum if paths else Path("data/out/cache_cum")

    try:
        import igraph  # noqa: F401
        import leidenalg  # noqa: F401
    except Exception as exc:
        raise SystemExit("Leiden requires: python-igraph + leidenalg installed.") from exc

    if args.use_ecg:
        try:
            import partition_igraph  # noqa: F401
        except ImportError as exc:
            raise SystemExit(
                "ECG requires: partition-igraph installed. "
                "Install: pip install partition-igraph>=0.0.7"
            ) from exc
        print(f"[ECG] Using ensemble clustering (ens_size={args.ecg_ens_size})")

    if args.use_ecg:
        _ecg_ens = args.ecg_ens_size

        def _ecg_wrapper(G, resolution, min_size, max_size):
            return run_ecg(
                G, resolution=resolution, min_size=min_size, max_size=max_size,
                ens_size=_ecg_ens,
            )

        global _cluster_fn
        _cluster_fn = _ecg_wrapper

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    mode_map = {
        "cumulative": ["cumulative"],
        "annual": ["annual"],
        "delta": ["delta"],
        "both": ["annual", "delta"],
        "all": ["cumulative", "annual", "delta"],
    }
    modes = mode_map.get(args.mode, ["cumulative"])

    # Cumulative run (with optional resolution sweep)
    if "cumulative" in modes:
        sweep_values: list[float] = []
        if args.res_sweep:
            try:
                sweep_values = [float(val.strip()) for val in args.res_sweep.split(",") if val.strip()]
            except ValueError as exc:
                raise SystemExit(f"Invalid --res-sweep values: {args.res_sweep}") from exc
        if sweep_values:
            print(f"[Sweep] Evaluating cumulative resolutions: {sweep_values}")
            sweep_summaries: list[dict[str, Any]] = []
            for res_val in sweep_values:
                sweep_result = run_cumulative(
                    graphs_dir=args.graphs_dir,
                    out_dir=out_dir,
                    resolution=res_val,
                    min_size=args.min,
                    max_size=args.max,
                    adaptive_max=args.adaptive_max,
                    max_fraction=args.max_fraction,
                    max_floor=args.max_floor,
                    max_ceiling=args.max_ceiling,
                    min_fraction=args.min_fraction,
                    max_floor_fraction=args.max_floor_fraction,
                    max_ceiling_fraction=args.max_ceiling_fraction,
                    min_absolute_floor=args.min_absolute_floor,
                    max_absolute_ceiling=args.max_absolute_ceiling,
                    avg_degree_threshold=args.avg_degree_threshold,
                    core_frac=args.core_frac,
                    overlap_min=args.overlap_min,
                    overlap_frac=args.overlap_frac,
                    window_quarters=args.window_quarters,
                    prune_age_years=args.prune_age_years,
                    prune_min_in_deg=args.prune_min_in_degree,
                    cache_dir=None,
                    resume=False,
                    force=False,
                    limit_quarters=args.limit_quarters,
                    write_outputs=False,
                    allow_external_pickle=args.allow_external_pickle,
                )
                sweep_summaries.append(sweep_result.get("summary", {"resolution": res_val}))
            sweep_path = out_dir / "resolution_sweep_cumulative.json"
            sweep_path.write_text(json.dumps(sweep_summaries, indent=2))
            print(f"[Sweep] Saved {sweep_path}")
            html_path = out_dir / "resolution_sweep.html"
            try:
                import subprocess
                python_exe = sys.executable
                cmd = [
                    python_exe,
                    str(Path(__file__).parent / "plot_resolution_sweep.py"),
                    "--input",
                    str(sweep_path),
                    "--out",
                    str(html_path),
                    "--static-out",
                    str(out_dir / "resolution_sweep.png"),
                    "--no-show",
                ]
                subprocess.check_call(cmd)
            except Exception as exc:
                print(f"[Sweep] Warning: failed to render HTML ({exc})")
            if args.res_sweep_only:
                return

        run_cumulative(
            graphs_dir=args.graphs_dir,
            out_dir=out_dir,
            resolution=args.res,
            min_size=args.min,
            max_size=args.max,
            adaptive_max=args.adaptive_max,
            max_fraction=args.max_fraction,
            max_floor=args.max_floor,
            max_ceiling=args.max_ceiling,
            min_fraction=args.min_fraction,
            max_floor_fraction=args.max_floor_fraction,
            max_ceiling_fraction=args.max_ceiling_fraction,
            min_absolute_floor=args.min_absolute_floor,
            max_absolute_ceiling=args.max_absolute_ceiling,
            core_frac=args.core_frac,
            overlap_min=args.overlap_min,
            overlap_frac=args.overlap_frac,
            window_quarters=args.window_quarters,
            prune_age_years=args.prune_age_years,
            prune_min_in_deg=args.prune_min_in_degree,
            cache_dir=args.cache_dir,
            resume=args.resume,
            force=args.force,
            limit_quarters=args.limit_quarters,
            write_outputs=True,
            allow_external_pickle=args.allow_external_pickle,
        )

    annual_json: dict = {"annual": []}
    delta_json: dict = {"delta": []}

    if "annual" in modes:
        annual_json = run_annual(
            resolution=args.annual_res,
            annual_min=args.annual_min,
            annual_max=args.annual_max,
            core_frac=args.core_frac,
            overlap_min=args.overlap_min,
            overlap_frac=args.overlap_frac,
            graphs_dir=args.graphs_dir,
            out_dir=out_dir,
            debug_events=args.debug_events,
            allow_external_pickle=args.allow_external_pickle,
        )

    if "delta" in modes:
        delta_json = run_quarterly(
            resolution=args.delta_res,
            delta_min=args.delta_min,
            delta_max=args.delta_max,
            core_frac=args.core_frac,
            overlap_min=args.overlap_min,
            overlap_frac=args.overlap_frac,
            align_mode=args.align_deltas,
            graphs_dir=args.graphs_dir,
            out_dir=out_dir,
            source=args.source,
            debug_events=args.debug_events,
            allow_external_pickle=args.allow_external_pickle,
        )

    if any(mode in ("annual", "delta") for mode in modes):
        _write_report(annual_json, delta_json, out_dir, list_events=True)


if __name__ == "__main__":
    main()




