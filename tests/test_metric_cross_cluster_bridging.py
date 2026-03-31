from __future__ import annotations

from pathlib import Path

import networkx as nx
import pandas as pd

from scripts import metric_cross_cluster_bridging as bridging


def test_build_undirected_collapses_directionality() -> None:
    graph = nx.DiGraph()
    graph.add_node("a", publication_date="2020-01-01")
    graph.add_node("b", publication_date="2020-02-01")
    graph.add_edge("a", "b")
    graph.add_edge("b", "a")

    undirected = bridging.build_undirected(graph)

    assert not undirected.is_directed()
    assert undirected.degree("a") == 1
    assert undirected.degree("b") == 1
    assert set(undirected.neighbors("a")) == {"b"}


def test_suggest_worker_count_scales_down_for_large_graphs(
    monkeypatch,
    tmp_path: Path,
) -> None:
    graph_path = tmp_path / "graph.pkl"
    with graph_path.open("wb") as handle:
        handle.truncate(200 * 1024 * 1024)

    monkeypatch.setattr(bridging, "MEMORY_UTILS_AVAILABLE", True)
    monkeypatch.setattr(
        bridging,
        "get_memory_info",
        lambda: {"available": 64.0, "used": 0.0, "total": 64.0, "percent": 0.0},
    )

    workers, reason = bridging.suggest_worker_count(
        graph_paths=[graph_path] * 16,
        requested_workers=16,
        memory_reserve_gb=8.0,
    )

    assert 1 <= workers < 16
    assert "estimated" in reason


def test_enrich_top_nodes_with_metadata(tmp_path: Path) -> None:
    ingest_path = tmp_path / "ingest.parquet"
    pd.DataFrame(
        [
            {"work_id": "W1", "title": "Paper One", "cited_by_count": 7},
            {"work_id": "W2", "title": "Paper Two", "cited_by_count": 13},
        ]
    ).to_parquet(ingest_path)

    quarters = [
        {
            "quarter": "2020Q1",
            "top_nodes": [
                {"node_id": "W2", "bridge_ratio": 0.8},
                {"node_id": "missing", "bridge_ratio": 0.5},
            ],
        }
    ]

    bridging.enrich_top_nodes_with_metadata(quarters, ingest_path)

    assert quarters[0]["top_nodes"][0]["title"] == "Paper Two"
    assert quarters[0]["top_nodes"][0]["cited_by_count"] == 13
    assert quarters[0]["top_nodes"][1]["title"] is None
    assert quarters[0]["top_nodes"][1]["cited_by_count"] is None
