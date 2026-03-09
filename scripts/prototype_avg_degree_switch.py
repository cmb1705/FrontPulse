#!/usr/bin/env python3
"""
Quick prototype: inspect per-quarter graph density and average degree for the
cumulative citation graphs captured in ``resolution_sweep_cumulative.json``.

Outputs a table highlighting when density-based vs. average-degree-based regime
switches would activate.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import pandas as pd

GRAPHS_DIR = Path("data/current_graphs")


def load_quarter_stats(graph_dir: Path) -> pd.DataFrame:
    files = sorted(graph_dir.glob("citation_graph_cumulative_*.pkl"))
    if not files:
        raise SystemExit("No cumulative graphs found (expected citation_graph_cumulative_*.pkl).")

    def _quarter_from_path(path: Path) -> str:
        stem = path.stem
        # filenames look like citation_graph_cumulative_2000Q1.pkl or citation_graph_cumulative_2000Q1_geom.pkl
        for token in stem.split("_"):
            if len(token) == 6 and token[:4].isdigit() and token[-2] == "Q" and token[-1].isdigit():
                return token
        raise ValueError(f"Cannot parse quarter from {path.name}")

    rows = []
    for path in files:
        q = _quarter_from_path(path)
        with path.open("rb") as fh:
            G = pickle.load(fh)
        n_nodes = G.number_of_nodes()
        n_edges = G.number_of_edges()
        if n_nodes > 1:
            density = n_edges / (n_nodes * (n_nodes - 1))
        else:
            density = 0.0
        avg_deg_directed = (n_edges / n_nodes) if n_nodes else 0.0
        avg_deg_undirected = (2 * n_edges / n_nodes) if n_nodes else 0.0
        rows.append(
            {
                "quarter_label": q,
                "quarter": pd.Period(q, freq="Q").to_timestamp(how="end"),
                "n_nodes": n_nodes,
                "n_edges": n_edges,
                "density": density,
                "avg_deg_directed": avg_deg_directed,
                "avg_deg_undirected": avg_deg_undirected,
            }
        )

    df = pd.DataFrame(rows).sort_values("quarter").reset_index(drop=True)
    return df


def summarize_thresholds(df: pd.DataFrame) -> None:
    density_thresholds = [0.0005, 0.0002, 0.00018, 0.00015]
    avg_degree_thresholds = [10, 12, 15, 18, 20]

    print("Density threshold activation windows:")
    for thr in density_thresholds:
        active = df[df["density"] >= thr]
        if active.empty:
            print(f"  >= {thr:.5f}: never triggers")
        else:
            print(
                f"  >= {thr:.5f}: {active.iloc[0]['quarter_label']}"
                f" -> {active.iloc[-1]['quarter_label']} (n={len(active)})"
            )

    print("\nAverage degree (directed) threshold activation windows:")
    for thr in avg_degree_thresholds:
        active = df[df["avg_deg_directed"] >= thr]
        if active.empty:
            print(f"  >= {thr:.0f}: never triggers")
        else:
            print(
                f"  >= {thr:.0f}: {active.iloc[0]['quarter_label']}"
                f" -> {active.iloc[-1]['quarter_label']} (n={len(active)})"
            )


def main() -> None:
    df = load_quarter_stats(GRAPHS_DIR)

    print("Quarterly graph stats (head/tail):")
    print(df.head(5)[["quarter", "n_nodes", "n_edges", "density", "avg_deg_directed"]].to_string(index=False))
    print("...")
    print(df.tail(5)[["quarter", "n_nodes", "n_edges", "density", "avg_deg_directed"]].to_string(index=False))
    print()
    summarize_thresholds(df)


if __name__ == "__main__":
    main()
