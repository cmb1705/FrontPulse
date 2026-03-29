#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import math
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from _path_bootstrap import ensure_repo_imports

_REPO = ensure_repo_imports()

from src import trusted_io  # noqa: E402


@dataclass(frozen=True)
class ParameterSet:
    alpha: float
    beta: float
    lambda_decay: float


def _parse_float_list(text: str) -> list[float]:
    parts = [p.strip() for p in str(text).split(",")]
    values: list[float] = []
    for part in parts:
        if not part:
            continue
        try:
            values.append(float(part))
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"Cannot parse float value from '{part}'") from exc
    if not values:
        raise argparse.ArgumentTypeError("At least one numeric value is required.")
    return values


def _quarter_sort_key(path: Path) -> tuple[int, int, str]:
    import re

    m = re.search(r"(\d{4})Q([1-4])", path.name)
    if m:
        return int(m.group(1)), int(m.group(2)), path.name
    return (0, 0, path.name)


def _collect_graphs(args: argparse.Namespace) -> list[Path]:
    paths: list[Path] = []
    if args.graphs:
        for raw in args.graphs:
            p = Path(raw)
            if not p.exists():
                print(f"[Warn] Graph not found: {p}", file=sys.stderr)
                continue
            paths.append(p)
    if not paths and args.graphs_dir:
        graphs_dir = Path(args.graphs_dir)
        if not graphs_dir.exists():
            raise SystemExit(f"--graphs-dir '{graphs_dir}' does not exist.")
        candidates = sorted(graphs_dir.glob("citation_graph_cumulative_*.pkl"), key=_quarter_sort_key)
        if not candidates:
            print(f"[Warn] No citation_graph_cumulative_*.pkl files found in {graphs_dir}", file=sys.stderr)
        else:
            if args.latest:
                candidates = candidates[-1:]
            elif args.limit and args.limit > 0:
                candidates = candidates[-args.limit :]
            paths.extend(candidates)
    if not paths:
        raise SystemExit("No graphs provided. Use --graphs or --graphs-dir.")
    deduped = []
    seen = set()
    for p in paths:
        if p not in seen:
            deduped.append(p)
            seen.add(p)
    return deduped


def _load_graph(path: Path) -> nx.DiGraph:
    G = trusted_io.load_trusted_binary(
        path, description="citation graph",
    )
    if not isinstance(G, nx.DiGraph):
        raise SystemExit(f"Graph at {path} is not a directed NetworkX graph.")
    return G


def _baseline_metrics(G: nx.DiGraph) -> dict[str, float]:
    total_weights: list[float] = []
    coupling_weights: list[float] = []
    citation_weights: list[float] = []
    coupling_edges = 0
    hybrid_edges = 0
    citation_edges = 0
    coupling_only_edges = 0

    for _, _, data in G.edges(data=True):
        wt = float(data.get("weight_total", data.get("weight", 0.0)) or 0.0)
        total_weights.append(wt)
        wc = float(data.get("weight_coupling", 0.0) or 0.0)
        if wc > 0:
            coupling_edges += 1
            coupling_weights.append(wc)
            str(data.get("edge_type", "")).lower()
            citation_weight = float(data.get("weight_citation", 0.0) or 0.0)
            if citation_weight > 0:
                hybrid_edges += 1
            else:
                coupling_only_edges += 1
        citation_weight = float(data.get("weight_citation", 0.0) or 0.0)
        if citation_weight > 0:
            citation_edges += 1
            citation_weights.append(citation_weight)

    return {
        "weight_total_sum": float(sum(total_weights)),
        "weight_total_mean": float(np.mean(total_weights)) if total_weights else 0.0,
        "weight_coupling_sum": float(sum(coupling_weights)),
        "weight_coupling_mean": float(np.mean(coupling_weights)) if coupling_weights else 0.0,
        "weight_coupling_max": float(max(coupling_weights)) if coupling_weights else 0.0,
        "weight_citation_sum": float(sum(citation_weights)),
        "weight_citation_mean": float(np.mean(citation_weights)) if citation_weights else 0.0,
        "total_edges": float(G.number_of_edges()),
        "coupling_active_edges": float(coupling_edges),
        "hybrid_edges": float(hybrid_edges),
        "coupling_only_edges": float(coupling_only_edges),
        "citation_edges": float(citation_edges),
    }


def _evaluate(
    G: nx.DiGraph,
    params: ParameterSet,
    *,
    tolerance: float,
) -> dict[str, float]:
    total_weight_sum = 0.0
    total_weight_values: list[float] = []
    coupling_weight_sum = 0.0
    coupling_weight_values: list[float] = []
    citation_weight_sum = 0.0
    citation_weight_values: list[float] = []

    total_edges = G.number_of_edges()
    citation_edges = 0
    coupling_active_edges = 0
    hybrid_edges = 0
    coupling_only_edges = 0

    for _, _, data in G.edges(data=True):
        citation_weight = float(data.get("weight_citation", 0.0) or 0.0)
        coupling_score = data.get("coupling_score")
        if coupling_score is None:
            coupling_score = 0.0
        try:
            coupling_score = float(coupling_score)
        except Exception:
            coupling_score = 0.0
        year_diff = data.get("coupling_year_diff", 0)
        try:
            year_diff = abs(float(year_diff))
        except Exception:
            year_diff = 0.0

        coupling_weight = 0.0
        if coupling_score > 0:
            try:
                coupling_weight = params.beta * coupling_score * math.exp(-params.lambda_decay * year_diff)
            except OverflowError:
                coupling_weight = 0.0

        total_weight = params.alpha * citation_weight + coupling_weight
        total_weight_sum += total_weight
        total_weight_values.append(total_weight)

        if citation_weight > tolerance:
            citation_edges += 1
            citation_weight_sum += params.alpha * citation_weight
            citation_weight_values.append(params.alpha * citation_weight)

        if coupling_weight > tolerance:
            coupling_active_edges += 1
            coupling_weight_sum += coupling_weight
            coupling_weight_values.append(coupling_weight)
            if citation_weight > tolerance:
                hybrid_edges += 1
            else:
                coupling_only_edges += 1

    return {
        "weight_total_sum": float(total_weight_sum),
        "weight_total_mean": float(np.mean(total_weight_values)) if total_weight_values else 0.0,
        "weight_total_max": float(max(total_weight_values)) if total_weight_values else 0.0,
        "weight_citation_sum": float(citation_weight_sum),
        "weight_citation_mean": float(np.mean(citation_weight_values)) if citation_weight_values else 0.0,
        "weight_coupling_sum": float(coupling_weight_sum),
        "weight_coupling_mean": float(np.mean(coupling_weight_values)) if coupling_weight_values else 0.0,
        "weight_coupling_max": float(max(coupling_weight_values)) if coupling_weight_values else 0.0,
        "total_edges": float(total_edges),
        "citation_edges": float(citation_edges),
        "coupling_active_edges": float(coupling_active_edges),
        "hybrid_edges": float(hybrid_edges),
        "coupling_only_edges": float(coupling_only_edges),
    }


def _deltas(new: dict[str, float], baseline: dict[str, float]) -> dict[str, float | None]:
    deltas: dict[str, float | None] = {}
    for key, new_val in new.items():
        base_val = baseline.get(key)
        if base_val is None:
            deltas[f"delta_{key}"] = None
            deltas[f"delta_{key}_pct"] = None
            continue
        delta_abs = float(new_val) - float(base_val)
        deltas[f"delta_{key}"] = delta_abs
        if base_val not in (None, 0, 0.0):
            deltas[f"delta_{key}_pct"] = (float(new_val) / float(base_val)) - 1.0
        else:
            deltas[f"delta_{key}_pct"] = None
    return deltas


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate sensitivity of bibliographic coupling weights to parameter choices."
    )
    parser.add_argument("--graphs", nargs="+", help="Specific graph pickle paths to evaluate.")
    parser.add_argument(
        "--graphs-dir",
        type=str,
        help="Directory containing citation_graph_cumulative_*.pkl graphs (used when --graphs not provided).",
    )
    parser.add_argument(
        "--latest",
        action="store_true",
        help="When scanning --graphs-dir, only evaluate the most recent cumulative graph.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit the number of graphs (most recent first) when using --graphs-dir.",
    )
    parser.add_argument(
        "--alphas",
        type=_parse_float_list,
        default=[1.0],
        help="Comma separated alpha values (default: 1.0).",
    )
    parser.add_argument(
        "--betas",
        type=_parse_float_list,
        default=[0.3],
        help="Comma separated beta values (default: 0.3).",
    )
    parser.add_argument(
        "--lambdas",
        type=_parse_float_list,
        default=[0.15],
        help="Comma separated lambda decay values (default: 0.15).",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-6,
        help="Numerical tolerance when classifying coupling edges (default: 1e-6).",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="data/out/coupling_param_sensitivity.csv",
        help="Output CSV path (default: data/out/coupling_param_sensitivity.csv).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting the output CSV if it already exists.",
    )

    args = parser.parse_args(argv)

    paths = _collect_graphs(args)
    combos = [
        ParameterSet(alpha=a, beta=b, lambda_decay=l)
        for a, b, l in itertools.product(args.alphas, args.betas, args.lambdas)
    ]
    if not combos:
        raise SystemExit("No parameter combinations to evaluate.")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and not args.overwrite:
        raise SystemExit(f"Output file {out_path} already exists. Use --overwrite to replace it.")

    rows: list[dict[str, object]] = []

    for graph_path in paths:
        print(f"[Sensitivity] Loading {graph_path}")
        G = _load_graph(graph_path)
        baseline = _baseline_metrics(G)

        graph_label = graph_path.stem
        for params in combos:
            metrics = _evaluate(G, params, tolerance=args.tolerance)
            delta_vals = _deltas(metrics, baseline)
            row: dict[str, object] = {
                "graph": str(graph_path),
                "graph_label": graph_label,
                "alpha": params.alpha,
                "beta": params.beta,
                "lambda_decay": params.lambda_decay,
            }
            row.update(baseline)
            row.update({f"new_{k}": v for k, v in metrics.items()})
            row.update(delta_vals)
            rows.append(row)

    df = pd.DataFrame(rows)
    df = df.sort_values(["graph_label", "alpha", "beta", "lambda_decay"]).reset_index(drop=True)
    df.to_csv(out_path, index=False)
    print(f"[Sensitivity] Wrote {len(df)} rows to {out_path}")


if __name__ == "__main__":
    main()
