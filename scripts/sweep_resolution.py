#run: python scripts/sweep_resolution.py --graph data/out/graphs/citation_graph_annual_2024.pkl --res 0.8,1.0,1.2,1.4 --out data/out/res_sweep_2024.json

# scripts/sweep_resolution.py
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

repo_root = Path(__file__).resolve().parents[1]
sys.path.append(str(repo_root))  # now `import src.*` works

try:
    from plotly.subplots import make_subplots  # type: ignore
    import plotly.graph_objects as go  # type: ignore
except Exception:  # pragma: no cover
    make_subplots = None  # type: ignore
    go = None  # type: ignore

from src import trusted_io
from src.community import run_leiden, compute_pia_flags
from src.alignment import variation_of_information


def _parse_resolutions(spec: str) -> List[float]:
    values: List[float] = []
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        try:
            values.append(float(token))
        except ValueError as exc:
            raise SystemExit(f"Invalid resolution '{token}' in --res") from exc
    if not values:
        raise SystemExit("No resolution values provided.")
    return values


def _plot_results(
    df: pd.DataFrame,
    html_out: Path | None,
    static_out: Path | None,
    show: bool,
) -> None:
    if make_subplots is None or go is None:
        print("[Plot] Plotly not available; skipping HTML export.")
        return

    fig = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=(
            "# communities",
            "Median cluster size",
            "Cluster size percentiles (p10/p90)",
            "PIA rate and nVI_QvQ20",
        ),
    )

    fig.add_trace(
        go.Scatter(
            x=df["resolution"],
            y=df["n_communities"],
            mode="lines+markers",
            name="# communities",
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=df["resolution"],
            y=df["median_size"],
            mode="lines+markers",
            name="Median size",
            marker=dict(color="#1f77b4"),
            line=dict(color="#1f77b4"),
        ),
        row=2,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=df["resolution"],
            y=df["p10"],
            mode="lines+markers",
            name="p10",
            marker=dict(color="#ff7f0e"),
            line=dict(color="#ff7f0e"),
        ),
        row=3,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=df["resolution"],
            y=df["p90"],
            mode="lines+markers",
            name="p90",
            marker=dict(color="#2ca02c"),
            line=dict(color="#2ca02c"),
        ),
        row=3,
        col=1,
    )

    ax4 = go.Scatter(
        x=df["resolution"],
        y=df["pia_rate"],
        mode="lines+markers",
        name="PIA rate",
        marker=dict(color="#d62728"),
        line=dict(color="#d62728"),
        yaxis="y4",
    )
    fig.add_trace(ax4, row=4, col=1)
    fig.add_trace(
        go.Scatter(
            x=df["resolution"],
            y=df["nVI_QvQ20"],
            mode="lines+markers",
            name="nVI_QvQ20",
            marker=dict(color="#9467bd"),
            line=dict(color="#9467bd"),
        ),
        row=4,
        col=1,
    )

    fig.update_layout(
        height=1200,
        margin=dict(l=60, r=20, t=80, b=40),
        hovermode="x unified",
    )
    fig.update_xaxes(title_text="Resolution", row=4, col=1)
    fig.update_yaxes(title_text="# communities", row=1, col=1)
    fig.update_yaxes(title_text="Median size", row=2, col=1)
    fig.update_yaxes(title_text="Cluster size", row=3, col=1)
    fig.update_yaxes(title_text="PIA rate", row=4, col=1)

    if html_out:
        html_out.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(html_out), include_plotlyjs="cdn")
        print(f"[Plot] Interactive HTML saved to {html_out}")

    if static_out:
        static_out.parent.mkdir(parents=True, exist_ok=True)
        try:
            fig.write_image(str(static_out), scale=2)
            print(f"[Plot] Static image saved to {static_out}")
        except ValueError as exc:  # pragma: no cover
            print(f"[Plot] Unable to export static image ({exc}). Install 'kaleido'.")

    if show and html_out is None:
        fig.show()


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Sweep Leiden CPM resolutions on a single graph and plot results."
    )
    ap.add_argument(
        "--graph",
        required=True,
        help="Graph pickle to evaluate (e.g., data/out/graphs/citation_graph_annual_2024.pkl)",
    )
    ap.add_argument(
        "--res",
        default="0.8,1.0,1.2,1.4",
        help="Comma-separated list of resolution parameters (default: %(default)s)",
    )
    ap.add_argument("--min", type=int, default=50, help="Minimum community size")
    ap.add_argument("--max", type=int, default=5000, help="Maximum community size")
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("data/out/res_sweep.json"),
        help="JSON file to write summary results (default: %(default)s)",
    )
    ap.add_argument(
        "--html-out",
        type=Path,
        default=None,
        help="Optional HTML output path for the interactive plot "
        "(default: derived from --out)",
    )
    ap.add_argument(
        "--static-out",
        type=Path,
        default=None,
        help="Optional static image output (requires kaleido)",
    )
    ap.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip interactive/static plot generation",
    )
    ap.add_argument(
        "--show",
        action="store_true",
        help="Display the interactive window (ignored when --html-out is provided)",
    )
    args = ap.parse_args()

    graph_path = Path(args.graph)
    if not graph_path.exists():
        raise SystemExit(f"Graph not found: {graph_path}")

    G = trusted_io.load_trusted_binary(
        graph_path, description="citation graph",
    )

    resolutions = _parse_resolutions(args.res)
    results = []
    prev_partition = None
    prev_size = None
    for r in resolutions:
        res = run_leiden(G, resolution=r, min_size=args.min, max_size=args.max)
        sizes = [len(c["nodes"]) for c in res["communities"]]
        med = float(np.median(sizes)) if sizes else 0.0
        p10 = float(np.percentile(sizes, 10)) if sizes else 0.0
        p90 = float(np.percentile(sizes, 90)) if sizes else 0.0
        row = {
            "resolution": r,
            "n_communities": len(sizes),
            "median_size": med,
            "p10": p10,
            "p90": p90,
        }

        label_map = {str(node): int(cid) for node, cid in res["partition"]}
        pia_result = compute_pia_flags(G, label_map, min_links=20, within_ratio=0.10)
        row["pia_rate"] = pia_result["totals"].get("pia_rate")

        if prev_partition is not None:
            vi_val = variation_of_information(prev_partition, label_map)
            n = len(set(prev_partition) | set(label_map))
            row["nVI_QvQ20"] = (vi_val / np.log2(n)) if n > 1 else None
        else:
            row["nVI_QvQ20"] = None
        prev_partition = label_map

        results.append(row)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"graph": str(graph_path), "results": results}, indent=2))
    print(f"Wrote {args.out}")

    if not args.no_plot:
        df = pd.DataFrame(results).sort_values("resolution")
        html_out = args.html_out or args.out.with_suffix(".html")
        _plot_results(
            df=df,
            html_out=html_out,
            static_out=args.static_out,
            show=args.show,
        )


if __name__ == "__main__":
    main()
