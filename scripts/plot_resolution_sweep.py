"""
Visualize resolution sweep metrics for cumulative community detection runs.

Loads data/out/resolution_sweep_cumulative.json (or a supplied path) and produces
interactive multi-panel plots comparing cluster counts, cluster sizes, PIA rates,
and quarter-over-quarter variation of information across resolutions.

Outputs an interactive Plotly dashboard (hover to inspect values, pan/zoom,
range slider). Optionally export to HTML and/or a static image.

Usage:
    python scripts/plot_resolution_sweep.py --out data/out/resolution_sweep.html
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

DEFAULT_INPUT = Path("data/out/resolution_sweep_cumulative.json")
DEFAULT_HTML_OUTPUT: Path | None = None


def load_sweep(path: Path) -> list[dict]:
    data = json.loads(path.read_text())
    if not isinstance(data, list):
        raise ValueError("resolution_sweep_cumulative JSON must contain a list.")
    if not data:
        raise ValueError("resolution_sweep_cumulative JSON is empty.")
    return data


def reshape_quarter_metrics(entries: Iterable[dict]) -> pd.DataFrame:
    rows = []
    for entry in entries:
        res = entry["resolution"]
        quarters: list[str] = entry["quarters"]
        clusters_by_q = entry["cluster_stats_by_quarter"]
        pia_by_q = entry.get("pia_by_quarter") or {}
        density_by_q = entry.get("graph_density_by_quarter") or {}
        raw_by_q = entry.get("raw_communities_by_quarter") or {}
        filtered_by_q = entry.get("filtered_communities_by_quarter") or {}
        nodes_by_q = entry.get("n_nodes_by_quarter") or {}
        edges_by_q = entry.get("n_edges_by_quarter") or {}
        for q in quarters:
            stats = clusters_by_q.get(q) or {}
            pia = pia_by_q.get(q) or {}
            rows.append(
                {
                    "resolution": res,
                    "quarter": pd.Period(q, freq="Q").to_timestamp(how="end"),
                    "n_clusters": stats.get("n_clusters"),
                    "median_cluster": stats.get("median"),
                    "mean_cluster": stats.get("mean"),
                    "p90_cluster": stats.get("90th_percentile"),
                    "pia_rate": pia.get("pia_rate"),
                    "nvi_qv20": (entry.get("nvi_qv20_by_quarter") or {}).get(q),
                    "density": density_by_q.get(q),
                    "raw_communities": raw_by_q.get(q),
                    "filtered_communities": filtered_by_q.get(q),
                    "n_nodes": nodes_by_q.get(q),
                    "n_edges": edges_by_q.get(q),
                }
            )
    return pd.DataFrame(rows)


def reshape_vi_metrics(entries: Iterable[dict]) -> pd.DataFrame:
    rows = []
    for entry in entries:
        res = entry["resolution"]
        quarters: list[str] = entry["quarters"]
        vi_qoq_by_q = entry.get("vi_qoq_by_quarter") or {}
        nvi_qoq_by_q = entry.get("nvi_qoq_by_quarter") or {}
        vi_qoq_2y_by_q = entry.get("vi_qoq_2y_by_quarter") or {}
        nvi_qoq_2y_by_q = entry.get("nvi_qoq_2y_by_quarter") or {}
        vi_yoy_by_q = entry.get("vi_yoy_by_quarter") or {}
        nvi_yoy_by_q = entry.get("nvi_yoy_by_quarter") or {}
        vi_qv20_by_q = entry.get("vi_qv20_by_quarter") or {}
        nvi_qv20_by_q = entry.get("nvi_qv20_by_quarter") or {}
        for q in quarters:
            rows.append(
                {
                    "resolution": res,
                    "quarter": pd.Period(q, freq="Q").to_timestamp(how="end"),
                    "vi_qoq": vi_qoq_by_q.get(q),
                    "nvi_qoq": nvi_qoq_by_q.get(q),
                    "vi_qoq_2y": vi_qoq_2y_by_q.get(q),
                    "nvi_qoq_2y": nvi_qoq_2y_by_q.get(q),
                    "vi_yoy": vi_yoy_by_q.get(q),
                    "nvi_yoy": nvi_yoy_by_q.get(q),
                    "vi_qv20": vi_qv20_by_q.get(q),
                    "nvi_qv20": nvi_qv20_by_q.get(q),
                }
            )
    return pd.DataFrame(rows)


def extract_final_nvi(entries: Iterable[dict]) -> pd.DataFrame:
    rows = []
    for entry in entries:
        rows.append(
            {
                "resolution": entry.get("resolution"),
                "nvi_qv20_final": entry.get("nvi_qv20_final"),
                "vi_qv20_final": entry.get("vi_qv20_final"),
            }
        )
    return pd.DataFrame(rows).sort_values("resolution")


def plot_resolution_sweep(
    quarter_df: pd.DataFrame,
    vi_df: pd.DataFrame,
    final_df: pd.DataFrame,
    *,
    html_output: Path | None = None,
    static_output: Path | None = None,
    show: bool = True,
) -> None:
    quarter_df = quarter_df.copy()
    quarter_df["quarter_label"] = quarter_df["quarter"].dt.to_period("Q").astype(str)
    vi_df = vi_df.copy()
    vi_df["quarter_label"] = vi_df["quarter"].dt.to_period("Q").astype(str)
    final_df = final_df.copy()

    quarter_numeric_cols = [
        "n_clusters",
        "median_cluster",
        "mean_cluster",
        "p90_cluster",
        "pia_rate",
        "nvi_qv20",
        "density",
        "raw_communities",
        "filtered_communities",
        "n_nodes",
        "n_edges",
    ]
    for col in quarter_numeric_cols:
        if col in quarter_df.columns:
            quarter_df[col] = pd.to_numeric(quarter_df[col], errors="coerce")

    vi_numeric_cols = [
        "vi_qoq",
        "nvi_qoq",
        "vi_qoq_2y",
        "nvi_qoq_2y",
        "vi_yoy",
        "nvi_yoy",
        "vi_qv20",
        "nvi_qv20",
    ]
    for col in vi_numeric_cols:
        if col in vi_df.columns:
            vi_df[col] = pd.to_numeric(vi_df[col], errors="coerce")
    if "nvi_qv20_final" in final_df.columns:
        final_df["nvi_qv20_final"] = pd.to_numeric(final_df["nvi_qv20_final"], errors="coerce")
    if "vi_qv20_final" in final_df.columns:
        final_df["vi_qv20_final"] = pd.to_numeric(final_df["vi_qv20_final"], errors="coerce")

    resolutions = sorted(quarter_df["resolution"].unique())

    fig = make_subplots(
        rows=6,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=(
            "Number of clusters",
            "Cluster size (mean & median)",
            "PIA rate (lower is better)",
            "Quarter-over-quarter normalized VI (nVI_QoQ)",
            "5-year normalized VI (nVI_QvQ20)",
            "Time window selector",
        ),
    )

    color_cycle = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
    ]
    color_map = {res: color_cycle[i % len(color_cycle)] for i, res in enumerate(resolutions)}

    for idx, res in enumerate(resolutions):
        res_label = f"{res:g}"
        res_df = quarter_df[quarter_df["resolution"] == res]
        if res_df.empty:
            continue

        clusters_custom = [
            [res_label, q_label, value]
            for q_label, value in zip(res_df["quarter_label"], res_df["n_clusters"])
        ]
        fig.add_trace(
            go.Scatter(
                x=res_df["quarter"],
                y=res_df["n_clusters"],
                mode="lines+markers",
                name=res_label if idx == 0 else None,
                legendgroup=res_label,
                marker={"color": color_map[res]},
                line={"color": color_map[res]},
                customdata=clusters_custom,
                hovertemplate=(
                    "<b>Clusters</b><br>"
                    "Quarter: %{customdata[1]}<br>"
                    "Count: %{customdata[2]:.0f}<br>"
                    "Resolution: %{customdata[0]}<extra></extra>"
                ),
                showlegend=(idx == 0),
            ),
            row=1,
            col=1,
        )

        median_custom = [
            [res_label, q_label, value]
            for q_label, value in zip(res_df["quarter_label"], res_df["median_cluster"])
        ]
        fig.add_trace(
            go.Scatter(
                x=res_df["quarter"],
                y=res_df["median_cluster"],
                mode="lines+markers",
                name=None,
                legendgroup=res_label,
                marker={"color": color_map[res]},
                line={"color": color_map[res]},
                customdata=median_custom,
                hovertemplate=(
                    "<b>Cluster size (median)</b><br>"
                    "Quarter: %{customdata[1]}<br>"
                    "Median size: %{customdata[2]:.0f}<br>"
                    "Resolution: %{customdata[0]}<extra></extra>"
                ),
                showlegend=False,
            ),
            row=2,
            col=1,
        )

        mean_custom = [
            [res_label, q_label, value]
            for q_label, value in zip(res_df["quarter_label"], res_df["mean_cluster"])
        ]
        fig.add_trace(
            go.Scatter(
                x=res_df["quarter"],
                y=res_df["mean_cluster"],
                mode="lines+markers",
                name=None,
                legendgroup=res_label,
                marker={"color": color_map[res]},
                line={"color": color_map[res], "dash": "dash"},
                customdata=mean_custom,
                hovertemplate=(
                    "<b>Cluster size (mean)</b><br>"
                    "Quarter: %{customdata[1]}<br>"
                    "Mean size: %{customdata[2]:.1f}<br>"
                    "Resolution: %{customdata[0]}<extra></extra>"
                ),
                showlegend=False,
            ),
            row=2,
            col=1,
        )

        pia_custom = [
            [res_label, q_label, value]
            for q_label, value in zip(res_df["quarter_label"], res_df["pia_rate"])
        ]
        fig.add_trace(
            go.Scatter(
                x=res_df["quarter"],
                y=res_df["pia_rate"],
                mode="lines+markers",
                name=None,
                legendgroup=res_label,
                marker={"color": color_map[res]},
                line={"color": color_map[res]},
                customdata=pia_custom,
                hovertemplate=(
                    "<b>PIA rate</b><br>"
                    "Quarter: %{customdata[1]}<br>"
                    "Rate: %{customdata[2]:.3f}<br>"
                    "Resolution: %{customdata[0]}<extra></extra>"
                ),
                showlegend=False,
            ),
            row=3,
            col=1,
        )

        vi_res = vi_df[vi_df["resolution"] == res]
        if vi_res.empty:
            continue
        vi_customdata = [
            [res_label, q_label, raw_vi, nvi]
            for q_label, raw_vi, nvi in zip(
                vi_res["quarter_label"],
                vi_res["vi_qoq"],
                vi_res["nvi_qoq"],
            )
        ]
        fig.add_trace(
            go.Scatter(
                x=vi_res["quarter"],
                y=vi_res["nvi_qoq"],
                mode="lines+markers",
                name=res_label if idx == 0 else None,
                legendgroup=f"{res_label}_vi",
                marker={"color": color_map[res]},
                line={"color": color_map[res]},
                customdata=vi_customdata,
                hovertemplate=(
                    "<b>Normalized VI (QoQ)</b><br>"
                    "Quarter: %{customdata[1]}<br>"
                    "nVI: %{customdata[3]:.3f}<br>"
                    "VI: %{customdata[2]:.3f}<br>"
                    "Resolution: %{customdata[0]}<extra></extra>"
                ),
                showlegend=(idx == 0),
            ),
            row=4,
            col=1,
        )

        nvi_series = vi_res["nvi_qv20"] if "nvi_qv20" in vi_res else pd.Series(dtype=float)
        vi_qv20_series = vi_res["vi_qv20"] if "vi_qv20" in vi_res else pd.Series(dtype=float)
        if hasattr(nvi_series, "notna") and nvi_series.notna().any():
            nvi_customdata = [
                [res_label, q_label, vi_val, nvi_val]
                for q_label, vi_val, nvi_val in zip(
                    vi_res["quarter_label"],
                    vi_qv20_series,
                    nvi_series,
                )
            ]
            fig.add_trace(
                go.Scatter(
                    x=vi_res["quarter"],
                    y=nvi_series,
                    mode="lines+markers",
                    name=f"{res_label} nVI (5y)" if idx == 0 else None,
                    legendgroup=f"{res_label}_nvi",
                    marker={"color": color_map[res]},
                    line={"color": color_map[res]},
                    customdata=nvi_customdata,
                    hovertemplate=(
                        "<b>Normalized VI (5-year)</b><br>"
                        "Quarter: %{customdata[1]}<br>"
                        "nVI: %{customdata[3]:.3f}<br>"
                        "VI: %{customdata[2]:.3f}<br>"
                        "Resolution: %{customdata[0]}<extra></extra>"
                    ),
                    showlegend=(idx == 0),
                ),
                row=5,
                col=1,
            )

    slider_x = sorted(quarter_df["quarter"].unique())
    if slider_x:
        fig.add_trace(
            go.Scatter(
                x=slider_x,
                y=[0.0] * len(slider_x),
                mode="lines",
                line={"color": "rgba(0,0,0,0)"},
                hoverinfo="skip",
                showlegend=False,
            ),
            row=6,
            col=1,
        )

    fig.update_layout(
        height=1600,
        margin={"l": 60, "r": 20, "t": 80, "b": 40},
        legend={"title": "Resolution", "orientation": "h", "x": 0, "y": 1.05},
        hovermode="x unified",
    )
    for axis_idx in range(1, 6):
        fig.update_xaxes(
            tickformat="%Y-Q%q",
            row=axis_idx,
            col=1,
        )
    fig.update_xaxes(
        row=6,
        col=1,
        showticklabels=False,
        showgrid=False,
        rangeslider={"visible": True, "bgcolor": "rgba(0,0,0,0)", "thickness": 0.15},
    )
    fig.update_yaxes(title_text="Communities", row=1, col=1)
    fig.update_yaxes(title_text="Papers per cluster", row=2, col=1)
    fig.update_yaxes(title_text="PIA rate (↓ better)", row=3, col=1)
    fig.update_yaxes(title_text="nVI (QoQ)", row=4, col=1)
    fig.update_yaxes(title_text="nVI (5-year)", row=5, col=1)
    fig.update_yaxes(visible=False, row=6, col=1)
    fig.update_xaxes(title_text="Quarter", row=5, col=1)
    fig.update_xaxes(title_text="", row=6, col=1)

    if html_output:
        html_output.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(html_output), include_plotlyjs="cdn")
        print(f"[Write] Interactive HTML saved to {html_output}")
    if static_output:
        static_output.parent.mkdir(parents=True, exist_ok=True)
        try:
            fig.write_image(str(static_output), scale=2)
            print(f"[Write] Static image saved to {static_output}")
        except Exception as exc:
            print(f"[Write] Warning: unable to export static image ({exc})")
    if show and not html_output:
        fig.show()


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Resolution sweep JSON (default: {DEFAULT_INPUT})",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_HTML_OUTPUT,
        help="Optional HTML output path for the interactive chart.",
    )
    ap.add_argument(
        "--static-out",
        type=Path,
        default=None,
        help="Optional static image output (requires 'kaleido').",
    )
    ap.add_argument(
        "--show",
        action="store_true",
        help="Display interactive viewer (ignored when --out is provided).",
    )
    ap.add_argument(
        "--no-show",
        action="store_true",
        help="Deprecated. Use --show if you want to display the figure.",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    entries = load_sweep(args.input)

    quarter_df = reshape_quarter_metrics(entries)
    if quarter_df.empty:
        raise SystemExit("No quarter-level metrics found in the sweep file.")
    vi_df = reshape_vi_metrics(entries)
    if vi_df.empty:
        raise SystemExit("No VI metrics found in the sweep file.")
    final_df = extract_final_nvi(entries)

    plot_resolution_sweep(
        quarter_df,
        vi_df,
        final_df,
        html_output=args.out,
        static_output=args.static_out,
        show=args.show and not args.out,
    )


if __name__ == "__main__":
    main()
