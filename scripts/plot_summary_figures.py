#!/usr/bin/env python3
from __future__ import annotations

"""
Regenerate cumulative summary graphics with readable axes.

Produces/overwrites:
  - cumulative_nvi.png
  - nvi_horizons_composite.png
  - nvi_heatmap.png
  - nvi_qoq_anomalies.png
  - communities_vs_qoq.png
  - vi_vs_resolution.png
  - clusters_vs_resolution.png
  - pia_vs_resolution.png
  - composite_vs_resolution.png
  - alerts_tripwire_alerts_by_quarter.png
  - alerts_tripwire_tail_strength.png
  - alerts_tripwire_top_fronts.png
"""

import argparse
import json
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _read_delta_df(path: Path) -> pd.DataFrame:
    data = json.loads(path.read_text())
    delta = data.get("delta") or []
    if not delta:
        raise RuntimeError(f"No 'delta' entries found in {path}")
    df = pd.DataFrame(delta)
    df["quarter"] = pd.PeriodIndex(df["quarter"].astype(str), freq="Q").astype(str)
    df = df.sort_values("quarter")
    return df


def _quarter_xticks(ax, labels: Sequence[str]) -> None:
    n = len(labels)
    if n == 0:
        return
    step = max(1, n // 12)
    idx = np.arange(0, n, step)
    ax.set_xticks(idx)
    ax.set_xticklabels([labels[i] for i in idx], rotation=45, ha="right")
    ax.margins(x=0.01)


def plot_nvi_horizons(df: pd.DataFrame, outdir: Path) -> None:
    metrics = [
        ("nVI_QoQ", "nVI QoQ"),
        ("nVI_QoQ_2y", "nVI QoQ (2y window)"),
        ("nVI_YoY", "nVI YoY"),
        ("nVI_QvQ20", "nVI 5-year"),
    ]
    quarters = df["quarter"].tolist()
    x = np.arange(len(quarters))

    plt.figure(figsize=(10, 6))
    for col, label in metrics:
        if col in df:
            plt.plot(x, df[col], label=label)
    plt.title("Cumulative nVI horizons")
    plt.ylabel("nVI")
    plt.xlabel("Quarter (labels on Q1)")
    _quarter_xticks(plt.gca(), quarters)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "cumulative_nvi.png", dpi=150)
    plt.close()


def plot_nvi_composite(df: pd.DataFrame, outdir: Path) -> None:
    metrics = [
        ("nVI_QoQ", "nVI QoQ"),
        ("nVI_QoQ_2y", "nVI QoQ (2y window)"),
        ("nVI_YoY", "nVI YoY"),
        ("nVI_QvQ20", "nVI 5-year"),
    ]
    quarters = df["quarter"].tolist()
    x = np.arange(len(quarters))

    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    axes = axes.flatten()
    for ax, (col, label) in zip(axes, metrics):
        if col in df:
            ax.plot(x, df[col], color="#1f77b4")
            ax.set_title(label)
            ax.set_ylabel("nVI")
            _quarter_xticks(ax, quarters)
    for ax in axes:
        ax.grid(alpha=0.2, linewidth=0.5)
    fig.suptitle("nVI horizons (individual views)", fontsize=14)
    fig.tight_layout(rect=[0, 0.02, 1, 0.96])
    fig.savefig(outdir / "nvi_horizons_composite.png", dpi=150)
    plt.close(fig)


def plot_nvi_heatmap(df: pd.DataFrame, outdir: Path) -> None:
    quarters = df["quarter"].tolist()
    metrics = ["nVI_QoQ", "nVI_QoQ_2y", "nVI_YoY", "nVI_QvQ20"]
    heat = df[metrics].to_numpy().T
    fig, ax = plt.subplots(figsize=(10, 3))
    im = ax.imshow(heat, aspect="auto", cmap="viridis")
    ax.set_yticks(np.arange(len(metrics)))
    ax.set_yticklabels(metrics)
    _quarter_xticks(ax, quarters)
    ax.set_title("nVI heatmap across horizons")
    fig.colorbar(im, ax=ax, label="nVI")
    fig.tight_layout()
    fig.savefig(outdir / "nvi_heatmap.png", dpi=150)
    plt.close(fig)


def plot_nvi_qoq_anomalies(df: pd.DataFrame, outdir: Path) -> None:
    quarters = df["quarter"].tolist()
    x = np.arange(len(quarters))
    values = df["nVI_QoQ"].to_numpy()
    thresh = np.nanmean(values) + 2 * np.nanstd(values)

    plt.figure(figsize=(10, 4.5))
    plt.plot(x, values, label="nVI QoQ", color="#1f77b4")
    plt.axhline(thresh, color="red", linestyle="--", alpha=0.6, label="Mean + 2σ")
    plt.scatter(x[values > thresh], values[values > thresh], color="red", zorder=5, label="Exceeds threshold")
    plt.title("nVI QoQ anomalies")
    plt.ylabel("nVI QoQ")
    _quarter_xticks(plt.gca(), quarters)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "nvi_qoq_anomalies.png", dpi=150)
    plt.close()


def plot_communities_vs_qoq(df: pd.DataFrame, outdir: Path) -> None:
    quarters = df["quarter"].tolist()
    x = np.arange(len(quarters))
    plt.figure(figsize=(10, 4.5))
    plt.plot(x, df["n_communities"], label="Communities per quarter", color="#2ca02c")
    plt.title("Communities vs quarter")
    plt.ylabel("# communities")
    _quarter_xticks(plt.gca(), quarters)
    plt.grid(alpha=0.2, linewidth=0.5)
    plt.tight_layout()
    plt.savefig(outdir / "communities_vs_qoq.png", dpi=150)
    plt.close()


def plot_denominators(df: pd.DataFrame, outdir: Path) -> None:
    quarters = df["quarter"].tolist()
    x = np.arange(len(quarters))
    denom_cols = [
        ("N_QoQ", "N QoQ"),
        ("N_QoQ_2y", "N QoQ (2y window)"),
        ("N_YoY", "N YoY"),
        ("N_QvQ20", "N 5-year"),
    ]
    available = [(col, label) for col, label in denom_cols if col in df]
    if not available:
        return
    plt.figure(figsize=(10, 4.5))
    for col, label in available:
        plt.plot(x, df[col], label=label)
    plt.title("Denominator sizes (N)")
    plt.ylabel("N")
    _quarter_xticks(plt.gca(), quarters)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "denominators_N.png", dpi=150)
    plt.close()


def _read_resolution_df(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise RuntimeError(f"{path} not found")
    data = json.loads(path.read_text())
    df = []
    for item in data:
        row = {}
        row["resolution"] = item.get("resolution")
        cluster_stats = item.get("cluster_size_stats") or {}
        row["clusters"] = cluster_stats.get("n_clusters")
        row["max_cluster_size"] = cluster_stats.get("max")
        row["vi_qoq_mean"] = item.get("vi_qoq_mean")
        pia = item.get("pia_summary") or {}
        row["pia_rate"] = pia.get("pia_rate")
        df.append(row)
    return pd.DataFrame(df).sort_values("resolution")


def plot_resolution_curves(df: pd.DataFrame, outdir: Path) -> None:
    if df.empty:
        return
    def plot_line(filename: str, ycol: str, ylabel: str, title: str) -> None:
        if ycol not in df:
            return
        plt.figure(figsize=(6, 4))
        plt.plot(df["resolution"], df[ycol], marker="o")
        plt.title(title)
        plt.xlabel("Resolution")
        plt.ylabel(ylabel)
        plt.xticks(df["resolution"], rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(outdir / filename, dpi=150)
        plt.close()

    plot_line("vi_vs_resolution.png", "vi_qoq_mean", "Mean nVI QoQ", "nVI vs resolution")
    plot_line("clusters_vs_resolution.png", "clusters", "# communities", "Communities vs resolution")
    plot_line("pia_vs_resolution.png", "pia_rate", "PIA rate", "PIA rate vs resolution")

    plt.figure(figsize=(8, 6))
    ax1 = plt.gca()
    ax1.plot(df["resolution"], df["clusters"], color="#2ca02c", marker="o", label="# communities")
    ax1.set_ylabel("# communities", color="#2ca02c")
    ax1.tick_params(axis="y", labelcolor="#2ca02c")
    ax1.set_xlabel("Resolution")
    ax2 = ax1.twinx()
    ax2.plot(df["resolution"], df["vi_qoq_mean"], color="#1f77b4", marker="s", label="Mean nVI QoQ")
    ax2.set_ylabel("Mean nVI QoQ", color="#1f77b4")
    ax2.tick_params(axis="y", labelcolor="#1f77b4")
    plt.title("Resolution composite view")
    ax1.set_xticks(df["resolution"])
    ax1.set_xticklabels(df["resolution"], rotation=45, ha="right")
    fig = plt.gcf()
    fig.tight_layout()
    fig.savefig(outdir / "composite_vs_resolution.png", dpi=150)
    plt.close(fig)


def plot_tripwire_figures(csv_path: Path, outdir: Path) -> None:
    if not csv_path.exists():
        return
    df = pd.read_csv(csv_path)
    if df.empty:
        return
    if "period" not in df:
        return
    df["period"] = pd.PeriodIndex(df["period"].astype(str), freq="Q").astype(str)
    df.sort_values("period", inplace=True)

    alert_counts = df.groupby("period")["alert"].sum().reset_index()
    plt.figure(figsize=(10, 4.5))
    plt.bar(alert_counts["period"].index, alert_counts["alert"], color="#d62728")
    plt.title("Alerts per quarter")
    plt.ylabel("# alerts")
    _quarter_xticks(plt.gca(), alert_counts["period"].tolist())
    plt.tight_layout()
    plt.savefig(outdir / "alerts_tripwire_alerts_by_quarter.png", dpi=150)
    plt.close()

    plt.figure(figsize=(10, 4.5))
    plt.plot(df["period"], df["q_value"], ".", alpha=0.3, label="q-values")
    plt.axhline(0.10, color="red", linestyle="--", label="α=0.10")
    _quarter_xticks(plt.gca(), df["period"].tolist())
    plt.ylabel("q-value")
    plt.title("Tripwire tail strength")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "alerts_tripwire_tail_strength.png", dpi=150)
    plt.close()

    top_fronts = (
        df[df["alert"]]
        .groupby("front_id")
        .agg(alerts=("alert", "sum"), mean_rr=("rr_obs_over_mu", "mean"))
        .reset_index()
        .sort_values("alerts", ascending=False)
        .head(15)
    )
    if not top_fronts.empty:
        plt.figure(figsize=(8, 6))
        plt.barh(top_fronts["front_id"].astype(str), top_fronts["alerts"], color="#9467bd")
        plt.xlabel("# alerts")
        plt.ylabel("Front ID")
        plt.title("Top alerting fronts")
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig(outdir / "alerts_tripwire_top_fronts.png", dpi=150)
        plt.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cumulative-json", default="data/out/communities_cumulative.json")
    ap.add_argument("--resolution-json", default="data/out/resolution_sweep_cumulative.json")
    ap.add_argument("--tripwire-csv", default="data/out/05_tripwire_detection/alerts_tripwire.csv")
    ap.add_argument("--out-dir", default="data/out")
    args = ap.parse_args()

    outdir = Path(args.out_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    df_cum = _read_delta_df(Path(args.cumulative_json))
    plot_nvi_horizons(df_cum, outdir)
    plot_nvi_composite(df_cum, outdir)
    plot_nvi_heatmap(df_cum, outdir)
    plot_nvi_qoq_anomalies(df_cum, outdir)
    plot_communities_vs_qoq(df_cum, outdir)
    plot_denominators(df_cum, outdir)

    df_res = _read_resolution_df(Path(args.resolution_json))
    if not df_res.empty:
        plot_resolution_curves(df_res, outdir)

    plot_tripwire_figures(Path(args.tripwire_csv), outdir)


if __name__ == "__main__":
    main()
