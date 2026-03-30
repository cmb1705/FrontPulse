from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from _path_bootstrap import ensure_repo_imports

REPO_ROOT = ensure_repo_imports()

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.domain_registry import (  # noqa: E402
    add_domain_args,
    apply_domain_path_defaults,
    resolve_script_paths,
)
from src.metrics.common import (  # noqa: E402
    create_metric_metadata,
    ensure_dir,
    get_metric_output_paths,
    iter_quarter_slices,
    update_manifest,
    write_metric_metadata,
    write_metric_parquet,
    write_placeholder_metric,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Measure quarterly topic diversity.")
    parser.add_argument("--slices-dir", default=None, type=Path)
    parser.add_argument("--out-dir", default=None, type=Path)
    parser.add_argument("--pattern", default="by_quarter__*.parquet")
    parser.add_argument("--topic-column", default="primary_topic_name")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--json-name", default="topic_diversity.json")
    parser.add_argument("--figure-name", default="topic_diversity.png")
    parser.add_argument("--top-topics", type=int, default=5, help="Number of top topics to plot in the stack chart.")
    add_domain_args(parser)
    return parser.parse_args()


def compute_diversity(args: argparse.Namespace) -> tuple[dict[str, object], list[Path]]:
    quarters: list[dict[str, object]] = []
    topic_shares: list[dict[str, object]] = []
    input_files: list[Path] = []  # Track input files for provenance
    top_n = max(1, int(args.top_topics))

    for idx, (quarter, path) in enumerate(iter_quarter_slices(args.slices_dir, args.pattern)):
        if args.limit is not None and idx >= args.limit:
            break
        input_files.append(path)  # Record input file
        df = pd.read_parquet(path, columns=[args.topic_column])
        series = df[args.topic_column].dropna()
        series = series.astype(str)
        counts = series.value_counts()
        total = int(counts.sum())
        if total == 0:
            quarters.append(
                {
                    "quarter": quarter,
                    "works": 0,
                    "unique_topics": 0,
                    "shannon_entropy": None,
                    "effective_topics": None,
                    "herfindahl_index": None,
                    "diversity_index": None,
                    "top_topics": [],
                }
            )
            continue

        probabilities = counts / total
        entropy = float(-(probabilities * np.log(probabilities)).sum())
        effective_topics = float(np.exp(entropy))
        herfindahl = float((probabilities ** 2).sum())
        diversity_index = 1.0 - herfindahl
        top_topics = [
            {"topic": topic, "share": float(probabilities.loc[topic]), "count": int(counts.loc[topic])}
            for topic in probabilities.nlargest(top_n).index
        ]
        quarters.append(
            {
                "quarter": quarter,
                "works": total,
                "unique_topics": int(counts.size),
                "shannon_entropy": entropy,
                "effective_topics": effective_topics,
                "herfindahl_index": herfindahl,
                "diversity_index": diversity_index,
                "top_topics": top_topics,
            }
        )
        for topic, share in probabilities.items():
            topic_shares.append({"quarter": quarter, "topic": topic, "share": float(share)})

    payload = {
        "metric": "topic_diversity",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "parameters": {
            "topic_column": args.topic_column,
            "top_topics": args.top_topics,
        },
        "quarters": quarters,
        "topic_shares": topic_shares,
    }
    return payload, input_files


def render_plot(payload: dict[str, object], out_path: Path, top_topics: int) -> None:
    quarters = [row["quarter"] for row in payload["quarters"]]
    effective = [row["effective_topics"] or 0 for row in payload["quarters"]]
    diversity = [row["diversity_index"] or 0 for row in payload["quarters"]]

    # Compute rolling averages for trendlines (4-quarter = 1 year smoothing)
    effective_series = pd.Series(effective)
    diversity_series = pd.Series(diversity)
    effective_trend = effective_series.rolling(window=4, center=True, min_periods=1).mean()
    diversity_trend = diversity_series.rolling(window=4, center=True, min_periods=1).mean()

    share_df = pd.DataFrame(payload["topic_shares"])
    if not share_df.empty:
        topic_order = (
            share_df.groupby("topic")["share"]
            .mean()
            .sort_values(ascending=False)
            .head(top_topics)
            .index
            .tolist()
        )
        share_df["topic_group"] = share_df["topic"].where(share_df["topic"].isin(topic_order), "Other")
        pivot = (
            share_df.groupby(["quarter", "topic_group"])["share"]
            .sum()
            .unstack(fill_value=0.0)
            .reindex(index=quarters, fill_value=0.0)
        )
    else:
        pivot = pd.DataFrame(index=quarters)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    # Plot effective topics on left y-axis
    ax1.plot(quarters, effective, color="#1f77b4", linewidth=1, alpha=0.3)
    ax1.plot(quarters, effective_trend, label="Effective number of topics (trend)", color="#1f77b4", linewidth=2.5)
    ax1.set_ylabel("Effective number of topics", color="#1f77b4")
    ax1.tick_params(axis='y', labelcolor="#1f77b4")
    ax1.set_title("Topic Diversity Over Time (4-quarter rolling average)")

    # Plot diversity index on right y-axis (secondary)
    ax1_2 = ax1.twinx()
    ax1_2.plot(quarters, diversity, color="#ff7f0e", linewidth=1, alpha=0.3, linestyle="--")
    ax1_2.plot(quarters, diversity_trend, label="Diversity index (1 - HHI) (trend)", color="#ff7f0e", linewidth=2.5, linestyle="--")
    ax1_2.set_ylabel("Diversity index (1 - HHI)", color="#ff7f0e")
    ax1_2.tick_params(axis='y', labelcolor="#ff7f0e")
    ax1_2.set_ylim(0.5, 1.0)  # Typical range for diversity index

    # Combine legends
    lines1 = ax1.get_lines() + ax1_2.get_lines()
    labels1 = [line.get_label() for line in lines1]
    ax1.legend(lines1, labels1, loc="upper right")

    if not pivot.empty:
        pivot.plot.area(ax=ax2, linewidth=0)
        ax2.set_ylabel("Topic share")
        ax2.set_title("Topic Composition (Top segments)")
        ax2.set_ylim(0, 1)
    else:
        ax2.set_ylabel("Topic share")
        ax2.set_title("No topic data available")

    ax2.tick_params(axis="x", rotation=75)
    ax2.set_xlabel("Quarter")
    step = max(1, len(quarters) // 16)
    ax2.set_xticks(range(0, len(quarters), step))
    ax2.set_xticklabels([quarters[i] for i in range(0, len(quarters), step)])

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def write_standardized_outputs(
    payload: dict[str, object],
    input_files: list[Path],
    args: argparse.Namespace,
) -> None:
    """
    Write standardized parquet outputs and metadata for topic diversity metric.
    """
    metric_name = "topic_diversity"

    # Convert quarters data to DataFrame for global level
    if not payload["quarters"]:
        return

    df_global = pd.DataFrame([
        {
            "quarter": row["quarter"],
            "value": row["diversity_index"],  # Primary metric: diversity index (1 - HHI)
            "shannon_entropy": row["shannon_entropy"],
            "effective_topics": row["effective_topics"],
            "herfindahl_index": row["herfindahl_index"],
            "unique_topics": row["unique_topics"],
            "works": row["works"],
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
        description="Quarterly topic diversity tracking distribution across research topics",
        formula="diversity_index = 1 - HHI, where HHI = sum(p_i^2); effective_topics = exp(shannon_entropy)",
        units="dimensionless diversity score (0-1); dimensionless effective count; nats for entropy",
        parameters={
            "slices_dir": str(args.slices_dir),
            "pattern": args.pattern,
            "topic_column": args.topic_column,
            "top_topics": args.top_topics,
            "num_input_files": len(input_files),
        },
        input_files=input_files,  # Track all input slice files
        level="global",
        column_descriptions={
            "quarter": "Quarter identifier (YYYYQN format)",
            "value": "Diversity index (1 - Herfindahl-Hirschman Index)",
            "shannon_entropy": "Shannon entropy of topic distribution (in nats)",
            "effective_topics": "Effective number of topics (exp(shannon_entropy))",
            "herfindahl_index": "Herfindahl-Hirschman Index (sum of squared shares)",
            "unique_topics": "Count of distinct topics in quarter",
            "works": "Total works in quarter",
        },
    )

    write_metric_metadata(metadata, paths_global["metadata"])

    # Update central manifest (Task 1.2)
    manifest_path = args.out_dir / "manifest.json"
    update_manifest(manifest_path, metric_name, "global", metadata, paths_global)

    placeholder_reason = (
        "Per-front/per-lineage topic diversity metrics require membership exports; "
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
        "slices_dir": ("slices", "", "data/current_ingest/slices"),
        "out_dir": ("out", "metrics", "data/out/metrics"),
    })
    args.slices_dir = Path(args.slices_dir)
    args.out_dir = Path(args.out_dir)
    ensure_dir(args.out_dir)
    payload, input_files = compute_diversity(args)

    # Legacy JSON output (backward compatibility)
    json_path = args.out_dir / args.json_name
    json_path.write_text(json.dumps(payload, indent=2))
    print(f"Wrote {json_path}")

    # Figure output
    figure_path = args.out_dir / args.figure_name
    if payload["quarters"]:
        render_plot(payload, figure_path, args.top_topics)
        print(f"Wrote {figure_path}")
    else:
        if figure_path.exists():
            figure_path.unlink()

    # Standardized parquet outputs with provenance tracking (Task 1.1 + 1.2)
    write_standardized_outputs(payload, input_files, args)


if __name__ == "__main__":
    main()
