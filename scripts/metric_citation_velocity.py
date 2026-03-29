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

from src.domain_registry import add_domain_args, resolve_script_paths  # noqa: E402
from src.metrics.common import (  # noqa: E402
    create_metric_metadata,
    ensure_dir,
    get_metric_output_paths,
    iter_quarter_slices,
    quarter_end,
    update_manifest,
    write_metric_metadata,
    write_metric_parquet,
    write_placeholder_metric,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute quarterly citation velocity statistics.")
    parser.add_argument("--slices-dir", default=None, type=Path)
    parser.add_argument("--out-dir", default=None, type=Path)
    parser.add_argument("--pattern", default="by_quarter__*.parquet")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--json-name", default="citation_velocity.json")
    parser.add_argument("--figure-name", default="citation_velocity.png")
    parser.add_argument("--min-age-years", type=float, default=0.25, help="Floor for age denominator.")
    parser.add_argument("--recent-window", type=float, default=2.0, help="Years defining a recent work.")
    parser.add_argument("--high-velocity-threshold", type=float, default=5.0, help="Velocity threshold for alert share.")
    parser.add_argument("--top-k", type=int, default=5, help="Number of top works to record per quarter.")
    add_domain_args(parser)
    return parser.parse_args()


def _resolve_publication_dates(df: pd.DataFrame) -> pd.Series:
    pub_dates = pd.to_datetime(df["publication_date"], errors="coerce", utc=True)
    missing = pub_dates.isna() & df["publication_year"].notna()
    if missing.any():
        years = (
            df.loc[missing, "publication_year"]
            .round()
            .astype("Int64")
            .astype(str)
        )
        filler = years + "-12-31"
        pub_dates.loc[missing] = pd.to_datetime(filler, errors="coerce", utc=True)
    return pub_dates


def compute_velocity(args: argparse.Namespace) -> tuple[dict[str, object], list[Path]]:
    quarters: list[dict[str, object]] = []
    input_files: list[Path] = []  # Track input files for provenance

    for idx, (quarter, path) in enumerate(iter_quarter_slices(args.slices_dir, args.pattern)):
        if args.limit is not None and idx >= args.limit:
            break
        input_files.append(path)  # Record input file
        df = pd.read_parquet(
            path,
            columns=["work_id", "title", "publication_date", "publication_year", "cited_by_count"],
        )
        if df.empty:
            quarters.append(
                {
                    "quarter": quarter,
                    "works": 0,
                    "mean_velocity": None,
                    "median_velocity": None,
                    "p90_velocity": None,
                    "share_high_velocity": None,
                    "recent_share": None,
                    "top_works": [],
                }
            )
            continue
        pub_dates = _resolve_publication_dates(df)
        quarter_end_ts = quarter_end(quarter).tz_localize("UTC")
        age_days = (quarter_end_ts - pub_dates).dt.days
        age_years = age_days / 365.25
        age_years = age_years.clip(lower=0)

        age_for_velocity = np.maximum(age_years.to_numpy(dtype="float64"), args.min_age_years)
        cited = df["cited_by_count"].fillna(0).to_numpy(dtype="float64")
        velocity = cited / age_for_velocity

        valid_mask = np.isfinite(velocity)
        velocity = velocity[valid_mask]
        age_years = age_years[valid_mask]
        cited = cited[valid_mask]
        df_valid = df.loc[valid_mask].reset_index(drop=True)

        if len(df_valid) == 0:
            quarters.append(
                {
                    "quarter": quarter,
                    "works": 0,
                    "mean_velocity": None,
                    "median_velocity": None,
                    "p90_velocity": None,
                    "share_high_velocity": None,
                    "recent_share": None,
                    "top_works": [],
                }
            )
            continue

        mean_velocity = float(np.mean(velocity))
        median_velocity = float(np.median(velocity))
        p90_velocity = float(np.percentile(velocity, 90))
        share_high_velocity = float(np.mean(velocity >= args.high_velocity_threshold))
        recent_share = float(np.mean(age_years <= args.recent_window))

        top_idx = np.argsort(-velocity)[: args.top_k]
        top_works = []
        for i in top_idx:
            title_val = df_valid.loc[i, "title"]
            title_serializable = None if pd.isna(title_val) else str(title_val)
            top_works.append(
                {
                    "work_id": str(df_valid.loc[i, "work_id"]),
                    "title": title_serializable,
                    "citation_velocity": float(velocity[i]),
                    "cited_by_count": float(cited[i]),
                    "age_years": float(age_years.iat[i]),
                }
            )

        quarters.append(
            {
                "quarter": quarter,
                "works": int(len(df_valid)),
                "mean_velocity": mean_velocity,
                "median_velocity": median_velocity,
                "p90_velocity": p90_velocity,
                "share_high_velocity": share_high_velocity,
                "recent_share": recent_share,
                "top_works": top_works,
            }
        )

    payload = {
        "metric": "citation_velocity",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "parameters": {
            "min_age_years": args.min_age_years,
            "recent_window": args.recent_window,
            "high_velocity_threshold": args.high_velocity_threshold,
            "top_k": args.top_k,
        },
        "quarters": quarters,
    }
    return payload, input_files


def render_plot(payload: dict[str, object], out_path: Path) -> None:
    quarters = [row["quarter"] for row in payload["quarters"]]
    medians = [row["median_velocity"] or 0 for row in payload["quarters"]]
    p90s = [row["p90_velocity"] or 0 for row in payload["quarters"]]
    shares = [row["share_high_velocity"] or 0 for row in payload["quarters"]]

    # Compute rolling averages for trendlines (4-quarter = 1 year smoothing)
    medians_series = pd.Series(medians)
    p90s_series = pd.Series(p90s)
    shares_series = pd.Series(shares)
    medians_trend = medians_series.rolling(window=4, center=True, min_periods=1).mean()
    p90s_trend = p90s_series.rolling(window=4, center=True, min_periods=1).mean()
    shares_trend = shares_series.rolling(window=4, center=True, min_periods=1).mean()

    fig, ax1 = plt.subplots(figsize=(12, 6))
    # Plot quarterly data with transparency
    ax1.plot(quarters, medians, color="#1f77b4", linewidth=1, alpha=0.3)
    ax1.plot(quarters, p90s, color="#ff7f0e", linewidth=1, alpha=0.3)
    # Plot trendlines prominently
    ax1.plot(quarters, medians_trend, label="Median velocity (trend)", color="#1f77b4", linewidth=2.5)
    ax1.plot(quarters, p90s_trend, label="90th percentile velocity (trend)", color="#ff7f0e", linewidth=2.5)
    ax1.set_ylabel("Citations per year")
    ax1.tick_params(axis="x", rotation=75)

    ax2 = ax1.twinx()
    # Plot quarterly share with transparency, then trendline
    ax2.plot(quarters, shares, color="#2ca02c", linewidth=1, alpha=0.3, linestyle="--")
    ax2.plot(quarters, shares_trend, label="Share ≥ threshold (trend)", color="#2ca02c", linewidth=2.5, linestyle="--")
    ax2.set_ylabel("Share of works above threshold", color="#2ca02c")
    ax2.tick_params(axis='y', labelcolor="#2ca02c")
    # Use a narrower range to show variation better (shares typically 0.5-0.9)
    shares_arr = [s for s in shares if s is not None and s > 0]
    if shares_arr:
        min_share = max(0, min(shares_arr) - 0.05)
        max_share = min(1, max(shares_arr) + 0.05)
        ax2.set_ylim(min_share, max_share)

    ax1.set_xlabel("Quarter")
    ax1.set_title("Citation Velocity Trends (4-quarter rolling average)")
    lines = ax1.get_lines() + ax2.get_lines()
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, loc="upper right")

    step = max(1, len(quarters) // 16)
    ax1.set_xticks(range(0, len(quarters), step))
    ax1.set_xticklabels([quarters[i] for i in range(0, len(quarters), step)])

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def write_standardized_outputs(
    payload: dict[str, object],
    input_files: list[Path],
    args: argparse.Namespace,
) -> None:
    """
    Write standardized parquet outputs and metadata for citation velocity metric.
    """
    metric_name = "citation_velocity"

    # Convert quarters data to DataFrame for global level
    if not payload["quarters"]:
        return

    df_global = pd.DataFrame([
        {
            "quarter": row["quarter"],
            "value": row["median_velocity"],  # Primary metric: median citation velocity
            "mean_velocity": row["mean_velocity"],
            "p90_velocity": row["p90_velocity"],
            "share_high_velocity": row["share_high_velocity"],
            "recent_share": row["recent_share"],
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
        description="Quarterly citation velocity tracking how rapidly works accumulate citations",
        formula="velocity = citations / max(age_years, min_age_years); aggregated by median/mean/p90",
        units="citations per year",
        parameters={
            "slices_dir": str(args.slices_dir),
            "pattern": args.pattern,
            "min_age_years": args.min_age_years,
            "recent_window": args.recent_window,
            "high_velocity_threshold": args.high_velocity_threshold,
            "num_input_files": len(input_files),
        },
        input_files=input_files,  # Track all input slice files
        level="global",
        column_descriptions={
            "quarter": "Quarter identifier (YYYYQN format)",
            "value": "Median citation velocity across works",
            "mean_velocity": "Mean citation velocity",
            "p90_velocity": "90th percentile citation velocity",
            "share_high_velocity": f"Share of works with velocity >= {args.high_velocity_threshold}",
            "recent_share": f"Share of works published within {args.recent_window} years",
            "works": "Number of works in quarter",
        },
    )

    write_metric_metadata(metadata, paths_global["metadata"])

    # Update central manifest (Task 1.2)
    manifest_path = args.out_dir / "manifest.json"
    update_manifest(manifest_path, metric_name, "global", metadata, paths_global)

    placeholder_reason = (
        "Per-front/per-lineage citation velocity metrics require membership exports; "
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
    args.slices_dir = args.slices_dir or (paths.slices if paths else Path("data/current_ingest/slices"))
    args.out_dir = args.out_dir or (paths.out / "metrics" if paths else Path("data/out/metrics"))
    ensure_dir(args.out_dir)
    payload, input_files = compute_velocity(args)

    # Legacy JSON output (backward compatibility)
    json_path = args.out_dir / args.json_name
    json_path.write_text(json.dumps(payload, indent=2))
    print(f"Wrote {json_path}")

    # Figure output
    figure_path = args.out_dir / args.figure_name
    if payload["quarters"]:
        render_plot(payload, figure_path)
        print(f"Wrote {figure_path}")
    else:
        if figure_path.exists():
            figure_path.unlink()

    # Standardized parquet outputs with provenance tracking (Task 1.1 + 1.2)
    write_standardized_outputs(payload, input_files, args)


if __name__ == "__main__":
    main()
