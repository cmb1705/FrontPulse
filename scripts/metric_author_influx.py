from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path

from _path_bootstrap import ensure_repo_imports

REPO_ROOT = ensure_repo_imports()

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

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
    parser = argparse.ArgumentParser(description="Compute quarterly author influx metrics.")
    parser.add_argument("--slices-dir", default=None, type=Path)
    parser.add_argument("--out-dir", default=None, type=Path)
    parser.add_argument("--pattern", default="by_quarter__*.parquet")
    parser.add_argument("--limit", type=int, default=None, help="Optional cap on number of quarters.")
    parser.add_argument("--json-name", default="author_influx.json")
    parser.add_argument("--figure-name", default="author_influx.png")
    add_domain_args(parser)
    return parser.parse_args()


def normalize_author_ids(raw_value: object) -> Iterable[str]:
    if raw_value is None:
        return []
    if isinstance(raw_value, str):
        return [val.strip() for val in raw_value.split(",") if val.strip()]
    if isinstance(raw_value, (list, tuple, set)):
        return [str(val).strip() for val in raw_value if str(val).strip()]
    if pd.isna(raw_value):
        return []
    return [str(raw_value).strip()] if str(raw_value).strip() else []


def compute_author_influx(args: argparse.Namespace) -> tuple[dict[str, object], list[Path]]:
    entries: list[dict[str, object]] = []
    seen_authors: set[str] = set()
    cumulative_counts: list[int] = []
    input_files: list[Path] = []  # Track input files for provenance

    for idx, (quarter, path) in enumerate(iter_quarter_slices(args.slices_dir, args.pattern)):
        if args.limit is not None and idx >= args.limit:
            break
        input_files.append(path)  # Record input file
        schema = pq.read_schema(path)
        columns = list(schema.names)

        chosen_col: str | None = None
        fallback_single_author = False
        for candidate in ("author_ids", "author_id", "authors", "first_author_id"):
            if candidate in columns:
                chosen_col = candidate
                if candidate == "first_author_id":
                    fallback_single_author = True
                break
        if chosen_col is None:
            raise ValueError(
                f"Unable to locate author identifier column in {path}. Columns: {', '.join(columns)}"
            )

        try:
            df = pd.read_parquet(path, columns=[chosen_col])
        except Exception:
            # Fall back to loading entire file if column pushdown unsupported
            df = pd.read_parquet(path)
            if chosen_col not in df.columns:
                if "first_author_id" in df.columns:
                    chosen_col = "first_author_id"
                    fallback_single_author = True
                else:
                    raise

        quarter_authors: set[str] = set()
        for raw in df[chosen_col].dropna():
            quarter_authors.update(normalize_author_ids(raw))
        total_authors = len(quarter_authors)
        new_authors = len(quarter_authors - seen_authors)
        returning_authors = total_authors - new_authors
        seen_authors.update(quarter_authors)
        cumulative_counts.append(len(seen_authors))
        entries.append(
            {
                "quarter": quarter,
                "total_authors": total_authors,
                "new_authors": new_authors,
                "returning_authors": returning_authors,
                "new_author_rate": (new_authors / total_authors) if total_authors else None,
                "cumulative_unique_authors": cumulative_counts[-1],
            }
        )
        if fallback_single_author:
            print(f"[Author Influx] {quarter}: only single-author ID available; using '{chosen_col}'.")

    payload = {
        "metric": "author_influx",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "quarters": entries,
    }
    return payload, input_files


def render_plot(payload: dict[str, object], out_path: Path) -> None:
    quarters = [row["quarter"] for row in payload["quarters"]]
    new_counts = [row["new_authors"] for row in payload["quarters"]]
    returning_counts = [row["returning_authors"] for row in payload["quarters"]]
    rates = [row["new_author_rate"] if row["new_author_rate"] is not None else 0 for row in payload["quarters"]]

    # Compute rolling averages for trendlines (4-quarter = 1 year smoothing)
    new_series = pd.Series(new_counts)
    returning_series = pd.Series(returning_counts)
    rates_series = pd.Series(rates)
    new_trend = new_series.rolling(window=4, center=True, min_periods=1).mean()
    returning_trend = returning_series.rolling(window=4, center=True, min_periods=1).mean()
    rates_trend = rates_series.rolling(window=4, center=True, min_periods=1).mean()

    fig, ax1 = plt.subplots(figsize=(12, 6))
    # Plot quarterly data with transparency
    ax1.plot(quarters, new_counts, color="#1f77b4", linewidth=1, alpha=0.3)
    ax1.plot(quarters, returning_counts, color="#ff7f0e", linewidth=1, alpha=0.3)
    # Plot trendlines prominently
    ax1.plot(quarters, new_trend, label="New authors (trend)", color="#1f77b4", linewidth=2.5)
    ax1.plot(quarters, returning_trend, label="Returning authors (trend)", color="#ff7f0e", linewidth=2.5)
    ax1.set_ylabel("Authors")
    ax1.tick_params(axis="x", rotation=75)

    ax2 = ax1.twinx()
    # Plot quarterly rate with transparency, then trendline
    ax2.plot(quarters, rates, color="#2ca02c", linewidth=1, alpha=0.3, linestyle="--")
    ax2.plot(quarters, rates_trend, label="New author rate (trend)", color="#2ca02c", linewidth=2.5, linestyle="--")
    ax2.set_ylabel("Share of first-time authors")
    ax2.set_ylim(0, 1)

    ax1.set_xlabel("Quarter")
    ax1.set_title("Quarterly Author Influx (4-quarter rolling average)")
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
    Write standardized parquet outputs and metadata for author influx metric.
    """
    metric_name = "author_influx"

    # Convert quarters data to DataFrame for global level
    if not payload["quarters"]:
        return

    df_global = pd.DataFrame([
        {
            "quarter": row["quarter"],
            "value": row["new_author_rate"],  # Primary metric: share of first-time authors
            "new_authors": row["new_authors"],
            "returning_authors": row["returning_authors"],
            "total_authors": row["total_authors"],
            "cumulative_unique_authors": row["cumulative_unique_authors"],
        }
        for row in payload["quarters"]
    ])

    # Get output paths
    paths_global = get_metric_output_paths(metric_name, args.out_dir, "global")

    # Write global parquet
    write_metric_parquet(df_global, paths_global["parquet"], "global", metric_name)

    # Create metadata with provenance tracking
    metadata = create_metric_metadata(
        metric_name=metric_name,
        description="Quarterly author influx tracking new vs. returning authors in PSC field",
        formula="new_author_rate = new_authors / total_authors; where new_authors are those not seen in prior quarters",
        units="dimensionless (rate); counts for auxiliary columns",
        parameters={
            "slices_dir": str(args.slices_dir),
            "pattern": args.pattern,
            "num_input_files": len(input_files),
        },
        input_files=input_files,  # Track all input slice files
        level="global",
        column_descriptions={
            "quarter": "Quarter identifier (YYYYQN format)",
            "value": "New author rate (share of first-time authors)",
            "new_authors": "Count of authors appearing for first time",
            "returning_authors": "Count of authors seen in previous quarters",
            "total_authors": "Total distinct authors in quarter",
            "cumulative_unique_authors": "Cumulative count of unique authors seen to date",
        },
    )

    write_metric_metadata(metadata, paths_global["metadata"])

    # Update central manifest (Task 1.2)
    manifest_path = args.out_dir / "manifest.json"
    update_manifest(manifest_path, metric_name, "global", metadata, paths_global)

    placeholder_reason = (
        "Per-front/per-lineage author assignments are not yet exported; "
        "placeholder emitted to satisfy schema requirements."
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
    payload, input_files = compute_author_influx(args)

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
