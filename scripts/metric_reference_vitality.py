from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
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
    parser = argparse.ArgumentParser(description="Evaluate quarterly reference vitality.")
    parser.add_argument("--slices-dir", default=None, type=Path)
    parser.add_argument("--ingest-path", default=None, type=Path)
    parser.add_argument("--out-dir", default=None, type=Path)
    parser.add_argument("--pattern", default="by_quarter__*.parquet")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--young-years", type=float, default=3.0, help="Threshold for recent references.")
    parser.add_argument("--old-years", type=float, default=10.0, help="Threshold for aging references.")
    parser.add_argument("--top-k", type=int, default=5, help="Top citing works by vitality per quarter.")
    parser.add_argument("--json-name", default="reference_vitality.json")
    parser.add_argument("--figure-name", default="reference_vitality.png")
    add_domain_args(parser)
    return parser.parse_args()


def normalize_date(raw_date: object, raw_year: object) -> pd.Timestamp | None:
    ts = pd.to_datetime(raw_date, errors="coerce")
    if pd.isna(ts) and raw_year is not None and not pd.isna(raw_year):
        try:
            year = int(raw_year)
            ts = pd.Timestamp(year=year, month=12, day=31)
        except Exception:
            ts = pd.NaT
    if pd.isna(ts):
        return None
    if getattr(ts, "tzinfo", None) is not None:
        ts = ts.tz_localize(None)
    return ts


def references_iter(raw_value: object) -> Iterable[str]:
    if raw_value is None or (isinstance(raw_value, float) and np.isnan(raw_value)):
        return []
    if isinstance(raw_value, str):
        return [raw_value.strip()] if raw_value.strip() else []
    if isinstance(raw_value, (list, tuple, set)):
        return [str(item).strip() for item in raw_value if str(item).strip()]
    if hasattr(raw_value, "tolist"):
        return [str(item).strip() for item in raw_value.tolist() if str(item).strip()]
    return []


def load_reference_lookup(path: Path) -> tuple[dict[str, pd.Timestamp], dict[str, int]]:
    df = pd.read_parquet(path, columns=["work_id", "publication_year", "publication_date"])
    df["publication_date"] = pd.to_datetime(df["publication_date"], errors="coerce")
    df["publication_date"] = df["publication_date"].apply(
        lambda ts: ts.tz_localize(None) if isinstance(ts, pd.Timestamp) and ts.tzinfo is not None else ts
    )
    date_map: dict[str, pd.Timestamp] = df.set_index("work_id")["publication_date"].to_dict()
    year_map: dict[str, int] = (
        df.set_index("work_id")["publication_year"]
        .dropna()
        .astype(int)
        .to_dict()
    )
    return date_map, year_map


def resolve_reference_date(
    ref_id: str,
    date_map: dict[str, pd.Timestamp],
    year_map: dict[str, int],
) -> pd.Timestamp | None:
    ts = date_map.get(ref_id)
    if ts is not None and not pd.isna(ts):
        return ts
    year = year_map.get(ref_id)
    if year is None:
        return None
    try:
        return pd.Timestamp(year=year, month=12, day=31)
    except Exception:
        return None


def compute_reference_vitality(args: argparse.Namespace) -> tuple[dict[str, object], list[Path]]:
    ref_date_map, ref_year_map = load_reference_lookup(args.ingest_path)
    quarters: list[dict[str, object]] = []
    input_files: list[Path] = []  # Track input files for provenance

    for idx, (quarter, path) in enumerate(iter_quarter_slices(args.slices_dir, args.pattern)):
        if args.limit is not None and idx >= args.limit:
            break
        input_files.append(path)  # Record input file
        df = pd.read_parquet(
            path,
            columns=[
                "work_id",
                "title",
                "publication_year",
                "publication_date",
                "referenced_works",
            ],
        )
        vitality_scores: list[float] = []
        ages_years: list[float] = []
        per_work: list[dict[str, object]] = []
        works_processed = 0
        works_missing_refs = 0

        for _, row in df.iterrows():
            cited_date = normalize_date(row["publication_date"], row["publication_year"])
            references = list(references_iter(row["referenced_works"]))
            if not references:
                works_missing_refs += 1
                continue
            if cited_date is None:
                works_missing_refs += 1
                continue

            local_scores: list[float] = []
            local_ages: list[float] = []
            for ref in references:
                ref_date = resolve_reference_date(ref, ref_date_map, ref_year_map)
                if ref_date is None:
                    continue
                delta_days = (cited_date - ref_date).days
                if delta_days < 0:
                    continue
                age_years = delta_days / 365.25
                score = 1.0 / (age_years + 1.0)
                local_scores.append(score)
                local_ages.append(age_years)

            if not local_scores:
                works_missing_refs += 1
                continue

            works_processed += 1
            vitality_scores.extend(local_scores)
            ages_years.extend(local_ages)
            per_work.append(
                {
                    "work_id": str(row["work_id"]),
                    "title": row["title"],
                    "mean_vitality": float(np.mean(local_scores)),
                    "references_considered": int(len(local_scores)),
                    "publication_year": int(row["publication_year"]) if not pd.isna(row["publication_year"]) else None,
                }
            )

        if not vitality_scores:
            quarters.append(
                {
                    "quarter": quarter,
                    "works_with_refs": 0,
                    "references_considered": 0,
                    "vitality": None,
                    "mean_reference_age": None,
                    "median_reference_age": None,
                    "young_share": None,
                    "old_share": None,
                    "weighted_recent_share": None,
                    "works_missing_refs": int(works_missing_refs),
                    "top_citing_works": [],
                }
            )
            continue

        scores_arr = np.array(vitality_scores, dtype=float)
        ages_arr = np.array(ages_years, dtype=float)
        vitality_value = float(scores_arr.mean())
        mean_age = float(ages_arr.mean())
        median_age = float(np.median(ages_arr))
        young_share = float(np.mean(ages_arr <= args.young_years))
        old_share = float(np.mean(ages_arr >= args.old_years))
        weighted_recent = float(scores_arr[ages_arr <= args.young_years].sum() / scores_arr.sum())
        top_k = max(1, int(args.top_k))
        top_citing = (
            sorted(per_work, key=lambda x: (x["mean_vitality"], x["references_considered"]), reverse=True)[:top_k]
            if per_work
            else []
        )
        quarters.append(
            {
                "quarter": quarter,
                "works_with_refs": int(works_processed),
                "references_considered": int(len(vitality_scores)),
                "vitality": vitality_value,
                "mean_reference_age": mean_age,
                "median_reference_age": median_age,
                "young_share": young_share,
                "old_share": old_share,
                "weighted_recent_share": weighted_recent,
                "works_missing_refs": int(works_missing_refs),
                "top_citing_works": top_citing,
            }
        )

    payload = {
        "metric": "reference_vitality",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "parameters": {
            "young_years": args.young_years,
            "old_years": args.old_years,
            "top_k": args.top_k,
        },
        "quarters": quarters,
    }
    return payload, input_files


def render_plot(payload: dict[str, object], out_path: Path) -> None:
    quarters = [row["quarter"] for row in payload["quarters"]]
    vitality = [row["vitality"] or 0 for row in payload["quarters"]]
    median_age = [row["median_reference_age"] or 0 for row in payload["quarters"]]
    young_share = [row["young_share"] or 0 for row in payload["quarters"]]
    old_share = [row["old_share"] or 0 for row in payload["quarters"]]

    # Compute rolling averages for trendlines (4-quarter = 1 year smoothing)
    vitality_series = pd.Series(vitality)
    median_age_series = pd.Series(median_age)
    young_series = pd.Series(young_share)
    old_series = pd.Series(old_share)
    vitality_trend = vitality_series.rolling(window=4, center=True, min_periods=1).mean()
    median_age_trend = median_age_series.rolling(window=4, center=True, min_periods=1).mean()
    young_trend = young_series.rolling(window=4, center=True, min_periods=1).mean()
    old_trend = old_series.rolling(window=4, center=True, min_periods=1).mean()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    # Plot quarterly data with transparency
    ax1.plot(quarters, vitality, color="#1f77b4", linewidth=1, alpha=0.3)
    # Plot trendline prominently
    ax1.plot(quarters, vitality_trend, label="Reference vitality (trend)", color="#1f77b4", linewidth=2.5)
    ax1_2 = ax1.twinx()
    ax1_2.plot(quarters, median_age, color="#ff7f0e", linewidth=1, alpha=0.3, linestyle="--")
    ax1_2.plot(quarters, median_age_trend, label="Median reference age (trend)", color="#ff7f0e", linewidth=2.5, linestyle="--")
    ax1.set_ylabel("Vitality (mean 1/(age+1))")
    ax1_2.set_ylabel("Median age (years)")
    ax1.set_title("Reference Vitality and Age Profile (4-quarter rolling average)")
    lines1 = ax1.get_lines() + ax1_2.get_lines()
    labels1 = [line.get_label() for line in lines1]
    ax1.legend(lines1, labels1, loc="upper right")

    # Plot quarterly data with transparency
    ax2.plot(quarters, young_share, color="#2ca02c", linewidth=1, alpha=0.3)
    ax2.plot(quarters, old_share, color="#d62728", linewidth=1, alpha=0.3, linestyle="-.")
    # Plot trendlines prominently
    ax2.plot(quarters, young_trend, label="Share ≤ young threshold (trend)", color="#2ca02c", linewidth=2.5)
    ax2.plot(quarters, old_trend, label="Share ≥ old threshold (trend)", color="#d62728", linewidth=2.5, linestyle="-.")
    ax2.set_ylabel("Fraction of references")
    ax2.set_ylim(0, 1)
    ax2.set_title("Reference Age Distribution (4-quarter rolling average)")
    ax2.legend(loc="upper right")

    ax2.tick_params(axis="x", rotation=75)
    ax2.set_xlabel("Quarter")
    step = max(1, len(quarters) // 16)
    ax2.set_xticks(range(0, len(quarters), step))
    ax2.set_xticklabels([quarters[i] for i in range(0, len(quarters), step)])

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def clean_for_json(obj):
    """Recursively clean pandas NA types for JSON serialization."""
    if isinstance(obj, dict):
        return {k: clean_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_for_json(item) for item in obj]
    elif pd.isna(obj):
        return None
    else:
        return obj


def write_standardized_outputs(
    payload: dict[str, object],
    input_files: list[Path],
    args: argparse.Namespace,
) -> None:
    """
    Write standardized parquet outputs and metadata for reference vitality metric.
    """
    metric_name = "reference_vitality"

    # Convert quarters data to DataFrame for global level
    if not payload["quarters"]:
        return

    df_global = pd.DataFrame([
        {
            "quarter": row["quarter"],
            "value": row["vitality"],  # Primary metric: mean reference vitality (1/(age+1))
            "mean_reference_age": row["mean_reference_age"],
            "median_reference_age": row["median_reference_age"],
            "young_share": row["young_share"],
            "old_share": row["old_share"],
            "weighted_recent_share": row["weighted_recent_share"],
            "works_with_refs": row["works_with_refs"],
            "references_considered": row["references_considered"],
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
        description="Quarterly reference vitality tracking how recent cited works are",
        formula="vitality = mean(1/(age_years+1)) across all references; age computed from publication dates",
        units="dimensionless vitality score (0-1); years for age metrics; dimensionless (rate) for shares",
        parameters={
            "slices_dir": str(args.slices_dir),
            "pattern": args.pattern,
            "young_years": args.young_years,
            "old_years": args.old_years,
            "ingest_path": str(args.ingest_path),
            "num_input_files": len(input_files),
        },
        input_files=input_files + ([args.ingest_path] if args.ingest_path.exists() else []),  # Track slices + ingest
        level="global",
        column_descriptions={
            "quarter": "Quarter identifier (YYYYQN format)",
            "value": "Mean reference vitality score (1/(age+1) averaged across all references)",
            "mean_reference_age": "Mean age of cited references in years",
            "median_reference_age": "Median age of cited references in years",
            "young_share": f"Share of references <= {args.young_years} years old",
            "old_share": f"Share of references >= {args.old_years} years old",
            "weighted_recent_share": f"Vitality-weighted share of recent references (<= {args.young_years} years)",
            "works_with_refs": "Count of works with analyzable references",
            "references_considered": "Total count of references analyzed",
        },
    )

    write_metric_metadata(metadata, paths_global["metadata"])

    # Update central manifest (Task 1.2)
    manifest_path = args.out_dir / "manifest.json"
    update_manifest(manifest_path, metric_name, "global", metadata, paths_global)

    placeholder_reason = (
        "Per-front/per-lineage reference vitality metrics require membership exports; "
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
        "ingest_path": ("ingest", "ingest.parquet",
                         "data/current_ingest/ingest.parquet"),
        "out_dir": ("out", "metrics", "data/out/metrics"),
    })
    args.slices_dir = Path(args.slices_dir)
    args.ingest_path = Path(args.ingest_path)
    args.out_dir = Path(args.out_dir)
    ensure_dir(args.out_dir)
    payload, input_files = compute_reference_vitality(args)
    # Clean NA types before JSON serialization
    payload = clean_for_json(payload)

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
