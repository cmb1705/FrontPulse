#!/usr/bin/env python3
"""
Analyze lag between detected inflections and curated milestones.

Creates a CSV with inflection↔milestone matches plus a JSON summary
containing coverage statistics and lag distributions.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from utils.quarter_utils import quarter_to_int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Milestone/inflection lag analysis.")
    parser.add_argument(
        "--inflections",
        default="data/out/02_lineage_tracking/inflection_labels.csv",
        help="Inflection labels CSV (default: %(default)s)",
    )
    parser.add_argument(
        "--milestones",
        required=True,
        help="Milestone-lineage mapping CSV",
    )
    parser.add_argument(
        "--out",
        default="data/out/analysis/milestone_inflection_analysis.csv",
        help="Output CSV path for per-inflection lag table (default: %(default)s)",
    )
    parser.add_argument(
        "--summary",
        default="data/out/analysis/milestone_inflection_summary.json",
        help="Output JSON summary path (default: %(default)s)",
    )
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def bucket_lag(lag: int | None) -> str:
    if lag is None:
        return "no_milestone"
    abs_lag = abs(lag)
    if abs_lag < 4:
        return "<4q"
    if abs_lag < 8:
        return "4-8q"
    if abs_lag < 12:
        return "8-12q"
    return ">=12q"


def build_milestone_lookup(df: pd.DataFrame) -> dict[int, list[dict[str, object]]]:
    lookup: dict[int, list[dict[str, object]]] = {}
    for _, row in df.iterrows():
        try:
            lineage_id = int(row["lineage_id"])
            event_quarter = str(row["event_quarter"])
        except Exception:
            continue
        entry = {
            "event_id": row.get("event_id"),
            "event_quarter": event_quarter,
            "event_quarter_int": quarter_to_int(event_quarter),
            "mapped_fronts": row.get("mapped_fronts"),
            "category": row.get("category"),
        }
        lookup.setdefault(lineage_id, []).append(entry)
    return lookup


def find_nearest(
    lineage_id: int,
    inflection_q: int,
    lookup: dict[int, list[dict[str, object]]],
) -> tuple[dict[str, object] | None, int | None]:
    candidates = lookup.get(lineage_id)
    if not candidates:
        return None, None
    best = min(candidates, key=lambda entry: abs(entry["event_quarter_int"] - inflection_q))
    lag = inflection_q - int(best["event_quarter_int"])
    return best, lag


def main() -> None:
    args = parse_args()
    inflection_path = Path(args.inflections)
    milestone_path = Path(args.milestones)
    out_path = Path(args.out)
    summary_path = Path(args.summary)

    inflections = pd.read_csv(inflection_path)
    required_cols = {"lineage_id", "quarter"}
    missing = required_cols - set(inflections.columns)
    if missing:
        raise ValueError(f"Inflection file missing columns: {missing}")

    milestones = pd.read_csv(milestone_path)
    if milestones.empty:
        raise ValueError(f"No milestone entries found in {milestone_path}")
    milestone_lookup = build_milestone_lookup(milestones)

    inflections = inflections.copy()
    inflections["lineage_id"] = inflections["lineage_id"].astype(int)
    inflections["inflection_quarter_int"] = inflections["quarter"].astype(str).apply(quarter_to_int)

    records: list[dict[str, object]] = []
    for _, row in inflections.iterrows():
        lineage_id = int(row["lineage_id"])
        inflection_q = int(row["inflection_quarter_int"])
        milestone_entry, lag = find_nearest(lineage_id, inflection_q, milestone_lookup)

        record = {
            "lineage_id": lineage_id,
            "inflection_quarter": row["quarter"],
            "inflection_type": row.get("inflection_type"),
            "inflection_score": row.get("inflection_score"),
            "nearest_milestone_id": milestone_entry["event_id"] if milestone_entry else None,
            "milestone_quarter": milestone_entry["event_quarter"] if milestone_entry else None,
            "milestone_category": milestone_entry["category"] if milestone_entry else None,
            "mapped_fronts": milestone_entry["mapped_fronts"] if milestone_entry else None,
            "lag_since_milestone": lag,
            "lag_bucket": bucket_lag(lag),
        }
        records.append(record)

    analysis_df = pd.DataFrame(records)
    ensure_dir(out_path)
    analysis_df.to_csv(out_path, index=False)

    coverage = analysis_df["nearest_milestone_id"].notna().mean()
    lag_counts = analysis_df["lag_bucket"].value_counts(dropna=False).to_dict()
    front_stats = (
        analysis_df.dropna(subset=["mapped_fronts"])
        .groupby("mapped_fronts")["lag_since_milestone"]
        .agg(["count", "median"])
        .reset_index()
        .rename(columns={"count": "matches", "median": "median_lag"})
        .to_dict(orient="records")
    )

    category_stats: list[dict[str, object]] = []
    category_df = analysis_df.dropna(subset=["milestone_category"])
    if not category_df.empty:
        grouped = category_df.groupby(["milestone_category", "lag_bucket"]).size().unstack(fill_value=0)
        for category, row in grouped.iterrows():
            total = int(row.sum())
            probs = {bucket: float(count) / total for bucket, count in row.items()}
            category_stats.append(
                {
                    "category": category,
                    "total_matches": total,
                    "lag_bucket_counts": {bucket: int(count) for bucket, count in row.items()},
                    "lag_bucket_probabilities": probs,
                }
            )

    summary = {
        "inflection_count": len(analysis_df),
        "milestone_matches": int(analysis_df["nearest_milestone_id"].notna().sum()),
        "milestone_coverage": coverage,
        "lag_bucket_counts": lag_counts,
        "front_stats": front_stats,
        "category_stats": category_stats,
        "inputs": {
            "inflections": str(inflection_path),
            "milestones": str(milestone_path),
        },
        "outputs": {
            "analysis_csv": str(out_path),
            "summary_json": str(summary_path),
        },
    }
    ensure_dir(summary_path)
    summary_path.write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
