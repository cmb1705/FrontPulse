#!/usr/bin/env python3
"""
Label QA queue generator.

Combines holdout predictions, feature slices, and refreshed label diagnostics
to surface the samples that need manual review (high-probability false positives,
high-magnitude false negatives, and persistence disagreements).

Example:
    python scripts/prepare_label_qa_queue.py \
        --predictions data/out/experiments/time_forward_holdout_2020/breakthrough_predictions.csv \
        --features data/out/experiments/time_forward_holdout_2020/inputs/lineage_features_predict_2020q1_2025q3.parquet \
        --labels data/out/02_lineage_tracking/inflection_labels.csv \
        --diagnostics data/out/02_lineage_tracking/inflection_label_diagnostics.csv \
        --output-dir data/out/analysis/label_qa_queue \
        --markdown data/out/analysis/label_qa_queue.md
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

MAGNITUDE_COLUMNS = [
    "growth_vs_field",
    "acceleration_vs_field",
    "new_works_over_p75",
    "relative_new_works",
    "relative_cumulative_works",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare label QA review queue.")
    parser.add_argument("--predictions", required=True, help="MSD predictions CSV for the holdout slice.")
    parser.add_argument("--features", required=True, help="Feature parquet used for the prediction slice.")
    parser.add_argument("--labels", required=True, help="Inflection labels CSV (latest refresh).")
    parser.add_argument("--diagnostics", required=True, help="Label diagnostics CSV with residual/lag flags.")
    parser.add_argument(
        "--output-dir",
        default="data/out/analysis/label_qa_queue",
        help="Directory for intermediate CSV exports.",
    )
    parser.add_argument(
        "--markdown",
        default="data/out/analysis/label_qa_queue.md",
        help="Markdown summary path for reviewers.",
    )
    parser.add_argument("--top-n", type=int, default=50, help="Number of high-probability false positives to surface.")
    parser.add_argument(
        "--magnitude-quantile",
        type=float,
        default=0.75,
        help="Quantile cutoff for high-magnitude false negatives (default: 0.75).",
    )
    return parser.parse_args()


def compute_magnitude_score(df: pd.DataFrame) -> pd.Series:
    cols = [c for c in MAGNITUDE_COLUMNS if c in df.columns]
    if not cols:
        raise ValueError("No magnitude columns found in feature slice; check MAGNITUDE_COLUMNS.")
    return df[cols].astype(float).replace([np.inf, -np.inf], np.nan).mean(axis=1, skipna=True)


def df_to_markdown(df: pd.DataFrame, columns: Sequence[str], max_rows: int = 10) -> str:
    if df.empty:
        return "_None_\n"
    subset = df.loc[:, columns].head(max_rows)
    header = " | ".join(columns)
    divider = " | ".join(["---"] * len(columns))
    rows = [" | ".join(str(subset.iloc[i, j]) for j in range(len(columns))) for i in range(len(subset))]
    return "\n".join([header, divider, *rows]) + "\n"


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    predictions = pd.read_csv(args.predictions)
    features = pd.read_parquet(args.features)
    labels = pd.read_csv(args.labels)
    diagnostics = pd.read_csv(args.diagnostics)

    merge_cols = ["lineage_id", "quarter"]
    merged = predictions.merge(features, on=merge_cols, how="left", suffixes=("", "_feat"))
    merged["magnitude_score"] = compute_magnitude_score(merged)

    labels_lookup = labels[merge_cols + ["inflection_type", "inflection_score"]].rename(
        columns={
            "inflection_type": "label_inflection_type",
            "inflection_score": "label_inflection_score",
        }
    )
    diag_lookup = diagnostics[
        merge_cols + ["logistic_residual", "lag_since_milestone", "lag_abs", "flag_high_residual", "flag_lag_drift"]
    ]
    merged = merged.merge(labels_lookup, on=merge_cols, how="left")
    merged = merged.merge(diag_lookup, on=merge_cols, how="left")

    # High-probability false positives (top-N)
    fp_candidates = merged[merged["is_inflection_true"] == 0].copy()
    fp_candidates.sort_values("inflection_probability", ascending=False, inplace=True)
    top_fp = fp_candidates.head(args.top_n)
    top_fp_path = output_dir / "high_probability_false_positives.csv"
    top_fp.to_csv(top_fp_path, index=False)

    # High-magnitude false negatives
    fn = merged[merged["is_inflection_true"] == 1].copy()
    magnitude_cutoff = fn["magnitude_score"].quantile(args.magnitude_quantile)
    fn_high = fn[fn["magnitude_score"] >= magnitude_cutoff].copy()
    fn_path = output_dir / "high_magnitude_false_negatives.csv"
    fn_high.to_csv(fn_path, index=False)

    # Persistence disagreements
    persistence_mismatch = merged[
        merged["is_inflection_pred"] != merged["is_inflection_pred_persistent"]
    ].copy()
    persistence_path = output_dir / "persistence_disagreements.csv"
    persistence_mismatch.to_csv(persistence_path, index=False)

    # Markdown summary
    md_lines: list[str] = []
    md_lines.append("# Label QA Queue\n")
    md_lines.append("## Summary\n")
    md_lines.append(f"- Predictions file: `{args.predictions}`")
    md_lines.append(f"- Features slice: `{args.features}`")
    md_lines.append(f"- Refreshed labels: `{args.labels}`")
    md_lines.append(f"- Diagnostics: `{args.diagnostics}`")
    md_lines.append(f"- High-probability false positives exported to `{top_fp_path}`")
    md_lines.append(f"- High-magnitude false negatives exported to `{fn_path}`")
    md_lines.append(f"- Persistence disagreements exported to `{persistence_path}`\n")

    md_lines.append("## Top High-Probability False Positives (first 10)\n")
    md_lines.append(
        df_to_markdown(
            top_fp,
            [
                "lineage_id",
                "quarter",
                "inflection_probability",
                "magnitude_score",
                "growth_vs_field",
                "acceleration_vs_field",
                "new_works_over_p75",
            ],
        )
    )

    md_lines.append(f"## High-Magnitude False Negatives (>= {args.magnitude_quantile:.0%} magnitude quantile)\n")
    md_lines.append(
        df_to_markdown(
            fn_high,
            [
                "lineage_id",
                "quarter",
                "magnitude_score",
                "inflection_probability",
                "label_inflection_type",
                "logistic_residual",
                "lag_since_milestone",
            ],
        )
    )

    md_lines.append("## Persistence vs. Detection Disagreements (first 10)\n")
    md_lines.append(
        df_to_markdown(
            persistence_mismatch,
            [
                "lineage_id",
                "quarter",
                "inflection_probability",
                "is_inflection_pred",
                "is_inflection_pred_persistent",
            ],
        )
    )

    markdown_path = Path(args.markdown)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text("\n".join(md_lines))
    print(f"Wrote label QA summary to {markdown_path}")


if __name__ == "__main__":
    main()
