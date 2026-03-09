"""
Simple heuristics baseline for inflection detection on the Nov 9 evaluation dataset.

Workflow:
- Compute percentile-based thresholds on a training window (default <=2019Q4) for growth metrics.
- Evaluate a small grid of rule variants on the training window and select the variant with the best PR-AUC
  (tie-break on F1 at threshold 0, then precision).
- Apply the chosen rule to the full dataset, emit predictions and scores, and persist metadata.

Outputs (written to --output-dir):
- breakthrough_predictions.csv: lineage-quarter rows with heuristic scores and predictions.
- metrics.json: training-set metrics for the chosen rule and the variant grid summary.
- run_metadata.json: parameters, timestamps, input file hash, and selected variant details.

Notes:
- Assumes input CSV matches `data/out/experiments/msd_training/msd_inflection/leakage_free/breakthrough_predictions.csv`.
- Optimized for vectorized operations; multiprocessing/GPU not required given dataset size.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score


TRAIN_CUTOFF_DEFAULT = "2019Q4"
WINDOW_REQUIREMENT_DEFAULT = 2
WINDOW_SIZE_DEFAULT = 3
MIN_CUMULATIVE_DEFAULT = 10
MIN_GROUP_SIZE_DEFAULT = 100
GROUPING_METHOD_DEFAULT = "size"


def quarter_to_int(q: str) -> int:
    """Convert 'YYYYQX' to a sortable int (year*4 + quarter-1)."""
    year = int(q[:4])
    quarter = int(q[-1])
    return year * 4 + (quarter - 1)


@dataclass
class VariantSpec:
    name: str
    params: Dict[str, float]
    rule_type: str  # "single_ga" or "momentum_ga"


@dataclass
class VariantMetrics:
    variant: VariantSpec
    pr_auc: float
    roc_auc: float
    f1_at_zero: float
    precision_at_zero: float
    recall_at_zero: float
    positives: int
    negatives: int


def hash_file(path: Path) -> str:
    """Return SHA256 hash for provenance."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def build_variants(
    train_df: pd.DataFrame,
    ga_percentiles: Tuple[int, ...],
) -> List[VariantSpec]:
    ga_values = train_df["growth_acceleration_z"].dropna()
    ga_thresholds = {p: float(np.nanpercentile(ga_values, p)) for p in ga_percentiles}

    variants: List[VariantSpec] = []
    for ga_p, ga_thr in ga_thresholds.items():
        variants.append(
            VariantSpec(
                name=f"single_ga{ga_p}",
                params={"ga_threshold": ga_thr},
                rule_type="single_ga",
            )
        )
        variants.append(
            VariantSpec(
                name=f"momentum_ga{ga_p}",
                params={"ga_threshold": ga_thr},
                rule_type="momentum_ga",
            )
        )
    return variants


def compute_scores(df: pd.DataFrame, variant: VariantSpec, zscore_stats: Optional[Tuple[float, float]]) -> pd.Series:
    """Compute continuous heuristic score per row."""
    if variant.rule_type == "single_ga":
        ga_thr = variant.params["ga_threshold"]
        score = df["growth_acceleration_z"] - ga_thr
    elif variant.rule_type == "momentum_ga":
        ga_thr = variant.params["ga_threshold"]
        momentum = (df["growth_acceleration_z_qoq_delta"] > 0).astype(float)
        score = (df["growth_acceleration_z"] - ga_thr) + momentum
    else:
        raise ValueError(f"Unknown rule_type: {variant.rule_type}")
    return score


def evaluate_variant(
    df: pd.DataFrame,
    variant: VariantSpec,
    zscore_stats: Optional[Tuple[float, float]],
) -> VariantMetrics:
    score = compute_scores(df, variant, zscore_stats)
    score_filled = score.fillna(-np.inf)
    y_true = df["is_inflection_true"].astype(int)
    y_pred = (score_filled >= 0).astype(int)

    positives = int(y_true.sum())
    negatives = int((1 - y_true).sum())

    pr_auc = float(average_precision_score(y_true, score_filled))
    roc_auc = float(roc_auc_score(y_true, score_filled))
    f1 = float(f1_score(y_true, y_pred))
    precision = float(precision_score(y_true, y_pred, zero_division=0))
    recall = float(recall_score(y_true, y_pred))

    return VariantMetrics(
        variant=variant,
        pr_auc=pr_auc,
        roc_auc=roc_auc,
        f1_at_zero=f1,
        precision_at_zero=precision,
        recall_at_zero=recall,
        positives=positives,
        negatives=negatives,
    )


def select_best_variant(metrics: List[VariantMetrics]) -> VariantMetrics:
    """Choose best by PR-AUC, then F1, then precision."""
    metrics_sorted = sorted(
        metrics,
        key=lambda m: (m.pr_auc, m.f1_at_zero, m.precision_at_zero),
        reverse=True,
    )
    return metrics_sorted[0]


def attach_detection_lag(pred_df: pd.DataFrame) -> pd.DataFrame:
    """Compute detection lag per lineage relative to the first predicted positive quarter."""
    pred_df = pred_df.copy()
    pred_df["quarter_int"] = pred_df["quarter"].apply(quarter_to_int)

    first_pred = (
        pred_df[pred_df["is_inflection_pred"] == 1]
        .groupby("lineage_id")["quarter_int"]
        .min()
        .rename("first_pred_quarter")
    )
    true_infl = (
        pred_df[pred_df["is_inflection_true"] == 1]
        .groupby("lineage_id")["quarter_int"]
        .min()
        .rename("true_inflection_quarter")
    )
    merged = pred_df.merge(first_pred, on="lineage_id", how="left").merge(true_infl, on="lineage_id", how="left")

    merged["detection_lag_quarters"] = merged["first_pred_quarter"] - merged["true_inflection_quarter"]
    merged.loc[merged["first_pred_quarter"].isna(), "detection_lag_quarters"] = np.nan

    merged.drop(columns=["quarter_int", "first_pred_quarter", "true_inflection_quarter"], inplace=True)
    return merged


def run(args: argparse.Namespace) -> None:
    start_ts = time.time()
    input_path = Path(args.input_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)
    df["quarter_int"] = df["quarter"].apply(quarter_to_int)

    train_cutoff_int = quarter_to_int(args.train_cutoff)
    train_df = df[df["quarter_int"] <= train_cutoff_int].copy()

    # Group assignments for pooled standardization
    lineage_max_cum = (
        train_df.groupby("lineage_id")["cumulative_works"]
        .max()
        .reset_index()
        .rename(columns={"cumulative_works": "max_cumulative_works"})
    )

    if args.grouping_method == "size":
        p33 = float(np.nanpercentile(lineage_max_cum["max_cumulative_works"], 33))
        p67 = float(np.nanpercentile(lineage_max_cum["max_cumulative_works"], 67))

        def size_bucket(x: float) -> str:
            if pd.isna(x):
                return "unknown"
            if x < p33:
                return "small"
            if x < p67:
                return "medium"
            return "large"

        lineage_max_cum["group_label"] = lineage_max_cum["max_cumulative_works"].apply(size_bucket)
        group_thresholds = {"p33": p33, "p67": p67}
    else:
        # Fallback: treat all as one group
        lineage_max_cum["group_label"] = "all"
        group_thresholds = {}

    df = df.merge(lineage_max_cum[["lineage_id", "group_label"]], on="lineage_id", how="left")
    train_df = df[df["quarter_int"] <= train_cutoff_int].copy()

    # Pooled stats per group
    group_stats = (
        train_df.groupby("group_label")["growth_acceleration"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(columns={"mean": "ga_mean", "std": "ga_std", "count": "n_quarters"})
    )
    global_mean = float(train_df["growth_acceleration"].mean())
    global_std = float(train_df["growth_acceleration"].std(ddof=0))

    def resolve_stats(label: str) -> Tuple[float, float]:
        row = group_stats[group_stats["group_label"] == label]
        if row.empty or row.iloc[0]["n_quarters"] < args.min_group_size:
            mean, std = global_mean, global_std
        else:
            mean = float(row.iloc[0]["ga_mean"])
            std = float(row.iloc[0]["ga_std"])
        if std == 0 or np.isnan(std):
            std = np.nan
        return mean, std

    means = []
    stds = []
    for lbl in df["group_label"]:
        m, s = resolve_stats(lbl)
        means.append(m)
        stds.append(s)
    df["ga_mean_group"] = means
    df["ga_std_group"] = stds
    df["growth_acceleration_z"] = (df["growth_acceleration"] - df["ga_mean_group"]) / df["ga_std_group"]
    df["growth_acceleration_z"] = df["growth_acceleration_z"].fillna(0.0)

    # Momentum term on standardized series
    df["growth_acceleration_z_qoq_delta"] = (
        df.sort_values(["lineage_id", "quarter_int"])
        .groupby("lineage_id")["growth_acceleration_z"]
        .diff()
    )
    train_df = df[df["quarter_int"] <= train_cutoff_int].copy()

    variants = build_variants(
        train_df=train_df,
        ga_percentiles=tuple(args.ga_percentiles),
    )

    variant_metrics: List[VariantMetrics] = [
        evaluate_variant(train_df, variant, None) for variant in variants
    ]
    best = select_best_variant(variant_metrics)

    df["heuristic_score"] = compute_scores(df, best.variant, None)
    def apply_soft_window(group: pd.DataFrame) -> pd.Series:
        arr = group["heuristic_score"].fillna(-np.inf).to_numpy()
        hits = arr >= 0
        persisted = np.zeros_like(hits, dtype=int)
        win = args.window_size
        req = args.window_requirement
        for i in range(len(hits)):
            if i + 1 < win:
                persisted[i] = 0
                continue
            window_hits = hits[i - win + 1 : i + 1]
            if window_hits.sum() >= req:
                persisted[i] = 1
        mask = pd.Series(persisted, index=group.index)
        mask &= group["cumulative_works"].fillna(0) >= args.min_cumulative_works
        return mask.astype(int)

    df["is_inflection_pred"] = (
        df.sort_values(["lineage_id", "quarter_int"])
        .groupby("lineage_id", group_keys=False)
        .apply(apply_soft_window)
    )
    df = attach_detection_lag(df)

    output_cols = [
        "lineage_id",
        "quarter",
        "is_inflection_true",
        "is_inflection_pred",
        "heuristic_score",
        "detection_lag_quarters",
        "growth_acceleration",
        "new_works_roll_mean_4q",
    ]
    predictions_path = output_dir / "breakthrough_predictions.csv"
    df[output_cols].to_csv(predictions_path, index=False)

    metrics_output = {
        "best_variant": {
            "name": best.variant.name,
            "rule_type": best.variant.rule_type,
            "params": best.variant.params,
            "pr_auc": best.pr_auc,
            "roc_auc": best.roc_auc,
            "f1_at_zero": best.f1_at_zero,
            "precision_at_zero": best.precision_at_zero,
            "recall_at_zero": best.recall_at_zero,
        },
        "all_variants": [
            {
                "name": m.variant.name,
                "rule_type": m.variant.rule_type,
                "params": m.variant.params,
                "pr_auc": m.pr_auc,
                "roc_auc": m.roc_auc,
                "f1_at_zero": m.f1_at_zero,
                "precision_at_zero": m.precision_at_zero,
                "recall_at_zero": m.recall_at_zero,
                "positives": m.positives,
                "negatives": m.negatives,
            }
            for m in variant_metrics
        ],
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics_output, indent=2))

    end_ts = time.time()
    run_metadata = {
        "start_time": start_ts,
        "end_time": end_ts,
        "duration_sec": end_ts - start_ts,
        "input_csv": str(input_path),
        "input_sha256": hash_file(input_path),
        "output_predictions": str(predictions_path),
        "train_cutoff": args.train_cutoff,
        "ga_percentiles": args.ga_percentiles,
        "min_cumulative_works": args.min_cumulative_works,
        "window_requirement": args.window_requirement,
        "window_size": args.window_size,
        "grouping_method": args.grouping_method,
        "group_thresholds": group_thresholds,
        "group_stats": group_stats.to_dict(orient="records"),
        "rows_total": int(len(df)),
        "rows_training": int(len(train_df)),
        "selected_variant": best.variant.name,
        "std_stats_lineages": int(lineage_max_cum["lineage_id"].nunique()),
    }
    (output_dir / "run_metadata.json").write_text(json.dumps(run_metadata, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simple heuristics baseline for inflection detection.")
    parser.add_argument(
        "--input-csv",
        default="data/out/experiments/msd_training/msd_inflection/leakage_free/breakthrough_predictions.csv",
        help="Input predictions CSV with labels and growth metrics.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/out/experiments/baselines/simple_heuristics",
        help="Directory to write predictions and metrics.",
    )
    parser.add_argument(
        "--train-cutoff",
        default=TRAIN_CUTOFF_DEFAULT,
        help="Inclusive training cutoff quarter for threshold estimation (format YYYYQ#).",
    )
    parser.add_argument(
        "--ga-percentiles",
        type=int,
        nargs="+",
        default=[60, 65, 70, 75, 80, 85, 90, 95],
        help="Percentiles for growth_acceleration thresholds.",
    )
    parser.add_argument(
        "--window-requirement",
        type=int,
        default=WINDOW_REQUIREMENT_DEFAULT,
        help="Minimum positives within window to emit detection (soft persistence).",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=WINDOW_SIZE_DEFAULT,
        help="Window size (quarters) for soft persistence.",
    )
    parser.add_argument(
        "--min-cumulative-works",
        type=int,
        default=MIN_CUMULATIVE_DEFAULT,
        help="Minimum cumulative works required to emit a detection (context-aware guard).",
    )
    parser.add_argument(
        "--grouping-method",
        choices=["size"],
        default=GROUPING_METHOD_DEFAULT,
        help="How to pool standardization stats; current support: size-based buckets.",
    )
    parser.add_argument(
        "--min-group-size",
        type=int,
        default=MIN_GROUP_SIZE_DEFAULT,
        help="Minimum quarters per group to trust pooled stats; fallback to global otherwise.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args)
