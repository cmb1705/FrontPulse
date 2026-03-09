#!/usr/bin/env python3
"""
Sweep decision thresholds for MSD predictions.

Loads `breakthrough_predictions.csv` (output of scripts/multi_signal_detector.py)
and reports precision/recall/F1 across a range of probability thresholds.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Dict

import numpy as np
import pandas as pd

from utils.quarter_utils import quarter_to_int


def summarize_detection_lag(df: pd.DataFrame, preds: pd.Series) -> Dict[str, float]:
    pred_df = df.copy()
    pred_df["is_milestone_pred"] = preds
    positives = pred_df[pred_df["is_milestone_true"] == 1]
    if positives.empty:
        return {"lag_coverage": 0.0}

    def sorted_quarters(series: pd.Series) -> List[str]:
        return sorted(series.tolist(), key=quarter_to_int)

    actual_map = positives.groupby("lineage_id")["quarter"].apply(sorted_quarters).to_dict()
    predicted_map = (
        pred_df[pred_df["is_milestone_pred"] == 1]
        .groupby("lineage_id")["quarter"]
        .apply(sorted_quarters)
        .to_dict()
    )

    lags = []
    for lineage_id, actual_quarters in actual_map.items():
        preds_quarters = predicted_map.get(lineage_id)
        if not preds_quarters:
            continue
        lag = quarter_to_int(preds_quarters[0]) - quarter_to_int(actual_quarters[0])
        lags.append(lag)

    coverage = len(lags) / len(actual_map) if actual_map else 0.0
    if not lags:
        return {"lag_coverage": coverage}

    arr = np.array(lags)
    return {
        "lag_coverage": coverage,
        "lag_median": float(np.median(arr)),
        "lag_mean": float(np.mean(arr)),
        "lag_share_le_0": float((arr <= 0).mean()),
        "lag_share_le_2": float((arr <= 2).mean()),
    }


def compute_metrics(df: pd.DataFrame, threshold: float) -> Dict[str, float]:
    preds = (df["milestone_probability"] >= threshold).astype(int)
    labels = df["is_milestone_true"].astype(int)

    tp = int(((preds == 1) & (labels == 1)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    lag_stats = summarize_detection_lag(df, preds)

    metrics = {
        "threshold": threshold,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "fpr": fpr,
    }
    metrics.update(lag_stats)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep MSD decision thresholds.")
    parser.add_argument(
        "--predictions",
        default="data/out/experiments/msd_multisignal/breakthrough_predictions.csv",
        help="Path to MSD predictions CSV (default: data/out/experiments/msd_multisignal/breakthrough_predictions.csv)",
    )
    parser.add_argument("--min-threshold", type=float, default=0.1)
    parser.add_argument("--max-threshold", type=float, default=0.9)
    parser.add_argument("--step", type=float, default=0.05)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to write sweep results as CSV",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.predictions)
    required_cols = {"milestone_probability", "is_milestone_true"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Predictions file is missing required columns: {missing}")

    thresholds = []
    t = args.min_threshold
    while t <= args.max_threshold + 1e-9:
        thresholds.append(round(t, 4))
        t += args.step

    results: List[Dict[str, float]] = [compute_metrics(df, thr) for thr in thresholds]
    results_df = pd.DataFrame(results)

    print("=" * 70)
    print("MSD Threshold Sweep")
    print("=" * 70)
    print(results_df.to_string(index=False, float_format=lambda x: f"{x:0.3f}"))

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(args.output, index=False)
        print(f"\nResults written to {args.output}")


if __name__ == "__main__":
    main()
