#!/usr/bin/env python3
"""
Find the detection threshold that maximizes recall subject to a precision floor.

Usage:
    python scripts/find_best_threshold.py \
        --predictions data/out/experiments/msd_lag4plus_train2005_2020/breakthrough_predictions.csv \
        --min-precision 0.20 \
        --start 0.005 --end 0.20 --num 200

The script prints the best threshold and writes a CSV with the sweep metrics.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Find best threshold given a precision floor.")
    parser.add_argument("--predictions", type=Path, required=True,
                        help="Path to breakthrough_predictions.csv")
    parser.add_argument("--min-precision", type=float, default=0.20,
                        help="Minimum precision required (default: 0.20)")
    parser.add_argument("--start", type=float, default=0.005,
                        help="Sweep start threshold (default: 0.005)")
    parser.add_argument("--end", type=float, default=0.25,
                        help="Sweep end threshold (default: 0.25)")
    parser.add_argument("--num", type=int, default=200,
                        help="Number of thresholds in sweep (default: 200)")
    parser.add_argument("--output-csv", type=Path, default=None,
                        help="Optional path to save sweep metrics CSV")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.predictions)
    if "is_inflection_true" not in df.columns or "inflection_probability" not in df.columns:
        raise ValueError("Predictions file must contain 'is_inflection_true' and 'inflection_probability' columns.")

    y_true = df["is_inflection_true"].fillna(0).astype(int).values
    probs = df["inflection_probability"].fillna(0).values

    thresholds = np.linspace(args.start, args.end, args.num)
    rows = []
    best_row: Optional[dict] = None

    for thresh in thresholds:
        preds = (probs >= thresh).astype(int)
        precision = precision_score(y_true, preds, zero_division=0)
        recall = recall_score(y_true, preds, zero_division=0)
        f1 = f1_score(y_true, preds, zero_division=0)
        row = {
            "threshold": float(thresh),
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "detections": int(preds.sum()),
        }
        rows.append(row)
        if precision >= args.min_precision:
            if best_row is None or recall > best_row["recall"]:
                best_row = row

    sweep_df = pd.DataFrame(rows)
    if args.output_csv:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        sweep_df.to_csv(args.output_csv, index=False)

    if best_row:
        print(f"Best threshold meeting precision >= {args.min_precision:.2f}:")
        print(f"  threshold={best_row['threshold']:.4f}  precision={best_row['precision']:.3f} "
              f"recall={best_row['recall']:.3f}  f1={best_row['f1']:.3f}  detections={best_row['detections']}")
    else:
        print(f"No threshold achieved precision >= {args.min_precision:.2f}.")
        max_recall_row = sweep_df.loc[sweep_df['recall'].idxmax()]
        print(f"Max recall observed at threshold={max_recall_row['threshold']:.4f} "
              f"(precision={max_recall_row['precision']:.3f}, recall={max_recall_row['recall']:.3f}).")


if __name__ == "__main__":
    main()
