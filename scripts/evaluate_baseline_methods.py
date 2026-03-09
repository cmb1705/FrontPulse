"""
Unified evaluation harness for baseline methods.

Reads one or more prediction files (with labels and scores), computes standard metrics,
and generates comparative PR/ROC plots and lag statistics.

Config JSON schema (list of methods):
[
  {
    "name": "simple_heuristics",
    "path": "data/out/experiments/baselines/simple_heuristics/breakthrough_predictions.csv",
    "score_column": "heuristic_score",
    "pred_column": "is_inflection_pred"
  },
  {
    "name": "msd_lightgbm",
    "path": "data/out/experiments/msd_training/msd_inflection/leakage_free/breakthrough_predictions.csv",
    "score_column": "inflection_probability",
    "pred_column": "is_inflection_pred"
  }
]

Outputs:
- leaderboard.json (metrics per method)
- pr_curves.png, roc_curves.png (comparative figures)
- run_metadata.json (parameters, timestamps, input hashes)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def quarter_to_int(q: str) -> int:
    year = int(q[:4])
    quarter = int(q[-1])
    return year * 4 + (quarter - 1)


def hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass
class MethodConfig:
    name: str
    path: Path
    score_column: str
    pred_column: Optional[str] = None


@dataclass
class MethodMetrics:
    name: str
    pr_auc: float
    roc_auc: Optional[float]
    best_threshold: float
    best_f1: float
    best_precision: float
    best_recall: float
    default_precision: float
    default_recall: float
    default_f1: float
    positives: int
    negatives: int
    detection_lag_median: Optional[float]
    detection_lag_mean: Optional[float]
    detection_lag_pct_le_2: Optional[float]
    detection_lag_coverage: Optional[float]


def load_config(path: Path) -> List[MethodConfig]:
    raw = json.loads(path.read_text())
    configs: List[MethodConfig] = []
    for entry in raw:
        configs.append(
            MethodConfig(
                name=entry["name"],
                path=Path(entry["path"]),
                score_column=entry["score_column"],
                pred_column=entry.get("pred_column"),
            )
        )
    return configs


def best_threshold_from_scores(y_true: np.ndarray, scores: np.ndarray) -> Tuple[float, float, float, float]:
    unique_scores = np.unique(scores[np.isfinite(scores)])
    if len(unique_scores) > 400:
        unique_scores = np.linspace(np.nanmin(scores), np.nanmax(scores), 400)

    best = {"f1": -1.0, "threshold": 0.0, "precision": 0.0, "recall": 0.0}
    for thr in unique_scores:
        y_pred = (scores >= thr).astype(int)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        if f1 > best["f1"]:
            best = {"f1": f1, "threshold": float(thr), "precision": precision, "recall": recall}
    return best["threshold"], best["f1"], best["precision"], best["recall"]


def compute_detection_lag(df: pd.DataFrame, pred_col: str) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
    if pred_col not in df.columns:
        return None, None, None, None
    df = df.copy()
    df["quarter_int"] = df["quarter"].apply(quarter_to_int)

    first_pred = (
        df[df[pred_col] == 1]
        .groupby("lineage_id")["quarter_int"]
        .min()
        .rename("first_pred_quarter")
    )
    true_inflection = (
        df[df["is_inflection_true"] == 1]
        .groupby("lineage_id")["quarter_int"]
        .min()
        .rename("true_inflection_quarter")
    )

    merged = true_inflection.to_frame().merge(first_pred, on="lineage_id", how="left")
    merged["detection_lag"] = merged["first_pred_quarter"] - merged["true_inflection_quarter"]

    valid = merged["detection_lag"].dropna()
    if valid.empty:
        return None, None, None, None
    median = float(valid.median())
    mean = float(valid.mean())
    pct_le_2 = float((valid <= 2).mean())
    coverage = float(len(valid) / len(merged))
    return median, mean, pct_le_2, coverage


def evaluate_method(config: MethodConfig) -> Tuple[MethodMetrics, Dict[str, List[float]]]:
    df = pd.read_csv(config.path)
    scores = df[config.score_column].replace([np.inf, -np.inf], np.nan)
    mask = scores.notna()
    scores_clean = scores[mask].to_numpy()
    y_true = df.loc[mask, "is_inflection_true"].astype(int).to_numpy()

    pr_auc = float(average_precision_score(y_true, scores_clean))

    try:
        roc_auc = float(roc_auc_score(y_true, scores_clean))
    except ValueError:
        roc_auc = None

    best_thr, best_f1, best_precision, best_recall = best_threshold_from_scores(y_true, scores_clean)

    if config.pred_column and config.pred_column in df.columns:
        default_pred = df[config.pred_column].astype(int)
    else:
        default_pred = (scores >= 0).astype(int)

    default_precision = float(precision_score(df["is_inflection_true"], default_pred, zero_division=0))
    default_recall = float(recall_score(df["is_inflection_true"], default_pred, zero_division=0))
    default_f1 = float(f1_score(df["is_inflection_true"], default_pred, zero_division=0))

    det_median, det_mean, det_pct_le_2, det_coverage = compute_detection_lag(df, config.pred_column or "is_inflection_pred")

    pr_curve = precision_recall_curve(y_true, scores_clean)
    try:
        roc = roc_curve(y_true, scores_clean)
    except ValueError:
        roc = (np.array([np.nan]), np.array([np.nan]), np.array([np.nan]))

    metrics = MethodMetrics(
        name=config.name,
        pr_auc=pr_auc,
        roc_auc=roc_auc,
        best_threshold=best_thr,
        best_f1=best_f1,
        best_precision=best_precision,
        best_recall=best_recall,
        default_precision=default_precision,
        default_recall=default_recall,
        default_f1=default_f1,
        positives=int(df["is_inflection_true"].sum()),
        negatives=int(len(df) - df["is_inflection_true"].sum()),
        detection_lag_median=det_median,
        detection_lag_mean=det_mean,
        detection_lag_pct_le_2=det_pct_le_2,
        detection_lag_coverage=det_coverage,
    )

    curves = {
        "precision": pr_curve[0].tolist(),
        "recall": pr_curve[1].tolist(),
        "pr_thresholds": pr_curve[2].tolist(),
        "fpr": roc[0].tolist(),
        "tpr": roc[1].tolist(),
        "roc_thresholds": roc[2].tolist(),
    }
    return metrics, curves


def plot_curves(curves: Dict[str, Dict[str, List[float]]], output_dir: Path) -> None:
    fig, ax = plt.subplots()
    for name, data in curves.items():
        ax.plot(data["recall"], data["precision"], label=name)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves")
    ax.legend()
    (output_dir / "pr_curves.png").parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "pr_curves.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots()
    for name, data in curves.items():
        if any(math.isnan(v) for v in data["fpr"]):
            continue
        ax.plot(data["fpr"], data["tpr"], label=name)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves")
    ax.legend()
    fig.savefig(output_dir / "roc_curves.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def run(args: argparse.Namespace) -> None:
    start_ts = time.time()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    configs = load_config(Path(args.methods_config))
    leaderboard: Dict[str, Dict[str, float]] = {}
    curves_for_plot: Dict[str, Dict[str, List[float]]] = {}
    input_hashes = {}

    for cfg in configs:
        metrics, curves = evaluate_method(cfg)
        leaderboard[cfg.name] = metrics.__dict__
        curves_for_plot[cfg.name] = curves
        input_hashes[cfg.name] = {"path": str(cfg.path), "sha256": hash_file(cfg.path)}

    plot_curves(curves_for_plot, output_dir)

    (output_dir / "leaderboard.json").write_text(json.dumps(leaderboard, indent=2))

    run_metadata = {
        "start_time": start_ts,
        "end_time": time.time(),
        "duration_sec": time.time() - start_ts,
        "methods_config": args.methods_config,
        "input_hashes": input_hashes,
        "output_dir": str(output_dir),
    }
    (output_dir / "run_metadata.json").write_text(json.dumps(run_metadata, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate baseline methods and plot PR/ROC curves.")
    parser.add_argument(
        "--methods-config",
        required=True,
        help="Path to JSON list defining methods (name, path, score_column, optional pred_column).",
    )
    parser.add_argument(
        "--output-dir",
        default="data/out/experiments/baselines/figures",
        help="Directory for leaderboard and figures.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args)
