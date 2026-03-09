"""
Citation burst detection baseline (Kleinberg 2002 approximation).

Approach:
- Use quarterly citation_velocity_roll_1q as the proxy event count per lineage-quarter.
- Run Kleinberg's burst-detection (Poisson with multi-state intensity) via dynamic programming.
- Tune (s, gamma, k) on a training window (<=2019Q4) to maximize PR-AUC (tie-break on F1).
- Emit per-quarter burst levels, binary predictions (level > 0), and detection lags.

Outputs:
- breakthrough_predictions.csv (lineage_id, quarter, is_inflection_true, burst_level, is_inflection_pred, detection_lag_quarters)
- metrics.json (best params, training metrics, grid summary)
- run_metadata.json (parameters, timestamps, hashes)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score


TRAIN_CUTOFF_DEFAULT = "2019Q4"
MIN_NONZERO_DEFAULT = 8
MIN_BURST_LEN_DEFAULT = 3
MAX_BURST_LEN_DEFAULT = 8
PERSISTENCE_DEFAULT = 2


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


def kleinberg_burst(
    counts: np.ndarray,
    s: float,
    gamma: float,
    n_states: int,
) -> np.ndarray:
    """
    Minimal Kleinberg burst detector for discrete counts.

    Emission cost: lambda - x*log(lambda) (ignoring factorial term).
    Transition cost: gamma * |i - j|.
    """
    n = len(counts)
    if n == 0:
        return np.array([], dtype=int)
    total = counts.sum()
    base_rate = max(total / n, 1e-12)
    lambdas = np.array([base_rate * (s ** i) for i in range(n_states)])

    cost = np.zeros((n_states, n))
    backptr = np.zeros((n_states, n), dtype=int)

    # Initialization
    emit0 = lambdas - counts[0] * np.log(lambdas + 1e-12)
    cost[:, 0] = emit0

    for t in range(1, n):
        x = counts[t]
        emit = lambdas - x * np.log(lambdas + 1e-12)
        for i in range(n_states):
            transition_costs = cost[:, t - 1] + gamma * np.abs(np.arange(n_states) - i)
            best_prev = np.argmin(transition_costs)
            backptr[i, t] = best_prev
            cost[i, t] = emit[i] + transition_costs[best_prev]

    states = np.zeros(n, dtype=int)
    states[-1] = int(np.argmin(cost[:, -1]))
    for t in range(n - 2, -1, -1):
        states[t] = backptr[states[t + 1], t + 1]
    return states


@dataclass
class BurstParams:
    s: float
    gamma: float
    n_states: int


@dataclass
class BurstMetrics:
    params: BurstParams
    pr_auc: float
    roc_auc: float
    f1_at_zero: float
    precision_at_zero: float
    recall_at_zero: float


def _burst_mask(states: np.ndarray, *, min_len: int, max_len: int, persistence: int) -> np.ndarray:
    """Convert burst states into a binary detection mask with duration and persistence constraints."""
    n = len(states)
    mask = np.zeros(n, dtype=int)
    start = None
    for i, s in enumerate(states):
        if s > 0 and start is None:
            start = i
        if (s == 0 or i == n - 1) and start is not None:
            end = i if s == 0 else i + 1
            length = end - start
            if min_len <= length <= max_len:
                for j in range(start + persistence - 1, end):
                    if j < n:
                        mask[j] = 1
            start = None
    return mask


def evaluate_counts(
    df: pd.DataFrame,
    params: BurstParams,
    min_len: int,
    max_len: int,
    persistence: int,
) -> BurstMetrics:
    predictions = []
    scores = []
    y_true = []

    for _, group in df.groupby("lineage_id"):
        counts = group["citation_count"].to_numpy()
        states = kleinberg_burst(counts, params.s, params.gamma, params.n_states)
        scores.append(states)
        predictions.append(_burst_mask(states, min_len=min_len, max_len=max_len, persistence=persistence))
        y_true.append(group["is_inflection_true"].to_numpy())

    score_vec = np.concatenate(scores)
    pred_vec = np.concatenate(predictions)
    y_vec = np.concatenate(y_true).astype(int)

    pr_auc = float(average_precision_score(y_vec, score_vec))
    try:
        roc_auc = float(roc_auc_score(y_vec, score_vec))
    except ValueError:
        roc_auc = float("nan")

    f1 = float(f1_score(y_vec, pred_vec))
    precision = float(precision_score(y_vec, pred_vec, zero_division=0))
    recall = float(recall_score(y_vec, pred_vec))

    return BurstMetrics(
        params=params,
        pr_auc=pr_auc,
        roc_auc=roc_auc,
        f1_at_zero=f1,
        precision_at_zero=precision,
        recall_at_zero=recall,
    )


def select_best(metrics: List[BurstMetrics]) -> BurstMetrics:
    return sorted(metrics, key=lambda m: (m.pr_auc, m.f1_at_zero, m.precision_at_zero), reverse=True)[0]


def compute_detection_lag(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["quarter_int"] = df["quarter"].apply(quarter_to_int)

    first_pred = (
        df[df["is_inflection_pred"] == 1]
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

    merged = df.merge(first_pred, on="lineage_id", how="left").merge(true_inflection, on="lineage_id", how="left")
    merged["detection_lag_quarters"] = merged["first_pred_quarter"] - merged["true_inflection_quarter"]
    merged.loc[merged["first_pred_quarter"].isna(), "detection_lag_quarters"] = np.nan
    merged.drop(columns=["quarter_int", "first_pred_quarter", "true_inflection_quarter"], inplace=True)
    return merged


def run(args: argparse.Namespace) -> None:
    start_ts = time.time()
    labels_path = Path(args.labels_csv)
    features_path = Path(args.features_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    labels = pd.read_csv(labels_path)[["lineage_id", "quarter", "is_inflection_true"]]
    features = pd.read_csv(features_path, usecols=["lineage_id", "quarter", "citation_velocity_roll_1q"])

    df = labels.merge(features, on=["lineage_id", "quarter"], how="left")
    df["citation_count"] = df["citation_velocity_roll_1q"].fillna(0.0).round().clip(lower=0).astype(int)
    df["quarter_int"] = df["quarter"].apply(quarter_to_int)

    train_df = df[df["quarter_int"] <= quarter_to_int(args.train_cutoff)].copy()
    # Identify lineages with sufficient signal for tuning
    valid_lineages = train_df.groupby("lineage_id")["citation_count"].apply(lambda s: (s > 0).sum())
    keep_ids = valid_lineages[valid_lineages >= args.min_nonzero].index
    train_df = train_df[train_df["lineage_id"].isin(keep_ids)]

    param_grid = [
        BurstParams(s, gamma, n_states)
        for s in args.s_grid
        for gamma in args.gamma_grid
        for n_states in args.states_grid
    ]

    grid_metrics: List[BurstMetrics] = [
        evaluate_counts(train_df, p, args.min_burst_len, args.max_burst_len, args.persistence) for p in param_grid
    ]
    best = select_best(grid_metrics)

    # Full run with best params
    all_scores = []
    all_preds = []
    indices = []
    for lineage_id, group in df.sort_values(["lineage_id", "quarter_int"]).groupby("lineage_id"):
        counts = group["citation_count"].to_numpy()
        if lineage_id not in keep_ids:
            states = np.zeros_like(counts)
        else:
            states = kleinberg_burst(counts, best.params.s, best.params.gamma, best.params.n_states)
        all_scores.append(states)
        all_preds.append(_burst_mask(states, min_len=args.min_burst_len, max_len=args.max_burst_len, persistence=args.persistence))
        indices.append(group.index.to_numpy())

    score_vec = np.concatenate(all_scores)
    pred_vec = np.concatenate(all_preds)
    idx_vec = np.concatenate(indices)

    df_sorted = df.loc[idx_vec].copy()
    df_sorted["burst_level"] = score_vec
    df_sorted["is_inflection_pred"] = pred_vec
    df_with_lag = compute_detection_lag(df_sorted)

    output_cols = [
        "lineage_id",
        "quarter",
        "is_inflection_true",
        "is_inflection_pred",
        "burst_level",
        "detection_lag_quarters",
        "citation_velocity_roll_1q",
    ]
    predictions_path = output_dir / "breakthrough_predictions.csv"
    df_with_lag[output_cols].to_csv(predictions_path, index=False)

    metrics_output = {
        "best_params": {
            "s": best.params.s,
            "gamma": best.params.gamma,
            "n_states": best.params.n_states,
            "pr_auc": best.pr_auc,
            "roc_auc": best.roc_auc,
            "f1_at_zero": best.f1_at_zero,
            "precision_at_zero": best.precision_at_zero,
            "recall_at_zero": best.recall_at_zero,
        },
        "grid": [
            {
                "s": m.params.s,
                "gamma": m.params.gamma,
                "n_states": m.params.n_states,
                "pr_auc": m.pr_auc,
                "roc_auc": m.roc_auc,
                "f1_at_zero": m.f1_at_zero,
                "precision_at_zero": m.precision_at_zero,
                "recall_at_zero": m.recall_at_zero,
            }
            for m in grid_metrics
        ],
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics_output, indent=2))

    run_metadata = {
        "start_time": start_ts,
        "end_time": time.time(),
        "duration_sec": time.time() - start_ts,
        "labels_csv": str(labels_path),
        "labels_sha256": hash_file(labels_path),
        "features_csv": str(features_path),
        "features_sha256": hash_file(features_path),
        "output_predictions": str(predictions_path),
        "train_cutoff": args.train_cutoff,
        "s_grid": args.s_grid,
        "gamma_grid": args.gamma_grid,
        "states_grid": args.states_grid,
        "min_nonzero": args.min_nonzero,
        "min_burst_len": args.min_burst_len,
        "max_burst_len": args.max_burst_len,
        "persistence": args.persistence,
    }
    (output_dir / "run_metadata.json").write_text(json.dumps(run_metadata, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Kleinberg citation burst baseline.")
    parser.add_argument(
        "--labels-csv",
        default="data/out/experiments/msd_training/msd_inflection/leakage_free/breakthrough_predictions.csv",
        help="CSV with lineage_id, quarter, is_inflection_true.",
    )
    parser.add_argument(
        "--features-csv",
        default="data/out/02_lineage_tracking/lineage_multisignal_features.csv",
        help="CSV with citation_velocity_roll_1q per lineage-quarter.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/out/experiments/baselines/kleinberg_burst",
        help="Directory for outputs.",
    )
    parser.add_argument(
        "--train-cutoff",
        default=TRAIN_CUTOFF_DEFAULT,
        help="Inclusive training cutoff quarter for parameter tuning.",
    )
    parser.add_argument(
        "--s-grid",
        type=float,
        nargs="+",
        default=[1.5, 2.0, 2.5, 3.0],
        help="Intensity multiplier grid.",
    )
    parser.add_argument(
        "--gamma-grid",
        type=float,
        nargs="+",
        default=[0.5, 1.0, 1.5],
        help="Transition penalty grid.",
    )
    parser.add_argument(
        "--states-grid",
        type=int,
        nargs="+",
        default=[2, 3],
        help="Number of burst states.",
    )
    parser.add_argument(
        "--min-nonzero",
        type=int,
        default=MIN_NONZERO_DEFAULT,
        help="Minimum nonzero citation quarters per lineage to include.",
    )
    parser.add_argument(
        "--min-burst-len",
        type=int,
        default=MIN_BURST_LEN_DEFAULT,
        help="Minimum burst duration (quarters) to keep.",
    )
    parser.add_argument(
        "--max-burst-len",
        type=int,
        default=MAX_BURST_LEN_DEFAULT,
        help="Maximum burst duration (quarters) to keep.",
    )
    parser.add_argument(
        "--persistence",
        type=int,
        default=PERSISTENCE_DEFAULT,
        help="Minimum consecutive quarters in-state to emit detection.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args)
