"""
Semantic changepoint baseline using quarterly embeddings.

Approach:
- Compute cosine-distance changes between consecutive quarterly embeddings per lineage.
- Tune global change-score threshold (percentiles over training window) and optional ±dilation window
  to maximize PR-AUC on training quarters (<=2019Q4).
- Emit per-quarter change scores, binary predictions, and detection lags.

Outputs:
- breakthrough_predictions.csv (lineage_id, quarter, is_inflection_true, change_score, is_inflection_pred, detection_lag_quarters)
- metrics.json (best params + grid summary)
- run_metadata.json (timestamps, hashes, parameters)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

TRAIN_CUTOFF_DEFAULT = "2019Q4"


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
class ParamSet:
    penalty: float
    dilation: int
    min_segment: int


@dataclass
class ParamMetrics:
    params: ParamSet
    pr_auc: float
    roc_auc: float
    f1_at_zero: float
    precision_at_zero: float
    recall_at_zero: float


def load_embeddings(npz_path: Path) -> pd.DataFrame:
    data = np.load(npz_path, allow_pickle=True)
    lineage_ids = data["lineage_ids"]
    quarters = data["quarters"]
    embeddings = data["embeddings"]
    return pd.DataFrame(
        {
            "lineage_id": lineage_ids,
            "quarter": quarters,
            "embedding": list(embeddings),
        }
    )


def compute_change_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Add change_score column (1 - cosine similarity to previous quarter)."""
    rows = []
    for _lineage_id, group in df.sort_values(["lineage_id", "quarter_int"]).groupby("lineage_id"):
        emb = np.stack(group["embedding"].to_numpy())
        norms = np.linalg.norm(emb, axis=1)
        dot = np.sum(emb[1:] * emb[:-1], axis=1)
        denom = norms[1:] * norms[:-1] + 1e-12
        cos_sim = dot / denom
        change = np.concatenate([[0.0], 1 - cos_sim])
        g = group.copy()
        g["change_score"] = change
        rows.append(g)
    return pd.concat(rows, ignore_index=True)


def apply_dilation(preds: np.ndarray, dilation: int) -> np.ndarray:
    if dilation == 0:
        return preds
    dilated = preds.copy()
    for shift in range(1, dilation + 1):
        dilated[shift:] = np.maximum(dilated[shift:], preds[:-shift])
        dilated[:-shift] = np.maximum(dilated[:-shift], preds[shift:])
    return dilated


def evaluate_params(df: pd.DataFrame, params: ParamSet, min_size: int) -> ParamMetrics:
    try:
        import ruptures as rpt
    except ImportError as exc:
        raise SystemExit("ruptures is required for semantic changepoint baseline. Install with `pip install ruptures`.") from exc

    preds_list: list[np.ndarray] = []
    scores_list: list[np.ndarray] = []
    y_list: list[np.ndarray] = []

    for _, g in df.sort_values(["lineage_id", "quarter_int"]).groupby("lineage_id"):
        emb = np.stack(g["embedding"].to_numpy())
        n = len(emb)
        if n < 2 * min_size:
            preds = np.zeros(n, dtype=int)
            preds_list.append(preds)
            scores_list.append(np.zeros(n))
            y_list.append(g["is_inflection_true"].astype(int).to_numpy())
            continue
        algo = rpt.Pelt(model="rbf", min_size=min_size, jump=1)
        cp = algo.fit(emb).predict(pen=params.penalty)
        breakpoints = [c - 1 for c in cp[:-1] if c - 1 >= 0]
        preds = np.zeros(n, dtype=int)
        for bp in breakpoints:
            preds[bp] = 1
        dists = g["change_score"].to_numpy()
        if params.dilation > 0:
            preds = apply_dilation(preds, params.dilation)
        preds_list.append(preds)
        scores_list.append(dists)
        y_list.append(g["is_inflection_true"].astype(int).to_numpy())

    score_vec = np.concatenate(scores_list)
    pred_vec = np.concatenate(preds_list)
    y_vec = np.concatenate(y_list)

    pr_auc = float(average_precision_score(y_vec, score_vec))
    try:
        roc_auc = float(roc_auc_score(y_vec, score_vec))
    except ValueError:
        roc_auc = float("nan")
    f1 = float(f1_score(y_vec, pred_vec, zero_division=0))
    precision = float(precision_score(y_vec, pred_vec, zero_division=0))
    recall = float(recall_score(y_vec, pred_vec, zero_division=0))

    return ParamMetrics(
        params=params,
        pr_auc=pr_auc,
        roc_auc=roc_auc,
        f1_at_zero=f1,
        precision_at_zero=precision,
        recall_at_zero=recall,
    )


def select_best(metrics: list[ParamMetrics]) -> ParamMetrics:
    return sorted(metrics, key=lambda m: (m.pr_auc, m.f1_at_zero, m.precision_at_zero), reverse=True)[0]


def compute_detection_lag(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["quarter_int"] = df["quarter"].apply(quarter_to_int)

    first_pred_per_lineage = (
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

    merged = df.merge(first_pred_per_lineage, on="lineage_id", how="left").merge(true_inflection, on="lineage_id", how="left")
    merged["detection_lag_quarters"] = merged["first_pred_quarter"] - merged["true_inflection_quarter"]
    merged.loc[merged["first_pred_quarter"].isna(), "detection_lag_quarters"] = np.nan
    merged.drop(columns=["quarter_int", "first_pred_quarter", "true_inflection_quarter"], inplace=True)
    return merged


def run(args: argparse.Namespace) -> None:
    start_ts = time.time()
    labels_path = Path(args.labels_csv)
    embeddings_path = Path(args.embeddings_npz)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    labels = pd.read_csv(labels_path)[["lineage_id", "quarter", "is_inflection_true"]]
    labels["quarter_int"] = labels["quarter"].apply(quarter_to_int)

    embeddings_df = load_embeddings(embeddings_path)
    embeddings_df["quarter_int"] = embeddings_df["quarter"].astype(str).apply(quarter_to_int)

    merged = labels.merge(embeddings_df, on=["lineage_id", "quarter", "quarter_int"], how="inner")
    merged = compute_change_scores(merged)

    train_df = merged[merged["quarter_int"] <= quarter_to_int(args.train_cutoff)].copy()

    grid = [ParamSet(pen, d, args.min_segment) for pen in args.penalties for d in args.dilation]
    metrics_grid: list[ParamMetrics] = [evaluate_params(train_df, params, args.min_segment) for params in grid]
    best = select_best(metrics_grid)

    merged_sorted = merged.sort_values(["lineage_id", "quarter_int"]).copy()
    merged_sorted["change_score"] = merged_sorted["change_score"].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    try:
        import ruptures as rpt
    except ImportError as exc:
        raise SystemExit("ruptures is required for semantic changepoint baseline. Install with `pip install ruptures`.") from exc

    preds_list: list[np.ndarray] = []
    idx_list: list[np.ndarray] = []
    for _, g in merged_sorted.groupby("lineage_id"):
        emb = np.stack(g["embedding"].to_numpy())
        n = len(emb)
        if n < 2 * args.min_segment:
            preds = np.zeros(n, dtype=int)
        else:
            algo = rpt.Pelt(model="rbf", min_size=args.min_segment, jump=1)
            cp = algo.fit(emb).predict(pen=best.params.penalty)
            breakpoints = [c - 1 for c in cp[:-1] if c - 1 >= 0]
            preds = np.zeros(n, dtype=int)
            for bp in breakpoints:
                preds[bp] = 1
            if best.params.dilation > 0:
                preds = apply_dilation(preds, best.params.dilation)
        preds_list.append(preds)
        idx_list.append(g.index.to_numpy())

    pred_vec = np.concatenate(preds_list)
    idx_vec = np.concatenate(idx_list)
    merged_sorted["is_inflection_pred"] = 0
    merged_sorted.loc[idx_vec, "is_inflection_pred"] = pred_vec
    merged_with_lag = compute_detection_lag(merged_sorted)

    output_cols = [
        "lineage_id",
        "quarter",
        "is_inflection_true",
        "is_inflection_pred",
        "change_score",
        "detection_lag_quarters",
    ]
    predictions_path = output_dir / "breakthrough_predictions.csv"
    merged_with_lag[output_cols].to_csv(predictions_path, index=False)

    metrics_output = {
        "best_params": {
            "penalty": best.params.penalty,
            "dilation": best.params.dilation,
            "min_segment": best.params.min_segment,
            "pr_auc": best.pr_auc,
            "roc_auc": best.roc_auc,
            "f1_at_zero": best.f1_at_zero,
            "precision_at_zero": best.precision_at_zero,
            "recall_at_zero": best.recall_at_zero,
        },
        "grid": [
            {
                "penalty": m.params.penalty,
                "dilation": m.params.dilation,
                "min_segment": m.params.min_segment,
                "pr_auc": m.pr_auc,
                "roc_auc": m.roc_auc,
                "f1_at_zero": m.f1_at_zero,
                "precision_at_zero": m.precision_at_zero,
                "recall_at_zero": m.recall_at_zero,
            }
            for m in metrics_grid
        ],
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics_output, indent=2))

    run_metadata = {
        "start_time": start_ts,
        "end_time": time.time(),
        "duration_sec": time.time() - start_ts,
        "labels_csv": str(labels_path),
        "labels_sha256": hash_file(labels_path),
        "embeddings_npz": str(embeddings_path),
        "embeddings_sha256": hash_file(embeddings_path),
        "output_predictions": str(predictions_path),
        "train_cutoff": args.train_cutoff,
        "penalties": args.penalties,
        "dilation": args.dilation,
        "min_segment": args.min_segment,
    }
    (output_dir / "run_metadata.json").write_text(json.dumps(run_metadata, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Semantic changepoint baseline using quarterly embeddings.")
    parser.add_argument(
        "--labels-csv",
        default="data/out/experiments/msd_training/msd_inflection/leakage_free/breakthrough_predictions.csv",
        help="CSV with lineage_id, quarter, is_inflection_true.",
    )
    parser.add_argument(
        "--embeddings-npz",
        default="data/out/experiments/msd_training/phase1_quarterly_embeddings/quarterly_embeddings.npz",
        help="NPZ with arrays: lineage_ids, quarters, embeddings.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/out/experiments/baselines/semantic_changepoint",
        help="Directory for outputs.",
    )
    parser.add_argument(
        "--train-cutoff",
        default=TRAIN_CUTOFF_DEFAULT,
        help="Inclusive training cutoff quarter for threshold tuning.",
    )
    parser.add_argument(
        "--percentiles",
        type=float,
        nargs="+",
        default=[90, 95, 97, 99],
        help="Percentile thresholds over training change scores.",
    )
    parser.add_argument(
        "--dilation",
        type=int,
        nargs="+",
        default=[0, 1],
        help="Neighborhood dilation (quarters) around detected changepoints.",
    )
    parser.add_argument(
        "--min-segment",
        type=int,
        default=4,
        help="Minimum segment length between changepoints within a lineage.",
    )
    parser.add_argument(
        "--penalties",
        type=float,
        nargs="+",
        default=[1.0, 2.0, 4.0, 8.0],
        help="Penalty grid for PELT changepoint detection.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args)
