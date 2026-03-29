#!/usr/bin/env python3
"""
GPU-only XGBoost CV runner using DeviceQuantileDMatrix and inplace_predict.

Minimal wrapper to evaluate the MSD dataset with a single XGBoost configuration
on GPU, bypassing sklearn pipelines/calibrators to avoid device-mismatch issues.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import xgboost as xgb
from _path_bootstrap import ensure_repo_imports
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold

REPO_ROOT = ensure_repo_imports()

from scripts.multi_signal_detector import (  # type: ignore  # noqa: E402
    construct_labels,
    engineer_features,
    load_and_merge_signals,
    select_features,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GPU XGBoost CV on MSD dataset.")
    parser.add_argument("--tight-mapping", type=Path, required=True)
    parser.add_argument("--semantic-velocity", type=Path, required=True)
    parser.add_argument("--multisignal", type=Path, required=True)
    parser.add_argument("--timeseries", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--feature-config", type=Path, default=Path("config/features/feature_groups.yaml"))
    parser.add_argument("--threshold", type=float, default=0.07)
    parser.add_argument("--lag-min", type=int, default=2)
    parser.add_argument("--lag-max", default="8")
    parser.add_argument("--n-splits", type=int, default=3)
    # XGB params
    parser.add_argument("--n-estimators", type=int, default=800)
    parser.add_argument("--max-depth", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--subsample", type=float, default=0.8)
    parser.add_argument("--colsample-bytree", type=float, default=0.8)
    parser.add_argument("--min-child-weight", type=float, default=1.0)
    parser.add_argument("--reg-lambda", type=float, default=1.0)
    return parser.parse_args()


def build_dmatrix(X: np.ndarray, y: np.ndarray) -> xgb.DMatrix:
    """Build a GPU-backed DMatrix if possible."""
    try:
        return xgb.DeviceQuantileDMatrix(X, label=y)
    except Exception:
        # Fallback to standard DMatrix (will rely on device parameter during training)
        return xgb.DMatrix(X, label=y)


def train_and_eval(
    X: np.ndarray,
    y: np.ndarray,
    threshold: float,
    params: dict,
    n_splits: int,
) -> tuple[dict, list[dict]]:
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    metrics = []
    for train_idx, val_idx in skf.split(X, y):
        dtrain = build_dmatrix(X[train_idx], y[train_idx])
        dval = build_dmatrix(X[val_idx], y[val_idx])
        booster = xgb.train(
            params,
            dtrain,
            num_boost_round=params.pop("num_boost_round", 800),
            evals=[(dval, "validation")],
            verbose_eval=False,
        )
        # inplace_predict keeps it GPU-aware when possible
        y_prob = booster.predict(dval)
        y_pred = (y_prob >= threshold).astype(int)

        fold_metrics = {
            "precision": precision_score(y[val_idx], y_pred, zero_division=0),
            "recall": recall_score(y[val_idx], y_pred, zero_division=0),
            "f1": f1_score(y[val_idx], y_pred, zero_division=0),
            "pr_auc": average_precision_score(y[val_idx], y_prob),
            "roc_auc": roc_auc_score(y[val_idx], y_prob) if len(np.unique(y[val_idx])) > 1 else 0.0,
        }
        metrics.append(fold_metrics)

    # Aggregate
    agg = {k: float(np.mean([m[k] for m in metrics])) for k in metrics[0]}
    return agg, metrics


def main() -> None:
    args = parse_args()

    features_df, labels_df, tight_mapping = load_and_merge_signals(
        args.tight_mapping,
        args.semantic_velocity,
        args.multisignal,
        args.timeseries,
        labels_path=args.labels,
        n_samples=None,
    )

    features_df = construct_labels(
        features_df,
        labels_df,
        tight_mapping,
        lag_min=args.lag_min,
        lag_max=None if args.lag_max == "none" else args.lag_max,
    )
    features_df = engineer_features(features_df)
    X, feature_names = select_features(features_df)
    y = features_df["is_milestone"].to_numpy()
    X_np = np.asarray(X, dtype=np.float32)

    params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "tree_method": "hist",
        "device": "cuda",
        "predictor": "gpu_predictor",
        "eta": args.learning_rate,
        "max_depth": args.max_depth,
        "subsample": args.subsample,
        "colsample_bytree": args.colsample_bytree,
        "min_child_weight": args.min_child_weight,
        "lambda": args.reg_lambda,
        "num_boost_round": args.n_estimators,
    }

    agg, folds = train_and_eval(
        X_np,
        y,
        threshold=args.threshold,
        params=params,
        n_splits=args.n_splits,
    )

    print("GPU XGBoost CV Results:")
    print(f"  Precision: {agg['precision']:.4f}")
    print(f"  Recall:    {agg['recall']:.4f}")
    print(f"  F1:        {agg['f1']:.4f}")
    print(f"  PR-AUC:    {agg['pr_auc']:.4f}")
    print(f"  ROC-AUC:   {agg['roc_auc']:.4f}")


if __name__ == "__main__":
    main()
