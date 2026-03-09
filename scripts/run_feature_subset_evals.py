#!/usr/bin/env python3
"""
Feature subset evaluation runner.

Loads the MSD dataset via the canonical pipeline, applies time-forward splits,
then trains/evaluates the standard LightGBM detector for each feature subset
defined in a YAML config. Reports metrics at the canonical threshold (0.07) and
across a sweep of thresholds, producing comparison tables and PR-curve plots.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

# Ensure repo root on path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Repo imports
from scripts.multi_signal_detector import (  # type: ignore
    load_and_merge_signals,
    construct_labels,
    engineer_features,
)
from scripts.utils.feature_registry import FeatureRegistry  # type: ignore
from utils.quarter_utils import filter_by_quarter  # type: ignore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate MSD feature subsets.")

    # Data inputs
    parser.add_argument("--tight-mapping", type=Path,
                        default=Path("data/out/experiments/stage0_tight_mapping/milestone_lineage_mapping_tight.csv"))
    parser.add_argument("--semantic-velocity", type=Path,
                        default=Path("data/out/experiments/stage1_quarterly_embeddings/semantic_velocity.csv"))
    parser.add_argument("--multisignal", type=Path,
                        default=Path("data/out/02_lineage_tracking/lineage_multisignal_features.csv"))
    parser.add_argument("--timeseries", type=Path,
                        default=Path("data/out/02_lineage_tracking/lineage_timeseries.csv"))
    parser.add_argument("--labels", type=Path, default=None)

    # Feature + config paths
    parser.add_argument("--feature-config", type=Path,
                        default=Path("config/features/feature_groups.yaml"))
    parser.add_argument("--subset-config", type=Path,
                        default=Path("config/features/feature_subset_configs.yaml"),
                        help="YAML describing feature subsets to evaluate.")
    parser.add_argument("--splits-config", type=Path,
                        default=Path("config/splits/msd_timeforward.yaml"))
    parser.add_argument("--threshold", type=float, default=0.07)
    parser.add_argument("--output-dir", type=Path,
                        default=Path("data/out/analysis/feature_signal_pruning/subset_eval"))

    # Execution controls
    parser.add_argument("--sample-limit", type=int, default=None,
                        help="Optional per-split sample limit (stratified).")
    parser.add_argument("--max-configs", type=int, default=None,
                        help="Limit number of configs processed.")
    parser.add_argument("--n-estimators", type=int, default=800)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--export-feature-list", action="store_true",
                        help="Write feature lists for each config under output_dir/feature_lists.")

    return parser.parse_args()


def load_subset_config(path: Path) -> Dict[str, Dict]:
    if not path.exists():
        raise FileNotFoundError(f"Subset config not found: {path}")
    with path.open("r", encoding="utf-8") as fp:
        data = yaml.safe_load(fp) or {}
    return data.get("configs", data)


def load_split_config(path: Path) -> Dict[str, Tuple[Optional[str], Optional[str]]]:
    with path.open("r", encoding="utf-8") as fp:
        data = yaml.safe_load(fp) or {}
    splits = data.get("splits", data)
    resolved = {}
    for name, span in splits.items():
        start = span.get("start")
        end = span.get("end")
        resolved[name] = (start, end)
    return resolved


def stratified_sample(df: pd.DataFrame, n_samples: Optional[int], seed: int) -> pd.DataFrame:
    if n_samples is None or len(df) <= n_samples:
        return df
    positives = df[df["is_milestone"] == 1]
    negatives = df[df["is_milestone"] == 0]
    if positives.empty:
        return df.head(n_samples).copy()
    ratio = len(positives) / len(df)
    n_pos = max(1, int(n_samples * ratio))
    n_neg = n_samples - n_pos
    pos_sample = positives.sample(n=min(len(positives), n_pos), random_state=seed)
    neg_sample = negatives.sample(n=min(len(negatives), n_neg), random_state=seed)
    return pd.concat([pos_sample, neg_sample]).sample(frac=1, random_state=seed).reset_index(drop=True)


def build_matrix(df: pd.DataFrame, features: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    X = df[features].replace([np.inf, -np.inf], 0).fillna(0).astype(np.float32).values
    y = df["is_milestone"].astype(int).values
    return X, y


def train_model(X: np.ndarray, y: np.ndarray, seed: int,
                n_estimators: int, learning_rate: float) -> LGBMClassifier:
    clf = LGBMClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=-1,
        num_leaves=255,
        subsample=0.8,
        colsample_bytree=0.8,
        class_weight="balanced",
        random_state=seed,
        n_jobs=-1,
    )
    clf.fit(X, y)
    return clf


def evaluate_model(model: LGBMClassifier, X: np.ndarray, y: np.ndarray,
                   threshold: float, thresholds_sweep: List[float]) -> Dict:
    probs = model.predict_proba(X)[:, 1]
    preds = (probs >= threshold).astype(int)
    metrics = {
        "precision": precision_score(y, preds, zero_division=0),
        "recall": recall_score(y, preds, zero_division=0),
        "f1": f1_score(y, preds, zero_division=0),
        "roc_auc": roc_auc_score(y, probs) if y.sum() > 0 else 0.0,
        "pr_auc": average_precision_score(y, probs) if y.sum() > 0 else 0.0,
        "detections": int(preds.sum()),
    }
    sweep_rows = []
    for thr in thresholds_sweep:
        sweep_preds = (probs >= thr).astype(int)
        sweep_rows.append({
            "threshold": thr,
            "precision": precision_score(y, sweep_preds, zero_division=0),
            "recall": recall_score(y, sweep_preds, zero_division=0),
            "f1": f1_score(y, sweep_preds, zero_division=0),
            "detections": int(sweep_preds.sum()),
        })
    curve = precision_recall_curve(y, probs)
    return metrics, sweep_rows, curve


def plot_pr_curve(curve: Tuple[np.ndarray, np.ndarray, np.ndarray],
                  label: str, output_path: Path) -> None:
    precision, recall, _ = curve
    plt.figure(figsize=(5, 4))
    plt.step(recall, precision, where="post", label=label)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title(f"Precision-Recall Curve ({label})")
    plt.legend()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    registry = FeatureRegistry(args.feature_config)
    subset_config = load_subset_config(args.subset_config)
    split_config = load_split_config(args.splits_config)

    # Load dataset via MSD pipeline
    features_df, labels_df, tight_mapping = load_and_merge_signals(
        args.tight_mapping,
        args.semantic_velocity,
        args.multisignal,
        args.timeseries,
        labels_path=args.labels,
        n_samples=None,
    )
    features_df = construct_labels(features_df, labels_df, tight_mapping)
    features_df = engineer_features(features_df)

    # Build splits
    splits: Dict[str, pd.DataFrame] = {}
    for split_name, (start, end) in split_config.items():
        split_df = filter_by_quarter(features_df, start, end, label=split_name)
        split_df = stratified_sample(split_df, args.sample_limit, args.random_seed)
        splits[split_name] = split_df.reset_index(drop=True)

    train_df = splits.get("train")
    dev_df = splits.get("dev")
    test_df = splits.get("test")
    if train_df is None or dev_df is None or test_df is None:
        raise ValueError("Split config must define train/dev/test sections.")

    thresholds_sweep = [round(x / 100, 2) for x in range(1, 21)]
    results_rows = []
    sweep_rows_all = []
    feature_lists_dir = output_dir / "feature_lists"
    if args.export_feature_list:
        feature_lists_dir.mkdir(parents=True, exist_ok=True)

    processed = 0
    for config_name, spec in subset_config.items():
        if args.max_configs is not None and processed >= args.max_configs:
            break
        include_groups = spec.get("include_groups")
        include_patterns = spec.get("include_patterns")
        exclude_groups = spec.get("exclude_groups")
        exclude_patterns = spec.get("exclude_patterns")
        columns = registry.resolve_features(
            include_groups=include_groups or ["all_predictors"],
            include_patterns=include_patterns,
            exclude_groups=exclude_groups,
            exclude_patterns=exclude_patterns,
        )
        available_cols = [c for c in columns if c in train_df.columns]
        if not available_cols:
            print(f"[WARN] Config '{config_name}' has no available columns. Skipping.")
            continue
        if args.export_feature_list:
            (feature_lists_dir / f"{config_name}.txt").write_text("\n".join(available_cols), encoding="utf-8")

        # Prepare matrices
        X_train, y_train = build_matrix(train_df, available_cols)
        X_dev, y_dev = build_matrix(dev_df, available_cols)
        X_test, y_test = build_matrix(test_df, available_cols)

        print(f"[Subset] Training config '{config_name}' with {len(available_cols)} features.")
        model = train_model(X_train, y_train, args.random_seed,
                            args.n_estimators, args.learning_rate)

        for split_label, X_split, y_split in [
            ("dev", X_dev, y_dev),
            ("test", X_test, y_test),
        ]:
            metrics, sweep_data, curve = evaluate_model(
                model, X_split, y_split, args.threshold, thresholds_sweep)
            row = {
                "config": config_name,
                "dataset": split_label,
                "features": len(available_cols),
                **metrics,
            }
            results_rows.append(row)
            for sweep_row in sweep_data:
                sweep_row.update({
                    "config": config_name,
                    "dataset": split_label,
                })
                sweep_rows_all.append(sweep_row)
            plot_path = plots_dir / f"pr_curve_{config_name}_{split_label}.png"
            plot_pr_curve(curve, f"{config_name}-{split_label}", plot_path)

        processed += 1

    results_df = pd.DataFrame(results_rows)
    sweep_df = pd.DataFrame(sweep_rows_all)
    results_csv = output_dir / "subset_eval_results.csv"
    thresholds_csv = output_dir / "subset_eval_thresholds.csv"
    results_df.to_csv(results_csv, index=False)
    sweep_df.to_csv(thresholds_csv, index=False)

    summary = {
        "threshold": args.threshold,
        "configs_evaluated": processed,
        "best_by_recall": results_df.sort_values("recall", ascending=False).head(5).to_dict("records"),
    }
    (output_dir / "subset_eval_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    summary_lines = ["# Feature Subset Evaluation", "", f"Threshold: {args.threshold}", "",
                     "## Top Configs by Recall"]
    for entry in summary["best_by_recall"]:
        summary_lines.append(
            f"- {entry['config']} ({entry['dataset']}): recall={entry['recall']:.3f}, precision={entry['precision']:.3f}, pr_auc={entry['pr_auc']:.3f}"
        )
    (output_dir / "subset_eval_summary.md").write_text("\n".join(summary_lines), encoding="utf-8")
    print(f"[Done] Saved subset evaluation results to {output_dir}")


if __name__ == "__main__":
    main()
