#!/usr/bin/env python3
"""Quarterly MSD retraining with model versioning.

Retrains the Multi-Signal Detector on the latest labeled data and saves
the result as a versioned model artifact in the model registry.  Supports
two modes:

- **full**: Train a fresh model from scratch on all available data.
- **incremental**: Warm-start from the previous CatBoost model, adding
  new boosting rounds on the updated dataset.

Usage::

    # Full retrain (default)
    python scripts/retrain_msd.py

    # Incremental warm-start from latest version
    python scripts/retrain_msd.py --mode incremental

    # Compare against a specific previous version
    python scripts/retrain_msd.py --compare-to v_20260323_001
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.model_registry import (  # noqa: E402
    ModelVersion,
    _next_version_id,
    compare_versions,
    get_latest_version_id,
    load_model_version,
    save_versioned_model,
)

logger = logging.getLogger(__name__)

_DEFAULT_REGISTRY = "data/out/models/msd"
_DEFAULT_MULTISIGNAL = "data/out/02_lineage_tracking/lineage_multisignal_features.csv"
_DEFAULT_TIMESERIES = "data/out/02_lineage_tracking/lineage_timeseries.csv"
_DEFAULT_LABELS = "data/out/02_lineage_tracking/onset_labels_msd.csv"
_DEFAULT_TIGHT_MAPPING = (
    "data/out/experiments/stage0_tight_mapping/milestone_lineage_mapping_tight.csv"
)
_DEFAULT_SEMANTIC_VELOCITY = (
    "data/out/experiments/stage1_quarterly_embeddings/semantic_velocity.csv"
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Retrain MSD with model versioning.",
    )
    parser.add_argument(
        "--registry-dir", default=_DEFAULT_REGISTRY,
        help="Model registry directory (default: %(default)s)",
    )
    parser.add_argument(
        "--mode", choices=["full", "incremental"], default="full",
        help="Retrain mode: full (from scratch) or incremental (warm-start). "
             "Default: full",
    )
    parser.add_argument(
        "--previous-version", default=None,
        help="Explicit previous version ID for warm-start. "
             "If omitted, uses latest version in registry.",
    )
    parser.add_argument(
        "--compare-to", default=None,
        help="Version ID to compare metrics against. "
             "If omitted, compares against previous version.",
    )

    # Data inputs
    parser.add_argument(
        "--multisignal", default=_DEFAULT_MULTISIGNAL,
        help="Multisignal features CSV (default: %(default)s)",
    )
    parser.add_argument(
        "--timeseries", default=_DEFAULT_TIMESERIES,
        help="Lineage timeseries CSV (default: %(default)s)",
    )
    parser.add_argument(
        "--labels", default=_DEFAULT_LABELS,
        help="Onset labels CSV (default: %(default)s)",
    )
    parser.add_argument(
        "--tight-mapping", default=_DEFAULT_TIGHT_MAPPING,
        help="Tight mapping CSV (default: %(default)s)",
    )
    parser.add_argument(
        "--semantic-velocity", default=_DEFAULT_SEMANTIC_VELOCITY,
        help="Semantic velocity CSV (default: %(default)s)",
    )

    # Training config
    parser.add_argument(
        "--model", default="catboost",
        help="Model type (default: catboost)",
    )
    parser.add_argument("--use-cv", action="store_true", help="Use cross-validation")
    parser.add_argument("--cv-folds", type=int, default=5, help="CV folds (default: 5)")
    parser.add_argument("--leakage-safe", action="store_true", help="Exclude leaky features")
    parser.add_argument(
        "--train-end", default=None,
        help="Inclusive end quarter for training data (e.g., 2023Q4)",
    )
    parser.add_argument(
        "--cat-iterations", type=int, default=1000,
        help="CatBoost boosting iterations (default: 1000)",
    )
    parser.add_argument("--cat-depth", type=int, default=8, help="CatBoost tree depth")
    parser.add_argument("--cat-learning-rate", type=float, default=0.05)
    parser.add_argument("--cat-l2", type=float, default=1.0)
    parser.add_argument("--cat-border-count", type=int, default=128)
    parser.add_argument(
        "--incremental-iterations", type=int, default=200,
        help="Additional boosting rounds for incremental mode (default: 200)",
    )
    parser.add_argument("--notes", default="", help="Free-text notes for this version")
    parser.add_argument("--verbose", "-v", action="store_true")
    return parser.parse_args()


def _load_data(args: argparse.Namespace) -> pd.DataFrame:
    """Load and merge feature data using the MSD pipeline functions."""
    # Import MSD functions (heavy imports deferred)
    sys.path.insert(0, str(_REPO / "scripts"))
    from multi_signal_detector import (
        construct_labels,
        engineer_features,
        filter_by_quarter,
        load_and_merge_signals,
    )

    labels_path = Path(args.labels) if args.labels else None
    features_df, labels_df, tight_mapping = load_and_merge_signals(
        Path(args.tight_mapping),
        Path(args.semantic_velocity),
        Path(args.multisignal),
        Path(args.timeseries),
        labels_path=labels_path,
    )
    features_df = construct_labels(features_df, labels_df, tight_mapping)
    features_df = engineer_features(features_df)

    if args.train_end:
        features_df = filter_by_quarter(
            features_df, None, args.train_end, label="Training slice",
        )

    return features_df


def _build_catboost_model(
    args: argparse.Namespace,
    init_model: Any = None,
) -> Any:
    """Build a CatBoost classifier, optionally warm-started."""
    from catboost import CatBoostClassifier

    iterations = args.cat_iterations
    if init_model is not None:
        iterations = args.incremental_iterations
        logger.info(
            "Warm-starting with %d additional iterations", iterations,
        )

    return CatBoostClassifier(
        iterations=iterations,
        depth=args.cat_depth,
        learning_rate=args.cat_learning_rate,
        l2_leaf_reg=args.cat_l2,
        border_count=args.cat_border_count,
        random_seed=42,
        verbose=False,
        loss_function="Logloss",
        eval_metric="Logloss",
        auto_class_weights="Balanced",
        thread_count=-1,
        task_type="CPU",
    )


def _train_and_evaluate(
    features_df: pd.DataFrame,
    args: argparse.Namespace,
    init_model: Any = None,
) -> dict[str, Any]:
    """Train model and compute evaluation metrics.

    Returns dict with 'pipeline', 'metrics', 'feature_names'.
    """
    from multi_signal_detector import select_features
    from sklearn.preprocessing import StandardScaler

    X, feature_names = select_features(
        features_df,
        leakage_safe=args.leakage_safe,
    )
    y = features_df["is_milestone"]

    print(f"   Dataset: {len(X)} samples, {len(feature_names)} features")
    print(f"   Positive class: {y.sum()} ({(y.sum() / len(y)) * 100:.2f}%)")

    # Build pipeline steps
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X.values)

    # Build model
    if args.model == "catboost":
        model = _build_catboost_model(args, init_model=init_model)
        if init_model is not None:
            model.fit(X_scaled, y, init_model=init_model)
        else:
            model.fit(X_scaled, y)
    else:
        raise ValueError(
            f"Incremental retraining only supported for catboost, got: {args.model}"
        )

    # Wrap in pipeline for compatibility with existing save/load
    from imblearn.pipeline import Pipeline as ImbPipeline
    pipeline = ImbPipeline([
        ("scaler", scaler),
        ("classifier", model),
    ])

    # Evaluate
    metrics: dict[str, Any] = {}
    if args.use_cv:
        from sklearn.metrics import average_precision_score, make_scorer
        from sklearn.model_selection import StratifiedKFold, cross_validate

        cv = StratifiedKFold(n_splits=args.cv_folds, shuffle=True, random_state=42)
        pr_auc_scorer = make_scorer(
            average_precision_score, needs_proba=True, response_method="predict_proba",
        )

        # For CV metrics, retrain fresh models per fold (not warm-started)
        base_model = _build_catboost_model(args, init_model=None)
        cv_pipeline = ImbPipeline([
            ("scaler", StandardScaler()),
            ("classifier", base_model),
        ])

        scoring = {
            "precision": "precision",
            "recall": "recall",
            "f1": "f1",
            "roc_auc": "roc_auc",
            "pr_auc": pr_auc_scorer,
        }
        cv_results = cross_validate(
            cv_pipeline, X.values, y,
            cv=cv, scoring=scoring, n_jobs=1,
        )
        for metric_name in ["precision", "recall", "f1", "roc_auc", "pr_auc"]:
            key = f"test_{metric_name}"
            values = cv_results[key]
            metrics[f"cv_{metric_name}_mean"] = round(float(np.mean(values)), 4)
            metrics[f"cv_{metric_name}_std"] = round(float(np.std(values)), 4)
    else:
        # Simple holdout evaluation
        from sklearn.metrics import (
            average_precision_score,
            f1_score,
            precision_score,
            recall_score,
            roc_auc_score,
        )

        y_prob = pipeline.predict_proba(X.values)[:, 1]
        y_pred = (y_prob >= 0.15).astype(int)

        metrics["precision_test"] = round(float(precision_score(y, y_pred, zero_division=0)), 4)
        metrics["recall_test"] = round(float(recall_score(y, y_pred, zero_division=0)), 4)
        metrics["f1_test"] = round(float(f1_score(y, y_pred, zero_division=0)), 4)
        metrics["roc_auc_test"] = round(float(roc_auc_score(y, y_prob)), 4)
        metrics["pr_auc_test"] = round(float(average_precision_score(y, y_prob)), 4)

    return {
        "pipeline": pipeline,
        "metrics": metrics,
        "feature_names": feature_names,
        "n_train_samples": len(X),
    }


def _format_comparison(comparison: dict[str, Any]) -> str:
    """Format a version comparison as a readable table."""
    lines = [
        f"Comparison: {comparison['current_version']} vs {comparison['previous_version']}",
        "",
        f"{'Metric':<25} {'Delta':>10}",
        "-" * 37,
    ]
    for key, delta in comparison["deltas"].items():
        sign = "+" if delta > 0 else ""
        marker = " *" if key == comparison.get("primary_metric") else ""
        lines.append(f"{key:<25} {sign}{delta:>9.4f}{marker}")

    if comparison["improved"] is True:
        lines.append("\nResult: IMPROVED (primary metric increased)")
    elif comparison["improved"] is False:
        lines.append("\nResult: REGRESSED (primary metric decreased)")
    else:
        lines.append("\nResult: No primary metric available for comparison")

    return "\n".join(lines)


def main() -> None:
    """Run MSD retraining with model versioning."""
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    registry_dir = Path(args.registry_dir)
    now_str = datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    print("=" * 60)
    print("MSD Retraining Pipeline")
    print("=" * 60)
    print(f"   Mode: {args.mode}")
    print(f"   Model: {args.model}")
    print(f"   Registry: {registry_dir}")

    # Load previous model for incremental mode
    init_model = None
    parent_version_id: str | None = None

    if args.mode == "incremental":
        prev_id = args.previous_version or get_latest_version_id(registry_dir)
        if prev_id is None:
            print("   No previous version found; falling back to full retrain.")
            args.mode = "full"
        else:
            print(f"   Warm-start from: {prev_id}")
            prev_pipeline, _prev_version = load_model_version(registry_dir, prev_id)
            init_model = prev_pipeline.named_steps["classifier"]
            parent_version_id = prev_id

    print()

    # Load data
    print("Loading data...")
    features_df = _load_data(args)

    # Train
    print("\nTraining model...")
    result = _train_and_evaluate(features_df, args, init_model=init_model)

    # Build version metadata
    version_id = _next_version_id(registry_dir)
    train_q_range = None
    if "quarter" in features_df.columns:
        q_sorted = sorted(features_df["quarter"].unique())
        train_q_range = f"{q_sorted[0]}-{q_sorted[-1]}"

    config: dict[str, Any] = {
        "model": args.model,
        "leakage_safe": args.leakage_safe,
        "use_cv": args.use_cv,
        "cv_folds": args.cv_folds if args.use_cv else None,
        "train_end": args.train_end,
    }
    if args.model == "catboost":
        config.update({
            "cat_iterations": args.cat_iterations,
            "cat_depth": args.cat_depth,
            "cat_learning_rate": args.cat_learning_rate,
            "cat_l2": args.cat_l2,
            "cat_border_count": args.cat_border_count,
        })
        if args.mode == "incremental":
            config["incremental_iterations"] = args.incremental_iterations

    version = ModelVersion(
        version_id=version_id,
        created_at=now_str,
        model_type=args.model,
        train_quarters=train_q_range,
        n_train_samples=result["n_train_samples"],
        n_features=len(result["feature_names"]),
        feature_names=result["feature_names"],
        metrics=result["metrics"],
        config=config,
        parent_version=parent_version_id,
        retrain_mode=args.mode,
        notes=args.notes,
    )

    # Save
    print(f"\nSaving version {version_id}...")
    version_dir = save_versioned_model(result["pipeline"], version, registry_dir)
    print(f"   Saved to {version_dir}")

    # Print metrics
    print(f"\nMetrics ({version_id}):")
    for key, val in sorted(result["metrics"].items()):
        print(f"   {key}: {val}")

    # Compare against previous version
    compare_id = args.compare_to or parent_version_id or get_latest_version_id(registry_dir)
    # Don't compare against ourselves
    if compare_id and compare_id != version_id:
        try:
            _, prev_version = load_model_version(registry_dir, compare_id)
            comparison = compare_versions(version, prev_version)
            print(f"\n{_format_comparison(comparison)}")

            # Save comparison report
            comp_path = version_dir / "comparison.json"
            comp_path.write_text(json.dumps(comparison, indent=2))
        except FileNotFoundError:
            print(f"\n   Could not load {compare_id} for comparison.")

    print(f"\nDone. Version {version_id} registered.")


if __name__ == "__main__":
    main()
