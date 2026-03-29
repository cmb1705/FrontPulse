"""Optuna hyperparameter optimization for the Multi-Signal Detector.

Searches across model types (gradient_boosting, lightgbm, catboost, random_forest)
and their respective hyperparameters using Bayesian optimization (TPE sampler).
Optimizes for CV PR-AUC, which measures ranking quality independent of threshold.

Usage:
    python scripts/optuna_msd_search.py --n-trials 50 --output-dir data/out/experiments/optuna_search
"""
from __future__ import annotations

import argparse
import json
import logging
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import optuna
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

# Suppress convergence and future warnings during optimization
warnings.filterwarnings("ignore", category=FutureWarning)
optuna.logging.set_verbosity(optuna.logging.WARNING)

from _path_bootstrap import ensure_repo_imports  # noqa: E402

REPO = ensure_repo_imports()

from src.domain_registry import add_domain_args, resolve_script_paths  # noqa: E402

LOG = logging.getLogger("optuna_msd")

# ---------------------------------------------------------------------------
# Data loading (reuses MSD patterns)
# ---------------------------------------------------------------------------

def load_data(
    labels_path: str,
    multisignal_path: str,
    timeseries_path: str,
    semantic_velocity_path: str,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Load and merge all data sources, returning features and labels."""
    # Load labels
    labels_df = pd.read_csv(labels_path)
    labels_df["lineage_id"] = labels_df["lineage_id"].astype(int)
    labels_df["quarter"] = labels_df["quarter"].astype(str)

    # Load features
    features_df = pd.read_csv(multisignal_path)
    features_df["lineage_id"] = features_df["lineage_id"].astype(int)
    features_df["quarter"] = features_df["quarter"].astype(str)

    # Load timeseries for growth features
    ts_df = pd.read_csv(timeseries_path)
    ts_df["lineage_id"] = ts_df["lineage_id"].astype(int)
    ts_df["quarter"] = ts_df["quarter"].astype(str)
    ts_group = ts_df.groupby("lineage_id")["new_works"]
    ts_df["total_works"] = ts_group.cumsum()
    ts_df["growth_rate"] = ts_group.pct_change().fillna(0).replace(
        [np.inf, -np.inf], 0,
    )

    # Merge timeseries columns into features
    ts_cols = ["lineage_id", "quarter", "total_works", "growth_rate"]
    features_df = features_df.merge(
        ts_df[ts_cols], on=["lineage_id", "quarter"], how="left", suffixes=("", "_ts"),
    )
    for col in ["total_works", "growth_rate"]:
        if col + "_ts" in features_df.columns:
            features_df[col] = features_df[col].fillna(features_df[col + "_ts"])
            features_df.drop(columns=[col + "_ts"], inplace=True)

    # Load semantic velocity
    sv_path = Path(semantic_velocity_path)
    if sv_path.exists():
        sv_df = pd.read_csv(sv_path)
        sv_df["lineage_id"] = sv_df["lineage_id"].astype(int)
        sv_df["quarter"] = sv_df["quarter"].astype(str)
        features_df = features_df.merge(
            sv_df[["lineage_id", "quarter", "semantic_velocity"]],
            on=["lineage_id", "quarter"], how="left",
        )
        features_df["semantic_velocity"] = features_df["semantic_velocity"].fillna(0)

    # Merge labels
    features_df = features_df.merge(
        labels_df[["lineage_id", "quarter", "is_inflection_onset"]],
        on=["lineage_id", "quarter"], how="left",
    )
    features_df["is_inflection_onset"] = (
        features_df["is_inflection_onset"].fillna(0).astype(int)
    )

    # Engineer derived features
    features_df["velocity_acceleration"] = (
        features_df.get("semantic_velocity", pd.Series(0, index=features_df.index))
        .diff().fillna(0)
    )
    features_df["growth_acceleration"] = (
        features_df["growth_rate"].diff().fillna(0)
    )
    features_df["novelty_momentum"] = (
        features_df.get("novelty_rate", pd.Series(0, index=features_df.index))
        .diff().fillna(0)
    )
    if "awakening_intensity" in features_df.columns:
        features_df["is_awakening"] = (
            features_df["awakening_intensity"] > 0
        ).astype(int)
    if "cross_domain_refs" in features_df.columns and "within_lineage_refs" in features_df.columns:
        total_refs = features_df["cross_domain_refs"] + features_df["within_lineage_refs"]
        features_df["citation_balance"] = np.where(
            total_refs > 0,
            features_df["cross_domain_refs"] / total_refs,
            0.5,
        )

    y = features_df["is_inflection_onset"].values

    return features_df, y


# Leakage-safe feature lists (matches MSD script logic)
LEAKAGE_UNSAFE = {
    "cd_index", "cd_min", "cd_max", "n_papers_cd", "disruption_intensity",
    "logistic_carrying_capacity", "logistic_growth_rate", "logistic_midpoint_idx",
    "logistic_midpoint_quarter", "logistic_fit_r2",
}

FIELD_PREFIXES = ("field_", "relative_", "growth_vs_field", "acceleration_vs_field",
                  "new_works_over_p75")

CORE_FEATURES = [
    "semantic_velocity", "velocity_acceleration", "growth_rate", "growth_acceleration",
    "new_works", "total_works", "novel_terms", "novelty_rate", "novelty_momentum",
    "dormancy_length", "awakening_intensity", "is_awakening",
    "cross_domain_share", "cross_domain_refs", "within_lineage_refs", "citation_balance",
]

CONTEXT_PREFIXES = (
    "author_influx_", "citation_velocity_", "reference_vitality_",
    "topic_diversity_", "cross_cluster_bridging_",
)


def select_leakage_safe_features(df: pd.DataFrame) -> list[str]:
    """Select leakage-safe features matching MSD --leakage-safe logic."""
    available = set(df.columns)
    features = [f for f in CORE_FEATURES if f in available]

    # Add context features (trailing windows only)
    for col in df.columns:
        if any(col.startswith(prefix) for prefix in CONTEXT_PREFIXES) and col not in LEAKAGE_UNSAFE:
            features.append(col)

    # Exclude field and leakage-unsafe features
    features = [
        f for f in features
        if f not in LEAKAGE_UNSAFE
        and not any(f.startswith(p) for p in FIELD_PREFIXES)
    ]

    return sorted(set(features))


def build_model(
    model_type: str, params: dict[str, Any],
) -> Any:
    """Build a sklearn-compatible model from type and params."""
    if model_type == "gradient_boosting":
        return GradientBoostingClassifier(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            min_samples_leaf=params["min_samples_leaf"],
            learning_rate=params["learning_rate"],
            subsample=params.get("subsample", 1.0),
            random_state=42,
        )
    elif model_type == "random_forest":
        return RandomForestClassifier(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            min_samples_leaf=params["min_samples_leaf"],
            max_features=params.get("max_features", "sqrt"),
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )
    elif model_type == "lightgbm":
        import lightgbm as lgb
        return lgb.LGBMClassifier(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            min_child_samples=params["min_samples_leaf"],
            learning_rate=params["learning_rate"],
            subsample=params.get("subsample", 1.0),
            colsample_bytree=params.get("colsample_bytree", 1.0),
            reg_alpha=params.get("reg_alpha", 0.0),
            reg_lambda=params.get("reg_lambda", 0.0),
            is_unbalance=True,
            random_state=42,
            verbose=-1,
            n_jobs=-1,
        )
    elif model_type == "catboost":
        from catboost import CatBoostClassifier
        return CatBoostClassifier(
            iterations=params["n_estimators"],
            depth=min(params["max_depth"], 16),  # CatBoost max depth is 16
            learning_rate=params["learning_rate"],
            l2_leaf_reg=params.get("l2_leaf_reg", 3.0),
            border_count=params.get("border_count", 128),
            auto_class_weights="Balanced",
            random_state=42,
            verbose=0,
            thread_count=-1,
        )
    elif model_type == "xgboost":
        import xgboost as xgb
        return xgb.XGBClassifier(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            min_child_weight=params.get("min_child_weight", 1),
            learning_rate=params["learning_rate"],
            subsample=params.get("subsample", 1.0),
            colsample_bytree=params.get("colsample_bytree", 1.0),
            reg_alpha=params.get("reg_alpha", 0.0),
            reg_lambda=params.get("reg_lambda", 1.0),
            scale_pos_weight=params.get("scale_pos_weight", 1.0),
            random_state=42,
            verbosity=0,
            n_jobs=-1,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def cv_evaluate(
    model: Any, X: np.ndarray, y: np.ndarray, n_folds: int = 5,
) -> dict[str, float]:
    """Run stratified CV and return mean metrics."""
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    roc_aucs = []
    pr_aucs = []

    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # SMOTE on training fold
        try:
            from imblearn.over_sampling import SMOTE
            smote = SMOTE(random_state=42)
            X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
        except ImportError:
            X_train_res, y_train_res = X_train, y_train

        model_clone = model.__class__(**model.get_params())
        model_clone.fit(X_train_res, y_train_res)

        if hasattr(model_clone, "predict_proba"):
            y_prob = model_clone.predict_proba(X_test)[:, 1]
        else:
            y_prob = model_clone.decision_function(X_test)

        roc_aucs.append(roc_auc_score(y_test, y_prob))
        pr_aucs.append(average_precision_score(y_test, y_prob))

    return {
        "cv_roc_auc_mean": float(np.mean(roc_aucs)),
        "cv_roc_auc_std": float(np.std(roc_aucs)),
        "cv_pr_auc_mean": float(np.mean(pr_aucs)),
        "cv_pr_auc_std": float(np.std(pr_aucs)),
    }


# ---------------------------------------------------------------------------
# Optuna objective
# ---------------------------------------------------------------------------

def create_objective(
    X: np.ndarray, y: np.ndarray, n_folds: int = 5,
):
    """Create an Optuna objective function that optimizes CV PR-AUC."""

    def objective(trial: optuna.Trial) -> float:
        # Choose model type
        model_type = trial.suggest_categorical(
            "model_type",
            ["gradient_boosting", "lightgbm", "catboost", "xgboost", "random_forest"],
        )

        # Common hyperparameters
        n_estimators = trial.suggest_int("n_estimators", 50, 800)
        max_depth = trial.suggest_int("max_depth", 2, 8)
        min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 50)

        params: dict[str, Any] = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "min_samples_leaf": min_samples_leaf,
        }

        # Model-specific hyperparameters
        if model_type in ("gradient_boosting", "lightgbm", "catboost", "xgboost"):
            params["learning_rate"] = trial.suggest_float(
                "learning_rate", 0.005, 0.3, log=True,
            )

        if model_type in ("gradient_boosting", "lightgbm", "xgboost"):
            params["subsample"] = trial.suggest_float("subsample", 0.5, 1.0)

        if model_type in ("lightgbm", "xgboost"):
            params["colsample_bytree"] = trial.suggest_float(
                "colsample_bytree", 0.3, 1.0,
            )
            params["reg_alpha"] = trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True)
            params["reg_lambda"] = trial.suggest_float(
                "reg_lambda", 1e-8, 10.0, log=True,
            )

        if model_type == "catboost":
            params["l2_leaf_reg"] = trial.suggest_float(
                "l2_leaf_reg", 1e-2, 10.0, log=True,
            )
            params["border_count"] = trial.suggest_int("border_count", 32, 255)

        if model_type == "xgboost":
            # Scale pos weight to handle imbalance (alternative to SMOTE)
            params["scale_pos_weight"] = trial.suggest_float(
                "scale_pos_weight", 1.0, 100.0, log=True,
            )
            params["min_child_weight"] = trial.suggest_int("min_child_weight", 1, 20)

        if model_type == "random_forest":
            params["max_features"] = trial.suggest_categorical(
                "max_features", ["sqrt", "log2", 0.3, 0.5, 0.7],
            )

        try:
            model = build_model(model_type, params)
            metrics = cv_evaluate(model, X, y, n_folds=n_folds)
            return metrics["cv_pr_auc_mean"]
        except Exception as e:
            LOG.warning("Trial failed: %s", e)
            return 0.0

    return objective


def main() -> None:
    """Run Optuna hyperparameter optimization."""
    parser = argparse.ArgumentParser(
        description="Optuna hyperparameter search for MSD onset detection",
    )
    parser.add_argument(
        "--labels", default=None,
    )
    parser.add_argument(
        "--multisignal",
        default=None,
    )
    parser.add_argument(
        "--timeseries",
        default=None,
    )
    parser.add_argument(
        "--semantic-velocity",
        default=None,
    )
    parser.add_argument("--n-trials", type=int, default=50)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument(
        "--output-dir", default=None,
    )
    add_domain_args(parser)
    args = parser.parse_args()

    paths = resolve_script_paths(args, REPO)
    if args.labels is None:
        args.labels = str(paths.lineage_tracking / "onset_labels_msd.csv") if paths else "data/out/02_lineage_tracking/onset_labels_msd.csv"
    if args.multisignal is None:
        args.multisignal = str(paths.lineage_tracking / "lineage_multisignal_features.csv") if paths else "data/out/02_lineage_tracking/lineage_multisignal_features.csv"
    if args.timeseries is None:
        args.timeseries = str(paths.lineage_tracking / "lineage_timeseries.csv") if paths else "data/out/02_lineage_tracking/lineage_timeseries.csv"
    if args.semantic_velocity is None:
        args.semantic_velocity = str(paths.experiments / "stage1_quarterly_embeddings" / "semantic_velocity.csv") if paths else "data/out/experiments/stage1_quarterly_embeddings/semantic_velocity.csv"
    if args.output_dir is None:
        args.output_dir = str(paths.experiments / "optuna_search") if paths else "data/out/experiments/optuna_search"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    # Load data
    LOG.info("Loading data...")
    features_df, y = load_data(
        args.labels, args.multisignal, args.timeseries, args.semantic_velocity,
    )

    # Select leakage-safe features
    feature_cols = select_leakage_safe_features(features_df)
    LOG.info("Selected %d leakage-safe features", len(feature_cols))

    X = features_df[feature_cols].fillna(0).values
    LOG.info("Dataset: %d samples, %d features, %d positives (%.2f%%)",
             X.shape[0], X.shape[1], y.sum(), 100 * y.mean())

    # Create Optuna study
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10),
        study_name="msd_onset_hpo",
    )

    objective = create_objective(X, y, n_folds=args.cv_folds)

    LOG.info("Starting Optuna search (%d trials)...", args.n_trials)
    study.optimize(objective, n_trials=args.n_trials, show_progress_bar=True)

    # Results
    best = study.best_trial
    LOG.info("Best trial: #%d", best.number)
    LOG.info("Best PR-AUC: %.4f", best.value)
    LOG.info("Best params: %s", json.dumps(best.params, indent=2))

    # Print top 10 trials
    print("\n" + "=" * 80)
    print("TOP 10 TRIALS (by CV PR-AUC)")
    print("=" * 80)
    trials_sorted = sorted(study.trials, key=lambda t: t.value or 0, reverse=True)
    header = "{:<6} {:<20} {:>8} {:>8} {:>6} {:>6} {:>8}".format(
        "Trial", "Model", "PR-AUC", "depth", "leaf", "n_est", "lr")
    print(header)
    print("-" * 80)
    for t in trials_sorted[:10]:
        if t.value is not None and t.value > 0:
            p = t.params
            lr_str = "{:.4f}".format(p.get("learning_rate", 0))
            row = "{:<6} {:<20} {:>8.4f} {:>8} {:>6} {:>6} {:>8}".format(
                t.number, p["model_type"], t.value, p["max_depth"],
                p["min_samples_leaf"], p["n_estimators"], lr_str)
            print(row)

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Best trial details
    best_result = {
        "best_trial_number": best.number,
        "best_pr_auc": best.value,
        "best_params": best.params,
        "n_trials": args.n_trials,
        "n_features": len(feature_cols),
        "n_samples": int(X.shape[0]),
        "n_positives": int(y.sum()),
        "feature_names": feature_cols,
    }
    (output_dir / "best_trial.json").write_text(
        json.dumps(best_result, indent=2, default=str),
    )

    # Full trial history
    trial_records = []
    for t in study.trials:
        if t.value is not None:
            record = {"trial": t.number, "pr_auc": t.value}
            record.update(t.params)
            trial_records.append(record)
    pd.DataFrame(trial_records).to_csv(
        output_dir / "trial_history.csv", index=False,
    )

    # Run final evaluation of best model with full metrics
    LOG.info("Running final evaluation of best model...")
    best_model = build_model(best.params["model_type"], best.params)
    final_metrics = cv_evaluate(best_model, X, y, n_folds=args.cv_folds)
    final_metrics.update(best.params)
    (output_dir / "final_evaluation.json").write_text(
        json.dumps(final_metrics, indent=2, default=str),
    )

    print("\nFinal best model evaluation:")
    print("  Model:   {}".format(best.params["model_type"]))
    print("  ROC-AUC: {:.4f} +/- {:.4f}".format(
        final_metrics["cv_roc_auc_mean"], final_metrics["cv_roc_auc_std"]))
    print("  PR-AUC:  {:.4f} +/- {:.4f}".format(
        final_metrics["cv_pr_auc_mean"], final_metrics["cv_pr_auc_std"]))
    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
