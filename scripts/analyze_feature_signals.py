#!/usr/bin/env python3
"""
Feature Signal Diagnostics Orchestrator.

Loads the multisignal dataset using the MSD pipeline, selects a configurable set
of features via the feature registry, and runs multiple diagnostics:

1. L1 logistic regression coefficients (per-fold survival analysis)
2. Linear SVM (L1) coefficients
3. LightGBM & XGBoost feature importances (+ optional SHAP)
4. Univariate metrics (mutual information, AUROC vs label)
5. Stability summaries and consolidated drop recommendations

Outputs CSV/JSON/Markdown artifacts under the requested output directory.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from lightgbm import LGBMClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.utils import Bunch
from sklearn.utils.validation import check_is_fitted
import warnings

try:
    from xgboost import XGBClassifier
except ModuleNotFoundError:
    XGBClassifier = None

try:
    import shap  # type: ignore
except ModuleNotFoundError:
    shap = None

# Ensure repository root on path for imports
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.utils.feature_registry import FeatureRegistry  # noqa: E402
from scripts.multi_signal_detector import (  # noqa: E402
    load_and_merge_signals,
    construct_labels,
    engineer_features,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze feature signal quality.")

    # Data sources
    parser.add_argument("--tight-mapping", type=Path,
                        default=Path("data/out/experiments/stage0_tight_mapping/milestone_lineage_mapping_tight.csv"))
    parser.add_argument("--semantic-velocity", type=Path,
                        default=Path("data/out/experiments/stage1_quarterly_embeddings/semantic_velocity.csv"))
    parser.add_argument("--multisignal", type=Path,
                        default=Path("data/out/02_lineage_tracking/lineage_multisignal_features.csv"))
    parser.add_argument("--timeseries", type=Path,
                        default=Path("data/out/02_lineage_tracking/lineage_timeseries.csv"))
    parser.add_argument("--labels", type=Path, default=None,
                        help="Optional label CSV with lineage_id, quarter, is_inflection_onset.")

    # Feature selection
    parser.add_argument("--feature-config", type=Path,
                        default=Path("config/features/feature_groups.yaml"))
    parser.add_argument("--include-groups", nargs="+", default=None,
                        help="Feature groups to include (defaults to all_predictors).")
    parser.add_argument("--include-pattern", action="append", default=None,
                        help="Wildcard pattern(s) to include (e.g., logistic_*).")
    parser.add_argument("--exclude-groups", nargs="+", default=None,
                        help="Groups to exclude.")
    parser.add_argument("--exclude-pattern", action="append", default=None,
                        help="Wildcard pattern(s) to exclude.")

    # Execution controls
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--sample-limit", type=int, default=None,
                        help="Limit dataset to N samples (stratified) for fast analysis.")
    parser.add_argument("--tree-sample-limit", type=int, default=20000,
                        help="Sample size for heavy tree/SHAP computations.")
    parser.add_argument("--output-dir", type=Path,
                        default=Path("data/out/analysis/feature_signal_pruning"))
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--disable-shap", action="store_true")
    parser.add_argument("--n-jobs", type=int, default=1,
                        help="Parallel jobs for per-feature metrics.")

    return parser.parse_args()


def stratified_sample(df: pd.DataFrame, n_samples: int, seed: int) -> pd.DataFrame:
    if n_samples is None or n_samples >= len(df):
        return df
    positives = df[df["is_milestone"] == 1]
    negatives = df[df["is_milestone"] == 0]
    pos_ratio = len(positives) / len(df) if len(df) else 0
    n_pos = max(1, int(n_samples * pos_ratio))
    n_neg = n_samples - n_pos
    pos_sample = positives.sample(n=min(len(positives), n_pos), random_state=seed)
    neg_sample = negatives.sample(n=min(len(negatives), n_neg), random_state=seed)
    sampled = pd.concat([pos_sample, neg_sample]).sample(frac=1, random_state=seed).reset_index(drop=True)
    return sampled


def prepare_matrix(df: pd.DataFrame, feature_names: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    X = df[feature_names].replace([np.inf, -np.inf], 0).fillna(0).astype(np.float32).values
    y = df["is_milestone"].astype(int).values
    return X, y


def determine_cv_folds(y: np.ndarray, requested: int) -> int:
    positives = int(y.sum())
    if positives < 2:
        raise ValueError("Dataset must contain at least two positive samples for cross-validation.")
    effective = min(requested, positives)
    if effective < 2:
        effective = 2
    if effective != requested:
        print(f"[Diag] Adjusted CV folds from {requested} to {effective} based on {positives} positives.")
    return effective


def run_l1_logistic(X: np.ndarray, y: np.ndarray, feature_names: List[str],
                    cv_folds: int, seed: int) -> pd.DataFrame:
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    rows = []
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    for fold_idx, (train_idx, _) in enumerate(skf.split(X, y)):
        model = Pipeline([
            ("scaler", StandardScaler(with_mean=True)),
            ("clf", LogisticRegression(
                penalty="l1", solver="saga", C=1.0,
                max_iter=5000, class_weight="balanced",
                random_state=seed + fold_idx))
        ])
        model.fit(X[train_idx], y[train_idx])
        check_is_fitted(model)
        coefs = model.named_steps["clf"].coef_[0]
        rows.append(coefs)
    coef_df = pd.DataFrame(rows, columns=feature_names)
    coef_df["fold"] = range(cv_folds)
    return coef_df


def run_linear_svc(X: np.ndarray, y: np.ndarray, feature_names: List[str],
                   cv_folds: int, seed: int) -> pd.DataFrame:
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    rows = []
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    for fold_idx, (train_idx, _) in enumerate(skf.split(X, y)):
        model = Pipeline([
            ("scaler", StandardScaler(with_mean=True)),
            ("clf", LinearSVC(
                C=1.0,
                penalty="l1",
                dual=False,
                max_iter=5000,
                class_weight="balanced",
                random_state=seed + fold_idx))
        ])
        model.fit(X[train_idx], y[train_idx])
        check_is_fitted(model)
        coefs = model.named_steps["clf"].coef_[0]
        rows.append(coefs)
    coef_df = pd.DataFrame(rows, columns=feature_names)
    coef_df["fold"] = range(cv_folds)
    return coef_df


def run_lightgbm_importance(X: np.ndarray, y: np.ndarray, feature_names: List[str],
                            seed: int) -> pd.DataFrame:
    clf = LGBMClassifier(
        n_estimators=600,
        learning_rate=0.05,
        max_depth=-1,
        num_leaves=255,
        subsample=0.8,
        colsample_bytree=0.8,
        class_weight="balanced",
        random_state=seed,
        n_jobs=-1,
    )
    clf.fit(X, y)
    importances = clf.feature_importances_
    df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances
    }).sort_values("importance", ascending=False)
    return df


def run_xgboost_importance(X: np.ndarray, y: np.ndarray, feature_names: List[str],
                           seed: int) -> Optional[pd.DataFrame]:
    if XGBClassifier is None:
        return None
    clf = XGBClassifier(
        n_estimators=600,
        learning_rate=0.05,
        max_depth=8,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        tree_method="hist",
        random_state=seed,
        n_jobs=-1,
        objective="binary:logistic",
        eval_metric="logloss",
        scale_pos_weight=(len(y) - y.sum()) / max(y.sum(), 1),
        device="cuda" if os.environ.get("MSD_USE_GPU", "0") == "1" else "cpu",
    )
    clf.fit(X, y)
    importances = clf.feature_importances_
    df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances
    }).sort_values("importance", ascending=False)
    return df


def compute_univariate_metrics(X: np.ndarray, y: np.ndarray, feature_names: List[str],
                               seed: int, n_jobs: int) -> pd.DataFrame:
    mi = mutual_info_classif(X, y, discrete_features=False, random_state=seed)

    def compute_auc(idx: int) -> Tuple[str, float]:
        col = X[:, idx]
        if np.allclose(col, col[0]):
            return feature_names[idx], 0.5
        try:
            return feature_names[idx], roc_auc_score(y, col)
        except ValueError:
            return feature_names[idx], 0.5

    if n_jobs and n_jobs > 1:
        auc_results = Parallel(n_jobs=n_jobs)(
            delayed(compute_auc)(i) for i in range(X.shape[1])
        )
    else:
        auc_results = [compute_auc(i) for i in range(X.shape[1])]

    auc_scores = {name: score for name, score in auc_results}
    df = pd.DataFrame({
        "feature": feature_names,
        "mutual_information": mi,
        "single_feature_auc": [auc_scores[name] for name in feature_names],
    })
    return df


def compute_stability(rank_tables: Dict[str, pd.DataFrame], top_k: int = 25) -> Dict[str, Dict]:
    summary: Dict[str, Dict] = {}
    for name, df in rank_tables.items():
        if "feature" not in df.columns or "score" not in df.columns:
            continue
        df = df.copy()
        df["rank"] = df["score"].rank(ascending=False, method="dense")
        summary[name] = {
            "top_features": df.sort_values("rank").head(top_k)[["feature", "score", "rank"]].to_dict("records"),
            "bottom_features": df.sort_values("rank", ascending=False).head(top_k)[["feature", "score", "rank"]].to_dict("records"),
        }
    return summary


def write_markdown_report(path: Path,
                          dataset_info: Dict[str, float],
                          stability_summary: Dict[str, Dict],
                          recommendations: List[str]) -> None:
    lines = [
        "# Feature Signal Diagnostics Report",
        "",
        "## Dataset",
        f"- Samples: {dataset_info['samples']:,}",
        f"- Positives: {dataset_info['positives']:,} ({dataset_info['positive_rate']:.2%})",
        f"- Features analyzed: {dataset_info['features']}",
        "",
        "## Recommendations",
    ]
    if recommendations:
        lines.extend([f"- {rec}" for rec in recommendations])
    else:
        lines.append("- Pending deeper review.")
    lines.append("")
    lines.append("## Top Features by Method")
    for method, payload in stability_summary.items():
        lines.append(f"### {method}")
        for entry in payload.get("top_features", [])[:10]:
            lines.append(f"- {entry['feature']}: score={entry['score']:.4f}, rank={int(entry['rank'])}")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    registry = FeatureRegistry(args.feature_config)
    include_groups = args.include_groups or ["all_predictors"]
    feature_names = registry.resolve_features(
        include_groups=include_groups,
        include_patterns=args.include_pattern,
        exclude_groups=args.exclude_groups,
        exclude_patterns=args.exclude_pattern,
    )

    start_time = time.time()
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

    if "is_milestone" not in features_df.columns:
        raise RuntimeError("Dataset missing 'is_milestone' column after labeling.")

    features_df = stratified_sample(features_df, args.sample_limit, args.random_seed)
    available_features = [f for f in feature_names if f in features_df.columns]
    dropped = sorted(set(feature_names) - set(available_features))
    if dropped:
        print(f"[WARN] Dropped {len(dropped)} requested features not present in dataset.")
    X, y = prepare_matrix(features_df, available_features)

    dataset_info = {
        "samples": len(features_df),
        "positives": int(features_df["is_milestone"].sum()),
        "positive_rate": float(features_df["is_milestone"].mean()),
        "features": len(available_features),
    }
    cv_folds = determine_cv_folds(y, args.cv_folds)

    # Diagnostics
    print("[Diag] Running L1 logistic regression...")
    l1_logistic = run_l1_logistic(X, y, available_features, cv_folds, args.random_seed)
    l1_logistic.to_parquet(output_dir / "l1_logistic_coefficients.parquet", index=False)

    print("[Diag] Running L1 LinearSVC...")
    l1_svc = run_linear_svc(X, y, available_features, cv_folds, args.random_seed)
    l1_svc.to_parquet(output_dir / "l1_linearsvc_coefficients.parquet", index=False)

    tree_sample = stratified_sample(features_df, args.tree_sample_limit, args.random_seed)
    tree_X, tree_y = prepare_matrix(tree_sample, available_features)

    print("[Diag] Training LightGBM for importances...")
    lgb_importance = run_lightgbm_importance(tree_X, tree_y, available_features, args.random_seed)
    lgb_importance.to_csv(output_dir / "lightgbm_importance.csv", index=False)

    xgb_importance = None
    if XGBClassifier is not None:
        print("[Diag] Training XGBoost for importances...")
        xgb_importance = run_xgboost_importance(tree_X, tree_y, available_features, args.random_seed)
        if xgb_importance is not None:
            xgb_importance.to_csv(output_dir / "xgboost_importance.csv", index=False)
    else:
        print("[Diag] Skipping XGBoost importances (package not available).")

    if shap is not None and not args.disable_shap:
        print("[Diag] Computing SHAP values (LightGBM)...")
        shap_model = LGBMClassifier(
            n_estimators=400,
            learning_rate=0.05,
            class_weight="balanced",
            random_state=args.random_seed
        ).fit(tree_X, tree_y)
        explainer = shap.TreeExplainer(shap_model)
        shap_raw = explainer.shap_values(tree_X)
        shap_values = shap_raw[1] if isinstance(shap_raw, list) else shap_raw
        shap.summary_plot(
            shap_values,
            tree_X,
            feature_names=available_features,
            show=False,
        )
        shap_path = output_dir / "shap_summary.png"
        import matplotlib.pyplot as plt  # Imported lazily
        plt.tight_layout()
        plt.savefig(shap_path, dpi=200)
        plt.close()
        np.save(output_dir / "shap_values.npy", shap_values)
    else:
        print("[Diag] SHAP disabled or package unavailable.")

    print("[Diag] Computing univariate metrics...")
    univariate = compute_univariate_metrics(X, y, available_features, args.random_seed, args.n_jobs)
    univariate.to_csv(output_dir / "univariate_metrics.csv", index=False)

    # Stability summary
    rank_tables = {
        "L1_Logistic": pd.DataFrame({
            "feature": available_features,
            "score": l1_logistic.drop(columns=["fold"]).abs().mean(axis=0).values,
        }),
        "L1_LinearSVC": pd.DataFrame({
            "feature": available_features,
            "score": l1_svc.drop(columns=["fold"]).abs().mean(axis=0).values,
        }),
        "LightGBM": lgb_importance.rename(columns={"importance": "score"}),
    }
    if xgb_importance is not None:
        rank_tables["XGBoost"] = xgb_importance.rename(columns={"importance": "score"})
    stability_summary = compute_stability(rank_tables)

    recommendations = []
    low_signal = univariate[univariate["single_feature_auc"] < 0.55]
    if not low_signal.empty:
        recommendations.append(f"Review {len(low_signal)} low-AUC features (<0.55) for removal.")
    zero_coefs = (l1_logistic.drop(columns=["fold"]).abs().sum(axis=0) == 0)
    zero_features = zero_coefs[zero_coefs].index.tolist()
    if zero_features:
        recommendations.append(f"{len(zero_features)} features pruned by all L1 folds.")

    runtime = time.time() - start_time
    serialized_args = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    meta = {
        "dataset": dataset_info,
        "dropped_features": dropped,
        "runtime_sec": runtime,
        "arguments": serialized_args,
    }
    (output_dir / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    write_markdown_report(output_dir / "analysis_report.md", dataset_info, stability_summary, recommendations)
    report_payload = {
        "dataset": dataset_info,
        "stability": stability_summary,
        "recommendations": recommendations,
    }
    (output_dir / "analysis_report.json").write_text(json.dumps(report_payload, indent=2), encoding="utf-8")
    print(f"[Done] Diagnostics complete in {runtime/60:.2f} minutes.")


if __name__ == "__main__":
    main()
