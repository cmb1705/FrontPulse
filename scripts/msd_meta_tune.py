#!/usr/bin/env python3
"""
MSD Meta-Learning: Automated Hyperparameter Tuning with Optuna (Task 3.2)

Performs automated model selection and hyperparameter tuning for the Multi-Signal
Detector using Optuna. Searches across model families and hyperparameters to
optimize breakthrough detection recall with precision constraints.

Usage:
    # Quick test (10 trials)
    python scripts/msd_meta_tune.py --n-trials 10 --output-dir data/out/experiments/msd_tuning_test

    # Full tuning (100 trials)
    python scripts/msd_meta_tune.py --n-trials 100

    # Resume previous study
    python scripts/msd_meta_tune.py --n-trials 50 --study-name my_study --resume

Architecture:
    - Search space: 4 model families (LogReg, RF, GBM, LightGBM)
    - Hyperparameters: depth, n_estimators, learning_rate, regularization
    - Calibration: none, sigmoid (Platt), isotonic
    - Optimization: Maximize recall with precision >= 0.20 constraint
    - Evaluation: 5-fold stratified cross-validation per trial
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from optuna.samplers import TPESampler
# Optional pruning callback (only available if optuna-integration[pytorch_lightning] installed)
try:
    from optuna.integration import PyTorchLightningPruningCallback  # type: ignore
except ModuleNotFoundError:
    PyTorchLightningPruningCallback = None  # type: ignore
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score, precision_score, f1_score, average_precision_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
try:
    from sklearn.frozen import FrozenEstimator  # type: ignore
except ImportError:
    FrozenEstimator = None  # type: ignore

# Optional gradient boosting libraries
try:
    import xgboost as xgb  # type: ignore
    from xgboost import XGBClassifier  # type: ignore
except ModuleNotFoundError:
    xgb = None
    XGBClassifier = None

try:
    from catboost import CatBoostClassifier  # type: ignore
except ModuleNotFoundError:
    CatBoostClassifier = None

_XGB_GPU_AVAILABLE: Optional[bool] = None
_CATBOOST_GPU_AVAILABLE: Optional[bool] = None
_PREFIT_CAL_WARNING_EMITTED = False

def _xgb_supports_gpu() -> bool:
    global _XGB_GPU_AVAILABLE
    if _XGB_GPU_AVAILABLE is not None:
        return _XGB_GPU_AVAILABLE
    if XGBClassifier is None:
        _XGB_GPU_AVAILABLE = False
        return False
    try:
        test_model = XGBClassifier(
            n_estimators=1,
            max_depth=1,
            learning_rate=0.1,
            tree_method='hist',
            device='cuda',
            use_label_encoder=False,
            verbosity=0,
        )
        test_model.fit(np.array([[0.0]]), np.array([0.0]))
        _XGB_GPU_AVAILABLE = True
    except Exception:
        _XGB_GPU_AVAILABLE = False
    return _XGB_GPU_AVAILABLE

def _catboost_supports_gpu() -> bool:
    global _CATBOOST_GPU_AVAILABLE
    if _CATBOOST_GPU_AVAILABLE is not None:
        return _CATBOOST_GPU_AVAILABLE
    if CatBoostClassifier is None:
        _CATBOOST_GPU_AVAILABLE = False
        return False
    try:
        test_model = CatBoostClassifier(
            iterations=1,
            depth=1,
            learning_rate=0.1,
            task_type='GPU',
            verbose=False,
        )
        test_model.fit(np.array([[0.0]]), np.array([0.0]))
        _CATBOOST_GPU_AVAILABLE = True
    except Exception:
        _CATBOOST_GPU_AVAILABLE = False
    return _CATBOOST_GPU_AVAILABLE

# Supported model registry (extended dynamically if optional deps available)
MODEL_TYPES_ALL = ['logistic', 'random_forest', 'gradient_boosting', 'lightgbm']
if XGBClassifier is not None:
    MODEL_TYPES_ALL.append('xgboost')
if CatBoostClassifier is not None:
    MODEL_TYPES_ALL.append('catboost')

MODEL_TYPES = MODEL_TYPES_ALL.copy()


def _wrap_for_prefit_calibration(estimator: Any) -> tuple[Any, Dict[str, Any]]:
    """
    Prepare arguments for CalibratedClassifierCV when the estimator is already fit.

    Returns:
        frozen_estimator (or original estimator) and extra keyword arguments.
    """
    global _PREFIT_CAL_WARNING_EMITTED
    if FrozenEstimator is not None:
        frozen = FrozenEstimator(estimator)
        if hasattr(estimator, "_estimator_type"):
            setattr(frozen, "_estimator_type", getattr(estimator, "_estimator_type"))
        return frozen, {}

    if not _PREFIT_CAL_WARNING_EMITTED:
        warnings.warn(
            "sklearn>=1.6 deprecates cv='prefit'. Consider upgrading scikit-learn "
            "to use FrozenEstimator for calibration to silence this warning.",
            FutureWarning,
            stacklevel=3,
        )
        _PREFIT_CAL_WARNING_EMITTED = True
    return estimator, {'cv': 'prefit'}


def _parse_lag_max_arg(value):
    if value is None:
        return None
    if isinstance(value, int):
        return value
    value_str = str(value).strip().lower()
    if value_str in {"none", "null", "inf", "infinite"}:
        return None
    return int(value_str)

# Import from multi_signal_detector
sys.path.insert(0, str(Path(__file__).parent))
from multi_signal_detector import (
    load_and_merge_signals,
    construct_labels,
    engineer_features,
    select_features,
    summarize_detection_lag,
)


def create_model_pipeline(
    trial: optuna.Trial,
    use_smote: bool = True,
    model_n_jobs: int = 1,
) -> tuple[ImbPipeline, Dict[str, Any]]:
    """
    Create model pipeline based on Optuna trial suggestions.

    Returns:
        pipeline: Configured sklearn/imblearn pipeline
        config: Dictionary of hyperparameters for logging
    """
    # Model family selection
    model_type = trial.suggest_categorical('model_type', MODEL_TYPES)

    # Calibration selection
    calibrate = False
    calibration_method = None

    config = {
        'model_type': model_type,
        'calibrate': calibrate,
        'calibration_method': calibration_method,
        'use_smote': use_smote,
    }

    # Build pipeline steps
    steps = [('scaler', StandardScaler())]

    # SMOTE (optional)
    if use_smote:
        steps.append(('smote', SMOTE(random_state=42, k_neighbors=5)))

    # Model-specific hyperparameters
    if model_type == 'logistic':
        C = trial.suggest_float('logistic_C', 1e-4, 1e2, log=True)
        model = LogisticRegression(
            C=C,
            class_weight='balanced',
            max_iter=1000,
            random_state=42
        )
        config['C'] = C

    elif model_type == 'random_forest':
        n_estimators = trial.suggest_int('rf_n_estimators', 50, 300)
        max_depth = trial.suggest_int('rf_max_depth', 3, 15)
        min_samples_leaf = trial.suggest_int('rf_min_samples_leaf', 1, 20)
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            class_weight='balanced',
            random_state=42,
            n_jobs=model_n_jobs
        )
        config.update({
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'min_samples_leaf': min_samples_leaf,
        })

    elif model_type == 'gradient_boosting':
        n_estimators = trial.suggest_int('gbm_n_estimators', 50, 300)
        max_depth = trial.suggest_int('gbm_max_depth', 3, 10)
        learning_rate = trial.suggest_float('gbm_learning_rate', 0.01, 0.3, log=True)
        model = GradientBoostingClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=42
        )
        config.update({
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'learning_rate': learning_rate,
        })

    elif model_type == 'lightgbm':
        # Original broad search space for LightGBM (explore wide, let PR-AUC decide).
        n_estimators = trial.suggest_int('lgb_n_estimators', 200, 1200)
        max_depth = trial.suggest_int('lgb_max_depth', 4, 18)
        learning_rate = trial.suggest_float('lgb_learning_rate', 0.003, 0.2, log=True)
        min_child_samples = trial.suggest_int('lgb_min_child_samples', 5, 60)
        num_leaves = trial.suggest_int('lgb_num_leaves', 31, 512)
        subsample = trial.suggest_float('lgb_subsample', 0.6, 1.0)
        colsample_bytree = trial.suggest_float('lgb_colsample_bytree', 0.6, 1.0)
        model = lgb.LGBMClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            min_child_samples=min_child_samples,
            num_leaves=num_leaves,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            class_weight='balanced',
            random_state=42,
            n_jobs=model_n_jobs,
            device='gpu',
            gpu_platform_id=0,
            gpu_device_id=0,
            tree_learner='data',
            verbose=-1,
        )
        config.update({
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'learning_rate': learning_rate,
            'min_child_samples': min_child_samples,
            'num_leaves': num_leaves,
            'subsample': subsample,
            'colsample_bytree': colsample_bytree,
        })

    elif model_type == 'xgboost':
        if XGBClassifier is None:
            raise RuntimeError("xgboost is not installed. Please `pip install xgboost` to enable this option.")
        n_estimators = trial.suggest_int('xgb_n_estimators', 200, 1200)
        max_depth = trial.suggest_int('xgb_max_depth', 3, 16)
        learning_rate = trial.suggest_float('xgb_learning_rate', 0.003, 0.3, log=True)
        subsample = trial.suggest_float('xgb_subsample', 0.5, 1.0)
        colsample_bytree = trial.suggest_float('xgb_colsample_bytree', 0.5, 1.0)
        min_child_weight = trial.suggest_float('xgb_min_child_weight', 1e-2, 10, log=True)
        reg_lambda = trial.suggest_float('xgb_reg_lambda', 1e-3, 100, log=True)
        # Prefer GPU if available
        use_gpu = _xgb_supports_gpu()
        tree_method = 'hist'
        extra_params: Dict[str, Any] = {}
        if use_gpu:
            extra_params['device'] = 'cuda'
        model = XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            min_child_weight=min_child_weight,
            reg_lambda=reg_lambda,
            objective='binary:logistic',
            eval_metric='logloss',
            random_state=42,
            tree_method=tree_method,
            n_jobs=model_n_jobs,
            use_label_encoder=False,
            **extra_params,
        )
        config.update({
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'learning_rate': learning_rate,
            'subsample': subsample,
            'colsample_bytree': colsample_bytree,
            'min_child_weight': min_child_weight,
            'reg_lambda': reg_lambda,
            'tree_method': tree_method,
            'device': extra_params.get('device', 'cpu'),
        })

    elif model_type == 'catboost':
        if CatBoostClassifier is None:
            raise RuntimeError("catboost is not installed. Please `pip install catboost` to enable this option.")
        iterations = trial.suggest_int('cat_iterations', 500, 2000)
        depth = trial.suggest_int('cat_depth', 5, 7)
        learning_rate = trial.suggest_float('cat_learning_rate', 0.01, 0.05, log=True)
        l2_leaf_reg = trial.suggest_float('cat_l2_leaf_reg', 1.0, 50.0, log=True)
        border_count = trial.suggest_int('cat_border_count', 32, 128)
        task_type = 'GPU' if _catboost_supports_gpu() else 'CPU'
        model = CatBoostClassifier(
            iterations=iterations,
            depth=depth,
            learning_rate=learning_rate,
            l2_leaf_reg=l2_leaf_reg,
            border_count=border_count,
            random_seed=42,
            verbose=False,
            loss_function='Logloss',
            eval_metric='Logloss',
            thread_count=model_n_jobs,
            task_type=task_type,
        )
        config.update({
            'iterations': iterations,
            'depth': depth,
            'learning_rate': learning_rate,
            'l2_leaf_reg': l2_leaf_reg,
            'border_count': border_count,
            'task_type': task_type,
        })

    steps.append(('classifier', model))

    # Create pipeline
    pipeline = ImbPipeline(steps)

    # Note: Calibration applied post-pipeline fit (see objective function)

    return pipeline, config


def objective(
    trial: optuna.Trial,
    X: pd.DataFrame,
    y: pd.Series,
    metadata: Optional[pd.DataFrame] = None,
    cv_folds: int = 5,
    min_precision: float = 0.20,
    use_smote: bool = True,
    model_n_jobs: int = 1,
    use_pruning: bool = False,
) -> float:
    """
    Optuna objective function: maximize recall with precision >= min_precision.

    Returns:
        Negative recall (for minimization) if precision < threshold, else recall.
    """
    pipeline, config = create_model_pipeline(
        trial,
        use_smote=use_smote,
        model_n_jobs=model_n_jobs,
    )

    # Cross-validation
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    pruning_callback = None
    if use_pruning and PyTorchLightningPruningCallback is not None:
        pruning_callback = PyTorchLightningPruningCallback(trial, monitor='val_recall')

    recalls = []
    precisions = []
    f1_scores = []
    pr_aucs = []

    fold_predictions: list[pd.DataFrame] = []

    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X, y)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        # Train pipeline
        pipeline.fit(X_train, y_train)

        # Calibration (if requested)
        if config['calibrate']:
            from sklearn.calibration import CalibratedClassifierCV
            calibrator_estimator, calibrator_kwargs = _wrap_for_prefit_calibration(pipeline)
            calibrator = CalibratedClassifierCV(
                calibrator_estimator,
                method=config['calibration_method'],
                **calibrator_kwargs,
            )
            calibrator.fit(X_val, y_val)
            y_pred = calibrator.predict(X_val)
            y_prob = calibrator.predict_proba(X_val)[:, 1]
        else:
            if config['model_type'] == 'xgboost' and config.get('device') == 'cuda' and xgb is not None:
                booster = pipeline.named_steps['classifier']
                # Use model’s predict_proba (XGBoost will handle device internally)
                y_prob = booster.predict_proba(X_val)[:, 1]
                y_pred = (y_prob >= 0.5).astype(int)
            else:
                y_pred = pipeline.predict(X_val)
                y_prob = pipeline.predict_proba(X_val)[:, 1]

        # Metrics
        recall = recall_score(y_val, y_pred, zero_division=0)
        precision = precision_score(y_val, y_pred, zero_division=0)
        f1 = f1_score(y_val, y_pred, zero_division=0)
        pr_auc = average_precision_score(y_val, y_prob) if y_val.sum() > 0 else 0.0

        if metadata is not None:
            fold_meta = metadata.iloc[val_idx].reset_index(drop=True)
            fold_pred_df = pd.DataFrame({
                'lineage_id': fold_meta['lineage_id'],
                'quarter': fold_meta['quarter'],
                'is_milestone_true': y_val.values,
                'is_milestone_pred': y_pred,
                'inflection_probability': y_prob,
            })
            fold_predictions.append(fold_pred_df)

        recalls.append(recall)
        precisions.append(precision)
        f1_scores.append(f1)
        pr_aucs.append(pr_auc)

        if use_pruning:
            current_mean_recall = float(np.mean(recalls))
            trial.report(current_mean_recall, fold_idx)
            if trial.should_prune():
                raise optuna.TrialPruned()

    # Aggregate metrics
    mean_recall = np.mean(recalls)
    mean_precision = np.mean(precisions)
    mean_f1 = np.mean(f1_scores)
    mean_pr_auc = np.mean(pr_aucs)

    # Aggregate lag metrics (optional)
    lag_summary: Dict[str, float] = {}
    if fold_predictions:
        concatenated = pd.concat(fold_predictions, ignore_index=True)
        lag_summary = summarize_detection_lag(concatenated)
    else:
        lag_summary = {
            "detection_lag_count": 0.0,
            "detection_lag_coverage": 0.0,
        }

    # Log metrics
    trial.set_user_attr('recall', mean_recall)
    trial.set_user_attr('precision', mean_precision)
    trial.set_user_attr('f1', mean_f1)
    trial.set_user_attr('pr_auc', mean_pr_auc)
    for key, value in lag_summary.items():
        try:
            trial.set_user_attr(key, float(value))
        except (TypeError, ValueError):
            # Fallback for non-numeric entries (e.g., None for quarters)
            trial.set_user_attr(key, value)

    # Objective: maximize PR-AUC, with a soft guard on precision.
    if mean_precision < min_precision:
        # If below the precision floor, return a small negative score to deprioritize.
        return -1.0 * (min_precision - mean_precision)
    return mean_pr_auc


def run_tuning(
    X: pd.DataFrame,
    y: pd.Series,
    metadata: Optional[pd.DataFrame],
    n_trials: int = 50,
    cv_folds: int = 5,
    min_precision: float = 0.20,
    use_smote: bool = True,
    study_name: str = 'msd_meta_tuning',
    storage: str = None,
    resume: bool = False,
    optuna_n_jobs: int = 1,
    model_n_jobs: int = 1,
    use_pruning: bool = False,
) -> optuna.Study:
    """
    Run Optuna hyperparameter tuning.

    Args:
        X: Feature matrix
        y: Target labels
        n_trials: Number of Optuna trials
        cv_folds: Number of CV folds
        min_precision: Minimum acceptable precision
        use_smote: Whether to use SMOTE
        study_name: Name for Optuna study
        storage: Optuna storage URL (for persistence)
        resume: Whether to resume existing study

    Returns:
        Completed Optuna study
    """
    # Create or load study
    if resume and storage:
        study = optuna.load_study(study_name=study_name, storage=storage)
        print(f"Resumed study '{study_name}' from {storage}")
        print(f"Previous trials: {len(study.trials)}")
    else:
        sampler = TPESampler(seed=42)
        study = optuna.create_study(
            study_name=study_name,
            direction='maximize',
            sampler=sampler,
            storage=storage,
            load_if_exists=resume
        )
        print(f"Created new study: {study_name}")

    print(f"\n" + "="*70)
    print(f"MSD Meta-Learning: Optuna Hyperparameter Tuning")
    print(f"="*70)
    print(f"Trials: {n_trials}")
    print(f"CV folds: {cv_folds}")
    print(f"Min precision: {min_precision:.2f}")
    print(f"Use SMOTE: {use_smote}")
    print(f"Dataset: {len(X)} samples, {y.sum()} positive ({y.sum()/len(y)*100:.1f}%)")
    print(f"="*70 + "\n")

    # Run optimization
    study.optimize(
        lambda trial: objective(
            trial, X, y, metadata,
            cv_folds=cv_folds,
            min_precision=min_precision,
            use_smote=use_smote,
            model_n_jobs=model_n_jobs,
        ),
        n_trials=n_trials,
        n_jobs=optuna_n_jobs,
        show_progress_bar=True,
    )

    return study


def save_study_results(
    study: optuna.Study,
    output_dir: Path,
    feature_names: list,
) -> None:
    """
    Save study results to disk for reproducibility.

    Outputs:
        - best_config.json: Best hyperparameters
        - study_results.json: All trials with metrics
        - optimization_history.json: Trial-by-trial progress
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Best trial
    best_trial = study.best_trial
    best_config = {
        'trial_number': best_trial.number,
        'value': best_trial.value,
        'params': best_trial.params,
        'user_attrs': best_trial.user_attrs,
        'datetime_complete': best_trial.datetime_complete.isoformat() if best_trial.datetime_complete else None,
    }

    with open(output_dir / 'best_config.json', 'w') as f:
        json.dump(best_config, f, indent=2)
    print(f"\nSaved best config to {output_dir / 'best_config.json'}")

    # All trials
    trials_data = []
    for trial in study.trials:
        trial_data = {
            'number': trial.number,
            'value': trial.value,
            'params': trial.params,
            'user_attrs': trial.user_attrs,
            'state': trial.state.name,
            'datetime_start': trial.datetime_start.isoformat() if trial.datetime_start else None,
            'datetime_complete': trial.datetime_complete.isoformat() if trial.datetime_complete else None,
            'duration': (trial.datetime_complete - trial.datetime_start).total_seconds() if (trial.datetime_complete and trial.datetime_start) else None,
        }
        trials_data.append(trial_data)

    with open(output_dir / 'study_results.json', 'w') as f:
        json.dump({'trials': trials_data, 'n_trials': len(trials_data)}, f, indent=2)
    print(f"Saved all trials to {output_dir / 'study_results.json'}")

    # Optimization history
    history = {
        'best_values': [study.trials[i].value for i in range(len(study.trials))],
        'best_params': [study.trials[i].params for i in range(len(study.trials))],
    }

    with open(output_dir / 'optimization_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    print(f"Saved optimization history to {output_dir / 'optimization_history.json'}")

    # Feature names
    with open(output_dir / 'feature_names.json', 'w') as f:
        json.dump(feature_names, f, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="MSD Meta-Learning: Automated Hyperparameter Tuning"
    )

    # Input paths (same as multi_signal_detector.py)
    parser.add_argument(
        '--tight-mapping',
        default='data/out/experiments/stage0_tight_mapping/milestone_lineage_mapping_tight.csv'
    )
    parser.add_argument(
        '--semantic-velocity',
        default='data/out/experiments/stage1_quarterly_embeddings/semantic_velocity.csv'
    )
    parser.add_argument(
        '--multisignal',
        default='data/out/02_lineage_tracking/lineage_multisignal_features.csv'
    )
    parser.add_argument(
        '--timeseries',
        default='data/out/02_lineage_tracking/lineage_timeseries.csv'
    )
    parser.add_argument(
        '--labels',
        default='data/out/02_lineage_tracking/inflection_labels.csv',
        help='Label CSV with columns lineage_id, quarter, is_inflection_onset.'
    )

    # Tuning configuration
    parser.add_argument('--n-trials', type=int, default=50,
                       help='Number of Optuna trials (default: 50)')
    parser.add_argument('--cv-folds', type=int, default=5,
                       help='Number of CV folds (default: 5)')
    parser.add_argument('--min-precision', type=float, default=0.05,
                       help='Minimum acceptable precision (default: 0.05)')
    parser.add_argument('--no-smote', action='store_true',
                       help='Disable SMOTE oversampling')
    parser.add_argument('--optuna-n-jobs', type=int, default=4,
                       help='Number of parallel Optuna workers (default: 4)')
    parser.add_argument('--estimator-n-jobs', type=int, default=16,
                       help='Threads per estimator (default: 16)')
    parser.add_argument('--prune', action='store_true',
                       help='Enable Optuna pruning (reports mean recall per fold)')
    parser.add_argument('--n-jobs', type=int, default=4,
                       help='Parallel Optuna trial workers (default: 4)')
    parser.add_argument('--model-n-jobs', type=int, default=16,
                       help='Threads per estimator (default: 16)')

    # Optuna study configuration
    parser.add_argument('--study-name', default='msd_meta_tuning',
                       help='Optuna study name (default: msd_meta_tuning)')
    parser.add_argument('--storage', default=None,
                       help='Optuna storage URL (e.g., sqlite:///optuna.db)')
    parser.add_argument('--resume', action='store_true',
                       help='Resume previous study')

    # Output configuration
    parser.add_argument('--output-dir',
                       default='data/out/experiments/msd_meta_tuning',
                       help='Output directory for results')

    # Detection window (same as MSD)
    parser.add_argument('--lag-min', type=int, default=2)
    parser.add_argument('--lag-max', default="8",
                       help="Maximum lag (quarters) after milestone; use 'none' for no cap.")
    parser.add_argument('--model-types', type=str, default=None,
                       help="Comma-separated list of model types to search (e.g., 'catboost,lightgbm').")

    args = parser.parse_args()
    args.lag_max = _parse_lag_max_arg(args.lag_max)
    if args.model_types:
        args.model_types = [m.strip() for m in args.model_types.split(',') if m.strip()]
    else:
        args.model_types = None

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.storage and args.storage.startswith("sqlite:///"):
        storage_path = Path(args.storage.replace("sqlite:///", ""))
        storage_path.parent.mkdir(parents=True, exist_ok=True)

    global MODEL_TYPES
    if args.model_types:
        MODEL_TYPES = args.model_types
    else:
        MODEL_TYPES = MODEL_TYPES_ALL.copy()

    print("="*70)
    print("MSD Meta-Learning: Automated Model Selection & Tuning")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"   Trials: {args.n_trials}")
    print(f"   CV folds: {args.cv_folds}")
    print(f"   Min precision: {args.min_precision:.2f}")
    print(f"   SMOTE: {not args.no_smote}")
    print(f"   Optuna workers: {args.optuna_n_jobs}")
    print(f"   Model threads: {args.estimator_n_jobs}")
    if args.prune and PyTorchLightningPruningCallback is None:
        print("WARNING: optuna-integration[pytorch_lightning] not installed; pruning disabled.")
        args.prune = False
    print(f"   Pruning enabled: {args.prune}")
    print(f"   Study name: {args.study_name}")
    print(f"   Output dir: {output_dir}")
    print()

    # Load and prepare data (same as MSD)
    tight_mapping_path = Path(args.tight_mapping)
    semantic_velocity_path = Path(args.semantic_velocity)
    multisignal_path = Path(args.multisignal)
    timeseries_path = Path(args.timeseries)

    # Verify inputs exist
    for path in [tight_mapping_path, semantic_velocity_path, multisignal_path, timeseries_path]:
        if not path.exists():
            print(f"ERROR: Input file not found: {path}")
            return

    labels_path = Path(args.labels) if args.labels else None

    # Load data
    features_df, labels_df, tight_mapping = load_and_merge_signals(
        tight_mapping_path,
        semantic_velocity_path,
        multisignal_path,
        timeseries_path,
        labels_path=labels_path,
        n_samples=None
    )

    # Construct labels
    features_df = construct_labels(features_df, labels_df, tight_mapping, lag_min=args.lag_min, lag_max=args.lag_max)

    # Engineer features
    features_df = engineer_features(features_df)
    features_df = features_df.reset_index(drop=True)

    # Select features
    X, feature_names = select_features(features_df)
    metadata = features_df[['lineage_id', 'quarter']].reset_index(drop=True)
    y = features_df['is_milestone'].reset_index(drop=True)

    print(f"\n   Dataset: {len(X)} samples, {len(feature_names)} features")
    print(f"   Positive class: {y.sum()} ({y.sum()/len(y)*100:.2f}%)")

    # Run tuning
    study = run_tuning(
        X, y, metadata,
        n_trials=args.n_trials,
        cv_folds=args.cv_folds,
        min_precision=args.min_precision,
        use_smote=not args.no_smote,
        study_name=args.study_name,
        storage=args.storage,
        resume=args.resume,
        optuna_n_jobs=args.optuna_n_jobs,
        model_n_jobs=args.estimator_n_jobs,
        use_pruning=args.prune,
    )

    # Print results
    print("\n" + "="*70)
    print("TUNING COMPLETE")
    print("="*70)

    best_trial = study.best_trial
    print(f"\nBest Trial: #{best_trial.number}")
    print(f"   Recall:    {best_trial.user_attrs['recall']:.3f}")
    print(f"   Precision: {best_trial.user_attrs['precision']:.3f}")
    print(f"   F1:        {best_trial.user_attrs['f1']:.3f}")
    print(f"   PR-AUC:    {best_trial.user_attrs['pr_auc']:.3f}")

    print(f"\nBest Hyperparameters:")
    for key, value in best_trial.params.items():
        print(f"   {key}: {value}")

    # Save results
    save_study_results(study, output_dir, feature_names)

    print(f"\n" + "="*70)
    print(f"Results saved to: {output_dir}")
    print(f"="*70)
    print(f"\nNext steps:")
    print(f"1. Review best config: {output_dir / 'best_config.json'}")
    print(f"2. Retrain MSD with best config: python scripts/multi_signal_detector.py \\")
    print(f"      --model {best_trial.params['model_type']} \\")
    if 'rf_n_estimators' in best_trial.params:
        print(f"      --n-estimators {best_trial.params['rf_n_estimators']} \\")
    if 'gbm_n_estimators' in best_trial.params:
        print(f"      --n-estimators {best_trial.params['gbm_n_estimators']} \\")
    if 'lgb_n_estimators' in best_trial.params:
        print(f"      --n-estimators {best_trial.params['lgb_n_estimators']} \\")
    if 'gbm_learning_rate' in best_trial.params:
        print(f"      --learning-rate {best_trial.params['gbm_learning_rate']:.6f} \\")
    if 'lgb_learning_rate' in best_trial.params:
        print(f"      --learning-rate {best_trial.params['lgb_learning_rate']:.6f} \\")
    if 'rf_max_depth' in best_trial.params:
        print(f"      --max-depth {best_trial.params['rf_max_depth']} \\")
        print(f"      --min-samples-leaf {best_trial.params['rf_min_samples_leaf']} \\")
    if 'gbm_max_depth' in best_trial.params:
        print(f"      --max-depth {best_trial.params['gbm_max_depth']} \\")
    if 'lgb_max_depth' in best_trial.params:
        print(f"      --max-depth {best_trial.params['lgb_max_depth']} \\")
    if 'lgb_min_child_samples' in best_trial.params:
        print(f"      --min-samples-leaf {best_trial.params['lgb_min_child_samples']} \\")
    if best_trial.params.get('calibrate'):
        print(f"      --calibrate --calibration-method {best_trial.params['calibration_method']} \\")
    print(f"      --use-cv --cv-folds {args.cv_folds}")


if __name__ == '__main__':
    main()
