from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from multi_signal_detector import (
    FeatureNameSafeLGBMClassifier,
    evaluate_with_cv,
    select_features,
)


def test_select_features_accepts_curated_list():
    df = pd.DataFrame(
        {
            "semantic_velocity": [0.15, 0.25],
            "growth_rate": [0.01, 0.02],
            "dummy": [1, 2],
        }
    )
    required = ["semantic_velocity", "growth_rate"]
    X, cols = select_features(df, required_features=required)
    assert cols == required
    assert list(X.columns) == required


def _feature_name_warnings(caught: list[warnings.WarningMessage]) -> list[warnings.WarningMessage]:
    return [warning for warning in caught if "valid feature names" in str(warning.message)]


def test_feature_name_safe_lgbm_classifier_suppresses_numpy_warning():
    rng = np.random.default_rng(42)
    X = rng.normal(size=(80, 4))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)

    model = FeatureNameSafeLGBMClassifier(
        n_estimators=10,
        max_depth=3,
        learning_rate=0.1,
        random_state=42,
        verbose=-1,
    )
    model.fit(X, y)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.predict(X[:8])
        model.predict_proba(X[:8])

    assert not _feature_name_warnings(caught)


def test_evaluate_with_cv_lightgbm_pipeline_is_warning_free():
    rng = np.random.default_rng(7)
    X_array = rng.normal(size=(120, 6))
    X = pd.DataFrame(X_array, columns=[f"feature_{idx}" for idx in range(X_array.shape[1])])
    y = pd.Series((X_array[:, 0] + 0.4 * X_array[:, 1] > 0).astype(int))

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        results = evaluate_with_cv(
            X,
            y,
            model_type="lightgbm",
            use_smote=False,
            cv_folds=3,
            random_state=7,
            max_depth=3,
            min_samples_leaf=5,
            n_estimators=10,
            learning_rate=0.1,
        )
        results["pipeline"].predict(X.values)
        results["pipeline"].predict_proba(X.values)

    assert not _feature_name_warnings(caught)
