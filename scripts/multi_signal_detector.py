#!/usr/bin/env python3
"""
Multi-Signal Detector (MSD): Supervised Ensemble Breakthrough Detector

Trains a supervised model to detect milestone-associated lineage-quarters
using the multi-signal feature stack derived from Stage 1.

SCOPE:
- Input: Tight mapping labels + multi-signal features
- Output: Trained model + predictions + evaluation metrics
- Goal: Improve recall beyond Stage 0 baseline (6.8%)

ARCHITECTURE:
- Binary classification (is_milestone_quarter: 0/1)
- Features: semantic_velocity, growth, novelty, dormancy, citations, CD index
- Models: Logistic Regression, Random Forest, XGBoost
- Class imbalance handling: SMOTE + class weights
- Evaluation: Precision, Recall, F1, ROC-AUC, PR-AUC

SOFTWARE DEVELOPMENT WORKFLOW:
1. Define scope and objectives ✓
2. Draft prototype code (this script)
3. Smoke test with synthetic data
4. Micro-benchmark and debug
5. Optimize
6. Scale test on medium data slice
7. Full production run
8. Post-run validation
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import Any

from _path_bootstrap import ensure_repo_imports

_REPO = ensure_repo_imports()

import lightgbm as lgb  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402

from src.domain_registry import (  # noqa: E402
    add_domain_args,
    apply_domain_path_defaults,
    resolve_script_paths,
)

try:
    from catboost import CatBoostClassifier  # type: ignore
except ModuleNotFoundError:
    CatBoostClassifier = None  # type: ignore
from imblearn.over_sampling import SMOTE  # noqa: E402
from imblearn.pipeline import Pipeline as ImbPipeline  # noqa: E402
from sklearn.base import clone  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    average_precision_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

try:
    from sklearn.frozen import FrozenEstimator  # type: ignore
except ImportError:
    FrozenEstimator = None  # type: ignore

from persistence_utils import ensure_persistence_column  # noqa: E402
from utils.quarter_utils import (  # noqa: E402
    describe_quarter_range,
    filter_by_quarter,
    int_to_quarter,
    quarter_to_int,
    snapshot_dataset,
)

from src.run_provenance import collect_run_provenance, save_run_provenance  # noqa: E402
from src.trusted_io import save_trusted_pickle  # noqa: E402

_PREFIT_CALIBRATION_WARNING_EMITTED = False


def _wrap_prefit_calibrator(estimator: Any) -> tuple[Any, dict[str, Any]]:
    """Return estimator/kwargs combo suitable for calibrating a pre-fit model."""
    global _PREFIT_CALIBRATION_WARNING_EMITTED
    if FrozenEstimator is not None:
        frozen = FrozenEstimator(estimator)
        if hasattr(estimator, "_estimator_type"):
            frozen._estimator_type = estimator._estimator_type
        return frozen, {}

    if not _PREFIT_CALIBRATION_WARNING_EMITTED:
        warnings.warn(
            "sklearn>=1.6 deprecates CalibratedClassifierCV(cv='prefit'). "
            "Install a version with sklearn.frozen to silence this warning.",
            FutureWarning,
            stacklevel=3,
        )
        _PREFIT_CALIBRATION_WARNING_EMITTED = True
    return estimator, {'cv': 'prefit'}


def _parse_lag_max_arg(value: str | None) -> int | None:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    value_str = value.strip().lower()
    if value_str in {"none", "null", "inf", "infinite"}:
        return None
    return int(value_str)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def save_fold_diagnostics(
    cv_results: dict,
    output_dir: Path,
) -> Path:
    """Save per-fold diagnostics, PR curve, and calibration data to JSON.

    Args:
        cv_results: Dict from evaluate_with_cv containing fold_diagnostics,
            oof_y_true, oof_y_prob, and per-fold metric arrays.
        output_dir: Experiment output directory.

    Returns:
        Path to saved diagnostics file.
    """
    from sklearn.calibration import calibration_curve
    from sklearn.metrics import precision_recall_curve

    output_dir.mkdir(parents=True, exist_ok=True)

    diagnostics: dict[str, Any] = {}

    # Per-fold confusion matrices
    diagnostics["per_fold"] = cv_results.get("fold_diagnostics", [])

    # Per-fold metric arrays (for paired statistical tests)
    for metric in ("precision", "recall", "f1", "roc_auc", "average_precision", "mcc", "f2"):
        key = f"test_{metric}"
        if key in cv_results:
            diagnostics[f"fold_{metric}"] = cv_results[key].tolist()

    # Out-of-fold PR curve
    oof_y_true = cv_results.get("oof_y_true", [])
    oof_y_prob = cv_results.get("oof_y_prob", [])
    if oof_y_true and oof_y_prob:
        pr_precision, pr_recall, pr_thresholds = precision_recall_curve(
            oof_y_true, oof_y_prob
        )
        diagnostics["pr_curve"] = {
            "precision": pr_precision.tolist(),
            "recall": pr_recall.tolist(),
            "thresholds": pr_thresholds.tolist(),
        }

        # Calibration curve (reliability diagram data)
        try:
            prob_true, prob_pred = calibration_curve(
                oof_y_true, oof_y_prob, n_bins=10, strategy="uniform"
            )
            diagnostics["calibration"] = {
                "prob_true": prob_true.tolist(),
                "prob_pred": prob_pred.tolist(),
                "n_bins": 10,
            }
        except ValueError:
            pass  # Too few samples for calibration curve

    out_path = output_dir / "fold_diagnostics.json"
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(diagnostics, fh, indent=2, default=str)
    print(f"   Saved fold diagnostics to {out_path}")
    return out_path


def compute_confusion_stats(
    pred_df: pd.DataFrame,
    pred_col: str = "is_milestone_pred",
    true_col: str = "is_milestone_true",
    prefix: str = "",
) -> dict[str, float]:
    """Compute TP/FP/FN/TN confusion stats from prediction columns.

    Args:
        pred_df: DataFrame with prediction and truth columns.
        pred_col: Column name for binary predictions.
        true_col: Column name for ground truth labels.
        prefix: Key prefix for output dict (e.g. ``"persistent_"``).

    Returns:
        Dict of confusion metrics with optional prefix.
    """
    tp = int(((pred_df[pred_col] == 1) & (pred_df[true_col] == 1)).sum())
    fp = int(((pred_df[pred_col] == 1) & (pred_df[true_col] == 0)).sum())
    fn = int(((pred_df[pred_col] == 0) & (pred_df[true_col] == 1)).sum())
    tn = int(((pred_df[pred_col] == 0) & (pred_df[true_col] == 0)).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    return {
        f"{prefix}tp_threshold": tp,
        f"{prefix}fp_threshold": fp,
        f"{prefix}fn_threshold": fn,
        f"{prefix}tn_threshold": tn,
        f"{prefix}precision_threshold": precision,
        f"{prefix}recall_threshold": recall,
        f"{prefix}f1_threshold": f1,
        f"{prefix}fpr_threshold": fpr,
    }


def summarize_detection_lag(pred_df: pd.DataFrame) -> dict[str, float]:
    positives = pred_df[pred_df['is_milestone_true'] == 1]
    if positives.empty:
        return {
            "detection_lag_count": 0,
            "detection_lag_coverage": 0.0,
        }

    def sorted_quarters(series: pd.Series) -> list[str]:
        return sorted(series.tolist(), key=quarter_to_int)

    actual_map = positives.groupby('lineage_id')['quarter'].apply(sorted_quarters).to_dict()
    predicted_map = (
        pred_df[pred_df['is_milestone_pred'] == 1]
        .groupby('lineage_id')['quarter']
        .apply(sorted_quarters)
        .to_dict()
    )

    lags: list[int] = []
    for lineage_id, actual_quarters in actual_map.items():
        preds = predicted_map.get(lineage_id)
        if not preds:
            continue
        lag = quarter_to_int(preds[0]) - quarter_to_int(actual_quarters[0])
        lags.append(lag)

    coverage = len(lags) / len(actual_map) if actual_map else 0.0
    if not lags:
        return {
            "detection_lag_count": 0,
            "detection_lag_coverage": coverage,
        }

    arr = np.array(lags)
    return {
        "detection_lag_count": len(lags),
        "detection_lag_coverage": coverage,
        "detection_lag_median": float(np.median(arr)),
        "detection_lag_mean": float(np.mean(arr)),
        "detection_lag_std": float(np.std(arr)),
        "detection_lag_share_le_0": float((arr <= 0).mean()),
        "detection_lag_share_le_2": float((arr <= 2).mean()),
    }


def add_detection_lag_column(pred_df: pd.DataFrame) -> pd.DataFrame:
    """
    Annotate predictions with per-lineage detection lag (first positive detection vs. first true inflection).
    """
    df = pred_df.copy()
    df["quarter_index"] = df["quarter"].apply(quarter_to_int)

    detection_lag = np.full(len(df), np.nan)

    true_first = (
        df[df["is_inflection_true"] == 1]
        .groupby("lineage_id")["quarter_index"]
        .min()
    )
    pred_first = (
        df[df["is_inflection_pred"] == 1]
        .groupby("lineage_id")["quarter_index"]
        .min()
    )

    for lineage_id, pred_quarter in pred_first.items():
        true_quarter = true_first.get(lineage_id)
        if pd.isna(true_quarter):
            continue
        lag_value = pred_quarter - true_quarter
        mask = (
            (df["lineage_id"] == lineage_id)
            & (df["is_inflection_pred"] == 1)
            & (df["quarter_index"] == pred_quarter)
        )
        detection_lag[mask] = lag_value

    df["detection_lag_quarters"] = detection_lag
    df = df.drop(columns=["quarter_index"])
    return df


def resolve_input_path(preferred: Path, legacy_tokens: list[tuple[str, str]]) -> Path:
    """
    Resolve preferred path, falling back to legacy replacements when needed.
    """
    if preferred.exists():
        return preferred

    for old_token, legacy_token in legacy_tokens:
        legacy_path = Path(str(preferred).replace(old_token, legacy_token))
        if legacy_path.exists():
            print(f"[WARN] Input file not found: {preferred}. Using legacy path {legacy_path}.")
            return legacy_path
    return preferred


# ---------------------------------------------------------------------------
# 1. Data Integration
# ---------------------------------------------------------------------------


def load_and_merge_signals(
    tight_mapping_path: Path,
    semantic_velocity_path: Path,
    multisignal_features_path: Path,
    lineage_timeseries_path: Path,
    labels_path: Path | None = None,
    n_samples: int = None,  # noqa: ARG001
) -> tuple[pd.DataFrame, pd.DataFrame | None, pd.DataFrame]:
    """
    Load and merge all signal sources into unified feature matrix.

    Returns DataFrame with columns:
        - lineage_id, quarter
        - is_milestone (label: 0/1)
        - semantic_velocity, growth_rate, novel_terms, novelty_rate,
          dormancy_length, awakening_intensity, cross_domain_share,
          cd_index, ...
    """
    print("[1/8] Loading and merging signal data...")

    # Load tight mapping (labels)
    tight_mapping_path = resolve_input_path(
        tight_mapping_path,
        [('stage0_', 'phase0_')]
    )
    if not tight_mapping_path.exists():
        raise FileNotFoundError(f"Tight mapping file not found: {tight_mapping_path}")

    tight_mapping = pd.read_csv(tight_mapping_path)
    print(f"   Loaded tight mapping: {len(tight_mapping)} milestone-lineage pairs")
    print(f"   Milestones: {tight_mapping['event_id'].nunique()}")
    print(f"   Lineages: {tight_mapping['lineage_id'].nunique()}")

    # Load semantic velocity (Stage 1)
    semantic_velocity_path = resolve_input_path(
        semantic_velocity_path,
        [('stage1_', 'phase1_')]
    )
    if not semantic_velocity_path.exists():
        raise FileNotFoundError(f"Semantic velocity file not found: {semantic_velocity_path}")

    semantic_velocity = pd.read_csv(semantic_velocity_path)
    print(f"   Loaded semantic velocity: {len(semantic_velocity)} lineage-quarters")

    # Load multi-signal features (Codex script)
    multisignal = pd.read_csv(multisignal_features_path)
    print(f"   Loaded multi-signal features: {len(multisignal)} lineage-quarters")
    print(f"   Feature columns: {list(multisignal.columns)}")

    # Load lineage timeseries (growth metrics)
    timeseries = pd.read_csv(lineage_timeseries_path)
    print(f"   Loaded lineage timeseries: {len(timeseries)} lineage-quarters")

    # Merge semantic velocity
    merged = multisignal.copy()
    merged = merged.merge(
        semantic_velocity[['lineage_id', 'quarter', 'semantic_velocity']],
        on=['lineage_id', 'quarter'],
        how='left'
    )

    # Compute total_works (cumulative sum of new_works per lineage)
    merged = merged.sort_values(['lineage_id', 'quarter'])
    merged['total_works'] = merged.groupby('lineage_id')['new_works'].cumsum()

    # Compute growth_rate (quarter-over-quarter change in total_works)
    merged['growth_rate'] = merged.groupby('lineage_id')['total_works'].pct_change()
    merged['growth_rate'] = merged['growth_rate'].fillna(0)

    print("   Computed total_works and growth_rate from new_works")

    print(f"   Merged dataset: {len(merged)} lineage-quarters")
    print(f"   Feature columns: {len(merged.columns)}")

    # Note: Sampling moved to after label construction for stratification

    labels_df = None
    if labels_path is not None and labels_path.exists():
        labels_df = pd.read_csv(labels_path)
        labels_df['lineage_id'] = labels_df['lineage_id'].astype(int)
        labels_df['quarter'] = labels_df['quarter'].astype(str)
        print(f"   Loaded labels: {len(labels_df)} rows")

    return merged, labels_df, tight_mapping


def construct_labels(
    features_df: pd.DataFrame,
    labels_df: pd.DataFrame | None,
    tight_mapping: pd.DataFrame,
    lag_min: int = 2,
    lag_max: int | None = 8
) -> pd.DataFrame:
    """
    Construct binary labels for each lineage-quarter.

    Label = 1 if (lineage_id, quarter) is within the fixed lag window after event
    Label = 0 otherwise

    Uses fixed lag window [lag_min, lag_max] quarters after event_quarter.
    Filters to rank==1 to ensure one lineage per milestone (no duplicates).

    Args:
        lag_min: Minimum lag (in quarters) after event_quarter to start detection.
                 Default 2 to avoid forecasting (detect impact 2+ quarters after event).
        lag_max: Maximum lag (in quarters) after event_quarter to end detection.
                 Default 8 (2 years). Set to None for no upper bound.
    """
    if labels_df is not None:
        print("[2/8] Constructing labels from onset/inflection file...")
        df = labels_df.copy()
        # Accept both new (is_onset) and legacy (is_inflection_onset) column names
        label_col: str | None = None
        for candidate in ("is_onset", "is_inflection_onset"):
            if candidate in df.columns:
                label_col = candidate
                break
        if label_col is None:
            raise ValueError(
                "Label file must contain 'is_onset' or 'is_inflection_onset' column."
            )
        required = {'lineage_id', 'quarter', label_col}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Label file missing columns: {missing}")
        df['lineage_id'] = df['lineage_id'].astype(int)
        df['quarter'] = df['quarter'].astype(str)
        features_df = features_df.merge(
            df[['lineage_id', 'quarter', label_col]],
            on=['lineage_id', 'quarter'],
            how='left'
        )
        features_df[label_col] = features_df[label_col].fillna(0).astype(int)
        n_pos = int(features_df[label_col].sum())
        pct = features_df[label_col].mean() * 100
        print(f"   Onset positives: {n_pos} ({pct:.2f}%)")
        features_df.rename(columns={label_col: 'is_milestone'}, inplace=True)
        return features_df

    print("[2/8] Constructing labels...")

    # Filter to rank==1 to get one lineage per milestone
    tight_mapping_rank1 = tight_mapping[tight_mapping['rank'] == 1].copy()
    print(f"   Filtered to rank==1: {len(tight_mapping_rank1)} milestone-lineage pairs")
    print(f"   Milestones: {tight_mapping_rank1['event_id'].nunique()}")
    print(f"   Lineages: {tight_mapping_rank1['lineage_id'].nunique()}")

    # Generate positive examples using actual detection windows
    positive_examples = set()

    def add_quarters(quarter: str, n: int):
        """Add n quarters to a quarter string."""
        year, q = int(quarter[:4]), int(quarter[5])
        year += (q + n - 1) // 4
        q = ((q + n - 1) % 4) + 1
        return f"{year}Q{q}"

    def generate_quarters(start_quarter: str, end_quarter: str):
        """Generate all quarters between start and end (inclusive)."""
        # Parse quarters
        start_year, start_q = int(start_quarter[:4]), int(start_quarter[5])
        end_year, end_q = int(end_quarter[:4]), int(end_quarter[5])

        # Generate quarters
        quarters = []
        current_year, current_q = start_year, start_q

        while (current_year < end_year) or (current_year == end_year and current_q <= end_q):
            quarters.append(f"{current_year}Q{current_q}")
            current_q += 1
            if current_q > 4:
                current_q = 1
                current_year += 1

        return quarters

    max_quarter_int = features_df["quarter"].apply(quarter_to_int).max()
    max_quarter_str = int_to_quarter(max_quarter_int)

    for _, row in tight_mapping_rank1.iterrows():
        lineage_id = row['lineage_id']
        event_quarter = row['event_quarter']

        # Fixed lag window: [event + lag_min, event + lag_max]
        # Detect IMPACT (2-8 quarters after event), not PREDICT event
        window_start = add_quarters(event_quarter, lag_min)
        window_end = add_quarters(event_quarter, lag_max) if lag_max is not None else max_quarter_str

        # Generate quarters in fixed detection window
        quarters = generate_quarters(window_start, window_end)

        for quarter in quarters:
            positive_examples.add((lineage_id, quarter))

    print(f"   Generated {len(positive_examples)} positive examples (lineage-quarter pairs)")
    if lag_max is None:
        print(f"   Using lag window >= {lag_min}Q after event (no upper bound)")
    else:
        print(f"   Using fixed lag window [{lag_min}Q, {lag_max}Q] after event (detect impact, not predict event)")

    # Label the feature matrix
    features_df['is_milestone'] = features_df.apply(
        lambda row: 1 if (row['lineage_id'], row['quarter']) in positive_examples else 0,
        axis=1
    )

    n_positive = features_df['is_milestone'].sum()
    n_negative = len(features_df) - n_positive

    print(f"   Positive examples: {n_positive} ({n_positive/len(features_df)*100:.2f}%)")
    print(f"   Negative examples: {n_negative} ({n_negative/len(features_df)*100:.2f}%)")
    print(f"   Class imbalance ratio: 1:{n_negative/max(n_positive, 1):.1f}")

    return features_df


# ---------------------------------------------------------------------------
# 2. Feature Engineering
# ---------------------------------------------------------------------------


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer additional features from raw signals.

    - Temporal windows (rolling means, changes)
    - Feature interactions
    - Normalized features
    """
    print("[3/8] Engineering features...")

    df = df.sort_values(['lineage_id', 'quarter'])

    # Velocity acceleration (change in semantic velocity)
    df['velocity_acceleration'] = df.groupby('lineage_id')['semantic_velocity'].diff()

    # Growth acceleration
    df['growth_acceleration'] = df.groupby('lineage_id')['growth_rate'].diff()

    # Novelty momentum (product of novelty rate and new papers)
    df['novelty_momentum'] = df['novelty_rate'] * df['n_new_papers']

    # Awakening flag (dormancy followed by activity)
    df['is_awakening'] = (df['awakening_intensity'] > 0).astype(int)

    # Citation balance (cross-domain vs within-lineage)
    df['citation_balance'] = df['cross_domain_refs'] / (df['within_lineage_refs'] + 1)

    # Disruption intensity (CD index * new papers)
    df['disruption_intensity'] = df['cd_index'] * df['n_papers_cd']

    # Fill NaN values with 0 (for first quarters)
    df = df.fillna(0)

    print(f"   Engineered features: {len(df.columns)} total columns")

    return df


def select_features(
    df: pd.DataFrame,
    required_features: list[str] | None = None,
    field_feature_mode: str = "auto",
    leakage_safe: bool = False,
) -> tuple[pd.DataFrame, list[str]]:
    """Select final feature set for training.

    Automatically includes context features if available.
    Context features are from Task 2.1 (field-normalized metrics).

    Args:
        df: Feature DataFrame.
        required_features: If provided, use exactly these features.
        field_feature_mode: "auto", "off", or "only" for field features.
        leakage_safe: If True, exclude features that use future data
            (logistic fits, CD disruption index, field-relative metrics).
            See ``docs/implementation/leakage_audit.md`` for details.
    """
    if required_features is not None:
        missing = [feat for feat in required_features if feat not in df.columns]
        if missing:
            raise ValueError(
                f"Required features missing from DataFrame: {missing[:5]}{'...' if len(missing) > 5 else ''}"
            )
        X = df[required_features].copy()
        X = X.replace([np.inf, -np.inf], 0)
        return X, required_features

    print("[4/8] Selecting features...")

    if field_feature_mode not in {"auto", "off", "only"}:
        raise ValueError(f"Unknown field_feature_mode: {field_feature_mode}")

    # Features that require future data (leakage-prone).
    # cd_index/cd_min/cd_max use future citations; disruption_intensity
    # is derived from cd_index.  See docs/implementation/leakage_audit.md.
    _LEAKAGE_PRONE_CORE = {
        'cd_index', 'cd_min', 'cd_max', 'disruption_intensity',
    }

    # Core features
    core_features = [
        # Semantic
        'semantic_velocity',
        'velocity_acceleration',
        # Growth
        'growth_rate',
        'growth_acceleration',
        'new_works',
        'total_works',
        # Novelty
        'novel_terms',
        'novelty_rate',
        'novelty_momentum',
        # Dormancy/Awakening
        'dormancy_length',
        'awakening_intensity',
        'is_awakening',
        # Cross-domain citations
        'cross_domain_share',
        'cross_domain_refs',
        'within_lineage_refs',
        'citation_balance',
        # Disruption (excluded in leakage-safe mode)
        'cd_index',
        'cd_min',
        'cd_max',
        'disruption_intensity',
    ]

    if leakage_safe:
        core_features = [f for f in core_features if f not in _LEAKAGE_PRONE_CORE]
        print("   [LEAKAGE-SAFE] Excluded CD/disruption features from core")
        if field_feature_mode != "off":
            print("   [LEAKAGE-SAFE] Forcing field_feature_mode='off' "
                  "(field baselines computed on full corpus)")
            field_feature_mode = "off"

    feature_columns: list[str] = []
    if field_feature_mode != "only":
        feature_columns.extend(core_features)

    # Context features (35 features from Task 2.1)
    # Automatically include if present (backward compatible)
    context_feature_patterns = []
    for metric in ['author_influx', 'citation_velocity', 'reference_vitality',
                   'topic_diversity', 'cross_cluster_bridging']:
        for suffix in ['_z', '_qoq_delta', '_roll_1q', '_roll_2q', '_roll_4q',
                      '_max_dev_4q', '_min_dev_4q']:
            context_feature_patterns.append(f"{metric}{suffix}")

    # Check which context features are available
    available_context = [col for col in context_feature_patterns if col in df.columns]

    context_included: list[str] = []
    if field_feature_mode != "only":
        if available_context:
            print(f"   Found {len(available_context)} context features (Task 2.1 integration)")
            feature_columns.extend(available_context)
            context_included = available_context
        else:
            print("   No context features found (baseline mode)")

    # Convergence features (14 features from cross-front detection)
    # Automatically include if present (backward compatible)
    available_convergence = sorted([col for col in df.columns if col.startswith("conv_")])

    convergence_included: list[str] = []
    if field_feature_mode != "only":
        if available_convergence:
            print(f"   Found {len(available_convergence)} convergence features")
            feature_columns.extend(available_convergence)
            convergence_included = available_convergence
        else:
            print("   No convergence features found (single-lineage mode)")

    # Field-relative features
    field_feature_candidates = [
        'relative_new_works',
        'relative_cumulative_works',
        'growth_vs_field',
        'acceleration_vs_field',
        'new_works_over_p75',
    ]
    field_feature_candidates.extend(sorted([col for col in df.columns if col.startswith("field_")]))
    # De-duplicate while preserving order
    seen = set()
    field_feature_candidates = [c for c in field_feature_candidates if not (c in seen or seen.add(c))]
    available_field_features = [col for col in field_feature_candidates if col in df.columns]

    field_included: list[str] = []
    if field_feature_mode == "off":
        pass
    else:
        if not available_field_features and field_feature_mode == "only":
            raise ValueError("Field feature mode 'only' selected but no field_* columns found in dataset.")
        if available_field_features:
            if field_feature_mode == "only":
                feature_columns = available_field_features.copy()
            else:
                feature_columns.extend(available_field_features)
            field_included = available_field_features
            print(f"   Field-relative features enabled ({len(field_included)} columns)")
        else:
            print("   No field-relative features found.")
            if field_feature_mode == "only":
                raise ValueError("Field feature mode 'only' selected but field features are missing.")

    # Filter to available columns
    available_features = [col for col in feature_columns if col in df.columns]

    print(f"   Selected {len(available_features)} features total:")
    if field_feature_mode != "only":
        core_count = len([c for c in available_features if c in core_features])
        print(f"      Core features: {core_count}")
    if context_included:
        print(f"      Context features: {len(context_included)}")
    if convergence_included:
        print(f"      Convergence features: {len(convergence_included)}")
    if field_included:
        print(f"      Field-relative features: {len(field_included)}")

    # Show first 10 features as sample
    print("   Sample features:")
    for feat in available_features[:10]:
        print(f"      - {feat}")
    if len(available_features) > 10:
        print(f"      ... and {len(available_features) - 10} more")

    X = df[available_features].copy()

    # Check for infinite values
    X = X.replace([np.inf, -np.inf], 0)

    return X, available_features


# ---------------------------------------------------------------------------
# 3. Model Training
# ---------------------------------------------------------------------------


def train_models(
    X: pd.DataFrame,
    y: pd.Series,
    model_type: str = 'random_forest',
    use_smote: bool = True,
    calibrate: bool = False,
    calibration_method: str = 'sigmoid',
    test_size: float = 0.2,
    random_state: int = 42,
    max_depth: int = 10,
    min_samples_leaf: int = 1,
    n_estimators: int = 100,
    learning_rate: float = 0.1,
    cat_l2: float = 1.0,
    cat_border_count: int = 128,
    cat_thread_count: int = -1,
    cat_task_type: str = 'CPU',
    external_test_data: tuple[pd.DataFrame, pd.Series] | None = None
) -> dict:
    """
    Train ensemble models with class imbalance handling and optional calibration.

    Args:
        calibrate: If True, calibrate probabilities using CalibratedClassifierCV.
                   Recommended to fix poor probability calibration (ROC-AUC 0.74 but PR-AUC 0.09).
        calibration_method: 'sigmoid' (Platt scaling) or 'isotonic'. Default 'sigmoid'.

    Args:
        n_estimators: Number of trees/boosting iterations for ensemble models.
        learning_rate: Boosting shrinkage factor (GBM/LightGBM only).

    Returns dict with:
        - model: trained model
        - scaler: fitted StandardScaler
        - X_train, X_test, y_train, y_test
        - predictions, probabilities
    """
    print(f"[5/8] Training {model_type} model...")

    def _to_matrix(data: pd.DataFrame | np.ndarray) -> np.ndarray:
        return data.values if isinstance(data, pd.DataFrame) else data

    if external_test_data is not None:
        X_train_df, y_train = X, y
        X_test_df, y_test = external_test_data
        X_train_matrix = _to_matrix(X_train_df)
        X_test_matrix = _to_matrix(X_test_df)
        print(f"   Train: {len(X_train_df)} samples ({y_train.sum()} positive) [external holdout mode]")
        print(f"   Holdout: {len(X_test_df)} samples ({y_test.sum()} positive)")
    else:
        # Split data (stratified)
        from sklearn.model_selection import train_test_split

        X_train_df, X_test_df, y_train, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=random_state
        )

        X_train_matrix = _to_matrix(X_train_df)
        X_test_matrix = _to_matrix(X_test_df)

        print(f"   Train: {len(X_train_df)} samples ({y_train.sum()} positive)")
        print(f"   Test: {len(X_test_df)} samples ({y_test.sum()} positive)")

    # Build pipeline
    steps = []

    # Scaling
    scaler = StandardScaler()
    steps.append(('scaler', scaler))

    # SMOTE (if enabled)
    if use_smote and y_train.sum() > 5:  # Need at least 6 positive samples
        smote = SMOTE(random_state=random_state, k_neighbors=min(5, y_train.sum() - 1))
        steps.append(('smote', smote))
        print("   Using SMOTE for oversampling")

    # Model
    if model_type == 'logistic':
        model = LogisticRegression(
            class_weight='balanced',
            max_iter=1000,
            random_state=random_state
        )
    elif model_type == 'random_forest':
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            class_weight='balanced',
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
            n_jobs=-1
        )
        print(
            "   Random Forest regularization: "
            f"n_estimators={n_estimators}, max_depth={max_depth}, min_samples_leaf={min_samples_leaf}"
        )
    elif model_type == 'gradient_boosting':
        # GBM doesn't support class_weight, use scale_pos_weight via sample_weight
        model = GradientBoostingClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,  # Use same max_depth as RF
            learning_rate=learning_rate,
            random_state=random_state
        )
        print(
            "   Gradient Boosting regularization: "
            f"n_estimators={n_estimators}, max_depth={max_depth}, learning_rate={learning_rate}"
        )
    elif model_type == 'lightgbm':
        model = lgb.LGBMClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            num_leaves=2**max_depth - 1,  # Common pattern: full binary tree
            min_child_samples=min_samples_leaf,  # Equivalent to min_samples_leaf
            class_weight='balanced',
            learning_rate=learning_rate,
            random_state=random_state,
            n_jobs=-1,
            verbose=-1  # Suppress output
        )
        print(
            "   LightGBM regularization: "
            f"n_estimators={n_estimators}, max_depth={max_depth}, "
            f"min_child_samples={min_samples_leaf}, learning_rate={learning_rate}"
        )
    elif model_type == 'catboost' or model_type == 'catboost':
        if CatBoostClassifier is None:
            raise ValueError("CatBoost is not installed. Please `pip install catboost` to use this model.")
        model = CatBoostClassifier(
            iterations=n_estimators,
            depth=max_depth,
            learning_rate=learning_rate,
            l2_leaf_reg=cat_l2,
            border_count=cat_border_count,
            random_seed=random_state,
            verbose=False,
            loss_function='Logloss',
            eval_metric='Logloss',
            auto_class_weights='Balanced',
            thread_count=cat_thread_count,
            task_type=cat_task_type
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    steps.append(('classifier', model))

    # Create pipeline
    if use_smote:
        pipeline = ImbPipeline(steps)
    else:
        from sklearn.pipeline import Pipeline
        pipeline = Pipeline(steps)

    # Train
    print(f"   Training {model_type}...")
    pipeline.fit(X_train_matrix, y_train)

    # Calibration (optional)
    if calibrate:
        from sklearn.calibration import CalibratedClassifierCV
        if external_test_data is not None:
            print(f"   Calibrating probabilities using {calibration_method} (3-fold CV on training data)...")
            calibrated_pipeline = CalibratedClassifierCV(
                pipeline,
                method=calibration_method,
                cv=3
            )
            calibrated_pipeline.fit(X_train_matrix, y_train)
        else:
            print(f"   Calibrating probabilities using {calibration_method} method...")
            prefitted_estimator, calibrator_kwargs = _wrap_prefit_calibrator(pipeline)
            calibrated_pipeline = CalibratedClassifierCV(
                prefitted_estimator,
                method=calibration_method,
                **calibrator_kwargs,
            )
            calibrated_pipeline.fit(X_test_matrix, y_test)
        pipeline = calibrated_pipeline
        print("   Calibration complete")

    # Predictions
    y_pred_train = pipeline.predict(X_train_matrix)
    y_pred_test = pipeline.predict(X_test_matrix)

    y_prob_train = pipeline.predict_proba(X_train_matrix)[:, 1]
    y_prob_test = pipeline.predict_proba(X_test_matrix)[:, 1]

    print("   Training complete")

    return {
        'pipeline': pipeline,
        'model': model,
        'scaler': scaler,
        'X_train': X_train_matrix,
        'X_test': X_test_matrix,
        'y_train': y_train,
        'y_test': y_test,
        'y_pred_train': y_pred_train,
        'y_pred_test': y_pred_test,
        'y_prob_train': y_prob_train,
        'y_prob_test': y_prob_test,
    }


def evaluate_with_cv(
    X: pd.DataFrame,
    y: pd.Series,
    model_type: str = 'random_forest',
    use_smote: bool = True,
    calibrate: bool = False,
    calibration_method: str = 'sigmoid',
    cv_folds: int = 5,
    random_state: int = 42,
    max_depth: int = 10,
    min_samples_leaf: int = 1,
    n_estimators: int = 100,
    learning_rate: float = 0.1,
    cat_l2: float = 1.0,
    cat_border_count: int = 128,
    cat_thread_count: int = -1,
    cat_task_type: str = 'CPU',
) -> dict:
    """
    Evaluate model using k-fold cross-validation for robust performance estimates.

    Benefits:
    - More robust metrics (especially with small positive class)
    - Avoids single train/test split bias
    - Each sample used for both training and testing
    - Better calibration (each fold calibrated independently)

    Args:
        cv_folds: Number of cross-validation folds. Default 5.
                  With 170 positives, 5 folds = ~34 positives per test fold.
    """
    print(f"[5/8] Evaluating {model_type} model with {cv_folds}-fold cross-validation...")

    # Build pipeline (same as train_models but without train/test split)
    steps = []

    scaler = StandardScaler()
    steps.append(('scaler', scaler))

    # Note: SMOTE is applied within each fold automatically by pipeline
    if use_smote:
        smote = SMOTE(random_state=random_state, k_neighbors=5)
        steps.append(('smote', smote))

    # Model
    if model_type == 'logistic':
        model = LogisticRegression(
            class_weight='balanced',
            max_iter=1000,
            random_state=random_state
        )
    elif model_type == 'random_forest':
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            class_weight='balanced',
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
            n_jobs=-1
        )
        print(
            "   Random Forest regularization: "
            f"n_estimators={n_estimators}, max_depth={max_depth}, min_samples_leaf={min_samples_leaf}"
        )
    elif model_type == 'gradient_boosting':
        model = GradientBoostingClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,  # Use same max_depth as RF
            learning_rate=learning_rate,
            random_state=random_state
        )
        print(
            "   Gradient Boosting regularization: "
            f"n_estimators={n_estimators}, max_depth={max_depth}, learning_rate={learning_rate}"
        )
    elif model_type == 'lightgbm':
        model = lgb.LGBMClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            num_leaves=2**max_depth - 1,  # Common pattern: full binary tree
            min_child_samples=min_samples_leaf,  # Equivalent to min_samples_leaf
            class_weight='balanced',
            learning_rate=learning_rate,
            random_state=random_state,
            n_jobs=-1,
            verbose=-1  # Suppress output
        )
        print(
            "   LightGBM regularization: "
            f"n_estimators={n_estimators}, max_depth={max_depth}, "
            f"min_child_samples={min_samples_leaf}, learning_rate={learning_rate}"
        )
    elif model_type == 'catboost':
        if CatBoostClassifier is None:
            raise ValueError("CatBoost is not installed. Please `pip install catboost` to use this model.")
        model = CatBoostClassifier(
            iterations=n_estimators,
            depth=max_depth,
            learning_rate=learning_rate,
            l2_leaf_reg=cat_l2,
            border_count=cat_border_count,
            random_seed=random_state,
            verbose=False,
            loss_function='Logloss',
            eval_metric='Logloss',
            auto_class_weights='Balanced',
            thread_count=cat_thread_count,
            task_type=cat_task_type
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    steps.append(('classifier', model))

    # Create pipeline
    if use_smote:
        pipeline = ImbPipeline(steps)
    else:
        from sklearn.pipeline import Pipeline
        pipeline = Pipeline(steps)

    # Stratified k-fold CV
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

    # Scoring metrics
    from sklearn.metrics import fbeta_score, make_scorer, matthews_corrcoef
    scoring = {
        'precision': 'precision',
        'recall': 'recall',
        'f1': 'f1',
        'roc_auc': 'roc_auc',
        'average_precision': 'average_precision',  # PR-AUC
        'mcc': make_scorer(matthews_corrcoef),
        'f2': make_scorer(fbeta_score, beta=2),
    }

    print(f"   Running {cv_folds}-fold stratified CV...")
    print(f"   Total samples: {len(X)} ({y.sum()} positive)")
    print(f"   Each test fold: ~{len(X)//cv_folds} samples (~{y.sum()//cv_folds} positive)")

    # Run CV -- CatBoost conflicts with joblib cloning, so run single-threaded
    X_matrix = X.values if isinstance(X, pd.DataFrame) else X
    n_jobs_cv = 1 if model_type == 'catboost' else -1

    cv_results = cross_validate(
        pipeline, X_matrix, y,
        cv=cv,
        scoring=scoring,
        return_train_score=True,
        n_jobs=n_jobs_cv
    )

    # Collect per-fold predictions for diagnostics (PR curves, calibration, etc.)
    fold_diagnostics: list[dict[str, Any]] = []
    oof_y_true: list[int] = []
    oof_y_prob: list[float] = []
    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X_matrix, y)):
        fold_pipeline = clone(pipeline)
        fold_pipeline.fit(X_matrix[train_idx], y.iloc[train_idx])
        y_test_fold = y.iloc[test_idx]
        y_prob_fold = fold_pipeline.predict_proba(X_matrix[test_idx])[:, 1]
        y_pred_fold = (y_prob_fold >= 0.5).astype(int)

        oof_y_true.extend(y_test_fold.tolist())
        oof_y_prob.extend(y_prob_fold.tolist())

        tp = int(((y_pred_fold == 1) & (y_test_fold.values == 1)).sum())
        fp = int(((y_pred_fold == 1) & (y_test_fold.values == 0)).sum())
        fn = int(((y_pred_fold == 0) & (y_test_fold.values == 1)).sum())
        tn = int(((y_pred_fold == 0) & (y_test_fold.values == 0)).sum())
        fold_diagnostics.append({
            "fold": fold_idx,
            "n_test": len(test_idx),
            "n_positive": int(y_test_fold.sum()),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        })
    cv_results["fold_diagnostics"] = fold_diagnostics
    cv_results["oof_y_true"] = oof_y_true
    cv_results["oof_y_prob"] = oof_y_prob

    # Aggregate results
    print(f"\n   === CROSS-VALIDATION RESULTS ({cv_folds} folds) ===")
    print("\n   Test Performance (mean ± std):")
    for metric in ['precision', 'recall', 'f1', 'roc_auc', 'average_precision', 'mcc', 'f2']:
        test_scores = cv_results[f'test_{metric}']
        mean_score = test_scores.mean()
        std_score = test_scores.std()
        print(f"      {metric:20s}: {mean_score:.3f} ± {std_score:.3f}")

    print("\n   Train Performance (mean ± std):")
    for metric in ['precision', 'recall', 'f1']:
        train_scores = cv_results[f'train_{metric}']
        mean_score = train_scores.mean()
        std_score = train_scores.std()
        print(f"      {metric:20s}: {mean_score:.3f} ± {std_score:.3f}")

    # Train final model on full data
    print("\n   Training final model on full dataset...")
    pipeline.fit(X_matrix, y)

    # Optionally calibrate final model
    if calibrate:
        from sklearn.calibration import CalibratedClassifierCV
        print(f"   Calibrating final model using {calibration_method} with CV...")
        pipeline = CalibratedClassifierCV(
            pipeline,
            method=calibration_method,
            cv=3  # Use 3-fold CV for calibration
        )
        pipeline.fit(X_matrix, y)

    print("   Training complete")

    # Package results
    metrics = {
        'cv_precision_mean': float(cv_results['test_precision'].mean()),
        'cv_precision_std': float(cv_results['test_precision'].std()),
        'cv_recall_mean': float(cv_results['test_recall'].mean()),
        'cv_recall_std': float(cv_results['test_recall'].std()),
        'cv_f1_mean': float(cv_results['test_f1'].mean()),
        'cv_f1_std': float(cv_results['test_f1'].std()),
        'cv_roc_auc_mean': float(cv_results['test_roc_auc'].mean()),
        'cv_roc_auc_std': float(cv_results['test_roc_auc'].std()),
        'cv_pr_auc_mean': float(cv_results['test_average_precision'].mean()),
        'cv_pr_auc_std': float(cv_results['test_average_precision'].std()),
        'cv_mcc_mean': float(cv_results['test_mcc'].mean()),
        'cv_mcc_std': float(cv_results['test_mcc'].std()),
        'cv_f2_mean': float(cv_results['test_f2'].mean()),
        'cv_f2_std': float(cv_results['test_f2'].std()),
        'cv_folds': cv_folds,
    }

    return {
        'pipeline': pipeline,
        'model': model,
        'scaler': scaler,
        'metrics': metrics,
        'cv_results': cv_results,
        'X': X_matrix,
        'y': y
    }


# ---------------------------------------------------------------------------
# 4. Evaluation
# ---------------------------------------------------------------------------


def evaluate_model(results: dict, feature_names: list[str]) -> dict:
    """
    Evaluate model performance with metrics appropriate for imbalanced data.
    """
    print("[6/8] Evaluating model...")

    y_train = results['y_train']
    y_test = results['y_test']
    y_pred_train = results['y_pred_train']
    y_pred_test = results['y_pred_test']
    results['y_prob_train']
    y_prob_test = results['y_prob_test']

    # Classification reports
    print("\n   === TRAIN SET ===")
    print(classification_report(y_train, y_pred_train, target_names=['Negative', 'Positive'], zero_division=0))

    print("\n   === TEST SET ===")
    print(classification_report(y_test, y_pred_test, target_names=['Negative', 'Positive'], zero_division=0))

    # Confusion matrices
    confusion_matrix(y_train, y_pred_train)
    cm_test = confusion_matrix(y_test, y_pred_test)

    print("\n   Confusion Matrix (Test):")
    print(f"      TN={cm_test[0,0]:4d}  FP={cm_test[0,1]:4d}")
    print(f"      FN={cm_test[1,0]:4d}  TP={cm_test[1,1]:4d}")

    # ROC-AUC and PR-AUC
    if y_test.sum() > 0:
        roc_auc_test = roc_auc_score(y_test, y_prob_test)
        pr_auc_test = average_precision_score(y_test, y_prob_test)

        print(f"\n   ROC-AUC (Test): {roc_auc_test:.3f}")
        print(f"   PR-AUC (Test):  {pr_auc_test:.3f}")
    else:
        roc_auc_test = 0.0
        pr_auc_test = 0.0

    # Feature importance (if available)
    pipeline = results['pipeline']

    # Handle calibrated vs uncalibrated pipelines
    from sklearn.calibration import CalibratedClassifierCV
    if isinstance(pipeline, CalibratedClassifierCV):
        # For calibrated models, get base estimator
        base_pipeline = pipeline.estimator
        classifier = base_pipeline.named_steps['classifier']
        print("\n   Note: Feature importance from base model (before calibration)")
    else:
        # Regular pipeline
        classifier = pipeline.named_steps['classifier']

    if hasattr(classifier, 'feature_importances_'):
        importances = classifier.feature_importances_
        indices = np.argsort(importances)[::-1]

        print("\n   Feature Importance (Top 10):")
        for i in range(min(10, len(feature_names))):
            idx = indices[i]
            print(f"      {i+1:2d}. {feature_names[idx]:30s} {importances[idx]:.4f}")

    elif hasattr(classifier, 'coef_'):
        coefs = np.abs(classifier.coef_[0])
        indices = np.argsort(coefs)[::-1]

        print("\n   Feature Coefficients (Top 10):")
        for i in range(min(10, len(feature_names))):
            idx = indices[i]
            print(f"      {i+1:2d}. {feature_names[idx]:30s} {coefs[idx]:.4f}")

    # Metrics summary
    from sklearn.metrics import f1_score, precision_score, recall_score

    metrics = {
        'precision_train': float(precision_score(y_train, y_pred_train, zero_division=0)),
        'recall_train': float(recall_score(y_train, y_pred_train, zero_division=0)),
        'f1_train': float(f1_score(y_train, y_pred_train, zero_division=0)),
        'precision_test': float(precision_score(y_test, y_pred_test, zero_division=0)),
        'recall_test': float(recall_score(y_test, y_pred_test, zero_division=0)),
        'f1_test': float(f1_score(y_test, y_pred_test, zero_division=0)),
        'roc_auc_test': float(roc_auc_test),
        'pr_auc_test': float(pr_auc_test),
        'confusion_matrix_test': cm_test.tolist(),
    }

    return metrics


# ---------------------------------------------------------------------------
# 5. Prediction and Export
# ---------------------------------------------------------------------------


def generate_predictions(
    df: pd.DataFrame,
    X: pd.DataFrame,
    pipeline,
    feature_names: list[str],  # noqa: ARG001
    output_dir: Path,
    threshold: float = 0.70,
    persistence_window: int = 2,
) -> pd.DataFrame:
    """
    Generate predictions for all lineage-quarters and export.

    Args:
        threshold: Decision threshold for binary classification. Default 0.70.
                   At 0.70: 39.6% precision, 59.4% recall (2:3 TP:FP ratio).
                   At 0.50: 7.3% precision, 88.8% recall (1:13 ratio - unusable).
        persistence_window: Require detections to sustain >= threshold for this many
                            consecutive quarters (set to 1 to disable).
    """
    print(f"[7/8] Generating predictions (threshold={threshold:.2f})...")

    # Predict probabilities
    X_matrix = X.values if isinstance(X, pd.DataFrame) else X

    y_prob = pipeline.predict_proba(X_matrix)[:, 1]

    # Apply custom threshold (not default 0.50)
    y_pred = (y_prob >= threshold).astype(int)

    # Create predictions DataFrame (onset-primary schema)
    predictions_df = pd.DataFrame({
        'lineage_id': df['lineage_id'],
        'quarter': df['quarter'],
        'is_onset_true': df['is_milestone'],
        'is_onset_pred': y_pred,
        'onset_probability': y_prob,
    })
    # Legacy column aliases for backward compatibility with downstream scripts
    predictions_df['is_inflection_true'] = predictions_df['is_onset_true']
    predictions_df['is_inflection_pred'] = predictions_df['is_onset_pred']
    predictions_df['inflection_probability'] = predictions_df['onset_probability']
    predictions_df['is_milestone_true'] = predictions_df['is_onset_true']
    predictions_df['is_milestone_pred'] = predictions_df['is_onset_pred']
    predictions_df['milestone_probability'] = predictions_df['onset_probability']
    predictions_df = add_detection_lag_column(predictions_df)

    print(f"   Generated {len(predictions_df)} predictions")
    print(f"   Predicted positives: {y_pred.sum()} ({y_pred.sum()/len(y_pred)*100:.2f}%)")
    print(f"   Using threshold {threshold:.2f} (not default 0.50)")

    # Enforce persistence for downstream QA/visualizations
    predictions_df = ensure_persistence_column(
        predictions_df,
        threshold=threshold,
        window=persistence_window,
        column_name="is_onset_pred_persistent",
    )
    # Legacy aliases
    predictions_df["is_inflection_pred_persistent"] = predictions_df["is_onset_pred_persistent"]
    predictions_df["is_milestone_pred_persistent"] = predictions_df["is_onset_pred_persistent"]

    persistent_hits = int(predictions_df["is_onset_pred_persistent"].sum())
    print(
        f"   Persistent detections (>={persistence_window}Q): "
        f"{persistent_hits} ({persistent_hits/len(predictions_df)*100:.2f}%)"
    )

    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / 'breakthrough_predictions.csv'
    predictions_df.to_csv(predictions_path, index=False)

    print(f"   Saved predictions to {predictions_path}")

    return predictions_df


def save_model_and_metrics(
    pipeline,
    metrics: dict,
    feature_names: list[str],
    output_dir: Path,
    threshold: float = 0.70,
    lag_min: int = 2,
    lag_max: int | None = 8,
    calibrated: bool = False,
    calibration_method: str = None
):
    """
    Save trained model and evaluation metrics.
    """
    print("[8/8] Saving model and metrics...")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Save model (trusted artifact -- not portable)
    model_path = output_dir / 'breakthrough_detector_model.pkl'
    save_trusted_pickle(
        pipeline, model_path, description="MSD trained model"
    )
    print(f"   Saved model to {model_path}")

    # Add configuration metadata to metrics
    metrics['threshold'] = threshold
    metrics['lag_min'] = lag_min
    metrics['lag_max'] = lag_max
    metrics['calibrated'] = calibrated
    if calibrated:
        metrics['calibration_method'] = calibration_method
    metrics['note'] = f"Using threshold {threshold:.2f}."
    if calibrated:
        metrics['note'] += f" Probabilities calibrated using {calibration_method}."

    # Save metrics
    metrics_path = output_dir / 'evaluation_metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"   Saved metrics to {metrics_path}")

    # Save feature list
    features_path = output_dir / 'feature_names.json'
    with open(features_path, 'w') as f:
        json.dump(feature_names, f, indent=2)
    print(f"   Saved feature names to {features_path}")


# ---------------------------------------------------------------------------
# Main CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Multi-Signal Detector (MSD): Supervised Ensemble Breakthrough Detector"
    )

    # Input paths
    parser.add_argument(
        '--labels',
        default=None,
        help='Optional label CSV (e.g., inflection labels) with columns lineage_id, quarter, is_inflection_onset.'
    )
    parser.add_argument(
        '--tight-mapping',
        default=None,
    )
    parser.add_argument(
        '--semantic-velocity',
        default=None,
    )
    parser.add_argument(
        '--multisignal',
        default=None,
    )
    parser.add_argument(
        '--timeseries',
        default=None,
    )
    parser.add_argument(
        '--output-dir',
        default=None,
    )

    # Model configuration
    model_choices = ['logistic', 'random_forest', 'gradient_boosting', 'lightgbm']
    if CatBoostClassifier is not None:
        model_choices.append('catboost')
    parser.add_argument('--model', choices=model_choices,
                       default='random_forest')
    parser.add_argument('--no-smote', action='store_true', help='Disable SMOTE oversampling')
    parser.add_argument('--calibrate', action='store_true',
                       help='Calibrate probabilities using CalibratedClassifierCV (recommended for better PR-AUC)')
    parser.add_argument('--calibration-method', choices=['sigmoid', 'isotonic'], default='sigmoid',
                       help='Calibration method: sigmoid (Platt scaling) or isotonic. Default: sigmoid')
    parser.add_argument('--use-cv', action='store_true',
                       help='Use k-fold cross-validation for evaluation (more robust metrics)')
    parser.add_argument('--cv-folds', type=int, default=5,
                       help='Number of CV folds. Default 5 (~34 positives per fold)')
    parser.add_argument('--test-size', type=float, default=0.2)
    parser.add_argument('--random-state', type=int, default=42)

    # Regularization parameters
    parser.add_argument('--max-depth', type=int, default=10,
                       help='Maximum tree depth for RF/GBM. Default 10. Use 5-7 for regularization to reduce overfitting.')
    parser.add_argument('--min-samples-leaf', type=int, default=1,
                       help='Minimum samples per leaf node for RF. Default 1. Use 5-10 for regularization to reduce overfitting.')
    parser.add_argument('--cat-iterations', type=int, default=1000,
                        help='Number of boosting iterations for CatBoost.')
    parser.add_argument('--cat-depth', type=int, default=8,
                        help='Tree depth for CatBoost.')
    parser.add_argument('--cat-learning-rate', type=float, default=0.05,
                        help='Learning rate for CatBoost.')
    parser.add_argument('--cat-l2', type=float, default=1.0,
                        help='L2 regularization term for CatBoost.')
    parser.add_argument('--cat-border-count', type=int, default=128,
                        help='Border count (feature binning) for CatBoost.')
    parser.add_argument('--cat-thread-count', type=int, default=-1,
                        help='Thread count for CatBoost (-1 = use all cores).')
    parser.add_argument('--cat-task-type', choices=['CPU', 'GPU'], default='CPU',
                        help='CatBoost task type (CPU or GPU).')
    parser.add_argument('--n-estimators', type=int, default=100,
                       help='Number of trees / boosting iterations for RF, GBM, LightGBM (default: 100).')
    parser.add_argument('--learning-rate', type=float, default=0.1,
                       help='Learning rate for GBM/LightGBM. Ignored for logistic/RF. Default 0.1.')

    # Testing
    parser.add_argument('--n-samples', type=int, default=None,
                       help='Limit to N samples for testing')

    # Temporal filtering (time-forward experiments)
    parser.add_argument('--train-start', default=None,
                       help='Inclusive quarter for training data start (e.g., 2003Q1). Default = earliest quarter.')
    parser.add_argument('--train-end', default=None,
                       help='Inclusive quarter for training data end (e.g., 2019Q4). Default = latest quarter.')
    parser.add_argument('--predict-start', default=None,
                       help='Inclusive quarter for prediction/evaluation slice start (e.g., 2020Q1). Default = earliest.')
    parser.add_argument('--predict-end', default=None,
                       help='Inclusive quarter for prediction/evaluation slice end. Default = latest quarter.')

    # Detection window configuration
    parser.add_argument('--lag-min', type=int, default=2,
                       help='Minimum lag (in quarters) after event to start detection. Default 2 (no forecasting)')
    parser.add_argument('--lag-max', default="8",
                       help="Maximum lag (in quarters) after event to end detection (default 8). "
                            "Use 'none' for no upper bound.")

    # Threshold configuration
    parser.add_argument('--threshold', type=float, default=0.70,
                       help='Decision threshold for binary classification (default 0.70).')
    parser.add_argument('--persistence-window', type=int, default=2,
                       help='Quarters that detections must stay above threshold to count (set 1 to disable).')
    parser.add_argument('--disable-field-features', action='store_true',
                       help='Exclude field-relative columns even if present in the multisignal matrix.')
    parser.add_argument('--field-features-only', action='store_true',
                       help='Use only field-relative columns (ablation mode).')
    parser.add_argument('--leakage-safe', action='store_true',
                       help='Exclude features that use future data (logistic fits, '
                            'CD disruption index, field-relative metrics). '
                            'Use for prospective evaluation experiments.')

    add_domain_args(parser)
    args = parser.parse_args()

    paths = resolve_script_paths(args, _REPO)
    apply_domain_path_defaults(args, paths, {
        "tight_mapping": (
            "experiments",
            "stage0_tight_mapping/milestone_lineage_mapping_tight.csv",
            "data/out/experiments/stage0_tight_mapping/milestone_lineage_mapping_tight.csv",
        ),
        "semantic_velocity": (
            "experiments",
            "stage1_quarterly_embeddings/semantic_velocity.csv",
            "data/out/experiments/stage1_quarterly_embeddings/semantic_velocity.csv",
        ),
        "multisignal": (
            "lineage_tracking",
            "lineage_multisignal_features.csv",
            "data/out/02_lineage_tracking/lineage_multisignal_features.csv",
        ),
        "timeseries": (
            "lineage_tracking",
            "lineage_timeseries.csv",
            "data/out/02_lineage_tracking/lineage_timeseries.csv",
        ),
        "output_dir": (
            "experiments",
            "multi_signal_detector",
            "data/out/experiments/multi_signal_detector",
        ),
    })
    if args.disable_field_features and args.field_features_only:
        parser.error("Cannot combine --disable-field-features with --field-features-only.")
    if args.model == 'catboost':
        args.n_estimators = args.cat_iterations
        args.max_depth = args.cat_depth
        args.learning_rate = args.cat_learning_rate
    args.lag_max = _parse_lag_max_arg(args.lag_max)

    print("="*70)
    if args.n_samples:
        print(f"MSD: ENSEMBLE DETECTOR [TEST MODE: {args.n_samples} samples]")
    else:
        print("Multi-Signal Detector (MSD): Supervised Ensemble Breakthrough Detector [PRODUCTION]")
    print("="*70)
    print()

    print("Configuration:")
    print(f"   Model: {args.model}")
    if args.model in ['random_forest', 'gradient_boosting']:
        print(f"   Regularization: max_depth={args.max_depth}, min_samples_leaf={args.min_samples_leaf if args.model == 'random_forest' else 'N/A'}")
    print(f"   SMOTE: {not args.no_smote}")
    print(f"   Calibration: {args.calibrate} ({args.calibration_method if args.calibrate else 'N/A'})")
    print(f"   Evaluation: {'Cross-validation' if args.use_cv else 'Train/test split'}")
    if args.use_cv:
        print(f"   CV folds: {args.cv_folds}")
    else:
        print(f"   Test size: {args.test_size}")
    if args.lag_max is None:
        detection_window = f"[{args.lag_min}Q, inf)"
    else:
        detection_window = f"[{args.lag_min}Q, {args.lag_max}Q]"
    print(f"   Detection window: {detection_window} after event (detect impact, not predict event)")
    print(f"   Threshold: {args.threshold:.2f}")
    persistence_label = "quarters" if args.persistence_window != 1 else "quarter"
    print(f"   Persistence window: {args.persistence_window} {persistence_label} (>= threshold)")
    print(f"   Sample limit: {args.n_samples if args.n_samples else 'None (full dataset)'}")
    print(f"   Train quarters: {describe_quarter_range(args.train_start, args.train_end)}")
    print(f"   Prediction quarters: {describe_quarter_range(args.predict_start, args.predict_end)}")
    print()

    # Paths
    tight_mapping_path = Path(args.tight_mapping)
    semantic_velocity_path = Path(args.semantic_velocity)
    multisignal_path = Path(args.multisignal)
    timeseries_path = Path(args.timeseries)
    output_dir = Path(args.output_dir)

    # Verify inputs exist
    for path in [tight_mapping_path, semantic_velocity_path, multisignal_path, timeseries_path]:
        if not path.exists():
            print(f"ERROR: Input file not found: {path}")
            return

    # Step 1: Load and merge signals
    labels_path = Path(args.labels) if getattr(args, "labels", None) else None
    features_df, labels_df, tight_mapping = load_and_merge_signals(
        tight_mapping_path,
        semantic_velocity_path,
        multisignal_path,
        timeseries_path,
        labels_path=labels_path,
        n_samples=None,  # Always load full dataset for proper labeling
    )

    # Step 2: Construct labels
    features_df = construct_labels(features_df, labels_df, tight_mapping, lag_min=args.lag_min, lag_max=args.lag_max)

    # Stratified sampling for testing (after labeling)
    if args.n_samples and args.n_samples < len(features_df):
        print(f"\n[SAMPLING] Stratified sampling to {args.n_samples} samples...")

        n_positive = features_df['is_milestone'].sum()
        n_negative = len(features_df) - n_positive

        # Calculate how many positives and negatives we need
        positive_ratio = n_positive / len(features_df)
        n_positive_sample = max(1, int(args.n_samples * positive_ratio))
        n_negative_sample = args.n_samples - n_positive_sample

        print(f"   Original: {n_positive} positive, {n_negative} negative")
        print(f"   Sampling: {n_positive_sample} positive, {n_negative_sample} negative")

        # Separate positive and negative
        positive_df = features_df[features_df['is_milestone'] == 1]
        negative_df = features_df[features_df['is_milestone'] == 0]

        # Sample
        positive_sample = positive_df.sample(n=min(len(positive_df), n_positive_sample), random_state=42)
        negative_sample = negative_df.sample(n=min(len(negative_df), n_negative_sample), random_state=42)

        # Combine
        features_df = pd.concat([positive_sample, negative_sample]).sample(frac=1, random_state=42)
        print(f"   Sampled dataset: {len(features_df)} samples ({features_df['is_milestone'].sum()} positive)")

    # Step 3: Engineer features
    features_df = engineer_features(features_df)

    # Step 3.5: Quarter-based filtering (time-forward experiments)
    if args.train_start or args.train_end:
        train_df = filter_by_quarter(
            features_df,
            args.train_start,
            args.train_end,
            label="Training slice"
        )
    else:
        train_df = features_df.copy()
        print(f"   Training slice: full dataset ({len(train_df):,} rows)")

    if args.predict_start or args.predict_end:
        prediction_df = filter_by_quarter(
            features_df,
            args.predict_start,
            args.predict_end,
            label="Prediction slice"
        )
    else:
        prediction_df = features_df.copy()
        print(f"   Prediction slice: full dataset ({len(prediction_df):,} rows)")

    if train_df.empty:
        raise ValueError("Training slice has zero rows after quarter filtering.")
    if prediction_df.empty:
        raise ValueError("Prediction slice has zero rows after quarter filtering.")

    filters_applied = any([args.train_start, args.train_end, args.predict_start, args.predict_end])
    if filters_applied:
        inputs_dir = output_dir / "inputs"
        snapshot_dataset(train_df, inputs_dir, "lineage_features_train", args.train_start, args.train_end)
        snapshot_dataset(prediction_df, inputs_dir, "lineage_features_predict", args.predict_start, args.predict_end)

    # Step 4: Select features
    field_feature_mode = "off" if args.disable_field_features else ("only" if args.field_features_only else "auto")
    X, feature_names = select_features(
        train_df,
        field_feature_mode=field_feature_mode,
        leakage_safe=getattr(args, 'leakage_safe', False),
    )
    y = train_df['is_milestone']
    prediction_X, _ = select_features(prediction_df, required_features=feature_names)

    print(f"\n   Training dataset: {len(X)} samples, {len(feature_names)} features")
    print(f"   Positive class (train): {y.sum()} ({(y.sum()/len(y))*100:.2f}%)")
    has_external_holdout = not prediction_df.index.isin(train_df.index).all()
    if has_external_holdout:
        holdout_positive = int(prediction_df['is_milestone'].sum())
        holdout_rate = (holdout_positive / len(prediction_df)) * 100 if len(prediction_df) else 0.0
        print(f"   Holdout slice: {len(prediction_df)} samples, {holdout_positive} positives "
              f"({holdout_rate:.2f}%)")

    # Step 5 & 6: Train and evaluate model
    if args.use_cv:
        # Cross-validation mode (more robust)
        cv_results = evaluate_with_cv(
            X, y,
            model_type=args.model,
            use_smote=not args.no_smote,
            calibrate=args.calibrate,
            calibration_method=args.calibration_method,
            cv_folds=args.cv_folds,
            random_state=args.random_state,
            max_depth=args.max_depth,
            min_samples_leaf=args.min_samples_leaf,
            n_estimators=args.n_estimators,
            learning_rate=args.learning_rate,
            cat_l2=args.cat_l2,
            cat_border_count=args.cat_border_count,
            cat_thread_count=args.cat_thread_count,
            cat_task_type=args.cat_task_type
        )
        results = cv_results  # Use CV results structure
        metrics = cv_results['metrics']  # CV metrics already computed
    else:
        # Regular train/test split mode
        external_test_tuple = None
        if has_external_holdout:
            external_test_tuple = (
                prediction_X,
                prediction_df['is_milestone'].copy()
            )
        results = train_models(
            X, y,
            model_type=args.model,
            use_smote=not args.no_smote,
            calibrate=args.calibrate,
            calibration_method=args.calibration_method,
            test_size=args.test_size,
            random_state=args.random_state,
            max_depth=args.max_depth,
            min_samples_leaf=args.min_samples_leaf,
            n_estimators=args.n_estimators,
            learning_rate=args.learning_rate,
            cat_l2=args.cat_l2,
            cat_border_count=args.cat_border_count,
            cat_thread_count=args.cat_thread_count,
            cat_task_type=args.cat_task_type,
            external_test_data=external_test_tuple
        )
        # Step 6: Evaluate
        metrics = evaluate_model(results, feature_names)

    # Attach configuration metadata for reproducibility
    metrics['model_type'] = args.model
    model_params: dict[str, Any] = {}
    if args.model in {'random_forest', 'gradient_boosting', 'lightgbm'}:
        model_params.update({
            'n_estimators': args.n_estimators,
            'max_depth': args.max_depth,
            'min_samples_leaf': args.min_samples_leaf,
        })
    if args.model in {'gradient_boosting', 'lightgbm'}:
        model_params['learning_rate'] = args.learning_rate
    if args.model == 'catboost':
        model_params.update({
            'iterations': args.cat_iterations,
            'depth': args.cat_depth,
            'learning_rate': args.cat_learning_rate,
            'l2_leaf_reg': args.cat_l2,
            'border_count': args.cat_border_count,
            'task_type': args.cat_task_type,
        })
    if model_params:
        metrics['model_params'] = model_params
    metrics['evaluation_mode'] = (
        'cross_validation'
        if args.use_cv
        else ('external_holdout' if has_external_holdout else 'train_test_split')
    )

    # Step 7: Generate predictions
    # In CV mode, pipeline already trained on full training set; predictions use requested slice
    pred_X = prediction_X
    predictions_df = generate_predictions(
        prediction_df,
        pred_X,
        results['pipeline'],
        feature_names,
        output_dir,
        threshold=args.threshold,
        persistence_window=args.persistence_window,
    )

    # Raw threshold confusion stats (ranking performance)
    threshold_stats = compute_confusion_stats(predictions_df)
    lag_stats = summarize_detection_lag(predictions_df)
    metrics.update(threshold_stats)
    metrics.update(lag_stats)

    # Persistence-filtered confusion stats (operational alert performance)
    persistent_stats = compute_confusion_stats(
        predictions_df,
        pred_col="is_onset_pred_persistent",
        true_col="is_onset_true",
        prefix="persistent_",
    )
    metrics.update(persistent_stats)
    metrics["persistent_positive_count"] = int(predictions_df["is_onset_pred_persistent"].sum())

    # Step 8: Save model and metrics
    save_model_and_metrics(
        results['pipeline'],
        metrics,
        feature_names,
        output_dir,
        threshold=args.threshold,
        lag_min=args.lag_min,
        lag_max=args.lag_max,
        calibrated=args.calibrate,
        calibration_method=args.calibration_method if args.calibrate else None
    )

    # Step 8b: Save fold diagnostics (PR curves, calibration, per-fold stats)
    if args.use_cv and "cv_results" in results:
        save_fold_diagnostics(results["cv_results"], output_dir)

    # Step 8c: Save run provenance for reproducibility
    input_files = {
        "labels": labels_path,
        "tight_mapping": tight_mapping_path,
        "semantic_velocity": semantic_velocity_path,
        "multisignal": multisignal_path,
        "timeseries": timeseries_path,
    }
    provenance = collect_run_provenance(
        args,
        input_files,
        output_dir,
        repo_root=_REPO,
        extra={
            "n_features": len(feature_names),
            "n_train_samples": len(X),
            "n_positive_train": int(y.sum()),
            "has_external_holdout": has_external_holdout,
        },
    )
    prov_path = save_run_provenance(provenance, output_dir)
    print(f"   Saved run provenance to {prov_path}")

    # Summary
    print("\n" + "="*70)
    print("MSD COMPLETE")
    print("="*70)

    if args.use_cv:
        print(f"\nCross-Validation Performance ({args.cv_folds} folds):")
        print(f"   Precision: {metrics['cv_precision_mean']:.3f} ± {metrics['cv_precision_std']:.3f}")
        print(f"   Recall:    {metrics['cv_recall_mean']:.3f} ± {metrics['cv_recall_std']:.3f}")
        print(f"   F1 Score:  {metrics['cv_f1_mean']:.3f} ± {metrics['cv_f1_std']:.3f}")
        print(f"   ROC-AUC:   {metrics['cv_roc_auc_mean']:.3f} ± {metrics['cv_roc_auc_std']:.3f}")
        print(f"   PR-AUC:    {metrics['cv_pr_auc_mean']:.3f} ± {metrics['cv_pr_auc_std']:.3f}")
        print(f"   MCC:       {metrics['cv_mcc_mean']:.3f} ± {metrics['cv_mcc_std']:.3f}")
        print(f"   F2 Score:  {metrics['cv_f2_mean']:.3f} ± {metrics['cv_f2_std']:.3f}")

        print("\nStage 0 Baseline: 6.8% recall")
        print(f"MSD Recall:   {metrics['cv_recall_mean']*100:.1f}%")

        if metrics['cv_recall_mean'] > 0.068:
            improvement = (metrics['cv_recall_mean'] - 0.068) / 0.068 * 100
            print(f"Improvement:      +{improvement:.1f}%")
    else:
        print("\nTest Set Performance:")
        print(f"   Precision: {metrics['precision_test']:.3f}")
        print(f"   Recall:    {metrics['recall_test']:.3f}")
        print(f"   F1 Score:  {metrics['f1_test']:.3f}")
        print(f"   ROC-AUC:   {metrics['roc_auc_test']:.3f}")
        print(f"   PR-AUC:    {metrics['pr_auc_test']:.3f}")

        print("\nStage 0 Baseline: 6.8% recall")
        print(f"MSD Recall:   {metrics['recall_test']*100:.1f}%")

        if metrics['recall_test'] > 0.068:
            improvement = (metrics['recall_test'] - 0.068) / 0.068 * 100
            print(f"Improvement:      +{improvement:.1f}%")

    # Print both raw and persistent threshold stats
    print(f"\nThreshold Analysis (t={metrics.get('threshold', 0.70):.2f}):")
    print(f"   Raw:        TP={metrics['tp_threshold']}, FP={metrics['fp_threshold']}, "
          f"FN={metrics['fn_threshold']}, TN={metrics['tn_threshold']}")
    print(f"   Raw P/R/F1: {metrics['precision_threshold']:.3f} / "
          f"{metrics['recall_threshold']:.3f} / {metrics['f1_threshold']:.3f}")
    print(f"   Persistent: TP={metrics['persistent_tp_threshold']}, "
          f"FP={metrics['persistent_fp_threshold']}, "
          f"FN={metrics['persistent_fn_threshold']}, "
          f"TN={metrics['persistent_tn_threshold']}")
    print(f"   Pers P/R/F1: {metrics['persistent_precision_threshold']:.3f} / "
          f"{metrics['persistent_recall_threshold']:.3f} / "
          f"{metrics['persistent_f1_threshold']:.3f}")


if __name__ == '__main__':
    main()

