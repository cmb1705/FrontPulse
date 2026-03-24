"""Longitudinal assessment history for MSD predictions.

Maintains a table of (lineage_id, quarter_assessed, quarter_predicted,
probability, actual_outcome) that grows with each quarterly run.  Previous
predictions are backfilled with hindsight when ground-truth labels become
available, enabling calibration tracking over time.

The canonical CSV schema::

    lineage_id          int     Lineage identifier
    quarter_predicted   str     Quarter the prediction applies to (YYYYQN)
    quarter_assessed    str     Quarter the model was run (YYYYQN)
    model_version       str     Registry version ID (v_YYYYMMDD_NNN)
    probability         float   Predicted onset probability [0, 1]
    predicted_label     int     Binary prediction (threshold-applied)
    threshold_used      float   Decision threshold at assessment time
    actual_outcome      int     Ground truth (0/1), -1 = unknown
    outcome_source      str     How outcome was determined (empty if unknown)
    backfilled_at       str     ISO timestamp when outcome was filled in

Primary key: (lineage_id, quarter_predicted, model_version).
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

ASSESSMENT_COLUMNS = [
    "lineage_id",
    "quarter_predicted",
    "quarter_assessed",
    "model_version",
    "probability",
    "predicted_label",
    "threshold_used",
    "actual_outcome",
    "outcome_source",
    "backfilled_at",
]

OUTCOME_UNKNOWN = -1


def create_empty_history() -> pd.DataFrame:
    """Return an empty DataFrame with the canonical assessment schema."""
    return pd.DataFrame(columns=ASSESSMENT_COLUMNS)


def record_assessments(
    predictions: pd.DataFrame,
    quarter_assessed: str,
    model_version: str,
    threshold: float,
    probability_column: str = "inflection_probability",
) -> pd.DataFrame:
    """Convert a predictions DataFrame into assessment history rows.

    Args:
        predictions: DataFrame from MSD ``generate_predictions()`` with at
            least ``lineage_id``, ``quarter``, and a probability column.
        quarter_assessed: The quarter in which the model was run (YYYYQN).
        model_version: Registry version ID that produced these predictions.
        threshold: Decision threshold applied to probabilities.
        probability_column: Column name holding predicted probabilities.

    Returns:
        DataFrame with assessment schema columns, one row per
        (lineage_id, quarter) in the input.
    """
    required = {"lineage_id", "quarter", probability_column}
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"Predictions missing required columns: {missing}")

    probs = predictions[probability_column].values
    labels = (probs >= threshold).astype(int)

    df = pd.DataFrame({
        "lineage_id": predictions["lineage_id"].values,
        "quarter_predicted": predictions["quarter"].values,
        "quarter_assessed": quarter_assessed,
        "model_version": model_version,
        "probability": np.round(probs, 6),
        "predicted_label": labels,
        "threshold_used": threshold,
        "actual_outcome": OUTCOME_UNKNOWN,
        "outcome_source": "",
        "backfilled_at": "",
    })
    return df


def backfill_outcomes(
    history: pd.DataFrame,
    labels: pd.DataFrame,
    label_column: str = "is_inflection_onset",
    source: str = "onset_labels",
) -> tuple[pd.DataFrame, int]:
    """Update unknown outcomes in the history with ground-truth labels.

    Only rows where ``actual_outcome == -1`` and a matching label exists
    are updated.  Rows already resolved are left untouched.

    Args:
        history: Current assessment history DataFrame.
        labels: Ground-truth labels with ``lineage_id``, ``quarter``,
            and *label_column* columns.
        label_column: Binary label column in the labels DataFrame.
        source: Description of the label source (stored in ``outcome_source``).

    Returns:
        Tuple of (updated history, count of rows backfilled).
    """
    if history.empty:
        return history, 0

    required = {"lineage_id", "quarter", label_column}
    missing = required - set(labels.columns)
    if missing:
        raise ValueError(f"Labels missing required columns: {missing}")

    # Build lookup: (lineage_id, quarter) -> outcome
    label_map: dict[tuple[Any, str], int] = {}
    for _, row in labels.iterrows():
        key = (row["lineage_id"], row["quarter"])
        label_map[key] = int(row[label_column])

    # For lineages in the label file but NOT in the positive set,
    # the implicit outcome is 0 (no onset detected).
    all_label_lineages = set(labels["lineage_id"].unique())

    now_str = datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    updated = history.copy()
    count = 0

    unknown_mask = updated["actual_outcome"] == OUTCOME_UNKNOWN
    for idx in updated.index[unknown_mask]:
        lid = updated.at[idx, "lineage_id"]
        qp = updated.at[idx, "quarter_predicted"]
        key = (lid, qp)

        if key in label_map:
            updated.at[idx, "actual_outcome"] = label_map[key]
            updated.at[idx, "outcome_source"] = source
            updated.at[idx, "backfilled_at"] = now_str
            count += 1
        elif lid in all_label_lineages:
            # Lineage has labels but this quarter is not an onset -> 0
            updated.at[idx, "actual_outcome"] = 0
            updated.at[idx, "outcome_source"] = f"{source}:implicit_negative"
            updated.at[idx, "backfilled_at"] = now_str
            count += 1

    return updated, count


def append_assessments(
    existing: pd.DataFrame,
    new_rows: pd.DataFrame,
) -> pd.DataFrame:
    """Append new assessment rows, deduplicating on primary key.

    If a row with the same (lineage_id, quarter_predicted, model_version)
    already exists, the existing row is kept (no overwrite).

    Args:
        existing: Current assessment history.
        new_rows: New rows to append.

    Returns:
        Combined DataFrame with duplicates removed.
    """
    if existing.empty:
        return new_rows.copy()
    if new_rows.empty:
        return existing.copy()

    combined = pd.concat([existing, new_rows], ignore_index=True)
    pk = ["lineage_id", "quarter_predicted", "model_version"]
    combined = combined.drop_duplicates(subset=pk, keep="first")
    return combined.reset_index(drop=True)


def load_history(path: Path) -> pd.DataFrame:
    """Load assessment history from CSV, or return empty if not found."""
    if not path.exists():
        logger.info("No existing history at %s, starting fresh", path)
        return create_empty_history()
    df = pd.read_csv(path)
    # Ensure schema alignment
    for col in ASSESSMENT_COLUMNS:
        if col not in df.columns:
            df[col] = "" if col in {"outcome_source", "backfilled_at"} else OUTCOME_UNKNOWN
    return df


def save_history(history: pd.DataFrame, path: Path) -> None:
    """Save assessment history to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    history[ASSESSMENT_COLUMNS].to_csv(path, index=False)
    logger.info("Saved %d assessment rows to %s", len(history), path)


def compute_calibration_stats(
    history: pd.DataFrame,
    n_bins: int = 10,
) -> dict[str, Any]:
    """Compute calibration statistics from resolved assessments.

    Only rows where ``actual_outcome`` is 0 or 1 (not -1) are used.

    Args:
        history: Assessment history DataFrame.
        n_bins: Number of probability bins for calibration curve.

    Returns:
        Dictionary with calibration metrics:
        - ``n_resolved``: count of resolved predictions
        - ``n_unknown``: count of still-unknown predictions
        - ``bins``: list of dicts with bin_center, predicted_mean,
          observed_rate, count per bin
        - ``brier_score``: mean squared error of probabilities
        - ``calibration_error``: expected calibration error (ECE)
    """
    resolved = history[history["actual_outcome"].isin([0, 1])].copy()
    n_unknown = int((history["actual_outcome"] == OUTCOME_UNKNOWN).sum())

    if resolved.empty:
        return {
            "n_resolved": 0,
            "n_unknown": n_unknown,
            "bins": [],
            "brier_score": None,
            "calibration_error": None,
        }

    probs = resolved["probability"].values.astype(float)
    outcomes = resolved["actual_outcome"].values.astype(float)

    # Brier score
    brier = float(np.mean((probs - outcomes) ** 2))

    # Bin edges
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins_data = []
    ece_sum = 0.0

    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (probs >= lo) & (probs < hi) if i < n_bins - 1 else (probs >= lo) & (probs <= hi)
        count = int(mask.sum())
        if count == 0:
            bins_data.append({
                "bin_center": round((lo + hi) / 2, 3),
                "predicted_mean": None,
                "observed_rate": None,
                "count": 0,
            })
            continue
        pred_mean = float(np.mean(probs[mask]))
        obs_rate = float(np.mean(outcomes[mask]))
        bins_data.append({
            "bin_center": round((lo + hi) / 2, 3),
            "predicted_mean": round(pred_mean, 4),
            "observed_rate": round(obs_rate, 4),
            "count": count,
        })
        ece_sum += abs(pred_mean - obs_rate) * count

    ece = ece_sum / len(resolved) if len(resolved) > 0 else 0.0

    return {
        "n_resolved": len(resolved),
        "n_unknown": n_unknown,
        "bins": bins_data,
        "brier_score": round(brier, 6),
        "calibration_error": round(ece, 6),
    }


def summarize_history(history: pd.DataFrame) -> dict[str, Any]:
    """Generate a summary of the assessment history state.

    Args:
        history: Assessment history DataFrame.

    Returns:
        Dictionary with summary statistics.
    """
    if history.empty:
        return {
            "total_rows": 0,
            "n_versions": 0,
            "n_lineages": 0,
            "n_quarters_assessed": 0,
            "n_resolved": 0,
            "n_unknown": 0,
            "resolution_rate": 0.0,
        }

    n_resolved = int(history["actual_outcome"].isin([0, 1]).sum())
    n_unknown = int((history["actual_outcome"] == OUTCOME_UNKNOWN).sum())
    total = len(history)

    return {
        "total_rows": total,
        "n_versions": int(history["model_version"].nunique()),
        "n_lineages": int(history["lineage_id"].nunique()),
        "n_quarters_assessed": int(history["quarter_assessed"].nunique()),
        "n_resolved": n_resolved,
        "n_unknown": n_unknown,
        "resolution_rate": round(n_resolved / max(total, 1), 4),
    }
