"""Probabilistic forward-looking onset estimates with conformal intervals.

Produces per-lineage onset probability estimates for future quarters (1-4Q
horizon) with confidence bounds derived from conformal prediction on
historical assessment residuals.

The approach:

1. Load the most recent MSD predictions for active lineages.
2. For each horizon step h in {1, 2, 3, 4}, apply a simple persistence
   forecast: the predicted probability decays or grows based on the
   lineage's recent trajectory (trend extrapolation from last 2-4 quarters).
3. Calibrate intervals using nonconformity scores from resolved assessments
   in the history table.  The nonconformity score for a resolved prediction
   is ``|probability - actual_outcome|``.  The (1 - alpha) quantile of these
   scores gives the interval half-width, providing distribution-free
   coverage guarantees.

Output schema (per row)::

    lineage_id          int     Lineage identifier
    quarter_target      str     Future quarter being estimated (YYYYQN)
    horizon             int     Quarters ahead (1-4)
    point_estimate      float   Predicted onset probability [0, 1]
    lower_bound         float   Lower confidence bound [0, 1]
    upper_bound         float   Upper confidence bound [0, 1]
    confidence_level    float   Nominal coverage (e.g., 0.90)
    basis_quarter       str     Quarter of the base prediction
    basis_model_version str     Model version used for base prediction
    method              str     Estimation method identifier
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

HORIZON_COLUMNS = [
    "lineage_id",
    "quarter_target",
    "horizon",
    "point_estimate",
    "lower_bound",
    "upper_bound",
    "confidence_level",
    "basis_quarter",
    "basis_model_version",
    "method",
]

# ---------------------------------------------------------------------------
# Quarter arithmetic
# ---------------------------------------------------------------------------


def next_quarter(quarter: str, steps: int = 1) -> str:
    """Advance a YYYYQN quarter string by *steps* quarters.

    Args:
        quarter: Quarter string in YYYYQN format (e.g., '2025Q1').
        steps: Number of quarters to advance (can be negative).

    Returns:
        New quarter string.
    """
    year = int(quarter[:4])
    q = int(quarter[5])
    total = (year * 4 + (q - 1)) + steps
    new_year = total // 4
    new_q = (total % 4) + 1
    return f"{new_year}Q{new_q}"


def quarter_diff(q_later: str, q_earlier: str) -> int:
    """Return the number of quarters between two quarter strings.

    Args:
        q_later: Later quarter (YYYYQN).
        q_earlier: Earlier quarter (YYYYQN).

    Returns:
        Integer difference (positive if q_later is after q_earlier).
    """
    y1, q1 = int(q_later[:4]), int(q_later[5])
    y0, q0 = int(q_earlier[:4]), int(q_earlier[5])
    return (y1 * 4 + q1) - (y0 * 4 + q0)


# ---------------------------------------------------------------------------
# Conformal calibration
# ---------------------------------------------------------------------------


def compute_nonconformity_scores(
    history: pd.DataFrame,
) -> np.ndarray:
    """Compute nonconformity scores from resolved assessment history.

    The nonconformity score is ``|predicted_probability - actual_outcome|``
    for each resolved (outcome in {0, 1}) row.

    Args:
        history: Assessment history DataFrame with ``probability`` and
            ``actual_outcome`` columns.

    Returns:
        1-D array of nonconformity scores.  Empty array if no resolved rows.
    """
    resolved = history[history["actual_outcome"].isin([0, 1])].copy()
    if resolved.empty:
        return np.array([], dtype=float)

    probs = resolved["probability"].values.astype(float)
    outcomes = resolved["actual_outcome"].values.astype(float)
    return np.abs(probs - outcomes)


def conformal_interval_width(
    scores: np.ndarray,
    alpha: float = 0.10,
) -> float:
    """Compute the conformal prediction interval half-width.

    Uses the ceil((n+1)(1-alpha))/n quantile of nonconformity scores to
    guarantee marginal coverage >= (1 - alpha).

    Args:
        scores: Nonconformity scores from resolved predictions.
        alpha: Significance level (default 0.10 for 90% coverage).

    Returns:
        Interval half-width.  Returns 0.5 (maximum uncertainty) if
        fewer than 2 scores are available.
    """
    n = len(scores)
    if n < 2:
        return 0.5  # maximum uncertainty when no calibration data

    # Finite-sample correction for conformal coverage
    quantile_level = min(np.ceil((n + 1) * (1 - alpha)) / n, 1.0)
    return float(np.quantile(scores, quantile_level))


# ---------------------------------------------------------------------------
# Trend extrapolation
# ---------------------------------------------------------------------------


def compute_probability_trend(
    history: pd.DataFrame,
    lineage_id: int,
    n_lookback: int = 4,
) -> float:
    """Estimate per-quarter probability trend for a lineage.

    Fits a simple linear slope to the last *n_lookback* quarterly
    probabilities.  If fewer than 2 data points exist, returns 0.0
    (no trend).

    Args:
        history: Assessment history DataFrame.
        lineage_id: Lineage to compute trend for.
        n_lookback: Number of recent quarters to consider.

    Returns:
        Per-quarter slope of probability.
    """
    rows = history[history["lineage_id"] == lineage_id].copy()
    if len(rows) < 2:
        return 0.0

    # Sort by quarter_predicted, take last n_lookback
    rows = rows.sort_values("quarter_predicted").tail(n_lookback)
    if len(rows) < 2:
        return 0.0

    # Convert quarters to ordinal indices
    quarters_sorted = sorted(rows["quarter_predicted"].unique())
    q_to_idx = {q: i for i, q in enumerate(quarters_sorted)}

    # Average probability per quarter (in case multiple model versions)
    qp_probs = rows.groupby("quarter_predicted")["probability"].mean()
    if len(qp_probs) < 2:
        return 0.0

    x = np.array([q_to_idx[q] for q in qp_probs.index])
    y = qp_probs.values.astype(float)

    # Simple linear regression slope
    x_mean = x.mean()
    y_mean = y.mean()
    denom = ((x - x_mean) ** 2).sum()
    if denom == 0:
        return 0.0
    return float(((x - x_mean) * (y - y_mean)).sum() / denom)


# ---------------------------------------------------------------------------
# Forward estimates
# ---------------------------------------------------------------------------


def generate_horizon_estimates(
    latest_predictions: pd.DataFrame,
    history: pd.DataFrame,
    max_horizon: int = 4,
    alpha: float = 0.10,
    trend_lookback: int = 4,
    trend_damping: float = 0.7,
) -> pd.DataFrame:
    """Generate forward-looking probability estimates with conformal intervals.

    For each active lineage in *latest_predictions*, extrapolates onset
    probability for 1 to *max_horizon* quarters ahead using trend-adjusted
    persistence.  Confidence intervals are derived from conformal prediction
    calibrated on resolved assessment history.

    Args:
        latest_predictions: DataFrame with at least ``lineage_id``,
            ``quarter``, ``inflection_probability``, plus optionally
            ``model_version``.
        history: Assessment history DataFrame (from ``load_history()``).
        max_horizon: Maximum forecast horizon in quarters (1-4).
        alpha: Significance level for confidence intervals (0.10 = 90%).
        trend_lookback: Quarters of history to use for trend estimation.
        trend_damping: Per-step damping factor for trend extrapolation.
            Applied as ``trend * damping^h`` at horizon h.

    Returns:
        DataFrame with ``HORIZON_COLUMNS`` schema, one row per
        (lineage_id, horizon) combination.
    """
    required = {"lineage_id", "quarter", "inflection_probability"}
    missing = required - set(latest_predictions.columns)
    if missing:
        raise ValueError(f"Predictions missing required columns: {missing}")

    # Compute conformal interval width from historical residuals
    scores = compute_nonconformity_scores(history)
    half_width = conformal_interval_width(scores, alpha=alpha)
    confidence_level = 1.0 - alpha

    logger.info(
        "Conformal calibration: %d scores, half-width=%.3f at %.0f%% level",
        len(scores), half_width, confidence_level * 100,
    )

    rows: list[dict[str, Any]] = []

    for _, pred in latest_predictions.iterrows():
        lid = pred["lineage_id"]
        base_q = pred["quarter"]
        base_prob = float(pred["inflection_probability"])
        model_ver = pred.get("model_version", "unknown")

        # Compute per-quarter trend from history
        trend = compute_probability_trend(
            history, lid, n_lookback=trend_lookback,
        )

        for h in range(1, max_horizon + 1):
            target_q = next_quarter(base_q, h)

            # Trend-adjusted persistence forecast with damping
            damped_trend = trend * (trend_damping ** h)
            point_est = np.clip(base_prob + damped_trend * h, 0.0, 1.0)

            # Conformal interval
            lower = max(0.0, point_est - half_width)
            upper = min(1.0, point_est + half_width)

            rows.append({
                "lineage_id": lid,
                "quarter_target": target_q,
                "horizon": h,
                "point_estimate": round(float(point_est), 6),
                "lower_bound": round(lower, 6),
                "upper_bound": round(upper, 6),
                "confidence_level": confidence_level,
                "basis_quarter": base_q,
                "basis_model_version": model_ver,
                "method": "trend_persistence_conformal",
            })

    result = pd.DataFrame(rows, columns=HORIZON_COLUMNS)
    logger.info(
        "Generated %d horizon estimates for %d lineages",
        len(result), latest_predictions["lineage_id"].nunique(),
    )
    return result


def summarize_horizon_estimates(
    estimates: pd.DataFrame,
) -> dict[str, Any]:
    """Summarize horizon estimate statistics.

    Args:
        estimates: DataFrame from ``generate_horizon_estimates()``.

    Returns:
        Dictionary with summary statistics.
    """
    if estimates.empty:
        return {
            "n_lineages": 0,
            "n_estimates": 0,
            "max_horizon": 0,
            "mean_point_estimate": 0.0,
            "mean_interval_width": 0.0,
        }

    widths = estimates["upper_bound"] - estimates["lower_bound"]
    return {
        "n_lineages": int(estimates["lineage_id"].nunique()),
        "n_estimates": len(estimates),
        "max_horizon": int(estimates["horizon"].max()),
        "mean_point_estimate": round(float(estimates["point_estimate"].mean()), 4),
        "mean_interval_width": round(float(widths.mean()), 4),
    }
