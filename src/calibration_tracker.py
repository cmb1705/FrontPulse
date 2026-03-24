"""Calibration refinement and degradation tracking for MSD predictions.

Fits isotonic calibration on resolved prediction-outcome pairs, tracks
Expected Calibration Error (ECE) and Brier score across model versions,
and alerts when calibration degrades beyond historical norms.

Workflow:
1. Load assessment history with resolved outcomes.
2. Fit isotonic calibration on resolved (probability, outcome) pairs.
3. Apply calibration to transform raw probabilities to calibrated ones.
4. Compute and store per-version calibration metrics.
5. Detect degradation by comparing current metrics to trailing baselines.

The isotonic calibrator maps raw probabilities -> calibrated probabilities
via a monotonic step function fitted on historical data.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

logger = logging.getLogger(__name__)

# Degradation detection: alert if ECE exceeds trailing mean + k * std
DEFAULT_DEGRADATION_K = 2.0
MIN_RESOLVED_FOR_CALIBRATION = 20


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class CalibrationSnapshot:
    """Calibration metrics for a single model version."""

    model_version: str
    n_resolved: int = 0
    brier_score: float | None = None
    ece: float | None = None
    mean_predicted: float | None = None
    mean_observed: float | None = None
    is_calibrated: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Convert to serializable dictionary."""
        return asdict(self)


@dataclass
class DegradationAlert:
    """Alert when calibration degrades beyond historical norms."""

    model_version: str
    metric: str
    current_value: float
    baseline_mean: float
    baseline_std: float
    threshold: float
    severity: str = "warning"  # "warning" or "critical"

    def to_dict(self) -> dict[str, Any]:
        """Convert to serializable dictionary."""
        return asdict(self)


@dataclass
class CalibrationHistory:
    """Tracks calibration snapshots across model versions."""

    snapshots: list[CalibrationSnapshot] = field(default_factory=list)

    def add_snapshot(self, snapshot: CalibrationSnapshot) -> None:
        """Add or replace a calibration snapshot for a model version."""
        self.snapshots = [
            s for s in self.snapshots
            if s.model_version != snapshot.model_version
        ]
        self.snapshots.append(snapshot)

    def get_trailing_stats(
        self, metric: str, n_trailing: int = 5,
    ) -> tuple[float, float]:
        """Compute trailing mean and std for a metric.

        Args:
            metric: Attribute name on CalibrationSnapshot.
            n_trailing: Number of recent snapshots to consider.

        Returns:
            Tuple of (mean, std). Returns (0.0, 1.0) if insufficient data.
        """
        values = [
            getattr(s, metric) for s in self.snapshots[-n_trailing:]
            if getattr(s, metric, None) is not None
        ]
        if len(values) < 2:
            return 0.0, 1.0
        return float(np.mean(values)), float(np.std(values))

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {"snapshots": [s.to_dict() for s in self.snapshots]}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CalibrationHistory:
        """Deserialize from dictionary."""
        snapshots = [
            CalibrationSnapshot(**s) for s in data.get("snapshots", [])
        ]
        return cls(snapshots=snapshots)


# ---------------------------------------------------------------------------
# Isotonic calibration
# ---------------------------------------------------------------------------


def fit_isotonic_calibrator(
    probabilities: np.ndarray,
    outcomes: np.ndarray,
) -> IsotonicRegression:
    """Fit an isotonic regression calibrator on resolved predictions.

    Args:
        probabilities: Raw predicted probabilities.
        outcomes: Binary ground-truth outcomes (0 or 1).

    Returns:
        Fitted IsotonicRegression model.

    Raises:
        ValueError: If fewer than MIN_RESOLVED_FOR_CALIBRATION samples.
    """
    if len(probabilities) < MIN_RESOLVED_FOR_CALIBRATION:
        raise ValueError(
            f"Need at least {MIN_RESOLVED_FOR_CALIBRATION} resolved predictions"
            f" for calibration, got {len(probabilities)}"
        )

    ir = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    ir.fit(probabilities, outcomes)
    logger.info("Fitted isotonic calibrator on %d samples", len(probabilities))
    return ir


def apply_calibration(
    raw_probabilities: np.ndarray,
    calibrator: IsotonicRegression,
) -> np.ndarray:
    """Apply isotonic calibration to raw probabilities.

    Args:
        raw_probabilities: Uncalibrated probability array.
        calibrator: Fitted IsotonicRegression model.

    Returns:
        Calibrated probability array clipped to [0, 1].
    """
    calibrated = calibrator.predict(raw_probabilities)
    return np.clip(calibrated, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Calibration metrics
# ---------------------------------------------------------------------------


def compute_ece(
    probabilities: np.ndarray,
    outcomes: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Compute Expected Calibration Error.

    Args:
        probabilities: Predicted probabilities.
        outcomes: Binary ground-truth outcomes.
        n_bins: Number of bins for calibration curve.

    Returns:
        ECE value (lower is better).
    """
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    total = len(probabilities)

    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        if i < n_bins - 1:
            mask = (probabilities >= lo) & (probabilities < hi)
        else:
            mask = (probabilities >= lo) & (probabilities <= hi)

        count = mask.sum()
        if count == 0:
            continue

        pred_mean = probabilities[mask].mean()
        obs_rate = outcomes[mask].mean()
        ece += abs(pred_mean - obs_rate) * count

    return float(ece / max(total, 1))


def compute_brier_score(
    probabilities: np.ndarray,
    outcomes: np.ndarray,
) -> float:
    """Compute Brier score (mean squared error of probabilities).

    Args:
        probabilities: Predicted probabilities.
        outcomes: Binary ground-truth outcomes.

    Returns:
        Brier score (lower is better, 0.0 = perfect).
    """
    return float(np.mean((probabilities - outcomes) ** 2))


def compute_calibration_snapshot(
    history: pd.DataFrame,
    model_version: str,
    calibrator: IsotonicRegression | None = None,
) -> CalibrationSnapshot:
    """Compute calibration metrics for a specific model version.

    Args:
        history: Assessment history DataFrame.
        model_version: Version to compute metrics for.
        calibrator: Optional fitted calibrator. If provided, metrics are
            computed on calibrated probabilities.

    Returns:
        CalibrationSnapshot with metrics.
    """
    version_rows = history[history["model_version"] == model_version].copy()
    resolved = version_rows[version_rows["actual_outcome"].isin([0, 1])].copy()

    if len(resolved) < 2:
        return CalibrationSnapshot(
            model_version=model_version,
            n_resolved=len(resolved),
        )

    probs = resolved["probability"].values.astype(float)
    outcomes = resolved["actual_outcome"].values.astype(float)

    if calibrator is not None:
        probs = apply_calibration(probs, calibrator)

    return CalibrationSnapshot(
        model_version=model_version,
        n_resolved=len(resolved),
        brier_score=round(compute_brier_score(probs, outcomes), 6),
        ece=round(compute_ece(probs, outcomes), 6),
        mean_predicted=round(float(probs.mean()), 6),
        mean_observed=round(float(outcomes.mean()), 6),
        is_calibrated=calibrator is not None,
    )


# ---------------------------------------------------------------------------
# Degradation detection
# ---------------------------------------------------------------------------


def check_degradation(
    current: CalibrationSnapshot,
    cal_history: CalibrationHistory,
    k: float = DEFAULT_DEGRADATION_K,
    n_trailing: int = 5,
) -> list[DegradationAlert]:
    """Check if current calibration has degraded beyond historical norms.

    Compares ECE and Brier score against trailing statistics. An alert
    fires if the current value exceeds ``trailing_mean + k * trailing_std``.

    Args:
        current: Current model version's calibration snapshot.
        cal_history: Historical calibration snapshots.
        k: Number of standard deviations for threshold.
        n_trailing: Number of historical snapshots for baseline.

    Returns:
        List of DegradationAlert objects (empty if no degradation).
    """
    alerts: list[DegradationAlert] = []

    for metric in ("ece", "brier_score"):
        current_val = getattr(current, metric)
        if current_val is None:
            continue

        mean, std = cal_history.get_trailing_stats(metric, n_trailing)
        threshold = mean + k * std

        if current_val > threshold and mean > 0:
            severity = "critical" if current_val > mean + 3 * std else "warning"
            alerts.append(DegradationAlert(
                model_version=current.model_version,
                metric=metric,
                current_value=current_val,
                baseline_mean=round(mean, 6),
                baseline_std=round(std, 6),
                threshold=round(threshold, 6),
                severity=severity,
            ))

    return alerts


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def save_calibration_history(
    cal_history: CalibrationHistory,
    path: Path,
) -> None:
    """Save calibration history to JSON.

    Args:
        cal_history: CalibrationHistory to save.
        path: Output JSON file path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cal_history.to_dict(), indent=2))
    logger.info("Saved calibration history (%d snapshots) to %s",
                len(cal_history.snapshots), path)


def load_calibration_history(path: Path) -> CalibrationHistory:
    """Load calibration history from JSON.

    Args:
        path: Path to JSON file.

    Returns:
        CalibrationHistory (empty if file not found).
    """
    if not path.exists():
        logger.info("No calibration history at %s, starting fresh", path)
        return CalibrationHistory()

    data = json.loads(path.read_text())
    return CalibrationHistory.from_dict(data)
