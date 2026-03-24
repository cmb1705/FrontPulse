"""Tests for calibration refinement and degradation tracking."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.assessment_history import OUTCOME_UNKNOWN
from src.calibration_tracker import (
    CalibrationHistory,
    CalibrationSnapshot,
    apply_calibration,
    check_degradation,
    compute_brier_score,
    compute_calibration_snapshot,
    compute_ece,
    fit_isotonic_calibrator,
    load_calibration_history,
    save_calibration_history,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_resolved_history(
    n: int = 100,
    model_version: str = "v_test",
) -> pd.DataFrame:
    """Create a history with resolved outcomes for calibration testing."""
    rng = np.random.default_rng(42)
    probs = rng.uniform(0, 1, n)
    outcomes = (rng.uniform(0, 1, n) < probs).astype(int)
    return pd.DataFrame({
        "lineage_id": list(range(n)),
        "quarter_predicted": ["2025Q1"] * n,
        "quarter_assessed": ["2025Q2"] * n,
        "model_version": [model_version] * n,
        "probability": probs,
        "predicted_label": (probs >= 0.5).astype(int),
        "threshold_used": [0.15] * n,
        "actual_outcome": outcomes,
        "outcome_source": ["test"] * n,
        "backfilled_at": ["2025-06-01"] * n,
    })


# ---------------------------------------------------------------------------
# Isotonic calibration
# ---------------------------------------------------------------------------


class TestIsotonicCalibration:
    """Tests for isotonic calibration fitting and application."""

    def test_fit_requires_minimum_samples(self) -> None:
        probs = np.array([0.5, 0.6])
        outcomes = np.array([0, 1])
        with pytest.raises(ValueError, match="at least"):
            fit_isotonic_calibrator(probs, outcomes)

    def test_fit_and_apply_preserves_range(self) -> None:
        rng = np.random.default_rng(0)
        n = 100
        probs = rng.uniform(0, 1, n)
        outcomes = (rng.uniform(0, 1, n) < probs).astype(int)

        calibrator = fit_isotonic_calibrator(probs, outcomes)
        calibrated = apply_calibration(probs, calibrator)

        assert (calibrated >= 0.0).all()
        assert (calibrated <= 1.0).all()

    def test_calibrated_output_is_monotonic(self) -> None:
        rng = np.random.default_rng(1)
        n = 200
        probs = rng.uniform(0, 1, n)
        outcomes = (rng.uniform(0, 1, n) < probs).astype(int)

        calibrator = fit_isotonic_calibrator(probs, outcomes)

        # Test on sorted input -- output should also be non-decreasing
        test_probs = np.linspace(0.0, 1.0, 50)
        calibrated = apply_calibration(test_probs, calibrator)

        diffs = np.diff(calibrated)
        assert (diffs >= -1e-10).all(), "Calibrated output must be monotonic"


# ---------------------------------------------------------------------------
# ECE and Brier score
# ---------------------------------------------------------------------------


class TestCalibrationMetrics:
    """Tests for ECE and Brier score computation."""

    def test_perfect_brier_is_zero(self) -> None:
        probs = np.array([1.0, 0.0, 1.0, 0.0])
        outcomes = np.array([1, 0, 1, 0])
        assert compute_brier_score(probs, outcomes) == pytest.approx(0.0)

    def test_worst_brier_is_one(self) -> None:
        probs = np.array([0.0, 1.0])
        outcomes = np.array([1, 0])
        assert compute_brier_score(probs, outcomes) == pytest.approx(1.0)

    def test_perfect_ece_is_zero(self) -> None:
        # All predictions in one bin, perfectly calibrated
        probs = np.array([0.5, 0.5, 0.5, 0.5])
        outcomes = np.array([1, 0, 1, 0])
        ece = compute_ece(probs, outcomes, n_bins=10)
        assert ece == pytest.approx(0.0)

    def test_ece_bounded_by_one(self) -> None:
        rng = np.random.default_rng(42)
        probs = rng.uniform(0, 1, 100)
        outcomes = rng.integers(0, 2, 100).astype(float)
        ece = compute_ece(probs, outcomes)
        assert 0.0 <= ece <= 1.0


# ---------------------------------------------------------------------------
# Calibration snapshot
# ---------------------------------------------------------------------------


class TestCalibrationSnapshot:
    """Tests for per-version calibration snapshot computation."""

    def test_snapshot_with_resolved_data(self) -> None:
        history = _make_resolved_history(100, "v_001")
        snapshot = compute_calibration_snapshot(history, "v_001")
        assert snapshot.n_resolved == 100
        assert snapshot.brier_score is not None
        assert snapshot.ece is not None
        assert not snapshot.is_calibrated

    def test_snapshot_insufficient_data(self) -> None:
        history = pd.DataFrame({
            "lineage_id": [1],
            "model_version": ["v_001"],
            "probability": [0.5],
            "actual_outcome": [OUTCOME_UNKNOWN],
        })
        snapshot = compute_calibration_snapshot(history, "v_001")
        assert snapshot.n_resolved == 0
        assert snapshot.brier_score is None

    def test_snapshot_wrong_version_empty(self) -> None:
        history = _make_resolved_history(50, "v_001")
        snapshot = compute_calibration_snapshot(history, "v_999")
        assert snapshot.n_resolved == 0

    def test_snapshot_serialization(self) -> None:
        snapshot = CalibrationSnapshot(
            model_version="v_test",
            n_resolved=100,
            brier_score=0.05,
            ece=0.03,
        )
        d = snapshot.to_dict()
        assert d["model_version"] == "v_test"
        assert d["brier_score"] == 0.05


# ---------------------------------------------------------------------------
# Degradation detection
# ---------------------------------------------------------------------------


class TestDegradationDetection:
    """Tests for calibration degradation alerting."""

    def _make_history_with_snapshots(
        self, ece_values: list[float],
    ) -> CalibrationHistory:
        cal = CalibrationHistory()
        for i, ece in enumerate(ece_values):
            cal.add_snapshot(CalibrationSnapshot(
                model_version=f"v_{i:03d}",
                n_resolved=100,
                brier_score=0.05,
                ece=ece,
            ))
        return cal

    def test_no_alert_within_bounds(self) -> None:
        cal_hist = self._make_history_with_snapshots([0.03, 0.04, 0.035, 0.04, 0.03])
        current = CalibrationSnapshot(
            model_version="v_new", n_resolved=100,
            brier_score=0.05, ece=0.04,
        )
        alerts = check_degradation(current, cal_hist)
        # ECE 0.04 is within normal range
        ece_alerts = [a for a in alerts if a.metric == "ece"]
        assert len(ece_alerts) == 0

    def test_alert_on_high_ece(self) -> None:
        cal_hist = self._make_history_with_snapshots([0.03, 0.03, 0.03, 0.03, 0.03])
        current = CalibrationSnapshot(
            model_version="v_bad", n_resolved=100,
            brier_score=0.05, ece=0.20,
        )
        alerts = check_degradation(current, cal_hist)
        ece_alerts = [a for a in alerts if a.metric == "ece"]
        assert len(ece_alerts) == 1
        assert ece_alerts[0].severity in ("warning", "critical")

    def test_no_alert_with_empty_history(self) -> None:
        cal_hist = CalibrationHistory()
        current = CalibrationSnapshot(
            model_version="v_first", n_resolved=100,
            brier_score=0.10, ece=0.08,
        )
        alerts = check_degradation(current, cal_hist)
        assert len(alerts) == 0

    def test_no_alert_when_metrics_none(self) -> None:
        cal_hist = self._make_history_with_snapshots([0.03, 0.04])
        current = CalibrationSnapshot(
            model_version="v_new", n_resolved=0,
        )
        alerts = check_degradation(current, cal_hist)
        assert len(alerts) == 0


# ---------------------------------------------------------------------------
# Calibration history persistence
# ---------------------------------------------------------------------------


class TestCalibrationHistoryPersistence:
    """Tests for save/load calibration history."""

    def test_round_trip(self, tmp_path: Path) -> None:
        cal = CalibrationHistory()
        cal.add_snapshot(CalibrationSnapshot(
            model_version="v_001", n_resolved=50,
            brier_score=0.05, ece=0.03,
        ))
        cal.add_snapshot(CalibrationSnapshot(
            model_version="v_002", n_resolved=100,
            brier_score=0.04, ece=0.02,
        ))

        path = tmp_path / "cal_history.json"
        save_calibration_history(cal, path)
        loaded = load_calibration_history(path)

        assert len(loaded.snapshots) == 2
        assert loaded.snapshots[0].model_version == "v_001"
        assert loaded.snapshots[1].brier_score == 0.04

    def test_load_nonexistent_returns_empty(self, tmp_path: Path) -> None:
        loaded = load_calibration_history(tmp_path / "no_such.json")
        assert len(loaded.snapshots) == 0

    def test_add_replaces_existing_version(self) -> None:
        cal = CalibrationHistory()
        cal.add_snapshot(CalibrationSnapshot(
            model_version="v_001", n_resolved=50, ece=0.05,
        ))
        cal.add_snapshot(CalibrationSnapshot(
            model_version="v_001", n_resolved=100, ece=0.03,
        ))
        assert len(cal.snapshots) == 1
        assert cal.snapshots[0].n_resolved == 100

    def test_trailing_stats_with_few_snapshots(self) -> None:
        cal = CalibrationHistory()
        cal.add_snapshot(CalibrationSnapshot(
            model_version="v_001", n_resolved=50, ece=0.05,
        ))
        mean, std = cal.get_trailing_stats("ece")
        # Only 1 snapshot, returns default (0.0, 1.0)
        assert mean == 0.0
        assert std == 1.0
