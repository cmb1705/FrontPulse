"""Tests for forward-looking horizon estimates with conformal intervals."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.assessment_history import OUTCOME_UNKNOWN
from src.horizon_estimates import (
    HORIZON_COLUMNS,
    compute_nonconformity_scores,
    compute_probability_trend,
    conformal_interval_width,
    generate_horizon_estimates,
    next_quarter,
    quarter_diff,
    summarize_horizon_estimates,
)

# ---------------------------------------------------------------------------
# Quarter arithmetic
# ---------------------------------------------------------------------------


class TestQuarterArithmetic:
    """Tests for quarter string manipulation."""

    def test_next_quarter_simple(self) -> None:
        assert next_quarter("2025Q1", 1) == "2025Q2"
        assert next_quarter("2025Q2", 1) == "2025Q3"
        assert next_quarter("2025Q3", 1) == "2025Q4"

    def test_next_quarter_year_boundary(self) -> None:
        assert next_quarter("2025Q4", 1) == "2026Q1"
        assert next_quarter("2024Q4", 2) == "2025Q2"

    def test_next_quarter_multiple_steps(self) -> None:
        assert next_quarter("2025Q1", 4) == "2026Q1"
        assert next_quarter("2025Q1", 8) == "2027Q1"

    def test_next_quarter_negative(self) -> None:
        assert next_quarter("2025Q2", -1) == "2025Q1"
        assert next_quarter("2025Q1", -1) == "2024Q4"

    def test_quarter_diff(self) -> None:
        assert quarter_diff("2025Q2", "2025Q1") == 1
        assert quarter_diff("2026Q1", "2025Q1") == 4
        assert quarter_diff("2025Q1", "2025Q2") == -1


# ---------------------------------------------------------------------------
# Nonconformity scores
# ---------------------------------------------------------------------------


class TestNonconformityScores:
    """Tests for conformal calibration scores."""

    def test_perfect_predictions_score_zero(self) -> None:
        history = pd.DataFrame({
            "probability": [1.0, 0.0, 1.0, 0.0],
            "actual_outcome": [1, 0, 1, 0],
        })
        scores = compute_nonconformity_scores(history)
        np.testing.assert_array_almost_equal(scores, [0.0, 0.0, 0.0, 0.0])

    def test_worst_predictions_score_one(self) -> None:
        history = pd.DataFrame({
            "probability": [0.0, 1.0],
            "actual_outcome": [1, 0],
        })
        scores = compute_nonconformity_scores(history)
        np.testing.assert_array_almost_equal(scores, [1.0, 1.0])

    def test_ignores_unknown_outcomes(self) -> None:
        history = pd.DataFrame({
            "probability": [0.8, 0.2, 0.5],
            "actual_outcome": [1, 0, OUTCOME_UNKNOWN],
        })
        scores = compute_nonconformity_scores(history)
        assert len(scores) == 2

    def test_empty_history_returns_empty(self) -> None:
        history = pd.DataFrame({
            "probability": [0.5],
            "actual_outcome": [OUTCOME_UNKNOWN],
        })
        scores = compute_nonconformity_scores(history)
        assert len(scores) == 0


# ---------------------------------------------------------------------------
# Conformal interval width
# ---------------------------------------------------------------------------


class TestConformalIntervalWidth:
    """Tests for conformal interval computation."""

    def test_few_scores_returns_max_uncertainty(self) -> None:
        assert conformal_interval_width(np.array([]), alpha=0.10) == 0.5
        assert conformal_interval_width(np.array([0.1]), alpha=0.10) == 0.5

    def test_tight_scores_give_narrow_interval(self) -> None:
        scores = np.array([0.05, 0.03, 0.04, 0.02, 0.06, 0.01, 0.03, 0.04, 0.05, 0.02])
        width = conformal_interval_width(scores, alpha=0.10)
        assert width < 0.10  # scores are all small

    def test_wide_scores_give_wide_interval(self) -> None:
        scores = np.array([0.8, 0.9, 0.7, 0.85, 0.95])
        width = conformal_interval_width(scores, alpha=0.10)
        assert width > 0.5

    def test_lower_alpha_gives_wider_interval(self) -> None:
        scores = np.linspace(0.0, 1.0, 100)
        w_90 = conformal_interval_width(scores, alpha=0.10)
        w_95 = conformal_interval_width(scores, alpha=0.05)
        assert w_95 >= w_90


# ---------------------------------------------------------------------------
# Trend computation
# ---------------------------------------------------------------------------


class TestProbabilityTrend:
    """Tests for per-lineage probability trend estimation."""

    def test_increasing_trend_positive(self) -> None:
        history = pd.DataFrame({
            "lineage_id": [1, 1, 1, 1],
            "quarter_predicted": ["2024Q1", "2024Q2", "2024Q3", "2024Q4"],
            "probability": [0.1, 0.2, 0.3, 0.4],
        })
        trend = compute_probability_trend(history, lineage_id=1)
        assert trend > 0

    def test_decreasing_trend_negative(self) -> None:
        history = pd.DataFrame({
            "lineage_id": [1, 1, 1, 1],
            "quarter_predicted": ["2024Q1", "2024Q2", "2024Q3", "2024Q4"],
            "probability": [0.8, 0.6, 0.4, 0.2],
        })
        trend = compute_probability_trend(history, lineage_id=1)
        assert trend < 0

    def test_single_observation_returns_zero(self) -> None:
        history = pd.DataFrame({
            "lineage_id": [1],
            "quarter_predicted": ["2024Q1"],
            "probability": [0.5],
        })
        trend = compute_probability_trend(history, lineage_id=1)
        assert trend == 0.0

    def test_missing_lineage_returns_zero(self) -> None:
        history = pd.DataFrame({
            "lineage_id": [1, 1],
            "quarter_predicted": ["2024Q1", "2024Q2"],
            "probability": [0.3, 0.4],
        })
        trend = compute_probability_trend(history, lineage_id=999)
        assert trend == 0.0

    def test_constant_probability_zero_trend(self) -> None:
        history = pd.DataFrame({
            "lineage_id": [1, 1, 1],
            "quarter_predicted": ["2024Q1", "2024Q2", "2024Q3"],
            "probability": [0.5, 0.5, 0.5],
        })
        trend = compute_probability_trend(history, lineage_id=1)
        assert trend == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Generate horizon estimates
# ---------------------------------------------------------------------------


class TestGenerateHorizonEstimates:
    """Tests for the main estimation function."""

    def _make_predictions(self, n: int = 3) -> pd.DataFrame:
        return pd.DataFrame({
            "lineage_id": list(range(1, n + 1)),
            "quarter": ["2025Q1"] * n,
            "inflection_probability": np.linspace(0.1, 0.9, n),
            "model_version": ["v_test"] * n,
        })

    def _make_empty_history(self) -> pd.DataFrame:
        from src.assessment_history import create_empty_history
        return create_empty_history()

    def test_output_schema(self) -> None:
        preds = self._make_predictions()
        estimates = generate_horizon_estimates(
            preds, self._make_empty_history(), max_horizon=4,
        )
        assert list(estimates.columns) == HORIZON_COLUMNS

    def test_row_count(self) -> None:
        preds = self._make_predictions(5)
        estimates = generate_horizon_estimates(
            preds, self._make_empty_history(), max_horizon=4,
        )
        assert len(estimates) == 5 * 4  # 5 lineages x 4 horizons

    def test_horizons_are_1_to_max(self) -> None:
        preds = self._make_predictions(1)
        estimates = generate_horizon_estimates(
            preds, self._make_empty_history(), max_horizon=3,
        )
        assert sorted(estimates["horizon"].unique()) == [1, 2, 3]

    def test_bounds_within_zero_one(self) -> None:
        preds = self._make_predictions(10)
        estimates = generate_horizon_estimates(
            preds, self._make_empty_history(), max_horizon=4,
        )
        assert (estimates["lower_bound"] >= 0.0).all()
        assert (estimates["upper_bound"] <= 1.0).all()
        assert (estimates["lower_bound"] <= estimates["upper_bound"]).all()

    def test_point_estimate_within_bounds(self) -> None:
        preds = self._make_predictions(5)
        estimates = generate_horizon_estimates(
            preds, self._make_empty_history(), max_horizon=4,
        )
        assert (estimates["point_estimate"] >= estimates["lower_bound"]).all()
        assert (estimates["point_estimate"] <= estimates["upper_bound"]).all()

    def test_quarter_targets_advance(self) -> None:
        preds = pd.DataFrame({
            "lineage_id": [1],
            "quarter": ["2025Q1"],
            "inflection_probability": [0.5],
        })
        estimates = generate_horizon_estimates(
            preds, self._make_empty_history(), max_horizon=4,
        )
        targets = estimates["quarter_target"].tolist()
        assert targets == ["2025Q2", "2025Q3", "2025Q4", "2026Q1"]

    def test_missing_columns_raises(self) -> None:
        bad_preds = pd.DataFrame({"lineage_id": [1]})
        with pytest.raises(ValueError, match="missing required columns"):
            generate_horizon_estimates(
                bad_preds, self._make_empty_history(),
            )

    def test_calibrated_intervals_narrower_with_good_history(self) -> None:
        """Well-calibrated history should produce narrower intervals."""
        preds = self._make_predictions(3)
        empty_history = self._make_empty_history()

        # History with tight calibration (low nonconformity scores)
        good_history = pd.DataFrame({
            "lineage_id": list(range(50)),
            "quarter_predicted": ["2024Q4"] * 50,
            "quarter_assessed": ["2025Q1"] * 50,
            "model_version": ["v_test"] * 50,
            "probability": [0.05] * 25 + [0.95] * 25,
            "predicted_label": [0] * 25 + [1] * 25,
            "threshold_used": [0.5] * 50,
            "actual_outcome": [0] * 25 + [1] * 25,
            "outcome_source": ["test"] * 50,
            "backfilled_at": [""] * 50,
        })

        est_uncal = generate_horizon_estimates(preds, empty_history)
        est_cal = generate_horizon_estimates(preds, good_history)

        avg_width_uncal = (est_uncal["upper_bound"] - est_uncal["lower_bound"]).mean()
        avg_width_cal = (est_cal["upper_bound"] - est_cal["lower_bound"]).mean()
        assert avg_width_cal < avg_width_uncal


# ---------------------------------------------------------------------------
# Summarize
# ---------------------------------------------------------------------------


class TestSummarizeHorizonEstimates:
    """Tests for summary statistics."""

    def test_empty_estimates(self) -> None:
        empty = pd.DataFrame(columns=HORIZON_COLUMNS)
        summary = summarize_horizon_estimates(empty)
        assert summary["n_lineages"] == 0
        assert summary["n_estimates"] == 0

    def test_summary_values(self) -> None:
        estimates = pd.DataFrame({
            "lineage_id": [1, 1, 2, 2],
            "quarter_target": ["2025Q2", "2025Q3", "2025Q2", "2025Q3"],
            "horizon": [1, 2, 1, 2],
            "point_estimate": [0.3, 0.4, 0.5, 0.6],
            "lower_bound": [0.1, 0.2, 0.3, 0.4],
            "upper_bound": [0.5, 0.6, 0.7, 0.8],
            "confidence_level": [0.9] * 4,
            "basis_quarter": ["2025Q1"] * 4,
            "basis_model_version": ["v_test"] * 4,
            "method": ["test"] * 4,
        })
        summary = summarize_horizon_estimates(estimates)
        assert summary["n_lineages"] == 2
        assert summary["n_estimates"] == 4
        assert summary["max_horizon"] == 2
        assert summary["mean_point_estimate"] == pytest.approx(0.45)
        assert summary["mean_interval_width"] == pytest.approx(0.4)
