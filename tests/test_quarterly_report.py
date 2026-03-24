"""Tests for quarterly briefing report generator."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.assessment_history import OUTCOME_UNKNOWN, create_empty_history
from src.quarterly_report import (
    EXTENDED_MONITORING_THRESHOLD,
    WATCH_LIST_THRESHOLD,
    classify_alerts,
    generate_quarterly_report,
    summarize_report_stats,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_predictions(n: int = 20) -> pd.DataFrame:
    """Create synthetic predictions spanning all tiers."""
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "lineage_id": list(range(1, n + 1)),
        "quarter": ["2025Q1"] * n,
        "inflection_probability": rng.uniform(0.0, 1.0, n),
        "model_version": ["v_test"] * n,
    })


def _make_history() -> pd.DataFrame:
    """Create a small assessment history with some resolved outcomes."""
    return pd.DataFrame({
        "lineage_id": [1, 2, 3, 4],
        "quarter_predicted": ["2024Q4"] * 4,
        "quarter_assessed": ["2025Q1"] * 4,
        "model_version": ["v_test"] * 4,
        "probability": [0.8, 0.3, 0.1, 0.5],
        "predicted_label": [1, 1, 0, 1],
        "threshold_used": [0.15] * 4,
        "actual_outcome": [1, 0, 0, OUTCOME_UNKNOWN],
        "outcome_source": ["test", "test", "test", ""],
        "backfilled_at": ["2025-01-01", "2025-01-01", "2025-01-01", ""],
    })


# ---------------------------------------------------------------------------
# classify_alerts
# ---------------------------------------------------------------------------


class TestClassifyAlerts:
    """Tests for two-tier alert classification."""

    def test_watch_list_classification(self) -> None:
        preds = pd.DataFrame({
            "lineage_id": [1],
            "inflection_probability": [0.20],
        })
        result = classify_alerts(preds)
        assert result["alert_tier"].iloc[0] == "watch_list"

    def test_extended_monitoring_classification(self) -> None:
        preds = pd.DataFrame({
            "lineage_id": [1],
            "inflection_probability": [0.10],
        })
        result = classify_alerts(preds)
        assert result["alert_tier"].iloc[0] == "extended_monitoring"

    def test_below_threshold_classification(self) -> None:
        preds = pd.DataFrame({
            "lineage_id": [1],
            "inflection_probability": [0.03],
        })
        result = classify_alerts(preds)
        assert result["alert_tier"].iloc[0] == "below_threshold"

    def test_boundary_at_watch_threshold(self) -> None:
        preds = pd.DataFrame({
            "lineage_id": [1],
            "inflection_probability": [WATCH_LIST_THRESHOLD],
        })
        result = classify_alerts(preds)
        assert result["alert_tier"].iloc[0] == "watch_list"

    def test_boundary_at_extended_threshold(self) -> None:
        preds = pd.DataFrame({
            "lineage_id": [1],
            "inflection_probability": [EXTENDED_MONITORING_THRESHOLD],
        })
        result = classify_alerts(preds)
        assert result["alert_tier"].iloc[0] == "extended_monitoring"

    def test_all_tiers_present(self) -> None:
        preds = pd.DataFrame({
            "lineage_id": [1, 2, 3],
            "inflection_probability": [0.50, 0.10, 0.01],
        })
        result = classify_alerts(preds)
        tiers = set(result["alert_tier"])
        assert tiers == {"watch_list", "extended_monitoring", "below_threshold"}


# ---------------------------------------------------------------------------
# generate_quarterly_report
# ---------------------------------------------------------------------------


class TestGenerateQuarterlyReport:
    """Tests for report generation."""

    def test_report_is_markdown_string(self) -> None:
        preds = _make_predictions()
        report = generate_quarterly_report(
            preds, create_empty_history(),
            quarter_assessed="2025Q1", model_version="v_test",
        )
        assert isinstance(report, str)
        assert report.startswith("# Quarterly Briefing Report")

    def test_report_contains_all_sections(self) -> None:
        preds = _make_predictions()
        history = _make_history()
        report = generate_quarterly_report(
            preds, history,
            quarter_assessed="2025Q1", model_version="v_test",
        )
        assert "## Alert Summary" in report
        assert "## Watch List" in report
        assert "## Extended Monitoring" in report
        assert "## Assessment Updates" in report
        assert "## Calibration Diagnostics" in report
        assert "## Model Performance" in report

    def test_report_includes_model_version(self) -> None:
        preds = _make_predictions(5)
        report = generate_quarterly_report(
            preds, create_empty_history(),
            quarter_assessed="2025Q1", model_version="v_20260323_001",
        )
        assert "v_20260323_001" in report

    def test_report_with_horizon_estimates(self) -> None:
        preds = _make_predictions(5)
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
        report = generate_quarterly_report(
            preds, create_empty_history(),
            quarter_assessed="2025Q1", model_version="v_test",
            horizon_estimates=estimates,
        )
        assert "## Forward-Looking Estimates" in report
        assert "Point Est." in report

    def test_report_with_calibration_stats(self) -> None:
        preds = _make_predictions(5)
        cal = {
            "n_resolved": 100,
            "n_unknown": 50,
            "brier_score": 0.05,
            "calibration_error": 0.03,
            "bins": [
                {"bin_center": 0.05, "predicted_mean": 0.04, "observed_rate": 0.03, "count": 20},
                {"bin_center": 0.95, "predicted_mean": 0.92, "observed_rate": 0.90, "count": 10},
            ],
        }
        report = generate_quarterly_report(
            preds, create_empty_history(),
            quarter_assessed="2025Q1", model_version="v_test",
            calibration_stats=cal,
        )
        assert "Brier score" in report
        assert "0.0500" in report

    def test_report_with_no_alerts(self) -> None:
        preds = pd.DataFrame({
            "lineage_id": [1, 2],
            "quarter": ["2025Q1", "2025Q1"],
            "inflection_probability": [0.01, 0.02],
        })
        report = generate_quarterly_report(
            preds, create_empty_history(),
            quarter_assessed="2025Q1", model_version="v_test",
        )
        assert "No lineages above watch list threshold" in report

    def test_report_with_model_comparison(self) -> None:
        preds = _make_predictions(5)
        comparison = {
            "improved": True,
            "primary_metric": "cv_pr_auc_mean",
            "deltas": {"cv_pr_auc_mean": 0.02, "cv_roc_auc_mean": 0.01},
        }
        report = generate_quarterly_report(
            preds, create_empty_history(),
            quarter_assessed="2025Q1", model_version="v_002",
            model_comparison=comparison, previous_version="v_001",
        )
        assert "improved" in report.lower()
        assert "v_001" in report


# ---------------------------------------------------------------------------
# summarize_report_stats
# ---------------------------------------------------------------------------


class TestSummarizeReportStats:
    """Tests for report summary statistics."""

    def test_tier_counts_sum_to_total(self) -> None:
        preds = _make_predictions(50)
        stats = summarize_report_stats(preds)
        total = (
            stats["watch_list_count"]
            + stats["extended_monitoring_count"]
            + stats["below_threshold_count"]
        )
        assert total == stats["total_lineages"]

    def test_probability_range(self) -> None:
        preds = _make_predictions(20)
        stats = summarize_report_stats(preds)
        assert 0.0 <= stats["mean_probability"] <= 1.0
        assert 0.0 <= stats["max_probability"] <= 1.0
