"""Tests for longitudinal assessment history tracking."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.assessment_history import (
    ASSESSMENT_COLUMNS,
    OUTCOME_UNKNOWN,
    append_assessments,
    backfill_outcomes,
    compute_calibration_stats,
    create_empty_history,
    load_history,
    record_assessments,
    save_history,
    summarize_history,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_predictions(n: int = 5) -> pd.DataFrame:
    """Create a synthetic predictions DataFrame."""
    return pd.DataFrame({
        "lineage_id": list(range(1, n + 1)),
        "quarter": ["2025Q1"] * n,
        "inflection_probability": np.linspace(0.05, 0.95, n),
    })


def _make_history_with_unknowns() -> pd.DataFrame:
    """Create a history with some unknown outcomes."""
    return pd.DataFrame({
        "lineage_id": [1, 2, 3, 4],
        "quarter_predicted": ["2025Q1", "2025Q1", "2025Q1", "2025Q1"],
        "quarter_assessed": ["2025Q2"] * 4,
        "model_version": ["v_20260323_001"] * 4,
        "probability": [0.8, 0.3, 0.1, 0.6],
        "predicted_label": [1, 1, 0, 1],
        "threshold_used": [0.15] * 4,
        "actual_outcome": [OUTCOME_UNKNOWN, OUTCOME_UNKNOWN, OUTCOME_UNKNOWN, 1],
        "outcome_source": ["", "", "", "manual"],
        "backfilled_at": ["", "", "", "2026-01-01T00:00:00Z"],
    })


def _make_labels() -> pd.DataFrame:
    """Create synthetic ground-truth labels."""
    return pd.DataFrame({
        "lineage_id": [1, 2],
        "quarter": ["2025Q1", "2025Q1"],
        "is_inflection_onset": [1, 0],
    })


# ---------------------------------------------------------------------------
# record_assessments
# ---------------------------------------------------------------------------


class TestRecordAssessments:
    """Tests for converting predictions to assessment rows."""

    def test_output_has_correct_columns(self) -> None:
        preds = _make_predictions()
        result = record_assessments(
            preds, quarter_assessed="2025Q2",
            model_version="v_20260323_001", threshold=0.15,
        )
        assert list(result.columns) == ASSESSMENT_COLUMNS

    def test_row_count_matches_predictions(self) -> None:
        preds = _make_predictions(10)
        result = record_assessments(
            preds, quarter_assessed="2025Q2",
            model_version="v_20260323_001", threshold=0.15,
        )
        assert len(result) == 10

    def test_threshold_applied_correctly(self) -> None:
        preds = pd.DataFrame({
            "lineage_id": [1, 2, 3],
            "quarter": ["2025Q1"] * 3,
            "inflection_probability": [0.10, 0.15, 0.20],
        })
        result = record_assessments(
            preds, quarter_assessed="2025Q2",
            model_version="v_test", threshold=0.15,
        )
        labels = result["predicted_label"].tolist()
        assert labels == [0, 1, 1]

    def test_all_outcomes_unknown(self) -> None:
        preds = _make_predictions()
        result = record_assessments(
            preds, quarter_assessed="2025Q2",
            model_version="v_test", threshold=0.15,
        )
        assert (result["actual_outcome"] == OUTCOME_UNKNOWN).all()

    def test_custom_probability_column(self) -> None:
        preds = pd.DataFrame({
            "lineage_id": [1],
            "quarter": ["2025Q1"],
            "my_prob": [0.75],
        })
        result = record_assessments(
            preds, quarter_assessed="2025Q2",
            model_version="v_test", threshold=0.5,
            probability_column="my_prob",
        )
        assert result["probability"].iloc[0] == pytest.approx(0.75, abs=1e-5)
        assert result["predicted_label"].iloc[0] == 1

    def test_missing_columns_raises(self) -> None:
        preds = pd.DataFrame({"lineage_id": [1]})
        with pytest.raises(ValueError, match="missing required columns"):
            record_assessments(
                preds, quarter_assessed="2025Q2",
                model_version="v_test", threshold=0.15,
            )


# ---------------------------------------------------------------------------
# backfill_outcomes
# ---------------------------------------------------------------------------


class TestBackfillOutcomes:
    """Tests for backfilling unknown outcomes with ground truth."""

    def test_backfills_matching_labels(self) -> None:
        history = _make_history_with_unknowns()
        labels = _make_labels()
        updated, count = backfill_outcomes(history, labels)
        # Labels cover lineages 1 and 2 only; lineage 3 not in label file
        assert count == 2
        assert updated.loc[0, "actual_outcome"] == 1  # lineage 1 positive
        assert updated.loc[1, "actual_outcome"] == 0  # lineage 2 explicit negative

    def test_implicit_negative_for_unlabeled_quarter(self) -> None:
        history = _make_history_with_unknowns()
        # Labels include lineage 3 but NOT for 2025Q1 specifically
        labels = pd.DataFrame({
            "lineage_id": [3],
            "quarter": ["2024Q4"],
            "is_inflection_onset": [1],
        })
        updated, count = backfill_outcomes(history, labels)
        # Lineage 3 is in label file but 2025Q1 is not a positive -> implicit 0
        assert count == 1
        assert updated.loc[2, "actual_outcome"] == 0
        assert "implicit_negative" in updated.loc[2, "outcome_source"]

    def test_already_resolved_rows_untouched(self) -> None:
        history = _make_history_with_unknowns()
        labels = pd.DataFrame({
            "lineage_id": [4],
            "quarter": ["2025Q1"],
            "is_inflection_onset": [0],
        })
        updated, count = backfill_outcomes(history, labels)
        # Lineage 4 was already resolved (actual_outcome=1), should not change
        assert count == 0
        assert updated.loc[3, "actual_outcome"] == 1
        assert updated.loc[3, "outcome_source"] == "manual"

    def test_empty_history_returns_zero(self) -> None:
        history = create_empty_history()
        labels = _make_labels()
        updated, count = backfill_outcomes(history, labels)
        assert count == 0
        assert updated.empty

    def test_missing_label_columns_raises(self) -> None:
        history = _make_history_with_unknowns()
        bad_labels = pd.DataFrame({"lineage_id": [1]})
        with pytest.raises(ValueError, match="missing required columns"):
            backfill_outcomes(history, bad_labels)

    def test_custom_label_column(self) -> None:
        history = _make_history_with_unknowns()
        labels = pd.DataFrame({
            "lineage_id": [1],
            "quarter": ["2025Q1"],
            "custom_label": [1],
        })
        updated, count = backfill_outcomes(
            history, labels, label_column="custom_label",
        )
        assert count == 1
        assert updated.loc[0, "actual_outcome"] == 1


# ---------------------------------------------------------------------------
# append_assessments
# ---------------------------------------------------------------------------


class TestAppendAssessments:
    """Tests for dedup-safe append of assessment rows."""

    def test_append_to_empty(self) -> None:
        existing = create_empty_history()
        new_rows = record_assessments(
            _make_predictions(3), quarter_assessed="2025Q2",
            model_version="v_test", threshold=0.15,
        )
        result = append_assessments(existing, new_rows)
        assert len(result) == 3

    def test_deduplicates_on_primary_key(self) -> None:
        rows = record_assessments(
            _make_predictions(3), quarter_assessed="2025Q2",
            model_version="v_test", threshold=0.15,
        )
        result = append_assessments(rows, rows)
        assert len(result) == 3  # no duplicates

    def test_keeps_first_on_duplicate(self) -> None:
        preds = pd.DataFrame({
            "lineage_id": [1],
            "quarter": ["2025Q1"],
            "inflection_probability": [0.5],
        })
        first = record_assessments(
            preds, quarter_assessed="2025Q2",
            model_version="v_test", threshold=0.15,
        )
        second = record_assessments(
            preds, quarter_assessed="2025Q2",
            model_version="v_test", threshold=0.50,
        )
        result = append_assessments(first, second)
        assert len(result) == 1
        assert result["threshold_used"].iloc[0] == 0.15  # keeps first

    def test_different_versions_kept(self) -> None:
        preds = _make_predictions(2)
        v1 = record_assessments(
            preds, quarter_assessed="2025Q2",
            model_version="v_001", threshold=0.15,
        )
        v2 = record_assessments(
            preds, quarter_assessed="2025Q2",
            model_version="v_002", threshold=0.15,
        )
        result = append_assessments(v1, v2)
        assert len(result) == 4


# ---------------------------------------------------------------------------
# load / save round-trip
# ---------------------------------------------------------------------------


class TestLoadSaveHistory:
    """Tests for CSV I/O of assessment history."""

    def test_round_trip(self, tmp_path: Path) -> None:
        rows = record_assessments(
            _make_predictions(5), quarter_assessed="2025Q2",
            model_version="v_test", threshold=0.15,
        )
        csv_path = tmp_path / "history.csv"
        save_history(rows, csv_path)
        loaded = load_history(csv_path)
        assert len(loaded) == 5
        assert list(loaded.columns) == ASSESSMENT_COLUMNS

    def test_load_nonexistent_returns_empty(self, tmp_path: Path) -> None:
        loaded = load_history(tmp_path / "no_such.csv")
        assert loaded.empty
        assert list(loaded.columns) == ASSESSMENT_COLUMNS

    def test_load_adds_missing_columns(self, tmp_path: Path) -> None:
        # Write a CSV missing some columns
        partial = pd.DataFrame({
            "lineage_id": [1],
            "quarter_predicted": ["2025Q1"],
            "probability": [0.5],
        })
        csv_path = tmp_path / "partial.csv"
        partial.to_csv(csv_path, index=False)
        loaded = load_history(csv_path)
        assert "actual_outcome" in loaded.columns
        assert "outcome_source" in loaded.columns


# ---------------------------------------------------------------------------
# compute_calibration_stats
# ---------------------------------------------------------------------------


class TestComputeCalibrationStats:
    """Tests for calibration metric computation."""

    def test_no_resolved_returns_none_scores(self) -> None:
        history = record_assessments(
            _make_predictions(5), quarter_assessed="2025Q2",
            model_version="v_test", threshold=0.15,
        )
        cal = compute_calibration_stats(history)
        assert cal["n_resolved"] == 0
        assert cal["brier_score"] is None
        assert cal["calibration_error"] is None

    def test_perfect_predictions_brier_zero(self) -> None:
        history = pd.DataFrame({
            "lineage_id": [1, 2, 3, 4],
            "quarter_predicted": ["2025Q1"] * 4,
            "quarter_assessed": ["2025Q2"] * 4,
            "model_version": ["v_test"] * 4,
            "probability": [1.0, 0.0, 1.0, 0.0],
            "predicted_label": [1, 0, 1, 0],
            "threshold_used": [0.5] * 4,
            "actual_outcome": [1, 0, 1, 0],
            "outcome_source": ["test"] * 4,
            "backfilled_at": [""] * 4,
        })
        cal = compute_calibration_stats(history)
        assert cal["n_resolved"] == 4
        assert cal["brier_score"] == pytest.approx(0.0)

    def test_bins_count_sums_to_resolved(self) -> None:
        rng = np.random.default_rng(42)
        n = 100
        history = pd.DataFrame({
            "lineage_id": list(range(n)),
            "quarter_predicted": ["2025Q1"] * n,
            "quarter_assessed": ["2025Q2"] * n,
            "model_version": ["v_test"] * n,
            "probability": rng.uniform(0, 1, n),
            "predicted_label": [0] * n,
            "threshold_used": [0.5] * n,
            "actual_outcome": rng.integers(0, 2, n),
            "outcome_source": ["test"] * n,
            "backfilled_at": [""] * n,
        })
        cal = compute_calibration_stats(history, n_bins=10)
        total_in_bins = sum(b["count"] for b in cal["bins"])
        assert total_in_bins == n

    def test_mixed_resolved_and_unknown(self) -> None:
        history = pd.DataFrame({
            "lineage_id": [1, 2, 3],
            "quarter_predicted": ["2025Q1"] * 3,
            "quarter_assessed": ["2025Q2"] * 3,
            "model_version": ["v_test"] * 3,
            "probability": [0.8, 0.2, 0.5],
            "predicted_label": [1, 0, 1],
            "threshold_used": [0.5] * 3,
            "actual_outcome": [1, 0, OUTCOME_UNKNOWN],
            "outcome_source": ["test", "test", ""],
            "backfilled_at": ["", "", ""],
        })
        cal = compute_calibration_stats(history)
        assert cal["n_resolved"] == 2
        assert cal["n_unknown"] == 1
        assert cal["brier_score"] is not None


# ---------------------------------------------------------------------------
# summarize_history
# ---------------------------------------------------------------------------


class TestSummarizeHistory:
    """Tests for history summary statistics."""

    def test_empty_history(self) -> None:
        summary = summarize_history(create_empty_history())
        assert summary["total_rows"] == 0
        assert summary["resolution_rate"] == 0.0

    def test_summary_counts(self) -> None:
        history = _make_history_with_unknowns()
        summary = summarize_history(history)
        assert summary["total_rows"] == 4
        assert summary["n_versions"] == 1
        assert summary["n_lineages"] == 4
        assert summary["n_resolved"] == 1
        assert summary["n_unknown"] == 3
        assert summary["resolution_rate"] == pytest.approx(0.25)
