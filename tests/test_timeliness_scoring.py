"""Tests for src.timeliness_scoring -- timeliness evaluation utilities.

Uses synthetic detection scenarios to exercise NAB, EDD, and ARL
metrics as defined in the evaluation contract (Section 3.3).
"""

from __future__ import annotations

import pytest

from src.timeliness_scoring import (
    PROFILES,
    TimelinessResult,
    compute_arl0,
    compute_arl1,
    compute_edd,
    compute_nab_scores,
    nab_score_single_window,
    score_timeliness,
)

pytestmark = pytest.mark.unit


# ── NAB scoring ────────────────────────────────────────────────────────────

class TestNABSingleWindow:

    def test_early_detection_scores_positive(self):
        """Detection before onset (position < 0) should score > 0."""
        profile = PROFILES["standard"]
        score = nab_score_single_window(-0.5, profile)
        assert score > 0

    def test_late_detection_scores_negative(self):
        """Detection well after onset (position near 1) should score < 0."""
        profile = PROFILES["standard"]
        score = nab_score_single_window(0.9, profile)
        assert score < 0

    def test_on_time_detection_scores_zero(self):
        """Detection exactly at onset (position = 0) scores 0 (sigmoid midpoint)."""
        profile = PROFILES["standard"]
        score = nab_score_single_window(0.0, profile)
        assert score == pytest.approx(0.0)

    def test_missed_detection_is_negative(self):
        """Missed detection (None) should return -A_fn."""
        profile = PROFILES["standard"]
        score = nab_score_single_window(None, profile)
        assert score == pytest.approx(-profile.A_fn)


class TestNABScores:

    def test_perfect_detector_scores_100(self):
        """All onsets detected at earliest position -> score 100."""
        positions = [-1.0, -1.0, -1.0]
        scores = compute_nab_scores(positions, n_false_positives=0)
        for name, score in scores.items():
            assert score == pytest.approx(100.0), f"Profile {name} should be 100"

    def test_null_detector_scores_0(self):
        """All onsets missed, no FP -> score 0."""
        positions = [None, None, None]
        scores = compute_nab_scores(positions, n_false_positives=0)
        for name, score in scores.items():
            assert score == pytest.approx(0.0), f"Profile {name} should be 0"

    def test_false_positives_reduce_score(self):
        """Adding false positives should reduce the score."""
        positions = [-1.0, -1.0]
        scores_clean = compute_nab_scores(positions, n_false_positives=0)
        scores_noisy = compute_nab_scores(positions, n_false_positives=5)
        for name in scores_clean:
            assert scores_noisy[name] < scores_clean[name]

    def test_all_three_profiles_present(self):
        """Standard, reward_low_FP, reward_low_FN profiles should all appear."""
        scores = compute_nab_scores([0.0], n_false_positives=0)
        assert set(scores.keys()) == {"standard", "reward_low_FP", "reward_low_FN"}

    def test_empty_detection_list(self):
        """No true onsets -> all profiles score 0."""
        scores = compute_nab_scores([], n_false_positives=0)
        for score in scores.values():
            assert score == pytest.approx(0.0)


# ── EDD ────────────────────────────────────────────────────────────────────

class TestEDD:

    def test_basic_edd(self):
        assert compute_edd([2, 4, 6]) == pytest.approx(4.0)

    def test_zero_lag(self):
        assert compute_edd([0, 0, 0]) == pytest.approx(0.0)

    def test_empty_returns_none(self):
        assert compute_edd([]) is None

    def test_single_detection(self):
        assert compute_edd([3]) == pytest.approx(3.0)


# ── ARL ────────────────────────────────────────────────────────────────────

class TestARL:

    def test_arl0_basic(self):
        """100 quarters, 5 false alarms -> ARL0 = 20."""
        assert compute_arl0(100, 5) == pytest.approx(20.0)

    def test_arl0_no_false_alarms(self):
        """No false alarms -> ARL0 is None (infinite)."""
        assert compute_arl0(100, 0) is None

    def test_arl1_same_as_edd(self):
        """ARL1 should equal EDD when detection always occurs."""
        lags = [1, 3, 5]
        assert compute_arl1(lags) == compute_edd(lags)


# ── Aggregate scorer ──────────────────────────────────────────────────────

class TestScoreTimeliness:

    def test_perfect_detection(self):
        """All onsets detected on-time, no false alarms."""
        result = score_timeliness(
            true_onset_quarters=[100, 200, 300],
            detection_quarters=[100, 200, 300],
            alert_quarters=[100, 200, 300],
            total_quarters=400,
        )
        assert result.n_true_onsets == 3
        assert result.n_detected == 3
        assert result.n_false_alarms == 0
        assert result.edd == pytest.approx(0.0)
        assert result.arl0 is None  # No false alarms
        assert result.nab_scores["standard"] > 50  # Good score

    def test_all_missed(self):
        """No detections at all."""
        result = score_timeliness(
            true_onset_quarters=[100, 200],
            detection_quarters=[None, None],
            alert_quarters=[],
            total_quarters=400,
        )
        assert result.n_detected == 0
        assert result.edd is None
        assert result.arl1 is None
        assert result.nab_scores["standard"] == pytest.approx(0.0)

    def test_mixed_detection(self):
        """Some detected, some missed, some false alarms."""
        result = score_timeliness(
            true_onset_quarters=[100, 200],
            detection_quarters=[102, None],  # First detected 2Q late, second missed
            alert_quarters=[102, 50, 350],  # 102 is TP, 50 and 350 are FP
            total_quarters=400,
        )
        assert result.n_detected == 1
        assert result.n_false_alarms == 2
        assert result.edd == pytest.approx(2.0)
        assert result.arl0 == pytest.approx(200.0)

    def test_mismatched_lengths_raises(self):
        with pytest.raises(ValueError, match="same length"):
            score_timeliness(
                true_onset_quarters=[100, 200],
                detection_quarters=[100],
                alert_quarters=[],
                total_quarters=400,
            )

    def test_result_is_frozen(self):
        result = score_timeliness(
            true_onset_quarters=[100],
            detection_quarters=[100],
            alert_quarters=[100],
            total_quarters=100,
        )
        with pytest.raises(AttributeError):
            result.edd = 5.0  # type: ignore[misc]
