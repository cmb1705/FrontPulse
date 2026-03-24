"""Tests for src.maturation_detector -- maturation detection utilities.

Uses synthetic count series to exercise every edge case described in
``docs/implementation/maturation_label_specification.md`` (v1.0).
"""

from __future__ import annotations

import pytest

from src.maturation_detector import MaturationResult, detect_maturation

pytestmark = pytest.mark.unit


# -- Helpers -----------------------------------------------------------------


def _qtr_labels(start_year: int, start_q: int, n: int) -> list[str]:
    """Generate *n* consecutive quarter labels starting from *start_year*Q*start_q*."""
    labels: list[str] = []
    y, q = start_year, start_q
    for _ in range(n):
        labels.append(f"{y}Q{q}")
        q += 1
        if q > 4:
            q = 1
            y += 1
    return labels


# -- Saturation detection ----------------------------------------------------


class TestSaturation:
    """Maturation subtype: growth rate drops to near-zero but lineage stays active."""

    def test_clear_saturation_detected(self):
        """Growth accelerates then flattens -- saturation should fire."""
        # Growth phase then plateau at ~20.
        counts = [1, 2, 4, 8, 16, 20, 20, 20, 20, 20, 20]
        quarters = _qtr_labels(2015, 1, len(counts))
        result = detect_maturation(quarters, counts)

        assert result.detected is True
        assert result.maturation_type == "saturation"
        assert result.reason == "sustained_saturation"
        assert result.quarter is not None
        assert result.peak_quarter is not None
        assert result.peak_count is not None
        assert result.confirmation_length >= 3

    def test_saturation_needs_min_count(self):
        """Flat series below min_count should not trigger saturation."""
        # Flatline at 1 -- below default min_count=3.
        counts = [0, 0, 1, 2, 2, 1, 1, 1, 1, 1]
        quarters = _qtr_labels(2015, 1, len(counts))
        result = detect_maturation(quarters, counts)

        # Should either not detect or detect dormancy, not saturation.
        if result.detected:
            assert result.maturation_type != "saturation"


# -- Senescence detection ----------------------------------------------------


class TestSenescence:
    """Maturation subtype: sustained negative growth."""

    def test_clear_senescence_detected(self):
        """Growth then steady decline -- senescence should fire."""
        counts = [1, 3, 6, 12, 20, 15, 10, 6, 3, 2]
        quarters = _qtr_labels(2015, 1, len(counts))
        result = detect_maturation(quarters, counts)

        assert result.detected is True
        assert result.maturation_type == "senescence"
        assert result.reason == "sustained_senescence"
        assert result.growth_rate is not None
        assert result.growth_rate < 0


# -- Dormancy entry detection ------------------------------------------------


class TestDormancyEntry:
    """Maturation subtype: activity drops below floor.

    Dormancy fires when smoothed counts stay below ``activity_floor``
    for ``c`` consecutive quarters.  In typical decline series senescence
    fires chronologically *earlier* because sustained negative growth
    rates appear before the smoothed count drops below floor.  Dormancy
    fires in the "instant crash" pattern: counts fall to zero in one
    step so the rolling mean shows only one quarter of strongly negative
    growth before going flat at zero -- not enough for senescence to
    sustain ``c`` quarters, but dormancy sustains easily.
    """

    def test_clear_dormancy_detected(self):
        """Growth then instant crash to zero -- dormancy_entry should fire.

        With counts [3, 10, 30, 0, 0, ...] and w=3, the smoothed
        values decline through only two quarters of strong negative
        growth (not 3), so senescence cannot sustain.  Dormancy fires
        once the smoothed count reaches 0.0.
        """
        counts = [3, 10, 30, 0, 0, 0, 0, 0, 0]
        quarters = _qtr_labels(2015, 1, len(counts))
        result = detect_maturation(quarters, counts)

        assert result.detected is True
        assert result.maturation_type == "dormancy_entry"
        assert result.reason == "sustained_dormancy_entry"

    def test_gradual_decline_triggers_senescence_not_dormancy(self):
        """A gradual decline triggers senescence before dormancy (Sec 2.3).

        When counts decline slowly, the growth rate stays below
        -g_min for enough quarters that senescence fires first,
        before the smoothed count drops below activity_floor.
        """
        counts = [2, 5, 10, 20, 30, 20, 12, 6, 3, 1]
        quarters = _qtr_labels(2015, 1, len(counts))
        result = detect_maturation(quarters, counts)

        assert result.detected is True
        assert result.maturation_type == "senescence"


# -- No maturation -----------------------------------------------------------


class TestNoMaturation:

    def test_still_growing(self):
        """Monotonically growing series should not trigger maturation (Sec 3.2)."""
        counts = [1, 2, 4, 8, 16, 32, 64, 100, 150, 200]
        quarters = _qtr_labels(2015, 1, len(counts))
        result = detect_maturation(quarters, counts)

        assert result.detected is False
        assert result.reason == "still_growing"
        assert result.peak_quarter is not None
        assert result.peak_count is not None

    def test_never_reached_threshold(self):
        """Lineage never exceeded min_count -- skip (Sec 3.3)."""
        counts = [0, 0, 1, 1, 0, 1, 0, 0, 1, 0]
        quarters = _qtr_labels(2015, 1, len(counts))
        result = detect_maturation(quarters, counts)

        assert result.detected is False
        assert result.reason == "never_reached_threshold"


# -- Insufficient history ----------------------------------------------------


class TestInsufficientHistory:

    def test_too_few_quarters(self):
        """Fewer than w + c quarters -> skip (Sec 3.1)."""
        counts = [1, 2, 3, 4, 5]
        quarters = _qtr_labels(2020, 1, len(counts))
        result = detect_maturation(quarters, counts)

        assert result.detected is False
        assert result.reason == "insufficient_history"

    def test_exactly_enough_quarters(self):
        """Exactly w + c quarters should NOT be skipped."""
        counts = [1, 2, 4, 8, 12, 18]
        quarters = _qtr_labels(2020, 1, len(counts))
        result = detect_maturation(quarters, counts)

        assert result.reason != "insufficient_history"


# -- Late maturation ---------------------------------------------------------


class TestLateMaturation:

    def test_late_maturation_flagged(self):
        """Maturation in last c quarters is flagged (Sec 3.5)."""
        # Growth then plateau right at the end of series.
        counts = [1, 3, 6, 12, 20, 30, 30, 30, 30]
        quarters = _qtr_labels(2015, 1, len(counts))
        result = detect_maturation(quarters, counts)

        if result.detected:
            # If maturation fires near end, late_maturation should be True.
            mat_idx = quarters.index(result.quarter)
            if (mat_idx + 3) >= len(quarters):
                assert result.late_maturation is True


# -- Parameter validation ----------------------------------------------------


class TestParameterValidation:

    def test_mismatched_lengths(self):
        with pytest.raises(ValueError, match="same length"):
            detect_maturation(["2020Q1", "2020Q2"], [1, 2, 3])

    def test_smoothing_window_out_of_range(self):
        quarters = _qtr_labels(2020, 1, 10)
        counts = [1] * 10
        with pytest.raises(ValueError, match="smoothing_window"):
            detect_maturation(quarters, counts, smoothing_window=1)

    def test_growth_threshold_out_of_range(self):
        quarters = _qtr_labels(2020, 1, 10)
        counts = [1] * 10
        with pytest.raises(ValueError, match="growth_threshold"):
            detect_maturation(quarters, counts, growth_threshold=0.01)

    def test_confirmation_quarters_out_of_range(self):
        quarters = _qtr_labels(2020, 1, 10)
        counts = [1] * 10
        with pytest.raises(ValueError, match="confirmation_quarters"):
            detect_maturation(quarters, counts, confirmation_quarters=1)

    def test_min_count_out_of_range(self):
        quarters = _qtr_labels(2020, 1, 10)
        counts = [1] * 10
        with pytest.raises(ValueError, match="min_count"):
            detect_maturation(quarters, counts, min_count=0)

    def test_activity_floor_out_of_range(self):
        quarters = _qtr_labels(2020, 1, 10)
        counts = [1] * 10
        with pytest.raises(ValueError, match="activity_floor"):
            detect_maturation(quarters, counts, activity_floor=5)


# -- MaturationResult dataclass ----------------------------------------------


class TestMaturationResult:

    def test_frozen(self):
        """MaturationResult should be immutable."""
        result = MaturationResult(
            quarter="2020Q1",
            detected=True,
            maturation_type="saturation",
            reason="sustained_saturation",
            growth_rate=0.02,
            smoothed_count=20.0,
            peak_quarter="2019Q4",
            peak_count=20.0,
            confirmation_length=3,
            late_maturation=False,
        )
        with pytest.raises(AttributeError):
            result.quarter = "2020Q2"  # type: ignore[misc]
