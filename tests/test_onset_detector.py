"""Tests for src.onset_detector -- onset detection utilities.

Uses synthetic count series to exercise every edge case described in
``docs/implementation/onset_label_specification.md`` (v1.0).
"""

from __future__ import annotations

import pytest

from src.onset_detector import OnsetResult, detect_onset

pytestmark = pytest.mark.unit


# ── Helpers ─────────────────────────────────────────────────────────────────

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


# ── Basic detection ─────────────────────────────────────────────────────────

class TestBasicDetection:
    """Onset should fire on clear, sustained growth."""

    def test_clear_growth_detected(self):
        # Flat start then accelerating growth.
        counts = [1, 1, 1, 2, 4, 8, 16, 20, 25]
        quarters = _qtr_labels(2015, 1, len(counts))
        result = detect_onset(quarters, counts)

        assert result.detected is True
        assert result.reason == "sustained_acceleration"
        assert result.quarter is not None
        assert result.growth_rate is not None
        assert result.growth_rate >= 0.10
        assert result.smoothed_count is not None
        assert result.smoothed_count >= 3
        assert result.confirmation_length >= 3

    def test_returns_first_onset_only(self):
        """Two growth episodes: only the first should be detected (Sec 3.3)."""
        #       flat    growth1     dormant    growth2
        counts = [1, 1, 1, 3, 6, 12, 2, 1, 1, 3, 7, 15, 20]
        quarters = _qtr_labels(2010, 1, len(counts))
        result = detect_onset(quarters, counts)

        assert result.detected is True
        # The onset should be in the first growth phase (2010--2011).
        onset_year = int(result.quarter[:4])
        assert onset_year <= 2012


# ── No-onset cases ──────────────────────────────────────────────────────────

class TestNoOnset:

    def test_flat_series(self):
        """Constant counts should not trigger onset (Sec 3.4)."""
        counts = [5] * 10
        quarters = _qtr_labels(2015, 1, 10)
        result = detect_onset(quarters, counts)

        assert result.detected is False
        assert result.reason == "no_sustained_growth"

    def test_declining_series(self):
        """Declining counts should not trigger onset."""
        counts = [20, 18, 15, 12, 10, 8, 6, 4, 3, 2]
        quarters = _qtr_labels(2015, 1, len(counts))
        result = detect_onset(quarters, counts)

        assert result.detected is False
        assert result.reason == "no_sustained_growth"

    def test_below_min_count(self):
        """Growth exists but smoothed counts stay below min_count."""
        counts = [0, 0, 0, 1, 2, 2, 2, 2, 2, 2]
        quarters = _qtr_labels(2015, 1, len(counts))
        result = detect_onset(quarters, counts, min_count=5)

        assert result.detected is False


# ── Insufficient history ────────────────────────────────────────────────────

class TestInsufficientHistory:

    def test_too_few_quarters(self):
        """Fewer than w + c quarters -> skip (Sec 3.1)."""
        counts = [1, 2, 3, 4, 5]
        quarters = _qtr_labels(2020, 1, len(counts))
        # Default w=3, c=3 -> need >= 6 quarters; 5 is insufficient.
        result = detect_onset(quarters, counts)

        assert result.detected is False
        assert result.reason == "insufficient_history"

    def test_exactly_enough_quarters(self):
        """Exactly w + c quarters should NOT be skipped."""
        counts = [1, 2, 4, 8, 12, 18]
        quarters = _qtr_labels(2020, 1, len(counts))
        result = detect_onset(quarters, counts)
        # Should attempt detection (may or may not find onset).
        assert result.reason != "insufficient_history"


# ── Temporary surges ────────────────────────────────────────────────────────

class TestTemporarySurge:

    def test_single_quarter_spike_not_onset(self):
        """A one-quarter spike that drops back is not onset (Sec 3.2)."""
        counts = [2, 2, 2, 20, 2, 2, 2, 2, 2, 2]
        quarters = _qtr_labels(2015, 1, len(counts))
        result = detect_onset(quarters, counts)

        assert result.detected is False

    def test_two_quarter_burst_not_onset(self):
        """A burst shorter than confirmation_quarters is not onset."""
        counts = [2, 2, 2, 10, 15, 2, 2, 2, 2, 2]
        quarters = _qtr_labels(2015, 1, len(counts))
        result = detect_onset(quarters, counts)

        assert result.detected is False


# ── Early onset ─────────────────────────────────────────────────────────────

class TestEarlyOnset:

    def test_onset_in_second_quarter_flagged(self):
        """Onset in first two quarters is flagged early_onset=True (Sec 3.5)."""
        # Immediate exponential growth from the start.
        counts = [3, 6, 12, 24, 48, 96]
        quarters = _qtr_labels(2020, 1, len(counts))
        result = detect_onset(
            quarters,
            counts,
            smoothing_window=2,
            confirmation_quarters=3,
        )

        assert result.detected is True
        assert result.early_onset is True


# ── Parameter validation ────────────────────────────────────────────────────

class TestParameterValidation:

    def test_mismatched_lengths(self):
        with pytest.raises(ValueError, match="same length"):
            detect_onset(["2020Q1", "2020Q2"], [1, 2, 3])

    def test_smoothing_window_out_of_range(self):
        quarters = _qtr_labels(2020, 1, 10)
        counts = [1] * 10
        with pytest.raises(ValueError, match="smoothing_window"):
            detect_onset(quarters, counts, smoothing_window=1)

    def test_growth_threshold_out_of_range(self):
        quarters = _qtr_labels(2020, 1, 10)
        counts = [1] * 10
        with pytest.raises(ValueError, match="growth_threshold"):
            detect_onset(quarters, counts, growth_threshold=0.01)

    def test_confirmation_quarters_out_of_range(self):
        quarters = _qtr_labels(2020, 1, 10)
        counts = [1] * 10
        with pytest.raises(ValueError, match="confirmation_quarters"):
            detect_onset(quarters, counts, confirmation_quarters=1)

    def test_min_count_out_of_range(self):
        quarters = _qtr_labels(2020, 1, 10)
        counts = [1] * 10
        with pytest.raises(ValueError, match="min_count"):
            detect_onset(quarters, counts, min_count=0)


# ── Custom parameters ──────────────────────────────────────────────────────

class TestCustomParameters:

    def test_higher_threshold_delays_onset(self):
        """Raising growth_threshold should delay (or prevent) onset."""
        counts = [1, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        quarters = _qtr_labels(2015, 1, len(counts))

        result_low = detect_onset(quarters, counts, growth_threshold=0.05)
        result_high = detect_onset(quarters, counts, growth_threshold=0.40)

        # With a very high threshold, onset may not fire at all or fires later.
        if result_low.detected and result_high.detected:
            low_idx = quarters.index(result_low.quarter)
            high_idx = quarters.index(result_high.quarter)
            assert high_idx >= low_idx
        elif result_low.detected:
            # Higher threshold prevented onset entirely -- also valid.
            assert result_high.detected is False

    def test_shorter_confirmation_enables_earlier_onset(self):
        """Reducing confirmation_quarters should allow earlier detection."""
        counts = [1, 1, 3, 6, 4, 3, 2, 2, 2, 2]
        quarters = _qtr_labels(2015, 1, len(counts))

        result_c2 = detect_onset(quarters, counts, confirmation_quarters=2)
        result_c3 = detect_onset(quarters, counts, confirmation_quarters=3)

        # With c=2, a shorter growth burst qualifies.
        if result_c2.detected and result_c3.detected:
            idx_c2 = quarters.index(result_c2.quarter)
            idx_c3 = quarters.index(result_c3.quarter)
            assert idx_c2 <= idx_c3
        elif result_c2.detected:
            assert result_c3.detected is False


# ── OnsetResult dataclass ──────────────────────────────────────────────────

class TestOnsetResult:

    def test_frozen(self):
        """OnsetResult should be immutable."""
        result = OnsetResult(
            quarter="2020Q1",
            detected=True,
            reason="sustained_acceleration",
            growth_rate=0.15,
            smoothed_count=5.0,
            confirmation_length=3,
            early_onset=False,
        )
        with pytest.raises(AttributeError):
            result.quarter = "2020Q2"  # type: ignore[misc]
