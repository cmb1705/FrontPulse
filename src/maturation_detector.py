"""Maturation detection utilities for lineage growth deceleration.

Implements the maturation labeling rules from
``docs/implementation/maturation_label_specification.md`` (v1.0).

The core function :func:`detect_maturation` is **pure** -- no side effects,
no file I/O -- so it can be tested with small synthetic count series.

Maturation is the complement to onset detection: where onset finds the
first quarter of sustained growth acceleration (the lower S-curve elbow),
maturation finds the first quarter of sustained growth deceleration (the
upper elbow).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class MaturationResult:
    """Result of maturation detection for a single lineage.

    Attributes:
        quarter: Quarter label where maturation was detected, or ``None``.
        detected: ``True`` if maturation was found.
        maturation_type: One of ``saturation``, ``senescence``,
            ``dormancy_entry``, or ``None`` if not detected.
        reason: Detection reason or skip reason.
        growth_rate: Growth rate at the maturation quarter.
        smoothed_count: Smoothed count at the maturation quarter.
        peak_quarter: Quarter of peak smoothed count.
        peak_count: Peak smoothed count value.
        confirmation_length: Number of consecutive qualifying quarters
            in the confirmation run.
        late_maturation: ``True`` if maturation fires in the last ``c``
            quarters of available data (flagged for QA review).
    """

    quarter: Optional[str]
    detected: bool
    maturation_type: Optional[str]
    reason: str
    growth_rate: Optional[float]
    smoothed_count: Optional[float]
    peak_quarter: Optional[str]
    peak_count: Optional[float]
    confirmation_length: int
    late_maturation: bool


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def detect_maturation(
    quarters: list[str],
    counts: list[int],
    *,
    smoothing_window: int = 3,
    growth_threshold: float = 0.10,
    confirmation_quarters: int = 3,
    min_count: int = 3,
    activity_floor: int = 1,
) -> MaturationResult:
    """Detect the first quarter of sustained growth deceleration.

    Implements Section 2.3 of the maturation label specification. Uses a
    trailing rolling mean for smoothing, quarter-over-quarter growth rates,
    peak detection, and a confirmation period to filter transient dips.

    Args:
        quarters: Chronologically ordered quarter labels.
        counts: New-works counts corresponding to each quarter.
        smoothing_window: Rolling mean window size (trailing).
        growth_threshold: Magnitude of growth rate threshold. For
            saturation, growth must stay within ``[-g_min, +g_min]``.
            For senescence, growth must be below ``-g_min``.
        confirmation_quarters: Consecutive quarters of qualifying
            deceleration required to confirm maturation.
        min_count: Minimum smoothed count for the lineage to be
            considered as having had a meaningful growth phase.
        activity_floor: Below this smoothed count, the lineage is
            considered to have entered dormancy.

    Returns:
        A :class:`MaturationResult` describing the detection outcome.

    Raises:
        ValueError: If *quarters* and *counts* have different lengths or
            if parameter values are out of the allowed range.
    """
    _validate_inputs(
        quarters,
        counts,
        smoothing_window,
        growth_threshold,
        confirmation_quarters,
        min_count,
        activity_floor,
    )

    n = len(quarters)

    # Edge case: insufficient history (spec Section 3.1).
    if n < smoothing_window + confirmation_quarters:
        return _no_maturation("insufficient_history")

    # Step 1 -- Trailing rolling mean (spec Section 2.3, rule 1).
    smoothed = _trailing_rolling_mean(counts, smoothing_window)

    # Edge case: lineage never reached meaningful activity (spec Section 3.3).
    if max(smoothed) < min_count:
        return _no_maturation("never_reached_threshold")

    # Step 2 -- Growth rates (spec Section 2.3, rule 2).
    growth_rates = _compute_growth_rates(smoothed)

    # Step 3 -- Peak detection (spec Section 2.3, rule 3).
    peak_idx = _find_peak(smoothed)
    peak_quarter = quarters[peak_idx]
    peak_count = smoothed[peak_idx]

    # Step 4 -- Deceleration test (spec Section 2.3, rule 4).
    # Scan from peak onward for sustained deceleration.
    for t in range(peak_idx, n):
        # Try each subtype in priority order: dormancy > senescence > saturation.
        for subtype, checker in [
            ("dormancy_entry", lambda idx: smoothed[idx] < activity_floor),
            ("senescence", lambda idx: growth_rates[idx] < -growth_threshold),
            (
                "saturation",
                lambda idx: (
                    abs(growth_rates[idx]) < growth_threshold
                    and smoothed[idx] >= min_count
                ),
            ),
        ]:
            run_length = 0
            confirmed = True
            for j in range(confirmation_quarters):
                idx = t + j
                if idx >= n:
                    confirmed = False
                    break
                if not checker(idx):
                    confirmed = False
                    break
                run_length += 1

            if not confirmed:
                continue

            # Maturation fires (spec Section 2.3, rule 5).
            is_late = (t + confirmation_quarters) >= n
            return MaturationResult(
                quarter=quarters[t],
                detected=True,
                maturation_type=subtype,
                reason=f"sustained_{subtype}",
                growth_rate=growth_rates[t],
                smoothed_count=smoothed[t],
                peak_quarter=peak_quarter,
                peak_count=peak_count,
                confirmation_length=run_length,
                late_maturation=is_late,
            )

    # No maturation found (spec Section 3.2).
    return MaturationResult(
        quarter=None,
        detected=False,
        maturation_type=None,
        reason="still_growing",
        growth_rate=None,
        smoothed_count=None,
        peak_quarter=peak_quarter,
        peak_count=peak_count,
        confirmation_length=0,
        late_maturation=False,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _no_maturation(reason: str) -> MaturationResult:
    """Return a negative maturation result with the given reason."""
    return MaturationResult(
        quarter=None,
        detected=False,
        maturation_type=None,
        reason=reason,
        growth_rate=None,
        smoothed_count=None,
        peak_quarter=None,
        peak_count=None,
        confirmation_length=0,
        late_maturation=False,
    )


def _validate_inputs(
    quarters: list[str],
    counts: list[int],
    smoothing_window: int,
    growth_threshold: float,
    confirmation_quarters: int,
    min_count: int,
    activity_floor: int,
) -> None:
    """Raise ``ValueError`` on invalid parameter combinations."""
    if len(quarters) != len(counts):
        raise ValueError(
            f"quarters ({len(quarters)}) and counts ({len(counts)}) "
            "must have the same length"
        )
    if not 2 <= smoothing_window <= 6:
        raise ValueError(
            f"smoothing_window must be 2--6, got {smoothing_window}"
        )
    if not 0.05 <= growth_threshold <= 0.50:
        raise ValueError(
            f"growth_threshold must be 0.05--0.50, got {growth_threshold}"
        )
    if not 2 <= confirmation_quarters <= 6:
        raise ValueError(
            f"confirmation_quarters must be 2--6, got {confirmation_quarters}"
        )
    if not 1 <= min_count <= 10:
        raise ValueError(f"min_count must be 1--10, got {min_count}")
    if not 0 <= activity_floor <= 3:
        raise ValueError(
            f"activity_floor must be 0--3, got {activity_floor}"
        )


def _trailing_rolling_mean(
    values: list[int],
    window: int,
) -> list[float]:
    """Compute a trailing rolling mean over *values*.

    For early indices where fewer than *window* values are available,
    the mean is computed over the available values (partial window).
    """
    n = len(values)
    result = [0.0] * n
    for i in range(n):
        start = max(0, i - window + 1)
        segment = values[start : i + 1]
        result[i] = sum(segment) / len(segment)
    return result


def _compute_growth_rates(smoothed: list[float]) -> list[float]:
    """Compute quarter-over-quarter growth rates.

    ``g[t] = (s[t] - s[t-1]) / max(s[t-1], 1)``

    The rate at index 0 is defined as 0.0 (no prior quarter).
    """
    n = len(smoothed)
    rates = [0.0] * n
    for t in range(1, n):
        prev = max(smoothed[t - 1], 1.0)
        rates[t] = (smoothed[t] - smoothed[t - 1]) / prev
    return rates


def _find_peak(smoothed: list[float]) -> int:
    """Return the index of the maximum smoothed count."""
    peak_idx = 0
    for i in range(1, len(smoothed)):
        if smoothed[i] > smoothed[peak_idx]:
            peak_idx = i
    return peak_idx
