"""Onset detection utilities for lineage growth acceleration.

Implements the onset labeling rules from
``docs/implementation/onset_label_specification.md`` (v1.0).

The core function :func:`detect_onset` is **pure** -- no side effects, no
file I/O -- so it can be tested with small synthetic count series.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class OnsetResult:
    """Result of onset detection for a single lineage.

    Attributes:
        quarter: Quarter label where onset was detected, or ``None``.
        detected: ``True`` if onset was found.
        reason: Detection reason (``sustained_acceleration``) or skip
            reason (``insufficient_history``, ``no_sustained_growth``).
        growth_rate: Quarter-over-quarter growth rate at the onset quarter.
        smoothed_count: Smoothed count at the onset quarter.
        confirmation_length: Number of consecutive positive-growth quarters
            in the confirmation run.
        early_onset: ``True`` if onset fires in the first two quarters of
            the lineage's existence (flagged for QA review).
    """

    quarter: str | None
    detected: bool
    reason: str
    growth_rate: float | None
    smoothed_count: float | None
    confirmation_length: int
    early_onset: bool


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def detect_onset(
    quarters: list[str],
    counts: list[int],
    *,
    smoothing_window: int = 3,
    growth_threshold: float = 0.10,
    confirmation_quarters: int = 3,
    min_count: int = 3,
) -> OnsetResult:
    """Detect the first quarter of sustained growth acceleration.

    Implements Section 2.2 of the onset label specification.  Uses a
    trailing rolling mean for smoothing, quarter-over-quarter growth rates,
    and a confirmation period to filter transient spikes.

    Args:
        quarters: Chronologically ordered quarter labels
            (e.g. ``["2015Q1", "2015Q2", ...]``).
        counts: New-works counts corresponding to each quarter.
        smoothing_window: Rolling mean window size (trailing).
        growth_threshold: Minimum growth rate at the onset quarter.
        confirmation_quarters: Consecutive quarters of positive growth
            required to confirm onset.
        min_count: Minimum smoothed count to avoid noise triggers.

    Returns:
        An :class:`OnsetResult` describing the detection outcome.

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
    )

    n = len(quarters)

    # Edge case: insufficient history (spec Section 3.1).
    if n < smoothing_window + confirmation_quarters:
        return OnsetResult(
            quarter=None,
            detected=False,
            reason="insufficient_history",
            growth_rate=None,
            smoothed_count=None,
            confirmation_length=0,
            early_onset=False,
        )

    # Step 1 -- Trailing rolling mean (spec Section 2.2, rule 1).
    smoothed = _trailing_rolling_mean(counts, smoothing_window)

    # Step 2 -- Quarter-over-quarter growth rate (spec Section 2.2, rule 2).
    growth_rates = _compute_growth_rates(smoothed)

    # Step 3 -- Acceleration test (spec Section 2.2, rule 3).
    # Scan from t=1 onward (growth rate undefined at t=0).
    for t in range(1, n):
        if smoothed[t] < min_count:
            continue
        if growth_rates[t] < growth_threshold:
            continue

        # Confirm c consecutive quarters of positive growth starting at t.
        run_length = 0
        confirmed = True
        for j in range(confirmation_quarters):
            idx = t + j
            if idx >= n:
                confirmed = False
                break
            if growth_rates[idx] <= 0:
                confirmed = False
                break
            run_length += 1

        if not confirmed:
            continue

        # Onset fires (spec Section 2.2, rule 4).
        return OnsetResult(
            quarter=quarters[t],
            detected=True,
            reason="sustained_acceleration",
            growth_rate=growth_rates[t],
            smoothed_count=smoothed[t],
            confirmation_length=run_length,
            early_onset=t <= 1,
        )

    # No onset found (spec Section 3.4).
    return OnsetResult(
        quarter=None,
        detected=False,
        reason="no_sustained_growth",
        growth_rate=None,
        smoothed_count=None,
        confirmation_length=0,
        early_onset=False,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _validate_inputs(
    quarters: list[str],
    counts: list[int],
    smoothing_window: int,
    growth_threshold: float,
    confirmation_quarters: int,
    min_count: int,
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
