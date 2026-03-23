"""Timeliness scoring utilities for detector evaluation.

Implements the timeliness metrics defined in the prospective evaluation
contract (``docs/implementation/evaluation_contract.md``, Section 3.3):

* **NAB score** -- Numenta Anomaly Benchmark profile-weighted scoring.
* **EDD** -- Expected Detection Delay (quarters from onset to first alert).
* **ARL** -- Average Run Length (ARL0 for false-alarm interval, ARL1 for
  detection delay).

All functions are **pure** -- no side effects, no file I/O.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Dict, List, Optional

# ---------------------------------------------------------------------------
# NAB profile weights
# ---------------------------------------------------------------------------

# Profiles define how strongly early detection is rewarded and how
# severely late detection and false positives are penalized.  The
# sigmoid parameters follow the original NAB paper (Lavin & Ahmad 2015).

@dataclass(frozen=True)
class NABProfile:
    """Scoring profile for the NAB metric.

    Attributes:
        name: Human-readable profile name.
        A_tp: Reward scaling for true positives.
        A_fp: Penalty scaling for false positives.
        A_fn: Penalty scaling for false negatives (missed detections).
    """

    name: str
    A_tp: float
    A_fp: float
    A_fn: float


PROFILES: Dict[str, NABProfile] = {
    "standard": NABProfile("standard", A_tp=1.0, A_fp=0.22, A_fn=1.0),
    "reward_low_FP": NABProfile("reward_low_FP", A_tp=1.0, A_fp=1.0, A_fn=1.0),
    "reward_low_FN": NABProfile("reward_low_FN", A_tp=1.0, A_fp=0.11, A_fn=2.0),
}


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TimelinessResult:
    """Aggregated timeliness metrics for one detector evaluation run.

    Attributes:
        nab_scores: NAB score per profile name (0--100 scale).
        edd: Expected Detection Delay in quarters, or ``None`` if no
            true onset was ever detected.
        arl0: Average Run Length between false alarms (quarters), or
            ``None`` if there are no false alarms.
        arl1: Average Run Length from onset to first detection (quarters),
            or ``None`` if no true onset was detected.
        n_true_onsets: Number of true onset events in the evaluation set.
        n_detected: Number of true onsets that were detected.
        n_false_alarms: Total false positive alerts.
    """

    nab_scores: Dict[str, float]
    edd: Optional[float]
    arl0: Optional[float]
    arl1: Optional[float]
    n_true_onsets: int
    n_detected: int
    n_false_alarms: int


# ---------------------------------------------------------------------------
# NAB scoring
# ---------------------------------------------------------------------------


def _sigmoid(y: float) -> float:
    """Scaled sigmoid: 2 / (1 + exp(5*y)) - 1.

    Maps relative position *y* in [-1, 1] to a score in [-1, 1].
    Early detections (y < 0) score positively; late detections (y > 0)
    score negatively.
    """
    import math

    return 2.0 / (1.0 + math.exp(5.0 * y)) - 1.0


def nab_score_single_window(
    detection_position: Optional[float],
    profile: NABProfile,
) -> float:
    """Score a single anomaly window (one true onset event).

    Args:
        detection_position: Relative position of the first detection
            within the window, in [-1, 1].  ``None`` if the onset was
            missed entirely.
        profile: NAB scoring profile.

    Returns:
        Score contribution for this window (can be negative).
    """
    if detection_position is None:
        # Missed detection: penalize by A_fn.
        return -profile.A_fn

    return profile.A_tp * _sigmoid(detection_position)


def nab_score_false_positive(profile: NABProfile) -> float:
    """Penalty for a single false positive alert outside any anomaly window."""
    return -profile.A_fp


def compute_nab_scores(
    detection_positions: Sequence[Optional[float]],
    n_false_positives: int,
    profiles: Optional[Dict[str, NABProfile]] = None,
) -> Dict[str, float]:
    """Compute NAB scores across all profiles.

    Args:
        detection_positions: One entry per true onset.  Each is the
            relative position of the first detection in [-1, 1], or
            ``None`` if the onset was missed.
        n_false_positives: Count of false positive alerts.
        profiles: Profile dict (defaults to :data:`PROFILES`).

    Returns:
        Dict mapping profile name to normalized NAB score (0--100).
    """
    if profiles is None:
        profiles = PROFILES

    results: Dict[str, float] = {}
    for name, profile in profiles.items():
        raw = 0.0
        for pos in detection_positions:
            raw += nab_score_single_window(pos, profile)
        raw += n_false_positives * nab_score_false_positive(profile)

        # Perfect score: every onset detected at position -1 (earliest).
        perfect = len(detection_positions) * nab_score_single_window(-1.0, profile)
        # Null detector score: every onset missed, no false positives.
        null = len(detection_positions) * nab_score_single_window(None, profile)

        # Normalize to [0, 100] scale.
        denom = perfect - null
        if denom == 0:
            results[name] = 0.0
        else:
            results[name] = 100.0 * (raw - null) / denom

    return results


# ---------------------------------------------------------------------------
# EDD and ARL
# ---------------------------------------------------------------------------


def compute_edd(detection_lags: Sequence[int]) -> Optional[float]:
    """Compute Expected Detection Delay.

    Args:
        detection_lags: Detection lag in quarters for each true onset
            that **was** detected.  Only includes detected onsets.

    Returns:
        Mean lag (quarters), or ``None`` if the list is empty.
    """
    if not detection_lags:
        return None
    return sum(detection_lags) / len(detection_lags)


def compute_arl0(
    total_quarters: int,
    n_false_alarms: int,
) -> Optional[float]:
    """Compute ARL0: average quarters between false alarms.

    Higher is better (fewer false alarms per unit time).

    Args:
        total_quarters: Total number of lineage-quarters evaluated.
        n_false_alarms: Count of false positive alerts.

    Returns:
        ARL0, or ``None`` if there are no false alarms.
    """
    if n_false_alarms == 0:
        return None
    return total_quarters / n_false_alarms


def compute_arl1(detection_lags: Sequence[int]) -> Optional[float]:
    """Compute ARL1: average quarters from onset to first detection.

    Equivalent to EDD when detection always occurs.  Lower is better.

    Args:
        detection_lags: Detection lag in quarters for each true onset
            that **was** detected.

    Returns:
        Mean lag (quarters), or ``None`` if the list is empty.
    """
    return compute_edd(detection_lags)


# ---------------------------------------------------------------------------
# Aggregate scorer
# ---------------------------------------------------------------------------


def score_timeliness(
    true_onset_quarters: List[int],
    detection_quarters: List[Optional[int]],
    alert_quarters: List[int],
    total_quarters: int,
    window_size: int = 8,
) -> TimelinessResult:
    """Compute all timeliness metrics for a detector evaluation run.

    This is the primary entry point for timeliness scoring.

    Args:
        true_onset_quarters: Quarter-int of each true onset event.
        detection_quarters: For each true onset, the quarter-int of the
            first detection, or ``None`` if the onset was missed.  Must
            have the same length as *true_onset_quarters*.
        alert_quarters: All quarter-ints where the detector fired a
            positive alert (including true and false positives).
        total_quarters: Total number of lineage-quarters in the
            evaluation set (used for ARL0).
        window_size: NAB anomaly window size in quarters.  Detections
            within this window around a true onset are scored; earlier
            detections score higher.

    Returns:
        A :class:`TimelinessResult` with all timeliness metrics.

    Raises:
        ValueError: If inputs have mismatched lengths.
    """
    if len(true_onset_quarters) != len(detection_quarters):
        raise ValueError(
            f"true_onset_quarters ({len(true_onset_quarters)}) and "
            f"detection_quarters ({len(detection_quarters)}) must have "
            "the same length"
        )

    n_true = len(true_onset_quarters)

    # Identify true positive and missed detections.
    detection_lags: List[int] = []
    detection_positions: List[Optional[float]] = []

    for onset_q, detect_q in zip(true_onset_quarters, detection_quarters):
        if detect_q is None:
            detection_positions.append(None)
        else:
            lag = detect_q - onset_q
            detection_lags.append(lag)
            # Map lag to relative position in [-1, 1] for NAB.
            # -1 = earliest possible (window_size quarters early),
            #  0 = on-time, +1 = latest (window_size quarters late).
            position = lag / max(window_size, 1)
            position = max(-1.0, min(1.0, position))
            detection_positions.append(position)

    n_detected = len(detection_lags)

    # Count false alarms: alerts not within any onset window.
    onset_set = set(true_onset_quarters)
    # An alert is a true positive if it falls within [onset, onset + window_size]
    # for any true onset.
    false_alarm_count = 0
    for alert_q in alert_quarters:
        is_tp = any(
            onset_q <= alert_q <= onset_q + window_size
            for onset_q in onset_set
        )
        if not is_tp:
            false_alarm_count += 1

    # Compute metrics.
    nab = compute_nab_scores(detection_positions, false_alarm_count)
    edd = compute_edd(detection_lags)
    arl0 = compute_arl0(total_quarters, false_alarm_count)
    arl1 = compute_arl1(detection_lags)

    return TimelinessResult(
        nab_scores=nab,
        edd=edd,
        arl0=arl0,
        arl1=arl1,
        n_true_onsets=n_true,
        n_detected=n_detected,
        n_false_alarms=false_alarm_count,
    )
