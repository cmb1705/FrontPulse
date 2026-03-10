# Onset Label Specification

Version: 1.0
Date: 2026-03-10
Status: Authoritative

This document defines the onset labeling rules that replace midpoint-oriented
logistic fits for prospective early-warning evaluation.  All onset-related
code and experiments must conform to this specification.

## 1. Motivation

The current labeling pipeline (`scripts/label_inflection_points.py`) uses two
retrospective methods:

1. **Logistic fit**: fits a logistic growth curve to the full cumulative
   publication series and marks the midpoint (inflection of the S-curve).
2. **Derivative heuristic**: second-derivative analysis of smoothed counts.

Both methods require **future data** to locate the inflection point accurately.
A logistic fit only converges reliably after the growth phase is largely
complete, and the derivative heuristic benefits from bilateral smoothing.
This creates temporal leakage: the label encodes information that would not
be available at detection time.

Onset labeling replaces the midpoint with the **first quarter of sustained
growth**, which can be defined using only past and current data.

## 2. Onset Definition

### 2.1 Plain Language

The **onset quarter** for a lineage is the first quarter in which publication
counts begin a sustained acceleration that is distinguishable from noise.

### 2.2 Operational Rules

Given a lineage's quarterly new-works series `n[t]` for quarters
`t = 1, 2, ..., T`:

1. **Smoothing**: compute a rolling mean `s[t]` with window `w` (default: 3
   quarters, centered when labels are retrospective, trailing when used for
   prospective detection).

2. **Growth rate**: compute the quarter-over-quarter growth rate
   `g[t] = (s[t] - s[t-1]) / max(s[t-1], 1)`.

3. **Acceleration test**: the onset fires at the first quarter `t*` where:
   - `g[t*] >= g_min` (minimum growth rate threshold)
   - `g[t*]` is positive for at least `c` consecutive quarters starting at
     `t*` (confirmation period)
   - `s[t*] >= n_min` (minimum absolute count to avoid noise triggers)

4. **Onset quarter**: `q_onset = t*` (the first quarter of the confirmed
   acceleration run).

### 2.3 Default Parameters

| Parameter | Symbol | Default | Allowed Range | Notes |
|-----------|--------|---------|---------------|-------|
| Smoothing window | `w` | 3 | 2--6 | Odd values for centered mode |
| Min growth rate | `g_min` | 0.10 | 0.05--0.50 | 10% quarter-over-quarter |
| Confirmation quarters | `c` | 3 | 2--6 | Must sustain growth |
| Min absolute count | `n_min` | 3 | 1--10 | Smoothed count floor |

### 2.4 Leakage Status

| Property | Onset (this spec) | Midpoint (current) |
|----------|--------------------|--------------------|
| Uses future data | No (trailing window only) | Yes (full series fit) |
| Stable under truncation | Yes | No -- shifts as more data arrives |
| Prospective-safe | Yes | No |

## 3. Edge Cases

### 3.1 Sparse Starts

If a lineage has fewer than `w + c` quarters of data, onset detection is
skipped and the lineage is labelled `onset = NaN` with reason
`insufficient_history`.

### 3.2 Temporary Surges

A single-quarter spike that does not sustain growth for `c` quarters is
**not** an onset.  The confirmation period filter handles this automatically.

### 3.3 Multiple Acceleration Episodes

Some lineages experience more than one growth phase (dormancy followed by
reactivation).  Rules:

- **First onset only**: the onset label marks the first qualifying episode.
  Later acceleration episodes are not onset events.
- **Reactivation tracking**: if downstream tasks need to detect
  reactivations, a separate `reactivation_quarter` column should be added
  by a future task.  This specification covers onset only.

### 3.4 Declining or Flat Lineages

Lineages that never sustain `c` consecutive quarters of growth above
`g_min` receive `onset = NaN` with reason `no_sustained_growth`.

### 3.5 Very Early Onset

If onset fires in the first or second quarter of a lineage's existence,
the detection is flagged as `early_onset = True` for QA review, but it is
still valid.

### 3.6 Field Growth Guard

If the field as a whole is experiencing above-average growth in the onset
quarter, the detection may be a field-level artifact rather than a
lineage-specific signal.  The onset detector should optionally accept a
`--min-field-growth-ratio` parameter (default: disabled) to filter these.

## 4. Output Schema

The onset labeling pipeline must produce a CSV with these columns:

| Column | Type | Description |
|--------|------|-------------|
| `lineage_id` | int | Persistent lineage identifier |
| `onset_quarter` | str | Quarter of onset (e.g., "2015Q3"), or empty |
| `onset_detected` | int | 1 if onset was found, 0 otherwise |
| `onset_reason` | str | Detection reason or skip reason |
| `onset_growth_rate` | float | Growth rate at onset quarter |
| `onset_smoothed_count` | float | Smoothed count at onset quarter |
| `onset_confirmation_length` | int | Consecutive growth quarters |
| `early_onset` | bool | True if onset is in first 2 quarters |
| `smoothing_window` | int | Window parameter used |
| `growth_threshold` | float | Growth rate threshold used |
| `confirmation_window` | int | Confirmation period used |

## 5. Relationship to Existing Labels

### 5.1 Coexistence

Onset labels and midpoint labels are independent columns.  The labeling
pipeline should emit both when run in comparison mode, enabling the
comparison memo (P1.8).

### 5.2 Expected Timing Relationship

Onset typically precedes midpoint by several quarters because onset marks
the beginning of growth while midpoint marks the center of the S-curve.
The lag between onset and midpoint is itself a useful diagnostic.

### 5.3 Migration Path

- Phase 1 experiments use **onset labels** for training and evaluation.
- Midpoint labels are retained only as a **retrospective reference**.
- No new code should default to midpoint labels without explicit opt-in.

## 6. Implementation Guidance for P1.2

The onset detector utilities (`scripts/onset_detector.py` or equivalent)
should implement:

```python
def detect_onset(
    quarters: list[str],
    counts: list[int],
    *,
    smoothing_window: int = 3,
    growth_threshold: float = 0.10,
    confirmation_quarters: int = 3,
    min_count: int = 3,
) -> OnsetResult:
    """Detect the first quarter of sustained growth acceleration."""
    ...
```

Where `OnsetResult` is a dataclass or named tuple containing `quarter`,
`detected`, `reason`, `growth_rate`, `smoothed_count`, and
`confirmation_length`.

The function must be **pure** (no side effects, no file I/O) and testable
with small synthetic count series.

## 7. QA Checklist

Before onset labels are accepted for training:

- [ ] Total onset count is reported
- [ ] Percentage of lineages with onset vs no onset
- [ ] Distribution of onset quarters (histogram)
- [ ] Comparison of onset vs midpoint timing (lag distribution)
- [ ] Sparse-series skip count and reasons
- [ ] Manual spot-check of 5--10 representative lineages
