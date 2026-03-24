# Maturation Label Specification

Version: 1.0
Date: 2026-03-24
Status: Draft

This document defines the maturation labeling rules that complement the onset
label specification (v1.0). Where onset detection identifies the first quarter
of sustained growth acceleration (the lower elbow of the S-curve), maturation
detection identifies the first quarter of sustained growth deceleration (the
upper elbow).

## 1. Motivation

The current FrontPulse pipeline detects onset events -- when a research lineage
begins accelerating. But the lifecycle of a research front has a second critical
transition: when growth slows and the lineage enters saturation, decline, or
dormancy. Detecting this "upper elbow" serves three purposes:

1. **Lifecycle completion**: onset + maturation together define the full growth
   envelope of a research lineage, enabling lifecycle-stage classification
   (pre-onset, growth, maturation, post-maturation).

2. **False positive reduction**: lineages that have already matured should not
   trigger onset alerts on noise fluctuations. Maturation labels provide a
   filtering signal for the MSD to reduce false positives in the horizon
   scanner.

3. **Convergence context**: maturation is when researchers are most likely to
   migrate to adjacent lineages, making it a leading indicator for cross-front
   convergence events.

## 2. Maturation Definition

### 2.1 Plain Language

The **maturation quarter** for a lineage is the first quarter in which
publication growth rate transitions from positive to sustained non-positive,
indicating that the lineage has passed its peak growth phase. Three subtypes
capture distinct post-peak trajectories.

### 2.2 Maturation Subtypes

| Subtype | Description | Growth Pattern |
|---------|-------------|----------------|
| `saturation` | S-curve plateau; growth rate drops to near-zero but the lineage remains active | g[t] falls within [-g_min, +g_min] for c quarters |
| `senescence` | Active decline; publication count decreases | g[t] < -g_min for c quarters |
| `dormancy_entry` | Activity ceases; new works count drops to zero or near-zero | s[t] < n_floor for c quarters |

### 2.3 Operational Rules

Given a lineage's quarterly new-works series `n[t]` for quarters
`t = 1, 2, ..., T`:

1. **Smoothing**: compute a trailing rolling mean `s[t]` with window `w`
   (default: 3 quarters, trailing only -- prospective-safe).

2. **Growth rate**: compute the quarter-over-quarter growth rate
   `g[t] = (s[t] - s[t-1]) / max(s[t-1], 1)`.

3. **Peak detection**: identify the quarter `t_peak` with the highest smoothed
   count `s[t_peak]`. Maturation can only occur at or after `t_peak`.

4. **Deceleration test**: scanning from `t_peak` onward, maturation fires at
   the first quarter `t*` where one of these conditions holds for `c`
   consecutive quarters starting at `t*`:

   - **Saturation**: `|g[t]| < g_min` for all quarters in the window, AND
     `s[t] >= n_min` (still active, just flat).
   - **Senescence**: `g[t] < -g_min` for all quarters in the window (sustained
     decline).
   - **Dormancy entry**: `s[t] < n_floor` for all quarters in the window
     (activity effectively ceased).

   Priority: if multiple subtypes could match at the same quarter, prefer
   `dormancy_entry` > `senescence` > `saturation` (most severe first).

5. **Maturation quarter**: `q_maturation = t*` (the first quarter of the
   confirmed deceleration/plateau run).

### 2.4 Default Parameters

| Parameter | Symbol | Default | Allowed Range | Notes |
|-----------|--------|---------|---------------|-------|
| Smoothing window | `w` | 3 | 2--6 | Trailing only (prospective-safe) |
| Min growth rate | `g_min` | 0.10 | 0.05--0.50 | Same threshold as onset |
| Confirmation quarters | `c` | 3 | 2--6 | Must sustain deceleration |
| Min absolute count | `n_min` | 3 | 1--10 | Smoothed count floor for saturation |
| Activity floor | `n_floor` | 1 | 0--3 | Below this = dormancy entry |

### 2.5 Leakage Status

| Property | Maturation (this spec) | Notes |
|----------|----------------------|-------|
| Uses future data | No (trailing window only) | Same as onset |
| Stable under truncation | Mostly -- peak may shift with new data | See Section 3.6 |
| Prospective-safe | Yes | Trailing smoothing + backward-only scan |

## 3. Edge Cases

### 3.1 Sparse Series

If a lineage has fewer than `w + c` quarters of data, maturation detection is
skipped. Label: `maturation_detected = 0`, reason = `insufficient_history`.

### 3.2 No Peak (Monotonically Growing)

If the lineage is still growing at the end of the series (no sustained
deceleration detected), maturation is not assigned. Label:
`maturation_detected = 0`, reason = `still_growing`.

### 3.3 Lineages That Never Grew

Lineages that never exceeded `n_min` smoothed count should not receive
maturation labels, since they never had a meaningful growth phase. Label:
`maturation_detected = 0`, reason = `never_reached_threshold`.

### 3.4 Multiple Deceleration Episodes

Some lineages experience growth-decline-regrowth cycles. Rules:

- **First maturation only**: the maturation label marks the first qualifying
  deceleration episode after the peak.
- This is symmetric with the onset spec, which marks only the first onset.

### 3.5 Very Late Maturation

If maturation fires in the last `c` quarters of available data, it is
flagged as `late_maturation = True` for QA review, since the confirmation
window is at the series boundary and may be truncated by data availability
rather than actual trajectory change.

### 3.6 Peak Instability Under Truncation

The peak quarter `t_peak` may shift as new data arrives (a new quarter might
exceed the previous maximum). This means maturation detection is not fully
stable under truncation: adding a quarter of higher-than-peak growth would
retroactively move the peak forward and potentially invalidate a previously
detected maturation.

Mitigation: in operational (prospective) mode, once a maturation label is
assigned, it is not retroactively removed. The assessment history
(FP-79g.3) tracks when labels were first assigned versus when they might
have been invalidated.

## 4. Interaction with Existing Features

### 4.1 Dormancy Features

The existing `dormancy_length` and `awakening_intensity` features (computed
in `scripts/compute_lineage_multisignal_features.py`) track consecutive
zero-output quarters and rebound intensity. These are complementary to but
distinct from maturation detection:

- **Dormancy features**: measure current dormancy state (how long silent,
  how strong the rebound).
- **Maturation detection**: identifies the transition point where growth
  stops -- which may or may not lead to dormancy.

A lineage can mature (enter saturation) without going dormant.

### 4.2 Onset Interaction

Maturation can only meaningfully occur after onset. In practice, the
maturation detector does not require an onset label -- it independently
finds the peak and scans for deceleration. But when both labels exist,
the interval (onset_quarter, maturation_quarter) defines the lineage's
active growth window.

### 4.3 Lifecycle Stage Feature

With both onset and maturation labels, a categorical lifecycle stage can
be derived per lineage-quarter:

| Stage | Condition |
|-------|-----------|
| `pre_onset` | Before onset quarter (or no onset detected) |
| `growth` | Between onset and maturation quarters |
| `post_maturation` | After maturation quarter |
| `never_grew` | No onset and no maturation detected |

This categorical feature can be one-hot encoded for MSD integration.

## 5. Output Schema

The maturation labeling pipeline must produce a CSV with these columns:

| Column | Type | Description |
|--------|------|-------------|
| `lineage_id` | int | Persistent lineage identifier |
| `maturation_quarter` | str | Quarter of maturation, or empty |
| `maturation_detected` | int | 1 if maturation was found, 0 otherwise |
| `maturation_type` | str | `saturation`, `senescence`, `dormancy_entry`, or empty |
| `maturation_reason` | str | Detection reason or skip reason |
| `maturation_growth_rate` | float | Growth rate at maturation quarter |
| `maturation_smoothed_count` | float | Smoothed count at maturation quarter |
| `maturation_peak_quarter` | str | Quarter of peak smoothed count |
| `maturation_peak_count` | float | Peak smoothed count value |
| `maturation_confirmation_length` | int | Consecutive qualifying quarters |
| `late_maturation` | bool | True if maturation is in last c quarters |
| `smoothing_window` | int | Window parameter used |
| `growth_threshold` | float | Growth rate threshold used |
| `confirmation_window` | int | Confirmation period used |
| `activity_floor` | int | Activity floor parameter used |

### 5.1 MSD-Compatible Sibling File

Following the onset pattern, the labeling pipeline should also emit an
MSD-compatible file `maturation_labels_msd.csv` with:

| Column | Type | Description |
|--------|------|-------------|
| `lineage_id` | int | Persistent lineage identifier |
| `quarter` | str | Maturation quarter |
| `is_maturation` | int | Always 1 (only detected maturations included) |
| `maturation_type` | str | Subtype classification |

## 6. Relationship to Onset Labels

### 6.1 Expected Timing Relationship

Maturation typically follows onset by multiple quarters. The lag between
onset and maturation represents the active growth window duration. This
interval is itself a useful diagnostic:

- Short interval: rapid boom-bust cycle
- Long interval: sustained growth phase
- No maturation after onset: still actively growing

### 6.2 Coexistence

Onset and maturation labels are independent columns. The labeling pipeline
should support `--mode maturation` to produce maturation labels, parallel
to the existing `--mode onset`.

## 7. Implementation Guidance

The maturation detector should implement:

```python
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
    """Detect the first quarter of sustained growth deceleration."""
    ...
```

Where `MaturationResult` is a frozen dataclass containing `quarter`,
`detected`, `maturation_type`, `reason`, `growth_rate`, `smoothed_count`,
`peak_quarter`, `peak_count`, `confirmation_length`, and `late_maturation`.

The function must be **pure** (no side effects, no file I/O) and testable
with small synthetic count series.

## 8. QA Checklist

Before maturation labels are accepted for MSD integration:

- [ ] Total maturation count is reported
- [ ] Percentage of lineages with maturation vs no maturation
- [ ] Distribution of maturation quarters (histogram)
- [ ] Breakdown by maturation subtype (saturation/senescence/dormancy_entry)
- [ ] Comparison of maturation vs onset timing (lag distribution)
- [ ] Sparse-series skip count and reasons
- [ ] Manual spot-check of 5--10 representative lineages
- [ ] Verify maturation never precedes onset for lineages with both labels
