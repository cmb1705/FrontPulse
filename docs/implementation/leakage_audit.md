# Feature Leakage Audit

Version: 2.0
Date: 2026-03-23
Status: Authoritative

This document classifies every feature group by its temporal leakage risk
and defines the leakage-safe feature path for prospective experiments.

## 1. Leakage Classification

### 1.1 Safe (no future data)

| Group | Features | Mechanism |
|-------|----------|-----------|
| `quarter_flags` | 4 | Binary quarter indicators |
| `lineage_activity` | 9 | Trailing rolling means, YoY deltas |
| Core (safe subset) | 14 of 18 | Trailing velocities, growth rates, novelty, cross-domain citations |

### 1.2 Unsafe (uses future data)

| Group | Features | Leakage mechanism |
|-------|----------|-------------------|
| `logistic_fits` | 4 | S-curve fitted on full cumulative history; inflection point and carrying capacity encode where the series actually peaked |
| Core CD features | 4 | `cd_index` uses future citations to compute a disruption score; `cd_min`, `cd_max`, `disruption_intensity` derived from it |
| `field_baseline` | 20 | Field-wide aggregates computed once on entire corpus without temporal split |
| `field_relative` | 5 | Ratios against `field_baseline` (inherits leakage) |

| `convergence` | 14 | Trailing-only pairwise signals per quarter |

### 1.3 Conditionally Safe

| Group | Features | Condition |
|-------|----------|-----------|
| `author_flow` | 7 | Safe if author counts use only publication-date-based lookback |
| `citation_velocity` | 7 | Safe if citation counts use only publication-date-based lookback |
| `reference_vitality` | 7 | Safe if reference lists use only publication-date-based lookback |
| `topic_diversity` | 7 | Safe if topic labels use only publication-date-based lookback |
| `cross_cluster_bridging` | 7 | Safe if bridging scores use only publication-date-based lookback |

Context metrics (35 total) use trailing rolling windows and are
considered safe under the assumption that the underlying Task 2.1
metrics are computed without future data.  If a context metric's
source uses future citations (e.g., citation velocity based on
inbound citations from later papers), it becomes unsafe.

### 1.4 Label-Derived (conditionally safe)

| Group | Features | Condition |
|-------|----------|-----------|
| `lifecycle` | 6 | Safe for retrospective analysis; LEAKS in time-forward holdout unless computed from training-only labels (see Section 5.5) |

## 2. Leakage-Safe Feature Path

### 2.1 Strict Mode (`leakage_safe`)

30 features: core (minus CD/disruption) + lineage\_activity + quarter\_flags.

**CLI usage:**

```bash
python scripts/multi_signal_detector.py --leakage-safe [...]
```

**Config reference:** `config/features/feature_subset_configs.yaml` ->
`leakage_safe`.

### 2.2 Extended Mode (`leakage_safe_extended`)

65 features: strict mode + all 35 context metrics.

**Config reference:** `config/features/feature_subset_configs.yaml` ->
`leakage_safe_extended`.

## 3. Excluded Features Detail

### 3.1 Logistic Fit Parameters

| Feature | Why it leaks |
|---------|-------------|
| `logistic_carrying_capacity` | L parameter requires the series to plateau before the fit converges |
| `logistic_growth_rate` | k parameter is biased by post-prediction trajectory |
| `logistic_midpoint_idx` | x0 directly encodes the inflection point we are trying to predict |
| `logistic_fit_r2` | Goodness of fit depends on having the full S-curve |

**Computation location:** `scripts/compute_lineage_multisignal_features.py`,
logistic curve fitting section.

### 3.2 CD Disruption Index

| Feature | Why it leaks |
|---------|-------------|
| `cd_index` | Counts future citations (papers citing this work after publication) |
| `cd_min` | Cumulative minimum of `cd_index` |
| `cd_max` | Cumulative maximum of `cd_index` |
| `disruption_intensity` | Product of `cd_index` and paper count |

**Computation location:** `scripts/compute_lineage_multisignal_features.py`,
CD index section.

### 3.3 Field Baseline and Relative Metrics

All 25 field features are computed once on the full corpus.  In a
time-forward evaluation, the field baseline for quarter T should use
only data up to T, but the current pipeline computes it globally.

**Remediation (future):** Recompute field baselines per temporal split.
Until then, exclude from prospective experiments.

## 4. Impact on Current Results

The `strong_signals_top20` feature set contains 7 leakage-prone features
out of 21 (33%).  Performance claims based on this set likely overestimate
prospective accuracy.  The leakage-safe experiments (FP-wq7.6) will
quantify the actual performance gap.

## 5. Onset Detector and Label Pipeline Audit (FP-yfu)

Date: 2026-03-23

### 5.1 Trailing Rolling Mean (onset_detector.py)

**Finding: NO LEAKAGE.** The ``_trailing_rolling_mean()`` function uses
strictly trailing partial windows.  At index ``i``, it averages
``values[max(0, i-window+1) : i+1]``, which never accesses future indices.
Partial windows at early indices are conservative (fewer samples), which
may slightly bias early growth rates but never introduces future data.

### 5.2 Quarter Sorting (label_inflection_points.py)

**Finding: NO LEAKAGE.** The ``detect_onset_for_lineage`` wrapper sorts
by ``quarter_int`` (chronological integer encoding).  The sort determines
processing order only and does not incorporate any future-dependent quantity.

### 5.3 Confirmation Period Look-Ahead

**Finding: ACCEPTABLE (retrospective labeling).** The confirmation period
checks ``c`` future quarters of positive growth starting from the candidate
onset quarter ``t``.  At labeling time, this looks ahead within the
historical series.  This is acceptable because:

- Labels are computed retrospectively from the full historical series and
  then **frozen** for all experiments.
- The confirmation period is part of the label-generation process, analogous
  to how ground-truth event labels are defined in any supervised learning task.
- In prospective deployment, onset detection would naturally wait ``c``
  quarters after the candidate before confirming, which is the expected
  operational behavior (detection lag = ``c`` quarters).

**No code change needed**, but evaluation methodology must ensure labels are
computed before any train/test split is applied.

### 5.4 Field Growth Guard

**Finding: NOT IMPLEMENTED for onset mode.** The specification (Section 3.6)
says ``--min-field-growth-ratio`` should be "default: disabled", so this is
by design.  The guard is implemented in the retrospective detection mode but
not in the onset mode function ``detect_onset_for_lineage()``.

**Recommendation:** Low priority.  If field-level growth artifacts are
observed in onset labels, add a post-processing filter that suppresses
onsets where the field growth ratio exceeds a threshold.

### 5.5 Lifecycle Features from Onset Labels

**Finding: LEAKAGE RISK in time-forward holdout evaluation.**

The lifecycle stage features (``lifecycle_pre_onset``, ``lifecycle_growth``,
``lifecycle_post_maturation``, ``lifecycle_never_grew``) are derived from
onset/maturation labels.  When computed from **full-dataset** labels:

1. For a lineage whose onset falls in the **test period**, the lifecycle
   features encode that this lineage experienced an onset (e.g.,
   ``lifecycle_growth=1`` at the onset quarter).
2. The MSD target ``is_milestone`` is 1 only at the onset quarter.
3. The transition from ``lifecycle_pre_onset=1`` to ``lifecycle_growth=1``
   exactly marks the onset quarter in the feature matrix.
4. The model could learn to read the onset timing directly from lifecycle
   features, bypassing genuine signal detection.

**Severity:** Moderate.  Does not affect retrospective analysis but
invalidates time-forward holdout results if lifecycle features are included.

**Remediation:** When running time-forward holdout evaluation:

- Compute lifecycle features using **only training-set onset labels**.
- For lineages whose onset falls in the test period, set lifecycle features
  to ``lifecycle_pre_onset=1`` (onset not yet known).
- Alternatively, exclude lifecycle features from the time-forward holdout
  entirely and use them only for retrospective analysis.

**Implementation note:** The ``build_lifecycle_features()`` function in
``scripts/compute_lineage_multisignal_features.py`` would need a
``--train-end`` parameter to filter onset labels by temporal cutoff.  This
is a future enhancement for the time-forward holdout task (FP-wq7.7).

### 5.6 Convergence Features

**Finding: SAFE.** All 14 convergence features (``conv_*``) use
trailing-only data: per-quarter author sets, citation references, term sets,
and trailing semantic similarity.  No future data is accessed.  Marked as
``leakage_status: safe`` in ``feature_groups.yaml``.

## 6. Annotation Convention

Feature groups in `config/features/feature_groups.yaml` now carry a
`leakage_status` field:

- `safe` -- no future data used
- `unsafe` -- uses future data; excluded in leakage-safe mode
- `mixed` -- group contains both safe and unsafe features
