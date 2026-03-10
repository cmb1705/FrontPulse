# Feature Leakage Audit

Version: 1.0
Date: 2026-03-09
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

## 5. Annotation Convention

Feature groups in `config/features/feature_groups.yaml` now carry a
`leakage_status` field:

- `safe` -- no future data used
- `unsafe` -- uses future data; excluded in leakage-safe mode
- `mixed` -- group contains both safe and unsafe features
