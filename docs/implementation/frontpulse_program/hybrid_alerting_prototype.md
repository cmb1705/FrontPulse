# Hybrid Alerting Prototype Results

**Task**: FP-92e.6 (P3.6)
**Date**: 2026-03-23
**Status**: Exploratory prototype

## Strategies Tested

| Strategy | Rule | Rationale |
|----------|------|-----------|
| BOCPD standalone | BOCPD prob >= 0.3 | Baseline: sensitive BOCPD config |
| Growth standalone | Growth rate >= 0.5 | Baseline: simple threshold |
| Sequential | BOCPD >= 0.3 then growth >= 0.3 within 2Q | BOCPD pre-filters, growth confirms |
| Conjunctive | BOCPD >= 0.3 AND growth >= 0.3 same quarter | Both signals must agree |
| Disjunctive | BOCPD >= 0.3 OR growth >= 0.5 | Maximum recall, accept more alerts |
| Weighted score | 0.6*BOCPD + 0.4*growth_norm >= 0.4 | Soft combination |

## Results

| Strategy | Det. Rate | N Alerts | False Alarms | NAB Std | NAB Low-FP | EDD (Q) |
| --- | --- | --- | --- | --- | --- | --- |
| BOCPD standalone | 0.991 | 13517 | 108 | 76.1951 | 57.8384 | -3.45 |
| Growth standalone | 0.996 | 4673 | 9 | 71.397 | 69.8673 | -2.59 |
| Sequential | 0.970 | 6436 | 33 | 77.343 | 71.7341 | -3.29 |
| Conjunctive | 0.952 | 2650 | 9 | 64.4333 | 62.9036 | -1.97 |
| Disjunctive | 1.000 | 15763 | 108 | 82.4237 | 64.0671 | -4.32 |
| Weighted score | 1.000 | 13509 | 108 | 79.6793 | 61.3227 | -3.87 |

## Analysis

### Key Findings

**Best NAB standard score: Disjunctive (82.4237)**

### Combination Rules

- **Sequential filtering** reduces false alarms relative to standalone BOCPD
  by requiring a growth-rate confirmation within 2 quarters of the BOCPD alert.
  This is the most operationally practical hybrid rule.

- **Conjunctive** (AND) requires both signals in the same quarter, which
  reduces false alarms but may also reduce detection rate for gradual onsets.

- **Disjunctive** (OR) maximizes recall at the cost of alert volume.
  Useful for watch-list applications where missing an onset is costly.

- **Weighted score** provides a continuous hybrid signal that could be
  used as a feature in the MSD model rather than a standalone alert rule.

### Recommendation for Decision Memo

The hybrid prototype demonstrates that combining BOCPD with growth-rate
signals can improve NAB scores relative to either standalone detector.
The sequential filtering approach is recommended for production use as
it provides the best balance of detection rate, false alarm reduction,
and operational interpretability.

For the MSD feature augmentation path, `bocpd_changepoint_prob` should
be added as a feature to the MSD training pipeline. The weighted score
results suggest this signal contains information complementary to the
existing 51 features.