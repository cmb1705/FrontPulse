# Deployment Unit Decision Memo

**Task**: FP-ax4.8 (P2.8)
**Status**: Decision made
**Date**: 2026-03-23
**Depends on**: FP-ax4.7 (detection unit benchmark)

## Decision

**Detection unit**: Lineage (via CatBoost MSD, 51 leakage-safe features).
**Deployment unit**: Front (via aggregation of lineage-level alerts).
**Architecture**: Hybrid -- detect at lineage level, aggregate to front level.

## Evidence Summary

The FP-ax4.7 benchmark established:

| Factor | Lineage-Level | Front-Level |
|--------|--------------|-------------|
| Entities | 5,179 | 22 |
| Training labels | 231 onsets (4.5%) | ~15-18 (derived) |
| ML feasibility | Yes (ROC-AUC 0.895) | No (insufficient samples) |
| Alert volume (t=0.07) | 1,148 | ~15-20 (after aggregation) |
| Interpretability | Low (numeric IDs) | High (named fronts) |
| Mapping coverage | 100% | 3.4% (177/5,179 lineages) |

Front-level ML training is infeasible with 22 fronts and ~15 onset labels.
Lineage-level detection has proven performance (0.895 holdout ROC-AUC).

## Supported Paths

### 1. Production Path: Hybrid Detection + Aggregation

```
Lineage timeseries --> CatBoost MSD (lineage-level)
    |
    v
Lineage onset probabilities --> Threshold (t=0.07 / t=0.15)
    |
    v
Lineage alerts --> Front aggregation (via mapping)
    |
    v
Front-level alerts --> BOCPD / timeliness / dashboard
```

- A front is flagged when ANY constituent lineage exceeds the threshold.
- Two-tier thresholds: t=0.15 (watch list, ~464 lineage alerts) and
  t=0.07 (extended monitoring, ~1,148 lineage alerts).
- Front-level consumers receive onset_detected, onset_quarter,
  lifecycle_stage, and the triggering lineage IDs.

### 2. Research Path: Lineage-Level Analysis

Lineage-level detection remains the primary research output for:

- Feature importance analysis (SHAP on 51 features)
- Onset timing studies (detection lag distributions)
- Convergence channel evaluation (conv_* features)
- Cross-domain generalization (PSC vs CRISPR comparison)

Lineage-level results feed into publications and dissertation analysis.

## Unsupported Paths

### Front-Level ML (Rejected)

Training a separate CatBoost model on front-level features is not supported
because:

1. 22 fronts cannot produce reliable 5-fold stratified CV.
2. ~15 front-level onsets is below the minimum for meaningful classification.
3. Any model would memorize the training set.

**Condition for revisiting**: If automated front discovery produces 100+
fronts with 30+ onsets, re-run `scripts/benchmark_detection_units.py` to
reassess feasibility.

### Pure Lineage-Level Deployment (Rejected for Operations)

Deploying raw lineage-level alerts (1,148 at t=0.07) to operational consumers
is not supported because:

1. No human analyst can review 1,148 lineage alerts per quarter.
2. Lineage IDs are opaque without front context.
3. Downstream BOCPD and timeliness scoring expect front-level series.

Lineage-level results remain available for research use.

## Implications for Downstream Work

### BOCPD (Phase 3, FP-92e)

BOCPD operates on front-level time series (new_works per front per quarter).
The front onset series from `scripts/compute_front_level_features.py` is the
input.  BOCPD does NOT depend on lineage-level MSD probabilities directly.

### Timeliness Scoring (FP-92e.4)

Timeliness scoring compares detected onset quarters against ground truth.
For front-level evaluation, the ground truth onset = earliest lineage onset
among mapped lineages.  The detected onset = first quarter where any
constituent lineage exceeds the MSD threshold.

### Horizon Scanner (FP-79g)

The operational horizon scanner pipeline reports at the front level.
Lineage-level alerts are aggregated before presentation.

## Front-Level Series Contract

The canonical front-level series schema is defined in
`docs/implementation/frontpulse_program/front_level_series_contract.md`.
All consumers must use this contract.

## Artifacts

| Artifact | Path |
|----------|------|
| Benchmark script | `scripts/benchmark_detection_units.py` |
| Benchmark document | `docs/implementation/frontpulse_program/detection_unit_benchmark.md` |
| This decision memo | `docs/implementation/frontpulse_program/deployment_unit_decision.md` |
| Front series contract | `docs/implementation/frontpulse_program/front_level_series_contract.md` |
| Front feature script | `scripts/compute_front_level_features.py` |
