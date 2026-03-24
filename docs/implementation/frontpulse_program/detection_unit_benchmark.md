# Detection Unit Benchmark: Lineage vs Front

**Task**: FP-ax4.7 (P2.7)
**Status**: Complete
**Date**: 2026-03-23

## Purpose

Compare lineage-level and front-level onset detection to determine the
best unit of analysis for operational deployment.  The evaluation uses the
same time-forward holdout contract (train <= 2019Q4, predict >= 2020Q1)
and metrics (ROC-AUC, PR-AUC, detection lag, operational alert volume).

## Data Characteristics

| Dimension | Lineage-Level | Front-Level |
|-----------|--------------|-------------|
| Entities | 5,179 lineages | 22 fronts |
| Observations | 21,609 (lineage, quarter) | ~1,980 (front, quarter) |
| Onsets detected | 231 (4.5%) | ~15-18 (derived from lineage onsets) |
| Mapping coverage | 100% (all lineages) | 3.4% (177 of 5,179 lineages mapped) |
| Mean entity size | 1 lineage | 8.0 lineages per front |
| Quarters span | 2003Q1-2025Q3 (90 quarters) | Same timespan |

## Feasibility Assessment

**Front-level ML training is NOT feasible** with current data:

1. **Insufficient entities**: 22 fronts is far below the ~50 minimum needed
   for reliable 5-fold stratified cross-validation.
2. **Insufficient positive labels**: ~15-18 front-level onsets cannot support
   stratified CV without extreme class imbalance artifacts.
3. **Overfitting risk**: With 22 entities and 14+ convergence features alone,
   any model would memorize rather than generalize.

This assessment would change if the mapping covered more fronts (e.g., through
automated front discovery rather than milestone-based mapping).

## Lineage-Level MSD Results (Reference)

From CatBoost MSD with leakage-safe features (51 features):

| Metric | 5-Fold CV | Time-Forward Holdout |
|--------|-----------|---------------------|
| ROC-AUC | 0.931 | 0.895 |
| PR-AUC | 0.170 | 0.147 |
| Precision @ t=0.07 | 0.156 | 0.078 |
| Recall @ t=0.07 | 0.939 | 0.767 |
| Detection lag (median) | 0.0Q | 0.0Q |
| Detection lag (mean) | -2.5Q | -2.3Q |
| Share at/before onset | 99.1% | 92.1% |

## Tradeoff Analysis

### Lineage-Level Advantages

- **Statistical power**: 5,179 entities and 231 onsets provide robust training
  data for CatBoost.  Cross-validation is reliable with stratified folds.
- **Early detection**: Individual lineages can show onset signals before the
  aggregate front-level metrics shift, providing lead time.
- **Resolution**: Finer-grained detection can distinguish which specific
  lineages within a front are driving growth.
- **Proven performance**: ROC-AUC 0.895 on time-forward holdout, with 92%
  of detections at or before the onset quarter.

### Front-Level Advantages

- **Operational manageability**: 22 fronts (vs 5,179 lineages) are reviewable
  by a human analyst.  At threshold t=0.07, lineage-level produces 1,148
  alerts; front-level aggregation would produce ~15-20.
- **Interpretability**: Fronts have meaningful names ("perovskite silicon
  tandems", "stability engineering") whereas lineages are numbered IDs
  requiring manual inspection.
- **Noise reduction**: Ephemeral lineages (52.1% have lifespan <= 1 quarter)
  are filtered out by the mapping, which requires milestone association.
- **Consumer alignment**: Downstream consumers (BOCPD changepoint detector,
  timeliness scoring, policy reports) operate at the front level.

### Hybrid Is Best

Neither pure approach dominates.  The hybrid architecture addresses all
requirements:

```
Lineage-level detection (statistical power)
    |
    v
Front-level aggregation (operational clarity)
    |
    v
Consumers: BOCPD, timeliness, dashboard, reports
```

**Detect at lineage level** to exploit the full 5,179-entity training set
and achieve 0.895 ROC-AUC.  **Aggregate to front level** by flagging a front
when any constituent lineage exceeds the detection threshold.

This preserves sensitivity while delivering interpretable alerts.

## Recommendation

**Detection unit**: Lineage (via CatBoost MSD).
**Deployment unit**: Front (via aggregation of lineage-level alerts).

A front is flagged when ANY constituent lineage triggers at threshold t=0.07
(extended monitoring) or t=0.15 (watch list).  The two-tier threshold
structure from the holdout evaluation carries forward unchanged.

## Future Work

If the mapping is expanded (e.g., by automated front discovery producing 100+
fronts), re-evaluate front-level ML feasibility using this benchmark script:

```
python scripts/benchmark_detection_units.py
```

The script automatically assesses whether front-level sample sizes meet
minimum thresholds for ML training.

## Artifacts

| File | Description |
|------|-------------|
| `scripts/benchmark_detection_units.py` | Benchmark orchestration script |
| `data/out/experiments/detection_unit_benchmark/benchmark_report.json` | Structured results (generated on run) |
| `data/out/experiments/detection_unit_benchmark/benchmark_report.txt` | Human-readable comparison (generated on run) |
