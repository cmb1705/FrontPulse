# Detector Stack Decision Memo

**Task**: FP-92e.7 (P3.7)
**Date**: 2026-03-23
**Status**: Decision made
**Scope**: Recommend the deployment-facing detector stack for FrontPulse

## 1. Context

Phase 3 evaluated three detector families on PSC lineage data (5,179
lineages, 91 quarters, 231 known onsets) using the timeliness scoring
contract (NAB, EDD, ARL metrics):

1. **BOCPD** -- Bayesian Online Changepoint Detection (Poisson-Gamma)
2. **MSD** -- Multi-Signal Detector (CatBoost, 51 leakage-safe features)
3. **Simple baselines** -- growth-rate thresholds and random detector

Phase 3 also prototyped hybrid combination rules and produced calibration
data for BOCPD parameters.

## 2. Evidence Summary

### 2.1 Standalone Detector Performance

| Detector | NAB Std | Det. Rate | EDD (Q) | Alerts | ARL0 (Q) |
|----------|---------|-----------|---------|--------|-----------|
| BOCPD (sensitive) | 76.2 | 99.1% | -3.45 | 13,517 | 200 |
| BOCPD (default) | 67.3 | 90.9% | -2.68 | 9,874 | 204 |
| Growth-rate (0.5) | 71.4 | 99.6% | -2.59 | 4,673 | 2,401 |
| Growth-rate (1.0) | 63.6 | 96.5% | -1.74 | 3,651 | 3,601 |
| Random (5%) | 23.2 | 67.5% | +4.08 | 1,070 | 5,402 |
| MSD CatBoost (CV) | -- | 30.7% | 0.0 | -- | -- |
| MSD CatBoost (holdout) | -- | 76.7% | 0.0 | -- | -- |

**Key observations**:

- BOCPD has the earliest detection (EDD = -3.45Q at sensitive config).
- Growth-rate (0.5) is a surprisingly strong baseline (NAB 71.4) with
  far fewer alerts than BOCPD.
- MSD has higher precision but lower recall in CV (30.7%); holdout recall
  is higher at looser threshold (76.7% at threshold=0.07).
- MSD NAB scores are not directly comparable because MSD was evaluated
  with a different methodology (precision/recall/F1 rather than NAB).

### 2.2 Hybrid Combination Results

| Strategy | NAB Std | Det. Rate | EDD (Q) | Alerts |
|----------|---------|-----------|---------|--------|
| Disjunctive (BOCPD OR growth) | **82.4** | 100% | -4.32 | 15,763 |
| Weighted (0.6*BOCPD + 0.4*growth) | 79.7 | 100% | -3.87 | 13,509 |
| Sequential (BOCPD -> growth) | 77.3 | 97.0% | -3.29 | 6,436 |
| BOCPD standalone | 76.2 | 99.1% | -3.45 | 13,517 |
| Conjunctive (BOCPD AND growth) | 64.4 | 95.2% | -1.97 | 2,650 |

**Key observations**:

- Hybrid OR achieves the highest NAB score (82.4) with 100% detection.
- Sequential filtering (BOCPD pre-filter + growth confirmation) achieves
  NAB 77.3 with the best alert-count tradeoff (6,436 alerts).
- Adding BOCPD to growth-rate improves NAB by 11 points (71.4 -> 82.4).

### 2.3 BOCPD Calibration Findings

- **Hazard rate** and **detection window** are the most influential parameters.
- **Prior parameters** (alpha0, beta0) have modest effect.
- Default config (hz=1/50, dw=3, thr=0.5) is conservative and general-purpose.
- Sensitive config (hz=1/20, dw=5, thr=0.3) is recommended for watch-list use.
- BOCPD detects all changepoints, not specifically onsets: 92% false alarm rate
  is structural (most lineages experience non-onset regime changes).

## 3. Decision

### 3.1 Production Detector Stack

**Recommended: Hybrid sequential (BOCPD + growth-rate confirmation)**

```
Layer 1: BOCPD (sensitive config)
    -> Flags quarters with P(recent changepoint) >= 0.3
    -> Provides early warning (median 3.5 quarters before onset label)

Layer 2: Growth-rate confirmation
    -> Confirms BOCPD alerts with growth_rate >= 0.3 within 2 quarters
    -> Reduces false alarm volume by ~52% vs standalone BOCPD

Layer 3 (future): MSD classifier
    -> Can be applied to BOCPD-flagged lineages for precision classification
    -> Requires bocpd_changepoint_prob as MSD feature (Phase 4 integration)
```

**Rationale**:

1. **NAB 77.3** -- best score among non-maximum-alert strategies.
2. **97% detection rate** -- catches nearly all onsets.
3. **6,436 alerts** -- 52% fewer than disjunctive, operationally manageable.
4. **EDD = -3.29Q** -- early detection preserved.
5. **No ML dependency** -- runs without trained models or feature engineering.
6. **Domain-portable** -- same BOCPD parameters work on any count series.

### 3.2 Research Detector (Unchanged)

The MSD CatBoost classifier remains the research-grade detector for:

- Precision-critical analyses where false positives are costly.
- Feature importance studies (SHAP analysis of 51 features).
- Historical retrospective evaluation (not online detection).

MSD is not recommended for production alerting because:

- It requires feature engineering pipeline (80+ features per lineage-quarter).
- CV precision (18.5%) is low for operational use.
- It detects onsets at the onset quarter (lag = 0), not prospectively.

### 3.3 Supported Deployment Paths

| Path | Stack | Use Case |
|------|-------|----------|
| **Production hybrid** | BOCPD + growth confirmation | Quarterly monitoring, watch-lists |
| **Research analysis** | MSD CatBoost | Feature studies, retrospective analysis |
| **Maximum recall** | BOCPD OR growth-rate | When missing an onset is unacceptable |
| **Simplest viable** | Growth-rate (0.5) | When no BOCPD infra is available |

### 3.4 Not Supported

| Path | Reason |
|------|--------|
| Front-level ML | Only 22 fronts (infeasible for ML training) |
| BOCPD standalone (default) | Lower NAB than hybrid or growth-rate baseline |
| Random/naive baselines | For benchmarking context only, not deployment |

## 4. Open Follow-Up Questions

### 4.1 MSD Feature Augmentation

Adding `bocpd_changepoint_prob` as a feature to MSD training could improve
MSD's recall. The weighted score results (NAB 79.7) suggest BOCPD signal
contains complementary information. This should be tested in Phase 4.

**Action**: Add `bocpd_changepoint_prob` to `compute_lineage_multisignal_features.py`
output and retrain MSD. Compare PR-AUC with and without the feature.

### 4.2 CRISPR Domain Transfer

The BOCPD parameters were calibrated on PSC data. The sensitive config
(hz=1/20, dw=5) assumes changepoints every ~5 years, which may not
transfer to a younger domain like CRISPR where fronts evolve faster.

**Action**: After CRISPR ingestion runs, recalibrate BOCPD on CRISPR
lineage timeseries and compare parameter optima.

### 4.3 Alert Fatigue Management

The sequential strategy produces 6,436 alerts across 21,608 lineage-quarters
(30% alert rate). For operational use, additional filtering is needed:

- Minimum sustained alert duration (e.g., 2+ consecutive quarters).
- Minimum new_works count threshold (e.g., >= 3 papers/quarter).
- Priority ranking by composite score or BOCPD probability magnitude.

### 4.4 Maturation Detection Integration

The BOCPD detector may also detect maturation-phase regime changes (growth
slowdown). Extending the hybrid architecture to detect both onset and
maturation changepoints is a natural follow-on. This aligns with the
lifecycle stage model in the front-level series contract.

## 5. Artifacts Produced in Phase 3

| File | Description |
|------|-------------|
| `src/bocpd.py` | BOCPD implementation (detect_changepoints, run_bocpd_on_fronts) |
| `scripts/run_bocpd_detector.py` | CLI wrapper for BOCPD on front-level series |
| `scripts/calibrate_bocpd.py` | Parameter calibration on PSC data |
| `scripts/benchmark_bocpd_vs_msd.py` | Detector comparison benchmark |
| `scripts/prototype_hybrid_alerting.py` | Hybrid alerting prototype |
| `tests/test_bocpd.py` | 17 unit tests for BOCPD module |
| `docs/.../bocpd_package_decision.md` | Package selection and interface spec |
| `docs/.../bocpd_calibration_report.md` | Calibration results (15 configs) |
| `docs/.../detector_benchmark_report.md` | 5-detector comparison |
| `docs/.../hybrid_alerting_prototype.md` | 6 hybrid strategy results |
| `docs/.../detector_stack_decision.md` | This memo |
