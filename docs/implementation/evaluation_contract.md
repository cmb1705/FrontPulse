# Prospective Evaluation Contract

Version: 1.0
Date: 2026-03-10
Status: Authoritative

All detector comparison experiments in FrontPulse must cite this document.
Changes to the contract require a versioned update with rationale.

## 1. Evaluation Regimes

This contract distinguishes three evaluation regimes.  Every experiment
report must state which regime was used.

### 1.1 Cross-Validation (CV)

- **Purpose**: hyperparameter selection and model comparison
- **Split method**: stratified k-fold (k=5 default, random_state=42)
- **Label leakage risk**: high -- CV does not respect temporal order
- **Use case**: comparing model families, tuning threshold
- **Reporting**: mean and std of all metrics across folds

### 1.2 Retrospective Replay

- **Purpose**: full-history evaluation of a trained model
- **Split method**: train on all available data, predict on the same period
- **Label leakage risk**: high -- training and test overlap
- **Use case**: illustrative diagnostics and visual timelines only
- **Reporting**: metrics must be labelled "retrospective"; never used for
  claims about prospective performance

### 1.3 Time-Forward Holdout

- **Purpose**: simulate prospective detection as closely as possible
- **Split method**: strict temporal cutoff; all training data precedes all
  test data
- **Label leakage risk**: low (if features are also leakage-free)
- **Default splits** (from `config/splits/`):

| Split name | Train | Dev | Test |
|------------|-------|-----|------|
| 3-way (`msd_timeforward.yaml`) | 2003Q1--2017Q4 | 2018Q1--2021Q4 | 2022Q1--2025Q3 |
| 2-way (`msd_timeforward_holdout_2020.yaml`) | 2003Q1--2019Q4 | -- | 2020Q1--2025Q3 |

- **Reporting**: this is the primary regime for performance claims

## 2. Detection Windows

### 2.1 Definitions

- **Inflection quarter** (`q_inflection`): the quarter when exponential
  growth begins in a lineage's publication count, as determined by the
  labeling pipeline.
- **Detection quarter** (`q_detection`): the first quarter in which a
  detector reports a positive alert for the lineage.
- **Detection lag** (`lag`): `q_detection - q_inflection`, measured in
  quarters.
  - `lag < 0`: early detection (before inflection -- unusual)
  - `lag = 0`: on-time detection
  - `lag > 0`: late detection (normal in early-warning context)

### 2.2 Lag Interpretation

| Lag category | Value | Interpretation |
|--------------|-------|----------------|
| Early | lag < 0 | Detected before inflection; flag for review |
| On-time | lag = 0 | Ideal |
| Acceptable | 0 < lag <= 3 | Within 3 quarters |
| Tolerable | 3 < lag <= 8 | Late but still useful |
| Missed | lag > 8 or no detection | Effectively missed |

### 2.3 Persistence Window

A detection is confirmed only if the alert remains above the probability
threshold for `persistence_window` consecutive quarters (default: 2).
Single-quarter spikes are treated as noise unless they exceed a separate
high-confidence threshold.

## 3. Metrics

### 3.1 Classification Metrics

All classification metrics are computed at the **lineage-quarter** level
(binary: is this quarter the inflection quarter?).

| Metric | Definition | Notes |
|--------|------------|-------|
| Precision | TP / (TP + FP) | At operating threshold |
| Recall | TP / (TP + FN) | At operating threshold |
| F1 | Harmonic mean of precision and recall | |
| PR-AUC | Area under precision-recall curve | Primary ranking metric |
| ROC-AUC | Area under ROC curve | Secondary metric |

### 3.2 Lag Metrics

| Metric | Definition | Notes |
|--------|------------|-------|
| Median lag | Median of `lag` across true positives | Primary timeliness metric |
| Mean lag | Mean of `lag` across true positives | Sensitive to outliers |
| Lag coverage | Fraction of true inflections detected at all | TP / (TP + FN) |
| Pct lag <= 0Q | Fraction of TPs with lag <= 0 | Early or on-time |
| Pct lag <= 2Q | Fraction of TPs with lag <= 2 | Within 2 quarters |
| Pct lag <= 4Q | Fraction of TPs with lag <= 4 | Within 1 year |

### 3.3 Timeliness Metrics (to be implemented in P3.4)

These metrics are defined here for contract purposes.  Implementation is
deferred to Phase 3 (`FP-92e.4`).

#### NAB Score (Numenta Anomaly Benchmark)

- Application-profile-weighted score that rewards early detection and
  penalizes late detection and false positives.
- Three standard profiles: `standard`, `reward_low_FP`, `reward_low_FN`.
- Null-detector baseline (always negative) scores 0.0.
- Perfect detector scores 100.0.
- Report all three profiles; use `standard` as the primary.

#### Expected Detection Delay (EDD)

- The expected number of quarters between the true inflection and the
  detector's first alert, conditioned on detection occurring.
- Lower is better.  Undefined if the inflection is never detected.

#### Average Run Length (ARL)

- **ARL0**: average quarters between false alarms (higher is better).
- **ARL1**: average quarters from inflection to first detection (lower is
  better; equivalent to EDD when detection always occurs).
- Report both ARL0 and ARL1.

### 3.4 Reporting Requirements

Every experiment report must include:

1. Evaluation regime (CV / retrospective / time-forward)
2. Split specification (quarter ranges, config file path)
3. Classification metrics at the operating threshold
4. PR-AUC and ROC-AUC (threshold-free)
5. Lag distribution summary (median, mean, pct <= 2Q, coverage)
6. Operating threshold and persistence window used
7. Number of true inflections in the test set

When timeliness scoring is available (post-P3.4), also include:

8. NAB score (standard profile)
9. EDD
10. ARL0 and ARL1

## 4. False Alarm Accounting

### 4.1 False Positive Definition

A false positive occurs when the detector alerts on a lineage-quarter that
is not labelled as an inflection quarter.

### 4.2 Cost Model

FrontPulse operates in a **low false-alarm tolerance** regime: each alert
generates manual review work.  The contract requires:

- Precision >= 0.20 at the operating threshold (at least 1 in 5 alerts is
  a true inflection)
- FPR (false positive rate) reported alongside precision and recall
- False positives broken down by milestone-linked vs orphan lineages

### 4.3 Threshold Policy

The **operating threshold** and **evaluation threshold** may differ:

- **Operating threshold**: conservative (e.g., 0.70) for production use
- **Evaluation threshold**: tuned for recall (e.g., 0.07) for research analysis

Both must be reported.  Claims about detection performance must specify
which threshold applies.

## 5. Baseline Comparison Protocol

### 5.1 Registered Baselines

All detector experiments must compare against these baselines
(defined in `config/baselines/methods.json`):

1. **MSD LightGBM** -- supervised multi-signal detector (current best)
2. **Simple heuristics** -- single-feature threshold rules
3. **Kleinberg burst** -- citation burst detection
4. **Semantic changepoint** -- unsupervised changepoint detection

### 5.2 Comparison Rules

- Use the **same test set** for all methods being compared
- Report **all metrics** from Section 3 for every method
- Use the **same persistence window** across methods
- A new method must improve PR-AUC **and** median lag (or improve one
  without regressing the other beyond 10%) to be considered an improvement

## 6. Data Integrity Requirements

- Features must be **leakage-free**: no feature may use information from
  quarters after the prediction target
- Labels must be **temporally anchored**: the inflection quarter is fixed
  at labeling time and does not shift with the test set boundary
- Train/test contamination: no lineage may appear in both train and test
  sets for the same quarter range
- Feature coverage: report the percentage of lineage-quarters with
  complete feature vectors (target: >= 90%)
