# Stability Baseline Report

Version: 1.0
Date: 2026-03-23
Status: Authoritative
Task: FP-ax4.1 (P2.1 Quantify lineage lifespan and instability baseline)

This report quantifies lineage lifespan distributions and partition
instability on the frozen PSC baseline (5,179 lineages, 91 quarters,
2003Q1--2025Q3).  It establishes the before-state that later stability
interventions (ECG ensemble clustering, stable-lineage filtering) must
improve upon.

## 1. Data Sources

| Asset | Location | Records |
|-------|----------|---------|
| Lineage timeseries | `data/out/02_lineage_tracking/lineage_timeseries.csv` | 21,608 rows |
| Lineage registry | `data/out/02_lineage_tracking/lineage_registry.json` | 5,179 lineages |
| Partition files | `data/out/cache_cum/partitions_cum/part_YYYYQN.json` | 91 quarters |
| Onset labels | `data/out/02_lineage_tracking/onset_labels.csv` | 231 detected |

All data was generated from the PSC baseline freeze (`psc_baseline_freeze.md`).

## 2. Lineage Lifespan Distribution

### 2.1 Summary Statistics

| Metric | Value |
|--------|-------|
| Total lineages | 5,179 |
| Mean lifespan | 4.2 quarters |
| Median lifespan | 1 quarter |
| Std deviation | 7.6 quarters |
| Min / Max | 1 / 67 quarters |

### 2.2 Percentile Distribution

| Percentile | Lifespan (quarters) |
|------------|---------------------|
| P5 | 1 |
| P10 | 1 |
| P25 | 1 |
| P50 (median) | 1 |
| P75 | 3 |
| P90 | 10 |
| P95 | 18 |
| P99 | 39 |

### 2.3 Lifespan Buckets

| Lifespan range | Count | Percentage |
|----------------|-------|------------|
| 1 quarter (ephemeral) | 2,698 | 52.1% |
| 2 quarters | 788 | 15.2% |
| 3--4 quarters | 657 | 12.7% |
| 5--8 quarters | 460 | 8.9% |
| 9--16 quarters | 282 | 5.4% |
| 17--32 quarters | 186 | 3.6% |
| 33--64 quarters | 104 | 2.0% |
| 65+ quarters | 4 | 0.1% |

### 2.4 Key Finding: Ephemeral Majority

Over half (52.1%) of all lineages appear for exactly one quarter and
then disappear.  Only 24.5% survive 4+ quarters, and only 12.7% survive
8+ quarters.  This extreme right-skew means that the majority of lineages
are too short-lived to support meaningful onset detection (which requires
at least `w + c = 6` quarters of history).

### 2.5 Lifespan vs Onset Detection

| Cohort | N | Mean lifespan | Median lifespan |
|--------|---|---------------|-----------------|
| With onset detected | 231 | 24.9 quarters | 22.0 quarters |
| Without onset | 4,948 | 3.2 quarters | 1.0 quarter |

Lineages with detected onsets live approximately 6x longer than the
population average and 22x longer than the median.  Onset detection
inherently selects for long-lived lineages via the confirmation period.

Onset lineage lifespan percentiles: P10=8, P25=11, P50=22, P75=35, P90=45.

## 3. Partition Instability

### 3.1 Variation of Information (VI)

VI measures how much the partition changes between consecutive quarters.
Higher values indicate more instability.  VI is computed at the partition
level (all nodes in a quarter), not per-lineage.

| Metric | Value |
|--------|-------|
| Quarters with VI data | 90 |
| Mean VI | 1.909 bits |
| Median VI | 1.916 bits |
| Std deviation | 0.662 bits |
| Min VI | 0.431 bits (2004Q2) |
| Max VI | 2.862 bits (2023Q1) |

### 3.2 VI Temporal Trend

| Period | Mean VI |
|--------|---------|
| Early (2003--2010) | 1.140 |
| Middle (2010--2017) | 2.000 |
| Late (2017--2025) | 2.588 |

Partition instability increases monotonically over time.  This is
expected: as the corpus grows, more papers enter the system each quarter,
creating more opportunity for communities to split, merge, or reorganize.
The 2.3x increase from early to late periods indicates that instability
is a growing problem, not a stable property of the clustering method.

### 3.3 Paper Identity Alignment (PIA)

PIA measures the fraction of papers in a lineage that change their
community assignment between consecutive quarters.  This is a
per-lineage metric (unlike VI which is partition-level).

| Metric | Value |
|--------|-------|
| Rows with PIA data | 13,352 |
| Mean PIA rate | 0.095 (9.5%) |
| Median PIA rate | 0.031 (3.1%) |
| Std deviation | 0.158 |
| Mean PIA rate per quarter | 0.085 |

The distribution is heavily right-skewed: most lineages have low paper
turnover (median 3.1%), but a long tail of lineages experience
substantial reassignment (up to 100% in extreme cases).

## 4. Activity Metrics

| Metric | Value |
|--------|-------|
| Mean new works per lineage-quarter | 4.3 |
| Median new works | 2.0 |
| Rows with 0 new works | 0 (0%) |

## 5. Implications for Downstream Tasks

### 5.1 Stable-Lineage Filter (P2.2)

A minimum-lifespan filter of 4 quarters would retain 1,268 lineages
(24.5%), excluding the ephemeral majority.  A filter of 8 quarters
retains 656 lineages (12.7%) but captures most onset-detectable
lineages (P10 onset lifespan = 8 quarters).

**Recommendation:** Filter at 8 quarters for onset detection experiments;
filter at 4 quarters for exploratory stability analysis.

### 5.2 ECG Ensemble Clustering (P2.4)

The increasing VI trend suggests that a single Leiden resolution may
not be appropriate across the full temporal range.  ECG ensemble
clustering could reduce instability by averaging across multiple
resolutions, but the benefit should be measured against this baseline.

**Success criterion:** ECG reduces mean VI by at least 15% (from 1.91
to 1.62 bits or less) and reduces the late-period mean from 2.59 to
under 2.20.

### 5.3 Front-Level Aggregation (P2.5)

Given that 52% of lineages are ephemeral, front-level aggregation
(grouping related lineages into research fronts) would smooth out
the instability by absorbing community splits and merges.  The PIA
rate distribution suggests that most lineages (median 3.1% turnover)
are reasonably stable, but aggregation would particularly help the
high-PIA tail.

### 5.4 Detection Pipeline Impact

The onset detector requires `w + c = 6` quarters of history minimum.
Since 75.5% of lineages have fewer than 4 quarters, the vast majority
are automatically excluded from onset detection.  This is not a flaw
in the detector but a property of the data: most lineages are noise
or transient topics that never achieve sustained growth.

## 6. Reproducibility

All statistics in this report can be reproduced from the frozen PSC
baseline artifacts listed in Section 1.  The VI and PIA values are
pre-computed in the lineage timeseries CSV.  Lifespan distributions
are computed by grouping the timeseries by `lineage_id` and counting
rows.

## 7. Change Log

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2026-03-23 | Initial baseline from frozen PSC data |
