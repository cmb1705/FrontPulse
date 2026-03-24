# Front-Level Onset Series Contract

Version: 1.0
Date: 2026-03-24
Status: Authoritative
Task: FP-ax4.5 (P2.5 Formalize front-level onset series contract)

This document defines the canonical front-level time-series schema that all
detector families (MSD, BOCPD, future hybrids) consume.  It stabilizes the
interface between front aggregation and detection so downstream work can
proceed without format negotiation.

## 1. Scope

The contract covers:

- The canonical CSV schema for front-level quarterly series.
- Required and optional columns, their types, and semantics.
- Output file path conventions.
- Prerequisite mapping and filtering rules.
- How onset annotations attach to front-level records.

The contract does **not** cover:

- How lineage-to-front mappings are produced (that is the mapping pipeline).
- How features are computed from the series (that is FP-ax4.6).
- How detectors consume the series internally (that is per-detector design).

## 2. Primary Key

Every record is uniquely identified by `(front_id, quarter)`.

| Column | Type | Description |
|--------|------|-------------|
| `front_id` | str | Canonical front identifier from `config/front_aliases.yaml` or `config/front_aliases_crispr.yaml`. Must match a top-level key in the active alias file. |
| `quarter` | str | Standard quarter string in `YYYYQN` format (e.g., `2009Q3`). Sorted chronologically. |

## 3. Core Series Columns (Required)

These columns are always present and non-null for every `(front_id, quarter)`
record.

| Column | Type | Description |
|--------|------|-------------|
| `new_works` | int | Sum of `new_works` across all lineages mapped to this front in this quarter. Zero if no lineages are active. |
| `cumulative_works` | int | Running cumulative sum of `new_works` for this front across all quarters up to and including this one. |
| `n_lineages` | int | Number of distinct lineages contributing to this front in this quarter (lineages with `new_works > 0`). |
| `n_lineages_cumulative` | int | Number of distinct lineages that have ever contributed to this front up to and including this quarter. |

## 4. Growth and Acceleration Columns (Required)

Computed from the core series using trailing-only operations.  These parallel
the lineage-level growth features but operate on front-level aggregates.

| Column | Type | Description |
|--------|------|-------------|
| `growth_rate` | float | Quarter-over-quarter growth rate of `new_works`: `(new_works[t] - new_works[t-1]) / max(new_works[t-1], 1)`. Null for the first quarter of a front. |
| `growth_acceleration` | float | Change in growth rate: `growth_rate[t] - growth_rate[t-1]`. Null for the first two quarters. |
| `new_works_roll_mean_4q` | float | Trailing 4-quarter rolling mean of `new_works`. Partial windows allowed for early quarters. |
| `new_works_roll_std_4q` | float | Trailing 4-quarter rolling standard deviation of `new_works`. Partial windows allowed. |
| `new_works_yoy_delta` | int | Year-over-year delta: `new_works[t] - new_works[t-4]`. Null for the first 4 quarters. |

## 5. Onset Annotation Columns (Required When Labels Exist)

These columns attach onset information from the label pipeline to the
front-level series.  They are present only when onset labels have been
computed for the underlying lineages.

### 5.1 Front-Level Onset Derivation

A front's onset is derived from its constituent lineages' onset labels:

1. For each front, collect all lineages mapped to it that have
   `onset_detected == 1` in `onset_labels.csv`.
2. The front's onset quarter is the **earliest** onset quarter among its
   mapped lineages.  This represents the first detectable growth signal
   in any sub-community of the front.
3. If no mapped lineage has a detected onset, the front has no onset
   (`onset_quarter` is null, `onset_detected` is 0).

| Column | Type | Description |
|--------|------|-------------|
| `onset_detected` | int (0/1) | Whether any lineage mapped to this front has a detected onset. |
| `onset_quarter` | str or null | The earliest onset quarter among mapped lineages. Null if `onset_detected == 0`. |
| `quarters_since_onset` | int or null | Number of quarters elapsed since `onset_quarter` for this record. 0 at the onset quarter itself. Null before onset or if no onset. |
| `is_onset_quarter` | int (0/1) | 1 only at the front's onset quarter; 0 everywhere else. This is the primary label for MSD-style binary classification at front level. |

### 5.2 Lifecycle Stage Column

| Column | Type | Description |
|--------|------|-------------|
| `lifecycle_stage` | str | One of: `pre_onset`, `growth`, `mature`, `decline`, `no_onset`. Derived from onset and maturation timing. See Section 5.3 for stage definitions. |

### 5.3 Lifecycle Stage Definitions

| Stage | Condition |
|-------|-----------|
| `no_onset` | Front has `onset_detected == 0`. |
| `pre_onset` | Quarter is before `onset_quarter`. |
| `growth` | Quarter is at or after `onset_quarter` and before maturation (if maturation is defined), or within `growth_window` quarters after onset (default: 12 quarters / 3 years). |
| `mature` | Quarter is after the growth window but cumulative works are still increasing. |
| `decline` | Quarter where `new_works_roll_mean_4q` has declined for 4+ consecutive quarters from its peak. |

The `growth_window` parameter (default 12 quarters) is configurable and
should be documented when the aggregation script is updated.

## 6. Optional Extension Columns

These columns may be added by downstream scripts but are not required by the
base contract.  Detectors must not assume their presence.

| Column | Type | Description |
|--------|------|-------------|
| `semantic_velocity_mean` | float | Mean semantic velocity across constituent lineages. |
| `conv_composite_score_mean` | float | Mean convergence composite score across constituent lineages. |
| `n_onset_lineages` | int | Number of mapped lineages with detected onsets. |
| `front_label` | str | Human-readable front name from `description` field in alias config. |

## 7. Output File Paths

All front-level series outputs live under `data/out/04_front_aggregation/`.

| File | Format | Description |
|------|--------|-------------|
| `front_onset_series.csv` | Long CSV | Canonical front-level series conforming to this contract. Primary output. |
| `front_onset_series_wide.csv` | Wide CSV | Pivoted view with quarters as rows and fronts as columns (new_works only). For visualization. |
| `front_timeseries_delta.csv` | Wide CSV | Legacy output from existing aggregation script. Retained for backward compatibility. |
| `front_timeseries_cumulative.csv` | Wide CSV | Legacy cumulative output. Retained for backward compatibility. |

The **canonical file** for all detector consumers is `front_onset_series.csv`.
Legacy files are produced by the existing `scripts/aggregate_lineages_to_fronts.py`
and may be retired once all consumers migrate.

## 8. Prerequisite: Lineage-to-Front Mapping

### 8.1 Mapping Source

The aggregation script requires a lineage-to-front mapping CSV with at minimum:

| Column | Type | Description |
|--------|------|-------------|
| `lineage_id` | int | Lineage ID from the lineage registry. |
| `primary_front` | str | Front ID matching a key in the active alias config. |
| `confidence` | str | Mapping confidence: `high`, `medium`, or `low`. |

### 8.2 Mapping Paths

| Path | Description |
|------|-------------|
| `data/out/03_milestone_mapping/lineage_front_mappings_selected.csv` | Curated mapping (preferred). |
| `data/out/03_milestone_mapping/lineage_front_mappings.csv` | Full mapping (fallback). |
| `data/out/experiments/stage0_tight_mapping/milestone_lineage_mapping_tight.csv` | Tight milestone mapping (PSC baseline). |

### 8.3 Filtering Rules

Before aggregation, mappings must be filtered to include only lineages with
`confidence` in `{high, medium}`.  Low-confidence mappings introduce noise
from weakly associated lineages.

The tight milestone mapping (`stage0_tight_mapping`) uses a different schema
with `similarity` scores instead of confidence labels.  When using this
mapping, filter by `similarity >= 0.65` (equivalent to medium+ confidence).

### 8.4 Unmapped Lineages

Lineages that do not appear in any mapping are excluded from front-level
aggregation.  This is by design: the front-level series covers only the
curated research fronts defined in the alias configuration, not the full
lineage population.

## 9. Temporal Coverage

The series spans all quarters present in `lineage_timeseries.csv` for which
at least one mapped lineage has data.  Quarters where a front has no active
lineages are included with `new_works = 0` and `n_lineages = 0` to maintain
a continuous time axis.

For PSC, this is 2003Q1 through 2025Q3 (91 quarters).

## 10. Consumer Contracts

### 10.1 MSD (CatBoost Classifier)

The MSD consumes front-level features computed from this series (FP-ax4.6).
It requires:

- `(front_id, quarter)` as the join key.
- `is_onset_quarter` as the binary target variable.
- All Section 4 growth columns as base features.
- Section 5 lifecycle stage for train/test split validation (lifecycle
  features from training labels only in time-forward holdout).

### 10.2 BOCPD (Bayesian Online Changepoint Detection)

BOCPD consumes the raw count series directly.  It requires:

- `new_works` as the observed count per quarter, sorted chronologically
  per front.
- `onset_quarter` as ground truth for timeliness evaluation.
- Continuous quarterly cadence (no gaps) so the hazard function can
  assume uniform timesteps.

### 10.3 Timeliness Scoring

The timeliness scoring utilities (`src/timeliness_scoring.py`) require:

- A truth series with `is_onset_quarter == 1` at the true onset.
- A prediction series with detector-specific alarm indicators.
- Both series keyed by `(front_id, quarter)`.

## 11. Feature Aggregation Semantics

When lineage-level features are aggregated to front level, the reduction
operation depends on the feature type.

### 11.1 Sum Aggregation (Additive Quantities)

These features represent absolute counts that are meaningful when summed
across lineages within a front.

| Lineage Feature | Front Column | Notes |
|-----------------|-------------|-------|
| `new_works` | `new_works` | Total new papers across all lineages in the front. |
| `novel_terms` | `novel_terms` | Total novel terms introduced. |
| `cross_domain_refs` | `cross_domain_refs` | Total cross-domain references. |
| `within_lineage_refs` | `within_lineage_refs` | Total within-lineage references. |
| `n_new_papers` | `n_new_papers` | Total new papers (may differ from new_works due to counting). |

### 11.2 Mean Aggregation (Rates and Scores)

These features represent rates or intensities where averaging across
lineages is semantically appropriate.

| Lineage Feature | Front Column | Notes |
|-----------------|-------------|-------|
| `novelty_rate` | `novelty_rate` | Mean novelty rate across lineages. |
| `novelty_momentum` | `novelty_momentum` | Mean novelty momentum. |
| `cross_domain_share` | `cross_domain_share` | Mean cross-domain citation share. |
| `citation_balance` | `citation_balance` | Mean citation balance. |
| `semantic_velocity` | `semantic_velocity` | Mean semantic drift velocity. |
| `velocity_acceleration` | `velocity_acceleration` | Mean velocity acceleration. |
| `dormancy_length` | `dormancy_length` | Mean dormancy length (informational). |
| `awakening_intensity` | `awakening_intensity` | Mean awakening intensity. |

Context features (`*_z`, `*_roll_2q`, `*_roll_4q`) and convergence features
(`conv_*`) are also averaged.

### 11.3 Features That Do Not Survive Aggregation

| Feature | Reason |
|---------|--------|
| `logistic_*` (4 features) | S-curve fit is per-lineage; no meaningful front-level analog. Also leakage-unsafe. |
| `cd_index`, `cd_min`, `cd_max` | Disruption index is per-paper; aggregation would dilute signal. Also leakage-unsafe. |
| `is_awakening` | Binary per-lineage flag; front-level awakening needs different definition. |
| `dormancy_length` (as count) | Included as mean but loses per-lineage dormancy semantics. |

### 11.4 Recomputed at Front Level

Growth and acceleration columns are recomputed on the front-level aggregated
`new_works` series rather than averaged from lineage-level growth rates.
This avoids the statistical pitfall of averaging ratios with different
denominators.

## 12. Versioning

The contract version is tracked in this document's header.  Schema changes
that add required columns or change column semantics require a version bump
and explicit migration notes.

Adding optional extension columns (Section 6) does not require a version
change.

## 12. Example Record

```csv
front_id,quarter,new_works,cumulative_works,n_lineages,n_lineages_cumulative,growth_rate,growth_acceleration,new_works_roll_mean_4q,new_works_roll_std_4q,new_works_yoy_delta,onset_detected,onset_quarter,quarters_since_onset,is_onset_quarter,lifecycle_stage
core_psc,2009Q3,5,5,2,2,,,5.0,0.0,,1,2009Q3,0,1,growth
core_psc,2009Q4,8,13,3,4,0.6,,6.5,2.12,,1,2009Q3,1,0,growth
core_psc,2010Q1,12,25,4,5,0.5,-0.1,8.33,3.51,,1,2009Q3,2,0,growth
core_psc,2010Q2,20,45,5,7,0.667,0.167,11.25,6.29,,1,2009Q3,3,0,growth
core_psc,2010Q3,35,80,6,8,0.75,0.083,18.75,11.9,30,1,2009Q3,4,0,growth
```

## 13. Change Log

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2026-03-24 | Initial contract from Phase 2 work. |
