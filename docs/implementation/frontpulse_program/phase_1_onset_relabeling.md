# FrontPulse Phase 1 Onset Relabeling

## Objective

Replace midpoint-oriented retrospective labels with a prospective onset labeling
framework that can support meaningful early-warning evaluation.

## Core Research Question

What is the first quarter in which a front or lineage enters a sustained growth
regime that should count as an actionable onset signal?

## Required Design Constraints

- The label cannot depend on full-history logistic fits when the goal is
  prospective detection.
- The implementation must separate label specification from model training.
- The team needs both machine-readable outputs and an interpretable QA summary.

## Work Packages

### P1.1 Write onset label specification

- Define onset in plain language and operational rules.
- Specify allowed parameter ranges for rolling windows, thresholds, and
  confirmation periods.
- Clarify edge cases such as sparse starts, temporary surges, and fronts with
  multiple plausible acceleration episodes.

Deliverable:

- An onset labeling specification note that later code and evaluation cite.

### P1.2 Implement standalone onset detector utilities

- Build reusable onset-detection primitives in a dedicated module instead of
  burying logic inside one reporting script.
- Keep the detector functions testable with small synthetic count series.
- Support multiple strategies if needed, but make one default strategy explicit.

Deliverable:

- `scripts/onset_detector.py` or equivalent shared module with tested core
  functions.

### P1.3 Add onset mode to labeling pipeline

- Extend `scripts/label_inflection_points.py` with an onset mode.
- Preserve midpoint or retrospective modes only when clearly flagged.
- Emit a schema that downstream training scripts can consume without special
  casing.

Deliverable:

- CLI support for onset label generation and stable output paths.

### P1.4 Generate PSC onset labels and QA summary

- Run the labeling workflow on the PSC development corpus.
- Summarize label counts, timing shifts, sparse cases, and obvious anomalies.
- Capture how onset labels differ from the existing midpoint-oriented labels.

Deliverable:

- PSC onset label artifact and QA summary.

### P1.5 Remove forward-looking logistic features from training bundles

- Audit the current feature pipeline for forward-looking logistic signals.
- Add an explicit leakage-safe feature mode or exclusion flag.
- Update configs so baseline experiments cannot silently mix retrospective-only
  features into prospective runs.

Deliverable:

- Leakage-safe feature bundle definition and updated config references.

### P1.6 Retrain MSD baselines on onset labels

- Retrain the existing detector family using onset labels and leakage-safe
  features.
- Keep the baseline family small enough to interpret.
- Preserve comparable output tables for later benchmarking.

Deliverable:

- Updated onset-trained baseline results.

### P1.7 Run time-forward holdout with lag analysis

- Evaluate the onset-trained models using the prospective evaluation contract.
- Report precision, recall, calibration, and lag-aware metrics together.
- Quantify the difference between CV performance and holdout behavior.

Deliverable:

- Holdout report with lag-aware metrics and error analysis.

### P1.8 Publish onset vs midpoint comparison memo

- Summarize what changed scientifically and operationally.
- Decide whether midpoint labels remain only as retrospective reference labels.
- Record open questions before the team moves to front-level work.

Deliverable:

- Decision memo on label transition.

## Phase Outputs

- Onset specification
- Onset detector module
- PSC onset labels
- Leakage-safe feature bundle
- Onset-based baseline report
- Comparison memo

## Exit Gate

Phase 1 is complete only when the team can train and evaluate at least one
detector on onset labels without relying on future-derived features, and the
difference between onset and midpoint labels is documented.

## Risks To Watch

- Turning onset into a vague concept rather than an operational rule
- Shipping an onset detector without a QA summary for sparse or noisy series
- Comparing new results to old baselines without locking the feature bundle
