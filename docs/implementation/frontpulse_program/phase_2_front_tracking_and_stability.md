# FrontPulse Phase 2 Front Tracking And Stability

## Objective

Move the detector program onto a stable unit of analysis by measuring the
current instability problem, improving community robustness where justified, and
formalizing front-level series as the deployment target.

## Why this phase exists

The current lineage unit is too volatile to support a practical prospective
detector. The right response is not to jump directly to a new clustering method.
The first response is to measure the instability, then decide whether
front-level aggregation and optional clustering improvements are enough.

## Work Packages

### P2.1 Quantify lineage lifespan and instability baseline

- Measure lineage lifespan distributions and quarter-over-quarter partition
  instability from a frozen PSC baseline.
- Use existing VI utilities where possible and extend reporting only where
  current outputs are insufficient.
- Produce a baseline report that later ECG experiments can compare against.

Deliverable:

- Stability baseline report with lifespan, VI, and related diagnostics.

### P2.2 Implement stable-lineage filter

- Add a reproducible filter for minimum lineage history length.
- Support this as a proof-of-concept path, not the long-term deployment target.
- Make the resulting cohort easy to compare with the unrestricted lineage set.

Deliverable:

- CLI or script that emits stable-lineage subsets plus summary counts.

### P2.3 Build community stability reporting utilities

- Package the stability diagnostics into a reusable reporting path.
- Include quarter-over-quarter and year-over-year views where helpful.
- Keep the report lean enough to rerun during clustering experiments.

Deliverable:

- A stability reporting script and stable output schema.

### P2.4 Prototype ECG ensemble clustering behind a flag

- Introduce ECG as an optional clustering path only after the baseline report is
  available.
- Compare output stability and practical runtime cost against the current path.
- Avoid making ECG the default until it demonstrates material benefit.

Deliverable:

- Flagged ECG path plus comparison note.

### P2.5 Formalize front-level onset series contract

- Define the canonical front-level time-series schema for counts, deltas, and
  any onset-aligned annotations.
- Clarify naming, output paths, and which mapping filters are required before
  aggregation.
- Make the contract stable enough for both MSD and BOCPD consumers.

Deliverable:

- Front series contract and updated aggregation outputs.

### P2.6 Add front-level feature computation path

- Extend feature computation so fronts, not only lineages, can feed later
  detectors.
- Record how front-level features differ from lineage-level features and which
  ones remain meaningful after aggregation.
- Preserve a simple baseline path to avoid overgrowing the feature surface.

Deliverable:

- Front-level feature artifacts and configuration path.

### P2.7 Benchmark lineage-level vs front-level detection

- Compare the stable-lineage proof path and the front-level deployment path on
  the same evaluation contract.
- Report not only precision and recall but also timeliness and operational
  interpretability.

Deliverable:

- Comparative evaluation report.

### P2.8 Decide deployment unit of analysis

- Use the empirical evidence from P2.1 through P2.7 to choose the deployment
  target.
- Record whether stable lineages remain a research-only diagnostic or continue
  as a supported detector path.

Deliverable:

- Decision memo on deployment unit of analysis.

## Phase Outputs

- Stability baseline report
- Stable-lineage filter
- Community stability reporting utility
- Optional ECG prototype
- Front-level series contract
- Front-level feature path
- Deployment-unit decision memo

## Exit Gate

Phase 2 is complete only when the front-level series contract is stable and the
team has evidence for why front-level detection is the preferred deployment
target, rather than only an intuition that raw lineages are unstable.

## Risks To Watch

- Replacing clustering logic before measuring the baseline problem
- Expanding the front-level schema without a stable detector consumer contract
- Keeping both lineage and front pipelines indefinitely without a clear decision
