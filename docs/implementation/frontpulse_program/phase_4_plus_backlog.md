# FrontPulse Phase 4 Plus Backlog

## Purpose

This document keeps the downstream work visible without diluting the immediate
execution wave. These phases are real, but they do not belong on the first
critical path.

## Phase 4 Cross-Domain And LLM-Assisted Validation

Entry criteria:

- Phase 1 onset labels are stable.
- Phase 2 front-level outputs are stable.
- Phase 3 timeliness benchmark exists.

Key goals:

- validate the detector logic on at least one non-PSC field,
- replace manual-only validation with repeatable assisted workflows,
- and prove the pipeline is not PSC-specific.

Preferred order:

1. Define domain selection rubric.
2. Stand up CRISPR as the first transfer field.
3. Build automated milestone extraction and review support.
4. Compare PSC and CRISPR detector behavior on the same evaluation frame.

## Phase 5 Feature Pruning And Model Simplification

Entry criteria:

- Onset labels and front-level feature paths are stable.

Key goals:

- reduce the feature surface,
- identify the smallest defensible prospective feature bundle,
- and turn ablation results into a publication-quality simplification story.

## Phase 6 BERTopic Comparison Track

Entry criteria:

- Baseline front-level detector stack is already running.

Key goals:

- test whether dynamic topic modeling yields a more stable unit than Leiden
  lineages,
- compare temporal stability and topic coherence,
- and determine whether BERTopic belongs in the mainline roadmap or stays a
  research track.

## Phase 7 Dashboard And Monitoring

Entry criteria:

- Detector stack choice is made,
- front-level outputs are stable,
- and alert metrics have a settled schema.

Key goals:

- provide a lightweight operator-facing surface,
- expose detector outputs without requiring notebook work,
- and support quarterly refresh review.

## Backlog Rule

No downstream phase should pull work onto the active critical path unless:

- it removes a confirmed blocker in Phases 0 through 3, or
- it materially improves the validity of the detector benchmark now in flight.
