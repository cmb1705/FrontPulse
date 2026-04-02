# FrontPulse Phase 0 Foundation

## Objective

Establish a stable, branded, leakage-aware baseline so later detector work is
measured against a reproducible starting point rather than a moving target.

## Why this phase exists

The next research loop only works if the team can answer three simple
questions:

1. What exact baseline are later comparisons using?
2. Which outputs are trusted, and how are they serialized?
3. Which metrics define success for a prospective detector?

Without those answers, later onset and BOCPD work will create artifacts that
cannot be compared cleanly.

## Work Packages

### P0.1 Finalize public branding and repo identity

- Rename the public-facing `README.md` heading from `2YP` to `FrontPulse`.
- Audit the top-level docs for stale `2YP` branding where it affects public
  interpretation.
- Leave historical terminology intact where it refers to legacy artifact names
  or previously published experiment directories.

Acceptance:

- Public-facing repo language consistently uses `FrontPulse`.
- Legacy names are preserved only where needed for artifact compatibility.

### P0.2 Freeze PSC baseline artifacts and manifests

- Record the exact configuration bundle used for the current PSC baseline.
- Produce a reproducible command manifest for the current inflection workflow.
- Capture baseline outputs, metrics, and known caveats in a short reference note.

Acceptance:

- A new team member can recreate the baseline without reconstructing prior chat
  history.
- The frozen baseline identifies which outputs are retrospective-only.

### P0.3 Standardize trusted artifact IO and model persistence

- Review where model and analysis artifacts are written and loaded.
- Replace or constrain ambiguous persistence patterns in
  `scripts/multi_signal_detector.py`.
- Document which artifacts remain trusted-only and which must be portable or
  shareable.

Acceptance:

- Model persistence has a documented standard.
- Artifact loading paths distinguish between trusted local bundles and portable
  exports.

### P0.4 Define prospective evaluation contract

- Write the authoritative definition for early-warning evaluation.
- Specify detection windows, lag handling, false-alarm accounting, and the
  baseline split policy.
- Define how NAB, Expected Detection Delay, and Average Run Length will be
  reported together.

Acceptance:

- All later detector comparisons can point to one evaluation contract.
- The contract explicitly distinguishes CV, retrospective replay, and
  time-forward holdout.

### P0.5 Expand smoke and regression harness for next phases

- Identify the minimal commands that must pass before and after label or
  detector changes.
- Add targeted tests for onset labeling primitives, front aggregation schema,
  and detector CLI wiring as later phases land.
- Keep the suite narrow enough to run during active development.

Acceptance:

- The team has a documented smoke set for rapid iteration.
- The baseline suite can catch schema drift before large experiments run.

## Phase Outputs

- Branded public repo entry points
- Baseline freeze memo and command manifest
- Artifact serialization policy
- Prospective evaluation contract
- Smoke and regression harness checklist

## Exit Gate

Phase 0 is complete only when:

- the public repo branding is correct,
- the baseline is frozen and documented,
- artifact handling is explicit,
- and the evaluation contract exists in a written form that later phases can
  cite.

## Risks To Watch

- Over-documenting the baseline without freezing exact commands
- Treating artifact persistence as a security issue only, instead of a
  reproducibility issue
- Letting Phase 0 expand into a general cleanup pass instead of a targeted
  baseline lock
