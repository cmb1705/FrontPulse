# FrontPulse Phase 3 BOCPD And Timeliness

## Objective

Add a front-level BOCPD detector and benchmark it with a timeliness-centered
evaluation framework against the onset-trained MSD and simpler baselines.

## Why this phase exists

Prospective detection is not only about identifying whether an event was found.
It is about whether the system found it early enough to be useful. This phase
turns the detector program into a proper early-warning benchmark.

## Work Packages

### P3.1 Select BOCPD package and detector interface

- Review the available BOCPD implementation options against the actual data
  contract the repo will support.
- Define the wrapper interface before implementing model-specific logic.
- Capture package choice, dependency implications, and failure fallback.

Deliverable:

- Package decision memo and detector interface note.

### P3.2 Implement front-level BOCPD wrapper

- Build a CLI and reusable code path that consumes the front-series contract.
- Keep model inputs explicit: counts, cadence, priors, and hazard controls.
- Emit outputs in a format comparable to the onset-trained MSD outputs.

Deliverable:

- BOCPD detector script or module with reproducible outputs.

### P3.3 Calibrate priors and hazard parameters

- Estimate practical default priors for PSC front count series.
- Compare at least one alternative configuration to avoid a single arbitrary
  parameter choice.
- Record calibration logic so later domain transfers are interpretable.

Deliverable:

- Calibration report and default parameter set.

### P3.4 Implement timeliness scoring utilities

- Add scoring utilities for NAB, Expected Detection Delay, and Average Run
  Length.
- Ensure the utilities accept the same truth and prediction contract used by all
  detector families.
- Keep the code path reusable for later cross-domain work.

Deliverable:

- Reusable timeliness scoring script and schema.

### P3.5 Benchmark BOCPD against MSD and simple baselines

- Evaluate BOCPD, onset-trained MSD, and lightweight baselines on the same
  front-level dataset and evaluation contract.
- Report tradeoffs between alert speed, false alarms, and operational clarity.
- Include failure-case analysis, not only ranking tables.

Deliverable:

- Side-by-side benchmark report.

### P3.6 Prototype BOCPD plus MSD hybrid alerting

- Explore whether BOCPD probability and MSD signal can be combined into a
  stronger alerting rule.
- Keep this work exploratory until the standalone detectors are well
  understood.

Deliverable:

- Hybrid detector experiment note and artifact set.

### P3.7 Publish detector-stack decision memo

- Recommend the next deployment-facing detector stack.
- Explicitly state whether the project proceeds with MSD only, BOCPD only, or a
  hybrid path for the next loop.

Deliverable:

- Detector-stack decision memo.

## Phase Outputs

- BOCPD package decision
- Detector wrapper
- Prior calibration defaults
- Timeliness scoring utilities
- Benchmark report
- Hybrid experiment note
- Detector-stack decision memo

## Exit Gate

Phase 3 is complete only when the project can compare at least two detector
families on front-level series using the same timeliness-aware evaluation
contract.

## Risks To Watch

- Choosing a BOCPD library before locking the front-series contract
- Treating timeliness metrics as a reporting add-on instead of a design driver
- Adding a hybrid detector before the standalone detector behavior is understood
