# FrontPulse Program Review And Execution Map

## Purpose

This document reviews the initial proposal in `frontpulse_next.md` and turns it
into an execution-ready program for a mixed team of researchers and developers.
The goal is not to restate the proposal verbatim. The goal is to tighten the
critical path, remove stale assumptions, and define a document set that can
drive implementation and `bd` tasking.

## Review Of `frontpulse_next.md`

### What is strong

- The core diagnosis is correct: midpoint-style retrospective detection is not a
  coherent basis for prospective early warning.
- The plan correctly shifts the unit of analysis toward fronts rather than raw
  quarter-by-quarter lineages.
- The proposed BOCPD track is well matched to front-level count series, where
  time series length and continuity are adequate.
- The plan identifies timeliness as a first-class evaluation concern instead of
  relying only on point classification metrics.

### What needs correction based on the current repo state

1. Phase 0 in the draft overstates the remaining security cleanup.
   `config/datasources.yaml` already has `mailto: null`, hardcoded `2YP` paths
   have been removed from the primary scripts, and trusted pickle boundaries now
   exist in `src/trusted_io.py`.
2. The remaining serialization task is narrower than the draft suggests. The
   immediate problem is artifact standardization and safe model persistence,
   especially around `scripts/multi_signal_detector.py`, not a repo-wide
   unmitigated deserialization problem.
3. Branding is still incomplete. The public repo root `README.md` still opens
   with `# 2YP`, so there is a real Phase 0 branding task left.
4. Phase 2 should begin with instrumentation, not algorithm replacement.
   Before introducing ECG or additional clustering machinery, the team should
   quantify the current lineage lifespan and quarter-over-quarter partition
   instability on a frozen baseline run.
5. Phase 3 should not start with model integration. It should first define the
   time-series contract, detector interface, and timeliness metrics so BOCPD,
   MSD, and simple baselines can be compared on the same evaluation frame.

### What to preserve from the original proposal

- Front-level tracking is the deployment target.
- Onset detection replaces midpoint detection.
- Timeliness metrics are mandatory.
- Cross-domain validation remains downstream from the critical path.
- BERTopic and dashboard work remain parallel or later-phase work, not the
  first proof point.

## Program Design Principles

1. Keep the first implementation loop narrow: prove prospective onset detection
   on PSC before opening cross-domain or dashboard work.
2. Separate research decisions from plumbing decisions. Label definitions,
   evaluation windows, and deployment unit of analysis should produce explicit
   decision memos.
3. Require phase exit gates. Each phase should produce artifacts that can be
   inspected without rerunning the whole pipeline.
4. Treat front-level aggregation as a product surface, not only an analysis
   helper. It needs schema, tests, and reproducible output locations.
5. Avoid hidden leakage. Any feature, label, or evaluation rule that uses
   future information must be explicitly marked retrospective-only.
6. Keep the planning artifacts synchronized with `bd` so a compacted agent can
   resume from the documents alone.

## Critical Path

The working critical path is:

1. Phase 0 foundation and baseline freeze
2. Phase 1 onset relabeling and no-leakage model baseline
3. Phase 2 front-level and stability transition
4. Phase 3 BOCPD and timeliness benchmarking

Everything else remains downstream or parallel until that path is proven.

## Document Map

| Document | Purpose |
| --- | --- |
| `task_register.md` | Canonical task inventory and dependency map |
| `phase_0_foundation.md` | Branding, baseline freeze, artifact standards, and test harness |
| `phase_1_onset_relabeling.md` | Onset label specification, implementation, and evaluation |
| `phase_2_front_tracking_and_stability.md` | Stability instrumentation, front-level schema, and deployment-unit decision |
| `phase_3_bocpd_and_timeliness.md` | BOCPD implementation and timeliness-centered benchmarking |
| `phase_4_plus_backlog.md` | Deferred workstreams for validation, pruning, BERTopic, and dashboarding |

## Operating Rhythm

- Planning unit: one phase file plus the corresponding `bd` epic.
- Delivery unit: one task with a concrete artifact and acceptance gate.
- Review unit: one decision memo or evaluation report per phase.
- Coordination unit: researchers own label logic, evaluation definitions, and
  domain framing; developers own pipeline changes, reproducibility, and test
  harnesses; both sides review decision memos.

## Expected First-Pass Outputs

By the end of Phases 0 through 3, the repo should contain:

- A branded public-facing baseline and frozen experiment manifest
- A prospective onset label set for PSC
- A front-level feature and alerting pipeline
- A BOCPD detector that consumes the same front series contract as the MSD path
- A benchmark report centered on timeliness, not just precision and recall
