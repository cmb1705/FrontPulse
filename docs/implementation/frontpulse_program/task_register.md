# FrontPulse Task Register

## Purpose

This register is the source of truth for the next execution wave. Task IDs in
this file map directly to the `bd` epics and tasks created for the repo.

## Epic Map

| Phase | Epic Title | Bead ID |
| --- | --- | --- |
| Phase 0 | Foundation and baseline lock | `FP-wng` |
| Phase 1 | Onset relabeling and leakage-safe baseline | `FP-wq7` |
| Phase 2 | Front tracking and stability transition | `FP-ax4` |
| Phase 3 | BOCPD and timeliness benchmarking | `FP-92e` |

## Phase 0

| Task | Bead ID | Title | Owner Mix | Depends On | Primary Outputs | Primary Files |
| --- | --- | --- | --- | --- | --- | --- |
| P0.1 | `FP-wng.1` | Finalize public branding and repo identity | Research + Eng | None | Renamed headings, repo-facing language audit | `README.md`, top-level docs |
| P0.2 | `FP-wng.2` | Freeze PSC baseline artifacts and manifests | Research + Eng | None | Baseline manifest, reproducible command log, metric snapshot | `config/`, `docs/implementation/` |
| P0.3 | `FP-wng.3` | Standardize trusted artifact IO and model persistence | Eng | None | Artifact policy, model serialization contract | `scripts/multi_signal_detector.py`, `src/trusted_io.py` |
| P0.4 | `FP-wng.4` | Define prospective evaluation contract | Research | P0.2 | Metric definitions, detection windows, split rules | New evaluation spec doc |
| P0.5 | `FP-wng.5` | Expand smoke and regression harness for next phases | Eng | P0.2, P0.3 | Smoke commands, targeted regression tests | `tests/`, `dev/experiments/tests/` |

## Phase 1

| Task | Bead ID | Title | Owner Mix | Depends On | Primary Outputs | Primary Files |
| --- | --- | --- | --- | --- | --- | --- |
| P1.1 | `FP-wq7.1` | Write onset label specification | Research | P0.4 | Labeling ADR, parameter ranges, edge-case rules | New onset spec note |
| P1.2 | `FP-wq7.2` | Implement standalone onset detector utilities | Eng | P1.1 | Detector module with unit-tested primitives | New `scripts/onset_detector.py` |
| P1.3 | `FP-wq7.3` | Add onset mode to labeling pipeline | Eng | P1.2 | CLI and output schema for onset labels | `scripts/label_inflection_points.py` |
| P1.4 | `FP-wq7.4` | Generate PSC onset labels and QA summary | Research + Eng | P1.3 | Onset label dataset, distribution report | `data/out/...`, report doc |
| P1.5 | `FP-wq7.5` | Remove forward-looking logistic features from training bundles | Eng | P1.1 | Feature flags and leakage-safe subsets | `scripts/compute_lineage_multisignal_features.py`, `config/features/` |
| P1.6 | `FP-wq7.6` | Retrain MSD baselines on onset labels | Eng | P1.4, P1.5 | Updated model artifacts and baseline tables | `scripts/multi_signal_detector.py` |
| P1.7 | `FP-wq7.7` | Run time-forward holdout with lag analysis | Research + Eng | P1.6 | Holdout report with lag-aware metrics | evaluation outputs |
| P1.8 | `FP-wq7.8` | Publish onset vs midpoint comparison memo | Research | P1.4, P1.7 | Decision memo on label transition | New memo doc |

## Phase 2

| Task | Bead ID | Title | Owner Mix | Depends On | Primary Outputs | Primary Files |
| --- | --- | --- | --- | --- | --- | --- |
| P2.1 | `FP-ax4.1` | Quantify lineage lifespan and instability baseline | Research + Eng | P0.2 | Baseline stability report | `scripts/communities.py`, `src/alignment.py` |
| P2.2 | `FP-ax4.2` | Implement stable-lineage filter | Eng | P2.1 | Filtered lineage set and CLI | New `scripts/filter_stable_lineages.py` |
| P2.3 | `FP-ax4.3` | Build community stability reporting utilities | Eng | P2.1 | VI/NMI/lifespan report pipeline | New `scripts/community_stability_report.py` |
| P2.4 | `FP-ax4.4` | Prototype ECG ensemble clustering behind a flag | Eng | P2.1 | Optional ECG path and benchmark note | `src/community.py`, `scripts/communities.py` |
| P2.5 | `FP-ax4.5` | Formalize front-level onset series contract | Research + Eng | P1.8, P2.1 | Schema and output contract for front series | `scripts/aggregate_lineages_to_fronts.py` |
| P2.6 | `FP-ax4.6` | Add front-level feature computation path | Eng | P2.5 | Front-level feature tables | feature scripts and configs |
| P2.7 | `FP-ax4.7` | Benchmark lineage-level vs front-level detection | Research + Eng | P2.2, P2.6 | Comparative evaluation report | evaluation outputs |
| P2.8 | `FP-ax4.8` | Decide deployment unit of analysis | Research | P2.7 | Signed-off decision memo | New memo doc |

## Phase 3

| Task | Bead ID | Title | Owner Mix | Depends On | Primary Outputs | Primary Files |
| --- | --- | --- | --- | --- | --- | --- |
| P3.1 | `FP-92e.1` | Select BOCPD package and detector interface | Eng | P0.4, P2.5 | Package decision memo and wrapper contract | requirements and new interface note |
| P3.2 | `FP-92e.2` | Implement front-level BOCPD wrapper | Eng | P3.1 | Detector CLI and reusable module | New `scripts/bocpd_detector.py` |
| P3.3 | `FP-92e.3` | Calibrate priors and hazard parameters | Research + Eng | P3.2 | Calibration notebook or report and defaults | calibration scripts/config |
| P3.4 | `FP-92e.4` | Implement timeliness scoring utilities | Eng | P0.4 | NAB, EDD, and ARL scoring tools | New `scripts/nab_scorer.py` |
| P3.5 | `FP-92e.5` | Benchmark BOCPD against MSD and simple baselines | Research + Eng | P3.3, P3.4, P2.8 | Side-by-side benchmark report | evaluation outputs |
| P3.6 | `FP-92e.6` | Prototype BOCPD plus MSD hybrid alerting | Eng | P3.5 | Hybrid scoring experiment | new hybrid script or extension |
| P3.7 | `FP-92e.7` | Publish detector-stack decision memo | Research + Eng | P3.5, P3.6 | Deployment recommendation memo | new memo doc |

## Downstream Backlog

The following workstreams stay documented but intentionally sit outside the
immediate `bd` execution wave:

- Phase 4: cross-domain and LLM-assisted validation
- Phase 5: feature pruning and model simplification
- Phase 6: BERTopic comparison track
- Phase 7: dashboard and quarterly monitoring layer
