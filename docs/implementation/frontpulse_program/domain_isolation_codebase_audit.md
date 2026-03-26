# Codebase Audit: Files Affected by Domain Data Isolation

**Task:** FP-i6q
**Date:** 2026-03-25
**Methodology:** grep for `current_ingest`, `current_graphs`, `data/out`, `outdir_suffix`
across all `.py` files, config YAML/JSON, and test files.

---

## Summary

| Category | Files | Must Change | Can Defer |
|----------|-------|-------------|-----------|
| A. Pipeline core | 3 | 3 | 0 |
| B. Source library (src/) | 7 | 2 | 5 |
| C. Feature computation scripts | 11 | 11 | 0 |
| D. Analysis/experiment scripts | 9 | 9 | 0 |
| E. Horizon scanner scripts | 4 | 4 | 0 |
| F. One-time / utility scripts | 18 | 0 | 18 |
| G. Dev / archived scripts | 8 | 0 | 8 |
| H. Config files | 5 | 2 | 3 |
| I. Test files | 1 | 1 | 0 |
| **Total** | **66** | **32** | **34** |

"Must Change" = references domain-specific data paths in argparse defaults or
code logic; needs `--domain` support for correct multi-domain operation.

"Can Defer" = one-off analysis scripts, dev experiments, or documentation-only
references; backward-compat symlinks cover these during transition.

---

## Category A: Pipeline Core (3 files -- all must change)

### A.1 run.py (entry point)

| Line(s) | Path Reference | Type |
|---------|----------------|------|
| 402 | `--ingest-dir` default `data/current_ingest` | argparse default |
| 403 | `--graphs-dir` default `data/current_graphs` | argparse default |
| 401 | `--outdir` required (no default) | argparse required |
| 461 | `--coupling-cache-dir` default `data/out/cache_coupling` | argparse default |
| 1457-1462 | `resolve_domain_args()` -- config only, not data paths | code logic |

**Change needed:** Replace hardcoded defaults with `None`; derive from
`resolve_pipeline_paths()` when `--domain` is set. Make `--outdir` optional
when `--domain` is provided.

### A.2 scripts/communities.py

| Line(s) | Path Reference | Type |
|---------|----------------|------|
| 1417 | `--graphs-dir` default `data/current_graphs` | argparse default |
| ~1418 | `--out-dir` default `data/out` | argparse default |
| ~1419 | `--cache-dir` default `data/out/cache_cum` | argparse default |

**Change needed:** Add `--domain`; derive graphs-dir, out-dir, cache-dir.

### A.3 scripts/run_build_pipeline.py

| Line(s) | Path Reference | Type |
|---------|----------------|------|
| 362 | `--raw-dir` default `data/current_ingest/raw` | argparse default |
| 368 | `--graphs-dir` default `data/current_graphs` | argparse default |
| 480-481 | `--slices-dir` default `data/current_ingest/slices` | argparse default |

**Change needed:** Add `--domain`; derive all three path args.

---

## Category B: Source Library -- src/ (7 files, 2 must change)

### B.1 src/domain_registry.py (MUST CHANGE)

| Line(s) | Reference | Type |
|---------|-----------|------|
| 35 | `outdir_suffix: str = ""` | dataclass field |
| 76 | `outdir_suffix="_crispr"` | registry entry |

**Change needed:** Add `DomainDataPaths` dataclass, `resolve_data_paths()`,
`resolve_pipeline_paths()`, `add_domain_args()`, `resolve_script_paths()`.
Remove `outdir_suffix`.

### B.2 src/settings.py (MUST CHANGE)

**Change needed:** Per-domain settings path (`data/{domain}/settings.json`)
instead of project-root `.2yp_settings.json`.

### B.3 src/pipeline.py (defer -- documentation only)

| Line(s) | Reference |
|---------|-----------|
| 22-24 | Docstring examples: `data/out`, `data/current_ingest`, `data/current_graphs` |
| 134-136 | Docstring examples (same) |

### B.4 src/audit_logger.py (defer -- documentation only)

| Line(s) | Reference |
|---------|-----------|
| 26 | Docstring example: `data/out/models/msd_model.pkl` |
| 538 | Docstring example: `data/out/models/msd_model_20251106.pkl` |

### B.5 src/model_registry.py (defer -- documentation only)

| Line(s) | Reference |
|---------|-----------|
| 14 | Docstring: `data/out/models/msd/` |
| 103 | Docstring: `data/out/models/msd` |

### B.6 src/validate_slices.py (defer -- docstring + argparse help)

| Line(s) | Reference |
|---------|-----------|
| 13-14 | Docstring example paths |
| 288 | argparse help string |

### B.7 src/metrics/common.py (defer -- docstring only)

| Line(s) | Reference |
|---------|-----------|
| 117 | Docstring: `data/out/metrics` |

### B.8 src/logging_config.py (defer -- docstring only)

| Line(s) | Reference |
|---------|-----------|
| 52 | Docstring: `data/out/logs/pipeline.log` |

---

## Category C: Feature Computation Scripts (11 files -- all must change)

These are the core feature computation pipeline. All need `--domain` support.

### C.1 scripts/compute_lineage_multisignal_features.py

| Line(s) | Path Reference |
|---------|----------------|
| 1018 | `--raw-dir` default `data/current_ingest/raw` |
| ~1016 | `--registry` default `data/out/02_lineage_tracking/lineage_registry.json` |
| ~1017 | `--timeseries` default `data/out/02_lineage_tracking/lineage_timeseries.csv` |
| ~1019 | `--partitions-dir` default `data/out/cache_cum/partitions_cum` |
| ~1020 | `--reference-cache` default `data/out/cache_lineage/reference_data.pkl` |
| ~1021 | `--metrics-dir` default `data/out/metrics` |
| ~1022 | `--field-metrics` default `data/out/04_front_aggregation/field_metrics.parquet` |
| ~1030 | `--out` default `data/out/02_lineage_tracking/lineage_multisignal_features.csv` |

**6+ path args to derive from domain.**

### C.2 scripts/compute_convergence_features.py

| Line(s) | Path Reference |
|---------|----------------|
| ~296 | `--registry` default `data/out/02_lineage_tracking/lineage_registry.json` |
| ~297 | `--timeseries` default `data/out/02_lineage_tracking/lineage_timeseries.csv` |
| ~298 | `--quarterly-embeddings` default `data/out/experiments/stage1_.../quarterly_embeddings.npz` |
| ~299 | `--partitions-dir` default `data/out/cache_cum/partitions_cum` |
| 300 | `--slices-dir` default `data/current_ingest/slices` |
| ~305 | `--out` default `data/out/02_lineage_tracking/convergence_features.csv` |

### C.3 scripts/compute_lineage_ctfidf.py

| Line(s) | Path Reference |
|---------|----------------|
| 1008 | `--raw` default `data/current_ingest/raw` |
| ~1006 | `--registry` default `data/out/02_lineage_tracking/lineage_registry.json` |
| ~1007 | `--partitions` default `data/out/cache_cum/partitions_cum` |
| ~1015 | `--output-root` default `data/out` |

### C.4 scripts/compute_lineage_npmi.py

| Line(s) | Path Reference |
|---------|----------------|
| 1485 | `--raw` default `data/current_ingest/raw` |
| ~1483 | `--registry` default `data/out/02_lineage_tracking/lineage_registry.json` |
| ~1484 | `--partitions` default `data/out/cache_cum/partitions_cum` |
| ~1488 | `--ctfidf-terms` default `data/out/02_lineage_tracking/lineage_ctfidf_terms.csv` |

### C.5 scripts/compute_lineage_embeddings.py

| Line(s) | Path Reference |
|---------|----------------|
| 1057 | `--graphs-dir` default `data/current_graphs` |
| 1063 | `--raw` default `data/current_ingest/raw` |
| ~1055 | `--registry` default `data/out/02_lineage_tracking/lineage_registry.json` |
| ~1060 | `--output-embeddings` default `data/out/02_lineage_tracking/lineage_embeddings.npz` |

### C.6 scripts/compute_front_level_features.py

| Path References |
|----------------|
| `--multisignal` default `data/out/02_lineage_tracking/lineage_multisignal_features.csv` |
| `--out` default `data/out/04_front_aggregation/front_onset_series.csv` |

### C.7 scripts/aggregate_field_metrics.py

| Path References |
|----------------|
| `--timeseries` default `data/out/02_lineage_tracking/lineage_timeseries.csv` |
| `--out-csv` default `data/out/04_front_aggregation/field_metrics.csv` |
| `--out-parquet` default `data/out/04_front_aggregation/field_metrics.parquet` |

### C.8 scripts/metric_author_influx.py

| Path References |
|----------------|
| `--slices-dir` default `data/current_ingest/slices` |
| `--out-dir` default `data/out/metrics` |

### C.9 scripts/metric_citation_velocity.py

| Path References |
|----------------|
| `--slices-dir` default `data/current_ingest/slices` |
| `--out-dir` default `data/out/metrics` |

### C.10 scripts/metric_reference_vitality.py

| Path References |
|----------------|
| `--slices-dir` default `data/current_ingest/slices` |
| `--ingest-path` default `data/current_ingest/ingest.parquet` |
| `--out-dir` default `data/out/metrics` |

### C.11 scripts/metric_topic_diversity.py

| Path References |
|----------------|
| `--slices-dir` default `data/current_ingest/slices` |
| `--out-dir` default `data/out/metrics` |

---

## Category D: Analysis/Training Scripts (9 files -- all must change)

### D.1 scripts/multi_signal_detector.py

| Path References |
|----------------|
| `--multisignal` default `data/out/02_lineage_tracking/lineage_multisignal_features.csv` |
| `--timeseries` default `data/out/02_lineage_tracking/lineage_timeseries.csv` |
| `--tight-mapping` default `data/out/experiments/stage0_.../milestone_lineage_mapping_tight.csv` |
| `--semantic-velocity` default `data/out/experiments/stage1_.../semantic_velocity.csv` |
| `--output-dir` default `data/out/experiments/multi_signal_detector/` |

### D.2 scripts/label_inflection_points.py

| Path References |
|----------------|
| `--timeseries` default `data/out/02_lineage_tracking/lineage_timeseries.csv` |
| `--milestones` default `data/out/experiments/stage0_.../milestone_lineage_mapping_tight.csv` |
| `--field-metrics` default `data/out/04_front_aggregation/field_metrics.parquet` |
| `--out` default `data/out/02_lineage_tracking/inflection_labels.csv` |

### D.3 scripts/optuna_msd_search.py

| Path References |
|----------------|
| `--labels` default `data/out/02_lineage_tracking/onset_labels_msd.csv` |
| `--multisignal` default `data/out/02_lineage_tracking/lineage_multisignal_features.csv` |
| `--timeseries` default `data/out/02_lineage_tracking/lineage_timeseries.csv` |
| `--output-dir` default `data/out/experiments/optuna_search/` |

### D.4 scripts/filter_stable_lineages.py

| Path References |
|----------------|
| `--timeseries` default `data/out/02_lineage_tracking/lineage_timeseries.csv` |
| `--out-dir` default `data/out/02_lineage_tracking` |

### D.5 scripts/run_bocpd_detector.py

| Path References |
|----------------|
| `--input` default `data/out/02_lineage_tracking/lineage_multisignal_features.csv` |
| `--output` default `data/out/experiments/bocpd_detections.csv` |

### D.6 scripts/map_lineages_to_fronts.py

| Path References |
|----------------|
| `--registry` default `data/out/02_lineage_tracking/lineage_registry.json` |
| `--graphs-dir` default `data/current_graphs` |
| `--out` default `data/out/03_milestone_mapping/lineage_to_front_mapping.json` |

### D.7 scripts/run_all_metrics.py

| Path References |
|----------------|
| `--ingest-dir` default `data/current_ingest` |
| `--graphs-dir` default `data/current_graphs` |
| `--out-dir` default `data/out/metrics` |

### D.8 scripts/run_metric_refresh.py

| Path References |
|----------------|
| `--slices-dir` default `data/current_ingest/slices` |

### D.9 scripts/metric_cross_cluster_bridging.py

| Path References |
|----------------|
| `--graphs-dir` default `data/current_graphs` |
| Line 115: hardcoded `data/current_ingest/ingest.parquet` |

---

## Category E: Horizon Scanner Scripts (4 files -- all must change)

### E.1 scripts/update_assessment_history.py

| Path References |
|----------------|
| `--history` default `data/out/assessments/assessment_history.csv` |
| `--labels` default `data/out/02_lineage_tracking/onset_labels_msd.csv` |

### E.2 scripts/generate_horizon_estimates.py

| Path References |
|----------------|
| `--predictions` default `data/out/experiments/multi_signal_detector/breakthrough_predictions.csv` |
| `--history` default `data/out/assessments/assessment_history.csv` (via update script) |
| `--out` default `data/out/assessments/horizon_estimates.csv` |

### E.3 scripts/generate_quarterly_report.py

| Path References |
|----------------|
| `--predictions` default `data/out/experiments/multi_signal_detector/breakthrough_predictions.csv` |
| `--history` default `data/out/assessments/assessment_history.csv` |
| `--horizon-estimates` default `data/out/assessments/horizon_estimates.csv` |

### E.4 scripts/refine_calibration.py

| Path References |
|----------------|
| `--history` default `data/out/assessments/assessment_history.csv` |
| `--cal-history` default `data/out/assessments/calibration_history.json` |

---

## Category F: One-Time / Utility Scripts (18 files -- all deferrable)

These scripts are used for ad-hoc analysis, benchmarking, or one-time operations.
Backward-compat symlinks cover them during the transition period.

| Script | Path Types |
|--------|------------|
| `scripts/build_lite_graphs.py` | `--graphs-dir` default `data/current_graphs` |
| `scripts/export_graphml.py` | `--graphs-dir` default `data/current_graphs` |
| `scripts/extract_abstracts.py` | Hardcoded `data/current_ingest/raw` |
| `scripts/extract_manufacturing_terms.py` | Hardcoded `data/current_ingest/raw` |
| `scripts/extract_milestone_dois.py` | Default `data/current_ingest/ingest.parquet` |
| `scripts/check_abstracts.py` | Hardcoded raw JSONL path |
| `scripts/profile_single_lineage.py` | Hardcoded `data/current_ingest/raw` |
| `scripts/prototype_avg_degree_switch.py` | Hardcoded `data/current_graphs` |
| `scripts/stage0_semantic_milestone_mapping.py` | `data/out/` paths |
| `scripts/stage1_quarterly_embeddings.py` | `data/current_ingest/ingest.parquet` |
| `scripts/stage1_quarterly_embeddings_optimized.py` | `data/current_ingest/raw` |
| `scripts/stage1_microbenchmark.py` | `data/current_ingest/raw` |
| `scripts/stage1_smoke_test.py` | `data/current_ingest/raw` |
| `scripts/stage5_ensemble_mapping.py` | `data/out/` paths |
| `scripts/test_ctfidf_smoke.py` | `data/current_ingest/raw` |
| `scripts/test_npmi_smoke.py` | `data/current_ingest/raw` |
| `scripts/generate_stability_report.py` | `data/out/` paths |
| `scripts/build_long_timeseries.py` | Hardcoded `data/out/04_front_aggregation/` paths |

---

## Category G: Dev / Archived Scripts (8 files -- all deferrable)

| Script | Notes |
|--------|-------|
| `dev/analysis/analyze_historical_coupling.py` | Hardcoded `data/current_graphs` |
| `dev/analysis/compare_graph_sizes.py` | Hardcoded `data/current_graphs` |
| `dev/analysis/ref_resolution.py` | Hardcoded `data/out` |
| `dev/diagnostics/graph_checks.py` | Hardcoded `data/out/graphs` |
| `dev/maintenance/cleanup_GraphML.py` | Hardcoded `data/out/graphs` |
| `dev/temp/temp_analyze_lgbm.py` | Hardcoded `data/out/experiments/` |
| `dev/experiments/test_parallel_safety.py` | Hardcoded `data/out/cache_coupling` |
| `dev/experiments/examples/pipeline_usage.py` | Documentation examples |

---

## Category H: Config Files (5 files, 2 must change)

### H.1 config/multisignal_config.yaml (MUST CHANGE)

Contains hardcoded paths used by feature pipeline:
- Line 44: `graphs_dir: "data/out/graphs"`
- Line 225: `abstract_cache_dir: "data/out/cache_lineage"`
- Line 230-233: `metrics_dir`, `slices_dir`, `experiments_dir`, `features_output`

**Change needed:** Add domain-aware path resolution or make paths relative
to a domain base.

### H.2 config/features/feature_groups.yaml (MUST CHANGE)

- Line 17: `source: data/out/04_front_aggregation/field_metrics.csv`

**Change needed:** Make source path relative or templated.

### H.3 config/baselines/methods.json (defer)

Contains experiment result paths -- historical reference only.

### H.4 config/baselines/methods_with_post_nov9_trials.json (defer)

Same as above -- historical reference.

### H.5 config/splits/msd_timeforward_holdout_2020.yaml (defer)

Contains experiment output paths -- historical reference.

---

## Category I: Test Files (1 file)

### I.1 tests/test_smoke.py

| Line | Reference |
|------|-----------|
| 424 | `"--outdir", "data/out"` in parse_args test |

**Change needed:** Update test to verify domain-derived path resolution.

---

## Cross-Cutting Concerns

### Hardcoded vs. Argparse Default

| Pattern | Count | Fix Strategy |
|---------|-------|--------------|
| argparse `default="data/..."` | ~55 occurrences | Change to `default=None`, derive from `--domain` |
| Hardcoded `Path("data/...")` in code | ~15 occurrences | Add `--domain` or rely on symlinks |
| Docstring examples | ~12 occurrences | Update to `data/{domain}/...` pattern |
| Config file paths | ~10 occurrences | Template or make relative |

### Shared Pattern: `sys.path.insert` + `REPO`

Most scripts use this pattern to establish the project root:
```python
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
```

The `REPO` variable is already available for `resolve_script_paths(args, REPO)`.

### Migration-Sensitive Files

These files have logic that constructs paths dynamically (not just defaults):

1. **`run.py` lines 527-540** (`setup_directories`): Derives `slices_dir` and
   `raw_dir` from `ingest_dir` -- must be updated to use `DomainDataPaths`.

2. **`scripts/metric_cross_cluster_bridging.py` line 115**: Hardcodes
   `data/current_ingest/ingest.parquet` in code body (not argparse).

3. **`scripts/run_build_pipeline.py`**: Orchestrates sub-scripts with path
   forwarding -- must forward domain-derived paths to child processes.

4. **`scripts/stage1_quarterly_embeddings_optimized.py` line 464**: Hardcodes
   `data/current_ingest/raw` in function body.

---

## Recommended Implementation Order

Based on dependency analysis and risk:

1. `src/domain_registry.py` -- foundation (all scripts depend on this)
2. `run.py` -- pipeline entry point
3. `scripts/communities.py` -- called by run.py via subprocess
4. Core feature scripts (C.1-C.11) -- produce data consumed downstream
5. Training scripts (D.1-D.3) -- consume feature data
6. Horizon scanner scripts (E.1-E.4) -- consume predictions
7. Remaining analysis scripts (D.4-D.9)
8. Config files (H.1-H.2)
9. Tests (I.1)
10. One-time scripts (F.1-F.18) and dev scripts (G.1-G.8) -- last, via symlinks

---

## File Count Summary

- **Python files with data path references:** 66 total
  - **Must change for multi-domain:** 32
  - **Deferrable (symlink-covered):** 34
- **Config files with data path references:** 5 total
  - **Must change:** 2
  - **Deferrable:** 3
- **Test files:** 1
- **New files to create:** 3 (DomainDataPaths in domain_registry, migration script, test file)
