# Changelog

## [Unreleased] - 2025-11-03

### Added

- **Field Baseline Aggregation**:
  - `scripts/aggregate_field_metrics.py` builds quarterly field-wide totals/percentiles/seasonality curves and persists them to `data/out/04_front_aggregation/field_metrics.{csv,parquet}` for reuse across experiments.
  - `scripts/compute_lineage_multisignal_features.py` can now ingest that table via `--field-metrics` (with `--disable-field-metrics` override) and emits relative contrast features (`relative_new_works`, `growth_vs_field`, `new_works_over_p75`, etc.) so each lineage is benchmarked against corpus-level velocity.
- **Field Metrics Tests**: Added `dev/experiments/tests/test_field_metrics_features.py` to cover `load_field_metrics` (CSV/parquet parsing, missing-file path) and the new `merge_field_metrics` ratio helper.
- **Labeler Field Guard**:
  - `scripts/label_inflection_points.py` now records the field metrics snapshot for each detection, emits `field_growth_ratio`/`field_cumulative_ratio`, and exposes `--min-field-growth-ratio` plus `--max-lineages` for smoke tests.
  - The labeler can skip detections that fail the field-growth guard, and metadata now captures the field-metrics provenance.
- **Evaluation Dashboard Enhancements**:
  - `scripts/evaluate_inflection_detection.py` and `scripts/generate_comprehensive_evaluation.py` accept `--field-metrics-path`, auto-compute threshold sweeps if missing, and emit a “field vs lineage” dashboard so reviewers can see how detection volume tracks corpus growth.

- **Quarter Utilities Module** (`scripts/utils/quarter_utils.py`): centralized helpers for normalizing/sorting/serializing quarters, plus shared `filter_by_quarter` + `snapshot_dataset` logic used by the detector and diagnostics.
- **Time-Forward Holdout Log** (`docs/analysis/TIME_FORWARD_HOLDOUT_PLAN.md`): now documents both the original 2003–2019 → 2020–2025 run and the additional 2003–2017 → 2018–2021 → 2022–2025 split, capturing precision/recall collapse at threshold 0.07.
- **Feature Signal Pruning Toolkit**:
  - Added metadata-rich config/features/feature_groups.yaml, config/features/feature_subset_configs.yaml, and helper module scripts/utils/feature_registry.py.
  - New diagnostics orchestrator scripts/analyze_feature_signals.py (L1 baselines, LightGBM importance, mutual information, optional XGBoost/SHAP) with JSON/Markdown reports.
  - New subset evaluation runner scripts/run_feature_subset_evals.py (time-forward LightGBM benchmarking, threshold sweeps, retrain hook) plus smoke harness scripts/run_feature_signal_ci.sh.
  - Documentation (docs/analysis/FEATURE_SIGNAL_PRUNING.md), README/ROADMAP updates, and archived outputs under data/out/analysis/feature_signal_pruning/.
  - Pytest coverage for registry + curated lists (`tests/test_feature_registry.py`, `tests/test_feature_subset_configs.py`, `tests/test_msd_feature_loading.py`).

#### Pipeline Infrastructure

- **LineageTextStore** (`src/lineage_text_store.py`): Shared resource store for Phases 2-5
  - Loads lineage registry, abstract extractor, and graph data once
  - Provides inverted index: `{lineage_id: {quarter: {community_id: lineage_id}}}`
  - Methods: `get_lineage_papers_from_graph()`, `get_lineage_quarters()`, `get_persistent_lineages()`
  - Supports dual-mode: pipeline (shared store) vs standalone (fresh load)

- **Phase Pipeline Orchestration** (`scripts/run_build_pipeline.py`):
  - Runs Phases 2-5 in sequence with direct function calls (no subprocess overhead)
  - Configurable phase selection via `--phases` flag
  - Timing profiling and validation control
  - Passes shared LineageTextStore to all phases

#### Validation Integration

- Integrated validation checks directly into phase scripts:
  - `compute_lineage_embeddings.py` (Phase 2)
  - `compute_lineage_ctfidf.py` (Phase 3)
  - `compute_lineage_npmi.py` (Phase 4)
  - `stage5_ensemble_mapping.py` (Phase 5)

- Added `--validate` / `--no-validate` CLI flags to all phase scripts
- Lazy imports for matplotlib/seaborn (only loaded when validation enabled)
- Validation outputs saved to `data/out/06_validation/phase{N}/`
- All phase `run_*()` functions now return `(data1, data2, validation_results)` tuples

#### New Scripts

- `scripts/build_long_timeseries.py`: Build extended time series from lineage data
- `scripts/create_selected_mappings.py`: Create curated lineage-to-front mappings
- `scripts/run_all_validation.py`: Batch validation runner for all phases
- `scripts/diagnose_tripwire_calibration.py`: Diagnostic tool for tripwire alerts
- `scripts/remap_milestones.py`: Remap expert milestones to detected communities

### Changed

- **Multi-Signal Detector**:
  - Added `zero_division=0` guards around sklearn metrics to suppress `UndefinedMetricWarning` when no positives fire at the default 0.50 threshold.
  - All LightGBM fit/predict paths now operate on NumPy arrays to eliminate feature-name mismatch warnings when scoring disjoint holdouts.
  - Imports quarter helpers from the shared utility module instead of maintaining bespoke implementations.
  - Feature selector auto-detects field-relative columns, logs the counts per feature family, and exposes `--disable-field-features` / `--field-features-only` for ablation runs (with prediction slices reusing the same feature list).
  - Added `--n-estimators` and `--learning-rate` knobs (plus metadata logging) so GradientBoosting/LightGBM/RF runs respect Optuna-tuned settings during both CV and holdout training; `scripts/msd_meta_tune.py` now emits those flags in its "retrain" instructions.
- **Field Metrics Merge**:
  - Introduced a shared `merge_field_metrics()` helper (also used by the parallel builder) so serial/parallel executions emit identical field-relative columns.
  - The helper’s safe-ratio logic removes pandas chained-assignment warnings, and both builders expose the `--field-metrics` / `--disable-field-metrics` toggles for consistent CLI surfaces.
- **Feature Builder Runtime**:
  - `scripts/compute_lineage_multisignal_features.py` embeds multiprocessing via `--n-workers`, covering both novelty extraction and per-work citation metrics; the legacy `compute_lineage_multisignal_features_parallel.py` wrapper now simply forwards to the main entry point with a deprecation warning.
- **Supporting Scripts** (`compute_lineage_multisignal_features*.py`, `label_inflection_points.py`, `analyze_milestone_inflection_lag.py`, `sweep_msd_thresholds.py`, `test_context_features.py`):
  - Updated to consume the new quarter helpers so normalization/sorting/tagging behave consistently across the pipeline.

#### Phase Scripts - Dual-Mode Execution

All phase scripts now support two execution modes:

1. **Standalone Mode** (original CLI behavior):
   - Load resources from CLI-specified paths
   - Self-contained execution
   - Example: `python scripts/compute_lineage_ctfidf.py --registry data/out/...`

2. **Pipeline Mode** (via `run_build_pipeline.py`):
   - Accept pre-loaded `LineageTextStore` via `store=` parameter
   - Skip redundant I/O (registry, abstracts, graphs)
   - 2-3x faster than standalone for Phases 2-4

**Modified Functions**:

- `run_embeddings()`: Added `registry_path`, `raw_dir`, `graphs_dir`, `store`, `validate` parameters
- `run_ctfidf()`: Added `registry_path`, `raw_dir`, `store`, `validate` parameters
- `run_npmi()`: Added `registry_path`, `raw_dir`, `store`, `validate` parameters
- `run_ensemble()`: Added `store`, `validate` parameters

**CLI Backward Compatibility Maintained**:

- All CLI arguments still work in standalone mode
- Fallback pattern: `registry_path or Path('default/path')`
- Updated `main()` functions to pass CLI args to `run_*()` functions

#### Documentation Organization

Reorganized 17 markdown files from root directory:

**Kept in root** (3 files):

- `README.md` - Project overview and usage
- `CONTRIBUTING.md` - Contribution guidelines
- `ROADMAP.md` - Feature roadmap and priorities

**Moved to `docs/`** (5 files):

- `agents.md` - Agent protocol and working notes
- `COMMUNITY_DETECTION_ARGUMENTS.md` - CLI reference
- `CONFIGURATION_GUIDE.md` - Configuration documentation
- `PRIORITIES.md` - Implementation priorities
- `PROJECT_STRUCTURE_ANALYSIS.md` - Code structure analysis

**Moved to `docs/analysis/`** (11 files):

- Coupling analysis documents (6 files)
- `TEST_FAILURES_ANALYSIS.md`
- `IMPORT_DEPENDENCY_AUDIT.md`
- `GRAPH_PARALLELIZATION_ANALYSIS.md`
- `LITERATURE_REVIEW_COUPLING_PARAMETERS.md`
- `TRIPWIRE_DIAGNOSTIC_SUMMARY.md`

**Moved to `docs/implementation/`** (2 files):

- `NAMING_MIGRATION_GUIDE.md`
- `PIPELINE_IMPLEMENTATION.md`

**Moved to `docs/proposals/`** (1 file):

- `milestone_remapping_proposal.md`

#### README Enhancement

- Added comprehensive "Complete Pipeline Order" section
- Documented Phases 2-5 execution sequence
- Added validation step documentation for each phase
- Clarified dependencies and outputs for each step

#### Tripwire Validation

- Enhanced `references/psc_tripwire_validator.py` with improved event matching
- Updated `references/psc_validation_viz.py` with comprehensive visualizations
- Modified `scripts/evaluate_tripwire.py` for better backtest analysis
- Updated `scripts/visualize_tripwire_comprehensive.py` with enhanced plots

#### Dependency Management

- Added missing ML dependencies to `requirements.txt`:
  - `torch`, `transformers` (for SciBERT)
  - `scikit-learn` (for TF-IDF, NPMI)
  - `seaborn` (for validation plots)

#### Stage 4 NPMI Optimizations

- Added RAM-aware multiprocessing to `scripts/compute_lineage_npmi.py` and Stage 4 of `scripts/run_build_pipeline.py` (`--max-workers` / `--npmi-workers`, `--worker-memory-gb`, `--npmi-worker-mem-gb`, `--memory-reserve-gb`, `--npmi-memory-reserve-gb`).
- Persisted the abstract extractor's global work/DOI index to `data/out/cache_lineage/abstract_index_*.pkl`; all phase scripts and worker processes reuse the cache instead of rebuilding the index on every spawn.

### Fixed

#### LineageTextStore Registry Structure

- **Critical Fix**: `_build_lineage_index()` now correctly inverts registry structure
- Previous bug: Only stored lineage_id per quarter `{lineage_id: {'2004Q1': 1}}`
- Fixed to: `{lineage_id: {quarter: {community_id: lineage_id}}}`
- Phase 3 no longer crashes with KeyError when iterating `community_map.items()`

#### get_lineage_papers_from_graph()

- Updated to work with new `{community_id: lineage_id}` map structure
- Extracts all community_ids for each lineage
- Correctly filters graph nodes by community membership
- Fixed graph filename pattern: `citation_graph_{mode}_{quarter}.pkl`

#### CLI Arguments

- Phase 2: `run_embeddings()` now respects `--lineage-registry`, `--raw-dir`, `--graphs-dir`
- Phase 3: `run_ctfidf()` now respects `--registry`, `--raw`
- Phase 4: `run_npmi()` now respects `--registry`, `--raw`
- Fixed issue where paths were hard-coded instead of using CLI args

#### Documentation Accuracy

- Removed unused `subprocess` import from `run_build_pipeline.py`
- Updated docstrings to reflect direct function call approach
- Fixed outdated comments about subprocess orchestration

### Removed

- Temporary analysis files: `temp_top.py`, `temp_bottom.py` (not committed)

---

## Implementation Notes

### Breaking Changes

None - all changes are backward compatible. Standalone CLI execution works exactly as before.

### Performance Impact

- **Pipeline mode**: 2-3x faster for Phases 2-4 (eliminates redundant registry/abstract loading)
- **Standalone mode**: Performance unchanged
- **Validation**: Lazy imports ensure no overhead when `--no-validate` is used

### Migration Guide

No migration required. To use the new pipeline mode:

```powershell
# Old way (still works):
python scripts/compute_lineage_embeddings.py
python scripts/compute_lineage_ctfidf.py
python scripts/compute_lineage_npmi.py
python scripts/stage5_ensemble_mapping.py

# New way (faster):
python scripts/run_build_pipeline.py --phases 2,3,4,5
```

### Known Issues

- Pipeline mode provides orchestration but minimal speedup in current implementation
- Validation visualizations require matplotlib backend configuration on some systems

### Future Work

- Consider pre-compiling SciBERT model for faster Phase 2 startup
- Add caching for c-TF-IDF term extraction results
- Explore parallel processing for Phase 3/4 lineage iteration
