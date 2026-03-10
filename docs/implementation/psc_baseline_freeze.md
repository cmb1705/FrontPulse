# PSC Baseline Freeze

Frozen: 2026-03-10

This document records the exact configuration, commands, outputs, and known
caveats for the PSC (perovskite solar cells) baseline.  All later detector
comparisons should cite this document as the reproducible starting point.

## Scope

The baseline covers the full FrontPulse pipeline applied to perovskite solar
cells literature from OpenAlex, ending with community lineage tracking and
MSD inflection detection.  Outputs are **retrospective only** -- there is no
prospective or online detection component in this baseline.

## Configuration Bundle

### Data Source (`config/datasources.yaml`)

- **OpenAlex Topic Filter**: `T10247` (Perovskite Solar Cells)
- **Date Range**: 2003-01-01 to present (open-ended)
- **Entity Type**: Works
- **Pagination**: 200 per page (OpenAlex max)

### Schema (`config/schema.yaml`)

- 40+ fields with explicit type coercion (Int16, Int32, boolean, string, datetime64)
- Non-null constraints on `work_id`, `publication_date`
- Range check: `cited_by_count >= 0`
- Primary index: `work_id`

### Temporal Slicing (`config/slices.yaml`)

- `last_8q`: rolling 8-quarter window (via `@cutoff` parameter)
- `by_quarter`: grouped by `pub_qtr`

### Community Detection (`config/defaults.yaml`)

| Parameter | Value | Notes |
|-----------|-------|-------|
| Resolution | 0.001 | Leiden algorithm; permissive |
| Min size | 10 | Nodes per cluster |
| Max size | 0 | Unlimited |
| Max ceiling | 15,000 | Hard cap |

### Bibliographic Coupling (`config/defaults.yaml`)

Coupling is **disabled by default** in the baseline.  When enabled:

| Parameter | Value | Source |
|-----------|-------|--------|
| alpha | 1.0 | Defaults (citation weight) |
| beta | 0.3 | Defaults (coupling weight) |
| lambda_decay | 0.15 | Temporal decay |
| min_shared_refs | 5 | Edge threshold |
| min_coupling_score | 0.25 | Literature standard (Glanzel and Czerwon 1996) |
| max_year_diff | 5 | Literature standard |

Safety constraint: `graph_workers` forced to 1 when coupling cache is active
to prevent race conditions.

### MSD Model (`config/multisignal_config.yaml`)

| Parameter | Value |
|-----------|-------|
| Default model | LightGBM |
| n_estimators | 100 |
| max_depth | 7 |
| learning_rate | 0.1 |
| class_weight | balanced |
| SMOTE | enabled (k=5) |
| Calibration | sigmoid (Platt scaling) |
| Test size | 0.25 |
| CV folds | 5 |
| Random state | 42 |
| Threshold optimization | recall with precision >= 0.20 |

### Core Feature Set (18 features)

semantic_velocity, velocity_acceleration, growth_rate, growth_acceleration,
new_works, total_works, novel_terms, novelty_rate, novelty_momentum,
awakening_intensity, cross_domain_share, cross_domain_refs,
within_lineage_refs, citation_balance, cd_index, cd_min, cd_max,
disruption_intensity.

Context features (35 additional): z-scores, quarter-on-quarter deltas,
rolling averages (1/2/4 quarters), burst detection for each metric.

## Baseline Commands

### Full pipeline run (ingest + graphs + communities)

```powershell
python run.py `
    --config config/datasources.yaml `
    --schema config/schema.yaml `
    --slices config/slices.yaml `
    --outdir data/out `
    --ingest-dir data/current_ingest `
    --graphs-dir data/current_graphs `
    --graph-mode both `
    --communities both `
    --raw-compression gzip `
    --log-level INFO `
    --use-last
```

### Reuse cached ingest (no OpenAlex API calls)

```powershell
python run.py `
    --config config/datasources.yaml `
    --schema config/schema.yaml `
    --slices config/slices.yaml `
    --outdir data/out `
    --skip-ingest `
    --graph-mode cumulative `
    --communities both
```

### Community lineage tracking (Phases 2-5)

```powershell
python scripts/run_build_pipeline.py --phases 2,3,4,5
```

Or individually:

```powershell
python scripts/communities.py --mode cumulative --graphs-dir data/current_graphs --out-dir data/out --resume
python scripts/compute_lineage_embeddings.py
python scripts/compute_lineage_ctfidf.py
python scripts/compute_lineage_npmi.py
python scripts/stage5_ensemble_mapping.py
```

### MSD training and evaluation

```powershell
python scripts/multi_signal_detector.py `
    --model lightgbm `
    --use-cv --cv-folds 5 `
    --calibrate --calibration-method sigmoid

python scripts/evaluate_inflection_detection.py --threshold 0.07 --n-timeline-plots 20
```

## Expected Outputs

| Path | Description |
|------|-------------|
| `data/current_ingest/ingest.parquet` | Full cached corpus |
| `data/current_ingest/raw/openalex_raw_*.jsonl.gz` | Chunked raw snapshots |
| `data/current_ingest/slices/by_quarter__*.parquet` | Per-quarter slices |
| `data/current_graphs/citation_graph_*.pkl` | Serialized NetworkX graphs |
| `data/out/manifest.json` | Pipeline execution metadata |
| `data/out/02_lineage_tracking/lineage_registry.json` | Persistent lineage IDs |
| `data/out/02_lineage_tracking/lineage_timeseries.csv` | Lineage time series |
| `data/out/02_lineage_tracking/lineage_metrics.csv` | Per-quarter lineage metrics |
| `data/out/02_lineage_tracking/lineage_embeddings.npz` | SciBERT embeddings |
| `data/out/02_lineage_tracking/lineage_ctfidf_terms.csv` | c-TF-IDF terms |
| `data/out/02_lineage_tracking/lineage_npmi_pairs.csv` | NPMI co-occurrence |
| `data/out/03_milestone_mapping/lineage_front_mappings.csv` | Ensemble mappings |
| `data/out/experiments/multi_signal_detector/` | Model, predictions, metrics |

## Known Caveats

1. **Retrospective only**.  All outputs reflect post-hoc analysis.  No
   detection windows, prospective splits, or time-forward holdouts are
   defined by this baseline.

2. **Coupling disabled**.  The baseline uses citation-only graphs.  Coupling
   augmentation is available but was not part of the standard run.

3. **Coupling cache race condition**.  If coupling is enabled with caching,
   `graph_workers` must be 1 (sequential) to avoid data corruption.  The
   pipeline enforces this automatically.

4. **Phase orchestration overhead**.  Phases 2-5 run via subprocess isolation
   by default.  `run_build_pipeline.py` supports shared `LineageTextStore`
   for in-memory execution but this is not the default path.

5. **Feature count discrepancy**.  The baseline core feature set has 18
   features, while `config/features/feature_groups.yaml` references 20.
   The delta is tracked by a pre-existing test failure
   (`test_core_group_contains_expected_features`).

6. **Email requirement**.  OpenAlex ingestion requires a contact email via
   `--mailto` or `config/settings.yaml`.  This is not committed to version
   control for privacy.

7. **Threshold sensitivity**.  The MSD default threshold (0.70) differs
   from the evaluation threshold used in holdout analysis (0.07).  The
   evaluation threshold was tuned for recall; the model threshold is
   conservative.

## Reproducibility Notes

- Python 3.9+ required; 3.10 tested
- OS: Windows (PowerShell commands shown; adapt for bash)
- Memory: 16GB+ recommended (8GB minimum with warnings)
- Default parallel workers: 12 (adjust via `DEFAULT_PARALLEL_WORKERS` in `run.py`)
- Random state: 42 throughout (numpy, sklearn, train/test split)
