# FrontPulse

Dynamic research-front monitoring on OpenAlex bibliometric data. The pipeline ingests scholarly works, builds citation graphs, detects community lineages via Leiden clustering, maps lineages to research fronts, and identifies inflection points where exponential growth begins.

## Environment Setup

1. Create and activate a virtualenv:

   ```powershell
   python -m venv .venv
   .\.venv\Scriptsctivate
   pip install -r requirements.txt
   ```

   For development (linting, type checking, testing):

   ```powershell
   pip install -r requirements-dev.txt
   ```

   Community detection requires `python-igraph` and `leidenalg` (included in `requirements.txt`). If installation fails on Windows:

   ```powershell
   conda install -c conda-forge python-igraph leidenalg
   ```

2. Verify installation:

   ```powershell
   python -c "import igraph, leidenalg; print('OK')"
   ```

3. Configure settings (optional):

   ```powershell
   copy config\settings.yaml.template config\settings.yaml
   ```

   Settings priority (highest to lowest): CLI arguments > `config/settings.yaml` > `.2yp_settings.json` (legacy) > built-in defaults. The template is version-controlled; your `config/settings.yaml` is git-ignored.

## Project Layout

```text
.
├─ run.py               # Primary pipeline driver (11 modular phases)
├─ config/
│  ├─ datasources.yaml  # OpenAlex query configuration
│  ├─ schema.yaml        # Data schema definitions
│  ├─ slices.yaml        # Temporal slice definitions
│  ├─ defaults.yaml      # Global defaults (workers, coupling params)
│  └─ settings.yaml.template
├─ src/                  # Core library modules
│  ├─ ingest.py          # OpenAlex data ingestion
│  ├─ transform.py       # Data transformations, time variables
│  ├─ validate.py        # Schema enforcement
│  ├─ slicing.py         # Temporal/categorical partitioning
│  ├─ graph_build.py     # Citation graph construction
│  ├─ community.py       # Leiden clustering
│  ├─ alignment.py       # Community alignment across time
│  ├─ config.py          # Configuration loading
│  ├─ logging_config.py  # Centralized logging
│  └─ memory_utils.py    # Memory monitoring
├─ scripts/              # Analysis, processing, and pipeline scripts
├─ data/
│  ├─ current_ingest/    # Cached parquet ingest and per-slice outputs
│  ├─ current_graphs/    # Graph exports (annual, delta, cumulative)
│  ├─ out/               # Reports, manifests, registries, experiment outputs
│  └─ archive/           # Timestamped snapshots
├─ tests/                # Pytest test suite
└─ docs/                 # Technical guides
```

### Key Data Artifacts

Data is organized per domain under `data/{domain_id}/` (e.g., `data/psc/`, `data/crispr/`).
Use `--domain psc` or `--domain crispr` for automatic path resolution.

- `data/{domain}/ingest/ingest.parquet` -- full cached corpus
- `data/{domain}/ingest/raw/*.jsonl` -- chunked OpenAlex raw snapshots with byte-offset indexes
- `data/{domain}/ingest/slices/*.parquet` -- per-slice parquet files
- `data/{domain}/graphs/citation_graph_*` -- exported graphs (annual, delta, cumulative)
- `data/{domain}/out/` -- JSON/CSV manifests, community summaries, debug reports
  - `lineage_timeseries.csv` -- time series tracking community lineages
  - `lineage_metrics.csv` -- per-quarter metrics for each lineage
  - `lineage_registry.json` -- mapping of quarter-specific community IDs to persistent lineage IDs
  - `communities_cumulative.json` -- full community detection results per quarter
- `data/{domain}/out/metrics/` -- scientometric context metrics with provenance tracking
- `data/{domain}/archive/<YYYYMMDD_HHMMSS>/` -- timestamped snapshots

## Terminology

- **Community lineages** (`lineage_id`): Persistent cluster identities tracked across time via PageRank core overlap.
- **Research fronts**: Intellectual problem domains within the field (named by domain experts). Multiple lineages map to each front (many-to-one).
- **Inflection point**: The quarter when exponential growth begins in a lineage's publication count.

## Primary Pipeline (`run.py`)

The pipeline runs 11 modular phases: Setup, Settings, Preflight, Ingest, Slicing, Graph Building, Slices Storage, Manifest Handling, Community Detection, Manifest Writing, Archival.

```text
usage: run.py --config CONFIG --schema SCHEMA --slices SLICES --outdir OUTDIR
              [--ingest-dir INGEST_DIR] [--graphs-dir GRAPHS_DIR]
              [--graph-mode {none,annual,delta,both,cumulative}]
              [--interactive] [--use-last] [--skip-preflight] [--skip-ingest]
              [--archive] [--archive-only]
              [--communities {none,annual,delta,both}]
              [--log-level {DEBUG,INFO,WARNING,ERROR,CRITICAL}]
              [--enable-coupling] [--coupling-*]
```

Key flags:

- `--domain` -- domain selector (`psc` or `crispr`); auto-resolves all data paths
- `--outdir` (default: domain-derived or `data/out`) -- report destination
- `--ingest-dir` (default: domain-derived or `data/current_ingest`) -- cached ingest + slices
- `--graphs-dir` (default: domain-derived or `data/current_graphs`) -- graph outputs
- `--mailto` -- contact email for OpenAlex (falls back to `config/settings.yaml`)
- `--skip-ingest` -- reuse `ingest.parquet` instead of re-downloading
- `--rebuild-ingest-from-raw` -- reconstruct ingest from raw NDJSON (no API calls)
- `--archive` -- snapshot ingest/graphs/out to `data/archive/<timestamp>/`
- `--graph-mode` -- choose graph types: `annual`, `delta`, `both`, `cumulative` (default)
- `--enable-coupling` -- enable bibliographic coupling augmentation

### Typical Workflows

**Initial full run** (ingest + graphs + communities):

```powershell
python run.py `
    --config config/datasources.yaml `
    --schema config/schema.yaml `
    --slices config/slices.yaml `
    --outdir data/out `
    --graph-mode both `
    --communities both `
    --use-last
```

**Reuse cached ingest** (skip remote download):

```powershell
python run.py --config config/datasources.yaml --schema config/schema.yaml `
    --slices config/slices.yaml --skip-ingest
```

**Rebuild from raw snapshot** (no OpenAlex requests):

```powershell
python run.py --config config/datasources.yaml --schema config/schema.yaml `
    --slices config/slices.yaml --rebuild-ingest-from-raw --skip-raw --use-last
```

**Archive current outputs**:

```powershell
python run.py --domain psc --archive-only
```

### Outputs

- `data/{domain}/ingest/ingest.parquet`, `slices/by_quarter__*.parquet`
- `data/{domain}/ingest/raw/openalex_raw_*.jsonl` + `_index.csv` + manifest
- `data/{domain}/graphs/citation_graph_*`
- `data/{domain}/out/manifest.json`
- `data/{domain}/out/logs/` -- rotating pipeline logs
- `data/{domain}/out/communities_*.json`, `front_id_registry*.json`, `front_timeseries.csv`
- `data/{domain}/out/eval/` -- backtest outputs (alerts, validation, diagnostic plots)

## Complete Pipeline Order (Steps 1-10)

After the primary pipeline produces community lineages, run these scripts **in order** to map lineages to research fronts and generate alerts.

> **Domain isolation**: All scripts accept `--domain psc` or `--domain crispr`
> for automatic path resolution under `data/{domain}/`. The `data/out/` paths
> shown below are PSC examples; substitute your domain or use `--domain`
> to override.

### Step 1: Community Detection

```powershell
python scripts/communities.py --mode cumulative --domain psc --resume
```

Outputs: `lineage_registry.json`, `lineage_timeseries.csv`, `lineage_metrics.csv` in `data/{domain}/out/02_lineage_tracking/`

Community modes: `cumulative` (default), `annual`, `delta`, `both`, `all`. Resolution sweeps available via `--res-sweep`.

### Step 2: SciBERT Embeddings

```powershell
python scripts/compute_lineage_embeddings.py
```

Computes semantic embeddings for persistent lineages using SciBERT. Requires PyTorch and transformers.

Outputs: `data/out/02_lineage_tracking/lineage_embeddings.npz`

### Step 3: c-TF-IDF Term Extraction

```powershell
python scripts/compute_lineage_ctfidf.py
```

Extracts distinctive terms for each lineage using c-TF-IDF with domain normalization.

Outputs: `data/out/02_lineage_tracking/lineage_ctfidf_terms.csv`

### Step 4: NPMI Co-occurrence

```powershell
python scripts/compute_lineage_npmi.py
```

Discovers co-occurring term pairs using Normalized Pointwise Mutual Information.

Outputs: `data/out/02_lineage_tracking/lineage_npmi_pairs.csv`

### Step 5: Ensemble Mapping

```powershell
python scripts/stage5_ensemble_mapping.py
```

Combines Phase 2-4 signals via ensemble voting to assign lineages to research fronts.

Outputs:

- `data/out/03_milestone_mapping/lineage_front_mappings.csv` -- mappings with confidence scores
- `data/out/mapping_evidence/*.json` -- evidence bundles for each decision

Confidence levels: **High** (3/3 phases agree), **Medium** (2/3 agree), **Low** (no consensus). Multi-label support for lineages spanning multiple fronts.

### Step 6: Curate Mappings (Manual Review)

```powershell
# Automatic filtering to medium+ confidence (recommended)
python scripts/create_selected_mappings.py

# Or manually edit lineage_front_mappings.csv
```

### Step 7: Aggregate Lineages to Fronts

```powershell
python scripts/aggregate_lineages_to_fronts.py
```

Rolls up lineage-level metrics to front-level time series.

Outputs: `data/out/04_front_aggregation/front_timeseries_{delta,cumulative}.csv` (wide and long formats)

### Step 8: Tripwire Detection

```powershell
python tripwire_nb_fdr.py `
    --timeseries data/out/04_front_aggregation/front_timeseries_delta_long.csv `
    --front-col front --date-col quarter --count-col count `
    --out data/out/05_tripwire_detection/alerts_tripwire.csv `
    --lookback 8 --min-history 6 --min-count 2
```

Generates statistical alerts for unusual activity spikes at the front level. Use `front_timeseries_delta_long.csv` (new works per quarter), not cumulative data.

### Step 9: Evaluate Against Milestones

```powershell
python scripts/evaluate_tripwire.py --milestones <path-to-milestone-catalog>
```

Runs backtest against a user-provided milestone catalog. Outputs alerts, validation matches, and diagnostic figures under `data/out/06_validation/`.

### Step 10: Field-Level Aggregation (Optional)

```powershell
python scripts/aggregate_front_alerts_to_field.py
```

Summarizes front-level alerts and metrics at the field level.

### Phase Validation

Validate any phase's outputs:

```powershell
python scripts/validate_stage2.py   # Phase 2
python scripts/validate_stage3.py   # Phase 3
python scripts/validate_stage4.py   # Phase 4
python scripts/validate_stage5.py   # Phase 5

# Or run all validations at once
python scripts/run_all_validation.py
python scripts/run_all_validation.py --skip-phases 2,3  # skip specific phases
```

## Inflection Detection Pipeline (S1-S6)

Detects inflection points -- quarters when exponential growth begins in a community lineage. Uses multi-signal gradient boosting with leakage-free feature engineering.

| Stage | Script | Purpose |
|-------|--------|---------|
| S1 | `scripts/label_inflection_points.py` | Dual-pathway labeling (logistic + derivative) |
| S2 | `scripts/analyze_milestone_inflection_lag.py` | Milestone-inflection relationship analysis |
| S3 | `scripts/compute_lineage_multisignal_features.py` | 55 features (20 core + 35 context), no temporal leakage |
| S4 | `scripts/multi_signal_detector.py` | LightGBM + isotonic calibration |
| S5 | `scripts/evaluate_inflection_detection.py` | Lag-focused evaluation with threshold sweeps |
| S6 | Documentation | Artifact manifest, reproducibility protocol |

### Running the Inflection Pipeline

**Label generation**:

```powershell
python scripts/label_inflection_points.py `
    --timeseries data/out/02_lineage_tracking/lineage_timeseries.csv `
    --milestones <path-to-milestone-catalog> `
    --out data/out/02_lineage_tracking/inflection_labels.csv `
    --plot-dir data/out/figures/inflection_qc
```

**Feature computation** (55 features: 20 core + 35 context):

```powershell
python scripts/compute_lineage_multisignal_features.py `
    --enable-context-features --n-workers 12
```

Features include causal logistic fits (no future leakage), derivatives, accelerations, rolling statistics, and field-relative ratios. Use `--field-metrics` to add corpus-level benchmarks, `--enable-milestone-proximity --milestones <path>` for milestone features.

**Model training**:

```powershell
python scripts/multi_signal_detector.py --use-cv --cv-folds 5
```

Supports LightGBM, XGBoost, CatBoost, Random Forest, and Logistic Regression. Use `--disable-field-features` / `--field-features-only` for ablation studies.

**Evaluation**:

```powershell
python scripts/evaluate_inflection_detection.py --threshold 0.07 --n-timeline-plots 20
```

Generates lag distribution dashboards, timeline plots, false positive analysis, threshold sweep tables, and evaluation summary JSON.

**Hyperparameter tuning** (optional):

```powershell
python scripts/msd_meta_tune.py --n-trials 80 `
    --output-dir data/out/experiments/msd_meta_tuning
python scripts/multi_signal_detector.py `
    --config data/out/experiments/msd_meta_tuning/best_config.json --use-cv --cv-folds 5
```

### Baseline Comparisons

Three baseline methods for benchmarking the detector:

1. **Simple heuristics** (`scripts/baseline_simple_heuristics.py`) -- single-feature threshold rules
2. **Kleinberg burst detection** (`scripts/baseline_kleinberg_burst.py`) -- citation burst detection
3. **Semantic changepoint** (`scripts/baseline_semantic_changepoint.py`) -- unsupervised changepoint detection

```powershell
python scripts/baseline_simple_heuristics.py
python scripts/baseline_kleinberg_burst.py
python scripts/baseline_semantic_changepoint.py
python scripts/evaluate_baseline_methods.py
```

Outputs stored in `data/out/experiments/baselines/`.

## Multi-Signal Context Metrics

The pipeline computes five field-level scientometric context metrics (via `scripts/run_metric_refresh.py`):

1. **Author Influx** -- new vs. returning author rates
2. **Citation Velocity** -- citation accumulation speed
3. **Reference Vitality** -- proportion of recent references
4. **Topic Diversity** -- concept breadth across descriptors
5. **Cross-Cluster Bridging** -- network bridging behavior

From each metric, 7 context features are derived per lineage-quarter (z-scores, deltas, rolling averages, burst detection), yielding 35 context features alongside 20 core lineage features.

```powershell
# Regenerate metrics
python scripts/run_metric_refresh.py --domain psc

# Aggregate field baselines
python scripts/aggregate_field_metrics.py
```

Configuration: [config/multisignal_config.yaml](config/multisignal_config.yaml)

## Feature Signal Pruning

Two CLI utilities support feature QA and ablation analysis:

```powershell
# Feature diagnostics (linear baselines + LightGBM importances + univariate screens)
python scripts/analyze_feature_signals.py `
    --sample-limit 2000 --tree-sample-limit 2000 --cv-folds 3 `
    --disable-shap --output-dir data/out/analysis/feature_signal_pruning/diagnostics_smoke

# Feature subset evaluation (trains on curated bundles with threshold sweeps)
python scripts/run_feature_subset_evals.py `
    --sample-limit 3000 --max-configs 2 `
    --output-dir data/out/analysis/feature_signal_pruning/subset_smoke
```

## Performance Configuration

Default: **12 parallel workers** (optimized for 16+ core systems).

Adjust the global default in `run.py`:

```python
DEFAULT_PARALLEL_WORKERS = 12  # Change to 4-8 for smaller systems
```

Or override per-run:

```powershell
python run.py --config config.yaml --graph-workers 6 --coupling-workers 6
```

Stage 4 (NPMI) supports RAM-aware multiprocessing:

```powershell
python scripts/run_build_pipeline.py `
    --stages 4 --npmi-workers 6 `
    --npmi-worker-mem-gb 3 --npmi-memory-reserve-gb 4
```

See [Performance Configuration Guide](docs/PERFORMANCE_CONFIG.md) for detailed recommendations.

## Testing

```powershell
pytest                                    # Run all tests
pytest -v                                 # Verbose output
pytest tests/test_transform.py            # Single file
pytest -m unit                            # Unit tests only
pytest -m integration                     # Integration tests
pytest --cov=src --cov-report=html        # With coverage
```

Markers: `@pytest.mark.unit`, `@pytest.mark.integration`, `@pytest.mark.slow`. Shared fixtures in `tests/conftest.py`.

## Code Quality

- Python 3.9+ with type hints on all public functions
- PEP 8 compliance (line length 100, enforced by black/ruff)
- Google-style docstrings on all public functions
- Structured logging with rotating file handlers (`data/out/logs/`)
- Modular architecture: 11 pipeline phases, each <150 lines

## Archival

Running with `--archive` (or `--archive-only`) produces a timestamped snapshot:

```text
data/archive/20250115_143512/
├─ ingest/
├─ graphs/
└─ out/
```

Use this to freeze outputs before experimenting with new settings or code changes.
