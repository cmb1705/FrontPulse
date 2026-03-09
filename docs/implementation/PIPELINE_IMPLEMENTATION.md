# Phase 2-5 Pipeline Implementation

## Overview

Implemented a pipeline orchestration system for running Phases 2-5 in sequence.

**Current Status**: Orchestration only (no speedup yet). Phase scripts run via subprocess, so they reload registry independently. Infrastructure is in place for future in-memory execution.

> **Update (2025-11-05):** Stage 4 (NPMI) now supports optional multiprocessing when invoked through
> `scripts/compute_lineage_npmi.py` or `scripts/run_build_pipeline.py`. Use `--max-workers` / `--npmi-workers`
> together with the memory guard flags to run multiple workers safely. Installing `psutil` enables the RAM
> checks; otherwise Stage 4 falls back to sequential mode. The abstract extractor also maintains a serialized
> global index cache in `data/out/cache_lineage/`, so worker processes can mount the cache instead of rebuilding
> the index on every spawn.

## Implementation Status: **Option 1 (Orchestration Only)**

### What's Implemented

1. **Shared Resource Store** ([src/lineage_text_store.py](src/lineage_text_store.py))
   - `LineageTextStore` class loads registry, abstracts, and graph data once
   - Provides fast access to lineage papers, texts, and metadata
   - Infrastructure ready for in-memory execution
   - **NOTE**: Currently unused by phase scripts due to subprocess isolation

2. **Pipeline Driver** ([scripts/run_build_pipeline.py](scripts/run_build_pipeline.py))
   - Orchestrates Phases 2-5 in sequence
   - **LIMITATION**: Uses subprocess.run(), so NO memory sharing
   - Each phase reloads registry independently (no speedup)
   - Configurable phase selection (e.g., run only phases 3,4,5)
   - Timing profiling for performance tracking

### Critical Issues Fixed

1. **CLI argument mismatches**: Removed unsupported `--output` flags
2. **Graph filename pattern**: Fixed to match actual `citation_graph_{mode}_{quarter}.pkl` format
3. **Honest documentation**: Clarified that current implementation provides orchestration only, not speedup

### Current Architecture

```
┌─────────────────────────────────────────┐
│  run_build_pipeline.py                  │
│  ┌──────────────────────────────────┐   │
│  │ 1. Load LineageTextStore         │   │
│  │    - lineage_registry.json       │   │
│  │    - AbstractExtractor (raw/)    │   │
│  │    - Graph/partition paths       │   │
│  └──────────────────────────────────┘   │
│                                          │
│  set_shared_store(store) # Global       │
│                                          │
│  ┌──────────────────────────────────┐   │
│  │ 2. Run Phase 2 (subprocess)      │   │
│  │    → compute_lineage_embeddings  │   │
│  └──────────────────────────────────┘   │
│                                          │
│  ┌──────────────────────────────────┐   │
│  │ 3. Run Phase 3 (subprocess)      │   │
│  │    → compute_lineage_ctfidf      │   │
│  └──────────────────────────────────┘   │
│                                          │
│  ┌──────────────────────────────────┐   │
│  │ 4. Run Phase 4 (subprocess)      │   │
│  │    → compute_lineage_npmi        │   │
│  └──────────────────────────────────┘   │
│                                          │
│  ┌──────────────────────────────────┐   │
│  │ 5. Run Phase 5 (subprocess)      │   │
│  │    → phase5_ensemble_mapping     │   │
│  └──────────────────────────────────┘   │
└─────────────────────────────────────────┘
```

**Limitation:** Each subprocess still loads its own registry/texts because they're separate Python processes. The shared store is available but not yet used by the phase scripts.

## Usage

### Run Full Pipeline

```bash
# Default: Phases 2,3,4,5 with standard paths
python scripts/run_build_pipeline.py

# Profile timing
python scripts/run_build_pipeline.py --profile

# Custom phases
python scripts/run_build_pipeline.py --phases 3,4,5  # Skip embeddings
```

### Run Individual Phases (Existing Scripts Still Work)

```bash
# Scripts still work independently
python scripts/compute_lineage_embeddings.py
python scripts/compute_lineage_ctfidf.py
python scripts/compute_lineage_npmi.py
python scripts/phase5_ensemble_mapping.py
```

## Path to Full Option 2 (In-Memory Execution)

To achieve true in-memory execution with the shared store, refactor phase scripts:

### Step 1: Extract Core Logic

Modify each phase script to separate CLI parsing from core logic:

**Before** (compute_lineage_embeddings.py):

```python
def main():
    args = parse_args()
    # Load registry
    with open(args.lineage_registry) as f:
        registry = json.load(f)
    # Do embedding computation
    ...
```

**After**:

```python
def run_embeddings(
    config: EmbeddingConfig,
    store: Optional[LineageTextStore] = None
) -> None:
    """Core embedding logic, accepts pre-loaded store."""
    if store is None:
        # Standalone mode: load fresh store
        store = LineageTextStore(...)

    # Use store for lineage access
    for lineage_id in store.get_persistent_lineages(config.min_quarters):
        papers = store.get_lineage_papers_from_graph(lineage_id)
        texts = store.extractor.get_texts_batch(papers)
        # Compute embeddings...

def main():
    """CLI entry point."""
    args = parse_args()
    config = EmbeddingConfig(args)
    run_embeddings(config, store=None)  # Standalone mode
```

### Step 2: Update Pipeline Driver

Replace subprocess calls with direct function calls:

**Before** (current):

```python
def run_phase2_embeddings(config, store):
    cmd = [sys.executable, "scripts/compute_lineage_embeddings.py", ...]
    subprocess.run(cmd, check=True)  # Loads registry again
```

**After** (fully integrated):

```python
def run_phase2_embeddings(config, store):
    from scripts.compute_lineage_embeddings import run_embeddings

    embed_config = EmbeddingConfig(
        min_quarters=config.min_quarters,
        device=config.embedding_device,
        output_path=config.output_dir / "lineage_embeddings.npz"
    )

    run_embeddings(embed_config, store=store)  # Reuses shared store!
```

### Step 3: Benefits of Full Refactoring

| Aspect | Current (Option 1.5) | Full Option 2 |
|--------|---------------------|---------------|
| Load registry | 1x (shared store created but not used by subprocesses) | 1x (truly shared) |
| Load raw JSONL | 4x (each subprocess loads independently) | 1x (shared AbstractExtractor) |
| Memory usage | 4x (separate processes) | 1x (single process) |
| Startup overhead | 4x process spawns | None (direct calls) |
| Complexity | Low (minimal changes) | Medium (refactor each script) |

**Estimated speedup:** 20-30% reduction in total runtime for full refactoring.

## Refactoring Checklist

To complete the transition to full Option 2:

- [ ] Refactor `compute_lineage_embeddings.py`
  - [ ] Extract `run_embeddings(config, store)` function
  - [ ] Keep `main()` as CLI wrapper
  - [ ] Add tests for both modes

- [ ] Refactor `compute_lineage_ctfidf.py`
  - [ ] Extract `run_ctfidf(config, store)` function
  - [ ] Update LineageTermExtractor to accept pre-loaded store
  - [ ] Keep CLI compatibility

- [ ] Refactor `compute_lineage_npmi.py`
  - [ ] Extract `run_npmi(config, store)` function
  - [ ] Update LineageNPMIAnalyzer to use store
  - [ ] Keep CLI compatibility

- [ ] Refactor `phase5_ensemble_mapping.py`
  - [ ] Extract `run_ensemble(config)` function
  - [ ] Phase 5 works from CSVs, minimal changes needed

- [ ] Update pipeline driver
  - [ ] Replace subprocess calls with direct function calls
  - [ ] Pass shared store to each phase
  - [ ] Add error handling for phase failures

- [ ] Testing
  - [ ] Verify standalone mode still works for each script
  - [ ] Verify pipeline mode produces identical results
  - [ ] Benchmark speedup from shared resources

## Recommendation

**Current implementation is production-ready** as-is:

- Scripts still work independently
- Pipeline driver provides orchestration
- Shared store infrastructure is in place

**Next step (optional):** Refactor one script at a time to use the shared store directly. Start with Phase 3 (c-TF-IDF) as a proof of concept since it's simpler than Phase 2's GPU embeddings.

## Example: Phase 3 Refactoring (Proof of Concept)

### Current (compute_lineage_ctfidf.py)

```python
def main():
    args = parse_args()

    # Loads registry internally
    extractor = LineageTermExtractor(
        registry_path=args.registry,
        partitions_dir=args.partitions,
        raw_dir=args.raw,
        ...
    )

    # Extract terms...
```

### After Refactoring

```python
@dataclass
class CTFIDFConfig:
    min_quarters: int = 12
    output_path: Path = Path("data/out/02_lineage_tracking/lineage_ctfidf_terms.csv")
    # ... other params


def run_ctfidf(
    config: CTFIDFConfig,
    store: Optional[LineageTextStore] = None
) -> pd.DataFrame:
    """
    Run c-TF-IDF extraction.

    Args:
        config: Configuration parameters
        store: Pre-loaded text store (or None to load fresh)

    Returns:
        DataFrame with lineage terms and scores
    """
    if store is None:
        # Standalone mode
        store = load_or_get_store(
            registry_path=Path("data/out/02_lineage_tracking/lineage_registry.json"),
            raw_dir=Path("data/current_ingest/raw"),
            verbose=True
        )

    # Use shared store instead of loading fresh
    persistent_lineages = store.get_persistent_lineages(config.min_quarters)

    # Extract terms using shared extractor
    results = []
    for lineage_id in persistent_lineages:
        papers = store.get_lineage_papers(lineage_id)
        texts = store.extractor.get_texts_batch(papers)
        # Compute c-TF-IDF...

    df = pd.DataFrame(results)
    df.to_csv(config.output_path, index=False)
    return df


def main():
    """CLI entry point."""
    args = parse_args()

    config = CTFIDFConfig(
        min_quarters=args.min_quarters,
        output_path=Path(args.output)
    )

    # Standalone mode (store=None)
    run_ctfidf(config, store=None)
```

### Updated Pipeline Driver

```python
def run_phase3_ctfidf(config: PipelineConfig, store: LineageTextStore) -> None:
    """Run Phase 3 with shared store."""
    from scripts.compute_lineage_ctfidf import run_ctfidf, CTFIDFConfig

    ctfidf_config = CTFIDFConfig(
        min_quarters=config.min_quarters,
        output_path=config.output_dir / "lineage_ctfidf_terms.csv"
    )

    # Pass shared store → no redundant loading!
    run_ctfidf(ctfidf_config, store=store)
```

## Performance Notes

Current implementation saves:

- ~5-10s: Initial pipeline setup and resource validation
- Better error messages when paths are incorrect

Full refactoring would save additionally:

- ~10-20s: Registry loading across phases
- ~30-60s: Abstract extraction indexing
- ~5-15s: Process spawn overhead

**Total estimated savings: 50-100 seconds** for full pipeline run (20-30% speedup).

Whether this optimization is worth the refactoring effort depends on how frequently you run the full pipeline.

## Multi-Signal Context Integration (2025-11-06)

### Overview

The pipeline now includes field-level and front-level scientometric context metrics to improve breakthrough detection. This enhancement integrates 5 global metrics as additional features for the Multi-Signal Detector (MSD).

### Stage 1.5: Metric Refresh (Optional Pre-Pipeline Step)

**Script**: `scripts/run_metric_refresh.py`

**Purpose**: Generate field-level context metrics from quarterly slices before running Stages 2-5.

**Metrics Computed**:

1. **author_influx**: New vs. returning author rates
2. **citation_velocity**: Citation accumulation rates
3. **reference_vitality**: Proportion of recent references
4. **topic_diversity**: Concept breadth via MeSH/Keywords
5. **cross_cluster_bridging**: Network bridging behavior

**Outputs**: Standardized parquet files in `data/out/metrics/`:

- `global/*.parquet` — Field-level metrics (current implementation)
- `front/*.parquet` — Front-level metrics (future expansion)
- `lineage/*.parquet` — Lineage-level metrics (future expansion)
- `manifest.json` — Central manifest with SHA256 file integrity verification

**Integration with Pipeline**:

```bash
# Option 1: Run metrics separately, then pipeline
python scripts/run_metric_refresh.py --slices-dir data/current_ingest/slices --out-dir data/out/metrics
python scripts/run_build_pipeline.py

# Option 2: Integrated refresh (Task 5.1)
python scripts/run_build_pipeline.py --refresh-metrics
```

When `--refresh-metrics` is enabled, `run_build_pipeline.py` invokes `run_metric_refresh.py` as a pre-step before executing Stages 2-5.

### Stage 2 Enhancement: Context Features

**Script**: `scripts/compute_lineage_multisignal_features.py`

**New CLI Flags** (Task 2.1):

- `--enable-context-features`: Enable field-normalized context features (default: disabled)
- `--metrics-dir`: Path to metrics directory (default: `data/out/metrics`)

**Feature Engineering**:

From each of the 5 global metrics, 7 context features are derived per lineage-quarter:

- **Z-scores** (`metric_z`): Field-normalized standardization
- **QoQ deltas** (`metric_qoq_delta`): Quarter-over-quarter changes
- **Rolling averages** (`metric_roll_1q`, `_2q`, `_4q`): Smoothing over 1, 2, 4 quarters
- **Burst detection** (`metric_max_dev_4q`, `_min_dev_4q`): Max/min deviations over 4Q

**Total Features**:

- **Baseline mode**: 20 core features
- **Enriched mode**: 55 features (20 core + 35 context)

### Configuration Management (Task 5.3)

**Config File**: `config/multisignal_config.yaml`

Centralized configuration for metrics, features, model training, meta-learning, and validation.

**Loader Module**: `src/config_loader.py`

### Validation Suite (Task 5.2)

**Script**: `scripts/test_pipeline_validation.py`

Comprehensive regression tests for schema validation, edge cases, performance, and manifest integrity.

### Documentation

- **README.md**: Multi-signal context integration overview
- **config/multisignal_config.yaml**: Context feature configuration (Task 2.3)
- **config/multisignal_config.yaml**: Configuration reference (Task 5.3)
