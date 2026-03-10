# Artifact Persistence Policy

This document defines the serialization standards for FrontPulse model and
analysis artifacts.  All new code must follow these conventions; existing code
is being migrated incrementally.

## Trust Boundary Model

Artifacts fall into two categories based on their deserialization risk:

### Trusted Artifacts

Binary Python serialization can execute arbitrary code on load.  We mitigate
this by:

1. **Repo-local sandbox** -- `load_trusted_pickle` and `save_trusted_pickle`
   in `src/trusted_io.py` enforce that artifact paths resolve under the
   repository root by default.
2. **Explicit opt-in for external paths** -- callers must pass
   `allow_external=True` to load artifacts from outside the repo.
3. **Never share trusted artifacts externally** -- binary model files are
   excluded from portable exports and should not be committed to shared
   artifact stores without an integrity check.

Trusted artifacts include:

| Artifact | Produced by | Consumed by | Path pattern |
|----------|-------------|-------------|--------------|
| Citation graphs | `src/graph_build.py` | `scripts/communities.py`, `src/lineage_text_store.py` | `data/current_graphs/citation_graph_*.pkl` |
| MSD trained model | `scripts/multi_signal_detector.py` | `scripts/field_level_detector.py`, `scripts/generate_publication_figures.py` | `data/out/breakthrough_detector_model.pkl` |
| Abstract index cache | `scripts/extract_abstracts.py` | `scripts/compute_lineage_multisignal_features.py` | `data/out/cache_lineage/abstract_index_*.pkl` |
| Reference data cache | `scripts/compute_lineage_multisignal_features.py` | same script (speed optimization) | `data/out/cache_lineage/reference_cache_*.pkl` |

### Portable Artifacts (CSV, JSON, Parquet, NumPy)

These formats carry no code-execution risk on load and can be shared freely:

| Format | Use case | Examples |
|--------|----------|---------|
| CSV | Human-readable tabular outputs | `lineage_timeseries.csv`, `breakthrough_predictions.csv` |
| JSON | Configuration, metrics, metadata | `evaluation_metrics.json`, `feature_names.json`, `lineage_registry.json` |
| Parquet | Large tabular data (efficient) | `ingest.parquet`, `field_metrics.parquet` |
| NumPy `.npz` | Dense arrays (with `allow_pickle=False`) | `lineage_embeddings.npz`, `*.lite.npz` |

## API Reference

All trusted IO goes through `src/trusted_io.py`:

```python
from src.trusted_io import (
    load_trusted_pickle,       # Load with repo-local sandbox
    save_trusted_pickle,       # Save with repo-local sandbox
    require_repo_local_path,   # Path validation only
    repo_path,                 # Build repo-rooted paths
    REPO_ROOT,                 # Absolute path to repo root
)
```

### Saving a trusted artifact

```python
from src.trusted_io import save_trusted_pickle

save_trusted_pickle(
    model_pipeline,
    output_dir / "model.pkl",
    description="MSD trained model",
)
```

Parent directories are created automatically.  The function returns the
resolved absolute path.

### Loading a trusted artifact

```python
from src.trusted_io import load_trusted_pickle

model = load_trusted_pickle(
    output_dir / "model.pkl",
    description="MSD trained model",
)
```

To load from outside the repository (rare, requires justification):

```python
model = load_trusted_pickle(
    external_path,
    description="external model checkpoint",
    allow_external=True,
)
```

## Rules for New Code

1. **Never use bare serialization calls** in `src/` modules.
   Always route through `save_trusted_pickle` / `load_trusted_pickle`.

2. **Scripts may use bare calls only for ephemeral caches** that are
   regenerated on every run and never shared.  Prefer the trusted IO
   functions even for caches.

3. **NumPy loads must use `allow_pickle=False`** unless loading a trusted
   artifact-containing `.npz` file.

4. **Portable exports** (CSV, JSON, Parquet) do not require sandbox checks
   but should use `pathlib.Path` for all path construction.

5. **Model artifacts are trusted-only**.  If a model must be shared, export
   its predictions or coefficients as CSV/JSON instead of the binary model.

## Migration Status

| Module | Status | Notes |
|--------|--------|-------|
| `src/trusted_io.py` | Complete | `save_trusted_pickle` added |
| `src/graph_build.py` | Migrated | Uses `save_trusted_pickle` |
| `src/lineage_text_store.py` | Migrated | Uses `load_trusted_pickle` |
| `scripts/multi_signal_detector.py` | Migrated | Uses `save_trusted_pickle` |
| `scripts/communities.py` | Already compliant | No change needed |
| `scripts/build_lite_graphs.py` | Already compliant | No change needed |
| `scripts/compute_lineage_multisignal_features.py` | Pending | Cache artifact, low priority |
| `scripts/extract_abstracts.py` | Pending | Cache artifact, low priority |
| `scripts/metric_cross_cluster_bridging.py` | Pending | Graph load, medium priority |
