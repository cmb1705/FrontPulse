# Pipeline API Examples

> **Note**: These examples use legacy shared paths (`data/out/`, `data/current_ingest/`).
> For domain-isolated runs, use `--domain psc` or `--domain crispr` to auto-resolve
> paths under `data/{domain}/`. See `src/domain_registry.py` for details.

This directory contains example scripts demonstrating how to use the `Pipeline` class for programmatic execution of the FrontPulse research front monitoring pipeline.

## Quick Start

```python
from src.pipeline import Pipeline

# Initialize pipeline
pipeline = Pipeline(
    config_path="config/datasources.yaml",
    schema_path="config/schema.yaml",
    slices_path="config/slices.yaml",
    outdir="data/out",
    ingest_dir="data/current_ingest",
    graphs_dir="data/current_graphs"
)

# Run full pipeline
results = pipeline.run()
```

## Examples

### 1. Full Pipeline (`pipeline_usage.py`)

Comprehensive examples demonstrating:

1. **Full Pipeline Execution**: Run all phases from ingest to community detection
2. **Individual Phase Control**: Execute phases separately for fine-grained control
3. **Bibliographic Coupling**: Enable and configure coupling for enhanced graph quality
4. **Notebook Workflow**: Lightweight workflow suitable for Jupyter notebooks

Run examples:

```bash
# Show available examples
python examples/pipeline_usage.py

# Run specific example
python examples/pipeline_usage.py 1   # Full pipeline
python examples/pipeline_usage.py 2   # Individual phases
python examples/pipeline_usage.py 3   # With coupling
python examples/pipeline_usage.py 4   # Notebook workflow
```

## Pipeline API Reference

### Class: `Pipeline`

**Constructor Parameters:**

- `config_path`: Path to datasources YAML config
- `schema_path`: Path to schema YAML config
- `slices_path`: Path to slices YAML config
- `outdir`: Output directory for reports and manifests
- `ingest_dir`: Directory for cached ingest data
- `graphs_dir`: Directory for graph exports
- `raw_dir`: Optional directory for raw NDJSON snapshots
- `graph_mode`: Graph building mode (`'annual'`, `'delta'`, `'cumulative'`, `'both'`)
- `coupling_config`: Optional `CouplingConfig` for bibliographic coupling
- `mailto`: Email for OpenAlex API (helps with rate limits)
- `log_level`: Logging level (`'DEBUG'`, `'INFO'`, `'WARNING'`, `'ERROR'`)

**Methods:**

#### `ingest(skip_cache=False, rebuild_from_raw=False, raw_manifest_path=None) -> pd.DataFrame`

Execute ingest phase to fetch or load works data.

```python
# Fetch from API
df = pipeline.ingest()

# Use cached data
df = pipeline.ingest(skip_cache=True)

# Rebuild from raw snapshot
df = pipeline.ingest(
    rebuild_from_raw=True,
    raw_manifest_path=Path("data/current_ingest/raw/manifest.json")
)
```

#### `slice(df=None) -> Dict[str, pd.DataFrame]`

Apply temporal/categorical slicing to DataFrame.

```python
slices = pipeline.slice()
# Returns: {'by_quarter__2020Q1': DataFrame, ...}
```

#### `build_graphs(df=None, mode=None) -> Dict[str, List[pathlib.Path]]`

Build citation graphs in specified mode.

```python
# Build cumulative graphs
graphs = pipeline.build_graphs(mode='cumulative')

# Build annual graphs
graphs = pipeline.build_graphs(mode='annual')

# Build both annual and delta
graphs = pipeline.build_graphs(mode='both')
```

#### `detect_communities(mode=None, resume=False) -> Dict[str, Any]`

Run community detection on built graphs.

```python
# Detect communities in cumulative mode
communities = pipeline.detect_communities(mode='cumulative')

# Resume from cache (cumulative only)
communities = pipeline.detect_communities(mode='cumulative', resume=True)
```

#### `run(skip_ingest=False, run_communities=True) -> PipelineResults`

Execute full pipeline from ingest to community detection.

```python
results = pipeline.run()

# Access results
print(f"Records: {len(results.df)}")
print(f"Slices: {len(results.slices)}")
print(f"Graphs: {results.graphs}")
print(f"Communities: {results.communities}")
print(f"Errors: {results.errors}")
```

## Jupyter Notebook Usage

The Pipeline API is designed for interactive use in Jupyter notebooks:

```python
# In a Jupyter cell
from src.pipeline import Pipeline

pipeline = Pipeline(
    config_path="config/datasources.yaml",
    schema_path="config/schema.yaml",
    slices_path="config/slices.yaml",
    outdir="data/out",
    ingest_dir="data/current_ingest",
    graphs_dir="data/current_graphs",
    log_level="WARNING"  # Less verbose for notebooks
)

# Load cached data
df = pipeline.ingest(skip_cache=True)

# Explore
df.head()
df['pub_qtr'].value_counts()

# Build specific graphs
from src.graph_build import build_direct_citation_graph
latest_quarter = df['pub_qtr'].max()
df_recent = df[df['pub_qtr'] == latest_quarter]
G = build_direct_citation_graph(df_recent)

# Analyze with NetworkX
import networkx as nx
print(f"Nodes: {G.number_of_nodes()}")
print(f"Edges: {G.number_of_edges()}")
print(f"Density: {nx.density(G):.4f}")
```

## Configuration Examples

### Coupling Configuration

```python
from src.graph_build import CouplingConfig
from pathlib import Path

coupling_config = CouplingConfig(
    enabled=True,
    alpha=1.0,              # Citation weight
    beta=0.3,               # Coupling base weight
    lambda_decay=0.15,      # Temporal decay factor
    min_shared_refs=5,      # Minimum shared references
    min_coupling_score=0.05,# Minimum coupling score
    cache_dir=Path("data/out/cache_coupling"),
    workers=4               # Parallel workers
)

pipeline = Pipeline(
    # ... other args ...
    coupling_config=coupling_config
)
```

### Custom Logging

```python
# Debug level for troubleshooting
pipeline = Pipeline(
    # ... other args ...
    log_level="DEBUG"
)

# Warning level for production
pipeline = Pipeline(
    # ... other args ...
    log_level="WARNING"
)
```

## Error Handling

```python
try:
    results = pipeline.run()
except ValueError as e:
    print(f"Configuration error: {e}")
except FileNotFoundError as e:
    print(f"Missing file: {e}")
except Exception as e:
    print(f"Pipeline failed: {e}")
    # Check results.errors for details
    print(f"Errors encountered: {results.errors}")
```

## Testing

The Pipeline class is designed to be testable:

```python
import pytest
from src.pipeline import Pipeline

def test_pipeline_initialization():
    pipeline = Pipeline(
        config_path="config/datasources.yaml",
        schema_path="config/schema.yaml",
        slices_path="config/slices.yaml",
        outdir="data/out",
        ingest_dir="data/current_ingest",
        graphs_dir="data/current_graphs"
    )
    assert pipeline.config.graph_mode == "cumulative"

def test_pipeline_ingest_cached():
    pipeline = Pipeline(
        # ... config ...
    )
    df = pipeline.ingest(skip_cache=True)
    assert len(df) > 0
    assert 'work_id' in df.columns
```

## Additional Resources

- **Main Documentation**: See `README.md` in the repository root
- **API Documentation**: See docstrings in `src/pipeline.py`
- **CLI Usage**: See `agents.md` for command-line workflows
- **Architecture**: See `ROADMAP.md` for design rationale (ARCH-1)
