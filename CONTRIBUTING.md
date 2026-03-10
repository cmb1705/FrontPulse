# Contributing to FrontPulse

Thank you for your interest in contributing to the FrontPulse research-front monitoring pipeline! This guide will help you get started with development, understand our coding conventions, and successfully contribute to the project.

## Table of Contents

- [Development Setup](#development-setup)
- [Project Architecture](#project-architecture)
- [Code Style Guide](#code-style-guide)
- [Testing Requirements](#testing-requirements)
- [Adding New Features](#adding-new-features)
- [Pull Request Process](#pull-request-process)
- [Documentation](#documentation)
- [Getting Help](#getting-help)

---

## Development Setup

### Prerequisites

- Python 3.10 or higher
- Git
- 16GB+ RAM recommended (8GB minimum)
- Windows, macOS, or Linux

### Initial Setup

1. **Fork and clone the repository:**

   ```bash
   git clone https://github.com/YOUR_USERNAME/FrontPulse.git
   cd FrontPulse
   ```

2. **Create a virtual environment:**

   ```bash
   python -m venv .venv
   ```

3. **Activate the virtual environment:**

   **Windows (PowerShell):**

   ```powershell
   .\.venv\Scripts\activate
   ```

   **macOS/Linux:**

   ```bash
   source .venv/bin/activate
   ```

4. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

   **Note for Windows users:** If `python-igraph` or `leidenalg` installation fails, you may need to use conda:

   ```bash
   conda install -c conda-forge python-igraph leidenalg
   ```

5. **Verify installation:**

   ```bash
   python -c "import igraph, leidenalg; print('Community detection: OK')"
   pytest
   ```

### Performance Configuration

The pipeline defaults to 12 parallel workers optimized for high-performance systems. If you're developing on a system with fewer resources, adjust `DEFAULT_PARALLEL_WORKERS` in `run.py` line 49:

```python
DEFAULT_PARALLEL_WORKERS = 6  # Adjust for your system
```

See [docs/PERFORMANCE_CONFIG.md](docs/PERFORMANCE_CONFIG.md) for detailed recommendations.

---

## Project Architecture

### Directory Structure

```text
.
├─ config/              # YAML configs (datasources, schema, slices)
├─ data/                # Runtime and output directories
│  ├─ current_ingest/   # Cached parquet ingest and slice outputs
│  ├─ current_graphs/   # Graph exports (annual, delta, cumulative)
│  ├─ out/              # Reports, manifests, alerts
│  └─ archive/          # Timestamped snapshots
├─ docs/                # Documentation and guides
├─ scripts/             # Auxiliary utilities and metrics
├─ src/                 # Core pipeline modules
│  ├─ metrics/          # Metric computation utilities
│  ├─ community.py      # Community detection and alignment
│  ├─ graph_build.py    # Citation and coupling graph construction
│  ├─ ingest.py         # Data source orchestration
│  ├─ logging_config.py # Centralized logging framework
│  ├─ memory_utils.py   # Memory monitoring and profiling
│  ├─ slicing.py        # Temporal/categorical partitioning
│  ├─ transform.py      # DataFrame transformations
│  └─ validate.py       # Schema validation and type coercion
├─ tests/               # Automated test suite
└─ run.py               # Main pipeline orchestrator
```

### Pipeline Phases

The main pipeline (`run.py`) consists of 11 modular phases:

1. **Setup** - Directories, logging, configuration
2. **Settings** - Configuration management
3. **Preflight** - Optional validation checks
4. **Ingest** - Data fetching with deduplication
5. **Slicing** - Temporal/categorical partitioning
6. **Graph Building** - Citation and coupling networks
7. **Slices Storage** - Parquet exports with statistics
8. **Manifest Handling** - Continuity tracking
9. **Community Detection** - Leiden clustering
10. **Manifest Writing** - Pipeline summary
11. **Archival** - Timestamped snapshots

Each phase is implemented as a standalone function for testability and maintainability.

---

## Code Style Guide

### Python Style

We follow **PEP 8** with the following conventions:

#### Type Hints

- **Always use type hints** for function parameters and return values
- Use `from __future__ import annotations` for forward references
- Accept `str | Path` for file path parameters (not just `str`)
- Use `Optional[T]` or `T | None` for nullable types

```python
from __future__ import annotations
from pathlib import Path
from typing import Dict, Optional

def process_data(
    input_path: str | Path,
    config: Dict[str, Any],
    max_records: Optional[int] = None
) -> pd.DataFrame:
    """Process input data according to configuration."""
    ...
```

#### Imports

- Standard library imports first
- Third-party imports second
- Local imports last
- Within each group, sort alphabetically
- Use absolute imports from `src/`

```python
from __future__ import annotations
import logging
from pathlib import Path
from typing import Dict, List

import pandas as pd
import yaml

from .logging_config import get_logger
from .validate import enforce_schema
```

#### Naming Conventions

- **Functions and variables**: `snake_case`
- **Classes**: `PascalCase`
- **Constants**: `UPPER_SNAKE_CASE`
- **Private functions**: `_leading_underscore`
- **Type variables**: `PascalCase` (e.g., `T`, `DataFrame`)

#### Docstrings

Use Google-style docstrings for all public functions:

```python
def apply_slices(
    df: pd.DataFrame,
    slices_yaml: str | Path,
    *,
    cutoff: Optional[pd.Timestamp] = None
) -> Dict[str, pd.DataFrame]:
    """
    Apply slicing rules from YAML configuration to create data partitions.

    Args:
        df: Input DataFrame to slice
        slices_yaml: Path to YAML file defining slice specifications
        cutoff: Optional timestamp for query expressions. Defaults to 2 years
            before today.

    Returns:
        Dictionary mapping slice names to DataFrames. For grouped slices,
        names are suffixed with group keys (e.g., "by_quarter__2020Q1").

    Raises:
        ValueError: If slice configuration is invalid

    Example:
        >>> slices = apply_slices(df, "config/slices.yaml")
        >>> slices["by_quarter__2020Q1"]  # DataFrame for 2020 Q1
    """
```

#### Path Handling

- Use `pathlib.Path` exclusively for file operations
- No string concatenation for paths
- Accept both `str` and `Path` in function signatures

```python
# Good
from pathlib import Path

def save_report(output_dir: str | Path, name: str) -> Path:
    output_path = Path(output_dir) / f"{name}.json"
    output_path.write_text(json.dumps(data))
    return output_path

# Bad
def save_report(output_dir: str, name: str) -> str:
    output_path = output_dir + "/" + name + ".json"  # Don't do this!
```

#### Error Handling

- Use specific exception types
- Log errors with appropriate severity
- Provide informative error messages

```python
import logging

logger = logging.getLogger(__name__)

def load_config(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    try:
        return yaml.safe_load(path.read_text())
    except yaml.YAMLError as e:
        logger.error(f"Invalid YAML in {path}: {e}")
        raise ValueError(f"Failed to parse configuration: {e}") from e
```

### Commit Message Style

Follow the [Conventional Commits](https://www.conventionalcommits.org/) format:

```text
<type>: <description> [<issue-id>]

<optional body>

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>
```

**Types:**

- `feat`: New feature or enhancement
- `fix`: Bug fix
- `perf`: Performance improvement
- `refactor`: Code restructuring without behavior change
- `test`: Test additions or modifications
- `docs`: Documentation updates
- `chore`: Maintenance tasks (dependencies, configs)

**Examples:**

```text
feat: Add progress indicators for long operations (LP-1)

Added tqdm progress bars to graph construction, coupling computation,
and edge addition. Progress bars auto-disable for datasets <1000 items.
```

```text
perf: Optimize DataFrame operations with itertuples (PERF-2)

Replaced .iterrows() with .itertuples() in graph_build.py for 2-3x
speedup on large datasets.
```

---

## Testing Requirements

All contributions must include tests. We use **pytest** with a comprehensive test suite.

### Running Tests

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run specific test file
pytest tests/test_transform.py

# Run with coverage report
pytest --cov=src --cov-report=html
```

### Test Markers

- `@pytest.mark.unit` - Fast unit tests (default)
- `@pytest.mark.integration` - Integration tests
- `@pytest.mark.slow` - Tests taking >5 seconds
- `@pytest.mark.requires_api` - Tests requiring OpenAlex API

```bash
# Run only unit tests
pytest -m unit

# Run excluding slow tests
pytest -m "not slow"
```

### Writing Tests

1. **Place tests in `tests/` directory** with `test_` prefix
2. **Use fixtures from `tests/conftest.py`** for sample data
3. **Test both success and failure cases**
4. **Provide descriptive test names**

```python
import pytest
import pandas as pd
from src.validate import enforce_schema

def test_enforce_schema_coerces_types(tmp_path):
    """Test that enforce_schema correctly coerces column types."""
    schema_path = tmp_path / "schema.yaml"
    schema_path.write_text("""
    coerce_types:
      pub_year: int64
      is_oa: boolean
    """)

    df = pd.DataFrame({
        "pub_year": ["2020", "2021"],
        "is_oa": ["true", "false"]
    })

    result = enforce_schema(df, schema_path)

    assert result["pub_year"].dtype == "int64"
    assert result["is_oa"].dtype == "boolean"

def test_enforce_schema_raises_on_null_violation(tmp_path):
    """Test that non-null constraints are enforced."""
    schema_path = tmp_path / "schema.yaml"
    schema_path.write_text("""
    constraints:
      non_null: [work_id]
    """)

    df = pd.DataFrame({"work_id": [None, "W123"]})

    with pytest.raises(ValueError, match="Nulls found in non-null column"):
        enforce_schema(df, schema_path)
```

### Test Coverage Expectations

- **New features**: 80%+ coverage for new code
- **Bug fixes**: Add regression test reproducing the bug
- **Refactoring**: Maintain or improve existing coverage

See `tests/conftest.py` for shared fixtures and test examples.

---

## Adding New Features

### Adding a New Metric to `src/metrics/`

1. **Create a new metric script** in `src/metrics/`:

   ```python
   # src/metrics/metric_example.py
   from __future__ import annotations
   from pathlib import Path
   from typing import Dict, List

   import pandas as pd
   from .common import list_quarter_files, ensure_dir

   def compute_example_metric(
       graphs_dir: Path,
       output_dir: Path
   ) -> Dict[str, float]:
       """
       Compute example metric for quarterly graphs.

       Args:
           graphs_dir: Directory containing quarterly GraphML files
           output_dir: Directory to save metric outputs

       Returns:
           Dictionary mapping quarters to metric values
       """
       ensure_dir(output_dir)
       results = {}

       for quarter, graph_path in list_quarter_files(graphs_dir, "*.graphml"):
           # Load graph and compute metric
           # ...
           results[quarter] = metric_value

       # Save results
       df = pd.DataFrame.from_dict(results, orient="index", columns=["metric_value"])
       df.to_csv(output_dir / "example_metric.csv")

       return results
   ```

2. **Add tests** in `tests/test_metrics.py`

3. **Document usage** in metric docstring

4. **Create a standalone script** in `scripts/` if the metric can be run independently:

   ```python
   # scripts/compute_example_metric.py
   """
   Compute example metric for archived quarterly graphs.

   Usage:
       python scripts/compute_example_metric.py data/archive/20250129_143022/graphs
   """
   from pathlib import Path
   import sys

   from src.metrics.metric_example import compute_example_metric

   if __name__ == "__main__":
       if len(sys.argv) < 2:
           print(__doc__)
           sys.exit(1)

       graphs_dir = Path(sys.argv[1])
       output_dir = graphs_dir.parent / "metrics"

       results = compute_example_metric(graphs_dir, output_dir)
       print(f"Computed metric for {len(results)} quarters")
   ```

### Adding a New Data Source

To add a new data source type (beyond OpenAlex, CSV, Parquet):

1. **Update `src/ingest.py`** `_read_one()` function:

   ```python
   def _read_one(src: Dict[str, Any]) -> Tuple[pd.DataFrame, Optional[List[Dict[str, Any]]]]:
       kind = src.get("kind", "csv").lower()

       # ... existing code ...

       elif kind == "your_source":
           # Implement data loading logic
           df = your_loader_function(src["param1"], src["param2"])
           return df, None
       else:
           raise ValueError(f"Unsupported kind: {kind}")
   ```

2. **Update `config/datasources.yaml`** with example configuration:

   ```yaml
   sources:
     primary:
       kind: your_source
       param1: value1
       param2: value2
   ```

3. **Add tests** to `tests/test_ingest.py`

4. **Document** the new source type in README.md

### Adding a New Pipeline Phase

To add a new phase to the main pipeline:

1. **Define the phase function** in `run.py`:

   ```python
   def phase_your_feature(
       config: PipelineConfig,
       state: PipelineState
   ) -> None:
       """
       Your feature description.

       Args:
           config: Pipeline configuration
           state: Mutable pipeline state
       """
       logger.info("Running your feature phase")
       # Implementation
   ```

2. **Register the phase** in the `PHASES` list in `run()`:

   ```python
   PHASES = [
       # ... existing phases ...
       ("Your Feature", phase_your_feature),
   ]
   ```

3. **Add CLI arguments** if needed in `parse_args()`

4. **Update documentation** in README.md

---

## Pull Request Process

### Before Submitting

1. **Run tests** and ensure they pass:

   ```bash
   pytest
   ```

2. **Check code formatting** (follow PEP 8):

   ```bash
   # Optional: use black for auto-formatting
   black src/ tests/
   ```

3. **Update documentation** if you've:
   - Added new features
   - Changed CLI arguments
   - Modified configuration files
   - Added new dependencies

4. **Commit your changes** with descriptive commit messages

### Submitting the Pull Request

1. **Push to your fork:**

   ```bash
   git push origin your-feature-branch
   ```

2. **Create a pull request** on GitHub with:
   - Clear title describing the change
   - Reference to related issues (e.g., "Closes #42")
   - Description of changes made
   - Test results (pytest output)
   - Screenshots if UI/output changed

3. **PR Description Template:**

   ```markdown
   ## Description
   Brief description of what this PR does.

   ## Related Issues
   Closes #42
   Related to #38

   ## Changes Made
   - Added feature X
   - Fixed bug Y
   - Updated documentation for Z

   ## Test Results
   ```

   pytest output here

   ```

   ## Checklist
   - [ ] Tests pass locally
   - [ ] Added tests for new features
   - [ ] Updated documentation
   - [ ] Followed code style guide
   ```

### Review Process

- Maintainers will review your PR within 3-5 business days
- Address review feedback by pushing new commits
- Once approved, a maintainer will merge your PR
- Your contribution will be acknowledged in release notes

---

## Documentation

### Updating Documentation

When contributing, update relevant documentation:

- **README.md** - User-facing features, CLI usage, quickstart
- **docs/** - Technical guides, performance tuning, architecture
- **Docstrings** - All public functions and classes
- **tests/conftest.py** - Testing fixtures and examples

### Documentation Style

- Use **Markdown** for all documentation
- Include **code examples** where appropriate
- Keep explanations **clear and concise**
- Use **relative links** for internal references: `[Performance Guide](docs/PERFORMANCE_CONFIG.md)`

---

## Getting Help

### Resources

- **README.md** - Project overview and quickstart
- **tests/conftest.py** - Testing fixtures
- **docs/PERFORMANCE_CONFIG.md** - Performance tuning

### Communication

- **GitHub Issues** - Bug reports, feature requests
- **GitHub Discussions** - Questions, ideas, general discussion
- **Pull Requests** - Code review and feedback

### Reporting Bugs

When reporting bugs, include:

1. **Python version** (`python --version`)
2. **Operating system**
3. **Steps to reproduce**
4. **Expected vs actual behavior**
5. **Relevant logs** (with `--log-level DEBUG`)
6. **Configuration files** used (with sensitive data redacted)

---

## Thank You

Your contributions help make FrontPulse better for everyone. Whether you're fixing a typo, adding a feature, or improving documentation, we appreciate your time and effort.

Happy coding!
