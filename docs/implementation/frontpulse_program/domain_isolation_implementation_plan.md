# Implementation Plan: Domain-Isolated Data Directories

**Task:** FP-0xc
**Date:** 2026-03-25
**Prerequisites:** Design spec (FP-gqm), codebase audit (FP-i6q), path trace (FP-r8u)

This plan is executable by another programmer without ambiguity. Each step
specifies what changes, why, and the before/after for key code sections.

---

## Build Sequence (Dependency Order)

```
Step 1: src/domain_registry.py (foundation -- everything depends on this)
    |
Step 2: tests/test_domain_data_paths.py (test new APIs)
    |
Step 3: run.py (pipeline entry point)
    |
Step 4: scripts/communities.py (called by run.py as subprocess)
    |
Step 5: src/settings.py (per-domain settings)
    |
Step 6: scripts/migrate_domain_layout.py (one-time migration tool)
    |
    |--- [RUN MIGRATION HERE: move existing PSC and CRISPR data] ---
    |
Step 7: Core feature scripts (11 files, parallel-safe)
    |
Step 8: Analysis/training scripts (9 files, parallel-safe)
    |
Step 9: Horizon scanner scripts (4 files, parallel-safe)
    |
Step 10: Config files (2 files)
    |
Step 11: Test updates (1 file)
    |
Step 12: Backward-compat symlinks + cleanup
```

---

## Step 1: src/domain_registry.py

**Why:** Foundation. Every script will call `resolve_script_paths()` from here.

### 1a. Add DomainDataPaths dataclass

Insert after the existing `DomainConfig` class (after line 55):

```python
@dataclass(frozen=True)
class DomainDataPaths:
    """All data directory paths for a single research domain.

    Derived from convention: data/{domain_id}/{ingest,graphs,out,archive}/.
    Frozen to prevent accidental mutation after resolution.
    """

    domain_id: str
    base: Path
    ingest: Path
    raw: Path
    slices: Path
    graphs: Path
    out: Path
    lineage_tracking: Path
    front_aggregation: Path
    cache_cum: Path
    cache_lineage: Path
    cache_coupling: Path
    assessments: Path
    experiments: Path
    archive: Path

    def ensure_dirs(self) -> None:
        """Create all directories in the tree if they don't exist."""
        for field_name in self.__dataclass_fields__:
            val = getattr(self, field_name)
            if isinstance(val, Path):
                val.mkdir(parents=True, exist_ok=True)

    def validate_paths(self) -> list[str]:
        """Check that all paths are under the domain base directory."""
        warnings = []
        for field_name in self.__dataclass_fields__:
            val = getattr(self, field_name)
            if isinstance(val, Path):
                try:
                    val.relative_to(self.base)
                except ValueError:
                    warnings.append(
                        f"{field_name}={val} is outside domain base {self.base}"
                    )
        return warnings

    def to_dict(self) -> dict[str, str]:
        """Serialize to string dict for manifest storage."""
        return {
            k: str(v) for k, v in self.__dict__.items()
            if k != "domain_id"
        }
```

### 1b. Add resolve_data_paths() to DomainConfig

Add method to `DomainConfig` class:

```python
def resolve_data_paths(self, project_root: Path) -> DomainDataPaths:
    """Derive all data directory paths from domain_id convention."""
    base = project_root / "data" / self.domain_id
    return DomainDataPaths(
        domain_id=self.domain_id,
        base=base,
        ingest=base / "ingest",
        raw=base / "ingest" / "raw",
        slices=base / "ingest" / "slices",
        graphs=base / "graphs",
        out=base / "out",
        lineage_tracking=base / "out" / "02_lineage_tracking",
        front_aggregation=base / "out" / "04_front_aggregation",
        cache_cum=base / "out" / "cache_cum",
        cache_lineage=base / "out" / "cache_lineage",
        cache_coupling=base / "out" / "cache_coupling",
        assessments=base / "out" / "assessments",
        experiments=base / "out" / "experiments",
        archive=base / "archive",
    )
```

### 1c. Add script integration helpers

Add after `resolve_domain_args()`:

```python
def add_domain_args(parser: argparse.ArgumentParser) -> None:
    """Add --domain argument to any script's parser."""
    parser.add_argument(
        "--domain", default=None,
        choices=sorted(DOMAIN_REGISTRY),
        help="Research domain (derives all data paths from convention)",
    )


def resolve_script_paths(
    args: argparse.Namespace,
    project_root: Path,
) -> DomainDataPaths | None:
    """Resolve paths for scripts that accept --domain.

    Returns DomainDataPaths if --domain is set, None otherwise.
    Scripts fall back to their explicit CLI args when None is returned.
    """
    domain_id = getattr(args, "domain", None)
    if domain_id is None:
        return None
    domain = get_domain(domain_id)
    return domain.resolve_data_paths(project_root)


def resolve_pipeline_paths(
    domain_id: str | None,
    cli_ingest_dir: str | None,
    cli_graphs_dir: str | None,
    cli_outdir: str | None,
    cli_coupling_cache: str | None,
    project_root: Path,
) -> DomainDataPaths | None:
    """Resolve data paths for run.py with CLI overrides.

    If domain_id is set, derives paths from convention. Individual CLI
    arguments override specific directories when provided (not None).

    Returns None if domain_id is None (caller must use explicit paths).
    """
    if domain_id is None:
        return None

    domain = get_domain(domain_id)
    paths = domain.resolve_data_paths(project_root)

    # Apply CLI overrides by constructing a new frozen instance
    overrides: dict[str, Path] = {}
    if cli_ingest_dir is not None:
        ingest = Path(cli_ingest_dir)
        overrides["ingest"] = ingest
        overrides["raw"] = ingest / "raw"
        overrides["slices"] = ingest / "slices"
    if cli_graphs_dir is not None:
        overrides["graphs"] = Path(cli_graphs_dir)
    if cli_outdir is not None:
        out = Path(cli_outdir)
        overrides["out"] = out
        overrides["lineage_tracking"] = out / "02_lineage_tracking"
        overrides["front_aggregation"] = out / "04_front_aggregation"
        overrides["cache_cum"] = out / "cache_cum"
        overrides["cache_lineage"] = out / "cache_lineage"
        overrides["cache_coupling"] = out / "cache_coupling"
        overrides["assessments"] = out / "assessments"
        overrides["experiments"] = out / "experiments"
    if cli_coupling_cache is not None:
        overrides["cache_coupling"] = Path(cli_coupling_cache)

    if overrides:
        d = {k: getattr(paths, k) for k in paths.__dataclass_fields__}
        d.update(overrides)
        return DomainDataPaths(**d)

    return paths
```

### 1d. Remove outdir_suffix from DomainConfig

Before:
```python
outdir_suffix: str = ""
```

After: Remove the field entirely. Remove `outdir_suffix="_crispr"` from the
CRISPR registry entry.

### 1e. Add import

Add `import argparse` to the imports at the top of the file.

---

## Step 2: tests/test_domain_data_paths.py

**Why:** Validate the new APIs before any script changes depend on them.

Create new file `tests/test_domain_data_paths.py`:

```python
"""Tests for domain-isolated data path resolution."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.domain_registry import (
    DOMAIN_REGISTRY,
    DomainDataPaths,
    get_domain,
    resolve_pipeline_paths,
    resolve_script_paths,
)


class TestDomainDataPaths:
    """Tests for DomainDataPaths dataclass."""

    def test_resolve_data_paths_convention(self) -> None:
        domain = get_domain("psc")
        root = Path("/project")
        paths = domain.resolve_data_paths(root)
        assert paths.ingest == root / "data" / "psc" / "ingest"
        assert paths.graphs == root / "data" / "psc" / "graphs"
        assert paths.out == root / "data" / "psc" / "out"

    def test_all_paths_under_base(self) -> None:
        domain = get_domain("psc")
        root = Path("/project")
        paths = domain.resolve_data_paths(root)
        warnings = paths.validate_paths()
        assert len(warnings) == 0

    def test_ensure_dirs_creates_tree(self, tmp_path: Path) -> None:
        domain = get_domain("crispr")
        paths = domain.resolve_data_paths(tmp_path)
        paths.ensure_dirs()
        assert paths.ingest.exists()
        assert paths.graphs.exists()
        assert paths.assessments.exists()

    def test_frozen_immutable(self) -> None:
        domain = get_domain("psc")
        paths = domain.resolve_data_paths(Path("/project"))
        with pytest.raises(AttributeError):
            paths.ingest = Path("/other")  # type: ignore[misc]

    def test_to_dict_serialization(self) -> None:
        domain = get_domain("psc")
        paths = domain.resolve_data_paths(Path("/project"))
        d = paths.to_dict()
        assert isinstance(d, dict)
        assert "ingest" in d
        assert isinstance(d["ingest"], str)

    def test_crispr_uses_crispr_subdirectory(self) -> None:
        domain = get_domain("crispr")
        root = Path("/project")
        paths = domain.resolve_data_paths(root)
        assert "crispr" in str(paths.base)
        assert paths.lineage_tracking == root / "data" / "crispr" / "out" / "02_lineage_tracking"


class TestResolvePipelinePaths:
    """Tests for run.py path resolution helper."""

    def test_domain_only_derives_all_paths(self) -> None:
        root = Path("/project")
        paths = resolve_pipeline_paths("psc", None, None, None, None, root)
        assert paths is not None
        assert paths.ingest == root / "data" / "psc" / "ingest"

    def test_cli_ingest_override(self) -> None:
        root = Path("/project")
        paths = resolve_pipeline_paths("psc", "/custom/ingest", None, None, None, root)
        assert paths is not None
        assert paths.ingest == Path("/custom/ingest")
        assert paths.slices == Path("/custom/ingest/slices")
        # Non-overridden paths stay at convention
        assert paths.graphs == root / "data" / "psc" / "graphs"

    def test_cli_outdir_override(self) -> None:
        root = Path("/project")
        paths = resolve_pipeline_paths("psc", None, None, "/custom/out", None, root)
        assert paths is not None
        assert paths.out == Path("/custom/out")
        assert paths.assessments == Path("/custom/out/assessments")

    def test_no_domain_returns_none(self) -> None:
        paths = resolve_pipeline_paths(None, None, None, None, None, Path("/project"))
        assert paths is None

    def test_invalid_domain_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown domain"):
            resolve_pipeline_paths("nonexistent", None, None, None, None, Path("/p"))


class TestResolveScriptPaths:
    """Tests for per-script domain resolution."""

    def test_with_domain(self) -> None:
        from argparse import Namespace
        args = Namespace(domain="psc")
        root = Path("/project")
        paths = resolve_script_paths(args, root)
        assert paths is not None
        assert paths.domain_id == "psc"

    def test_without_domain(self) -> None:
        from argparse import Namespace
        args = Namespace(domain=None)
        paths = resolve_script_paths(args, Path("/project"))
        assert paths is None

    def test_missing_domain_attr(self) -> None:
        from argparse import Namespace
        args = Namespace()
        paths = resolve_script_paths(args, Path("/project"))
        assert paths is None
```

---

## Step 3: run.py

**Why:** Pipeline entry point; all runs flow through here.

### 3a. Change argparse defaults

Before (lines 396-404):
```python
ap.add_argument("--domain", default=None, choices=["psc", "crispr"],
    help="...")
ap.add_argument("--config", default=None, help="...")
ap.add_argument("--outdir", required=True, help="output directory")
ap.add_argument("--ingest-dir", default="data/current_ingest", help="...")
ap.add_argument("--graphs-dir", default="data/current_graphs", help="...")
```

After:
```python
ap.add_argument("--domain", default=None,
    choices=sorted(DOMAIN_REGISTRY),
    help="Research domain (derives all data paths from convention)")
ap.add_argument("--config", default=None,
    help="config/datasources.yaml (or use --domain)")
ap.add_argument("--outdir", default=None,
    help="Output directory (derived from --domain if not set)")
ap.add_argument("--ingest-dir", default=None,
    help="Ingest directory (derived from --domain if not set)")
ap.add_argument("--graphs-dir", default=None,
    help="Graphs directory (derived from --domain if not set)")
```

Note: Requires adding `from src.domain_registry import DOMAIN_REGISTRY` at top.

### 3b. Change main() domain resolution

Before (lines 1456-1465):
```python
if args.domain is not None:
    from src.domain_registry import resolve_domain_args
    args.config = resolve_domain_args(
        args.domain, args.config,
        project_root=pathlib.Path(__file__).resolve().parent,
    )
elif args.config is None:
    print("Error: either --domain or --config must be specified.")
    sys.exit(1)

outdir = pathlib.Path(args.outdir)
```

After:
```python
project_root = pathlib.Path(__file__).resolve().parent

if args.domain is not None:
    from src.domain_registry import resolve_domain_args, resolve_pipeline_paths
    args.config = resolve_domain_args(
        args.domain, args.config, project_root=project_root,
    )
    domain_paths = resolve_pipeline_paths(
        args.domain,
        args.ingest_dir, args.graphs_dir, args.outdir,
        getattr(args, "coupling_cache_dir", None),
        project_root,
    )
    # Apply domain-derived paths where CLI args were not explicitly set
    if args.ingest_dir is None:
        args.ingest_dir = str(domain_paths.ingest)
    if args.graphs_dir is None:
        args.graphs_dir = str(domain_paths.graphs)
    if args.outdir is None:
        args.outdir = str(domain_paths.out)
    if not hasattr(args, "coupling_cache_dir") or args.coupling_cache_dir is None:
        args.coupling_cache_dir = domain_paths.cache_coupling
elif args.config is None:
    print("Error: either --domain or --config must be specified.")
    sys.exit(1)

# Validate that required paths are set
if args.outdir is None:
    print("Error: --outdir is required when --domain is not set.")
    sys.exit(1)
if args.ingest_dir is None:
    args.ingest_dir = "data/current_ingest"  # legacy fallback
if args.graphs_dir is None:
    args.graphs_dir = "data/current_graphs"  # legacy fallback

outdir = pathlib.Path(args.outdir)
```

### 3c. Update coupling_cache_dir default

Before (line 461):
```python
ap.add_argument("--coupling-cache-dir", type=pathlib.Path,
    default=pathlib.Path("data/out/cache_coupling"), ...)
```

After:
```python
ap.add_argument("--coupling-cache-dir", type=pathlib.Path,
    default=None,
    help="Coupling cache dir (derived from --domain if not set)")
```

And in main(), after domain resolution, add fallback:
```python
if args.coupling_cache_dir is None:
    args.coupling_cache_dir = pathlib.Path(args.outdir) / "cache_coupling"
```

---

## Step 4: scripts/communities.py

**Why:** Called by run.py as subprocess; receives paths via CLI.

### 4a. Add domain args

Before (line ~1417):
```python
ap.add_argument("--graphs-dir", type=Path, default=Path("data/current_graphs"))
```

After:
```python
from src.domain_registry import add_domain_args, resolve_script_paths
add_domain_args(ap)
ap.add_argument("--graphs-dir", type=Path, default=None)
```

### 4b. Resolve in main

After parsing args, add domain fallback:
```python
paths = resolve_script_paths(args, REPO)
if args.graphs_dir is None:
    args.graphs_dir = paths.graphs if paths else Path("data/current_graphs")
if args.out_dir is None:
    args.out_dir = paths.out if paths else Path("data/out")
```

---

## Step 5: src/settings.py

**Why:** Watermark collision between domains.

### 5a. Per-domain settings path

Add parameter to `load_settings()` and `save_settings()`:

Before:
```python
def load_settings() -> dict:
    ...  # looks for config/settings.yaml then .2yp_settings.json
```

After:
```python
def load_settings(domain_settings_path: Path | None = None) -> dict:
    """Load settings, optionally from a domain-specific path."""
    if domain_settings_path and domain_settings_path.exists():
        return yaml.safe_load(domain_settings_path.read_text()) or {}
    ...  # existing logic as fallback
```

And in `save_settings()`:
```python
def save_settings(settings: dict, domain_settings_path: Path | None = None) -> None:
    if domain_settings_path:
        domain_settings_path.parent.mkdir(parents=True, exist_ok=True)
        domain_settings_path.write_text(yaml.dump(settings, default_flow_style=False))
        return
    ...  # existing .2yp_settings.json logic
```

### 5b. Wire into run.py

In `main()`, after domain path resolution:
```python
domain_settings = None
if args.domain is not None:
    domain_settings = domain_paths.base / "settings.yaml"
settings = load_settings(domain_settings_path=domain_settings)
...
save_settings(settings, domain_settings_path=domain_settings)
```

---

## Step 6: scripts/migrate_domain_layout.py

**Why:** Move existing data into domain-isolated directories.

Create new file. Key logic:

```python
"""One-time migration: move existing data into domain-isolated directories.

Usage:
    python scripts/migrate_domain_layout.py --dry-run
    python scripts/migrate_domain_layout.py
    python scripts/migrate_domain_layout.py --domain psc
"""

MOVES = {
    "psc": [
        ("data/current_ingest", "data/psc/ingest"),
        ("data/current_graphs", "data/psc/graphs"),
        ("data/out", "data/psc/out"),
    ],
    "crispr": [
        ("data/out_crispr", "data/crispr/out"),
        # data/crispr/ingest and data/crispr/graphs are empty
        # (CRISPR raw data needs re-ingestion into isolated tree)
    ],
}
```

Features:
- `--dry-run`: print planned moves without executing
- Verify no destination conflicts
- Use `shutil.move()` (same-filesystem = instant rename)
- Write `data/.migration_log.json` with timestamp and file counts
- Optionally create backward-compat symlinks (junction points on Windows)

---

## Step 7: Core Feature Scripts (11 files)

**Pattern for each script:** Add `--domain`, derive paths with fallback.

### Template (apply to each script)

```python
# At top, add imports:
from src.domain_registry import add_domain_args, resolve_script_paths

# In argparse setup:
add_domain_args(ap)

# Change defaults from hardcoded to None:
ap.add_argument("--registry", default=None)  # was: "data/out/02_.../lineage_registry.json"
ap.add_argument("--slices-dir", default=None)  # was: "data/current_ingest/slices"

# After parse_args():
paths = resolve_script_paths(args, REPO)

# Resolve each path with domain fallback:
registry = Path(args.registry) if args.registry else (
    paths.lineage_tracking / "lineage_registry.json"
    if paths else Path("data/out/02_lineage_tracking/lineage_registry.json")
)
```

### Per-script path mappings

**C.1 compute_lineage_multisignal_features.py:**

| Arg | Domain Path | Legacy Default |
|-----|-------------|----------------|
| `--registry` | `paths.lineage_tracking / "lineage_registry.json"` | `data/out/02_lineage_tracking/lineage_registry.json` |
| `--timeseries` | `paths.lineage_tracking / "lineage_timeseries.csv"` | `data/out/02_lineage_tracking/lineage_timeseries.csv` |
| `--raw-dir` | `paths.raw` | `data/current_ingest/raw` |
| `--partitions-dir` | `paths.cache_cum / "partitions_cum"` | `data/out/cache_cum/partitions_cum` |
| `--reference-cache` | `paths.cache_lineage / "reference_data.pkl"` | `data/out/cache_lineage/reference_data.pkl` |
| `--metrics-dir` | `paths.out / "metrics"` | `data/out/metrics` |
| `--field-metrics` | `paths.front_aggregation / "field_metrics.parquet"` | `data/out/04_front_aggregation/field_metrics.parquet` |
| `--out` | `paths.lineage_tracking / "lineage_multisignal_features.csv"` | `data/out/02_lineage_tracking/lineage_multisignal_features.csv` |

**C.2 compute_convergence_features.py:**

| Arg | Domain Path |
|-----|-------------|
| `--registry` | `paths.lineage_tracking / "lineage_registry.json"` |
| `--timeseries` | `paths.lineage_tracking / "lineage_timeseries.csv"` |
| `--quarterly-embeddings` | `paths.experiments / "stage1_quarterly_embeddings/quarterly_embeddings.npz"` |
| `--partitions-dir` | `paths.cache_cum / "partitions_cum"` |
| `--slices-dir` | `paths.slices` |
| `--out` | `paths.lineage_tracking / "convergence_features.csv"` |

**C.3-C.11:** Follow the same pattern. The audit doc (FP-i6q) lists every arg
per script. Map each to the corresponding `DomainDataPaths` attribute.

### Special case: metric_cross_cluster_bridging.py line 115

Change hardcoded path:
```python
# Before:
ingest_path = REPO_ROOT / "data" / "current_ingest" / "ingest.parquet"

# After:
ingest_path = Path(args.ingest_path) if args.ingest_path else (
    paths.ingest / "ingest.parquet"
    if paths else REPO_ROOT / "data" / "current_ingest" / "ingest.parquet"
)
```

Add new argparse arg:
```python
parser.add_argument("--ingest-path", default=None, type=Path)
```

---

## Step 8: Analysis/Training Scripts (9 files)

Same pattern as Step 7. Key scripts:

**multi_signal_detector.py:**

| Arg | Domain Path |
|-----|-------------|
| `--multisignal` | `paths.lineage_tracking / "lineage_multisignal_features.csv"` |
| `--timeseries` | `paths.lineage_tracking / "lineage_timeseries.csv"` |
| `--tight-mapping` | `paths.experiments / "stage0_tight_mapping/milestone_lineage_mapping_tight.csv"` |
| `--semantic-velocity` | `paths.experiments / "stage1_quarterly_embeddings/semantic_velocity.csv"` |
| `--output-dir` | `paths.experiments / "multi_signal_detector"` |

**label_inflection_points.py:**

| Arg | Domain Path |
|-----|-------------|
| `--timeseries` | `paths.lineage_tracking / "lineage_timeseries.csv"` |
| `--milestones` | `paths.experiments / "stage0_tight_mapping/milestone_lineage_mapping_tight.csv"` |
| `--field-metrics` | `paths.front_aggregation / "field_metrics.parquet"` |
| `--out` | `paths.lineage_tracking / "inflection_labels.csv"` |

---

## Step 9: Horizon Scanner Scripts (4 files)

**update_assessment_history.py:**

| Arg | Domain Path |
|-----|-------------|
| `--history` | `paths.assessments / "assessment_history.csv"` |
| `--labels` | `paths.lineage_tracking / "onset_labels_msd.csv"` |
| `--predictions` | (user-specified, no default change needed) |

**generate_horizon_estimates.py:**

| Arg | Domain Path |
|-----|-------------|
| `--predictions` | `paths.experiments / "multi_signal_detector/breakthrough_predictions.csv"` |
| `--history` | `paths.assessments / "assessment_history.csv"` |
| `--out` | `paths.assessments / "horizon_estimates.csv"` |

**generate_quarterly_report.py:**

| Arg | Domain Path |
|-----|-------------|
| `--predictions` | `paths.experiments / "multi_signal_detector/breakthrough_predictions.csv"` |
| `--history` | `paths.assessments / "assessment_history.csv"` |
| `--horizon-estimates` | `paths.assessments / "horizon_estimates.csv"` |

**refine_calibration.py:**

| Arg | Domain Path |
|-----|-------------|
| `--history` | `paths.assessments / "assessment_history.csv"` |
| `--cal-history` | `paths.assessments / "calibration_history.json"` |

---

## Step 10: Config Files (2 files)

### 10a. config/multisignal_config.yaml

The path fields (`graphs_dir`, `metrics_dir`, `slices_dir`, etc.) are NOT
used at runtime for path resolution (confirmed in path trace). Update them
as documentation to use `{domain}` placeholder:

```yaml
# Before:
graphs_dir: "data/out/graphs"
slices_dir: "data/current_ingest/slices"

# After (documentation only):
# Note: paths are resolved at runtime via --domain or explicit CLI args.
# These values are illustrative defaults for the PSC domain.
graphs_dir: "data/psc/graphs"
slices_dir: "data/psc/ingest/slices"
```

### 10b. config/features/feature_groups.yaml

Update the `source:` field (documentation only):

```yaml
# Before:
source: data/out/04_front_aggregation/field_metrics.csv

# After:
source: data/{domain}/out/04_front_aggregation/field_metrics.csv
```

---

## Step 11: Test Updates

### 11a. tests/test_smoke.py line 424

Before:
```python
"--outdir", "data/out",
```

After:
```python
"--outdir", "data/psc/out",
```

Or better, make it domain-aware if the test uses `--domain`.

### 11b. tests/test_domain_registry.py

Update existing tests that reference `outdir_suffix`:
- Remove or update any test that asserts `outdir_suffix` exists on DomainConfig
- Add test for `resolve_data_paths()` on both domains

### 11c. Add to tests/test_smoke.py

```python
def test_domain_data_paths_importable():
    from src.domain_registry import DomainDataPaths
    assert DomainDataPaths is not None

def test_all_domains_resolve_data_paths():
    from src.domain_registry import DOMAIN_REGISTRY
    for domain in DOMAIN_REGISTRY.values():
        paths = domain.resolve_data_paths(Path(__file__).resolve().parents[1])
        assert paths.domain_id == domain.domain_id
        warnings = paths.validate_paths()
        assert len(warnings) == 0
```

---

## Step 12: Backward-Compat Symlinks + Cleanup

### 12a. Create symlinks (after migration, temporary)

On Windows, use directory junctions (`mklink /J`):
```cmd
mklink /J data\current_ingest data\psc\ingest
mklink /J data\current_graphs data\psc\graphs
mklink /J data\out data\psc\out
```

On Unix:
```bash
ln -s data/psc/ingest data/current_ingest
ln -s data/psc/graphs data/current_graphs
ln -s data/psc/out data/out
```

### 12b. Remove symlinks (after all scripts updated)

Delete the symlinks once all 34 deferrable scripts have been updated or
confirmed to no longer reference old paths.

### 12c. Update documentation

- Update `CLAUDE.md` memory entries referencing `data/out/` paths
- Update `docs/implementation/frontpulse_program/README.md` if it references paths
- Update any shell scripts or Makefiles

---

## Verification Checklist

After each step, verify:

- [ ] `python -m pytest tests/ -x -q --tb=short -k "not test_core_group"` passes
- [ ] `ruff check src/ scripts/ tests/` passes
- [ ] No circular imports introduced
- [ ] `python -c "from src.domain_registry import DomainDataPaths"` works

After migration:

- [ ] `ls data/psc/ingest/ingest.parquet` exists
- [ ] `ls data/psc/graphs/citation_graph_cumulative_2025Q4.pkl` exists
- [ ] `ls data/psc/out/02_lineage_tracking/lineage_registry.json` exists
- [ ] `python run.py --domain psc --schema config/schema.yaml --slices config/slices.yaml --skip-ingest --skip-raw --skip-graphs --skip-communities` succeeds

After full script updates:

- [ ] `python run.py --domain psc ...` uses `data/psc/` paths
- [ ] `python run.py --domain crispr ...` uses `data/crispr/` paths
- [ ] `python run.py --config config/datasources.yaml --outdir data/scratch/out --ingest-dir data/scratch/ingest --graphs-dir data/scratch/graphs` still works (no --domain)
- [ ] Settings watermark stored per-domain

---

## Estimated Scope

| Step | Files Changed | Lines Changed (est.) |
|------|--------------|---------------------|
| 1 | 1 | ~120 new |
| 2 | 1 | ~100 new |
| 3 | 1 | ~40 changed |
| 4 | 1 | ~15 changed |
| 5 | 1 | ~20 changed |
| 6 | 1 | ~100 new |
| 7 | 11 | ~10-15 per file = ~130 |
| 8 | 9 | ~10-15 per file = ~110 |
| 9 | 4 | ~10-15 per file = ~50 |
| 10 | 2 | ~10 changed |
| 11 | 2 | ~30 changed |
| 12 | 0 (manual) | N/A |
| **Total** | **34 files** | **~725 lines** |

---

## Risk Mitigation

1. **Test after every step** -- run full suite between each step
2. **Symlinks bridge the gap** -- deferrable scripts work throughout
3. **Legacy fallbacks** -- every domain resolution returns legacy default when
   `--domain` is not set
4. **Dry-run migration** -- always test migration before executing
5. **Git safety** -- commit after each step; easy rollback if needed
