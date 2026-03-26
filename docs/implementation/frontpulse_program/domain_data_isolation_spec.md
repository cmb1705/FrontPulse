# Domain-Isolated Data Directory Structure

**Epic:** FP-yb9
**Status:** Design spec (FP-gqm)
**Date:** 2026-03-25

---

## Problem Statement

FrontPulse supports multiple research domains (PSC, CRISPR, future topics) but
stores intermediate pipeline artifacts in shared directories. When a CRISPR
pipeline run executes after a PSC run, it overwrites PSC data in
`data/current_ingest/slices/` and `data/current_graphs/`. The output directory
is partially isolated via `DomainConfig.outdir_suffix` (e.g., `data/out_crispr/`)
but ingest and graph directories are not.

**Concrete failure observed:** Running `python run.py --domain crispr --outdir
data/out_crispr` overwrote PSC quarterly parquet slices in
`data/current_ingest/slices/` because `--ingest-dir` defaults to
`data/current_ingest` regardless of domain.

## Design Goals

1. Each domain gets a self-contained data tree -- no cross-domain file sharing
2. `DomainConfig` is the single source of truth for all data paths
3. Explicit CLI path overrides remain available (escape hatch)
4. Backward-compatible: existing PSC data migrates without re-ingestion
5. New domains auto-create their directory tree on first run
6. No new pip dependencies

## Non-Goals

- Per-run versioning within a domain (archive handles this)
- Domain-specific config/schema files (already handled by domain registry)
- Shared reference data across domains (each domain is self-contained)

---

## Canonical Directory Layout

```
data/
  {domain_id}/
    ingest/                     was: data/current_ingest/
      ingest.parquet
      raw/
        openalex_raw_*.ndjson.gz
        openalex_raw_*_manifest.json
      slices/
        by_quarter__YYYYQN.parquet
        last_8q.parquet
    graphs/                     was: data/current_graphs/
      citation_graph_cumulative_YYYYQN.pkl
      citation_graph_annual_YYYY.pkl
      citation_graph_delta_YYYYQN.pkl
    out/                        was: data/out/ or data/out_crispr/
      02_lineage_tracking/
        lineage_registry.json
        lineage_timeseries.csv
        lineage_multisignal_features.csv
        onset_labels_msd.csv
        convergence_features.csv
      03_milestone_mapping/
      04_front_aggregation/
        field_metrics.parquet
      cache_cum/
        partitions_cum/
        cores_cum/
      cache_lineage/
        reference_data.pkl
      cache_coupling/
      assessments/
        assessment_history.csv
        horizon_estimates.csv
        quarterly_report_YYYYQN.md
        calibration_history.json
      experiments/
        multi_signal_detector/
        optuna_search/
        stage0_tight_mapping/
        stage1_quarterly_embeddings/
      communities_cumulative.json
      manifest.json
      logs/
    archive/                    was: data/archive/
      YYYYMMDD_HHMMSS/
        ingest/
        graphs/
        out/
```

**Examples:**

| Before (current) | After |
|-------------------|-------|
| `data/current_ingest/` | `data/psc/ingest/` |
| `data/current_graphs/` | `data/psc/graphs/` |
| `data/out/` | `data/psc/out/` |
| `data/out_crispr/` | `data/crispr/out/` |
| `data/current_ingest/slices/` | `data/psc/ingest/slices/` |

---

## DomainConfig API Changes

### Current State

```python
@dataclass
class DomainConfig:
    domain_id: str
    display_name: str
    datasources_config: str
    front_aliases_config: str | None
    description: str
    outdir_suffix: str = ""          # REMOVE

    def resolve_paths(self, project_root: Path) -> dict[str, Path | None]:
        # Only resolves config file paths
```

### Proposed State

```python
@dataclass
class DomainConfig:
    domain_id: str
    display_name: str
    datasources_config: str
    front_aliases_config: str | None
    description: str
    # outdir_suffix removed -- convention replaces it

    def resolve_paths(self, project_root: Path) -> dict[str, Path | None]:
        # Unchanged: config file resolution only
        ...

    def resolve_data_paths(self, project_root: Path) -> DomainDataPaths:
        """Derive all data directory paths from domain_id convention.

        Returns a DomainDataPaths with every directory the pipeline
        reads from or writes to for this domain.
        """
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

### DomainDataPaths Dataclass

```python
@dataclass(frozen=True)
class DomainDataPaths:
    """All data directory paths for a single research domain."""

    domain_id: str
    base: Path              # data/{domain_id}/
    ingest: Path            # data/{domain_id}/ingest/
    raw: Path               # data/{domain_id}/ingest/raw/
    slices: Path            # data/{domain_id}/ingest/slices/
    graphs: Path            # data/{domain_id}/graphs/
    out: Path               # data/{domain_id}/out/
    lineage_tracking: Path  # data/{domain_id}/out/02_lineage_tracking/
    front_aggregation: Path # data/{domain_id}/out/04_front_aggregation/
    cache_cum: Path         # data/{domain_id}/out/cache_cum/
    cache_lineage: Path     # data/{domain_id}/out/cache_lineage/
    cache_coupling: Path    # data/{domain_id}/out/cache_coupling/
    assessments: Path       # data/{domain_id}/out/assessments/
    experiments: Path       # data/{domain_id}/out/experiments/
    archive: Path           # data/{domain_id}/archive/

    def ensure_dirs(self) -> None:
        """Create all directories in the tree if they don't exist."""
        for field_name in self.__dataclass_fields__:
            path = getattr(self, field_name)
            if isinstance(path, Path) and field_name != "domain_id":
                path.mkdir(parents=True, exist_ok=True)

    def to_dict(self) -> dict[str, str]:
        """Serialize to string dict for manifest storage."""
        return {k: str(v) for k, v in self.__dict__.items()}
```

---

## Path Resolution Strategy

### Priority Order (highest to lowest)

1. **Explicit CLI argument** -- `--ingest-dir /custom/path` always wins
2. **Domain convention** -- `--domain psc` derives `data/psc/ingest/`
3. **Error** -- neither specified, pipeline refuses to start

### Helper Function

Add to `src/domain_registry.py`:

```python
def resolve_pipeline_paths(
    domain_id: str | None,
    cli_ingest_dir: str | None,
    cli_graphs_dir: str | None,
    cli_outdir: str | None,
    project_root: Path,
) -> DomainDataPaths:
    """Resolve data paths from domain convention with CLI overrides.

    If domain_id is set, all paths derive from convention. Individual
    CLI arguments override specific directories when provided explicitly
    (i.e., not left at argparse default).

    If domain_id is None, all three directory args must be specified.
    """
```

### Argparse Changes in run.py

```python
# Before:
ap.add_argument("--outdir", required=True)
ap.add_argument("--ingest-dir", default="data/current_ingest")
ap.add_argument("--graphs-dir", default="data/current_graphs")

# After:
ap.add_argument("--outdir", default=None,
    help="Output directory (derived from --domain if not set)")
ap.add_argument("--ingest-dir", default=None,
    help="Ingest directory (derived from --domain if not set)")
ap.add_argument("--graphs-dir", default=None,
    help="Graphs directory (derived from --domain if not set)")
```

With `--domain psc`, all three are derived automatically:
```bash
python run.py --domain psc --schema config/schema.yaml --slices config/slices.yaml
# Resolves to:
#   ingest-dir = data/psc/ingest/
#   graphs-dir = data/psc/graphs/
#   outdir     = data/psc/out/
```

Explicit overrides still work:
```bash
python run.py --domain psc --ingest-dir /mnt/fast/psc_ingest ...
# ingest-dir = /mnt/fast/psc_ingest (override)
# graphs-dir = data/psc/graphs/     (convention)
# outdir     = data/psc/out/        (convention)
```

Without `--domain`, explicit paths are required (backward compat for ad-hoc runs):
```bash
python run.py --config config/datasources.yaml \
    --ingest-dir data/scratch/ingest \
    --graphs-dir data/scratch/graphs \
    --outdir data/scratch/out
```

---

## Script Integration Pattern

### Problem: 72+ scripts have hardcoded default paths

Most scripts follow this pattern:
```python
ap.add_argument("--registry",
    default="data/out/02_lineage_tracking/lineage_registry.json")
ap.add_argument("--slices-dir",
    default="data/current_ingest/slices")
```

### Solution: Domain-aware defaults

Add a shared argument group factory:

```python
# In src/domain_registry.py

def add_domain_args(parser: argparse.ArgumentParser) -> None:
    """Add --domain argument and override group to any script's parser."""
    parser.add_argument("--domain", default=None, choices=sorted(DOMAIN_REGISTRY),
        help="Research domain (derives all data paths from convention)")
    parser.add_argument("--data-root", default=None,
        help="Override data/{domain}/ base directory")

def resolve_script_paths(
    args: argparse.Namespace,
    project_root: Path,
) -> DomainDataPaths | None:
    """Resolve paths for scripts that accept --domain.

    Returns DomainDataPaths if --domain is set, None otherwise.
    Scripts fall back to their explicit CLI args when None.
    """
    if args.domain is None:
        return None
    domain = get_domain(args.domain)
    paths = domain.resolve_data_paths(project_root)
    if getattr(args, "data_root", None):
        # Allow override of the base directory
        paths = _rebase_paths(paths, Path(args.data_root))
    return paths
```

### Script adoption pattern (incremental)

Each script adds domain awareness with minimal change:

```python
# Before (compute_lineage_multisignal_features.py):
ap.add_argument("--registry",
    default="data/out/02_lineage_tracking/lineage_registry.json")
ap.add_argument("--partitions-dir",
    default="data/out/cache_cum/partitions_cum")
ap.add_argument("--slices-dir",
    default="data/current_ingest/slices")

# After:
from src.domain_registry import add_domain_args, resolve_script_paths

add_domain_args(ap)
ap.add_argument("--registry", default=None)
ap.add_argument("--partitions-dir", default=None)
ap.add_argument("--slices-dir", default=None)

args = ap.parse_args()
paths = resolve_script_paths(args, REPO)

# Resolve with domain fallback:
registry = args.registry or (
    str(paths.lineage_tracking / "lineage_registry.json")
    if paths else "data/out/02_lineage_tracking/lineage_registry.json"
)
```

This is **incremental**: scripts that haven't been updated yet continue to work
with explicit CLI paths. Domain-aware scripts gain automatic path resolution.

---

## Guard Rails

### Cross-Domain Write Prevention

The `DomainDataPaths.ensure_dirs()` method only creates directories under
`data/{domain_id}/`. The pipeline should log a warning if any output path
falls outside the domain's base directory:

```python
def validate_paths(self) -> list[str]:
    """Check that all paths are under the domain base directory."""
    warnings = []
    for field_name in self.__dataclass_fields__:
        path = getattr(self, field_name)
        if isinstance(path, Path):
            try:
                path.relative_to(self.base)
            except ValueError:
                warnings.append(
                    f"{field_name}={path} is outside domain base {self.base}"
                )
    return warnings
```

### Settings File Isolation

Currently `.2yp_settings.json` at project root stores the last-used topic ID
and watermark. With domain isolation, settings should be per-domain:
`data/{domain_id}/settings.json`. This prevents CRISPR's watermark from
overwriting PSC's.

---

## Migration Strategy

### One-Time Migration Script: `scripts/migrate_domain_layout.py`

```
Usage: python scripts/migrate_domain_layout.py [--dry-run] [--domain psc]

Actions (PSC domain):
  1. Move data/current_ingest/  -> data/psc/ingest/
  2. Move data/current_graphs/  -> data/psc/graphs/
  3. Move data/out/             -> data/psc/out/
  4. Leave data/out_crispr/     -> data/crispr/out/ (separate pass)

Actions (CRISPR domain):
  5. Move data/out_crispr/      -> data/crispr/out/
  6. data/crispr/ingest/ and data/crispr/graphs/ remain empty
     (CRISPR raw data needs re-ingestion into isolated tree)
```

**Safety:**
- `--dry-run` prints planned moves without executing
- Verifies no destination conflicts before moving
- Writes `data/.migration_log.json` with timestamp and file counts
- Does not delete source dirs until all moves succeed
- Preserves file timestamps

**Backward Compatibility Symlinks (optional, temporary):**
```
data/current_ingest -> data/psc/ingest     (symlink)
data/current_graphs -> data/psc/graphs     (symlink)
data/out            -> data/psc/out        (symlink)
```

These symlinks allow scripts not yet updated to `--domain` to continue
working. Remove them once all scripts are migrated.

---

## Affected Files Inventory

### Must Change (core pipeline)

| File | Change | Scope |
|------|--------|-------|
| `src/domain_registry.py` | Add `DomainDataPaths`, `resolve_data_paths()`, `resolve_pipeline_paths()`, `add_domain_args()`, `resolve_script_paths()` | ~80 lines new |
| `run.py` | Replace hardcoded defaults with domain-derived paths; `--outdir` no longer required when `--domain` set | ~30 lines changed |
| `src/settings.py` | Per-domain settings path (`data/{domain}/settings.json`) | ~10 lines changed |

### Should Change (core scripts with hardcoded data/out paths)

| File | Change |
|------|--------|
| `scripts/compute_lineage_multisignal_features.py` | Add `--domain`, derive 6 path args |
| `scripts/multi_signal_detector.py` | Add `--domain`, derive 4 path args |
| `scripts/label_inflection_points.py` | Add `--domain`, derive 3 path args |
| `scripts/compute_convergence_features.py` | Add `--domain`, derive 4 path args |
| `scripts/update_assessment_history.py` | Add `--domain`, derive 2 path args |
| `scripts/generate_horizon_estimates.py` | Add `--domain`, derive 2 path args |
| `scripts/generate_quarterly_report.py` | Add `--domain`, derive 3 path args |
| `scripts/refine_calibration.py` | Add `--domain`, derive 2 path args |
| `scripts/communities.py` | Add `--domain`, derive 2 path args |
| `scripts/compute_lineage_ctfidf.py` | Add `--domain`, derive 4 path args |
| `scripts/compute_lineage_embeddings.py` | Add `--domain`, derive 3 path args |
| `scripts/compute_lineage_npmi.py` | Add `--domain`, derive 4 path args |
| `scripts/aggregate_field_metrics.py` | Add `--domain`, derive 2 path args |
| `scripts/filter_stable_lineages.py` | Add `--domain`, derive 1 path arg |
| `scripts/build_lite_graphs.py` | Add `--domain`, derive 1 path arg |
| `scripts/run_all_metrics.py` | Add `--domain`, derive 3 path args |

### Can Defer (analysis/experiment scripts)

All scripts in `scripts/` that are used for one-off analysis, benchmarking,
or visualization. These can be updated lazily -- they still accept explicit
CLI paths. Approximately 50 scripts in this category.

### New Files

| File | Purpose |
|------|---------|
| `scripts/migrate_domain_layout.py` | One-time data migration |
| `tests/test_domain_data_paths.py` | Unit tests for DomainDataPaths |

---

## DOMAIN_REGISTRY Update

```python
DOMAIN_REGISTRY: dict[str, DomainConfig] = {
    "psc": DomainConfig(
        domain_id="psc",
        display_name="Polymer Solar Cells (PSC)",
        datasources_config="config/datasources.yaml",
        front_aliases_config="config/front_aliases.yaml",
        description="Baseline domain: polymer solar cell research fronts (T10247)",
        # outdir_suffix removed
    ),
    "crispr": DomainConfig(
        domain_id="crispr",
        display_name="CRISPR and Genetic Engineering",
        datasources_config="config/datasources_crispr.yaml",
        front_aliases_config="config/front_aliases_crispr.yaml",
        description="Second domain: CRISPR gene editing research fronts (T10878)",
        # outdir_suffix removed
    ),
}
```

---

## Testing Plan

### Unit Tests (test_domain_data_paths.py)

- `test_resolve_data_paths_convention` -- paths match `data/{id}/...` pattern
- `test_resolve_data_paths_all_under_base` -- validate_paths returns no warnings
- `test_ensure_dirs_creates_tree` -- all dirs created in tmp_path
- `test_resolve_pipeline_paths_domain_only` -- no CLI overrides
- `test_resolve_pipeline_paths_with_override` -- CLI ingest-dir overrides convention
- `test_resolve_pipeline_paths_no_domain_no_paths_raises` -- ValueError
- `test_domain_data_paths_frozen` -- cannot assign new values
- `test_to_dict_serialization` -- round-trip to dict
- `test_add_domain_args_adds_choices` -- parser has --domain with correct choices

### Integration Tests

- `test_run_py_domain_psc_derives_paths` -- mock run with --domain psc, verify paths
- `test_migration_dry_run` -- migration script reports moves without acting
- `test_settings_per_domain` -- settings saved/loaded from domain path

### Smoke Tests (add to test_smoke.py)

- `test_domain_data_paths_importable`
- `test_all_registered_domains_have_valid_paths`
- `test_backward_compat_no_domain_explicit_paths`

---

## Rollout Sequence

### Phase 1: Core Infrastructure
1. Add `DomainDataPaths` and `resolve_data_paths()` to `src/domain_registry.py`
2. Add `resolve_pipeline_paths()`, `add_domain_args()`, `resolve_script_paths()`
3. Write `tests/test_domain_data_paths.py`
4. Update `run.py` to use domain-derived paths

### Phase 2: Migration
5. Write `scripts/migrate_domain_layout.py`
6. Run migration for PSC data (with dry-run first)
7. Run migration for CRISPR data
8. Create backward-compat symlinks

### Phase 3: Script Updates (Core Pipeline)
9. Update 16 core scripts to accept `--domain`
10. Update `src/settings.py` for per-domain settings
11. Remove `outdir_suffix` from `DomainConfig`

### Phase 4: Cleanup
12. Update existing tests that reference `data/out/` paths
13. Remove backward-compat symlinks after verifying all scripts work
14. Update `CLAUDE.md` and memory files

---

## Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| Migration corrupts data | `--dry-run` first; `.migration_log.json` for audit |
| Scripts not yet updated break | Backward-compat symlinks bridge the gap |
| Disk space doubles during migration | Move (not copy) by default; same filesystem = instant |
| Hardcoded paths in notebooks or ad-hoc scripts | Symlinks cover this; grep audit catches remaining refs |
| Windows symlink permissions | Use junction points (`mklink /J`) instead of symlinks on Windows |
| `outdir_suffix` removal breaks existing CRISPR commands | Migration moves `out_crispr/` to `crispr/out/` first |

---

## Decision Record

**Why convention over configuration?**

The alternative is adding `ingest_dir`, `graphs_dir`, `out_dir` fields to
`DomainConfig` and letting users configure arbitrary paths per domain. This
was rejected because:
- More fields to maintain per domain
- Path typos cause silent cross-domain writes
- Convention (`data/{domain_id}/`) is predictable and self-documenting
- Explicit CLI overrides serve the same escape-hatch purpose

**Why not `data/domains/{domain_id}/`?**

One extra nesting level with no benefit. `data/psc/` is shorter and equally
clear. The `data/` directory already contains only pipeline data.

**Why frozen dataclass for DomainDataPaths?**

Prevents accidental mutation after resolution. All path customization happens
at construction time through `resolve_pipeline_paths()`.
