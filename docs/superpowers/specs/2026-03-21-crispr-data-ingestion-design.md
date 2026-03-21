# CRISPR Data Ingestion Configuration

**Date**: 2026-03-21
**Status**: Approved
**Scope**: Add CRISPR gene editing as a second research domain for the FrontPulse pipeline

---

## Context

FrontPulse tracks research front evolution via citation graph analysis, Leiden
clustering, and onset detection. The pipeline was developed and validated on
Perovskite Solar Cells (PSC, OpenAlex topic T10247). CRISPR gene editing is the
primary validation domain for cross-domain generalization, selected for its
staggered inflection points (2012 Cas9, 2013 eukaryotic, 2015 in vivo, 2017 base
editing, 2019 prime editing, 2023 FDA approval).

The goal is NOT to detect the onset of CRISPR itself (that is trivially known),
but to track the research fronts that evolved within the CRISPR ecosystem and
detect when each front begins its own exponential growth phase.

## Design Decisions

### Data source strategy: Approach A (topic filter only)

Use `topics.id: T10878` ("CRISPR and Genetic Engineering") as the sole API
filter. Estimated corpus: ~238K works from 2000-01-01 onward (verified via
OpenAlex API count query on 2026-03-21).

**Rationale**: The citation graph naturally separates core CRISPR methodology from
applied fronts via Leiden clustering. Peripheral papers with weak citation
connections to the CRISPR core will form small, short-lived lineages that the
stability filter (Phase 2) removes. No need for multi-source deduplication or
keyword search supplements.

**Alternatives considered**:

- *Two-pass pull (topic + keyword search)*: Marginal recall gain, requires
  pipeline changes for multi-source deduplication. Rejected.
- *Concept-based (C98108389)*: Tighter focus (~78K works), but OpenAlex is
  deprecating concepts in favor of topics. Rejected.

### API authentication: environment variable

OpenAlex has migrated from `mailto` polite pool to API key authentication. Keys
are passed as the `api_key` query parameter.

**Key management**: The API key is loaded from the `OPENALEX_API_KEY` environment
variable (or a `.env` file loaded via `python-dotenv`). It never appears in any
committed configuration file.

### Output isolation: separate directories per domain

Each research domain writes to its own output directories, selected via CLI
arguments. No code changes needed for isolation -- the pipeline already supports
`--outdir`, `--ingest-dir`, and `--graphs-dir`.

| Domain | Ingest Dir             | Graphs Dir             | Output Dir         |
| ------ | ---------------------- | ---------------------- | ------------------ |
| PSC    | `data/current_ingest/` | `data/current_graphs/` | `data/out/`        |
| CRISPR | `data/crispr_ingest/`  | `data/crispr_graphs/`  | `data/out_crispr/` |

---

## Deliverables

### 1. `config/datasources_crispr.yaml` (new file)

Mirrors the structure of `config/datasources.yaml` (PSC) with CRISPR-specific
filter values. Includes `mailto: null` and `merges: []` for structural parity
with the existing config.

```yaml
sources:
  primary:
    kind: openalex
    entity: works
    mailto: null
    per_page: 200
    max_records: null
    filters:
      topics.id: T10878
      from_publication_date: '2000-01-01'
      to_publication_date: ''
    select: null
    sort: null
merges: []
```

Authentication is handled by the `OPENALEX_API_KEY` environment variable, not
the config file. The `mailto` field is retained for structural consistency and
backward compatibility.

### 2. `config/front_aliases_crispr.yaml` (new placeholder)

Initial sketch of expected CRISPR research fronts. Will be refined after
inspecting Leiden clustering output. This file is not consumed by the pipeline
until front-mapping is implemented (Phase 2+); it serves as documentation of
known research fronts for future reference.

```yaml
fronts:
  cas9_methodology:
    canonical: "Cas9 Editing Methodology"
    aliases: ["CRISPR-Cas9", "RNA-guided endonuclease"]
  eukaryotic_applications:
    canonical: "Eukaryotic/Mammalian Applications"
  in_vivo_editing:
    canonical: "In Vivo Therapeutic Editing"
  base_editing:
    canonical: "Base Editing"
    aliases: ["cytosine base editor", "adenine base editor", "CBE", "ABE"]
  prime_editing:
    canonical: "Prime Editing"
  diagnostics:
    canonical: "CRISPR Diagnostics"
    aliases: ["SHERLOCK", "DETECTR", "Cas13"]
  clinical_therapeutics:
    canonical: "Clinical/FDA-Approved Therapies"
    aliases: ["Casgevy", "exa-cel", "sickle cell"]
```

### 3. `src/openalex.py` (modify)

Update `fetch_openalex()` to support API key authentication:

- Add optional `api_key: str | None = None` parameter to the function signature
- Make `mailto` optional (`mailto: str | None = None`) instead of required
- If `api_key` is provided, add `params["api_key"] = api_key` to every request
  and remove `params["mailto"]` (the key replaces mailto for authentication)
- If only `mailto` is provided (no api_key), preserve current behavior
- If neither is provided, raise `ValueError` with a clear message
- Update the User-Agent header (line 80):
  - Rebrand from `"2YP-RF-Ingest/0.1"` to `"FrontPulse/1.0"`
  - When `mailto` is available: `"FrontPulse/1.0 (+mailto:{mailto})"`
  - When only `api_key` is available: `"FrontPulse/1.0"`

### 4. `src/ingest.py` (modify)

Update `_read_one()` (not `ingest()`) to pass the API key through. The call
chain is: `run.py main()` -> `ingest()` -> `_read_one()` -> `fetch_openalex()`.

Changes to `_read_one()` (lines 50-67):

- Read `api_key` from the source config: `api_key = src.get("api_key")`
- Relax the mailto-only validation at lines 52-57: accept either `mailto` or
  `api_key` (raise ValueError only if BOTH are missing)
- Pass `api_key=api_key` kwarg to `fetch_openalex()`

### 5. `run.py` (modify)

The mailto validation block at lines 657-663 currently enforces that a mailto
value is present, aborting if not. This must be relaxed for API-key-only runs.

Changes:

- After loading settings, load API key from environment:
  `api_key = os.environ.get("OPENALEX_API_KEY")`
- Modify the validation block (lines 657-663): accept either a valid `mailto`
  OR a valid `OPENALEX_API_KEY`. Only abort if neither is available.
- Inject `api_key` into `settings["source_overrides"]` via
  `build_source_overrides()` so it reaches `_read_one()` -> `fetch_openalex()`
- Update `build_source_overrides()` (line 157-168) to include the `api_key`
  key in its returned dictionary

### 6. `requirements.txt` (modify)

Add `python-dotenv` dependency for `.env` file loading:

```
python-dotenv>=1.0
```

Add a `load_dotenv()` call near the top of `run.py` (after imports) so that
`.env` values are available via `os.environ` before any settings are loaded.

### 7. `.env.template` (new file, committed)

Template for users to copy to `.env` (which is already gitignored per line 75
of `.gitignore`):

```
# OpenAlex API key -- get yours at https://openalex.org/settings/api
OPENALEX_API_KEY=
```

---

## What is NOT in scope

- Domain selector UI at pipeline start (captured as FP-dvd, P3 backlog)
- Multi-source deduplication (not needed for Approach A)
- CRISPR-specific analysis scripts
- Front alias refinement (depends on clustering output)
- Any changes to `config/schema.yaml` or `config/slices.yaml` (these are
  field-agnostic and work for CRISPR without modification)
- Changes to evaluation configs

---

## Run sequence

After implementation:

```bash
# 1. Set up API key
cp .env.template .env
# Edit .env to add your OPENALEX_API_KEY

# 2. Preflight test with small pull (use datasource max_records override)
# Create a temporary datasource config with max_records: 100, or use
# the interactive --configure mode to set max_records for a test run.

# 3. Run full pipeline for CRISPR
python run.py \
  --config config/datasources_crispr.yaml \
  --schema config/schema.yaml \
  --slices config/slices.yaml \
  --outdir data/out_crispr \
  --ingest-dir data/crispr_ingest \
  --graphs-dir data/crispr_graphs \
  --graph-mode cumulative

# 4. Inspect results
# - data/crispr_ingest/ingest.parquet  (~238K rows)
# - data/out_crispr/02_lineage_tracking/lineage_timeseries.csv
# - Assess corpus size, front count, lineage stability

# 5. If corpus is too large, add primary_topic_only post-filter
```

---

## Estimated effort

| Deliverable | Effort |
| --- | --- |
| datasources_crispr.yaml | 10 min |
| front_aliases_crispr.yaml | 15 min |
| openalex.py API key + User-Agent update | 30 min |
| ingest.py _read_one() key passthrough | 15 min |
| run.py mailto/api_key validation | 20 min |
| requirements.txt + dotenv integration | 10 min |
| .env.template | 5 min |
| Testing (preflight + small pull) | 30 min |
| **Total** | **~2.25 hours** |

---

## Verification

1. Small test pull succeeds with API key auth (set `max_records: 100` in a
   temporary datasource config or via interactive mode)
2. Returned works have `topics` containing T10878
3. Data writes to `data/crispr_ingest/`, not `data/current_ingest/`
4. Existing PSC config (`config/datasources.yaml`) and data are untouched
5. Pipeline runs without `mailto` when `OPENALEX_API_KEY` is set
6. Pipeline still works with `mailto` only (no api_key) for backward
   compatibility with PSC workflows
7. Pipeline errors clearly if neither key nor mailto is provided
8. User-Agent header reads "FrontPulse/1.0" (not "2YP")
