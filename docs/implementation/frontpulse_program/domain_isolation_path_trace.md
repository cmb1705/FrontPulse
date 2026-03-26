# Pipeline Path Flow Trace

**Task:** FP-r8u
**Date:** 2026-03-25

Traces how data directory paths propagate through the full pipeline execution
chain at runtime. Complements the static codebase audit (FP-i6q).

---

## Pipeline Execution Stages and Path Propagation

### Stage 1: Entry Point (run.py)

```
CLI args parsed:
  --domain psc          -> resolve_domain_args() -> args.config = "config/datasources.yaml"
  --ingest-dir          -> default "data/current_ingest"
  --graphs-dir          -> default "data/current_graphs"
  --raw-dir             -> default None (derived as ingest_dir/"raw")
  --outdir              -> REQUIRED
  --coupling-cache-dir  -> default "data/out/cache_coupling"

setup_directories(args) returns:
  ingest_dir  = Path(args.ingest_dir)             # data/current_ingest
  slices_dir  = ingest_dir / "slices"             # data/current_ingest/slices
  graphs_dir  = Path(args.graphs_dir)             # data/current_graphs
  raw_dir     = Path(args.raw_dir) or ingest_dir/"raw"  # data/current_ingest/raw
```

**Key issue:** `--domain` resolves the config file but NOT the data directories.
All four directory paths use their hardcoded defaults regardless of domain.

### Stage 2: Ingest Phase

```
run_ingest_phase(args, logger, ingest_dir, raw_dir, cache_path, settings)
  cache_path = ingest_dir / "ingest.parquet"     # data/current_ingest/ingest.parquet

  Path writes:
    ingest_dir / "ingest.parquet"                 # flattened dataset
    raw_dir / "openalex_raw_*.ndjson.gz"          # raw chunks (if --skip-raw not set)
    raw_dir / "*_manifest.json"                   # raw manifest
```

**Path mechanism:** Function parameters from `setup_directories()`.

### Stage 3: Slicing Phase

```
apply_slices(df, slices_config, ...)
save_slices_with_stats(sliced, slices_dir, ingest_dir, ...)

  Path writes:
    slices_dir / "by_quarter__YYYYQN.parquet"     # data/current_ingest/slices/
    slices_dir / "last_8q.parquet"                # data/current_ingest/slices/
```

**Path mechanism:** `slices_dir` derived from `ingest_dir / "slices"`.

### Stage 4: Graph Building Phase

```
build_graphs_phase(args, df, years, quarters, graphs_dir, coupling_cfg, logger)

  Path writes:
    graphs_dir / "citation_graph_cumulative_YYYYQN.pkl"
    graphs_dir / "citation_graph_annual_YYYY.pkl"
    graphs_dir / "citation_graph_delta_YYYYQN.pkl"

  Coupling cache:
    Path(args.coupling_cache_dir) / "*.pkl"       # data/out/cache_coupling/
```

**Path mechanism:** `graphs_dir` from function parameter; coupling cache from CLI arg.

### Stage 5: Community Detection (subprocess)

```
run_community_detection(args, graphs_dir, outdir, logger)

  Subprocess call:
    python scripts/communities.py
      --mode cumulative
      --graphs-dir {graphs_dir}                   # passed as string
      --out-dir {outdir}                          # passed as string
      --resume
      [resolution and size args from CLI]

  communities.py internally:
    cache_dir = out_dir / "cache_cum"             # derived from --out-dir
    Path writes:
      out_dir / "communities_cumulative.json"
      out_dir / "front_id_registry_cumulative.json"
      out_dir / "front_metrics_cumulative.csv"
      out_dir / "front_timeseries_cumulative.csv"
      cache_dir / "partitions_cum/part_YYYYQN.json"
      cache_dir / "cores_cum/cores_YYYYQN.json"
```

**Path mechanism:** CLI args passed via `subprocess.check_call()`.

### Stage 6: Manifest Writing

```
write_pipeline_manifest(manifest_path, ...)
  manifest_path = outdir / "manifest.json"

  Sections:
    manifest["raw"]          = raw snapshot metadata
    manifest["slices"]       = slice metadata
    manifest["graphs"]       = graph file references
    manifest["communities"]  = community detection outputs
    manifest["coupling"]     = coupling config + cache stats
```

**Path mechanism:** `outdir` from CLI; manifest stores relative paths.

### Stage 7: Settings Watermark

```
save_settings(settings)
  Writes to: .2yp_settings.json (project root)
  Updates: last_ingested_date = max_date

  Also checks: config/settings.yaml
```

**Path mechanism:** Hardcoded to project root. NOT domain-aware.

---

## Downstream Pipeline (Post-run.py)

These scripts are run separately after `run.py` completes.

### Feature Computation Chain

```
run_all_metrics.py --ingest-dir ... --graphs-dir ... --out-dir ...
  ├── metric_author_influx.py     --slices-dir {ingest_dir}/slices --out-dir {out}/metrics
  ├── metric_citation_velocity.py --slices-dir {ingest_dir}/slices --out-dir {out}/metrics
  ├── metric_reference_vitality.py --slices-dir ... --ingest-path ... --out-dir {out}/metrics
  ├── metric_topic_diversity.py   --slices-dir ... --out-dir {out}/metrics
  └── metric_cross_cluster_bridging.py --graphs-dir ... --out-dir {out}/metrics
       (also hardcodes: data/current_ingest/ingest.parquet at line 115)

compute_lineage_multisignal_features.py
  --registry       {out}/02_lineage_tracking/lineage_registry.json
  --timeseries     {out}/02_lineage_tracking/lineage_timeseries.csv
  --raw-dir        {ingest}/raw
  --partitions-dir {out}/cache_cum/partitions_cum
  --reference-cache {out}/cache_lineage/reference_data.pkl
  --metrics-dir    {out}/metrics
  --field-metrics  {out}/04_front_aggregation/field_metrics.parquet
  --out            {out}/02_lineage_tracking/lineage_multisignal_features.csv
```

**Path mechanism:** All argparse defaults. Each script resolves independently.

### Labeling and Training Chain

```
label_inflection_points.py
  --timeseries     {out}/02_lineage_tracking/lineage_timeseries.csv
  --milestones     {out}/experiments/stage0_.../milestone_lineage_mapping_tight.csv
  --field-metrics  {out}/04_front_aggregation/field_metrics.parquet
  --out            {out}/02_lineage_tracking/inflection_labels.csv

multi_signal_detector.py
  --multisignal    {out}/02_lineage_tracking/lineage_multisignal_features.csv
  --timeseries     {out}/02_lineage_tracking/lineage_timeseries.csv
  --labels         (user-specified onset labels file)
  --output-dir     {out}/experiments/multi_signal_detector/
```

**Path mechanism:** All argparse defaults pointing to `data/out/`.

### Horizon Scanner Chain

```
update_assessment_history.py record
  --predictions    {out}/experiments/multi_signal_detector/breakthrough_predictions.csv
  --history        {out}/assessments/assessment_history.csv

update_assessment_history.py backfill
  --labels         {out}/02_lineage_tracking/onset_labels_msd.csv
  --history        {out}/assessments/assessment_history.csv

generate_horizon_estimates.py
  --predictions    {out}/experiments/multi_signal_detector/breakthrough_predictions.csv
  --history        {out}/assessments/assessment_history.csv
  --out            {out}/assessments/horizon_estimates.csv

generate_quarterly_report.py
  --predictions    {out}/experiments/.../breakthrough_predictions.csv
  --history        {out}/assessments/assessment_history.csv
  --horizon-estimates {out}/assessments/horizon_estimates.csv
  --out            {out}/assessments/quarterly_report_YYYYQN.md

refine_calibration.py
  --history        {out}/assessments/assessment_history.csv
  --cal-history    {out}/assessments/calibration_history.json
```

**Path mechanism:** All argparse defaults pointing to `data/out/`.

---

## Alternative Orchestrator: run_build_pipeline.py

Uses a different path propagation pattern (PipelineConfig object, not subprocess):

```
PipelineConfig:
  output_root    = Path(args.output_dir)          # default: data/out
  lineage_dir    = output_root / "02_lineage_tracking"
  mapping_dir    = output_root / "03_milestone_mapping"
  validation_root = output_root / "06_validation"
  raw_dir        = Path(args.raw)                 # default: data/current_ingest/raw
  graphs_dir     = Path(args.graphs)              # default: data/current_graphs
  partitions_dir = Path(args.partitions)          # default: data/out/cache_cum/partitions_cum

  Stages 2-5 receive paths via config attributes (direct Python calls).
```

**Key difference:** No subprocess isolation. All stages share the same process.

---

## Config Files with Runtime Path References

| Config File | Used at Runtime? | Paths Referenced |
|-------------|------------------|-----------------|
| `config/datasources.yaml` | Yes (ingest) | API filters and CSV/parquet source paths |
| `config/datasources_crispr.yaml` | Yes (ingest) | Same pattern for CRISPR domain |
| `config/baselines/methods.json` | Yes (evaluation) | Prediction CSV paths for baseline comparison |
| `config/features/feature_groups.yaml` | Metadata only | `source:` field (documentation, not resolved) |
| `config/multisignal_config.yaml` | Params only | Path fields exist but NOT used for resolution |
| `config/defaults.yaml` | Params only | No path references |

---

## Path Dependency Graph

```
run.py (setup_directories)
  │
  ├── ingest_dir ──────────┬── slices_dir (derived)
  │                        ├── raw_dir (derived)
  │                        └── cache_path = ingest_dir/ingest.parquet
  │
  ├── graphs_dir ──────────── build_graphs_phase, run_community_detection
  │
  ├── outdir ──────────────┬── manifest.json
  │                        ├── communities.py --out-dir
  │                        ├── logs/
  │                        └── community outputs (JSON, CSV)
  │
  └── coupling_cache_dir ── data/out/cache_coupling/

[DOWNSTREAM SCRIPTS - each resolves independently]

run_all_metrics.py
  ├── ingest_dir ── slices_dir, ingest.parquet
  ├── graphs_dir ── citation graphs
  └── out_dir ──── metrics/, manifest.json

compute_lineage_multisignal_features.py
  ├── raw_dir (ingest/raw)
  ├── out/02_lineage_tracking/ (registry, timeseries)
  ├── out/cache_cum/partitions_cum/
  ├── out/cache_lineage/
  ├── out/metrics/
  └── out/04_front_aggregation/

multi_signal_detector.py
  ├── out/02_lineage_tracking/ (features, timeseries)
  ├── out/experiments/stage0_*/ (milestone mapping)
  └── out/experiments/multi_signal_detector/ (output)

horizon scanner scripts
  ├── out/experiments/multi_signal_detector/ (predictions)
  ├── out/assessments/ (history, estimates, reports)
  └── out/02_lineage_tracking/ (labels)
```

---

## Critical Observations for Domain Isolation

### 1. Three path domains need isolation

| Domain | Current Root | Proposed Root |
|--------|-------------|---------------|
| Ingest + slices + raw | `data/current_ingest/` | `data/{domain}/ingest/` |
| Graphs | `data/current_graphs/` | `data/{domain}/graphs/` |
| Outputs (all of `data/out/`) | `data/out/` | `data/{domain}/out/` |

### 2. Path resolution is decentralized

Each script independently resolves paths via argparse defaults. There is no
shared path resolution layer. This means:
- 32 scripts need argparse default changes
- The `--domain` arg must be added to each script individually
- A shared helper (`resolve_script_paths()`) can minimize per-script changes

### 3. Two propagation patterns exist

| Pattern | Used By | Mechanism |
|---------|---------|-----------|
| CLI subprocess args | run.py -> communities.py, run_all_metrics.py -> metric_* | Path strings in subprocess.check_call |
| Object attributes | run_build_pipeline.py -> stages 2-5 | PipelineConfig dataclass |

Both patterns need updating, but the approach differs:
- Subprocess callers: must forward domain-derived paths
- Object callers: PipelineConfig gains domain awareness

### 4. Settings file is a cross-domain collision risk

`.2yp_settings.json` stores `last_ingested_date` watermark globally. If PSC
is ingested through 2026Q1 and then CRISPR runs, the watermark gets
overwritten. Per-domain settings files (`data/{domain}/settings.json`) are
needed.

### 5. Coupling cache should be per-domain

`data/out/cache_coupling/` stores bibliographic coupling intermediates. These
are domain-specific (PSC coupling scores differ from CRISPR). Must move to
`data/{domain}/out/cache_coupling/`.

### 6. Manifest is already domain-isolated

`manifest.json` lives in `outdir/`, which is already per-domain when run with
`--outdir data/out_crispr`. After migration, it naturally goes to
`data/{domain}/out/manifest.json`.

### 7. One hardcoded path in runtime code

`scripts/metric_cross_cluster_bridging.py` line 115 hardcodes
`data/current_ingest/ingest.parquet`. This is the only case where a data path
is hardcoded in code logic (not argparse defaults). Must be changed to accept
a CLI arg or derive from `--domain`.
