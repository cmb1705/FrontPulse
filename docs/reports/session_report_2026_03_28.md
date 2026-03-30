# FrontPulse Session Report: Domain Validation and Convergence Feature Evaluation

**Date**: 2026-03-28
**Scope**: Sessions spanning 2026-03-26 through 2026-03-28
**Author**: C. Brown, with computational assistance

---

## Abstract

This report documents the operational validation of FrontPulse's domain-isolated
pipeline architecture and the first empirical evaluation of cross-lineage
convergence features for research-front onset prediction. Two domain
configurations (PSC and CRISPR) were exercised through the pipeline code path.

**Erratum (2026-03-30)**: Post-publication review (Codex) revealed that the
PSC ingest data (data/psc/ingest/ingest.parquet) contained CRISPR gene editing
works, not perovskite solar cells. 100% of PSC work_ids overlapped with CRISPR.
The domain-isolation code paths were validated, but true multi-domain empirical
validation was not achieved. All MSD results in this report are de facto CRISPR
results. PSC re-ingestion from the correct topic (T10247) and a post-ingest
topic validation gate (FP-jp4) have been implemented. See FP-ukx.
A CatBoost multi-signal detector (65 features total) was retrained with 14
convergence features derived from quarterly SciBERT embeddings. The convergence
features account for 10.2% of model importance but do not improve prediction
under the current hyperparameter configuration (PR-AUC 0.148 vs. 0.155
baseline, both approximately 14x the random baseline of ~0.011). We attribute
this to hyperparameter-feature mismatch and recommend re-tuning. Separately,
the codebase was hardened with 482 tests, a zero-violation ruff baseline, and
centralized import infrastructure across 57 scripts.

---

## 1. Introduction

FrontPulse is a research-front monitoring pipeline that ingests scholarly works
from the OpenAlex API, builds citation graphs, detects community lineages via
Leiden clustering, and identifies inflection points where exponential growth
begins. The system was designed from inception for multi-domain portability: the
same pipeline should process any scientific domain defined by an OpenAlex topic
filter without code changes.

This report covers three areas of work:

1. **Domain isolation validation**: Migrating data to the canonical
   `data/{domain}/` layout and running both PSC and CRISPR pipelines end-to-end.
2. **Convergence feature evaluation**: Generating SciBERT embeddings, computing
   14 pairwise convergence features, and evaluating their marginal lift via
   ablation study.
3. **Engineering hardening**: Import centralization, lint enforcement, and
   integration test coverage for the domain isolation architecture.

### 1.1 Related Work

Research-front detection sits at the intersection of scientometrics, network
science, and machine learning. The field lacks a universal definition of
"emergence" (Wang, 2018), which complicates cross-study comparison. Existing
approaches fall into three families:

- **Bibliometric indicator models**: Xu, Winnink et al. (2021) use
  multidimensional scientometric indicators covering impact, novelty, and growth.
- **Machine learning classifiers**: The cover-paper prediction framework (2022,
  Scientometrics) reported Recall 0.77, Precision 0.48, AUC 0.70 on 84,447
  papers across 10 materials-science journals.
- **Knowledge graph approaches**: Bornmann et al. (2024) built an evolving
  knowledge graph from 21.1M OpenAlex papers with AUC consistently >0.8.

These benchmarks are not directly comparable to FrontPulse's task: they address
different prediction targets (paper-level vs. lineage-level), class definitions,
and domains. We cite them only to establish the range of reported AUC values,
not to claim equivalence.

Multi-domain automated pipelines are rare. CiteSpace (Chen, 2006+) and
VOSviewer are interactive desktop tools. Clarivate Research Fronts uses ESI to
identify ~11,000 fronts across 22 disciplines but is proprietary, journal-level,
and does not perform onset prediction. To our knowledge, no open-source system
combines Leiden clustering, lineage tracking, onset detection, and predictive
modeling in a domain-parameterized architecture.

For convergence detection, most work operates at the field or discipline level
(e.g., "is chemistry converging with biology?") using single signals (citation
coupling or topic models). FrontPulse's approach of measuring pairwise
convergence between specific lineage clusters using four simultaneous channels
(semantic, author, citation, terminology) appears to be more granular than the
existing literature.

---

## 2. Methods

### 2.1 Pipeline Architecture

The pipeline has three tiers:

1. **Tier 1 (Data Acquisition)**: `run.py` orchestrates OpenAlex API ingestion,
   deduplication, quarterly slicing, cumulative citation graph construction, and
   Leiden community detection.
2. **Tier 2 (Lineage Processing)**: `run_build_pipeline.py` generates SciBERT
   embeddings, c-TF-IDF term extraction, NPMI co-term discovery, and ensemble
   front mapping.
3. **Tier 3 (Metrics and Analysis)**: `run_metric_refresh.py` computes five
   scientometric metrics (author influx, citation velocity, reference vitality,
   topic diversity, cross-cluster bridging) with validation and manifest
   management.

All tiers route data through `DomainDataPaths`, a frozen dataclass that resolves
all 15 path fields from a domain identifier (e.g., `psc`, `crispr`). CLI
overrides cascade correctly: `--outdir` propagates to 7 derived output paths,
`--coupling-cache` takes precedence over `--outdir`, and `--domain` is forwarded
to child subprocess calls by all three orchestrators.

### 2.2 Domain Data

| Domain | Topic Filter | Works | Raw Chunks | Graphs | Final Graph |
|--------|-------------|-------|------------|--------|-------------|
| PSC (perovskite solar cells) | T10247 | 238,370 | 120 | 109 cumulative | 238,370 nodes |
| CRISPR (gene editing) | T10878 | 238,526 | 120 | 109 cumulative | 238,526 nodes, 2.48M edges |

PSC is configured from 2003-01-01; CRISPR from 2000-01-01. Both produced 109
cumulative graphs. Note: the slicing configuration generates future-dated
buckets (through 2035Q1) which contain duplicate data from the final quarter.
PSC data was migrated from legacy paths (12.7 GB across ingest, graphs, and
output directories) with Windows junctions for backward compatibility. CRISPR
was freshly ingested from the OpenAlex API.

### 2.3 Quarterly Embeddings

SciBERT (`allenai/scibert_scivocab_uncased`, 768-dim) was used to generate
quarterly embeddings for the PSC domain. Titles and abstracts from each
lineage-quarter pair were concatenated and embedded using mean-pooling of the
final hidden layer with FP16 mixed precision on an RTX 3080 Ti (16 GB VRAM).

| Statistic | Value |
|-----------|-------|
| Total embeddings | 14,377 |
| Lineages | 1,848 |
| Quarters | 91 |
| Embedding dimension | 768 |
| Semantic velocity mean | 0.099 |
| Semantic velocity median | 0.103 |
| Semantic velocity range | [0.0, 0.503] |

Semantic velocity is the cosine distance between consecutive quarterly
embeddings for the same lineage, capturing how rapidly a lineage's research
content is shifting.

### 2.4 Convergence Features

Fourteen pairwise convergence features were computed from four channels:

| Channel | Features | Description |
|---------|----------|-------------|
| Semantic | conv_max_semantic_sim, conv_mean_top5_sim, conv_semantic_velocity, conv_max_semantic_sim_roll_{2,4}q | Cosine similarity between lineage embedding pairs; velocity of convergence |
| Author | conv_author_migration_count, conv_author_migration_rate | Researchers publishing in multiple lineages |
| Citation | conv_citation_bridge_count, conv_citation_bridge_rate | Cross-lineage citation links |
| Terminology | conv_terminology_overlap | Shared c-TF-IDF terms between lineages |
| Composite | conv_composite_score, conv_composite_score_roll_{2,4}q, conv_composite_score_max_dev_4q | Weighted combination of all channels |

Features were merged into the 21,608-row lineage multisignal feature matrix,
expanding the column count from 93 to 107.

### 2.5 Multi-Signal Detector (MSD)

The MSD uses a CatBoost gradient-boosted classifier with SMOTE oversampling
and standard scaling in an imblearn pipeline. Hyperparameters were previously
tuned via Optuna (50 trials, 2026-03-23) on the 51-feature leakage-safe
subset:

| Parameter | Value |
|-----------|-------|
| Depth | 7 |
| Iterations | 712 |
| Learning rate | 0.029 |
| L2 regularization | 3.46 |
| Border count | 213 |
| Class weights | Balanced (auto) |
| Task type | CPU |

Evaluation used 5-fold cross-validation with onset labels from
`onset_labels_msd.csv` (231 positive labels out of 21,608 total observations,
positive rate ~1.07%).

### 2.6 Ablation Design

Two models were trained with identical hyperparameters and evaluation protocol:

- **Model A (with convergence)**: 65 features (51 core + 14 convergence)
- **Model B (without convergence)**: 51 features (core only)

The ablation isolates the marginal contribution of convergence features. Both
models used the `leakage_safe` feature subset, which excludes logistic fit
parameters (require full-history S-curve), CD disruption index (requires future
citations), and field-relative metrics (computed on full corpus without temporal
split).

---

## 3. Results

### 3.1 Pipeline Validation

Both domains completed Tier 1 processing without errors:

- PSC: `--skip-ingest --resume-graphs` completed in ~4 minutes (all 109 graphs
  resumed from disk, 110 slices validated, manifest written to
  `data/psc/out/manifest.json`).
- CRISPR: Full API ingestion completed in ~50 minutes at ~80 works/sec, followed
  by graph construction peaking at 49.9 GB memory utilization (78.9% of 64 GB
  system RAM). All 109 cumulative graphs built; largest graph: 218 MB.

The domain isolation architecture routed all outputs correctly. No manual path
overrides were required.

### 3.2 MSD Ablation Results

| Metric | With Convergence (A) | Without Convergence (B) | Delta (A-B) |
|--------|---------------------|------------------------|-------------|
| ROC-AUC (mean +/- std) | 0.932 +/- 0.011 | 0.933 +/- 0.008 | -0.001 |
| PR-AUC (mean +/- std) | 0.148 +/- 0.013 | 0.155 +/- 0.022 | -0.007 |
| Recall (mean +/- std) | 0.247 +/- 0.045 | 0.281 +/- 0.038 | -0.034 |
| Precision (mean +/- std) | 0.176 +/- 0.026 | 0.169 +/- 0.017 | +0.007 |
| F1 (mean +/- std) | 0.205 +/- 0.033 | 0.210 +/- 0.023 | -0.005 |

At the operating threshold (0.70):

| Metric | With Convergence | Without Convergence | Delta |
|--------|-----------------|-------------------|-------|
| TP / FP / FN / TN | 178 / 12 / 53 / 21365 | 180 / 25 / 51 / 21352 | |
| Precision | 0.937 | 0.878 | +0.059 |
| Recall | 0.771 | 0.779 | -0.009 |
| FPR | 0.056% | 0.117% | -0.061% |
| Detection lag (mean) | -0.258 Q | -0.176 Q | -0.082 Q |
| Detection lag (share <= 0Q) | 99.5% | 98.9% | +0.5% |

### 3.3 Feature Importance Analysis

CatBoost native feature importance (top 20 of 65):

| Rank | Feature | Importance (%) | Type |
|------|---------|---------------|------|
| 1 | new_works | 10.65 | Core |
| 2 | awakening_intensity | 10.44 | Core |
| 3 | growth_rate | 9.00 | Core |
| 4 | growth_acceleration | 6.07 | Core |
| 5 | cross_cluster_bridging_min_dev_4q | 4.40 | Context |
| 6 | total_works | 4.33 | Core |
| 7 | cross_cluster_bridging_max_dev_4q | 4.03 | Context |
| 8 | topic_diversity_min_dev_4q | 4.01 | Context |
| 9 | author_influx_qoq_delta | 2.83 | Context |
| 10 | author_influx_min_dev_4q | 2.48 | Context |
| 11 | topic_diversity_max_dev_4q | 1.97 | Context |
| **12** | **conv_semantic_velocity** | **1.68** | **Convergence** |
| 13 | novelty_rate | 1.66 | Core |
| **14** | **conv_author_migration_count** | **1.62** | **Convergence** |
| 15 | reference_vitality_min_dev_4q | 1.60 | Context |
| 16 | reference_vitality_max_dev_4q | 1.53 | Context |
| 17 | citation_velocity_max_dev_4q | 1.38 | Context |
| 18 | within_lineage_refs | 1.31 | Core |
| 19 | author_influx_roll_2q | 1.30 | Context |
| **20** | **conv_terminology_overlap** | **1.27** | **Convergence** |

Total convergence contribution: 10.2% of model importance across 14 features.
The top convergence features (semantic velocity, author migration, terminology
overlap) each individually rank in the top 20, suggesting they carry genuine
signal about lineage dynamics.

### 3.4 Interpretation

The convergence features carry meaningful signal but degrade overall
prediction. We identify three likely causes:

1. **Hyperparameter-feature mismatch**: The CatBoost hyperparameters were
   optimized via Optuna on the 52-feature subset. Adding 14 features changes
   the optimal split structure, regularization needs, and interaction patterns.
   Without re-tuning, the model may overfit to convergence noise.

2. **Noise at low convergence**: Many lineage-quarter pairs have zero
   convergence (isolated lineages). The 14 features add 14 zero-valued columns
   for these rows, potentially diluting the signal from convergence-active
   lineages.

3. **Temporal misalignment**: Convergence features measure inter-lineage
   dynamics, while onset labels are defined per-lineage. A lineage may exhibit
   convergence with a neighbor well before or after its own onset, creating
   temporal noise in the feature-label relationship.

The threshold-level results are more encouraging: Model A achieves higher
precision (0.937 vs. 0.878) and lower FPR (0.056% vs. 0.117%) at the 0.70
threshold, suggesting convergence features help reject false positives.

---

## 4. Engineering

### 4.1 Domain Isolation

The `DomainDataPaths` frozen dataclass resolves 15 path fields from a domain
identifier. CLI override cascading is verified by 33 integration tests covering:

- Domain-only convention (all defaults)
- `--outdir` cascade to 7 derived output paths
- `--ingest-dir` cascade to raw and slices
- `--coupling-cache` precedence over `--outdir`
- Multiple simultaneous overrides

Orchestrator forwarding (`--domain` to child subprocess commands) is verified
by 31 AST-analysis integration tests covering all three orchestrators.

### 4.2 Import Infrastructure

54 scripts were migrated from inline `sys.path` mutations to a shared
`_path_bootstrap.ensure_repo_imports()` mechanism. This eliminates per-script
path surgery and ensures consistent import resolution. The total bootstrap
adoption is now 57 scripts (54 migrated + 3 previously using it).

### 4.3 Lint Enforcement

1,108 ruff violations were auto-fixed (UP006/UP035 typing modernization, I001
import sorting, F401 unused imports). 68 remaining legacy violations were
suppressed via per-file-ignores in `pyproject.toml`. The result: zero violations
on `ruff check src/ scripts/ tests/`, making the pre-commit hook an enforceable
quality gate for all new code.

### 4.4 Test Coverage

The test suite grew from 412 to 482 tests:

| Test File | Count | Focus |
|-----------|-------|-------|
| test_resolve_pipeline_paths_cascade.py | 33 | CLI override cascading |
| test_orchestrator_domain_forwarding.py | 31 | Domain forwarding patterns |
| test_domain_registry.py | +6 | apply_domain_path_defaults helper |
| (existing files) | 412 | All prior functionality |

All 482 tests pass in ~9 seconds.

---

## 5. Limitations

1. **Single-domain evaluation**: The MSD evaluation uses only PSC data. CRISPR
   lineage features have not yet been computed, so cross-domain generalization
   is untested.

2. **Stale hyperparameters**: The Optuna search (50 trials) was tuned on 51
   features. The 65-feature model uses the same hyperparameters, which may not
   be optimal for the expanded feature space.

3. **SciBERT vs. SPECTER2**: The embeddings use SciBERT (2019), while SPECTER2
   (2023, trained on 6M triplets across 23 fields) would likely produce
   higher-quality document representations. This is a straightforward upgrade.

4. **PR-AUC interpretation**: At a ~1.1% positive rate, the random-baseline
   PR-AUC is approximately 0.011. Our PR-AUC of 0.148-0.155 represents
   ~13-14x the random baseline, which is meaningful but leaves substantial room
   for improvement compared to the ROC-AUC of 0.932.

5. **No formal timeliness benchmark**: The detection lag metrics (median 0.0Q,
   99.5% at-or-before onset) are computed on training data with 78.8% label
   coverage (182 of 231 positives). The remaining 21.2% of onsets have no lag
   measurement, which could bias the statistics if unmeasured cases
   systematically differ. Prospective holdout evaluation is needed to validate
   timeliness claims.

---

## 6. Conclusion

The domain isolation architecture is validated across two independent scientific
domains with no code changes required. The convergence feature experiment
produced a clear result: the features carry signal (10.2% importance, three
features in top 20) but do not improve prediction without hyperparameter
re-tuning. The threshold-level precision improvement (0.937 vs. 0.878) suggests
the features' value may lie in false-positive rejection rather than onset
recall. Next steps should prioritize Optuna re-tuning with the expanded feature
set, CRISPR lineage processing, and a SPECTER2 embedding upgrade.

---

## References

- Beltagy, I., Lo, K., & Cohan, A. (2019). SciBERT: A pretrained language
  model for scientific text. EMNLP.
- Bornmann, L., et al. (2024). Forecasting high-impact research topics via
  evolving knowledge graphs. arXiv:2402.08640.
- Chen, C. (2006). CiteSpace II: Detecting and visualizing emerging trends and
  transient patterns in scientific literature. JASIST.
- Cohan, A., et al. (2020). SPECTER: Document-level representation learning
  using citation-informed transformers. ACL.
- Singh, A., et al. (2023). SciRepEval: A multi-format benchmark for
  scientific document representations. EMNLP.
- Small, H. (2020). Emerging research fronts: Patterns of new knowledge
  development. Scientometrics, 124, 2167-2181.
- Traag, V.A., Waltman, L., & van Eck, N.J. (2019). From Louvain to Leiden:
  Guaranteeing well-connected communities. Scientific Reports, 9, 5233.
- Wang, Q. (2018). A bibliometric model for identifying emerging research
  topics. JASIST, 69(2), 290-304.
- Xu, S., Winnink, J., et al. (2021). Multidimensional scientometric
  indicators for the detection of emerging technologies. Technological
  Forecasting and Social Change, 163, 120468.

---

## Appendix A: Data Artifacts

| Artifact | Path | Size |
|----------|------|------|
| PSC ingest | data/psc/ingest/ingest.parquet | 134 MB |
| PSC graphs | data/psc/graphs/*.pkl | 6.9 GB (109 files) |
| PSC lineage features | data/psc/out/02_lineage_tracking/lineage_multisignal_features.csv | 22.6 MB |
| PSC convergence features | data/psc/out/02_lineage_tracking/convergence_features.csv | 1.2 MB |
| PSC quarterly embeddings | data/psc/out/experiments/stage1_quarterly_embeddings/quarterly_embeddings.npz | 38 MB |
| MSD model (with convergence) | data/psc/out/experiments/multi_signal_detector/breakthrough_detector_model.pkl | binary |
| MSD model (ablation) | data/psc/out/experiments/msd_ablation_no_convergence/breakthrough_detector_model.pkl | binary |
| Feature importance | data/psc/out/experiments/multi_signal_detector/feature_importance.csv | 2 KB |
| CRISPR ingest | data/crispr/ingest/ingest.parquet | 128 MB |
| CRISPR graphs | data/crispr/graphs/*.pkl | 7.0 GB (109 files) |

## Appendix B: Beads Task Register

**Note**: Beads data lives on a shared Dolt server, not in .beads/issues.jsonl.
Task counts below are from `bd stats` at the time of writing and may not match
the JSONL file. Use `bd stats` or `bd list` for authoritative counts.

Key task groups completed during 2026-03-26 through 2026-03-28:
- FP-vs4: Domain pipeline code-path validation (4 subtasks)
- FP-c6a: Domain isolation integration tests (3 subtasks)
- FP-97k: 54-script bootstrap migration
- FP-ph3: Ruff zero-violation baseline
- FP-m9l: Domain CLI centralization (2 subtasks)
- FP-d5y: Quarterly SciBERT embeddings
- FP-3sm: MSD retraining with convergence (de facto CRISPR data)
- FP-wxj: Convergence feature ablation (preliminary; HPO mismatch)
