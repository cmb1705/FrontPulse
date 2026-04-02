# FrontPulse Next-Stage Research Plan

> **Execution note**: Save this plan to `frontpulse_next.md` in the repo root, then create beads epics/tasks for Phases 0-3.

## Context

FrontPulse (formerly 2YP) is a field-agnostic bibliometric pipeline that ingests OpenAlex data, builds citation graphs, detects community lineages via Leiden clustering, maps lineages to research fronts, and attempts to detect inflection points (onset of exponential growth). The pipeline was validated on perovskite solar cells (PSC) as a case study.

**What happened**: The inflection detector (LightGBM + isotonic calibration) achieved 95.7% recall / 83.6% precision in cross-validation, but suffered a 93% degradation in time-forward holdout (1.1% recall). Post-mortem analysis revealed:
- **Label circularity**: logistic fitting used both for labeling AND as model features
- **Midpoint detection** instead of onset: the system detects the S-curve midpoint (requires future data to fit), making real-time prediction incoherent
- **Community instability**: median lineage lifespan of 2 quarters, making per-lineage tracking impractical
- **Core-features-only ablation** outperformed full 55-feature model (97.6% vs 83.6% precision)

**What we want**: Transform the retrospective detection framework into a prospective early-warning system with publishable methodology.

---

## First Pass Scope (Ralph Loop Execution)

**Phases 0-3 only** (~18 days, 25 tasks). This is the critical path that transforms the system from retrospective midpoint detection to prospective onset detection at the front level.

Phases 4-7 (validation, feature pruning, BERTopic, dashboard) follow in subsequent passes once the core detection architecture is proven.

**Ralph loop strategy**: Each phase becomes a beads epic. Within each epic, tasks are created with dependencies. The ralph loop picks up ready tasks, executes them, closes them, and moves to the next. Context compaction is expected -- each task should be self-contained enough to resume after compaction via `bd prime`.

---

## Assessment of 6 Goals

### Goal 1: Onset Detection (replaces Midpoint) -- IMPLEMENT
**Status**: Feasible, high priority, genuine literature gap (no paper compares onset vs inflection for research fronts)

**What changes**:
- Replace logistic midpoint labeling with derivative-based onset criteria
- Onset = first quarter of sustained acceleration (N consecutive quarters above threshold)
- Eliminates label circularity (onset defined by past data only, no future leakage)
- Detection lag becomes meaningful (onset-to-midpoint gap measurable)

**Key insight from research**: CDC EARS C1/C2/C3 algorithms map directly to quarterly publication monitoring. Farrington Flexible method has highest sensitivity/specificity in epidemiological literature.

### Goal 2: BOCPD Integration -- IMPLEMENT (at front level)
**Status**: Feasible ONLY at front level or stable-lineage level, NOT at raw lineage level

**Why**: BOCPD requires minimum ~6-8 data points for reliable detection. Median lineage lifespan is 2 quarters. Front-level time series are continuous (20+ years of quarterly data).

**Best package**: `changepoint-online` v1.2.1 (Lancaster group, FOCuS algorithm, native Poisson change-in-rate, O(log n)). Alternative: `hildensia/bayesian_changepoint_detection` (~750 GitHub stars, PyTorch, GPU support).

**Conjugate priors**: Poisson-Gamma for count data; consider Negative Binomial for overdispersed counts.

### Goal 3: Unit of Analysis -- IMPLEMENT Strategy B (front-level)
**Status**: Strategy B (front-level tracking) is the practical path

- **Strategy A** (filter to stable lineages): Good proof-of-concept, ~200-500 lineages with 5+ quarters. Implement as stepping stone.
- **Strategy B** (front-level): Fronts smooth over lineage deaths/splits. Already partially implemented via `aggregate_lineages_to_fronts.py`. This is the deployment target.
- **Strategy C** (decouple from community detection): PhD-thesis-level redesign. Out of scope but BERTopic dynamic topic modeling could serve as parallel track for comparison.

### Goal 4: Community Instability Mitigation -- IMPLEMENT (incremental improvements)
**Status**: Partially addressable without architectural overhaul

**Approaches** (ranked by effort):
1. **ECG ensemble clustering** (`partition-igraph`): Drop-in replacement for single-run Leiden. Reduces partition volatility by averaging over multiple runs.
2. **Multi-resolution analysis** (`PyGenStability`): Principled resolution parameter selection instead of fixed 0.001.
3. **BERTopic comparison track**: Run BERTopic dynamic topic modeling on abstracts, compare topic stability vs Leiden community stability. If topics are more stable, consider hybrid approach.

**Not addressable**: Fundamental instability from quarterly citation graph snapshots. Cumulative graphs (already used) partially mitigate this.

### Goal 5: Validation Gaps -- IMPLEMENT (automated approaches)
**Status**: LLM-based labeling makes expert campaigns feasible without human experts

**Research finding**: GPT-4 achieves 88.4% agreement with human annotators vs 86% human inter-annotator agreement, at ~7x lower cost. With active learning: ~42x cheaper.

**Automated approaches**:
- **LLM-based milestone validation**: Use Claude/GPT-4 to classify whether detected inflections correspond to real scientific breakthroughs (replacing 3-rater expert panel)
- **Cross-domain generalization**: OpenAlex is fully parameterizable. Best validation fields: CRISPR (1987-2020, well-documented milestones), graphene (2004 Nobel), quantum computing (4 phases)
- **Automated milestone extraction**: Wikipedia timelines + Wikidata provide machine-readable milestone catalogs. Nobel Laureate Publication Dataset (Li et al. 2019, Scientific Data) for curated histories.

### Goal 6: Retrospective Lessons -- INFORM (no implementation, guides other goals)
**Status**: Lessons already captured. They inform the implementation priorities above.

Key lessons driving this plan:
- Complexity did not correlate with value (20 features > 55 features)
- Label circularity was the root cause of holdout collapse
- The most informative experiments were failures
- Unit of analysis was unstable the whole time

---

## What's Missing (additions to the 6 goals)

### M1: Evaluation Framework Overhaul
The current metrics (precision/recall/F1) don't measure what matters for an early-warning system: **timeliness**.

**Add**: NAB (Numenta Anomaly Benchmark) scoring -- linear decay reward in detection window, 0-100 scale. Also: Average Run Length (ARL0 for false alarm rate, ARL1 for detection speed), Expected Detection Delay (EDD).

### M2: Feature Simplification
The 55-feature model underperformed the 20-feature model. The plan should include aggressive feature pruning based on the ablation results, not just label-circularity fixes.

### M3: Reproducibility and Publication Target
Target journals: Quantitative Science Studies (IF 3.5, MIT Press, open access) or Journal of Informetrics. Required for publication: FAIR-compliant data artifacts, baseline comparisons (already done), ablation studies (partially done), cross-domain validation (new).

### M4: Dashboard/Monitoring Interface
For practical deployment, need a lightweight dashboard showing front-level alerts. Streamlit is the pragmatic choice (Python-native, minimal frontend complexity).

---

## What's Unrealistic (scoped out)

1. **Strategy C (decouple from community detection)**: PhD-thesis-level redesign. Acknowledged but deferred.
2. **Full 3-rater expert campaign**: 3-6 months calendar time, requires domain experts. Replaced by LLM-based automated labeling.
3. **Causal milestone analysis (DiD/IV/synthetic control)**: Requires clean causal identification that doesn't exist in observational bibliometric data. Deferred.
4. **Real-time streaming deployment**: The quarterly cadence of OpenAlex snapshots makes true real-time impossible. Quarterly batch processing is the natural unit.

---

## Implementation Plan (Beads Epic Structure)

### Phase 0: Foundation (prerequisites)
**Goal**: Fix known bugs, establish clean baseline for experiments

| Task | Description | Files | Effort |
|------|-------------|-------|--------|
| 0.1 | Fix unsafe deserialization (2YP-093) | `scripts/multi_signal_detector.py` | 0.5d |
| 0.2 | Remove personal contact info (2YP-8po) | `config/datasources.yaml` | 0.25d |
| 0.3 | Eliminate hardcoded paths (2YP-lle) | 6 scripts | 0.5d |
| 0.4 | Update README heading to FrontPulse | `README.md` | 0.1d |
| 0.5 | Run full test suite, fix any failures | `tests/` | 0.5d |

### Phase 1: Onset Relabeling (Goal 1)
**Goal**: Replace logistic midpoint labels with derivative-based onset labels

| Task | Description | Files | Effort |
|------|-------------|-------|--------|
| 1.1 | Implement onset labeling function | `scripts/label_inflection_points.py` | 1d |
| 1.2 | CDC EARS C2 algorithm adaptation | New: `scripts/onset_detector.py` | 1d |
| 1.3 | Generate onset labels for full corpus | Run labeling pipeline | 0.5d |
| 1.4 | Compare onset vs midpoint label distributions | Analysis script | 0.5d |
| 1.5 | Update feature computation to exclude logistic features | `scripts/compute_lineage_multisignal_features.py` | 0.5d |
| 1.6 | Retrain MSD with onset labels + pruned features | `scripts/multi_signal_detector.py` | 0.5d |
| 1.7 | Run time-forward holdout with onset labels | Evaluation | 0.5d |

**Onset definition**: First quarter where rolling 3Q growth rate exceeds field-relative threshold AND next 2 quarters maintain positive acceleration. Tunable parameters: rolling window (2-4Q), threshold multiplier (1.0-2.0 sigma above field median), confirmation window (1-3Q).

**Key functions to modify**:
- `label_inflection_points.py:fit_logistic()` (L206-236): Replace with onset detection
- `label_inflection_points.py:derivative_detection()` (L239-262): Adapt as primary method
- `compute_lineage_multisignal_features.py:compute_growth_features()` (L178): Remove `logistic_*` features from model input

### Phase 2: Stable Lineage Filtering + Front-Level Tracking (Goals 3, 4)
**Goal**: Implement Strategy A (proof-of-concept) then Strategy B (deployment target)

| Task | Description | Files | Effort |
|------|-------------|-------|--------|
| 2.1 | Lineage stability filter (min N quarters) | New: `scripts/filter_stable_lineages.py` | 0.5d |
| 2.2 | ECG ensemble clustering integration | `src/community.py`, `scripts/communities.py` | 1.5d |
| 2.3 | Stability metrics dashboard (VI, NMI, lifespan distributions) | New: `scripts/community_stability_report.py` | 1d |
| 2.4 | Front-level onset detection pipeline | Extend `scripts/aggregate_lineages_to_fronts.py` | 1d |
| 2.5 | Front-level feature computation | Extend feature pipeline | 1d |
| 2.6 | Compare lineage-level vs front-level detection performance | Analysis | 0.5d |

**Key existing code to reuse**:
- `aggregate_lineages_to_fronts.py` (120 lines): Already sums new_works per front per quarter
- `alignment.py:variation_of_information()` (L107-158): Existing VI computation
- `alignment.py:pagerank_core()` (L13-44): Keep for lineage matching, add ECG on top

**ECG integration**: `partition-igraph` package provides `ECG()` function. Replace single `run_leiden()` call with ECG ensemble (10-50 runs, community co-membership matrix, final consensus partition).

### Phase 3: BOCPD Integration (Goal 2)
**Goal**: Bayesian online changepoint detection at front level

| Task | Description | Files | Effort |
|------|-------------|-------|--------|
| 3.1 | Install and benchmark `changepoint-online` | Requirements, smoke test | 0.5d |
| 3.2 | BOCPD wrapper for front-level time series | New: `scripts/bocpd_detector.py` | 1.5d |
| 3.3 | Poisson-Gamma prior calibration on historical data | Calibration script | 1d |
| 3.4 | Compare BOCPD vs MSD vs baselines | Evaluation framework | 1d |
| 3.5 | NAB scoring implementation | New: `scripts/nab_scorer.py` | 1d |
| 3.6 | Ensemble: BOCPD probability + MSD features | Hybrid model | 1d |

**Architecture**: BOCPD runs per-front (not per-lineage). Each front has 80+ quarterly observations (2003Q1-2025Q3). Poisson-Gamma conjugate prior with quarterly publication counts as observations. Output: posterior probability of changepoint at each quarter. Threshold: probability > 0.5 triggers alert.

**NAB scoring**: Implement Numenta Anomaly Benchmark scoring with configurable detection window (e.g., +/- 4 quarters around true onset). Linear reward decay within window, penalty outside.

### Phase 4: Automated Validation (Goal 5) -- LITRIS-Integrated
**Goal**: Replace manual expert validation with automated approaches, leveraging the LITRIS literature review platform

| Task | Description | Files | Effort |
|------|-------------|-------|--------|
| 4.1 | LITRIS-powered inflection validator | New: `scripts/litris_validate_inflections.py` | 1.5d |
| 4.2 | Wikipedia/Wikidata milestone scraper | New: `scripts/scrape_milestones.py` | 1d |
| 4.3 | CRISPR domain configuration + OpenAlex query | `config/datasources_crispr.yaml` | 0.5d |
| 4.4 | Run full pipeline on CRISPR corpus | Pipeline execution | 1d |
| 4.5 | Cross-domain comparison report (PSC vs CRISPR) | Analysis | 1d |
| 4.6 | Immunotherapy domain (stretch, backup validation) | `config/datasources_immunotherapy.yaml` | 1d |

**LITRIS integration approach** (reuses D:\git_repos\LITRIS components):

Key reusable components from LITRIS:
- `src/analysis/llm_factory.py`: Multi-provider LLM factory (Anthropic, OpenAI, Google, Ollama, llama.cpp)
- `src/analysis/schemas.py`: Pydantic models for structured extraction with quality_rating (1-5)
- `src/query/agentic.py`: Multi-round agentic search with gap analysis
- `src/query/deep_review.py`: Literature synthesis with citation verification (anti-hallucination)
- `src/analysis/clustering.py`: UMAP + HDBSCAN topic clustering
- `src/analysis/gap_detection.py`: Heuristic gap detection for emerging topics
- MCP tools: `litris_search_agentic`, `litris_deep_review`, `litris_similar`

Validation workflow per detected inflection:
1. **Identify papers**: Use LITRIS `SearchEngine.search()` with year_min/max around onset quarter, quality_min=4
2. **Agentic search**: Multi-round `litris_search_agentic` to find papers discussing the research front's breakthrough
3. **Deep review**: Generate integrated review via `litris_deep_review` with citation verification
4. **Quality filter**: LITRIS's quality_rating (1-5) filters to high-confidence papers only
5. **LLM classification**: Use LITRIS's `create_llm_client()` factory to classify whether the inflection corresponds to a genuine breakthrough (structured output via Pydantic schema)
6. **Synthetic inter-rater**: Run 3x with temperature variation using LITRIS's `ConsensusStrategy.MAJORITY_VOTE` from `llm_council.py`

This replaces the need for a standalone `anthropic` package dependency -- LITRIS already has multi-provider LLM support.

**Milestone scraping**:
- Wikipedia: parse "Timeline of X" articles (exist for CRISPR, graphene, quantum computing, etc.)
- Wikidata: SPARQL query for `significant_event` properties on scientific topic entities
- Nobel Prize data: nobelprize.org API for laureate publication histories
- Output: milestone CSV with (field, milestone_description, year, quarter_estimate, source_url)

**Validation domain selection** (11 fields evaluated, scored on 7 criteria out of 35):

| Rank | Field | Score | Key Advantage |
|------|-------|-------|---------------|
| 1 | **CRISPR Gene Editing** | **34/35** | Best milestone catalog (Broad Institute timeline); 30-50K papers; 10+ research fronts with staggered inflections; Nobel 2020 + FDA 2023 |
| 2 | Immunotherapy / Checkpoint Inhibitors | 32/35 | FDA approval dates as precise ground truth; long timeline (1990s-present) |
| 3 | Quantum Computing | 31/35 | Best Wikipedia timeline; diverse sub-fields; but arXiv-heavy, no Nobel |

**CRISPR is the primary validation domain** because it has staggered inflection points (2012 Cas9, 2013 eukaryotic, 2015 in vivo, 2017 base editing/diagnostics, 2019 prime editing, 2023 FDA approval) -- ideal for testing multi-event detection. The Broad Institute maintains a ready-made milestone catalog.

Ground truth sources:
- Broad Institute CRISPR Timeline (broadinstitute.org)
- Addgene CRISPR History (addgene.org/crispr/history/)
- 2020 Nobel Prize in Chemistry documentation
- 2023 FDA approval of Casgevy
- PMC review: "CRISPR: History of Discovery" (PMC9377665)

Note: PSC is not an ideal validation domain despite NREL efficiency chart (only ~15 years of history, fewer distinct research fronts than CRISPR). The PSC corpus remains valuable as the development dataset.

### Phase 5: Feature Pruning + Model Simplification (Goal 6 lessons applied)
**Goal**: Reduce model complexity based on ablation evidence

| Task | Description | Files | Effort |
|------|-------------|-------|--------|
| 5.1 | Formal ablation study (20 vs 35 vs 55 features) | `scripts/run_feature_subset_evals.py` | 0.5d |
| 5.2 | SHAP analysis with grouped correlated features | `scripts/analyze_feature_signals.py` | 0.5d |
| 5.3 | Define minimal feature set for publication | `config/features/feature_groups.yaml` | 0.25d |
| 5.4 | Retrain final model with minimal features + onset labels | `scripts/multi_signal_detector.py` | 0.5d |
| 5.5 | Publication-quality evaluation tables | Analysis/reporting | 0.5d |

**SHAP best practice**: Use `TreeSHAP` with `feature_perturbation="interventional"` and `shap.utils.hclust()` for grouped correlated features (time-series derivatives are highly correlated).

### Phase 6: BERTopic Comparison Track (Goal 4 extension)
**Goal**: Evaluate topic modeling as alternative to community detection

| Task | Description | Files | Effort |
|------|-------------|-------|--------|
| 6.1 | BERTopic dynamic topic model on PSC abstracts | New: `scripts/bertopic_comparison.py` | 1.5d |
| 6.2 | Topic stability vs community stability comparison | Analysis | 1d |
| 6.3 | Topic-level onset detection | Extend onset detector | 0.5d |
| 6.4 | Side-by-side evaluation report | Documentation | 0.5d |

**BERTopic configuration**: Use `BERTopic(embedding_model="allenai/scibert_scivocab_uncased")` with dynamic topic modeling via `.topics_over_time()`. Compare topic coherence (c_v score) and temporal stability (topic persistence across quarters) vs Leiden community stability (VI, NMI).

### Phase 7: Dashboard + Monitoring (M4)
**Goal**: Lightweight Streamlit dashboard for front-level alerts

| Task | Description | Files | Effort |
|------|-------------|-------|--------|
| 7.1 | Streamlit app skeleton (front selector, time range) | New: `app/dashboard.py` | 1d |
| 7.2 | Front-level growth curves with onset markers | Visualization | 0.5d |
| 7.3 | BOCPD probability heatmap | Visualization | 0.5d |
| 7.4 | Alert table with NAB scores | Component | 0.5d |
| 7.5 | Quarterly refresh automation (schedule script) | New: `scripts/quarterly_refresh.py` | 0.5d |

---

## Dependencies and Execution Order

```
Phase 0 (Foundation)
  |
  v
Phase 1 (Onset Relabeling)  -----> Phase 5 (Feature Pruning)
  |                                    |
  v                                    v
Phase 2 (Stable/Front-level) -----> Phase 3 (BOCPD) -----> Phase 7 (Dashboard)
  |
  v
Phase 4 (Automated Validation)

Phase 6 (BERTopic) -- independent, can run in parallel with Phases 2-5
```

**Critical path**: Phase 0 -> Phase 1 -> Phase 2 -> Phase 3 -> Phase 7
**Parallel track**: Phase 6 (BERTopic) runs independently
**Phase 4** depends on Phase 1 (onset labels) but not on Phase 3
**Phase 5** depends on Phase 1 (onset labels) + Phase 2 (front-level features)

---

## New Dependencies (requirements.txt additions)

| Package | Version | Purpose | Phase |
|---------|---------|---------|-------|
| `changepoint-online` | >=1.2 | BOCPD/FOCuS online changepoint detection | 3 |
| `partition-igraph` | >=0.0.7 | ECG ensemble community detection | 2 |
| `bertopic` | >=0.16 | Dynamic topic modeling comparison | 6 |
| `shap` | >=0.43 | TreeSHAP with interventional perturbation | 5 |
| `streamlit` | >=1.30 | Dashboard interface | 7 |
| (LITRIS) | local | LLM validation via LITRIS multi-provider factory | 4 |
| `wikipedia-api` | >=0.7 | Milestone scraping from Wikipedia | 4 |
| `SPARQLWrapper` | >=2.0 | Wikidata milestone queries | 4 |

Note: `bertopic`, `streamlit`, `anthropic` are optional dependencies (not needed for core pipeline).

---

## Beads Epic/Task Structure

### Epic: FP-FOUNDATION (Phase 0) -- Bug Fixes and Baseline
- 5 tasks (0.1-0.5)
- No dependencies
- Total effort: ~1.85 days

### Epic: FP-ONSET (Phase 1) -- Onset Relabeling
- 7 tasks (1.1-1.7), no external dependencies
- Blocked by: Phase 0 completion
- Total effort: ~5 days

### Epic: FP-STABILITY (Phase 2) -- Community Stability + Front-Level Tracking
- 6 tasks (2.1-2.6)
- Blocked by: Phase 1 (onset labels needed for evaluation)
- Total effort: ~5.5 days

### Epic: FP-BOCPD (Phase 3) -- Bayesian Changepoint Detection
- 6 tasks (3.1-3.6)
- Blocked by: Phase 2 (front-level time series needed)
- Total effort: ~6 days

### Epic: FP-VALIDATION (Phase 4) -- Automated Cross-Domain Validation
- 6 tasks (4.1-4.6)
- Blocked by: Phase 1 (onset labels)
- Can run parallel with Phases 2-3
- Total effort: ~6 days

### Epic: FP-FEATURES (Phase 5) -- Feature Simplification
- 5 tasks (5.1-5.5)
- Blocked by: Phase 1 + Phase 2
- Total effort: ~2.25 days

### Epic: FP-BERTOPIC (Phase 6) -- Topic Model Comparison
- 4 tasks (6.1-6.4)
- No dependencies (can start after Phase 0)
- Total effort: ~3.5 days

### Epic: FP-DASHBOARD (Phase 7) -- Monitoring Dashboard
- 5 tasks (7.1-7.5)
- Blocked by: Phase 3 (BOCPD results to display)
- Total effort: ~3 days

**Grand total**: ~33.1 days of implementation effort across 44 tasks

---

## Verification Plan

### Per-Phase Verification

**Phase 1**:
- Onset labels CSV generated with expected schema
- Label count comparison (onset vs midpoint: expect more onset labels, earlier timing)
- Time-forward holdout recall > 10% (improvement over 1.1% midpoint baseline)
- `pytest tests/` passes

**Phase 2**:
- ECG partitions show lower VI than single-run Leiden (quantified improvement)
- Front-level time series generated for all mapped fronts
- Stable lineage filter produces 200-500 lineages with 5+ quarter histories

**Phase 3**:
- BOCPD detects known PSC milestones (e.g., 2012Q3 Miyasaka breakthrough)
- NAB score computed for all fronts
- Comparison table: BOCPD vs MSD vs baselines on same evaluation set

**Phase 4**:
- LLM validation kappa >= 0.50 (self-consistency across 3 runs)
- CRISPR pipeline runs end-to-end without PSC-specific code
- At least 10 milestones scraped per domain from Wikipedia/Wikidata

**Phase 5**:
- Minimal feature set defined (expect 12-20 features)
- SHAP summary plot generated
- Final model precision >= 90% with onset labels

**Phase 6**:
- BERTopic produces coherent topics (c_v > 0.4)
- Topic persistence measured across quarters
- Side-by-side comparison with Leiden communities

**Phase 7**:
- Streamlit app launches without errors
- Front selector shows all mapped fronts
- Growth curves render with onset markers
- BOCPD probability heatmap renders

### End-to-End Verification
```powershell
# Full test suite
pytest -v

# Onset labeling
python scripts/label_inflection_points.py --mode onset --out data/out/02_lineage_tracking/onset_labels.csv

# Feature computation (pruned)
python scripts/compute_lineage_multisignal_features.py --exclude-logistic-features

# Model training
python scripts/multi_signal_detector.py --labels onset --use-cv --cv-folds 5

# BOCPD detection
python scripts/bocpd_detector.py --timeseries data/out/04_front_aggregation/front_timeseries_delta_long.csv

# Cross-domain
python run.py --config config/datasources_crispr.yaml --schema config/schema.yaml --slices config/slices.yaml --skip-ingest

# Dashboard
streamlit run app/dashboard.py
```

---

## Key Files to Modify

| File | Changes |
|------|---------|
| `scripts/label_inflection_points.py` | Add onset detection mode, CDC EARS adaptation |
| `scripts/compute_lineage_multisignal_features.py` | Flag to exclude logistic features, front-level features |
| `scripts/multi_signal_detector.py` | Replace unsafe serialization with joblib, onset label support |
| `src/community.py` | ECG ensemble clustering integration |
| `scripts/communities.py` | ECG option, stability metrics |
| `src/alignment.py` | Enhanced stability tracking |
| `scripts/aggregate_lineages_to_fronts.py` | Front-level feature computation |
| `config/multisignal_config.yaml` | Onset parameters, feature flags |
| `requirements.txt` | New dependencies |

## New Files to Create

| File | Purpose |
|------|---------|
| `scripts/onset_detector.py` | CDC EARS-adapted onset detection |
| `scripts/bocpd_detector.py` | BOCPD wrapper for front-level detection |
| `scripts/nab_scorer.py` | NAB timeliness scoring |
| `scripts/filter_stable_lineages.py` | Lineage stability filter |
| `scripts/community_stability_report.py` | VI/NMI stability dashboard |
| `scripts/llm_validate_inflections.py` | LLM-based validation |
| `scripts/scrape_milestones.py` | Wikipedia/Wikidata milestone extraction |
| `scripts/bertopic_comparison.py` | BERTopic dynamic topic modeling |
| `config/datasources_crispr.yaml` | CRISPR domain configuration |
| `app/dashboard.py` | Streamlit monitoring dashboard |
