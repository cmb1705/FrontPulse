# Implementation Plan: Precursor and Recombination Graphs

**Source**: [precursor_recombination_graph.md](precursor_recombination_graph.md)
**Status**: Planned (P4 -- blocked on current P0-P3 work)
**Beads Epic**: FP-prg (see beads for subtasks)
**Date**: 2026-03-30

## Prerequisites

This work is blocked until the following are complete:

| Blocker | ID | Why it must finish first |
|---------|----|-------------------------|
| Convergence ablation re-run | FP-07w (P0) | Determines whether conv signal is worth preserving in richer form |
| CRISPR Tier 2+3 pipeline | FP-t9g (P1) | Cross-domain motif validation needs two complete domains |
| SPECTER2 upgrade | FP-jtc (P2) | Embedding quality directly affects semantic convergence edges |
| Prospective holdout evaluation | FP-043 (P3) | Validates baseline detector before building on top of it |

## Phase 1: Proposal Refinement

**Goal**: Harden the draft proposal into a publication-ready research design document.

### Deliverables
- Finalized conceptual model with formal notation
- Concrete artifact schema (edge store columns, primary keys, storage format)
- Defined lag window parameters (justified, not arbitrary)
- Scoring method selection (event enrichment + permutation vs. hazard model)
- Domain selection for first feasibility pass (PSC, CRISPR, or both)
- Updated success criteria with quantitative thresholds where possible
- Risk mitigation plan with concrete guardrails

### Key decisions to make
1. **Target granularity**: Score precursor edges against lineage onsets, front onsets, or both?
2. **Edge sparsity threshold**: Top-k neighbors (k=5? 10? 20?) and minimum interaction strength
3. **Lag window**: 2-8 quarters (proposal default) vs. adaptive by domain age
4. **Directionality**: Temporal precedence only, or also asymmetric channel rules?
5. **Negative precursors**: Allow competitive/inhibitory relationships, or only constructive?
6. **BOCPD integration**: Treat changepoints as auxiliary targets alongside onset labels?

### Acceptance criteria
- Proposal document is self-contained and reviewable without supplementary explanation
- All "Open Questions" from the draft have documented answers or explicit deferral rationale
- Schema definitions are concrete enough to implement directly

## Phase 2: Feasibility Study and Literature Review

**Goal**: Determine whether the approach is technically viable and situate it in the literature.

### Literature review scope
1. Temporal knowledge recombination in science of science
2. Dynamic innovation networks and lead-lag structures
3. Temporal motif mining in citation and collaboration networks
4. Null models for temporal interaction networks with varying edge volume
5. Mechanism vs. association claims in observational network studies

### Feasibility study scope
1. **Edge volume estimation**: How many non-trivial edges per quarter across channels? What is the pairwise explosion factor at lineage level?
2. **Signal density**: What fraction of edges show any activity? How sparse is the top-k semantic neighbor set?
3. **Convergence artifact audit**: Can current `compute_convergence_features.py` emit edge-level data without major refactoring, or does it need structural changes?
4. **Onset coverage**: How many onset events have sufficient preceding convergence data (>= 4 quarters of edge history)?
5. **Storage budget**: Projected Parquet size for edge store at PSC scale (5,179 lineages, 109 quarters)

### Deliverables
- Literature review memo (5-10 pages, annotated bibliography)
- Feasibility report with quantitative findings from items 1-5
- Go/no-go recommendation with stated conditions

### Acceptance criteria
- Literature review covers >= 15 relevant publications
- Feasibility report includes concrete numbers (edge counts, storage, onset coverage)
- Clear recommendation on whether to proceed, pivot, or abandon

## Phase 3: Edge Persistence Layer

**Goal**: Extend the convergence pipeline to emit sparse quarterly edge artifacts.

### Implementation

**New artifact**: `data/{domain}/out/02_lineage_tracking/convergence_edges.parquet`

**Schema**:
```
quarter: str (YYYYQN)
source_lineage_id: int
target_lineage_id: int
semantic_sim: float  (top-k only)
semantic_rank: int
author_overlap_count: int
author_overlap_rate: float
citation_bridge_count: int
citation_bridge_rate: float
terminology_overlap: float
composite_score: float
n_active_channels: int
```

**Primary key**: `(quarter, source_lineage_id, target_lineage_id)`

### Design constraints
- Sparse: only top-k semantic neighbors + edges with non-trivial interaction in >= 1 channel
- Lineage-level first; front roll-up is a separate downstream step
- Parquet for schema stability and columnar efficiency
- Must not break existing `conv_*` node-level output (additive artifact)

### Modifications
- `scripts/compute_convergence_features.py`: Add `--emit-edges` flag to persist pairwise data
- `src/convergence.py`: Factor out edge-level computation from node-level aggregation
- Domain-aware output path via existing `resolve_script_paths`

### Acceptance criteria
- Edge artifact generated for at least one domain (PSC or CRISPR)
- Existing `conv_*` features unchanged (regression test)
- Edge store passes schema validation and primary key uniqueness

## Phase 4: Temporal Precursor Scoring

**Goal**: Score lagged source-target relationships against onset events.

### Approach (initial: event enrichment + permutation)

For each candidate pair `(i, j)` and lag `tau` in `[2, 8]` quarters:
1. Identify quarters where target `j` enters onset
2. Check whether edge `(i, j)` was active at `t - tau`
3. Compare observed frequency against matched non-onset control windows
4. Compute enrichment ratio and permutation p-value

### New artifacts
- `data/{domain}/out/05_precursor_graph/precursor_scores.parquet`
- Columns: `source_id, target_id, channel, lag_quarters, enrichment_ratio, p_value, n_onset_events`

### New code
- `scripts/score_precursor_edges.py`
- `src/precursor_scoring.py` (core statistical logic)

### Acceptance criteria
- Precursor scores computed for at least one domain
- Enrichment ratios distinguishable from null at p < 0.05 for at least some edges
- Results reproducible with fixed random seed

## Phase 5: Motif Discovery

**Goal**: Identify recurring multi-source patterns preceding emergence.

### Motif classes (ordered by complexity)
1. **Single precursor**: `A -> C` (one source leads one target)
2. **Two-source recombination**: `(A, B) -> C` (two sources jointly precede target)
3. **Channel complementarity**: semantic + author leading same target
4. **Serial chain**: `A -> B -> C` (transitive precursor path)

### Approach
- Start with dyadic motifs (classes 1-2)
- Enumerate candidate motifs from precursor score table
- Score motif recurrence across onset events within domain
- Test motif stability across domains (Phase B validation)

### New artifacts
- `data/{domain}/out/05_precursor_graph/motif_catalog.parquet`
- `data/{domain}/out/05_precursor_graph/motif_summary.json`

### New code
- `scripts/find_precursor_motifs.py`
- `src/motif_mining.py`

### Acceptance criteria
- At least one recurring motif family identified across multiple onset events
- Motif catalog includes lag, channel, and confidence information

## Phase 6: Explanation Surface and Reporting

**Goal**: Produce human-readable mechanism maps and summary cards.

### Outputs
- Precursor rankings per target node
- Motif catalog with lags and channels
- Front-level mechanism maps (lineage motifs aggregated to curated fronts)
- Mechanism cards (short summaries per onset event)

### New code
- `scripts/generate_precursor_report.py`
- Extension to `src/quarterly_report.py` or standalone `src/precursor_report.py`

### Acceptance criteria
- Mechanism cards generated for all onset events with sufficient precursor data
- Report includes at least one cross-domain comparison (if Phase B complete)

## Phase 7: Retrospective Validation

**Goal**: Validate precursor structure against null models.

### Phase A: Within-domain
- Extract 2-8 quarter windows before onset events
- Score precursor edges and motifs against matched non-onset controls
- Report enrichment ratio, temporal concentration, permutation significance

### Phase B: Cross-domain motif stability
- Learn motif families in one domain
- Test whether same channel patterns recur in the other
- Goal: **pattern transfer**, not node-level transfer

### Phase C: Predictive utility (optional)
- Add precursor-derived features to MSD
- Test whether precursor density improves early-warning ranking

### Acceptance criteria
- Phase A: enrichment above null for top precursor edges
- Phase B: at least one motif family recurs across domains
- Phase C: clear answer on whether precursor features add detector value

## Estimated Effort

| Phase | Sessions | Dependencies |
|-------|----------|-------------|
| 1: Proposal refinement | 1 | None (can start once epic unblocked) |
| 2: Lit review + feasibility | 1-2 | Phase 1 |
| 3: Edge persistence | 1-2 | Phase 2 go decision |
| 4: Precursor scoring | 2 | Phase 3 |
| 5: Motif discovery | 1-2 | Phase 4 |
| 6: Reporting | 1-2 | Phase 5 |
| 7: Validation | 1-2 | Phases 4-6 + two domains with onset labels |
| **Total** | **8-13** | |

## Risk Register

| Risk | Impact | Mitigation |
|------|--------|-----------|
| Causal overclaim | Credibility | Use "precursor" / "lead-lag" language only |
| Field-wide growth confound | False signal | Matched non-onset controls + field-growth controls |
| Lineage instability | Noisy edges | Score at lineage level, report by stability cohort |
| Pairwise explosion | Scale | Top-k sparsity, Parquet, defer higher-order motifs |
| Mapping coverage gaps | Interpretation | Front roll-up is interpretation layer, not analysis layer |
| Convergence signal too weak | Wasted effort | Phase 2 feasibility gate before implementation |
