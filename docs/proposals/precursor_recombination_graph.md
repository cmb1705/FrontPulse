# Proposal: Precursor and Recombination Graphs for Research-Front Emergence

**Status**: Draft
**Date**: 2026-03-30
**Priority**: P2 research track after current domain-repair and cross-domain execution work

## Problem Statement

FrontPulse currently treats emergence primarily as a **node-level detection**
problem: given a lineage-quarter or front-quarter record, predict whether that
entity is entering onset. That framing is useful, but it leaves a major gap.
The system can say that a lineage or front appears to be emerging, but it
cannot say **which upstream areas made that emergence possible**.

This matters because many research fronts do not emerge in isolation. They
emerge when previously separate areas become connected through:

- growing semantic similarity,
- shared authors moving between areas,
- citation bridges across clusters,
- and shared terminology or problem framing.

FrontPulse already computes these interactions as quarterly convergence signals,
but it collapses them into per-lineage summary columns (`conv_*`) for classifier
input. That loses the most interesting structure: **who is converging with
whom, in what sequence, and how far ahead of onset**.

The proposed direction is to keep that structure and make it first-class.

## Proposed Capability

Build a **dynamic precursor / recombination graph** that models emergence as a
temporal network process rather than only a node-level classification problem.

In this graph:

- **Nodes** are lineages first, with optional roll-up to curated fronts.
- **Edges** represent quarterly cross-lineage interaction signals across the
  four existing convergence channels: semantic similarity, author migration,
  citation bridging, and terminology overlap.
- **Directed precursor edges** encode temporal precedence: activity in source
  area `A` tends to occur before onset in target area `B` by `tau` quarters.
- **Motifs** capture multi-source recombination patterns, such as two distinct
  precursor areas jointly preceding emergence in a third area.

The goal is not only to flag what is emerging, but to identify the **mechanism
of emergence**:

- which upstream areas are acting as precursors,
- which channels are most informative,
- whether the pattern is reuse, recombination, migration, or citation
  integration,
- and whether similar precursor motifs recur across domains.

## Core Research Questions

1. Do onset events tend to follow identifiable cross-lineage precursor patterns
   rather than appearing as isolated node-level events?
2. Which interaction channels lead earliest and most reliably: semantic,
   author, citation, or terminology?
3. Are there recurrent precursor motifs, such as two-source recombination,
   that appear across multiple emergence events?
4. Does preserving the pairwise structure of convergence signals yield better
   explanatory power than collapsing them into scalar `conv_*` features?
5. Can precursor motifs become a reusable prior for later discovery-oriented
   work, ranking not just what is emerging but what combinations are likely to
   produce the next wave of emergence?

## Why This Is Different

This is not just another detector tuning pass.

It changes the object of study from:

- "Which node is entering onset?"

to:

- "Which prior network interactions make onset likely, and how do new fronts
  get assembled from existing ones?"

That is a qualitatively different claim.

It also differs from the broad-domain generative discovery proposal:

- **Generative discovery** asks: "Which unflagged areas should we watch?"
- **Precursor graphs** ask: "What upstream structure systematically produces
  emergence?"

The two ideas are compatible, but they are not the same. Generative discovery
is a ranked-output capability. Precursor graphs are a **mechanism-learning**
capability.

## Why Now

This direction is newly plausible because the repo already contains most of the
substrate:

- domain-aware path routing and multi-domain execution,
- onset labels and a prospective evaluation contract,
- front-level quarterly contracts and BOCPD infrastructure,
- and convergence utilities that already compute the relevant interaction
  channels.

The main missing step is architectural, not conceptual:

- stop discarding the pairwise convergence structure too early,
- persist it as an edge-level quarterly artifact,
- and score lead-lag relationships against onset outcomes.

This also directly addresses a weakness already visible in the convergence
ablation work: the `conv_*` features appear to carry signal, but some of that
signal may be diluted when reduced to zero-heavy scalar summaries. The proposal
tests whether the real value of convergence is not just as a tabular feature,
but as a temporal interaction graph.

## Conceptual Model

Let:

- `V` be the set of lineages (or fronts),
- `t` index quarters,
- `C` be the set of interaction channels
  `{semantic, author, citation, terminology}`,
- `E_t(i, j, c)` be the interaction strength between nodes `i` and `j` in
  quarter `t` for channel `c`,
- `y_j,t` indicate whether node `j` enters onset at quarter `t`.

The precursor problem is to estimate whether some interaction pattern involving
node `i` at quarter `t - tau` tends to precede onset in node `j` at quarter
`t`, where `tau` is a lead window such as 2-8 quarters.

This creates three levels of structure:

1. **Interaction edges**:
   undirected or symmetric channel scores observed within a quarter.
2. **Directed precursor edges**:
   source-target relationships inferred from temporal precedence.
3. **Motifs**:
   higher-order patterns such as `(A, B) -> C`, where two precursor areas jointly
   precede emergence in a target.

The proposal is explicitly about **precursor structure**, not strict causal
identification. The output should be interpreted as "historically leading and
mechanistically suggestive," not "proven causal."

## Concrete Example

Suppose lineage `C` enters onset in 2017Q2.

During the prior 4-6 quarters:

- lineage `A` becomes one of `C`'s closest semantic neighbors,
- authors begin publishing in both `A` and `C`,
- papers in `C` increasingly cite both `A` and lineage `B`,
- and the terminology overlap between `B` and `C` rises sharply.

The current pipeline would mostly compress this into a handful of
lineage-quarter `conv_*` values attached to `C`.

Under this proposal, the system would instead produce a mechanism statement like:

> Emergence in `C` was preceded by a two-source recombination pattern from
> `A` and `B`, with strongest evidence in semantic convergence and author
> migration, peaking 3-5 quarters before onset.

That output is much closer to the actual scientific question:

- not only whether an area is emerging,
- but how new areas are assembled from prior knowledge components.

## Relation to the Existing FrontPulse Stack

This proposal builds on current components rather than replacing them.

### Existing assets it reuses

- `src.convergence` already computes quarterly interaction signals across four
  channels.
- `scripts/compute_convergence_features.py` already orchestrates quarter-level
  extraction of semantic neighbors, author overlap, citation bridges, and
  terminology overlap.
- onset labels already provide target events for temporal scoring.
- the front-level series contract already defines a stable quarterly interface
  for downstream detector and reporting work.
- `src.bocpd` already provides a changepoint-oriented temporal reference that
  can later be used as an alternative target or companion signal.

### What changes

The main change is to preserve and score **edge-level** interaction data before
it is reduced to node-level features.

### What does not change

- The onset detector remains valid.
- The MSD and BOCPD tracks remain valid.
- Domain-isolated execution remains the base architecture.
- The proposal can start as a research track without destabilizing the mainline
  detector stack.

## Proposed Architecture

### 1. Edge Persistence Layer

Extend the convergence pipeline so that it emits a sparse quarterly edge store
in addition to node-level `conv_*` features.

Proposed artifact:

- `data/{domain}/out/02_lineage_tracking/convergence_edges.parquet`

Primary key:

- `(quarter, source_lineage_id, target_lineage_id)`

Candidate columns:

- `semantic_sim`
- `semantic_rank`
- `author_overlap_count`
- `author_overlap_rate`
- `citation_bridge_count`
- `citation_bridge_rate`
- `terminology_overlap`
- `composite_score`
- `n_active_channels`

Design choices:

- Keep only sparse top-k semantic neighbors and edges with non-trivial
  interaction in at least one channel.
- Store lineages first; front-level roll-ups can be derived later.
- Prefer Parquet over CSV for scale and schema stability.

### 2. Temporal Precursor Scoring Layer

Construct lagged source-target candidate relationships from the edge store.

For each candidate pair `(i, j)` and lag `tau` in a chosen window:

- compare edge activity at `t - tau`
- against onset events at `t` for target `j`
- and score whether that relationship is enriched beyond a null expectation.

Candidate scoring families:

1. **Event enrichment score**
   Compare observed precursor frequency before onsets against matched non-onset
   windows.
2. **Discrete-time hazard model**
   Estimate whether lagged edge activity increases the hazard of onset for the
   target node.
3. **Permutation significance**
   Preserve quarter counts and degree structure while shuffling targets or event
   times, then compute empirical p-values.

The initial implementation should prefer the simplest interpretable approach:

- event enrichment plus permutation baseline.

That keeps the first feasibility pass statistically honest and easier to debug.

### 3. Motif Discovery Layer

Single precursor edges are interesting, but the main research upside is likely
in **motifs**.

Candidate motif classes:

- **Two-source recombination**: `(A, B) -> C`
- **Serial precursor chain**: `A -> B -> C`
- **Channel complementarity**: semantic convergence plus author migration
  preceding the same target
- **Bridge-then-growth pattern**: citation or author bridge spike followed by
  semantic tightening and then onset

This layer should answer:

- whether emergence is usually driven by one upstream area or by combinations,
- whether certain channels tend to appear together,
- and whether motif families recur across domains.

### 4. Front-Level Roll-Up Layer

Because curated fronts are easier to interpret than raw lineage IDs, the system
should support a second view that aggregates lineage-level precursor patterns to
the front level.

Important constraint:

- front roll-up is for interpretation and reporting,
- not necessarily for initial scoring.

The lineage graph preserves granularity.
The front graph produces human-readable mechanism maps.

### 5. Explanation Surface

The proposal should end with a usable reporting layer, not only raw graph files.

Proposed outputs:

- precursor rankings per target node,
- motif catalog with lags and channels,
- front-level mechanism maps,
- and short "mechanism cards" summarizing why a target was flagged.

Example mechanism card:

- Target: `front_x`
- Onset quarter: `2018Q1`
- Top precursors: `lineage_14`, `lineage_92`
- Lead window: `3-5 quarters`
- Dominant channels: `semantic`, `author`
- Motif type: `two_source_recombination`
- Confidence: `high relative to null model`

## Proposed Deliverables

### Data artifacts

- edge-level quarterly convergence store
- lagged precursor score table
- motif catalog
- optional front-level precursor graph

### Code artifacts

- `scripts/build_convergence_edges.py` or an extension of
  `scripts/compute_convergence_features.py`
- `scripts/score_precursor_edges.py`
- `scripts/find_precursor_motifs.py`
- `scripts/generate_precursor_report.py`

### Documentation artifacts

- precursor graph contract
- feasibility report
- retrospective evaluation report
- literature review memo

## Retrospective Validation Design

The validation objective is not only "does this look interesting?" It is:

- does the precursor graph capture repeatable temporal structure above null?

### Phase A: Within-domain retrospective validation

Use existing onset labels in one domain.

For each onset event:

- extract preceding 2-8 quarter windows,
- score which precursor edges and motifs were active,
- compare against matched non-onset control windows.

Primary tests:

- enrichment ratio of top precursor edges before true onsets,
- temporal concentration of signal in the lead window,
- permutation significance against null edge/event alignments.

### Phase B: Cross-domain motif stability

After PSC and CRISPR both have valid downstream artifacts:

- learn motif families in one domain,
- test whether the same channel patterns recur in the other.

The goal is not one-to-one node transfer.
The goal is **pattern transfer**:

- for example, whether two-source semantic plus author precursor motifs are
  generally associated with later emergence across domains.

### Phase C: Predictive utility test

Use precursor scores as an auxiliary signal for later detector work.

Possible tests:

- add precursor-derived features to the MSD,
- rank watch-list targets by precursor pressure,
- or use precursor density as a front-level early-warning prior.

This is not required for the proposal to succeed, but it is an important
secondary test.

## Success Criteria

The proposal is successful if it produces all of the following:

1. A reproducible edge-level interaction artifact for at least one domain.
2. Evidence that top precursor edges occur before onset more often than expected
   under a matched null model.
3. At least one recurring motif family that appears in multiple onset cases.
4. A front-level explanation surface that makes emergence interpretable in terms
   of upstream areas and channels.
5. A clear answer on whether precursor structure adds predictive value,
   explanatory value, or both.

The proposal does **not** require large classifier lift to be worthwhile.
If the main gain is explanatory and mechanistic, that is still a meaningful
research contribution.

## Why This Over Other Improvements

| Alternative | What it improves | Why precursor graphs are more interesting |
|-------------|------------------|-------------------------------------------|
| SPECTER2 upgrade | Embedding quality | Better inputs, but still mostly incremental |
| More HPO tuning | Detector score | Improves ranking but not scientific explanation |
| Dashboard work | Usability | Surfaces results but does not change the claim |
| Broad-domain transfer | Novel target discovery | Finds candidates, but not how they were assembled |
| **Precursor / recombination graph** | **Mechanism of emergence** | **Explains which upstream interactions produce emergence** |

## Alignment with Dissertation

This direction fits the project's broader intellectual frame better than a
purely operational detector improvement.

It pushes FrontPulse toward a stronger claim:

- not only that AI can monitor research dynamics,
- but that it can identify **how scientific novelty is assembled through
  recombination and bridge formation**.

That is a stronger contribution for a dissertation or article because it speaks
to mechanism, not only ranking performance.

## Technical Feasibility

Feasibility is promising because the proposal mostly composes artifacts the repo
already knows how to generate.

### What is already available

- quarterly lineage partitions,
- quarterly slices,
- embeddings,
- onset labels,
- convergence channels,
- domain-isolated output paths,
- and timeliness-aware evaluation infrastructure.

### What is newly required

- edge persistence instead of only node-level reduction,
- lagged statistical scoring,
- sparse graph storage and reporting,
- and motif mining logic.

### Scale expectations

The main scale risk is pairwise growth.

Mitigations:

- keep sparse top-k semantic neighbors,
- keep only active edges,
- work lineage-level first on one domain,
- and aggregate to fronts only after edge scoring.

This is a research problem that should start sparse and interpretable, not
dense and maximal.

## Key Risks and Guardrails

### Risk 1: Causal overclaim

The precursor graph may tempt causal language that the data cannot support.

Guardrail:

- use "precursor," "lead-lag," "mechanistically suggestive," and
  "historically associated" language unless a stronger design is later added.

### Risk 2: Spurious field-wide growth effects

Many areas may co-rise because the whole field is expanding.

Guardrail:

- compare against matched non-onset windows,
- include field-growth controls,
- and prefer within-domain null models that preserve temporal volume.

### Risk 3: Lineage instability

Some precursor edges may be artifacts of volatile communities.

Guardrail:

- score at lineage level but report robustness by lifespan and stability cohort,
- and roll up to fronts for interpretation where appropriate.

### Risk 4: Pairwise explosion

Even sparse graphs can grow quickly.

Guardrail:

- restrict to top-k neighbors and active channels,
- use Parquet edge stores,
- and defer higher-order motif mining until the dyadic layer is validated.

### Risk 5: Mapping coverage limits front-level interpretation

Curated front mappings cover only a subset of all lineages.

Guardrail:

- treat front-level graphs as an interpretation surface,
- not the only valid analysis layer.

## Estimated Effort

| Phase | Effort | Description |
|-------|--------|-------------|
| A: Proposal refinement | 1 session | Tighten definitions, choose first target domain, define lag windows |
| B: Literature review | 1-2 sessions | Dynamic networks, knowledge recombination, temporal motifs, emergence studies |
| C: Edge persistence prototype | 1-2 sessions | Emit sparse quarterly edge artifacts from convergence pipeline |
| D: Precursor scoring prototype | 2 sessions | Lead-lag enrichment and permutation baseline |
| E: Motif discovery prototype | 1-2 sessions | Dyads first, then two-source recombination motifs |
| F: Feasibility analysis | 1-2 sessions | Scale, artifact quality, stability, first case studies |
| G: Writeup | 1-2 sessions | Proposal-to-report transition with figures and examples |

**Total**: 8-12 sessions for a serious first pass.

## Immediate Next Steps

The next steps after this proposal should be:

1. **Literature review**
   Focus on temporal knowledge recombination, dynamic innovation networks,
   lead-lag structures in science, and temporal motif mining.
2. **Feasibility study**
   Verify that current convergence artifacts can be lifted to sparse edge-level
   storage without unacceptable scale or data-quality failure.
3. **Domain choice**
   Decide whether the first feasibility pass should run on CRISPR, PSC, or both.
4. **Scoring design choice**
   Decide whether the initial statistical frame should be event enrichment plus
   permutation, hazard modeling, or a simpler null-vs-observed motif count
   comparison.

## Questions for the Literature Review

The literature review should answer these before implementation hardens:

1. How have prior scientometric studies modeled knowledge recombination as a
   temporal network process rather than only a node-ranking problem?
2. What is the strongest prior work on precursor or lead-lag structures in
   scientific emergence?
3. What null models are standard for temporal interaction networks where edge
   volume changes over time?
4. How have temporal motifs been used in innovation or citation-network work?
5. What claims can be defended about mechanism versus association in this kind
   of design?

## Open Questions

- Should precursor scoring target lineage onsets, front onsets, or both?
- Should directionality be inferred purely from temporal precedence, or should
  there also be asymmetric channel rules?
- Should motifs allow negative or competitive precursor relationships, not only
  constructive recombination?
- What is the right lead window: 2-8 quarters, 4-12 quarters, or adaptive by
  domain age?
- Should BOCPD changepoints be treated as an auxiliary target in addition to
  onset labels?
- How much of the graph should remain lineage-level in final reporting, versus
  being rolled up to fronts?

## Bottom Line

If generative discovery is the project's "what should we watch?" track, this is
the "where do new fronts come from?" track.

It is riskier than another detector optimization pass, but it has a stronger
chance of producing a novel scientific claim. It turns FrontPulse from a system
that flags onset into a system that can describe **the recombination pathways
that precede emergence**.
