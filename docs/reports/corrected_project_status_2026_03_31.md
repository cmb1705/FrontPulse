# FrontPulse Corrected Project Status

**Date**: 2026-04-01

**Scope**: FP-ss7 audit-remediation closeout status

**Supersedes**: `docs/reports/convergence_ablation_final_2026_03_30.md` for current-project status claims

---

## Executive Summary

FrontPulse now has a defensible CRISPR artifact set for the corrected 198-label
workflow, but it does not yet have a valid cross-domain story. The CRISPR
Stage 2-5 mapping and validation pipeline has been rerun cleanly under the
domain-isolated CRISPR tree with SPECTER2 metadata on disk, the corrected
holdout and ablation artifacts are present, and the memo now cites only
verifiable artifacts. PSC remains incomplete from community detection onward,
so PSC-vs-CRISPR comparisons are still out of scope.

---

## Preserved Artifacts (Trusted, Not Regenerated)

| Domain | Artifact | Evidence | Status |
|--------|----------|----------|--------|
| CRISPR | Raw corpus | `data/crispr/ingest/` | Preserved-valid, topic T10878 |
| CRISPR | Cumulative graphs | `data/crispr/graphs/` | Preserved-valid, 109 quarterly graph files |
| CRISPR | Community partitions and caches | `data/crispr/out/cache_cum/` | Preserved-valid |
| CRISPR | Onset labels | `data/crispr/out/02_lineage_tracking/onset_labels_msd.csv` | Preserved-valid, 198 onset labels |
| CRISPR | Convergence features | `data/crispr/out/02_lineage_tracking/convergence_features.csv` | Preserved-valid |
| PSC | Raw corpus | `data/psc/ingest/` | Preserved-valid, topic T10247 |
| PSC | Cumulative graphs | `data/psc/graphs/` | Preserved-valid, 109 quarterly graph files |

## Regenerated Artifacts

| Artifact | Evidence | Current state |
|----------|----------|---------------|
| CRISPR front alias config | `config/front_aliases_crispr.yaml` | Fixed to 7 top-level CRISPR fronts with anchor DOIs |
| CRISPR Stage 2 embeddings | `data/crispr/out/02_lineage_tracking/lineage_embeddings.json` | Regenerated with `model=allenai/specter2_base`, `embedding_dim=768`, `n_lineages=542` |
| CRISPR Stage 2 similarity | `data/crispr/out/03_milestone_mapping/lineage_front_similarity.csv` | Regenerated; columns match only CRISPR fronts |
| CRISPR Stage 3 similarity | `data/crispr/out/03_milestone_mapping/lineage_front_term_similarity.csv` | Regenerated under CRISPR output tree |
| CRISPR Stage 4 similarity | `data/crispr/out/03_milestone_mapping/lineage_front_npmi_similarity.csv` | Regenerated under CRISPR output tree |
| CRISPR Stage 5 mappings | `data/crispr/out/03_milestone_mapping/lineage_front_mappings.csv` | Regenerated; 542 mapped lineages across 7 CRISPR fronts |
| CRISPR front aggregation | `data/crispr/out/04_front_aggregation/front_timeseries_delta_long.csv` | Regenerated; 720 front-quarter records across 7 fronts |
| CRISPR Stage 2-5 validation | `data/crispr/out/06_validation/stage2/` through `data/crispr/out/06_validation/stage5/` | Regenerated inside the CRISPR domain tree |
| 51-feature Optuna HPO | `data/crispr/out/experiments/optuna_search_51feat_100trials/` | Regenerated on corrected 198-label data |
| 65-feature Optuna HPO | `data/crispr/out/experiments/optuna_search_with_convergence_198labels/` | Regenerated on corrected 198-label data |
| CRISPR holdout | `data/crispr/out/experiments/msd_holdout_2020_onset_labels/` | Regenerated with raw and persistence-filtered metrics plus provenance |
| CRISPR ablation diagnostics | `data/crispr/out/experiments/ablation_51feat_198labels/` and `data/crispr/out/experiments/ablation_65feat_198labels/` | Regenerated diagnostics, plots, and Wilcoxon JSON |

---

## CRISPR Pipeline Audit (Fresh Stage 2-5 Rerun)

The clean CRISPR rerun used preserved CRISPR ingest and graphs, not OpenAlex.

- Stage 2 command: `D:\Git_Repos\FrontPulse\.venv\Scripts\python.exe -u scripts/run_build_pipeline.py --domain crispr --stages 2 --device cuda --no-validate`
- Stage 2 result: 147,141 unique papers embedded with SPECTER2 in 8,562.1 s; outputs written to `data/crispr/out/02_lineage_tracking/` and `data/crispr/out/03_milestone_mapping/`
- Stage 2-5 command: `D:\Git_Repos\FrontPulse\.venv\Scripts\python.exe -u scripts/run_build_pipeline.py --domain crispr --stages 2,3,4,5 --device cuda --npmi-workers 10`
- Aggregation command: `D:\Git_Repos\FrontPulse\.venv\Scripts\python.exe -u scripts/aggregate_lineages_to_fronts.py --domain crispr`

Post-run audit findings:

- `lineage_embeddings.json` identifies the embedding model from file inspection alone.
- Stage 2, Stage 3, and Stage 4 similarity matrices all expose exactly the 7 CRISPR front columns.
- Stage 5 validation now writes to `data/crispr/out/06_validation/stage5/`, not `data/out/`.
- `data/crispr/out/04_front_aggregation/` is no longer empty.
- Text audit across `data/crispr/out/03_milestone_mapping/`, `data/crispr/out/04_front_aggregation/`, and `data/crispr/out/06_validation/` found zero PSC front-key hits and no `perovskite` leakage.

---

## Corrected Metrics

### CRISPR Cross-Validation and Ablation

Dataset context:

- Source labels: 198 onset labels from `data/crispr/out/02_lineage_tracking/onset_labels_msd.csv`
- Sample count: 13,031 lineage-quarter rows
- Positive rate: 1.52%

Matched-HPO upper bound from Optuna best trials:

| Variant | Artifact | Model | PR-AUC | ROC-AUC |
|---------|----------|-------|--------|---------|
| 51-feature (no convergence) | `data/crispr/out/experiments/optuna_search_51feat_100trials/best_trial.json` | XGBoost | 0.237 +/- 0.050 | 0.922 +/- 0.021 |
| 65-feature (with convergence) | `data/crispr/out/experiments/optuna_search_with_convergence_198labels/best_trial.json` | LightGBM | 0.243 +/- 0.063 | 0.923 +/- 0.018 |

Independent 5-fold estimate from regenerated ablation diagnostics:

| Variant | Artifact | PR-AUC | ROC-AUC |
|---------|----------|--------|---------|
| 51-feature (no convergence) | `data/crispr/out/experiments/ablation_51feat_198labels/diagnostic_summary.json` | 0.179 +/- 0.041 | 0.900 +/- 0.016 |
| 65-feature (with convergence) | `data/crispr/out/experiments/ablation_65feat_198labels/diagnostic_summary.json` | 0.172 +/- 0.042 | 0.895 +/- 0.015 |

Paired test:

- Wilcoxon signed-rank on fold PR-AUC: `p = 0.8125`
- Artifact: `data/crispr/out/experiments/ablation_51feat_198labels/ablation_wilcoxon.json`
- Current conclusion: convergence features do not provide a statistically significant benefit on the corrected CRISPR workflow

### CRISPR Time-Forward Holdout

Artifact: `data/crispr/out/experiments/msd_holdout_2020_onset_labels/evaluation_metrics.json`

- Ranking metrics: PR-AUC 0.208, ROC-AUC 0.903
- Raw threshold metrics at `t = 0.70`: TP = 2, FP = 10, FN = 65, TN = 3,842
- Raw threshold precision/recall/F1: 0.167 / 0.030 / 0.051
- Persistence-filtered operational metrics with `persistence_window = 2`: TP = 0, FP = 0, FN = 67, TN = 3,852
- Operational interpretation: the detector still carries ranking signal on the holdout, but the current production threshold plus persistence rule suppresses all positive alerts

### CRISPR Stage 5 Mapping Status

Artifact: `data/crispr/out/06_validation/stage5/Stage5_validation_results.json`

- 542 mapped lineages across 7 CRISPR fronts
- Confidence distribution: 1 high, 453 medium, 88 low
- Review-needed count: 541
- Interpretation: the alias-clean rerun is now trustworthy as a CRISPR-only artifact set, but the mapping layer still needs manual curation before it should be treated as high-confidence front labeling

---

## Retired Claims

The following claims are retired and should not be reused:

1. ~~Convergence features improve recall by +20.9%~~
   The March 30 report used stale 231-label / PSC-contaminated comparisons and asymmetric HPO budgets.

2. ~~CRISPR Stage 2-5 rerun remains pending~~
   The rerun is now complete on disk under `data/crispr/out/`.

3. ~~Cross-domain PSC vs CRISPR comparison is ready~~
   PSC downstream artifacts remain stale from community detection onward.

---

## Remaining Gaps

1. **PSC rebuild remains the primary blocker** (`FP-2r1`).
   PSC output artifacts under `data/psc/out/` are still stale from the old contaminated workflow. No valid PSC-vs-CRISPR comparison should be claimed until PSC is rebuilt from community detection onward.

2. **CRISPR clinical_therapeutics anchor coverage needs repair** (`FP-olh`).
   The Stage 2 rerun logged missing anchor abstracts for `clinical_therapeutics`. The artifact set is still valid for the alias-clean rerun issue, but this front-specific semantic anchor needs correction or explicit exclusion.

3. **Stage 5 mappings still need curation**.
   Only 1 of 542 mappings is currently marked ready-to-use without review. The pipeline is now domain-correct, but semantic labeling quality is not equivalent to expert-curated fronts yet.

4. **Selection bias should stay explicit in reporting**.
   Best-trial Optuna metrics are useful as search diagnostics, not as the primary generalization estimate. The regenerated ablation diagnostics are the better unbiased summary for CRISPR model comparison.

---

## Valid Claims Today

- FrontPulse has a corrected CRISPR artifact stack for embeddings, mapping, front aggregation, holdout reporting, and ablation diagnostics under `data/crispr/out/`.
- The corrected CRISPR workflow supports statements about CRISPR-only model performance, CRISPR-only front aggregation, and the null ablation result for convergence features.
- FrontPulse does not yet support a valid PSC-vs-CRISPR comparison.
