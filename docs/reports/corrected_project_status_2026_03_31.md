# FrontPulse Corrected Project Status

**Date**: 2026-03-31
**Scope**: FP-ss7 audit remediation closeout
**Supersedes**: docs/reports/convergence_ablation_final_2026_03_30.md (stale)

---

## What Changed (Audit Remediation)

The Codex reviewer audit (FP-ss7) identified that the project was not at a
defensible stopping point. Key issues remediated:

1. **Data contamination**: Prior PSC experiments ran on CRISPR-contaminated
   feature matrices (21,608 rows, 231 labels). PSC was re-ingested from T10247
   (192,222 perovskite works) but downstream artifacts were never regenerated.

2. **HPO asymmetry**: Convergence ablation compared 65-feat (100 trials) vs
   51-feat (50 trials). The +20% claim was an artifact of unequal search budget.

3. **Silent stale inputs**: Pipeline scripts warned and backfilled zeros when
   upstream metric files were missing, allowing experiments to silently proceed
   on incomplete data.

4. **Missing provenance**: Experiment artifacts did not record their exact run
   configuration, making reproduction impossible without reading source code.

5. **Schema ambiguity**: Onset-label experiments used inflection-era column
   names, making it impossible to distinguish the two workflows from file
   inspection alone.

---

## Preserved Artifacts (Valid, Not Regenerated)

| Domain | Artifact | Status |
|--------|----------|--------|
| CRISPR | Raw corpus (data/crispr/ingest/) | Valid -- 238,526 works, T10878 |
| CRISPR | Cumulative graphs (data/crispr/graphs/) | Valid -- 109 quarterly PKL files |
| CRISPR | Community partitions (data/crispr/out/cache_cum/) | Valid |
| CRISPR | Onset labels (onset_labels_msd.csv) | Valid -- 198 onset labels |
| CRISPR | Convergence features (convergence_features.csv) | Valid -- 15,597 rows |
| PSC | Raw corpus (data/psc/ingest/) | Valid -- 192,222 works, T10247 |
| PSC | Cumulative graphs (data/psc/graphs/) | Valid -- 109 quarterly PKL files |

## Regenerated Artifacts

| Artifact | Old State | New State |
|----------|-----------|-----------|
| CRISPR front alias config | Wrapped in `fronts:` key (broken) | 7 top-level fronts with anchor DOIs |
| CRISPR Stage 2-5 pipeline | PSC front aliases in CRISPR outputs | CRISPR-specific front aliases |
| CRISPR lineage embeddings | SciBERT (stale) | SPECTER2 (correct model) |
| 51-feat Optuna HPO | 50 trials, 231 labels (stale) | 100 trials, 198 labels |
| 65-feat Optuna HPO | 100 trials, 231 labels (stale) | 100 trials, 198 labels |
| Convergence ablation | Copied legacy artifacts | Fresh per-fold diagnostics |

---

## Current Metrics (Corrected)

### CRISPR MSD Cross-Validation (5-fold, 198 onset labels, 13,031 samples)

**Optuna HPO results (100 trials each, selection-biased upper bound):**

| Variant | Model | PR-AUC | ROC-AUC | Trial |
|---------|-------|--------|---------|-------|
| 51-feat (no convergence) | XGBoost | 0.237 +/- 0.050 | 0.922 +/- 0.021 | 84 |
| 65-feat (with convergence) | LightGBM | 0.243 +/- 0.063 | 0.923 +/- 0.018 | 93 |

**Independent 5-fold CV (unbiased estimate):**

| Variant | PR-AUC | ROC-AUC |
|---------|--------|---------|
| 51-feat (no convergence) | 0.179 +/- 0.041 | 0.900 +/- 0.016 |
| 65-feat (with convergence) | 0.172 +/- 0.042 | 0.895 +/- 0.015 |

**Wilcoxon signed-rank test**: p = 0.81. Convergence features provide NO
statistically significant improvement.

### Interpretation

- The positive class rate is 1.52% (198/13,031). PR-AUC of 0.18 represents
  ~12x improvement over random baseline (0.0152).
- The gap between Optuna's reported PR-AUC (0.24) and independent CV (0.18)
  reflects selection bias from choosing the best of 100 trials.
- Convergence features add 14 columns but no measurable lift. The 65-feat
  model has higher variance, suggesting the additional features introduce noise.

---

## Retired Claims

The following claims from prior reports are now retired:

1. ~~Convergence features improve recall by +20.9%~~ -- Artifact of HPO
   asymmetry (50 vs 100 trials) and PSC-contaminated data (231 labels from
   wrong dataset).

2. ~~MSD achieves PR-AUC 0.162 with convergence features~~ -- This was
   the Optuna selection-biased result on contaminated data. Unbiased
   estimate on correct data is 0.172.

3. ~~Cross-domain comparison: PSC vs CRISPR~~ -- PSC downstream artifacts
   remain stale (communities onward not regenerated). No valid cross-domain
   comparison exists yet.

---

## Infrastructure Improvements (Code)

| Change | Module | Purpose |
|--------|--------|---------|
| Run provenance | src/run_provenance.py | CLI args + input hashes + git SHA saved per experiment |
| Artifact freshness | src/artifact_freshness.py | Hard failure on missing/stale upstream inputs |
| Onset schema | MSD output columns | is_onset_* primary columns with legacy aliases |
| Per-fold diagnostics | fold_diagnostics.json | PR curves, calibration, per-fold confusion matrices |
| Holdout reporting | evaluation_metrics.json | Both raw threshold and persistence-filtered metrics |
| Embedding cache | compute_lineage_embeddings.py | NPZ reuse when lineage set unchanged |
| Global embedding | compute_lineage_embeddings.py | Embed each paper once, aggregate per lineage |
| SPECTER2 default | --embedding-model flag | SPECTER2 replaces SciBERT for similarity scoring |
| CRISPR front aliases | front_aliases_crispr.yaml | 7 fronts with anchor DOIs (was broken wrapper) |

---

## Remaining Gaps

1. **PSC pipeline rebuild** (FP-2r1): PSC downstream artifacts (communities
   onward) are stale. Requires community detection + full pipeline rerun.
   Multi-session effort.

2. **CRISPR holdout rerun**: The corrected holdout with raw + persistent
   metrics has not been regenerated yet. Code infrastructure (FP-ss7.6) is
   in place; needs an MSD run with temporal split.

3. **CRISPR Stage 2 embeddings**: Global paper-level SPECTER2 embedding
   pipeline running. Pending completion and validation.

4. **Convergence features**: Given the null ablation result, convergence
   features may be removed from the default feature set in future work.
   They add complexity without measurable benefit.

5. **Selection bias in HPO reporting**: Optuna best-trial PR-AUC (0.24)
   overstates true performance (0.18). Consider using nested CV or
   reporting the independent CV estimate as the primary metric.
