# Lessons Learned

## 2026-03-28: Scope ruff --unsafe-fixes carefully (FP-97k)
When using `ruff check --fix --unsafe-fixes scripts/`, it modifies ALL files in the
directory, not just the ones you intend to commit. This caused 136 files to be modified
when only 54 were migration targets. Fix: always specify exact file paths for ruff fixes
rather than broad directories, or immediately revert non-target files.

## 2026-03-28: Pre-commit hook gates on full file, not diff (FP-97k)
The ruff pre-commit hook checks entire staged files, not just changed lines. So touching
a file with pre-existing violations triggers failures. Strategy: fix pre-existing
violations in touched files as part of the commit.

## 2026-03-28: torch.compile requires Triton (FP-d5y)
torch.compile uses the Triton backend which is not available on Windows. Always pass
--no-compile when running PyTorch scripts on Windows. The --no-compile flag exists
in stage1_quarterly_embeddings_optimized.py for this reason.

## 2026-03-28: torchvision version must match torch version (FP-d5y)
After reinstalling torch with CUDA support (2.6.0+cu124), torchvision remained at
0.25.0 (built for torch 2.10.0), causing RuntimeError on operator registration.
Fix: install torchvision matching the torch version (0.21.0+cu124 for torch 2.6.0).

## 2026-03-30: Optuna search confirms convergence feature value (FP-25k)
After 16 trials with 65 features (including convergence), Optuna found PR-AUC 0.205
vs 0.155 baseline -- a 32.3% improvement. This proves the prior session's negative
ablation result was caused by HPO mismatch, not by the features themselves being
uninformative. Always re-tune hyperparameters when adding features.

## 2026-03-30: CRISPR Tier 2 requires community detection (FP-cf4)
The CRISPR pipeline was initially run with --skip-communities, which meant no
partition files (cache_cum/partitions_cum/) were generated. Stages 2-5 of
run_build_pipeline.py require these files. Must run communities.py --domain crispr
before Tier 2 can proceed.

## 2026-03-30: PSC data contained CRISPR works -- always validate post-ingest (FP-ukx)
The PSC ingest.parquet had 100% overlap with CRISPR data and 0 perovskite rows.
Root cause unknown (possibly saved settings overrode the config topic filter).
All PSC-domain results (MSD, ablation, convergence, embeddings) were actually CRISPR.
Fix: added validate_topic_alignment() to run.py (FP-jp4). Always verify primary_topic_name
after ingest. The validation gate checks >30% topic match against datasource config.
