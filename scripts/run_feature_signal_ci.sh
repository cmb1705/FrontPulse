#!/usr/bin/env bash
set -euo pipefail

# Lightweight CI hook for feature-signal diagnostics + subset evaluation.
# Runs small-sample smoke tests to ensure the tooling executes end-to-end.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

python "$ROOT_DIR/scripts/analyze_feature_signals.py" \
  --sample-limit 3000 \
  --tree-sample-limit 3000 \
  --cv-folds 3 \
  --disable-shap \
  --n-jobs 2 \
  --output-dir "$ROOT_DIR/data/out/analysis/feature_signal_pruning/ci_diagnostics"

python "$ROOT_DIR/scripts/run_feature_subset_evals.py" \
  --sample-limit 3000 \
  --max-configs 2 \
  --output-dir "$ROOT_DIR/data/out/analysis/feature_signal_pruning/ci_subset"
