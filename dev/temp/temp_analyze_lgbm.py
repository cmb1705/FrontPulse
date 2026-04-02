#!/usr/bin/env python3
"""Quick analysis of LightGBM predictions at threshold 0.70"""

import pandas as pd

# Load predictions
lgbm_pred = pd.read_csv('data/out/experiments/phase2_lightgbm/breakthrough_predictions.csv')
rf_pred = pd.read_csv('data/out/experiments/phase2_cv/breakthrough_predictions.csv')

# Analyze LightGBM at threshold 0.70
threshold = 0.70
y_true = lgbm_pred['is_milestone_true']
y_pred_lgbm = (lgbm_pred['milestone_probability'] >= threshold).astype(int)

tp = ((y_true == 1) & (y_pred_lgbm == 1)).sum()
fp = ((y_true == 0) & (y_pred_lgbm == 1)).sum()
fn = ((y_true == 1) & (y_pred_lgbm == 0)).sum()
tn = ((y_true == 0) & (y_pred_lgbm == 0)).sum()

precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

print("=" * 70)
print("LIGHTGBM PERFORMANCE AT THRESHOLD 0.70")
print("=" * 70)
print("\nConfusion Matrix:")
print(f"  TP: {tp:4d}    FP: {fp:4d}")
print(f"  FN: {fn:4d}    TN: {tn:4d}")
print("\nMetrics:")
print(f"  Precision: {precision*100:5.1f}%")
print(f"  Recall:    {recall*100:5.1f}%")
print(f"  F1 Score:  {f1*100:5.1f}%")
print(f"  TP:FP Ratio: {tp}:{fp} = 1:{fp/tp:.1f}" if tp > 0 else "  TP:FP Ratio: 0:0")

# Probability distribution
print("\nProbability Distribution:")
print(f"  Mean:   {lgbm_pred['milestone_probability'].mean():.3f}")
print(f"  Median: {lgbm_pred['milestone_probability'].median():.3f}")
print(f"  75th %: {lgbm_pred['milestone_probability'].quantile(0.75):.3f}")
print(f"  95th %: {lgbm_pred['milestone_probability'].quantile(0.95):.3f}")
print(f"  Max:    {lgbm_pred['milestone_probability'].max():.3f}")

# Positives distribution
pos = lgbm_pred[lgbm_pred['is_milestone_true'] == 1]
print(f"\nPositive samples (n={len(pos)}) probability distribution:")
print(f"  Mean:   {pos['milestone_probability'].mean():.3f}")
print(f"  Median: {pos['milestone_probability'].median():.3f}")
print(f"  Max:    {pos['milestone_probability'].max():.3f}")
print(f"  >= 0.70: {(pos['milestone_probability'] >= 0.70).sum()}")

# Random Forest comparison
print("\n" + "=" * 70)
print("RANDOM FOREST PERFORMANCE AT THRESHOLD 0.70 (for comparison)")
print("=" * 70)
y_pred_rf = (rf_pred['milestone_probability'] >= threshold).astype(int)

tp_rf = ((rf_pred['is_milestone_true'] == 1) & (y_pred_rf == 1)).sum()
fp_rf = ((rf_pred['is_milestone_true'] == 0) & (y_pred_rf == 1)).sum()
fn_rf = ((rf_pred['is_milestone_true'] == 1) & (y_pred_rf == 0)).sum()

precision_rf = tp_rf / (tp_rf + fp_rf) if (tp_rf + fp_rf) > 0 else 0
recall_rf = tp_rf / (tp_rf + fn_rf) if (tp_rf + fn_rf) > 0 else 0
f1_rf = 2 * precision_rf * recall_rf / (precision_rf + recall_rf) if (precision_rf + recall_rf) > 0 else 0

print("\nConfusion Matrix:")
print(f"  TP: {tp_rf:4d}    FP: {fp_rf:4d}")
print(f"  FN: {fn_rf:4d}")
print("\nMetrics:")
print(f"  Precision: {precision_rf*100:5.1f}%")
print(f"  Recall:    {recall_rf*100:5.1f}%")
print(f"  F1 Score:  {f1_rf*100:5.1f}%")
print(f"  TP:FP Ratio: {tp_rf}:{fp_rf} = 1:{fp_rf/tp_rf:.1f}" if tp_rf > 0 else "  TP:FP Ratio: 0:0")

print("\n" + "=" * 70)
print("COMPARISON")
print("=" * 70)
print(f"\n{'Metric':<20s} {'LightGBM':>12s} {'Random Forest':>15s} {'Difference':>15s}")
print(f"{'-'*20} {'-'*12} {'-'*15} {'-'*15}")
print(f"{'Precision':<20s} {precision*100:>11.1f}% {precision_rf*100:>14.1f}% {(precision-precision_rf)*100:>14.1f}%")
print(f"{'Recall':<20s} {recall*100:>11.1f}% {recall_rf*100:>14.1f}% {(recall-recall_rf)*100:>14.1f}%")
print(f"{'F1 Score':<20s} {f1*100:>11.1f}% {f1_rf*100:>14.1f}% {(f1-f1_rf)*100:>14.1f}%")

print("\n" + "=" * 70)
