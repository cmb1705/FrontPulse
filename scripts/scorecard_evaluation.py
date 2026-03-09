"""
Scorecard Evaluation Script
Calculates performance metrics for all prediction files in the scorecard.
"""

import argparse
import json
from datetime import datetime
from pathlib import Path
import shutil
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    auc,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix
)

warnings.filterwarnings('ignore')


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for scorecard evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate prediction files listed in a scorecard JSON.")
    parser.add_argument(
        "--scorecard",
        type=Path,
        default=REPO_ROOT / "data" / "out" / "scorecard.json",
        help="Path to the scorecard JSON file.",
    )
    return parser.parse_args()


def calculate_metrics(y_true, y_pred, y_prob):
    """
    Calculate comprehensive classification metrics.

    Args:
        y_true: True binary labels
        y_pred: Predicted binary labels
        y_prob: Predicted probabilities

    Returns:
        dict: Dictionary of metrics
    """
    metrics = {}

    # Handle edge cases where all predictions are the same class
    try:
        # Basic classification metrics
        metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
        metrics['recall'] = recall_score(y_true, y_pred, zero_division=0)
        metrics['f1_score'] = f1_score(y_true, y_pred, zero_division=0)

        # Confusion matrix metrics
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        metrics['true_positives'] = int(tp)
        metrics['false_positives'] = int(fp)
        metrics['true_negatives'] = int(tn)
        metrics['false_negatives'] = int(fn)
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        # Predicted class counts
        metrics['predicted_positives'] = int(y_pred.sum())
        metrics['predicted_negatives'] = int(len(y_pred) - y_pred.sum())

        # Probability-based metrics
        if len(np.unique(y_true)) > 1:  # Need both classes for these metrics
            metrics['average_precision'] = average_precision_score(y_true, y_prob)

            # PR-AUC
            precision_vals, recall_vals, _ = precision_recall_curve(y_true, y_prob)
            metrics['pr_auc'] = auc(recall_vals, precision_vals)

            # ROC-AUC
            metrics['roc_auc'] = roc_auc_score(y_true, y_prob)
        else:
            metrics['average_precision'] = 0.0
            metrics['pr_auc'] = 0.0
            metrics['roc_auc'] = 0.0

        # Support metrics
        metrics['n_samples'] = len(y_true)
        metrics['n_positives'] = int(y_true.sum())
        metrics['n_negatives'] = int(len(y_true) - y_true.sum())
        metrics['positive_rate'] = float(y_true.mean())

    except Exception as e:
        print(f"Error calculating metrics: {e}")
        # Return default metrics on error
        metrics = {
            'precision': 0.0,
            'recall': 0.0,
            'f1_score': 0.0,
            'average_precision': 0.0,
            'pr_auc': 0.0,
            'roc_auc': 0.0,
            'true_positives': 0,
            'false_positives': 0,
            'true_negatives': 0,
            'false_negatives': 0,
            'specificity': 0.0,
            'predicted_positives': int(y_pred.sum()) if 'y_pred' in locals() else 0,
            'predicted_negatives': int(len(y_pred) - y_pred.sum()) if 'y_pred' in locals() else 0,
            'n_samples': len(y_true),
            'n_positives': 0,
            'n_negatives': 0,
            'positive_rate': 0.0
        }

    return metrics


def infer_split_type(experiment: str) -> str:
    """
    Rough heuristic to tag experiments as prospective/holdout vs retrospective.
    """
    exp = experiment.lower()
    prospective_terms = ['time_forward', 'holdout', 'dev', 'test', 'split', 'monitoring']
    if any(term in exp for term in prospective_terms):
        return 'prospective'
    return 'retrospective'


def evaluate_prediction_file(filepath):
    """
    Evaluate a single prediction file.

    Args:
        filepath: Path to the prediction CSV file

    Returns:
        dict: Evaluation results with metrics for inflection and milestone
    """
    try:
        df = pd.read_csv(filepath)

        # Get file metadata
        file_path = Path(filepath)
        file_stats = file_path.stat()

        result = {
            'file_path': str(filepath),
            'file_size_bytes': file_stats.st_size,
            'modified_timestamp': datetime.fromtimestamp(file_stats.st_mtime).isoformat(),
            'evaluated_timestamp': datetime.now().isoformat(),
            'status': 'success'
        }

        # Check if file has required columns
        required_cols_inflection = ['is_inflection_true', 'is_inflection_pred', 'inflection_probability']
        required_cols_milestone = ['is_milestone_true', 'is_milestone_pred', 'milestone_probability']

        warnings_list = []

        # Evaluate inflection predictions
        if all(col in df.columns for col in required_cols_inflection):
            inflection_metrics = calculate_metrics(
                df['is_inflection_true'].values,
                df['is_inflection_pred'].values,
                df['inflection_probability'].values
            )
            if inflection_metrics.get('predicted_positives', 0) == 0:
                warnings_list.append('inflection_predictions_all_negative')
            result['inflection_metrics'] = inflection_metrics
        else:
            result['inflection_metrics'] = None
            result['missing_inflection_columns'] = [c for c in required_cols_inflection if c not in df.columns]

        # Evaluate milestone predictions
        if all(col in df.columns for col in required_cols_milestone):
            milestone_metrics = calculate_metrics(
                df['is_milestone_true'].values,
                df['is_milestone_pred'].values,
                df['milestone_probability'].values
            )
            if milestone_metrics.get('predicted_positives', 0) == 0:
                warnings_list.append('milestone_predictions_all_negative')
            result['milestone_metrics'] = milestone_metrics
        else:
            result['milestone_metrics'] = None
            result['missing_milestone_columns'] = [c for c in required_cols_milestone if c not in df.columns]

        # Additional data quality metrics
        result['data_quality'] = {
            'total_rows': len(df),
            'unique_lineages': df['lineage_id'].nunique() if 'lineage_id' in df.columns else None,
            'quarter_range': f"{df['quarter'].min()} to {df['quarter'].max()}" if 'quarter' in df.columns else None,
            'columns': list(df.columns)
        }
        if warnings_list:
            result['warnings'] = warnings_list

    except FileNotFoundError:
        result = {
            'file_path': str(filepath),
            'status': 'file_not_found',
            'error': 'File does not exist'
        }
    except Exception as e:
        result = {
            'file_path': str(filepath),
            'status': 'error',
            'error': str(e)
        }

    return result


def main():
    """Main execution function."""
    args = parse_args()
    scorecard_path = args.scorecard.resolve()

    with open(scorecard_path, 'r') as f:
        scorecard = json.load(f)

    print(f"Evaluating {len(scorecard['prediction_files'])} prediction files...")
    print("=" * 80)

    # Evaluate each file
    results = []
    for i, file_entry in enumerate(scorecard['prediction_files'], 1):
        filepath = file_entry['path']
        experiment = file_entry['experiment']

        print(f"\n[{i}/{len(scorecard['prediction_files'])}] {experiment}")
        print(f"  File: {filepath}")

        result = evaluate_prediction_file(filepath)
        result['experiment'] = experiment
        result['split_type'] = infer_split_type(experiment)
        results.append(result)

        # Print summary
        if result['status'] == 'success':
            if result.get('inflection_metrics'):
                inf_m = result['inflection_metrics']
                print(f"  Inflection - AP: {inf_m['average_precision']:.4f}, "
                      f"PR-AUC: {inf_m['pr_auc']:.4f}, "
                      f"ROC-AUC: {inf_m['roc_auc']:.4f}, "
                      f"F1: {inf_m['f1_score']:.4f}")
            if result.get('milestone_metrics'):
                mil_m = result['milestone_metrics']
                print(f"  Milestone  - AP: {mil_m['average_precision']:.4f}, "
                      f"PR-AUC: {mil_m['pr_auc']:.4f}, "
                      f"ROC-AUC: {mil_m['roc_auc']:.4f}, "
                      f"F1: {mil_m['f1_score']:.4f}")
        else:
            print(f"  Status: {result['status']}")
            if 'error' in result:
                print(f"  Error: {result['error']}")

    # Deduplicate prediction_file entries by path hash to avoid counting copies.
    def sha256_file(path: Path) -> str:
        import hashlib
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()

    seen = {}
    dedup_results = []
    for r in results:
        fp = Path(r.get('file_path', ''))
        digest = sha256_file(fp) if fp.exists() else None
        if digest and digest in seen:
            continue
        if digest:
            seen[digest] = r.get('experiment', '')
        dedup_results.append(r)

    scorecard['results'] = dedup_results
    scorecard['prediction_files'] = [pf for pf in scorecard.get('prediction_files', []) if not any(dup in pf.get('experiment','') for dup in [])]
    scorecard['metadata']['last_evaluated'] = datetime.now().isoformat()
    scorecard['metadata']['total_evaluated'] = len(dedup_results)
    scorecard['metadata']['successful_evaluations'] = sum(1 for r in dedup_results if r['status'] == 'success')

    # Save updated scorecard (make a timestamped backup first)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    backup_path = scorecard_path.parent / f"scorecard_backup_{timestamp}.json"
    try:
        shutil.copy(scorecard_path, backup_path)
        print(f"Backup written: {backup_path}")
    except Exception as e:
        print(f"Warning: could not create backup: {e}")

    with open(scorecard_path, 'w') as f:
        json.dump(scorecard, f, indent=2)

    print("\n" + "=" * 80)
    print(f"\nScorecard updated: {scorecard_path}")
    print(f"Total files evaluated: {len(results)}")
    print(f"Successful: {scorecard['metadata']['successful_evaluations']}")
    print(f"Failed: {len(results) - scorecard['metadata']['successful_evaluations']}")


if __name__ == '__main__':
    main()
