import argparse
from pathlib import Path

from temp_top import (
    compute_precision_recall_metrics,
    create_summary_dashboard,
    load_data,
    plot_detection_performance_heatmap,
    plot_lead_time_analysis,
    plot_precision_at_k_curve,
    plot_precision_recall_analysis,
    plot_timeline_with_milestones,
    plot_zscore_distributions,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate tripwire validation visualizations.",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        required=True,
        help="Directory containing tripwire validation outputs.",
    )
    return parser.parse_args()


def main():
    """Generate all visualizations."""
    args = parse_args()
    outdir: Path = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    alerts, validation = load_data(outdir)

    print("Computing precision/recall metrics...")
    metrics = compute_precision_recall_metrics(alerts, validation)

    print("\n=== PRECISION/RECALL METRICS ===")
    print(f"True Positives (TP): {metrics['tp']}")
    print(f"False Positives (FP): {metrics['fp']}")
    print(f"False Negatives (FN): {metrics['fn']}")
    print(f"True Negatives (TN): {metrics['tn']}")
    print(f"\nPrecision: {metrics['precision']:.1%}")
    print(f"Recall: {metrics['recall']:.1%}")
    print(f"F1-Score: {metrics['f1']:.1%}")
    print(f"\nFalse Positive Alerts: {len(metrics['false_positive_alerts'])}")

    print("\nGenerating visualizations...")
    print("  [1/7] Summary dashboard...")
    create_summary_dashboard(metrics, validation, alerts, outdir)

    print("  [2/7] Precision-recall analysis...")
    plot_precision_recall_analysis(metrics, outdir)

    print("  [3/7] Z-score distributions...")
    plot_zscore_distributions(alerts, outdir)

    print("  [4/7] Detection performance heatmap...")
    plot_detection_performance_heatmap(validation, outdir)

    print("  [5/7] Precision@K curve...")
    plot_precision_at_k_curve(alerts, validation, outdir)

    print("  [6/7] Lead time analysis...")
    plot_lead_time_analysis(validation, outdir)

    print("  [7/7] Timeline with milestones...")
    plot_timeline_with_milestones(alerts, validation, outdir)

    print(f"\nAll visualizations saved to {outdir}/")
    print("  - 00_dashboard_summary.png")
    print("  - 01_timeline_comprehensive.png")
    print("  - 02_precision_recall_analysis.png")
    print("  - 03_zscore_distributions.png")
    print("  - 04_detection_performance_heatmap.png")
    print("  - 05_precision_at_k_curve.png")
    print("  - 06_lead_time_analysis.png")

if __name__ == '__main__':
    main()
