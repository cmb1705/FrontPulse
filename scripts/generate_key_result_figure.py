"""
Generate a clean, publication-ready key result figure showing the performance gap.

Creates a simple but impactful visualization for:
- Paper figures
- Presentations
- Executive summaries
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def create_key_result_figure(leaderboard_path: Path, output_path: Path):
    """Create a clean key result figure showing MSD vs baselines."""

    # Load leaderboard
    with open(leaderboard_path) as f:
        leaderboard = json.load(f)

    # Method configuration
    methods = {
        'MSD\n(Multi-Signal)': {
            'key': 'msd_lightgbm',
            'color': '#2E7D32',
            'hatch': None
        },
        'Semantic\nChangepoint': {
            'key': 'semantic_changepoint',
            'color': '#1565C0',
            'hatch': '//'
        },
        'Simple\nHeuristics': {
            'key': 'simple_heuristics',
            'color': '#F57C00',
            'hatch': '\\\\'
        },
        'Kleinberg\nBurst': {
            'key': 'kleinberg_burst',
            'color': '#C62828',
            'hatch': 'xx'
        }
    }

    # Create figure with 2x2 subplot
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Baseline Method Performance Comparison', fontsize=16, fontweight='bold', y=0.98)

    # === 1. PR-AUC Comparison ===
    pr_aucs = [leaderboard[m['key']]['pr_auc'] * 100 for m in methods.values()]
    colors = [m['color'] for m in methods.values()]
    hatches = [m['hatch'] for m in methods.values()]

    bars1 = ax1.bar(range(len(methods)), pr_aucs, color=colors, edgecolor='black', linewidth=1.5, alpha=0.8)

    # Apply hatching to baselines
    for bar, hatch in zip(bars1[1:], hatches[1:]):  # Skip MSD
        bar.set_hatch(hatch)

    # Add value labels
    for i, (bar, val) in enumerate(zip(bars1, pr_aucs)):
        ax1.text(i, val + 2, f'{val:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')

    # Add performance gap annotation
    msd_val = pr_aucs[0]
    best_baseline = max(pr_aucs[1:])
    gap = msd_val - best_baseline

    ax1.annotate('', xy=(0, msd_val), xytext=(1, best_baseline),
                arrowprops={'arrowstyle': '<->', 'lw': 2, 'color': 'red'})
    ax1.text(0.5, (msd_val + best_baseline) / 2, f'{gap:.1f}% gap\n({msd_val/best_baseline:.1f}× higher)',
            ha='center', va='center', fontsize=10, fontweight='bold', color='red',
            bbox={'boxstyle': 'round,pad=0.5', 'facecolor': 'white', 'edgecolor': 'red', 'linewidth': 2})

    ax1.set_ylabel('PR-AUC (%)', fontsize=12, fontweight='bold')
    ax1.set_title('A. PR-AUC Performance', fontsize=13, fontweight='bold', loc='left')
    ax1.set_xticks(range(len(methods)))
    ax1.set_xticklabels(methods.keys(), fontsize=10)
    ax1.grid(True, alpha=0.3, linestyle='--', axis='y')
    ax1.set_ylim([0, 105])

    # === 2. Precision & Recall ===
    x = np.arange(len(methods))
    width = 0.35

    precisions = [leaderboard[m['key']]['default_precision'] * 100 for m in methods.values()]
    recalls = [leaderboard[m['key']]['default_recall'] * 100 for m in methods.values()]

    bars_p = ax2.bar(x - width/2, precisions, width, label='Precision', color='#4CAF50', edgecolor='black', linewidth=1, alpha=0.8)
    bars_r = ax2.bar(x + width/2, recalls, width, label='Recall', color='#2196F3', edgecolor='black', linewidth=1, alpha=0.8)

    # Add value labels
    for bars in [bars_p, bars_r]:
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 2,
                    f'{height:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax2.set_ylabel('Performance (%)', fontsize=12, fontweight='bold')
    ax2.set_title('B. Precision & Recall', fontsize=13, fontweight='bold', loc='left')
    ax2.set_xticks(x)
    ax2.set_xticklabels(methods.keys(), fontsize=10)
    ax2.legend(loc='upper right', fontsize=10, framealpha=0.95)
    ax2.grid(True, alpha=0.3, linestyle='--', axis='y')
    ax2.set_ylim([0, 105])

    # === 3. Coverage Comparison ===
    coverages = [leaderboard[m['key']]['detection_lag_coverage'] * 100 for m in methods.values()]

    bars3 = ax3.barh(range(len(methods)), coverages, color=colors, edgecolor='black', linewidth=1.5, alpha=0.8)

    # Apply hatching
    for bar, hatch in zip(bars3[1:], hatches[1:]):
        bar.set_hatch(hatch)

    # Add value labels
    for i, (bar, val) in enumerate(zip(bars3, coverages)):
        ax3.text(val + 2, i, f'{val:.1f}%', va='center', ha='left', fontsize=11, fontweight='bold')

    ax3.set_xlabel('Breakthrough Coverage (%)', fontsize=12, fontweight='bold')
    ax3.set_title('C. Coverage of True Breakthroughs', fontsize=13, fontweight='bold', loc='left')
    ax3.set_yticks(range(len(methods)))
    ax3.set_yticklabels(methods.keys(), fontsize=10)
    ax3.grid(True, alpha=0.3, linestyle='--', axis='x')
    ax3.set_xlim([0, 105])
    ax3.invert_yaxis()

    # === 4. Detection Lag ===
    lags = [leaderboard[m['key']]['detection_lag_median'] for m in methods.values()]
    lag_colors = ['green' if l <= 0 else 'red' for l in lags]

    bars4 = ax4.barh(range(len(methods)), lags, color=lag_colors, edgecolor='black', linewidth=1.5, alpha=0.6)

    # Add value labels
    for i, (bar, val) in enumerate(zip(bars4, lags)):
        label = f'{abs(val):.0f}Q {"early" if val < 0 else "on-time" if val == 0 else "late"}'
        ax4.text(val + 0.3 if val > 0 else val - 0.3, i, label,
                va='center', ha='left' if val > 0 else 'right',
                fontsize=10, fontweight='bold')

    ax4.axvline(x=0, color='black', linestyle='-', linewidth=1.5, alpha=0.7)
    ax4.set_xlabel('Median Detection Lag (quarters)', fontsize=12, fontweight='bold')
    ax4.set_title('D. Detection Timing', fontsize=13, fontweight='bold', loc='left')
    ax4.set_yticks(range(len(methods)))
    ax4.set_yticklabels(methods.keys(), fontsize=10)
    ax4.grid(True, alpha=0.3, linestyle='--', axis='x')
    ax4.set_xlim([-10, 3])
    ax4.invert_yaxis()

    # Add legend for early/late
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='green', edgecolor='black', label='Early/On-time detection', alpha=0.6),
        Patch(facecolor='red', edgecolor='black', label='Late detection', alpha=0.6)
    ]
    ax4.legend(handles=legend_elements, loc='lower left', fontsize=9, framealpha=0.95)

    plt.tight_layout()

    # Save figure
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(output_path.with_suffix('.pdf'), bbox_inches='tight', facecolor='white')
    print(f"Saved key result figure to {output_path}")
    print(f"Saved PDF version to {output_path.with_suffix('.pdf')}")

    plt.close()


def main():
    """Generate key result figure."""
    figures_dir = Path("data/out/experiments/baselines/figures")
    leaderboard_path = figures_dir / "leaderboard.json"
    output_path = figures_dir / "key_result_comparison.png"

    if not leaderboard_path.exists():
        print(f"Error: Leaderboard not found at {leaderboard_path}")
        return

    create_key_result_figure(leaderboard_path, output_path)
    print("\nKey result figure generation complete!")


if __name__ == "__main__":
    main()
