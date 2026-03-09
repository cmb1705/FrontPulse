#!/usr/bin/env python3
"""
Parameter sweep and random search for tripwire_nb_fdr.py.

Runs tripwire + validation for a grid of (lookback, min_count, alpha)
and an additional random search, logging summary metrics and producing
an exploratory visualization.
"""

from __future__ import annotations

import itertools
import json
import random
import subprocess
from pathlib import Path
from typing import Dict, List

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# Locations
TIMESERIES = Path("data/out/04_front_aggregation/front_timeseries_delta_long.csv")
ALERTS_PATH = Path("data/out/05_tripwire_detection/alerts_tripwire.csv")
VALIDATION_DIR = Path("data/out/06_validation")
VALIDATION_RESULTS_PATH = VALIDATION_DIR / "validation_results.csv"

# Parameter grids (doubled resolution)
GRID_LOOKBACKS = [6, 8, 10, 12, 14]
GRID_MIN_COUNTS = [1, 2, 3, 4]
GRID_ALPHAS = [0.03, 0.05, 0.07, 0.10, 0.15]

# Random search settings
RANDOM_ITERS = 50
LOOKBACK_RANGE = (6, 16)      # inclusive, integer
MIN_COUNT_RANGE = (1, 6)      # inclusive, integer
ALPHA_RANGE = (0.02, 0.20)    # continuous


def run_tripwire(lookback: int, min_count: int, alpha: float) -> None:
    """Execute tripwire_nb_fdr.py with selected parameters."""
    cmd = [
        "python",
        "tripwire_nb_fdr.py",
        "--timeseries", str(TIMESERIES),
        "--out", str(ALERTS_PATH),
        "--lookback", str(lookback),
        "--min-history", str(max(6, lookback - 2)),
        "--min-count", str(min_count),
        "--alpha", f"{alpha:.4f}",
    ]
    subprocess.run(cmd, check=True)


def run_validation() -> None:
    """Execute evaluate_tripwire.py to compute metrics."""
    cmd = [
        "python",
        "scripts/evaluate_tripwire.py",
        "--counts", str(TIMESERIES),
        "--metrics", "data/out/front_metrics_cumulative.csv",
        "--outdir", str(VALIDATION_DIR),
        "--start", "2008-Q1",
        "--end", "2025-Q3",
        "--history", "8",
    ]
    subprocess.run(cmd, check=True)


def summarize_metrics(params: Dict) -> Dict:
    """Return summary metrics from the latest validation run."""
    validation_df = pd.read_csv(VALIDATION_RESULTS_PATH)
    alerts_df = pd.read_csv(ALERTS_PATH)

    events_total = len(validation_df)
    hits = int(validation_df["detected"].sum())

    significant_alerts = int(alerts_df["alert"].sum())
    precision = hits / significant_alerts if significant_alerts else 0.0
    recall = hits / events_total if events_total else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    avg_lead = (
        validation_df.loc[validation_df["detected"], "lead_time_quarters"].mean()
        if hits
        else None
    )

    result = {
        "lookback": params["lookback"],
        "min_count": params["min_count"],
        "alpha": params["alpha"],
        "detections": hits,
        "events": events_total,
        "significant_alerts": significant_alerts,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "avg_lead": avg_lead,
    }
    return result


def sweep_grid() -> List[Dict]:
    """Run the structured grid search."""
    results = []
    for lookback, min_count, alpha in itertools.product(
        GRID_LOOKBACKS, GRID_MIN_COUNTS, GRID_ALPHAS
    ):
        print(
            f"[Grid] lookback={lookback}, min_count={min_count}, alpha={alpha:.3f}"
        )
        params = {"lookback": lookback, "min_count": min_count, "alpha": alpha}
        run_tripwire(**params)
        run_validation()
        results.append(summarize_metrics(params))
    return results


def random_search() -> List[Dict]:
    """Run a random search over the parameter space."""
    results = []
    for i in range(RANDOM_ITERS):
        lookback = random.randint(*LOOKBACK_RANGE)
        min_count = random.randint(*MIN_COUNT_RANGE)
        alpha = random.uniform(*ALPHA_RANGE)
        alpha = round(alpha, 4)

        print(
            f"[Random {i+1}/{RANDOM_ITERS}] "
            f"lookback={lookback}, min_count={min_count}, alpha={alpha:.4f}"
        )

        params = {"lookback": lookback, "min_count": min_count, "alpha": alpha}
        run_tripwire(**params)
        run_validation()
        results.append(summarize_metrics(params))
    return results


def visualise(df: pd.DataFrame, path: Path) -> None:
    """Create a facet scatter plot highlighting F1 across parameters."""
    sns.set_theme(style="whitegrid")
    df_plot = df.copy()
    df_plot["alpha"] = df_plot["alpha"].round(3)

    g = sns.FacetGrid(
        df_plot, col="alpha", col_wrap=3, height=3.2, sharex=False, sharey=False
    )
    g.map_dataframe(
        sns.scatterplot,
        x="lookback",
        y="min_count",
        hue="f1",
        size="recall",
        palette="viridis",
        sizes=(40, 200),
    )
    g.add_legend(title="F1 (color) / Recall (size)")
    g.set_axis_labels("Lookback (quarters)", "Min count")
    g.fig.subplots_adjust(top=0.9)
    g.fig.suptitle("Tripwire parameter sweep")
    path.parent.mkdir(parents=True, exist_ok=True)
    g.savefig(path, dpi=200)
    plt.close(g.fig)


def main() -> None:
    results: List[Dict] = []

    # Structured grid
    print("=== Running grid search ===")
    results.extend(sweep_grid())

    # Random search
    print("\n=== Running random search ===")
    results.extend(random_search())

    df = pd.DataFrame(results)
    output_dir = VALIDATION_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "tripwire_param_sweep_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved results to {csv_path}")

    json_path = output_dir / "tripwire_param_sweep_results.json"
    json_path.write_text(json.dumps(results, indent=2))

    # Best combinations
    top = df.sort_values("f1", ascending=False).head(10)
    print("\nTop configurations (by F1):")
    print(top[["lookback", "min_count", "alpha", "detections", "precision", "recall", "f1"]])

    # Visualization
    viz_path = output_dir / "tripwire_param_sweep.png"
    visualise(df, viz_path)
    print(f"Visualization written to {viz_path}")


if __name__ == "__main__":
    main()
