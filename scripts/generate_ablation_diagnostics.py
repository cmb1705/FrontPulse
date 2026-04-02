#!/usr/bin/env python3
"""Generate review plots and statistical summaries for ablation experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, pstdev
from typing import Any


def load_fold_diagnostics(experiment_dir: Path) -> dict[str, Any]:
    """Load the fold diagnostics JSON for a single experiment directory."""
    path = experiment_dir / "fold_diagnostics.json"
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def _metric_summary(values: list[float]) -> dict[str, float] | None:
    """Return simple summary stats for a metric array."""
    if not values:
        return None
    return {
        "mean": float(mean(values)),
        "std": float(pstdev(values)),
        "min": float(min(values)),
        "max": float(max(values)),
        "n": len(values),
    }


def build_experiment_summary(experiment_dir: Path, diagnostics: dict[str, Any]) -> dict[str, Any]:
    """Build a compact machine-readable summary for one ablation experiment."""
    summary: dict[str, Any] = {
        "experiment_dir": str(experiment_dir),
        "experiment_name": diagnostics.get("name", experiment_dir.name),
        "model_type": diagnostics.get("model_type"),
        "n_features": diagnostics.get("n_features"),
        "n_samples": diagnostics.get("n_samples"),
        "n_positives": diagnostics.get("n_positives"),
        "pr_curve_source": "out_of_fold_aggregate",
        "calibration_source": "out_of_fold_aggregate",
        "fold_metric_summary": {},
    }
    for metric_name in ("fold_pr_auc", "fold_roc_auc"):
        metric_summary = _metric_summary([float(value) for value in diagnostics.get(metric_name, [])])
        if metric_summary is not None:
            summary["fold_metric_summary"][metric_name] = metric_summary
    return summary


def write_json(output_path: Path, payload: dict[str, Any]) -> None:
    """Write a JSON artifact with stable formatting."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def render_pr_curve(diagnostics: dict[str, Any], experiment_dir: Path) -> Path:
    """Render the stored out-of-fold PR curve to PNG."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pr_curve = diagnostics.get("pr_curve", {})
    precision = pr_curve.get("precision", [])
    recall = pr_curve.get("recall", [])
    if not precision or not recall:
        raise ValueError(f"PR curve missing from {experiment_dir / 'fold_diagnostics.json'}")

    output_path = experiment_dir / "pr_curve.png"
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(recall, precision, color="#0b7285", linewidth=2.2)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"PR Curve: {diagnostics.get('name', experiment_dir.name)}")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.05)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def render_calibration_curve(diagnostics: dict[str, Any], experiment_dir: Path) -> Path:
    """Render the stored calibration bins to PNG."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    calibration = diagnostics.get("calibration", {})
    prob_true = calibration.get("prob_true", [])
    prob_pred = calibration.get("prob_pred", [])
    if not prob_true or not prob_pred:
        raise ValueError(f"Calibration data missing from {experiment_dir / 'fold_diagnostics.json'}")

    output_path = experiment_dir / "calibration_curve.png"
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot([0, 1], [0, 1], linestyle="--", color="#6c757d", linewidth=1.2, label="Perfectly calibrated")
    ax.plot(prob_pred, prob_true, marker="o", color="#c92a2a", linewidth=2.0, label="Observed")
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Observed frequency")
    ax.set_title(f"Calibration Curve: {diagnostics.get('name', experiment_dir.name)}")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.grid(alpha=0.25)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def compute_wilcoxon_summary(
    left_dir: Path,
    left_diagnostics: dict[str, Any],
    right_dir: Path,
    right_diagnostics: dict[str, Any],
) -> dict[str, Any]:
    """Compute a paired Wilcoxon signed-rank summary on fold PR-AUC."""
    from scipy.stats import wilcoxon

    left_values = [float(value) for value in left_diagnostics.get("fold_pr_auc", [])]
    right_values = [float(value) for value in right_diagnostics.get("fold_pr_auc", [])]
    if not left_values or len(left_values) != len(right_values):
        raise ValueError("fold_pr_auc arrays must exist and have equal length for Wilcoxon comparison.")

    result = wilcoxon(left_values, right_values, alternative="two-sided", zero_method="wilcox")
    deltas = [right - left for left, right in zip(left_values, right_values)]
    winner = (
        right_diagnostics.get("name", right_dir.name)
        if mean(right_values) > mean(left_values)
        else left_diagnostics.get("name", left_dir.name)
    )
    return {
        "metric": "fold_pr_auc",
        "test": "wilcoxon_signed_rank",
        "alternative": "two-sided",
        "left_experiment": {
            "experiment_dir": str(left_dir),
            "name": left_diagnostics.get("name", left_dir.name),
            "values": left_values,
            "mean": float(mean(left_values)),
        },
        "right_experiment": {
            "experiment_dir": str(right_dir),
            "name": right_diagnostics.get("name", right_dir.name),
            "values": right_values,
            "mean": float(mean(right_values)),
        },
        "fold_deltas_right_minus_left": deltas,
        "mean_delta_right_minus_left": float(mean(deltas)),
        "median_like_middle_delta_right_minus_left": float(sorted(deltas)[len(deltas) // 2]),
        "statistic": float(result.statistic),
        "p_value": float(result.pvalue),
        "better_model_by_mean_pr_auc": winner,
    }


def run_ablation_diagnostics(experiment_dirs: list[Path]) -> dict[str, list[Path] | Path | None]:
    """Generate plot and JSON artifacts for one or two experiment directories."""
    generated_paths: list[Path] = []
    loaded: dict[Path, dict[str, Any]] = {}

    for experiment_dir in experiment_dirs:
        diagnostics = load_fold_diagnostics(experiment_dir)
        loaded[experiment_dir] = diagnostics
        generated_paths.append(render_pr_curve(diagnostics, experiment_dir))
        generated_paths.append(render_calibration_curve(diagnostics, experiment_dir))
        summary_path = experiment_dir / "diagnostic_summary.json"
        write_json(summary_path, build_experiment_summary(experiment_dir, diagnostics))
        generated_paths.append(summary_path)

    comparison_path: Path | None = None
    if len(experiment_dirs) == 2:
        left_dir, right_dir = experiment_dirs
        comparison = compute_wilcoxon_summary(
            left_dir,
            loaded[left_dir],
            right_dir,
            loaded[right_dir],
        )
        for experiment_dir in experiment_dirs:
            comparison_path = experiment_dir / "ablation_wilcoxon.json"
            write_json(comparison_path, comparison)
            generated_paths.append(comparison_path)

    return {
        "generated_paths": generated_paths,
        "comparison_path": comparison_path,
    }


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Render PR/calibration plots and Wilcoxon summaries from ablation fold diagnostics."
    )
    parser.add_argument(
        "experiment_dirs",
        nargs="+",
        type=Path,
        help="One or two experiment directories containing fold_diagnostics.json",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entry point."""
    args = parse_args()
    experiment_dirs = [path.resolve() for path in args.experiment_dirs]
    if len(experiment_dirs) not in {1, 2}:
        raise SystemExit("Provide one or two experiment directories.")

    result = run_ablation_diagnostics(experiment_dirs)
    for path in result["generated_paths"]:
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
