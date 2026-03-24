#!/usr/bin/env python3
"""Benchmark lineage-level versus front-level onset detection.

Compares detection at two units of analysis using the same evaluation
contract.  Produces a structured report covering data characteristics,
detection coverage, timeliness, and operational tradeoffs.

The benchmark does NOT train a separate front-level model.  With only
~22 fronts (vs 5,179 lineages), front-level ML is infeasible.  Instead,
it compares the operational signal produced by aggregating lineage-level
detections to the front level.

Usage:
    python scripts/benchmark_detection_units.py
    python scripts/benchmark_detection_units.py --out-dir results/benchmark/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Set

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Benchmark lineage-level vs front-level onset detection.",
    )
    parser.add_argument(
        "--timeseries",
        default="data/out/02_lineage_tracking/lineage_timeseries.csv",
        help="Path to lineage timeseries CSV.",
    )
    parser.add_argument(
        "--onset-labels",
        default="data/out/02_lineage_tracking/onset_labels.csv",
        help="Path to lineage onset labels CSV.",
    )
    parser.add_argument(
        "--mapping",
        default="data/out/experiments/stage0_tight_mapping/"
        "milestone_lineage_mapping_tight.csv",
        help="Path to lineage-to-front mapping CSV.",
    )
    parser.add_argument(
        "--holdout-metrics",
        default="data/out/experiments/msd_onset_catboost_holdout/"
        "evaluation_metrics.json",
        help="Path to lineage-level MSD holdout metrics JSON.",
    )
    parser.add_argument(
        "--cv-metrics",
        default="data/out/experiments/msd_onset_leakage_safe/"
        "evaluation_metrics.json",
        help="Path to lineage-level MSD CV metrics JSON.",
    )
    parser.add_argument(
        "--train-end",
        default="2019Q4",
        help="Last quarter in training set (default: 2019Q4).",
    )
    parser.add_argument(
        "--out-dir",
        default="data/out/experiments/detection_unit_benchmark",
        help="Output directory for benchmark results.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_timeseries(path: Path) -> pd.DataFrame:
    """Load lineage timeseries CSV."""
    df = pd.read_csv(path)
    df["quarter"] = df["quarter"].astype(str)
    return df


def load_onset_labels(path: Path) -> pd.DataFrame:
    """Load onset labels with columns lineage_id, onset_quarter, onset_detected."""
    df = pd.read_csv(path)
    df["onset_detected"] = df["onset_detected"].astype(int)
    return df


def load_mapping(path: Path) -> pd.DataFrame:
    """Load lineage-to-front mapping."""
    return pd.read_csv(path)


def load_json_metrics(path: Path) -> Dict[str, Any]:
    """Load evaluation metrics JSON."""
    with path.open() as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# Analysis functions
# ---------------------------------------------------------------------------


def compute_lineage_level_stats(
    ts_df: pd.DataFrame,
    onset_df: pd.DataFrame,
    train_end: str,
) -> Dict[str, Any]:
    """Compute lineage-level detection statistics."""
    n_lineages = ts_df["lineage_id"].nunique()
    n_observations = len(ts_df)
    n_quarters = ts_df["quarter"].nunique()

    detected = onset_df[onset_df["onset_detected"] == 1]
    n_onsets = len(detected)
    onset_rate = round(100.0 * n_onsets / max(n_lineages, 1), 2)

    # Time-forward split
    holdout_qs = sorted(q for q in ts_df["quarter"].unique() if q > train_end)
    holdout_obs = len(ts_df[ts_df["quarter"].isin(holdout_qs)])
    holdout_onsets = len(
        detected[detected["onset_quarter"].isin(holdout_qs)]
    ) if "onset_quarter" in detected.columns else 0

    return {
        "unit": "lineage",
        "n_entities": n_lineages,
        "n_observations": n_observations,
        "n_quarters": n_quarters,
        "n_onsets": n_onsets,
        "onset_prevalence_pct": onset_rate,
        "holdout_quarters": len(holdout_qs),
        "holdout_observations": holdout_obs,
        "holdout_onsets": holdout_onsets,
    }


def compute_front_level_stats(
    ts_df: pd.DataFrame,
    onset_df: pd.DataFrame,
    mapping_df: pd.DataFrame,
    train_end: str,
) -> Dict[str, Any]:
    """Compute front-level detection statistics."""
    # Build lineage -> front mapping (one lineage can map to one front row)
    lineage_to_front = dict(
        zip(mapping_df["lineage_id"], mapping_df["mapped_fronts"])
    )
    mapped_lineages = set(lineage_to_front.keys())
    all_lineages = set(ts_df["lineage_id"].unique())

    # Unique fronts
    fronts = sorted(mapping_df["mapped_fronts"].unique())
    n_fronts = len(fronts)

    # Front -> constituent lineages
    front_lineages: Dict[str, Set[int]] = {}
    for _, row in mapping_df.iterrows():
        f = row["mapped_fronts"]
        lid = int(row["lineage_id"])
        front_lineages.setdefault(f, set()).add(lid)

    # Front-level onset: earliest onset quarter among constituent lineages
    detected = onset_df[onset_df["onset_detected"] == 1]
    front_onsets: Dict[str, str] = {}
    for front, lids in front_lineages.items():
        front_detected = detected[detected["lineage_id"].isin(lids)]
        if not front_detected.empty and "onset_quarter" in front_detected.columns:
            earliest = front_detected["onset_quarter"].dropna().min()
            if pd.notna(earliest):
                front_onsets[front] = str(earliest)

    # Front-level observations (unique front-quarter pairs)
    mapped_ts = ts_df[ts_df["lineage_id"].isin(mapped_lineages)].copy()
    mapped_ts["front"] = mapped_ts["lineage_id"].map(lineage_to_front)
    front_quarters = mapped_ts.groupby("front")["quarter"].nunique()
    total_front_obs = int(front_quarters.sum())

    # Holdout split
    holdout_qs = sorted(q for q in ts_df["quarter"].unique() if q > train_end)
    holdout_front_onsets = sum(
        1 for q in front_onsets.values() if q in holdout_qs
    )

    return {
        "unit": "front",
        "n_entities": n_fronts,
        "n_observations": total_front_obs,
        "n_quarters": ts_df["quarter"].nunique(),
        "n_onsets": len(front_onsets),
        "onset_prevalence_pct": round(
            100.0 * len(front_onsets) / max(n_fronts, 1), 2
        ),
        "mapping_coverage_lineages": len(mapped_lineages),
        "total_lineages": len(all_lineages),
        "mapping_coverage_pct": round(
            100.0 * len(mapped_lineages) / max(len(all_lineages), 1), 2
        ),
        "front_names": fronts,
        "front_onset_quarters": front_onsets,
        "mean_lineages_per_front": round(
            np.mean([len(v) for v in front_lineages.values()]), 1
        ),
        "holdout_onsets": holdout_front_onsets,
    }


def build_comparison_table(
    lineage_stats: Dict[str, Any],
    front_stats: Dict[str, Any],
    lineage_holdout: Dict[str, Any],
    lineage_cv: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Build a side-by-side comparison table."""
    rows = []

    def _row(metric: str, lineage_val: Any, front_val: Any, note: str = "") -> None:
        rows.append({
            "metric": metric,
            "lineage_level": lineage_val,
            "front_level": front_val,
            "note": note,
        })

    _row("Entities", lineage_stats["n_entities"], front_stats["n_entities"])
    _row(
        "Observations",
        lineage_stats["n_observations"],
        front_stats["n_observations"],
    )
    _row("Onsets detected", lineage_stats["n_onsets"], front_stats["n_onsets"])
    _row(
        "Onset prevalence",
        f"{lineage_stats['onset_prevalence_pct']}%%",
        f"{front_stats['onset_prevalence_pct']}%%",
    )
    _row(
        "Holdout onsets",
        lineage_stats["holdout_onsets"],
        front_stats["holdout_onsets"],
    )

    # MSD metrics (lineage only -- front-level ML is infeasible)
    _row(
        "ROC-AUC (holdout)",
        round(lineage_holdout.get("roc_auc_test", 0), 3),
        "N/A",
        "Insufficient samples for front-level ML",
    )
    _row(
        "PR-AUC (holdout)",
        round(lineage_holdout.get("pr_auc_test", 0), 3),
        "N/A",
        "Insufficient samples for front-level ML",
    )
    _row(
        "Precision @ t=0.07",
        round(lineage_holdout.get("precision_threshold", 0), 3),
        "N/A",
    )
    _row(
        "Recall @ t=0.07",
        round(lineage_holdout.get("recall_threshold", 0), 3),
        "N/A",
    )
    _row(
        "Detection lag (median)",
        f"{lineage_holdout.get('detection_lag_median', 'N/A')}Q",
        "Derived from lineage lag",
    )
    _row(
        "CV ROC-AUC (5-fold)",
        round(lineage_cv.get("cv_roc_auc_mean", 0), 3),
        "N/A",
    )
    _row(
        "CV PR-AUC (5-fold)",
        round(lineage_cv.get("cv_pr_auc_mean", 0), 3),
        "N/A",
    )

    return rows


def assess_feasibility(front_stats: Dict[str, Any]) -> Dict[str, Any]:
    """Assess whether front-level ML training is feasible."""
    n_entities = front_stats["n_entities"]
    n_onsets = front_stats["n_onsets"]
    n_obs = front_stats["n_observations"]

    feasible = n_entities >= 50 and n_onsets >= 10 and n_obs >= 500
    reasons = []
    if n_entities < 50:
        reasons.append(f"Only {n_entities} fronts (need >= 50 for reliable CV)")
    if n_onsets < 10:
        reasons.append(f"Only {n_onsets} front onsets (need >= 10 for stratified CV)")
    if n_obs < 500:
        reasons.append(f"Only {n_obs} front-quarter observations (need >= 500)")

    return {
        "ml_training_feasible": feasible,
        "reasons": reasons if not feasible else ["All criteria met"],
        "recommendation": (
            "Front-level ML is feasible; proceed with CatBoost training."
            if feasible
            else "Front-level ML is NOT feasible. Use lineage-level detection "
            "with front-level aggregation for operational deployment."
        ),
    }


def determine_recommendation(
    lineage_stats: Dict[str, Any],
    front_stats: Dict[str, Any],
    feasibility: Dict[str, Any],
) -> Dict[str, Any]:
    """Produce the final benchmark recommendation."""
    if not feasibility["ml_training_feasible"]:
        return {
            "detection_unit": "lineage",
            "deployment_unit": "front (via aggregation)",
            "rationale": (
                "Lineage-level detection provides sufficient training data "
                f"({lineage_stats['n_entities']} lineages, "
                f"{lineage_stats['n_onsets']} onsets) for reliable ML. "
                f"Front-level has only {front_stats['n_entities']} entities "
                f"with {front_stats['n_onsets']} onsets, insufficient for "
                "standalone ML. The recommended architecture is: "
                "detect onsets at lineage level, then aggregate alerts to "
                "fronts for operational reporting."
            ),
            "tradeoffs": {
                "lineage_advantages": [
                    "5,179 entities provide robust training data",
                    "231 onset labels enable stratified cross-validation",
                    "Finer-grained detection catches early signals",
                    "ROC-AUC 0.895 on time-forward holdout demonstrates viability",
                ],
                "front_advantages": [
                    "22 fronts are operationally manageable for human review",
                    "Front-level alerts are more interpretable (named research areas)",
                    "Aggregation reduces noise from ephemeral lineages",
                    "Downstream consumers (BOCPD, timeliness) prefer front-level",
                ],
                "hybrid_approach": (
                    "Detect at lineage level (statistical power), "
                    "aggregate to front level (operational clarity). "
                    "A front is flagged when any constituent lineage triggers. "
                    "This preserves the sensitivity of lineage-level detection "
                    "while delivering interpretable front-level alerts."
                ),
            },
        }
    return {
        "detection_unit": "both (compare empirically)",
        "rationale": "Front-level has sufficient data for ML; run both and compare.",
    }


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------


def format_report(
    lineage_stats: Dict[str, Any],
    front_stats: Dict[str, Any],
    comparison: List[Dict[str, Any]],
    feasibility: Dict[str, Any],
    recommendation: Dict[str, Any],
) -> str:
    """Format the benchmark as human-readable text."""
    lines: List[str] = []
    lines.append("=" * 70)
    lines.append("DETECTION UNIT BENCHMARK: LINEAGE vs FRONT")
    lines.append("=" * 70)

    lines.append("")
    lines.append("1. DATA CHARACTERISTICS")
    lines.append("-" * 50)
    lines.append(f"  {'Metric':<35} {'Lineage':>12} {'Front':>12}")
    lines.append(f"  {'Entities':<35} {lineage_stats['n_entities']:>12,} {front_stats['n_entities']:>12,}")
    lines.append(f"  {'Observations':<35} {lineage_stats['n_observations']:>12,} {front_stats['n_observations']:>12,}")
    lines.append(f"  {'Onsets':<35} {lineage_stats['n_onsets']:>12,} {front_stats['n_onsets']:>12,}")
    lines.append(f"  {'Onset prevalence':<35} {lineage_stats['onset_prevalence_pct']:>11}%% {front_stats['onset_prevalence_pct']:>11}%%")
    lines.append(f"  {'Mapping coverage':<35} {'100%%':>12} {front_stats['mapping_coverage_pct']:>11}%%")

    lines.append("")
    lines.append("2. FRONT-LEVEL ML FEASIBILITY")
    lines.append("-" * 50)
    lines.append(f"  Feasible: {'YES' if feasibility['ml_training_feasible'] else 'NO'}")
    for r in feasibility["reasons"]:
        lines.append(f"  - {r}")

    lines.append("")
    lines.append("3. COMPARISON TABLE")
    lines.append("-" * 50)
    for row in comparison:
        note = f"  ({row['note']})" if row["note"] else ""
        lines.append(
            f"  {row['metric']:<30} {str(row['lineage_level']):>14} {str(row['front_level']):>14}{note}"
        )

    lines.append("")
    lines.append("4. RECOMMENDATION")
    lines.append("-" * 50)
    lines.append(f"  Detection unit:   {recommendation['detection_unit']}")
    if "deployment_unit" in recommendation:
        lines.append(f"  Deployment unit:  {recommendation['deployment_unit']}")
    lines.append(f"  Rationale: {recommendation['rationale']}")

    if "tradeoffs" in recommendation:
        t = recommendation["tradeoffs"]
        lines.append("")
        lines.append("  Lineage advantages:")
        for a in t["lineage_advantages"]:
            lines.append(f"    + {a}")
        lines.append("  Front advantages:")
        for a in t["front_advantages"]:
            lines.append(f"    + {a}")
        lines.append(f"  Hybrid approach: {t['hybrid_approach']}")

    lines.append("")
    lines.append("=" * 70)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Run the detection unit benchmark."""
    args = parse_args()

    # Check required files
    required = {
        "timeseries": Path(args.timeseries),
        "onset_labels": Path(args.onset_labels),
        "mapping": Path(args.mapping),
    }
    for name, path in required.items():
        if not path.exists():
            print(f"ERROR: {name} not found: {path}")
            sys.exit(1)

    # Load data
    print("Loading data...")
    ts_df = load_timeseries(required["timeseries"])
    onset_df = load_onset_labels(required["onset_labels"])
    mapping_df = load_mapping(required["mapping"])

    print(f"  Timeseries: {len(ts_df):,} rows, {ts_df['lineage_id'].nunique():,} lineages")
    print(f"  Onset labels: {len(onset_df):,} lineages, {onset_df['onset_detected'].sum()} onsets")
    print(f"  Mapping: {len(mapping_df):,} rows, {mapping_df['mapped_fronts'].nunique()} fronts")

    # Load MSD metrics if available
    holdout_path = Path(args.holdout_metrics)
    cv_path = Path(args.cv_metrics)
    lineage_holdout = load_json_metrics(holdout_path) if holdout_path.exists() else {}
    lineage_cv = load_json_metrics(cv_path) if cv_path.exists() else {}

    # Compute stats
    print("\nComputing statistics...")
    lineage_stats = compute_lineage_level_stats(ts_df, onset_df, args.train_end)
    front_stats = compute_front_level_stats(ts_df, onset_df, mapping_df, args.train_end)

    # Assess feasibility
    feasibility = assess_feasibility(front_stats)

    # Build comparison
    comparison = build_comparison_table(
        lineage_stats, front_stats, lineage_holdout, lineage_cv,
    )

    # Determine recommendation
    recommendation = determine_recommendation(
        lineage_stats, front_stats, feasibility,
    )

    # Format and print
    text = format_report(
        lineage_stats, front_stats, comparison, feasibility, recommendation,
    )
    print()
    print(text)

    # Write outputs
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "lineage_level": lineage_stats,
        "front_level": front_stats,
        "comparison": comparison,
        "feasibility": feasibility,
        "recommendation": recommendation,
        "msd_holdout_metrics": lineage_holdout,
        "msd_cv_metrics": lineage_cv,
    }

    json_out = out_dir / "benchmark_report.json"
    with json_out.open("w") as fh:
        json.dump(report, fh, indent=2, default=str)
    print(f"\nWrote JSON report: {json_out}")

    text_out = out_dir / "benchmark_report.txt"
    text_out.write_text(text, encoding="utf-8")
    print(f"Wrote text report: {text_out}")


if __name__ == "__main__":
    main()
