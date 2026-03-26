#!/usr/bin/env python3
"""Update the MSD assessment history with new predictions and/or labels.

Supports two operations:

1. **Record**: Load model predictions and append to the history table.
2. **Backfill**: Apply ground-truth labels to resolve unknown outcomes.

Both operations can run in a single invocation.

Usage::

    # Record new predictions from a model run
    python scripts/update_assessment_history.py record \\
        --predictions data/out/experiments/msd_latest/breakthrough_predictions.csv \\
        --model-version v_20260323_001 \\
        --quarter-assessed 2025Q1

    # Backfill outcomes with onset labels
    python scripts/update_assessment_history.py backfill \\
        --labels data/out/02_lineage_tracking/onset_labels_msd.csv

    # Show calibration stats
    python scripts/update_assessment_history.py stats
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import pandas as pd  # noqa: E402

from src.assessment_history import (  # noqa: E402
    append_assessments,
    backfill_outcomes,
    compute_calibration_stats,
    load_history,
    record_assessments,
    save_history,
    summarize_history,
)
from src.domain_registry import add_domain_args, resolve_script_paths  # noqa: E402

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Update MSD assessment history.",
    )
    parser.add_argument(
        "--history", default=None,
        help="Path to assessment history CSV",
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    add_domain_args(parser)

    sub = parser.add_subparsers(dest="command")

    # Record subcommand
    rec = sub.add_parser("record", help="Record new model predictions")
    rec.add_argument(
        "--predictions", required=True,
        help="Path to breakthrough_predictions.csv from MSD",
    )
    rec.add_argument(
        "--model-version", required=True,
        help="Model version ID (e.g., v_20260323_001)",
    )
    rec.add_argument(
        "--quarter-assessed", required=True,
        help="Quarter the model was run (e.g., 2025Q1)",
    )
    rec.add_argument(
        "--threshold", type=float, default=0.15,
        help="Decision threshold (default: 0.15)",
    )
    rec.add_argument(
        "--probability-column", default="inflection_probability",
        help="Column name for probabilities (default: inflection_probability)",
    )

    # Backfill subcommand
    bf = sub.add_parser("backfill", help="Backfill outcomes with labels")
    bf.add_argument(
        "--labels", default=None,
        help="Path to onset labels CSV",
    )
    bf.add_argument(
        "--label-column", default="is_inflection_onset",
        help="Binary label column (default: is_inflection_onset)",
    )
    bf.add_argument(
        "--source", default="onset_labels",
        help="Label source name (default: onset_labels)",
    )

    # Stats subcommand
    sub.add_parser("stats", help="Show calibration statistics")

    return parser.parse_args()


def cmd_record(args: argparse.Namespace) -> None:
    """Record new predictions into the assessment history."""
    history = load_history(Path(args.history))

    predictions = pd.read_csv(args.predictions)
    logger.info(
        "Loaded %d predictions from %s", len(predictions), args.predictions,
    )

    new_rows = record_assessments(
        predictions,
        quarter_assessed=args.quarter_assessed,
        model_version=args.model_version,
        threshold=args.threshold,
        probability_column=args.probability_column,
    )

    history = append_assessments(history, new_rows)
    save_history(history, Path(args.history))

    summary = summarize_history(history)
    print(f"Recorded {len(new_rows)} assessments for {args.model_version}")
    print(f"History: {summary['total_rows']} rows, {summary['n_versions']} versions")


def cmd_backfill(args: argparse.Namespace) -> None:
    """Backfill unknown outcomes with ground-truth labels."""
    history = load_history(Path(args.history))
    if history.empty:
        print("No history to backfill.")
        return

    labels = pd.read_csv(args.labels)
    logger.info("Loaded %d labels from %s", len(labels), args.labels)

    history, n_filled = backfill_outcomes(
        history, labels,
        label_column=args.label_column,
        source=args.source,
    )

    save_history(history, Path(args.history))
    summary = summarize_history(history)
    print(f"Backfilled {n_filled} outcomes")
    print(
        f"Resolution: {summary['n_resolved']}/{summary['total_rows']} "
        f"({summary['resolution_rate']:.1%})"
    )


def cmd_stats(args: argparse.Namespace) -> None:
    """Print calibration statistics for the assessment history."""
    history = load_history(Path(args.history))
    if history.empty:
        print("No history found.")
        return

    summary = summarize_history(history)
    print("Assessment History Summary")
    print("=" * 40)
    for key, val in summary.items():
        print(f"  {key}: {val}")

    cal = compute_calibration_stats(history)
    print(f"\nCalibration ({cal['n_resolved']} resolved predictions)")
    print("-" * 40)
    if cal["brier_score"] is not None:
        print(f"  Brier score: {cal['brier_score']:.4f}")
        print(f"  ECE:         {cal['calibration_error']:.4f}")
        print(f"\n  {'Bin':>6}  {'Pred':>6}  {'Obs':>6}  {'Count':>6}")
        for b in cal["bins"]:
            pred = f"{b['predicted_mean']:.3f}" if b["predicted_mean"] is not None else "  N/A"
            obs = f"{b['observed_rate']:.3f}" if b["observed_rate"] is not None else "  N/A"
            print(f"  {b['bin_center']:>6.3f}  {pred:>6}  {obs:>6}  {b['count']:>6}")
    else:
        print("  No resolved predictions yet.")

    # Save stats JSON alongside history
    stats_path = Path(args.history).with_suffix(".stats.json")
    stats_path.write_text(json.dumps({"summary": summary, "calibration": cal}, indent=2))
    print(f"\nStats saved to {stats_path}")


def main() -> None:
    """Run assessment history update."""
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    paths = resolve_script_paths(args, _REPO)
    if args.history is None:
        args.history = str(paths.assessments / "assessment_history.csv") if paths else "data/out/assessments/assessment_history.csv"
    # Resolve backfill labels default if applicable
    if args.command == "backfill" and getattr(args, "labels", None) is None:
        args.labels = str(paths.lineage_tracking / "onset_labels_msd.csv") if paths else "data/out/02_lineage_tracking/onset_labels_msd.csv"

    if args.command == "record":
        cmd_record(args)
    elif args.command == "backfill":
        cmd_backfill(args)
    elif args.command == "stats":
        cmd_stats(args)
    else:
        print("Usage: update_assessment_history.py {record|backfill|stats}")
        sys.exit(1)


if __name__ == "__main__":
    main()
