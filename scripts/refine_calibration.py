#!/usr/bin/env python3
"""Refine calibration using operational assessment data.

Fits isotonic calibration on resolved prediction-outcome pairs, computes
per-version calibration metrics, and alerts on degradation.

Usage::

    python scripts/refine_calibration.py \\
        --history data/out/assessments/assessment_history.csv \\
        --model-version v_20260323_001 \\
        --cal-history data/out/assessments/calibration_history.json
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from _path_bootstrap import ensure_repo_imports

_REPO = ensure_repo_imports()

from src.assessment_history import load_history  # noqa: E402
from src.calibration_tracker import (  # noqa: E402
    MIN_RESOLVED_FOR_CALIBRATION,
    check_degradation,
    compute_calibration_snapshot,
    fit_isotonic_calibrator,
    load_calibration_history,
    save_calibration_history,
)
from src.domain_registry import add_domain_args, resolve_script_paths  # noqa: E402

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Refine MSD calibration with operational data.",
    )
    parser.add_argument(
        "--history", default=None,
        help="Path to assessment history CSV",
    )
    parser.add_argument(
        "--model-version", required=True,
        help="Model version to evaluate (e.g., v_20260323_001)",
    )
    parser.add_argument(
        "--cal-history", default=None,
        help="Path to calibration history JSON",
    )
    parser.add_argument(
        "--fit-calibrator", action="store_true",
        help="Fit isotonic calibrator on all resolved data",
    )
    parser.add_argument(
        "--degradation-k", type=float, default=2.0,
        help="Std devs for degradation threshold (default: %(default)s)",
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    add_domain_args(parser)
    return parser.parse_args()


def main() -> None:
    """Run calibration refinement."""
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    paths = resolve_script_paths(args, _REPO)
    if args.history is None:
        args.history = str(paths.assessments / "assessment_history.csv") if paths else "data/out/assessments/assessment_history.csv"
    if args.cal_history is None:
        args.cal_history = str(paths.assessments / "calibration_history.json") if paths else "data/out/assessments/calibration_history.json"

    # Load assessment history
    history = load_history(Path(args.history))
    if history.empty:
        print("No assessment history found.")
        sys.exit(0)

    resolved = history[history["actual_outcome"].isin([0, 1])]
    print(f"Assessment history: {len(history)} total, {len(resolved)} resolved")

    # Optionally fit isotonic calibrator on all resolved data
    calibrator = None
    if args.fit_calibrator and len(resolved) >= MIN_RESOLVED_FOR_CALIBRATION:
        probs = resolved["probability"].values.astype(float)
        outcomes = resolved["actual_outcome"].values.astype(float)
        calibrator = fit_isotonic_calibrator(probs, outcomes)
        print(f"Fitted isotonic calibrator on {len(resolved)} samples")
    elif args.fit_calibrator:
        print(
            f"Insufficient resolved predictions ({len(resolved)})"
            f" for calibration (need {MIN_RESOLVED_FOR_CALIBRATION})"
        )

    # Compute snapshot for current version
    snapshot = compute_calibration_snapshot(
        history, args.model_version, calibrator=calibrator,
    )

    print(f"\nCalibration for {args.model_version}:")
    print(f"  Resolved predictions: {snapshot.n_resolved}")
    if snapshot.brier_score is not None:
        print(f"  Brier score: {snapshot.brier_score:.4f}")
        print(f"  ECE:         {snapshot.ece:.4f}")
        print(f"  Calibrated:  {snapshot.is_calibrated}")
    else:
        print("  Insufficient data for metrics")

    # Load calibration history and check degradation
    cal_history = load_calibration_history(Path(args.cal_history))

    alerts = check_degradation(
        snapshot, cal_history, k=args.degradation_k,
    )

    if alerts:
        print("\n-- DEGRADATION ALERTS --")
        for alert in alerts:
            print(
                f"  [{alert.severity.upper()}] {alert.metric}:"
                f" {alert.current_value:.4f}"
                f" (baseline {alert.baseline_mean:.4f}"
                f" +/- {alert.baseline_std:.4f},"
                f" threshold {alert.threshold:.4f})"
            )
    else:
        print("\nNo calibration degradation detected.")

    # Save updated history
    cal_history.add_snapshot(snapshot)
    save_calibration_history(cal_history, Path(args.cal_history))
    print(f"\nCalibration history saved ({len(cal_history.snapshots)} versions)")


if __name__ == "__main__":
    main()
