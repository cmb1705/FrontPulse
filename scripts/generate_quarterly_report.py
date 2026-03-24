#!/usr/bin/env python3
"""Generate a quarterly briefing report for the operational horizon scanner.

Assembles predictions, assessment history, horizon estimates, and calibration
data into a structured markdown report with two-tier alerting.

Usage::

    python scripts/generate_quarterly_report.py \\
        --predictions data/out/experiments/msd_latest/breakthrough_predictions.csv \\
        --quarter 2025Q1 \\
        --model-version v_20260323_001 \\
        --out data/out/assessments/quarterly_report_2025Q1.md
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import pandas as pd  # noqa: E402

from src.assessment_history import (  # noqa: E402
    compute_calibration_stats,
    load_history,
)
from src.quarterly_report import (  # noqa: E402
    generate_quarterly_report,
    summarize_report_stats,
)

logger = logging.getLogger(__name__)

_DEFAULT_PREDICTIONS = "data/out/experiments/msd_latest/breakthrough_predictions.csv"
_DEFAULT_HISTORY = "data/out/assessments/assessment_history.csv"
_DEFAULT_ESTIMATES = "data/out/assessments/horizon_estimates.csv"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate quarterly briefing report.",
    )
    parser.add_argument(
        "--predictions", default=_DEFAULT_PREDICTIONS,
        help="Path to latest MSD predictions CSV (default: %(default)s)",
    )
    parser.add_argument(
        "--history", default=_DEFAULT_HISTORY,
        help="Path to assessment history CSV (default: %(default)s)",
    )
    parser.add_argument(
        "--horizon-estimates", default=_DEFAULT_ESTIMATES,
        help="Path to horizon estimates CSV (default: %(default)s)",
    )
    parser.add_argument(
        "--quarter", required=True,
        help="Quarter being assessed (e.g., 2025Q1)",
    )
    parser.add_argument(
        "--model-version", required=True,
        help="Model version ID (e.g., v_20260323_001)",
    )
    parser.add_argument(
        "--out", required=True,
        help="Output path for the markdown report",
    )
    parser.add_argument(
        "--probability-column", default="inflection_probability",
        help="Probability column name (default: %(default)s)",
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Run quarterly report generation."""
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    # Load predictions
    pred_path = Path(args.predictions)
    if not pred_path.exists():
        logger.error("Predictions file not found: %s", pred_path)
        sys.exit(1)
    predictions = pd.read_csv(pred_path)
    logger.info("Loaded %d predictions from %s", len(predictions), pred_path)

    # Load assessment history
    history = load_history(Path(args.history))
    logger.info("Loaded %d history rows", len(history))

    # Load horizon estimates (optional)
    horizon_estimates = None
    est_path = Path(args.horizon_estimates)
    if est_path.exists():
        horizon_estimates = pd.read_csv(est_path)
        logger.info("Loaded %d horizon estimates", len(horizon_estimates))

    # Compute calibration stats
    cal_stats = compute_calibration_stats(history) if not history.empty else None

    # Generate report
    report = generate_quarterly_report(
        predictions=predictions,
        history=history,
        quarter_assessed=args.quarter,
        model_version=args.model_version,
        horizon_estimates=horizon_estimates,
        calibration_stats=cal_stats,
        probability_column=args.probability_column,
    )

    # Save report
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")
    logger.info("Report saved to %s", out_path)

    # Print summary
    stats = summarize_report_stats(predictions, args.probability_column)
    print(f"Report generated: {out_path}")
    print(f"  Watch list:           {stats['watch_list_count']} lineages")
    print(f"  Extended monitoring:  {stats['extended_monitoring_count']} lineages")
    print(f"  Below threshold:      {stats['below_threshold_count']} lineages")


if __name__ == "__main__":
    main()
