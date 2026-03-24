#!/usr/bin/env python3
"""Generate forward-looking onset probability estimates for active lineages.

Produces per-lineage estimates for the next 1-4 quarters with conformal
prediction intervals calibrated on historical assessment data.

Usage::

    python scripts/generate_horizon_estimates.py \\
        --predictions data/out/experiments/msd_latest/breakthrough_predictions.csv \\
        --history data/out/assessments/assessment_history.csv \\
        --out data/out/assessments/horizon_estimates.csv
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

from src.assessment_history import load_history  # noqa: E402
from src.horizon_estimates import (  # noqa: E402
    generate_horizon_estimates,
    summarize_horizon_estimates,
)

logger = logging.getLogger(__name__)

_DEFAULT_PREDICTIONS = "data/out/experiments/msd_latest/breakthrough_predictions.csv"
_DEFAULT_HISTORY = "data/out/assessments/assessment_history.csv"
_DEFAULT_OUT = "data/out/assessments/horizon_estimates.csv"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate forward-looking onset probability estimates.",
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
        "--out", default=_DEFAULT_OUT,
        help="Output path for horizon estimates CSV (default: %(default)s)",
    )
    parser.add_argument(
        "--max-horizon", type=int, default=4,
        help="Maximum forecast horizon in quarters (default: %(default)s)",
    )
    parser.add_argument(
        "--alpha", type=float, default=0.10,
        help="Significance level for confidence intervals (default: %(default)s)",
    )
    parser.add_argument(
        "--trend-lookback", type=int, default=4,
        help="Quarters of history for trend estimation (default: %(default)s)",
    )
    parser.add_argument(
        "--trend-damping", type=float, default=0.7,
        help="Per-step damping factor for trend extrapolation (default: %(default)s)",
    )
    parser.add_argument(
        "--probability-column", default="inflection_probability",
        help="Probability column name in predictions (default: %(default)s)",
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Run horizon estimate generation."""
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

    # Rename probability column if needed
    if args.probability_column != "inflection_probability":
        if args.probability_column in predictions.columns:
            predictions = predictions.rename(
                columns={args.probability_column: "inflection_probability"},
            )
        else:
            logger.error(
                "Column '%s' not found in predictions", args.probability_column,
            )
            sys.exit(1)

    # Load assessment history
    history = load_history(Path(args.history))
    logger.info("Loaded %d history rows from %s", len(history), args.history)

    # Generate estimates
    estimates = generate_horizon_estimates(
        latest_predictions=predictions,
        history=history,
        max_horizon=args.max_horizon,
        alpha=args.alpha,
        trend_lookback=args.trend_lookback,
        trend_damping=args.trend_damping,
    )

    # Save output
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    estimates.to_csv(out_path, index=False)
    logger.info("Saved %d estimates to %s", len(estimates), out_path)

    # Print summary
    summary = summarize_horizon_estimates(estimates)
    print("Horizon Estimates Summary")
    print("=" * 40)
    for key, val in summary.items():
        print(f"  {key}: {val}")


if __name__ == "__main__":
    main()
