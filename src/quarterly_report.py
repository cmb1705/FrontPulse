"""Quarterly briefing report generator for the operational horizon scanner.

Produces a structured markdown report with two-tier alerting:

1. **Watch list** (t >= 0.15): High-confidence onset alerts for immediate
   attention.  ~6:1 FP:TP ratio at ~55% recall.
2. **Extended monitoring** (t >= 0.07): Broader net for background tracking.
   ~12:1 FP:TP ratio at ~77% recall.

The report includes:
- Newly detected onsets per tier
- Updated assessments from previous quarters (backfill results)
- Top-N emerging lineages ranked by onset probability
- Forward-looking horizon estimates with confidence intervals
- Model performance versus previous quarters
- Calibration diagnostics

Output schema: a single markdown string suitable for stakeholder review.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

# Two-tier threshold system (settled project decision)
WATCH_LIST_THRESHOLD = 0.15
EXTENDED_MONITORING_THRESHOLD = 0.07


# ---------------------------------------------------------------------------
# Tier classification
# ---------------------------------------------------------------------------


def classify_alerts(
    predictions: pd.DataFrame,
    probability_column: str = "inflection_probability",
) -> pd.DataFrame:
    """Classify predictions into alert tiers.

    Args:
        predictions: DataFrame with lineage_id, quarter, and probability.
        probability_column: Column with onset probabilities.

    Returns:
        Copy of predictions with ``alert_tier`` column added:
        'watch_list', 'extended_monitoring', or 'below_threshold'.
    """
    df = predictions.copy()
    probs = df[probability_column]

    tiers = pd.Series("below_threshold", index=df.index)
    tiers[probs >= EXTENDED_MONITORING_THRESHOLD] = "extended_monitoring"
    tiers[probs >= WATCH_LIST_THRESHOLD] = "watch_list"
    df["alert_tier"] = tiers
    return df


# ---------------------------------------------------------------------------
# Report section builders
# ---------------------------------------------------------------------------


def _section_header(
    quarter_assessed: str,
    model_version: str,
    generated_at: str,
) -> str:
    """Build the report header."""
    lines = [
        f"# Quarterly Briefing Report: {quarter_assessed}",
        "",
        f"**Model version:** {model_version}",
        f"**Generated:** {generated_at}",
        "",
        "---",
        "",
    ]
    return "\n".join(lines)


def _section_alert_summary(classified: pd.DataFrame) -> str:
    """Build the alert tier summary section."""
    watch = classified[classified["alert_tier"] == "watch_list"]
    extended = classified[classified["alert_tier"] == "extended_monitoring"]
    below = classified[classified["alert_tier"] == "below_threshold"]

    lines = [
        "## Alert Summary",
        "",
        "| Tier | Threshold | Count | Description |",
        "|------|-----------|-------|-------------|",
        f"| Watch list | >= {WATCH_LIST_THRESHOLD} | {len(watch)} |"
        " High-confidence alerts for immediate review |",
        f"| Extended monitoring | >= {EXTENDED_MONITORING_THRESHOLD} |"
        f" {len(extended)} | Background tracking, broader net |",
        f"| Below threshold | < {EXTENDED_MONITORING_THRESHOLD} |"
        f" {len(below)} | No alert |",
        "",
    ]
    return "\n".join(lines)


def _section_watch_list(
    classified: pd.DataFrame,
    probability_column: str = "inflection_probability",
    top_n: int = 20,
) -> str:
    """Build the watch list detail section."""
    watch = classified[classified["alert_tier"] == "watch_list"].copy()
    if watch.empty:
        return "## Watch List\n\nNo lineages above watch list threshold.\n\n"

    watch = watch.sort_values(probability_column, ascending=False).head(top_n)

    lines = [
        "## Watch List (High-Confidence Alerts)",
        "",
        f"Top {min(top_n, len(watch))} lineages with probability"
        f" >= {WATCH_LIST_THRESHOLD}:",
        "",
        "| Rank | Lineage ID | Probability | Quarter |",
        "|------|------------|-------------|---------|",
    ]

    for rank, (_, row) in enumerate(watch.iterrows(), 1):
        prob = row[probability_column]
        lines.append(
            f"| {rank} | {row['lineage_id']} | {prob:.4f} | {row['quarter']} |"
        )

    lines.append("")
    return "\n".join(lines)


def _section_extended_monitoring(
    classified: pd.DataFrame,
    probability_column: str = "inflection_probability",
    top_n: int = 20,
) -> str:
    """Build the extended monitoring section."""
    extended = classified[classified["alert_tier"] == "extended_monitoring"].copy()
    if extended.empty:
        return "## Extended Monitoring\n\nNo additional lineages in extended tier.\n\n"

    extended = extended.sort_values(probability_column, ascending=False).head(top_n)

    lines = [
        "## Extended Monitoring (Background Tracking)",
        "",
        f"Top {min(top_n, len(extended))} lineages with probability in"
        f" [{EXTENDED_MONITORING_THRESHOLD}, {WATCH_LIST_THRESHOLD}):",
        "",
        "| Rank | Lineage ID | Probability | Quarter |",
        "|------|------------|-------------|---------|",
    ]

    for rank, (_, row) in enumerate(extended.iterrows(), 1):
        prob = row[probability_column]
        lines.append(
            f"| {rank} | {row['lineage_id']} | {prob:.4f} | {row['quarter']} |"
        )

    lines.append("")
    return "\n".join(lines)


def _section_assessment_updates(
    history: pd.DataFrame,
) -> str:
    """Build the assessment updates section showing backfilled outcomes."""
    if history.empty:
        return "## Assessment Updates\n\nNo assessment history available.\n\n"

    # Find rows backfilled since last quarter
    backfilled = history[
        (history["backfilled_at"] != "")
        & (history["actual_outcome"].isin([0, 1]))
    ].copy()

    if backfilled.empty:
        return "## Assessment Updates\n\nNo outcomes backfilled yet.\n\n"

    n_positive = int((backfilled["actual_outcome"] == 1).sum())
    n_negative = int((backfilled["actual_outcome"] == 0).sum())

    lines = [
        "## Assessment Updates",
        "",
        f"**Resolved predictions:** {len(backfilled)}"
        f" ({n_positive} positive, {n_negative} negative)",
        "",
    ]

    # Show accuracy of resolved predictions
    if not backfilled.empty:
        correct = (backfilled["predicted_label"] == backfilled["actual_outcome"]).sum()
        accuracy = correct / len(backfilled) if len(backfilled) > 0 else 0
        lines.append(f"**Prediction accuracy (resolved):** {accuracy:.1%}")
        lines.append("")

    return "\n".join(lines)


def _section_horizon_estimates(
    estimates: pd.DataFrame | None,
    top_n: int = 10,
) -> str:
    """Build the forward-looking estimates section."""
    if estimates is None or estimates.empty:
        return "## Forward-Looking Estimates\n\nNo horizon estimates available.\n\n"

    # Show horizon-1 estimates ranked by point estimate
    h1 = estimates[estimates["horizon"] == 1].copy()
    if h1.empty:
        return "## Forward-Looking Estimates\n\nNo 1-quarter estimates available.\n\n"

    h1 = h1.sort_values("point_estimate", ascending=False).head(top_n)

    lines = [
        "## Forward-Looking Estimates (Next Quarter)",
        "",
        f"Top {min(top_n, len(h1))} lineages by predicted onset probability:",
        "",
        "| Lineage ID | Point Est. | 90% CI | Target Quarter |",
        "|------------|------------|--------|----------------|",
    ]

    for _, row in h1.iterrows():
        ci = f"[{row['lower_bound']:.3f}, {row['upper_bound']:.3f}]"
        lines.append(
            f"| {row['lineage_id']} | {row['point_estimate']:.4f}"
            f" | {ci} | {row['quarter_target']} |"
        )

    lines.append("")
    return "\n".join(lines)


def _section_calibration(
    calibration_stats: dict[str, Any] | None,
) -> str:
    """Build the calibration diagnostics section."""
    if calibration_stats is None:
        return "## Calibration Diagnostics\n\nNo calibration data available.\n\n"

    lines = [
        "## Calibration Diagnostics",
        "",
    ]

    n_res = calibration_stats.get("n_resolved", 0)
    n_unk = calibration_stats.get("n_unknown", 0)
    lines.append(f"**Resolved predictions:** {n_res}")
    lines.append(f"**Pending predictions:** {n_unk}")
    lines.append("")

    brier = calibration_stats.get("brier_score")
    ece = calibration_stats.get("calibration_error")
    if brier is not None:
        lines.append(f"**Brier score:** {brier:.4f}")
        lines.append(f"**Expected Calibration Error (ECE):** {ece:.4f}")
        lines.append("")

        bins = calibration_stats.get("bins", [])
        if bins:
            lines.append("| Bin Center | Predicted | Observed | Count |")
            lines.append("|------------|-----------|----------|-------|")
            for b in bins:
                pred = f"{b['predicted_mean']:.3f}" if b["predicted_mean"] is not None else "N/A"
                obs = f"{b['observed_rate']:.3f}" if b["observed_rate"] is not None else "N/A"
                lines.append(f"| {b['bin_center']:.3f} | {pred} | {obs} | {b['count']} |")
            lines.append("")
    else:
        lines.append("No resolved predictions available for calibration.")
        lines.append("")

    return "\n".join(lines)


def _section_model_comparison(
    current_version: str,
    previous_version: str | None,
    comparison: dict[str, Any] | None,
) -> str:
    """Build the model version comparison section."""
    if comparison is None or previous_version is None:
        return (
            "## Model Performance\n\n"
            f"**Current model:** {current_version}\n"
            "No previous version available for comparison.\n\n"
        )

    lines = [
        "## Model Performance",
        "",
        f"**Current:** {current_version}",
        f"**Previous:** {previous_version}",
        "",
    ]

    improved = comparison.get("improved")
    if improved is True:
        lines.append("Performance **improved** versus previous version.")
    elif improved is False:
        lines.append("Performance **regressed** versus previous version.")
    else:
        lines.append("Unable to compare (no common metrics).")

    lines.append("")

    deltas = comparison.get("deltas", {})
    if deltas:
        lines.append("| Metric | Delta |")
        lines.append("|--------|-------|")
        for metric, delta in sorted(deltas.items()):
            sign = "+" if delta > 0 else ""
            lines.append(f"| {metric} | {sign}{delta:.4f} |")
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main report assembly
# ---------------------------------------------------------------------------


def generate_quarterly_report(
    predictions: pd.DataFrame,
    history: pd.DataFrame,
    quarter_assessed: str,
    model_version: str,
    horizon_estimates: pd.DataFrame | None = None,
    calibration_stats: dict[str, Any] | None = None,
    model_comparison: dict[str, Any] | None = None,
    previous_version: str | None = None,
    probability_column: str = "inflection_probability",
    watch_top_n: int = 20,
    extended_top_n: int = 20,
    horizon_top_n: int = 10,
) -> str:
    """Generate a complete quarterly briefing report as markdown.

    Args:
        predictions: Latest MSD predictions with lineage_id, quarter,
            and probability column.
        history: Assessment history DataFrame.
        quarter_assessed: Current quarter (YYYYQN).
        model_version: Current model version ID.
        horizon_estimates: Optional forward-looking estimates DataFrame.
        calibration_stats: Optional calibration statistics dict.
        model_comparison: Optional version comparison dict.
        previous_version: Previous model version ID for comparison.
        probability_column: Column name for onset probabilities.
        watch_top_n: Max lineages to show in watch list.
        extended_top_n: Max lineages to show in extended monitoring.
        horizon_top_n: Max lineages to show in horizon estimates.

    Returns:
        Complete markdown report string.
    """
    generated_at = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    # Classify predictions into tiers
    classified = classify_alerts(predictions, probability_column)

    sections = [
        _section_header(quarter_assessed, model_version, generated_at),
        _section_alert_summary(classified),
        _section_watch_list(classified, probability_column, watch_top_n),
        _section_extended_monitoring(classified, probability_column, extended_top_n),
        _section_assessment_updates(history),
        _section_horizon_estimates(horizon_estimates, horizon_top_n),
        _section_calibration(calibration_stats),
        _section_model_comparison(model_version, previous_version, model_comparison),
    ]

    report = "\n".join(sections)
    logger.info(
        "Generated quarterly report for %s (%d characters)",
        quarter_assessed, len(report),
    )
    return report


def summarize_report_stats(
    predictions: pd.DataFrame,
    probability_column: str = "inflection_probability",
) -> dict[str, Any]:
    """Compute summary statistics for report metadata.

    Args:
        predictions: Classified predictions DataFrame.
        probability_column: Probability column name.

    Returns:
        Dictionary with tier counts and probability statistics.
    """
    classified = classify_alerts(predictions, probability_column)
    return {
        "total_lineages": len(classified),
        "watch_list_count": int((classified["alert_tier"] == "watch_list").sum()),
        "extended_monitoring_count": int(
            (classified["alert_tier"] == "extended_monitoring").sum()
        ),
        "below_threshold_count": int(
            (classified["alert_tier"] == "below_threshold").sum()
        ),
        "mean_probability": round(float(classified[probability_column].mean()), 4),
        "max_probability": round(float(classified[probability_column].max()), 4),
    }
