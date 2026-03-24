"""Unit tests for src.stability_report."""

from __future__ import annotations

import pandas as pd
import pytest

from src.stability_report import (
    compute_activity_stats,
    compute_lifespan_stats,
    compute_pia_stats,
    compute_stability_report,
    compute_vi_stats,
    format_report_text,
)


def _make_timeseries(
    lineage_quarters: dict[int, list[str]],
    extra_columns: dict[str, list] | None = None,
) -> pd.DataFrame:
    """Build a minimal timeseries DataFrame from lineage_id -> quarter lists."""
    rows = []
    for lid, quarters in lineage_quarters.items():
        for q in quarters:
            rows.append({"lineage_id": lid, "quarter": q, "new_works": 5})
    df = pd.DataFrame(rows)
    if extra_columns:
        for col, values in extra_columns.items():
            df[col] = values
    return df


# -- compute_lifespan_stats ---------------------------------------------------


class TestLifespanStats:
    def test_basic_stats(self) -> None:
        df = _make_timeseries({
            1: ["2020Q1", "2020Q2", "2020Q3"],
            2: ["2020Q1"],
            3: ["2020Q1", "2020Q2"],
        })
        result = compute_lifespan_stats(df)
        assert result["total_lineages"] == 3
        assert result["mean_lifespan"] == 2.0
        assert result["median_lifespan"] == 2
        assert result["min_lifespan"] == 1
        assert result["max_lifespan"] == 3

    def test_percentiles_present(self) -> None:
        df = _make_timeseries({i: [f"2020Q{q}" for q in range(1, 5)] for i in range(20)})
        result = compute_lifespan_stats(df)
        assert "percentiles" in result
        assert "p50" in result["percentiles"]
        assert result["percentiles"]["p50"] == 4

    def test_buckets_present(self) -> None:
        df = _make_timeseries({
            1: ["2020Q1"],
            2: ["2020Q1", "2020Q2"],
            3: ["2020Q1", "2020Q2", "2020Q3"],
        })
        result = compute_lifespan_stats(df)
        assert "buckets" in result
        labels = [b["label"] for b in result["buckets"]]
        assert "1_quarter" in labels
        assert "2_quarters" in labels

    def test_empty_dataframe(self) -> None:
        df = pd.DataFrame(columns=["lineage_id", "quarter", "new_works"])
        result = compute_lifespan_stats(df)
        assert result["total_lineages"] == 0

    def test_bucket_counts_sum_to_total(self) -> None:
        df = _make_timeseries({
            1: ["2020Q1"],
            2: ["2020Q1", "2020Q2"],
            3: [f"2020Q{q}" for q in [1, 2, 3, 4]] + [f"2021Q{q}" for q in [1, 2, 3, 4]],
        })
        result = compute_lifespan_stats(df)
        total_in_buckets = sum(b["count"] for b in result["buckets"])
        assert total_in_buckets == result["total_lineages"]


# -- compute_vi_stats ---------------------------------------------------------


class TestVIStats:
    def test_basic_vi(self) -> None:
        rows = []
        for q in ["2020Q1", "2020Q2", "2020Q3", "2020Q4"]:
            rows.append({"lineage_id": 1, "quarter": q, "VI_vs_prev_quarter": 0.5})
        df = pd.DataFrame(rows)
        result = compute_vi_stats(df)
        assert result["available"] is True
        assert result["n_quarters"] == 4
        assert result["mean_vi"] == 0.5

    def test_missing_column(self) -> None:
        df = pd.DataFrame({"lineage_id": [1], "quarter": ["2020Q1"]})
        result = compute_vi_stats(df)
        assert result["available"] is False

    def test_temporal_trend_with_enough_quarters(self) -> None:
        rows = []
        quarters = [f"20{y}Q{q}" for y in range(20, 23) for q in range(1, 5)]
        for i, q in enumerate(quarters):
            rows.append({
                "lineage_id": 1,
                "quarter": q,
                "VI_vs_prev_quarter": 0.1 * (i + 1),
            })
        df = pd.DataFrame(rows)
        result = compute_vi_stats(df)
        assert "temporal_trend" in result
        assert result["temporal_trend"]["early_mean"] < result["temporal_trend"]["late_mean"]

    def test_no_temporal_trend_with_few_quarters(self) -> None:
        rows = [
            {"lineage_id": 1, "quarter": "2020Q1", "VI_vs_prev_quarter": 0.3},
            {"lineage_id": 1, "quarter": "2020Q2", "VI_vs_prev_quarter": 0.4},
        ]
        df = pd.DataFrame(rows)
        result = compute_vi_stats(df)
        assert result["available"] is True
        assert "temporal_trend" not in result


# -- compute_pia_stats --------------------------------------------------------


class TestPIAStats:
    def test_basic_pia(self) -> None:
        df = pd.DataFrame({
            "lineage_id": [1, 1, 2, 2],
            "quarter": ["2020Q1", "2020Q2", "2020Q1", "2020Q2"],
            "pia_rate": [0.8, 0.9, 0.7, 0.85],
        })
        result = compute_pia_stats(df)
        assert result["available"] is True
        assert result["n_records"] == 4
        assert 0.0 <= result["mean_pia"] <= 1.0

    def test_missing_column(self) -> None:
        df = pd.DataFrame({"lineage_id": [1], "quarter": ["2020Q1"]})
        result = compute_pia_stats(df)
        assert result["available"] is False

    def test_all_nan(self) -> None:
        df = pd.DataFrame({
            "lineage_id": [1, 2],
            "quarter": ["2020Q1", "2020Q2"],
            "pia_rate": [float("nan"), float("nan")],
        })
        result = compute_pia_stats(df)
        assert result["available"] is False


# -- compute_activity_stats ---------------------------------------------------


class TestActivityStats:
    def test_basic_activity(self) -> None:
        df = pd.DataFrame({
            "lineage_id": [1, 1, 2],
            "quarter": ["2020Q1", "2020Q2", "2020Q1"],
            "new_works": [10, 0, 5],
        })
        result = compute_activity_stats(df)
        assert result["available"] is True
        assert result["n_records"] == 3
        assert result["zero_count"] == 1
        assert result["zero_pct"] == pytest.approx(33.3, abs=0.1)

    def test_missing_column(self) -> None:
        df = pd.DataFrame({"lineage_id": [1], "quarter": ["2020Q1"]})
        result = compute_activity_stats(df)
        assert result["available"] is False

    def test_no_zeros(self) -> None:
        df = pd.DataFrame({
            "lineage_id": [1, 2],
            "quarter": ["2020Q1", "2020Q1"],
            "new_works": [5, 10],
        })
        result = compute_activity_stats(df)
        assert result["zero_count"] == 0
        assert result["zero_pct"] == 0.0


# -- compute_stability_report ------------------------------------------------


class TestStabilityReport:
    def test_all_sections_present(self) -> None:
        df = pd.DataFrame({
            "lineage_id": [1, 1, 2],
            "quarter": ["2020Q1", "2020Q2", "2020Q1"],
            "new_works": [5, 3, 7],
            "VI_vs_prev_quarter": [0.2, 0.3, 0.1],
            "pia_rate": [0.8, 0.9, 0.7],
        })
        report = compute_stability_report(df)
        assert "lifespan" in report
        assert "vi" in report
        assert "pia" in report
        assert "activity" in report

    def test_missing_optional_columns(self) -> None:
        df = _make_timeseries({1: ["2020Q1", "2020Q2"]})
        report = compute_stability_report(df)
        assert report["vi"]["available"] is False
        assert report["pia"]["available"] is False
        assert report["activity"]["available"] is True


# -- format_report_text -------------------------------------------------------


class TestFormatReportText:
    def test_returns_string(self) -> None:
        df = _make_timeseries({1: ["2020Q1", "2020Q2"]})
        report = compute_stability_report(df)
        text = format_report_text(report)
        assert isinstance(text, str)
        assert "COMMUNITY STABILITY REPORT" in text

    def test_empty_report(self) -> None:
        report = {
            "lifespan": {"total_lineages": 0},
            "vi": {"available": False, "reason": "No VI data"},
            "pia": {"available": False, "reason": "No PIA data"},
            "activity": {"available": False, "reason": "No activity data"},
        }
        text = format_report_text(report)
        assert "No lineage data available" in text
        assert "Not available" in text
