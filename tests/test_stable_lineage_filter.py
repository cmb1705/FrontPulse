"""Unit tests for src.stable_lineage_filter."""

from __future__ import annotations

import pandas as pd
import pytest

from src.stable_lineage_filter import (
    compute_lineage_lifespans,
    filter_stable_lineages,
    summarize_filter,
)


def _make_timeseries(lineage_quarters: dict[int, list[str]]) -> pd.DataFrame:
    """Build a minimal timeseries DataFrame from lineage_id -> quarter lists."""
    rows = []
    for lid, quarters in lineage_quarters.items():
        for q in quarters:
            rows.append({"lineage_id": lid, "quarter": q, "new_works": 1})
    return pd.DataFrame(rows)


# -- compute_lineage_lifespans ------------------------------------------------


class TestComputeLifespans:
    def test_single_quarter_lineage(self) -> None:
        df = _make_timeseries({1: ["2020Q1"]})
        result = compute_lineage_lifespans(df)
        assert result == {1: 1}

    def test_multi_quarter_lineages(self) -> None:
        df = _make_timeseries({
            1: ["2020Q1", "2020Q2", "2020Q3"],
            2: ["2020Q1"],
            3: ["2020Q1", "2020Q2"],
        })
        result = compute_lineage_lifespans(df)
        assert result == {1: 3, 2: 1, 3: 2}

    def test_empty_dataframe(self) -> None:
        df = pd.DataFrame(columns=["lineage_id", "quarter", "new_works"])
        result = compute_lineage_lifespans(df)
        assert result == {}


# -- filter_stable_lineages ---------------------------------------------------


class TestFilterStableLineages:
    def test_min_quarters_filters_correctly(self) -> None:
        df = _make_timeseries({
            1: ["2020Q1", "2020Q2", "2020Q3", "2020Q4"],
            2: ["2020Q1"],
            3: ["2020Q1", "2020Q2"],
        })
        result = filter_stable_lineages(df, min_quarters=3)
        assert result == {1}

    def test_min_quarters_includes_exact_match(self) -> None:
        df = _make_timeseries({
            1: ["2020Q1", "2020Q2", "2020Q3"],
        })
        result = filter_stable_lineages(df, min_quarters=3)
        assert result == {1}

    def test_max_quarters_upper_bound(self) -> None:
        df = _make_timeseries({
            1: ["2020Q1", "2020Q2", "2020Q3", "2020Q4"],
            2: ["2020Q1", "2020Q2"],
            3: ["2020Q1"],
        })
        result = filter_stable_lineages(df, min_quarters=2, max_quarters=3)
        assert result == {2}

    def test_default_threshold_is_8(self) -> None:
        quarters = [f"2020Q{q}" for q in [1, 2, 3, 4]] + [
            f"2021Q{q}" for q in [1, 2, 3, 4]
        ]
        df = _make_timeseries({
            1: quarters,       # 8 quarters -- should pass
            2: quarters[:7],   # 7 quarters -- should fail
        })
        result = filter_stable_lineages(df)
        assert result == {1}

    def test_empty_result_when_none_qualify(self) -> None:
        df = _make_timeseries({1: ["2020Q1"], 2: ["2020Q1"]})
        result = filter_stable_lineages(df, min_quarters=5)
        assert result == set()


# -- summarize_filter ---------------------------------------------------------


class TestSummarizeFilter:
    def test_summary_structure(self) -> None:
        df = _make_timeseries({
            1: ["2020Q1", "2020Q2", "2020Q3"],
            2: ["2020Q1"],
        })
        summary = summarize_filter(df, min_quarters=2)
        assert summary["total_lineages"] == 2
        assert summary["stable_lineages"] == 1
        assert summary["stable_pct"] == 50.0
        assert summary["total_records"] == 4
        assert summary["stable_records"] == 3
        assert summary["min_quarters"] == 2
        assert summary["max_quarters"] is None

    def test_summary_with_max_quarters(self) -> None:
        df = _make_timeseries({
            1: ["2020Q1", "2020Q2", "2020Q3", "2020Q4"],
            2: ["2020Q1", "2020Q2"],
            3: ["2020Q1"],
        })
        summary = summarize_filter(df, min_quarters=2, max_quarters=3)
        assert summary["stable_lineages"] == 1
        assert summary["max_quarters"] == 3

    def test_summary_all_stable(self) -> None:
        df = _make_timeseries({
            1: ["2020Q1", "2020Q2", "2020Q3"],
            2: ["2020Q1", "2020Q2", "2020Q3", "2020Q4"],
        })
        summary = summarize_filter(df, min_quarters=2)
        assert summary["stable_lineages"] == 2
        assert summary["stable_pct"] == 100.0
