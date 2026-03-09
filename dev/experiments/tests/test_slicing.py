"""Tests for src/slicing.py - Temporal and categorical slicing."""
from __future__ import annotations

import pandas as pd
import pytest

from src.slicing import apply_slices


@pytest.mark.unit
class TestApplySlices:
    """Test suite for apply_slices function."""

    def test_basic_slicing(self, sample_works_df, sample_slices_yaml):
        """Test basic slicing functionality."""
        # Add required columns
        df = sample_works_df.copy()
        df["pub_year"] = pd.to_datetime(df["publication_date"]).dt.year

        result = apply_slices(df, sample_slices_yaml)

        assert isinstance(result, dict)
        assert "recent" in result
        assert "high_impact" in result

    def test_recent_slice_filters_correctly(self, sample_works_df, sample_slices_yaml):
        """Test that 'recent' slice filters by year >= 2021."""
        df = sample_works_df.copy()
        df["pub_year"] = pd.to_datetime(df["publication_date"]).dt.year

        result = apply_slices(df, sample_slices_yaml)

        recent_df = result["recent"]
        assert len(recent_df) == 3  # 2021, 2022, 2023
        assert all(recent_df["pub_year"] >= 2021)

    def test_high_impact_slice_filters_correctly(self, sample_works_df, sample_slices_yaml):
        """Test that 'high_impact' slice filters by cited_by_count > 10."""
        df = sample_works_df.copy()
        df["pub_year"] = pd.to_datetime(df["publication_date"]).dt.year

        result = apply_slices(df, sample_slices_yaml)

        high_impact_df = result["high_impact"]
        assert len(high_impact_df) == 1  # Only W002 with 25 citations
        assert all(high_impact_df["cited_by_count"] > 10)

    def test_cutoff_parameter(self, sample_works_df, sample_slices_yaml):
        """Test that cutoff parameter limits data by date."""
        df = sample_works_df.copy()
        df["publication_date"] = pd.to_datetime(df["publication_date"])
        df["pub_year"] = df["publication_date"].dt.year

        cutoff = pd.Timestamp("2021-12-31")
        result = apply_slices(df, sample_slices_yaml, cutoff=cutoff)

        # All slices should only include data <= cutoff
        for slice_name, slice_df in result.items():
            if len(slice_df) > 0:
                assert all(slice_df["publication_date"] <= cutoff)

    def test_empty_slice_result(self, sample_works_df, sample_slices_yaml):
        """Test that slices can return empty DataFrames."""
        df = sample_works_df.copy()
        df["pub_year"] = 2010  # All data from 2010
        df["cited_by_count"] = 5  # All low citations

        result = apply_slices(df, sample_slices_yaml)

        # Recent slice should be empty (requires >= 2021)
        assert len(result["recent"]) == 0
        # High impact should be empty (requires > 10 citations)
        assert len(result["high_impact"]) == 0

    def test_preserves_dataframe_columns(self, sample_works_df, sample_slices_yaml):
        """Test that all original columns are preserved in slices."""
        df = sample_works_df.copy()
        df["pub_year"] = pd.to_datetime(df["publication_date"]).dt.year
        original_columns = set(df.columns)

        result = apply_slices(df, sample_slices_yaml)

        for slice_df in result.values():
            if len(slice_df) > 0:
                assert original_columns.issubset(set(slice_df.columns))

    def test_no_data_mutation(self, sample_works_df, sample_slices_yaml):
        """Test that original DataFrame is not modified."""
        df = sample_works_df.copy()
        df["pub_year"] = pd.to_datetime(df["publication_date"]).dt.year
        original_len = len(df)
        original_columns = list(df.columns)

        _ = apply_slices(df, sample_slices_yaml)

        assert len(df) == original_len
        assert list(df.columns) == original_columns

    def test_multiple_filters_in_slice(self, sample_works_df, temp_dir):
        """Test slice with multiple filter conditions."""
        slices_path = temp_dir / "multi_filter.yaml"
        slices_content = """
slices:
  recent_high_impact:
    filter:
      - pub_year >= 2021
      - cited_by_count > 15
        """
        slices_path.write_text(slices_content)

        df = sample_works_df.copy()
        df["pub_year"] = pd.to_datetime(df["publication_date"]).dt.year

        result = apply_slices(df, str(slices_path))

        # Should only include W002 (2021, 25 citations)
        assert len(result["recent_high_impact"]) == 1
        assert result["recent_high_impact"]["id"].iloc[0] == "W002"

    def test_empty_input_dataframe(self, sample_slices_yaml):
        """Test handling of empty input DataFrame."""
        df = pd.DataFrame({
            "id": [],
            "pub_year": [],
            "cited_by_count": []
        })

        result = apply_slices(df, sample_slices_yaml)

        assert isinstance(result, dict)
        for slice_df in result.values():
            assert len(slice_df) == 0
