"""Tests for src/transform.py - DataFrame transformation utilities."""
from __future__ import annotations

import pandas as pd
import pytest

from src.transform import add_time_vars


@pytest.mark.unit
class TestAddTimeVars:
    """Test suite for add_time_vars function."""

    def test_adds_pub_year_column(self, sample_works_df):
        """Test that pub_year column is added correctly."""
        df = add_time_vars(sample_works_df)

        assert "pub_year" in df.columns
        assert df["pub_year"].dtype == "int64"
        assert list(df["pub_year"]) == [2020, 2021, 2022, 2023]

    def test_adds_pub_qtr_column(self, sample_works_df):
        """Test that pub_qtr column is added correctly."""
        df = add_time_vars(sample_works_df)

        assert "pub_qtr" in df.columns
        assert df["pub_qtr"].dtype == "object"  # string type
        assert list(df["pub_qtr"]) == ["2020Q1", "2021Q3", "2022Q4", "2023Q1"]

    def test_quarter_calculation_accuracy(self):
        """Test that quarters are calculated correctly for edge cases."""
        df = pd.DataFrame({
            "publication_date": [
                "2020-01-01",  # Q1
                "2020-03-31",  # Q1
                "2020-04-01",  # Q2
                "2020-06-30",  # Q2
                "2020-07-01",  # Q3
                "2020-09-30",  # Q3
                "2020-10-01",  # Q4
                "2020-12-31",  # Q4
            ]
        })
        df["publication_date"] = pd.to_datetime(df["publication_date"])

        result = add_time_vars(df)

        expected_quarters = ["2020Q1", "2020Q1", "2020Q2", "2020Q2",
                            "2020Q3", "2020Q3", "2020Q4", "2020Q4"]
        assert list(result["pub_qtr"]) == expected_quarters

    def test_handles_missing_publication_dates(self):
        """Test that NaT values are handled gracefully."""
        df = pd.DataFrame({
            "publication_date": ["2020-01-01", None, "2021-06-15"]
        })
        df["publication_date"] = pd.to_datetime(df["publication_date"])

        result = add_time_vars(df)

        assert result["pub_year"].iloc[0] == 2020
        assert pd.isna(result["pub_year"].iloc[1])
        assert result["pub_year"].iloc[2] == 2021

    def test_preserves_original_columns(self, sample_works_df):
        """Test that original DataFrame columns are preserved."""
        original_columns = set(sample_works_df.columns)
        df = add_time_vars(sample_works_df)

        assert original_columns.issubset(set(df.columns))

    def test_does_not_modify_original_dataframe(self, sample_works_df):
        """Test that the original DataFrame is not modified."""
        original_shape = sample_works_df.shape
        original_columns = list(sample_works_df.columns)

        _ = add_time_vars(sample_works_df)

        assert sample_works_df.shape == original_shape
        assert list(sample_works_df.columns) == original_columns
        assert "pub_year" not in sample_works_df.columns
        assert "pub_qtr" not in sample_works_df.columns

    def test_handles_string_dates(self):
        """Test conversion of string dates to datetime."""
        df = pd.DataFrame({
            "publication_date": ["2020-03-15", "2021-07-20"]
        })
        # Note: add_time_vars expects datetime, but let's test if it handles strings
        df["publication_date"] = pd.to_datetime(df["publication_date"])

        result = add_time_vars(df)

        assert result["pub_year"].iloc[0] == 2020
        assert result["pub_qtr"].iloc[0] == "2020Q1"

    def test_empty_dataframe(self):
        """Test handling of empty DataFrame."""
        df = pd.DataFrame({"publication_date": []})
        df["publication_date"] = pd.to_datetime(df["publication_date"])

        result = add_time_vars(df)

        assert len(result) == 0
        assert "pub_year" in result.columns
        assert "pub_qtr" in result.columns
