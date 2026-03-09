"""Tests for src/validate.py - Schema validation and type coercion."""
from __future__ import annotations

import pandas as pd
import pytest

from src.validate import enforce_schema


@pytest.mark.unit
class TestEnforceSchema:
    """Test suite for enforce_schema function."""

    def test_enforces_data_types(self, sample_schema_yaml):
        """Test that data types are enforced according to schema."""
        df = pd.DataFrame({
            "id": ["W001", "W002"],
            "title": ["Title 1", "Title 2"],
            "publication_date": ["2020-01-01", "2021-06-15"],
            "cited_by_count": ["10", "25"]  # String instead of int
        })

        result = enforce_schema(df, sample_schema_yaml)

        assert result["id"].dtype == "object"  # str
        assert result["title"].dtype == "object"  # str
        assert pd.api.types.is_datetime64_any_dtype(result["publication_date"])
        assert result["cited_by_count"].dtype == "int64"

    def test_enforces_non_null_constraints(self, sample_schema_yaml):
        """Test that non-nullable columns raise ValueError when null."""
        df = pd.DataFrame({
            "id": [None, "W002"],  # id is non-nullable
            "title": ["Title 1", "Title 2"],
            "publication_date": ["2020-01-01", "2021-06-15"],
            "cited_by_count": [10, 25]
        })

        with pytest.raises(ValueError, match="non-null constraint"):
            enforce_schema(df, sample_schema_yaml)

    def test_allows_null_in_nullable_columns(self, sample_schema_yaml):
        """Test that nullable columns accept null values."""
        df = pd.DataFrame({
            "id": ["W001", "W002"],
            "title": ["Title 1", "Title 2"],
            "publication_date": ["2020-01-01", None],  # nullable
            "cited_by_count": [10, None]  # nullable
        })

        result = enforce_schema(df, sample_schema_yaml)

        assert result["publication_date"].iloc[0] == pd.Timestamp("2020-01-01")
        assert pd.isna(result["publication_date"].iloc[1])
        assert result["cited_by_count"].iloc[0] == 10
        assert pd.isna(result["cited_by_count"].iloc[1])

    def test_handles_extra_columns(self, sample_schema_yaml):
        """Test that extra columns not in schema are preserved."""
        df = pd.DataFrame({
            "id": ["W001", "W002"],
            "title": ["Title 1", "Title 2"],
            "publication_date": ["2020-01-01", "2021-06-15"],
            "cited_by_count": [10, 25],
            "extra_column": ["data1", "data2"]  # Not in schema
        })

        result = enforce_schema(df, sample_schema_yaml)

        assert "extra_column" in result.columns
        assert list(result["extra_column"]) == ["data1", "data2"]

    def test_handles_missing_optional_columns(self, sample_schema_yaml):
        """Test that missing nullable columns are handled gracefully."""
        df = pd.DataFrame({
            "id": ["W001", "W002"],
            "title": ["Title 1", "Title 2"],
            # publication_date missing (nullable in schema)
            "cited_by_count": [10, 25]
        })

        # Should not raise error for missing nullable columns
        result = enforce_schema(df, sample_schema_yaml)

        assert "id" in result.columns
        assert "title" in result.columns

    def test_datetime_conversion(self, sample_schema_yaml):
        """Test that datetime conversion works for various formats."""
        df = pd.DataFrame({
            "id": ["W001", "W002", "W003"],
            "title": ["T1", "T2", "T3"],
            "publication_date": [
                "2020-01-15",
                "2021-06-30T10:30:00",
                "2022-12-01"
            ],
            "cited_by_count": [10, 20, 30]
        })

        result = enforce_schema(df, sample_schema_yaml)

        assert pd.api.types.is_datetime64_any_dtype(result["publication_date"])
        assert result["publication_date"].iloc[0] == pd.Timestamp("2020-01-15")
        assert result["publication_date"].iloc[1].year == 2021
        assert result["publication_date"].iloc[1].month == 6

    def test_empty_dataframe(self, sample_schema_yaml):
        """Test handling of empty DataFrame."""
        df = pd.DataFrame({
            "id": [],
            "title": [],
            "publication_date": [],
            "cited_by_count": []
        })

        result = enforce_schema(df, sample_schema_yaml)

        assert len(result) == 0
        assert "id" in result.columns
        assert "title" in result.columns

    def test_preserves_dataframe_index(self, sample_schema_yaml):
        """Test that DataFrame index is preserved."""
        df = pd.DataFrame({
            "id": ["W001", "W002"],
            "title": ["Title 1", "Title 2"],
            "publication_date": ["2020-01-01", "2021-06-15"],
            "cited_by_count": [10, 25]
        }, index=[5, 10])

        result = enforce_schema(df, sample_schema_yaml)

        assert list(result.index) == [5, 10]
