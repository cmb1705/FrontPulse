"""Tests for field metrics loading and merge helpers."""
from pathlib import Path
import sys

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.compute_lineage_multisignal_features import (  # noqa: E402
    load_field_metrics,
    merge_field_metrics,
)


@pytest.mark.unit
def test_load_field_metrics_reads_csv_and_casts_quarter(tmp_path):
    """Field metrics loader should read CSV/parquet files and normalize quarter column."""
    path = tmp_path / "field_metrics.csv"
    df = pd.DataFrame({
        "quarter": ["2024Q1", "2024Q2"],
        "total_new_works": [100, 120],
        "cumulative_new_works": [100, 220],
    })
    df.to_csv(path, index=False)

    result = load_field_metrics(path)

    assert list(result["quarter"]) == ["2024Q1", "2024Q2"]
    assert result["quarter"].dtype == object  # normalized to string
    assert "total_new_works" in result.columns


@pytest.mark.unit
def test_load_field_metrics_missing_file_returns_empty(tmp_path):
    """Missing field metrics files should return an empty DataFrame."""
    missing_path = tmp_path / "does_not_exist.parquet"
    result = load_field_metrics(missing_path)
    assert result.empty


@pytest.mark.unit
def test_merge_field_metrics_adds_relative_columns():
    """Merging should append contrastive ratios/deltas without NaNs or infs."""
    features = pd.DataFrame({
        "lineage_id": [1, 2],
        "quarter": ["2024Q1", "2024Q2"],
        "new_works": [10.0, 5.0],
        "cumulative_works": [50.0, 60.0],
        "growth_rate_diff": [4.0, 2.0],
        "growth_acceleration": [1.5, -0.5],
    })
    field_metrics = pd.DataFrame({
        "quarter": ["2024Q1", "2024Q2"],
        "total_new_works": [20.0, 0.0],  # second row zero to test safe divide
        "cumulative_new_works": [200.0, 0.0],
        "total_new_works_diff": [1.0, 0.0],
        "cumulative_acceleration": [0.25, 0.0],
        "new_works_p75": [8.0, 0.0],
    })

    merged = merge_field_metrics(features.copy(), field_metrics)

    assert "relative_new_works" in merged
    assert "relative_cumulative_works" in merged
    assert "growth_vs_field" in merged
    assert "acceleration_vs_field" in merged
    assert "new_works_over_p75" in merged

    assert merged.loc[0, "relative_new_works"] == pytest.approx(0.5)
    assert merged.loc[0, "relative_cumulative_works"] == pytest.approx(0.25)
    assert merged.loc[0, "growth_vs_field"] == pytest.approx(3.0)
    assert merged.loc[0, "acceleration_vs_field"] == pytest.approx(1.25)
    assert merged.loc[0, "new_works_over_p75"] == pytest.approx(1.25)

    # Zero denominators should yield zeros (no inf/nan)
    assert merged.loc[1, "relative_new_works"] == 0
    assert merged.loc[1, "relative_cumulative_works"] == 0
    assert merged.loc[1, "new_works_over_p75"] == 0
