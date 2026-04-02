from __future__ import annotations

import importlib
from pathlib import Path

import pandas as pd

agg = importlib.import_module("scripts.aggregate_lineages_to_fronts")


def test_run_aggregation_prefers_selected_mappings(tmp_path: Path) -> None:
    selected_path = tmp_path / "lineage_front_mappings_selected.csv"
    full_path = tmp_path / "lineage_front_mappings.csv"
    timeseries_path = tmp_path / "lineage_timeseries.csv"
    output_dir = tmp_path / "front_aggregation"

    pd.DataFrame(
        {
            "lineage_id": [1],
            "primary_front": ["selected_front"],
            "confidence": ["High"],
        }
    ).to_csv(selected_path, index=False)
    pd.DataFrame(
        {
            "lineage_id": [1],
            "primary_front": ["full_front"],
            "confidence": ["Low"],
        }
    ).to_csv(full_path, index=False)
    pd.DataFrame(
        {
            "lineage_id": [1, 1],
            "quarter": ["2020Q1", "2020Q2"],
            "new_works": [2, 3],
        }
    ).to_csv(timeseries_path, index=False)

    outputs = agg.run_aggregation(
        mappings_path_selected=selected_path,
        mappings_path_full=full_path,
        timeseries_path=timeseries_path,
        output_dir=output_dir,
    )

    delta = pd.read_csv(outputs["delta"], index_col=0)
    delta_long = pd.read_csv(outputs["delta_long"])

    assert "selected_front" in delta.columns
    assert "full_front" not in delta.columns
    assert output_dir.joinpath("front_timeseries_cumulative.csv").exists()
    assert int(delta_long["count"].sum()) == 5
