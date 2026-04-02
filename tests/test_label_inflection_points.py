from __future__ import annotations

import pandas as pd

from scripts import label_inflection_points as labels


def test_prepare_timeseries_densifies_missing_quarters(tmp_path) -> None:
    source = tmp_path / "lineage_timeseries.csv"
    pd.DataFrame(
        [
            {"lineage_id": 1, "quarter": "2020Q1", "new_works": 2},
            {"lineage_id": 1, "quarter": "2020Q3", "new_works": 5},
            {"lineage_id": 2, "quarter": "2021Q2", "new_works": 3},
        ]
    ).to_csv(source, index=False)

    dense = labels.prepare_timeseries(source)

    lineage_one = dense[dense["lineage_id"] == 1].reset_index(drop=True)
    assert lineage_one["quarter"].tolist() == ["2020Q1", "2020Q2", "2020Q3"]
    assert lineage_one["new_works"].tolist() == [2.0, 0.0, 5.0]
    assert lineage_one["quarter_order"].tolist() == [0, 1, 2]

    lineage_two = dense[dense["lineage_id"] == 2].reset_index(drop=True)
    assert lineage_two["quarter"].tolist() == ["2021Q2"]
    assert lineage_two["new_works"].tolist() == [3.0]
    assert lineage_two["quarter_order"].tolist() == [0]


def test_prepare_timeseries_aggregates_duplicate_quarters(tmp_path) -> None:
    source = tmp_path / "lineage_timeseries.csv"
    pd.DataFrame(
        [
            {"lineage_id": 7, "quarter": "2022Q1", "new_works": 1},
            {"lineage_id": 7, "quarter": "2022Q1", "new_works": 2},
            {"lineage_id": 7, "quarter": "2022Q2", "new_works": 4},
        ]
    ).to_csv(source, index=False)

    dense = labels.prepare_timeseries(source)

    assert dense["quarter"].tolist() == ["2022Q1", "2022Q2"]
    assert dense["new_works"].tolist() == [3.0, 4.0]
    assert dense["quarter_order"].tolist() == [0, 1]
