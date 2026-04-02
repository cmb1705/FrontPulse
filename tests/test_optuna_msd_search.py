from __future__ import annotations

import importlib
from pathlib import Path

import pandas as pd

optuna_search = importlib.import_module("scripts.optuna_msd_search")


def _write_csv(path: Path, rows: list[dict[str, object]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_load_data_accepts_is_onset_labels(tmp_path: Path) -> None:
    labels_path = _write_csv(
        tmp_path / "onset_labels_msd.csv",
        [{"lineage_id": 1, "quarter": "2020Q1", "is_onset": 1}],
    )
    multisignal_path = _write_csv(
        tmp_path / "lineage_multisignal_features.csv",
        [{
            "lineage_id": 1,
            "quarter": "2020Q1",
            "new_works": 5,
            "novelty_rate": 0.2,
            "cross_domain_refs": 1,
            "within_lineage_refs": 1,
        }],
    )
    timeseries_path = _write_csv(
        tmp_path / "lineage_timeseries.csv",
        [{"lineage_id": 1, "quarter": "2020Q1", "new_works": 5}],
    )

    features_df, y = optuna_search.load_data(
        str(labels_path),
        str(multisignal_path),
        str(timeseries_path),
        str(tmp_path / "missing_semantic_velocity.csv"),
    )

    assert features_df.loc[0, "is_inflection_onset"] == 1
    assert y.tolist() == [1]
