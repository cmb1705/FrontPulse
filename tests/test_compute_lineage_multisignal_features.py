from __future__ import annotations

import importlib
from argparse import Namespace
from pathlib import Path

import pandas as pd

from src.domain_registry import apply_domain_path_defaults, get_domain

multisignal = importlib.import_module("scripts.compute_lineage_multisignal_features")


def _write_metric(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(path, index=False)


def test_load_global_metrics_uses_plain_metric_names(tmp_path: Path) -> None:
    metrics_dir = tmp_path / "metrics"

    for idx, (_metric_name, filename) in enumerate(multisignal.GLOBAL_METRIC_FILENAMES.items(), start=1):
        _write_metric(
            metrics_dir / "global" / filename,
            [
                {"quarter": "2020Q1", "value": float(idx)},
                {"quarter": "2020Q2", "value": float(idx + 10)},
            ],
        )

    loaded = multisignal.load_global_metrics(metrics_dir)

    assert set(loaded["2020Q1"]) == set(multisignal.GLOBAL_METRIC_FILENAMES)
    assert loaded["2020Q1"]["author_influx"] == 1.0
    assert loaded["2020Q2"]["cross_cluster_bridging"] == 15.0


def test_domain_defaults_cover_auxiliary_feature_inputs(tmp_path: Path) -> None:
    paths = get_domain("psc").resolve_data_paths(tmp_path)
    args = Namespace(
        registry=None,
        timeseries=None,
        raw_dir=None,
        partitions_dir=None,
        reference_cache=None,
        out=None,
        metrics_dir=None,
        field_metrics=None,
        milestones=None,
        onset_labels=None,
        maturation_labels=None,
        convergence_features=None,
    )

    apply_domain_path_defaults(args, paths, multisignal.MULTISIGNAL_DOMAIN_DEFAULTS)

    assert Path(args.onset_labels) == paths.lineage_tracking / "onset_labels.csv"
    assert Path(args.maturation_labels) == paths.lineage_tracking / "maturation_labels.csv"
    assert Path(args.convergence_features) == paths.lineage_tracking / "convergence_features.csv"
