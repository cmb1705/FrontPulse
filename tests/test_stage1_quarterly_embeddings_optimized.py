from __future__ import annotations

import importlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

stage1 = importlib.import_module("scripts.stage1_quarterly_embeddings_optimized")


def _write_registry(path: Path, registry: dict[str, dict[str, int]]) -> Path:
    path.write_text(json.dumps(registry), encoding="utf-8")
    return path


def test_resolve_tight_mapping_path_prefers_active_experiments_tree(tmp_path: Path) -> None:
    output_dir = tmp_path / "psc" / "out" / "experiments" / "stage1_quarterly_embeddings"
    tight_mapping_path = output_dir.parent / "stage0_tight_mapping" / "milestone_lineage_mapping_tight.csv"
    tight_mapping_path.parent.mkdir(parents=True, exist_ok=True)
    tight_mapping_path.write_text("lineage_id\n1\n", encoding="utf-8")

    resolved = stage1.resolve_tight_mapping_path(output_dir)

    assert resolved == tight_mapping_path


def test_build_abstract_cache_path_uses_requested_cache_root(tmp_path: Path) -> None:
    raw_dir = tmp_path / "psc" / "ingest" / "raw"
    cache_dir = tmp_path / "psc" / "out" / "cache_lineage"

    cache_path = stage1.build_abstract_cache_path(raw_dir, cache_dir)

    assert cache_path.parent == cache_dir
    assert cache_path.name.startswith("abstract_index_raw_")
    assert cache_path.suffix == ".pkl"


def test_test_coverage_skips_stale_tight_mapping(tmp_path: Path) -> None:
    registry_path = _write_registry(
        tmp_path / "lineage_registry.json",
        {"2020Q1": {"1": 1, "2": 2}},
    )
    tight_mapping_path = tmp_path / "stage0_tight_mapping.csv"
    pd.DataFrame({"lineage_id": [1, 3]}).to_csv(tight_mapping_path, index=False)

    summary = stage1.test_coverage(
        embeddings={(1, "2020Q1"): np.array([1.0, 2.0])},
        registry_path=registry_path,
        tight_mapping_path=tight_mapping_path,
    )

    coverage_audit = summary["coverage_audit"]
    assert coverage_audit["status"] == "skipped_stale_tight_mapping"
    assert coverage_audit["milestone_lineages"] == 2
    assert coverage_audit["registry_lineages"] == 2
    assert coverage_audit["stale_mapping_lineages"] == 1
    assert coverage_audit["stale_mapping_lineage_ids_preview"] == [3]


def test_save_embeddings_and_velocity_writes_sibling_legacy_copy(tmp_path: Path) -> None:
    output_dir = tmp_path / "psc" / "out" / "experiments" / "stage1_quarterly_embeddings"
    embeddings = {
        (1, "2020Q1"): np.array([1.0, 2.0], dtype=float),
        (1, "2020Q2"): np.array([3.0, 4.0], dtype=float),
    }
    velocity_df = pd.DataFrame(
        [
            {"lineage_id": 1, "quarter": "2020Q1", "semantic_velocity": 0.0},
            {"lineage_id": 1, "quarter": "2020Q2", "semantic_velocity": 0.25},
        ]
    )

    stage1.save_embeddings_and_velocity(
        embeddings,
        velocity_df,
        output_dir,
        summary_extras={"coverage_audit": {"status": "skipped_missing_tight_mapping"}},
        legacy_output_dirs=stage1.resolve_legacy_output_dirs(output_dir),
    )

    legacy_dir = output_dir.parent / "phase1_quarterly_embeddings"
    assert (output_dir / "quarterly_embeddings.npz").exists()
    assert (output_dir / "semantic_velocity.csv").exists()
    assert (legacy_dir / "quarterly_embeddings.npz").exists()
    assert (legacy_dir / "semantic_velocity.csv").exists()

    summary = json.loads((output_dir / "quarterly_embeddings_summary.json").read_text(encoding="utf-8"))
    assert summary["n_embeddings"] == 2
    assert summary["coverage_audit"]["status"] == "skipped_missing_tight_mapping"
