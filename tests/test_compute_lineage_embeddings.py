from __future__ import annotations

import importlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

cle = importlib.import_module("scripts.compute_lineage_embeddings")


class _FakeExtractor:
    def __init__(self) -> None:
        self.paper_texts = {
            "W1": "crispr genome editing",
            "W2": "gene editing delivery",
            "W3": "cas9 screening",
        }
        self.doi_texts = {
            "10.1000/front-a": "crispr anchor front",
        }

    def get_texts_batch(self, work_ids: list[str], include_title: bool = True) -> dict[str, str]:
        return {work_id: self.paper_texts[work_id] for work_id in work_ids if work_id in self.paper_texts}

    def get_text_by_doi(self, doi: str, include_title: bool = True) -> str | None:
        return self.doi_texts.get(doi)

    def close(self) -> None:
        return None


class _FakeEmbedder:
    DEFAULT_MODEL = "fake/specter2"

    def __init__(
        self,
        raw_dir: Path | None = None,
        model_name: str = DEFAULT_MODEL,
        device: str | None = None,
        extractor: object | None = None,
    ) -> None:
        self.model_name = model_name
        self.device = device or "cpu"
        self.auto_batch_size = 4
        self.extractor = extractor or _FakeExtractor()
        self.model = type(
            "FakeModel",
            (),
            {"config": type("FakeConfig", (), {"_commit_hash": "fake-rev-1", "_name_or_path": model_name})()},
        )()
        self.tokenizer = type("FakeTokenizer", (), {"init_kwargs": {"revision": "fake-rev-1"}})()

    def embed_texts_batch(
        self,
        texts: list[str],
        batch_size: int = 0,
        max_length: int = 512,
    ) -> np.ndarray:
        rows = []
        for text in texts:
            width = float(len(text.split()))
            rows.append(np.array([width, width + 1.0, width + 2.0], dtype=float))
        return np.vstack(rows)


def _write_embedding_inputs(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    lineage_metrics_path = tmp_path / "lineage_metrics.csv"
    pd.DataFrame(
        {
            "lineage_id": [1, 2],
            "quarter": ["2020Q1", "2020Q1"],
        }
    ).to_csv(lineage_metrics_path, index=False)

    registry_path = tmp_path / "lineage_registry.json"
    registry_path.write_text("{}", encoding="utf-8")

    front_config_path = tmp_path / "front_aliases.yaml"
    front_config_path.write_text(
        yaml.safe_dump(
            {
                "front_a": {
                    "anchor_dois": ["10.1000/front-a"],
                }
            }
        ),
        encoding="utf-8",
    )

    output_root = tmp_path / "out"
    output_root.mkdir()

    return lineage_metrics_path, registry_path, front_config_path, output_root


def test_run_embeddings_writes_model_metadata(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    lineage_metrics_path, registry_path, front_config_path, output_root = _write_embedding_inputs(tmp_path)
    output_path = output_root / "02_lineage_tracking" / "lineage_embeddings.npz"
    partitions_dir = tmp_path / "partitions"
    partitions_dir.mkdir()

    monkeypatch.setattr(cle, "LineageEmbedder", _FakeEmbedder)
    monkeypatch.setattr(
        cle,
        "load_lineage_papers_fast",
        lambda lineage_id, lineage_registry, partitions_dir: {
            1: ["W1", "W2"],
            2: ["W3"],
        }[int(lineage_id)],
    )

    cle.run_embeddings(
        min_quarters=1,
        device="cpu",
        output_path=output_path,
        lineage_metrics_path=lineage_metrics_path,
        front_config_path=front_config_path,
        partitions_dir=partitions_dir,
        output_root=output_root,
        registry_path=registry_path,
        raw_dir=tmp_path / "raw",
        validate=False,
        model_name="fake/specter2",
    )

    metadata_path = output_path.with_suffix(".json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    assert metadata["model"] == "fake/specter2"
    assert metadata["model_version"] == "fake-rev-1"
    assert metadata["embedding_dim"] == 3
    assert metadata["summary"]["embedding_dim"] == 3
    assert metadata["summary"]["n_lineages"] == 2
    assert metadata["lineages"][0]["lineage_id"] == 1
    assert metadata["generated_at"].endswith("Z")
    assert (output_root / "03_milestone_mapping" / "lineage_front_similarity.csv").exists()


def test_run_embeddings_recomputes_if_cached_metadata_lacks_model(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    lineage_metrics_path, registry_path, front_config_path, output_root = _write_embedding_inputs(tmp_path)
    output_path = output_root / "02_lineage_tracking" / "lineage_embeddings.npz"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        embeddings=np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        lineage_ids=np.array([1, 2]),
    )
    output_path.with_suffix(".json").write_text(
        json.dumps(
            {
                "lineages": [
                    {"lineage_id": 1, "n_papers": 2, "n_with_text": 2, "coverage": 1.0},
                    {"lineage_id": 2, "n_papers": 1, "n_with_text": 1, "coverage": 1.0},
                ],
                "summary": {
                    "n_lineages": 2,
                    "embedding_dim": 3,
                    "avg_papers_per_lineage": 1.5,
                    "avg_coverage": 1.0,
                },
            }
        ),
        encoding="utf-8",
    )
    partitions_dir = tmp_path / "partitions"
    partitions_dir.mkdir()

    calls = {"count": 0}

    def _load_lineage_papers(lineage_id: int, lineage_registry: dict, partitions_dir: Path) -> list[str]:
        calls["count"] += 1
        return {
            1: ["W1", "W2"],
            2: ["W3"],
        }[int(lineage_id)]

    monkeypatch.setattr(cle, "LineageEmbedder", _FakeEmbedder)
    monkeypatch.setattr(cle, "load_lineage_papers_fast", _load_lineage_papers)

    cle.run_embeddings(
        min_quarters=1,
        device="cpu",
        output_path=output_path,
        lineage_metrics_path=lineage_metrics_path,
        front_config_path=front_config_path,
        partitions_dir=partitions_dir,
        output_root=output_root,
        registry_path=registry_path,
        raw_dir=tmp_path / "raw",
        validate=False,
        model_name="fake/specter2",
    )

    metadata = json.loads(output_path.with_suffix(".json").read_text(encoding="utf-8"))

    assert calls["count"] > 0, "legacy metadata should force recomputation instead of cache reuse"
    assert metadata["model"] == "fake/specter2"
    assert metadata["model_version"] == "fake-rev-1"
