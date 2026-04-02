from __future__ import annotations

import importlib
from pathlib import Path

import pandas as pd

stage5 = importlib.import_module("scripts.stage5_ensemble_mapping")


def test_stage5_validation_honors_output_root(tmp_path: Path, monkeypatch) -> None:
    mappings_df = pd.DataFrame(
        {
            "lineage_id": [1, 2],
            "primary_front": ["cas9_methodology", "diagnostics"],
            "confidence": ["high", "medium"],
            "review_needed": [False, True],
            "alternative_fronts": ["", ""],
        }
    )
    evidence_dict = {1: {"Stage2": {}}, 2: {"Stage2": {}}}
    validation_dir = tmp_path / "06_validation" / "stage5"

    def _touch_png(_df: pd.DataFrame, output_path: Path) -> None:
        output_path.write_bytes(b"png")

    def _touch_report(_checks: dict, _df: pd.DataFrame, output_path: Path) -> None:
        output_path.write_text("ok\n", encoding="utf-8")

    monkeypatch.setattr(stage5, "_generate_Stage5_confidence_viz", _touch_png)
    monkeypatch.setattr(stage5, "_generate_Stage5_front_distribution", _touch_png)
    monkeypatch.setattr(stage5, "_generate_Stage5_report", _touch_report)

    checks = stage5.run_Stage5_validation(
        mappings_df,
        evidence_dict,
        validation_dir=validation_dir,
    )

    assert checks["total_lineages"] == 2
    assert validation_dir.joinpath("Stage5_validation_results.json").exists()
    assert validation_dir.joinpath("Stage5_confidence_distribution.png").exists()
    assert validation_dir.joinpath("Stage5_front_distribution.png").exists()
    assert validation_dir.joinpath("Stage5_validation_report.md").exists()
