from __future__ import annotations

import json
from pathlib import Path

from scripts.generate_ablation_diagnostics import run_ablation_diagnostics


def _write_fold_diagnostics(experiment_dir: Path, name: str, fold_pr_auc: list[float]) -> None:
    experiment_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "name": name,
        "model_type": "lightgbm",
        "n_features": 51 if "51" in name else 65,
        "n_samples": 120,
        "n_positives": 18,
        "per_fold": [
            {
                "fold": idx,
                "pr_auc": value,
                "roc_auc": 0.85 + idx * 0.01,
                "precision": 0.20 + idx * 0.01,
                "recall": 0.40 + idx * 0.01,
                "f1": 0.28 + idx * 0.01,
                "mcc": 0.18 + idx * 0.01,
                "f2": 0.32 + idx * 0.01,
            }
            for idx, value in enumerate(fold_pr_auc)
        ],
        "pr_curve": {
            "precision": [1.0, 0.8, 0.5],
            "recall": [0.0, 0.6, 1.0],
            "thresholds": [0.35, 0.70],
        },
        "calibration": {
            "prob_true": [0.05, 0.25, 0.75],
            "prob_pred": [0.10, 0.30, 0.70],
            "n_bins": 3,
        },
        "fold_pr_auc": fold_pr_auc,
        "fold_roc_auc": [0.85, 0.83, 0.84, 0.86, 0.87],
    }
    with open(experiment_dir / "fold_diagnostics.json", "w", encoding="utf-8") as handle:
        json.dump(payload, handle)


def test_run_ablation_diagnostics_writes_plots_and_stats(tmp_path: Path) -> None:
    left_dir = tmp_path / "ablation_51feat_198labels"
    right_dir = tmp_path / "ablation_65feat_198labels"
    _write_fold_diagnostics(
        left_dir,
        "ablation_51feat_198labels",
        [0.23309392442575647, 0.14045992268196195, 0.14933539723237438, 0.14849263525582782, 0.22376567420743354],
    )
    _write_fold_diagnostics(
        right_dir,
        "ablation_65feat_198labels",
        [0.24512602232856362, 0.11833543025528645, 0.14804015196235706, 0.17998516736956943, 0.16625079292481046],
    )

    result = run_ablation_diagnostics([left_dir, right_dir])

    generated_paths = [Path(path) for path in result["generated_paths"]]
    assert generated_paths

    for experiment_dir in (left_dir, right_dir):
        pr_curve = experiment_dir / "pr_curve.png"
        calibration_curve = experiment_dir / "calibration_curve.png"
        diagnostic_summary = experiment_dir / "diagnostic_summary.json"
        comparison = experiment_dir / "ablation_wilcoxon.json"

        for path in (pr_curve, calibration_curve, diagnostic_summary, comparison):
            assert path.exists()
            assert path.stat().st_size > 0

        assert pr_curve.read_bytes().startswith(b"\x89PNG")
        assert calibration_curve.read_bytes().startswith(b"\x89PNG")

        summary = json.loads(diagnostic_summary.read_text(encoding="utf-8"))
        assert summary["pr_curve_source"] == "out_of_fold_aggregate"
        assert summary["fold_metric_summary"]["fold_pr_auc"]["n"] == 5

        wilcoxon_summary = json.loads(comparison.read_text(encoding="utf-8"))
        assert wilcoxon_summary["metric"] == "fold_pr_auc"
        assert wilcoxon_summary["test"] == "wilcoxon_signed_rank"
        assert wilcoxon_summary["p_value"] == 0.8125
