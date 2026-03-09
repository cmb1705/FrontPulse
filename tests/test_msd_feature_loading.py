import pandas as pd
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "scripts"))

from multi_signal_detector import select_features  # noqa: E402


def test_select_features_accepts_curated_list():
    df = pd.DataFrame(
        {
            "semantic_velocity": [0.15, 0.25],
            "growth_rate": [0.01, 0.02],
            "dummy": [1, 2],
        }
    )
    required = ["semantic_velocity", "growth_rate"]
    X, cols = select_features(df, required_features=required)
    assert cols == required
    assert list(X.columns) == required
