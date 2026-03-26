from __future__ import annotations

import pandas as pd
from multi_signal_detector import select_features


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
