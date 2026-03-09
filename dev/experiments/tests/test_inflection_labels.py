import numpy as np

from scripts.label_inflection_points import derivative_detection, fit_logistic


def test_fit_logistic_detects_known_inflection():
    x = np.arange(0, 40)
    y = 120 / (1 + np.exp(-0.35 * (x - 16)))
    noise = np.random.default_rng(123).normal(scale=0.5, size=len(y))
    cumulative = (y + noise).clip(min=0)

    result = fit_logistic(cumulative, min_points=12, r2_threshold=0.6)
    assert result is not None
    assert abs(result.idx - 16) <= 2


def test_derivative_detection_flags_acceleration():
    # slow growth then sharp acceleration
    slow = np.linspace(0, 10, 20)
    fast = np.linspace(10, 60, 20)
    cumulative = np.concatenate([slow, fast])

    result = derivative_detection(cumulative, window=3, threshold_k=1.0)
    assert result is not None
    assert 15 <= result.idx <= 25
