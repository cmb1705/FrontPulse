"""Unit tests for src.bocpd."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.bocpd import BOCPDConfig, detect_changepoints, run_bocpd_on_fronts


# -- BOCPDConfig defaults ----------------------------------------------------


class TestBOCPDConfig:
    def test_default_values(self) -> None:
        cfg = BOCPDConfig()
        assert cfg.alpha0 == 1.0
        assert cfg.beta0 == 0.1
        assert cfg.hazard_rate == pytest.approx(0.02)
        assert cfg.max_run_length == 40
        assert cfg.threshold == 0.5

    def test_custom_values(self) -> None:
        cfg = BOCPDConfig(alpha0=2.0, beta0=0.5, hazard_rate=0.1, threshold=0.3)
        assert cfg.alpha0 == 2.0
        assert cfg.hazard_rate == 0.1
        assert cfg.threshold == 0.3


# -- detect_changepoints -----------------------------------------------------


class TestDetectChangepoints:
    def test_empty_input(self) -> None:
        result = detect_changepoints(np.array([]))
        assert len(result.changepoint_prob) == 0
        assert result.n_alerts == 0
        assert result.alert_indices == []

    def test_single_observation(self) -> None:
        result = detect_changepoints(np.array([5]))
        assert len(result.changepoint_prob) == 1
        assert 0.0 <= result.changepoint_prob[0] <= 1.0

    def test_constant_series_low_changepoint_prob(self) -> None:
        """A constant series should have low changepoint probability after warmup."""
        counts = np.full(20, 10)
        result = detect_changepoints(counts)
        # After warmup (first few quarters), changepoint prob should be low
        late_probs = result.changepoint_prob[5:]
        assert np.mean(late_probs) < 0.3, (
            f"Constant series should have low changepoint prob, got mean={np.mean(late_probs):.3f}"
        )

    def test_step_change_detected(self) -> None:
        """A sharp step change should produce elevated changepoint probability."""
        counts = np.concatenate([
            np.full(15, 5),    # Stable low regime
            np.full(15, 50),   # Sharp increase
        ])
        config = BOCPDConfig(hazard_rate=1 / 20, detection_window=3, threshold=0.3)
        result = detect_changepoints(counts, config)
        # After the step change, the cumulative short-run-length
        # probability should be higher than during the stable regime
        stable_mean = np.mean(result.changepoint_prob[5:14])
        post_change_mean = np.mean(result.changepoint_prob[16:22])
        assert post_change_mean > stable_mean, (
            f"Post-change CP prob ({post_change_mean:.3f}) should exceed "
            f"stable regime ({stable_mean:.3f})"
        )

    def test_output_shapes_match_input(self) -> None:
        counts = np.array([1, 2, 3, 4, 5])
        result = detect_changepoints(counts)
        assert result.changepoint_prob.shape == (5,)
        assert result.alert.shape == (5,)
        assert result.map_run_length.shape == (5,)

    def test_probabilities_in_valid_range(self) -> None:
        np.random.seed(42)
        counts = np.random.poisson(lam=10, size=30)
        result = detect_changepoints(counts)
        assert np.all(result.changepoint_prob >= -1e-10)
        assert np.all(result.changepoint_prob <= 1.0 + 1e-10)

    def test_alert_binary(self) -> None:
        counts = np.array([1, 2, 50, 3, 4])
        result = detect_changepoints(counts)
        assert set(np.unique(result.alert)).issubset({0, 1})

    def test_alert_indices_consistent(self) -> None:
        counts = np.array([1, 1, 1, 100, 1, 1])
        config = BOCPDConfig(threshold=0.01)  # Low threshold to trigger alerts
        result = detect_changepoints(counts, config)
        expected_indices = list(np.where(result.alert == 1)[0])
        assert result.alert_indices == expected_indices

    def test_n_alerts_matches_alert_array(self) -> None:
        counts = np.random.poisson(lam=5, size=20)
        result = detect_changepoints(counts)
        assert result.n_alerts == int(np.sum(result.alert))

    def test_high_hazard_rate_more_changepoints(self) -> None:
        """Higher hazard rate should produce more changepoint detections."""
        np.random.seed(123)
        counts = np.random.poisson(lam=10, size=30)
        low_hz = detect_changepoints(counts, BOCPDConfig(hazard_rate=0.01, threshold=0.3))
        high_hz = detect_changepoints(counts, BOCPDConfig(hazard_rate=0.2, threshold=0.3))
        # High hazard should have higher mean CP probability
        assert np.mean(high_hz.changepoint_prob) >= np.mean(low_hz.changepoint_prob)

    def test_map_run_length_non_negative(self) -> None:
        counts = np.array([5, 5, 5, 50, 50, 50])
        result = detect_changepoints(counts)
        assert np.all(result.map_run_length >= 0)

    def test_max_run_length_respected(self) -> None:
        """Run length should not exceed max_run_length."""
        counts = np.full(60, 5)
        config = BOCPDConfig(max_run_length=20)
        result = detect_changepoints(counts, config)
        assert np.all(result.map_run_length <= 20)


# -- run_bocpd_on_fronts -----------------------------------------------------


class TestRunBOCPDOnFronts:
    def test_basic_output_columns(self) -> None:
        df = pd.DataFrame({
            "front_id": ["A", "A", "A", "B", "B", "B"],
            "quarter": ["2020Q1", "2020Q2", "2020Q3"] * 2,
            "new_works": [5, 5, 5, 10, 10, 10],
        })
        result = run_bocpd_on_fronts(df)
        assert "front_id" in result.columns
        assert "quarter" in result.columns
        assert "bocpd_changepoint_prob" in result.columns
        assert "bocpd_alert" in result.columns
        assert "bocpd_map_run_length" in result.columns

    def test_output_length_matches_input(self) -> None:
        df = pd.DataFrame({
            "front_id": ["A"] * 5 + ["B"] * 3,
            "quarter": [f"2020Q{i}" for i in [1, 2, 3, 4]] + ["2021Q1"]
            + [f"2020Q{i}" for i in [1, 2, 3]],
            "new_works": [5, 5, 5, 5, 5, 10, 10, 10],
        })
        result = run_bocpd_on_fronts(df)
        assert len(result) == len(df)

    def test_probabilities_valid(self) -> None:
        df = pd.DataFrame({
            "front_id": ["X"] * 10,
            "quarter": [f"20{20 + i // 4}Q{i % 4 + 1}" for i in range(10)],
            "new_works": list(range(1, 11)),
        })
        result = run_bocpd_on_fronts(df)
        assert (result["bocpd_changepoint_prob"] >= 0.0).all()
        assert (result["bocpd_changepoint_prob"] <= 1.0).all()
