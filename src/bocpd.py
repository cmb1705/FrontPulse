"""Bayesian Online Changepoint Detection for front-level count series.

Implements the Adams-MacKay (2007) BOCPD algorithm with Poisson-Gamma
conjugate priors, designed for quarterly new_works count data from the
front-level series contract.

Usage:
    from src.bocpd import detect_changepoints, BOCPDConfig, run_bocpd_on_fronts

    result = detect_changepoints(counts, config=BOCPDConfig())
    # result.changepoint_prob is a per-quarter array of P(changepoint)

References:
    Adams, R.P. and MacKay, D.J.C. (2007). Bayesian Online Changepoint
    Detection. arXiv:0710.3742.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.special import gammaln


@dataclass(frozen=True)
class BOCPDConfig:
    """Configuration for BOCPD detector.

    Attributes:
        alpha0: Gamma shape prior for Poisson rate. Higher values express
            stronger prior belief about the count rate.
        beta0: Gamma rate prior. alpha0/beta0 is the prior mean count rate.
        hazard_rate: Constant hazard function value. P(changepoint) per
            quarter. 1/hazard_rate is the expected run length between
            changepoints.
        max_run_length: Maximum run length to track. Truncates the
            run-length distribution for numerical stability.
        detection_window: Number of recent quarters to aggregate for
            changepoint probability. P(r_t <= detection_window) gives
            the probability a changepoint occurred recently.
        threshold: Changepoint probability threshold for binary alert.
    """

    alpha0: float = 1.0
    beta0: float = 0.1
    hazard_rate: float = 1 / 50
    max_run_length: int = 40
    detection_window: int = 3
    threshold: float = 0.5


@dataclass
class BOCPDResult:
    """BOCPD detection result for a single series.

    Attributes:
        changepoint_prob: Per-quarter recent-changepoint probability.
            Computed as P(r_t <= detection_window), the probability that
            a changepoint occurred within the last few quarters.
        alert: Binary alert array (1 if changepoint_prob >= threshold).
        map_run_length: MAP run length per quarter.
        n_alerts: Total number of alerts.
        alert_indices: Indices where alert=1.
    """

    changepoint_prob: np.ndarray
    alert: np.ndarray
    map_run_length: np.ndarray
    n_alerts: int
    alert_indices: list[int]


def _log_neg_binomial_pmf(x: int, alpha: np.ndarray, beta: np.ndarray) -> np.ndarray:
    """Log probability of x under negative binomial (Poisson-Gamma predictive).

    When the Poisson rate lambda ~ Gamma(alpha, beta), the predictive
    distribution for a new observation x is NegBin(alpha, beta/(beta+1)).

    Args:
        x: Observed count (non-negative integer).
        alpha: Gamma shape parameters (array, one per run length).
        beta: Gamma rate parameters (array, one per run length).

    Returns:
        Log-probability array, same shape as alpha.
    """
    p = beta / (beta + 1.0)
    return (
        gammaln(alpha + x)
        - gammaln(alpha)
        - gammaln(x + 1)
        + alpha * np.log(p)
        + x * np.log(1.0 - p)
    )


def detect_changepoints(
    counts: np.ndarray,
    config: BOCPDConfig | None = None,
) -> BOCPDResult:
    """Run BOCPD on a 1-d count series.

    Implements the Adams-MacKay (2007) algorithm with Poisson-Gamma
    conjugate priors and constant hazard function.

    Args:
        counts: 1-d array of non-negative integer counts (one per quarter).
        config: BOCPD configuration. Uses defaults if None.

    Returns:
        BOCPDResult with per-quarter changepoint probabilities and alerts.
    """
    if config is None:
        config = BOCPDConfig()

    T = len(counts)
    if T == 0:
        return BOCPDResult(
            changepoint_prob=np.array([]),
            alert=np.array([], dtype=int),
            map_run_length=np.array([], dtype=int),
            n_alerts=0,
            alert_indices=[],
        )

    max_rl = config.max_run_length
    H = config.hazard_rate

    # Sufficient statistics for each run length hypothesis
    # alpha[r] and beta[r] are the Gamma parameters for run length r
    alpha = np.full(max_rl + 1, config.alpha0)
    beta = np.full(max_rl + 1, config.beta0)

    # Run-length distribution (probability of each run length)
    # Start with r=0 having probability 1
    rl_dist = np.zeros(max_rl + 1)
    rl_dist[0] = 1.0

    changepoint_prob = np.zeros(T)
    map_run_length = np.zeros(T, dtype=int)

    for t in range(T):
        x_t = int(counts[t])

        # Predictive probabilities for each run length
        log_pred = _log_neg_binomial_pmf(x_t, alpha[: max_rl + 1], beta[: max_rl + 1])
        pred = np.exp(log_pred - np.max(log_pred))  # Numerical stability

        # Growth probabilities: P(r_t = r_{t-1}+1)
        growth = rl_dist * pred * (1.0 - H)

        # Changepoint probability: P(r_t = 0)
        cp = np.sum(rl_dist * pred) * H

        # Shift run-length distribution (grow by 1)
        new_rl = np.zeros(max_rl + 1)
        new_rl[0] = cp
        new_rl[1 : max_rl + 1] = growth[: max_rl]

        # Normalize
        total = np.sum(new_rl)
        if total > 0:
            new_rl /= total

        rl_dist = new_rl

        # Recent-changepoint probability: P(run_length <= detection_window)
        # This varies with data (unlike P(r=0) which equals hazard rate
        # for constant hazard). It captures mass shifting to short runs
        # after a regime change.
        dw = min(config.detection_window, max_rl)
        changepoint_prob[t] = float(np.sum(rl_dist[: dw + 1]))
        map_run_length[t] = int(np.argmax(rl_dist))

        # Update sufficient statistics
        new_alpha = np.zeros(max_rl + 1)
        new_beta = np.zeros(max_rl + 1)
        # For r=0 (new segment): reset to prior
        new_alpha[0] = config.alpha0
        new_beta[0] = config.beta0
        # For r>0: update from previous
        new_alpha[1 : max_rl + 1] = alpha[: max_rl] + x_t
        new_beta[1 : max_rl + 1] = beta[: max_rl] + 1.0

        alpha = new_alpha
        beta = new_beta

    alert = (changepoint_prob >= config.threshold).astype(int)
    alert_indices = list(np.where(alert == 1)[0])

    return BOCPDResult(
        changepoint_prob=changepoint_prob,
        alert=alert,
        map_run_length=map_run_length,
        n_alerts=int(np.sum(alert)),
        alert_indices=alert_indices,
    )


def run_bocpd_on_fronts(
    series_df: pd.DataFrame,
    config: BOCPDConfig | None = None,
) -> pd.DataFrame:
    """Run BOCPD on all fronts in a front-level series DataFrame.

    Args:
        series_df: DataFrame with columns front_id, quarter, new_works.
            Must be sorted by (front_id, quarter).
        config: BOCPD configuration. Uses defaults if None.

    Returns:
        DataFrame with columns: front_id, quarter, bocpd_changepoint_prob,
        bocpd_alert, bocpd_map_run_length.
    """
    if config is None:
        config = BOCPDConfig()

    results: list[dict] = []
    for front_id, group in series_df.groupby("front_id"):
        group = group.sort_values("quarter")
        counts = group["new_works"].values.astype(float)
        quarters = group["quarter"].values

        result = detect_changepoints(counts, config)

        for i, q in enumerate(quarters):
            results.append({
                "front_id": front_id,
                "quarter": q,
                "bocpd_changepoint_prob": float(result.changepoint_prob[i]),
                "bocpd_alert": int(result.alert[i]),
                "bocpd_map_run_length": int(result.map_run_length[i]),
            })

    return pd.DataFrame(results)
