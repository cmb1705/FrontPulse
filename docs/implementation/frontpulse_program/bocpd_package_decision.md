# BOCPD Package Decision and Detector Interface

**Task**: FP-92e.1 (P3.1)
**Status**: Decision made
**Date**: 2026-03-23

## Package Decision

### Primary: Custom Poisson-Gamma BOCPD

Implement a lean BOCPD detector using conjugate Poisson-Gamma priors.
This is the natural model for the FrontPulse data contract: count-valued
time series (new_works per front per quarter).

**Rationale**:

1. **Exact conjugacy**: The Poisson-Gamma model provides closed-form
   posterior updates. No MCMC or variational approximation needed.
   - Observation: `new_works_t ~ Poisson(lambda)`
   - Prior: `lambda ~ Gamma(alpha, beta)`
   - Posterior: `Gamma(alpha + sum(x), beta + n)`
2. **Minimal dependencies**: Pure NumPy/SciPy implementation (~80 lines
   of core logic). No new pip packages required.
3. **Sequential processing**: True online algorithm. Processes one quarter
   at a time, maintaining run-length distribution. Suitable for both batch
   replay and future streaming use.
4. **Interpretable outputs**: Produces per-quarter changepoint probability
   (marginal probability of run length = 0), directly compatible with
   the timeliness scoring interface.

### Comparison baseline: ruptures (offline)

ruptures v1.1.10 is already installed in the environment. Use it as an
offline changepoint detection baseline for benchmarking (P3.5):

- PELT algorithm with `rbf` or `l2` cost function
- Provides optimal offline segmentation for comparison
- Not suitable for online/prospective detection

### Rejected alternatives

| Package | Reason for rejection |
|---------|---------------------|
| `bayesian_changepoint_detection` | Unmaintained (last commit 2019), Gaussian-only observation model, would need forking for Poisson data |
| `bocpd` (PyPI) | Minimal adoption, undocumented, no count-data support |
| `changepoynt` | Heavy dependencies (TensorFlow), over-engineered for quarterly count data |
| ruptures as primary | Offline-only; cannot produce sequential per-quarter probabilities needed for prospective evaluation |

### Fallback plan

If the custom BOCPD proves unreliable (e.g., numerical instability with
large counts or long run lengths):

1. Cap run length at 40 quarters (10 years) to bound the distribution.
2. Use log-space computation for predictive probabilities.
3. If still unstable, fall back to ruptures PELT as a non-Bayesian detector,
   converting segmentation boundaries to binary changepoint labels.

## Detector Interface

The BOCPD detector must conform to this interface for interoperability
with the evaluation contract and timeliness scoring utilities.

### Input contract

```python
@dataclass
class BOCPDConfig:
    """Configuration for BOCPD detector."""
    # Prior parameters (Poisson-Gamma conjugate)
    alpha0: float = 1.0      # Gamma shape prior
    beta0: float = 0.1       # Gamma rate prior
    # Hazard function
    hazard_rate: float = 1/50  # P(changepoint) per quarter (1/expected_run_length)
    # Run length truncation
    max_run_length: int = 40   # Cap for numerical stability
    # Output threshold
    threshold: float = 0.5     # Changepoint probability threshold for binary alert
```

### Core function

```python
def detect_changepoints(
    counts: np.ndarray,          # 1-d array of new_works per quarter
    config: BOCPDConfig | None,  # None uses defaults
) -> BOCPDResult:
    """Run BOCPD on a single front's count series.

    Returns per-quarter changepoint probabilities and binary alerts.
    """
```

### Output contract

```python
@dataclass
class BOCPDResult:
    """BOCPD detection result for a single front."""
    # Per-quarter outputs (same length as input counts)
    changepoint_prob: np.ndarray   # P(run_length=0) per quarter, range [0,1]
    alert: np.ndarray              # Binary: 1 if changepoint_prob >= threshold
    # Run-length distribution (optional, for diagnostics)
    map_run_length: np.ndarray     # MAP run length per quarter
    # Summary
    n_alerts: int
    alert_quarters: list[int]      # Indices where alert=1
```

### CLI entry point

```python
def run_bocpd_on_fronts(
    series_df: pd.DataFrame,       # Front-level series (front_id, quarter, new_works)
    config: BOCPDConfig | None,
) -> pd.DataFrame:
    """Run BOCPD on all fronts. Returns DataFrame with:
    front_id, quarter, bocpd_changepoint_prob, bocpd_alert
    """
```

### Compatibility with evaluation contract

The BOCPD output DataFrame must support merging with ground-truth onset
labels on `(front_id, quarter)` for timeliness evaluation:

```python
# Merge BOCPD alerts with ground truth
merged = pd.merge(
    bocpd_output[["front_id", "quarter", "bocpd_alert"]],
    ground_truth[["front_id", "onset_quarter"]],
    on="front_id",
)
# Feed to timeliness scoring
result = score_timeliness(
    true_onsets=merged["onset_quarter"],
    detected_quarters=merged.loc[merged["bocpd_alert"] == 1, "quarter"],
)
```

## Dependency Implications

### No new pip packages required

The custom BOCPD implementation uses only:
- `numpy` (already installed)
- `scipy.special` for `gammaln` (already installed via scipy)
- `pandas` for DataFrame I/O (already installed)

### ruptures (already available)

ruptures is installed but not in `requirements.txt`. Add it for
reproducibility:

```
ruptures>=1.1.10  # Offline changepoint detection baseline
```

### Test dependencies

No additional test dependencies. Standard pytest fixtures suffice.

## Implementation Sequence

```
P3.1 (this task) -- package decision + interface
    |
    v
P3.2 -- implement src/bocpd.py + CLI script
    |
    v
P3.3 -- calibrate priors on PSC front data
    |
    v
P3.5 -- benchmark BOCPD vs MSD vs baselines
```

## Algorithm Sketch

The Adams-MacKay (2007) BOCPD with Poisson-Gamma conjugate:

1. Initialize run-length distribution: `P(r_0 = 0) = 1`
2. For each new observation `x_t`:
   a. Compute predictive probability `P(x_t | r_t)` for each run length
      using negative binomial (Poisson-Gamma predictive)
   b. Compute growth probability: `P(r_t = r_{t-1}+1)` = `(1-H) * P(r_{t-1}) * P(x_t|r_t)`
   c. Compute changepoint probability: `P(r_t = 0)` = `H * sum(P(r_{t-1}) * P(x_t|r_t))`
   d. Normalize to get `P(r_t | x_{1:t})`
   e. Update sufficient statistics: `alpha_r += x_t`, `beta_r += 1`
3. Output `P(r_t = 0)` as changepoint probability for quarter t

Where `H = hazard_rate` is the constant hazard function (geometric prior
on run lengths).
