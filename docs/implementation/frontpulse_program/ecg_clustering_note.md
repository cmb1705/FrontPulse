# ECG Ensemble Clustering: Integration Note

**Task**: FP-ax4.4 (P2.4)
**Status**: Prototype available behind flag
**Date**: 2026-03-23

## What Is ECG?

ECG (Ensemble Clustering for Graphs) runs an ensemble of randomized
single-level Louvain partitions, aggregates co-membership votes into
edge weights, then runs a final Leiden/Louvain on the re-weighted graph.

The key insight: edges that consistently appear within communities across
many randomized runs are "strong" community edges. Edges that vary are
"weak." The final partition uses this evidence to produce more stable
communities.

Reference: Poulin and Theberge, "Ensemble clustering for graphs:
comparisons and applications," Appl Netw Sci 4, 51 (2019).

## Integration

### Usage

```bash
# Default: Leiden (unchanged behavior)
python scripts/communities.py --mode cumulative

# ECG: ensemble clustering (experimental)
python scripts/communities.py --mode cumulative --use-ecg
python scripts/communities.py --mode cumulative --use-ecg --ecg-ens-size 32
```

### Implementation

- `src/community.py`: Added `run_ecg()` function parallel to `run_leiden()`.
  Same input/output contract. Returns additional ECG-specific fields:
  `original_modularity` and `community_strength_index`.
- `scripts/communities.py`: Added `--use-ecg` and `--ecg-ens-size` flags.
  Module-level dispatch variable `_cluster_fn` defaults to `run_leiden`;
  overridden to `run_ecg` when flag is set.
- `requirements.txt`: Added `partition-igraph>=0.0.7`.
- Default behavior is unchanged. No existing pipeline behavior is modified.

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--use-ecg` | False | Enable ECG ensemble clustering |
| `--ecg-ens-size` | 16 | Number of ensemble members |
| `--res` | 0.001 | Resolution parameter (passed to final partition) |

## Expected Stability Improvement

ECG addresses the partition volatility observed in the stability baseline
(FP-ax4.1): median lineage lifespan of 2 quarters, 52.1% ephemeral
lineages. By consensus-weighting edges, ECG should:

1. **Reduce quarter-to-quarter VI**: Communities are anchored by edges with
   consistent co-membership across ensemble runs.
2. **Increase lineage lifespan**: More stable communities lead to longer-lived
   lineages (fewer spurious births/deaths).
3. **Lower PIA variation**: Paper identity alignment should improve when
   communities are less noisy.

## Runtime Considerations

ECG runs `ens_size` Louvain partitions before the final partition. Expected
overhead:

- **ens_size=16** (default): ~16x single Louvain time, plus edge re-weighting
  and final partition. Total ~20x single Leiden.
- **ens_size=32**: ~35x single Leiden.

For the PSC dataset (~70K nodes, ~300K edges at peak), single Leiden takes
~1-2 seconds per quarter. With 90 quarters:

| Configuration | Estimated Total |
|--------------|----------------|
| Leiden (baseline) | ~2-3 minutes |
| ECG ens_size=16 | ~30-45 minutes |
| ECG ens_size=32 | ~60-90 minutes |

This is acceptable for batch processing but not for interactive use.

## Comparison Protocol

To compare ECG against Leiden:

1. Run Leiden baseline: `python scripts/communities.py --mode cumulative`
2. Run ECG: `python scripts/communities.py --mode cumulative --use-ecg`
3. Compare using stability report:
   ```bash
   python scripts/generate_stability_report.py \
       --timeseries data/out/02_lineage_tracking/lineage_timeseries.csv
   ```
4. Key metrics to compare:
   - Mean lineage lifespan (expect ECG > Leiden)
   - Ephemeral lineage fraction (expect ECG < Leiden)
   - Mean VI between consecutive quarters (expect ECG < Leiden)
   - Mean PIA rate (expect ECG >= Leiden)

## Decision Criteria

ECG becomes the default clustering path if:

1. Mean lineage lifespan increases by >= 50% (e.g., 2Q -> 3Q median).
2. VI between consecutive quarters decreases by >= 20%.
3. Runtime stays under 2 hours for full cumulative pipeline.
4. No regression in downstream onset detection metrics (MSD ROC-AUC).

Until these criteria are met, Leiden remains the default.
