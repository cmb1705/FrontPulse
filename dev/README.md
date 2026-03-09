# Development Scripts and Utilities

This directory contains one-off scripts, tests, and utilities used during development that are not part of the main pipeline.

## Directory Structure

### `analysis/`

Historical analysis and comparison scripts:

- `analyze_historical_coupling.py` - Coupling metric analysis across historical data
- `analyze_historical_years.py` - Year-based historical analysis
- `compare_graph_sizes.py` - Graph size comparison utilities
- `ref_resolution.py` - Reference resolution analysis

### `diagnostics/`

Validation and debugging tools:

- `graph_checks.py` - Graph integrity checks
- `check_parquet/` - Parquet file validation utilities

### `experiments/`

Testing scripts and experimental code:

- `test_coupling_parameters.py` - Coupling parameter testing
- `test_parallel_safety.py` - Parallel processing safety tests
- `tests/` - Unit and integration tests
- `scratch_test/` - Experimental test scripts
- `examples/` - Example scripts and demonstrations

### `maintenance/`

Cleanup and maintenance utilities:

- `cleanup_GraphML.py` - GraphML file cleanup

### `temp/`

Temporary analysis scripts:

- `temp_analyze_lgbm.py` - LightGBM analysis
- `temp_bottom.py` - Bottom-level analysis
- `temp_scan.py` - Scanning utilities
- `temp_top.py` - Top-level analysis

## Main Pipeline Scripts

The main pipeline scripts remain in the project root:

- `run.py` - Main pipeline orchestrator
- `tripwire_nb_fdr.py` - Step 8: Tripwire Detection (negative binomial/FDR-based anomaly detection)
