# Performance Configuration Quick Reference

## Global Parallel Workers Setting

All parallel operations in the FrontPulse pipeline now use a single, centralized default that can be easily adjusted.

### Quick Start

**Location**: [run.py](../run.py) line 49

```python
DEFAULT_PARALLEL_WORKERS = 12
```

### What This Controls

This single setting controls the default number of workers for:

1. **Graph Building** (`--graph-workers`)
   - Annual graph construction
   - Delta graph construction
   - Cumulative graph construction

2. **Coupling Calculations** (`--coupling-workers`)
   - Parallel computation of bibliographic coupling pairs
   - Shared reference counting

3. **CouplingConfig Default** (`src/graph_build.py`)
   - Default workers for coupling when programmatically creating configs

### Recommended Values by System

| System Specs | CPU Cores | RAM | Recommended Value |
|--------------|-----------|-----|-------------------|
| **Minimum** | 4 cores | 8 GB | `4` |
| **Standard** | 8 cores | 16 GB | `6-8` |
| **High-Performance** | 16+ cores | 32+ GB | **`12`** (default) |
| **Workstation** | 32+ cores | 64+ GB | `16-20` |

### How to Change

#### Option 1: Change Global Default (Recommended)

Edit `run.py` line 49:

```python
# For 8-core systems
DEFAULT_PARALLEL_WORKERS = 8

# For 4-core systems
DEFAULT_PARALLEL_WORKERS = 4

# For high-end workstations
DEFAULT_PARALLEL_WORKERS = 16
```

**Pros**:

- Changes apply to all parallel operations automatically
- Only need to update one place
- All CLI help text updates automatically

#### Option 2: Override via Command Line

Use `--graph-workers` and `--coupling-workers` arguments:

```bash
# Override for a single run (8-core system)
python run.py --config config.yaml --graph-workers 8 --coupling-workers 8

# Override for a single run (4-core system)
python run.py --config config.yaml --graph-workers 4 --coupling-workers 4
```

**Pros**:

- No code changes needed
- Can experiment with different values
- Good for one-off adjustments

### Performance Considerations

#### CPU Usage

- Each worker is a separate process
- Target: 75-90% CPU utilization across all cores
- If CPU usage is low, increase workers
- If CPU usage is >95%, consider reducing workers slightly

#### Memory Usage

- Each worker needs its own memory for the DataFrame
- Rough estimate: `Total RAM / (workers + 2)` should be ≥ 2 GB
- Example: 32 GB RAM → up to 12-14 workers comfortable
- Example: 16 GB RAM → up to 6-8 workers comfortable

#### Disk I/O

- Many workers can saturate disk bandwidth
- SSDs handle this much better than HDDs
- If disk is bottleneck (check task manager), reduce workers

### Monitoring Performance

Run with verbose logging to see worker utilization:

```bash
python run.py --config config.yaml --log-level DEBUG
```

Look for messages like:

```
INFO: Building annual graphs for 10 years (parallel workers: 12)
INFO: Completed annual graph for year 2015
INFO: Completed annual graph for year 2016
...
```

### Current System Optimization

Your system (AMD Ryzen 9 6900HX, 16 cores, 32GB RAM):

```python
DEFAULT_PARALLEL_WORKERS = 12  # ✅ Already optimal!
```

**Why 12 instead of 16?**

- Leaves headroom for OS and other processes
- Each worker may spawn its own threads
- Avoids context switching overhead
- Empirically tested sweet spot for this CPU class

### Testing Different Values

Quick test script to find optimal workers for your system:

```bash
# Test with 8 workers
time python run.py --config config.yaml --graph-mode annual --graph-workers 8

# Test with 10 workers
time python run.py --config config.yaml --graph-mode annual --graph-workers 10

# Test with 12 workers (default)
time python run.py --config config.yaml --graph-mode annual --graph-workers 12

# Test with 14 workers
time python run.py --config config.yaml --graph-mode annual --graph-workers 14
```

Compare wall-clock time and choose the fastest. Diminishing returns typically start around 75% of core count.

---

## Related Documentation

- [Configuration Guide](CONFIGURATION_GUIDE.md)
- [Pipeline Implementation](implementation/PIPELINE_IMPLEMENTATION.md)
