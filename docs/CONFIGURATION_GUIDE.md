# 2YP Configuration Guide

## Critical: Graph Workers and Coupling Cache

### ⚠️ IMPORTANT SAFETY CONSTRAINT

**If coupling is enabled with cache, graph workers are AUTOMATICALLY limited to 1.**

This is enforced by the system to prevent cache corruption.

---

## The Problem

Multiple graph workers writing to the same coupling cache simultaneously creates race conditions:

```
Time  Worker 1 (Q1)          Worker 2 (Q2)
----  ---------------------  ---------------------
T0    Read cache            Read cache
T1    Compute Q1 edges      Compute Q2 edges
T2    Write cache           [waiting]
T3    [done]                Write cache ← Overwrites Q1!
```

**Result:** Last write wins, cache corrupted, incremental builds broken.

---

## Safe Configurations

### Option 1: Coupling with Cache (Sequential, Safe)

**Configuration:**

```yaml
# config/defaults.yaml
coupling:
  enabled: true
  cache_dir: data/out/cache_coupling  # Cache enabled

graphs:
  default_workers: 2  # Will be forced to 1 automatically
```

**Command line:**

```bash
python run.py \
  --graph-workers 4  # System forces to 1, logs warning
```

**Result:**

- ✅ Safe: No cache corruption
- ✅ Incremental: Cache benefits preserved
- ❌ Slow: Sequential graph building

**Best for:** Incremental updates when you want cache benefits

---

### Option 2: Coupling without Cache (Parallel, Fast)

**Configuration:**

```yaml
# config/defaults.yaml
coupling:
  enabled: true
  cache_dir: null  # No cache

graphs:
  default_workers: 4  # Parallel allowed
```

**Command line:**

```bash
python run.py \
  --graph-workers 4  # Full parallelization
```

**Result:**

- ✅ Safe: No shared state
- ✅ Fast: Parallel graph building (~4x speedup)
- ❌ No cache: Full coupling recomputation each time

**Best for:** Full rebuilds, parameter tuning, one-off analyses

---

### Option 3: No Coupling (Parallel, Fastest)

**Configuration:**

```yaml
# config/defaults.yaml
coupling:
  enabled: false  # Coupling disabled

graphs:
  default_workers: 4  # Parallel allowed
```

**Command line:**

```bash
python run.py \
  --graph-workers 4  # Full parallelization
```

**Result:**

- ✅ Safe: No coupling computation
- ✅ Fastest: Pure citation network only
- ❌ No coupling: Missing bibliographic coupling edges

**Best for:** Citation-only analysis, maximum speed

---

## Automatic Enforcement

The system automatically detects unsafe configurations and forces safe mode:

### Example: Coupling Cache Detected

```bash
$ python run.py --graph-workers 4
[WARNING] COUPLING CACHE SAFETY: Forcing graph_workers=1 (sequential mode)
          to prevent cache corruption. Parallel graph building with shared
          coupling cache causes race conditions where workers overwrite each
          other's cache. To use parallel building: disable coupling cache
          (set cache_dir=null) or disable coupling entirely.
```

**No manual intervention needed** - system protects you automatically.

---

## Performance Comparison

### Full Rebuild (all quarters from scratch)

| Configuration | Time | Safety | Use Case |
|---------------|------|--------|----------|
| **Coupling + Cache + Workers=1** | 8 hours | ✅ Safe | Not recommended for full rebuild |
| **Coupling + No Cache + Workers=4** | 2 hours | ✅ Safe | ✅ **Recommended for full rebuild** |
| **No Coupling + Workers=4** | 30 min | ✅ Safe | Citation-only analysis |

### Incremental Update (new quarter added)

| Configuration | Time | Safety | Use Case |
|---------------|------|--------|----------|
| **Coupling + Cache + Workers=1** | 15 min | ✅ Safe | ✅ **Recommended for incremental** |
| **Coupling + No Cache + Workers=4** | 2 hours | ✅ Safe | Not recommended (loses benefit) |
| **No Coupling + Workers=4** | 5 min | ✅ Safe | Citation-only incremental |

---

## Workflow Recommendations

### First-Time Setup / Parameter Changes

When building graphs for the first time or after changing coupling parameters:

1. **Disable coupling cache** for speed:

   ```yaml
   coupling:
     cache_dir: null
   ```

2. **Use maximum parallelization**:

   ```bash
   python run.py --graph-workers 4
   ```

3. **Re-enable cache for future incremental builds** (optional):

   ```yaml
   coupling:
     cache_dir: data/out/cache_coupling
   ```

### Regular Incremental Updates

When adding new quarters to existing graphs:

1. **Keep cache enabled**:

   ```yaml
   coupling:
     cache_dir: data/out/cache_coupling
   ```

2. **Accept sequential mode** (automatic):

   ```bash
   python run.py  # System forces workers=1
   ```

3. **Benefit from incremental cache** - only new edges computed

---

## Configuration Matrix

| Coupling | Cache | Workers | Automatic Action | Safe? | Fast? |
|----------|-------|---------|------------------|-------|-------|
| ✅ Yes | ✅ Yes | >1 | **Force to 1** | ✅ Yes | ❌ No |
| ✅ Yes | ✅ Yes | 1 | Keep 1 | ✅ Yes | ❌ No |
| ✅ Yes | ❌ No | >1 | Allow parallel | ✅ Yes | ✅ Yes |
| ✅ Yes | ❌ No | 1 | Keep 1 | ✅ Yes | ❌ No |
| ❌ No | - | >1 | Allow parallel | ✅ Yes | ✅ Yes |
| ❌ No | - | 1 | Keep 1 | ✅ Yes | ❌ No |

**Key:** System only intervenes when coupling + cache + workers>1 detected.

---

## Command Line Examples

### Example 1: Full Rebuild with Parallelization

```bash
# Temporarily disable cache in config or via environment
python run.py \
  --config config/datasources.yaml \
  --schema config/schema.yaml \
  --slices config/slices.yaml \
  --graph-mode cumulative \
  --graph-workers 4 \
  --coupling-workers 12
```

**Set in config first:**

```yaml
coupling:
  cache_dir: null  # Disable cache
```

### Example 2: Incremental Update with Cache

```bash
# Keep cache enabled in config
python run.py \
  --config config/datasources.yaml \
  --schema config/schema.yaml \
  --slices config/slices.yaml \
  --graph-mode cumulative \
  --graph-workers 4  # System forces to 1, logs warning
  --coupling-workers 12
```

**Config:**

```yaml
coupling:
  cache_dir: data/out/cache_coupling  # Cache enabled
```

**Output:**

```
[WARNING] COUPLING CACHE SAFETY: Forcing graph_workers=1 (sequential mode)...
```

### Example 3: Citation-Only (Maximum Speed)

```bash
python run.py \
  --config config/datasources.yaml \
  --schema config/schema.yaml \
  --slices config/slices.yaml \
  --graph-mode cumulative \
  --graph-workers 4 \
  --no-coupling  # Disable coupling
```

---

## Troubleshooting

### "Why is my parallel build running sequentially?"

**Check coupling cache:**

```bash
grep -A 3 "coupling:" config/defaults.yaml
```

If you see `cache_dir: data/out/cache_coupling`, the system is protecting you from cache corruption.

**Solution:** Disable cache for parallel builds:

```yaml
coupling:
  cache_dir: null
```

### "My cache seems corrupted/ineffective"

**Symptom:** Incremental builds recompute everything, no speed benefit.

**Likely cause:** Ran with `graph_workers > 1` before automatic enforcement was added.

**Solution:**

1. Delete corrupted cache:

   ```bash
   rm -rf data/out/cache_coupling/*
   ```

2. Rebuild with cache enabled and `workers=1`:

   ```bash
   python run.py --graph-workers 1
   ```

### "Can I ever use parallel with coupling?"

**Yes, but only without cache:**

```yaml
coupling:
  enabled: true
  cache_dir: null  # ← This enables parallelization
```

**Trade-off:** No incremental benefits, full recomputation each time.

---

## Technical Details

### Why Does This Happen?

Coupling cache stores:

```
data/out/cache_coupling/
├── coupling_edges.parquet  ← All workers read/write this
├── coupling_nodes.json     ← All workers read/write this
└── coupling_config.json    ← All workers read/write this
```

**All workers share one cache directory** → simultaneous writes → last writer overwrites → corruption.

### Future Enhancement

Planned: Quarter-specific cache directories to enable safe parallel caching:

```
data/out/cache_coupling/
├── cache_2018Q1/  ← Worker 1
├── cache_2018Q2/  ← Worker 2
├── cache_2018Q3/  ← Worker 3
└── cache_2018Q4/  ← Worker 4
```

This would provide:

- ✅ Safe parallelization
- ✅ Per-quarter incremental benefits
- ✅ Best of both worlds

**Status:** Not yet implemented. For now, use sequential mode with cache.

---

## Summary

1. **Coupling + Cache = Sequential mode (automatic)**
   - System forces `graph_workers=1`
   - Safe for incremental builds
   - Slower but preserves cache integrity

2. **Coupling + No Cache = Parallel allowed**
   - Set `cache_dir: null`
   - Safe for full rebuilds
   - Fast but no incremental benefit

3. **No Coupling = Parallel allowed**
   - Set `enabled: false`
   - Safe for citation-only analysis
   - Fastest option

**The system protects you automatically** - just be aware of the trade-offs for your use case.

---

## See Also

- [config/defaults.yaml](../config/defaults.yaml) - Configuration file
- [PERFORMANCE_CONFIG.md](PERFORMANCE_CONFIG.md) - Performance tuning guide
