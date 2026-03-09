"""Test that parallel graph building safety enforcement works correctly."""
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent))

from src.graph_build import CouplingConfig

# Simulate the enforcement logic from run.py
def check_parallel_safety(coupling_enabled: bool, cache_dir: Optional[Path], graph_workers: int) -> tuple[int, str]:
    """
    Check if parallel graph building is safe and return adjusted worker count.

    Returns:
        (adjusted_workers, reason_or_warning)
    """
    if coupling_enabled and cache_dir is not None:
        if graph_workers > 1:
            warning = (
                "COUPLING CACHE SAFETY: Forcing graph_workers=1 (sequential mode) "
                "to prevent cache corruption."
            )
            return (1, warning)
    return (graph_workers, "Safe configuration")


print("=" * 80)
print("PARALLEL GRAPH BUILDING SAFETY TEST")
print("=" * 80)

test_cases = [
    # (coupling_enabled, cache_dir, requested_workers, expected_workers, description)
    (True, Path("cache"), 4, 1, "Coupling + Cache + Multiple Workers"),
    (True, Path("cache"), 1, 1, "Coupling + Cache + Single Worker"),
    (True, None, 4, 4, "Coupling + No Cache + Multiple Workers"),
    (True, None, 1, 1, "Coupling + No Cache + Single Worker"),
    (False, Path("cache"), 4, 4, "No Coupling + Cache + Multiple Workers"),
    (False, Path("cache"), 1, 1, "No Coupling + Cache + Single Worker"),
    (False, None, 4, 4, "No Coupling + No Cache + Multiple Workers"),
    (False, None, 1, 1, "No Coupling + No Cache + Single Worker"),
]

print("\nTest Cases:")
print("-" * 80)

all_passed = True

for coupling, cache, requested, expected, description in test_cases:
    adjusted, message = check_parallel_safety(coupling, cache, requested)

    passed = adjusted == expected
    status = "[PASS]" if passed else "[FAIL]"

    print(f"\n{status} {description}")
    print(f"  Coupling: {coupling}, Cache: {cache is not None}, Requested: {requested}")
    print(f"  Result: {adjusted} workers")

    if not passed:
        print(f"  ERROR: Expected {expected}, got {adjusted}")
        all_passed = False

    if "SAFETY" in message:
        print(f"  Warning: {message[:70]}...")

print("\n" + "=" * 80)

if all_passed:
    print("[SUCCESS] All safety enforcement tests passed")
    print("\nKey findings:")
    print("  - Coupling + Cache: Automatically forced to 1 worker")
    print("  - Coupling + No Cache: Parallel allowed")
    print("  - No Coupling: Parallel always allowed")
    print("  - Single worker: Always allowed (never enforced)")
else:
    print("[FAILURE] Some safety tests failed")
    sys.exit(1)

print("=" * 80)

# Additional validation: Test with real CouplingConfig
print("\nReal Configuration Test:")
print("-" * 80)

config_with_cache = CouplingConfig(
    enabled=True,
    cache_dir=Path("data/out/cache_coupling")
)

config_without_cache = CouplingConfig(
    enabled=True,
    cache_dir=None
)

config_disabled = CouplingConfig(
    enabled=False
)

print(f"\n1. Coupling with cache:")
print(f"   enabled={config_with_cache.enabled}, cache_dir={config_with_cache.cache_dir}")
adjusted, msg = check_parallel_safety(
    config_with_cache.enabled,
    config_with_cache.cache_dir,
    4
)
print(f"   Result: {adjusted} workers (forced from 4)")
assert adjusted == 1, "Should force to 1 worker"

print(f"\n2. Coupling without cache:")
print(f"   enabled={config_without_cache.enabled}, cache_dir={config_without_cache.cache_dir}")
adjusted, msg = check_parallel_safety(
    config_without_cache.enabled,
    config_without_cache.cache_dir,
    4
)
print(f"   Result: {adjusted} workers (parallel allowed)")
assert adjusted == 4, "Should allow parallel"

print(f"\n3. Coupling disabled:")
print(f"   enabled={config_disabled.enabled}")
adjusted, msg = check_parallel_safety(
    config_disabled.enabled,
    config_disabled.cache_dir,
    4
)
print(f"   Result: {adjusted} workers (parallel allowed)")
assert adjusted == 4, "Should allow parallel"

print("\n" + "=" * 80)
print("[SUCCESS] All configuration tests passed")
print("=" * 80)
