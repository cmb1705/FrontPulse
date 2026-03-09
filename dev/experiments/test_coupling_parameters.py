"""Test that coupling parameters are correctly loaded and applied."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.graph_build import CouplingConfig, _COUPLING_DEFAULTS

print("=" * 80)
print("COUPLING PARAMETER VERIFICATION")
print("=" * 80)

print("\nCoupling Defaults:")
for key, value in _COUPLING_DEFAULTS.items():
    print(f"  {key}: {value}")

print("\nCouplingConfig Instance:")
config = CouplingConfig(enabled=True)
print(f"  enabled: {config.enabled}")
print(f"  alpha: {config.alpha}")
print(f"  beta: {config.beta}")
print(f"  lambda_decay: {config.lambda_decay}")
print(f"  min_shared_refs: {config.min_shared_refs}")
print(f"  min_coupling_score: {config.min_coupling_score}")
print(f"  max_year_diff: {config.max_year_diff}")
print(f"  workers: {config.workers}")

print("\n" + "=" * 80)
print("LITERATURE STANDARDS VERIFICATION")
print("=" * 80)

checks = []

# Check min_shared_refs
if config.min_shared_refs == 5:
    print("[PASS] min_shared_refs = 5 (literature standard)")
    checks.append(True)
else:
    print(f"[FAIL] min_shared_refs = {config.min_shared_refs} (expected 5)")
    checks.append(False)

# Check min_coupling_score
if config.min_coupling_score == 0.25:
    print("[PASS] min_coupling_score = 0.25 (core documents threshold, Glanzel & Czerwon)")
    checks.append(True)
else:
    print(f"[FAIL] min_coupling_score = {config.min_coupling_score} (expected 0.25)")
    checks.append(False)

# Check max_year_diff
if config.max_year_diff == 5:
    print("[PASS] max_year_diff = 5 years (research front detection standard)")
    checks.append(True)
else:
    print(f"[FAIL] max_year_diff = {config.max_year_diff} (expected 5)")
    checks.append(False)

print("\n" + "=" * 80)
if all(checks):
    print("[SUCCESS] ALL PARAMETERS MATCH LITERATURE STANDARDS (Option A)")
    print("\nExpected impact:")
    print("  - 2018Q1: ~3M edges (vs. 16.2M baseline)")
    print("  - File size: ~300MB (vs. 1.6GB baseline)")
    print("  - Edge reduction: ~82%")
    print("  - Preserves recent, strong coupling relationships")
else:
    print("[FAILURE] SOME PARAMETERS DO NOT MATCH LITERATURE STANDARDS")
    sys.exit(1)

print("=" * 80)
