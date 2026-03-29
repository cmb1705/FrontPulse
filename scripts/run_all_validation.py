#!/usr/bin/env python3
"""
Master Validation Runner

Runs all Stage 2-5 validation scripts and tripwire visualizations in sequence.
This ensures data quality and generates comprehensive validation reports for the
entire lineage-to-front mapping and tripwire detection pipeline.

Usage:
    python scripts/run_all_validation.py [--skip-stages PHASES]

Options:
    --skip-stages: Comma-separated list of stages to skip (e.g., "2,3")
"""

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def run_script(script_path: Path, description: str) -> bool:
    """
    Run a validation script and report status.

    Returns True if successful, False otherwise.
    """
    print(f"\n{'='*70}")
    print(f"{description}")
    print(f"{'='*70}\n")

    try:
        subprocess.run(
            [sys.executable, str(script_path)],
            check=True,
            capture_output=False,
            text=True
        )
        print(f"\n[OK] {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n[FAIL] {description} FAILED (exit code {e.returncode})")
        return False
    except Exception as e:
        print(f"\n[ERROR] {description} ERROR: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Run all validation scripts")
    parser.add_argument(
        "--skip-stages",
        type=str,
        default="",
        help="Comma-separated list of stages to skip (e.g., '2,3')"
    )
    args = parser.parse_args()

    # Parse skip list
    skip_stages = set()
    if args.skip_stages:
        skip_stages = set(args.skip_stages.split(','))

    print("="*70)
    print("MASTER VALIDATION RUNNER")
    print("="*70)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Skipping stages: {', '.join(skip_stages) if skip_stages else 'None'}")
    print()

    scripts_dir = Path("scripts")
    results = {}

    # Stage 2: SciBERT Embeddings
    if '2' not in skip_stages:
        results['stage2'] = run_script(
            scripts_dir / "validate_stage2.py",
            "Stage 2 Validation (SciBERT Embeddings)"
        )

    # Stage 3: c-TF-IDF Terms
    if '3' not in skip_stages:
        results['stage3'] = run_script(
            scripts_dir / "validate_stage3.py",
            "Stage 3 Validation (c-TF-IDF Terms)"
        )

    # Stage 4: NPMI Co-occurrence
    if '4' not in skip_stages:
        results['stage4'] = run_script(
            scripts_dir / "validate_stage4.py",
            "Stage 4 Validation (NPMI Co-occurrence)"
        )

    # Stage 5: Ensemble Mapping
    if '5' not in skip_stages:
        results['stage5'] = run_script(
            scripts_dir / "validate_stage5.py",
            "Stage 5 Validation (Ensemble Mapping)"
        )

    # Tripwire Validation & Visualization
    # Note: This requires tripwire_alerts.csv and validation_results.csv from evaluate_tripwire.py
    tripwire_files_exist = (
        Path("data/out/06_validation/tripwire_alerts.csv").exists() and
        Path("data/out/06_validation/validation_results.csv").exists()
    )

    if tripwire_files_exist:
        results['tripwire_viz'] = run_script(
            scripts_dir / "visualize_tripwire_comprehensive.py",
            "Tripwire Visualization"
        )
    else:
        print(f"\n{'='*70}")
        print("Tripwire Visualization (SKIPPED)")
        print(f"{'='*70}\n")
        print("[SKIP] Tripwire validation files not found.")
        print("Run scripts/evaluate_tripwire.py first to generate:")
        print("  - data/out/06_validation/tripwire_alerts.csv")
        print("  - data/out/06_validation/validation_results.csv")
        results['tripwire_viz'] = None

    # Summary
    print(f"\n\n{'='*70}")
    print("VALIDATION SUMMARY")
    print(f"{'='*70}\n")

    passed = sum(1 for v in results.values() if v is True)
    failed = sum(1 for v in results.values() if v is False)
    skipped = sum(1 for v in results.values() if v is None)

    print("Results:")
    for name, status in results.items():
        status_str = "[PASS]" if status is True else "[FAIL]" if status is False else "[SKIP]"
        print(f"  {name:20s} {status_str}")

    print(f"\nTotal: {passed} passed, {failed} failed, {skipped} skipped")
    print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Exit code based on results
    if failed > 0:
        print("\n[WARNING] Some validations failed. Review output above for details.")
        sys.exit(1)
    else:
        print("\n[OK] All validations passed successfully!")
        sys.exit(0)


if __name__ == '__main__':
    main()
