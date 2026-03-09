#!/usr/bin/env python3
"""
Apply provenance tracking to remaining metric scripts.

This script modifies metric scripts to track input files with SHA256 hashes,
matching the pattern implemented in metric_author_influx.py.
"""

import re
from pathlib import Path

def patch_metric_script(script_path: Path, compute_func_name: str) -> bool:
    """
    Patch a metric script to add provenance tracking.

    Returns True if changes were made, False otherwise.
    """
    content = script_path.read_text()
    original_content = content

    # Step 1: Add Tuple to imports if not present
    if "from typing import" in content and "Tuple" not in content:
        content = re.sub(
            r'(from typing import [^)]+)',
            lambda m: m.group(1).rstrip(')') + (', Tuple' if not m.group(1).endswith(',') else ' Tuple') + (')'  if ')' in m.group(0) else ''),
            content
        )

    # Step 2: Modify compute function to return (payload, input_files)
    # Find the compute function
    compute_pattern = rf'def {compute_func_name}\([^)]+\)( -> [^:]+)?:'
    if re.search(compute_pattern, content):
        # Change return type
        content = re.sub(
            rf'def {compute_func_name}\([^)]+\) -> Dict\[str, object\]:',
            f'def {compute_func_name}(args: argparse.Namespace) -> Tuple[Dict[str, object], List[Path]]:',
            content
        )

        # Add input_files tracking at start of function
        func_start_pattern = rf'(def {compute_func_name}\([^)]+\)[^:]*:\s+(?:[^\n]+\n)*?\s+)(\w+.*?=.*?\[\])'
        content = re.sub(
            func_start_pattern,
            lambda m: m.group(1) + m.group(2) + '\n    input_files: List[Path] = []  # Track input files for provenance\n',
            content,
            count=1
        )

        # Add input file tracking in loop
        # Find the for loop that iterates over quarter slices
        loop_pattern = r'for idx, \(quarter, path\) in enumerate\(iter_quarter_slices\([^)]+\)\):'
        if re.search(loop_pattern, content):
            content = re.sub(
                r'(for idx, \(quarter, path\) in enumerate\(iter_quarter_slices\([^)]+\)\):\s+if [^\n]+\n\s+break\s+)',
                r'\1input_files.append(path)  # Record input file\n        ',
                content
            )

        # Modify return statement to return (payload, input_files)
        # Find the return statement in compute function
        content = re.sub(
            rf'(def {compute_func_name}.*?)(    return \{{\s*"metric":)',
            lambda m: m.group(1).replace('    return {', '    payload = {', 1) if '    return {' in m.group(0) else m.group(0),
            content,
            flags=re.DOTALL
        )
        content = re.sub(
            r'(payload = \{\s*"metric"[^\}]+\})\s*$',
            r'\1\n    return payload, input_files',
            content,
            flags=re.MULTILINE
        )

    # Step 3: Modify write_standardized_outputs signature
    content = re.sub(
        r'def write_standardized_outputs\(\s*payload: Dict\[str, object\],\s*args: argparse\.Namespace,\s*\)',
        'def write_standardized_outputs(\n    payload: Dict[str, object],\n    input_files: List[Path],\n    args: argparse.Namespace,\n)',
        content
    )

    # Step 4: Update create_metric_metadata call
    # Change input_files=[] to input_files=input_files
    # Also add num_input_files to parameters
    content = re.sub(
        r'input_files=\[\],  # Input files are numerous quarterly slices',
        'input_files=input_files,  # Track all input slice files',
        content
    )

    # Add num_input_files to parameters dict
    content = re.sub(
        r'(parameters=\{[^}]+)(\s+\})',
        r'\1,\n            "num_input_files": len(input_files),\2',
        content
    )

    # Step 5: Update main() to unpack tuple and pass input_files
    content = re.sub(
        rf'payload = {compute_func_name}\(args\)',
        f'payload, input_files = {compute_func_name}(args)',
        content
    )

    content = re.sub(
        r'write_standardized_outputs\(payload, args\)',
        'write_standardized_outputs(payload, input_files, args)',
        content
    )

    # Step 6: Update comment
    content = re.sub(
        r'# Standardized parquet outputs \(Task 1\.1\)',
        '# Standardized parquet outputs with provenance tracking (Task 1.1 + 1.2)',
        content
    )

    if content != original_content:
        script_path.write_text(content)
        return True
    return False


def main():
    scripts = [
        ("scripts/metric_citation_velocity.py", "compute_velocity"),
        ("scripts/metric_reference_vitality.py", "compute_vitality"),
        ("scripts/metric_topic_diversity.py", "compute_diversity"),
        ("scripts/metric_cross_cluster_bridging.py", "compute_bridging"),
    ]

    for script_path, compute_func in scripts:
        path = Path(script_path)
        if path.exists():
            print(f"Patching {script_path}...")
            if patch_metric_script(path, compute_func):
                print(f"  ✓ Applied provenance tracking")
            else:
                print(f"  ✗ No changes made (already patched or different structure)")
        else:
            print(f"  ✗ File not found: {script_path}")


if __name__ == "__main__":
    main()
