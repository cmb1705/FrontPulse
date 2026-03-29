#!/usr/bin/env python3
"""
Systematic script to update all data/out/ paths in scripts to new directory structure.
"""

from pathlib import Path

# Define path mappings (order matters - do more specific paths first!)
PATH_MAPPINGS = [
    # Evaluation/validation directories (must come before general patterns)
    ('data/out/06_validation/archived/2025_11_01_intercept_fix/', 'data/out/06_validation/archived/2025_11_01_intercept_fix/'),
    ('data/out/06_validation/archived/2025_11_01_floor_10/', 'data/out/06_validation/archived/2025_11_01_floor_10/'),
    ('data/out/06_validation/archived/2025_11_01_floor_02/', 'data/out/06_validation/archived/2025_11_01_floor_02/'),

    # Lineage tracking files (02_lineage_tracking)
    ('data/out/02_lineage_tracking/lineage_registry.json', 'data/out/02_lineage_tracking/lineage_registry.json'),
    ('data/out/02_lineage_tracking/lineage_timeseries.csv', 'data/out/02_lineage_tracking/lineage_timeseries.csv'),
    ('data/out/02_lineage_tracking/lineage_metrics.csv', 'data/out/02_lineage_tracking/lineage_metrics.csv'),
    ('data/out/02_lineage_tracking/lineage_embeddings.npz', 'data/out/02_lineage_tracking/lineage_embeddings.npz'),
    ('data/out/02_lineage_tracking/lineage_ctfidf_terms.csv', 'data/out/02_lineage_tracking/lineage_ctfidf_terms.csv'),
    ('data/out/02_lineage_tracking/lineage_npmi_pairs.csv', 'data/out/02_lineage_tracking/lineage_npmi_pairs.csv'),
    ('data/out/02_lineage_tracking/lineage_stratification_summary.json', 'data/out/02_lineage_tracking/lineage_stratification_summary.json'),

    # Milestone mapping files (03_milestone_mapping)
    ('data/out/03_milestone_mapping/lineage_front_similarity.csv', 'data/out/03_milestone_mapping/lineage_front_similarity.csv'),
    ('data/out/03_milestone_mapping/lineage_front_term_similarity.csv', 'data/out/03_milestone_mapping/lineage_front_term_similarity.csv'),
    ('data/out/03_milestone_mapping/lineage_front_npmi_similarity.csv', 'data/out/03_milestone_mapping/lineage_front_npmi_similarity.csv'),
    ('data/out/03_milestone_mapping/lineage_front_mappings.csv', 'data/out/03_milestone_mapping/lineage_front_mappings.csv'),

    # Front aggregation files (04_front_aggregation)
    ('data/out/04_front_aggregation/front_timeseries', 'data/out/04_front_aggregation/front_timeseries'),
    ('data/out/04_front_aggregation/front_metrics', 'data/out/04_front_aggregation/front_metrics'),

    # Validation files (06_validation)
    # Note: study-specific milestone catalogs now live under _local/psc/references/
]

def update_file(filepath: Path) -> tuple[int, bool]:
    """
    Update a single Python file with new paths.
    Returns (num_replacements, was_modified)
    """
    try:
        content = filepath.read_text(encoding='utf-8')
        original_content = content
        replacements = 0

        for old_path, new_path in PATH_MAPPINGS:
            if old_path in content:
                count = content.count(old_path)
                content = content.replace(old_path, new_path)
                replacements += count

        if content != original_content:
            filepath.write_text(content, encoding='utf-8')
            return replacements, True

        return 0, False

    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return 0, False

def main():
    scripts_dir = Path('scripts')

    print("=" * 80)
    print("UPDATING DATA/OUT PATHS IN SCRIPTS")
    print("=" * 80)

    # Find all Python files in scripts/
    python_files = list(scripts_dir.glob('*.py'))
    print(f"\nFound {len(python_files)} Python files to check")

    total_replacements = 0
    modified_files = []

    for filepath in sorted(python_files):
        replacements, modified = update_file(filepath)
        if modified:
            modified_files.append(filepath.name)
            total_replacements += replacements
            print(f"[OK] {filepath.name}: {replacements} replacements")

    print("\n" + "=" * 80)
    print(f"SUMMARY: Updated {len(modified_files)} files with {total_replacements} path changes")
    print("=" * 80)

    if modified_files:
        print("\nModified files:")
        for filename in modified_files:
            print(f"  - {filename}")

if __name__ == '__main__':
    main()
