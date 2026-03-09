"""Quick script to check abstract availability in raw JSONL."""
import json
from pathlib import Path

jsonl_path = Path("data/current_ingest/raw/openalex_raw_20251021T000808Z_part0000.jsonl")

count_with_abstract = 0
total = 0
sample_abstract = None
sample_id = None

with open(jsonl_path, 'r') as f:
    for i, line in enumerate(f):
        if i >= 100:  # Check first 100 records
            break
        total += 1
        data = json.loads(line)

        if 'abstract_inverted_index' in data and data['abstract_inverted_index']:
            count_with_abstract += 1
            if sample_abstract is None:
                sample_abstract = data['abstract_inverted_index']
                sample_id = data['id']

print(f"Checked {total} records")
print(f"Records with abstract_inverted_index: {count_with_abstract}")
print(f"Percentage: {count_with_abstract/total*100:.1f}%")

if sample_abstract:
    print(f"\nSample abstract from {sample_id}:")
    print(f"Type: {type(sample_abstract)}")
    print(f"First 10 terms: {list(sample_abstract.keys())[:10]}")
    print(f"\nInverted index sample (first term):")
    first_term = list(sample_abstract.keys())[0]
    print(f"  '{first_term}': {sample_abstract[first_term]}")
else:
    print("\nNo abstracts found in sample")
