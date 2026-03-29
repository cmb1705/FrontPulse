"""Analyze Phase 3 improvements after alias expansion."""
import pandas as pd

# Load Phase 3 similarity matrix
df = pd.read_csv('data/out/03_milestone_mapping/lineage_front_term_similarity.csv', index_col=0)

print("=" * 70)
print("PHASE 3 IMPROVEMENT ANALYSIS")
print("=" * 70)

print("\nOverall Statistics:")
print(f"  Mean score per lineage: {df.mean(axis=1).mean():.4f}")
print(f"  Max score per lineage: {df.max(axis=1).mean():.4f}")
print(f"  Min score per lineage: {df.min(axis=1).mean():.4f}")
print(f"  Std dev per lineage: {df.std(axis=1).mean():.4f}")

print("\nFront Coverage (lineages with non-zero scores):")
print(f"{'Front':<30} {'Coverage':<15} {'Mean Score'}")
print("-" * 70)
for col in df.columns:
    count = (df[col] > 0).sum()
    mean_score = df[df[col] > 0][col].mean() if count > 0 else 0.0
    print(f"{col:<30} {count:>3}/99 ({100*count/99:>5.1f}%)   {mean_score:.4f}")

print("\nKey Improvements:")
print(f"  scalable_manufacturing: {(df['scalable_manufacturing'] > 0).sum()}/99 ({100*(df['scalable_manufacturing'] > 0).sum()/99:.1f}%)")
print(f"  large_area_modules: {(df['large_area_modules'] > 0).sum()}/99 ({100*(df['large_area_modules'] > 0).sum()/99:.1f}%)")

print("\n" + "=" * 70)
