# ref_resolution.py
import pandas as pd
from pathlib import Path
import numpy as np
try:
    import pyarrow as pa
except Exception:
    pa = None

# 1) Build a corpus set of all work_ids (fast: from manifest or glob)
all_ids = set()
for fp in Path("data/out").glob("by_quarter__*.parquet"):
    df = pd.read_parquet(fp, engine="pyarrow", columns=["work_id"])
    all_ids.update(df["work_id"].dropna().astype(str))

def safe_len(x):
    if x is None: return 0
    if isinstance(x, list): return len(x)
    if isinstance(x, str):
        s = x.strip()
        if s.startswith("[") and s.endswith("]"):
            inner = [t.strip() for t in s[1:-1].split(",") if t.strip()]
            return len(inner)
        return len([t for t in s.split(",") if t.strip()])
    return 0

def iter_refs(x):
    if x is None:
        return []
    # PyArrow ListScalar from parquet
    if pa is not None and isinstance(x, pa.lib.ListScalar):
        v = x.as_py()            # -> Python list or None
        return [] if v is None else [str(t).split("/")[-1] for t in v]
    # NumPy arrays
    if isinstance(x, np.ndarray):
        return [str(t).split("/")[-1] for t in x.tolist()]
    # Native Python list
    if isinstance(x, list):
        return [str(t).split("/")[-1] for t in x]
    # String fallbacks (comma- or bracketed)
    if isinstance(x, str):
        s = x.strip()
        if not s:
            return []
        if s.startswith("[") and s.endswith("]"):
            inner = [t.strip(" '\"") for t in s[1:-1].split(",") if t.strip()]
            return [i.split("/")[-1] for i in inner]
        return [t.strip().split("/")[-1] for t in s.split(",") if t.strip()]
    return []

rows = []
for fp in sorted(Path("data/out").glob("by_quarter__*.parquet")):
    q = fp.stem.split("__")[1]
    df = pd.read_parquet(fp, engine="pyarrow", columns=["work_id","referenced_works"])
    # optional: ensure Python objects not Arrow scalars inside series
    # df = df.copy()  # (not required if iter_refs handles ListScalar)
    df["refs"] = df["referenced_works"].apply(iter_refs)
    tot_refs = int(df["refs"].apply(len).sum())
    in_corpus = int(df["refs"].apply(lambda L: sum(1 for r in L if r in all_ids)).sum())
    rows.append({"quarter": q, "rows": len(df), "total_refs": tot_refs,
                 "in_corpus_refs": in_corpus,
                 "in_corpus_share": (in_corpus / tot_refs) if tot_refs else 0.0})
rate = pd.DataFrame(rows).sort_values("quarter")
print(rate.tail(8))
rate.to_csv("data/out/ref_resolution_by_quarter.csv", index=False)