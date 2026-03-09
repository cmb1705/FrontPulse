"""Slice validation utilities for quarterly parquet files.

Validates temporal slices (by_quarter__YYYYQN.parquet) to ensure:
- All expected columns are present
- work_id values are unique and non-null
- publication_date values fall within the quarter window
- ref_count matches actual referenced_works length
- No data corruption or format issues

Can be run standalone via CLI or imported for programmatic validation.

Usage:
    python src/validate_slices.py "data/current_ingest/slices/by_quarter__*.parquet" \\
        --json data/out/slice_validation.json --strict
"""

from __future__ import annotations
import argparse, json, re, sys
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd
import numpy as np

try:
    import pyarrow as pa
except Exception:
    pa = None

EXPECTED_COLS = [
    "work_id","title","publication_date","publication_year","type","language",
    "cited_by_count","referenced_works","ref_count","pub_qtr"
]

def quarter_bounds(label: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Convert quarter label to start/end timestamps.

    Args:
        label: Quarter string in format "YYYYQN" (e.g., "2020Q3").

    Returns:
        Tuple of (start_timestamp, end_timestamp) for the quarter.

    Examples:
        >>> quarter_bounds("2020Q1")
        (Timestamp('2020-01-01 00:00:00'), Timestamp('2020-03-31 00:00:00'))
    """
    p = pd.Period(label, freq="Q")
    return (p.start_time.normalize(), p.end_time.normalize())

def safe_len(x) -> int:
    """Safely compute length of various types including PyArrow scalars.

    Handles multiple input types from parquet columns: None, NaN, lists, tuples,
    sets, numpy arrays, pandas Series, PyArrow ListScalars, comma-separated strings.
    Returns 0 for null/empty values.

    This function is critical for validating referenced_works columns which may
    be stored in various formats depending on parquet engine and schema.

    Args:
        x: Value to measure (any type).

    Returns:
        Integer length. Returns 0 for None, NaN, or empty collections.

    Examples:
        >>> safe_len([1, 2, 3])
        3
        >>> safe_len(None)
        0
        >>> safe_len("A,B,C")
        3
        >>> safe_len("[item1, item2]")
        2
    """
    if x is None:
        return 0
    if isinstance(x, (list, tuple, set)):
        return len(x)
    if pa is not None:
        try:
            import pyarrow.lib as _palib
            if isinstance(x, (_palib.ListScalar, _palib.ListValue)):
                v = x.as_py()
                return 0 if v is None else len(v)
        except Exception:
            try:
                if hasattr(x, "as_py"):
                    v = x.as_py()
                    return 0 if v is None else len(v)
            except Exception:
                pass
    if isinstance(x, np.ndarray):
        return int(x.size)
    if isinstance(x, pd.Series):
        return int(len(x))
    if hasattr(x, "__len__") and not isinstance(x, (str, bytes)):
        try:
            return len(x)
        except Exception:
            pass
    if isinstance(x, str):
        s = x.strip()
        if not s:
            return 0
        if s.startswith("[") and s.endswith("]"):
            inner = s[1:-1].strip()
            return 0 if not inner else len([t for t in inner.split(",") if t.strip()])
        return len([t for t in s.split(",") if t.strip()])
    try:
        if pd.isna(x):
            return 0
    except Exception:
        pass
    return 0

def validate_file(path: Path, limit: int = 25) -> Dict[str, Any]:
    """Validate a single parquet slice file and return comprehensive diagnostics.

    Performs multiple validation checks:
    1. Schema validation (expected columns present)
    2. Quarter window validation (dates fall within labeled quarter)
    3. work_id validation (non-null, unique)
    4. ref_count consistency (matches referenced_works length)
    5. Language distribution analysis

    Returns detailed diagnostics including anomaly counts and sample records
    for investigation.

    Args:
        path: Path to parquet file (should match pattern by_quarter__YYYYQN.parquet).
        limit: Maximum number of sample anomaly rows to include in report (default: 25).

    Returns:
        Dictionary with validation results containing:
        - file: Path string
        - rows: Total row count
        - missing_cols: List of expected columns not found
        - quarter_check: Quarter window validation stats
        - null_work_id: Count of rows with null work_id
        - dup_work_id: Count of duplicate work_id values
        - ref_mismatch_count: Count of ref_count mismatches
        - sample_*: Sample rows for each anomaly type (up to limit)
        - language_distribution: Dictionary of language -> count
        - anomaly_flags: Boolean flags for each anomaly type

    Examples:
        >>> report = validate_file(Path("data/slices/by_quarter__2020Q1.parquet"))
        >>> report['rows']
        1523
        >>> report['anomaly_flags']
        {'missing_cols': False, 'out_of_window': 0, 'null_work_id': 0, ...}
    """
    m = re.search(r"by_quarter__([0-9]{4}Q[1-4])\.parquet$", path.name)
    qlabel = m.group(1) if m else None

    df = pd.read_parquet(path, engine="pyarrow")
    df.columns = df.columns.str.strip().str.replace(" ","_").str.replace("-","_").str.lower()
    if "publication_date" in df.columns:
        df["publication_date"] = pd.to_datetime(df["publication_date"], errors="coerce")
    else:
        df["publication_date"] = pd.NaT

    missing_cols = [c for c in EXPECTED_COLS if c not in df.columns]

    out_of_range = pd.DataFrame()
    qcheck_summary = None
    if qlabel:
        start, end = quarter_bounds(qlabel)
        mask = df["publication_date"].between(start, end, inclusive="both")
        out_of_range = df.loc[~mask, ["work_id","publication_date"]].head(limit)
        qcheck_summary = {
            "quarter": qlabel,
            "start": str(start.date()),
            "end": str(end.date()),
            "rows": int(len(df)),
            "in_window": int(mask.sum()),
            "out_of_window": int((~mask).sum())
        }

    has_wid = "work_id" in df.columns
    wid_null = (df["work_id"].isna() if has_wid else pd.Series([True]*len(df)))
    wid_dups = (df["work_id"].duplicated() if has_wid else pd.Series([False]*len(df)))
    null_ids = df.loc[wid_null, ["title","publication_date"]].head(limit)
    dup_ids = df.loc[wid_dups, ["work_id","publication_date"]].head(limit)

    ref_mismatch = pd.DataFrame()
    if "referenced_works" in df.columns:
        calc_len = df["referenced_works"].apply(safe_len)
        df["_calc_ref_len"] = calc_len
        if "ref_count" in df.columns:
            mismatch_mask = df["ref_count"].fillna(-1).astype("int64") != calc_len.astype("int64")
            ref_mismatch = df.loc[mismatch_mask, ["work_id","ref_count","_calc_ref_len"]].head(limit)

    lang_counts = df["language"].value_counts(dropna=False).to_dict() if "language" in df.columns else {}

    # summarize anomalies
    anomalies = {
        "missing_cols": bool(missing_cols),
        "out_of_window": int(qcheck_summary["out_of_window"]) if qcheck_summary else 0,
        "null_work_id": int(wid_null.sum()),
        "dup_work_id": int(wid_dups.sum()),
        "ref_mismatch": int(len(ref_mismatch))
    }

    return {
        "file": str(path),
        "rows": int(len(df)),
        "missing_cols": missing_cols,
        "quarter_check": qcheck_summary,
        "null_work_id": anomalies["null_work_id"],
        "dup_work_id": anomalies["dup_work_id"],
        "sample_out_of_window": out_of_range.to_dict(orient="records"),
        "sample_null_work_id": null_ids.to_dict(orient="records"),
        "sample_dup_work_id": dup_ids.to_dict(orient="records"),
        "ref_mismatch_count": anomalies["ref_mismatch"],
        "sample_ref_mismatch": ref_mismatch.to_dict(orient="records"),
        "language_distribution": lang_counts,
        "anomaly_flags": anomalies
    }

def print_report(r: Dict[str, Any], limit: int):
    """Print human-readable validation report to console.

    Formats validation results from validate_file() into a readable console
    report with summary statistics and sample anomaly records.

    Args:
        r: Validation report dictionary from validate_file().
        limit: Maximum number of sample rows to display per anomaly type.

    Returns:
        None. Output is printed to stdout.
    """
    qc = r.get("quarter_check") or {}
    print(f"\n== {Path(r['file']).name} ==")
    print(f"rows={r['rows']}  missing_cols={r['missing_cols']}")
    if qc:
        print(f"quarter={qc['quarter']}  window={qc['start']}..{qc['end']}  "
              f"in={qc['in_window']}  out={qc['out_of_window']}")
    print(f"null_work_id={r['null_work_id']}  dup_work_id={r['dup_work_id']}  "
          f"ref_mismatch={r['ref_mismatch_count']}")

    if r["missing_cols"]:
        print("  >> Missing columns detected. Downstream code may fail.")
    if r["sample_out_of_window"]:
        print(f"  >> Out-of-window sample (up to {limit}):")
        for row in r["sample_out_of_window"]:
            print(f"     {row}")
    if r["sample_null_work_id"]:
        print(f"  >> Null work_id sample (up to {limit}):")
        for row in r["sample_null_work_id"]:
            print(f"     {row}")
    if r["sample_dup_work_id"]:
        print(f"  >> Duplicate work_id sample (up to {limit}):")
        for row in r["sample_dup_work_id"]:
            print(f"     {row}")
    if r["sample_ref_mismatch"]:
        print(f"  >> ref_count vs referenced_works length mismatches (up to {limit}):")
        for row in r["sample_ref_mismatch"]:
            print(f"     {row}")

def main():
    """CLI entry point for slice validation.

    Validates one or more parquet slice files and prints reports to console.
    Optionally writes JSON report and exits with non-zero status on anomalies.

    Command-line Arguments:
        paths: One or more file paths or glob patterns (e.g., "data/slices/*.parquet")
        --limit INT: Maximum anomaly rows to show per file (default: 25)
        --json PATH: Write JSON validation report to file
        --strict: Exit with status 1 if any anomalies found (useful for CI/CD)

    Returns:
        Exit code 0 on success, 1 if anomalies found (with --strict), 2 if no files matched.

    Examples:
        $ python src/validate_slices.py "data/slices/by_quarter__*.parquet"
        $ python src/validate_slices.py "data/slices/*.parquet" --json report.json --strict
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+", help="Parquet files or globs (e.g., data/out/by_quarter__*.parquet)")
    ap.add_argument("--limit", type=int, default=25, help="Max anomaly rows to show per file")
    ap.add_argument("--json", type=Path, default=None, help="Optional JSON report path")
    ap.add_argument("--strict", action="store_true",
                    help="Exit non-zero if any anomalies are found")
    args = ap.parse_args()

    files: List[Path] = []
    for p in args.paths:
        files.extend(sorted(Path().glob(p)))
    if not files:
        print("No files matched.", file=sys.stderr)
        sys.exit(2)

    report: List[Dict[str, Any]] = []
    any_anomaly = False

    for f in files:
        try:
            r = validate_file(f, limit=args.limit)
            report.append(r)
            print_report(r, args.limit)
            a = r["anomaly_flags"]
            if a["missing_cols"] or a["out_of_window"] or a["null_work_id"] or a["dup_work_id"] or a["ref_mismatch"]:
                any_anomaly = True
        except Exception as e:
            any_anomaly = True
            print(f"\n== {f.name} ==\nERROR: {e}", file=sys.stderr)
            # continue to next file

    if args.json:
        args.json.write_text(json.dumps(report, indent=2))
        print(f"\nWrote JSON report to {args.json}")

    # Continue process regardless. Only affect exit code if --strict.
    if args.strict and any_anomaly:
        sys.exit(1)
    sys.exit(0)

if __name__ == "__main__":
    # Tip: pip install pandas pyarrow
    main()
