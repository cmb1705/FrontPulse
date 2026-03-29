# scripts/report_communities.py  (PowerShell-safe; run from repo root)
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def loadj(p: Path) -> dict:
    return json.loads(p.read_text()) if p.exists() else {}

def summarize_annual(annual: list[dict], overlap_min: int):
    rows = []
    all_events = []
    for e in annual:
        y = e.get("year")
        ncom = e.get("n_communities", 0)
        align = e.get("alignment") or {}
        matches = align.get("matches") or []  # list of [prev, curr, overlap]
        VI = align.get("VI")

        # Build maps for split/merge counts using all overlaps >= threshold
        prev_to_curr = defaultdict(list)
        curr_to_prev = defaultdict(list)
        for a, b, ov in matches:
            if ov >= overlap_min:
                prev_to_curr[a].append((b, ov))
                curr_to_prev[b].append((a, ov))

        splits = sum(1 for a, lst in prev_to_curr.items() if len({b for b, _ in lst}) > 1)
        merges = sum(1 for b, lst in curr_to_prev.items() if len({a for a, _ in lst}) > 1)

        # Collect event detail (optional dump)
        for a, lst in prev_to_curr.items():
            if len({b for b, _ in lst}) > 1:
                all_events.append({"year": y, "type": "split", "prev": int(a),
                                   "to": [{"curr": int(b), "overlap": int(ov)} for b, ov in sorted(lst, key=lambda x: -x[1])]})
        for b, lst in curr_to_prev.items():
            if len({a for a, _ in lst}) > 1:
                all_events.append({"year": y, "type": "merge", "curr": int(b),
                                   "from": [{"prev": int(a), "overlap": int(ov)} for a, ov in sorted(lst, key=lambda x: -x[1])]})

        rows.append({"year": y, "n_communities": ncom, "VI": VI, "splits": splits, "merges": merges})
    df = pd.DataFrame(rows).sort_values("year")
    return df, all_events

def summarize_delta(delta: list[dict]):
    rows = []
    for e in delta:
        vi = e.get("VI_vs_prev_quarter", e.get("VI_vs_ref_annual", e.get("VI_vs_latest_annual")))
        rows.append({
            "quarter": e.get("quarter"),
            "prev_quarter": e.get("prev_quarter"),
            "n_communities": e.get("n_communities", 0),
            "VI_vs_prev_quarter": vi,
            "ref_year": e.get("ref_year"),
        })
    df = pd.DataFrame(rows).sort_values("quarter")
    return df

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--annual", type=Path, default=Path("data/out/communities_annual.json"))
    ap.add_argument("--delta", type=Path,  default=Path("data/out/communities_delta.json"))
    ap.add_argument("--out-csv", type=Path, default=Path("data/out/community_report.csv"))
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--overlap-min", type=int, default=10, help="Min core-overlap to count link in split/merge")
    ap.add_argument("--list-events", action="store_true", help="Write detailed split/merge events to JSON")
    args = ap.parse_args()

    ca = loadj(args.annual); cd = loadj(args.delta)
    df_a, events = summarize_annual(ca.get("annual", []), overlap_min=args.overlap_min)
    df_d = summarize_delta(cd.get("delta", []))

    # Write CSV with both sections
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", encoding="utf-8") as f:
        f.write("# Annual summary\n"); df_a.to_csv(f, index=False)
        f.write("\n# Delta summary\n");  df_d.to_csv(f, index=False)
    print(f"Wrote {args.out_csv}")

    if args.list_events:
        ev_path = args.out_csv.parent / "communities_events.json"
        ev_path.write_text(json.dumps(events, indent=2))
        print(f"Wrote {ev_path}")

    if args.plot and not df_a.empty:
        plt.figure()
        plt.plot(df_a["year"], df_a["VI"])
        plt.title("Annual VI (lower = more stable)")
        plt.xlabel("Year"); plt.ylabel("VI")
        plt.tight_layout()
        plt.savefig(str(args.out_csv.parent / "annual_vi.png"), dpi=150)
        print(f"Saved {args.out_csv.parent/'annual_vi.png'}")

if __name__ == "__main__":
    main()
