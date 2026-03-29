from __future__ import annotations

import argparse
from collections.abc import Iterable
from pathlib import Path

import networkx as nx
from _path_bootstrap import ensure_repo_imports

REPO_ROOT = ensure_repo_imports()

from src.graph_lite import write_lite_graph  # noqa: E402
from src.trusted_io import load_trusted_pickle  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert NetworkX graph pickles into lightweight .lite.npz archives. "
                    "Publication dates and derived fields (pub_qtr, pub_year) are now included by default "
                    "to support downstream community analysis and new-work counting."
    )
    parser.add_argument("--graphs-dir", type=Path, default=Path("data/current_graphs"),
                        help="Directory containing citation_graph_* graph pickles.")
    parser.add_argument("--pattern", default="citation_graph_*_*.pkl",
                        help="Glob pattern for graph pickles to convert.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing lite files.")
    parser.add_argument("--skip-publication-dates", action="store_true",
                        help="Omit publication dates/metadata (NOT recommended - breaks downstream analysis)")
    parser.add_argument(
        "--allow-external-pickle",
        action="store_true",
        help="Allow loading pickle graphs from outside the repository root.",
    )
    return parser.parse_args()


def iter_graph_pickles(graphs_dir: Path, pattern: str) -> Iterable[Path]:
    return sorted(graphs_dir.glob(pattern))


def build_lite_for_file(
    pickle_path: Path,
    *,
    include_publication_dates: bool,
    force: bool,
    allow_external_pickle: bool,
) -> None:
    base = pickle_path.stem
    lite_path = pickle_path.with_name(f"{base}.lite.npz")
    if lite_path.exists() and not force:
        return
    G: nx.DiGraph = load_trusted_pickle(
        pickle_path,
        description="Lite-graph source pickle",
        allow_external=allow_external_pickle,
    )
    write_lite_graph(G, lite_path=lite_path, include_publication_dates=include_publication_dates)
    del G


def main() -> None:
    args = parse_args()
    pickles = list(iter_graph_pickles(args.graphs_dir, args.pattern))
    if not pickles:
        print(f"[Lite] No pickles found matching {args.pattern} in {args.graphs_dir}")
        return

    # Publication dates are now included by default (required for communities.py)
    include_dates = not args.skip_publication_dates
    if args.skip_publication_dates:
        print("[Lite] WARNING: Skipping publication dates. Downstream analysis may fail.")

    total = len(pickles)
    for idx, path in enumerate(pickles, start=1):
        try:
            build_lite_for_file(
                path,
                include_publication_dates=include_dates,
                force=args.force,
                allow_external_pickle=args.allow_external_pickle,
            )
        except Exception as exc:
            print(f"[Lite] Failed for {path}: {exc}")
        else:
            print(f"[Lite] {idx}/{total} -> {path.stem}.lite.npz")


if __name__ == "__main__":
    main()
