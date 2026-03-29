#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections.abc import Iterable
from pathlib import Path

from _path_bootstrap import ensure_repo_imports

REPO = ensure_repo_imports()

from src.graph_build import save_graph  # type: ignore  # noqa: E402
from src.trusted_io import load_trusted_pickle  # noqa: E402


def _iter_graphs(paths: Iterable[Path]) -> list[Path]:
    expanded: list[Path] = []
    for path in paths:
        if path.is_dir():
            expanded.extend(sorted(path.glob("*.pkl")))
        elif path.is_file():
            expanded.append(path)
        else:
            for matched in sorted(Path().glob(str(path))):
                if matched.is_file():
                    expanded.append(matched)
    return sorted({p.resolve() for p in expanded})


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Re-export GraphML files from existing NetworkX pickle graphs."
    )
    parser.add_argument(
        "graphs",
        nargs="*",
        help="Pickle paths or glob patterns (default: data/current_graphs/citation_graph_cumulative_*.pkl)",
    )
    parser.add_argument(
        "--graphs-dir",
        type=Path,
        default=Path("data/current_graphs"),
        help="Directory containing graph pickles (used when no graphs argument supplied).",
    )
    parser.add_argument(
        "--compression",
        choices=["none", "gzip"],
        default="none",
        help="GraphML compression mode (default: none).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing GraphML files.",
    )
    parser.add_argument(
        "--allow-external-pickle",
        action="store_true",
        help="Allow loading pickle graphs from outside the repository root.",
    )
    args = parser.parse_args(argv)

    raw_paths: list[Path]
    if args.graphs:
        raw_paths = [Path(p) for p in args.graphs]
    else:
        raw_paths = [args.graphs_dir / "citation_graph_cumulative_*.pkl"]

    graphs = _iter_graphs(raw_paths)
    if not graphs:
        print("[Export] No graph pickles found.")
        return

    compression = None if args.compression == "none" else args.compression

    for pkl_path in graphs:
        base = pkl_path.with_suffix("")
        graphml_path = base.with_suffix(".graphml")
        graphml_gz = base.with_suffix(".graphml.gz")
        if not args.force:
            if compression == "gzip" and graphml_gz.exists():
                print(f"[Export] {graphml_gz.name} exists; skipping.")
                continue
            if compression != "gzip" and graphml_path.exists():
                print(f"[Export] {graphml_path.name} exists; skipping.")
                continue
        print(f"[Export] {pkl_path.name} -> GraphML ({args.compression})")
        G = load_trusted_pickle(
            pkl_path,
            description="Graph export pickle",
            allow_external=args.allow_external_pickle,
        )
        save_graph(
            G,
            base,
            write_pickle=False,
            write_graphml=True,
            graphml_compression=compression,
        )


if __name__ == "__main__":
    main()
