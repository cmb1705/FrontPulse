#!/usr/bin/env python3
"""
Feature registry utilities for managing multisignal feature groups.

Loads `config/features/feature_groups.yaml` and provides helpers to resolve
grouped column sets, wildcard patterns, and curated feature lists for MSD
training, diagnostics, and ablation studies.
"""

from __future__ import annotations

import argparse
from fnmatch import fnmatch
from pathlib import Path
from typing import Iterable, List, Dict, Optional, Set

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "features" / "feature_groups.yaml"


def _deduplicate(sequence: Iterable[str]) -> List[str]:
    seen: Set[str] = set()
    ordered: List[str] = []
    for item in sequence:
        if item is None:
            continue
        if item not in seen:
            seen.add(item)
            ordered.append(item)
    return ordered


class FeatureRegistry:
    """Registry providing access to predefined feature groups and metadata."""

    def __init__(self, config_path: Path | str = DEFAULT_CONFIG):
        self.config_path = Path(config_path)
        if not self.config_path.exists():
            raise FileNotFoundError(f"Feature config not found: {self.config_path}")
        with self.config_path.open("r", encoding="utf-8") as fp:
            config = yaml.safe_load(fp) or {}
        self.group_defs: Dict[str, Dict] = config.get("groups", {})
        self.feature_metadata: Dict[str, Dict] = config.get("features", {})
        self.registry_metadata = config.get("metadata", {})
        self._group_cache: Dict[str, List[str]] = {}

    # ------------------------------------------------------------------
    # Group resolution helpers
    # ------------------------------------------------------------------
    def list_groups(self) -> List[str]:
        return sorted(self.group_defs.keys())

    def describe_group(self, name: str) -> Dict:
        return self.group_defs.get(name, {})

    def get_group(self, name: str) -> List[str]:
        if name not in self.group_defs:
            raise KeyError(f"Unknown feature group: {name}")
        if name in self._group_cache:
            return self._group_cache[name]

        definition = self.group_defs[name]
        collected: List[str] = []

        for included in definition.get("includes", []) or []:
            collected.extend(self.get_group(included))

        collected.extend(definition.get("columns", []) or [])
        resolved = _deduplicate(collected)
        self._group_cache[name] = resolved
        return resolved

    def available_features(self) -> List[str]:
        all_columns: List[str] = []
        for name in self.list_groups():
            all_columns.extend(self.get_group(name))
        deduped = _deduplicate(all_columns)
        if self.feature_metadata:
            extra = [feat for feat in self.feature_metadata if feat not in deduped]
            deduped.extend(extra)
        return deduped

    def list_features(self) -> List[str]:
        if self.feature_metadata:
            return sorted(self.feature_metadata.keys())
        return self.available_features()

    def describe_feature(self, name: str) -> Dict:
        return self.feature_metadata.get(name, {})

    # ------------------------------------------------------------------
    # Config resolution
    # ------------------------------------------------------------------
    def resolve_features(
        self,
        include_groups: Optional[Iterable[str]] = None,
        include_columns: Optional[Iterable[str]] = None,
        include_patterns: Optional[Iterable[str]] = None,
        exclude_groups: Optional[Iterable[str]] = None,
        exclude_columns: Optional[Iterable[str]] = None,
        exclude_patterns: Optional[Iterable[str]] = None,
    ) -> List[str]:
        """Resolve a curated feature list by combining groups, columns, and wildcard patterns."""
        include_groups = include_groups or []
        include_columns = include_columns or []
        include_patterns = include_patterns or []
        exclude_groups = exclude_groups or []
        exclude_columns = exclude_columns or []
        exclude_patterns = exclude_patterns or []

        selected: List[str] = []
        for group in include_groups:
            selected.extend(self.get_group(group))
        selected.extend(include_columns)

        if include_patterns:
            available = self.available_features()
            for pattern in include_patterns:
                matches = [col for col in available if fnmatch(col, pattern)]
                selected.extend(matches)

        selected = _deduplicate(selected)

        # Build exclusion set
        exclusions: Set[str] = set()
        for group in exclude_groups:
            exclusions.update(self.get_group(group))
        exclusions.update(exclude_columns)
        if exclude_patterns:
            available = self.available_features()
            for pattern in exclude_patterns:
                exclusions.update(col for col in available if fnmatch(col, pattern))

        filtered = [col for col in selected if col not in exclusions]
        return filtered

    # ------------------------------------------------------------------
    # CLI helpers
    # ------------------------------------------------------------------
    def to_dict(self) -> Dict[str, List[str]]:
        return {name: self.get_group(name) for name in self.list_groups()}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inspect multisignal feature groups.")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Path to feature_groups.yaml",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available group names and exit.",
    )
    parser.add_argument(
        "--show",
        metavar="GROUP",
        help="Print columns in a group.",
    )
    parser.add_argument(
        "--show-feature",
        metavar="FEATURE",
        help="Show metadata for a specific feature.",
    )
    parser.add_argument(
        "--resolve",
        metavar="GROUPS",
        nargs="+",
        help="Resolve combined feature list from named groups.",
    )
    parser.add_argument(
        "--include-pattern",
        action="append",
        dest="include_patterns",
        help="Wildcard pattern(s) to include (e.g., logistic_*).",
    )
    parser.add_argument(
        "--exclude-pattern",
        action="append",
        dest="exclude_patterns",
        help="Wildcard pattern(s) to exclude.",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    registry = FeatureRegistry(args.config)

    if args.list:
        for name in registry.list_groups():
            desc = registry.describe_group(name).get("description", "")
            print(f"{name:20s} {desc}")
        return

    if args.show:
        columns = registry.get_group(args.show)
        print("\n".join(columns))
        return

    if args.show_feature:
        meta = registry.describe_feature(args.show_feature)
        if not meta:
            raise SystemExit(f"No metadata recorded for feature: {args.show_feature}")
        for key, value in meta.items():
            if isinstance(value, list):
                print(f"{key}:")
                for item in value:
                    print(f"  - {item}")
            else:
                print(f"{key}: {value}")
        return

    if args.resolve:
        columns = registry.resolve_features(
            include_groups=args.resolve,
            include_patterns=args.include_patterns,
            exclude_patterns=args.exclude_patterns,
        )
        print("\n".join(columns))
        return

    # Default: show summary
    print(f"Loaded {len(registry.list_groups())} groups from {registry.config_path}")
    print(f"Total unique features tracked: {len(registry.available_features())}")


if __name__ == "__main__":
    main()
