"""
Extract titles and abstracts from raw OpenAlex JSONL for use in semantic analysis.

This utility provides fast access to text data for works using the byte-offset
indexed raw JSONL files, reconstructing abstracts from OpenAlex's inverted index format.

Usage:
    from scripts.extract_abstracts import AbstractExtractor

    extractor = AbstractExtractor('data/current_ingest/raw')
    text = extractor.get_text('W2057327293')  # Returns "Title. Abstract text..."

    # Batch extraction for a lineage
    work_ids = ['W123', 'W456', 'W789']
    texts = extractor.get_texts_batch(work_ids)

The extractor now supports an on-disk cache of the global work/DOI index. The first
load serializes the lookup tables to ``data/out/cache_lineage/abstract_index_*.pkl``;
subsequent loads (including Stage 4 worker processes) reuse the cache, eliminating the
need to rebuild the index for every process.
"""

from __future__ import annotations
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict

from src.raw_store import RawStore
from src import trusted_io

_CACHE_VERSION = 1


def _default_cache_path(raw_dir: Path) -> Path:
    """Return a deterministic cache path for a given raw directory."""
    cache_dir = Path("data/out/cache_lineage")
    digest = hashlib.md5(str(raw_dir.resolve()).encode("utf-8")).hexdigest()[:8]
    slug = raw_dir.resolve().name
    return cache_dir / f"abstract_index_{slug}_{digest}.pkl"


class AbstractExtractor:
    """
    Fast text extraction from raw OpenAlex JSONL using byte-offset indexes.

    Handles inverted index reconstruction and provides batch operations for
    efficient lineage-level text aggregation.
    """

    def __init__(self, raw_dir: Path | str, cache_path: Optional[Path | str] = None):
        """
        Initialize extractor with path to raw JSONL directory.

        Args:
            raw_dir: Directory containing openalex_raw_*.jsonl and *_index.csv files
            cache_path: Optional path for serialized abstract index cache (auto-generated if None)
        """
        self.raw_dir = Path(raw_dir)
        if cache_path is None:
            self.cache_path = _default_cache_path(self.raw_dir)
        else:
            self.cache_path = Path(cache_path)
        self._stores: Dict[str, RawStore] = {}
        self._load_stores()

    def _load_stores(self):
        """Load all raw stores from the directory."""
        # Find all JSONL files
        jsonl_files = sorted(self.raw_dir.glob('openalex_raw_*_part*.jsonl'))

        if not jsonl_files:
            raise FileNotFoundError(f"No raw JSONL files found in {self.raw_dir}")

        print(f"[AbstractExtractor] Loading {len(jsonl_files)} raw store indexes...")

        for jsonl_path in jsonl_files:
            # Extract base path (without .jsonl extension)
            base_path = jsonl_path.with_suffix('')
            try:
                store = RawStore.from_basepath(base_path)
                # Store by part number for quick lookup
                part_num = jsonl_path.stem.split('_part')[-1]
                self._stores[part_num] = store
                print(f"  Loaded part {part_num}: {len(store._index)} records")
            except Exception as e:
                print(f"  Warning: Could not load {jsonl_path.name}: {e}")

        manifest = self._build_manifest(jsonl_files)
        cache_loaded = self._try_load_cache(manifest)

        if not cache_loaded:
            print(f"[AbstractExtractor] Building global index (cache miss)...")
            self._work_to_store = {}
            self._doi_to_work = {}  # DOI → work_id mapping

            for store_id, store in self._stores.items():
                for work_id in store._index.keys():
                    self._work_to_store[work_id] = store_id

                    # Also build DOI → work_id mapping
                    work = store.get_json(work_id)
                    if work:
                        doi = work.get('doi')
                        if doi:
                            # Normalize DOI (remove https://doi.org/ prefix if present)
                            doi_clean = doi.replace('https://doi.org/', '').lower()
                            self._doi_to_work[doi_clean] = work_id

            if self.cache_path:
                self._save_cache(manifest)

        total_records = sum(len(s._index) for s in self._stores.values())
        print(f"[AbstractExtractor] Ready: {total_records} total records, {len(self._work_to_store)} indexed")
        print(f"[AbstractExtractor] DOI index: {len(self._doi_to_work)} DOIs mapped")

    def _build_manifest(self, jsonl_files: List[Path]) -> Dict[str, Tuple[float, int]]:
        manifest: Dict[str, Tuple[float, int]] = {}
        for jsonl_path in jsonl_files:
            base = jsonl_path.resolve()
            index_path = jsonl_path.with_name(jsonl_path.stem + '_index.csv').resolve()
            for candidate in (base, index_path):
                if candidate.exists():
                    stat = candidate.stat()
                    manifest[str(candidate)] = (stat.st_mtime, stat.st_size)
        return manifest

    def _try_load_cache(self, manifest: Dict[str, Tuple[float, int]]) -> bool:
        if not self.cache_path:
            return False
        try:
            data: Dict[str, Any] = trusted_io.load_trusted_pickle(
                self.cache_path, description="abstract index cache",
            )
        except FileNotFoundError:
            return False
        except Exception as exc:
            print(f"[AbstractExtractor] Warning: failed to load cache {self.cache_path}: {exc}")
            return False

        if data.get('version') != _CACHE_VERSION:
            return False

        cached_dir = data.get('raw_dir')
        if cached_dir != str(self.raw_dir.resolve()):
            return False

        if data.get('files') != manifest:
            return False

        work_to_store = data.get('work_to_store')
        doi_to_work = data.get('doi_to_work')
        if not isinstance(work_to_store, dict) or not isinstance(doi_to_work, dict):
            return False

        self._work_to_store = work_to_store
        self._doi_to_work = doi_to_work
        print(f"[AbstractExtractor] Loaded cached index from {self.cache_path}")
        return True

    def _save_cache(self, manifest: Dict[str, Tuple[float, int]]) -> None:
        if not self.cache_path:
            return
        try:
            payload = {
                'version': _CACHE_VERSION,
                'raw_dir': str(self.raw_dir.resolve()),
                'files': manifest,
                'work_to_store': self._work_to_store,
                'doi_to_work': self._doi_to_work,
            }
            trusted_io.save_trusted_pickle(
                payload, self.cache_path,
                description="abstract index cache",
            )
            print(f"[AbstractExtractor] Cached index written to {self.cache_path}")
        except Exception as exc:
            print(f"[AbstractExtractor] Warning: failed to write cache {self.cache_path}: {exc}")

    def _find_work(self, work_id: str) -> Optional[Dict]:
        """
        Find a work across all raw stores (O(1) lookup via global index).

        Args:
            work_id: OpenAlex work ID (e.g., "W2057327293")

        Returns:
            Work dictionary if found, None otherwise
        """
        # Use global index for O(1) lookup instead of O(n) search
        store_id = self._work_to_store.get(work_id)
        if store_id is None:
            return None

        store = self._stores.get(store_id)
        if store is None:
            return None

        return store.get_json(work_id)

    @staticmethod
    def reconstruct_abstract(inverted_index: Dict[str, List[int]]) -> str:
        """
        Reconstruct abstract text from OpenAlex inverted index format.

        Args:
            inverted_index: Dict mapping words to position lists
                Example: {"The": [0, 15], "cell": [3], "efficiency": [5, 20]}

        Returns:
            Reconstructed abstract text
        """
        if not inverted_index:
            return ""

        # Find the maximum position to size the array
        max_pos = max(max(positions) for positions in inverted_index.values() if positions)

        # Create array of words at each position
        words = [''] * (max_pos + 1)

        for word, positions in inverted_index.items():
            for pos in positions:
                words[pos] = word

        # Join words with spaces
        return ' '.join(words)

    def get_text(self, work_id: str, include_title: bool = True) -> Optional[str]:
        """
        Get combined title + abstract text for a work.

        Args:
            work_id: OpenAlex work ID (e.g., "W2057327293")
            include_title: Whether to include title (default True)

        Returns:
            "Title. Abstract text..." or None if work not found
        """
        work = self._find_work(work_id)

        if not work:
            return None

        title = work.get('display_name', '')
        abstract_index = work.get('abstract_inverted_index', {})

        parts = []
        if include_title and title:
            parts.append(title)

        if abstract_index:
            abstract = self.reconstruct_abstract(abstract_index)
            if abstract:
                parts.append(abstract)

        return '. '.join(parts) if parts else None

    def get_text_by_doi(self, doi: str, include_title: bool = True) -> Optional[str]:
        """
        Get combined title + abstract text for a work by DOI.

        Args:
            doi: DOI (e.g., "10.1021/ja809598r" or "https://doi.org/10.1021/ja809598r")
            include_title: Whether to include title (default True)

        Returns:
            "Title. Abstract text..." or None if work not found
        """
        # Normalize DOI (remove https://doi.org/ prefix if present, lowercase)
        doi_clean = doi.replace('https://doi.org/', '').lower()

        # Look up work_id from DOI
        work_id = self._doi_to_work.get(doi_clean)
        if not work_id:
            return None

        # Use existing get_text method
        return self.get_text(work_id, include_title=include_title)

    def get_texts_batch(
        self,
        work_ids: List[str],
        include_title: bool = True,
        verbose: bool = False
    ) -> Dict[str, str]:
        """
        Batch extraction for multiple works (optimized for lineage processing).

        Args:
            work_ids: List of OpenAlex work IDs
            include_title: Whether to include titles (default True)
            verbose: Print progress (default False)

        Returns:
            Dict mapping work_id -> text (only includes works with text)
        """
        texts = {}
        missing = []

        for i, work_id in enumerate(work_ids):
            if verbose and (i + 1) % 100 == 0:
                print(f"  Processed {i + 1}/{len(work_ids)} works...")

            text = self.get_text(work_id, include_title=include_title)
            if text:
                texts[work_id] = text
            else:
                missing.append(work_id)

        if verbose:
            print(f"  Extracted text for {len(texts)}/{len(work_ids)} works")
            if missing:
                print(f"  Missing text for {len(missing)} works (no abstract or not found)")

        return texts

    def get_metadata(self, work_id: str) -> Optional[Dict]:
        """
        Get metadata for a work (without reconstructing abstract).

        Useful for getting publication dates, citations, concepts, etc.

        Args:
            work_id: OpenAlex work ID

        Returns:
            Dictionary with selected metadata fields, or None if not found
        """
        work = self._find_work(work_id)

        if not work:
            return None

        return {
            'work_id': work.get('id'),
            'title': work.get('display_name'),
            'publication_year': work.get('publication_year'),
            'publication_date': work.get('publication_date'),
            'cited_by_count': work.get('cited_by_count'),
            'has_abstract': bool(work.get('abstract_inverted_index')),
            'keywords': [kw.get('display_name') for kw in work.get('keywords', [])],
            'concepts': [
                {'name': c.get('display_name'), 'score': c.get('score')}
                for c in work.get('concepts', [])
            ][:5],  # Top 5 concepts
            'topics': [
                {'name': t.get('display_name'), 'score': t.get('score')}
                for t in work.get('topics', [])
            ][:3],  # Top 3 topics
        }

    def close(self):
        """Close all raw stores."""
        for store in self._stores.values():
            store.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def main():
    """Demo usage of AbstractExtractor."""
    import sys

    if len(sys.argv) < 2:
        print("Usage: python scripts/extract_abstracts.py <work_id> [work_id2 ...]")
        print("\nExample:")
        print("  python scripts/extract_abstracts.py W2057327293")
        print("  python scripts/extract_abstracts.py W2057327293 W2741809807")
        sys.exit(1)

    work_ids = sys.argv[1:]

    with AbstractExtractor('data/current_ingest/raw') as extractor:
        for work_id in work_ids:
            print(f"\n{'='*70}")
            print(f"Work ID: {work_id}")
            print(f"{'='*70}")

            # Get metadata
            metadata = extractor.get_metadata(work_id)
            if metadata:
                print(f"\nTitle: {metadata['title']}")
                print(f"Year: {metadata['publication_year']}")
                print(f"Citations: {metadata['cited_by_count']}")
                print(f"Has abstract: {metadata['has_abstract']}")
                if metadata['keywords']:
                    print(f"Keywords: {', '.join(metadata['keywords'][:5])}")

            # Get full text
            text = extractor.get_text(work_id)
            if text:
                print(f"\nFull text ({len(text)} characters):")
                print(text[:500] + "..." if len(text) > 500 else text)
            else:
                print("\nNo text available (work not found or no abstract)")


if __name__ == "__main__":
    main()
