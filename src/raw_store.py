"""Raw NDJSON storage with byte-offset indexing for efficient random access."""
from __future__ import annotations

import csv
import gzip
import json
import pathlib
import shutil
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple


@dataclass(frozen=True)
class IndexEntry:
    work_id: Optional[str]
    offset: int
    length: int


class RawStore:
    """
    Lightweight accessor for NDJSON + byte-offset index archives.

    Usage:
        store = RawStore.from_basepath(Path(\"data/.../openalex_raw_part0000\"))
        record = store.get_json(\"W123\")
    """

    def __init__(self, ndjson_path: pathlib.Path, index: Dict[str, IndexEntry]) -> None:
        self._path = ndjson_path
        self._index = index
        self._fh = ndjson_path.open("rb")

    def __enter__(self) -> RawStore:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    @classmethod
    def from_basepath(cls, base: pathlib.Path) -> RawStore:
        ndjson_path, index_path = cls._resolve_paths(base)
        index = cls._load_index(index_path)
        return cls(ndjson_path, index)

    @classmethod
    def from_paths(cls, ndjson_path: pathlib.Path, index_path: pathlib.Path) -> RawStore:
        index = cls._load_index(index_path)
        return cls(ndjson_path, index)

    @staticmethod
    def _resolve_paths(base: pathlib.Path) -> Tuple[pathlib.Path, pathlib.Path]:
        if base.suffix in (".jsonl", ".jsonl.gz", ".jsonl.zst"):
            ndjson_path = base
            index_path = base.with_name(base.stem + "_index.csv")
        else:
            ndjson_path = base.with_suffix(".jsonl")
            index_path = base.with_name(base.name + "_index.csv")
        if not ndjson_path.exists():
            raise FileNotFoundError(f"NDJSON file not found: {ndjson_path}")
        if not index_path.exists():
            raise FileNotFoundError(f"Index CSV not found: {index_path}")
        return ndjson_path, index_path

    @staticmethod
    def _load_index(index_path: pathlib.Path) -> Dict[str, IndexEntry]:
        index: Dict[str, IndexEntry] = {}
        with index_path.open("r", newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                work_id = row.get("work_id") or None
                offset = int(row["offset"])
                length = int(row["length"])
                if work_id:
                    index[work_id] = IndexEntry(work_id, offset, length)
        return index

    def close(self) -> None:
        if not self._fh.closed:
            self._fh.close()

    def get_bytes(self, work_id: str) -> bytes:
        entry = self._index.get(work_id)
        if entry is None:
            raise KeyError(f"work_id {work_id!r} not found in raw index.")
        self._fh.seek(entry.offset)
        return self._fh.read(entry.length)

    def get_json(self, work_id: str) -> Dict[str, Any]:
        raw = self.get_bytes(work_id)
        return json.loads(raw.decode("utf-8"))

    def __contains__(self, work_id: str) -> bool:
        return work_id in self._index

    def iter_entries(self) -> Iterator[IndexEntry]:
        return iter(self._index.values())

    def iter_json(self) -> Iterator[Dict[str, Any]]:
        for entry in self.iter_entries():
            if entry.work_id is None:
                continue
            yield self.get_json(entry.work_id)

    @property
    def path(self) -> pathlib.Path:
        return self._path


def write_raw_chunks(
    records: Sequence[Dict[str, Any]],
    *,
    outdir: pathlib.Path,
    basename: str,
    chunk_size: int = 1000,
    compression: str = "gzip",
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Persist OpenAlex records to chunked NDJSON files with sidecar byte-offset indexes.

    Returns a manifest dictionary describing the chunks; the manifest is also written
    to ``<outdir>/<basename>_manifest.json``.
    """
    outdir.mkdir(parents=True, exist_ok=True)
    chunks: List[Dict[str, Any]] = []
    total_records = len(records)

    for part_index, chunk in enumerate(_iter_chunks(records, chunk_size)):
        chunk_base = outdir / f"{basename}_part{part_index:04d}"
        ndjson_path = chunk_base.with_suffix(".jsonl")
        index_path = chunk_base.with_name(chunk_base.name + "_index.csv")
        _write_ndjson_and_index(chunk, ndjson_path, index_path)
        compressed_path = _compress_file(ndjson_path, compression)
        chunk_info: Dict[str, Any] = {
            "basepath": str(chunk_base),
            "records": len(chunk),
            "ndjson_path": str(ndjson_path),
            "ndjson_bytes": ndjson_path.stat().st_size,
            "index_path": str(index_path),
            "index_bytes": index_path.stat().st_size,
            "compressed_path": str(compressed_path) if compressed_path else None,
            "compressed_bytes": compressed_path.stat().st_size if compressed_path else None,
        }
        chunks.append(chunk_info)

    manifest: Dict[str, Any] = {
        "created_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "basename": basename,
        "outdir": str(outdir),
        "records": total_records,
        "chunk_size": chunk_size,
        "compression": compression,
        "chunks": chunks,
        "metadata": metadata or {},
    }

    manifest_path = outdir / f"{basename}_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    manifest["manifest_path"] = str(manifest_path)
    return manifest


def _iter_chunks(records: Sequence[Dict[str, Any]], size: int) -> Iterable[Sequence[Dict[str, Any]]]:
    if size <= 0:
        yield records
        return
    for start in range(0, len(records), size):
        yield records[start : start + size]


def _write_ndjson_and_index(
    records: Sequence[Dict[str, Any]],
    ndjson_path: pathlib.Path,
    index_path: pathlib.Path,
) -> None:
    rows: List[IndexEntry] = []
    with ndjson_path.open("wb") as fout:
        for rec in records:
            start = fout.tell()
            payload = json.dumps(rec, sort_keys=True, separators=(",", ":")) + "\n"
            encoded = payload.encode("utf-8")
            fout.write(encoded)
            end = fout.tell()
            raw_id = rec.get("id") if isinstance(rec, dict) else None
            work_id = raw_id.rsplit("/", 1)[-1] if isinstance(raw_id, str) else None
            rows.append(IndexEntry(work_id=work_id, offset=start, length=end - start))

    with index_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["work_id", "offset", "length"])
        for entry in rows:
            writer.writerow([entry.work_id, entry.offset, entry.length])


def _compress_file(path: pathlib.Path, method: str) -> Optional[pathlib.Path]:
    if method == "none":
        return None
    if method == "gzip":
        target = path.with_suffix(path.suffix + ".gz")
        with path.open("rb") as fin, gzip.open(target, "wb") as fout:
            shutil.copyfileobj(fin, fout)
        return target
    raise ValueError(f"Unsupported compression method: {method}")

