from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional

import networkx as nx
import numpy as np
import pandas as pd


@dataclass
class LiteGraph:
    node_ids: np.ndarray  # dtype=str
    pub_ts: np.ndarray  # dtype=int64, -1 when unknown
    pub_qtr: np.ndarray  # dtype=str, e.g. "2023Q4", empty string when unknown
    pub_year: np.ndarray  # dtype=int32, -1 when unknown
    edge_src: np.ndarray  # dtype=int32
    edge_dst: np.ndarray  # dtype=int32
    weight_total: np.ndarray  # dtype=float32
    edge_type: np.ndarray  # dtype=uint8 (0=citation,1=coupling,2=hybrid)
    coupling_stats: Dict[str, object]

    @property
    def n_nodes(self) -> int:
        return int(self.node_ids.size)

    @property
    def n_edges(self) -> int:
        return int(self.edge_src.size)

    def to_networkx(self, *, include_publication_dates: bool) -> nx.DiGraph:
        """
        Materialize a NetworkX graph for downstream consumers.

        Rehydrates all required node attributes: publication_date, pub_qtr, pub_year.
        These are needed by scripts/communities.py for windowing and new-work counts.
        """
        G = nx.DiGraph()
        node_ids_list = self.node_ids.tolist()

        # Rehydrate node attributes
        for idx, node in enumerate(node_ids_list):
            attrs = {}

            # Publication date (if available and requested)
            if include_publication_dates and self.pub_ts.size == self.node_ids.size:
                ts_val = int(self.pub_ts[idx])
                if ts_val >= 0:
                    dt = pd.to_datetime(ts_val, unit="ns", utc=True).tz_convert(None)
                    attrs["publication_date"] = dt.isoformat()

            # pub_qtr (required for new_works counting in communities.py)
            if self.pub_qtr.size == self.node_ids.size:
                qtr_val = str(self.pub_qtr[idx])
                if qtr_val and qtr_val != "":
                    attrs["pub_qtr"] = qtr_val

            # pub_year (useful for windowing)
            if self.pub_year.size == self.node_ids.size:
                year_val = int(self.pub_year[idx])
                if year_val >= 0:
                    attrs["pub_year"] = year_val

            G.add_node(node, **attrs)

        for idx in range(self.edge_src.size):
            u = node_ids_list[int(self.edge_src[idx])]
            v = node_ids_list[int(self.edge_dst[idx])]
            wt = float(self.weight_total[idx])
            G.add_edge(u, v, weight_total=wt)

        if self.coupling_stats:
            # Ensure JSON-serialisable copy
            stats: Dict[str, object] = {}
            for key, value in self.coupling_stats.items():
                if isinstance(value, np.generic):
                    stats[key] = value.item()
                else:
                    stats[key] = value
            G.graph["coupling_stats"] = stats
        return G

    def save(self, path: Path, *, coupling_stats_path: Optional[Path] = None) -> None:
        arrays = {
            "node_ids": self.node_ids,
            "pub_ts": self.pub_ts,
            "pub_qtr": self.pub_qtr,
            "pub_year": self.pub_year,
            "edge_src": self.edge_src,
            "edge_dst": self.edge_dst,
            "weight_total": self.weight_total,
            "edge_type": self.edge_type,
        }
        np.savez_compressed(path, **arrays)
        stats_path = coupling_stats_path or path.with_suffix(".json")
        stats_path.write_text(json.dumps(self.coupling_stats, indent=2))

    @classmethod
    def load(cls, path: Path) -> "LiteGraph":
        data = np.load(path, allow_pickle=False)
        node_ids = data["node_ids"]
        pub_ts = data["pub_ts"]

        # Backward compatibility: pub_qtr and pub_year may not exist in old formats
        if "pub_qtr" in data:
            pub_qtr = data["pub_qtr"]
        else:
            # Derive from pub_ts if possible
            pub_qtr = np.array(["" for _ in range(node_ids.size)], dtype="U16")
            if pub_ts.size == node_ids.size:
                for idx in range(node_ids.size):
                    ts_val = int(pub_ts[idx])
                    if ts_val >= 0:
                        try:
                            dt = pd.to_datetime(ts_val, unit="ns", utc=True).tz_convert(None)
                            qtr = pd.Period(dt, freq="Q")
                            pub_qtr[idx] = str(qtr)
                        except Exception:
                            pass

        if "pub_year" in data:
            pub_year = data["pub_year"]
        else:
            # Derive from pub_ts if possible
            pub_year = np.full(node_ids.size, -1, dtype=np.int32)
            if pub_ts.size == node_ids.size:
                for idx in range(node_ids.size):
                    ts_val = int(pub_ts[idx])
                    if ts_val >= 0:
                        try:
                            dt = pd.to_datetime(ts_val, unit="ns", utc=True).tz_convert(None)
                            pub_year[idx] = dt.year
                        except Exception:
                            pass

        edge_src = data["edge_src"]
        edge_dst = data["edge_dst"]
        weight_total = data["weight_total"]
        edge_type = data["edge_type"]
        stats_path = path.with_suffix(".json")
        if stats_path.exists():
            coupling_stats = json.loads(stats_path.read_text())
        else:
            coupling_stats = {}
        return cls(
            node_ids=node_ids,
            pub_ts=pub_ts,
            pub_qtr=pub_qtr,
            pub_year=pub_year,
            edge_src=edge_src,
            edge_dst=edge_dst,
            weight_total=weight_total,
            edge_type=edge_type,
            coupling_stats=coupling_stats,
        )

    @classmethod
    def from_networkx(
        cls,
        G: nx.DiGraph,
        *,
        include_publication_dates: bool = True,
        include_edge_types: bool = True,
    ) -> "LiteGraph":
        node_ids = np.array(list(G.nodes()), dtype="U64")
        index_map = {node: idx for idx, node in enumerate(node_ids)}

        # Extract publication metadata
        pub_ts = np.full(node_ids.size, -1, dtype=np.int64)
        pub_qtr = np.array(["" for _ in range(node_ids.size)], dtype="U16")
        pub_year = np.full(node_ids.size, -1, dtype=np.int32)

        for node, data in G.nodes(data=True):
            idx = index_map[node]

            # Capture publication timestamp
            if include_publication_dates:
                ts_val = _parse_publication_ts(data.get("publication_date"))
                if ts_val is not None:
                    pub_ts[idx] = ts_val

            # Capture pub_qtr (required for new_works counting)
            qtr_val = data.get("pub_qtr")
            if qtr_val:
                pub_qtr[idx] = str(qtr_val)

            # Capture pub_year (useful for windowing)
            year_val = data.get("pub_year")
            if year_val is not None:
                try:
                    pub_year[idx] = int(year_val)
                except (ValueError, TypeError):
                    pass

        num_edges = G.number_of_edges()
        edge_src = np.empty(num_edges, dtype=np.int32)
        edge_dst = np.empty(num_edges, dtype=np.int32)
        weight_total = np.empty(num_edges, dtype=np.float32)
        edge_type = np.zeros(num_edges, dtype=np.uint8)

        for idx, (u, v, data) in enumerate(G.edges(data=True)):
            edge_src[idx] = index_map[u]
            edge_dst[idx] = index_map[v]
            wt = data.get("weight_total", data.get("weight", 1.0))
            weight_total[idx] = float(wt)
            if include_edge_types:
                et = data.get("edge_type", "citation")
                edge_type[idx] = 1 if et == "coupling" else 2 if et == "hybrid" else 0

        coupling_stats = G.graph.get("coupling_stats") or {}

        return cls(
            node_ids=node_ids,
            pub_ts=pub_ts,
            pub_qtr=pub_qtr,
            pub_year=pub_year,
            edge_src=edge_src,
            edge_dst=edge_dst,
            weight_total=weight_total,
            edge_type=edge_type,
            coupling_stats=dict(coupling_stats),
        )


def _parse_publication_ts(value: Optional[object]) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, (pd.Timestamp, )):
        ts = value.tz_localize(None) if value.tzinfo else value
        return int(ts.to_datetime64().astype("datetime64[ns]").astype(np.int64))
    try:
        ts = pd.to_datetime(value, errors="coerce")
    except Exception:
        return None
    if pd.isna(ts):
        return None
    if isinstance(ts, pd.Series):
        ts = ts.iloc[0]
    ts = ts.tz_localize(None) if getattr(ts, "tzinfo", None) else ts
    return int(ts.to_datetime64().astype("datetime64[ns]").astype(np.int64))


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def write_lite_graph(
    G: nx.DiGraph,
    *,
    lite_path: Path,
    include_publication_dates: bool = True,
) -> LiteGraph:
    ensure_parent_dir(lite_path)
    lite = LiteGraph.from_networkx(
        G,
        include_publication_dates=include_publication_dates,
        include_edge_types=True,
    )
    lite.save(lite_path)
    return lite

