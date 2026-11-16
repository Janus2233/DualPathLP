
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from . import config
from .io import RawEdges


@dataclass
class ProcessedEdges:

    name: str
    src: np.ndarray
    dst: np.ndarray
    ts: np.ndarray
    edge_idx: np.ndarray
    label: np.ndarray
    orig_src: np.ndarray
    orig_dst: np.ndarray
    raw_row_order: np.ndarray
    n_nodes_raw: int
    n_max_id: int
    stats: dict[str, object]

    @property
    def n_edges(self) -> int:
        return int(self.src.shape[0])

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "u": self.src,
                "i": self.dst,
                "ts": self.ts,
                "label": self.label,
                "idx": self.edge_idx,
            }
        )


def build_id_mapping(
    orig_src: np.ndarray, orig_dst: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    uniq = np.unique(np.concatenate([orig_src, orig_dst]))

    def remap(x: np.ndarray) -> np.ndarray:
        pos = np.searchsorted(uniq, x)
        return (pos + config.ID_BASE).astype(np.int64)

    return remap(orig_src), remap(orig_dst), uniq

def preprocess(raw: RawEdges) -> ProcessedEdges:
    orig_src, orig_dst = raw.src, raw.dst
    ts, row_order = raw.ts, raw.row_order

    n_raw = raw.n_edges
    keep = orig_src != orig_dst
    n_self_loops = int(n_raw - np.count_nonzero(keep))

    s, d, t, r = orig_src[keep], orig_dst[keep], ts[keep], row_order[keep]

    order = np.lexsort((r, t))
    s, d, t, r = s[order], d[order], t[order], r[order]

    n_kept = int(s.shape[0])
    if n_kept == 0:
        raise ValueError(f"{raw.spec.name} 丢弃自环后没有剩余边。")

    s_mapped, d_mapped, uniq_nodes = build_id_mapping(s, d)
    n_nodes_raw = int(uniq_nodes.shape[0])
    edge_idx = np.arange(1, n_kept + 1, dtype=np.int64)
    label = np.ones(n_kept, dtype=np.float64)

    result = ProcessedEdges(
        name=raw.spec.name,
        src=s_mapped,
        dst=d_mapped,
        ts=t,
        edge_idx=edge_idx,
        label=label,
        orig_src=s,
        orig_dst=d,
        raw_row_order=r,
        n_nodes_raw=n_nodes_raw,
        n_max_id=n_nodes_raw,
        stats={},
    )

    ts_arr = result.ts
    _, counts = np.unique(ts_arr, return_counts=True)
    result.stats = {
        "dataset": raw.spec.name,
        "n_edges_raw": n_raw,
        "n_self_loops_dropped": n_self_loops,
        "n_edges": n_kept,
        "self_loop_pct": round(100.0 * n_self_loops / n_raw, 3),
        "n_nodes": n_nodes_raw,
        "n_nodes_dropped_only_self_loops": raw.n_unique_nodes - n_nodes_raw,
        "n_unique_directed_pairs": int(
            len(np.unique(np.stack([s_mapped, d_mapped], axis=1), axis=0))
        ),
        "ts_min": float(ts_arr.min()),
        "ts_max": float(ts_arr.max()),
        "ts_is_monotonic": bool(np.all(np.diff(ts_arr) >= 0)),
        "ts_strictly_increasing": bool(np.all(np.diff(ts_arr) > 0)),
        "ts_rows_in_ties": int(counts[counts > 1].sum()),
        "has_self_loops": bool(np.any(s_mapped == d_mapped)),
        "id_min": int(min(s_mapped.min(), d_mapped.min())),
        "id_max": int(max(s_mapped.max(), d_mapped.max())),
        "idx_min": int(edge_idx.min()),
        "idx_max": int(edge_idx.max()),
    }
    return result


def write_processed(pe: ProcessedEdges) -> dict[str, Path]:
    import json

    paths = config.processed_paths(pe.name)
    paths["dir"].mkdir(parents=True, exist_ok=True)

    df = pe.to_frame()
    df.to_csv(paths["edges"], index=True)

    n_edges = pe.n_edges
    edge_feat = np.zeros((n_edges + 1, config.EDGE_FEAT_DIM), dtype=np.float32)
    node_feat = np.zeros((pe.n_max_id + 1, config.NODE_FEAT_DIM), dtype=np.float32)
    np.save(paths["edge_feat"], edge_feat)
    np.save(paths["node_feat"], node_feat)

    meta = {
        "dataset": pe.name,
        "source_file": pe.stats.get("dataset"),
        "n_edges": n_edges,
        "n_nodes": pe.n_nodes_raw,
        "edge_feat_shape": list(edge_feat.shape),
        "node_feat_shape": list(node_feat.shape),
        "protocol": {
            "train_ratio": config.TRAIN_RATIO,
            "val_ratio": config.VAL_RATIO,
            "test_ratio": config.TEST_RATIO,
            "new_node_ratio": config.NEW_NODE_RATIO,
            "new_node_seed": config.NEW_NODE_SEED,
            "id_base": config.ID_BASE,
            "self_loops": "dropped",
            "sort": "ts ascending, tie-break by original file row order",
        },
        "stats": pe.stats,
    }
    meta_path = paths["dir"] / f"meta_{pe.name}.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    return {**paths, "meta": meta_path}


def load_processed(name: str) -> pd.DataFrame:
    paths = config.processed_paths(name)
    if not paths["edges"].is_file():
        raise FileNotFoundError(
            f"未找到 {paths['edges']}；请先运行 python -m data.build --dataset {name}"
        )
    return pd.read_csv(paths["edges"])
