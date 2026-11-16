"""预处理：排序 → 丢自环 → 全图 1-based 重映射 → 产出 TGB 三件套。

严格对齐参考实现，不自行发明步骤或参数：

- 丢自环：``参考模型/TGN-SEAL/utils/preprocess_csv_data.py:16-17`` (``if i == u: continue``)
- 排序：  ``参考模型/TGN-SEAL/utils/preprocess_txt_data.py:126`` (``df.sort_values(by="ts")``)
- 重映射：``参考模型/TGN-SEAL/utils/preprocess_csv_data.py:41-56``（``rename`` + 整体 +1，1-based）
- 特征：  ``参考模型/TAMI/utils/DataLoader.py:81-83``（172 维零填充）
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from . import config
from .io import RawEdges


@dataclass
class ProcessedEdges:
    """预处理结果。``src``/``dst`` 是 **1-based** 的重映射 id，0 保留给 padding。"""

    name: str
    src: np.ndarray  # int64, 1-based
    dst: np.ndarray  # int64, 1-based
    ts: np.ndarray  # float64
    edge_idx: np.ndarray  # int64, 1-based
    label: np.ndarray  # float64, 恒为 1.0
    orig_src: np.ndarray  # int64, 原始 id（诊断用）
    orig_dst: np.ndarray
    raw_row_order: np.ndarray  # int64, 原始文件行号（诊断用）
    n_nodes_raw: int  # 重映射前的唯一节点数
    n_max_id: int  # 重映射后的最大 id（= n_nodes_raw，因为 1-based）
    stats: dict[str, object]

    @property
    def n_edges(self) -> int:
        return int(self.src.shape[0])

    def to_frame(self) -> pd.DataFrame:
        """与 TGB/DyGLib 完全一致的列布局（``,u,i,ts,label,idx``）。"""
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
    """全图 id 重映射：``np.unique`` 升序 → 0-based → +1（1-based）。

    与 ``TGN-SEAL/preprocess_csv_data.py:41-56`` 的 ``rename`` 等价。
    这是**传导式（transductive）**重映射：映射表建立在整个图之上，
    与参考实现一致（见 data/README.md §1）。

    Returns:
        (重映射后的 src, 重映射后的 dst, 升序的唯一原始 id 数组)
    """
    uniq = np.unique(np.concatenate([orig_src, orig_dst]))

    def remap(x: np.ndarray) -> np.ndarray:
        # 用 searchsorted 做向量化查询，等价于逐元素查 dict，但快得多。
        # 传入的 x 一定来自本图，故 pos 必然命中，无需处理未找到的情况。
        pos = np.searchsorted(uniq, x)
        return (pos + config.ID_BASE).astype(np.int64)

    return remap(orig_src), remap(orig_dst), uniq

def preprocess(raw: RawEdges) -> ProcessedEdges:
    """把一份原始边表变成 TGB 格式的边表。

    步骤与参考实现一致：
      1. 丢弃自环
      2. 按 ``ts`` 排序；并列时以原始行号 tie-break（可复现性要求，见 README §3）
      3. 全图重映射到 1-based 连续 id
      4. ``idx`` 从 1 开始（1-based）
    """
    orig_src, orig_dst = raw.src, raw.dst
    ts, row_order = raw.ts, raw.row_order

    n_raw = raw.n_edges
    keep = orig_src != orig_dst  # 丢自环
    n_self_loops = int(n_raw - np.count_nonzero(keep))

    s, d, t, r = orig_src[keep], orig_dst[keep], ts[keep], row_order[keep]

    # 排序：ts 升序，并列按原始行号升序。lexsort 的主键在**最后**，
    # 因此先用 row_order（次键），再用 t（主键）。
    order = np.lexsort((r, t))
    s, d, t, r = s[order], d[order], t[order], r[order]

    n_kept = int(s.shape[0])
    if n_kept == 0:
        raise ValueError(f"{raw.spec.name} 丢弃自环后没有剩余边。")

    # 全图重映射到 1-based
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
        n_max_id=n_nodes_raw,  # 1-based => 最大 id == 节点数
        stats={},
    )

    # --- 产出统计（供自检与文档核对） -------------------------------------- #
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
    """写出 TGB 三件套到 ``runs/data/<name>/``。

    - ``ml_<name>.csv``：边表，列 ``,u,i,ts,label,idx``（首列为 pandas 索引列，与参考实现一致）
    - ``ml_<name>.npy``：边特征 ``(E+1, 172)``，第 0 行为 padding
    - ``ml_<name>_node.npy``：节点特征 ``(Nmax+1, 172)``，第 0 行为 padding

    另外写出 ``meta_<name>.json`` 记录本次产生的参数与统计，便于回溯。
    """
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
    """读回已写出的边表（列 ``u,i,ts,label,idx``）。"""
    paths = config.processed_paths(name)
    if not paths["edges"].is_file():
        raise FileNotFoundError(
            f"未找到 {paths['edges']}；请先运行 python -m data.build --dataset {name}"
        )
    return pd.read_csv(paths["edges"])
