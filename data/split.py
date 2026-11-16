
from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import config
from .preprocess import ProcessedEdges


@dataclass
class SplitData:

    src_node_ids: np.ndarray
    dst_node_ids: np.ndarray
    node_interact_times: np.ndarray
    edge_ids: np.ndarray
    labels: np.ndarray

    @property
    def num_interactions(self) -> int:
        return int(self.src_node_ids.shape[0])

    @property
    def unique_node_ids(self) -> set[int]:
        return set(self.src_node_ids.tolist()) | set(self.dst_node_ids.tolist())

    @property
    def num_unique_nodes(self) -> int:
        return len(self.unique_node_ids)

    def describe(self) -> dict[str, object]:
        ts = self.node_interact_times
        return {
            "n_interactions": self.num_interactions,
            "n_unique_nodes": self.num_unique_nodes,
            "ts_min": float(ts.min()) if ts.size else None,
            "ts_max": float(ts.max()) if ts.size else None,
        }


@dataclass
class Splits:

    name: str
    full: SplitData
    train: SplitData
    val: SplitData
    test: SplitData
    new_node_val: SplitData
    new_node_test: SplitData
    val_time: float
    test_time: float
    new_test_node_set: set[int]
    new_node_set: set[int]
    stats: dict[str, object]

    def as_arrays(self) -> dict[str, np.ndarray]:
        out: dict[str, np.ndarray] = {
            "val_time": np.float64(self.val_time),
            "test_time": np.float64(self.test_time),
            "new_test_node_set": np.array(sorted(self.new_test_node_set), dtype=np.int64),
            "new_node_set": np.array(sorted(self.new_node_set), dtype=np.int64),
        }
        for prefix, d in (
            ("full", self.full),
            ("train", self.train),
            ("val", self.val),
            ("test", self.test),
            ("new_node_val", self.new_node_val),
            ("new_node_test", self.new_node_test),
        ):
            out[f"{prefix}_src"] = d.src_node_ids
            out[f"{prefix}_dst"] = d.dst_node_ids
            out[f"{prefix}_ts"] = d.node_interact_times
            out[f"{prefix}_eid"] = d.edge_ids
        return out


def _subset(pe: ProcessedEdges, mask: np.ndarray) -> SplitData:
    return SplitData(
        src_node_ids=pe.src[mask],
        dst_node_ids=pe.dst[mask],
        node_interact_times=pe.ts[mask],
        edge_ids=pe.edge_idx[mask],
        labels=pe.label[mask],
    )


def split(
    pe: ProcessedEdges,
    *,
    val_ratio: float = config.VAL_RATIO,
    test_ratio: float = config.TEST_RATIO,
    new_node_ratio: float = config.NEW_NODE_RATIO,
    new_node_seed: int = config.NEW_NODE_SEED,
) -> Splits:
    src, dst, ts = pe.src, pe.dst, pe.ts

    val_time, test_time = (
        float(x)
        for x in np.quantile(ts, [(1.0 - val_ratio - test_ratio), (1.0 - test_ratio)])
    )

    full_data = SplitData(
        src_node_ids=src,
        dst_node_ids=dst,
        node_interact_times=ts,
        edge_ids=pe.edge_idx,
        labels=pe.label,
    )

    random.seed(new_node_seed)

    node_set = set(src.tolist()) | set(dst.tolist())
    num_total_unique_node_ids = len(node_set)

    test_time_mask = ts > val_time
    test_node_set = set(src[test_time_mask].tolist()) | set(dst[test_time_mask].tolist())

    n_new = int(new_node_ratio * num_total_unique_node_ids)
    new_test_node_set = set(random.sample(sorted(test_node_set), n_new))

    new_test_source_mask = np.fromiter((v in new_test_node_set for v in src.tolist()), bool, src.size)
    new_test_destination_mask = np.fromiter((v in new_test_node_set for v in dst.tolist()), bool, dst.size)
    observed_edges_mask = ~new_test_source_mask & ~new_test_destination_mask

    val_mask = (ts <= test_time) & (ts > val_time)
    test_mask = ts > test_time

    train_mask = (ts <= val_time) & observed_edges_mask

    train_data = _subset(pe, train_mask)
    val_data = _subset(pe, val_mask)
    test_data = _subset(pe, test_mask)

    train_node_set = train_data.unique_node_ids
    new_node_set = node_set - train_node_set

    edge_contains_new_node_mask = np.fromiter(
        (
            (s in new_node_set) or (d in new_node_set)
            for s, d in zip(src.tolist(), dst.tolist())
        ),
        bool,
        src.size,
    )
    new_node_val_mask = val_mask & edge_contains_new_node_mask
    new_node_test_mask = test_mask & edge_contains_new_node_mask

    new_node_val_data = _subset(pe, new_node_val_mask)
    new_node_test_data = _subset(pe, new_node_test_mask)

    overlap = train_node_set & new_test_node_set
    if overlap:
        raise AssertionError(
            f"训练集与归纳新节点集合相交，共 {len(overlap)} 个节点；"
            "这会让归纳评测失效。请检查 observed_edges_mask 与 train_mask 的计算。"
        )

    splits = Splits(
        name=pe.name,
        full=full_data,
        train=train_data,
        val=val_data,
        test=test_data,
        new_node_val=new_node_val_data,
        new_node_test=new_node_test_data,
        val_time=val_time,
        test_time=test_time,
        new_test_node_set=new_test_node_set,
        new_node_set=new_node_set,
        stats={},
    )

    n = full_data.num_interactions
    splits.stats = {
        "dataset": pe.name,
        "val_ratio": val_ratio,
        "test_ratio": test_ratio,
        "new_node_ratio": new_node_ratio,
        "new_node_seed": new_node_seed,
        "val_time": val_time,
        "test_time": test_time,
        "n_total": n,
        "n_train": train_data.num_interactions,
        "n_val": val_data.num_interactions,
        "n_test": test_data.num_interactions,
        "n_new_node_val": new_node_val_data.num_interactions,
        "n_new_node_test": new_node_test_data.num_interactions,
        "train_pct": round(100.0 * train_data.num_interactions / n, 3),
        "val_pct": round(100.0 * val_data.num_interactions / n, 3),
        "test_pct": round(100.0 * test_data.num_interactions / n, 3),
        "n_nodes_total": num_total_unique_node_ids,
        "n_nodes_train": train_data.num_unique_nodes,
        "n_new_test_nodes_sampled": len(new_test_node_set),
        "n_new_nodes": len(new_node_set),
        "train_max_ts": float(train_data.node_interact_times.max()),
        "val_min_ts": float(val_data.node_interact_times.min()),
        "val_max_ts": float(val_data.node_interact_times.max()),
        "test_min_ts": float(test_data.node_interact_times.min()),
        "train_le_val": bool(train_data.node_interact_times.max() <= val_time),
        "val_lt_test": bool(val_data.node_interact_times.max() <= test_time),
        "test_gt_val": bool(test_data.node_interact_times.min() > val_time),
    }
    return splits


def write_splits(sp: Splits) -> Path:
    path = config.splits_path(sp.name)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **sp.as_arrays())
    return path


def load_splits(name: str) -> dict[str, np.ndarray]:
    path = config.splits_path(name)
    if not path.is_file():
        raise FileNotFoundError(
            f"未找到 {path}；请先运行 python -m data.build --dataset {name}"
        )
    with np.load(path) as z:
        return {k: z[k] for k in z.files}
