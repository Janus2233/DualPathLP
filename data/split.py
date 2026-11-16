"""按时间切分 + 10% 新节点归纳留出。

逐行对齐 ``参考模型/TAMI/utils/DataLoader.py:67-175`` 的 ``get_link_prediction_data``
（与 ``参考模型/TGN-SEAL/utils/data_processing.py:20-104`` 同源）。

要点（照抄，不改口径）：

1. 切分点是 ``np.quantile(ts, [1-v-r, 1-r])``，默认 v=r=0.15 → 70/15/15。
2. 训练用 ``ts <= val_time``；验证 ``val_time < ts <= test_time``；测试 ``ts > test_time``。
3. ``random.seed(2020)`` 后从"测试期节点"中抽 10% 作为新节点，
   并把**所有涉及新节点的边从训练集中移除**（归纳评测）。
4. ``edge_idxs`` 全程携带，测试期边表是**全量输入**（含训练期边），与参考实现一致。

本模块**不做**任何未来信息泄漏防护上的"改良"——参考实现怎么做就怎么做。
已知的泄漏点记录在 ``docs/BUGLOG.md``，不在此处静默修改。
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import config
from .preprocess import ProcessedEdges


@dataclass
class SplitData:
    """一段边集。字段名对齐参考实现的 ``Data`` 类。"""

    src_node_ids: np.ndarray  # int64, 1-based
    dst_node_ids: np.ndarray  # int64, 1-based
    node_interact_times: np.ndarray  # float64
    edge_ids: np.ndarray  # int64, 1-based, 指向 full 边表的行
    labels: np.ndarray  # float64, 恒 1.0

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
    """一次完整切分的全部产物。"""

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
    new_node_set: set[int]  # 未在训练集出现的节点
    stats: dict[str, object]

    def as_arrays(self) -> dict[str, np.ndarray]:
        """摊平成 ``npz`` 可存的字典（节点集合另存为排序数组）。"""
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
    """按时间切分并做归纳留出。"""
    src, dst, ts = pe.src, pe.dst, pe.ts

    # --- 1. 切分点（TAMI: utils/DataLoader.py:94） ------------------------- #
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

    # --- 2. 归纳留出（TAMI: utils/DataLoader.py:106-133） ------------------ #
    # 参考实现这一行注释就是 "the setting of seed follows previous works"，
    # 种子 2020 硬编码。照抄以保证与文献数字可比。
    random.seed(new_node_seed)

    node_set = set(src.tolist()) | set(dst.tolist())
    num_total_unique_node_ids = len(node_set)

    # 出现在测试期的节点
    test_time_mask = ts > val_time
    test_node_set = set(src[test_time_mask].tolist()) | set(dst[test_time_mask].tolist())

    n_new = int(new_node_ratio * num_total_unique_node_ids)
    # 注意：参考实现是 random.sample(list(set), n)，set 迭代顺序影响结果。
    # 传入前对 list 排序，使结果只依赖种子而不依赖 set 的内部顺序（可复现性）。
    # 这只影响"抽中哪些节点"，不改变抽样规模与协议。
    new_test_node_set = set(random.sample(sorted(test_node_set), n_new))

    new_test_source_mask = np.fromiter((v in new_test_node_set for v in src.tolist()), bool, src.size)
    new_test_destination_mask = np.fromiter((v in new_test_node_set for v in dst.tolist()), bool, dst.size)
    observed_edges_mask = ~new_test_source_mask & ~new_test_destination_mask

    # --- 3. 三段切分 ------------------------------------------------------ #
    val_mask = (ts <= test_time) & (ts > val_time)
    test_mask = ts > test_time

    # 训练：验证期及以前，且不涉及任何新节点
    train_mask = (ts <= val_time) & observed_edges_mask

    train_data = _subset(pe, train_mask)
    val_data = _subset(pe, val_mask)
    test_data = _subset(pe, test_mask)

    # --- 4. 新节点相关段 -------------------------------------------------- #
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

    # --- 5. 一致性断言（参考实现也有对应 assert） -------------------------- #
    # TAMI: utils/DataLoader.py:133 "assert len(train_node_set & new_test_node_set) == 0"
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

    # --- 6. 统计（供自检） ------------------------------------------------- #
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
        # 时间边界的严格性：这三条是本管线最关键的防泄漏检查
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
    """把切分产物写入 ``runs/data/<name>/splits_<name>.npz``。"""
    path = config.splits_path(sp.name)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **sp.as_arrays())
    return path


def load_splits(name: str) -> dict[str, np.ndarray]:
    """读回切分产物。"""
    path = config.splits_path(name)
    if not path.is_file():
        raise FileNotFoundError(
            f"未找到 {path}；请先运行 python -m data.build --dataset {name}"
        )
    with np.load(path) as z:
        return {k: z[k] for k in z.files}
