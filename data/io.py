"""原始数据读取、格式校验与完整性校验。

原始文件位于 ``dataset/``，**只读**。本模块不做任何写操作。
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import config


class DataIntegrityError(RuntimeError):
    """原始文件被改动、行数与登记不符，或格式非法。"""


@dataclass
class RawEdges:
    """一份原始边表的只读视图。

    ``row_order`` 是边在**原始文件中**的行号，用作并列时间戳的确定性 tie-break。
    """

    spec: config.DatasetSpec
    src: np.ndarray  # int64, 原始节点 id（未重映射）
    dst: np.ndarray  # int64, 原始节点 id（未重映射）
    ts: np.ndarray  # float64, 原始时间戳（未归一化）
    row_order: np.ndarray  # int64, 0-based 原始文件行号

    @property
    def n_edges(self) -> int:
        return int(self.src.shape[0])

    @property
    def n_self_loops(self) -> int:
        return int(np.count_nonzero(self.src == self.dst))

    @property
    def n_unique_nodes(self) -> int:
        return len(np.unique(np.concatenate([self.src, self.dst])))

    def is_ts_sorted(self) -> bool:
        """原始文件是否已按时间戳非降序排列。"""
        return bool(np.all(np.diff(self.ts) >= 0))


def sha256_of(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest().upper()


def verify_dataset(spec: config.DatasetSpec) -> str:
    """校验文件存在且 SHA256 与登记值一致；返回实际摘要。"""
    path = spec.path
    if not path.is_file():
        raise DataIntegrityError(f"原始数据文件不存在：{path}")
    actual = sha256_of(path)
    if actual != spec.sha256.upper():
        raise DataIntegrityError(
            f"{spec.filename} 的 SHA256 与登记值不符，原始文件已被改动。\n"
            f"  期望 {spec.sha256.upper()}\n  实际 {actual}"
        )
    return actual


def read_raw(spec: config.DatasetSpec, *, verify: bool = True) -> RawEdges:
    """读取原始三列文本 ``源节点 目标节点 时间戳``。

    与参考实现一致使用 ``np.loadtxt``（空白分隔、无表头）。
    TGN-SEAL 的 txt 管线：``utils/preprocess_txt_data.py``
    """
    if verify:
        verify_dataset(spec)

    try:
        arr = np.loadtxt(spec.path, dtype=np.float64, ndmin=2)
    except ValueError as exc:  # 列数不一致 / 非数值
        raise DataIntegrityError(f"{spec.filename} 无法按三列数值解析：{exc}") from exc

    if arr.shape[1] != 3:
        raise DataIntegrityError(
            f"{spec.filename} 期望 3 列（源节点 目标节点 时间戳），实际 {arr.shape[1]} 列。"
        )

    nan_rows = int(np.count_nonzero(~np.isfinite(arr).all(axis=1)))
    if nan_rows:
        raise DataIntegrityError(f"{spec.filename} 有 {nan_rows} 行含 NaN/Inf。")

    # 节点 id 必须是整数值
    node_part = arr[:, :2]
    if not np.all(node_part == np.floor(node_part)):
        bad = int(np.count_nonzero(node_part != np.floor(node_part)))
        raise DataIntegrityError(f"{spec.filename} 有 {bad} 个非整数节点 id。")

    return RawEdges(
        spec=spec,
        src=arr[:, 0].astype(np.int64),
        dst=arr[:, 1].astype(np.int64),
        ts=arr[:, 2].astype(np.float64),
        row_order=np.arange(arr.shape[0], dtype=np.int64),
    )


def describe(raw: RawEdges) -> dict[str, object]:
    """汇总一份原始边表的基本事实，用于自检与文档核对。"""
    ts = raw.ts
    vals, counts = np.unique(ts, return_counts=True)
    tied_rows = int(counts[counts > 1].sum())
    return {
        "dataset": raw.spec.name,
        "file": raw.spec.filename,
        "n_edges": raw.n_edges,
        "n_unique_nodes": raw.n_unique_nodes,
        "n_self_loops": raw.n_self_loops,
        "n_exact_duplicate_rows": raw.n_edges
        - len(np.unique(np.stack([raw.src, raw.dst, ts], axis=1), axis=0)),
        "ts_min": float(ts.min()),
        "ts_max": float(ts.max()),
        "ts_span_days": float((ts.max() - ts.min()) / 86400.0),
        "ts_n_unique": int(len(vals)),
        "ts_n_tied_timestamps": int(len(vals[counts > 1])),
        "ts_rows_in_ties": tied_rows,
        "ts_rows_in_ties_pct": round(100.0 * tied_rows / raw.n_edges, 2),
        "file_ts_sorted": raw.is_ts_sorted(),
        "ts_n_descending_steps": int(np.count_nonzero(np.diff(ts) < 0)),
    }
