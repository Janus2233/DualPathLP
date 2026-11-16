
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import config


class DataIntegrityError(RuntimeError):
    pass


@dataclass
class RawEdges:

    spec: config.DatasetSpec
    src: np.ndarray
    dst: np.ndarray
    ts: np.ndarray
    row_order: np.ndarray

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
    if verify:
        verify_dataset(spec)

    try:
        arr = np.loadtxt(spec.path, dtype=np.float64, ndmin=2)
    except ValueError as exc:
        raise DataIntegrityError(f"{spec.filename} 无法按三列数值解析：{exc}") from exc

    if arr.shape[1] != 3:
        raise DataIntegrityError(
            f"{spec.filename} 期望 3 列（源节点 目标节点 时间戳），实际 {arr.shape[1]} 列。"
        )

    nan_rows = int(np.count_nonzero(~np.isfinite(arr).all(axis=1)))
    if nan_rows:
        raise DataIntegrityError(f"{spec.filename} 有 {nan_rows} 行含 NaN/Inf。")

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
