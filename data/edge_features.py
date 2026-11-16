
from __future__ import annotations

import numpy as np

ALL_COLS: tuple[str, ...] = ("pair_prior_count", "deg_src_prior", "deg_dst_prior", "cn_prior")

FULL_DIM = len(ALL_COLS)


def causal_edge_features(src: np.ndarray, dst: np.ndarray, ts: np.ndarray, *,
                         cols: tuple[str, ...] = ALL_COLS,
                         log1p: bool = True) -> np.ndarray:
    src = np.asarray(src).astype(np.int64, copy=False)
    dst = np.asarray(dst).astype(np.int64, copy=False)
    ts = np.asarray(ts, dtype=np.float64)
    n = len(src)
    if not (len(dst) == n and len(ts) == n):
        raise ValueError(f"src/dst/ts 长度不一致：{len(src)}/{len(dst)}/{len(ts)}")
    if n > 1 and np.any(np.diff(ts) < 0):
        raise ValueError("ts 非非降序：本函数按索引推进状态，乱序会静默算错，故直接拒绝")

    bad = [c for c in cols if c not in ALL_COLS]
    if bad:
        raise ValueError(f"未知列 {bad}；可选 {list(ALL_COLS)}")

    out = np.zeros((n, len(cols)), dtype=np.float32)
    idx = {c: i for i, c in enumerate(cols)}

    pair_cnt: dict[tuple[int, int], int] = {}
    deg: dict[int, int] = {}
    adj: dict[int, set[int]] = {}

    for i in range(n):
        u = int(src[i]); v = int(dst[i])
        key = (u, v) if u <= v else (v, u)

        pc = pair_cnt.get(key, 0)
        du = deg.get(u, 0)
        dv = deg.get(v, 0)
        cn = 0
        au = adj.get(u)
        if au is not None and v in adj:
            av = adj[v]
            cn = len(au & av) if len(au) <= len(av) else len(av & au)

        vals = {"pair_prior_count": pc, "deg_src_prior": du, "deg_dst_prior": dv, "cn_prior": cn}
        for c, j in idx.items():
            out[i, j] = vals[c]

        pair_cnt[key] = pc + 1
        deg[u] = du + 1
        deg[v] = dv + 1
        adj.setdefault(u, set()).add(v)
        adj.setdefault(v, set()).add(u)

    if log1p:
        np.log1p(out, out=out)
    return out


def neighbor_feature_lookup(src: np.ndarray, dst: np.ndarray, ts: np.ndarray, *,
                            cols: tuple[str, ...] = ALL_COLS) -> tuple[dict, np.ndarray]:
    F = causal_edge_features(src, dst, ts, cols=cols)
    key2row: dict[tuple[int, int, float], int] = {}
    for i in range(len(src)):
        u, v, t = int(src[i]), int(dst[i]), float(ts[i])
        key2row.setdefault((u, v, t), i)
        key2row.setdefault((v, u, t), i)
    return key2row, F


def feature_names(cols: tuple[str, ...] = ALL_COLS, *, log1p: bool = True) -> list[str]:
    return [f"log1p_{c}" for c in cols] if log1p else list(cols)
