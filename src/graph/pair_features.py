
from __future__ import annotations

import numpy as np

__all__ = ["PAIR_DIM", "PAIR_GROUPS", "pair_features", "pair_features_batch", "parse_dims"]

PAIR_DIM = 8

PAIR_GROUPS: dict[str, list[int]] = {
    "cn": [0],
    "aa": [1],
    "jac": [2],
    "deg": [3, 4],
    "hist": [5, 6, 7],
}

PAIR_DEFAULT_SPEC = "aa,jac,deg,hist"


def parse_dims(spec: str | None) -> np.ndarray:
    if spec is None or spec.strip().lower() in ("", "default"):
        spec = PAIR_DEFAULT_SPEC
    if spec.strip().lower() == "all":
        return np.arange(PAIR_DIM, dtype=np.int64)
    idx: list[int] = []
    for tok in spec.split(","):
        tok = tok.strip().lower()
        if not tok:
            continue
        if tok in PAIR_GROUPS:
            idx.extend(PAIR_GROUPS[tok])
        else:
            i = int(tok)
            assert 0 <= i < PAIR_DIM, f"pair 维度下标越界：{i}"
            idx.append(i)
    assert idx, f"--pair-dims 解析为空：{spec!r}"
    return np.array(sorted(set(idx)), dtype=np.int64)


def pair_features(graph, u: int, v: int, t: float, window: float | None = None,
                  dims: np.ndarray | None = None) -> np.ndarray:
    nu, _ = graph.neighbors_before(u, t, None, window=window)
    nv, _ = graph.neighbors_before(v, t, None, window=window)
    su = np.unique(nu[nu != v]) if nu.size else nu
    sv = np.unique(nv[nv != u]) if nv.size else nv
    inter = np.intersect1d(su, sv)
    cn = int(inter.size)
    union = int(su.size + sv.size - cn)
    jac = (cn / union) if union > 0 else 0.0

    aa = 0.0
    for w in inter.tolist():
        dw = graph.count_before(int(w), t, window)
        aa += 1.0 / np.log(dw + 2.0)

    nu_all, tu_all = graph.neighbors_before(u, t, None)
    mask = nu_all == v
    n_uv = int(mask.sum())
    dt_uv = float(t - tu_all[mask].max()) if n_uv else 0.0

    return np.array([np.log1p(cn), aa, jac, np.log1p(su.size), np.log1p(sv.size),
                     np.log1p(n_uv), np.log1p(dt_uv) if n_uv else 0.0,
                     1.0 if n_uv else 0.0], dtype=np.float32)[
        np.arange(PAIR_DIM, dtype=np.int64) if dims is None else dims]


def pair_features_batch(graph, src, dst, ts, window: float | None = None,
                        dims: np.ndarray | None = None) -> np.ndarray:
    src_l, dst_l, ts_l = np.asarray(src).tolist(), np.asarray(dst).tolist(), np.asarray(ts).tolist()
    d = PAIR_DIM if dims is None else len(dims)
    out = np.zeros((len(src_l), d), dtype=np.float32)
    for i, (u, v, t) in enumerate(zip(src_l, dst_l, ts_l)):
        out[i] = pair_features(graph, int(u), int(v), float(t), window, dims)
    return out
