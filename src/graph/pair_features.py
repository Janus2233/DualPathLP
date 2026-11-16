"""候选对级别的**因果**结构特征（严格 `< t`，不引入任何未来信息）。

动机（见 `docs/INNOVATIONS.md` I5）：TGB message 类数据集的节点/边原始特征实测**恒为零**
（`EXPERIMENTS.md` 实测：college (1900,172)/(59836,172) 非零元素 = 0），
因此子图路线的全部信息只能来自「结构 + 时间」。`structural_features()` 给的是**单节点**的因果统计；
本模块补的是**候选对**层面的经典链路预测信号（共同邻居 / Adamic-Adar / Jaccard / 历史交互），
全部按候选时刻 `t` 截断。

维度（`PAIR_DIM = 8`，顺序固定，改动必须同步 `docs`）：

0. `log1p(共同邻居数)`（窗口内、去掉对端自身）
1. `Adamic-Adar`：`Σ_w 1/ln(deg_w + 2)`，`deg_w` 取**同一窗口内**的因果度数
2. `Jaccard`：`|N_u ∩ N_v| / |N_u ∪ N_v|`
3. `log1p(|N_u 去掉 v|)`、4. `log1p(|N_v 去掉 u|)`（窗口内因果度数，**剔除对端自身**，与 CN/Jaccard 同口径）
5. `log1p(u–v 历史交互次数)`（**完整因果历史**，不受窗口限制）
6. `log1p(t − 上次 u–v 交互时刻)`（无历史则 0）
7. `1{u–v 有历史交互}`

> 说明：5–7 用完整历史而非窗口 —— “这两人以前有没有来往”是链路预测里最强的单条信号之一，
> 用窗口截断会把它误判成“无历史”。这三项同样严格 `< t`，不构成泄漏。
"""

from __future__ import annotations

import numpy as np

__all__ = ["PAIR_DIM", "PAIR_GROUPS", "pair_features", "pair_features_batch", "parse_dims"]

PAIR_DIM = 8

#: 逐维消融用的分组（键 → 维度下标）。`--pair-dims` 接受这些组名的逗号分隔串，
#: 也接受 `all`（字面全部 8 维）与裸下标（如 `0,1`）；缺省值见 `PAIR_DEFAULT_SPEC`。
PAIR_GROUPS: dict[str, list[int]] = {
    "cn": [0],         # 共同邻居数
    "aa": [1],         # Adamic-Adar
    "jac": [2],        # Jaccard
    "deg": [3, 4],     # 两端窗口内度数
    "hist": [5, 6, 7],  # u–v 历史交互次数 / 距上次时间 / 是否有历史
}

#: **推荐默认组合**：逐维消融实测 `cn` 单独用比完全不用还差、去掉 `cn` 反而最好
#: （`EXPERIMENTS.md` M0-d：0.9600 / 0.9458，优于全 8 维的 0.9574 / 0.9413）。
PAIR_DEFAULT_SPEC = "aa,jac,deg,hist"


def parse_dims(spec: str | None) -> np.ndarray:
    """把 `--pair-dims` 的写法解析成维度下标数组（升序去重）。

    `None` / `""` / `"default"` → `PAIR_DEFAULT_SPEC`；`"all"` 表示字面上的全部 8 维。
    """
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
    """返回 (len(dims),) float32；`graph` 需提供 `neighbors_before` 与 `count_before`。

    ``dims=None`` 表示全部 8 维；传子集用于**逐维消融**（`parse_dims` 负责解析组名）。
    """
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
    """(B, len(dims)) float32。"""
    src_l, dst_l, ts_l = np.asarray(src).tolist(), np.asarray(dst).tolist(), np.asarray(ts).tolist()
    d = PAIR_DIM if dims is None else len(dims)
    out = np.zeros((len(src_l), d), dtype=np.float32)
    for i, (u, v, t) in enumerate(zip(src_l, dst_l, ts_l)):
        out[i] = pair_features(graph, int(u), int(v), float(t), window, dims)
    return out
