"""计划 3 · D2 的因果边特征（纯 numpy，无 torch；遵守 `data/` 层不引入 torch 的边界）。

用途
----
`NodeRoute` 预留了 `edge_feat_dim` 开关（`src/models/node_route.py:104`），
但**全仓库没有任何调用点传它** ⇒ 节点路线的消息 MLP 与时序注意力
在结构上只能看到 `[mem_src ‖ mem_dst ‖ time_enc(Δt)]`（病灶 2：被饿着）。

本模块产出**只用 `(src, dst, timestamp)` 三列即可算**的边特征，把 `edge_feat_dim` 从 0 变成 >0。

四维特征（全部**因果**：第 i 个事件只用**严格早于 i** 的事件）
--------------------------------------------------------
1. `pair_prior_count`  节点对历史交互频次（= 文献 TNCN 的 **0-hop CN**；对数化）
2. `deg_src_prior`     源节点在 t 之前的度数（对数化）
3. `deg_dst_prior`     目标节点在 t 之前的度数（对数化）
4. `cn_prior`          节点对的公共邻居数（只数**双方都已在 t 之前出现过**的邻居；对数化）

文献依据（证据强度见 `docs/PLAN3_LIT_MAP.md`）
--------------------------------------------
- **2406.07926**【已读原文】：0-hop CN = 节点对历史交互频次是其明确定义；
  且**表 7 证明"裸拼到节点表示上全部掉点"** ⇒ 本模块只负责**产出**特征，
  **不负责注入方式**；注入必须用**零初始化的可学习投影**，不得裸拼接。
- **2602.00596**【已读原文】：唯一做 node 通道 vs edge 通道对照的实验 ⇒
  信号必须落在**边通道**（注意力 key/value 侧）。
- **2504.11386**【已读原文】：在**完全无任何边属性**的图上并联信息流有正面实测（+2.57/+3.45 AP）。

为什么"严格早于"而不是"不晚于"
------------------------------
时间戳可能并列。若允许用**同一时间戳但更早索引**的事件，就引入了同一时刻的
互相可见性——这是本项目最忌讳的泄漏面（AGENTS.md §6「严格按时间划分」）。
故本模块一律按**索引**严格递增：算第 i 个事件的特征时，只把 `0..i-1` 写进状态。
这样"不依赖未来"是**构造性**保证，而不是靠约定。

用法
----
    from data.edge_features import causal_edge_features
    F = causal_edge_features(src, dst, ts)          # (E, 4) float32
"""

from __future__ import annotations

import numpy as np

#: 输出列的顺序（与 `causal_edge_features(..., cols=...)` 的选择顺序一致）
ALL_COLS: tuple[str, ...] = ("pair_prior_count", "deg_src_prior", "deg_dst_prior", "cn_prior")

#: 本模块输出的完整维度（`--edge-feat-dim` 应取这个数）
FULL_DIM = len(ALL_COLS)


def causal_edge_features(src: np.ndarray, dst: np.ndarray, ts: np.ndarray, *,
                         cols: tuple[str, ...] = ALL_COLS,
                         log1p: bool = True) -> np.ndarray:
    """逐事件产出因果边特征，返回 ``(E, len(cols))`` 的 float32 数组。

    参数
    ----
    src, dst : (E,) 整数节点 id
    ts       : (E,) 时间戳；**调用方必须保证已按 ts 非降序排好**
               （本函数只按索引推进状态，不重新排序；乱序会静默产出错误特征，
               故此处在入口显式断言单调，宁可报错也不静默算错）
    cols     : 取哪些列，见 ``ALL_COLS``
    log1p    : 是否对计数做 log1p（默认 True，与项目的其它计数特征一致）
    """
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

        # ---- 先只用**已处理**的状态产出特征（因果）----
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

        # ---- 再把本次事件写进状态（供 i 之后的事件使用）----
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
    """为“邻居槽位”建 `(u, v, t) -> 特征行号` 的索引，并返回特征矩阵。

    为什么需要它：`CausalGraph.neighbors_before(x, cut)` 只返回 `(邻居 id, 交互时刻)`，
    **不携带该次交互的边特征**。而 D2 要给注意力的每个邻居槽位配一行边特征，
    所以需要把 `(x, 邻居, t)` 映回事件行。

    重复键的处置（**关键，且已实测**）：同一 `(u, v, t)` 可能在流里出现多次
    （CollegeMsg 实测 119,590 个键里有 **80** 个重复，占 0.067%）。
    **保留最早一次**（`setdefault`）而不是覆盖：最早那次事件用的先验事件是其真子集，
    而邻居查询本身要求 `t < cut` ⇒ 附上的特征**只用到候选时刻之前的事件**，
    是**保守**而非泄漏。若改成覆盖，则会用上重复事件中较晚那次的先验（含同刻更晚索引的事件）
    —— 那就是泄漏。
    """
    F = causal_edge_features(src, dst, ts, cols=cols)
    key2row: dict[tuple[int, int, float], int] = {}
    for i in range(len(src)):
        u, v, t = int(src[i]), int(dst[i]), float(ts[i])
        key2row.setdefault((u, v, t), i)      # 保留最早 ⇒ 因果保守
        key2row.setdefault((v, u, t), i)
    return key2row, F


def feature_names(cols: tuple[str, ...] = ALL_COLS, *, log1p: bool = True) -> list[str]:
    """返回输出列名（供文档/日志用）。``log1p=True`` 时加上前缀以区分计数本身。"""
    return [f"log1p_{c}" for c in cols] if log1p else list(cols)
