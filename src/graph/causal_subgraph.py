"""因果封闭子图：构建、k-hop 提取、DRNL 标号、结构统计。

实现依据（全部来自 ``docs/DECISIONS.md``）：

- **D9.1**：因果 k-hop 封闭子图；每节点每跳取最近 ``m`` 条边；**严格 ``< t``**；
  邻接**无向登记**（正反都算）；src–dst 之间的目标边在标号前移除；正负候选同一口径。
- **D9.2**：DRNL 标号（不可达节点标 0），嵌入由模型侧负责。
- **R13 的教训**：不为正样本开"含同刻边"的后门——本模块只有一条时间规则：``ts < t``。

本模块**纯 numpy、不依赖 torch、不做 I/O**，以便自检（``src/checks/check_causal_subgraph.py``）
与主实验共用同一份实现。
"""

from __future__ import annotations

import numpy as np

__all__ = ["CausalGraph", "drnl_labels", "structural_features"]


class CausalGraph:
    """按时间升序组织的**无向**邻接，支持"某时刻之前的最近 m 个邻居"查询与 k-hop 子图提取。

    Args:
        src, dst, ts: 边数组（1-based 节点 id，与数据层产物一致）。
        n_nodes: 节点数（不含 0 号 padding 行时按 id 最大值给）。
    """

    def __init__(self, src, dst, ts, n_nodes: int):
        src = np.asarray(src, dtype=np.int64)
        dst = np.asarray(dst, dtype=np.int64)
        ts = np.asarray(ts, dtype=np.float64)
        # 无向登记
        u = np.concatenate([src, dst])
        v = np.concatenate([dst, src])
        t = np.concatenate([ts, ts])
        # **关键**：CSR 要求“先按 source 分组，组内再按时间升序”。
        # 早期版本只按时间排序（`argsort(t)`），却用计数排序建 offsets，
        # 导致 starts[x] 切出的是任意一段连续边（**不是 x 的邻居**），
        # 子图里因此出现自环与无关节点。见 docs/BUGLOG.md B14。
        order = np.lexsort((t, u))
        u, v, t = u[order], v[order], t[order]

        self.neighbor = v
        self.time = t
        self.source = u
        starts = np.zeros(n_nodes + 2, dtype=np.int64)
        np.add.at(starts, u + 1, 1)
        np.cumsum(starts, out=starts)
        self.starts = starts
        self.n_nodes = int(n_nodes)
    def validate_csr(self, n_probe: int = 64, seed: int = 0) -> None:
        """自检：抽查若干节点，确认 CSR 切片确实等于“source == x”的条目集合。

        这是对“计数排序必须配合按 source 分组”的直接验证；
        ``src/checks/check_causal_subgraph.py`` 会调它。
        """
        rng = np.random.RandomState(seed)
        xs = rng.choice(self.n_nodes, size=min(n_probe, self.n_nodes), replace=False) + 1
        for x in xs.tolist():
            lo, hi = int(self.starts[x]), int(self.starts[x + 1])
            if not np.all(self.source[lo:hi] == x):
                raise AssertionError(
                    f"CSR 损坏：节点 {x} 的切片 [{lo},{hi}) 含 source != {x} 的条目"
                )
            total = int((self.source == x).sum())
            if hi - lo != total:
                raise AssertionError(f"CSR 损坏：节点 {x} 的邻居数 {hi - lo} != 实际 {total}")
            if hi - lo > 1 and np.any(np.diff(self.time[lo:hi]) < 0):
                raise AssertionError(f"CSR 损坏：节点 {x} 的邻居未按时间升序")
        return None

    def brute_neighbors_before(self, x: int, cut: float, m: int | None = 10):
        """暴力法（O(E)，仅供自检对照）——与 CSR 路径相互独立。"""
        mask = (self.source == x) & (self.time < cut)
        nb, nt = self.neighbor[mask], self.time[mask]
        order = np.argsort(nt, kind="stable")
        nb, nt = nb[order], nt[order]
        if m:
            nb, nt = nb[-m:], nt[-m:]
        return nb, nt

    # ------------------------------------------------------------------ #
    # 邻居查询
    # ------------------------------------------------------------------ #

    def neighbors_before(self, x: int, cut: float, m: int | None = 10,
                         window: float | None = None):
        """返回 ``(neighbors, times)``：节点 ``x`` 在 ``cut`` **之前**的最近 ``m`` 条交互。

        - ``m=None`` 表示不截断；
        - ``window`` 非空时只取 ``[cut - window, cut)`` 内的交互（**硬时间窗**，D9.1 修订后的主线）。

        时间严格小于 ``cut``；越界切片返回空数组。
        """
        lo, hi = self.starts[x], self.starts[x + 1]
        if hi <= lo:
            empty_i = np.empty(0, dtype=np.int64)
            return empty_i, np.empty(0, dtype=np.float64)
        nb = self.neighbor[lo:hi]
        nt = self.time[lo:hi]
        j = int(np.searchsorted(nt, cut))  # 严格 < cut
        if j <= 0:
            return nb[:0], nt[:0]
        k = int(np.searchsorted(nt, cut - window)) if window is not None else 0
        if k >= j:
            return nb[:0], nt[:0]
        nb, nt = nb[k:j], nt[k:j]
        if m and len(nb) > m:
            nb, nt = nb[-m:], nt[-m:]
        return nb, nt

    def count_before(self, x: int, cut: float, window: float | None = None) -> int:
        """``cut`` 之前（或 ``[cut - window, cut)`` 内）的交互**条数**——只做二分，不复制切片。

        语义与 `neighbors_before(..., m=None)` 的长度严格一致（同一套边界口径），
        供 `src/graph/pair_features.py` 算 Adamic-Adar 度数用。
        """
        lo, hi = self.starts[x], self.starts[x + 1]
        if hi <= lo:
            return 0
        nt = self.time[lo:hi]
        j = int(np.searchsorted(nt, cut))          # 严格 < cut
        if j <= 0:
            return 0
        k = int(np.searchsorted(nt, cut - window)) if window is not None else 0
        return max(0, j - k)

    def last_time_before(self, x: int, cut: float) -> float:
        """节点 ``x`` 在 ``cut`` 之前的最后一次交互时间（没有则返回 ``nan``）。"""
        _, nt = self.neighbors_before(x, cut, m=1)
        return float(nt[-1]) if nt.size else float("nan")

    # ------------------------------------------------------------------ #
    # k-hop 子图提取
    # ------------------------------------------------------------------ #

    def extract(self, u: int, v: int, t: float, hop: int = 3, m: int = 10,
                max_nodes: int | None = None, window: float | None = None):
        """提取 (u, v) 在时刻 ``t`` 的因果 k-hop 封闭子图。

        Returns:
            ``(nodes, edge_index, edge_dt, center)``

            - ``nodes``: ``(N,)`` 子图内节点 id（``nodes[0] == u``、``nodes[1] == v``）
            - ``edge_index``: ``(2, E)`` 子图内边（**已移除** src–dst 之间的边）
            - ``edge_dt``: ``(E,)`` 每条边的 ``t - ts_e``（**恒 > 0**）
            - ``center``: ``(2,)`` 恒为 ``[0, 1]``（center pooling 用，保留以便将来重排）
        """
        idx = {int(u): 0, int(v): 1}
        nodes = [int(u), int(v)]
        frontier = [int(u), int(v)]
        ea: list[int] = []
        eb: list[int] = []
        edt: list[float] = []
        seen_edge: set[tuple[int, int, float]] = set()

        for _ in range(hop):
            new_frontier: list[int] = []
            for x in frontier:
                nb, nt = self.neighbors_before(x, t, m, window=window)
                for y, tt in zip(nb.tolist(), nt.tolist()):
                    key = (x, y, tt) if x <= y else (y, x, tt)
                    if key in seen_edge:
                        continue
                    seen_edge.add(key)
                    if y not in idx:
                        if max_nodes is not None and len(nodes) >= max_nodes:
                            continue
                        idx[y] = len(nodes)
                        nodes.append(y)
                        new_frontier.append(y)
                    ea.append(idx[x])
                    eb.append(idx[y])
                    edt.append(t - tt)
            frontier = new_frontier
            if not frontier:
                break

        if ea:
            a = np.asarray(ea, dtype=np.int64)
            b = np.asarray(eb, dtype=np.int64)
            d = np.asarray(edt, dtype=np.float64)
            keep = ~(((a == 0) & (b == 1)) | ((a == 1) & (b == 0)))  # 移除目标边
            a, b, d = a[keep], b[keep], d[keep]
            edge_index = np.stack([a, b], axis=0)
        else:
            edge_index = np.zeros((2, 0), dtype=np.int64)
            d = np.zeros(0, dtype=np.float64)

        return (
            np.asarray(nodes, dtype=np.int64),
            edge_index,
            d,
            np.asarray([0, 1], dtype=np.int64),
        )


def drnl_labels(nodes: np.ndarray, edge_index: np.ndarray, max_z: int = 100) -> np.ndarray:
    """DRNL 标号（SEAL 的定义）：``z = 1 + min(ds,dd) + floor(d/2)*(floor(d/2)+d%2-1)``。

    约定：``nodes[0]`` 与 ``nodes[1]`` 是目标两端，恒标 1；不可达节点标 0；结果截断到 ``max_z-1``。
    """
    n = len(nodes)
    z = np.zeros(n, dtype=np.int64)
    if n == 0:
        return z
    adj: list[list[int]] = [[] for _ in range(n)]
    for a, b in zip(edge_index[0].tolist(), edge_index[1].tolist()):
        adj[a].append(b)
        adj[b].append(a)

    def bfs(s: int) -> np.ndarray:
        dist = np.full(n, -1, dtype=np.int64)
        dist[s] = 0
        q = [s]
        while q:
            nq: list[int] = []
            for x in q:
                dx = dist[x]
                for y in adj[x]:
                    if dist[y] < 0:
                        dist[y] = dx + 1
                        nq.append(y)
            q = nq
        return dist

    ds = bfs(0)
    dd = bfs(1)
    reach = (ds >= 0) & (dd >= 0)
    d = ds + dd
    q = d // 2
    r = d % 2
    zz = 1 + np.minimum(ds, dd) + q * (q + r - 1)
    z[reach] = zz[reach]
    if n >= 1:
        z[0] = 1
    if n >= 2:
        z[1] = 1
    return np.clip(z, 0, max_z - 1)


def structural_features(graph: CausalGraph, nodes: np.ndarray, edge_index: np.ndarray,
                        edge_dt: np.ndarray, t: float, tau: float) -> np.ndarray:
    """候选②（D8.1 里被降级的那个）的**因果**结构统计特征，逐子图节点计算。

    三个维度（全部只用到 ``t`` 之前的边）：

    1. 子图内的度：``log1p(deg)``
    2. 时间衰减度：``log1p(Σ_e exp(−Δt_e / τ))``（τ = 训练期平均交互间隔）
    3. 陈旧度：``log1p((t − 该节点在 t 前最后一次交互) / τ)``，无历史则记为 0

    Returns: ``(N, 3)`` float32
    """
    n = len(nodes)
    out = np.zeros((n, 3), dtype=np.float32)
    if n == 0:
        return out
    deg = np.bincount(edge_index.ravel(), minlength=n).astype(np.float64)
    decay = np.zeros(n, dtype=np.float64)
    if edge_dt.size:
        w = np.exp(-edge_dt / max(tau, 1e-12))
        np.add.at(decay, edge_index[0], w)
        np.add.at(decay, edge_index[1], w)
    out[:, 0] = np.log1p(deg)
    out[:, 1] = np.log1p(decay)
    for k, node in enumerate(nodes.tolist()):
        last = graph.last_time_before(node, t)
        out[k, 2] = 0.0 if np.isnan(last) else np.log1p(max(t - last, 0.0) / max(tau, 1e-12))
    return out
