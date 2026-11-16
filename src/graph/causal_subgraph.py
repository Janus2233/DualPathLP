
from __future__ import annotations

import numpy as np

__all__ = ["CausalGraph", "drnl_labels", "structural_features"]


class CausalGraph:

    def __init__(self, src, dst, ts, n_nodes: int):
        src = np.asarray(src, dtype=np.int64)
        dst = np.asarray(dst, dtype=np.int64)
        ts = np.asarray(ts, dtype=np.float64)
        u = np.concatenate([src, dst])
        v = np.concatenate([dst, src])
        t = np.concatenate([ts, ts])
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
        mask = (self.source == x) & (self.time < cut)
        nb, nt = self.neighbor[mask], self.time[mask]
        order = np.argsort(nt, kind="stable")
        nb, nt = nb[order], nt[order]
        if m:
            nb, nt = nb[-m:], nt[-m:]
        return nb, nt


    def neighbors_before(self, x: int, cut: float, m: int | None = 10,
                         window: float | None = None):
        lo, hi = self.starts[x], self.starts[x + 1]
        if hi <= lo:
            empty_i = np.empty(0, dtype=np.int64)
            return empty_i, np.empty(0, dtype=np.float64)
        nb = self.neighbor[lo:hi]
        nt = self.time[lo:hi]
        j = int(np.searchsorted(nt, cut))
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
        lo, hi = self.starts[x], self.starts[x + 1]
        if hi <= lo:
            return 0
        nt = self.time[lo:hi]
        j = int(np.searchsorted(nt, cut))
        if j <= 0:
            return 0
        k = int(np.searchsorted(nt, cut - window)) if window is not None else 0
        return max(0, j - k)

    def last_time_before(self, x: int, cut: float) -> float:
        _, nt = self.neighbors_before(x, cut, m=1)
        return float(nt[-1]) if nt.size else float("nan")


    def extract(self, u: int, v: int, t: float, hop: int = 3, m: int = 10,
                max_nodes: int | None = None, window: float | None = None):
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
            keep = ~(((a == 0) & (b == 1)) | ((a == 1) & (b == 0)))
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
