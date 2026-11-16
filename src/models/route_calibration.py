
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

DEFAULT_GRID = 101


@dataclass(frozen=True)
class FitReport:

    route: str
    n: int
    n_distinct: int
    degenerate: bool
    vmin: float
    vmax: float

    def as_dict(self) -> dict:
        return {"route": self.route, "n": self.n, "n_distinct": self.n_distinct,
                "degenerate": self.degenerate, "vmin": self.vmin, "vmax": self.vmax}


class RouteCalibrator:

    def __init__(self, n_grid: int = DEFAULT_GRID):
        if n_grid < 2:
            raise ValueError("n_grid 至少为 2")
        self.n_grid = int(n_grid)
        self.grid = np.linspace(0.0, 1.0, self.n_grid)
        self.breaks: dict[str, np.ndarray] = {}
        self.reports: dict[str, FitReport] = {}

    def fit(self, route: str, scores: np.ndarray) -> FitReport:
        s = np.asarray(scores, dtype=np.float64).ravel()
        if s.size == 0:
            raise ValueError(f"路线 {route} 的拟合分数为空")
        if not np.all(np.isfinite(s)):
            raise ValueError(f"路线 {route} 的拟合分数含非有限值")
        n_distinct = int(np.unique(s).size)
        degenerate = n_distinct == 1
        if degenerate:
            breaks = np.full(self.n_grid, float(s[0]), dtype=np.float64)
        else:
            breaks = np.quantile(s, self.grid)
        self.breaks[route] = breaks
        rep = FitReport(route=route, n=int(s.size), n_distinct=n_distinct,
                        degenerate=degenerate, vmin=float(s.min()), vmax=float(s.max()))
        self.reports[route] = rep
        return rep

    def transform(self, route: str, scores: np.ndarray) -> np.ndarray:
        if route not in self.breaks:
            raise KeyError(f"路线 {route} 尚未 fit；已拟合：{sorted(self.breaks)}")
        s = np.asarray(scores, dtype=np.float64)
        br = self.breaks[route]
        if np.all(br == br[0]):
            return np.full(s.shape, 0.5, dtype=np.float64)
        out = np.interp(s, br, self.grid, left=0.0, right=1.0)
        return np.clip(out, 0.0, 1.0)

    def transform_torch(self, route: str, x):
        import torch
        if route not in self.breaks:
            raise KeyError(f"路线 {route} 尚未 fit；已拟合：{sorted(self.breaks)}")
        br = torch.as_tensor(self.breaks[route], dtype=x.dtype, device=x.device)
        grid = torch.as_tensor(self.grid, dtype=x.dtype, device=x.device)
        if bool((br == br[0]).all()):
            return torch.full_like(x, 0.5)
        j = torch.searchsorted(br, x, right=False).clamp(1, br.numel() - 1)
        x0, x1 = br[j - 1], br[j]
        y0, y1 = grid[j - 1], grid[j]
        t = (x - x0) / (x1 - x0).clamp_min(torch.finfo(x.dtype).tiny)
        y = y0 + t * (y1 - y0)
        y = torch.where(x <= br[0], grid[0].expand_as(y), y)
        y = torch.where(x >= br[-1], grid[-1].expand_as(y), y)
        return y.clamp(0.0, 1.0)

    def relative(self, route_a: str, a_scores: np.ndarray,
                 route_b: str, b_scores: np.ndarray) -> np.ndarray:
        return self.transform(route_a, a_scores) - self.transform(route_b, b_scores)

    def relative_torch(self, route_a: str, a_scores, route_b: str, b_scores):
        return (self.transform_torch(route_a, a_scores)
                - self.transform_torch(route_b, b_scores))

    def state_dict(self) -> dict:
        return {"n_grid": self.n_grid,
                "breaks": {k: v.tolist() for k, v in self.breaks.items()},
                "reports": {k: v.as_dict() for k, v in self.reports.items()}}

    def load_state_dict(self, st: dict) -> None:
        self.n_grid = int(st["n_grid"])
        self.grid = np.linspace(0.0, 1.0, self.n_grid)
        self.breaks = {k: np.asarray(v, dtype=np.float64) for k, v in st["breaks"].items()}
        self.reports = {k: FitReport(**v) for k, v in st.get("reports", {}).items()}


def selftest() -> int:
    fails: list[str] = []
    rng = np.random.default_rng(0)

    c = RouteCalibrator()
    s = np.concatenate([rng.beta(2, 7, size=4000), rng.random(200) * 0.5 + 0.5])
    rep = c.fit("node", s)
    if rep.degenerate:
        fails.append("例1 不应判为退化")
    q = c.transform("node", s)
    if q.min() < 0.0 or q.max() > 1.0:
        fails.append(f"例1 分位位置越界：[{q.min():.3f},{q.max():.3f}]")

    ss = np.sort(s)
    qq = c.transform("node", ss)
    if np.any(np.diff(qq) < -1e-12):
        fails.append("例2 校准非单调（分数升高反而位置下降）")

    if abs(float(c.transform("node", np.array([np.median(s)]))[0]) - 0.5) > 0.02:
        fails.append("例2 中位数应映到 ≈0.5")

    before = c.breaks["node"].copy()
    c.transform("node", rng.normal(size=100))
    if not np.array_equal(before, c.breaks["node"]):
        fails.append("例3 transform 改动了断点 ⇒ 校准没有被冻结")

    c2 = RouteCalibrator()
    c2.fit("sub", np.array([0.1, 0.2, 0.3, 0.4]))
    if not np.allclose(c2.transform("sub", np.array([0.25])), 0.5, atol=0.05):
        fails.append("例4 简单均匀分数下 0.25 应接近 0.5")

    c3 = RouteCalibrator()
    rep3 = c3.fit("flat", np.full(100, 0.7))
    if not rep3.degenerate or rep3.n_distinct != 1:
        fails.append("例5 常数分数应判为退化且 n_distinct=1")
    if not np.allclose(c3.transform("flat", np.array([0.1, 0.7, 0.9])), 0.5):
        fails.append("例5 退化路线的 transform 应恒为 0.5")

    try:
        c3.transform("nope", np.array([1.0]))
        fails.append("例6 未拟合的路线未报错")
    except KeyError:
        pass

    for bad, tag in ((np.array([]), "空"), (np.array([np.nan]), "NaN"),
                     (np.array([np.inf]), "Inf")):
        try:
            RouteCalibrator().fit("x", bad)
            fails.append(f"例6 {tag} 输入未报错")
        except ValueError:
            pass

    ca = RouteCalibrator()
    a = rng.random(3000)
    ca.fit("A", a)
    ca.fit("B", a.copy())
    rel = ca.relative("A", a, "B", a.copy())
    if rel.min() < -1.0 - 1e-12 or rel.max() > 1.0 + 1e-12:
        fails.append("例7 relative 越界")
    if np.abs(rel).max() > 1e-9:
        fails.append("例7 两路同分布时 relative 应恒为 0")

    st = ca.state_dict()
    cb = RouteCalibrator()
    cb.load_state_dict(st)
    if not np.allclose(cb.transform("A", a), ca.transform("A", a)):
        fails.append("例8 state_dict 往返后 transform 不一致")

    import torch
    cc = RouteCalibrator()
    sk = rng.beta(2, 5, size=5000) + rng.random(5000) * 0.3
    cc.fit("node", sk)
    cc.fit("sub", rng.random(5000))
    probe = np.concatenate([
        sk[:500],
        np.array([sk.min() - 1.0, sk.max() + 1.0, float(np.median(sk))], dtype=np.float64),
        rng.random(300),
    ])
    for route in ("node", "sub"):
        a_np = cc.transform(route, probe)
        a_th = cc.transform_torch(route, torch.as_tensor(probe, dtype=torch.float64)).numpy()
        if not np.allclose(a_np, a_th, atol=1e-9):
            d = float(np.abs(a_np - a_th).max())
            fails.append(f"例9 路线 {route} 的 numpy/torch 实现最大差 {d:.3e}（应一致）")
    cflat = RouteCalibrator(); cflat.fit("flat", np.full(50, 0.3))
    if not np.allclose(cflat.transform_torch("flat", torch.zeros(4)).numpy(), 0.5):
        fails.append("例9 退化路线的 torch 实现应恒为 0.5")
    try:
        cc.fit("nan", np.array([1.0, np.nan]))
        fails.append("例9 NaN 未报错")
    except ValueError:
        pass
    if fails:
        for f in fails:
            print(f"[FAIL] {f}")
        return 1
    print(f"[route_calibration] 9 组断言全部通过（网格 {DEFAULT_GRID} 点、单调、冻结、退化显式）")
    return 0


if __name__ == "__main__":
    raise SystemExit(selftest())
