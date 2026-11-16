"""计划 3 · D1 的**按路线分位数校准**（纯 numpy，无 torch）。

D1 想解决的问题
---------------
病灶 1：门控拿不到“该信哪一路”的判别信息。而 `|p_node − p_sub|` **已经在**冻结的
7 维 ctx 里（`src/models/dual_route.py:121`）却仍然坍塌（E1 复核结论 1）。
⇒ 缺的不是“分歧”这个标量，而是**参照系**：两路分数各自的尺度不同，
“这个候选上节点路线的这一分算高还是低”**需要拿它跟它自己那条路线的分布比**。

本模块给出的就是把分数映射成“它在这条路线自己的分布里的位置”（分位数位置），
再取两路位置之差送进门控。

为什么是**分位数**而不是 z-score
--------------------------------
z-score 假设分布近似正态；而 AP 场景下的分数分布明显偏斜（大量低分 + 长尾高分）。
分位数是**单调、无分布假设**的，且天然落在 [0,1]，两路可直接相减。

协议安全（关键，两条硬约束）
--------------------------
1. **只用训练流拟合、拟合后冻结**：`fit()` 之后 `transform()` 不再改动内部状态。
   刻意**不用验证集**——验证集目前只用于早停（选最佳轮次），若再拿它拟合一个会进入
   训练梯度的变换，等于把验证集的使用强度升级，属于协议层问题（Q2）。
2. **不依赖批内其他候选**：`transform()` 是**逐候选**的查表/插值，
   **不使用批统计量** ⇒ 最终分数不依赖批构成 ⇒ 不触 Q2。

与门控预热的配合（这条让 D1 的设计变得干净）
------------------------------------------
训练脚本里 `warmup = epoch <= --warmup-epochs`，而预热期 `g ≡ 0.5`（**完全绕过 ctx**）。
因此“在预热结束时用训练流拟合一次校准”满足：
  * 用的全是模型**已经见过**的数据；
  * 此刻门控**还没有从 ctx 学到任何东西** ⇒ 不存在“用校准去拟合门控、又用门控去拟合校准”的循环；
  * 拟合完成后冻结，训练/验证/测试三处用的是**同一个**变换 ⇒ 无训练-推理不一致。

退化处置（必须显式，不得静默）
----------------------------
若某条路线在拟合集上的分数是**常数**（`n_distinct == 1`），分位数映射无信息。
此时 `transform` 一律返回 **0.5**，并在 `FitReport.degenerate` 里**置真**——
调用方应当把它当失败信号报出来，而不是让它悄悄变成一个恒为 0 的输入维
（计划 2 的臂 C 就因“覆盖率退化成常数”而返工）。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: 分位数网格点数（101 ⇒ 每 1% 一个断点，线性插值）
DEFAULT_GRID = 101


@dataclass(frozen=True)
class FitReport:
    """一次拟合的诊断结果（**必须随结论一起报告**）。"""

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
    """按路线拟合的分位数校准；`fit` 一次后冻结。"""

    def __init__(self, n_grid: int = DEFAULT_GRID):
        if n_grid < 2:
            raise ValueError("n_grid 至少为 2")
        self.n_grid = int(n_grid)
        self.grid = np.linspace(0.0, 1.0, self.n_grid)
        self.breaks: dict[str, np.ndarray] = {}
        self.reports: dict[str, FitReport] = {}

    # ------------------------------------------------------------------ #
    def fit(self, route: str, scores: np.ndarray) -> FitReport:
        """用给定分数拟合该路线的分位点。**调用一次，之后冻结。**"""
        s = np.asarray(scores, dtype=np.float64).ravel()
        if s.size == 0:
            raise ValueError(f"路线 {route} 的拟合分数为空")
        if not np.all(np.isfinite(s)):
            raise ValueError(f"路线 {route} 的拟合分数含非有限值")
        n_distinct = int(np.unique(s).size)
        degenerate = n_distinct == 1
        if degenerate:
            # 常数分布：分位数无意义，退化为常量 0.5（由 transform 的常量 breaks 实现）
            breaks = np.full(self.n_grid, float(s[0]), dtype=np.float64)
        else:
            breaks = np.quantile(s, self.grid)
        self.breaks[route] = breaks
        rep = FitReport(route=route, n=int(s.size), n_distinct=n_distinct,
                        degenerate=degenerate, vmin=float(s.min()), vmax=float(s.max()))
        self.reports[route] = rep
        return rep

    # ------------------------------------------------------------------ #
    def transform(self, route: str, scores: np.ndarray) -> np.ndarray:
        """把分数映射为“它在该路线自己分布里的分位位置”，值域 [0,1]。

        逐候选、无批依赖；断点外按端点截断（`np.interp` 的默认行为）。
        未拟合的路线直接报错——不静默用别的路线顶上。
        """
        if route not in self.breaks:
            raise KeyError(f"路线 {route} 尚未 fit；已拟合：{sorted(self.breaks)}")
        s = np.asarray(scores, dtype=np.float64)
        br = self.breaks[route]
        if np.all(br == br[0]):                     # 退化：常量映射
            return np.full(s.shape, 0.5, dtype=np.float64)
        out = np.interp(s, br, self.grid, left=0.0, right=1.0)
        return np.clip(out, 0.0, 1.0)

    # ------------------------------------------------------------------ #
    def transform_torch(self, route: str, x):
        """与 `transform` **数学等价**的 torch 实现（前向用，避免 GPU→CPU 同步）。

        为什么要有两份：拟合/审计用 numpy（不依赖 torch）；前向在 GPU 上跑，
        逐 batch 把张量搬到 CPU 再搬回去会产生同步开销。两份实现必须逐位对得上——
        这一点由 `check_route_calibration` 的**例 9** 直接对账（而不是靠眼睛看公式）。
        """
        import torch                                   # 懒引用：numpy 路径不需要 torch
        if route not in self.breaks:
            raise KeyError(f"路线 {route} 尚未 fit；已拟合：{sorted(self.breaks)}")
        br = torch.as_tensor(self.breaks[route], dtype=x.dtype, device=x.device)
        grid = torch.as_tensor(self.grid, dtype=x.dtype, device=x.device)
        if bool((br == br[0]).all()):
            return torch.full_like(x, 0.5)
        # `torch.searchsorted(right=False)` 与 np.interp 的区间选择一致：
        # 找 br[j-1] <= x < br[j]，再在 [j-1, j] 上线性插值，两端截断。
        j = torch.searchsorted(br, x, right=False).clamp(1, br.numel() - 1)
        x0, x1 = br[j - 1], br[j]
        y0, y1 = grid[j - 1], grid[j]
        t = (x - x0) / (x1 - x0).clamp_min(torch.finfo(x.dtype).tiny)
        y = y0 + t * (y1 - y0)
        y = torch.where(x <= br[0], grid[0].expand_as(y), y)
        y = torch.where(x >= br[-1], grid[-1].expand_as(y), y)
        return y.clamp(0.0, 1.0)

    # ------------------------------------------------------------------ #
    def relative(self, route_a: str, a_scores: np.ndarray,
                 route_b: str, b_scores: np.ndarray) -> np.ndarray:
        """两路的**相对位置差** `q_a − q_b` ∈ [−1, 1]——这就是 D1 要送进门控的那一维。"""
        return self.transform(route_a, a_scores) - self.transform(route_b, b_scores)

    def relative_torch(self, route_a: str, a_scores, route_b: str, b_scores):
        """`relative` 的 torch 版（前向用），与 numpy 版等价。"""
        return (self.transform_torch(route_a, a_scores)
                - self.transform_torch(route_b, b_scores))

    # ------------------------------------------------------------------ #
    def state_dict(self) -> dict:
        """可落盘/可 resume 的纯数据形式（不含对象）。"""
        return {"n_grid": self.n_grid,
                "breaks": {k: v.tolist() for k, v in self.breaks.items()},
                "reports": {k: v.as_dict() for k, v in self.reports.items()}}

    def load_state_dict(self, st: dict) -> None:
        self.n_grid = int(st["n_grid"])
        self.grid = np.linspace(0.0, 1.0, self.n_grid)
        self.breaks = {k: np.asarray(v, dtype=np.float64) for k, v in st["breaks"].items()}
        self.reports = {k: FitReport(**v) for k, v in st.get("reports", {}).items()}


# --------------------------------------------------------------------------- #
def selftest() -> int:
    """免数据、免 GPU 的自检。"""
    fails: list[str] = []
    rng = np.random.default_rng(0)

    c = RouteCalibrator()
    # 明显偏斜的分布（AP 场景的典型形状）
    s = np.concatenate([rng.beta(2, 7, size=4000), rng.random(200) * 0.5 + 0.5])
    rep = c.fit("node", s)
    if rep.degenerate:
        fails.append("例1 不应判为退化")
    q = c.transform("node", s)
    if q.min() < 0.0 or q.max() > 1.0:
        fails.append(f"例1 分位位置越界：[{q.min():.3f},{q.max():.3f}]")

    # 单调性：分数升序 ⇒ 分位位置非降
    ss = np.sort(s)
    qq = c.transform("node", ss)
    if np.any(np.diff(qq) < -1e-12):
        fails.append("例2 校准非单调（分数升高反而位置下降）")

    # 中位数应映到 ≈0.5（分位数定义的自洽性）
    if abs(float(c.transform("node", np.array([np.median(s)]))[0]) - 0.5) > 0.02:
        fails.append("例2 中位数应映到 ≈0.5")

    # 冻结性：再 transform 别的数据不得改变内部断点
    before = c.breaks["node"].copy()
    c.transform("node", rng.normal(size=100))
    if not np.array_equal(before, c.breaks["node"]):
        fails.append("例3 transform 改动了断点 ⇒ 校准没有被冻结")

    # fit-once 语义：同一路线二次 fit 会覆盖（这是有意的），但必须由调用方控制
    c2 = RouteCalibrator()
    c2.fit("sub", np.array([0.1, 0.2, 0.3, 0.4]))
    if not np.allclose(c2.transform("sub", np.array([0.25])), 0.5, atol=0.05):
        fails.append("例4 简单均匀分数下 0.25 应接近 0.5")

    # 退化：常数分数必须返回 0.5 并在报告里置真（不得静默）
    c3 = RouteCalibrator()
    rep3 = c3.fit("flat", np.full(100, 0.7))
    if not rep3.degenerate or rep3.n_distinct != 1:
        fails.append("例5 常数分数应判为退化且 n_distinct=1")
    if not np.allclose(c3.transform("flat", np.array([0.1, 0.7, 0.9])), 0.5):
        fails.append("例5 退化路线的 transform 应恒为 0.5")

    # 未拟合的路线必须报错
    try:
        c3.transform("nope", np.array([1.0]))
        fails.append("例6 未拟合的路线未报错")
    except KeyError:
        pass

    # 空/非有限输入必须报错
    for bad, tag in ((np.array([]), "空"), (np.array([np.nan]), "NaN"),
                     (np.array([np.inf]), "Inf")):
        try:
            RouteCalibrator().fit("x", bad)
            fails.append(f"例6 {tag} 输入未报错")
        except ValueError:
            pass

    # relative ∈ [−1,1]，且两路同分布时≈0
    ca = RouteCalibrator()
    a = rng.random(3000)
    ca.fit("A", a)
    ca.fit("B", a.copy())          # 同分布
    rel = ca.relative("A", a, "B", a.copy())
    if rel.min() < -1.0 - 1e-12 or rel.max() > 1.0 + 1e-12:
        fails.append("例7 relative 越界")
    if np.abs(rel).max() > 1e-9:
        fails.append("例7 两路同分布时 relative 应恒为 0")

    # state_dict 往返
    st = ca.state_dict()
    cb = RouteCalibrator()
    cb.load_state_dict(st)
    if not np.allclose(cb.transform("A", a), ca.transform("A", a)):
        fails.append("例8 state_dict 往返后 transform 不一致")

    # 例 9 —— numpy 与 torch 两份实现必须对得上（否则前向与拟合就不是同一个变换）
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
    # 退化路线上两份也必须一致（都返回 0.5）
    cflat = RouteCalibrator(); cflat.fit("flat", np.full(50, 0.3))
    if not np.allclose(cflat.transform_torch("flat", torch.zeros(4)).numpy(), 0.5):
        fails.append("例9 退化路线的 torch 实现应恒为 0.5")
    # NaN 必须报错，不得静默返回垃圾
    try:
        cc.fit("nan", np.array([1.0, np.nan]))
        fails.append("例9 NaN 未报错")
    except ValueError:
        pass
    if fails:
        for f in fails:
            print(f"[FAIL] {f}")
        return 1
    print(f"[route_calibration] 9 组断言全部通过（网格 {DEFAULT_GRID} 点、单调、冻结、退化显式）")  # noqa: F541
    return 0


if __name__ == "__main__":
    raise SystemExit(selftest())
