"""双路线融合（实现 D7 / D10.1 / D10.2）。

    p = g · p_subgraph + (1 − g) · p_node,     g = σ(MLP(ctx))

`ctx`（D10.1 定稿，全部**推理时可得**）：

1. `log1p(子图节点数)`
2. `子图边数 / 节点数`
3. `log1p(Δt_u)`、`log1p(Δt_v)`：两端距上次交互的时间（记忆陈旧度）
4. `log1p(历史交互数_u)`、`log1p(历史交互数_v)`（**不用**“是否属于归纳留出新节点”——D10.1 排掉的坑）
5. `|p_node − p_subgraph|`（两路分歧）

对照下界：`const_gate=True` 时只学一个标量 g（= Neo-GNNs 那一类“可学但与上下文无关”的融合）。

**跨路线一致性监督（I6，`--lambda-cons`，默认 0 = 关闭）**：除各自的 BCE 与融合 BCE 外，
可额外加一项**对称散度**把两条路线的**分数分布**拉近（互蒸馏），而不是只靠门控在输出端加权：

    L = BCE(p_fused) + λ_n·BCE(p_node) + λ_s·BCE(p_sub) + λ_c·D(p_node ‖ p_sub)

`D` 默认取 **Jensen-Shannon 散度**（对称、有界于 `[0, ln 2]`、p=q 时梯度为 0），
实现全部在 logit 空间用 `logsigmoid/logaddexp` 计算，极端 logit（±50）下也不出 NaN。

> 与 D10.2 的张力：**每路各自校准要求两路独立，一致性要求两路趋同**——
> 因此它是**默认为关的消融**，不是主线；开与不开必须各自报数（见 `EXPERIMENTS.md`）。
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

__all__ = ["GatedFusion", "context_features", "ctx_dim", "CTX_DIM",
           "consistency_loss", "js_divergence", "CONS_KINDS"]

CTX_DIM = 7  # 见上方 1–5（其中 3、4 各占两维）


class GatedFusion(nn.Module):
    """标量 sigmoid 门控（D10.1）。`warmup=True` 时把 g 固定为 0.5（D10.2 的门控预热）。"""

    def __init__(self, ctx_dim: int = CTX_DIM, hidden: int = 32, const_gate: bool = False):
        super().__init__()
        self.const_gate = const_gate
        if const_gate:
            self.g_logit = nn.Parameter(torch.zeros(()))
        else:
            self.mlp = nn.Sequential(
                nn.Linear(ctx_dim, hidden), nn.ReLU(), nn.Linear(hidden, 1)
            )

    def forward(self, logit_node: torch.Tensor, logit_sub: torch.Tensor, ctx: torch.Tensor,
                warmup: bool = False):
        """返回 ``(p_fused, g, logit_fused)``；`p_fused` 是概率、`logit_fused` 供 BCE 使用。"""
        if warmup:
            g = torch.full_like(logit_node, 0.5)
        elif self.const_gate:
            g = torch.sigmoid(self.g_logit).expand_as(logit_node)
        else:
            # ctx 里的时间量是 float64（见 BUGLOG B15），进 MLP 前降到参数精度
            ctx = ctx.to(self.mlp[0].weight.dtype)
            g = torch.sigmoid(self.mlp(ctx)).squeeze(-1)
        p_node = torch.sigmoid(logit_node)
        p_sub = torch.sigmoid(logit_sub)
        p_fused = g * p_sub + (1.0 - g) * p_node
        # 融合分数用概率的 logit 形式喂给 BCEWithLogitsLoss（数值稳定）
        eps = 1e-6
        logit_fused = torch.logit(p_fused.clamp(eps, 1 - eps))
        return p_fused, g, logit_fused


class FeatureFusion(nn.Module):
    """特征级融合头（计划2 D0 的 `--fusion feat`）：不做「两概率凸组合」，直接学一个判别函数。

    输入 = ``[ctx（可含富上下文扩展）, logit_node, logit_sub]``，输出 = ``logit_fused``，**没有标量门 g**。
    它回答的问题是：融合的瓶颈在**门的输入**（`rich` 臂）还是在**融合的形式**（本臂）？
    代价：失去 ḡ 这个可解释量——论文里的「坍塌刻画」依赖它，所以本模式只作对照臂，
    除作者明确决定外不进主线。

    ``warmup`` 参数保留是为了与 `GatedFusion` 同签名：没有门可预热，故忽略。
    """

    def __init__(self, in_dim: int, hidden: int = 32, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim + 2, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, 1))

    def forward(self, logit_node: torch.Tensor, logit_sub: torch.Tensor, ctx: torch.Tensor,
                warmup: bool = False):
        """返回 ``(None, None, logit_fused)``——与 `GatedFusion` 的三元组签名保持一致。"""
        h = torch.cat([ctx.to(self.net[0].weight.dtype),
                       logit_node.unsqueeze(-1), logit_sub.unsqueeze(-1)], dim=-1)
        return None, None, self.net(h).squeeze(-1)


def context_features(sub_nodes: torch.Tensor, sub_edges: torch.Tensor,
                     dt_u: torch.Tensor, dt_v: torch.Tensor,
                     cnt_u: torch.Tensor, cnt_v: torch.Tensor,
                     logit_node: torch.Tensor, logit_sub: torch.Tensor,
                     use_dt: bool = True, pair_feat: torch.Tensor | None = None,
                     stat_feat: torch.Tensor | None = None,
                     meta_feat: torch.Tensor | None = None,
                     emb_feat: torch.Tensor | None = None,
                     calib_feat: torch.Tensor | None = None) -> torch.Tensor:
    """拼出 D10.1 的 7 维上下文（`use_dt=False` 时去掉两维陈旧度，返回 5 维，作 I4 消融）。

    计划2 起支持**富上下文**：按 `pair_feat`（候选对因果特征）/ `stat_feat`（子图内结构统计的均值±标准差）
    / `meta_feat`（路由元信息，如窗口覆盖率）三组依次追加，三者都为 None 时输出与改造前**逐位相同**。
    """
    ratio = torch.where(sub_nodes > 0, sub_edges / sub_nodes.clamp(min=1.0),
                        torch.zeros_like(sub_nodes))
    parts = [
        torch.log1p(sub_nodes),
        ratio,
    ]
    if use_dt:
        parts += [torch.log1p(dt_u), torch.log1p(dt_v)]
    parts += [
        torch.log1p(cnt_u),
        torch.log1p(cnt_v),
        (torch.sigmoid(logit_node) - torch.sigmoid(logit_sub)).abs(),
    ]
    out = torch.stack(parts, dim=-1)
    # 富上下文扩展（计划2 D0/D2）：按组追加，顺序固定 = pair → stat → meta。
    # 默认（三个都为 None）时 out 逐位不变 ⇒ 默认路径不受影响。
    # 计划3 · D1：最后一组是**按路线分位数校准**后的相对位置差（默认 None ⇒ 逐位不变）。
    for extra in (pair_feat, stat_feat, meta_feat, emb_feat, calib_feat):
        if extra is not None:
            out = torch.cat([out, extra], dim=-1)
    return out


def ctx_dim(use_dt: bool = True, extra: int = 0) -> int:
    """`context_features` 输出维度（随 `use_dt` 与 `extra`（富上下文扩展组）变化）。"""
    return (CTX_DIM if use_dt else CTX_DIM - 2) + int(extra)


# --------------------------------------------------------------------------- #
# I6：跨路线一致性监督（默认关闭）
# --------------------------------------------------------------------------- #

def _log_prob_pair(logit: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """返回 (log σ(x), log(1 − σ(x)))，用 `logsigmoid` 保证极端 logit 下不溢出。"""
    return torch.nn.functional.logsigmoid(logit), torch.nn.functional.logsigmoid(-logit)


def js_divergence(logit_a: torch.Tensor, logit_b: torch.Tensor) -> torch.Tensor:
    """两个 Bernoulli 分布之间的 **Jensen-Shannon 散度**（逐样本，返回 (B,)）。

    对称、非负、上界 `ln 2`、`a == b` 时**恒为 0 且梯度为 0**。
    全程在 logit 空间算，`logit = ±50` 也不产生 NaN/Inf（无 `log(0)`）。
    """
    la, l1a = _log_prob_pair(logit_a)
    lb, l1b = _log_prob_pair(logit_b)
    pa, pb = la.exp(), lb.exp()
    # m = (p_a + p_b) / 2，用 logaddexp 保证数值稳定
    lm = torch.logaddexp(la, lb) - math.log(2.0)
    l1m = torch.logaddexp(l1a, l1b) - math.log(2.0)
    kl_a = pa * (la - lm) + (1.0 - pa) * (l1a - l1m)
    kl_b = pb * (lb - lm) + (1.0 - pb) * (l1b - l1m)
    # 数值上可能出现 -1e-9 这类极小负值（浮点误差），clamp 到 0
    return (0.5 * (kl_a + kl_b)).clamp(min=0.0)


CONS_KINDS = ("jsd", "mse_logit", "mse_prob")


def consistency_loss(logit_node: torch.Tensor, logit_sub: torch.Tensor,
                     kind: str = "jsd") -> torch.Tensor:
    """跨路线一致性损失（标量）。`kind`：

    - `jsd`（默认）：对称 JS 散度，有界 `[0, ln 2]`，两路一致时梯度为 0；
    - `mse_logit`：`mean((a − b)²)`，对分歧惩罚更重、无上界；
    - `mse_prob`：`mean((σ(a) − σ(b))²)`，最温和。
    """
    assert kind in CONS_KINDS, f"未知的一致性损失类型：{kind}（可选 {CONS_KINDS}）"
    if kind == "jsd":
        return js_divergence(logit_node, logit_sub).mean()
    if kind == "mse_logit":
        return torch.mean((logit_node - logit_sub) ** 2)
    return torch.mean((torch.sigmoid(logit_node) - torch.sigmoid(logit_sub)) ** 2)
