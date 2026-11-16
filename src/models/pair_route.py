"""启发式特征分支（计划 2 的 D1 实验，`--route pair`）。

**它回答什么**：把「候选对级因果特征」（AA / Jaccard / 两端窗口内因果度 / u–v 历史交互）
单独喂给一个两层 MLP，能到多少 AP？——即**增益里有多少来自手工启发式**，
而不是来自子图消息传递。这是审稿人必问、而现稿只有消融间接回答的问题。

设计取舍（写清理由，便于复核）：
  * **不构造子图**：只算候选对特征，成本远低于子图路线（省掉 k-hop BFS / DRNL / 结构统计）。
  * **不做 BatchNorm**（这是被实测打回来的选择，见 BUGLOG B44）：初版加了 `BatchNorm1d`，结果
    ①评测时若忘了切 `.eval()`（BN 仍用批内统计）→ test AP **0.9991**，比这 7 维特征在**同一协议**下的
    理论上限还高（离线逻辑回归复现：test AP **0.9503**、逐 batch 平均 0.9503）；②补上 `.eval()` 后
    训练/评测统计不匹配 → test AP **0.3535、AUC 0.129**（排序反了）。两次都是 BN 在「每批恰好
    200 正 + 200 负」这种构造下的病态行为。而线性模型在**原始特征**上就能到 0.95 ⇒ 直接不归一化，
    让第一层 Linear 自己学每维尺度（等价于逻辑回归），既没有批依赖也没有模式不匹配。
  * 本模块只在 `--route pair` 时实例化；**默认路径（fused/node/sub）完全不经过它**。
  * **超参沿用旗舰配置**（lr / epochs / window 与同数据集旗舰一致，不单独调参）——
    偏保守：若启发式单独就能接近子图路线，那结论只会更强。
  * 本模块只在 `--route pair` 时实例化；**默认路径（fused/node/sub）完全不经过它**。
"""

from __future__ import annotations

import torch
import torch.nn as nn


class PairRoute(nn.Module):
    """候选对特征 → MLP → logit。

    Args:
        pair_dim: 输入维度（由 `--pair-dims` 决定，默认 `aa,jac,deg,hist` = 7）。
        hidden:   隐藏层宽度（默认 32，与子图路线同宽，便于对照）。
        dropout:  隐藏层后的 dropout 概率。
    """

    def __init__(self, pair_dim: int, hidden: int = 32, dropout: float = 0.1):
        super().__init__()
        if pair_dim <= 0:
            raise ValueError(f"pair_dim 必须为正，收到 {pair_dim}")
        self.pair_dim = int(pair_dim)
        self.net = nn.Sequential(
            nn.Linear(self.pair_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, pair_feat: torch.Tensor) -> torch.Tensor:
        """``pair_feat``: ``(B, pair_dim)`` → ``(B,)`` logits。"""
        return self.net(pair_feat).squeeze(-1)      # 不做归一化：第一层 Linear 自己学每维尺度（B44）
