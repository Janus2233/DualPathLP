
from __future__ import annotations

import torch
import torch.nn as nn


class PairRoute(nn.Module):

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
        return self.net(pair_feat).squeeze(-1)
