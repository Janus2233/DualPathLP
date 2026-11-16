
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

__all__ = ["TimeEncode", "linspace_freqs"]


def linspace_freqs(dim: int) -> np.ndarray:
    return 1.0 / 10 ** np.linspace(0, 9, dim)


class TimeEncode(nn.Module):

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.w = nn.Linear(1, dim)
        with torch.no_grad():
            self.w.weight.copy_(torch.from_numpy(linspace_freqs(dim)).reshape(dim, 1).float())
            self.w.bias.zero_()

    def forward(self, dt: torch.Tensor) -> torch.Tensor:
        if dt.dtype != self.w.weight.dtype:
            dt = dt.to(self.w.weight.dtype)
        return torch.cos(self.w(dt.unsqueeze(-1)))
