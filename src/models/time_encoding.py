"""时间编码 ``cos(W·Δt + b)``（两条路线共用）。

出处：``参考模型/TGN-SEAL/model/time_encoding.py:11-18,26``（与 TGAT 的 ``TimeEncode`` 同源）。
``W`` 初始化为 ``1/10^linspace(0,9,dim)``、``b = 0``。

按 `DECISIONS.md` §7.3 的建议，**两条路线共用同一套参数**（`src/models/subgraph_route.py` 与
`src/models/node_route.py` 各自持有一个实例；是否共享权重留待后续消融）。
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

__all__ = ["TimeEncode", "linspace_freqs"]


def linspace_freqs(dim: int) -> np.ndarray:
    return 1.0 / 10 ** np.linspace(0, 9, dim)


class TimeEncode(nn.Module):
    """``cos(W·Δt + b)``，``W`` 按参考实现初始化。"""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.w = nn.Linear(1, dim)
        with torch.no_grad():
            self.w.weight.copy_(torch.from_numpy(linspace_freqs(dim)).reshape(dim, 1).float())
            self.w.bias.zero_()

    def forward(self, dt: torch.Tensor) -> torch.Tensor:
        # 时间戳/时间差一律用 float64 传进来（float32 在 1e9 秒量级上 ULP≈128s，会丢精度），
        # 这里在进 Linear 前降到参数精度。见 BUGLOG B15。
        if dt.dtype != self.w.weight.dtype:
            dt = dt.to(self.w.weight.dtype)
        return torch.cos(self.w(dt.unsqueeze(-1)))
