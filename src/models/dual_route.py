
from __future__ import annotations

import math

import torch
import torch.nn as nn

__all__ = ["GatedFusion", "context_features", "ctx_dim", "CTX_DIM",
           "consistency_loss", "js_divergence", "CONS_KINDS"]

CTX_DIM = 7


class GatedFusion(nn.Module):

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
        if warmup:
            g = torch.full_like(logit_node, 0.5)
        elif self.const_gate:
            g = torch.sigmoid(self.g_logit).expand_as(logit_node)
        else:
            ctx = ctx.to(self.mlp[0].weight.dtype)
            g = torch.sigmoid(self.mlp(ctx)).squeeze(-1)
        p_node = torch.sigmoid(logit_node)
        p_sub = torch.sigmoid(logit_sub)
        p_fused = g * p_sub + (1.0 - g) * p_node
        eps = 1e-6
        logit_fused = torch.logit(p_fused.clamp(eps, 1 - eps))
        return p_fused, g, logit_fused


class FeatureFusion(nn.Module):

    def __init__(self, in_dim: int, hidden: int = 32, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim + 2, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, 1))

    def forward(self, logit_node: torch.Tensor, logit_sub: torch.Tensor, ctx: torch.Tensor,
                warmup: bool = False):
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
    for extra in (pair_feat, stat_feat, meta_feat, emb_feat, calib_feat):
        if extra is not None:
            out = torch.cat([out, extra], dim=-1)
    return out


def ctx_dim(use_dt: bool = True, extra: int = 0) -> int:
    return (CTX_DIM if use_dt else CTX_DIM - 2) + int(extra)



def _log_prob_pair(logit: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.nn.functional.logsigmoid(logit), torch.nn.functional.logsigmoid(-logit)


def js_divergence(logit_a: torch.Tensor, logit_b: torch.Tensor) -> torch.Tensor:
    la, l1a = _log_prob_pair(logit_a)
    lb, l1b = _log_prob_pair(logit_b)
    pa, pb = la.exp(), lb.exp()
    lm = torch.logaddexp(la, lb) - math.log(2.0)
    l1m = torch.logaddexp(l1a, l1b) - math.log(2.0)
    kl_a = pa * (la - lm) + (1.0 - pa) * (l1a - l1m)
    kl_b = pb * (lb - lm) + (1.0 - pb) * (l1b - l1m)
    return (0.5 * (kl_a + kl_b)).clamp(min=0.0)


CONS_KINDS = ("jsd", "mse_logit", "mse_prob")


def consistency_loss(logit_node: torch.Tensor, logit_sub: torch.Tensor,
                     kind: str = "jsd") -> torch.Tensor:
    assert kind in CONS_KINDS, f"未知的一致性损失类型：{kind}（可选 {CONS_KINDS}）"
    if kind == "jsd":
        return js_divergence(logit_node, logit_sub).mean()
    if kind == "mse_logit":
        return torch.mean((logit_node - logit_sub) ** 2)
    return torch.mean((torch.sigmoid(logit_node) - torch.sigmoid(logit_sub)) ** 2)
