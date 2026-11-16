
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

from src.models.time_encoding import TimeEncode

__all__ = ["SubgraphRoute", "collate_subgraphs"]


class SubgraphRoute(nn.Module):

    def __init__(
        self,
        hidden: int = 32,
        max_z: int = 100,
        num_layers: int = 2,
        time_dim: int = 64,
        feat_dim: int = 0,
        pair_dim: int = 0,
        dropout: float = 0.1,
        use_edge_gate: bool = True,
    ):
        super().__init__()
        self.hidden = hidden
        self.max_z = max_z
        self.feat_dim = feat_dim
        self.pair_dim = pair_dim
        self.use_edge_gate = use_edge_gate
        self.dropout = dropout

        self.z_embedding = nn.Embedding(max_z, hidden)
        self.time_encoder = TimeEncode(time_dim)
        if use_edge_gate:
            self.edge_gate = nn.Sequential(
                nn.Linear(time_dim, hidden), nn.ReLU(), nn.Linear(hidden, 1)
            )
        in_dim = hidden + feat_dim
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_dim, hidden))
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden, hidden))
        self.decoder = nn.Sequential(
            nn.Linear(hidden + pair_dim, hidden), nn.ReLU(), nn.Linear(hidden, 1)
        )

    def forward(
        self,
        z: torch.Tensor,
        edge_index: torch.Tensor,
        edge_dt: torch.Tensor,
        center: torch.Tensor,
        node_feat: torch.Tensor | None = None,
        pair_feat: torch.Tensor | None = None,
        return_pooled: bool = False,
    ) -> torch.Tensor:
        x = self.z_embedding(z)
        if self.feat_dim:
            assert node_feat is not None and node_feat.shape[1] == self.feat_dim
            x = torch.cat([x, node_feat], dim=1)

        if self.use_edge_gate and edge_dt.numel():
            edge_weight = torch.sigmoid(self.edge_gate(self.time_encoder(edge_dt)))
            edge_weight = edge_weight.squeeze(-1)
        else:
            edge_weight = None

        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index, edge_weight=edge_weight)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)

        h_src = x[center[:, 0]]
        h_dst = x[center[:, 1]]
        h = h_src * h_dst
        if self.pair_dim:
            assert pair_feat is not None and pair_feat.shape[1] == self.pair_dim
            h = torch.cat([h, pair_feat], dim=-1)
        logits = self.decoder(h).squeeze(-1)
        return (logits, h) if return_pooled else logits


def collate_subgraphs(subgraphs, device: torch.device, feat_dim: int = 0, pair_dim: int = 0):
    z_list, feat_list = [], []
    ei_list, dt_list = [], []
    pair_list = []
    centers = []
    node_offset = 0
    ptr = [0]
    for item in subgraphs:
        nodes, edge_index, edge_dt, z, feat = item[:5]
        n = len(nodes)
        z_list.append(z)
        if feat_dim:
            feat_list.append(feat if feat is not None else torch.zeros(n, feat_dim))
        if edge_index.shape[1]:
            ei_list.append(edge_index + node_offset)
        dt_list.append(edge_dt)
        if pair_dim:
            pair_list.append(item[5])
        centers.append([node_offset, node_offset + 1])
        node_offset += n
        ptr.append(node_offset)

    z_cat = torch.as_tensor(_cat(z_list), dtype=torch.long, device=device)
    edge_index_cat = (
        torch.as_tensor(_cat(ei_list, axis=1), dtype=torch.long, device=device)
        if ei_list
        else torch.zeros((2, 0), dtype=torch.long, device=device)
    )
    dt_cat = torch.as_tensor(_cat(dt_list), dtype=torch.float32, device=device)
    center_cat = torch.as_tensor(centers, dtype=torch.long, device=device)
    feat_cat = (
        torch.as_tensor(_cat(feat_list), dtype=torch.float32, device=device)
        if feat_dim
        else None
    )
    pair_cat = (
        torch.as_tensor(np.asarray(pair_list, dtype=np.float32), device=device)
        if pair_dim
        else None
    )
    if pair_dim:
        return z_cat, edge_index_cat, dt_cat, center_cat, feat_cat, ptr, pair_cat
    return z_cat, edge_index_cat, dt_cat, center_cat, feat_cat, ptr


def _cat(arrays, axis: int = 0):
    import numpy as np

    if not arrays:
        return np.zeros(0)
    return np.concatenate([np.asarray(a) for a in arrays], axis=axis)
