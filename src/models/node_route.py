
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.time_encoding import TimeEncode

__all__ = ["NodeMemory", "NodeRoute"]


class NodeMemory(nn.Module):

    def __init__(self, n_nodes: int, dim: int, device: torch.device):
        super().__init__()
        self.n_nodes = int(n_nodes)
        self.dim = int(dim)
        self.device = device
        self.register_buffer("memory", torch.zeros(self.n_nodes + 1, dim))
        self.register_buffer("last_update", torch.zeros(self.n_nodes + 1, dtype=torch.long))
        self.raw_messages: dict[int, tuple[torch.Tensor, float]] = {}
        self.watermark: float = float("-inf")

    def init_memory(self) -> None:
        self.memory.zero_()
        self.last_update.zero_()
        self.raw_messages.clear()
        self.watermark = float("-inf")

    def get(self, nodes: torch.Tensor) -> torch.Tensor:
        return self.memory[nodes]

    def store(self, nodes: torch.Tensor, messages: torch.Tensor, ts) -> None:
        nodes_cpu = nodes.detach().cpu()
        msgs = messages.detach()
        ts_cpu = ts.detach().cpu() if isinstance(ts, torch.Tensor) else ts
        ts_arr = np.atleast_1d(np.asarray(ts_cpu, dtype=np.float64))
        for k, n in enumerate(nodes_cpu.tolist()):
            t_k = float(ts_arr[k]) if ts_arr.size > 1 else float(ts_arr[0])
            prev = self.raw_messages.get(int(n))
            if prev is None or t_k >= prev[1]:
                self.raw_messages[int(n)] = (msgs[k], t_k)
        self.watermark = max(self.watermark, float(ts_arr.max()))

    def apply(self, updater: nn.GRUCell, grad: bool = False) -> int:
        if not self.raw_messages:
            return 0
        nodes = sorted(self.raw_messages.keys())
        idx = torch.tensor(nodes, dtype=torch.long, device=self.device)
        msgs = torch.stack([self.raw_messages[n][0] for n in nodes]).to(self.device)
        ts = torch.tensor([self.raw_messages[n][1] for n in nodes], dtype=torch.long)
        if not grad:
            with torch.no_grad():
                new_mem = updater(msgs, self.memory[idx])
            self.memory[idx] = new_mem.detach()
        else:
            new_mem = updater(msgs, self.memory[idx])
            self.memory = self.memory.index_copy(0, idx, new_mem)
        self.last_update[idx] = ts.to(self.device)
        self.raw_messages.clear()
        return len(nodes)

    def detach(self) -> None:
        self.memory = self.memory.detach()
        self.last_update.detach_()


class NodeRoute(nn.Module):

    def __init__(
        self,
        n_nodes: int,
        memory_dim: int = 172,
        time_dim: int = 100,
        edge_feat_dim: int = 0,
        edge_msg_dim: int = 0,
        n_neighbors: int = 10,
        n_heads: int = 2,
        dropout: float = 0.1,
        device: torch.device | None = None,
        message_function: str = "identity",
        decoder_dt: bool = True,
    ):
        super().__init__()
        self.n_nodes = int(n_nodes)
        self.memory_dim = memory_dim
        self.time_dim = time_dim
        self.edge_feat_dim = edge_feat_dim
        self.n_neighbors = n_neighbors
        self.n_heads = n_heads
        self.device = device or torch.device("cpu")
        self.message_function = message_function
        self.decoder_dt = bool(decoder_dt)

        self.edge_msg_dim = int(edge_msg_dim)
        if self.edge_msg_dim:
            self.edge_k = nn.Linear(self.edge_msg_dim, memory_dim)
            self.edge_v = nn.Linear(self.edge_msg_dim, memory_dim)
            nn.init.zeros_(self.edge_k.weight); nn.init.zeros_(self.edge_k.bias)
            nn.init.zeros_(self.edge_v.weight); nn.init.zeros_(self.edge_v.bias)

        raw_dim = 2 * memory_dim + edge_feat_dim + time_dim
        self.raw_dim = raw_dim
        self.memory = NodeMemory(self.n_nodes, memory_dim, self.device)
        self.memory_updater = nn.GRUCell(raw_dim, memory_dim)
        if message_function == "mlp":
            self.message_mlp = nn.Sequential(nn.Linear(raw_dim, raw_dim), nn.ReLU())
        self.time_encoder = TimeEncode(time_dim)

        key_dim = memory_dim + time_dim + edge_feat_dim
        self.q_proj = nn.Linear(memory_dim + time_dim, memory_dim)
        self.k_proj = nn.Linear(key_dim, memory_dim)
        self.v_proj = nn.Linear(key_dim, memory_dim)
        self.out_proj = nn.Linear(memory_dim * 2, memory_dim)
        self.dropout = nn.Dropout(dropout)

        dec_in = memory_dim * 2 + (time_dim if self.decoder_dt else 0)
        self.decoder = nn.Sequential(
            nn.Linear(dec_in, memory_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(memory_dim, 1),
        )


    def raw_message(self, src_mem, dst_mem, edge_feat, dt) -> torch.Tensor:
        parts = [src_mem, dst_mem]
        if self.edge_feat_dim:
            parts.append(edge_feat)
        parts.append(self.time_encoder(dt))
        msg = torch.cat(parts, dim=-1)
        if self.message_function == "mlp":
            msg = self.message_mlp(msg)
        return msg

    def store_messages(self, src, dst, ts, edge_feat=None) -> None:
        src = torch.as_tensor(src, dtype=torch.long, device=self.device)
        dst = torch.as_tensor(dst, dtype=torch.long, device=self.device)
        ts_t = torch.as_tensor(np.asarray(ts, dtype=np.float64), dtype=torch.float64,
                               device=self.device)
        dt_src = (ts_t - self.memory.last_update[src].double()).clamp(min=0.0)
        dt_dst = (ts_t - self.memory.last_update[dst].double()).clamp(min=0.0)
        ef = edge_feat if edge_feat is not None else None
        m_src = self.raw_message(self.memory.get(src), self.memory.get(dst), ef, dt_src)
        m_dst = self.raw_message(self.memory.get(dst), self.memory.get(src), ef, dt_dst)
        self.memory.store(src, m_src, ts_t)
        self.memory.store(dst, m_dst, ts_t)


    def embed(self, nodes: torch.Tensor, times: torch.Tensor, nbr_ids: list, nbr_ts: list,
              nbr_feat: list | None = None) -> torch.Tensor:
        b = len(nbr_ids)
        device = self.device
        k = self.n_neighbors
        mem = self.memory.get(nodes)
        q = self.q_proj(torch.cat([mem, self.time_encoder(torch.zeros(b, device=device))], dim=-1))

        nbr_np = np.zeros((b, k), dtype=np.int64)
        dt_np = np.zeros((b, k), dtype=np.float32)
        mask_np = np.zeros((b, k), dtype=bool)
        for i, (ids, tss, t0) in enumerate(zip(nbr_ids, nbr_ts, times.tolist())):
            if len(ids) == 0:
                continue
            m = min(len(ids), k)
            nbr_np[i, :m] = ids[-m:]
            dt_np[i, :m] = [t0 - x for x in tss[-m:]]
            mask_np[i, :m] = True
        nbr = torch.as_tensor(nbr_np, device=device)
        dt = torch.as_tensor(dt_np, device=device)
        mask = torch.as_tensor(mask_np, device=device)

        nbr_mem = self.memory.get(nbr.reshape(-1)).reshape(b, k, -1)
        kv_parts = [nbr_mem, self.time_encoder(dt.reshape(-1)).reshape(b, k, -1)]
        if self.edge_feat_dim and nbr_feat is not None:
            kv_parts.append(nbr_feat)
        kv = torch.cat(kv_parts, dim=-1)
        key = self.k_proj(kv)
        value = self.v_proj(kv)

        if self.edge_msg_dim and nbr_feat is not None:
            key = key + self.edge_k(nbr_feat)
            value = value + self.edge_v(nbr_feat)

        logits = (q.unsqueeze(1) * key).sum(-1) / (self.memory_dim ** 0.5)
        logits = logits.masked_fill(~mask, -1e9)
        attn = torch.softmax(logits, dim=-1)
        attn = self.dropout(attn)
        out = (attn.unsqueeze(-1) * value).sum(1)
        empty = ~mask.any(dim=-1, keepdim=True)
        out = torch.where(empty, torch.zeros_like(out), out)
        h = self.out_proj(torch.cat([mem, out], dim=-1))
        return h

    def score(self, h_src: torch.Tensor, h_dst: torch.Tensor, dt_last: torch.Tensor) -> torch.Tensor:
        parts = [h_src, h_dst] + ([self.time_encoder(dt_last)] if self.decoder_dt else [])
        return self.decoder(torch.cat(parts, dim=-1)).squeeze(-1)
