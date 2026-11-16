"""节点路线（实现 D8 / D8.1 / D8-配置表）。

结构：

    逐 batch 按时间顺序处理
      ① 用「当前记忆」算候选两端的时序嵌入（时序图注意力，k 个最近邻居）
      ② 打分 → 反传
      ③ 为 batch 内的边生成 raw message（detach），更新记忆（GRU）——供下一个 batch 用

**因果性靠"批边界"保证**：记忆只在 batch 结束时更新，因此任一 batch 用到的记忆只包含
**严格早于该 batch** 的边。`memory_watermark` 记录已进入记忆的最大时间戳，训练/评测脚本会断言
`min(候选 ts) >= watermark`（见 `src/checks/check_node_route.py`）。这正是 TGN 的
`memory_update_at_start` 语义（`参考模型/TGN-SEAL/train_self_supervised.py:104`）。

配置依据（`docs/DECISIONS.md` D8-配置表）：记忆维度 172、`identity` 消息、`last` 聚合、GRU 更新、
注意力 1 层 2 头、邻居数 k=10、时间编码 `cos(W·Δt+b)`（W ~ `1/10^linspace(0,9,dim)`）。

与参考实现的两处**有意差异**（均为省算力，且写在此处便于复核）：

1. **不带 172 维边特征**（我们的边特征实测全零，见 `EXPERIMENTS.md`；带进去只是给 GRU 喂死重量）。
   预留 `edge_feat_dim` 开关，设为 172 即可退回参考形态。
2. 邻居查询用我们自己的 `CausalGraph`（CSR 已修 B14），**不使用** TGN-SEAL 有泄漏的
   `get_node_max_ts`（`BUGLOG.md` R1）。
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.time_encoding import TimeEncode

__all__ = ["NodeMemory", "NodeRoute"]


class NodeMemory(nn.Module):
    """节点记忆：GRU 状态 + 最后更新时刻 + raw message 暂存。"""

    def __init__(self, n_nodes: int, dim: int, device: torch.device):
        super().__init__()
        self.n_nodes = int(n_nodes)
        self.dim = int(dim)
        self.device = device
        self.register_buffer("memory", torch.zeros(self.n_nodes + 1, dim))
        self.register_buffer("last_update", torch.zeros(self.n_nodes + 1, dtype=torch.long))
        self.raw_messages: dict[int, tuple[torch.Tensor, float]] = {}
        self.watermark: float = float("-inf")  # 已进入记忆的最大时间戳

    def init_memory(self) -> None:
        self.memory.zero_()
        self.last_update.zero_()
        self.raw_messages.clear()
        self.watermark = float("-inf")

    def get(self, nodes: torch.Tensor) -> torch.Tensor:
        return self.memory[nodes]

    def store(self, nodes: torch.Tensor, messages: torch.Tensor, ts) -> None:
        """暂存 raw message（聚合器 = last：同一节点在 batch 内只保留**时刻最大**的一条）。

        `ts` 可以是标量或与 `nodes` 等长的数组（每个节点各自的边时刻）。
        """
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
        """把暂存的 raw message 一次性写进记忆（GRU 更新）。返回更新的节点数。

        `grad=False`（默认＝冻结路径）：更新在 `no_grad` 下做、写回 `detach()`，
        与历史 run **逐位一致**。

        `grad=True`（BUGLOG **B64** 修复）：保留 GRU 的计算图写回记忆，使**下一个 batch**
        的损失能反传到 `memory_updater` 的参数。

        为什么必须“延迟一步”：记忆是在**打分之后**才更新的，所以当批损失永远不经过它；
        单去掉 `no_grad` 拿不到任何梯度。真正的通路是——本 batch 写入的记忆带着图，
        被**下一个 batch 的 `embed()` 读到**，于是那一步的 `backward()` 能落到 GRU 参数上。
        调用方必须在 `backward()` 之后调用 `detach()` 切断图，否则退化成跨 batch 的 BPTT
        （显存随轮数增长）。
        """
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
            # 必须用**越出式**写回：`self.memory` 是 buffer（叶、不需梯度），
            # 就地 `self.memory[idx] = ...` 不会被 autograd 记录；`index_copy` 会建
            # IndexCopyBackward 节点，梯度才能流到 `updater` 的参数。
            new_mem = updater(msgs, self.memory[idx])
            self.memory = self.memory.index_copy(0, idx, new_mem)
        self.last_update[idx] = ts.to(self.device)
        self.raw_messages.clear()
        return len(nodes)

    def detach(self) -> None:
        # 越出式：grad 模式下 `memory` 带 grad_fn，就地 `detach_()` 对非叶张量不安全
        self.memory = self.memory.detach()
        self.last_update.detach_()


class NodeRoute(nn.Module):
    """记忆 + 时序图注意力的节点路线；输出 `p_node` 的 logit。"""

    def __init__(
        self,
        n_nodes: int,
        memory_dim: int = 172,
        time_dim: int = 100,
        edge_feat_dim: int = 0,
        # 计划 3 · D2：**只走边通道**的边特征维度（0 = 关）。
        # 与 `edge_feat_dim` 的区别很重要：后者同时把边特征送进 `raw_message`⇒GRUCell
        # （即**节点表示侧**），而 2406.07926 表 7 已实测“把结构量拼到节点侧全都掉点”；
        # 2602.00596 的 node/edge 通道对照也表明信号该落在**边通道**。
        # ⇒ 本开关只注入注意力的 key/value，且用**零初始化投影**、不做裸拼接。
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
        # I4 消融：解码器是否吃 time_enc(Δt)（D8-配置表写的是吃；关掉即“陈旧度只进嵌入”）
        self.decoder_dt = bool(decoder_dt)

        self.edge_msg_dim = int(edge_msg_dim)
        # 计划 3 · D2：**只注入边通道**的两个零初始化投影。
        # 零初始化 ⇒ 初始化时 forward 与基线**逐位相同**（加 0 是精确运算），
        # 因此“打开开关”不会在 t=0 就把模型改坏；能不能学到东西由梯度决定。
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

        # 时序注意力：query = [mem_x ‖ time_enc(0)]；key/value = [mem_n ‖ time_enc(Δt) ‖ edge_feat]
        key_dim = memory_dim + time_dim + edge_feat_dim
        self.q_proj = nn.Linear(memory_dim + time_dim, memory_dim)
        self.k_proj = nn.Linear(key_dim, memory_dim)
        self.v_proj = nn.Linear(key_dim, memory_dim)
        self.out_proj = nn.Linear(memory_dim * 2, memory_dim)
        self.dropout = nn.Dropout(dropout)

        # p_node 解码器（D8-配置表：MLP([h_u ‖ h_v ‖ time_enc(Δt_last)]) → 1）
        dec_in = memory_dim * 2 + (time_dim if self.decoder_dt else 0)
        self.decoder = nn.Sequential(
            nn.Linear(dec_in, memory_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(memory_dim, 1),
        )

    # ------------------------------------------------------------------ #
    # 消息与记忆
    # ------------------------------------------------------------------ #

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
        """按参考实现为源端与目标端各存一条 raw message（detach）。

        时间差取 **`边时刻 − 该节点上次更新时刻`**（照 TGN 的 `get_raw_messages`），
        而不是 0 —— 这是“该节点多久没动”的信号，写错会明显削弱记忆。
        """
        src = torch.as_tensor(src, dtype=torch.long, device=self.device)
        dst = torch.as_tensor(dst, dtype=torch.long, device=self.device)
        # **必须 float64**：float32 在 1e9 秒量级的 ULP≈128s，会让记忆水位与 last_update 偏掉（BUGLOG B15）
        ts_t = torch.as_tensor(np.asarray(ts, dtype=np.float64), dtype=torch.float64,
                               device=self.device)
        dt_src = (ts_t - self.memory.last_update[src].double()).clamp(min=0.0)
        dt_dst = (ts_t - self.memory.last_update[dst].double()).clamp(min=0.0)
        ef = edge_feat if edge_feat is not None else None
        m_src = self.raw_message(self.memory.get(src), self.memory.get(dst), ef, dt_src)
        m_dst = self.raw_message(self.memory.get(dst), self.memory.get(src), ef, dt_dst)
        self.memory.store(src, m_src, ts_t)
        self.memory.store(dst, m_dst, ts_t)

    # ------------------------------------------------------------------ #
    # 嵌入（时序图注意力）
    # ------------------------------------------------------------------ #

    def embed(self, nodes: torch.Tensor, times: torch.Tensor, nbr_ids: list, nbr_ts: list,
              nbr_feat: list | None = None) -> torch.Tensor:
        """对 `nodes`（B 个）在 `times` 处做一次时序邻居注意力，返回 (B, memory_dim)。

        `nbr_ids[i]` / `nbr_ts[i]` 是节点 i 的邻居 id 与交互时间（均已按时间升序、只含 < t 的边）。
        """
        b = len(nbr_ids)
        device = self.device
        k = self.n_neighbors
        mem = self.memory.get(nodes)                                  # (B, d)
        q = self.q_proj(torch.cat([mem, self.time_encoder(torch.zeros(b, device=device))], dim=-1))

        # 先在 CPU/numpy 里把 (b, k) 的三块表填好，再**一次性**搬到 device：
        # 旧写法在循环里逐行 `torch.as_tensor(...)`，一个 batch 就是 2·b 次小张量创建 + H2D，
        # 在 GPU 上光 launch 开销就能吃掉大部分时间（node-only 跑一轮从 ~15 min 掉到 ~2 min）。
        nbr_np = np.zeros((b, k), dtype=np.int64)
        dt_np = np.zeros((b, k), dtype=np.float32)
        mask_np = np.zeros((b, k), dtype=bool)
        for i, (ids, tss, t0) in enumerate(zip(nbr_ids, nbr_ts, times.tolist())):
            if len(ids) == 0:
                continue
            m = min(len(ids), k)
            nbr_np[i, :m] = ids[-m:]
            dt_np[i, :m] = [t0 - x for x in tss[-m:]]      # float64 相减后落到 float32，与旧写法一致
            mask_np[i, :m] = True
        nbr = torch.as_tensor(nbr_np, device=device)
        dt = torch.as_tensor(dt_np, device=device)
        mask = torch.as_tensor(mask_np, device=device)

        nbr_mem = self.memory.get(nbr.reshape(-1)).reshape(b, k, -1)   # (B, k, d)
        kv_parts = [nbr_mem, self.time_encoder(dt.reshape(-1)).reshape(b, k, -1)]
        if self.edge_feat_dim and nbr_feat is not None:
            kv_parts.append(nbr_feat)
        kv = torch.cat(kv_parts, dim=-1)
        key = self.k_proj(kv)                                          # (B, k, d)
        value = self.v_proj(kv)

        # 计划 3 · D2：边通道的**零初始化加性注入**（不是拼接）。
        # - 只加到 key/value ⇒ 信号落在**边通道**，不进节点表示（2602.00596 的通道对照）
        # - 零初始化 ⇒ t=0 时 `key + 0` 与 `value + 0` **逐位等于**基线（加 0 是精确的）
        # - 加性而非拼接 ⇒ 不改变 k_proj/v_proj 的输入宽度，旧权重全部保留
        if self.edge_msg_dim and nbr_feat is not None:
            key = key + self.edge_k(nbr_feat)
            value = value + self.edge_v(nbr_feat)

        logits = (q.unsqueeze(1) * key).sum(-1) / (self.memory_dim ** 0.5)  # (B, k)
        logits = logits.masked_fill(~mask, -1e9)
        attn = torch.softmax(logits, dim=-1)
        attn = self.dropout(attn)
        out = (attn.unsqueeze(-1) * value).sum(1)                       # (B, d)
        # 空邻居行必须清零：mask 全 False 时 softmax 得到**均匀权重**，会把填充位（0 号节点）的记忆
        # 混进结果——那是“幻影邻居”。见 src/checks/check_node_route.py 第 5 项。
        empty = ~mask.any(dim=-1, keepdim=True)                         # (B, 1)
        out = torch.where(empty, torch.zeros_like(out), out)
        h = self.out_proj(torch.cat([mem, out], dim=-1))
        return h

    def score(self, h_src: torch.Tensor, h_dst: torch.Tensor, dt_last: torch.Tensor) -> torch.Tensor:
        """p_node 的 logit：MLP([h_u ‖ h_v ‖ time_enc(Δt_last)])；`decoder_dt=False` 时去掉最后一项。"""
        parts = [h_src, h_dst] + ([self.time_encoder(dt_last)] if self.decoder_dt else [])
        return self.decoder(torch.cat(parts, dim=-1)).squeeze(-1)
