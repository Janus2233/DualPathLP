"""M1 —— 双路线模型（D7 融合 + D8 节点路线 + D9 子图路线）的第一组实验。

与 M0 的区别：这里两条路线**同时**跑，用 D10.1 的门控融合，并按 D10.2 的契约训练
（每路各自 BCE + 融合 BCE、负样本两路共用、门控预热、联合端到端）。

协议仍是 D2（时间 70/15/15、训练 1:1 目标端替换、评测全图池固定种子、AP+AUC 每 batch 先算再平均）。

**单路线消融**：`--route {fused,node,sub}`。取 `node` / `sub` 时只训该路线（损失 = 该路自己的 BCE，
不做门控融合），并**跳过另一路的计算**（`node` 时不做子图提取，`sub` 时不做邻居查询与记忆读取）——
这是“融合必须胜过单路线基线”的证据来源，且与 fused 共用同一份数据、负样本与评测代码。

**因果性**：节点路线按时间顺序逐 batch 处理，记忆只在 batch 结束后更新，因此任一 batch 用到的记忆
只含更早的边；脚本在每个 batch 前断言 `min(ts) >= memory.watermark`。

用法（在 ``Model experiments/`` 下）：

    & $py -m src.train.m1_dual_route --dataset college_msg --epochs 12 --tag m1_college
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from data import metrics as data_metrics
from data import negatives as data_negatives
from src.graph.causal_subgraph import CausalGraph
from src.graph.pair_features import PAIR_DEFAULT_SPEC, PAIR_DIM, parse_dims
from src.models.dual_route import (CONS_KINDS, FeatureFusion, GatedFusion, consistency_loss,
                                   context_features, ctx_dim)
from src.models.node_route import NodeRoute
from src.models.route_calibration import RouteCalibrator   # 计划3 · D1
from src.models.subgraph_route import SubgraphRoute
from src.train.m0_subgraph_only import (build_subgraphs, load_processed, run_model as run_sub_model,
                                        run_model_pooled)

ROOT = Path(__file__).resolve().parents[2]
RUNS_DIR = ROOT / "runs" / "m1"


# ---------------------------------------------------------------------------- #
# 打分
# ---------------------------------------------------------------------------- #

def meta_coverage(g_sub, src, dst, ts, win) -> np.ndarray:
    """窗口覆盖率：``(B, 2)``，各端 = 硬窗内邻居数 / 完整因果历史邻居数（无历史记 0）。

    抽成模块级函数有两个理由（都是为了**可验证**）：
      1. 能单测：默认 ``win=None`` 时覆盖率退化成二值（有历史 1.0 / 无历史 0.0），
         这不是"看起来对"就能放过的性质，要有断言把它钉住
         （见 ``src/checks/check_meta_window.py``）；
      2. 诊断工具 ``src/checks/ctx_info.py`` 要算**同一套**覆盖率，
         复用此函数而不是抄一份 —— 抄一份就会在改动后悄悄漂移。

    ``win=None`` 表示"窗口即全部历史"，此时分子分母相同 ⇒ 覆盖率只能是 0 或 1。
    """
    cols = []
    for _col in (src, dst):
        vals = []
        for x, t0 in zip(np.asarray(_col).tolist(), np.asarray(ts).tolist()):
            tot = g_sub.count_before(int(x), float(t0))
            got = g_sub.count_before(int(x), float(t0), win) if win else tot
            vals.append(got / tot if tot else 0.0)
        cols.append(vals)
    return np.stack(cols, axis=1)


def _neighbor_feat_matrix(centers, nbr_ids: list, nbr_ts: list, key2row: dict, feats,
                         k: int, device) -> torch.Tensor:
    """把“逐候选的邻居列表”转成 `(B, k, EF)` 的 **加性边特征** 矩阵（计划 3 · D2）。

    `centers[i]` 是第 i 个候选的**中心节点**（节点路线用 src/dst），用来和邻居、时刻
    一起拼出查询键 `(中心, 邻居, t)`——`nbr_ids` 里只有邻居，拼不出键，所以必须传中心。

    必须与 `NodeRoute.embed` 的口径**逐位对齐**：那边取 `ids[-m:]` / `tss[-m:]`
    （**最近 m 条**，靠后对齐），这里用相同的后缀切片；空邻居行补 0。

    查不到就 **报错**，不静默补 0：覆盖率已实测 100%（CollegeMsg 766/766），
    静默补 0 会把“映射建错了”伪装成“这个邻居本来就没信息”。
    """
    b = len(nbr_ids)
    ef_dim = feats.shape[1]
    out = np.zeros((b, k, ef_dim), dtype=np.float32)
    miss: list[tuple[int, int, float]] = []
    for i, (x, ids, tss) in enumerate(zip(centers, nbr_ids, nbr_ts)):
        m = min(len(ids), k)
        if not m:
            continue
        for j in range(m):
            v = int(ids[len(ids) - m + j])
            t = float(tss[len(tss) - m + j])
            row = key2row.get((int(x), v, t))
            if row is None:
                miss.append((int(x), v, t))
                continue
            out[i, j] = feats[row]
    if miss:
        raise AssertionError(
            f"边特征查不到 {len(miss)} 个邻居槽位（前 3 例 (中心,邻居,t)：{miss[:3]}）")
    return torch.as_tensor(out, device=device)


def ctx_size_dims(stats: dict, b: int, per_candidate: bool) -> tuple[list, list, str]:
    """融合门 ctx 前两维（子图节点数、边/节点比）的来源选择。BUGLOG **B65**（E0）。

    返回 ``(每候选节点数, 每候选边数, 来源标记)``；来源标记 ∈
    ``{"per-candidate", "batch-mean"}``，供调用方判断**是否真的用上了逐候选口径**。

    两个口径的含义与差别：
      * **批均值广播**（`per_candidate=False`，默认）：`[stats["nodes_mean"]] * b`。
        这两维在**批内为常数** ⇒ 门控按它们分组时退化成"按测试流批次位置分组"，
        §3.5 的"逐候选门控"与 §4.8 的"按子图规模四分位"都不成立（B65 的病灶）。
      * **逐候选**（`per_candidate=True`）：取 `build_subgraphs` 落下来的
        `stats["nodes"] / ["edges"]`（`m0_subgraph_only.py:125-126`），长度必须等于 `b`。

    **为什么做成开关而不是无条件修复**（2026-10-12 作者决策）：E0 会改变**全部**融合结果，
    而现存 N=8 主表（含旗舰 `scalar` 臂）全部跑在旧口径上。无条件生效等于用一次代码改动
    追溯性重写 64 个 run 的口径；做成默认关的 flag 之后，
    "逐候选门 vs 批均值门"才成为一个**可对照的实验**（E0-only 那一格），
    而冻结路径保持逐位不变 —— 与本仓库"新行为一律 flag 门控、默认关"的既有纪律一致。

    **与 B65 无条件版逐位一致**：开关闭环里的分支体与兜底条件都没有改动，只是加了一道
    `per_candidate` 判断 ⇒ `m1p3_e1pilot_nw_s42`（跑在无条件版上）等价于
    "本开关开 + `--memory-grad onestep`"，仍然可比。

    兜底：`stats` 里没有逐候选列表（非子图路径）或长度与 `b` 不符 ⇒ 退回批均值广播，
    并**如实返回来源标记**；调用方在开关打开时必须拒绝该退回（见 `score_candidates`）。
    """
    if per_candidate:
        nn = stats.get("nodes")
        ne = stats.get("edges")
        if nn is not None and ne is not None and len(nn) == b and len(ne) == b:
            return list(nn), list(ne), "per-candidate"
    return ([stats.get("nodes_mean", 0.0)] * b,
            [stats.get("edges_mean", 0.0)] * b, "batch-mean")


def score_candidates(node_route, sub_route, g_nbr, g_sub, src, dst, ts, cfg, device, feat_dim,
                     need_node=True, need_sub=True, pair_route=None, need_pair=False,
                     edge_lookup=None, calib=None, calib_on=False):
    """对一批候选打各路分数并拼出上下文；返回 ``(logit_node, logit_sub, logit_pair, ctx, sub_stats)``。

    ``need_pair=True`` 时走**启发式特征分支**（`--route pair`）：只用候选对特征，不构造子图。

    ``need_node`` / ``need_sub`` 为 False 时**跳过**对应路线（单路线消融用：
    节点路线基线可省掉占大头的子图提取，子图路线基线可省掉邻居查询与记忆读取）；
    被跳过的路线返回全 0 分，且 ``ctx`` 返回 None（此时上游不会调用融合门）。
    """
    src_t = torch.as_tensor(src, dtype=torch.long, device=device)
    dst_t = torch.as_tensor(dst, dtype=torch.long, device=device)
    t_t = torch.as_tensor(np.asarray(ts, dtype=np.float64), dtype=torch.float64, device=device)
    b = len(np.asarray(src))

    if need_node:
        nb_u_ids, nb_u_ts, nb_v_ids, nb_v_ts = [], [], [], []
        for x, y, t0 in zip(np.asarray(src).tolist(), np.asarray(dst).tolist(),
                            np.asarray(ts).tolist()):
            ids, tss = g_nbr.neighbors_before(x, float(t0), cfg["k"])
            nb_u_ids.append(ids); nb_u_ts.append(tss)
            ids, tss = g_nbr.neighbors_before(y, float(t0), cfg["k"])
            nb_v_ids.append(ids); nb_v_ts.append(tss)

        # 计划 3 · D2：只给**边通道**准备邻居边特征（零初始化加性注入）。
        # 开关关（edge_lookup=None）时这两行什么都不做 ⇒ 默认路径逐位不变。
        nbf_u = nbf_v = None
        if edge_lookup is not None and getattr(node_route, "edge_msg_dim", 0):
            _k2r, _feats = edge_lookup
            _src_l = np.asarray(src).tolist()
            _dst_l = np.asarray(dst).tolist()
            nbf_u = _neighbor_feat_matrix(_src_l, nb_u_ids, nb_u_ts, _k2r, _feats,
                                          cfg["k"], device)
            nbf_v = _neighbor_feat_matrix(_dst_l, nb_v_ids, nb_v_ts, _k2r, _feats,
                                          cfg["k"], device)

        h_u = node_route.embed(src_t, t_t, nb_u_ids, nb_u_ts, nbr_feat=nbf_u)
        h_v = node_route.embed(dst_t, t_t, nb_v_ids, nb_v_ts, nbr_feat=nbf_v)
        # 时间差用 float64（float32 在 1e9 秒量级 ULP≈128s，见 BUGLOG B15）
        last_u = node_route.memory.last_update[src_t].double()
        last_v = node_route.memory.last_update[dst_t].double()
        dt_u = (t_t - last_u).clamp(min=0.0)
        dt_v = (t_t - last_v).clamp(min=0.0)
        logit_node = node_route.score(h_u, h_v, torch.maximum(dt_u, dt_v))

        # cnt_u/cnt_v 只喂**融合门的 ctx**；单跑节点路线（need_sub=False）时 ctx 为 None，
        # 再去查一遍全量邻居就是白做——省掉它能把 node-only 的邻居查询砍掉一半。
        if need_sub:
            cnt_u = torch.as_tensor([len(g_nbr.neighbors_before(int(x), float(t0), None)[0])
                                     for x, t0 in zip(np.asarray(src).tolist(),
                                                      np.asarray(ts).tolist())],
                                    dtype=torch.float32, device=device)
            cnt_v = torch.as_tensor([len(g_nbr.neighbors_before(int(y), float(t0), None)[0])
                                     for y, t0 in zip(np.asarray(dst).tolist(),
                                                      np.asarray(ts).tolist())],
                                    dtype=torch.float32, device=device)
        else:
            cnt_u = cnt_v = torch.zeros(b, dtype=torch.float32, device=device)
    else:
        dt_u = dt_v = torch.zeros(b, dtype=torch.float64, device=device)
        cnt_u = cnt_v = torch.zeros(b, dtype=torch.float32, device=device)
        logit_node = torch.zeros(b, device=device)

    h_sub_pooled = None      # 只在 --fusion-ctx emb 时被赋值（子图路线的池化表示）
    if need_sub:
        subs, stats = build_subgraphs(g_sub, src, dst, ts, cfg["hop"], cfg["m"], cfg["max_nodes"],
                                      cfg["input_mode"], cfg["tau"], cfg["max_z"],
                                      window=cfg["window"], pair_dim=cfg.get("pair_dim", 0),
                                      pair_dims=cfg.get("pair_dims"),
                                      collect_struct=("stat" in cfg.get("fusion_groups", ())))
        if "emb" in cfg.get("fusion_groups", ()):
            # emb 组需要子图路线的池化表示；默认走 run_sub_model（一字不动的那条路径）
            logit_sub, h_sub_pooled = run_model_pooled(sub_route, subs, device, feat_dim,
                                                       cfg.get("pair_dim", 0))
        else:
            logit_sub = run_sub_model(sub_route, subs, device, feat_dim, cfg.get("pair_dim", 0))
    else:
        stats = {"nodes_mean": 0.0, "edges_mean": 0.0}
        logit_sub = torch.zeros(b, dtype=torch.float32, device=device)
    logit_pair = None
    if need_pair:
        # 启发式特征分支（计划2 D1）：只算候选对特征，**不抽子图、不进记忆**。
        # 与子图路线内部的 pair 分支用**同一个** pair_features_batch 与同一套 --pair-dims，
        # 保证「手工启发式单独能到多少」这个对照与主线口径一致。
        from src.graph.pair_features import pair_features_batch
        pf = pair_features_batch(g_sub, src, dst, ts, window=cfg["window"],
                                 dims=cfg.get("pair_dims"))
        logit_pair = pair_route(torch.as_tensor(pf, dtype=torch.float32, device=device))

    ctx = None
    if need_node and need_sub:
        # 富上下文扩展（计划2 D2 的四个臂）：按 cfg 里选的组现算，未选时不构造任何额外张量
        extras: dict = {}
        for _g in cfg.get("fusion_groups", ()):
            if _g == "pair":
                # 与子图路线内部的 pair 分支用**同一个函数、同一套 --pair-dims**，保证口径一致
                from src.graph.pair_features import pair_features_batch
                pf = pair_features_batch(g_sub, src, dst, ts, window=cfg["window"],
                                         dims=cfg.get("pair_dims"))
                extras["pair_feat"] = torch.as_tensor(pf, dtype=torch.float32, device=device)
            elif _g == "pair-shuf":
                # **安慰剂对照**（用来回答一个必然会遇到的质疑）：与 `pair` 完全同分布、同维度、
                # 同参数量，**只把批次内的行做确定性置换** —— 每列的信息量、边缘分布都不变，
                # 唯一被破坏的是"这一行特征属于这一对候选"的对应关系。
                # 若臂 A 的增益来自"门控 MLP 变宽/参数变多"而非对级信息，那么本臂应当同样有增益；
                # 反之（本臂无增益）才能把增益归因给**信息**。
                # 置换用**固定种子**：不读也不推进全局 RNG ⇒ 不改变其他任何部分的可复现性。
                from src.graph.pair_features import pair_features_batch
                pf = pair_features_batch(g_sub, src, dst, ts, window=cfg["window"],
                                         dims=cfg.get("pair_dims"))
                perm = np.random.RandomState(20261003).permutation(len(pf))
                extras["pair_feat"] = torch.as_tensor(pf[perm], dtype=torch.float32, device=device)
            elif _g == "noise":
                # **无信息容量对照**：维度与参数量与 `pair` 完全相同，但内容是 iid 噪声 ——
                # 与候选、与标签都无关。它回答"门控 MLP 变宽本身能不能带来增益"：
                # 若臂 A 的增益只是"头更大"，本臂应当同样有增益；本臂无增益才支持"来自信息"。
                #
                # 为什么**不用**行置换当安慰剂（原计划，已实测否掉，见 BUGLOG B48）：本评测里正负
                # 分两次调用打分（各 200 行），行置换只能发生在**同一标签组内部**，于是每组的特征
                # 边缘分布完全不变 ⇒ 正负之间仍有"分布层面的标签信号"，且 AUC 对组内置换**恒等不变**
                # （实测置换前后 AUC 差 0.0000）。行置换去掉的只是"逐候选对应"，不是"信息"，
                # 作为容量对照不干净。噪声版本没有这个问题。
                # 尺度差异（噪声 N(0,1) vs 真特征量纲）在解释上无害：噪声在任何尺度下都可被忽略。
                # 注意：**不能**把 CPU generator 和 `device='cuda'` 一起传给 torch.randn
                # （实测 RuntimeError: Expected a 'cuda' device type for generator but found 'cpu'）。
                # 因此先在 CPU 上按固定种子生成，再搬到目标设备 —— 固定种子保证逐位可复现。
                gen = torch.Generator(device="cpu").manual_seed(20261003)
                extras["pair_feat"] = torch.randn(b, cfg.get("pair_dim", 0),
                                                  generator=gen, dtype=torch.float32).to(device)
            elif _g == "noise-emb":
                # **臂 F 的容量对照**：与 `emb` **同维（2*memory_dim + sub_hidden + pair_dim）**、
                # 同参数量，内容为 iid 噪声 ⇒ 与候选、与标签无关。
                # 为什么必须单列一个臂：`noise`（7 维）只能控制"融合头从 7 维变 14 维"这种小容量变化，
                # 而臂 F 的头是从 289 参数变到 12,545（输入 7→390 维）——用 7 维噪声控制不了它。
                # 判据已在 EXPERIMENTS.md 预先写死（noise ≥ 50% 臂F 增益 ⇒ 归因容量）。
                gen = torch.Generator(device="cpu").manual_seed(20261004)
                extras["emb_feat"] = torch.randn(b, cfg.get("emb_dim", 0),
                                                 generator=gen, dtype=torch.float32).to(device)
            elif _g == "meta":
                # 窗口覆盖率 = 该端落在硬窗内的邻居数 / 完整因果历史邻居数（0 = 该端无历史）。
                # 无硬窗时窗口即全部历史 ⇒ 覆盖率为 1（有历史）或 0（无历史）——**退化成二值**：
                # college_msg 的旗舰配置是 `--window-frac 0`（`窗 Δ=0.00 天`），本臂若照搬该窗口，
                # 两个覆盖率维度就几乎恒为 1.0，等于门控拿到的是一列常数（该数据集上必然无结论）。
                # 因此提供**特征专用**回退窗口 `--meta-window-frac`（train_span 的比例）：
                #   默认 0.0 ⇒ 不启用 ⇒ `--fusion-ctx meta` 的默认行为逐位不变；
                #   臂 C 显式传 0.05 ⇒ 四个数据集的覆盖率**用同一把尺子**（train_span 的 5%），
                #   而不是「三个数据集 5%、college 无窗」这种口径不齐的混合。
                win = cfg["window"] if cfg["window"] is not None else cfg.get("meta_window")
                extras["meta_feat"] = torch.as_tensor(
                    meta_coverage(g_sub, src, dst, ts, win),
                    dtype=torch.float32, device=device)
            elif _g == "emb":
                # 两路的学习表示：节点路线 h_u/h_v（各 memory_dim）+ 子图路线池化表示 h（sub_hidden+pair_dim）。
                # 这是 D2 里唯一「门能看到路线内部表示」的臂，用来区分「门的输入不够」与「融合形式不对」。
                if need_node and h_sub_pooled is not None:
                    extras["emb_feat"] = torch.cat([h_u, h_v, h_sub_pooled], dim=-1)
            elif _g == "stat" and "struct" in stats:
                # 子图内结构统计的 mean±std（由 build_subgraphs(collect_struct=True) 提供）
                extras["stat_feat"] = torch.as_tensor(
                    np.asarray(stats["struct"], dtype=np.float32).reshape(b, -1),
                    dtype=torch.float32, device=device)
        # 计划3 · D1：第 8 组 = **按路线分位数校准后的相对位置差** `q_node − q_sub`。
        # 开关关（calib_on=False）时不构造任何张量 ⇒ 默认路径逐位不变。
        # 校准尚未拟合时（预热期）该维**严格补 0**——预热期 g≡0.5、完全不读 ctx，
        # 所以补 0 不会给门控引入偏差；宽度恒定 ⇒ 门控权重形状不因“拟合与否”而变。
        if calib_on:
            if calib is None:
                extras["calib_feat"] = torch.zeros(b, 1, dtype=torch.float32, device=device)
            else:
                _rel = calib.relative_torch("node", torch.sigmoid(logit_node),
                                            "sub", torch.sigmoid(logit_sub))
                extras["calib_feat"] = _rel.unsqueeze(-1).to(torch.float32)
        # BUGLOG B65（E0）：ctx 前两维取逐候选量还是批均值广播，由 `--ctx-per-candidate` 决定。
        # 取法与"为什么必须是开关"见 `ctx_size_dims`。
        _nn, _ne, _ctx_src = ctx_size_dims(stats, b, bool(cfg.get("ctx_per_candidate")))
        if cfg.get("ctx_per_candidate") and _ctx_src != "per-candidate":
            # 走到这里说明开关开了却**静默退回**了旧口径 —— 那会把一个旧口径的 run
            # 记成"逐候选门控"，正是要防的那类不报错的错答案。宁可在这里炸。
            raise SystemExit(
                "--ctx-per-candidate 已打开，但 ctx 前两维退回了批均值广播"
                f"（stats 缺逐候选 nodes/edges，或长度与本批候选数 {b} 不符）"
                "⇒ 该 run 实际是旧口径，不能作为 E0 的证据。")
        ctx = context_features(
            torch.as_tensor(np.asarray(_nn, dtype=np.float32), device=device),
            torch.as_tensor(np.asarray(_ne, dtype=np.float32), device=device),
            dt_u, dt_v, cnt_u, cnt_v, logit_node, logit_sub,
            use_dt=cfg.get("ctx_use_dt", True), **extras)
        assert_ctx_ready(cfg.get("fusion_groups", ()), extras, ctx.shape[1],
                         bool(cfg.get("ctx_use_dt", True)), int(cfg.get("fusion_extra_dim", 0)))
    return logit_node, logit_sub, logit_pair, ctx, stats


# ---------------------------------------------------------------------------- #
# 训练 / 评测
# ---------------------------------------------------------------------------- #


def run_stream(node_route, sub_route, fusion, g_nbr, g_sub, src, dst, ts, sampler, *,
               cfg, device, feat_dim, criterion, optimizer=None, warmup=False,
               lambda_node=0.3, lambda_sub=0.3, lambda_cons=0.0, cons_kind="jsd",
               update_memory=True, collect_metrics=True,
               route="fused", collect_gate=False, pair_route=None, dump_scores=False,
               edge_lookup=None, calib=None, calib_on=False):
    """按时间顺序跑一段边（训练或评测）。返回 (metrics_dict, loss_mean, g_stats)。"""
    train = optimizer is not None
    if not train:
        sampler.reset_random_state()   # 评测：负样本跨轮固定（照 TAMI/TGN-SEAL）
    n = len(src)
    per_batch, losses, g_vals, n_nodes, n_edges = [], [], [], [], []
    cons_vals: list = []               # I6：lambda_cons > 0 时逐 batch 记录一致性项的值
    gate_rows: list = []               # collect_gate=True 时逐 batch 收集 (g, ctx, y)
    # 计划3 D0a：dump_scores=True 时逐 batch 收集 (p_node, p_sub, y, batch_id)，用于算
    # 「逐候选 oracle 上界」。默认 False ⇒ 默认路径逐位不变。
    score_rows: list = []
    need_pair = route == "pair"
    need_node, need_sub = route != "sub" and not need_pair, route != "node" and not need_pair
    if dump_scores and not (need_node and need_sub):
        raise SystemExit("--score-dump 需要 --route fused（要同时拿到节点/子图两路分数）")
    for lo in range(0, n, cfg["batch_size"]):
        hi = min(lo + cfg["batch_size"], n)
        b_src, b_dst, b_ts = src[lo:hi], dst[lo:hi], ts[lo:hi]
        # ---- 因果性断言：本 batch 候选时刻必须不早于已进记忆的水位
        wm = node_route.memory.watermark
        t_min = float(np.min(b_ts))
        assert t_min >= wm, (
            f"记忆水位晚于候选时刻（未来信息泄漏）：batch[{lo}:{hi}] min ts={t_min:.0f} "
            f"< 水位 {wm:.0f}（差 {wm - t_min:.0f}s）"
        )

        _, neg_dst = sampler.sample_batch(b_src)
        if train:
            node_route.train(); sub_route.train(); fusion.train()
            if pair_route is not None:      # 漏掉这一句 = 评测时 BatchNorm 仍用**批内统计**
                pair_route.train()          # （每批恰好 200 正 + 200 负）⇒ 相当于评测时做了批内对比
        else:                               # 归一化，AP 被虚高（BUGLOG B44）。
            node_route.eval(); sub_route.eval(); fusion.eval()
            if pair_route is not None:
                pair_route.eval()

        ctx_args = dict(cfg=cfg, device=device, feat_dim=feat_dim,
                        need_node=need_node, need_sub=need_sub,
                        pair_route=pair_route, need_pair=need_pair,
                        edge_lookup=edge_lookup,            # 计划3 D2：默认 None ⇒ 不变
                        calib=calib, calib_on=calib_on)     # 计划3 D1：默认 False ⇒ 不变
        ln_p, ls_p, lp_p, ctx_p, stats = score_candidates(node_route, sub_route, g_nbr, g_sub,
                                                           b_src, b_dst, b_ts, **ctx_args)
        with torch.set_grad_enabled(train):
            ln_n, ls_n, lp_n, ctx_n, _ = score_candidates(node_route, sub_route, g_nbr, g_sub,
                                                           b_src, neg_dst, b_ts, **ctx_args)
        ln = torch.cat([ln_p, ln_n]); ls = torch.cat([ls_p, ls_n])
        lp = torch.cat([lp_p, lp_n]) if lp_p is not None else None
        y = torch.cat([torch.ones(len(b_src), device=device),
                       torch.zeros(len(b_src), device=device)])

        g = None
        if route == "fused":
            ctx = torch.cat([ctx_p, ctx_n])
            _, g, logit_main = fusion(ln, ls, ctx, warmup=warmup)
        elif route == "node":
            logit_main = ln
        elif route == "pair":      # 启发式特征分支（D1）：分数就是 pair MLP 的输出
            logit_main = lp
        else:
            logit_main = ls

        if train:
            loss = criterion(logit_main, y)
            if route == "fused":            # D10.2：两路各自校准（单路线消融时无此项）
                loss = loss + lambda_node * criterion(ln, y) + lambda_sub * criterion(ls, y)
                if lambda_cons > 0.0:       # I6：跨路线一致性（默认关；关时不建图、不影响任何数）
                    cons = consistency_loss(ln, ls, cons_kind)
                    loss = loss + lambda_cons * cons
                    cons_vals.append(float(cons.detach()))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # BUGLOG B64：grad 模式下，本步 backward 已消费完上一步写入记忆的计算图，
            # 立刻切断，否则图会跨 batch 累积（退化成 BPTT，显存随轮数增长）。
            if node_route is not None and cfg.get("memory_grad", "none") == "onestep":
                node_route.memory.detach()
            losses.append(float(loss.detach()))

        if collect_metrics:
            with torch.no_grad():
                yy = y.cpu().numpy()
                scored = {}
                if route == "fused":
                    scored["fused"] = logit_main
                if need_node:
                    scored["node"] = ln
                if need_sub:
                    scored["sub"] = ls
                if need_pair:
                    scored["pair"] = lp
                per_batch.append({
                    k: data_metrics.get_link_prediction_metrics(
                        torch.sigmoid(v).detach().cpu().numpy(), yy)
                    for k, v in scored.items()
                })
                if g is not None:
                    g_vals.append(float(g.mean()))
                    if collect_gate:
                        gate_rows.append((g.detach().float().cpu().numpy(),
                                          ctx.detach().float().cpu().numpy(), yy))
                    if dump_scores:
                        score_rows.append((
                            torch.sigmoid(ln).detach().float().cpu().numpy(),
                            torch.sigmoid(ls).detach().float().cpu().numpy(),
                            yy,
                            np.full(len(yy), len(score_rows), dtype=np.int32),
                            # 源/目标节点 id：供离线做“按 source-node degree 分层”审计
                            # （2506.12588 的病灶③：pooled AP 可能被 hub 主导）。
                            # B60 修复（第 52 轮）：候选顺序是「**先正例后负例**」，
                            # 且负例**沿用与正例锚点相同的 b_src**（只有目标换成 neg_dst）
                            # ⇒ 逐候选 id 必须是 concat([b_src, b_src]) / concat([b_dst, neg_dst])。
                            # 旧写法只存了正边的 b_src/b_dst（**长度只有候选数的一半**），
                            # 会让 D7 的分层审计错位；若长度偶然相等还会**静默算错**。
                            np.concatenate([np.asarray(b_src, dtype=np.int64)] * 2),
                            np.concatenate([
                                np.asarray(b_dst, dtype=np.int64),
                                np.asarray(neg_dst.detach().cpu()
                                           if hasattr(neg_dst, "detach") else neg_dst,
                                           dtype=np.int64)]),
                            # 学到的融合分数：D7 审计要靠它直接回答
                            # “融合增益是不是被少数 hub候选 拉高的”。
                            torch.sigmoid(logit_main).detach().float().cpu().numpy()))
        n_nodes.append(stats["nodes_mean"]); n_edges.append(stats["edges_mean"])

        if update_memory:
            if cfg.get("memory_grad", "none") == "onestep":
                # BUGLOG B64 修复：保留 GRU 计算图。梯度不由本 batch 消费（记忆在打分之后
                # 才更新），而是由**下一个** batch 的 backward 经 `embed()` 读到记忆时回流。
                # `store_messages` 内部仍 detach 消息 ⇒ GRU 输入是常量，只有参数拿梯度。
                node_route.store_messages(b_src, b_dst, b_ts)
                node_route.memory.apply(node_route.memory_updater, grad=True)
            else:
                with torch.no_grad():
                    node_route.store_messages(b_src, b_dst, b_ts)
                    node_route.memory.apply(node_route.memory_updater)

    out = {}
    if collect_metrics:
        for key in ("fused", "node", "sub", "pair"):   # pair 只在 --route pair 时出现（下面的 in 判断兜住）
            if key in per_batch[0]:
                out[key] = data_metrics.aggregate_batch_metrics([b[key] for b in per_batch])
    return {
        "metrics": out,
        "loss": float(np.mean(losses)) if losses else None,
        "g_mean": float(np.mean(g_vals)) if g_vals else None,
        "cons_mean": float(np.mean(cons_vals)) if cons_vals else None,
        "gate_rows": gate_rows,
        "score_rows": score_rows,
        "sub_nodes_mean": float(np.mean(n_nodes)),
        "sub_edges_mean": float(np.mean(n_edges)),
        "n_batches": len(n_nodes),
    }


GATE_CTX_NAMES = ("log1p_sub_nodes", "sub_edge_node_ratio", "log1p_dt_u", "log1p_dt_v",
                  "log1p_cnt_u", "log1p_cnt_v", "abs_pnode_minus_psub")


def _rank(x: np.ndarray) -> np.ndarray:
    return np.argsort(np.argsort(x)).astype(np.float64)


#: 每个融合组"必须产出"的特征键（B51 的通用解药：分支被编辑吃掉时能按名字报错，而不是等矩阵乘法报形状）
FUSION_GROUP_KEY = {"pair": "pair_feat", "pair-shuf": "pair_feat", "noise": "pair_feat",
                    "noise-emb": "emb_feat", "emb": "emb_feat", "stat": "stat_feat",
                    "meta": "meta_feat"}


def assert_ctx_ready(fusion_groups, extras: dict, ctx_cols: int | None = None,
                     use_dt: bool = True, extra_dim: int = 0) -> None:
    """两道结构性护栏（第 11 轮加，起因见 BUGLOG B51）。

    **为什么要有它**：第 9 轮插入 `noise-emb` 分支时，一次"锚点替换"编辑**吃掉了
    `elif _g == "meta":` 这一行**，于是 meta 分支的代码体变成了 noise-emb 的代码体：
    `--fusion-ctx meta` 什么也不追加，直到融合头做矩阵乘法才报
    `mat1 and mat2 shapes cannot be multiplied (400x7 and 9x32)` ——
    **报得晚、且不指向真正的原因**。这两道断言把同一类错误变成"在源头、按名字"的报错。

    抽出成模块级函数是为了**可单测**：见 `src/checks/check_ctx_guard.py`（免 GPU）。
    """
    groups = list(fusion_groups)
    missing = [g for g in groups if FUSION_GROUP_KEY.get(g) not in extras]
    if missing:
        raise SystemExit(
            f"融合组 {missing} 没有产出任何上下文特征（分支丢失或条件不满足）；"
            f"fusion_groups={groups}、已产出={sorted(extras)}")
    if ctx_cols is not None:
        expect = ctx_dim(use_dt, extra=extra_dim)
        if ctx_cols != expect:
            raise SystemExit(
                f"上下文维数不匹配：实际 {ctx_cols}、融合头期望 {expect}"
                f"（fusion_groups={groups}, 已产出={sorted(extras)}）")


def save_ckpt(path, *, epoch, models, opt, best_val, best_state, log_epochs, sampler=None,
              args=None) -> None:
    """写"继续训练所需的**全部**状态"（计划 2 的可选中继点，默认不启用）。

    为什么必须连 RNG 一起存：训练负样本走**全局 numpy RNG**
    （`NegativeEdgeSampler` 在 `seed=None` 时用 `np.random.randint`，见 `data/negatives.py`），
    dropout 走 torch RNG。少存任何一个，续训轨迹就与"一口气跑完"不同。

    记忆**不需要**存：`main` 每轮开头都调 `node_route.memory.init_memory()`，记忆不跨轮携带，
    因此"从第 k+1 轮继续"在状态上是干净的。
    """
    import random as _random

    blob = {
        "epoch": int(epoch),
        "models": {k: v.state_dict() for k, v in models.items() if v is not None},
        "opt": opt.state_dict(),
        "best_val": float(best_val),
        "best_state": best_state,
        "log_epochs": list(log_epochs),
        # 存下**当时的配置**，供 load 时校验：同 tag 但配置变了就拒绝 resume，
        # 否则会把"另一套配置的训练状态"静默套到这次运行上（又一类不会报错的错答案）。
        "args": ({k: v for k, v in vars(args).items() if k != "resume"} if args is not None else None),
        # 采样器的**记账**也要存：否则续训后 summary.json 里的 n_neg_drawn 只统计最后一段，
        # 与"一口气跑完"对不上（数值本身不受影响，但逐字段比对会挂 —— 记账也要连续）。
        "sampler": ({"n_neg_drawn": int(sampler.n_neg_drawn),
                     "n_neg_rejected": int(sampler.n_neg_rejected)} if sampler is not None else None),
        "rng": {
            "torch": torch.get_rng_state(),
            "torch_cuda": (torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None),
            "numpy": np.random.get_state(),
            "python": _random.getstate(),
        },
    }
    tmp = path.with_suffix(".tmp")
    torch.save(blob, tmp)
    tmp.replace(path)          # 原子替换：中途被杀不会留下半个 checkpoint


def load_ckpt(path, *, models, opt, sampler=None, args=None) -> dict:
    """读回 checkpoint 并恢复模型/优化器/RNG；返回 blob（轮次、best_val、best_state、log）。"""
    import random as _random

    blob = torch.load(path, map_location="cpu", weights_only=False)
    if args is not None and blob.get("args"):
        now = {k: v for k, v in vars(args).items() if k != "resume"}
        old = blob["args"]
        diff = [k for k in sorted(set(now) | set(old)) if repr(now.get(k)) != repr(old.get(k))]
        if diff:
            raise SystemExit(
                f"中继点与当前配置不一致（{len(diff)} 项：{diff[:6]}）⇒ 拒绝 --resume。\n"
                f"  该目录：{path}\n"
                f"  处理：要么删掉 ckpt.pt 重新跑，要么换一个 --tag（tag 与配置应一一对应）。")
    for k, sd in blob["models"].items():
        if models.get(k) is not None:
            models[k].load_state_dict(sd)
    opt.load_state_dict(blob["opt"])
    if sampler is not None and blob.get("sampler"):
        sampler.n_neg_drawn = int(blob["sampler"]["n_neg_drawn"])
        sampler.n_neg_rejected = int(blob["sampler"]["n_neg_rejected"])
    rng = blob.get("rng") or {}
    if "torch" in rng:
        torch.set_rng_state(rng["torch"])
    if rng.get("torch_cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(rng["torch_cuda"])
    if "numpy" in rng:
        np.random.set_state(rng["numpy"])
    if "python" in rng:
        _random.setstate(rng["python"])
    return blob


def write_gate_stats(out_dir, gate_rows, tag: str, extra_layout=()) -> None:
    """门控是否真被**上下文**驱动（创新点 I1 的证据②）：g 与各 ctx 维的 Spearman + 分桶均值。

    只做统计，不改模型；输出 ``gate_stats.json``。判读方式：
    - 若 `std_g` ≈ 0 或所有相关性 ≈ 0 → 门控退化成一个常数，I1 的“上下文门控”名不副实；
    - 若 `g` 与 `abs_pnode_minus_psub` / 子图规模 / 陈旧度显著相关 → 门控确实在按样本调节。
    - 若 `g` 与 `abs_pnode_minus_psub` / 子图规模 / 陈旧度显著相关 → 门控确实在按样本调节。

    ``extra_layout``：(组名, 维数) 的序列，描述**追加在基础 7 维之后**的上下文排布
    （如 `[("pair", 7)]`）。计划 2 的四个臂的全部意义都在这些追加维上，而早先版本只统计基础 7 维，
    于是"门控有没有用上新输入"**无法从我自己的产物里回答**（M4 只能间接推断）。
    传空（默认）时输出与改造前**逐字节一致** ⇒ 冻结 run 的 gate_stats.json 不受影响。
    """
    g_all = np.concatenate([r[0] for r in gate_rows])
    c_all = np.concatenate([r[1] for r in gate_rows], axis=0)
    y_all = np.concatenate([r[2] for r in gate_rows])
    # 第 27 轮修：**常数门（`--const-gate`）下这些相关性是假的**。
    # `_rank` 用 `argsort(argsort(x))`，对**常数** x 会返回 0,1,2,…,n−1（平局按位置打破），
    # 于是 `corrcoef` 算出的是**完全由候选顺序驱动**的“相关性”——不是 NaN，是看上去合法的噪声。
    # 实测：D0b 的 CPU 冒烟（σ_g 恒为 0）打出了 sub_edge_node_ratio=+0.446 之类的数。
    # 若不拦，`gate_report` 会跨种子对它们投票，把“常数门在看什么”写成结论——而常数门什么都不看。
    # 用**容差**而不是 `> 0`：常数门的 g 在 float32 下 `np.std` 得到的是 5.96e-08 而不是恰好 0
    # （第 27 轮实测：写成 `> 0` 时守卫根本没触发，照样打出了假相关性）。
    # 真实的上下文门 σ_g 在 1e-2~1e-1 量级，1e-6 这条线能干净地分开两者；
    # 而 σ_g 小到 1e-6 以下的“门”本来也就没有候选间差异，相关性同样无意义。
    g_sd = float(np.std(g_all))
    corr_defined = g_sd > 1e-6
    if corr_defined:
        rg = _rank(g_all)
        corr = {n: float(np.corrcoef(rg, _rank(c_all[:, i]))[0, 1])
                for i, n in enumerate(GATE_CTX_NAMES)}
    else:
        # 写 **null**（不是 NaN）：JSON 规范里 NaN 非法，且下游 `max(abs(...))` 会**静默投出一票**
        corr = {n: None for n in GATE_CTX_NAMES}
    # ---- 追加维的统计（计划 2）：门控到底有没有用上新送进来的那些输入 ----
    n_base = len(GATE_CTX_NAMES)
    extra_rep: dict = {}
    if extra_layout and c_all.shape[1] > n_base:
        off = n_base
        for gname, ndim in extra_layout:
            if ndim <= 0 or off >= c_all.shape[1]:
                off += max(ndim, 0)
                continue
            hi = min(off + ndim, c_all.shape[1])
            if corr_defined:
                rhos = [float(np.corrcoef(rg, _rank(c_all[:, j]))[0, 1]) for j in range(off, hi)]
                order = sorted(range(len(rhos)), key=lambda k: -abs(rhos[k]))
                extra_rep[gname] = {
                    "dims": len(rhos),
                    "max_abs_rho": float(max(abs(r) for r in rhos)),
                    "mean_abs_rho": float(sum(abs(r) for r in rhos) / len(rhos)),
                    "top3": [[int(k), round(rhos[k], 6)] for k in order[:3]],
                    "all_rho": [round(r, 6) for r in rhos] if len(rhos) <= 16 else None,
                }
            else:
                # 常数门：相关性未定义（不是 0，也不是“无关”），显式写 null
                extra_rep[gname] = {"dims": hi - off, "max_abs_rho": None,
                                    "mean_abs_rho": None, "top3": None, "all_rho": None}
            off = hi
    q = np.quantile(c_all[:, 0], [0.25, 0.5, 0.75])
    buckets = [c_all[:, 0] <= q[0],
               (c_all[:, 0] > q[0]) & (c_all[:, 0] <= q[1]),
               (c_all[:, 0] > q[1]) & (c_all[:, 0] <= q[2]),
               c_all[:, 0] > q[2]]
    rep = {
        "tag": tag, "n": int(g_all.size),
        "mean_g": float(g_all.mean()), "std_g": float(g_all.std()),
        "g_pos_mean": float(g_all[y_all > 0.5].mean()),
        "g_neg_mean": float(g_all[y_all < 0.5].mean()),
        "spearman_g_vs_ctx": corr,
        "mean_g_by_subgraph_node_quartile": [float(g_all[b].mean()) for b in buckets],
        "ctx_quantiles": {n: [float(v) for v in np.quantile(c_all[:, i], [0.05, 0.5, 0.95])]
                          for i, n in enumerate(GATE_CTX_NAMES)},
    }
    if extra_rep:
        rep["spearman_g_vs_extra"] = extra_rep
    (out_dir / "gate_stats.json").write_text(
        json.dumps(rep, indent=2, ensure_ascii=False), encoding="utf-8")
    if corr_defined:
        top = sorted(corr.items(), key=lambda kv: -abs(kv[1]))[:3]
        tail = "与上下文最相关的前三维：" + ", ".join(f"{k}={v:+.3f}" for k, v in top)
    else:
        tail = "σ_g=0（常数门）⇒ **与上下文的相关性未定义**，不报（报出来的只会是候选顺序的噪声）"
    print(f"[{tag}] 门控统计（n={rep['n']}）ḡ={rep['mean_g']:.3f} σ={rep['std_g']:.3f} | " + tail)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="college_msg")
    ap.add_argument("--pair-features", action="store_true",
                    help="开候选对级因果特征（共同邻居/AA/Jaccard/历史交互），注入子图解码器")
    ap.add_argument("--pair-dims", default=PAIR_DEFAULT_SPEC,
                    help="pair 特征逐维消融：all | cn,aa,jac,deg,hist 的任意组合 | 裸下标如 0,1")
    ap.add_argument("--input", choices=["zeros", "struct"], default="struct")
    ap.add_argument("--hop", type=int, default=2)
    ap.add_argument("--m", type=int, default=10)
    ap.add_argument("--window-frac", type=float, default=0.05)
    ap.add_argument("--meta-window-frac", type=float, default=0.0,
                    help="仅 `--fusion-ctx meta`：覆盖率用的特征专用窗口（train_span 的比例）。"
                         "默认 0 = 沿用 --window-frac 的硬窗；数据集本身无硬窗（--window-frac 0，"
                         "如 college_msg）时覆盖率会退化成二值，此时可用本开关给一个统一的窗口。")
    ap.add_argument("--max-nodes", type=int, default=800)
    ap.add_argument("--k", type=int, default=10, help="节点路线邻居数")
    ap.add_argument("--memory-dim", type=int, default=172)
    ap.add_argument("--time-dim", type=int, default=100)
    ap.add_argument("--sub-hidden", type=int, default=32)
    ap.add_argument("--sub-layers", type=int, default=2)
    ap.add_argument("--max-z", type=int, default=100)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--batch-size", type=int, default=200)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--warmup-epochs", type=int, default=2, help="门控预热轮数（g≡0.5）")
    ap.add_argument("--lambda-node", type=float, default=0.3)
    ap.add_argument("--lambda-sub", type=float, default=0.3)
    ap.add_argument("--lambda-cons", type=float, default=0.0,
                    help="I6 消融：跨路线一致性（互蒸馏）损失权重；0 = 关闭（默认）")
    ap.add_argument("--cons-kind", choices=list(CONS_KINDS), default="jsd",
                    help="I6 一致性损失形态：jsd（对称有界，默认）/ mse_logit / mse_prob")
    ap.add_argument("--const-gate", action="store_true", help="常数门控（D10.1 的对照下界）")
    ap.add_argument("--memory-grad", choices=["none", "onestep"], default="none",
                    help="BUGLOG B64 修复：记忆写回是否保留 GRU 计算图。"
                         "none=冻结路径（默认，与历史 run 逐位一致）；"
                         "onestep=一步延迟的可微写回，使 memory_updater 真正被训练")
    ap.add_argument("--ctx-per-candidate", action="store_true",
                    help="BUGLOG B65 修复（E0）：融合门 ctx 前两维取**逐候选**子图规模/边节点比。"
                         "默认关 = 批均值广播，与全部既有冻结 run 逐位一致；"
                         "开 = 与 B65 无条件版逐位一致（等价于 m1p3_e1pilot_* 的口径）。"
                         "做成开关而不是无条件修复，是为了让『逐候选门 vs 批均值门』成为可对照的实验")
    ap.add_argument("--route", choices=["fused", "node", "sub", "pair"], default="fused",
                    help="单路线消融：只训该路线（fused=双路线门控融合，默认；pair=只用候选对特征的"
                         "启发式分支，计划2 D1）")
    ap.add_argument("--train-frac", type=float, default=1.0)
    ap.add_argument("--val-limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--deterministic", action="store_true",
                    help="开确定性算法（可复现性；若某算子无确定性实现会报错，正好定位噪声源）")
    ap.add_argument("--tag", default=None)
    # 计划 2 的可选中继点：默认 0/False ⇒ 默认路径行为与改造前**逐位一致**（不写任何文件）。
    ap.add_argument("--ckpt-every", type=int, default=0,
                    help="每 N 轮写一次 <out_dir>/ckpt.pt；**默认 0 = 关闭**")
    ap.add_argument("--resume", action="store_true",
                    help="若 <out_dir>/ckpt.pt 存在则从中断轮次继续（默认关；配合 --ckpt-every 使用）")
    ap.add_argument("--strict-negatives", action="store_true",
                    help="严格负采样：抽到的目标若与源构成「图里已存在的边」就重抽（默认关＝参考实现口径）")
    ap.add_argument("--gate-stats", action="store_true",
                    help="测试后额外输出 gate_stats.json（g 与 7 维上下文的 Spearman 与分桶均值）")
    ap.add_argument("--score-dump", action="store_true",
                    help="计划3 D0a：测试后额外输出 scores.npz（逐候选 p_node/p_sub/y/batch_id，"
                         "供 src.checks.oracle_headroom 算逐候选路由的 oracle 上界）。默认关")
    ap.add_argument("--edge-msg-dim", type=int, default=0,
                    help="计划3 D2：**只注入边通道**的边特征维度（0 = 关，默认）。"
                         "打开时用 data.edge_features 的四维因果特征，经**零初始化加性投影**"
                         "注入注意力 key/value（不拼接、不进节点表示）。默认 0 ⇒ 不建任何参数")
    ap.add_argument("--route-calib", action="store_true",
                    help="计划3 D1：给门控加一维**按路线分位数校准**后的相对位置差 "
                         "q_node−q_sub（取代'原始不可比的分歧'）。校准在**预热结束时用训练流**"
                         "拟合一次并冻结（预热期 g≡0.5 不读 ctx ⇒ 无循环、不碰验证集）。默认关")
    ap.add_argument("--no-decoder-dt", action="store_true",
                    help="I4 消融：节点解码器去掉 time_enc(Δt)，即“陈旧度只进嵌入、不进打分”")
    ap.add_argument("--no-edge-gate", action="store_true",
                    help="I2 消融：关掉子图内的边时间门控（GCNConv 不再吃 edge_weight）")
    ap.add_argument("--sub-time-dim", type=int, default=64,
                    help="子图路线的时间编码维度（默认 64；--share-time-encode 时须等于 --time-dim）")
    ap.add_argument("--share-time-encode", action="store_true",
                    help="消融：两条路线共享同一个 TimeEncode 实例（D8 实现注记 3）")
    ap.add_argument("--message-fn", choices=["identity", "mlp"], default="identity",
                    help="节点路线的 raw message 函数（D8 配置表定稿为 identity）")
    ap.add_argument("--ctx-no-dt", action="store_true",
                    help="I4 消融：融合门上下文去掉两维陈旧度（ctx 7 维 → 5 维）")
    ap.add_argument("--fusion", choices=["scalar", "rich", "feat"], default="scalar",
                    help="融合形式（计划2 D0）：scalar=现状的标量凸组合（默认）；rich=同型标量门但上下文更宽"
                         "（配合 --fusion-ctx）；feat=特征级融合头（不再有标量门 g，失去 ḡ 诊断）")
    ap.add_argument("--fusion-ctx", default="",
                    help="rich/feat 的上下文扩展组（逗号分隔）：pair=候选对因果特征 /"
                         " stat=子图内结构统计 mean±std / meta=窗口覆盖率。默认空 = 与现状一致")
    a = ap.parse_args()
    if a.lambda_cons > 0.0 and a.route != "fused":
        raise SystemExit("--lambda-cons > 0 只在 --route fused 下有意义："
                         "单路线消融里没有“两路”可以对齐（当前 --route " + a.route + "）")

    tag = a.tag or f"m1_{a.dataset}_hop{a.hop}_w{a.window_frac}"
    out_dir = RUNS_DIR / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(a.seed); np.random.seed(a.seed)

    if a.deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    device = torch.device(a.device if torch.cuda.is_available() else "cpu")

    d = load_processed(a.dataset)
    n_nodes = d["n_nodes"]
    train_span = float(d["train_ts"].max() - d["train_ts"].min())
    tau = train_span / max(len(d["train_ts"]) - 1, 1)
    window = a.window_frac * train_span if a.window_frac > 0 else None
    # 仅 `--fusion-ctx meta` 用的特征专用窗口（见 meta 分支的注释）：默认 None ⇒ 不发生任何影响。
    meta_window = a.meta_window_frac * train_span if a.meta_window_frac > 0 else None
    feat_dim = 3 if a.input == "struct" else 0
    pair_dims = parse_dims(a.pair_dims)
    pair_dim = len(pair_dims) if a.pair_features else 0

    # ---- 融合模式（计划2 D0）：在 cfg 之前解析，供 context_features 与融合头共用 ----
    # scalar = 现状（默认，逐位不变）；rich = 同型标量门 + 更宽上下文；feat = 特征级融合头（无 g）。
    # 扩展组：pair（候选对因果特征）/ stat（子图内结构统计的 mean±std，6 维）/ meta（窗口覆盖率，2 维）。
    fusion_groups = [g.strip() for g in a.fusion_ctx.split(",") if g.strip()]
    _known = ("pair", "pair-shuf", "noise", "noise-emb", "stat", "meta", "emb")
    _unknown = [g for g in fusion_groups if g not in _known]
    if _unknown:
        raise SystemExit(f"--fusion-ctx 未知组 {_unknown}；可选：{','.join(_known)}")
    if a.fusion == "scalar" and fusion_groups:
        raise SystemExit("--fusion-ctx 只在 --fusion rich/feat 下有意义（scalar 保持现状，避免误配）")
    if ("pair" in fusion_groups or "pair-shuf" in fusion_groups or "noise" in fusion_groups) \
            and pair_dim <= 0:
        raise SystemExit("--fusion-ctx pair/pair-shuf/noise 需要 --pair-features（后者决定追加几维）")
    if "stat" in fusion_groups and a.input != "struct":
        raise SystemExit("--fusion-ctx stat 需要 --input struct（zeros 档没有结构统计可汇总）")
    if "emb" in fusion_groups and a.route != "fused":
        raise SystemExit("--fusion-ctx emb 需要两路都在（--route fused）：单路线没有两路嵌入可拼）")
    emb_dim = 2 * a.memory_dim + a.sub_hidden + pair_dim   # emb 组与 noise-emb 组共用同一维度
    _pair_like = ("pair" in fusion_groups or "pair-shuf" in fusion_groups
                  or "noise" in fusion_groups)   # 三者都追加 pair_dim 维（noise 是容量对照）
    _emb_like = ("emb" in fusion_groups or "noise-emb" in fusion_groups)
    extra_dim = ((pair_dim if _pair_like else 0)
                 + (6 if "stat" in fusion_groups else 0)
                 + (2 if "meta" in fusion_groups else 0)
                 + ((2 * a.memory_dim + a.sub_hidden + pair_dim) if _emb_like else 0)
                 # 计划3 · D1：校准后的相对位置差占 1 维（默认关 ⇒ 不加，不变）
                 + (1 if a.route_calib else 0))
    cfg = dict(hop=a.hop, m=a.m, max_nodes=a.max_nodes, window=window, input_mode=a.input,
               meta_window=meta_window,
               tau=tau, max_z=a.max_z, k=a.k, batch_size=a.batch_size, pair_dim=pair_dim,
               pair_dims=pair_dims, ctx_use_dt=not a.ctx_no_dt,
               memory_grad=a.memory_grad, ctx_per_candidate=a.ctx_per_candidate,
               fusion_kind=a.fusion, fusion_groups=fusion_groups, fusion_extra_dim=extra_dim,
               emb_dim=emb_dim)

    print(f"[{tag}] dataset={a.dataset} device={device} 窗 Δ={window/86400 if window else 0:.2f} 天 "
          f"hop={a.hop} input={a.input} pair_dim={pair_dim} 路线={a.route} "
          f"门控={'常数' if a.const_gate else '上下文'} "
          f"ctx前两维={'逐候选(B65)' if a.ctx_per_candidate else '批均值(冻结口径)'} "
          f"记忆梯度={a.memory_grad} "
          f"解码器Δt={'on' if not a.no_decoder_dt else 'off'} 门控ctxΔt={'off' if a.ctx_no_dt else 'on'} "
          f"边时间门控={'off' if a.no_edge_gate else 'on'} "
          f"时间编码={'共享/dim=%d' % a.sub_time_dim if a.share_time_encode else '独立/dim=%d' % a.sub_time_dim} "
          f"预热={a.warmup_epochs} 轮 λ=({a.lambda_node},{a.lambda_sub})"
          + (f" λ_cons={a.lambda_cons}({a.cons_kind})" if a.lambda_cons > 0 else " 一致性=关"))

    g_train = CausalGraph(d["train_src"], d["train_dst"], d["train_ts"], n_nodes)
    g_full = CausalGraph(d["full_src"], d["full_dst"], d["full_ts"], n_nodes)
    # 计划 3 · D2：边通道的邻居特征索引。
    # **只在开关打开时构建**（默认 None）⇒ 默认路径既不建索引也不传特征。
    # 必须在 **full 流**上算：g_nbr = g_full，邻居槽位来自 full，特征行也必须来自 full。
    edge_lookup = None
    # 计划3 · D1：按路线分位数校准器（默认关 ⇒ None ⇒ 一切不变）。
    # 初始未拟合，预热结束时才 fit；在那之前 `calib_feat` 严格补 0。
    calibrator = RouteCalibrator() if a.route_calib else None
    # 关键：“已启用”≠“已拟合”。预热期二者不同：对象已建但**尚无断点**。
    # 传给前向的必须是“可用的校准器”，未拟合时传 None ⇒ ctx 那一维严格补 0。
    calib_use = None
    if a.edge_msg_dim:
        from data.edge_features import neighbor_feature_lookup
        _t0 = time.time()
        edge_lookup = neighbor_feature_lookup(d["full_src"], d["full_dst"], d["full_ts"])
        print(f"[{a.tag or 'm1'}] D2 边特征索引：{len(edge_lookup[0])} 个键、"
              f"{edge_lookup[1].shape[1]} 维，耗时 {time.time() - _t0:.1f}s")

    forbidden_pairs = None
    if a.strict_negatives:
        forbidden_pairs = set(zip(d["full_src"].tolist(), d["full_dst"].tolist()))
        print(f"[{tag or a.dataset}] 严格负采样：已存在边对 {len(forbidden_pairs)} 条将被拒绝")
    train_sampler = data_negatives.make_train_sampler(
        d["train_src"], d["train_dst"], forbidden_pairs=forbidden_pairs)
    val_sampler, test_sampler = data_negatives.make_eval_samplers(
        d["full_src"], d["full_dst"], forbidden_pairs=forbidden_pairs)

    node_route = NodeRoute(n_nodes=n_nodes, memory_dim=a.memory_dim, time_dim=a.time_dim,
                           n_neighbors=a.k, dropout=a.dropout, device=device,
                           decoder_dt=not a.no_decoder_dt,
                           message_function=a.message_fn,
                           # 计划3 · D2：默认 0 ⇒ 不建任何新参数，默认路径逐位不变
                           edge_msg_dim=a.edge_msg_dim).to(device)
    sub_route = SubgraphRoute(hidden=a.sub_hidden, max_z=a.max_z, num_layers=a.sub_layers,
                              time_dim=a.sub_time_dim, feat_dim=feat_dim, pair_dim=pair_dim,
                              dropout=a.dropout, use_edge_gate=not a.no_edge_gate).to(device)
    if a.share_time_encode:
        assert a.sub_time_dim == a.time_dim, (
            "共享时间编码要求两条路线维度相同："
            f"--sub-time-dim {a.sub_time_dim} != --time-dim {a.time_dim}")
        sub_route.time_encoder = node_route.time_encoder      # 同一 Module 实例 → 权重共享
    # 融合头（计划2 D0）：rich 与 scalar **同型**（都是标量门 g），只是 ctx 更宽；
    # feat 换成特征级头（没有标量门，因此也没有 ḡ 诊断）。
    # 关键：--fusion scalar（默认）时 extra_dim = 0 ⇒ 构造参数与改造前**完全一致**（默认路径逐位不变）。
    if a.fusion == "feat":
        fusion = FeatureFusion(in_dim=ctx_dim(not a.ctx_no_dt) + extra_dim,
                               hidden=a.sub_hidden, dropout=a.dropout).to(device)
    else:
        fusion = GatedFusion(ctx_dim=ctx_dim(not a.ctx_no_dt) + extra_dim,
                             const_gate=a.const_gate).to(device)
    # 共享时间编码时，同一个 TimeEncode 会同时出现在两条路线的 parameters() 里：
    # 必须按 id 去重，否则 Adam 会拿到重复参数组（torch 会警告，且等价于给该模块加倍学习率）。
    seen: dict[int, torch.nn.Parameter] = {}
    # 启发式特征分支（计划2 D1）：**只在 --route pair 时构造**。
    # 为什么不能无条件构造：任何 nn.Module 的初始化都会消耗全局 RNG；无条件构造会改变默认路径
    # （fused/node/sub）的参数初始化序列 → 与冻结的 N=8 结果不再逐位一致（违反 A1 的前提）。
    pair_route = None
    if a.route == "pair":
        if pair_dim <= 0:
            raise SystemExit("--route pair 需要 --pair-features（否则 pair_dim=0，该分支没有输入）")
        from src.models.pair_route import PairRoute      # 局部导入：默认路径连模块都不加载
        pair_route = PairRoute(pair_dim=pair_dim, hidden=a.sub_hidden,
                               dropout=a.dropout).to(device)
    all_params = (list(node_route.parameters()) + list(sub_route.parameters())
                  + list(fusion.parameters()))
    if pair_route is not None:
        # **必须显式加入优化器**：漏掉这一句 = 该分支权重永不更新，loss 恒为 ln2、AP≈随机，
        # 而训练脚本**不会报任何错**（BUGLOG B43 就是这么来的）。
        all_params += list(pair_route.parameters())
    for p in all_params:
        seen.setdefault(id(p), p)
    params = list(seen.values())
    n_params = sum(p.numel() for p in params)
    # 打印里带上 pair 分支：漏把某分支加进优化器时，这一行会立刻暴露它「有参数但不参与训练」
    _pair_txt = (f"pair {sum(p.numel() for p in pair_route.parameters()):,} / "
                 if pair_route is not None else "")
    print(f"[{tag}] 参数量 {n_params:,}（node {sum(p.numel() for p in node_route.parameters()):,} / "
          f"sub {sum(p.numel() for p in sub_route.parameters()):,} / {_pair_txt}"
          f"去重后优化器参数 {n_params:,}）")

    opt = torch.optim.Adam(params, lr=a.lr)
    criterion = F.binary_cross_entropy_with_logits

    tr_src, tr_dst, tr_ts = d["train_src"], d["train_dst"], d["train_ts"]
    if a.train_frac < 1.0:
        keep = int(len(tr_src) * a.train_frac)
        tr_src, tr_dst, tr_ts = tr_src[:keep], tr_dst[:keep], tr_ts[:keep]

    log = {"args": vars(a), "tag": tag, "n_params": n_params, "window_days":
           (window / 86400 if window else None), "epochs": []}
    best_val, best_state = -1.0, None

    # ---- 可选的中继点（计划 2）：默认 `--ckpt-every 0` ⇒ 下面整段都不执行 ----
    ckpt_path = out_dir / "ckpt.pt"
    start_epoch = 1
    if a.resume and ckpt_path.exists():
        _blob = load_ckpt(ckpt_path, models={"node": node_route, "sub": sub_route,
                                             "fusion": fusion, "pair": pair_route}, opt=opt,
                           sampler=train_sampler, args=a)
        start_epoch = int(_blob["epoch"]) + 1
        best_val = float(_blob.get("best_val", -1.0))
        best_state = _blob.get("best_state")
        log["epochs"] = list(_blob.get("log_epochs") or [])
        print(f"[{tag}] 从 ckpt.pt 恢复：已完成 {_blob['epoch']} 轮 ⇒ 从第 {start_epoch} 轮继续")

    for epoch in range(start_epoch, a.epochs + 1):
        te = time.perf_counter()
        node_route.memory.init_memory()          # 照参考实现：每轮重置记忆
        warmup = epoch <= a.warmup_epochs
        tr = run_stream(node_route, sub_route, fusion, g_train, g_train, tr_src, tr_dst, tr_ts,
                        train_sampler, cfg=cfg, device=device, feat_dim=feat_dim,
                        criterion=criterion, optimizer=opt, warmup=warmup, route=a.route,
                        lambda_node=a.lambda_node, lambda_sub=a.lambda_sub,
                        lambda_cons=a.lambda_cons, cons_kind=a.cons_kind,
                        update_memory=True, collect_metrics=True, pair_route=pair_route,
                        edge_lookup=edge_lookup,
                        calib=calib_use, calib_on=a.route_calib,
                        # D1 拟合需要两路分数，但只**在最后一个预热轮**收集：
                        # 更早的轮门控还未学任何 ctx，收集了也用不上；每轮收集会带来
                        # 逐 batch 的 GPU→CPU 同步，白白拖慢训练。
                        dump_scores=(a.route_calib and epoch == a.warmup_epochs))
        # 计划3 · D1：预热结束的那一轮，用**训练流**的两路分数拟合一次并**冻结**。
        # 为何此刻干净：预热期 g≡0.5、完全不读 ctx ⇒ 门控尚未从 ctx 学到任何东西，
        # 不存在“用校准拟合门控、又用门控拟合校准”的循环；也**不用验证集**（那会把
        # 验证集从“仅早停”升级成“参与训练梯度”，属协议层 Q2）。
        if a.route_calib and epoch == a.warmup_epochs:
            _rows = tr.get("score_rows") or []
            if not _rows:
                raise SystemExit("--route-calib：预热末轮未收到两路分数，无法拟合（不应发生）")
            _pn = np.concatenate([r[0] for r in _rows])
            _ps = np.concatenate([r[1] for r in _rows])
            _rn = calibrator.fit("node", _pn)
            _rs = calibrator.fit("sub", _ps)
            calib_use = calibrator          # 从此开始才真正送进门控（同一轮 val 已用上）
            print(f"[{tag}] D1 校准已拟合并冻结："
                  f"node n={_rn.n} distinct={_rn.n_distinct} degenerate={_rn.degenerate} | "
                  f"sub n={_rs.n} distinct={_rs.n_distinct} degenerate={_rs.degenerate}")
            if _rn.degenerate or _rs.degenerate:
                print(f"[{tag}] ⚠️ D1 校准在至少一条路线上**退化**（分数为常数）"
                      f"⇒ 该维恒为 0.5、等价于没有信息；结论必须按此限定。")
        g_txt = f"ḡ={tr['g_mean']:.3f} " if tr["g_mean"] is not None else ""
        c_txt = f"cons={tr['cons_mean']:.4f} " if tr["cons_mean"] is not None else ""
        msg = (f"epoch {epoch:>3} loss {tr['loss']:.4f} "
               + g_txt + c_txt
               + "train " + " ".join(f"{k}AP={v['average_precision']:.3f}"
                                     for k, v in tr["metrics"].items()) + " "
               f"子图 {tr['sub_nodes_mean']:.1f}节点 {time.perf_counter()-te:.1f}s")
        rec = {"epoch": epoch, "loss": tr["loss"], "g_mean": tr["g_mean"],
               "cons_mean": tr["cons_mean"],
               "train": tr["metrics"], "sub_nodes_mean": tr["sub_nodes_mean"]}
        if epoch % 2 == 0 or epoch == a.epochs:
            v_src, v_dst, v_ts = d["val_src"], d["val_dst"], d["val_ts"]
            if a.val_limit:
                v_src, v_dst, v_ts = v_src[:a.val_limit], v_dst[:a.val_limit], v_ts[:a.val_limit]
            val = run_stream(node_route, sub_route, fusion, g_full, g_full, v_src, v_dst, v_ts,
                             val_sampler, cfg=cfg, device=device, feat_dim=feat_dim,
                             criterion=criterion, optimizer=None, warmup=False, route=a.route,
                             update_memory=True, collect_metrics=True, pair_route=pair_route,
                             edge_lookup=edge_lookup,
                             calib=calib_use, calib_on=a.route_calib)
            rec["val"] = val["metrics"]; rec["val_g_mean"] = val["g_mean"]
            v_ap = "/".join(f"{v['average_precision']:.3f}"
                            for v in val["metrics"].values())
            msg += f" | valAP {'/'.join(val['metrics'].keys())} = {v_ap}"
            score = val["metrics"][a.route]["average_precision"]
            if score > best_val:
                best_val = score
                state = {**{f"n.{kk}": vv for kk, vv in node_route.state_dict().items()},
                         **{f"s.{kk}": vv for kk, vv in sub_route.state_dict().items()},
                         **{f"g.{kk}": vv for kk, vv in fusion.state_dict().items()}}
                if pair_route is not None:
                    state.update({f"p.{kk}": vv for kk, vv in pair_route.state_dict().items()})
                best_state = {k: v.detach().cpu().clone() for k, v in state.items()}
        print(msg)
        log["epochs"].append(rec)
        if a.ckpt_every and epoch % a.ckpt_every == 0 and epoch < a.epochs:
            save_ckpt(ckpt_path, epoch=epoch,
                      models={"node": node_route, "sub": sub_route, "fusion": fusion,
                              "pair": pair_route},
                      opt=opt, best_val=best_val, best_state=best_state,
                      log_epochs=log["epochs"], sampler=train_sampler, args=a)

    if best_state is not None:
        node_route.load_state_dict({k[2:]: v for k, v in best_state.items() if k.startswith("n.")})
        sub_route.load_state_dict({k[2:]: v for k, v in best_state.items() if k.startswith("s.")})
        fusion.load_state_dict({k[2:]: v for k, v in best_state.items() if k.startswith("g.")})
        if a.route == "pair":      # 启发式分支的最佳轮次同样要还原（否则早停形同虚设）
            pair_route.load_state_dict({k[2:]: v for k, v in best_state.items() if k.startswith("p.")})

    test = run_stream(node_route, sub_route, fusion, g_full, g_full,
                      d["test_src"], d["test_dst"], d["test_ts"], test_sampler,
                      cfg=cfg, device=device, feat_dim=feat_dim, criterion=criterion,
                      optimizer=None, warmup=False, update_memory=True, collect_metrics=True,
                      pair_route=pair_route,
                      route=a.route, collect_gate=a.gate_stats,
                      dump_scores=a.score_dump, edge_lookup=edge_lookup,
                      calib=calib_use, calib_on=a.route_calib)
    log["test"] = test["metrics"]; log["test_g_mean"] = test["g_mean"]; log["best_val_ap"] = best_val
    log["negatives"] = {"strict": bool(a.strict_negatives),
                        "train": train_sampler.stats(),
                        "val": val_sampler.stats(),
                        "test": test_sampler.stats()}
    print(f"[{tag}] TEST " + " | ".join(f"{k} {v}" for k, v in test["metrics"].items())
          + (f" | ḡ={test['g_mean']:.3f}" if test["g_mean"] is not None else ""))
    (out_dir / "summary.json").write_text(json.dumps(log, indent=2, ensure_ascii=False), encoding="utf-8")
    if ckpt_path.exists():
        # 跑完了就删掉中继点：避免"已完成 run 的陈旧 ckpt"在将来被 --resume 误用。
        # 放在 summary.json 之后 ⇒ 若在评测阶段被杀，ckpt 仍在，可用 --resume 直接补跑评测
        #（start_epoch = epochs+1 ⇒ 训练循环不执行、只重做测试）。
        ckpt_path.unlink()
        print(f"[{tag}] 已删除中继点 {ckpt_path.name}（run 已完成）")
    print(f"[{tag}] 结果写入 {out_dir / 'summary.json'}")
    if a.gate_stats and test["gate_rows"]:
        # 追加维的排布（与 score_candidates 里 extras 的拼接顺序一致）；无追加维时传空列表
        # ⇒ 输出与改造前逐字节一致（冻结 run 的 gate_stats.json 形状不变）。
        _layout = []
        for _g in fusion_groups:
            _layout.append((_g, {"pair": pair_dim, "pair-shuf": pair_dim, "noise": pair_dim,
                                 "noise-emb": emb_dim, "emb": emb_dim,
                                 "stat": 6, "meta": 2}.get(_g, 0)))
        write_gate_stats(out_dir, test["gate_rows"], tag, extra_layout=_layout)
    if a.score_dump:
        if not test["score_rows"]:
            raise SystemExit("--score-dump 打开了，但测试期没收到逐候选分数（不应发生）")
        _pn = np.concatenate([r[0] for r in test["score_rows"]])
        _ps = np.concatenate([r[1] for r in test["score_rows"]])
        _py = np.concatenate([r[2] for r in test["score_rows"]])
        _pb = np.concatenate([r[3] for r in test["score_rows"]])
        _psrc = np.concatenate([r[4] for r in test["score_rows"]])
        _pdst = np.concatenate([r[5] for r in test["score_rows"]])
        _pf = np.concatenate([r[6] for r in test["score_rows"]])
        _sp = out_dir / "scores.npz"
        np.savez_compressed(_sp, p_node=_pn, p_sub=_ps, y=_py, batch_id=_pb,
                            src=_psrc, dst=_pdst, p_fused=_pf)
        print(f"[{tag}] 逐候选分数写入 {_sp}（n={len(_py)}，批数={len(test['score_rows'])}）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
