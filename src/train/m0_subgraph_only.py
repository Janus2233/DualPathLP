"""M0 —— 子图路线**单独**跑：三种输入特征对照（回答 ``ARCHITECTURE.md`` §7.1 的空缺）。

实验目的（对应 ``docs/ARCHITECTURE.md`` §7.1）：

    D10.2 把"节点路线嵌入注入子图节点特征"默认关掉后，子图 GCN 该吃什么输入？
    本脚本用最小代价先量出两个候选的差距：
      --input zeros  ：子图节点特征全零，只有 DRNL 嵌入 + 边时间门控
      --input struct ：额外拼上因果结构统计（度 / 时间衰减度 / 陈旧度，见 D8.1 的候选②）

    第三个候选（只读节点路线记忆）需要节点路线存在，本脚本**不实现**，留到节点路线落地后测。

严格遵循的协议（D2/D4，冻结）：

- 训练段 = ``splits`` 的 ``train_*``；验证/测试同数据层产物。
- 训练负采样：训练期节点池、目标端替换、1:1（``data.negatives.make_train_sampler``）。
- 评测负采样：**全图**池、固定种子（val=0 / test=2），每轮评测前 ``reset_random_state()``。
- 指标：``data.metrics.get_link_prediction_metrics`` + ``aggregate_batch_metrics``（每 batch 先算再平均）。
- 子图：训练候选用**训练图**的邻接，验证/测试候选切**全图**邻接（照 TGN-SEAL 的做法）；
  但两者都只取 ``ts < t`` 的边（``CausalGraph`` 保证）。

用法（在仓库根目录下）：

    python -m src.train.m0_subgraph_only --dataset college_msg --input zeros --epochs 15 --tag m0_zeros
    ``collect_struct=True`` 时额外在 ``stats["struct"]`` 里放逐候选的结构统计 mean±std（6 维，
    计划2 D2 的 `stat` 组用）；默认 False ⇒ 与改造前逐位相同。
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
from data.config import DATASETS
from src.graph.causal_subgraph import CausalGraph, drnl_labels, structural_features
from src.graph.pair_features import PAIR_DEFAULT_SPEC, PAIR_DIM, pair_features, parse_dims
from src.models.subgraph_route import SubgraphRoute, collate_subgraphs

ROOT = Path(__file__).resolve().parents[2]  # Model experiments/
DATA_DIR = ROOT / "runs" / "data"
RUNS_DIR = ROOT / "runs" / "m0"


# ---------------------------------------------------------------------------- #
# 数据加载
# ---------------------------------------------------------------------------- #


def load_processed(dataset: str):
    import pandas as pd

    d = DATA_DIR / dataset
    df = pd.read_csv(d / f"ml_{dataset}.csv", index_col=0)
    sp = np.load(d / f"splits_{dataset}.npz", allow_pickle=True)

    def arr(name):
        return np.asarray(sp[name])

    out = {
        "full_src": arr("full_src"), "full_dst": arr("full_dst"), "full_ts": arr("full_ts"),
        "train_src": arr("train_src"), "train_dst": arr("train_dst"), "train_ts": arr("train_ts"),
        "val_src": arr("val_src"), "val_dst": arr("val_dst"), "val_ts": arr("val_ts"),
        "test_src": arr("test_src"), "test_dst": arr("test_dst"), "test_ts": arr("test_ts"),
    }
    out["n_nodes"] = int(max(df["u"].max(), df["i"].max()))
    out["n_edges_full"] = int(len(df))
    return out


# ---------------------------------------------------------------------------- #
# 子图批构造
# ---------------------------------------------------------------------------- #


def build_subgraphs(graph: CausalGraph, src, dst, ts, hop, m, max_nodes,
                    input_mode: str, tau: float, max_z: int, window: float | None = None,
                    pair_dim: int = 0, pair_dims=None, collect_struct: bool = False):
    """把一批候选边变成子图列表；同时返回自检统计（用来断言因果性）。

    ``window`` 非空时启用**硬时间窗**（D9.1 修订后的主线）：只保留 ``[t - window, t)`` 内的边。
    ``pair_dim > 0`` 时额外算**候选对级因果特征**（`src/graph/pair_features.py`），
    并作为 6 元组末位放进每个子图项。
    """
    subs = []
    stats = {"nodes": [], "edges": [], "min_dt": np.inf, "target_edge_present": 0}
    if collect_struct:
        stats["struct"] = []      # 计划2 D2 的 stat 组用；不选该组时连键都不建（默认路径逐位不变）
    for u, v, t in zip(np.asarray(src).tolist(), np.asarray(dst).tolist(), np.asarray(ts).tolist()):
        nodes, ei, edt, center = graph.extract(u, v, t, hop=hop, m=m, max_nodes=max_nodes,
                                                window=window)
        z = drnl_labels(nodes, ei, max_z=max_z)
        if edt.size:
            stats["min_dt"] = min(stats["min_dt"], float(edt.min()))
        if ei.shape[1] and np.any((ei[0] == 0) & (ei[1] == 1)) or (
            ei.shape[1] and np.any((ei[0] == 1) & (ei[1] == 0))
        ):
            stats["target_edge_present"] += 1
        feat = (
            structural_features(graph, nodes, ei, edt, t, tau) if input_mode == "struct" else None
        )
        if pair_dim:
            subs.append((nodes, ei, edt, z, feat,
                         pair_features(graph, int(u), int(v), float(t), window, pair_dims)))
        else:
            subs.append((nodes, ei, edt, z, feat))
        if collect_struct:
            # 计划2 D2 的 `stat` 组：把该候选子图内的 3 维结构统计汇总成 mean±std（6 维），
            # 供**融合门**判断「这个子图有多结构化」。默认 collect_struct=False ⇒ 默认路径完全不变。
            if feat is not None and len(feat):
                _sm = feat.mean(axis=0)
                _ss = feat.std(axis=0)
            else:
                _sm = np.zeros(3, dtype=np.float32)
                _ss = np.zeros(3, dtype=np.float32)
            stats["struct"].append(np.concatenate([_sm, _ss]).astype(np.float32))
        stats["nodes"].append(len(nodes))
        stats["edges"].append(int(ei.shape[1]))
    stats["nodes_mean"] = float(np.mean(stats["nodes"])) if stats["nodes"] else 0.0
    stats["edges_mean"] = float(np.mean(stats["edges"])) if stats["edges"] else 0.0
    return subs, stats


def run_model(model, subs, device, feat_dim, pair_dim=0):
    out = collate_subgraphs(subs, device, feat_dim=feat_dim, pair_dim=pair_dim)
    if pair_dim:
        z, ei, dt, center, feat, _, pair = out
        return model(z, ei, dt, center, feat, pair)
    z, ei, dt, center, feat, _ = out
    return model(z, ei, dt, center, feat)


def run_model_pooled(model, subs, device, feat_dim, pair_dim=0):
    """同 `run_model`，但同时返回**池化后的链接表示**（计划2 D2 的 `emb` 组用）。

    单列一个函数而不是给 `run_model` 加开关：`run_model` 是 m0/m1 共用的默认路径，
    保持它一字不动最安全（默认路径逐位不变的第一道保险）。
    """
    out = collate_subgraphs(subs, device, feat_dim=feat_dim, pair_dim=pair_dim)
    if pair_dim:
        z, ei, dt, center, feat, _, pair = out
        return model(z, ei, dt, center, feat, pair, return_pooled=True)
    z, ei, dt, center, feat, _ = out
    return model(z, ei, dt, center, feat, return_pooled=True)

# ---------------------------------------------------------------------------- #
# 评测（协议见模块 docstring）
# ---------------------------------------------------------------------------- #


@torch.no_grad()
def evaluate(model, graph, sampler, src, dst, ts, *, hop, m, max_nodes,
             input_mode, tau, max_z, batch_size, device, feat_dim, window=None, pair_dim=0,
             pair_dims=None):
    model.eval()
    sampler.reset_random_state()
    n = len(src)
    batch_metrics = []
    for lo in range(0, n, batch_size):
        hi = min(lo + batch_size, n)
        b_src = np.asarray(src[lo:hi])
        b_dst = np.asarray(dst[lo:hi])
        b_ts = np.asarray(ts[lo:hi])
        _, neg_dst = sampler.sample_batch(b_src)

        pos_subs, _ = build_subgraphs(graph, b_src, b_dst, b_ts, hop, m, max_nodes,
                                      input_mode, tau, max_z, window=window, pair_dim=pair_dim,
                                      pair_dims=pair_dims)
        neg_subs, _ = build_subgraphs(graph, b_src, neg_dst, b_ts, hop, m, max_nodes,
                                      input_mode, tau, max_z, window=window, pair_dim=pair_dim,
                                      pair_dims=pair_dims)
        logits = torch.cat(
            [run_model(model, pos_subs, device, feat_dim, pair_dim),
             run_model(model, neg_subs, device, feat_dim, pair_dim)]
        )
        probs = torch.sigmoid(logits).detach().cpu().numpy()
        labels = np.concatenate([np.ones(len(pos_subs)), np.zeros(len(neg_subs))])
        batch_metrics.append(data_metrics.get_link_prediction_metrics(probs, labels))
    model.train()
    return data_metrics.aggregate_batch_metrics(batch_metrics)


# ---------------------------------------------------------------------------- #
# 主流程
# ---------------------------------------------------------------------------- #


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="college_msg")
    ap.add_argument("--pair-dims", default=PAIR_DEFAULT_SPEC,
                    help="pair 特征逐维消融：all | cn,aa,jac,deg,hist 的任意组合 | 裸下标如 0,1")
    ap.add_argument("--input", choices=["zeros", "struct"], default="zeros")
    ap.add_argument("--hop", type=int, default=3)
    ap.add_argument("--m", type=int, default=10)
    ap.add_argument("--max-nodes", type=int, default=800)
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--time-dim", type=int, default=64)
    ap.add_argument("--max-z", type=int, default=100)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--window-frac", type=float, default=0.01,
                    help="硬时间窗 Δ = 该比例 × 训练期时间跨度（D9.1 修订后的主线）；0 = 关闭窗口")
    ap.add_argument("--no-edge-gate", action="store_true", help="关掉子图内的边时间门控（消融）")
    ap.add_argument("--pair-features", action="store_true",
                    help="开候选对级因果特征（共同邻居/AA/Jaccard/历史交互，见 src/graph/pair_features.py）")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--train-frac", type=float, default=0.25,
                    help="每 epoch 使用训练边的比例（快档调试用；1.0 = 全量）")
    ap.add_argument("--val-limit", type=int, default=0,
                    help=">0 时只用前 N 条验证边（过拟合自检等场合用；0 = 全量）")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--deterministic", action="store_true",
                    help="开确定性算法（可复现性；若某算子无确定性实现会报错，正好定位噪声源）")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--overfit", type=int, default=0,
                    help=">0 时进入过拟合自检：只用这么多条训练边，期望 AP 迅速接近 1")
    a = ap.parse_args()

    tag = a.tag or f"m0_{a.dataset}_{a.input}_hop{a.hop}" + ("_nogate" if a.no_edge_gate else "")
    out_dir = RUNS_DIR / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    if a.deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    device = torch.device(a.device if torch.cuda.is_available() else "cpu")

    d = load_processed(a.dataset)
    n_nodes = d["n_nodes"]
    train_span = float(d["train_ts"].max() - d["train_ts"].min())
    tau = train_span / max(len(d["train_ts"]) - 1, 1)  # 平均交互间隔（特征尺度）
    window = a.window_frac * train_span if a.window_frac > 0 else None
    feat_dim = 3 if a.input == "struct" else 0
    pair_dims = parse_dims(a.pair_dims)
    pair_dim = len(pair_dims) if a.pair_features else 0

    _w = f"Δ={window/86400:.2f} 天（跨度的 {a.window_frac:.1%}）" if window else "关闭（因果 k-hop）"
    print(f"[{tag}] 时间窗 {_w}")
    print(f"[{tag}] dataset={a.dataset} device={device} input={a.input} feat_dim={feat_dim} "
          f"pair_dim={pair_dim}")
    print(f"[{tag}] train={len(d['train_src'])} val={len(d['val_src'])} test={len(d['test_src'])} "
          f"tau={tau:.1f}s  train_frac={a.train_frac}")

    t0 = time.perf_counter()
    g_train = CausalGraph(d["train_src"], d["train_dst"], d["train_ts"], n_nodes)
    g_full = CausalGraph(d["full_src"], d["full_dst"], d["full_ts"], n_nodes)
    print(f"[{tag}] 邻接构建 {time.perf_counter() - t0:.1f}s")

    train_sampler = data_negatives.make_train_sampler(d["train_src"], d["train_dst"])
    val_sampler, test_sampler = data_negatives.make_eval_samplers(d["full_src"], d["full_dst"])

    model = SubgraphRoute(
        hidden=a.hidden, max_z=a.max_z, num_layers=a.layers, time_dim=a.time_dim,
        feat_dim=feat_dim, pair_dim=pair_dim, dropout=a.dropout, use_edge_gate=not a.no_edge_gate,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[{tag}] 参数量 {n_params:,}")
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    criterion = torch.nn.BCEWithLogitsLoss()

    # 训练候选（可选过拟合自检）
    tr_src, tr_dst, tr_ts = d["train_src"], d["train_dst"], d["train_ts"]
    if a.overfit:
        # **随机**子集：取最早的 n 条边会得到“没有历史 → 子图为空”的退化物（正负输入完全相同）
        _rng = np.random.RandomState(a.seed)
        _sel = np.sort(_rng.choice(len(tr_src), size=min(a.overfit, len(tr_src)), replace=False))
        tr_src, tr_dst, tr_ts = tr_src[_sel], tr_dst[_sel], tr_ts[_sel]
        print(f"[{tag}] 过拟合自检：随机抽 {len(tr_src)} 条训练边（seed={a.seed}）")
    log = {"args": vars(a), "tag": tag, "n_params": n_params, "tau": tau, "epochs": []}
    best_val_ap, best_state = -1.0, None

    for epoch in range(1, a.epochs + 1):
        te = time.perf_counter()
        n_use = len(tr_src) if a.train_frac >= 1.0 else max(a.batch_size, int(len(tr_src) * a.train_frac))
        sel = np.random.choice(len(tr_src), size=n_use, replace=False)
        sel.sort()
        losses, seen = [], 0
        epoch_pos: list[float] = []
        epoch_neg: list[float] = []
        for lo in range(0, len(sel), a.batch_size):
            idx = sel[lo : lo + a.batch_size]
            b_src, b_dst, b_ts = tr_src[idx], tr_dst[idx], tr_ts[idx]
            _, neg_dst = train_sampler.sample_batch(b_src)

            pos_subs, stats_p = build_subgraphs(g_train, b_src, b_dst, b_ts, a.hop, a.m,
                                                a.max_nodes, a.input, tau, a.max_z,
                                                window=window, pair_dim=pair_dim,
                                                pair_dims=pair_dims)
            neg_subs, stats_n = build_subgraphs(g_train, b_src, neg_dst, b_ts, a.hop, a.m,
                                                a.max_nodes, a.input, tau, a.max_z,
                                                window=window, pair_dim=pair_dim,
                                                pair_dims=pair_dims)
            if epoch == 1 and lo == 0:
                # 因果性自检：子图内每条边都必须严格早于候选时刻
                assert stats_p["min_dt"] > 0 and stats_n["min_dt"] > 0, "出现 Δt ≤ 0 的边（泄漏）"
                assert stats_p["target_edge_present"] == 0, "目标边未被移除"
                print(f"[{tag}] 首个子图批：pos 节点均值 {stats_p['nodes_mean']:.1f}"
                      f"（边 {stats_p['edges_mean']:.1f}）| neg {stats_n['nodes_mean']:.1f}"
                      f" | min Δt {stats_p['min_dt']:.1f}s")

            logits = torch.cat(
                [run_model(model, pos_subs, device, feat_dim, pair_dim),
                 run_model(model, neg_subs, device, feat_dim, pair_dim)]
            )
            labels = torch.cat([
                torch.ones(len(pos_subs), device=device),
                torch.zeros(len(neg_subs), device=device),
            ])
            loss = criterion(logits, labels)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss.detach()))
            seen += len(pos_subs) + len(neg_subs)
            # 累计训练集的正/负概率，用于每轮算训练集 AP（只看 loss 会看不出“到底学没学到”）
            with torch.no_grad():
                _p = torch.sigmoid(logits).detach().cpu().numpy()
            epoch_pos.extend(_p[: len(pos_subs)].tolist())
            epoch_neg.extend(_p[len(pos_subs) :].tolist())

        train_probs = np.asarray(epoch_pos + epoch_neg)
        train_labels = np.asarray([1.0] * len(epoch_pos) + [0.0] * len(epoch_neg))
        train_m = (
            data_metrics.get_link_prediction_metrics(train_probs, train_labels)
            if len(epoch_pos) and len(epoch_neg)
            else {}
        )
        msg = (f"epoch {epoch:>3} loss {np.mean(losses):.4f} "
               f"train_AP {train_m.get('average_precision', float('nan')):.3f} "
               f"候选 {seen} 用时 {time.perf_counter()-te:.1f}s")
        rec = {"epoch": epoch, "loss": float(np.mean(losses)), "n_cand": seen,
               "train_AP": train_m.get("average_precision"), "train_AUC": train_m.get("roc_auc")}

        if epoch % 5 == 0 or epoch == a.epochs or a.overfit:
            t1 = time.perf_counter()
            v_src, v_dst, v_ts = d["val_src"], d["val_dst"], d["val_ts"]
            if a.val_limit:
                v_src, v_dst, v_ts = v_src[: a.val_limit], v_dst[: a.val_limit], v_ts[: a.val_limit]
            val = evaluate(model, g_full, val_sampler, v_src, v_dst, v_ts,
                           hop=a.hop, m=a.m, max_nodes=a.max_nodes, input_mode=a.input, tau=tau,
                           max_z=a.max_z, batch_size=a.batch_size, device=device,
                           feat_dim=feat_dim, window=window, pair_dim=pair_dim,
                           pair_dims=pair_dims)
            rec["val"] = val
            msg += f"  | val {val} ({time.perf_counter()-t1:.1f}s)"
            # 指标字典的键是 average_precision / roc_auc（见 data/metrics.py）
            score = val.get("average_precision", val.get("ap", val.get("AP", 0.0)))
            if score > best_val_ap:
                best_val_ap = score
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(msg)
        log["epochs"].append(rec)

    if best_state is not None:
        model.load_state_dict(best_state)
    test = evaluate(model, g_full, test_sampler, d["test_src"], d["test_dst"], d["test_ts"],
                    hop=a.hop, m=a.m, max_nodes=a.max_nodes, input_mode=a.input, tau=tau,
                    max_z=a.max_z, batch_size=a.batch_size, device=device,
                    feat_dim=feat_dim, window=window, pair_dim=pair_dim,
                    pair_dims=pair_dims)
    log["test"] = test
    log["best_val_ap"] = best_val_ap
    print(f"[{tag}] TEST {test}")

    (out_dir / "summary.json").write_text(json.dumps(log, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[{tag}] 结果写入 {out_dir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
