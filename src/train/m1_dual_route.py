
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
from src.models.route_calibration import RouteCalibrator
from src.models.subgraph_route import SubgraphRoute
from src.train.m0_subgraph_only import (build_subgraphs, load_processed, run_model as run_sub_model,
                                        run_model_pooled)

ROOT = Path(__file__).resolve().parents[2]
RUNS_DIR = ROOT / "runs" / "m1"



def meta_coverage(g_sub, src, dst, ts, win) -> np.ndarray:
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
        last_u = node_route.memory.last_update[src_t].double()
        last_v = node_route.memory.last_update[dst_t].double()
        dt_u = (t_t - last_u).clamp(min=0.0)
        dt_v = (t_t - last_v).clamp(min=0.0)
        logit_node = node_route.score(h_u, h_v, torch.maximum(dt_u, dt_v))

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

    h_sub_pooled = None
    if need_sub:
        subs, stats = build_subgraphs(g_sub, src, dst, ts, cfg["hop"], cfg["m"], cfg["max_nodes"],
                                      cfg["input_mode"], cfg["tau"], cfg["max_z"],
                                      window=cfg["window"], pair_dim=cfg.get("pair_dim", 0),
                                      pair_dims=cfg.get("pair_dims"),
                                      collect_struct=("stat" in cfg.get("fusion_groups", ())))
        if "emb" in cfg.get("fusion_groups", ()):
            logit_sub, h_sub_pooled = run_model_pooled(sub_route, subs, device, feat_dim,
                                                       cfg.get("pair_dim", 0))
        else:
            logit_sub = run_sub_model(sub_route, subs, device, feat_dim, cfg.get("pair_dim", 0))
    else:
        stats = {"nodes_mean": 0.0, "edges_mean": 0.0}
        logit_sub = torch.zeros(b, dtype=torch.float32, device=device)
    logit_pair = None
    if need_pair:
        from src.graph.pair_features import pair_features_batch
        pf = pair_features_batch(g_sub, src, dst, ts, window=cfg["window"],
                                 dims=cfg.get("pair_dims"))
        logit_pair = pair_route(torch.as_tensor(pf, dtype=torch.float32, device=device))

    ctx = None
    if need_node and need_sub:
        extras: dict = {}
        for _g in cfg.get("fusion_groups", ()):
            if _g == "pair":
                from src.graph.pair_features import pair_features_batch
                pf = pair_features_batch(g_sub, src, dst, ts, window=cfg["window"],
                                         dims=cfg.get("pair_dims"))
                extras["pair_feat"] = torch.as_tensor(pf, dtype=torch.float32, device=device)
            elif _g == "pair-shuf":
                from src.graph.pair_features import pair_features_batch
                pf = pair_features_batch(g_sub, src, dst, ts, window=cfg["window"],
                                         dims=cfg.get("pair_dims"))
                perm = np.random.RandomState(20261003).permutation(len(pf))
                extras["pair_feat"] = torch.as_tensor(pf[perm], dtype=torch.float32, device=device)
            elif _g == "noise":
                gen = torch.Generator(device="cpu").manual_seed(20261003)
                extras["pair_feat"] = torch.randn(b, cfg.get("pair_dim", 0),
                                                  generator=gen, dtype=torch.float32).to(device)
            elif _g == "noise-emb":
                gen = torch.Generator(device="cpu").manual_seed(20261004)
                extras["emb_feat"] = torch.randn(b, cfg.get("emb_dim", 0),
                                                 generator=gen, dtype=torch.float32).to(device)
            elif _g == "meta":
                win = cfg["window"] if cfg["window"] is not None else cfg.get("meta_window")
                extras["meta_feat"] = torch.as_tensor(
                    meta_coverage(g_sub, src, dst, ts, win),
                    dtype=torch.float32, device=device)
            elif _g == "emb":
                if need_node and h_sub_pooled is not None:
                    extras["emb_feat"] = torch.cat([h_u, h_v, h_sub_pooled], dim=-1)
            elif _g == "stat" and "struct" in stats:
                extras["stat_feat"] = torch.as_tensor(
                    np.asarray(stats["struct"], dtype=np.float32).reshape(b, -1),
                    dtype=torch.float32, device=device)
        if calib_on:
            if calib is None:
                extras["calib_feat"] = torch.zeros(b, 1, dtype=torch.float32, device=device)
            else:
                _rel = calib.relative_torch("node", torch.sigmoid(logit_node),
                                            "sub", torch.sigmoid(logit_sub))
                extras["calib_feat"] = _rel.unsqueeze(-1).to(torch.float32)
        _nn, _ne, _ctx_src = ctx_size_dims(stats, b, bool(cfg.get("ctx_per_candidate")))
        if cfg.get("ctx_per_candidate") and _ctx_src != "per-candidate":
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




def run_stream(node_route, sub_route, fusion, g_nbr, g_sub, src, dst, ts, sampler, *,
               cfg, device, feat_dim, criterion, optimizer=None, warmup=False,
               lambda_node=0.3, lambda_sub=0.3, lambda_cons=0.0, cons_kind="jsd",
               update_memory=True, collect_metrics=True,
               route="fused", collect_gate=False, pair_route=None, dump_scores=False,
               edge_lookup=None, calib=None, calib_on=False):
    train = optimizer is not None
    if not train:
        sampler.reset_random_state()
    n = len(src)
    per_batch, losses, g_vals, n_nodes, n_edges = [], [], [], [], []
    cons_vals: list = []
    gate_rows: list = []
    score_rows: list = []
    need_pair = route == "pair"
    need_node, need_sub = route != "sub" and not need_pair, route != "node" and not need_pair
    if dump_scores and not (need_node and need_sub):
        raise SystemExit("--score-dump 需要 --route fused（要同时拿到节点/子图两路分数）")
    for lo in range(0, n, cfg["batch_size"]):
        hi = min(lo + cfg["batch_size"], n)
        b_src, b_dst, b_ts = src[lo:hi], dst[lo:hi], ts[lo:hi]
        wm = node_route.memory.watermark
        t_min = float(np.min(b_ts))
        assert t_min >= wm, (
            f"记忆水位晚于候选时刻（未来信息泄漏）：batch[{lo}:{hi}] min ts={t_min:.0f} "
            f"< 水位 {wm:.0f}（差 {wm - t_min:.0f}s）"
        )

        _, neg_dst = sampler.sample_batch(b_src)
        if train:
            node_route.train(); sub_route.train(); fusion.train()
            if pair_route is not None:
                pair_route.train()
        else:
            node_route.eval(); sub_route.eval(); fusion.eval()
            if pair_route is not None:
                pair_route.eval()

        ctx_args = dict(cfg=cfg, device=device, feat_dim=feat_dim,
                        need_node=need_node, need_sub=need_sub,
                        pair_route=pair_route, need_pair=need_pair,
                        edge_lookup=edge_lookup,
                        calib=calib, calib_on=calib_on)
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
        elif route == "pair":
            logit_main = lp
        else:
            logit_main = ls

        if train:
            loss = criterion(logit_main, y)
            if route == "fused":
                loss = loss + lambda_node * criterion(ln, y) + lambda_sub * criterion(ls, y)
                if lambda_cons > 0.0:
                    cons = consistency_loss(ln, ls, cons_kind)
                    loss = loss + lambda_cons * cons
                    cons_vals.append(float(cons.detach()))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
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
                            np.concatenate([np.asarray(b_src, dtype=np.int64)] * 2),
                            np.concatenate([
                                np.asarray(b_dst, dtype=np.int64),
                                np.asarray(neg_dst.detach().cpu()
                                           if hasattr(neg_dst, "detach") else neg_dst,
                                           dtype=np.int64)]),
                            torch.sigmoid(logit_main).detach().float().cpu().numpy()))
        n_nodes.append(stats["nodes_mean"]); n_edges.append(stats["edges_mean"])

        if update_memory:
            if cfg.get("memory_grad", "none") == "onestep":
                node_route.store_messages(b_src, b_dst, b_ts)
                node_route.memory.apply(node_route.memory_updater, grad=True)
            else:
                with torch.no_grad():
                    node_route.store_messages(b_src, b_dst, b_ts)
                    node_route.memory.apply(node_route.memory_updater)

    out = {}
    if collect_metrics:
        for key in ("fused", "node", "sub", "pair"):
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


FUSION_GROUP_KEY = {"pair": "pair_feat", "pair-shuf": "pair_feat", "noise": "pair_feat",
                    "noise-emb": "emb_feat", "emb": "emb_feat", "stat": "stat_feat",
                    "meta": "meta_feat"}


def assert_ctx_ready(fusion_groups, extras: dict, ctx_cols: int | None = None,
                     use_dt: bool = True, extra_dim: int = 0) -> None:
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
    import random as _random

    blob = {
        "epoch": int(epoch),
        "models": {k: v.state_dict() for k, v in models.items() if v is not None},
        "opt": opt.state_dict(),
        "best_val": float(best_val),
        "best_state": best_state,
        "log_epochs": list(log_epochs),
        "args": ({k: v for k, v in vars(args).items() if k != "resume"} if args is not None else None),
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
    tmp.replace(path)


def load_ckpt(path, *, models, opt, sampler=None, args=None) -> dict:
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
    g_all = np.concatenate([r[0] for r in gate_rows])
    c_all = np.concatenate([r[1] for r in gate_rows], axis=0)
    y_all = np.concatenate([r[2] for r in gate_rows])
    g_sd = float(np.std(g_all))
    corr_defined = g_sd > 1e-6
    if corr_defined:
        rg = _rank(g_all)
        corr = {n: float(np.corrcoef(rg, _rank(c_all[:, i]))[0, 1])
                for i, n in enumerate(GATE_CTX_NAMES)}
    else:
        corr = {n: None for n in GATE_CTX_NAMES}
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
    meta_window = a.meta_window_frac * train_span if a.meta_window_frac > 0 else None
    feat_dim = 3 if a.input == "struct" else 0
    pair_dims = parse_dims(a.pair_dims)
    pair_dim = len(pair_dims) if a.pair_features else 0

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
    emb_dim = 2 * a.memory_dim + a.sub_hidden + pair_dim
    _pair_like = ("pair" in fusion_groups or "pair-shuf" in fusion_groups
                  or "noise" in fusion_groups)
    _emb_like = ("emb" in fusion_groups or "noise-emb" in fusion_groups)
    extra_dim = ((pair_dim if _pair_like else 0)
                 + (6 if "stat" in fusion_groups else 0)
                 + (2 if "meta" in fusion_groups else 0)
                 + ((2 * a.memory_dim + a.sub_hidden + pair_dim) if _emb_like else 0)
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
    edge_lookup = None
    calibrator = RouteCalibrator() if a.route_calib else None
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
                           edge_msg_dim=a.edge_msg_dim).to(device)
    sub_route = SubgraphRoute(hidden=a.sub_hidden, max_z=a.max_z, num_layers=a.sub_layers,
                              time_dim=a.sub_time_dim, feat_dim=feat_dim, pair_dim=pair_dim,
                              dropout=a.dropout, use_edge_gate=not a.no_edge_gate).to(device)
    if a.share_time_encode:
        assert a.sub_time_dim == a.time_dim, (
            "共享时间编码要求两条路线维度相同："
            f"--sub-time-dim {a.sub_time_dim} != --time-dim {a.time_dim}")
        sub_route.time_encoder = node_route.time_encoder
    if a.fusion == "feat":
        fusion = FeatureFusion(in_dim=ctx_dim(not a.ctx_no_dt) + extra_dim,
                               hidden=a.sub_hidden, dropout=a.dropout).to(device)
    else:
        fusion = GatedFusion(ctx_dim=ctx_dim(not a.ctx_no_dt) + extra_dim,
                             const_gate=a.const_gate).to(device)
    seen: dict[int, torch.nn.Parameter] = {}
    pair_route = None
    if a.route == "pair":
        if pair_dim <= 0:
            raise SystemExit("--route pair 需要 --pair-features（否则 pair_dim=0，该分支没有输入）")
        from src.models.pair_route import PairRoute
        pair_route = PairRoute(pair_dim=pair_dim, hidden=a.sub_hidden,
                               dropout=a.dropout).to(device)
    all_params = (list(node_route.parameters()) + list(sub_route.parameters())
                  + list(fusion.parameters()))
    if pair_route is not None:
        all_params += list(pair_route.parameters())
    for p in all_params:
        seen.setdefault(id(p), p)
    params = list(seen.values())
    n_params = sum(p.numel() for p in params)
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
        node_route.memory.init_memory()
        warmup = epoch <= a.warmup_epochs
        tr = run_stream(node_route, sub_route, fusion, g_train, g_train, tr_src, tr_dst, tr_ts,
                        train_sampler, cfg=cfg, device=device, feat_dim=feat_dim,
                        criterion=criterion, optimizer=opt, warmup=warmup, route=a.route,
                        lambda_node=a.lambda_node, lambda_sub=a.lambda_sub,
                        lambda_cons=a.lambda_cons, cons_kind=a.cons_kind,
                        update_memory=True, collect_metrics=True, pair_route=pair_route,
                        edge_lookup=edge_lookup,
                        calib=calib_use, calib_on=a.route_calib,
                        dump_scores=(a.route_calib and epoch == a.warmup_epochs))
        if a.route_calib and epoch == a.warmup_epochs:
            _rows = tr.get("score_rows") or []
            if not _rows:
                raise SystemExit("--route-calib：预热末轮未收到两路分数，无法拟合（不应发生）")
            _pn = np.concatenate([r[0] for r in _rows])
            _ps = np.concatenate([r[1] for r in _rows])
            _rn = calibrator.fit("node", _pn)
            _rs = calibrator.fit("sub", _ps)
            calib_use = calibrator
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
        if a.route == "pair":
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
        ckpt_path.unlink()
        print(f"[{tag}] 已删除中继点 {ckpt_path.name}（run 已完成）")
    print(f"[{tag}] 结果写入 {out_dir / 'summary.json'}")
    if a.gate_stats and test["gate_rows"]:
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
