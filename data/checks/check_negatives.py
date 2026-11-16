"""负采样自检。

验证：

- 训练采样器的池 = **训练期**节点；评测采样器的池 = **全图**节点（与参考实现一致）
- 目标节点替换式：源节点保持为 batch 的源节点不变
- 评测采样器给定固定种子时**完全可复现**（重置后重抽结果一致）
- 采样出的负样本确实取自对应池（不越界）
- 训练池不含"训练期之外才出现的节点"（这是训练路径的防泄漏要求）
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]))
    from data import config, negatives, split
    from data.checks._common import Checker
else:
    from .. import config, negatives, split
    from ._common import Checker


def run(names: list[str]) -> bool:
    ok_all = True
    for name in names:
        c = Checker(f"check_negatives :: {name}")
        try:
            arr = split.load_splits(name)
        except FileNotFoundError as exc:
            c.check("切分产物存在", False, str(exc))
            ok_all &= c.report()
            continue

        train_src, train_dst = arr["train_src"], arr["train_dst"]
        full_src, full_dst = arr["full_src"], arr["full_dst"]

        train_sampler = negatives.make_train_sampler(train_src, train_dst)
        val_sampler, test_sampler = negatives.make_eval_samplers(full_src, full_dst)

        # --- 池的构成 -------------------------------------------------------- #
        train_pool = set(int(x) for x in train_sampler.unique_dst_node_ids)
        full_pool = set(int(x) for x in val_sampler.unique_dst_node_ids)
        full_nodes = set(int(x) for x in full_src) | set(int(x) for x in full_dst)

        def _set_cmp(label: str, actual: set, expected: set) -> None:
            # 不把两个大集合直接塞进 detail——失败时会打印上万个数字。
            if actual == expected:
                c.check(label, True, f"n={len(actual)}")
            else:
                c.check(
                    label,
                    False,
                    f"size {len(actual)} vs {len(expected)}; "
                    f"仅前者有 {len(actual - expected)} 个 例:{sorted(actual - expected)[:5]}; "
                    f"仅后者有 {len(expected - actual)} 个 例:{sorted(expected - actual)[:5]}",
                )

        _set_cmp(
            "训练采样池 == 训练期目标节点集合",
            train_pool,
            set(int(x) for x in train_dst),
        )
        _set_cmp(
            "评测采样池 == 全图目标节点集合",
            full_pool,
            set(int(x) for x in full_dst),
        )
        c.check("评测池 ⊇ 训练池", train_pool <= full_pool)
        c.check(
            "评测池严格大于训练池（含未来节点，与参考实现一致）",
            len(full_pool) > len(train_pool),
            f"train={len(train_pool)} full={len(full_pool)}",
        )

        # 训练池不得含"首次出现就在训练期之后"的节点（训练时根本不存在的节点）。
        #
        # 注意定义："未来节点"= **首次出现时间晚于训练期末**，而不是
        # "最后出现时间晚于训练期末"。后者在转导式设定下大量存在且是正常的
        # （同一个节点可以在训练、验证、测试期反复出现），早期版本的检查就是
        # 因为把两者混为一谈而误报。
        train_max_ts = float(arr["train_ts"].max())
        edge_nodes = np.concatenate([arr["full_src"], arr["full_dst"]])
        edge_times = np.concatenate([arr["full_ts"], arr["full_ts"]])
        sorter = np.argsort(edge_times, kind="stable")
        sorted_nodes = edge_nodes[sorter]
        sorted_times = edge_times[sorter]
        # 每个节点在时间升序中第一次出现的位置
        _, first_pos = np.unique(sorted_nodes, return_index=True)
        future_only = {
            int(n)
            for n, tt in zip(sorted_nodes[first_pos].tolist(), sorted_times[first_pos].tolist())
            if tt > train_max_ts
        }

        c.equal(
            "训练池不含首次出现即在训练期之后的节点",
            len(train_pool & future_only),
            0,
            f"首次出现晚于训练期末的节点数={len(future_only)}",
        )
        # ------------------------------------------------------------------ #
        # 下面只**记录两个协议事实**，不计为失败。两者是**不同**的量，别混：
        #
        #   future_only : 首次出现时间晚于训练期末的节点 —— 训练时根本不存在。
        #                 它与训练池的交集 **恒为 0**（上面那一条已验证）。
        #   later_nodes : 在训练期末之后还有过交互的节点 —— 它当然可能同时在
        #                 训练期就出现过，所以它与训练池的交集**非 0**。
        #
        # 后者意味着：**训练负样本有机会在将来变成正样本**（因为负样本是从训练池
        # 采的目标节点，而该节点未来可能真的与源节点连边）。
        # 这是参考实现（TAMI / TGN-SEAL 的 random 策略）的既有性质，我们照抄不改。
        # ------------------------------------------------------------------ #
        later_mask = arr["full_ts"] > train_max_ts
        later_nodes = set(int(x) for x in arr["full_src"][later_mask]) | set(
            int(x) for x in arr["full_dst"][later_mask]
        )
        n_reappear = len(train_pool & later_nodes)
        c.check(
            "记录：训练池∩future_only（应为 0，协议要求）",
            len(train_pool & future_only) == 0,
            f"{len(train_pool & future_only)}",
        )
        c.check(
            "记录：训练池∩later_nodes（非 0，协议固有，非失败）",
            True,
            f"{n_reappear} / {len(train_pool)}"
            f"（占 {100.0 * n_reappear / len(train_pool):.1f}%）；"
            "这些节点未来可能成为正样本，属参考实现的 random 策略固有性质",
        )

        # --- 目标替换式：源节点保持不变 -------------------------------------- #
        batch = np.asarray(train_src[:64], dtype=np.int64)
        neg_src, neg_dst = train_sampler.sample_batch(batch)
        c.check("neg_src 与 batch 源节点逐位相同", bool(np.array_equal(neg_src, batch)))
        c.equal("负样本数量与 batch 一致", neg_dst.size, batch.size)
        c.check(
            "负样本目标节点都在训练池内",
            bool(np.isin(neg_dst, train_sampler.unique_dst_node_ids).all()),
        )

        # --- 评测可复现性 ---------------------------------------------------- #
        s1 = val_sampler.sample_batch(batch)[1].copy()
        s2 = val_sampler.sample_batch(batch)[1].copy()
        c.check("未重置时连续两次采样不同（随机性存在）", not bool(np.array_equal(s1, s2)))
        val_sampler.reset_random_state()
        s3 = val_sampler.sample_batch(batch)[1]
        val_sampler.reset_random_state()
        s4 = val_sampler.sample_batch(batch)[1]
        c.check(
            "重置随机状态后采样完全可复现",
            bool(np.array_equal(s3, s4)),
            "评测负样本在不同 run 之间必须一致",
        )

        # --- 独立种子互不干扰 ------------------------------------------------ #
        a = val_sampler.sample_batch(batch)[1].copy()
        b = test_sampler.sample_batch(batch)[1].copy()
        c.check("val 与 test 采样器使用不同种子", not bool(np.array_equal(a, b)))

        # --- 采样规模与类型 -------------------------------------------------- #
        big = val_sampler.sample_batch(np.asarray(full_src[:1000], dtype=np.int64))[1]
        c.equal("大批量采样规模正确", big.size, 1000)
        c.check("负样本 id 均为正整数", bool((big >= 1).all()), f"min={big.min()}")
        # 强化边界检查：负样本必须落在**评测池**内（而不是仅仅不超过最大值）。
        # 原先写成 `big <= max(full_nodes)` 只是弱约束，几乎永远成立，等于没查。
        c.check(
            "负样本 id 都落在评测池内",
            bool(np.isin(big, val_sampler.unique_dst_node_ids).all()),
            f"池外个数={int((~np.isin(big, val_sampler.unique_dst_node_ids)).sum())}",
        )
        c.check(
            "负样本 id 不超过全图最大节点 id",
            bool((big <= max(full_nodes)).all()),
            f"max={big.max()} full_max={max(full_nodes)}",
        )

        ok_all &= c.report()
    return ok_all


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="负采样自检")
    p.add_argument("--dataset", choices=sorted(config.DATASETS), default=None)
    p.add_argument("--fast", action="store_true")
    args = p.parse_args(argv)

    names = [args.dataset] if args.dataset else list(
        config.FAST_DATASETS if args.fast else config.ALL_DATASETS
    )
    return 0 if run(names) else 1


if __name__ == "__main__":
    raise SystemExit(main())
