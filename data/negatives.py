"""负采样。逐行对齐 ``参考模型/TAMI/utils/utils.py:305-391`` 的 ``NegativeEdgeSampler``。

三种策略参考实现都支持（``random`` / ``historical`` / ``inductive``），默认 ``random``
（``TAMI/utils/load_configs.py``）。我们只实现 ``random``——其余两种在参考实现里
也要配合 ``possible_edges`` 全集构建，代价高且默认不用；需要时再加。

**关键语义（照抄，不自行发挥）**

``random_sample`` 是"源、目标各自独立从池中抽"；但调用方在 `random` 策略下
**丢弃抽到的源节点，改用 batch 自己的源节点**，即实际是**目标节点替换式**：

- 训练：``TAMI/train_link_prediction.py:178-179``
  ``_, batch_neg_dst_node_ids = train_neg_edge_sampler.sample(...)``
  ``batch_neg_src_node_ids = batch_src_node_ids``
- 评测：``TAMI/evaluate_models_utils.py:69-70``（同样的两行）

**池的选择**

- 训练池 = **训练期节点**（``TAMI/train_link_prediction.py:51``）
- 评测池 = **全图节点**（``full_data``，``TAMI/train_link_prediction.py:52,54``），
  且用固定种子（val=0, test=2）保证跨 run 用同一批负样本。

评测池含"未来才出现的节点"是参考实现的既有做法，我们**照抄不改**；
该做法的性质与影响记录在 ``docs/BUGLOG.md``，不在此静默修正。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import config


@dataclass
class NegativeEdgeSampler:
    """随机负边采样器（目标节点替换式）。

    Args:
        src_node_ids: 采样器"看到"的源节点集合（用于建池）。
        dst_node_ids: 采样器"看到"的目标节点集合（**真正决定池**）。
        seed: 固定种子。``None`` 表示用全局 ``np.random``（训练用）；评测必须给种子。
    """

    src_node_ids: np.ndarray
    dst_node_ids: np.ndarray
    seed: int | None = None
    negative_sample_strategy: str = "random"
    # 严格负采样（``--strict-negatives``）：把"图里已存在的边对"传进来，
    # 抽到它们就重抽。``None``（默认）= 完全保持参考实现口径，行为与加该功能前逐位一致。
    forbidden_pairs: set[tuple[int, int]] | None = None

    # 派生状态
    unique_src_node_ids: np.ndarray = field(init=False)
    unique_dst_node_ids: np.ndarray = field(init=False)
    _random_state: np.random.RandomState | None = field(init=False, default=None)
    # 严格模式的统计（用于把"拒绝率"写进 summary.json）
    n_neg_drawn: int = field(init=False, default=0)
    n_neg_rejected: int = field(init=False, default=0)

    # 抽到已存在边对时的最大重抽轮数；超过就把该位置留成原样（并在 stats 里体现）
    MAX_REJECT_ROUNDS: int = 20

    def __post_init__(self) -> None:
        if self.negative_sample_strategy != "random":
            raise NotImplementedError(
                f"只实现了 'random' 策略；收到 {self.negative_sample_strategy!r}。"
                "需要 historical/inductive 时请先与作者确认（见模块 docstring）。"
            )
        # 与参考实现一致：池 = 各自端点集合的 np.unique
        self.unique_src_node_ids = np.unique(np.asarray(self.src_node_ids, dtype=np.int64))
        self.unique_dst_node_ids = np.unique(np.asarray(self.dst_node_ids, dtype=np.int64))
        if self.unique_dst_node_ids.size == 0:
            raise ValueError("目标节点池为空，无法采样负样本。")
        if self.seed is not None:
            self._random_state = np.random.RandomState(self.seed)
        if self.forbidden_pairs is not None:
            self.forbidden_pairs = {(int(u), int(v)) for u, v in self.forbidden_pairs}

    # ------------------------------------------------------------------ #
    # 采样
    # ------------------------------------------------------------------ #

    def random_sample_dst(self, size: int) -> np.ndarray:
        """从目标池中独立均匀抽 ``size`` 个目标节点 id。"""
        n = self.unique_dst_node_ids.size
        if self._random_state is None:
            idx = np.random.randint(0, n, size)
        else:
            idx = self._random_state.randint(0, n, size)
        return self.unique_dst_node_ids[idx]

    def sample(self, size: int) -> tuple[np.ndarray, np.ndarray]:
        """返回 ``(neg_src, neg_dst)``。

        与参考实现一样，``random`` 策略下源节点也是独立抽的；
        调用方按惯例应当丢弃 ``neg_src`` 并改用 batch 的源节点。
        这里仍然返回它，是为了保留与参考实现完全一致的调用形态。
        """
        neg_dst = self.random_sample_dst(size)
        if self._random_state is None:
            neg_src = self.unique_src_node_ids[np.random.randint(0, self.unique_src_node_ids.size, size)]
        else:
            neg_src = self.unique_src_node_ids[
                self._random_state.randint(0, self.unique_src_node_ids.size, size)
            ]
        return neg_src, neg_dst

    def sample_batch(self, batch_src_node_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """训练/评测路径实际使用的形式：**保留源节点，只替换目标节点**。

        对应参考实现的两行::

            _, batch_neg_dst_node_ids = sampler.sample(size=len(batch_src_node_ids))
            batch_neg_src_node_ids = batch_src_node_ids

        注意：此处**不调 ``sample()``**。``sample()`` 会额外抽一批源节点，
        而它们在 ``random`` 策略下必定被丢弃——调用它等于白抽。
        直接抽目标节点，并显式保留 batch 自己的源节点，语义与参考实现完全一致。
        """
        src = np.asarray(batch_src_node_ids, dtype=np.int64)
        neg_dst = self.random_sample_dst(size=len(src))
        self.n_neg_drawn += len(src)
        if self.forbidden_pairs is None:
            return src, neg_dst
        # 严格模式：拒绝重抽。只对"命中"的位置重抽，且用同一个 RandomState，
        # 因此给定种子后仍然完全确定（评测里 reset_random_state 依然保证跨轮固定）。
        for _ in range(self.MAX_REJECT_ROUNDS):
            bad = np.fromiter(
                ((int(u), int(v)) in self.forbidden_pairs for u, v in zip(src, neg_dst)),
                dtype=bool, count=len(src))
            k = int(bad.sum())
            if k == 0:
                break
            self.n_neg_rejected += k
            self.n_neg_drawn += k
            neg_dst[bad] = self.random_sample_dst(size=k)
        return src, neg_dst

    def stats(self) -> dict:
        """严格负采样的统计（非严格模式下 rejected=0、strict=False）。"""
        return {
            "strict": self.forbidden_pairs is not None,
            "n_neg_drawn": int(self.n_neg_drawn),
            "n_neg_rejected": int(self.n_neg_rejected),
            "reject_rate": (self.n_neg_rejected / self.n_neg_drawn) if self.n_neg_drawn else 0.0,
        }

    def reset_random_state(self) -> None:
        """把随机状态重置到初始种子。参考实现评测前必做这一步。"""
        if self.seed is None:
            raise ValueError("无种子，无法重置；评测用采样器必须指定 seed。")
        self._random_state = np.random.RandomState(self.seed)


# ---------------------------------------------------------------------- #
# 便捷构造：与参考实现的三个采样器一一对应
# ---------------------------------------------------------------------- #


def make_train_sampler(train_src: np.ndarray, train_dst: np.ndarray, *,
                       forbidden_pairs: set[tuple[int, int]] | None = None) -> NegativeEdgeSampler:
    """训练用。池 = 训练期节点，不设种子（每 epoch 换一批）。

    ``TAMI/train_link_prediction.py:51``
    """
    return NegativeEdgeSampler(train_src, train_dst, seed=None, forbidden_pairs=forbidden_pairs)


def make_eval_samplers(
    full_src: np.ndarray, full_dst: np.ndarray, *,
    forbidden_pairs: set[tuple[int, int]] | None = None,
) -> tuple[NegativeEdgeSampler, NegativeEdgeSampler]:
    """验证/测试用。池 = **全图**，固定种子。

    ``TAMI/train_link_prediction.py:52,54``（val seed=0, test seed=2）
    """
    return (
        NegativeEdgeSampler(full_src, full_dst, seed=config.EVAL_VAL_SEED,
                            forbidden_pairs=forbidden_pairs),
        NegativeEdgeSampler(full_src, full_dst, seed=config.EVAL_TEST_SEED,
                            forbidden_pairs=forbidden_pairs),
    )
