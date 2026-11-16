
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import config


@dataclass
class NegativeEdgeSampler:

    src_node_ids: np.ndarray
    dst_node_ids: np.ndarray
    seed: int | None = None
    negative_sample_strategy: str = "random"
    forbidden_pairs: set[tuple[int, int]] | None = None

    unique_src_node_ids: np.ndarray = field(init=False)
    unique_dst_node_ids: np.ndarray = field(init=False)
    _random_state: np.random.RandomState | None = field(init=False, default=None)
    n_neg_drawn: int = field(init=False, default=0)
    n_neg_rejected: int = field(init=False, default=0)

    MAX_REJECT_ROUNDS: int = 20

    def __post_init__(self) -> None:
        if self.negative_sample_strategy != "random":
            raise NotImplementedError(
                f"只实现了 'random' 策略；收到 {self.negative_sample_strategy!r}。"
                "需要 historical/inductive 时请先与作者确认（见模块 docstring）。"
            )
        self.unique_src_node_ids = np.unique(np.asarray(self.src_node_ids, dtype=np.int64))
        self.unique_dst_node_ids = np.unique(np.asarray(self.dst_node_ids, dtype=np.int64))
        if self.unique_dst_node_ids.size == 0:
            raise ValueError("目标节点池为空，无法采样负样本。")
        if self.seed is not None:
            self._random_state = np.random.RandomState(self.seed)
        if self.forbidden_pairs is not None:
            self.forbidden_pairs = {(int(u), int(v)) for u, v in self.forbidden_pairs}


    def random_sample_dst(self, size: int) -> np.ndarray:
        n = self.unique_dst_node_ids.size
        if self._random_state is None:
            idx = np.random.randint(0, n, size)
        else:
            idx = self._random_state.randint(0, n, size)
        return self.unique_dst_node_ids[idx]

    def sample(self, size: int) -> tuple[np.ndarray, np.ndarray]:
        neg_dst = self.random_sample_dst(size)
        if self._random_state is None:
            neg_src = self.unique_src_node_ids[np.random.randint(0, self.unique_src_node_ids.size, size)]
        else:
            neg_src = self.unique_src_node_ids[
                self._random_state.randint(0, self.unique_src_node_ids.size, size)
            ]
        return neg_src, neg_dst

    def sample_batch(self, batch_src_node_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        src = np.asarray(batch_src_node_ids, dtype=np.int64)
        neg_dst = self.random_sample_dst(size=len(src))
        self.n_neg_drawn += len(src)
        if self.forbidden_pairs is None:
            return src, neg_dst
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
        return {
            "strict": self.forbidden_pairs is not None,
            "n_neg_drawn": int(self.n_neg_drawn),
            "n_neg_rejected": int(self.n_neg_rejected),
            "reject_rate": (self.n_neg_rejected / self.n_neg_drawn) if self.n_neg_drawn else 0.0,
        }

    def reset_random_state(self) -> None:
        if self.seed is None:
            raise ValueError("无种子，无法重置；评测用采样器必须指定 seed。")
        self._random_state = np.random.RandomState(self.seed)




def make_train_sampler(train_src: np.ndarray, train_dst: np.ndarray, *,
                       forbidden_pairs: set[tuple[int, int]] | None = None) -> NegativeEdgeSampler:
    return NegativeEdgeSampler(train_src, train_dst, seed=None, forbidden_pairs=forbidden_pairs)


def make_eval_samplers(
    full_src: np.ndarray, full_dst: np.ndarray, *,
    forbidden_pairs: set[tuple[int, int]] | None = None,
) -> tuple[NegativeEdgeSampler, NegativeEdgeSampler]:
    return (
        NegativeEdgeSampler(full_src, full_dst, seed=config.EVAL_VAL_SEED,
                            forbidden_pairs=forbidden_pairs),
        NegativeEdgeSampler(full_src, full_dst, seed=config.EVAL_TEST_SEED,
                            forbidden_pairs=forbidden_pairs),
    )
