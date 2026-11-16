"""评测指标。逐行对齐 ``参考模型/TAMI/utils/metrics.py:6-23``。

**主协议（默认，与 TAMI / TGN-SEAL 一致）**

1-vs-1 二分类：每批把正样本分数与负样本分数拼起来，算 AP 与 AUC；
然后把**每批的指标跨批平均**。

- 定义：``TAMI/utils/metrics.py:6-23``
  ``average_precision_score(labels, predicts)`` / ``roc_auc_score(labels, predicts)``
- 聚合：``TGN-SEAL/evaluation/evaluation.py:56-63`` 的 ``np.mean(val_ap), np.mean(val_auc)``

**关于 MRR：本协议下不要用 TGB 的 MRR。**

TAMI 里那行 ``evaluator.eval(input_dict)['mrr']`` 是把 **1 正 1 负**的 batch-local 字典
喂给 TGB 评测器（``TAMI/evaluate_models_utils.py:370-375``），得到的**不是全局排名 MRR**，
在 1-vs-1 设定下没有排名含义。因此本模块不提供它。

若将来要报真正的排名指标，必须**同时改变评测协议**（1-vs-N），
这属于决策事项（见仓库 ``AGENTS.md`` §2），本模块只预留 ``ranking_metrics`` 供那种协议使用。
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

#: 本协议下唯一对外承诺的指标名。
PRIMARY_METRICS: tuple[str, ...] = ("average_precision", "roc_auc")


def get_link_prediction_metrics(predicts: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    """单个 batch 的 AP 与 AUC。

    与 ``TAMI/utils/metrics.py:6-23`` 等价（那边用 torch.Tensor，这边用 ndarray）。
    """
    predicts = np.asarray(predicts, dtype=np.float64).ravel()
    labels = np.asarray(labels).ravel()

    if predicts.shape != labels.shape:
        raise ValueError(f"predicts 与 labels 形状不一致：{predicts.shape} vs {labels.shape}")
    if predicts.size == 0:
        raise ValueError("空 batch，无法计算指标。")

    n_pos = int(np.count_nonzero(labels == 1))
    n_neg = int(labels.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        # 参考实现在这种 batch 上会直接抛错或给出无意义值；显式报错更好定位。
        raise ValueError(
            f"batch 内只有单一类别（pos={n_pos}, neg={n_neg}），AP/AUC 无定义。"
        )

    return {
        "average_precision": float(average_precision_score(y_true=labels, y_score=predicts)),
        "roc_auc": float(roc_auc_score(y_true=labels, y_score=predicts)),
    }


def aggregate_batch_metrics(batch_metrics: list[dict[str, float]]) -> dict[str, float]:
    """跨 batch 平均。对应 ``TGN-SEAL/evaluation/evaluation.py:56-63``。

    注意：这是**先 per-batch 再平均**，与"把所有 batch 的分数拼起来算一次"
    （micro 式）不等价。参考实现用的是前者，我们照抄；两种口径不可混用比较。
    """
    if not batch_metrics:
        raise ValueError("没有 batch 指标可聚合。")
    keys = batch_metrics[0].keys()
    return {
        k: float(np.mean([m[k] for m in batch_metrics]))
        for k in keys
    }


def ranking_metrics(
    ranks: np.ndarray,
    *,
    k_values: tuple[int, ...] = (10, 20, 50),
) -> dict[str, float]:
    """真正的排名指标（MRR / Hits@K）。

    **只有在 1-vs-N 排名协议下才有意义。** 默认协议（1-vs-1）不要调用。

    Args:
        ranks: 每个正样本在候选集中的名次（1-based）。调用方负责保证
            排名口径一致（是否过滤已观测边、候选集如何构造）。
    """
    ranks = np.asarray(ranks, dtype=np.float64).ravel()
    if ranks.size == 0:
        raise ValueError("ranks 为空。")
    if np.any(ranks < 1):
        raise ValueError("ranks 必须是 1-based 的正整数。")

    out = {"mrr": float(np.mean(1.0 / ranks))}
    for k in k_values:
        out[f"hits@{k}"] = float(np.mean(ranks <= k))
    return out


def compute_rank(
    pos_score: np.ndarray,
    neg_scores: np.ndarray,
    *,
    higher_is_better: bool = True,
) -> np.ndarray:
    """1-vs-N：正样本在 ``[正] + N 个负`` 中的名次（1-based）。

    Args:
        pos_score: ``(B,)`` 正样本分数。
        neg_scores: ``(B, N)`` 对应负样本分数。
        higher_is_better: 分数越大越可能是边（点积/内积类打分器为 True）。

    Returns:
        ``(B,)`` int64 名次。名次 1 表示正样本排第一。
    """
    pos = np.asarray(pos_score, dtype=np.float64).reshape(-1, 1)
    neg = np.asarray(neg_scores, dtype=np.float64)
    if neg.ndim != 2 or neg.shape[0] != pos.shape[0]:
        raise ValueError(f"neg_scores 形状应为 (B, N) 且 B 与 pos_score 一致；得到 {neg.shape}")

    if higher_is_better:
        # 严格大于正样本分数的负样本个数 + 1；并列按平均名次会导致非整数，
        # 这里采用与 EAGLE 定义一致的写法：rank = #{neg > pos} + 1
        worse = np.count_nonzero(neg > pos, axis=1)
    else:
        worse = np.count_nonzero(neg < pos, axis=1)
    return (worse + 1).astype(np.int64)
