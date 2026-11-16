
from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

PRIMARY_METRICS: tuple[str, ...] = ("average_precision", "roc_auc")


def get_link_prediction_metrics(predicts: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    predicts = np.asarray(predicts, dtype=np.float64).ravel()
    labels = np.asarray(labels).ravel()

    if predicts.shape != labels.shape:
        raise ValueError(f"predicts 与 labels 形状不一致：{predicts.shape} vs {labels.shape}")
    if predicts.size == 0:
        raise ValueError("空 batch，无法计算指标。")

    n_pos = int(np.count_nonzero(labels == 1))
    n_neg = int(labels.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        raise ValueError(
            f"batch 内只有单一类别（pos={n_pos}, neg={n_neg}），AP/AUC 无定义。"
        )

    return {
        "average_precision": float(average_precision_score(y_true=labels, y_score=predicts)),
        "roc_auc": float(roc_auc_score(y_true=labels, y_score=predicts)),
    }


def aggregate_batch_metrics(batch_metrics: list[dict[str, float]]) -> dict[str, float]:
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
    pos = np.asarray(pos_score, dtype=np.float64).reshape(-1, 1)
    neg = np.asarray(neg_scores, dtype=np.float64)
    if neg.ndim != 2 or neg.shape[0] != pos.shape[0]:
        raise ValueError(f"neg_scores 形状应为 (B, N) 且 B 与 pos_score 一致；得到 {neg.shape}")

    if higher_is_better:
        worse = np.count_nonzero(neg > pos, axis=1)
    else:
        worse = np.count_nonzero(neg < pos, axis=1)
    return (worse + 1).astype(np.int64)
