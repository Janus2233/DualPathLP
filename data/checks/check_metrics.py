
from __future__ import annotations

import argparse
import sys

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]))
    from data import config, metrics
    from data.checks._common import Checker
else:
    from .. import config, metrics
    from ._common import Checker


def run() -> bool:
    c = Checker("check_metrics")

    m = metrics.get_link_prediction_metrics(
        predicts=[0.9, 0.8, 0.2, 0.1], labels=[1, 1, 0, 0]
    )
    c.close("完全可分 AP=1", m["average_precision"], 1.0, 1e-12)
    c.close("完全可分 AUC=1", m["roc_auc"], 1.0, 1e-12)

    m = metrics.get_link_prediction_metrics(
        predicts=[0.5, 0.5, 0.5, 0.5], labels=[1, 1, 0, 0]
    )
    c.close("全同分 AUC=0.5", m["roc_auc"], 0.5, 1e-12)

    m = metrics.get_link_prediction_metrics(
        predicts=[0.1, 0.2, 0.8, 0.9], labels=[1, 1, 0, 0]
    )
    c.close("完全反序 AUC=0", m["roc_auc"], 0.0, 1e-12)

    c.close(
        "AP 手算一致",
        metrics.get_link_prediction_metrics([0.9, 0.8, 0.2, 0.1], [1, 1, 0, 0])[
            "average_precision"
        ],
        1.0,
        1e-12,
    )
    c.close(
        "AP 手算一致（交错情形）",
        metrics.get_link_prediction_metrics([0.9, 0.15, 0.8, 0.1], [1, 1, 0, 0])[
            "average_precision"
        ],
        (1.0 + 2.0 / 3.0) / 2.0,
        1e-12,
    )

    try:
        metrics.get_link_prediction_metrics([0.1, 0.2], [1, 1])
        c.check("单类别 batch 抛错", False, "未抛错")
    except ValueError:
        c.check("单类别 batch 抛错", True)

    b1 = metrics.get_link_prediction_metrics([0.9, 0.1], [1, 0])
    b2 = metrics.get_link_prediction_metrics([0.9, 0.8, 0.2, 0.1], [1, 1, 0, 0])
    agg = metrics.aggregate_batch_metrics([b1, b2])
    c.close("跨 batch 平均 = 逐 batch 均值", agg["roc_auc"], (b1["roc_auc"] + b2["roc_auc"]) / 2, 1e-12)

    b_easy = metrics.get_link_prediction_metrics([0.9, 0.1], [1, 0])
    b_hard = metrics.get_link_prediction_metrics([0.3, 0.7], [1, 0])
    per_batch = metrics.aggregate_batch_metrics([b_easy, b_hard])["roc_auc"]
    micro = metrics.get_link_prediction_metrics([0.9, 0.1, 0.3, 0.7], [1, 0, 1, 0])[
        "roc_auc"
    ]
    c.close("异质 batch：per-batch 平均 = (1+0)/2", per_batch, 0.5, 1e-12)
    c.check(
        "聚合口径为 per-batch 平均而非 micro",
        abs(per_batch - micro) > 1e-9,
        f"per-batch={per_batch:.6f} micro={micro:.6f}",
    )

    ranks = np.array([1, 2, 5, 10, 100], dtype=np.int64)
    rm = metrics.ranking_metrics(ranks, k_values=(10, 50))
    c.close("MRR = mean(1/rank)", rm["mrr"], float(np.mean(1.0 / ranks)), 1e-12)
    c.close("Hits@10 手算一致", rm["hits@10"], 4 / 5, 1e-12)
    c.close("Hits@50 手算一致", rm["hits@50"], 4 / 5, 1e-12)

    pos = np.array([0.5, 0.5])
    neg = np.array([[0.9, 0.1], [0.1, 0.2]])
    r = metrics.compute_rank(pos, neg)
    c.check("compute_rank 名次正确", bool(np.array_equal(r, [2, 1])), f"got {r}")

    try:
        metrics.ranking_metrics(np.array([0]), k_values=(10,))
        c.check("rank<1 抛错", False, "未抛错")
    except ValueError:
        c.check("rank<1 抛错", True)

    return c.report()

def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="指标自检（与数据集无关）")
    p.add_argument("--dataset", choices=sorted(config.DATASETS), default=None)
    p.add_argument("--fast", action="store_true")
    p.parse_args(argv)
    return 0 if run() else 1

if __name__ == "__main__":
    raise SystemExit(main())
