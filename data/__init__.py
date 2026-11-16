"""数据层：动态图链接预测的数据准备、切分、负采样与指标。

协议全部对齐参考实现（TGB / DyGLib / TGN-SEAL / TAMI），见 ``data/README.md``。
本包只依赖 numpy / pandas / scikit-learn，不依赖 torch。
"""

from __future__ import annotations

__all__ = [
    "config",
    "io",
    "preprocess",
    "split",
    "negatives",
    "metrics",
]
