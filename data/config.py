
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "dataset"
RUNS_DIR = PROJECT_ROOT / "runs"
PROCESSED_DIR = RUNS_DIR / "data"


TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

NEW_NODE_RATIO = 0.10

NEW_NODE_SEED = 2020

NODE_FEAT_DIM = 172
EDGE_FEAT_DIM = 172

EVAL_VAL_SEED = 0
EVAL_TEST_SEED = 2

TRAIN_NEG_PER_POS = 1

ID_BASE = 1




@dataclass(frozen=True)
class DatasetSpec:

    name: str
    filename: str
    sha256: str
    n_rows: int
    n_nodes: int
    note: str = ""

    @property
    def path(self) -> Path:
        return DATASET_DIR / self.filename


DATASETS: dict[str, DatasetSpec] = {
    "college_msg": DatasetSpec(
        name="college_msg",
        filename="CollegeMsg.txt",
        sha256="E00BA2415373DEE52C00616065BCCEAA4750E78DE60D1855C76470600F10740F",
        n_rows=59_835,
        n_nodes=1_899,
        note="文件已按 ts 升序；无自环。",
    ),
    "email_eu_core": DatasetSpec(
        name="email_eu_core",
        filename="email-Eu-core-temporal.txt",
        sha256="1835C94DC6685DCF2A614A1B6A0275B323DD4002F7AA111FBE9F76DB8864391F",
        n_rows=332_334,
        n_nodes=986,
        note="文件未完全排序（2 处 ts 逆序）；无自环；49.5% 的行共享时间戳。",
    ),
    "reddit_hyperlinks": DatasetSpec(
        name="reddit_hyperlinks",
        filename="soc-redditHyperlinks-body.txt",
        sha256="81C77BE89D67E4135D545F7D74B8B961386D74763C314CE14C708F16A601EE4B",
        n_rows=286_561,
        n_nodes=35_776,
        note="文件已按 ts 升序；无自环；19.8% 的行共享时间戳。",
    ),
    "sx_mathoverflow": DatasetSpec(
        name="sx_mathoverflow",
        filename="sx-mathoverflow.txt",
        sha256="2BF52F784CDBA69730F912633AA1414B381059406DED2C38BC37658EAD5B0773",
        n_rows=506_550,
        n_nodes=24_818,
        note=(
            "文件未完全排序（629 处 ts 逆序）；含 22.9% 自环（116,109 条），"
            "按参考实现丢弃，其中 59 个节点仅出现在自环中。"
        ),
    ),
}

FAST_DATASETS: tuple[str, ...] = ("college_msg", "email_eu_core")
MAIN_DATASETS: tuple[str, ...] = ("sx_mathoverflow", "reddit_hyperlinks")
ALL_DATASETS: tuple[str, ...] = tuple(DATASETS)


def processed_paths(name: str) -> dict[str, Path]:
    if name not in DATASETS:
        raise KeyError(f"未知数据集 {name!r}；可选：{sorted(DATASETS)}")
    d = PROCESSED_DIR / name
    return {
        "dir": d,
        "edges": d / f"ml_{name}.csv",
        "edge_feat": d / f"ml_{name}.npy",
        "node_feat": d / f"ml_{name}_node.npy",
    }


def splits_path(name: str) -> Path:
    if name not in DATASETS:
        raise KeyError(f"未知数据集 {name!r}；可选：{sorted(DATASETS)}")
    return PROCESSED_DIR / name / f"splits_{name}.npz"


def as_int_array(x) -> np.ndarray:
    return np.asarray(x, dtype=np.int64)
