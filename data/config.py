"""数据层常量与数据集登记表。

所有协议常量都明确标注来源（参考实现的哪个文件），不自行发明。
路径一律相对本仓库根（``Model experiments/``），不写死本机绝对路径。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------- #
# 路径
# --------------------------------------------------------------------------- #

# data/config.py -> data/ -> 仓库根
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "dataset"
RUNS_DIR = PROJECT_ROOT / "runs"
PROCESSED_DIR = RUNS_DIR / "data"

# --------------------------------------------------------------------------- #
# 协议常量（每一项都对应参考实现的一个具体位置，见 data/README.md §1）
# --------------------------------------------------------------------------- #

#: 训练/验证/测试比例。TAMI: utils/load_configs.py:45-46 (val 0.15 / test 0.15)
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

#: 归纳留出：抽 10% 的"测试期节点"作为新节点，并把它们的全部边移出训练集。
#: TAMI: utils/DataLoader.py:106-133 / TGN-SEAL: utils/data_processing.py:41-56
NEW_NODE_RATIO = 0.10

#: 归纳留出用的种子。参考实现硬编码 2020，我们照抄以保证可比。
#: TAMI: utils/DataLoader.py:106 (random.seed(2020))
NEW_NODE_SEED = 2020

#: 特征维度。参考实现把所有数据集零填充到 172 维。
#: TAMI: utils/DataLoader.py:81-83 (NODE_FEAT_DIM = EDGE_FEAT_DIM = 172)
NODE_FEAT_DIM = 172
EDGE_FEAT_DIM = 172

#: 评测负采样器的固定种子，保证不同 run 用同一批评测负样本。
#: TGN-SEAL: train_self_supervised.py:196,199 (val seed=0, test seed=2)
EVAL_VAL_SEED = 0
EVAL_TEST_SEED = 2

#: 训练负采样：每个正样本配 1 个负样本。
#: TAMI: train_link_prediction.py:178 / TGN-SEAL: train_self_supervised.py (NUM_NEG=1)
TRAIN_NEG_PER_POS = 1

#: 节点 id / 边 id 从 1 开始，0 行留给 padding。
#: TGN-SEAL: utils/preprocess_csv_data.py:41-56 (new_df.u += 1; new_df.i += 1; new_df.idx += 1)
ID_BASE = 1


# --------------------------------------------------------------------------- #
# 数据集登记表
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class DatasetSpec:
    """一个原始数据集的登记信息。

    ``sha256`` 与仓库根 ``AGENTS.md`` 附录记录的校验值一致，用于防止原始文件被误改。
    """

    name: str
    filename: str
    sha256: str
    n_rows: int
    n_nodes: int
    note: str = ""

    @property
    def path(self) -> Path:
        return DATASET_DIR / self.filename


#: 四个 SNAP 数据集。``note`` 只记录我们实测到的、会影响管线的事实。
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

#: 快速自检档（小、秒级）；主实验档见 README §5。
FAST_DATASETS: tuple[str, ...] = ("college_msg", "email_eu_core")
MAIN_DATASETS: tuple[str, ...] = ("sx_mathoverflow", "reddit_hyperlinks")
ALL_DATASETS: tuple[str, ...] = tuple(DATASETS)


def processed_paths(name: str) -> dict[str, Path]:
    """返回某数据集三个产出文件的路径（与 TGB/DyGLib 布局一致）。"""
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
    """返回切分产物的路径（``npz``）。"""
    if name not in DATASETS:
        raise KeyError(f"未知数据集 {name!r}；可选：{sorted(DATASETS)}")
    return PROCESSED_DIR / name / f"splits_{name}.npz"


def as_int_array(x) -> np.ndarray:
    """统一的整型转换，避免不同平台默认 dtype 不一致。"""
    return np.asarray(x, dtype=np.int64)
