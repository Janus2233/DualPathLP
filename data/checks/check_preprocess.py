
from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]))
    from data import config, io
    from data import preprocess as pre_mod
    from data.checks._common import Checker
else:
    from .. import config, io
    from .. import preprocess as pre_mod
    from ._common import Checker

EXPECTED_COLUMNS = ["Unnamed: 0", "u", "i", "ts", "label", "idx"]


def run(names: list[str], *, check_raw_untouched: bool = True) -> bool:
    ok_all = True
    for name in names:
        c = Checker(f"check_preprocess :: {name}")
        paths = config.processed_paths(name)

        if not paths["edges"].is_file():
            c.check("预处理产物存在", False, f"缺少 {paths['edges']}；先跑 python -m data.build")
            ok_all &= c.report()
            continue

        df = pd.read_csv(paths["edges"])
        c.equal("CSV 列名与参考实现一致", list(df.columns), EXPECTED_COLUMNS)
        c.check("边数 > 0", len(df) > 0, f"{len(df)}")

        u = df["u"].to_numpy(np.int64)
        i = df["i"].to_numpy(np.int64)
        ts = df["ts"].to_numpy(np.float64)
        idx = df["idx"].to_numpy(np.int64)
        label = df["label"].to_numpy(np.float64)

        c.equal("节点 id 最小值为 1（1-based）", int(min(u.min(), i.min())), config.ID_BASE)
        c.equal("边 id 最小值为 1（1-based）", int(idx.min()), config.ID_BASE)
        c.check("无自环", not bool(np.any(u == i)), f"自环数={int(np.count_nonzero(u == i))}")

        c.check(
            "ts 单调不减",
            bool(np.all(np.diff(ts) >= 0)),
            f"逆序步数={int((np.diff(ts) < 0).sum())}",
        )
        c.check("label 恒为 1.0", bool(np.all(label == 1.0)))
        c.check("ts 无缺失", bool(np.isfinite(ts).all()))

        c.check("边 id 唯一", len(np.unique(idx)) == len(idx))
        c.check(
            "边 id 连续 1..E",
            bool(np.array_equal(np.sort(idx), np.arange(1, len(idx) + 1, dtype=np.int64))),
        )

        edge_feat = np.load(paths["edge_feat"])
        node_feat = np.load(paths["node_feat"])
        c.equal("边特征形状 (E+1, 172)", tuple(edge_feat.shape), (len(df) + 1, config.EDGE_FEAT_DIM))
        c.equal(
            "节点特征形状 (Nmax+1, 172)",
            tuple(node_feat.shape),
            (int(max(u.max(), i.max())) + 1, config.NODE_FEAT_DIM),
        )
        c.check("边特征第 0 行为 padding（全 0）", bool(np.all(edge_feat[0] == 0)))
        c.check("节点特征第 0 行为 padding（全 0）", bool(np.all(node_feat[0] == 0)))

        try:
            spec = config.DATASETS[name]
            raw = io.read_raw(spec, verify=False)

            pe2 = pre_mod.preprocess(raw)
            c.check(
                "预处理可复现（重跑结果逐位一致）",
                bool(np.array_equal(pe2.src, u) and np.array_equal(pe2.dst, i)),
                "重跑后的 (u,i) 与已存产物不一致",
            )
        except Exception as exc:
            c.check("预处理可复现（重跑结果逐位一致）", False, f"{type(exc).__name__}: {exc}")

        if check_raw_untouched:
            try:
                io.verify_dataset(config.DATASETS[name])
                c.check("原始文件 SHA256 未变", True)
            except Exception as exc:
                c.check("原始文件 SHA256 未变", False, str(exc))

        ok_all &= c.report()
    return ok_all


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="预处理产物自检")
    p.add_argument("--dataset", choices=sorted(config.DATASETS), default=None)
    p.add_argument("--fast", action="store_true")
    args = p.parse_args(argv)

    names = [args.dataset] if args.dataset else list(
        config.FAST_DATASETS if args.fast else config.ALL_DATASETS
    )
    return 0 if run(names) else 1


if __name__ == "__main__":
    raise SystemExit(main())
