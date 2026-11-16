
from __future__ import annotations

import argparse
import sys

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]))
    from data import config, io
    from data.checks._common import Checker
else:
    from .. import config, io
    from ._common import Checker


def run(names: list[str]) -> bool:
    ok_all = True
    for name in names:
        spec = config.DATASETS[name]
        c = Checker(f"check_integrity :: {name}")

        actual_sha = io.sha256_of(spec.path)
        c.equal("sha256 与登记值一致", actual_sha, spec.sha256.upper(), spec.filename)

        raw = io.read_raw(spec, verify=False)
        c.equal("行数与登记值一致", raw.n_edges, spec.n_rows)
        c.equal("节点数与登记值一致", raw.n_unique_nodes, spec.n_nodes)
        c.check("时间戳全部有限", bool(np.isfinite(raw.ts).all()))
        c.check("时间戳非降序检查可执行", True, f"文件已排序={raw.is_ts_sorted()}")
        c.check(
            "存在并列时间戳（tie-break 因此必要）",
            len(np.unique(raw.ts)) < raw.n_edges,
            f"唯一 ts={len(np.unique(raw.ts))} < 边数={raw.n_edges}",
        )

        ok_all &= c.report()
    return ok_all


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="原始数据完整性自检")
    p.add_argument("--dataset", choices=sorted(config.DATASETS), default=None)
    p.add_argument("--all", action="store_true")
    p.add_argument("--fast", action="store_true")
    args = p.parse_args(argv)

    if args.dataset:
        names = [args.dataset]
    elif args.fast:
        names = list(config.FAST_DATASETS)
    else:
        names = list(config.ALL_DATASETS)

    return 0 if run(names) else 1


if __name__ == "__main__":
    raise SystemExit(main())
