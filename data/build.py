"""数据层命令行入口。

用法（在 ``Model experiments/`` 下执行，使用指定 conda 环境解释器）::

    $py = "D:\\anaconda3\\envs\\py313_torch290_cu128\\python.exe"
    & $py -m data.build --dataset college_msg
    & $py -m data.build --all
    & $py -m data.build --fast
    & $py -m data.build --dataset sx_mathoverflow --stage preprocess

产出写入 ``runs/data/<name>/``（``runs/`` 不入库）。
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

from . import config, io, preprocess, split


def _env_info() -> dict[str, object]:
    """记录运行环境，便于结果回溯。"""
    import numpy as np
    import pandas as pd
    import sklearn

    return {
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "sklearn": sklearn.__version__,
    }


def build_one(
    name: str,
    *,
    stage: str = "all",
    verify: bool = True,
    quiet: bool = False,
) -> dict[str, object]:
    """跑单个数据集的预处理（+ 切分）。返回汇总信息。"""
    spec = config.DATASETS[name]
    t0 = time.perf_counter()

    if not quiet:
        print(f"\n=== {name}  ({spec.filename}) ===")

    raw = io.read_raw(spec, verify=verify)
    raw_desc = io.describe(raw)
    if not quiet:
        print(
            f"  原始：{raw_desc['n_edges']:,} 边 / {raw_desc['n_unique_nodes']:,} 节点"
            f" / 自环 {raw_desc['n_self_loops']:,}"
            f" / 文件已排序={raw_desc['file_ts_sorted']}"
            f" / 并列 ts 行 {raw_desc['ts_rows_in_ties_pct']}%"
        )

    pe = preprocess.preprocess(raw)
    paths = preprocess.write_processed(pe)
    if not quiet:
        s = pe.stats
        print(
            f"  预处理：丢自环 {s['n_self_loops_dropped']:,}"
            f" → {s['n_edges']:,} 边 / {s['n_nodes']:,} 节点"
            f" / ts 单调={s['ts_is_monotonic']}"
        )

    result: dict[str, object] = {
        "dataset": name,
        "raw": raw_desc,
        "preprocess": pe.stats,
        "outputs": {k: str(v) for k, v in paths.items()},
    }

    if stage in ("all", "split"):
        sp = split.split(pe)
        sp_path = split.write_splits(sp)
        result["split"] = sp.stats
        result["outputs"]["splits"] = str(sp_path)
        if not quiet:
            st = sp.stats
            print(
                f"  切分：train {st['n_train']:,} ({st['train_pct']}%)"
                f" / val {st['n_val']:,} ({st['val_pct']}%)"
                f" / test {st['n_test']:,} ({st['test_pct']}%)"
            )
            print(
                f"        归纳留出：新测试节点 {st['n_new_test_nodes_sampled']:,}"
                f" / 训练集不含的新节点 {st['n_new_nodes']:,}"
            )
            print(
                f"        时间边界：train_max={st['train_max_ts']:.0f}"
                f" <= val_time={st['val_time']:.0f}"
                f" < test_min={st['test_min_ts']:.0f}"
            )
            if not (st["train_le_val"] and st["val_lt_test"] and st["test_gt_val"]):
                print("        !! 时间边界检查未通过")

    result["elapsed_sec"] = round(time.perf_counter() - t0, 2)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m data.build",
        description="预处理 + 切分动态图链接预测数据集（协议对齐 TGB/DyGLib）",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dataset", choices=sorted(config.DATASETS), help="单个数据集")
    group.add_argument("--all", action="store_true", help="全部四个数据集")
    group.add_argument("--fast", action="store_true", help="快速档（college_msg + email_eu_core）")
    group.add_argument("--main", action="store_true", help="主实验档（sx_mathoverflow + reddit）")

    parser.add_argument(
        "--stage",
        choices=("preprocess", "split", "all"),
        default="all",
        help="只做预处理 / 只做切分（要求预处理已存在时用） / 全做（默认）",
    )
    parser.add_argument("--no-verify", action="store_true", help="跳过 SHA256 校验（不建议）")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--report", type=Path, default=None, help="把汇总 JSON 写到指定路径")
    args = parser.parse_args(argv)

    if args.dataset:
        names = [args.dataset]
    elif args.fast:
        names = list(config.FAST_DATASETS)
    elif args.main:
        names = list(config.MAIN_DATASETS)
    else:
        names = list(config.ALL_DATASETS)

    env = _env_info()
    if not args.quiet:
        print("环境：", json.dumps(env, ensure_ascii=False))

    report: dict[str, object] = {"env": env, "datasets": {}}
    failures: list[str] = []

    for name in names:
        try:
            report["datasets"][name] = build_one(
                name, stage=args.stage, verify=not args.no_verify, quiet=args.quiet
            )
        except Exception as exc:  # noqa: BLE001 - 需要在一个报告里汇总所有失败
            failures.append(name)
            report["datasets"][name] = {"dataset": name, "error": f"{type(exc).__name__}: {exc}"}
            print(f"!! {name} 失败：{type(exc).__name__}: {exc}", file=sys.stderr)

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        with open(args.report, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        if not args.quiet:
            print(f"\n报告已写入 {args.report}")

    if failures:
        print(f"\n失败的数据集：{failures}", file=sys.stderr)
        return 1
    if not args.quiet:
        print(f"\n完成：{len(names)} 个数据集。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
