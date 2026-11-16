
from __future__ import annotations

import argparse
import sys
import time

if __package__ in (None, ""):
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]))
    from data import config
    from data.checks import (
        check_integrity,
        check_metrics,
        check_negatives,
        check_preprocess,
        check_splits,
    )
else:
    from .. import config
    from . import (
        check_integrity,
        check_metrics,
        check_negatives,
        check_preprocess,
        check_splits,
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="数据层全部自检")
    p.add_argument("--fast", action="store_true", help="只跑小数据集")
    p.add_argument("--dataset", choices=sorted(config.DATASETS), default=None)
    p.add_argument("--skip-integrity", action="store_true", help="跳过 SHA256 校验（省时间）")
    p.add_argument("--no-self-test", action="store_true", help="跳过注入泄漏的自检")
    args = p.parse_args(argv)

    if args.dataset:
        names = [args.dataset]
    elif args.fast:
        names = list(config.FAST_DATASETS)
    else:
        names = list(config.ALL_DATASETS)

    t0 = time.perf_counter()
    print(f"数据层自检：{names}")

    results: dict[str, bool] = {}

    if not args.skip_integrity:
        results["check_integrity"] = check_integrity.run(names)

    results["check_preprocess"] = check_preprocess.run(names)
    results["check_splits"] = check_splits.run(names, self_test=not args.no_self_test)
    results["check_negatives"] = check_negatives.run(names)
    results["check_metrics"] = check_metrics.run()

    elapsed = time.perf_counter() - t0
    print("\n" + "=" * 62)
    print(f"汇总（{elapsed:.1f}s）")
    for name, ok in results.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    failed = [k for k, v in results.items() if not v]
    print("=" * 62)

    if failed:
        print(f"自检未通过：{failed}")
        return 1
    print("全部自检通过。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
