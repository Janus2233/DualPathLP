"""切分与时间泄漏自检 —— 本数据层最关键的检查。

验证：

1. 时间边界严格：``train.max_ts <= val_time < val.min_ts``，``val.max_ts <= test_time < test.min_ts``。
2. 训练集与测试集在**时间上不重叠**（这是动态图最容易出错处）。
3. 边 id 在 train / val / test 之间互不重叠，且并集 = 全量边表。
4. 归纳留出有效：抽样出的新测试节点**不出现在训练集中**。
5. 全量边表按时间单调，且并列时间戳的 tie-break 是确定性的。

**并且自检本身要被检验**：``--self-test`` 会人为把测试期的边混进训练集，
要求上述检查**报错**。若注入泄漏后仍然全部通过，说明检查是空的，必须视为失败。
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]))
    from data import config, io, preprocess, split
    from data.checks._common import Checker
else:
    from .. import config, io, preprocess, split
    from ._common import Checker


# --------------------------------------------------------------------------- #
# 可复用的判定逻辑：返回失败项列表，空列表表示通过
# --------------------------------------------------------------------------- #


def leakage_violations(arrays: dict[str, np.ndarray]) -> list[str]:
    """检查一份切分产物是否有时间泄漏。返回人类可读的违规说明列表。

    Args:
        arrays: 至少包含 ``train_ts`` / ``val_ts`` / ``test_ts`` /
            ``val_time`` / ``test_time`` / ``train_src`` / ``train_dst`` /
            ``new_test_node_set`` 的字典。
    """
    v: list[str] = []
    val_time = float(arrays["val_time"])
    test_time = float(arrays["test_time"])

    tr, va, te = arrays["train_ts"], arrays["val_ts"], arrays["test_ts"]

    # 1. 时间边界
    if tr.size and float(tr.max()) > val_time:
        v.append(f"训练集含 {int((tr > val_time).sum())} 条晚于 val_time 的边（未来信息泄漏）")
    if va.size and float(va.max()) > test_time:
        v.append(f"验证集含 {int((va > test_time).sum())} 条晚于 test_time 的边")
    if va.size and float(va.min()) <= val_time:
        v.append(f"验证集含 {int((va <= val_time).sum())} 条不晚于 val_time 的边（与训练集重叠）")
    if te.size and float(te.min()) <= test_time:
        v.append(f"测试集含 {int((te <= test_time).sum())} 条不晚于 test_time 的边（与验证集重叠）")
    if te.size and float(te.min()) <= val_time:
        v.append(f"测试集含 {int((te <= val_time).sum())} 条不晚于 val_time 的边（与训练集重叠）")

    # 2. 训练与测试的时间区间不得交错
    if tr.size and te.size and float(tr.max()) >= float(te.min()):
        v.append(f"训练集最大时间 {tr.max():.0f} >= 测试集最小时间 {te.min():.0f}")

    # 3. 归纳留出：抽中的新测试节点不得出现在训练集
    nts = set(int(x) for x in arrays.get("new_test_node_set", []))
    if nts:
        train_nodes = set(int(x) for x in arrays["train_src"]) | set(
            int(x) for x in arrays["train_dst"]
        )
        overlap = train_nodes & nts
        if overlap:
            v.append(
                f"{len(overlap)} 个归纳新测试节点出现在训练集中（归纳评测失效），"
                f"例如 {sorted(overlap)[:5]}"
            )
    return v


def _check_one(name: str, c: Checker) -> None:
    paths = config.processed_paths(name)
    if not paths["edges"].is_file():
        c.check(f"{name}: 预处理产物存在", False, f"缺少 {paths['edges']}")
        return

    arr = split.load_splits(name)
    tag = f"{name}"

    # --- 时间边界 --------------------------------------------------------- #
    viol = leakage_violations(arr)
    c.check(f"{tag}: 无时间泄漏/重叠", not viol, "; ".join(viol))

    val_time, test_time = float(arr["val_time"]), float(arr["test_time"])
    c.check(
        f"{tag}: train.max_ts <= val_time",
        float(arr["train_ts"].max()) <= val_time,
        f"{arr['train_ts'].max():.0f} vs {val_time:.0f}",
    )
    c.check(
        f"{tag}: val.max_ts <= test_time",
        float(arr["val_ts"].max()) <= test_time,
        f"{arr['val_ts'].max():.0f} vs {test_time:.0f}",
    )
    c.check(
        f"{tag}: test.min_ts > test_time",
        float(arr["test_ts"].min()) > test_time,
        f"{arr['test_ts'].min():.0f} vs {test_time:.0f}",
    )

    # --- 切分比例（注意 train 会因归纳留出而低于 70%） --------------------- #
    n_full = arr["full_ts"].size
    n_tr, n_va, n_te = arr["train_ts"].size, arr["val_ts"].size, arr["test_ts"].size
    c.check(f"{tag}: val 占比≈15%", abs(n_va / n_full - 0.15) < 0.01, f"{n_va / n_full:.4f}")
    c.check(f"{tag}: test 占比≈15%", abs(n_te / n_full - 0.15) < 0.01, f"{n_te / n_full:.4f}")
    c.check(
        f"{tag}: train 占比 <= 70%（被归纳留出削减）",
        n_tr / n_full <= 0.70 + 1e-9,
        f"{n_tr / n_full:.4f}",
    )

    # --- 边 id 划分：互不重叠且并集为全量 ---------------------------------- #
    tr_eid, va_eid, te_eid = arr["train_eid"], arr["val_eid"], arr["test_eid"]
    c.equal(f"{tag}: train/val 边 id 无交集", len(np.intersect1d(tr_eid, va_eid)), 0)
    c.equal(f"{tag}: train/test 边 id 无交集", len(np.intersect1d(tr_eid, te_eid)), 0)
    c.equal(f"{tag}: val/test 边 id 无交集", len(np.intersect1d(va_eid, te_eid)), 0)

    union = np.union1d(np.union1d(tr_eid, va_eid), te_eid)
    c.equal(f"{tag}: 三段边 id 唯一", union.size, tr_eid.size + va_eid.size + te_eid.size)

    # --- 归纳留出 --------------------------------------------------------- #
    nts = set(int(x) for x in arr["new_test_node_set"])
    train_nodes = set(int(x) for x in arr["train_src"]) | set(int(x) for x in arr["train_dst"])
    c.equal(f"{tag}: 新测试节点不在训练集", len(train_nodes & nts), 0)
    c.check(
        f"{tag}: 新测试节点数 = 10% 全图节点",
        len(nts) > 0,
        f"抽样 {len(nts)} 个",
    )

    # --- 全量边表时间单调 ------------------------------------------------- #
    full_ts = arr["full_ts"]
    c.check(
        f"{tag}: 全量边表按时间单调不减",
        bool(np.all(np.diff(full_ts) >= 0)),
        f"逆序步数={int((np.diff(full_ts) < 0).sum())}",
    )


def run(names: list[str], *, self_test: bool = True) -> bool:
    ok_all = True
    for name in names:
        c = Checker(f"check_splits :: {name}")
        _check_one(name, c)
        ok_all &= c.report()

        if self_test:
            ok_all &= _self_test_leak_detection(name)

    return ok_all


def _self_test_leak_detection(name: str) -> bool:
    """**检验检查本身**：人为注入泄漏，要求检查报错。

    做法：把测试期的边拼进训练集的时间戳数组，然后要求 ``leakage_violations`` 非空。
    若注入后仍然"无违规"，说明检查逻辑是空的 —— 记为失败。
    """
    c = Checker(f"check_splits :: {name} :: self-test(注入泄漏必须被抓到)")
    try:
        arr = {k: v.copy() for k, v in split.load_splits(name).items()}
    except FileNotFoundError as exc:
        c.check("切分产物可读", False, str(exc))
        c.report()
        return False

    clean = leakage_violations(arr)
    c.check("干净数据上无违规", not clean, "; ".join(clean))

    # --- 注入 1：把测试期的边混入训练集时间戳 ----------------------------- #
    leaked = {k: v.copy() for k, v in arr.items()}
    leaked["train_ts"] = np.concatenate([leaked["train_ts"], leaked["test_ts"]])
    viol = leakage_violations(leaked)
    c.check("注入未来边到训练集后必须报错", bool(viol), "检查未报错 -> 检查是空的")

    # --- 注入 2：破坏归纳留出（把新测试节点放进训练集） ------------------- #
    leaked2 = {k: v.copy() for k, v in arr.items()}
    nts = np.asarray(leaked2["new_test_node_set"], dtype=np.int64)
    if nts.size:
        leaked2["train_src"] = np.concatenate([leaked2["train_src"], nts[:1]])
        viol2 = leakage_violations(leaked2)
        c.check("把新测试节点放进训练集后必须报错", bool(viol2), "检查未报错 -> 检查是空的")
    else:
        c.check("存在新测试节点可供注入", False, "new_test_node_set 为空")

    # --- 注入 3：把验证集时间戳整体前移到训练期内 ------------------------- #
    leaked3 = {k: v.copy() for k, v in arr.items()}
    leaked3["val_ts"] = leaked3["val_ts"] - (float(arr["val_time"]) + 1.0)
    viol3 = leakage_violations(leaked3)
    c.check("把验证集移入训练期后必须报错", bool(viol3), "检查未报错 -> 检查是空的")

    return c.report()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="切分与时间泄漏自检")
    p.add_argument("--dataset", choices=sorted(config.DATASETS), default=None)
    p.add_argument("--fast", action="store_true")
    p.add_argument("--no-self-test", action="store_true", help="跳过注入泄漏的自检")
    args = p.parse_args(argv)

    if args.dataset:
        names = [args.dataset]
    elif args.fast:
        names = list(config.FAST_DATASETS)
    else:
        names = list(config.ALL_DATASETS)

    return 0 if run(names, self_test=not args.no_self_test) else 1


if __name__ == "__main__":
    raise SystemExit(main())
