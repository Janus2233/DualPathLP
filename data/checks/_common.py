
from __future__ import annotations

import sys

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass


class CheckFailure(AssertionError):
    pass


class Checker:

    def __init__(self, title: str) -> None:
        self.title = title
        self.records: list[tuple[str, bool, str]] = []

    def check(self, name: str, condition: bool, detail: str = "") -> bool:
        self.records.append((name, bool(condition), detail))
        return bool(condition)

    def equal(self, name: str, actual, expected, detail: str = "") -> bool:
        ok = actual == expected
        msg = f"actual={actual!r} expected={expected!r}"
        if detail:
            msg = f"{msg}  ({detail})"
        return self.check(name, ok, msg)

    def close(self, name: str, actual: float, expected: float, tol: float, detail: str = "") -> bool:
        ok = abs(actual - expected) <= tol
        msg = f"actual={actual:.6g} expected={expected:.6g} tol={tol:g}"
        if detail:
            msg = f"{msg}  ({detail})"
        return self.check(name, ok, msg)

    @property
    def failed(self) -> list[tuple[str, str]]:
        return [(n, d) for n, ok, d in self.records if not ok]

    def report(self) -> bool:
        print(f"\n=== {self.title} ===")
        n_pass = sum(1 for _, ok, _ in self.records if ok)
        for name, ok, detail in self.records:
            mark = "PASS" if ok else "FAIL"
            line = f"  [{mark}] {name}"
            if detail and not ok:
                line += f"  ->  {detail}"
            print(line)
        print(f"  --- {n_pass}/{len(self.records)} 通过 ---")
        if self.failed:
            print(f"  {len(self.failed)} 项失败：{[n for n, _ in self.failed]}")
            return False
        return True


def raise_if_failed(checker: Checker) -> None:
    if checker.failed:
        raise CheckFailure(
            f"{checker.title} 有 {len(checker.failed)} 项未通过：{[n for n, _ in checker.failed]}"
        )
