"""tests/_compat.py — pytest 미설치 환경(`.venv` 기본 상태) 호환 레이어.

`s1-01-setup-environment.sh`의 `uv sync --frozen` 실행 후에는 `.venv`에 `pytest`가 없으므로,
`pytest`가 설치되어 있으면 그대로 사용하고 없으면 동일한 인터페이스의 경량 shim
(`fixture`, `raises`, `approx`, `skip`, `monkeypatch`, `capsys`)으로 모든 테스트를 실행한다.
"""

from __future__ import annotations

import contextlib
import inspect
import io
import os
import re
import sys
from types import SimpleNamespace
from typing import Any


class _Skipped(BaseException):
    """pytest.skip 대체 예외."""


class _Approx:
    def __init__(self, expected: float, rel: float = 1e-6, abs_tol: float = 1e-12) -> None:
        self.expected = float(expected)
        self.rel = rel
        self.abs_tol = abs_tol

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, (int, float)):
            return False
        tol = max(self.rel * abs(self.expected), self.abs_tol)
        return abs(float(other) - self.expected) <= tol

    def __repr__(self) -> str:
        return f"approx({self.expected!r})"


class _MonkeyPatch:
    _MISSING = object()

    def __init__(self) -> None:
        self._attrs: list[tuple[Any, str, Any]] = []
        self._env: list[tuple[str, str | None]] = []

    def setattr(self, target: Any, name: str, value: Any, raising: bool = True) -> None:
        old = getattr(target, name, self._MISSING)
        if old is self._MISSING and raising:
            raise AttributeError(f"{target!r} has no attribute {name!r}")
        self._attrs.append((target, name, old))
        setattr(target, name, value)

    def setenv(self, name: str, value: str) -> None:
        self._env.append((name, os.environ.get(name)))
        os.environ[name] = str(value)

    def delenv(self, name: str, raising: bool = True) -> None:
        if name not in os.environ and raising:
            raise KeyError(name)
        self._env.append((name, os.environ.get(name)))
        os.environ.pop(name, None)

    def undo(self) -> None:
        for target, name, old in reversed(self._attrs):
            if old is self._MISSING:
                delattr(target, name)
            else:
                setattr(target, name, old)
        self._attrs.clear()
        for name, old_val in reversed(self._env):
            if old_val is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = old_val
        self._env.clear()


class _CapSys:
    def __init__(self) -> None:
        self._out = io.StringIO()
        self._err = io.StringIO()
        self._orig_out = sys.stdout
        self._orig_err = sys.stderr
        sys.stdout = self._out
        sys.stderr = self._err

    def readouterr(self) -> SimpleNamespace:
        out = self._out.getvalue()
        err = self._err.getvalue()
        self._out.seek(0)
        self._out.truncate(0)
        self._err.seek(0)
        self._err.truncate(0)
        return SimpleNamespace(out=out, err=err)

    def close(self) -> None:
        sys.stdout = self._orig_out
        sys.stderr = self._orig_err


class _PytestShim:
    @staticmethod
    def fixture(fn=None, **_kwargs):
        def _mark(f):
            f._is_fixture = True
            return f

        return _mark(fn) if fn is not None else _mark

    @staticmethod
    @contextlib.contextmanager
    def raises(expected_exc: type[BaseException] | tuple[type[BaseException], ...], match: str | None = None):
        try:
            yield
        except expected_exc as exc:
            if match is not None and not re.search(match, str(exc)):
                raise AssertionError(f"Regex {match!r} did not match {str(exc)!r}") from exc
            return
        raise AssertionError(f"DID NOT RAISE {expected_exc}")

    @staticmethod
    def approx(expected: float, rel: float = 1e-6, abs: float = 1e-12) -> _Approx:
        return _Approx(expected, rel=rel, abs_tol=abs)

    @staticmethod
    def skip(reason: str = "") -> None:
        raise _Skipped(reason)


try:
    import pytest as _real_pytest  # type: ignore[import-not-found]

    pytest: Any = _real_pytest
except ImportError:
    pytest = _PytestShim()


def _run_one_test(name: str, fn: Any, ns: dict[str, Any]) -> tuple[str, BaseException | None]:
    mp = _MonkeyPatch()
    cap: _CapSys | None = None
    teardowns: list[Any] = []
    cache: dict[str, Any] = {}

    def _resolve(param: str) -> Any:
        nonlocal cap
        if param in cache:
            return cache[param]
        if param == "monkeypatch":
            cache[param] = mp
            return mp
        if param == "capsys":
            if cap is None:
                cap = _CapSys()
            cache[param] = cap
            return cap
        fix_fn = ns.get(param)
        if callable(fix_fn):
            sub_args = [_resolve(p) for p in inspect.signature(fix_fn).parameters]
            res = fix_fn(*sub_args)
            if inspect.isgenerator(res):
                val = next(res)
                teardowns.append(res)
            else:
                val = res
            cache[param] = val
            return val
        raise RuntimeError(f"unknown fixture {param!r} for {name}")

    try:
        kwargs = {p: _resolve(p) for p in inspect.signature(fn).parameters}
        fn(**kwargs)
    except _Skipped as s:
        if cap is not None:
            cap.close()
            cap = None
        print(f"SKIP {name}: {s}")
        return ("SKIP", None)
    except Exception as e:  # noqa: BLE001
        if cap is not None:
            cap.close()
            cap = None
        print(f"FAIL {name}: {type(e).__name__}: {e}")
        return ("FAIL", e)
    else:
        if cap is not None:
            cap.close()
            cap = None
        print(f"PASS {name}")
        return ("PASS", None)
    finally:
        if cap is not None:
            cap.close()
        for gen in reversed(teardowns):
            with contextlib.suppress(StopIteration):
                next(gen)
        mp.undo()


def run_standalone(ns: dict[str, Any]) -> int:
    """단독 실행 러너 — 픽스처(`monkeypatch`, `capsys`, 모듈 내 `@pytest.fixture`)를 주입해 전 테스트를 실행."""
    failed: list[tuple[str, BaseException]] = []
    for name, fn in sorted(ns.items()):
        if not name.startswith("test_") or not callable(fn):
            continue
        status, err = _run_one_test(name, fn, ns)
        if status == "FAIL" and err is not None:
            failed.append((name, err))
    print(f"\n{len(failed)} failed")
    return 1 if failed else 0
