"""tests/test_evalset.py — 36-prep-evalset(표준 GSM8K 평가 세트) 단위 테스트.

pytest가 없는 환경에서도 단독 실행된다:
    .venv/bin/python tests/test_evalset.py

검증 대상:
  - gold 추출(데이터셋 정답의 `#### N`)
  - 프롬프트 포맷(Raon chat template + `#### <number>` 지시)
  - 결정론적 샘플링(같은 seed·n → 같은 문제)
  - 매니페스트 신선도(freshness) 판정 — 멱등성
"""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tests._compat import run_standalone


def _load_script(name: str):
    mod_name = name.replace("-", "_").removesuffix(".py")
    spec = importlib.util.spec_from_file_location(mod_name, ROOT / name)
    assert spec is not None and spec.loader is not None, f"{name} 로드 실패"
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


ev = _load_script("s7-36-prep-evalset.py")


def test_gold_extraction():
    assert ev._gold_of("bla bla\n#### 18") == "18"
    assert ev._gold_of("#### 1,234") == "1234"
    assert ev._gold_of("#### 3.5") == "3.5"
    assert ev._gold_of("no answer here") is None
    assert ev._gold_of("") is None


def test_prompt_format_uses_raon_chat_template():
    p = ev._format_prompt("  What is 2+2?  ")
    assert p.startswith("user\n")
    assert "What is 2+2?" in p
    assert p.rstrip().endswith("assistant\n<think></think>")
    assert "#### <number>" in p


def test_sampling_is_deterministic():
    rows = [{"idx": i, "question": f"q{i}", "gold": str(i)} for i in range(100)]
    a = ev.sample_rows(rows, 10, 42)
    b = ev.sample_rows(rows, 10, 42)
    c = ev.sample_rows(rows, 10, 43)
    assert [r["idx"] for r in a] == [r["idx"] for r in b]
    assert [r["idx"] for r in a] != [r["idx"] for r in c]
    assert len(a) == 10
    assert len({r["idx"] for r in a}) == 10  # 중복 없음


def test_sampling_full_pool_is_shuffled():
    """전체(n=0)도 셔플된 순서 — '앞의 N개'가 연속 블록이 되지 않게."""
    rows = [{"idx": i, "question": "q", "gold": "1"} for i in range(50)]
    full = ev.sample_rows(rows, 0, 1)
    assert len(full) == 50
    assert [r["idx"] for r in full] != list(range(50)), "전체도 셔플되어야 한다"
    assert sorted(r["idx"] for r in full) == list(range(50))
    # n=0과 n=50은 같은 순서, n=10은 그 앞부분
    assert [r["idx"] for r in ev.sample_rows(rows, 50, 1)] == [r["idx"] for r in full]
    assert [r["idx"] for r in ev.sample_rows(rows, 10, 1)] == [r["idx"] for r in full[:10]]
    assert len(ev.sample_rows(rows, 99, 1)) == 50


def test_build_writes_prompts_and_manifest(monkeypatch):
    rows = [
        {"idx": 0, "question": "Janet has 3 apples", "gold": "18"},
        {"idx": 1, "question": "A robe takes 2 bolts", "gold": "3"},
    ]
    monkeypatch.setattr(ev, "load_test_rows", lambda: rows)
    with tempfile.TemporaryDirectory() as d:
        monkeypatch.setattr(ev, "SHARED", Path(d))
        m = ev.build("smoke", 2, 7)
        assert m["written_n"] == 2
        assert m["pool_size"] == 2
        assert m["seed"] == 7
        out = ev.prompts_path("smoke")
        recs = [json.loads(x) for x in out.read_text(encoding="utf-8").splitlines() if x.strip()]
        assert len(recs) == 2
        assert recs[0]["bench"] == "gsm8k"
        assert recs[0]["gold"] == "18"
        assert recs[0]["dataset_idx"] == 0
        assert "assistant" in recs[0]["prompt"]


def test_freshness_is_idempotent(monkeypatch):
    rows = [{"idx": i, "question": f"q{i}", "gold": str(i)} for i in range(10)]
    monkeypatch.setattr(ev, "load_test_rows", lambda: rows)
    with tempfile.TemporaryDirectory() as d:
        monkeypatch.setattr(ev, "SHARED", Path(d))
        assert ev.is_fresh("smoke", 5, 7) is False  # 아직 없음
        ev.build("smoke", 5, 7)
        assert ev.is_fresh("smoke", 5, 7) is True  # 같은 파라미터 → fresh
        assert ev.is_fresh("smoke", 6, 7) is False  # n 변경 → stale
        assert ev.is_fresh("smoke", 5, 8) is False  # seed 변경 → stale


if __name__ == "__main__":
    raise SystemExit(run_standalone(dict(globals())))
