"""tests/test_sdlc.py — 9단계 SDLC 공통 모듈(sdlc.py · s1-07-acceptance.py · gates.inc) 단위 테스트.

pytest가 없는 환경에서도 단독 실행된다:
    .venv/bin/python tests/test_sdlc.py

검증 대상:
  - config/acceptance.toml 스키마 검증 + 기본값 폴백
  - Δ 판정(verdict_for_delta)
  - 단계 원장(ledger) 기록/병합/타깃 집계
  - s1-07-acceptance.py 산출물(acceptance-<mode>.json)
  - gates.inc 게이트(acceptance/ppl/eval)의 통과·차단 동작
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import sdlc
from tests._compat import run_standalone


def _load_script(name: str):
    mod_name = name.replace("-", "_").removesuffix(".py")
    spec = importlib.util.spec_from_file_location(mod_name, ROOT / name)
    assert spec is not None and spec.loader is not None, f"{name} 로드 실패"
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


acc = _load_script("s1-07-acceptance.py")


# ---------------------------------------------------------------- 수용 기준


def test_acceptance_toml_is_valid():
    data = sdlc.load_acceptance()
    assert sdlc.acceptance_errors(data) == [], sdlc.acceptance_errors(data)
    for t in ("awq-w3", "awq-w5", "nvfp4-awq", "nvfp4-gptq"):
        spec = data["targets"][t]
        assert spec["quant_format"]
        assert len(spec["size_gb"]) == 2
        assert spec["ppl_delta_warn"] < spec["ppl_delta_max"]


def test_acceptance_defaults_when_file_missing():
    data = sdlc.load_acceptance("/nonexistent/acceptance.toml")
    assert data["source"] == "default"
    assert sdlc.acceptance_errors(data) == []
    assert data["targets"]["nvfp4-awq"]["size_gb"] == [15.0, 28.0]


def test_acceptance_errors_detect_bad_schema():
    bad = {
        "version": "x",
        "targets": {
            "awq-w3": {"quant_format": 1, "size_gb": [30.0, 10.0], "ppl_delta_warn": 1.0, "ppl_delta_max": 0.3},
        },
        "gates": {},
    }
    errs = sdlc.acceptance_errors(bad)
    assert any("version" in e for e in errs)
    assert any("size_gb" in e for e in errs)
    assert any("quant_format" in e for e in errs)
    assert any("ppl_delta_warn" in e for e in errs)


def test_verdict_for_delta_thresholds():
    spec = {"ppl_delta_warn": 0.3, "ppl_delta_max": 1.0}
    assert sdlc.verdict_for_delta(0.0, spec) == "Accept"
    assert sdlc.verdict_for_delta(0.29, spec) == "Accept"
    assert sdlc.verdict_for_delta(0.31, spec) == "Conditional"
    assert sdlc.verdict_for_delta(0.99, spec) == "Conditional"
    assert sdlc.verdict_for_delta(1.0, spec) == "Discard"
    assert sdlc.verdict_for_delta(5.0, spec) == "Discard"


# ---------------------------------------------------------------- 원장


def test_ledger_roundtrip():
    with tempfile.TemporaryDirectory() as d:
        sdlc.update_ledger("smoke", 5, "quant", "pass", artifacts=["models/x"], shared=d)
        data = sdlc.read_ledger("smoke", shared=d)
        assert data["stages"]["5"]["status"] == "pass"
        assert data["stages"]["5"]["artifacts"] == ["models/x"]
        assert data["stages"]["5"]["id"] == 5


def test_ledger_target_aggregate_marks_failure():
    with tempfile.TemporaryDirectory() as d:
        sdlc.update_ledger("smoke", 5, "quant", "pass", target="awq-w3", shared=d)
        sdlc.update_ledger("smoke", 5, "quant", "pass", target="nvfp4-awq", shared=d)
        data = sdlc.read_ledger("smoke", shared=d)
        assert data["stages"]["5"]["status"] == "pass"
        assert set(data["stages"]["5"]["targets"]) == {"awq-w3", "nvfp4-awq"}
        sdlc.update_ledger("smoke", 5, "quant", "fail", target="nvfp4-gptq", shared=d)
        data = sdlc.read_ledger("smoke", shared=d)
        assert data["stages"]["5"]["status"] == "fail"


def test_ledger_missing_file_is_empty():
    with tempfile.TemporaryDirectory() as d:
        data = sdlc.read_ledger("full", shared=d)
        assert data["stages"] == {}


# ---------------------------------------------------------------- s1-07-acceptance


def test_07_acceptance_writes_snapshot():
    with tempfile.TemporaryDirectory() as d:
        env = dict(os.environ, RQ_SHARED=d)
        out = subprocess.run(
            [sys.executable, str(ROOT / "s1-07-acceptance.py"), "--mode", "smoke"],
            capture_output=True, text=True, env=env, cwd=str(ROOT), check=False,
        )
        assert out.returncode == 0, out.stderr
        p = Path(d) / "outputs" / "acceptance-smoke.json"
        assert p.is_file()
        payload = json.loads(p.read_text(encoding="utf-8"))
        assert payload["mode"] == "smoke"
        assert len(payload["stages"]) == 9
        assert "nvfp4-awq" in payload["targets"]


def test_07_acceptance_rejects_bad_file():
    with tempfile.TemporaryDirectory() as d:
        bad = Path(d) / "bad.toml"
        bad.write_text("[targets.awq-w3]\nquant_format = 1\n", encoding="utf-8")
        out = subprocess.run(
            [sys.executable, str(ROOT / "s1-07-acceptance.py"), "--mode", "smoke", "--path", str(bad)],
            capture_output=True, text=True, cwd=str(ROOT), check=False,
        )
        assert out.returncode == 1
        assert "acceptance 스키마 오류" in out.stderr


# ---------------------------------------------------------------- gates.inc


def _bash(script: str, env_extra: dict | None = None) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env.setdefault("RQ_SHARED", "/tmp/rq-test-shared")
    if env_extra:
        env.update(env_extra)
    return subprocess.run(
        ["bash", "-c", script], capture_output=True, text=True, env=env, cwd=str(ROOT), check=False
    )


def test_gate_ppl_blocks_nan_and_missing():
    with tempfile.TemporaryDirectory() as d:
        bad = Path(d) / "ppl.json"
        bad.write_text(json.dumps({"ppl": float("nan")}), encoding="utf-8")
        out = _bash(f'source gates.inc; rq_gate_ppl "{bad}" nvfp4-awq')
        assert out.returncode == 1, out.stdout
        out = _bash(f'source gates.inc; rq_gate_ppl "{Path(d) / "none.json"}" nvfp4-awq')
        assert out.returncode == 1


def test_gate_ppl_accepts_finite_with_matching_verdict():
    with tempfile.TemporaryDirectory() as d:
        good = Path(d) / "nvfp4-awq.json"
        good.write_text(json.dumps({"target": "nvfp4-awq", "ppl": 7.7, "n": 8}), encoding="utf-8")
        (Path(d) / "summary.json").write_text(
            json.dumps({"nvfp4-awq": {"delta": 0.05, "verdict": "Accept"}}), encoding="utf-8"
        )
        out = _bash(f'source gates.inc; rq_gate_ppl "{good}" nvfp4-awq')
        assert out.returncode == 0, out.stdout + out.stderr


def test_gate_ppl_rejects_verdict_mismatch():
    with tempfile.TemporaryDirectory() as d:
        bad = Path(d) / "nvfp4-awq.json"
        bad.write_text(json.dumps({"target": "nvfp4-awq", "ppl": 9.9, "n": 8}), encoding="utf-8")
        (Path(d) / "summary.json").write_text(
            json.dumps({"nvfp4-awq": {"delta": 2.0, "verdict": "Accept"}}), encoding="utf-8"
        )
        out = _bash(f'source gates.inc; rq_gate_ppl "{bad}" nvfp4-awq')
        assert out.returncode == 1


def test_gate_eval_blocks_empty():
    with tempfile.TemporaryDirectory() as d:
        empty = Path(d) / "e.jsonl"
        empty.write_text("", encoding="utf-8")
        out = _bash(f'source gates.inc; rq_gate_eval "{empty}"')
        assert out.returncode == 1
        full = Path(d) / "f.jsonl"
        full.write_text('{"output": "x"}\n', encoding="utf-8")
        out = _bash(f'source gates.inc; rq_gate_eval "{full}"')
        assert out.returncode == 0


def test_gate_acceptance_requires_snapshot():
    with tempfile.TemporaryDirectory() as d:
        out = _bash(f'source gates.inc; RQ_OUTPUTS="{d}"; rq_gate_acceptance smoke')
        assert out.returncode == 1
        (Path(d) / "acceptance-smoke.json").write_text(
            json.dumps({"version": 1, "targets": {t: {"quant_format": "x", "ppl_delta_max": 1.0}
                                                 for t in ("awq-w3", "awq-w5", "nvfp4-awq", "nvfp4-gptq")}}),
            encoding="utf-8",
        )
        out = _bash(f'source gates.inc; RQ_OUTPUTS="{d}"; rq_gate_acceptance smoke')
        assert out.returncode == 0, out.stdout + out.stderr


def test_rq_stage_of_mapping_covers_nine_stages():
    out = _bash(
        "source config/common.env smoke; "
        "for s in 10-setup 07-acceptance 13-calib 16-compat 19-dequant 28-nvfp4-awq "
        "34-ppl 37-eval 43-prep; do rq_stage_of $s; done"
    )
    assert out.returncode == 0, out.stderr
    ids = [line.split()[0] for line in out.stdout.strip().splitlines() if line.strip()]
    assert ids == ["0", "1", "2", "3", "4", "5", "6", "7", "8"]


if __name__ == "__main__":
    raise SystemExit(run_standalone(dict(globals())))
