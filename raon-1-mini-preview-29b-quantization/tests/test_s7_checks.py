"""tests/test_s7_checks.py — s7 검사 확장(GSM8K 통계·한국어 KMMLU·형식지시 준수·서빙) 단위 테스트.

pytest가 없는 환경에서도 단독 실행된다:
    .venv/bin/python tests/test_s7_checks.py

합성 픽스처만 사용한다(모델 로드·GPU·네트워크 없음).
"""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

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


def _isolate(mod: Any, d: str) -> None:
    """모듈과 sdlc의 SHARED를 임시 디렉터리로 고정한다(실제 shared 산출물 오염 방지)."""
    mod.SHARED = Path(d)
    sdlc.SHARED = Path(d)


g80 = _load_script("s4-13-gsm8k.py")
ko = _load_script("s4-19-kmmluko.py")
fmt = _load_script("s4-22-format.py")
sb = _load_script("s4-25-servebench.py")
ev = _load_script("s4-07-eval.py")
prep = _load_script("s4-31-prep-upload.py")


# ---------------------------------------------------------------- GSM8K 통계


def test_gsm8k_stats_answer_and_truncation_rate():
    with tempfile.TemporaryDirectory() as d:
        _isolate(g80, d)
        outdir = Path(d) / "outputs" / "eval-smoke"
        outdir.mkdir(parents=True)
        rows = [
            {"bench": "gsm8k", "output": "reasoning #### 18", "gold": "18"},
            {"bench": "gsm8k", "output": "no marker here", "gold": "5"},
            {"bench": "gsm8k", "output": "", "gold": "7"},
        ]
        (outdir / "nvfp4-awq.jsonl").write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8"
        )
        st = g80.gsm8k_stats("smoke", "nvfp4-awq")
        assert st["n"] == 3
        assert st["answer_rate"] == 0.333
        assert st["truncation_rate"] == 0.667


def test_gsm8k_check_verdict_against_contract():
    rows = {
        "base": {"benchmarks": {"gsm8k": {"n": 200, "acc": 0.945}}},
        "nvfp4-awq": {"benchmarks": {"gsm8k": {"n": 200, "acc": 0.935}}},
    }
    payload = g80.gsm8k_check("smoke", rows)
    assert payload["verdict"]["base"] == "ref"
    assert payload["verdict"]["nvfp4-awq"] == "PASS"
    assert payload["delta_vs_base"]["nvfp4-awq"] == 0.01


# ---------------------------------------------------------------- 한국어 KMMLU


def test_kmmluko_parse_letter():
    assert ko.parse_letter("정답은 B") == "B"
    assert ko.parse_letter("정답: C") == "C"
    assert ko.parse_letter("answer is D") == "D"
    assert ko.parse_letter("A. 부화") == "A"
    assert ko.parse_letter("답을 모르겠어요") is None


def test_kmmluko_score_target():
    rows = [
        {"bench": "kmmluko", "output": "정답은 A", "gold": "A"},
        {"bench": "kmmluko", "output": "정답은 C", "gold": "B"},
        {"bench": "kmmluko", "output": "잘 모르겠어요", "gold": "B"},
    ]
    res = ko.score_target(rows)
    assert res["n"] == 3
    assert res["acc"] == 0.333
    assert res["answer_rate"] == 0.667


def test_kmmluko_main_writes_checks_section():
    with tempfile.TemporaryDirectory() as d:
        _isolate(ko, d)
        outdir = Path(d) / "outputs" / "eval-smoke"
        outdir.mkdir(parents=True)
        for t, acc in (("base", "A"), ("nvfp4-awq", "B")):
            (outdir / f"{t}.jsonl").write_text(
                json.dumps({"bench": "kmmluko", "output": f"정답은 {acc}", "gold": acc}, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
        (outdir / "score.json").write_text(json.dumps({
            "base": {"benchmarks": {"gsm8k": {"n": 10, "acc": 0.9}}},
            "nvfp4-awq": {"benchmarks": {"gsm8k": {"n": 10, "acc": 0.9}}},
        }), encoding="utf-8")
        payload = ko.main("smoke")
        assert payload["verdict"]["base"] == "ref"
        assert payload["verdict"]["nvfp4-awq"] in ("PASS", "FAIL")
        checks = json.loads((outdir / "checks.json").read_text(encoding="utf-8"))
        assert "kmmluko" in checks


# ---------------------------------------------------------------- 형식/지시 준수


def test_eval_row_preserves_prompt_metadata():
    """응답 행은 프롬프트 행의 case/check를 보존해야 한다(잃으면 가짜 통과)."""
    row = ev.build_row({"bench": "format", "case": "json_only", "check": "has_json",
                        "prompt": "P", "gold": None}, '{"a": 1}')
    assert row["case"] == "json_only"
    assert row["check"] == "has_json"
    assert row["bench"] == "format"
    assert row["prompt"] == "P"
    assert row["output"] == '{"a": 1}'


def test_format_score_joins_prompt_metadata():
    """메타데이터가 없는 응답 행도 프롬프트 세트와 조인해 실제로 검사된다."""
    with tempfile.TemporaryDirectory() as d:
        _isolate(fmt, d)
        fmt.prepare("smoke")
        prompts = Path(d) / "outputs" / "eval-smoke" / "prompts-format.jsonl"
        rows = [json.loads(x) for x in prompts.read_text(encoding="utf-8").splitlines() if x.strip()]
        outdir = Path(d) / "outputs" / "eval-smoke"
        # 응답 행에 case/check가 없다(구 산출물 형태) — 조인 없이는 전부 통과가 된다.
        stripped = [{"bench": "format", "prompt": r["prompt"], "output": "x"} for r in rows]
        (outdir / "nvfp4-awq.jsonl").write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in stripped) + "\n", encoding="utf-8"
        )
        payload = fmt.score("smoke")
        t = payload["targets"]["nvfp4-awq"]
        # 조인이 됐어야 검사 키가 채워진다(모두 "no check" 통과가 아니다)
        assert all(c["check"] for c in t["cases"]), "검사 키가 회수되지 않음"
        assert t["cases"][0]["reason"] != "no check"


def test_format_score_fails_when_check_unknown():
    """검사 키를 못 찾으면 통과가 아니라 실패(가짜 통과 방지)."""
    with tempfile.TemporaryDirectory() as d:
        _isolate(fmt, d)
        outdir = Path(d) / "outputs" / "eval-smoke"
        outdir.mkdir(parents=True)
        (outdir / "nvfp4-awq.jsonl").write_text(
            json.dumps({"bench": "format", "prompt": "없는 프롬프트", "output": "hello"},
                       ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        payload = fmt.score("smoke")
        t = payload["targets"]["nvfp4-awq"]
        assert t["pass_rate"] == 0.0
        assert "검사 키 없음" in t["cases"][0]["reason"]


def test_score_json_scores_kmmluko_and_drops_format():
    """score.json: kmmluko는 정확도로, format(통과율 벤치)은 제외."""
    with tempfile.TemporaryDirectory() as d:
        _isolate(g80, d)
        outdir = Path(d) / "outputs" / "eval-smoke"
        outdir.mkdir(parents=True)
        rows = [
            {"bench": "gsm8k", "output": "#### 18", "gold": "18"},
            {"bench": "kmmluko", "output": "정답은 B", "gold": "B"},
            {"bench": "format", "output": "whatever", "gold": None},
        ]
        (outdir / "base.jsonl").write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8"
        )
        res = g80.score("smoke")
        b = res["base"]["benchmarks"]
        assert b["gsm8k"]["acc"] == 1.0
        assert b["kmmluko"]["acc"] == 1.0
        assert "format" not in b


def test_format_check_outputs():
    assert fmt.check_output("no_hangul", "The weather is nice.")[0] is True
    assert fmt.check_output("no_hangul", "오늘은 날씨가 좋습니다.")[0] is False
    assert fmt.check_output("has_hangul", "오늘은 날씨가 좋습니다.")[0] is True
    assert fmt.check_output("has_json", '{"name": "홍길동"}')[0] is True
    assert fmt.check_output("has_json", "홍길동입니다")[0] is False
    assert fmt.check_output("has_marker", "2+2=4\n#### 4")[0] is True
    assert fmt.check_output("very_short", "안녕")[0] is True
    assert fmt.check_output("very_short", "안녕하세요 반갑습니다 오늘 날씨가 정말 좋은군요 산책하러 가기 좋은 날씨네요")[0] is False
    assert fmt.check_output("asks_back", "무엇을 알려드릴까요?")[0] is True
    assert fmt.check_output("asks_back", "네 알겠습니다")[0] is False
    assert fmt.check_output("has_two_lines", "첫째 줄\n둘째 줄")[0] is True


def test_format_prepare_and_score():
    with tempfile.TemporaryDirectory() as d:
        _isolate(fmt, d)
        n = fmt.prepare("smoke")
        assert n == len(fmt.CASES)
        prompts = Path(d) / "outputs" / "eval-smoke" / "prompts-format.jsonl"
        rows = [json.loads(x) for x in prompts.read_text(encoding="utf-8").splitlines() if x.strip()]
        assert len(rows) == len(fmt.CASES)
        assert rows[0]["bench"] == "format"
        # 점수: 절반만 통과하는 응답
        outdir = Path(d) / "outputs" / "eval-smoke"
        scored = []
        for i, r in enumerate(rows):
            ok, _ = fmt.check_output(r["check"], "정답은 A" if i % 2 == 0 else "")
            scored.append({"bench": "format", "case": r["case"], "check": r["check"],
                           "output": "정답은 A" if ok else "", "ok": ok})
        (outdir / "nvfp4-awq.jsonl").write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in scored) + "\n", encoding="utf-8"
        )
        payload = fmt.score("smoke")
        assert payload["targets"]["nvfp4-awq"]["n"] == len(fmt.CASES)
        assert 0.0 <= payload["targets"]["nvfp4-awq"]["pass_rate"] <= 1.0


def test_format_verdict_against_contract():
    """계약 하한(format_pass_rate_min) 미달이면 verdict는 FAIL — 통과율을 보고 PASS로 돌리지 않는다."""
    good = {
        "no_hangul": "The weather was nice today, so I went for a walk.",
        "has_hangul": "오늘은 날씨가 좋아서 산책을 했습니다.",
        "has_json": '{"name": "홍길동", "age": 30}',
        "has_marker": "2+2 = 4\n#### 4",
        "short": "세종대왕은 훈민정음을 만들었다.",
        "very_short": "안녕",
        "has_bullets": "- 이유1\n- 이유2\n- 이유3",
        "asks_back": "어떤 도움이 필요하신가요?",
        "no_preamble": "2의 제곱근은 약 1.414입니다.",
        "has_two_lines": "첫째 줄\n둘째 줄",
    }
    # 마지막 2케이스를 실패시키기 위한 응답(각 검사 키의 실패 조건을 만족)
    bad = {
        "no_preamble": "안녕하세요! 도와드리겠습니다.",
        "has_two_lines": "한 줄만 출력",
    }
    with tempfile.TemporaryDirectory() as d:
        _isolate(fmt, d)
        fmt.prepare("smoke")
        prompts = Path(d) / "outputs" / "eval-smoke" / "prompts-format.jsonl"
        rows = [json.loads(x) for x in prompts.read_text(encoding="utf-8").splitlines() if x.strip()]
        outdir = Path(d) / "outputs" / "eval-smoke"
        # 10케이스 중 8개만 통과 → pass_rate 0.8 < 계약 하한 0.9
        scored = []
        for i, r in enumerate(rows):
            text = good[str(r["check"])] if i < 8 else bad[str(r["check"])]
            assert fmt.check_output(str(r["check"]), text)[0] is (i < 8)
            scored.append({"bench": "format", "case": r["case"], "check": r["check"], "output": text})
        (outdir / "nvfp4-awq.jsonl").write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in scored) + "\n", encoding="utf-8"
        )
        payload = fmt.score("smoke")
        t = payload["targets"]["nvfp4-awq"]
        assert t["pass_rate"] == 0.8
        assert t["verdict"] == "FAIL", "계약 하한 미달인데 PASS로 판정됨(가짜 통과)"
        assert payload["verdict"]["nvfp4-awq"] == "FAIL"


def test_card_format_verdict_is_not_false_pass():
    """모델 카드의 형식 판정은 채점 verdict를 따른다(통과율만 보고 PASS로 만들지 않는다)."""
    failing = {"format": {"targets": {"nvfp4-awq": {"n": 10, "pass_rate": 0.8, "pass_rate_min": 0.9}},
                          "verdict": {"nvfp4-awq": "FAIL"}}}
    assert prep._format_contract_verdict(failing, "nvfp4-awq") == "FAIL"
    # verdict가 없는 구 산출물에서도 계약 하한으로 계산한다(가짜 통과 방지)
    legacy = {"format": {"targets": {"nvfp4-awq": {"n": 10, "pass_rate": 0.8}}}}
    assert prep._format_contract_verdict(legacy, "nvfp4-awq") == "FAIL"
    passing = {"format": {"targets": {"nvfp4-awq": {"n": 10, "pass_rate": 1.0}},
                          "verdict": {"nvfp4-awq": "PASS"}}}
    assert prep._format_contract_verdict(passing, "nvfp4-awq") == "PASS"
    assert prep._format_contract_verdict({}, "nvfp4-awq") == "-"


# ---------------------------------------------------------------- 서빙 벤치


def test_servebench_skips_without_server():
    with tempfile.TemporaryDirectory() as d:
        _isolate(sb, d)
        payload = sb.main("smoke", 1, 1, "http://127.0.0.1:1/v1", 1, None, 16)
        assert payload["ok"] is False
        assert payload["status"] == "skipped"
        assert "published_reference" in payload


def test_servebench_infer_target():
    assert sb.infer_target("tayaee/Raon-1-mini-preview-nvfp4-awq") == "nvfp4-awq"
    assert sb.infer_target("tayaee/Raon-1-mini-preview-smoke") == "base"
    assert sb.infer_target("tayaee/Raon-1-mini-preview") == "base"
    assert sb.infer_target("tayaee/Raon-1-mini-preview-awq-w3") == "awq-w3"


def test_servebench_skip_preserves_history():
    with tempfile.TemporaryDirectory() as d:
        _isolate(sb, d)
        outdir = Path(d) / "outputs" / "eval-smoke"
        outdir.mkdir(parents=True)
        prior = {"servebench": {"ok": True, "targets": {"base": {"1": {"tok_per_s_p50": 42.0}}},
                                "hardware": {"gpu": "X"}}}
        (outdir / "checks.json").write_text(json.dumps(prior), encoding="utf-8")
        payload = sb.main("smoke", 1, 1, "http://127.0.0.1:1/v1", 1, None, 16)
        assert payload["ok"] is False
        assert payload["targets"] == {"base": {"1": {"tok_per_s_p50": 42.0}}}
        assert payload["hardware"] == {"gpu": "X"}


def test_servebench_hardware_keys():
    hw = sb.hardware_info("none")
    for k in ("gpu", "gpu_driver", "unified_mem_gb", "vllm_version",
              "tensor_parallel", "gpu_memory_utilization", "max_model_len", "quantization"):
        assert k in hw, k
    assert hw["quantization"] == "none"


def test_serve_section_comparison_table():
    sb_doc = {"ok": True, "hardware": {"gpu": "G", "unified_mem_gb": 128.0, "vllm_version": "v",
                                       "tensor_parallel": 1, "gpu_memory_utilization": 0.9,
                                       "max_model_len": 8192},
              "targets": {"base": {"1": {"tok_per_s_p50": 50.0, "req_latency_s_p50": 5.0},
                                   "8": {"tok_per_s_p50": 200.0}, "32": {"tok_per_s_p50": 300.0},
                                   "memory": {"before_load": {"mem_avail_gb": 114.0, "server_rss_gb": 0.0},
                                              "after_warmup": {"mem_avail_gb": 52.0, "server_rss_gb": 60.1},
                                              "before_teardown": {"mem_avail_gb": 51.0, "server_rss_gb": 61.2},
                                              "after_teardown": {"mem_avail_gb": 113.0, "server_rss_gb": 0.0}}},
                          "nvfp4-awq": {"1": {"tok_per_s_p50": 90.0, "req_latency_s_p50": 2.8}}}}
    out = prep._serve_section(sb_doc, ["base", "nvfp4-awq"], "full")
    assert "Hardware: G" in out and "TP=1" in out
    assert "50.0" in out and "90.0" in out and "C=32 tok/s" in out
    assert "before_load 114.0" in out and "RSS peak 61.2 GB" in out


if __name__ == "__main__":
    raise SystemExit(run_standalone(dict(globals())))
