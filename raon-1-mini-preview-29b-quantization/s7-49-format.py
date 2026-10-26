"""s7-49-format.py — 지시/형식 준수 스위트 (Tier 2: 생성 규약).

PPL은 teacher-forced, GSM8K/KMMLU는 "정답이 맞는가"만 본다. **"요청한 형식을 지켰는가"**
는 어느 검사도 하지 않는다 — 실제 배포에서 가장 흔한 실패 모드다(언어 지정 무시,
지시 무시, Rambling, JSON 미출력).

고정된 케이스(내장)를 prompts-format.jsonl로 만들어 s7-37-eval-base.sh/s7-38-eval-quantized.sh가 생성하고,
이 스크립트가 응답을 검사해 pass_rate를 산출한다.

  --prepare : 프롬프트 세트 생성(파이프라인의 eval 단계 앞에서 1회)
  (기본)    : 생성된 응답을 검사 → checks.json의 "format" 섹션

  uv run s7-49-format.py --mode smoke --prepare
  uv run s7-49-format.py --mode smoke
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from sdlc import load_acceptance, merge_check, target_spec

SHARED = Path(os.environ.get("RQ_SHARED", "/rosenas/data/AIML/raon-1-mini-preview-29b-quantization-shared"))
BENCH = "format"
#: 계약 파일(config/acceptance.toml)이 없을 때 쓰는 기본 통과율 하한
DEFAULT_PASS_RATE_MIN = 0.9

#: (케이스 이름, 프롬프트, 검사 키) — 프롬프트는 Raon chat template 형식
CASES: list[tuple[str, str, str]] = [
    ("translate_to_english", "이 문장을 영어로 번역해 주세요: 오늘은 날씨가 좋아서 산책을 했습니다.", "no_hangul"),
    ("translate_to_korean", "Translate this sentence into Korean: The cat sat on the mat.", "has_hangul"),
    ("json_only", "다음 정보를 JSON으로만 답해 주세요: name=홍길동, age=30", "has_json"),
    ("number_format", "2+2를 계산하고 최종 답은 `#### 4` 형식으로만 답하세요.", "has_marker"),
    ("one_sentence", "세종대왕의 주요 업적을 한 문장으로 요약해 주세요.", "short"),
    ("single_word", "안녕이라고 한 단어만 답해 주세요.", "very_short"),
    ("three_bullets", "사과를 먹는 이유를 3개만 bullet으로 답해 주세요.", "has_bullets"),
    ("empty_question", "", "asks_back"),
    ("no_preamble", "2의 제곱근을 구하세요. 답만 쓰세요.", "no_preamble"),
    ("keep_format", "다음 두 줄을 그대로 출력하세요:\n첫째 줄\n둘째 줄", "has_two_lines"),
]


def _hangul_count(text: str) -> int:
    return sum(1 for c in text if "가" <= c <= "힣")


def check_output(key: str, text: str) -> tuple[bool, str]:
    """검사 키별 판정. (통과?, 사유)"""
    t = text or ""
    stripped = t.strip()
    if key == "no_hangul":
        return (_hangul_count(t) == 0, "한글 0자여야 함")
    if key == "has_hangul":
        return (_hangul_count(t) > 0, "한글이 있어야 함")
    if key == "has_json":
        return ("{" in t and "}" in t, "JSON 중괄호가 있어야 함")
    if key == "has_marker":
        return ("####" in t, "#### 마커가 있어야 함")
    if key == "short":
        sents = len(re.findall(r"[.!?。！？]", stripped))
        return (sents <= 3 and len(stripped) <= 400, "3문장 이내·400자 이내")
    if key == "very_short":
        return (len(stripped) <= 30, "30자 이내")
    if key == "has_bullets":
        bullets = len(re.findall(r"(?m)^\s*(?:[-*•]|\d+[.)])\s+", t))
        return (bullets >= 2, "bullet 2개 이상")
    if key == "asks_back":
        ok = ("?" in t) or ("무엇" in t) or ("어떤" in t)
        return (ok, "되묻는 답이어야 함")
    if key == "no_preamble":
        head = stripped[:40]
        bad = ("안녕하세요" in head) or ("도와드리겠습니다" in head) or ("Sure" in head) or ("Here" in head)
        return (not bad, "사전 멘트 없이 바로 답")
    if key == "has_two_lines":
        lines = [ln for ln in t.splitlines() if ln.strip()]
        return (len(lines) >= 2, "2줄 이상 출력")
    return (True, "no check")


def outdir(mode: str) -> Path:
    return SHARED / "outputs" / f"eval-{mode}"


def _pass_rate_min(target: str) -> float:
    """계약 파일(acceptance.toml)의 통과율 하한. 없으면 기본값."""
    try:
        return float(target_spec(load_acceptance(), target).get("format_pass_rate_min", DEFAULT_PASS_RATE_MIN))
    except Exception:  # noqa: BLE001 — 계약 파일이 없으면 기본값
        return DEFAULT_PASS_RATE_MIN


def prompts_path(mode: str) -> Path:
    return outdir(mode) / f"prompts-{BENCH}.jsonl"


def meta_index(mode: str) -> dict[str, dict]:
    """프롬프트 세트의 (case, check)를 prompt 문자열로 색인한다.

    응답 행에는 검사 키가 없을 수 있다(예측 생성기가 case/check를 보존하지 않은 산출물).
    그 상태로 채점하면 "검사 키 없음"이 통과로 집계돼 **가짜 통과**가 되므로,
    프롬프트 세트와 prompt 문자열로 반드시 조인한다.
    """
    idx: dict[str, dict] = {}
    fp = prompts_path(mode)
    if not fp.is_file():
        return idx
    with fp.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            idx[str(r.get("prompt", ""))] = {"case": r.get("case"), "check": r.get("check")}
    return idx


def prepare(mode: str) -> int:
    out = prompts_path(mode)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for name, prompt, key in CASES:
            row = {"bench": BENCH, "case": name, "check": key,
                   "prompt": f"user\n{prompt}\nassistant\n<think></think>\n\n"}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(out)
    print(f"format prompts -> {out} ({len(CASES)} cases)")
    return len(CASES)


def score(mode: str) -> dict:
    rows_by_target: dict[str, list[dict]] = {}
    for fp in sorted(outdir(mode).glob("*.jsonl")):
        if fp.name.startswith("prompts-"):
            continue
        target = fp.stem
        rows: list[dict] = []
        with fp.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                if r.get("bench") == BENCH:
                    rows.append(r)
        if rows:
            rows_by_target[target] = rows

    results: dict[str, dict] = {}
    meta = meta_index(mode)
    for target, rows in rows_by_target.items():
        passed = 0
        details: list[dict] = []
        for r in rows:
            case = r.get("case")
            check = r.get("check")
            if not check or not case:
                # 응답 행에 메타데이터가 없으면 프롬프트 세트와 조인한다.
                m = meta.get(str(r.get("prompt", ""))) or {}
                case = case or m.get("case")
                check = check or m.get("check")
            if check:
                ok, why = check_output(str(check), r.get("output") or "")
            else:
                # 검사 키를 못 찾으면 통과로 보지 않는다(가짜 통과 방지).
                ok, why = False, "검사 키 없음(메타데이터 유실)"
            passed += 1 if ok else 0
            details.append({"case": case, "check": check, "ok": ok, "reason": why,
                            "output_head": (r.get("output") or "")[:120]})
        rate = round(passed / max(len(rows), 1), 3)
        min_rate = _pass_rate_min(target)
        results[target] = {
            "n": len(rows),
            "pass_rate": rate,
            "pass_rate_min": min_rate,
            # base는 기준선(ref) — 게이트 대상이 아니다.
            "verdict": "ref" if target == "base" else ("PASS" if rate >= min_rate else "FAIL"),
            "failed": [d["case"] or "(unknown)" for d in details if not d["ok"]],
            "cases": details,
        }
    payload = {
        "bench": BENCH,
        "cases": len(CASES),
        "targets": results,
        "verdict": {t: v["verdict"] for t, v in results.items()},
        # 계약 파일이 없을 때의 기본값 + 타깃별 실제 적용값(수치 불일치 방지)
        "contract": {"format_pass_rate_min": DEFAULT_PASS_RATE_MIN},
        "pass_rate_min": {t: v["pass_rate_min"] for t, v in results.items()},
    }
    merge_check(mode, BENCH, payload)
    print(f"{'target':12} {'n':>4} {'pass':>6}  failed")
    for t, v in sorted(results.items()):
        print(f"{t:12} {v['n']:>4} {v['pass_rate']:>6}  {','.join(v['failed']) or '-'}")
    return payload


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="지시/형식 준수 스위트 (Tier 2)")
    ap.add_argument("--mode", default=os.environ.get("MODE", "smoke"), choices=["smoke", "full"])
    ap.add_argument("--prepare", action="store_true", help="프롬프트 세트만 생성")
    a = ap.parse_args()
    if a.prepare:
        prepare(a.mode)
    else:
        score(a.mode)
