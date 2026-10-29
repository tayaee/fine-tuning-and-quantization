"""s4-20-aggregate-gsm8k__transformers_vllm_both.py — eval 산출물 집계 (완결율 + 벤치마크별 정확도).
입력: outputs/eval-<mode>/<target>.jsonl (최상위 glob만 — bench/ 하위는 상세 사본)
지표: 완결율(빈 출력 제외), 평균 출력 길이, thinking 포함률(<think> 흔적),
      벤치별 정확도 (kmmlu_redux / gsm8k / humaneval, gold가 있는 행만)
출력: outputs/eval-<mode>/score.json (stdout 표)

  uv run s4-20-aggregate-gsm8k__transformers_vllm_both.py --mode smoke
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from sdlc import load_acceptance, merge_check, target_spec

SHARED = os.environ.get("RQ_SHARED", "/rosenas/data/AIML/raon-1-mini-preview-29b-quantization-shared")
BENCHMARKS = ["kmmlu_redux", "gsm8k", "humaneval"]

#: 정확도로 채점할 수 있는 벤치. 통과율 벤치(format)는 s4-22가 담당하므로 여기서 제외한다
#: (제외하지 않으면 "채점 불가"가 acc=0.0으로 기록돼 정상 벤치와 구분되지 않는다).
SCORABLE_BENCHES = frozenset({"gsm8k", "kmmluko", "kmmlu_redux", "humaneval"})
#: 계약 파일(config/acceptance.toml)이 없을 때 쓰는 기본 허용폭
DEFAULT_DROP_MAX = 0.02

_ANSWER_RE = re.compile(r"(?:정답|답|answer)\s*(?:은|는|:|=)?\s*\**\s*\(?([A-D])\)?\b", re.IGNORECASE)
_BARE_MCQ_RE = re.compile(r"^\s*\**\(?([A-D])[\.\):]")
_HASH_NUM_RE = re.compile(r"####\s*([-+]?[\d,]*\.?\d+)")
_NUM_RE = re.compile(r"[-+]?\d[\d,]*\.?\d*")
_CODE_FENCE_RE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL)


def _norm_number(text: str) -> str:
    return text.replace(",", "").rstrip(".0").rstrip(".")


def _mcq_pred(output: str) -> str | None:
    """선택지 정답 추출: '정답은 B' / '답: C' 같은 명시 마커 → 없으면 첫 글자."""
    m = _ANSWER_RE.search(output)
    if m:
        return m.group(1).upper()
    m = _BARE_MCQ_RE.search(output)
    return m.group(1).upper() if m else None


def _gsm8k_pred(output: str) -> str | None:
    """정답 수 추출: '#### 42' 우선, 없으면 출력의 마지막 수."""
    m = _HASH_NUM_RE.search(output)
    if m:
        return _norm_number(m.group(1))
    nums = _NUM_RE.findall(output)
    return _norm_number(nums[-1]) if nums else None


def _humaneval_pass(output: str, gold: dict) -> bool:
    """코드 블록을 뽑아 gold의 test 하네스를 exec해 통과 여부를 본다."""
    blocks = _CODE_FENCE_RE.findall(output)
    code = blocks[-1] if blocks else output
    ns: dict = {}
    try:
        exec(compile(code, "<generated>", "exec"), ns)  # noqa: S102 — 벤치 정답 판정용 격리 실행
    except Exception:  # noqa: BLE001 — 모델이 뱉은 코드는 어떤 예외든 미통과
        return False
    test_src = gold.get("test") or ""
    entry = gold.get("entry_point", "")
    if not test_src or entry not in ns:
        return False
    try:
        exec(compile(test_src, "<humaneval_test>", "exec"), ns)  # noqa: S102 — 벤치 정답 판정용 격리 실행
        harness = ns.get("check") or ns.get("test")
        if harness is None:
            return False
        harness(ns[entry])
    except Exception:  # noqa: BLE001 — 실패하면 미통과
        return False
    return True


def score_answer(bench: str, gold, output: str) -> int:
    """벤치별 정답 일치 여부 (1/0). gold가 None/빈 값이면 채점 불가 → 0."""
    if gold is None or gold == "":
        return 0
    if bench in ("kmmlu_redux", "kmmluko"):
        pred = _mcq_pred(output)
        gold_list = gold if isinstance(gold, list) else [gold]
        return 1 if pred in {str(g).strip().upper() for g in gold_list} else 0
    if bench == "gsm8k":
        pred = _gsm8k_pred(output)
        return 1 if pred is not None and pred == _norm_number(str(gold)) else 0
    if bench == "humaneval":
        return 1 if isinstance(gold, dict) and _humaneval_pass(output, gold) else 0
    return 0


def score(mode: str) -> dict:
    outdir = OUTDIR(mode)
    rows: dict[str, dict] = {}
    for name in sorted(os.listdir(outdir)) if os.path.isdir(outdir) else []:
        if not name.endswith(".jsonl") or name.startswith("prompts-"):  # 최상위만 — prompts-*/bench/ 제외
            continue
        fp = os.path.join(outdir, name)
        with open(fp, encoding="utf-8") as f:
            recs = [json.loads(line) for line in f if line.strip()]
        t = name[:-6]
        outs = [r.get("output") or "" for r in recs]
        n = len(outs)
        done = sum(1 for o in outs if o.strip())
        think = sum(1 for o in outs if "<think>" in o.lower() or "생각" in o or "추론" in o)
        benches: dict[str, dict] = {}
        for r in recs:
            b = r.get("bench", "smoke")
            if b not in SCORABLE_BENCHES:
                continue  # 통과율 벤치(format) 등은 s4-19/s4-22가 checks.json에 기록한다
            slot = benches.setdefault(b, {"n": 0, "acc_sum": 0})
            slot["n"] += 1
            slot["acc_sum"] += score_answer(b, r.get("gold"), r.get("output") or "")
        row = {
            "n": n,
            "done_rate": round(done / max(n, 1), 3),
            "avg_len": round(sum(len(o) for o in outs) / max(n, 1), 1),
            "think_rate": round(think / max(n, 1), 3),
            "benchmarks": {
                b: {"n": s["n"], "acc": round(s["acc_sum"] / max(s["n"], 1), 3)}
                for b, s in sorted(benches.items())
            },
        }
        rows[t] = row
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "score.json"), "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    merge_check(mode, "gsm8k", gsm8k_check(mode, rows))
    print(f"{'target':12} {'n':>4} {'done':>6} {'avglen':>8} {'think':>6}  benchmarks")
    for t, r in rows.items():
        bs = " ".join(f"{b}={v['acc']}({v['n']})" for b, v in r["benchmarks"].items())
        print(f"{t:12} {r['n']:>4} {r['done_rate']:>6} {r['avg_len']:>8} {r['think_rate']:>6}  {bs}")
    return rows


def gsm8k_stats(mode: str, target: str) -> dict:
    """타깃별 GSM8K 통계: 정확도 · 답 마커율 · 절단률(답 마커 누락)."""
    fp = os.path.join(OUTDIR(mode), f"{target}.jsonl")
    if not os.path.isfile(fp):
        return {}
    n = marker = 0
    with open(fp, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("bench") not in (None, "gsm8k"):
                continue
            out = r.get("output") or ""
            n += 1
            if "####" in out:
                marker += 1
    return {"n": n, "answer_rate": round(marker / max(n, 1), 3) if n else 0.0,
            "truncation_rate": round((n - marker) / max(n, 1), 3) if n else 0.0}


def _drop_max(target: str, key: str, default: float) -> float:
    """계약 파일(acceptance.toml)이 없거나 타깃 항목이 없으면 기본값을 쓴다."""
    try:
        return float(target_spec(load_acceptance(), target).get(key, default))
    except Exception:  # noqa: BLE001 — 계약 파일이 없으면 기본값
        return default


def gsm8k_check(mode: str, rows: dict[str, dict]) -> dict:
    """GSM8K 검사 결과: 정확도·답 마커율·절단률 + 수용 기준(base 대비 하락 허용폭) 판정."""
    targets: dict[str, dict] = {}
    for t, row in rows.items():
        g = (row.get("benchmarks") or {}).get("gsm8k") or {}
        if not g:
            continue
        st = gsm8k_stats(mode, t)
        targets[t] = {
            "n": g.get("n"),
            "acc": g.get("acc"),
            "answer_rate": st.get("answer_rate"),
            "truncation_rate": st.get("truncation_rate"),
        }
    base = targets.get("base") or {}
    base_acc = base.get("acc")
    verdicts: dict[str, str] = {}
    deltas: dict[str, float] = {}
    drop_maxes: dict[str, float] = {}
    for t, v in targets.items():
        if t == "base":
            verdicts[t] = "ref"
            continue
        acc = v.get("acc")
        if acc is None or base_acc is None:
            verdicts[t] = "no-base"
            continue
        drop = round(float(base_acc) - float(acc), 4)
        deltas[t] = drop
        drop_max = _drop_max(t, "gsm8k_acc_drop_max", DEFAULT_DROP_MAX)
        drop_maxes[t] = drop_max
        verdicts[t] = "PASS" if drop <= drop_max else "FAIL"
    return {
        "targets": targets,
        "delta_vs_base": deltas,
        "verdict": verdicts,
        # 계약 파일이 없을 때의 기본값 + 타깃별 실제 적용값(수치 불일치 방지)
        "contract": {"gsm8k_acc_drop_max": DEFAULT_DROP_MAX},
        "drop_max": drop_maxes,
        "note": "answer_rate = fraction with a '#### N' marker; truncation_rate = 1 - answer_rate",
    }


def OUTDIR(mode: str) -> str:
    return f"{SHARED}/outputs/eval-{mode}"


def main(mode: str):
    return score(mode)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="eval 산출물 집계 (완결율 + 벤치마크 정확도)")
    ap.add_argument("--mode", default=os.environ.get("MODE", "smoke"), choices=["smoke", "full"])
    a = ap.parse_args()
    main(a.mode)