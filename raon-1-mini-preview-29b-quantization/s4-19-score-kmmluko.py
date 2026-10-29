"""s4-19-score-kmmluko.py — 표준 한국어 KMMLU 채점 + 수용 기준 대조 (Tier 2: 한국어 커버리지).

s4-07-answer-benchmarks-base.sh/s4-10-answer-benchmarks-quantized.sh가 생성한 응답(eval-<mode>/<target>.jsonl) 중 bench=="kmmluko" 행만 모아
4지선다 정답을 파싱해 정확도를 계산하고, Stage 1 계약(base 대비 하락 허용폭)과 대조한다.
결과는 checks.json의 "kmmluko" 섹션으로 병합한다(카드·릴리스 노트가 읽는다).

  uv run s4-19-score-kmmluko.py --mode smoke
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
BENCH = "kmmluko"
LETTERS = ("A", "B", "C", "D")
#: 계약 파일(config/acceptance.toml)이 없을 때 쓰는 기본 허용폭
DEFAULT_DROP_MAX = 0.03

#: "정답은 B" / "정답: B" / "answer is B" / 줄 첫 글자 B
_PATTERNS = (
    re.compile(r"정답\s*은\s*[:：]?\s*([ABCD])\b", re.IGNORECASE),
    re.compile(r"정답\s*[:：]\s*([ABCD])\b", re.IGNORECASE),
    re.compile(r"answer\s+is\s*[:：]?\s*([ABCD])\b", re.IGNORECASE),
    re.compile(r"^\s*\(?([ABCD])[\.\):\s]", re.IGNORECASE),
)


def outdir(mode: str) -> Path:
    return SHARED / "outputs" / f"eval-{mode}"


def parse_letter(text: str) -> str | None:
    """응답에서 정답 레터를 추출한다(없으면 None)."""
    t = (text or "").strip()
    for pat in _PATTERNS:
        m = pat.search(t)
        if m:
            return m.group(1).upper()
    return None


def target_rows(mode: str, target: str) -> list[dict]:
    fp = outdir(mode) / f"{target}.jsonl"
    if not fp.is_file():
        return []
    rows: list[dict] = []
    with fp.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("bench") == BENCH:
                rows.append(r)
    return rows


def score_target(rows: list[dict]) -> dict:
    n = len(rows)
    correct = 0
    answered = 0
    for r in rows:
        letter = parse_letter(r.get("output") or "")
        if letter:
            answered += 1
        if letter and letter == str(r.get("gold") or "").strip().upper():
            correct += 1
    return {
        "n": n,
        "acc": round(correct / max(n, 1), 3) if n else 0.0,
        "answer_rate": round(answered / max(n, 1), 3) if n else 0.0,
    }


def _drop_max(target: str, key: str, default: float) -> float:
    """계약 파일(acceptance.toml)이 없거나 타깃 항목이 없으면 기본값을 쓴다."""
    try:
        return float(target_spec(load_acceptance(), target).get(key, default))
    except Exception:  # noqa: BLE001 — 계약 파일이 없으면 기본값
        return default


def main(mode: str) -> dict:
    score_fp = outdir(mode) / "score.json"
    targets_in_score: list[str] = []
    if score_fp.is_file():
        try:
            data = json.loads(score_fp.read_text(encoding="utf-8"))
            targets_in_score = [k for k, v in data.items() if isinstance(v, dict) and "benchmarks" in v]
        except (OSError, json.JSONDecodeError):
            targets_in_score = []
    if not targets_in_score:
        targets_in_score = sorted(p.stem for p in outdir(mode).glob("*.jsonl") if not p.name.startswith("prompts-"))

    targets: dict[str, dict] = {}
    for t in targets_in_score:
        rows = target_rows(mode, t)
        if not rows:
            continue
        targets[t] = score_target(rows)

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
        drop_max = _drop_max(t, "kmmluko_acc_drop_max", DEFAULT_DROP_MAX)
        drop_maxes[t] = drop_max
        verdicts[t] = "PASS" if drop <= drop_max else "FAIL"

    payload = {
        "bench": BENCH,
        "dataset": "HAERAE-HUB/KMMLU",
        "targets": targets,
        "delta_vs_base": deltas,
        "verdict": verdicts,
        # 계약 파일이 없을 때의 기본값 + 타깃별 실제 적용값(수치 불일치 방지)
        "contract": {"kmmluko_acc_drop_max": DEFAULT_DROP_MAX},
        "drop_max": drop_maxes,
    }
    merge_check(mode, BENCH, payload)
    print(f"{'target':12} {'n':>4} {'acc':>6} {'answered':>8}  verdict")
    for t, v in sorted(targets.items()):
        print(f"{t:12} {v['n']:>4} {v['acc']:>6} {v['answer_rate']:>8}  {verdicts.get(t, '-')}")
    return payload


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="표준 한국어 KMMLU 채점 + 기준 대조")
    ap.add_argument("--mode", default=os.environ.get("MODE", "smoke"), choices=["smoke", "full"])
    a = ap.parse_args()
    main(a.mode)
