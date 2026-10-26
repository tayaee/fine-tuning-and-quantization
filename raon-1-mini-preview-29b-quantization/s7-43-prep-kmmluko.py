"""s7-43-prep-kmmluko.py — 표준 한국어 KMMLU 평가 세트 준비 (Tier 2: 한국어 커버리지).

GSM8K가 영어 수학이라 이 모델(한국어 우선)의 한국어 능력을 아무것도 재지 않는다.
여기서는 **표준 한국어 벤치마크** `HAERAE-HUB/KMMLU`(44개 과목 4지선다 MCQ)에서
과목별로 고르게 표본을 뽑아 `outputs/eval-<mode>/prompts-kmmluko.jsonl`을 만든다.
생성은 s7-37-eval-base.sh/s7-38-eval-quantized.sh(벤치 공통), 채점은 s7-46-kmmluko.sh.

  - 프롬프트: Raon chat template + 4지선다 + "정답은 X" 지시
  - gold: 정답 인덱스(0~3) → 채점 시 A/B/C/D로 매핑
  - 결정론: (과목 목록, seed, n)으로 항상 같은 문제들

  uv run s7-43-prep-kmmluko.py --mode full --n 200 --subjects 8
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import zlib
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

SHARED = Path(os.environ.get("RQ_SHARED", "/rosenas/data/AIML/raon-1-mini-preview-29b-quantization-shared"))

BENCH = "kmmluko"
DATASET = os.environ.get("KMMLU_DATASET", "HAERAE-HUB/KMMLU")
#: 표준 한국어 KMMLU의 과목 코드(44개 중 골라 쓴다). 한글 지식·추론 폭을 넓게 덮기 위해 diverse 선택.
ALL_SUBJECTS = [
    "Accounting", "Biology", "Chemistry", "Computer-Science", "Economics", "Education",
    "Electrical-Engineering", "Electronics-Engineering", "Energy-Management", "Environmental-Science",
    "Food-Processing", "Health", "Industrial-Engineer", "Information-Technology", "Interior-Architecture-and-Design",
    "Law", "Machine-Design-and-Manufacturing", "Management", "Marketing", "Materials-Engineering",
    "Mechanical-Engineering", "Political-Science-and-Sociology", "Psychology", "Public-Safety",
    "Real-Estate", "Social-Welfare", "Taxation", "Telecommunications-and-Wireless-Technology",
    "Korean-History", "Math", "Aviation-Engineering-and-Maintenance", "Agricultural-Sciences",
    "Civil-Engineering", "Construction", "Criminal-Law", "Ecology", "Fashion", "Gas-Technology-and-Engineering",
    "Geomatics", "Maritime-Engineering-and-Shipping", "Nondestructive-Testing", "Patent",
]
DEFAULT_SUBJECTS = [
    "Biology", "Chemistry", "Computer-Science", "Economics", "Korean-History",
    "Law", "Math", "Management",
]
SEED = int(os.environ.get("KMMLU_SEED", "20261024"))
INSTRUCTION = (
    "다음 객관식 문제를 읽고 알맞은 정답을 고르세요. "
    "설명 없이 첫 줄에 반드시 \"정답은 X\" (X는 A, B, C, D 중 하나) 형식으로만 답하세요."
)
LETTERS = ("A", "B", "C", "D")


def outdir(mode: str) -> Path:
    return SHARED / "outputs" / f"eval-{mode}"


def prompts_path(mode: str) -> Path:
    return outdir(mode) / f"prompts-{BENCH}.jsonl"


def manifest_path(mode: str) -> Path:
    return outdir(mode) / f"evalset-{BENCH}.json"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _format_prompt(question: str, options: list[str]) -> str:
    opts = "\n".join(f"{LETTERS[i]}. {o}" for i, o in enumerate(options))
    return f"user\n{INSTRUCTION}\n\n문제: {question.strip()}\n{opts}\nassistant\n<think></think>\n\n"


def load_rows(subjects: list[str], per_subject: int, seed: int) -> list[dict]:
    """과목별 표본을 결정론적으로 수집한다."""
    from datasets import load_dataset

    rows: list[dict] = []
    for subj in subjects:
        ds = load_dataset(DATASET, subj, split="test")
        items: list[Any] = list(ds)
        idxs = list(range(len(items)))
        random.Random(seed + zlib.crc32(subj.encode("utf-8")) % 100000).shuffle(idxs)
        for i in idxs[:per_subject]:
            r = items[i]
            # KMMLU는 컬럼이 평평하다(question/answer/A/B/C/D) — answer는 1-based.
            choices = [str(r.get(k, "")) for k in ("A", "B", "C", "D")]
            if not all(choices):
                continue
            try:
                ans = int(str(r.get("answer", "0")).strip()) - 1
            except (TypeError, ValueError):
                continue
            if not (0 <= ans < 4):
                continue
            rows.append({
                "subject": subj,
                "question": str(r.get("question", "")),
                "choices": choices,
                "answer": ans,
            })
    return rows


def build(mode: str, n: int, subjects: list[str], seed: int) -> dict:
    rows = load_rows(subjects, max(1, n // max(1, len(subjects))), seed)
    out = prompts_path(mode)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for r in rows:
            row = {
                "bench": BENCH,
                "prompt": _format_prompt(r["question"], r["choices"]),
                "gold": LETTERS[r["answer"]],
                "subject": r["subject"],
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(out)
    manifest = {
        "mode": mode,
        "bench": BENCH,
        "dataset": DATASET,
        "subjects": subjects,
        "seed": seed,
        "requested_n": n,
        "written_n": len(rows),
        "sha256": _sha256(out),
        "generated_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "instruction": INSTRUCTION,
    }
    mpath = manifest_path(mode)
    mpath.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def is_fresh(mode: str, n: int, subjects: list[str], seed: int) -> bool:
    mpath = manifest_path(mode)
    ppath = prompts_path(mode)
    if not mpath.is_file() or not ppath.is_file():
        return False
    try:
        m = json.loads(mpath.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if m.get("seed") != seed or m.get("dataset") != DATASET:
        return False
    if list(m.get("subjects") or []) != list(subjects):
        return False
    if int(m.get("requested_n", -1)) != int(n):
        return False
    if int(m.get("written_n", 0)) <= 0:
        return False  # 0행 세트는 유효하지 않다(로더가 깨졌을 때 생김)
    return m.get("sha256") == _sha256(ppath)


def main() -> int:
    ap = argparse.ArgumentParser(description="표준 한국어 KMMLU 평가 세트 준비")
    ap.add_argument("--mode", default=os.environ.get("MODE", "smoke"), choices=["smoke", "full"])
    ap.add_argument("--n", type=int, default=int(os.environ.get("KMMLU_N", "40")))
    ap.add_argument("--subjects", default=",".join(DEFAULT_SUBJECTS))
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()

    subjects = [s for s in a.subjects.replace(" ", "").split(",") if s]
    if not a.force and is_fresh(a.mode, a.n, subjects, a.seed):
        print(f"evalset OK (fresh) {prompts_path(a.mode)}")
        return 0
    if a.check:
        print(f"evalset STALE {prompts_path(a.mode)}")
        return 1
    m = build(a.mode, a.n, subjects, a.seed)
    print(f"evalset OK: {m['dataset']} subjects={len(subjects)} n={m['written_n']} -> {prompts_path(a.mode)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
