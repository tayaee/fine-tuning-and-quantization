"""s4-04-prep-gsm8k.py — 표준 GSM8K 평가 세트 준비 (Stage 7의 입력).

기존에는 `outputs/eval-<mode>/prompts-*.jsonl`을 **만드는 스크립트가 파이프라인에
없었다**(외부에서 수작업으로 만들어 둔 3~10행 샘플). 그래서 평가가 "표준 벤치마크"가
아니라 임의 표본을 재는 상태였다. 이 스크립트가 그 자리를 메워 **표준 GSM8K 테스트
셋**(`openai/gsm8k`, main/test, 1319문제)을 결정론적으로 샘플링해 평가 입력을 만든다.

  - 프롬프트: Raon chat template (`user\\n...\\nassistant\\n<think></think>\\n\\n`)
    + "최종 답은 `#### <number>` 형식" 지시 (기존 ad-hoc 프롬프트와 동일 형식)
  - gold: 데이터셋 정답의 `#### N` 숫자 (콤마 제거)
  - 결정론: (dataset, revision, seed, n) 조합이 같으면 같은 파일 → sha256 기록
  - 멱등: 매니페스트의 파라미터·해시가 일치하면 재생성 없이 skip

출력: outputs/eval-<mode>/prompts-gsm8k.jsonl
      outputs/eval-<mode>/evalset-<mode>.json (매니페스트)

  uv run s4-04-prep-gsm8k.py --mode full --n 200
  uv run s4-04-prep-gsm8k.py --check --mode full
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

SHARED = Path(os.environ.get("RQ_SHARED", "/rosenas/data/AIML/raon-1-mini-preview-29b-quantization-shared"))

#: 표준 GSM8K. revision은 2026-10-24 기준 main 브랜치 스냅샷(고정해야 재현된다).
GSM8K_DATASET = os.environ.get("GSM8K_DATASET", "openai/gsm8k")
GSM8K_CONFIG = os.environ.get("GSM8K_CONFIG", "main")
GSM8K_REVISION = os.environ.get("GSM8K_REVISION", "740312add88f781978c0658806c59bc2815b9866")
#: 샘플링 시드 — 표본이 바뀌지 않도록 고정.
GSM8K_SEED = int(os.environ.get("GSM8K_SEED", "20261024"))

BENCH = "gsm8k"
INSTRUCTION = (
    "Solve the following math word problem concisely. "
    "At the end of your response, write the final numerical answer strictly in the format \"#### <number>\"."
)
_HASH_ANSWER_RE = re.compile(r"####\s*([-+]?[\d,]*\.?\d+)")


def outdir(mode: str) -> Path:
    return SHARED / "outputs" / f"eval-{mode}"


def prompts_path(mode: str) -> Path:
    return outdir(mode) / f"prompts-{BENCH}.jsonl"


def manifest_path(mode: str) -> Path:
    return outdir(mode) / f"evalset-{mode}.json"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _gold_of(answer: str) -> str | None:
    """데이터셋 정답에서 `#### N` 추출 (콤마 제거)."""
    m = _HASH_ANSWER_RE.search(answer or "")
    if not m:
        return None
    return m.group(1).replace(",", "").strip()


def _format_prompt(question: str) -> str:
    return f"user\n{INSTRUCTION}\n\nQuestion: {question.strip()}\nassistant\n<think></think>\n\n"


def load_test_rows() -> list[dict]:
    """표준 GSM8K test split을 로드한다(오프라인 캐시 우선)."""
    from datasets import load_dataset

    ds = load_dataset(GSM8K_DATASET, GSM8K_CONFIG, revision=GSM8K_REVISION, split="test")
    rows: list[dict] = []
    for i, r in enumerate(ds):
        row = dict(r)  # pyright: ignore[reportArgumentType] — datasets 행은 매핑으로 접근
        gold = _gold_of(str(row.get("answer", "")))
        if gold is None:
            continue
        rows.append({"idx": i, "question": str(row.get("question", "")), "gold": gold})
    return rows


def sample_rows(rows: list[dict], n: int, seed: int) -> list[dict]:
    """결정론적 샘플(같은 seed·n이면 항상 같은 문제들).

    **항상 seed로 셔플한 순서**를 쓴다 — 그래야 "앞의 N개"가 임의의 연속 블록이 아니라
    무작위 표본이 된다(n=0=전체일 때도 순서가 섞여 재현 비교가 가능하다).
    """
    order = list(range(len(rows)))
    random.Random(seed).shuffle(order)
    picked = order if (n <= 0 or n >= len(rows)) else order[:n]
    return [rows[i] for i in picked]


def build(mode: str, n: int, seed: int) -> dict:
    rows = load_test_rows()
    picked = sample_rows(rows, n, seed)
    out = prompts_path(mode)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for r in picked:
            row = {
                "bench": BENCH,
                "prompt": _format_prompt(r["question"]),
                "gold": r["gold"],
                "dataset_idx": r["idx"],
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(out)
    manifest = {
        "mode": mode,
        "bench": BENCH,
        "dataset": GSM8K_DATASET,
        "config": GSM8K_CONFIG,
        "revision": GSM8K_REVISION,
        "seed": seed,
        "requested_n": n,
        "written_n": len(picked),
        "pool_size": len(rows),
        "sha256": _sha256(out),
        "generated_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "instruction": INSTRUCTION,
    }
    mpath = manifest_path(mode)
    mpath.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def is_fresh(mode: str, n: int, seed: int) -> bool:
    """매니페스트가 현재 파라미터와 일치하고 파일 해시도 같으면 fresh."""
    mpath = manifest_path(mode)
    ppath = prompts_path(mode)
    if not mpath.is_file() or not ppath.is_file():
        return False
    try:
        m = json.loads(mpath.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if m.get("seed") != seed or m.get("revision") != GSM8K_REVISION:
        return False
    if m.get("dataset") != GSM8K_DATASET or m.get("config") != GSM8K_CONFIG:
        return False
    if int(m.get("requested_n", -1)) != int(n):
        return False
    return m.get("sha256") == _sha256(ppath)


def main() -> int:
    ap = argparse.ArgumentParser(description="표준 GSM8K 평가 세트 준비")
    ap.add_argument("--mode", default=os.environ.get("MODE", "smoke"), choices=["smoke", "full"])
    ap.add_argument("--n", type=int, default=int(os.environ.get("GSM8K_N", "0")), help="0이면 전체(1319)")
    ap.add_argument("--seed", type=int, default=GSM8K_SEED)
    ap.add_argument("--force", action="store_true", help="freshness 무시")
    ap.add_argument("--check", action="store_true", help="파라미터/해시만 확인")
    a = ap.parse_args()

    if not a.force and is_fresh(a.mode, a.n, a.seed):
        print(f"evalset OK (fresh) {prompts_path(a.mode)}")
        return 0
    if a.check:
        print(f"evalset STALE {prompts_path(a.mode)}")
        return 1
    m = build(a.mode, a.n, a.seed)
    print(
        f"evalset OK: {m['dataset']}@{m['revision'][:12]} seed={m['seed']} "
        f"n={m['written_n']}/{m['pool_size']} -> {prompts_path(a.mode)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
