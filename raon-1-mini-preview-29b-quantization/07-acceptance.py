"""07-acceptance.py — Stage 1(요구사항·수용 기준) 실행기.

config/acceptance.toml(= 수용 기준 SSOT)을 읽어 스키마를 검증하고, 그 스냅샷을
outputs/acceptance-<mode>.json으로 남긴다. 이 파일이 Stage 1의 산출물이며,
이후 단계(5 양자화 → 6 PPL → 7 평가 → 8 릴리스)는 전부 이 계약에 대조된다.

  uv run 07-acceptance.py --mode smoke
  uv run 07-acceptance.py --check        # 검증만 (파일 미생성)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from sdlc import (  # sys.path 조작 후 import
    QUANT_TARGETS,
    STAGES,
    acceptance_errors,
    load_acceptance,
    write_json,
)

SHARED = Path(os.environ.get("RQ_SHARED", "/rosenas/data/AIML/raon-1-mini-preview-29b-quantization-shared"))


def acceptance_out(mode: str) -> Path:
    return SHARED / "outputs" / f"acceptance-{mode}.json"


def build_payload(mode: str, data: dict) -> dict:
    return {
        "mode": mode,
        "version": data.get("version"),
        "generated_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source": data.get("source"),
        "targets": {t: data["targets"].get(t, {}) for t in QUANT_TARGETS},
        "gates": data.get("gates", {}),
        "stages": [
            {"id": s["id"], "name": s["name"], "title": s["title"], "artifact": s["artifact"]} for s in STAGES
        ],
        "rule": "ΔPPL < ppl_delta_warn → Accept / < ppl_delta_max → Conditional / 그 이상 → Discard",
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Stage 1: 수용 기준 검증 + acceptance-<mode>.json 생성")
    ap.add_argument("--mode", default=os.environ.get("MODE", "smoke"), choices=["smoke", "full"])
    ap.add_argument("--path", default=None, help="acceptance.toml 경로(기본 config/acceptance.toml)")
    ap.add_argument("--check", action="store_true", help="검증만 하고 파일을 만들지 않는다")
    ap.add_argument("--print", dest="show", action="store_true", help="JSON을 stdout으로 출력")
    a = ap.parse_args()

    data = load_acceptance(a.path)
    errors = acceptance_errors(data)
    if errors:
        for e in errors:
            print(f"FAIL: acceptance 스키마 오류: {e}", file=sys.stderr)
        return 1

    payload = build_payload(a.mode, data)
    if a.show:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    if not a.check:
        out = write_json(acceptance_out(a.mode), payload)
        print(f"acceptance OK -> {out}")
    else:
        print("acceptance OK (check only)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
