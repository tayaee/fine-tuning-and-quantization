"""s8-64-verify-serving.py — Stage 8 배포 후 검증(서빙 스모크 결과 대조).

s8-58-serve-infer-all.sh가 각 타깃에 대해 남긴 logs/55.<slug>.log를 읽어
"모델이 실제로 서빙되고 응답을 냈는가"를 확인한다. 결과는 outputs/serve-<mode>.json
으로 남으며 gates.inc의 rq_gate_serving이 이 파일을 대조한다.

서빙을 아직 하지 않았다면(로그 없음) skipped로 기록하고 실패로 보지 않는다
(배포는 수동 단계이므로 파이프라인을 막지 않는다).

  uv run s8-64-verify-serving.py --mode smoke
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

SHARED = Path(os.environ.get("RQ_SHARED", "/rosenas/data/AIML/raon-1-mini-preview-29b-quantization-shared"))
BASE_MODEL = os.environ.get("BASE_MODEL", "KRAFTON/Raon-1-mini-preview")
TARGETS = ("base", "awq-w3", "awq-w5", "nvfp4-awq", "nvfp4-gptq")


def slugify(text: str) -> str:
    return re.sub(r"/", "__", text).lower()


def expected_slug(mode: str, target: str) -> str:
    slug = f"tayaee/{BASE_MODEL.split('/')[-1]}-{target}"
    if mode == "smoke":
        slug += "-smoke"
    return slugify(slug)


def inspect_log(path: Path) -> dict:
    if not path.is_file():
        return {"status": "missing"}
    text = path.read_text(encoding="utf-8", errors="replace")
    if "SERVE-FAIL" in text:
        return {"status": "fail", "reason": "SERVE-FAIL"}
    if "INFER-FAIL" in text:
        return {"status": "fail", "reason": "INFER-FAIL"}
    if "[OK]" not in text:
        return {"status": "fail", "reason": "OK 마커 없음"}
    n_resp = len(re.findall(r"^\[resp\]", text, re.MULTILINE))
    return {"status": "ok", "responses": n_resp}


def main() -> int:
    ap = argparse.ArgumentParser(description="배포 후 서빙 검증 (Stage 8)")
    ap.add_argument("--mode", default=os.environ.get("MODE", "smoke"), choices=["smoke", "full"])
    ap.add_argument("--log-dir", default=str(ROOT / "logs"))
    a = ap.parse_args()

    print("[s8-64-verify-serving] Stage 8/9 - verify the serving logs and write outputs/serve-<mode>.json")
    log_dir = Path(a.log_dir)
    results: dict[str, dict] = {}
    for t in TARGETS:
        slug = expected_slug(a.mode, t)
        results[t] = inspect_log(log_dir / f"55.{slug}.log")

    failed = [t for t, r in results.items() if r["status"] == "fail"]
    missing = [t for t, r in results.items() if r["status"] == "missing"]
    ok = bool(results) and not failed and not missing
    payload = {
        "mode": a.mode,
        "generated_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "ok": ok,
        "status": "ok" if ok else ("skipped" if not failed and missing else "fail"),
        "errors": [f"{t}: {results[t].get('reason', 'missing')}" for t in failed + missing],
        "targets": results,
    }
    out = SHARED / "outputs" / f"serve-{a.mode}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if failed:
        return 1
    if missing:
        print(f"SKIP: 서빙 로그가 없는 타겟 {missing} (s8-58-serve-infer-all.sh 실행 필요)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
