"""s7-19-verify-serving-log.py — Stage 7 CAT 감사 (서빙 chat·pi 결과 대조).

s7-13-chat.sh가 남긴 logs/s7-13.<slug>.log와 s7-16-pi.sh가 남긴
logs/s7-16-<slug>.log를 읽어 "모델이 실제로 서빙되고 chat·tool-call 응답을
냈는가"를 확인한다. 결과는 outputs/serve-<mode>.json으로 남으며
gates.inc의 rq_gate_serving이 이 파일을 대조한다.

서빙을 아직 하지 않았다면(로그 없음) skipped로 기록하고 실패로 보지 않는다
(CAT는 수동 단계이므로 파이프라인을 막지 않는다).

  uv run s7-19-verify-serving-log.py --mode smoke
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
TARGETS = ("awq-w3", "awq-w5", "nvfp4-awq", "nvfp4-gptq")
FAIL_MARKERS = ("SERVE-FAIL", "CHAT-FAIL", "PI-FAIL", "INFER-FAIL")


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
    for marker in FAIL_MARKERS:
        if marker in text:
            return {"status": "fail", "reason": marker}
    if "[OK]" not in text:
        return {"status": "fail", "reason": "OK 마커 없음"}
    n_resp = len(re.findall(r"^\[resp\]", text, re.MULTILINE))
    return {"status": "ok", "responses": n_resp}


def main() -> int:
    ap = argparse.ArgumentParser(description="CAT 서빙 검증 (Stage 7)")
    ap.add_argument("--mode", default=os.environ.get("MODE", "smoke"), choices=["smoke", "full"])
    ap.add_argument("--log-dir", default=str(ROOT / "logs"))
    a = ap.parse_args()

    print("[s7-19-verify-serving-log] Stage 7 - verify the serving logs and write outputs/serve-<mode>.json")
    log_dir = Path(a.log_dir)
    results: dict[str, dict] = {}
    for t in TARGETS:
        slug = expected_slug(a.mode, t)
        chat = inspect_log(log_dir / f"s7-13-{slug}.log")
        pi = inspect_log(log_dir / f"s7-16-{slug}.log")
        results[t] = {"chat": chat, "pi": pi,
                      "status": "fail" if "fail" in (chat["status"], pi["status"])
                                else ("missing" if "missing" in (chat["status"], pi["status"]) else "ok")}

    failed = [t for t, r in results.items() if r["status"] == "fail"]
    missing = [t for t, r in results.items() if r["status"] == "missing"]
    ok = bool(results) and not failed and not missing
    payload = {
        "mode": a.mode,
        "generated_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "ok": ok,
        "status": "ok" if ok else ("skipped" if not failed and missing else "fail"),
        "errors": [f"{t}: {results[t]}" for t in failed + missing],
        "targets": results,
    }
    out = SHARED / "outputs" / f"serve-{a.mode}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if failed:
        return 1
    if missing:
        print(f"SKIP: 서빙 로그가 없는 타겟 {missing} (s7-13-chat.sh/s7-16-pi.sh 실행 필요)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
