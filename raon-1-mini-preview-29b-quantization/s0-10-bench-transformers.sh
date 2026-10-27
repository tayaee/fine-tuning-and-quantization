#!/usr/bin/env bash
# s0-10-bench-transformers.sh [smoke|full] — transformers용 벤치 하네스 스캐폴드 (Stage 0 절차 4).
# 고정 probe 프롬프트 + 실행 파라미터 + 빈 baseline을 만든다. 실제 측정은 s4에서 한다.
set -euo pipefail
source "$(dirname "$0")/config/common.env" "${1:-${MODE:-smoke}}"
rq_repro_echo
rq_banner "s0-10-bench-transformers" "Stage 0 - scaffold the transformers bench harness (prompts/params, no measurement)"
export UV_NO_SYNC=1; unset VIRTUAL_ENV
OUTDIR="$RQ_SHARED/outputs/s0-bench-transformers"
mkdir -p "$OUTDIR"
uv run python - "$OUTDIR" <<'PY'
import json, sys
d = sys.argv[1]
prompts = [
    "다음을 한국어로 한 문장으로 요약하세요: 세종대왕은 훈민정음을 창제하여 백성이 쉽게 글을 읽고 쓸 수 있게 하였다.",
    "사과 3개가 있고 배는 사과보다 2개 많습니다. 배는 몇 개입니까? 단계별로 생각하세요.",
    "1부터 10까지 합을 구하는 파이썬 함수를 작성하세요.",
    "What is 2+2? 한국어로 답하세요.",
    "다음 JSON을 완성하세요: {\"name\": \"세종대왕\", \"born\":",
    "불릿 3개로 아침 루틴을 적으세요.",
    "100자를 넘지 않게 자기소개를 쓰세요.",
    "다음 문장을 영어에서 한국어로 번역하세요: The quick brown fox jumps over the lazy dog.",
]
with open(f"{d}/prompts.jsonl", "w", encoding="utf-8") as f:
    for i, p in enumerate(prompts):
        f.write(json.dumps({"id": i, "prompt": p}, ensure_ascii=False) + "\n")
params = {"runtime": "transformers", "trust_remote_code": True, "dtype": "bfloat16",
          "device_map": "auto", "max_new_tokens": 256, "do_sample": False,
          "note": "실측은 s4-28-bench-transformers에서. baseline은 첫 실측이 채운다."}
with open(f"{d}/params.json", "w", encoding="utf-8") as f:
    json.dump(params, f, ensure_ascii=False, indent=2)
base = f"{d}/baseline.json"
try:
    open(base, "x").write(json.dumps({"measured": False}, ensure_ascii=False, indent=2) + "\n")
except FileExistsError:
    pass
print(f"scaffold OK {d} prompts={len(prompts)}")
PY
rq_stage_record 0 base pass "$OUTDIR/prompts.jsonl"
echo ---- result ----
rq_log "s0-10 OK scaffold=$OUTDIR"
