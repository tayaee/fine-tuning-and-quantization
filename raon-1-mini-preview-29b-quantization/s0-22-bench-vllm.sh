#!/usr/bin/env bash
# s0-22-bench-vllm.sh [smoke|full] — vLLM endpoint로 벤치 실행 + 서버 정리 (Stage 0 절차 8).
# 고정 8프롬프트 하네스를 endpoint로 측정해 results.json을 남기고(최초 1회 baseline 충전),
# 마지막에 s0-13이 띄운 베이스 서버를 내린다 (GPU 반환). PORT(기본 8000).
set -euo pipefail
source "$(dirname "$0")/config/common.env" "${1:-${MODE:-smoke}}"
source "$RQ_ROOT/config/s0-bench.inc"
rq_repro_echo
rq_banner "s0-22-bench-vllm" "Stage 0 - run the bench suite on the vLLM endpoint, then stop the base server"
export UV_NO_SYNC=1; unset VIRTUAL_ENV
PORT="${PORT:-8000}"
OUTDIR="$RQ_SHARED/outputs/s0-bench-vllm"
mkdir -p "$OUTDIR"
if [ ! -s "$OUTDIR/prompts.jsonl" ]; then
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
params = {"runtime": "vllm-openai", "base_url": "http://localhost:8000/v1",
          "served_model": "tayaee/Raon-1-mini-preview-base",
          "max_tokens": 256, "temperature": 0.6, "top_p": 0.95,
          "note": "실측은 s0-22/s4-25에서. baseline은 첫 실측이 채운다."}
with open(f"{d}/params.json", "w", encoding="utf-8") as f:
    json.dump(params, f, ensure_ascii=False, indent=2)
base = f"{d}/baseline.json"
try:
    open(base, "x").write(json.dumps({"measured": False}, ensure_ascii=False, indent=2) + "\n")
except FileExistsError:
    pass
print(f"scaffold OK {d} prompts={len(prompts)}")
PY
fi
ACTUAL="$(curl -s --max-time 5 "http://localhost:${PORT}/v1/models" 2>/dev/null | "$RQ_PY" -c 'import json,sys; d=json.load(sys.stdin); print((d.get("data") or [{}])[0].get("id",""))' 2>/dev/null || true)"
if [ -z "$ACTUAL" ]; then
  echo "FAIL: PORT=$PORT 에 서버 없음 — s0-13-load-vllm.sh 먼저 실행" >&2
  rq_stage_record 0 base fail "bench-vllm:no-server"
  exit 1
fi
s0_bench_run "http://localhost:${PORT}/v1" "$ACTUAL" "$OUTDIR" \
  || { echo "FAIL: 벤치 실행 실패" >&2; rq_stage_record 0 base fail "bench-vllm:fail"; exit 1; }
# s0-13 서버 정리 (pidfile이 있으면 해당 프로세스만, 없으면 포트 점유 그대로 둔다).
PIDF="$RQ_ROOT/logs/s0-serve-base.pid"
if [ -f "$PIDF" ] && kill -0 "$(cat "$PIDF")" 2>/dev/null; then
  kill "$(cat "$PIDF")" 2>/dev/null || true
  rq_log "stopped base server pid=$(cat "$PIDF")"
fi
rm -f "$PIDF"
rq_stage_record 0 base pass "$OUTDIR/results.json"
echo ---- result ----
rq_log "s0-22 OK results=$OUTDIR/results.json"
