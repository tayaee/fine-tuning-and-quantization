#!/usr/bin/env bash
# s0-19-picall-vllm.sh [smoke|full] — 서빙 베이스모델로 pi 툴콜 검증 (Stage 0 절차 7).
# s0-13 서버(localhost:PORT)를 ~/.pi/agent/models.json에 등록하고 고정 질문으로
# 도구 호출 왕복을 확인한다. 대조군(s0-07)이 SKIP이면 본 판정도 참고용으로 기록된다.
set -euo pipefail
source "$(dirname "$0")/config/common.env" "${1:-${MODE:-smoke}}"
source "$RQ_ROOT/config/s0-pi.inc"
rq_repro_echo
rq_banner "s0-19-picall-vllm" "Stage 0 - pi tool-call check against the vLLM-served base model"
PORT="${PORT:-8000}"
ACTUAL="$(curl -s --max-time 5 "http://localhost:${PORT}/v1/models" 2>/dev/null | "$RQ_PY" -c 'import json,sys; d=json.load(sys.stdin); print((d.get("data") or [{}])[0].get("id",""))' 2>/dev/null || true)"
if [ -z "$ACTUAL" ]; then
  echo "FAIL: PORT=$PORT 에 서버 없음 — s0-13-load-vllm.sh 먼저 실행" >&2
  rq_stage_record 0 base fail "pi-vllm:no-server"
  exit 1
fi
s0_pi_register "raon-base-8000" "http://localhost:${PORT}/v1" "$ACTUAL" "EMPTY"
PROBE="$(s0_pi_probe_dir)"
if s0_pi_ask "raon-base-8000" "$ACTUAL" "$PROBE"; then
  rq_stage_record 0 base pass "pi-vllm:ok"
else
  rq_stage_record 0 base fail "pi-vllm:fail"
  exit 1
fi
