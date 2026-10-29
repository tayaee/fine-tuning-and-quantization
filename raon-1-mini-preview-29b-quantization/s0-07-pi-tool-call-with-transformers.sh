#!/usr/bin/env bash
# s0-07-pi-tool-call-with-transformers.sh [smoke|full] — transformers endpoint로 pi 툴콜 검증 (Stage 0 절차 3).
# 전제: s0-01-serve-base-from-hf-with-transformers.sh가 서버를 띄운 상태. PORT(기본 8002).
# pi 고정 질문 왕복을 검증한다. 판정 = 서버 TOOL-CALL 증거 + 답에 기대 수 포함.
set -euo pipefail
source "$(dirname "$0")/config/common.env" "${1:-${MODE:-smoke}}"
source "$RQ_ROOT/config/s0-pi.inc"
rq_repro_echo
rq_banner "s0-07-pi-tool-call-with-transformers" "Stage 0 - pi tool-call check against the transformers endpoint"
PORT="${PORT:-8002}"
ACTUAL="$(curl -s --max-time 5 "http://localhost:${PORT}/v1/models" 2>/dev/null | "$RQ_PY" -c 'import json,sys; d=json.load(sys.stdin); print((d.get("data") or [{}])[0].get("id",""))' 2>/dev/null || true)"
if [ -z "$ACTUAL" ]; then
  echo "FAIL: PORT=$PORT 에 서버 없음 — s0-01-serve-base-from-hf-with-transformers.sh 먼저 실행" >&2
  rq_stage_record 0 base fail "pi-transformers:no-server"
  exit 1
fi
s0_pi_register "raon-s0tf-$PORT" "http://localhost:${PORT}/v1" "$ACTUAL" "EMPTY"
PROBE="$(s0_pi_probe_dir)"
SERVLOG="$RQ_ROOT/logs/s0-01-serve-base.log"
if s0_pi_verify "raon-s0tf-$PORT" "$ACTUAL" "$PROBE" "$SERVLOG"; then
  rq_stage_record 0 base pass "pi-transformers:ok"
else
  rq_stage_record 0 base fail "pi-transformers:fail"
  exit 1
fi
