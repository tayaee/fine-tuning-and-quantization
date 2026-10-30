#!/usr/bin/env bash
# s7-35-pi-tool-call-test.sh [smoke|full] [model] — 서빙 모델로 pi 툴콜 검증 + 서버 정리 (Stage 7 CAT).
# model: HF repo id | short target (기본 nvfp4-awq-transformers). slug는 repo에서 유도한다.
# 전제: s7-30-chat.sh가 서버를 띄운 상태 (PORT, 기본 8000).
# pi 고정 질문 왕복을 검증하고 logs/s7-35-<slug>.log에 기록한 뒤 서버를 내린다.
set -euo pipefail
source "$(dirname "$0")/config/common.env" "${1:-${MODE:-smoke}}"
source "$RQ_ROOT/config/s0-pi.inc"
rq_repro_echo
rq_banner "s7-35-pi-tool-call-test" "Stage 7 - pi tool-call check against the served model, then stop the server"
export UV_NO_SYNC=1; unset VIRTUAL_ENV
rq_hf_parse "${2:-nvfp4-awq-transformers}" "$MODE"
REPO="$RQ_HF_REPO"
TARGET="${RQ_HF_TARGET:-$RQ_HF_REPO}"
PORT="${PORT:-8000}"
SLUG="$(printf '%s' "$REPO" | sed -e 's;/;__;g' | tr A-Z a-z)"
LOGF="$RQ_ROOT/logs/s7-35-${SLUG}.log"
PIDF="$RQ_ROOT/logs/s7-serve.pid"
ACTUAL="$(curl -s --max-time 5 "http://localhost:${PORT}/v1/models" 2>/dev/null | "$RQ_PY" -c 'import json,sys; d=json.load(sys.stdin); print((d.get("data") or [{}])[0].get("id",""))' 2>/dev/null || true)"
if [ -z "$ACTUAL" ]; then
  { echo "[SERVE-FAIL] $(date --iso-8601=s) mode=$MODE target=$TARGET (PORT=$PORT 서버 없음 — s7-30-chat.sh 먼저 실행)"; } > "$LOGF"
  rq_stage_record 7 cat-vllm fail "serving-pi:$TARGET"
  exit 1
fi
{
  echo "[model] $ACTUAL $(date --iso-8601=s) mode=$MODE target=$TARGET port=$PORT"
  s0_pi_register "raon-s7-$PORT" "http://localhost:${PORT}/v1" "$ACTUAL" "EMPTY"
  PROBE="$(s0_pi_probe_dir)"
  s0_pi_verify "raon-s7-$PORT" "$ACTUAL" "$PROBE"
} > "$LOGF" 2>&1 && rc=0 || rc=$?
if [ "$rc" -eq 0 ]; then
  echo "[OK] $TARGET" >> "$LOGF"; tail -n 2 "$LOGF"
  rq_stage_record 7 cat-vllm pass "$LOGF"
else
  echo "[PI-FAIL] rc=$rc $(date --iso-8601=s)" >> "$LOGF"
  rq_stage_record 7 cat-vllm fail "serving-pi:$TARGET"
fi
if [ -f "$PIDF" ] && kill -0 "$(cat "$PIDF")" 2>/dev/null; then
  kill "$(cat "$PIDF")" 2>/dev/null || true
  rq_log "stopped serving pid=$(cat "$PIDF")"
fi
rm -f "$PIDF"
[ "$rc" -eq 0 ]
