#!/usr/bin/env bash
# s0-01-serve-base-from-hf-with-transformers.sh [smoke|full] — 베이스모델을 transformers로 서브 (Stage 0 절차 1).
# BASE_MODEL(HF gated)을 transformers 직결 OpenAI 서버로 띄워 endpoint를 연다.
# s0-04/07/10이 이 endpoint로 chat·pi·벤치를 돌린다. 서버는 상주, s0-10이 내린다.
# PORT(기본 8002).
set -euo pipefail
source "$(dirname "$0")/config/common.env" "${1:-${MODE:-smoke}}"
rq_repro_echo
rq_banner "s0-01-serve-base-from-hf-with-transformers" "Stage 0 - serve the base model with transformers (endpoint for chat/pi/bench)"
export UV_NO_SYNC=1; unset VIRTUAL_ENV
RQ_BASE="${BASE_MODEL:?BASE_MODEL 미설정}"
PORT="${PORT:-8002}"
SERVLOG="$RQ_ROOT/logs/s0-01-serve-base.log"
PIDF="$RQ_ROOT/logs/s0-transformers.pid"
TIMEOUT="${S0_LOAD_TIMEOUT:-1200}"
if ss -ltn 2>/dev/null | grep -q ":${PORT} "; then
  rq_log "port $PORT 사용 중 — 기존 서버를 s0-04/07/10에 재사용 (pidfile=$PIDF)"
else
  rm -f "$SERVLOG"
  nohup uv run python "$RQ_ROOT/config/transformers-server.py" \
    --model-dir "$RQ_BASE" --model-id "$RQ_BASE" --port "$PORT" \
    >"$SERVLOG" 2>&1 &
  echo $! >"$PIDF"
  rq_log "serve start pid=$(cat "$PIDF") log=$SERVLOG (base=$RQ_BASE)"
fi
elapsed=0
while [ "$elapsed" -lt "$TIMEOUT" ]; do
  if curl -s --max-time 3 "http://localhost:${PORT}/v1/models" 2>/dev/null | grep -q '"id"'; then
    echo "Endpoint ready: http://localhost:${PORT}/v1 (${elapsed}s)."
    break
  fi
  if grep -qE "Traceback \(most recent|Error" "$SERVLOG" 2>/dev/null; then
    echo "serve died early (see $SERVLOG)"
    tail -n 5 "$SERVLOG"
    rq_stage_record 0 base fail "transformers-serve:failed"
    exit 1
  fi
  sleep 5; elapsed=$((elapsed + 5))
done
curl -s --max-time 3 "http://localhost:${PORT}/v1/models" 2>/dev/null | grep -q '"id"' \
  || { echo "FAIL: endpoint 미기동 (${TIMEOUT}s, 상세: $SERVLOG)" >&2;
       rq_stage_record 0 base fail "transformers-serve:timeout"; exit 1; }
rq_stage_record 0 base pass "transformers-served:$RQ_BASE"
echo ---- result ----
rq_log "s0-01 OK endpoint=http://localhost:${PORT}/v1 (server kept for s0-04/07/10, pidfile=$PIDF)"
