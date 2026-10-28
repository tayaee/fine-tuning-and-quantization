#!/usr/bin/env bash
# s6-16-pi-tool-call-test.sh [mode] [model] — 고객 transformers 로딩본으로 pi 툴콜 검증 (Stage 6).
# model: HF repo id | short target | all (기본 nvfp4-awq-transformers).
# 모델마다 transformers 기반 최소 OpenAI 서버(:8001)를 띄워 pi 고정 질문을 던지고 내린다.
# 서버는 tools 왕복(chat→tool_calls→tool 결과→최종)을 지원한다. 산출물 변경 없음.
set -euo pipefail
source "$(dirname "$0")/config/common.env" "${1:-${MODE:-smoke}}"
source "$RQ_ROOT/config/s6-customer.inc"
source "$RQ_ROOT/config/s0-pi.inc"
rq_repro_echo
rq_banner "s6-16-pi-tool-call-test" "Stage 6 - pi tool-call check against customer-loaded models (temp server)"
export UV_NO_SYNC=1; unset VIRTUAL_ENV
rq_hf_parse "${2:-nvfp4-awq-transformers}" "$MODE"
PORT="${PORT:-8001}"
FAIL=0
for _e in $RQ_HF_TARGETS_RUN; do
  case "$_e" in
    */*) REPO="$_e"; TAG="$(printf '%s' "$_e" | tr '/:' '__')" ;;
    *) REPO="$(s6_repo "$_e" "$MODE")"; TAG="$_e" ;;
  esac
  t="$TAG"
  LOGF="$RQ_ROOT/logs/s6-16-$t.log"
  _tgt="$(rq_target_of "$t")"
  if [ -n "$_tgt" ] && rq_skip_transformers "$_tgt"; then
    echo "[SKIP] mode=$MODE target=$t — transformers 적재 불가(vLLM 전용)" | tee "$LOGF"
    rq_stage_record 6 cat-tf pass "customer-skip:$t"
    continue
  fi
  SERVLOG="$RQ_ROOT/logs/s6-16-server-$t.log"
  OUTDIR="$RQ_OUTPUTS/s6-$MODE"
  mkdir -p "$OUTDIR"
  echo "=========================================="
  echo "Model: mode=$MODE target=$t (repo $REPO, port $PORT)"
  echo "=========================================="
  if ! LOCAL="$(s6_download "$REPO")"; then
    { echo "[SERVE-FAIL] $(date --iso-8601=s) mode=$MODE target=$t repo=$REPO (download 실패)"; } > "$LOGF"
    rq_stage_record 6 cat-tf fail "customer-pi:$t"
    FAIL=1; continue
  fi
  rm -f "$SERVLOG"
  nohup uv run python "$RQ_ROOT/config/transformers-server.py" --model-dir "$LOCAL" --model-id "$REPO" --port "$PORT" >"$SERVLOG" 2>&1 &
  SRVPID=$!
  rq_log "server start pid=$SRVPID (target=$t)"
  ok=0
  for _i in $(seq 1 120); do
    if curl -s --max-time 3 "http://localhost:${PORT}/v1/models" 2>/dev/null | grep -q '"id"'; then ok=1; break; fi
    kill -0 "$SRVPID" 2>/dev/null || break
    sleep 5
  done
  (
    echo "[model] $REPO $(date --iso-8601=s) mode=$MODE target=$t port=$PORT"
    if [ "$ok" = 1 ]; then
      s0_pi_register "raon-cust-$PORT" "http://localhost:${PORT}/v1" "$REPO" "EMPTY"
      PROBE="$(s0_pi_probe_dir)"
      EXPECT="$(s0_pi_expect "$PROBE")"
      (cd "$PROBE" && pi --provider "raon-cust-$PORT" --model "$REPO" --no-session \
        -p "$S0_PI_QUESTION" >"$OUTDIR/pi-answer-$t.txt" 2>&1) || rc=$?
      cat "$OUTDIR/pi-answer-$t.txt"
      if [ "${rc:-0}" -eq 0 ] && grep -q "TOOL-CALL" "$SERVLOG" 2>/dev/null \
        && grep -Eq "(^|[^0-9])$EXPECT([^0-9]|$)" "$OUTDIR/pi-answer-$t.txt"; then
        echo "pi tool-call OK (tool invoked, answer contains $EXPECT)"
        rc=0
      else
        echo "pi tool-call NOT verified (need: server TOOL-CALL + standalone $EXPECT in answer)"
        rc=1
      fi
    else
      echo "server did not come up (see $SERVLOG)"
      rc=1
    fi
    exit "${rc:-0}"
  ) > "$LOGF" 2>&1; rc=$?
  if [ "$rc" -eq 0 ]; then
    echo "[OK] $t" >> "$LOGF"; rq_stage_record 6 cat-tf pass "$LOGF"
  else
    echo "[PI-FAIL] rc=$rc $(date --iso-8601=s)" >> "$LOGF"
    rq_stage_record 6 cat-tf fail "customer-pi:$t"; FAIL=1
  fi
  tail -n 2 "$LOGF"
  kill "$SRVPID" 2>/dev/null || true
  wait "$SRVPID" 2>/dev/null || true
  rq_log "server stopped (target=$t)"
done
[ "$FAIL" -eq 0 ]
