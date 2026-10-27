#!/usr/bin/env bash
# s0-13-load-vllm.sh [smoke|full] — 베이스모델을 vLLM 포크로 서빙해 로딩 검증 (Stage 0 절차 5).
# BASE_MODEL(Hub MXFP8 릴리스)은 포크의 mxfp8 프로파일 지원 대상이다.
# 서버는 백그라운드로 남겨 s0-16/s0-19가 쓰고, s0-22가 내린다.
# VLLM_MAX_LEN(기본 32768, 카드값 262144로 오버라이드 가능) · PORT(기본 8000).
set -euo pipefail
source "$(dirname "$0")/config/common.env" "${1:-${MODE:-smoke}}"
rq_repro_echo
rq_banner "s0-13-load-vllm" "Stage 0 - serve the base model with the vLLM fork (load check)"
PORT="${PORT:-8000}"
SERVLOG="$RQ_ROOT/logs/s0-13-serve-base.log"
PIDF="$RQ_ROOT/logs/s0-serve-base.pid"
REPO="tayaee/${BASE_MODEL##*/}-base"
TIMEOUT="${S0_LOAD_TIMEOUT:-1200}"
# 포크 지원 매트릭스: B300(sm_103) + RTX 5090(sm_120, NVFP4). 그 외 GPU(예: GB10 sm_121)는
# 비호환 가능 — 경고만 하고 시도는 이어간다(실패 원인은 $SERVLOG에 보존).
_CC="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d ' .')"
case "$_CC" in
  10.*|12.0) ;;
  *) rq_log "WARN: compute_cap=${_CC:-unknown} — 포크 검증 매트릭스(sm_103/sm_120) 밖. 실패 가능." ;;
esac
if ss -ltn 2>/dev/null | grep -q ":${PORT} "; then
  rq_log "port $PORT 사용 중 — 기존 서버를 s0-16/s0-19에 재사용 (pidfile=$PIDF)"
else
  rm -f "$SERVLOG"
  # GB10(SM121) 등 비검증 GPU에서만 adaptive vsplit을 끈다(stocks FlashKDA 폴백).
  # 포크가 _set_default로만 채우므로 선설정이 이긴다. 검증 GPU(sm_103/sm_120)는
  # production 설정을 그대로 쓴다.
  case "$_CC" in
    10.*|12.0) ;;
    *) export RAON_KDA_PREFILL_VSPLIT=0 ;;
  esac
  VLLM_REF="${VLLM_REF:-raon-1-mini-preview}"
  VLLM_USE_PRECOMPILED=1 nohup uvx --from "vllm @ git+https://github.com/krafton-ai/vllm.git@${VLLM_REF}" \
    vllm serve "$BASE_MODEL" \
    --host 127.0.0.1 --port "$PORT" \
    --served-model-name "$REPO" \
    --tensor-parallel-size "$TP" \
    --max-model-len "${VLLM_MAX_LEN:-32768}" \
    --max-num-batched-tokens 16384 \
    --max-num-seqs 128 \
    --gpu-memory-utilization "${GPU_UTIL:-0.85}" \
    --kv-cache-dtype fp8_e4m3 \
    --enable-prefix-caching \
    --reasoning-parser raon_1 \
    --tool-call-parser raon_1 \
    --enable-auto-tool-choice \
    --trust-remote-code \
    --default-chat-template-kwargs '{"enable_thinking": true, "truncate_history_thinking": false}' \
    >"$SERVLOG" 2>&1 &
  echo $! >"$PIDF"
  rq_log "serve start pid=$(cat "$PIDF") log=$SERVLOG (base=$BASE_MODEL)"
fi
elapsed=0 slug="" died=0
while [ "$elapsed" -lt "$TIMEOUT" ]; do
  slug="$(PORT="$PORT" "$RQ_ROOT/model-slug.sh" 2>/dev/null | tr -d '[:space:]')"
  if [ -n "$slug" ] && [ "$slug" != "null" ]; then echo "Model loaded: $slug (${elapsed}s)."; break; fi
  # 엔진 사망은 로그에 즉시 남는다 — 전체 타임아웃까지 기다리지 않고 조기 중단한다.
  if grep -qE "Engine core initialization failed|Traceback \(most recent" "$SERVLOG" 2>/dev/null; then
    echo "serve died early (see $SERVLOG)"
    grep -E "Error|error" "$SERVLOG" 2>/dev/null | tail -n 3 || true
    slug=""; died=1; break
  fi
  sleep 5; elapsed=$((elapsed + 5))
done
if [ -z "$slug" ] || [ "$slug" = "null" ]; then
  if [ "$died" = 1 ]; then
    echo "FAIL: 베이스 서빙 기동 실패 (상세: $SERVLOG)" >&2
  else
    echo "FAIL: 베이스 서빙 로딩 타임아웃 (${TIMEOUT}s, 상세: $SERVLOG)" >&2
  fi
  if [ "$died" = 1 ]; then rq_stage_record 0 base fail "vllm-base-load:failed";
  else rq_stage_record 0 base fail "vllm-base-load:timeout"; fi
  exit 1
fi
rq_stage_record 0 base pass "vllm-base-loaded:$slug"
echo ---- result ----
rq_log "s0-13 OK slug=$slug (server kept for s0-16/s0-19, stop with s0-22)"
