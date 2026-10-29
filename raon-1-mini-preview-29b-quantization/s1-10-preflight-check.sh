#!/usr/bin/env bash
# s1-10-preflight-check.sh — 양자화 전제조건 점검. 32~36 실행 전 필수.
set -euo pipefail
source "$(dirname "$0")/config/common.env" "${1:-${MODE:-smoke}}"
rq_repro_echo
rq_banner "s1-10-preflight-check" "Stage 1 - pre-flight compatibility check (custom arch load, expert targets, SM>=10, vLLM/transformers versions)"
export UV_NO_SYNC=1; unset VIRTUAL_ENV

# 동시 build(8종 wrapper 병렬 실행)가 같은 probe를 중복 실행하지 않게 한다.
# probe는 읽기 전용이라 무해하지만 119초씩 들고, stamp 경합(모두 stamp 확인→
# 모두 재실행)을 막으려면 lock이 필요하다.
rq_lock "$RQ_OUTPUTS/compat"

# 이미 검증한 환경이면 skip. 판정 기준: probe 코드 해시 + GPU compute_cap +
# llmcompressor/vllm 버전 (하나라도 바뀌면 다시 검증한다).
STAMP="$RQ_OUTPUTS/compat.stamp"
LOG="$RQ_OUTPUTS/compat.log"
_probe_sha="$(sha256sum "$RQ_ROOT/s1-10-preflight-check.py" 2>/dev/null | awk '{print $1}')"
_cc="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | tr '\n' ',' | sed 's/,$//')"
_llm="$(uv pip show llmcompressor 2>/dev/null | awk '/^Version:/{print $2}')"
_vllm="$(uv pip show vllm 2>/dev/null | awk '/^Version:/{print $2}')"
if [ "${FORCE:-0}" != 1 ] \
  && rq_stamp_ok "$STAMP" probe "$_probe_sha" \
  && rq_stamp_ok "$STAMP" cc "$_cc" \
  && rq_stamp_ok "$STAMP" llmcompressor "$_llm" \
  && rq_stamp_ok "$STAMP" vllm "$_vllm"; then
  rq_log "skip: compat already verified (cc=$_cc llmcompressor=$_llm vllm=$_vllm)"
  echo ---- result ----
  (set -x; cat "$LOG")
  exit 0
fi

(set -x; uv run "$RQ_ROOT/s1-10-preflight-check.py") | tee "$LOG"
rq_stamp_write "$STAMP" probe "$_probe_sha" cc "$_cc" llmcompressor "$_llm" vllm "$_vllm"

echo ---- result ----
(set -x; tail -n 20 "$LOG")
