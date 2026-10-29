#!/usr/bin/env bash
# s0-13-serve-base-from-hf-with-vllm-fork.sh [smoke|full] — 베이스모델을 vLLM 포크로 서빙해 로딩 검증 (Stage 0 절차 5).
# sm_103/sm_120: 포크 직접 서빙(BASE_MODEL MXFP8).
# sm_121(GB10/DGX Spark): KRAFTON 카드의 NVFP8+sm121 컨테이너 경로
# (MXFP8 직접 서빙은 커널 체인으로 불가). 그 외 GPU는 SKIP.
# 서버는 백그라운드로 남겨 s0-16/s0-19가 쓰고, s0-22가 내린다.
# VLLM_MAX_LEN(기본 32768, 카드값 262144로 오버라이드 가능) · PORT(기본 8000).
set -euo pipefail
source "$(dirname "$0")/config/common.env" "${1:-${MODE:-smoke}}"
rq_repro_echo
rq_banner "s0-13-serve-base-from-hf-with-vllm-fork" "Stage 0 - serve the base model with the vLLM fork (load check)"
PORT="${PORT:-8000}"
SERVLOG="$RQ_ROOT/logs/s0-13-serve-base.log"
PIDF="$RQ_ROOT/logs/s0-serve-base.pid"
REPO="tayaee/${BASE_MODEL##*/}-base"
TIMEOUT="${S0_LOAD_TIMEOUT:-1200}"
# 포크 검증 매트릭스: B300(sm_103) + RTX 5090(sm_120, NVFP4).
# GB10 sm_121(DGX Spark)은 KRAFTON 카드의 NVFP4+sm121 컨테이너 경로로 검증한다
# (MXFP8 직접 서빙은 체인으로 불가: vsplit/MLA/prefill/MoE tcgen05+MODE=4 강제).
# nvidia-smi compute_cap의 점/공백을 지운 값으로 비교한다 ("12.1" → "121").
_CC="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d ' .')"
SM121=0
case "$_CC" in
  10*|120) ;;
  121) SM121=1 ;;
  *)
    {
      echo "SKIP: compute_cap=${_CC:-unknown} — 포크 검증 매트릭스(sm_103/sm_120/sm_121) 밖."
      echo "transformers half(s0-01/04/07/10)가 베이스 검증 증거다."
    } | tee "$SERVLOG"
    rq_stage_record 0 base skipped "vllm-half:unsupported-gpu-${_CC:-unknown}"
    rm -f "$PIDF"
    exit 0 ;;
esac
if ss -ltn 2>/dev/null | grep -q ":${PORT} "; then
  rq_log "port $PORT 사용 중 — 기존 서버를 s0-16/s0-19에 재사용 (pidfile=$PIDF)"
elif [ "$SM121" = 1 ]; then
  # GB10: KRAFTON 카드의 NVFP4+sm121 컨테이너 경로 (MXFP8 직접 서빙 불가).
  S0_NVFP4_REV="${S0_NVFP4_REV:-0b0f06ac2b3d03abea62dd221fe7839fddf36dfe}"
  S0_DRAFT_REV="${S0_DRAFT_REV:-6450d1546d17ee1bb1cb942f316c7fb91cdd868a}"
  S0_IMAGE="${S0_IMAGE:-ghcr.io/krafton-ai/raon-vllm-spark:sm121-4e09534}"
  S0_CNAME="${S0_CNAME:-s0-raon-spark}"
  S0_HF="${HF_HOME:-$RQ_SHARED/hf-cache}"
  S0_SNAP="$S0_HF/hub/models--KRAFTON--Raon-1-mini-preview-nvfp4/snapshots/$S0_NVFP4_REV"
  S0_DRAFT="$S0_HF/hub/models--KRAFTON--Raon-1-mini-preview-dflash2/snapshots/$S0_DRAFT_REV"
  command -v docker >/dev/null 2>&1 || {
    echo "SKIP: docker 없음 — sm121 컨테이너 경로 불가" | tee "$SERVLOG"
    rq_stage_record 0 base skipped "vllm-half:no-docker"; exit 0; }
  docker image inspect "$S0_IMAGE" >/dev/null 2>&1 || {
    echo "SKIP: 이미지 없음 — docker pull $S0_IMAGE 후 재실행" | tee "$SERVLOG"
    rq_stage_record 0 base skipped "vllm-half:no-image"; exit 0; }
  { [ -d "$S0_SNAP" ] && [ -d "$S0_DRAFT" ]; } || {
    echo "SKIP: 스냅샷 없음 — 아래 2개 다운로드 후 재실행" | tee "$SERVLOG"
    echo "hf download KRAFTON/Raon-1-mini-preview-nvfp4 --revision $S0_NVFP4_REV" | tee -a "$SERVLOG"
    echo "hf download KRAFTON/Raon-1-mini-preview-dflash2 --revision $S0_DRAFT_REV" | tee -a "$SERVLOG"
    rq_stage_record 0 base skipped "vllm-half:no-snapshot"; exit 0; }
  rm -f "$SERVLOG"
  docker rm -f "$S0_CNAME" >/dev/null 2>&1 || true
  {
    echo "container: $S0_IMAGE (sm121 fixes, NVFP4+$S0_NVFP4_REV draft $S0_DRAFT_REV)"
    echo "model: KRAFTON/Raon-1-mini-preview-nvfp4 (ModelOpt, Marlin)"
  } >"$SERVLOG"
  docker run -d --name "$S0_CNAME" --hostname raon-spark --gpus all --ipc=host \
    -p "127.0.0.1:${PORT}:8000" \
    -v "$S0_HF:/hf" -v raon-spark-cache:/cache \
    -e HF_HOME=/hf -e HF_HUB_OFFLINE=1 \
    -e MODEL_ID="/hf/hub/models--KRAFTON--Raon-1-mini-preview-nvfp4/snapshots/$S0_NVFP4_REV" \
    -e DRAFT_MODEL_ID="/hf/hub/models--KRAFTON--Raon-1-mini-preview-dflash2/snapshots/$S0_DRAFT_REV" \
    -e SPEC_TOKENS=7 -e MAX_MODEL_LEN=131072 \
    "$S0_IMAGE" >>"$SERVLOG" 2>&1
  echo "docker:$S0_CNAME" >"$PIDF"
  rq_log "serve start container=$S0_CNAME log=$SERVLOG (NVFP4 sm121)"
else
  rm -f "$SERVLOG"
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
_s0_cname=""
if [ -f "$PIDF" ]; then
  case "$(cat "$PIDF" 2>/dev/null)" in docker:*) _s0_cname="$(sed 's/^docker://' "$PIDF")" ;; esac
fi
while [ "$elapsed" -lt "$TIMEOUT" ]; do
  slug="$(PORT="$PORT" "$RQ_ROOT/model-slug.sh" 2>/dev/null | tr -d '[:space:]')"
  if [ -n "$slug" ] && [ "$slug" != "null" ]; then echo "Model loaded: $slug (${elapsed}s)."; break; fi
  # 엔진 사망은 로그에 즉시 남는다 — 전체 타임아웃까지 기다리지 않고 조기 중단한다.
  if grep -qE "Engine core initialization failed|Traceback \(most recent" "$SERVLOG" 2>/dev/null; then
    echo "serve died early (see $SERVLOG)"
    grep -E "Error|error" "$SERVLOG" 2>/dev/null | tail -n 3 || true
    slug=""; died=1; break
  fi
  # 컨테이너 경로: 컨테이너 사망 시 docker logs를 보존하고 중단한다.
  if [ -n "$_s0_cname" ]; then
    _st="$(docker inspect -f '{{.State.Status}}' "$_s0_cname" 2>/dev/null || echo "?")"
    if [ "$_st" != "running" ]; then
      echo "container exited (status=$_st), tail:" | tee -a "$SERVLOG"
      docker logs --tail 20 "$_s0_cname" 2>&1 | tee -a "$SERVLOG" | tail -n 5 || true
      slug=""; died=1; break
    fi
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
