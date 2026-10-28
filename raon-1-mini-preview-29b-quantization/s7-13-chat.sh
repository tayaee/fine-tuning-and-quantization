#!/usr/bin/env bash
# s7-13-chat.sh [smoke|full] [model] — vLLM 서빙 + chat 검증 (Stage 7 CAT).
# model: HF repo id ("tayaee/...-nvfp4-awq") | short target (기본 nvfp4-awq).
# SOURCE=local|hf (기본 local; hf면 Hub repo로 서빙). PORT(기본 8000).
# frozen 래퍼(s7-<num>-test-<target>-with-vllm-fork.sh)를 백그라운드로 띄워 chat 고정 질문을 검증하고
# 서버는 남긴다 (s7-16-pi-tool-call-test.sh가 재사용, 마지막에 내림).
# logs/s7-13-<slug>.log + 원시 서빙 로그 logs/s7-13-serve-<slug>.log.
set -euo pipefail
source "$(dirname "$0")/config/common.env" "${1:-${MODE:-smoke}}"
rq_repro_echo
rq_banner "s7-13-chat" "Stage 7 - serve with vLLM and verify chat (customer acceptance)"
export UV_NO_SYNC=1; unset VIRTUAL_ENV
rq_hf_parse "${2:-nvfp4-awq}" "$MODE"
REPO="$RQ_HF_REPO"
TARGET="$RQ_HF_TARGET"
SOURCE="${SOURCE:-local}"
PORT="${PORT:-8000}"
if [ -z "$TARGET" ]; then
  # 미등록 HF id: 로컬 SUF 매핑 불가 → SOURCE=hf로 직접 서빙만 된다.
  if [ "$SOURCE" != hf ]; then
    echo "FAIL: 알 수 없는 타깃 ($REPO) — SOURCE=hf로 서빙하라" >&2
    exit 2
  fi
  SUF=""; NUM=""; QUANT="compressed-tensors"; VTP_DEF=1
else
  case "$TARGET" in
    awq-w3)      SUF="raon-awq-w3-$MODE"; QUANT="compressed-tensors"; VTP_DEF=1; NUM=01 ;;
    awq-w5)      SUF="raon-awq-w5-$MODE"; QUANT="compressed-tensors"; VTP_DEF=1; NUM=04 ;;
    nvfp4-awq)   SUF="raon-nvfp4-awq-$MODE"; QUANT="compressed-tensors"; VTP_DEF=1; NUM=07 ;;
    nvfp4-gptq)  SUF="raon-nvfp4-gptq-$MODE"; QUANT="compressed-tensors"; VTP_DEF=1; NUM=10 ;;
    nvfp4-awq-modelopt) SUF="raon-nvfp4-awq-modelopt-$MODE"; QUANT=""; VTP_DEF=1; NUM=06 ;;
    *) echo "target: awq-w3|awq-w5|nvfp4-awq|nvfp4-gptq|nvfp4-awq-modelopt" >&2; exit 2 ;;
  esac
fi
if [ "$SOURCE" = "hf" ]; then MODEL="$REPO"; else MODEL="$RQ_MODELS/$SUF"; fi
SLUG="$(printf '%s' "$REPO" | sed -e 's;/;__;g' | tr A-Z a-z)"
LOGF="$RQ_ROOT/logs/s7-13-${SLUG}.log"
SERVLOG="$RQ_ROOT/logs/s7-13-serve-${SLUG}.log"
PIDF="$RQ_ROOT/logs/s7-serve.pid"
TIMEOUT="${S7_LOAD_TIMEOUT:-1200}"
# Preflight (2026-10-26 진단): 핀된 포크는 ModelOpt가 아닌 체크포인트의
# quantization_config를 삭제한다 → llmcompressor PTQ 4종 서빙 불가.
# 미리 알리고 서빙 로그를 남긴다.
if [ -f "$RQ_MODELS/$SUF/config.json" ] \
  && grep -q '"quant_method": "compressed-tensors"' "$RQ_MODELS/$SUF/config.json" 2>/dev/null; then
  echo "[warn] $MODE $TARGET — compressed-tensors PTQ 체크포인트는 핀된 포크 서빙과"
  echo "       비호환 가능 (MXFP8-release/ModelOpt 프로파일만 지원, 2026-10-26 확인)."
  echo "       서빙 시도 후 실패 원인은 $SERVLOG 에 보존된다."
fi
if ss -ltn 2>/dev/null | grep -q ":${PORT} "; then
  rq_log "port $PORT 사용 중 — 기존 서버를 재사용 (pidfile=$PIDF)"
else
  rm -f "$SERVLOG"
  WRAPPER="$RQ_ROOT/s7-$NUM-test-${TARGET}-with-vllm-fork.sh"
  if [ "$SOURCE" = "hf" ] && [ -f "$WRAPPER" ] \
    && [ "${PORT:-8000}" = 8000 ]; then
    rq_log "frozen card command (background): $WRAPPER"
    nohup "$WRAPPER" >"$SERVLOG" 2>&1 &
  else
    VLLM_REF="${VLLM_REF:-raon-1-mini-preview}"
    # ModelOpt NVFP4는 --quantization 없이 자동 감지 (빈 QUANT면 플래그 생략)
    QUANT_FLAG=(); [ -n "$QUANT" ] && QUANT_FLAG=(--quantization "$QUANT")
    VLLM_USE_PRECOMPILED=1 nohup uvx --from "vllm @ git+https://github.com/krafton-ai/vllm.git@${VLLM_REF}" \
      vllm serve "$MODEL" \
      --host 127.0.0.1 --port "$PORT" \
      --served-model-name "$REPO" \
      --tensor-parallel-size "${VTP:-$VTP_DEF}" \
      "${QUANT_FLAG[@]}" \
      --max-model-len "${VLLM_MAX_LEN:-262144}" \
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
  fi
  echo $! >"$PIDF"
  rq_log "serve start pid=$(cat "$PIDF") log=$SERVLOG"
fi
elapsed=0 slug="" died=0
while [ "$elapsed" -lt "$TIMEOUT" ]; do
  slug="$(PORT="$PORT" "$RQ_ROOT/model-slug.sh" 2>/dev/null | tr -d '[:space:]')"
  if [ -n "$slug" ] && [ "$slug" != "null" ]; then echo "Model loaded: $slug (${elapsed}s)."; break; fi
  if grep -qE "Engine core initialization failed|Traceback \(most recent" "$SERVLOG" 2>/dev/null; then
    echo "serve died early (see $SERVLOG)"; slug=""; died=1; break
  fi
  sleep 5; elapsed=$((elapsed + 5))
done
{
  echo "[model] $slug $(date --iso-8601=s) mode=$MODE target=$TARGET port=$PORT"
  if [ -z "$slug" ] || [ "$slug" = "null" ]; then
    if [ "$died" = 1 ]; then echo "[SERVE-FAIL] server died (see $SERVLOG)"
    else echo "[SERVE-FAIL] loading timeout ${TIMEOUT}s"; fi
    exit 1
  fi
  uv run python - "$PORT" <<'PY'
import sys
from openai import OpenAI
port = sys.argv[1]
client = OpenAI(base_url=f"http://localhost:{port}/v1", api_key="dummy")
model_id = client.models.list().data[0].id
r = client.chat.completions.create(
    model=model_id,
    messages=[{"role": "user", "content": "사과 3개가 있고 배는 사과보다 2개 많습니다. 배는 몇 개입니까?"}],
    max_tokens=256, temperature=0.6, top_p=0.95)
text = (r.choices[0].message.content or "")
print("CHAT", "OK" if text.strip() else "EMPTY", ":", text[:200])
if not text.strip():
    raise SystemExit("FAIL: 빈 응답")
PY
} > "$LOGF" 2>&1 && rc=0 || rc=$?
if [ "$rc" -eq 0 ]; then
  echo "[OK] $slug" >> "$LOGF"; tail -n 2 "$LOGF"
  rq_stage_record 7 cat-vllm pass "$LOGF"
else
  if ! grep -q "SERVE-FAIL" "$LOGF" 2>/dev/null; then
    echo "[CHAT-FAIL] rc=$rc $(date --iso-8601=s)" >> "$LOGF"
  fi
  rq_stage_record 7 cat-vllm fail "serving-chat:$TARGET"
  exit 1
fi
echo ---- result ----
rq_log "s7-13 OK slug=$slug (server kept for s7-16-pi-tool-call-test, pidfile=$PIDF)"
