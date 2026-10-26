#!/usr/bin/env bash
# s8-64-serve-vllm.sh [smoke|full] [target] — 전 타깃 커버 서빙 디스패처 (개별 래퍼 s8-52/55/58/61에 위임).
# README 모델카드의 vLLM 명령(production)을 그대로 실행.
# 인자 없이 실행하면 복사-붙여넣기용 실행 명령 목록 출력 (서빙 안 함).
# target: base|nvfp4-awq|nvfp4-gptq|awq-w5|awq-w3 (기본 nvfp4-gptq).
# SOURCE=local|hf (기본 local; hf면 tayaee/* repo로 서빙 → README 명령과 문자열 동일).
# VTP/PORT/VLLM_REF/VLLM_MAX_LEN/GPU_UTIL로 오버라이드. VLLM_USE_PRECOMPILED=1 고정.
# 예: ./s8-64-serve-vllm.sh                    # 목록만 출력
#     ./s8-64-serve-vllm.sh smoke nvfp4-gptq
#     SOURCE=hf ./s8-64-serve-vllm.sh full nvfp4-awq   # README 명령 as-is 검증
set -euo pipefail
if [ $# -eq 0 ]; then
  source "$(dirname "$0")/config/common.env" "${MODE:-smoke}"
  echo "[TP=$TP] copy & paste (terminal1):"
  echo "  README 카드와 동일한 독립 스크립트 ( Hub repo 직접 서빙 ):"
  for spec in "52:awq-w3" "55:awq-w5" "58:nvfp4-awq" "61:nvfp4-gptq"; do
    printf '  %-58s\n' "./s8-${spec%%:*}-vllm-serve-Raon-1-mini-preview-${spec#*:}.sh"
  done
  echo "  파라미터형 (local/hf, smoke/full, base 포함):"
  for m in full smoke; do
    for t in base awq-w3 awq-w5 nvfp4-awq nvfp4-gptq; do
      case "$t" in base) _suf="raon-bf16" ;; *) _suf="raon-$t-$m" ;; esac
      printf '%-38s # %s\n' "./s8-64-serve-vllm.sh $m $t" "$RQ_MODELS/$_suf"
    done
  done
  exit 0
fi
source "$(dirname "$0")/config/common.env" "${1:-${MODE:-smoke}}"
TARGET="${2:-nvfp4-gptq}"
SOURCE="${SOURCE:-local}"
PORT="${PORT:-8000}"
rq_repro_echo
rq_banner "s8-64-serve-vllm" "Stage 8/9 - serve the model with vLLM (OpenAI-compatible API)"

case "$TARGET" in
  base)        SUF="raon-bf16"; QUANT=""; VTP_DEF="$TP"; NUM="" ;;
  awq-w3)      SUF="raon-awq-w3-$MODE"; QUANT="compressed-tensors"; VTP_DEF=1; NUM=52 ;;
  awq-w5)      SUF="raon-awq-w5-$MODE"; QUANT="compressed-tensors"; VTP_DEF=1; NUM=55 ;;
  nvfp4-awq)   SUF="raon-nvfp4-awq-$MODE"; QUANT="compressed-tensors"; VTP_DEF=1; NUM=58 ;;
  nvfp4-gptq)  SUF="raon-nvfp4-gptq-$MODE"; QUANT="compressed-tensors"; VTP_DEF=1; NUM=61 ;;
  *) echo "target: base|nvfp4-awq|nvfp4-gptq|awq-w5|awq-w3" >&2; exit 1 ;;
esac
# 양자화 4종은 s7-56 README 카드와 동일 스펙(TP=1, compressed-tensors).
# base는 Hub 카드가 없으므로 infra TP를 따른다. VTP로 오버라이드 가능.
REPO="tayaee/${BASE_MODEL##*/}-${TARGET}"
[ "$MODE" = smoke ] && REPO="$REPO-smoke"
if [ "$SOURCE" = "hf" ]; then MODEL="$REPO"; else MODEL="$RQ_MODELS/$SUF"; fi

# 카드 명령과 완전히 일치하는 호출(full+hf+양자화4종+기본값)은 frozen 래퍼에 위임한다.
# 오버라이드(PORT/VTP/VLLM_REF/VLLM_MAX_LEN/GPU_UTIL)가 하나라도 있으면 아래 파라미터형 경로로 간다.
WRAPPER="$RQ_ROOT/s8-$NUM-vllm-serve-${BASE_MODEL##*/}-${TARGET}.sh"
if [ "$SOURCE" = hf ] && [ "$MODE" = full ] && [ -f "$WRAPPER" ] \
  && [ "${PORT:-8000}" = 8000 ] && [ "${VTP:-$VTP_DEF}" = "$VTP_DEF" ] \
  && [ "${VLLM_REF:-raon-1-mini-preview}" = raon-1-mini-preview ] \
  && [ "${VLLM_MAX_LEN:-262144}" = 262144 ] && [ "${GPU_UTIL:-0.85}" = 0.85 ]; then
  rq_log "delegating to frozen card command: $WRAPPER"
  set -x
  exec "$WRAPPER"
fi
VTP="${VTP:-$VTP_DEF}"
VLLM_REF="${VLLM_REF:-raon-1-mini-preview}"

rq_log "serving model=$MODEL quant=${QUANT:-none} tp=$VTP port=$PORT fork=krafton-ai/vllm@$VLLM_REF"
export VLLM_USE_PRECOMPILED=1
set -x
exec uvx --from "vllm @ git+https://github.com/krafton-ai/vllm.git@${VLLM_REF}" vllm serve "$MODEL" \
  --host 127.0.0.1 --port "$PORT" \
  --served-model-name "$REPO" \
  --tensor-parallel-size "$VTP" \
  ${QUANT:+--quantization "$QUANT"} \
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
  --default-chat-template-kwargs '{"enable_thinking": true, "truncate_history_thinking": false}'
