#!/usr/bin/env bash
# s8-52-serve-vllm.sh [smoke|full] [target] — vLLM OpenAI-호환 서빙 (TP=2 기본, 2x).
# 인자 없이 실행하면 복사-붙여넣기용 실행 명령 목록 출력 (서빙 안 함).
# target: base|nvfp4-awq|nvfp4-gptq|awq-w5|awq-w3 (기본 nvfp4-gptq).
# SOURCE=local|hf (기본 local; hf면 tayaee/* repo로 서빙). PORT 기본 8000.
# 예: ./s8-52-serve-vllm.sh                    # 목록만 출력
#     ./s8-52-serve-vllm.sh smoke nvfp4-gptq
#     SOURCE=hf ./s8-52-serve-vllm.sh full nvfp4-awq
set -euo pipefail
if [ $# -eq 0 ]; then
  source "$(dirname "$0")/config/common.env" smoke
  echo "[TP=$TP] copy & paste (terminal1):"
  for m in full smoke; do
    for t in base awq-w3 awq-w5 nvfp4-awq nvfp4-gptq; do
      printf '%-38s # %s\n' "./s8-52-serve-vllm.sh $m $t" "$RQ_MODELS/raon-$t-$m"
    done
  done
  exit 0
fi
source "$(dirname "$0")/config/common.env" "${1:-smoke}"
TARGET="${2:-nvfp4-gptq}"
SOURCE="${SOURCE:-local}"
PORT="${PORT:-8000}"
rq_repro_echo
rq_banner "s8-52-serve-vllm" "Stage 8/9 - serve the model with vLLM (OpenAI-compatible API)"

case "$TARGET" in
  base)        SUF="raon-bf16"; QUANT="" ;;
  awq-w5)      SUF="raon-awq-w5-$MODE"; QUANT="compressed-tensors" ;;
  awq-w3)      SUF="raon-awq-w3-$MODE"; QUANT="compressed-tensors" ;;
  nvfp4-awq)   SUF="raon-nvfp4-awq-$MODE"; QUANT="compressed-tensors" ;;
  nvfp4-gptq)  SUF="raon-nvfp4-gptq-$MODE"; QUANT="compressed-tensors" ;;
  *) echo "target: base|nvfp4-awq|nvfp4-gptq|awq-w5|awq-w3" >&2; exit 1 ;;
esac

REPO="tayaee/${BASE_MODEL##*/}-${TARGET}"
[ "$MODE" = smoke ] && REPO="$REPO-smoke"
if [ "$SOURCE" = "hf" ]; then MODEL="$REPO"; else MODEL="$RQ_MODELS/$SUF"; fi

rq_log "serving model=$MODEL quant=${QUANT:-none} tp=$TP port=$PORT"
export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-FLASH_ATTN}"
export VLLM_USE_FLASHINFER_SAMPLER=0
set -x
exec "$VLLM_BIN" serve "$MODEL" \
  --host 0.0.0.0 --port "$PORT" \
  --served-model-name "$REPO" \
  --tensor-parallel-size "$TP" \
  ${QUANT:+--quantization "$QUANT"} \
  --max-model-len "${VLLM_MAX_LEN:-8192}" \
  --gpu-memory-utilization "${GPU_UTIL:-0.9}" \
  --trust-remote-code
