#!/usr/bin/env bash
# s8-65-serve-vllm-production.sh [smoke|full] [target] — GB10/DGX Spark용 풀 컨텍스트 서빙.
# README 모델카드의 "Production serving" 명령과 동일 (벤더 B300 프로파일을 128GB에 맞게).
# target: base|nvfp4-awq|nvfp4-gptq|awq-w5|awq-w3 (기본 nvfp4-awq).
# SOURCE=hf|local (기본 hf — 게시된 Hub repo 서빙; local이면 로컬 체크포인트).
# PORT/GPU_UTIL/VLLM_REF 오버라이드 가능 (바쁜 노드면 GPU_UTIL=0.3).
# 예: ./s8-65-serve-vllm-production.sh full nvfp4-awq
#     SOURCE=local GPU_UTIL=0.3 ./s8-65-serve-vllm-production.sh full nvfp4-awq
set -euo pipefail
source "$(dirname "$0")/config/common.env" "${1:-${MODE:-smoke}}"
TARGET="$(rq_require_targets "./s8-65-serve-vllm-production.sh $MODE" "${2:-}" base awq-w3 awq-w5 nvfp4-awq nvfp4-gptq)" || exit 2
SOURCE="${SOURCE:-hf}"
PORT="${PORT:-8000}"
VLLM_REF="${VLLM_REF:-raon-1-mini-preview}"
rq_repro_echo
rq_banner "s8-65-serve-vllm-production" "Stage 8/9 - serve at full 262K context on GB10 (production flags)"

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

rq_log "serving model=$MODEL quant=${QUANT:-none} tp=1 port=$PORT fork=krafton-ai/vllm@$VLLM_REF"
export VLLM_USE_PRECOMPILED=1
set -x
exec uvx --from "vllm @ git+https://github.com/krafton-ai/vllm.git@${VLLM_REF}" vllm serve "$MODEL" \
  --host 127.0.0.1 --port "$PORT" \
  --served-model-name "$REPO" \
  --tensor-parallel-size 1 \
  ${QUANT:+--quantization "$QUANT"} \
  --max-model-len 262144 \
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
