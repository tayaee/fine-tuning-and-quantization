#!/usr/bin/env bash
# 60-serve-vllm.sh [mini|full] [target] (STRAT env, 기본 single) — Stage 8b. vLLM OpenAI-호환 서빙.
# 인자 없이 실행하면 복사-붙여넣기용 실행 명령 20줄을 PPL 실측순으로 보여준다.
# target: fft|fft-fp8|fft-awq|fft-gptq|qlora|qlora-fp8|qlora-awq|qlora-gptq (기본 fft, 품질순).
# (GGUF는 61-serve-gguf.sh llama-server 사용 — vLLM GGUF 미지원.)
# SOURCE=local|hf (기본 local; hf면 tayaee/* repo로 서빙). PORT 기본 8000.
# 예: ./60-serve-vllm.sh                       # 실행 명령 목록만 출력 (서빙 안 함)
#     ./60-serve-vllm.sh mini fft
#     SOURCE=hf ./60-serve-vllm.sh mini fft-gptq
#     QTYPE=q4_k_m ./61-serve-gguf.sh mini fft-gguf  # GGUF 변종 선택 (기본 q8_0)
#     62-infer-examples.py --model tayaee/Llama-3.2-1B-math-fft-gptq-mini (별도 터미널)
set -euo pipefail
if [ $# -eq 0 ]; then
  source "$(dirname "$0")/config/common.env" mini
  STRAT="${STRAT:-single}"
  # PPL 실측순 (출처: outputs/ppl-{full,mini}-single/summary.json).
  # 1차 method(transformers vs gguf — 엔진 상이, PPL 교차비교 금지),
  # 2차 fft vs qlora, 각 그룹 내 PPL 오름차순.
  echo "[STRAT=$STRAT] copy & paste (terminal1, PPL순):"
  echo "# transformers"
  for spec in \
    "full fft 4.549" "full fft-awq 4.489" "full fft-fp8 4.570" "full fft-gptq 4.604" \
    "full qlora 4.990" "full qlora-gptq 5.045" "full qlora-fp8 5.342" "full qlora-awq 5.840" \
    "mini fft 9.867" "mini fft-fp8 10.035" "mini fft-gptq 10.164" "mini fft-awq 10.635" \
    "mini qlora 9.867" "mini qlora-fp8 10.035" "mini qlora-gptq 10.164" "mini qlora-awq 10.683"; do
    read -r m t ppl <<< "$spec"
    case "$t" in
      fft)        SUF="synthetic-fft-$m-$STRAT" ;;
      qlora)      SUF="synthetic-qlora-$m-$STRAT-merged" ;;
      fft-gptq)   SUF="synthetic-fft-$m-$STRAT-gptq" ;;
      fft-awq)    SUF="synthetic-fft-$m-$STRAT-awq" ;;
      fft-fp8)    SUF="synthetic-fft-$m-$STRAT-fp8" ;;
      qlora-gptq) SUF="synthetic-qlora-$m-$STRAT-merged-gptq" ;;
      qlora-awq)  SUF="synthetic-qlora-$m-$STRAT-merged-awq" ;;
      qlora-fp8)  SUF="synthetic-qlora-$m-$STRAT-merged-fp8" ;;
    esac
    printf '%-34s # ppl %s | %s\n' "./60-serve-vllm.sh $m $t" "$ppl" "$M1B_MODELS/$SUF"
  done
  echo "# gguf → 61-serve-gguf.sh llama-server (llama.cpp, q8_0 서빙분 PPL)"
  for spec in \
    "full fft-gguf 2.780" "full qlora-gguf 2.747" \
    "mini fft-gguf 2.720" "mini qlora-gguf 2.720"; do
    read -r m t ppl <<< "$spec"
    case "$t" in
      fft-gguf)   SUF="synthetic-fft-$m-$STRAT-gguf/model-${QTYPE:-q8_0}.gguf" ;;
      qlora-gguf) SUF="synthetic-qlora-$m-$STRAT-merged-gguf/model-${QTYPE:-q8_0}.gguf" ;;
    esac
    printf '%-34s # ppl %s | %s\n' "./61-serve-gguf.sh $m $t" "$ppl" "$M1B_MODELS/$SUF"
  done
  exit 0
fi
source "$(dirname "$0")/config/common.env" "${1:-mini}"
STRAT="${STRAT:-single}"
case "$STRAT" in single|ddp|fsdp) ;; *) echo "STRAT must be single|ddp|fsdp" >&2; exit 1;; esac
TARGET="${2:-fft}"
SOURCE="${SOURCE:-local}"
PORT="${PORT:-8000}"
m1b_repro_echo

case "$TARGET" in
  fft)   SUF="synthetic-fft-$MODE-$STRAT"; QUANT="" ;;
  qlora) SUF="synthetic-qlora-$MODE-$STRAT-merged"; QUANT="" ;;
  fft-gptq|gptq)  SUF="synthetic-fft-$MODE-$STRAT-gptq"; QUANT="gptq"; TARGET="fft-gptq" ;;
  # llmcompressor 산출물(AWQ/FP8)은 compressed-tensors 포맷 (50-eval.py와 동일)
  fft-awq|awq)   SUF="synthetic-fft-$MODE-$STRAT-awq"; QUANT="compressed-tensors"; TARGET="fft-awq" ;;
  fft-fp8|fp8)   SUF="synthetic-fft-$MODE-$STRAT-fp8"; QUANT="compressed-tensors"; TARGET="fft-fp8" ;;
  qlora-gptq)  SUF="synthetic-qlora-$MODE-$STRAT-merged-gptq"; QUANT="gptq" ;;
  qlora-awq)   SUF="synthetic-qlora-$MODE-$STRAT-merged-awq"; QUANT="compressed-tensors" ;;
  qlora-fp8)   SUF="synthetic-qlora-$MODE-$STRAT-merged-fp8"; QUANT="compressed-tensors" ;;
  fft-gguf|gguf)  SUF="synthetic-fft-$MODE-$STRAT-gguf"; QUANT="gguf"; TARGET="fft-gguf" ;;
  qlora-gguf)  SUF="synthetic-qlora-$MODE-$STRAT-merged-gguf"; QUANT="gguf" ;;
  *) echo "target: fft|fft-fp8|fft-awq|fft-gptq|fft-gguf|qlora|qlora-fp8|qlora-awq|qlora-gptq|qlora-gguf" >&2; exit 1 ;;
esac

# HF repo명 규칙은 58-upload-hf.py repo_name()과 동일 (single 동결, full 무표기).
# models-tree.txt 설계 결정 3 참조.
MODEL_SLUG="${BASE_MODEL##*/}"
REPO="tayaee/${MODEL_SLUG}-math-${TARGET}"
[ "$STRAT" != single ] && REPO="$REPO-$STRAT"
[ "$MODE" != full ] && REPO="$REPO-$MODE"
if [ "$SOURCE" = "hf" ]; then MODEL="$REPO"; else MODEL="$M1B_MODELS/$SUF"; fi
case "$TARGET" in *-gguf) [ "$SOURCE" = "local" ] && MODEL="$MODEL/model-${QTYPE:-q8_0}.gguf" ;; esac

m1b_log "serving model=$MODEL quant=${QUANT:-none} tp=$TP port=$PORT"
# FLASH_ATTN 고정 + flashinfer sampler off: flashinfer JIT(sampling 커널) 빌드가
# GB10에서 깨짐 (ninja 실패 → Engine core init 실패)
export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-FLASH_ATTN}"
export VLLM_USE_FLASHINFER_SAMPLER=0
set -x
exec "$VLLM_BIN" serve "$MODEL" \
  --host 0.0.0.0 --port "$PORT" \
  --served-model-name "$REPO" \
  --tensor-parallel-size "$TP" \
  ${QUANT:+--quantization "$QUANT"} \
  --max-model-len 2048 \
  --gpu-memory-utilization "${GPU_UTIL:-0.5}" \
  --trust-remote-code
