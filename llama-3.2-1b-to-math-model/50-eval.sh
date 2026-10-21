#!/usr/bin/env bash
# 50-eval.sh [mini|full] [target...] (STRAT env, 기본 single) — Stage 7b wrapper. target 기본 all.
# all = base + fft/qlora FP 2종 + HF 양자화 6종 (GGUF 10종 제외, 52-eval-gguf.sh 별도 평가).
set -euo pipefail
source "$(dirname "$0")/config/common.env" "${1:-mini}"
STRAT="${STRAT:-single}"
case "$STRAT" in single|ddp|fsdp) ;; *) echo "STRAT must be single|ddp|fsdp" >&2; exit 1;; esac
TARGETS="${2:-all}"
[ "$TARGETS" = all ] && TARGETS="base fft qlora fft-gptq fft-awq fft-fp8 qlora-gptq qlora-awq qlora-fp8"
N="$EVAL_N"  # mini 10 / full 0=전체 (common.env)
TEST="$M1B_DATASETS/gsm8k-test.jsonl"
m1b_require "$TEST"
: "${VLLM_MAX_NUM_SEQS:=8}"
m1b_repro_echo
# 50-eval.py resolve()의 bash 미러 — base(HF)는 로컬 타임스탬프 없음
target_model() {
  case "$1" in
    base) echo "" ;;
    fft) echo "$M1B_MODELS/synthetic-fft-$MODE-$STRAT" ;;
    qlora) echo "$M1B_MODELS/synthetic-qlora-$MODE-$STRAT-merged" ;;
    fft-*) echo "$M1B_MODELS/synthetic-fft-$MODE-$STRAT-${1#fft-}" ;;
    qlora-*) echo "$M1B_MODELS/synthetic-qlora-$MODE-$STRAT-merged-${1#qlora-}" ;;
    gptq|awq|fp8) echo "$M1B_MODELS/synthetic-fft-$MODE-$STRAT-$1" ;;
    *) echo "" ;;
  esac
}
# VLLM_GPU_MEM_UTIL: 평가는 배치 작업이라 서빙급 예약 불필요 (1B 가중치 ~3GB +
# 짧은 프롬프트/512 생성. 22-teacher와 같은 0.5로 통일). 실행 중 잡에는 영향 없음.
: "${VLLM_GPU_MEM_UTIL:=0.5}"
m1b_log "tp=$TP targets=$TARGETS n=$N gpu_mem_util=$VLLM_GPU_MEM_UTIL max_num_seqs=$VLLM_MAX_NUM_SEQS"
m1b_lock "$M1B_OUTPUTS/eval-$MODE-$STRAT"
for t in $TARGETS; do
  out="$M1B_OUTPUTS/eval-$MODE-$STRAT/$t.jsonl"
  m="$(target_model "$t")"
  if [ -z "$m" ]; then
    if [ "${FORCE:-0}" != 1 ] && [ -f "$out" ]; then m1b_log "skip: fresh $out (base/HF, FORCE=1 to rebuild)"; continue; fi
  elif [ -f "$m/config.json" ] && [ -f "$out" ] && m1b_fresh "$out" "$m/config.json" "$TEST"; then
    m1b_log "skip: fresh $out (FORCE=1 to rebuild)"; continue
  fi
  (set -x; uv run "$M1B_ROOT/50-eval.py" --mode "$MODE" --strat "$STRAT" --target "$t" --n "$N" --tp "$TP" \
    --gpu-mem-util "$VLLM_GPU_MEM_UTIL" --max-num-seqs "$VLLM_MAX_NUM_SEQS")
done

echo ---- result ----
(set -x; ls -l "$M1B_OUTPUTS/eval-$MODE-$STRAT/")
(set -x; wc -l "$M1B_OUTPUTS/eval-$MODE-$STRAT/"*.jsonl)
