#!/usr/bin/env bash
# 50-eval.sh [smoke|full] [target...] — vLLM offline 스모크 생성. target 기본 all.
set -euo pipefail
source "$(dirname "$0")/config/common.env" "${1:-smoke}"
TARGETS="${2:-all}"
N="$EVAL_N"
: "${VLLM_GPU_MEM_UTIL:=0.9}"
rq_repro_echo
target_model() {
  case "$1" in
    base) echo "$RQ_MODELS/raon-bf16" ;;
    *) echo "$RQ_MODELS/raon-$1-$MODE" ;;
  esac
}
rq_log "tp=$TP targets=$TARGETS n=$N gpu_mem_util=$VLLM_GPU_MEM_UTIL"
rq_lock "$RQ_OUTPUTS/eval-$MODE"
for t in $TARGETS; do
  [ "$t" = all ] && { uv run "$RQ_ROOT/50-eval.py" --mode "$MODE" --target all --n "$N" --tp "$TP"; break; }
  out="$RQ_OUTPUTS/eval-$MODE/$t.jsonl"
  m="$(target_model "$t")"
  if [ -f "$m/config.json" ] && [ -f "$out" ] && [ "${FORCE:-0}" != 1 ] && rq_fresh "$out" "$m/config.json"; then
    rq_log "skip: fresh $out (FORCE=1 to rebuild)"; continue
  fi
  [ -f "$m/config.json" ] || { rq_log "missing model: $m (run 30-dequant-mxfp8.sh first)"; exit 1; }
  (set -x; uv run "$RQ_ROOT/50-eval.py" --mode "$MODE" --target "$t" --n "$N" --tp "$TP")
done

echo ---- result ----
(set -x; ls -l "$RQ_OUTPUTS/eval-$MODE/")
(set -x; wc -l "$RQ_OUTPUTS/eval-$MODE/"*.jsonl)
