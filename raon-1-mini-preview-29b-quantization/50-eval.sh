#!/usr/bin/env bash
# 50-eval.sh [smoke|full] [target...] — 벤치마크 프롬프트 생성. target 기본 all.
# 50-eval.py가 backend를 고른다(기본 auto: vLLM → 커스텀 아키텍처 미지원 시 transformers 폴백).
set -euo pipefail
source "$(dirname "$0")/config/common.env" "${1:-smoke}"
TARGETS="${2:-${TARGETS:-all}}"
N="$EVAL_N"
: "${VLLM_GPU_MEM_UTIL:=0.9}"
: "${RQ_EVAL_BACKEND:=auto}"
export VLLM_GPU_MEM_UTIL RQ_EVAL_BACKEND
rq_repro_echo
target_model() {
  case "$1" in
    base) echo "$RQ_MODELS/raon-bf16" ;;
    *) echo "$RQ_MODELS/raon-$1-$MODE" ;;
  esac
}
rq_log "tp=$TP targets=$TARGETS n=$N gpu_mem_util=$VLLM_GPU_MEM_UTIL backend=$RQ_EVAL_BACKEND"
rq_lock "$RQ_OUTPUTS/eval-$MODE"
# `all`을 명시 목록으로 펴야 한다: 그대로 넘기면 50-eval.py --target all이 29B 모델
# 5개를 한 프로세스에서 로드해 GPU/통합 메모리를 다 먹는다 (48-measure-ppl.sh와 동일).
EVAL_ALL_TARGETS="base awq-w3 awq-w5 nvfp4-awq nvfp4-gptq"
_ts=""
for t in $(echo "$TARGETS" | tr ',' ' '); do
  if [ "$t" = all ]; then
    _ts="$EVAL_ALL_TARGETS"
    break
  fi
  _ts="${_ts:+$_ts }$t"
done
[ -n "${_ts:-}" ] || _ts="$EVAL_ALL_TARGETS"
for t in $_ts; do
  out="$RQ_OUTPUTS/eval-$MODE/$t.jsonl"
  m="$(target_model "$t")"
  if [ -f "$m/config.json" ] && [ -f "$out" ] && [ "${FORCE:-0}" != 1 ] && rq_fresh "$out" "$m/config.json"; then
    rq_log "skip: fresh $out (FORCE=1 to rebuild)"; continue
  fi
  [ -f "$m/config.json" ] || { rq_log "missing model: $m (run 30-dequant-mxfp8.sh first)"; exit 1; }
  (set -x; uv run "$RQ_ROOT/50-eval.py" --mode "$MODE" --target "$t" --n "$N" --tp "$TP" --backend "$RQ_EVAL_BACKEND")
done

echo ---- result ----
(set -x; ls -l "$RQ_OUTPUTS/eval-$MODE/")
(set -x; wc -l "$RQ_OUTPUTS/eval-$MODE/"*.jsonl)