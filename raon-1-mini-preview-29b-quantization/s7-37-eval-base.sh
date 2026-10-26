#!/usr/bin/env bash
# s7-37-eval-base.sh [smoke|full] — 베이스 모델 벤치마크 응답 생성.
# s7-37-eval.py가 backend를 고른다(기본 auto: vLLM → 커스텀 아키텍처 미지원 시 transformers 폴백).
set -euo pipefail
source "$(dirname "$0")/config/common.env" "${1:-${MODE:-smoke}}"
N="$EVAL_N"
: "${VLLM_GPU_MEM_UTIL:=0.9}"
: "${RQ_EVAL_BACKEND:=auto}"
export VLLM_GPU_MEM_UTIL RQ_EVAL_BACKEND
rq_repro_echo
rq_banner "s7-37-eval-base" "Stage 7/9 - generate benchmark answers for the base model and save them"
# 평가 벤치(기본: 표준 GSM8K). s7-36-prep-gsm8k.sh가 prompts-<bench>.jsonl을 만든다.
BENCHES="${EVAL_BENCHMARKS:-gsm8k}"
rq_log "tp=$TP target=base n=$N benches=$BENCHES batch=${EVAL_BATCH:-1} gpu_mem_util=$VLLM_GPU_MEM_UTIL backend=$RQ_EVAL_BACKEND"
rq_lock "$RQ_OUTPUTS/eval-$MODE"
out="$RQ_OUTPUTS/eval-$MODE/base.jsonl"
m="$RQ_MODELS/raon-bf16"
# 요청한 벤치 응답이 없으면 신선하지 않다 (일부 벤치만 돌린 결과).
# 프롬프트 파일 mtime은 보지 않는다 (결정론 생성이라 내용이 같으면 재측정 불필요).
# 모델(config.json)이 바뀌면 재측정한다.
_missing_bench=""
for _b in $(echo "$BENCHES" | tr ',' ' '); do
  # 파일이 새것이어도 요청한 벤치 응답이 없으면 신선하지 않다(일부 벤치만 돌린 결과).
  if [ -f "$out" ] && ! grep -q "\"bench\": \"$_b\"" "$out"; then
    _missing_bench="${_missing_bench:+$_missing_bench,}$_b"
  fi
done
if [ -n "$_missing_bench" ]; then
  rq_log "stale: $out has no rows for bench(es) $_missing_bench"
fi
if [ -z "$_missing_bench" ] && [ -f "$m/config.json" ] && [ -f "$out" ] && [ "${FORCE:-0}" != 1 ] && rq_fresh "$out" "$m/config.json"; then
  rq_log "skip: fresh $out (FORCE=1 to rebuild)"
else
  [ -f "$m/config.json" ] || { rq_log "missing model: $m (run s4-19-dequant-mxfp8.sh first)"; exit 1; }
  (set -x; uv run "$RQ_ROOT/s7-37-eval.py" --mode "$MODE" --target base --n "$N" --tp "$TP" \
     --backend "$RQ_EVAL_BACKEND" --benchmarks "$BENCHES" --batch-size "${EVAL_BATCH:-1}")
fi

echo ---- result ----
(set -x; ls -l "$RQ_OUTPUTS/eval-$MODE/")
(set -x; wc -l "$RQ_OUTPUTS/eval-$MODE/"*.jsonl)
