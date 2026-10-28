#!/usr/bin/env bash
# s4-10-eval-quantized.sh [smoke|full] [targets] — 양자화 모델 벤치마크 응답 생성. targets 필수.
# base는 s4-07-eval-base.sh를 쓴다.
# s4-07-eval.py가 backend를 고른다(기본 auto: vLLM → 커스텀 아키텍처 미지원 시 transformers 폴백).
set -euo pipefail
source "$(dirname "$0")/config/common.env" "${1:-${MODE:-smoke}}"
TARGETS="$(rq_require_targets "./s4-10-eval-quantized.sh $MODE" "${2:-}" awq-w3 awq-w5 nvfp4-awq-modelopt nvfp4-awq nvfp4-gptq)" || exit 2
N="$EVAL_N"
: "${VLLM_GPU_MEM_UTIL:=0.9}"
: "${RQ_EVAL_BACKEND:=auto}"
export VLLM_GPU_MEM_UTIL RQ_EVAL_BACKEND
rq_repro_echo
rq_banner "s4-10-eval-quantized" "Stage 4 - generate benchmark answers for quantized models and save them"
target_model() {
  case "$1" in
    *) echo "$RQ_MODELS/raon-$1-$MODE" ;;
  esac
}
# 평가 벤치(기본: 표준 GSM8K). s4-04-prep-gsm8k.sh가 prompts-<bench>.jsonl을 만든다.
BENCHES="${EVAL_BENCHMARKS:-gsm8k}"
rq_log "tp=$TP targets=$TARGETS n=$N benches=$BENCHES batch=${EVAL_BATCH:-1} gpu_mem_util=$VLLM_GPU_MEM_UTIL backend=$RQ_EVAL_BACKEND"
rq_lock "$RQ_OUTPUTS/eval-$MODE"
# `all`을 명시 목록으로 펴야 한다: 그대로 넘기면 s4-07-eval.py --target all이 29B 모델
# 4개를 한 프로세스에서 로드해 GPU/통합 메모리를 다 먹는다 (s4-01-measure-ppl.sh와 동일).
EVAL_ALL_TARGETS="awq-w3 awq-w5 nvfp4-awq-modelopt nvfp4-awq nvfp4-gptq"
_ts=""
for t in $(echo "$TARGETS" | tr ',' ' '); do
  if [ "$t" = all ]; then
    _ts="$EVAL_ALL_TARGETS"
    break
  fi
  case "$t" in
    base) echo "base는 s4-07-eval-base.sh를 쓰라." >&2; exit 1 ;;
  esac
  _ts="${_ts:+$_ts }$t"
done
[ -n "${_ts:-}" ] || _ts="$EVAL_ALL_TARGETS"
for t in $_ts; do
  out="$RQ_OUTPUTS/eval-$MODE/$t.jsonl"
  m="$(target_model "$t")"
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
    rq_log "skip: fresh $out (FORCE=1 to rebuild)"; continue
  fi
  [ -f "$m/config.json" ] || { rq_log "missing model: $m (run the quant step first)"; exit 1; }
  (set -x; uv run "$RQ_ROOT/s4-07-eval.py" --mode "$MODE" --target "$t" --n "$N" --tp "$TP" \
     --backend "$RQ_EVAL_BACKEND" --benchmarks "$BENCHES" --batch-size "${EVAL_BATCH:-1}")
done

echo ---- result ----
(set -x; ls -l "$RQ_OUTPUTS/eval-$MODE/")
(set -x; wc -l "$RQ_OUTPUTS/eval-$MODE/"*.jsonl)
