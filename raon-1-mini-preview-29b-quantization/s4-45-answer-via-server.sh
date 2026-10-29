#!/usr/bin/env bash
# s4-45-answer-via-server.sh [mode] [target] — 실행 중인 서버로 벤치마크 응답 생성. target 필수.
# in-process로 읽을 수 없는 가중치(ModelOpt FP4 등)의 정확도를 실제 서빙 스택에서 잰다.
# s4-40-serve-...가 띄운 endpoint를 쓴다. 서버가 없으면 skip(기존 응답 보존).
# 출력 스키마는 s4-07/s4-10과 동일 → s4-20-aggregate 이후 그대로 동작한다.
set -euo pipefail
source "$(dirname "$0")/config/common.env" "${1:-${MODE:-smoke}}"
TARGETS="$(rq_require_targets "./s4-45-answer-via-server.sh $MODE" "${2:-}" base awq-w3-transformers awq-w5-transformers nvfp4-awq-vllm nvfp4-awq-transformers nvfp4-gptq-transformers)" || exit 2
N="$EVAL_N"
: "${BASE_URL:=}"
: "${PORT:=8000}"
# 서버의 served-model-name 이 타깃 slug 가 아닐 때 (예: 카드의 서빙 명령은 repo id 를 쓴다)
# → --model 로 명시해야 오귀인 차단을 통과한다. SERVER_MODEL 로 넘긴다.
: "${SERVER_MODEL:=}"
export BASE_URL PORT
rq_repro_echo
rq_banner "s4-45-answer-via-server" "Stage 4 - generate benchmark answers via the running server (for in-process-unloadable weights)"
BENCHES="${EVAL_BENCHMARKS:-gsm8k}"
rq_log "tp=$TP targets=$TARGETS n=$N benches=$BENCHES backend=server url=${BASE_URL:-localhost:$PORT} server_model=${SERVER_MODEL:-<auto>}"
rq_lock "$RQ_OUTPUTS/eval-$MODE"
EVAL_ALL_TARGETS="base awq-w3-transformers awq-w5-transformers nvfp4-awq-vllm nvfp4-awq-transformers nvfp4-gptq-transformers"
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
  m="$RQ_MODELS/raon-bf16"
  [ "$t" = base ] || m="$RQ_MODELS/raon-$t-$MODE"
  # 요청한 벤치 응답이 없으면 신선하지 않다 (일부 벤치만 돌린 결과).
  _missing_bench=""
  for _b in $(echo "$BENCHES" | tr ',' ' '); do
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
  (set -x; uv run "$RQ_ROOT/s4-45-answer-via-server.py" --mode "$MODE" --target "$t" --n "$N" \
    --benchmarks "$BENCHES" --base-url "$BASE_URL" --port "$PORT" \
    ${SERVER_MODEL:+--model "$SERVER_MODEL"})
done

echo ---- result ----
(set -x; ls -l "$RQ_OUTPUTS/eval-$MODE/")
(set -x; wc -l "$RQ_OUTPUTS/eval-$MODE/"*.jsonl)
