#!/usr/bin/env bash
# s4-00-measure-ppl__transformers-only.sh [smoke|full] [targets] — PPL 측정 + base 대비 Δ판정.
# targets 필수 (미지정 시 사용법 출력 후 종료, TARGETS=all로 전체).
# 타겟마다 별도 프로세스로 띄운다 (요구사항 1): 29B 모델 5개를 한 프로세스에서
# 순차 로드하면 GB10(128GB 통합 메모리)에서 두 번째 로드부터 OOM이 난다.
# 프로세스 격리로 OS 메모리까지 100% 회수한 뒤 마지막에 캐시를 집계한다.
set -euo pipefail
source "$(dirname "$0")/config/common.env" "${1:-${MODE:-smoke}}"
TARGETS="$(rq_require_targets "./s4-00-measure-ppl__transformers-only.sh $MODE" "${2:-}" base awq-w3-transformers awq-w5-transformers nvfp4-awq-vllm nvfp4-awq-transformers nvfp4-gptq-transformers)" || exit 2
N="${PPL_N:-8}"
rq_repro_echo
rq_banner "s4-00-measure-ppl__transformers-only" "Stage 4 - measure in-domain PPL and judge the delta against the bf16 base"
SUMMARY="$RQ_OUTPUTS/ppl-$MODE/summary.json"
rq_lock "$RQ_OUTPUTS/ppl-$MODE"
# s4-24/s4-26 서버가 떠 있으면 모델 이중 적재(OOM 위험) — 먼저 내리고 실행할 것.
for _pf in "$RQ_ROOT/logs/s4-transformers.pid" "$RQ_ROOT/logs/s4-vllm-fork.pid"; do
  _pid="$(cat "$_pf" 2>/dev/null || true)"
  if [ -n "$_pid" ] && kill -0 "$_pid" 2>/dev/null; then
    rq_log "WARN: local server alive (pid=$_pid, $_pf) — in-process 측정과 모델 이중 적재됨. s4-55-stop-serve-local-weight.sh로 먼저 내릴 것"
  fi
done
unset _pf _pid
target_model() {
  case "$1" in
    base|bf16|raon-bf16) echo "$RQ_MODELS/raon-bf16" ;;
    *) echo "$RQ_MODELS/raon-$1-$MODE" ;;
  esac
}
rq_log "ppl_n=$N targets=$TARGETS"
# `all`을 명시 목록으로 펴야 한다: 그대로 넘기면 s4-00-measure-ppl__transformers-only.py가 29B 모델 5개를
# 한 프로세스에서 로드해 (accept 기준 2가 막으려는) GB10 OOM이 재발한다.
PPL_ALL_TARGETS="base awq-w3-transformers awq-w5-transformers nvfp4-awq-vllm nvfp4-awq-transformers nvfp4-gptq-transformers"
_t=""
for t in $(echo "$TARGETS" | tr ',' ' '); do
  if [ "$t" = all ]; then
    _t="$PPL_ALL_TARGETS"
    break
  fi
  _t="${_t:+$_t }$t"
done
[ -n "${_t:-}" ] || _t="$PPL_ALL_TARGETS"
rq_log "ppl targets(1 process each): $_t"
for t in $_t; do
  _norm_t="$t"
  [ "$_norm_t" = bf16 ] || [ "$_norm_t" = raon-bf16 ] && _norm_t="base"
  out="$RQ_OUTPUTS/ppl-$MODE/${_norm_t}.json"
  m="$(target_model "$t")"
  if rq_skip_transformers "$_norm_t"; then
    rq_log "SKIP: $_norm_t — transformers 적재 불가(vLLM 전용), 품질 게이트는 s7 서빙"
    printf '{"target": "%s", "skipped": "transformers-unloadable (vLLM-only format)"}\n' "$_norm_t" > "$out"
    continue
  fi
  if [ -f "$m/config.json" ] && [ -f "$out" ] && [ "${FORCE:-0}" != 1 ] && rq_fresh "$out" "$m/config.json"; then
    rq_log "skip: fresh $out (FORCE=1 to rebuild)"; continue
  fi
  (set -x; uv run "$RQ_ROOT/s4-00-measure-ppl__transformers-only.py" --mode "$MODE" --n "$N" --targets "$t")
done
# 마지막 집계: 이번 실행에서 못 잰 타겟도 디렉터리의 캐시 json으로 판정된다.
(set -x; uv run "$RQ_ROOT/s4-00-measure-ppl__transformers-only.py" --mode "$MODE" --summarize-only)

echo ---- result ----
(set -x; cat "$SUMMARY")