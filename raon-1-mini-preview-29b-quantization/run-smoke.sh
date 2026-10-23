#!/usr/bin/env bash
# run-smoke.sh — smoke 파이프라인 원샷 (full과 동일 단계, 작은 규모).
#   setup → calib(8) → compat → validate → dequant → quant 4종 → PPL → eval+score.
#   규모는 config/common.env smoke값 (CALIB_N=8·SEQ_LEN=512·EVAL_N=8·PPL_N=8).
# 빠른 파이프라인 점검만 원하면 PIPELINE_TEST=1 ./run-smoke.sh
#   (setup → calib(2) → compat → validate, 5분 목표, 양자화 미실행).
# 사용: ./run-smoke.sh
#   PIPELINE_TEST=1 ./run-smoke.sh       # 빠른 점검 (양자화 제외)
#   SKIP_SETUP=1 ./run-smoke.sh         # 10-setup 생략
#   STEPS=setup,calib,compat ./run-smoke.sh  # 일부 스텝만
# 스텝명: setup calib compat validate dequant quant ppl eval (기본 STEPS=all)
# TARGETS: 양자화 타깃 부분집합 (콤마 구분, 기본 all)
#   예: TARGETS=awq-w3,awq-w5 STEPS=quant ./run-smoke.sh   # spark1 분담
#       TARGETS=nvfp4-awq,nvfp4-gptq STEPS=quant ./run-smoke.sh  # spark2 분담
# 2노드 smoke/full 모두 같은 분할을 쓴다 (dequant는 공유 1회, PPL/eval은 전 타깃 모인 뒤 1곳에서).
set -euo pipefail
source "$(dirname "$0")/run-smoke.inc"
ROOT="$(cd "$(dirname "$0")" && pwd)"; cd "$ROOT"
source "$ROOT/config/common.env" "$MODE"
# 기본값: full과 동일 단계 전부 실행 (작은 규모). 빠른 점검은 PIPELINE_TEST=1로 명시.
: "${PIPELINE_TEST:=0}"
if [ "$PIPELINE_TEST" = 1 ]; then
  export CALIB_N=2 SEQ_LEN=128 EVAL_N=1 PPL_N=2
  : "${STEPS:=setup,calib,compat,validate}"
fi
rq_repro_echo

TIMING_LOG="$ROOT/logs/timing-$MODE.log"
mkdir -p "$ROOT/logs"
RUN_T0=$(date +%s)
rq_step() {
  local _name="$1"; shift
  local _t0 _dt _rc=0
  _t0=$(date +%s)
  rq_log "step start: $_name"
  "$@" || _rc=$?
  _dt=$(($(date +%s) - _t0))
  rq_log "step done: $_name elapsed=${_dt}s rc=$_rc"
  if [ "$_rc" -ne 0 ] || [ "$_dt" -gt 0 ]; then
    printf '%s\t%s\t%s\t%ss\trc=%d\n' "$(date -u +%FT%TZ)" "$MODE" "$_name" "$_dt" "$_rc" >>"$TIMING_LOG"
  fi
  return $_rc
}
want_step() {
  case ",${STEPS:-all}," in *,all,*|*,"$1",*) return 0 ;; *) return 1 ;; esac
}
run_done() {
  rq_log "run-$MODE end rc=$? total=$(( $(date +%s) - RUN_T0 ))s (timing: $TIMING_LOG)"
}
trap 'RQ_RC=$?; rq_repro; run_done' EXIT

rq_log "run-$MODE start (steps=${STEPS:-all})"

if [ "${SKIP_SETUP:-0}" != 1 ] && want_step setup; then
  rq_step "10-setup" ./10-setup.sh
fi

want_step calib && rq_step "12-calib" ./12-prep-calibration.sh "$MODE"
want_step compat && rq_step "20-compat" ./20-compat-check.sh "$MODE"

want_step dequant && rq_step "30-dequant" ./30-dequant-mxfp8.sh "$MODE"

if want_step validate; then
  rq_step "validate-helps" bash -c '
    for s in 30-dequant-mxfp8 32-quant-awq-w5 33-quant-awq-w3 35-quant-nvfp4-awq 36-quant-nvfp4-gptq \
             48-measure-ppl 50-eval 54-score 58-upload-hf; do
      uv run "$0/$s.py" --help >/dev/null || exit 1
    done
    for s in 48-measure-ppl 50-eval 54-score 58-upload-hf 60-serve-vllm 62-infer-examples; do
      bash -n "$0/$s.sh" || exit 1
    done
    echo "validate OK (15 scripts)"' "$ROOT"
fi

if want_step quant; then
  _targets="${TARGETS:-all}"
  case ",${_targets}," in *,all,*) _targets="awq-w3,awq-w5,nvfp4-awq,nvfp4-gptq" ;; esac
  for _t in $(echo "$_targets" | tr ',' ' '); do
    case "$_t" in
      awq-w3) rq_step "33-awq-w3" ./33-quant-awq-w3.sh "$MODE" ;;
      awq-w5) rq_step "32-awq-w5" ./32-quant-awq-w5.sh "$MODE" ;;
      nvfp4-awq) rq_step "35-nvfp4-awq" ./35-quant-nvfp4-awq.sh "$MODE" ;;
      nvfp4-gptq) rq_step "36-nvfp4-gptq" ./36-quant-nvfp4-gptq.sh "$MODE" ;;
      *) echo "unknown quant target: $_t (awq-w3|awq-w5|nvfp4-awq|nvfp4-gptq)" >&2; exit 1 ;;
    esac
  done
fi

want_step ppl && rq_step "48-ppl" ./48-measure-ppl.sh "$MODE"
if want_step eval; then
  rq_step "50-eval" ./50-eval.sh "$MODE"
  rq_step "54-score" ./54-score.sh "$MODE"
fi

rq_log "run-$MODE done"
# --- 선택 (수동) ---
# ./58-upload-hf.sh full              # smoke는 업로드하지 않음
# ./60-serve-vllm.sh smoke nvfp4-gptq  # 터미널1 (상주)
# ./62-infer-examples.sh 3            # 터미널2
# ./64-serve-infer-all.sh smoke 3     # 전 타깃 순회
