#!/usr/bin/env bash
# run-full.sh — full 전 파이프라인 원샷 (calib 256·seq 2048, 밤샘 권장).
# NOTE: 28.8B MoE NVFP4-GPTQ/AWQ가 가장 오래 걸린다. 공식 KRAFTON NVFP4와 PPL 비교 권장.
# 제외: 58-upload-hf, 60-serve, 62-infer, 64-serve-infer-all → 맨 아래 주석 명령으로 개별 실행.
set -euo pipefail
source "$(dirname "$0")/run-full.inc"
ROOT="$(cd "$(dirname "$0")" && pwd)"; cd "$ROOT"
source "$ROOT/config/common.env" "$MODE"
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

if want_step quant; then
  rq_step "33-awq-w3" ./33-quant-awq-w3.sh "$MODE"
  rq_step "32-awq-w5" ./32-quant-awq-w5.sh "$MODE"
  rq_step "35-nvfp4-awq" ./35-quant-nvfp4-awq.sh "$MODE"
  rq_step "36-nvfp4-gptq" ./36-quant-nvfp4-gptq.sh "$MODE"
fi

want_step ppl && rq_step "48-ppl" ./48-measure-ppl.sh "$MODE"
if want_step eval; then
  rq_step "50-eval" ./50-eval.sh "$MODE"
  rq_step "54-score" ./54-score.sh "$MODE"
fi

rq_log "run-$MODE done"
# --- 선택 (수동) ---
# ./58-upload-hf.sh full
# ./60-serve-vllm.sh full nvfp4-awq     # 터미널1 (상주, TP=2)
# ./62-infer-examples.sh 3              # 터미널2
# ./64-serve-infer-all.sh full 3        # 전 타깃 순회
