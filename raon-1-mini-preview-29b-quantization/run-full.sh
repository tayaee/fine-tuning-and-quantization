#!/usr/bin/env bash
# run-full.sh — full 전 파이프라인 원샷 (calib 256·seq 2048, 밤샘 권장).
# NOTE: 28.8B MoE NVFP4-GPTQ/AWQ가 가장 오래 걸린다. 공식 KRAFTON NVFP4와 PPL 비교 권장.
# TARGETS: 양자화 타깃 부분집합 (콤마 구분, 기본 all) — 2노드 분산용. PPL/eval/prep은 base를 앞에 붙인다.
#   예: TARGETS=awq-w3,awq-w5 STEPS=dequant,quant ./run-full.sh        # spark1 분담
#       TARGETS=nvfp4-awq,nvfp4-gptq STEPS=dequant,quant ./run-full.sh  # spark2 분담
#   smoke와 동일한 TARGETS 분할을 쓴다 (dequant 공유 1회, PPL/eval은 전 타깃 모인 뒤 1곳에서).
# QUANT_CPU_ONLY=1 ./run-full.sh  # 양자화 단계만 GPU를 가리고 PPL/eval은 GPU 사용.
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

# TARGETS 정규화 — quant/ppl/eval/prep/upload이 같은 목록을 본다 (issue-4 요구사항 3).
_targets="${TARGETS:-all}"
case ",${_targets}," in *,all,*) _targets="awq-w3,awq-w5,nvfp4-awq,nvfp4-gptq" ;; esac
# base(30-dequant 복원본)를 항상 포함 — PPL Δ 판정과 모델 카드 비교표의 기준선이다.
_eval_targets="base,$_targets"

# 양자화만 CPU 강제(QUANT_CPU_ONLY=1): env로 자식 프로세스에만 빈 값을 넘겨
# 48-ppl/50-eval은 그대로 GPU를 본다 (부모 셸을 바꾸면 평가가 CPU로 떨어진다).
if [ "${QUANT_CPU_ONLY:-0}" = 1 ]; then
  _quant_cmd=(env CUDA_VISIBLE_DEVICES=)
  rq_log "QUANT_CPU_ONLY=1 — 양자화 단계만 GPU 숨김, PPL/eval은 GPU 사용"
else
  _quant_cmd=(env)
fi

if want_step quant; then
  for _t in $(echo "$_targets" | tr ',' ' '); do
    case "$_t" in
      awq-w3) rq_step "33-awq-w3" "${_quant_cmd[@]}" ./33-quant-awq-w3.sh "$MODE" ;;
      awq-w5) rq_step "32-awq-w5" "${_quant_cmd[@]}" ./32-quant-awq-w5.sh "$MODE" ;;
      nvfp4-awq) rq_step "35-nvfp4-awq" "${_quant_cmd[@]}" ./35-quant-nvfp4-awq.sh "$MODE" ;;
      nvfp4-gptq) rq_step "36-nvfp4-gptq" "${_quant_cmd[@]}" ./36-quant-nvfp4-gptq.sh "$MODE" ;;
      *) echo "unknown quant target: $_t (awq-w3|awq-w5|nvfp4-awq|nvfp4-gptq)" >&2; exit 1 ;;
    esac
  done
fi

want_step ppl && rq_step "48-ppl" ./48-measure-ppl.sh "$MODE" "$_eval_targets"
if want_step eval; then
  rq_step "50-eval" ./50-eval.sh "$MODE" "$_eval_targets"
  rq_step "54-score" ./54-score.sh "$MODE"
fi
want_step prep && rq_step "56-prep" ./56-prep-upload.sh "$MODE" "$_targets"

rq_log "run-$MODE done"
# --- 선택 (수동) ---
# ./58-upload-hf.sh full --dry-run   # 56-prep 산출물 검증만 (네트워크 없음)
# ./58-upload-hf.sh full              # smoke는 업로드하지 않음
# ./60-serve-vllm.sh full nvfp4-awq     # 터미널1 (상주, TP=2)
# ./62-infer-examples.sh 3              # 터미널2
# ./64-serve-infer-all.sh full 3        # 전 타깃 순회
