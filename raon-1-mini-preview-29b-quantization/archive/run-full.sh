#!/usr/bin/env bash
# run-full.sh — full 전 파이프라인 원샷 (calib 256·seq 2048, 밤샘 권장).
# NOTE: 28.8B MoE NVFP4-GPTQ/AWQ가 가장 오래 걸린다. 공식 KRAFTON NVFP4와 PPL 비교 권장.
# TARGETS: 양자화 타깃 부분집합 (콤마 구분) — 2노드 분산용. **미지정 시 가능한 값을 찍고 종료**. PPL/eval/prep은 base를 앞에 붙인다.
#   예: TARGETS=all ./run-full.sh                              # 4종 전부
#       TARGETS=awq-w3,awq-w5 STEPS=dequant,quant ./run-full.sh        # spark1 분담
#       TARGETS=nvfp4-awq,nvfp4-gptq STEPS=dequant,quant ./run-full.sh  # spark2 분담
#   smoke와 동일한 TARGETS 분할을 쓴다 (dequant 공유 1회, PPL/eval은 전 타깃 모인 뒤 1곳에서).
# QUANT_CPU_ONLY=1 ./run-full.sh  # 양자화 단계만 GPU를 가리고 PPL/eval은 GPU 사용.
# 제외: s5-01-upload-hf, s5-01-upload-hf, s7-13-chat, s7-16-pi, s7-19-verify-serving-log → 맨 아래 주석 명령으로 개별 실행.
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
  # SDLC 원장 기록 (실패도 기록한다 — 증빙이므로).
  local _stage _sid _sname _st _tgt
  _stage="$(rq_stage_of "$_name")"
  if [ -n "$_stage" ]; then
    _sid="${_stage%% *}"; _sname="${_stage##* }"
    if [ "$_rc" -eq 0 ]; then _st=pass; else _st=fail; fi
    _tgt="$(rq_target_of "$_name")"
    if [ -n "$_tgt" ]; then
      rq_stage_target "$_sid" "$_sname" "$_st" "$_tgt" "$_name"
    else
      rq_stage_record "$_sid" "$_sname" "$_st" "$_name"
    fi
  fi
  return $_rc
}
want_step() {
  case ",${STEPS:-all}," in *,all,*|*,"$1",*) return 0 ;; *) return 1 ;; esac
}
# 명시적으로 나열한 스텝만 실행 (STEPS=all 에는 포함되지 않는다).
#   STEPS=verify ./run-full.sh   → s4-34-verify + gate-release 만 추가
want_step_explicit() {
  case ",${STEPS:-all}," in *,"$1",*) return 0 ;; *) return 1 ;; esac
}
# 서빙 처리량 측정은 명시 요청 시에만(STEPS=serve) — 서버가 떠 있어야 측정된다.
want_step_serve() {
  case ",${STEPS:-all}," in *,serve,*) return 0 ;; *) return 1 ;; esac
}
run_done() {
  rq_log "run-$MODE end rc=$? total=$(( $(date +%s) - RUN_T0 ))s (timing: $TIMING_LOG)"
}
trap 'RQ_RC=$?; rq_repro; run_done' EXIT

rq_banner "run-$MODE" "pipeline: s1 env-chain -> s2 dequant -> s3 quant -> s4 measure+eval (TARGETS is required)"
rq_log "run-$MODE start (steps=${STEPS:-all})"

# TARGETS 정규화 — quant/ppl/eval/prep/upload이 같은 목록을 본다 (issue-4 요구사항 3).
# TARGETS는 **명시해야 한다**: 미지정 시 4종 전체(각 20~40분)를 조용히 돌지 않는다.
RQ_TARGETS_ALL="awq-w3 awq-w5 nvfp4-awq nvfp4-gptq"
if [ -z "${TARGETS:-}" ]; then
  echo "TARGETS is not specified — you must name the quantization targets (issue-12)." >&2
  echo "" >&2
  echo "Available values:" >&2
  for _t in $RQ_TARGETS_ALL; do echo "  - $_t" >&2; done
  echo "  - all            # all 4 targets above" >&2
  echo "" >&2
  echo "Examples:" >&2
  echo "  TARGETS=nvfp4-awq ./run-*.sh              # a single target" >&2
  echo "  TARGETS=awq-w3,awq-w5 ./run-*.sh           # split across 2 nodes" >&2
  echo "  TARGETS=all ./run-*.sh                     # all (slow per target)" >&2
  exit 2
fi
_targets="$TARGETS"
case ",${_targets}," in *,all,*) _targets="awq-w3,awq-w5,nvfp4-awq,nvfp4-gptq" ;; esac
# 알 수 없는 타깃은 즉시 거부한다(조용히 무시되면 "왜 안 돌아갔지"가 된다).
for _t in $(echo "$_targets" | tr ',' ' '); do
  case " $RQ_TARGETS_ALL " in
    *" $_t "*) ;;
    *) echo "Unknown TARGETS: $_t (allowed: $RQ_TARGETS_ALL | all)" >&2; exit 2 ;;
  esac
done

if [ "${SKIP_SETUP:-0}" != 1 ] && want_step setup; then
  rq_step "s1-01-setup" ./s1-01-setup.sh
fi

# Stage 1 — 수용 기준. 이게 없으면 뒤 단계의 판정 기준이 없다(s1-04-acceptance).
if want_step acceptance; then
  rq_step "s1-04-acceptance" ./s1-04-acceptance.sh "$MODE"
  rq_stage_record 1 acceptance pass "$RQ_OUTPUTS/acceptance-$MODE.json"
fi

want_step calib && rq_step "s1-07-calib" ./s1-07-prep-calibration.sh "$MODE"
if want_step compat; then
  rq_step "s1-10-compat" ./s1-10-compat-check.sh "$MODE"
  rq_stage_record 1 compat pass "$RQ_OUTPUTS/compat.stamp"
fi
if want_step dequant; then
  rq_step "s2-01-dequant" ./s2-01-dequant-mxfp8.sh "$MODE"
  rq_stage_record 2 dequant pass "$RQ_MODELS/raon-bf16"
fi

# base(s2-01-dequant 복원본)를 항상 포함 — PPL Δ 판정과 모델 카드 비교표의 기준선이다.
_eval_targets="base,$_targets"

# 양자화만 CPU 강제(QUANT_CPU_ONLY=1): env로 자식 프로세스에만 빈 값을 넘겨
# s4-01-ppl/s4-07-eval-base·s4-10-eval-quantized는 그대로 GPU를 본다 (부모 셸을 바꾸면 평가가 CPU로 떨어진다).
if [ "${QUANT_CPU_ONLY:-0}" = 1 ]; then
  _quant_cmd=(env CUDA_VISIBLE_DEVICES=)
  rq_log "QUANT_CPU_ONLY=1 — 양자화 단계만 GPU 숨김, PPL/eval은 GPU 사용"
else
  _quant_cmd=(env)
fi

if want_step quant; then
  for _t in $(echo "$_targets" | tr ',' ' '); do
    case "$_t" in
      awq-w3)
        rq_step "s3-01-quant-awq-w3" "${_quant_cmd[@]}" ./s3-01-quant-awq-w3.sh "$MODE"
        rq_step "gate-quant" rq_gate_quant "$RQ_MODELS/raon-$_t-$MODE" "$(rq_quant_format "$_t")"
        rq_stage_target 3 quant pass "$_t" "$RQ_MODELS/raon-$_t-$MODE" ;;
      awq-w5)
        rq_step "s3-04-quant-awq-w5" "${_quant_cmd[@]}" ./s3-04-quant-awq-w5.sh "$MODE"
        rq_step "gate-quant" rq_gate_quant "$RQ_MODELS/raon-$_t-$MODE" "$(rq_quant_format "$_t")"
        rq_stage_target 3 quant pass "$_t" "$RQ_MODELS/raon-$_t-$MODE" ;;
      nvfp4-awq)
        rq_step "s3-07-quant-nvfp4-awq" "${_quant_cmd[@]}" ./s3-07-quant-nvfp4-awq.sh "$MODE"
        rq_step "gate-quant" rq_gate_quant "$RQ_MODELS/raon-$_t-$MODE" "$(rq_quant_format "$_t")"
        rq_stage_target 3 quant pass "$_t" "$RQ_MODELS/raon-$_t-$MODE" ;;
      nvfp4-gptq)
        rq_step "s3-10-quant-nvfp4-gptq" "${_quant_cmd[@]}" ./s3-10-quant-nvfp4-gptq.sh "$MODE"
        rq_step "gate-quant" rq_gate_quant "$RQ_MODELS/raon-$_t-$MODE" "$(rq_quant_format "$_t")"
        rq_stage_target 3 quant pass "$_t" "$RQ_MODELS/raon-$_t-$MODE" ;;
      *) echo "unknown quant target: $_t (awq-w3|awq-w5|nvfp4-awq|nvfp4-gptq)" >&2; exit 1 ;;
    esac
  done
fi

if want_step ppl; then
  rq_step "s4-01-ppl" ./s4-01-measure-ppl.sh "$MODE" "$_eval_targets"
  rq_stage_record 4 measure pass "$RQ_OUTPUTS/ppl-$MODE/summary.json"
  # Stage 4 게이트: PPL이 유한하고 수용 기준(Δ) 판정과 일치하는지 확인한다.
  for _t in $(echo "$_targets" | tr ',' ' '); do
    rq_step "gate-ppl" rq_gate_ppl "$RQ_OUTPUTS/ppl-$MODE/$_t.json" "$_t"
  done
fi
if want_step eval; then
  # 평가 세트 준비(36 GSM8K / 43 한국어 KMMLU / 49 형식 준수) — 결정론적이라 멱등.
  want_step evalset && rq_step "s4-04-evalset" ./s4-04-prep-gsm8k.sh "$MODE"
  want_step evalset && rq_step "s4-16-prep-kmmluko" ./s4-16-prep-kmmluko.sh "$MODE"
  want_step evalset && rq_step "s4-22-format-prepare" ./s4-22-format.sh "$MODE" --prepare
  rq_step "s4-07-eval-base" ./s4-07-eval-base.sh "$MODE"
  rq_step "s4-10-eval-quantized" ./s4-10-eval-quantized.sh "$MODE" "$_targets"
  # 채점(40 GSM8K / 46 한국어 / 49 형식)
  rq_step "s4-13-gsm8k" ./s4-13-gsm8k.sh "$MODE"
  rq_step "s4-19-kmmluko" ./s4-19-kmmluko.sh "$MODE"
  rq_step "s4-22-format" ./s4-22-format.sh "$MODE"
  rq_stage_record 4 eval pass "$RQ_OUTPUTS/eval-$MODE/score.json"
  # Stage 4 게이트: 응답 생성 + GSM8K 정확도 + 한국어 정확도 + 형식 준수
  for _t in $(echo "$_eval_targets" | tr ',' ' '); do
    rq_step "gate-eval" rq_gate_eval "$RQ_OUTPUTS/eval-$MODE/$_t.jsonl" "$_t" "$MODE"
    rq_step "gate-kmmluko" rq_gate_kmmluko "$MODE" "$_t"
    rq_step "gate-format" rq_gate_format "$MODE" "$_t"
  done
  # Tier 3: 서빙 처리량(서버가 떠 있을 때만 측정 — 없으면 skip)
  want_step_serve && rq_step "s4-25-servebench" ./s4-25-servebench.sh "$MODE"
fi
if want_step prep; then
  rq_step "s4-31-prep" ./s4-31-prep-upload.sh "$MODE" "$_targets"
  rq_stage_record 4 eval pass "$RQ_OUTPUTS/prep-upload-$MODE.json"
fi

# Stage 8 릴리스 게이트는 명시 요청 시에만(업로드 전 감사 s4-34가 필요하므로 기본 경로 밖).
if want_step_explicit verify; then
  rq_step "s4-34-verify" ./s4-34-verify-upload.sh "$MODE" "$_targets"
  rq_step "gate-release" rq_gate_release "$MODE"
fi

rq_log "run-$MODE done"
# --- 선택 (수동) ---
# ./s5-01-upload-hf.sh full --dry-run   # s4-31-prep 산출물 검증만 (네트워크 없음)
# ./s5-01-upload-hf.sh full              # smoke는 업로드하지 않음
# ./s7-13-chat.sh full nvfp4-awq      # 서버 기동 + chat 검증 (서버 상주)
# ./s7-16-pi.sh full nvfp4-awq     # pi 검증 후 서버 정리
# ./s7-19-verify-serving-log.sh full   # 서빙 감사
