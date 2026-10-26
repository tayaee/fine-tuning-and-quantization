#!/usr/bin/env bash
# run-full.sh — full 전 파이프라인 원샷 (calib 256·seq 2048, 밤샘 권장).
# NOTE: 28.8B MoE NVFP4-GPTQ/AWQ가 가장 오래 걸린다. 공식 KRAFTON NVFP4와 PPL 비교 권장.
# TARGETS: 양자화 타깃 부분집합 (콤마 구분) — 2노드 분산용. **미지정 시 가능한 값을 찍고 종료**. PPL/eval/prep은 base를 앞에 붙인다.
#   예: TARGETS=all ./run-full.sh                              # 4종 전부
#       TARGETS=awq-w3,awq-w5 STEPS=dequant,quant ./run-full.sh        # spark1 분담
#       TARGETS=nvfp4-awq,nvfp4-gptq STEPS=dequant,quant ./run-full.sh  # spark2 분담
#   smoke와 동일한 TARGETS 분할을 쓴다 (dequant 공유 1회, PPL/eval은 전 타깃 모인 뒤 1곳에서).
# QUANT_CPU_ONLY=1 ./run-full.sh  # 양자화 단계만 GPU를 가리고 PPL/eval은 GPU 사용.
# 제외: s8-49-upload-hf, s8-64-serve, s8-67-infer, s8-70-serve-infer-all → 맨 아래 주석 명령으로 개별 실행.
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
  # SDLC 9단계 원장 기록 (실패도 기록한다 — 증빙이므로).
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
#   STEPS=verify ./run-full.sh   → 57-verify + gate-release 만 추가
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

rq_banner "run-$MODE" "9-stage pipeline: setup -> acceptance -> calib -> compat -> dequant -> quant -> ppl -> eval -> release (TARGETS is required)"
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
  rq_step "10-setup" ./s0-10-setup.sh
fi

# Stage 1 — 수용 기준. 이게 없으면 뒤 단계의 판정 기준이 없다(s1-07-acceptance).
if want_step acceptance; then
  rq_step "07-acceptance" ./s1-07-acceptance.sh "$MODE"
  rq_stage_record 1 acceptance pass "$RQ_OUTPUTS/acceptance-$MODE.json"
fi

want_step calib && rq_step "13-calib" ./s2-13-prep-calibration.sh "$MODE"
if want_step compat; then
  rq_step "16-compat" ./s3-16-compat-check.sh "$MODE"
  rq_stage_record 3 compat pass "$RQ_OUTPUTS/compat.stamp"
fi
if want_step dequant; then
  rq_step "19-dequant" ./s4-19-dequant-mxfp8.sh "$MODE"
  rq_stage_record 4 dequant pass "$RQ_MODELS/raon-bf16"
fi

# base(19-dequant 복원본)를 항상 포함 — PPL Δ 판정과 모델 카드 비교표의 기준선이다.
_eval_targets="base,$_targets"

# 양자화만 CPU 강제(QUANT_CPU_ONLY=1): env로 자식 프로세스에만 빈 값을 넘겨
# 34-ppl/s7-37-eval은 그대로 GPU를 본다 (부모 셸을 바꾸면 평가가 CPU로 떨어진다).
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
        rq_step "22-awq-w3" "${_quant_cmd[@]}" ./s5-22-quant-awq-w3.sh "$MODE"
        rq_step "gate-quant" rq_gate_quant "$RQ_MODELS/raon-$_t-$MODE" "$(rq_quant_format "$_t")"
        rq_stage_target 5 quant pass "$_t" "$RQ_MODELS/raon-$_t-$MODE" ;;
      awq-w5)
        rq_step "25-awq-w5" "${_quant_cmd[@]}" ./s5-25-quant-awq-w5.sh "$MODE"
        rq_step "gate-quant" rq_gate_quant "$RQ_MODELS/raon-$_t-$MODE" "$(rq_quant_format "$_t")"
        rq_stage_target 5 quant pass "$_t" "$RQ_MODELS/raon-$_t-$MODE" ;;
      nvfp4-awq)
        rq_step "28-nvfp4-awq" "${_quant_cmd[@]}" ./s5-28-quant-nvfp4-awq.sh "$MODE"
        rq_step "gate-quant" rq_gate_quant "$RQ_MODELS/raon-$_t-$MODE" "$(rq_quant_format "$_t")"
        rq_stage_target 5 quant pass "$_t" "$RQ_MODELS/raon-$_t-$MODE" ;;
      nvfp4-gptq)
        rq_step "31-nvfp4-gptq" "${_quant_cmd[@]}" ./s5-31-quant-nvfp4-gptq.sh "$MODE"
        rq_step "gate-quant" rq_gate_quant "$RQ_MODELS/raon-$_t-$MODE" "$(rq_quant_format "$_t")"
        rq_stage_target 5 quant pass "$_t" "$RQ_MODELS/raon-$_t-$MODE" ;;
      *) echo "unknown quant target: $_t (awq-w3|awq-w5|nvfp4-awq|nvfp4-gptq)" >&2; exit 1 ;;
    esac
  done
fi

if want_step ppl; then
  rq_step "34-ppl" ./s6-34-measure-ppl.sh "$MODE" "$_eval_targets"
  rq_stage_record 6 ppl pass "$RQ_OUTPUTS/ppl-$MODE/summary.json"
  # Stage 6 게이트: PPL이 유한하고 수용 기준(Δ) 판정과 일치하는지 확인한다.
  for _t in $(echo "$_targets" | tr ',' ' '); do
    rq_step "gate-ppl" rq_gate_ppl "$RQ_OUTPUTS/ppl-$MODE/$_t.json" "$_t"
  done
fi
if want_step eval; then
  # 평가 세트 준비(36 GSM8K / 43 한국어 KMMLU / 49 형식 준수) — 결정론적이라 멱등.
  want_step evalset && rq_step "36-evalset" ./s7-36-prep-gsm8k.sh "$MODE"
  want_step evalset && rq_step "43-prep-kmmluko" ./s7-43-prep-kmmluko.sh "$MODE"
  want_step evalset && rq_step "49-format-prepare" ./s7-49-format.sh "$MODE" --prepare
  rq_step "37-eval" ./s7-37-eval.sh "$MODE" "$_eval_targets"
  # 채점(40 GSM8K / 46 한국어 / 49 형식)
  rq_step "40-gsm8k" ./s7-40-gsm8k.sh "$MODE"
  rq_step "46-kmmluko" ./s7-46-kmmluko.sh "$MODE"
  rq_step "49-format" ./s7-49-format.sh "$MODE"
  rq_stage_record 7 eval pass "$RQ_OUTPUTS/eval-$MODE/score.json"
  # Stage 7 게이트: 응답 생성 + GSM8K 정확도 + 한국어 정확도 + 형식 준수
  for _t in $(echo "$_eval_targets" | tr ',' ' '); do
    rq_step "gate-eval" rq_gate_eval "$RQ_OUTPUTS/eval-$MODE/$_t.jsonl" "$_t" "$MODE"
    rq_step "gate-kmmluko" rq_gate_kmmluko "$MODE" "$_t"
    rq_step "gate-format" rq_gate_format "$MODE" "$_t"
  done
  # Tier 3: 서빙 처리량(서버가 떠 있을 때만 측정 — 없으면 skip)
  want_step_serve && rq_step "52-servebench" ./s7-52-servebench.sh "$MODE"
fi
if want_step prep; then
  rq_step "56-prep" ./s7-56-prep-upload.sh "$MODE" "$_targets"
  rq_stage_record 7 eval pass "$RQ_OUTPUTS/prep-upload-$MODE.json"
fi

# Stage 8 릴리스 게이트는 명시 요청 시에만(업로드 전 감사 46이 필요하므로 기본 경로 밖).
if want_step_explicit verify; then
  rq_step "57-verify" ./s7-57-verify-upload.sh "$MODE" "$_targets"
  rq_step "gate-release" rq_gate_release "$MODE"
fi

rq_log "run-$MODE done"
# --- 선택 (수동) ---
# ./s8-49-upload-hf.sh full --dry-run   # 56-prep 산출물 검증만 (네트워크 없음)
# ./s8-49-upload-hf.sh full              # smoke는 업로드하지 않음
# ./s8-64-serve-vllm.sh full nvfp4-awq     # 터미널1 (상주, README 카드 명령과 동일, 양자화 타깃 TP=1)
# ./s8-67-infer-examples.sh 3              # 터미널2
# ./s8-70-serve-infer-all.sh는 TARGETS 필수: TARGETS=all ./s8-70-serve-infer-all.sh full 3
