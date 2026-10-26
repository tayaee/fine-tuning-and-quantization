# build-ppl-eval.inc — 단일 (mode × target) "빌드 → PPL → eval" 공유 엔진.
#
# wrapper 8종이 이 파일을 source하고 rq_bpe_main을 호출한다:
#   s0-s7-build-{smoke,full}-{awq-w3|awq-w5|nvfp4-awq|nvfp4-gptq}.sh
# run-{smoke,full}.sh의 TARGETS/STEPS 조합을 한 타깃으로 좁힌 버전이다.
#
# 스텝(기본 STEPS):
#   setup(10) → calib(13) → compat(16) → dequant(19) → quant(22·25·28·31) → ppl(34) → eval(37+40)
#   → prep(56 모델카드) → verify(57 감사) — 빌드가 끝나면 검증된 모델 + 검증결과가 완료된다.
#
# 환경변수:
#   STEPS=quant,ppl        실행할 스텝만 (기본: setup,calib,compat,dequant,quant,ppl,eval,prep,verify)
#   FORCE=1                freshness 무시(강제 재계산)
#   SKIP_SETUP=1           s0-10-setup 생략
#   QUANT_CPU_ONLY=1       양자화만 GPU 숨김(PPL/eval은 GPU)
#   CALIB_N/PPL_N/EVAL_N   규모 오버라이드(config/common.env와 동일 규약)
#
# 주의: wrapper는 config/common.env를 source한 *뒤에* 이 파일을 source한다.
#       (common.env의 EXIT 트랩을 이 파일이 대체한다.)

BPE_STEPS_DEFAULT="setup,acceptance,calib,compat,dequant,quant,ppl,eval,prep,verify"
BPE_T0=$(date +%s)
BPE_TIMING=""

# 타깃 → "<양자화 스크립트> <기대 압축 포맷>"
rq_bpe_spec() {
  case "$1" in
    awq-w3) echo "s5-22-quant-awq-w3.sh pack-quantized" ;;
    awq-w5) echo "s5-25-quant-awq-w5.sh pack-quantized" ;;
    nvfp4-awq) echo "s5-28-quant-nvfp4-awq.sh nvfp4-pack-quantized" ;;
    nvfp4-gptq) echo "s5-31-quant-nvfp4-gptq.sh nvfp4-pack-quantized" ;;
    *) echo "" ;;
  esac
}

rq_bpe_want_step() {
  case ",${STEPS:-$BPE_STEPS_DEFAULT}," in *,all,*|*,"$1",*) return 0 ;; *) return 1 ;; esac
}

# 스텝 실행 + timing 기록(로그 포맷은 run-{smoke,full}.sh와 동일).
rq_bpe_step() {
  local mode="$1" timing="$2" name="$3"; shift 3
  local _t0 _dt _rc=0
  _t0=$(date +%s)
  rq_log "step start: $name"
  "$@" || _rc=$?
  _dt=$(( $(date +%s) - _t0 ))
  rq_log "step done: $name elapsed=${_dt}s rc=$_rc"
  printf '%s\t%s\t%s\t%ss\trc=%d\n' "$(date -u +%FT%TZ)" "$mode" "$name" "$_dt" "$_rc" >>"$timing"
  # SDLC 9단계 원장 기록(실패도 기록).
  local _stage _sid _sname _st _tgt
  _stage="$(rq_stage_of "$name")"
  if [ -n "$_stage" ]; then
    _sid="${_stage%% *}"; _sname="${_stage##* }"
    if [ "$_rc" -eq 0 ]; then _st=pass; else _st=fail; fi
    _tgt="$(rq_target_of "$name")"
    if [ -n "$_tgt" ]; then
      rq_stage_target "$_sid" "$_sname" "$_st" "$_tgt" "$name"
    else
      rq_stage_record "$_sid" "$_sname" "$_st" "$name"
    fi
  fi
  return "$_rc"
}

# 게이트는 gates.inc의 공용 구현을 쓴다(9단계 SDLC의 단계 간 계약).
#   rq_gate_acceptance : Stage 1 계약(수용 기준) 유효성
#   rq_gate_quant      : 양자화 산출물 계약(dense 폴백·미보정 global_scale 차단)
#   rq_gate_ppl        : PPL 유한값 + 수용 기준 판정 일치
#   rq_gate_eval       : eval 응답 생성 여부
#   rq_gate_release    : 모델카드·지표·업로드 전 감사(Stage 8)
# quant_common.verify_compressed_output는 양자화 스크립트 내부에서도 호출되지만,
# 그 검사는 "새로 빌드할 때"만 걸린다. freshness로 건너뛴 기존 산출물이
# 오염돼 있으면(=이번에 재빌드하지 않았어도) 게이트가 잡아낸다.

rq_bpe_done() {
  rq_log "build-ppl-eval end rc=${RQ_RC:-0} total=$(( $(date +%s) - BPE_T0 ))s (timing: ${BPE_TIMING:-none})"
}

rq_bpe_main() {
  : "${BPE_MODE:?build-ppl-eval.inc: BPE_MODE 미설정 (wrapper가 export할 것)}"
  : "${BPE_TARGET:?build-ppl-eval.inc: BPE_TARGET 미설정 (wrapper가 export할 것)}"
  local mode="$BPE_MODE" target="$BPE_TARGET"
  case "$mode" in
    smoke|full) ;;
    *) echo "잘못된 BPE_MODE: $mode (smoke|full)" >&2; return 1 ;;
  esac
  if [ "${MODE:-}" != "$mode" ]; then
    echo "MODE(${MODE:-unset}) != BPE_MODE($mode) — wrapper의 common.env 인자가 잘못됨" >&2
    return 1
  fi
  local spec qscript qformat
  spec="$(rq_bpe_spec "$target")"
  if [ -z "$spec" ]; then
    echo "알 수 없는 타깃: $target (awq-w3|awq-w5|nvfp4-awq|nvfp4-gptq)" >&2
    return 1
  fi
  qscript="${spec%% *}"
  qformat="${spec##* }"

  BPE_TIMING="$RQ_ROOT/logs/timing-build-$mode.log"
  mkdir -p "$RQ_ROOT/logs"
  local model_dir="$RQ_MODELS/raon-$target-$mode"
  local eval_targets="base,$target"
  rq_repro_echo
  rq_log "build-ppl-eval start mode=$mode target=$target steps=${STEPS:-$BPE_STEPS_DEFAULT}"

  # 양자화만 CPU 강제(QUANT_CPU_ONLY=1): GPU를 가리고 PPL/eval은 GPU 사용.
  local -a quant_cmd=(env)
  if [ "${QUANT_CPU_ONLY:-0}" = 1 ]; then
    quant_cmd=(env CUDA_VISIBLE_DEVICES=)
    rq_log "QUANT_CPU_ONLY=1 — 양자화 단계만 GPU 숨김, PPL/eval은 GPU 사용"
  fi

  if [ "${SKIP_SETUP:-0}" != 1 ] && rq_bpe_want_step setup; then
    rq_bpe_step "$mode" "$BPE_TIMING" "10-setup" ./s0-10-setup.sh
  fi
  # Stage 1 — 수용 기준(s1-07-acceptance). 게이트의 판정 기준이 여기서 정해진다.
  if rq_bpe_want_step acceptance; then
    rq_bpe_step "$mode" "$BPE_TIMING" "07-acceptance" ./s1-07-acceptance.sh "$mode"
    rq_stage_record 1 acceptance pass "$RQ_OUTPUTS/acceptance-$mode.json"
    rq_bpe_step "$mode" "$BPE_TIMING" "gate-acceptance" rq_gate_acceptance "$mode"
  fi
  if rq_bpe_want_step calib; then
    rq_bpe_step "$mode" "$BPE_TIMING" "13-calib" ./s2-13-prep-calibration.sh "$mode"
  fi
  if rq_bpe_want_step compat; then
    rq_bpe_step "$mode" "$BPE_TIMING" "16-compat" ./s3-16-compat-check.sh "$mode"
  fi
  if rq_bpe_want_step dequant; then
    rq_bpe_step "$mode" "$BPE_TIMING" "19-dequant" ./s4-19-dequant-mxfp8.sh "$mode"
  fi
  if rq_bpe_want_step quant; then
    rq_bpe_step "$mode" "$BPE_TIMING" "${qscript%.sh}" "${quant_cmd[@]}" "./$qscript" "$mode"
    rq_bpe_step "$mode" "$BPE_TIMING" "gate-quant" rq_gate_quant "$model_dir" "$qformat"
    rq_stage_target 5 quant pass "$target" "$model_dir"
  fi
  if rq_bpe_want_step ppl; then
    rq_bpe_step "$mode" "$BPE_TIMING" "34-ppl" ./s6-34-measure-ppl.sh "$mode" "$eval_targets"
    rq_bpe_step "$mode" "$BPE_TIMING" "gate-ppl" rq_gate_ppl "$RQ_OUTPUTS/ppl-$mode/$target.json" "$target"
    rq_stage_record 6 ppl pass "$RQ_OUTPUTS/ppl-$mode/summary.json"
  fi
  if rq_bpe_want_step eval; then
    if rq_bpe_want_step evalset; then
      rq_bpe_step "$mode" "$BPE_TIMING" "36-evalset" ./s7-36-prep-gsm8k.sh "$mode"
      rq_bpe_step "$mode" "$BPE_TIMING" "43-prep-kmmluko" ./s7-43-prep-kmmluko.sh "$mode"
      rq_bpe_step "$mode" "$BPE_TIMING" "49-format-prepare" ./s7-49-format.sh "$mode" --prepare
    fi
    rq_bpe_step "$mode" "$BPE_TIMING" "s7-37-eval" ./s7-37-eval.sh "$mode" "$eval_targets"
    rq_bpe_step "$mode" "$BPE_TIMING" "40-gsm8k" ./s7-40-gsm8k.sh "$mode"
    rq_bpe_step "$mode" "$BPE_TIMING" "46-kmmluko" ./s7-46-kmmluko.sh "$mode"
    rq_bpe_step "$mode" "$BPE_TIMING" "49-format" ./s7-49-format.sh "$mode"
    rq_bpe_step "$mode" "$BPE_TIMING" "gate-eval" rq_gate_eval \
      "$RQ_OUTPUTS/eval-$mode/$target.jsonl" "$target" "$mode"
    rq_bpe_step "$mode" "$BPE_TIMING" "gate-kmmluko" rq_gate_kmmluko "$mode" "$target"
    rq_bpe_step "$mode" "$BPE_TIMING" "gate-format" rq_gate_format "$mode" "$target"
    # Tier 3: 서빙 처리량(STEPS=serve로 명시 요청 시에만 — 서버가 떠 있어야 측정된다)
    case ",${STEPS:-$BPE_STEPS_DEFAULT}," in *,serve,*)
      rq_bpe_step "$mode" "$BPE_TIMING" "52-servebench" ./s7-52-servebench.sh "$mode" ;;
    esac
    rq_stage_record 7 eval pass "$RQ_OUTPUTS/eval-$mode/score.json"
  fi
  if rq_bpe_want_step prep; then
    rq_bpe_step "$mode" "$BPE_TIMING" "56-prep" ./s7-56-prep-upload.sh "$mode" "$target"
    rq_stage_record 7 eval pass "$RQ_OUTPUTS/prep-upload-$mode.json"
  fi
  if rq_bpe_want_step verify; then
    rq_bpe_step "$mode" "$BPE_TIMING" "57-verify" ./s7-57-verify-upload.sh "$mode" "$target"
  fi

  rq_log "build-ppl-eval done mode=$mode target=$target"
}

trap 'RQ_RC=$?; rq_repro; rq_bpe_done' EXIT
