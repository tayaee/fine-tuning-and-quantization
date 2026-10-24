# build-ppl-eval.inc — 단일 (mode × target) "빌드 → PPL → eval" 공유 엔진.
#
# wrapper 8종이 이 파일을 source하고 rq_bpe_main을 호출한다:
#   10-16,19,{22|25|28|31}-40-build-{smoke,full}-{awq-w3|awq-w5|nvfp4-awq|nvfp4-gptq}.sh
# run-{smoke,full}.sh의 TARGETS/STEPS 조합을 한 타깃으로 좁힌 버전이다.
#
# 스텝(기본 STEPS):
#   setup(10) → calib(13) → compat(16) → dequant(19) → quant(22·25·28·31) → ppl(34) → eval(37+40)
#   선택: prep(43 모델카드) · verify(46 업로드 전 감사)
#
# 환경변수:
#   STEPS=quant,ppl        실행할 스텝만 (기본: setup,calib,compat,dequant,quant,ppl,eval)
#   FORCE=1                freshness 무시(강제 재계산)
#   SKIP_SETUP=1           10-setup 생략
#   QUANT_CPU_ONLY=1       양자화만 GPU 숨김(PPL/eval은 GPU)
#   CALIB_N/PPL_N/EVAL_N   규모 오버라이드(config/common.env와 동일 규약)
#
# 주의: wrapper는 config/common.env를 source한 *뒤에* 이 파일을 source한다.
#       (common.env의 EXIT 트랩을 이 파일이 대체한다.)

BPE_STEPS_DEFAULT="setup,calib,compat,dequant,quant,ppl,eval"
BPE_T0=$(date +%s)
BPE_TIMING=""

# 타깃 → "<양자화 스크립트> <기대 압축 포맷>"
rq_bpe_spec() {
  case "$1" in
    awq-w3) echo "22-quant-awq-w3.sh pack-quantized" ;;
    awq-w5) echo "25-quant-awq-w5.sh pack-quantized" ;;
    nvfp4-awq) echo "28-quant-nvfp4-awq.sh nvfp4-pack-quantized" ;;
    nvfp4-gptq) echo "31-quant-nvfp4-gptq.sh nvfp4-pack-quantized" ;;
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
  return "$_rc"
}

# 게이트 1 — 양자화 산출물 계약(dense 폴백·미보정 global_scale 차단).
# quant_common.verify_compressed_output는 32/33/35/36 스크립트 내부에서도 호출되지만,
# 그 검사는 "새로 빌드할 때"만 걸린다. freshness로 건너뛴 기존 산출물이
# 오염돼 있으면(=이번에 재빌드하지 않았어도) 여기서 잡아낸다.
rq_bpe_gate_quant() {
  local model_dir="$1" fmt="$2"
  ( set -x; cd "$RQ_ROOT" && uv run python -c '
import sys
from quant_common import verify_compressed_output
try:
    verify_compressed_output(sys.argv[1], sys.argv[2])
except Exception as e:
    print("FAIL: %s" % e, file=sys.stderr)
    raise SystemExit(1)
' "$model_dir" "$fmt" )
}

# 게이트 2 — PPL 측정값 유효성. NaN/미측정이면 '오염된 모델'을 eval에 흘려보내지 않는다.
# (34-measure-ppl.py는 NaN이면 <target>.json을 아예 쓰지 않는다.)
rq_bpe_gate_ppl() {
  local f="$1"
  if [ ! -f "$f" ]; then
    echo "FAIL: $f 없음 — PPL 미측정(유효 토큰 0/NaN). 양자화 산출물이 오염됐을 가능성" >&2
    return 1
  fi
  ( set -x; uv run python -c '
import json, math, sys
d = json.load(open(sys.argv[1]))
ppl = d.get("ppl")
if ppl is None or not math.isfinite(float(ppl)):
    raise SystemExit("FAIL: %s ppl=%r (비유한 값)" % (sys.argv[1], ppl))
print("PPL OK %s ppl=%s n=%s seq_len=%s" % (sys.argv[1], ppl, d.get("n"), d.get("seq_len")))
' "$f" )
}

# 게이트 3 — eval 응답 생성 여부(빈 파일이면 모델이 아무것도 못 냈다).
rq_bpe_gate_eval() {
  local f="$1"
  if [ ! -s "$f" ]; then
    echo "FAIL: $f 없음/빈 파일 — eval 응답 미생성" >&2
    return 1
  fi
  local n
  n=$(wc -l <"$f")
  if [ "${n:-0}" -lt 1 ]; then
    echo "FAIL: $f 라인 0 — eval 응답 미생성" >&2
    return 1
  fi
  echo "eval OK $f lines=$n"
}

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
    rq_bpe_step "$mode" "$BPE_TIMING" "10-setup" ./10-setup.sh
  fi
  if rq_bpe_want_step calib; then
    rq_bpe_step "$mode" "$BPE_TIMING" "13-calib" ./13-prep-calibration.sh "$mode"
  fi
  if rq_bpe_want_step compat; then
    rq_bpe_step "$mode" "$BPE_TIMING" "16-compat" ./16-compat-check.sh "$mode"
  fi
  if rq_bpe_want_step dequant; then
    rq_bpe_step "$mode" "$BPE_TIMING" "19-dequant" ./19-dequant-mxfp8.sh "$mode"
  fi
  if rq_bpe_want_step quant; then
    rq_bpe_step "$mode" "$BPE_TIMING" "${qscript%.sh}" "${quant_cmd[@]}" "./$qscript" "$mode"
    rq_bpe_step "$mode" "$BPE_TIMING" "gate-quant" rq_bpe_gate_quant "$model_dir" "$qformat"
  fi
  if rq_bpe_want_step ppl; then
    rq_bpe_step "$mode" "$BPE_TIMING" "34-ppl" ./34-measure-ppl.sh "$mode" "$eval_targets"
    rq_bpe_step "$mode" "$BPE_TIMING" "gate-ppl" rq_bpe_gate_ppl "$RQ_OUTPUTS/ppl-$mode/$target.json"
  fi
  if rq_bpe_want_step eval; then
    rq_bpe_step "$mode" "$BPE_TIMING" "37-eval" ./37-eval.sh "$mode" "$eval_targets"
    rq_bpe_step "$mode" "$BPE_TIMING" "gate-eval" rq_bpe_gate_eval "$RQ_OUTPUTS/eval-$mode/$target.jsonl"
    rq_bpe_step "$mode" "$BPE_TIMING" "40-score" ./40-score.sh "$mode"
  fi
  if rq_bpe_want_step prep; then
    rq_bpe_step "$mode" "$BPE_TIMING" "43-prep" ./43-prep-upload.sh "$mode" "$target"
  fi
  if rq_bpe_want_step verify; then
    rq_bpe_step "$mode" "$BPE_TIMING" "46-verify" ./46-verify-upload.sh "$mode" "$target"
  fi

  rq_log "build-ppl-eval done mode=$mode target=$target"
}

trap 'RQ_RC=$?; rq_repro; rq_bpe_done' EXIT
